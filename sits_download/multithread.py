import pandas as pd
from typing import List, Callable, Union, Tuple
import numpy as np
from tqdm.std import tqdm
from returns.result import Success, Failure
import pathlib
import pandera as pa
import anyio
import logging
import logging.handlers
import queue
import warnings
import hashlib
from functools import partial
import concurrent.futures


def create_gdf_hash(gdf):
    gdf_copy = gdf[["geometry", "start_date", "end_date"]].copy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gdf_copy["centroid_x"] = gdf_copy.geometry.centroid.x
        gdf_copy["centroid_y"] = gdf_copy.geometry.centroid.y

    gdf_copy = gdf_copy.drop(columns=["geometry"])

    hash = pd.util.hash_pandas_object(gdf_copy)
    return hashlib.sha1(hash.values).hexdigest()


def multithread_download_sits(
    gdf: pd.DataFrame,
    output_filestem: str,
    save_columns: List[str],
    sits_download_function: Callable,
    result_shape: Tuple[int, ...],
    num_threads_rush: int = 80,
    num_threads_retry: int = 20,
    **kwargs,
) -> Tuple[int, int]:
    all_raw = np.zeros((len(gdf), *result_shape), dtype=np.float16)
    indexes_with_errors: List[int] = []

    def process_download(n, geometry, start_date, end_date):
        result = sits_download_function(geometry, start_date, end_date, **kwargs)

        match result:
            case Success((raw)):
                num_timestamps = raw.shape[0]
                all_raw[n, :num_timestamps] = raw

            case Failure(error_message):
                if error_message.startswith(
                    "Too Many Requests"
                ) or error_message.startswith("Quota exceeded"):
                    indexes_with_errors.append(n)
                return f"{n}-{error_message}"

    def run_downloads(indexes, num_threads, desc):
        error_count = 0
        with (
            open(f"{output_filestem}.log", "a") as logfile,
            concurrent.futures.ThreadPoolExecutor(num_threads) as executor,
        ):
            futures = {
                executor.submit(
                    process_download,
                    n,
                    gdf.iloc[n].geometry,
                    gdf.iloc[n].start_date,
                    gdf.iloc[n].end_date,
                ): n
                for n in indexes
            }
            pbar = tqdm(
                concurrent.futures.as_completed(futures), total=len(futures), desc=desc
            )
            for future in pbar:
                error_message = future.result()
                if error_message:
                    error_count += 1
                    pbar.set_postfix({"Errors": error_count})
                    logfile.write(f"{error_message}\n")
                    logfile.flush()

    run_downloads(gdf.index, num_threads=num_threads_rush, desc="Downloading")

    if indexes_with_errors:
        run_downloads(
            indexes_with_errors,
            num_threads=num_threads_retry,
            desc="Re-running failed downloads",
        )

    save_data = {
        "result": all_raw,
        **{col: gdf[col].values for col in save_columns if col in gdf.columns},
    }
    np.savez_compressed(f"{output_filestem}.npz", **save_data)  # type: ignore

    return 0, 0


async def download_sits_anyio(
    gdf: pd.DataFrame,
    output_filestem: str,
    sits_download_function: Callable[..., Union[Success, Failure]],
    result_shape: Tuple[int, ...],
    save_columns: List[str],
    initial_concurrency: int = 80,
    retry_concurrency: int = 20,
    initial_timeout: float = 20,
    retry_timeout: float = 40,
    **kwargs,
) -> Tuple[int, int]:
    log_queue: queue.Queue[logging.LogRecord] = queue.Queue(-1)

    file_handler = logging.FileHandler(f"{output_filestem}.log", mode="a")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)

    queue_listener = logging.handlers.QueueListener(log_queue, file_handler)
    queue_listener.start()

    queue_handler = logging.handlers.QueueHandler(log_queue)
    logger = logging.getLogger(f"logger_{output_filestem}")
    logger.setLevel(logging.ERROR)
    logger.addHandler(queue_handler)
    logger.propagate = False

    all_raw = np.zeros((len(gdf), *result_shape), dtype=np.float16)

    async def run_download(indices, concurrency, timeout):
        sem = anyio.Semaphore(concurrency)

        async def download_task(i: int):
            async with sem:
                row = gdf.iloc[i]
                try:
                    with anyio.fail_after(timeout):
                        result = await anyio.to_thread.run_sync(
                            partial(
                                sits_download_function,
                                row.geometry,
                                row.start_date,
                                row.end_date,
                                **kwargs,
                            )
                        )
                except TimeoutError as e:
                    all_raw[i] = -1
                    logger.error(f"TimeoutError at index {i}: {str(e)}")
                    return

                match result:
                    case Success(raw):
                        num_timestamps = raw.shape[0]
                        all_raw[i, :num_timestamps] = raw
                    case Failure(error_message):
                        logger.error(f"Failure at index {i}: {error_message}")
                        if error_message.startswith(
                            ("Too Many Requests", "Quota exceeded")
                        ):
                            all_raw[i] = -2
                        else:
                            all_raw[i] = -3

        async with anyio.create_task_group() as tg:
            for i in indices:
                tg.start_soon(download_task, i)

    await run_download(range(len(gdf)), initial_concurrency, initial_timeout)

    retry_indices = np.where((all_raw[..., 0] == -1) | (all_raw[..., 0] == -2))[
        0
    ].tolist()

    print("Retrying indices:", len(retry_indices))  # type: ignore

    if retry_indices:
        await run_download(retry_indices, retry_concurrency, retry_timeout)

    save_data = {
        "result": all_raw,
        **{col: gdf[col].values for col in save_columns if col in gdf.columns},
    }
    np.savez_compressed(f"{output_filestem}.npz", **save_data)  # type: ignore

    timeout_errors = int(np.sum(all_raw[..., 0] == -1))
    gee_errors = int(np.sum(all_raw[..., 0] == -2))

    queue_listener.stop()

    return timeout_errors, gee_errors


async def async_download_large_gdf_in_chunks(
    gdf: pd.DataFrame,
    sits_download_function: Callable,
    result_shape,
    save_columns: List[str],
    final_output_filestem: str,
    chunk_size: int = 3000,
    initial_concurrency: int = 60,
    retry_concurrency: int = 20,
    initial_timeout: float = 20,
    retry_timeout: float = 10,
    **kwargs,
):
    schema = pa.DataFrameSchema(
        {
            "geometry": pa.Column("geometry", nullable=False),
            "start_date": pa.Column(pa.DateTime, nullable=False),
            "end_date": pa.Column(pa.DateTime, nullable=False),
            **{col: pa.Column() for col in save_columns},
        }
    )

    schema.validate(gdf, lazy=True)

    hashlib_gdf = create_gdf_hash(gdf)
    function_name = sits_download_function.__name__
    kwargs_str = "_".join(
        f"{k}_{str(v).replace('/', '_').replace(' ', '_')}" for k, v in kwargs.items()
    )
    output_path = (
        pathlib.Path("data/temp")
        / f"{function_name}_{hashlib_gdf}_{chunk_size}_{kwargs_str}"
    )
    output_path.mkdir(parents=True, exist_ok=True)

    existing_chunks = {
        int(f.stem) for f in output_path.glob("*.npz") if f.stem.isdigit()
    }

    chunks = [
        gdf.iloc[i : i + chunk_size].reset_index(drop=True)
        for i in range(0, len(gdf), chunk_size)
    ]

    print(
        f"Number of chunks: {len(chunks)}, skipping {len(existing_chunks)} already downloaded chunks"
    )

    pbar = tqdm(enumerate(chunks), total=len(chunks))
    all_timeout_errors = 0
    all_gee_errors = 0

    for idx, chunk in pbar:
        if idx in existing_chunks:
            continue

        output_filestem = str(output_path) + "/" + f"{idx}"

        timeout_errors, gee_errors = await download_sits_anyio(
            gdf=chunk,
            output_filestem=output_filestem,
            sits_download_function=sits_download_function,
            result_shape=result_shape,
            save_columns=save_columns,
            initial_concurrency=initial_concurrency,
            initial_timeout=initial_timeout,
            retry_concurrency=retry_concurrency,
            retry_timeout=retry_timeout,
            **kwargs,
        )

        all_timeout_errors += timeout_errors
        all_gee_errors += gee_errors

        pbar.set_postfix(
            {"Timeout Errors": all_timeout_errors, "GEE Errors": all_gee_errors}
        )

    chunk_files = sorted(output_path.glob("*.npz"), key=lambda f: int(f.stem))

    with np.load(chunk_files[0], allow_pickle=True) as data:
        all_arrays = {key: data[key] for key in data.files}

    for file in chunk_files[1:]:
        with np.load(file, allow_pickle=True) as data:
            for key in data.files:
                all_arrays[key] = np.concatenate((all_arrays[key], data[key]), axis=0)

    np.savez_compressed(f"{final_output_filestem}.npz", **all_arrays)

    print(f"Final file saved to: {final_output_filestem}.npz")


def download_large_gdf_in_chunks(
    gdf: pd.DataFrame,
    sits_download_function: Callable,
    result_shape,
    save_columns: List[str],
    final_output_filestem: str,
    chunk_size: int = 10000,
    initial_concurrency: int = 80,
    retry_concurrency: int = 20,
    **kwargs,
):
    if len(gdf) == 0:
        print("Empty GeoDataFrame, nothing to download")
        return

    schema = pa.DataFrameSchema(
        {
            "geometry": pa.Column("geometry", nullable=False),
            "start_date": pa.Column(pa.DateTime, nullable=False),
            "end_date": pa.Column(pa.DateTime, nullable=False),
            **{col: pa.Column() for col in save_columns},
        }
    )

    schema.validate(gdf, lazy=True)

    hashlib_gdf = create_gdf_hash(gdf)
    function_name = sits_download_function.__name__
    kwargs_str = "_".join(
        f"{k}_{str(v).replace('/', '_').replace(' ', '_')}" for k, v in kwargs.items()
    )
    output_path = (
        pathlib.Path("data/temp")
        / f"{function_name}_{hashlib_gdf}_{chunk_size}_{kwargs_str}"
    )
    output_path.mkdir(parents=True, exist_ok=True)

    existing_chunks = {
        int(f.stem) for f in output_path.glob("*.npz") if f.stem.isdigit()
    }

    chunks = [
        gdf.iloc[i : i + chunk_size].reset_index(drop=True)
        for i in range(0, len(gdf), chunk_size)
    ]

    print(
        f"Number of chunks: {len(chunks)}, skipping {len(existing_chunks)} already downloaded chunks"
    )

    for idx, chunk in enumerate(chunks):
        if idx in existing_chunks:
            continue

        output_filestem = str(output_path) + "/" + f"{idx}"

        multithread_download_sits(
            gdf=chunk,
            output_filestem=output_filestem,
            sits_download_function=sits_download_function,
            save_columns=save_columns,
            result_shape=result_shape,
            num_threads_rush=initial_concurrency,
            num_threads_retry=retry_concurrency,
            **kwargs,
        )

    chunk_files = sorted(output_path.glob("*.npz"), key=lambda f: int(f.stem))

    with np.load(chunk_files[0], allow_pickle=True) as data:
        all_arrays = {key: data[key] for key in data.files}

    for file in chunk_files[1:]:
        with np.load(file, allow_pickle=True) as data:
            for key in data.files:
                all_arrays[key] = np.concatenate((all_arrays[key], data[key]), axis=0)

    np.savez_compressed(f"{final_output_filestem}.npz", **all_arrays)

    print(f"Final file saved to: {final_output_filestem}.npz")
