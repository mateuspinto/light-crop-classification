import ee
import numpy as np
import pandas as pd
from typing import List, Union
from returns.result import Result, Success, Failure
from shapely import Polygon

GEE_INDICES = {
    "NDVI": "NDVI = (i.NIR - i.RED) / (i.NIR + i.RED)",
    "EVI2": "EVI2 = 2.5 * (i.NIR - i.RED) / (i.NIR + 2.4 * i.RED + 1)",
}


def fix_gee_compute_values(values: List[Union[float, None]], dtype: type) -> np.ndarray:
    fixed_values = np.array(
        [np.nan if x is None else x for x in values], dtype=np.float16
    )
    fixed_values = np.nan_to_num(fixed_values, nan=0).astype(dtype)

    return fixed_values


def remove_invalid_timestamps_and_merge(
    band_values: np.ndarray, doy_values: np.ndarray
) -> np.ndarray:
    if len(doy_values) == 0:
        return np.array([])

    valid_indices = ~(band_values == 0).any(axis=1)

    valid_band_values = band_values[valid_indices]
    valid_doy_values = doy_values[valid_indices].reshape(-1, 1)

    merged_values = np.hstack([valid_band_values, valid_doy_values])

    return merged_values


def ee_map_doys(img):
    img = ee.Image(img)
    return ee.Number(img.date().getRelative("day", "year").add(1))


def ee_map_valid_pixels(
    img: ee.Image, ee_geometry: ee.Geometry, scale: float
) -> ee.Image:
    mask = ee.Image(img).select([0]).gt(0)

    valid_pixels = ee.Number(
        mask.rename("valid")
        .reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=ee_geometry,
            scale=scale,
            maxPixels=1e8,
            bestEffort=True,
        )
        .get("valid")
    )

    return ee.Image(img.set("ZZ_USER_VALID_PIXELS", valid_pixels))


def ee_cloud_probability_mask(
    img: ee.Image, threshold: float, invert: bool = False
) -> ee.Image:
    if invert:
        mask = img.select(["cloud"]).gte(threshold)
    else:
        mask = img.select(["cloud"]).lt(threshold)

    return img.updateMask(mask).select(img.bandNames().remove("cloud"))


def download_sits_s2_w_cloud_score_plus(
    geometry: Polygon,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    use_sr: bool = False,
) -> Result[np.ndarray, str]:
    ee_geometry = ee.Geometry(geometry.__geo_interface__)
    ee_geometry = ee.Geometry(
        ee.Algorithms.If(
            ee_geometry.buffer(-10).area().gte(35000),
            ee_geometry.buffer(-10),
            ee_geometry,
        )
    )

    ee_start_date = ee.Date(start_date.strftime("%Y-%m-%d"))
    ee_end_date = ee.Date(end_date.strftime("%Y-%m-%d"))

    filter = ee.Filter.And(
        ee.Filter.bounds(ee_geometry), ee.Filter.date(ee_start_date, ee_end_date)
    )

    s2_img = (
        ee.ImageCollection(
            "COPERNICUS/S2_SR_HARMONIZED" if use_sr else "COPERNICUS/S2_HARMONIZED"
        )
        .filter(filter)
        .select(
            ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"],
            [
                "0_blue",
                "1_green",
                "2_red",
                "3_re1",
                "4_re2",
                "5_re3",
                "6_nir",
                "7_re4",
                "8_swir1",
                "9_swir2",
            ],
        )
    )

    s2_cloud_mask = (
        ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
        .filter(filter)
        .select(["cs_cdf"], ["cloud"])
    )

    s2_img = s2_img.combine(s2_cloud_mask)

    s2_img = s2_img.map(lambda img: ee_cloud_probability_mask(img, 0.7, True))
    s2_img = s2_img.map(lambda img: ee_map_valid_pixels(img, ee_geometry, 10)).filter(
        ee.Filter.gte("ZZ_USER_VALID_PIXELS", 20)
    )
    s2_img = (
        s2_img.map(
            lambda img: img.set("ZZ_USER_TIME_DUMMY", img.date().format("YYYY-MM-dd"))
        )
        .sort("ZZ_USER_TIME_DUMMY")
        .distinct("ZZ_USER_TIME_DUMMY")
    )

    def map_bands(img):
        img = ee.Image(img)

        stats = img.reduceRegion(
            reducer=ee.Reducer.median(),
            geometry=ee_geometry,
            scale=10,
            maxPixels=1e2,
            bestEffort=True,
        )
        return stats.values()

    try:
        band_values_raw = ee.data.computeValue(s2_img.toList(10000).map(map_bands))
        doy_values_raw = ee.data.computeValue(s2_img.toList(10000).map(ee_map_doys))

        bands_values = (
            fix_gee_compute_values(band_values_raw, dtype=np.uint16) / 10000
        ).astype(np.float16)
        doy_values = fix_gee_compute_values(doy_values_raw, dtype=np.uint16)

        result = remove_invalid_timestamps_and_merge(bands_values, doy_values)

        if result.shape[0] == 0:
            return Failure("No valid timestamps found")

        return Success((result))
    except Exception as e:
        return Failure(str(e))


def download_agricultural_score(
    geometry: Polygon,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    use_sr: bool = True,
) -> Result[np.ndarray, str]:
    ee_geometry = ee.Geometry(geometry.__geo_interface__)
    ee_geometry = ee.Geometry(
        ee.Algorithms.If(
            ee_geometry.buffer(-10).area().gte(35000),
            ee_geometry.buffer(-10),
            ee_geometry,
        )
    )

    ee_start_date = ee.Date(start_date.strftime("%Y-%m-%d"))
    ee_end_date = ee.Date(end_date.strftime("%Y-%m-%d"))

    filter = ee.Filter.And(
        ee.Filter.bounds(ee_geometry), ee.Filter.date(ee_start_date, ee_end_date)
    )

    s2_img = (
        ee.ImageCollection(
            "COPERNICUS/S2_SR_HARMONIZED" if use_sr else "COPERNICUS/S2_HARMONIZED"
        )
        .filter(filter)
        .select(
            [
                "B4",
                "B8",
            ],
            [
                "RED",
                "NIR",
            ],
        )
        .map(
            lambda img: ee.Image(img).addBands(ee.Image(img).divide(10000), None, True)
        )
    )

    s2_img = s2_img.map(
        lambda img: ee.Image(img).addBands(
            ee.Image(img).expression(GEE_INDICES["EVI2"], {"i": ee.Image(img)}),
            None,
            True,
        )
    ).select(["EVI2"])

    s2_cloud_mask = (
        ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
        .filter(filter)
        .select(["cs_cdf"], ["cloud"])
    )

    s2_img = s2_img.combine(s2_cloud_mask)

    s2_img = s2_img.map(lambda img: ee_cloud_probability_mask(img, 0.7, True))
    s2_img = s2_img.map(lambda img: ee_map_valid_pixels(img, ee_geometry, 10)).filter(
        ee.Filter.gte("ZZ_USER_VALID_PIXELS", 20)
    )

    s2_img = (
        s2_img.map(
            lambda img: img.set("ZZ_USER_TIME_DUMMY", img.date().format("YYYY-MM-dd"))
        )
        .sort("ZZ_USER_TIME_DUMMY")
        .distinct("ZZ_USER_TIME_DUMMY")
    )

    def get_max_evi2(img):
        img = ee.Image(img)
        median_evi2 = img.reduceRegion(
            reducer=ee.Reducer.median(),
            geometry=ee_geometry,
            scale=10,
            maxPixels=1e8,
            bestEffort=True,
        ).get("EVI2")
        return img.set("median_evi2", median_evi2)

    s2_img = s2_img.map(get_max_evi2)
    max_evi2_img = s2_img.sort("median_evi2", False).first()

    if max_evi2_img is None:
        return Failure("No images available")

    def calculate_std_dev(img):
        img = ee.Image(img)
        stats = img.reduceRegion(
            reducer=ee.Reducer.stdDev(),
            geometry=ee_geometry,
            scale=10,
            maxPixels=1e8,
            bestEffort=True,
        )
        doy = ee.Number(img.date().getRelative("day", "year").add(1))

        return ee.List([stats.get("EVI2"), img.get("median_evi2"), doy])

    try:
        score = fix_gee_compute_values(
            ee.data.computeValue(calculate_std_dev(max_evi2_img)), np.float16
        )
        return Success((score))
    except Exception as e:
        return Failure(str(e))
