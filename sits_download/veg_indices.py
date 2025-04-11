import numpy as np


def get_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    return (nir - red) / (nir + red)


def get_gndvi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    return (nir - green) / (nir + green)


def get_ndwi(nir: np.ndarray, swir: np.ndarray) -> np.ndarray:
    return (nir - swir) / (nir + swir)


def get_savi(red: np.ndarray, nir: np.ndarray, L: float = 0.5) -> np.ndarray:
    return ((nir - red) / (nir + red + L)) * (1 + L)


def get_evi(blue: np.ndarray, red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    return 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)


def get_evi2(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    return 2.5 * (nir - red) / (nir + 2.4 * red + 1)


def get_msavi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    return (2 * nir + 1 - np.sqrt((2 * nir + 1) ** 2 - 8 * (nir - red))) / 2


def get_ndre(red_edge: np.ndarray, nir: np.ndarray) -> np.ndarray:
    return (nir - red_edge) / (nir + red_edge)


def get_mcari(green: np.ndarray, red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    return ((nir - red) - 0.2 * (nir - green)) * (nir / red)


def get_gci(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    return (nir / green) - 1


def get_bsi(
    blue: np.ndarray, red: np.ndarray, nir: np.ndarray, swir: np.ndarray
) -> np.ndarray:
    return ((swir + red) - (nir + blue)) / ((swir + red) + (nir + blue))


def get_ci_red(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    return (nir / red) - 1


def get_ci_green(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    return (nir / green) - 1


def get_osavi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    return (nir - red) / (nir + red + 0.16)


def get_arvi(blue: np.ndarray, red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    return (nir - (2 * red - blue)) / (nir + (2 * red - blue))


def get_vhvv(vv: np.ndarray, vh: np.ndarray) -> np.ndarray:
    return vh / vv
