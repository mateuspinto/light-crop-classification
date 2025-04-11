import ee
import numpy as np
from typing import List


def download_ee_img_as_numpy(
    ee_img: ee.Image, ee_geometry: ee.Geometry, scale: int
) -> np.ndarray:
    ee_img = ee.Image(ee_img)
    ee_geometry = ee.Geometry(ee_geometry).bounds()

    projection = ee.Projection("EPSG:4326").atScale(scale).getInfo()
    chip_size = round(ee_geometry.perimeter(0.1).getInfo() / (4 * scale))

    scale_y = -projection["transform"][0]
    scale_x = projection["transform"][4]

    list_of_coordinates: List = ee.Array.cat(ee_geometry.coordinates(), 1).getInfo()

    x_min = list_of_coordinates[0][0]
    y_max = list_of_coordinates[0][1]
    coordinates = [x_min, y_max]

    chip_size = 1 if chip_size == 0 else chip_size

    img_in_bytes = ee.data.computePixels(
        {
            "expression": ee_img,
            "fileFormat": "NUMPY_NDARRAY",
            "grid": {
                "dimensions": {"width": chip_size, "height": chip_size},
                "affineTransform": {
                    "scaleX": scale_x,
                    "scaleY": scale_y,
                    "translateX": coordinates[0],
                    "translateY": coordinates[1],
                },
                "crsCode": projection["crs"],
            },
        }
    )

    img_in_array = np.array(img_in_bytes.tolist()).astype(np.float32)
    img_in_array[np.isinf(img_in_array)] = 0
    img_in_array[np.isnan(img_in_array)] = 0

    return img_in_array
