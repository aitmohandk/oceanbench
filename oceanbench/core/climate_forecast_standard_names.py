# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from enum import Enum
# from huggingface_hub import DatasetCard
from scipy import datasets
import xarray
from typing import Dict, Optional
from loguru import logger


class StandardDimension(Enum):
    DEPTH = "depth"
    TIME = "time"
    LATITUDE = "latitude"
    LONGITUDE = "longitude"

    def dimension_name_from_dataset_standard_names(self, dataset: xarray.Dataset) -> Optional[str]:
        return _get_variable_name_from_standard_name(dataset, self.value)


class StandardVariable(Enum):
    SEA_SURFACE_HEIGHT_ABOVE_GEOID = "sea_surface_height_above_geoid"
    SEA_WATER_POTENTIAL_TEMPERATURE = "sea_water_potential_temperature"
    SEA_WATER_SALINITY = "sea_water_salinity"
    NORTHWARD_SEA_WATER_VELOCITY = "northward_sea_water_velocity"
    EASTWARD_SEA_WATER_VELOCITY = "eastward_sea_water_velocity"
    UPWARD_SEA_WATER_VELOCITY = "upward_sea_water_velocity"
    MIXED_LAYER_THICKNESS = "ocean_mixed_layer_thickness"
    GEOSTROPHIC_NORTHWARD_SEA_WATER_VELOCITY = "geostrophic_northward_sea_water_velocity"
    GEOSTROPHIC_EASTWARD_SEA_WATER_VELOCITY = "geostrophic_eastward_sea_water_velocity"
    MEAN_DYNAMIC_TOPOGRAPHY = "mean_dynamic_topography"
    SEA_SURFACE_HEIGHT_ABOVE_SEA_LEVEL = "sea_surface_height_above_sea_level"
    SEA_SURFACE_TEMPERATURE = "sea_surface_temperature"
    SEA_SURFACE_SALINITY = "sea_surface_salinity"

    def variable_name_from_dataset_standard_names(
        self, dataset: xarray.Dataset,
    ) -> str:
        return _get_variable_name_from_standard_name(dataset, self.value)


def _get_variable_name_from_standard_name(dataset: xarray.Dataset | xarray.DataArray, standard_name: str) -> Optional[str]:
    """
    Retourne le nom de la variable correspondant au standard_name dans un Dataset ou DataArray.
    """
    standard_name = standard_name.lower()
    if isinstance(dataset, xarray.Dataset):
        # Parcourir toutes les variables du Dataset
        for variable_name in dataset.data_vars:
            var = dataset[variable_name]
            var_std_name = var.attrs.get("standard_name", '').lower()
            if not var_std_name:
                var_std_name = var.attrs.get("std_name", '').lower()
            if var_std_name == standard_name:
                return str(variable_name)
        # Si aucune variable ne correspond
        return None
    elif isinstance(dataset, xarray.DataArray):
        var_std_name = dataset.attrs.get("standard_name", '').lower()
        if not var_std_name:
            var_std_name = dataset.attrs.get("std_name", '').lower()
        if var_std_name == standard_name:
            # Pour un DataArray, le nom est dans .name
            return str(dataset.name)
        return None
    else:
        raise TypeError(f"Expected xarray.Dataset or xarray.DataArray, got {type(dataset)}")