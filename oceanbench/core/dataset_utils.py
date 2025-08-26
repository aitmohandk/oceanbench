# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from datetime import datetime
from enum import Enum
from typing import Optional
import numpy as np
import xarray

from oceanbench.core.climate_forecast_standard_names import (
    StandardDimension,
    StandardVariable,
)


class Variable(Enum):
    SEA_SURFACE_HEIGHT_ABOVE_GEOID = StandardVariable.SEA_SURFACE_HEIGHT_ABOVE_GEOID
    SEA_WATER_POTENTIAL_TEMPERATURE = StandardVariable.SEA_WATER_POTENTIAL_TEMPERATURE
    SEA_WATER_SALINITY = StandardVariable.SEA_WATER_SALINITY
    NORTHWARD_SEA_WATER_VELOCITY = StandardVariable.NORTHWARD_SEA_WATER_VELOCITY
    EASTWARD_SEA_WATER_VELOCITY = StandardVariable.EASTWARD_SEA_WATER_VELOCITY
    UPWARD_SEA_WATER_VELOCITY = StandardVariable.UPWARD_SEA_WATER_VELOCITY
    MIXED_LAYER_THICKNESS = StandardVariable.MIXED_LAYER_THICKNESS
    GEOSTROPHIC_NORTHWARD_SEA_WATER_VELOCITY = StandardVariable.GEOSTROPHIC_NORTHWARD_SEA_WATER_VELOCITY
    GEOSTROPHIC_EASTWARD_SEA_WATER_VELOCITY = StandardVariable.GEOSTROPHIC_EASTWARD_SEA_WATER_VELOCITY
    MEAN_DYNAMIC_TOPOGRAPHY = StandardVariable.MEAN_DYNAMIC_TOPOGRAPHY
    SEA_SURFACE_HEIGHT_ABOVE_SEA_LEVEL = StandardVariable.SEA_SURFACE_HEIGHT_ABOVE_SEA_LEVEL
    SEA_SURFACE_TEMPERATURE = StandardVariable.SEA_SURFACE_TEMPERATURE
    SEA_SURFACE_SALINITY = StandardVariable.SEA_SURFACE_SALINITY
    def key(self) -> str:
        return self.value.value

    def variable_name_from_dataset(self, dataset: xarray.Dataset) -> str:
        return (
            self.value.variable_name_from_dataset_standard_names(dataset)
            if isinstance(self.value, StandardVariable)
            else self.value
        )


class Dimension(Enum):
    DEPTH = StandardDimension.DEPTH
    TIME = StandardDimension.TIME
    LATITUDE = StandardDimension.LATITUDE
    LONGITUDE = StandardDimension.LONGITUDE

    def dimension_name_from_dataset(self, dataset: xarray.Dataset) -> Optional[str]:
        return self.value.dimension_name_from_dataset_standard_names(dataset)


class DepthLevel(Enum):
    SURFACE = 4.940250e-01
    MINUS_50_METERS = 4.737369e01
    MINUS_200_METERS = 2.224752e02
    MINUS_550_METERS = 5.410889e02


def get_variable(dataset: xarray.Dataset, variable: Variable) -> xarray.DataArray:
    return dataset[variable.variable_name_from_dataset(dataset)]


def get_dimension(dataset: xarray.Dataset, dimension: Dimension) -> xarray.DataArray:
    return dataset[dimension.dimension_name_from_dataset(dataset)]



def select_variable_day_and_depth(
    dataset: xarray.Dataset,
    variable: Variable,
    depth_level: DepthLevel,
    lead_day: int,
) -> xarray.DataArray:
    depth_name = StandardDimension.DEPTH.dimension_name_from_dataset_standard_names(dataset)
    time_name = StandardDimension.TIME.dimension_name_from_dataset_standard_names(dataset)
    try:
        var_data = get_variable(dataset, variable)
        
        # Vérifier si la dimension time existe et contient des valeurs
        if time_name and time_name in var_data.dims and var_data.sizes.get(time_name, 0) > 0:
            # Vérifier que lead_day est dans les limites
            if lead_day < var_data.sizes[time_name]:
                new_dataset = var_data.isel({time_name: lead_day})
            else:
                # lead_day trop grand, prendre le dernier temps disponible
                new_dataset = var_data.isel({time_name: -1})
        else:
            # Pas de dimension time ou dimension vide, retourner la variable telle quelle
            new_dataset = var_data
        
        # Appliquer la sélection de profondeur si applicable
        if depth_name and depth_name in new_dataset.coords:
            new_dataset = new_dataset.sel({depth_name: depth_level.value})
            
        return new_dataset
    except Exception as exception:
        try:
            start_datetime = datetime.fromisoformat(str(get_variable(dataset, variable)[0].values))
        except:
            start_datetime = "unknown"
            
        details = (
            f"start_datetime={start_datetime}, variable={variable.value},"
            + f" depth={depth_level.value}, lead_day={lead_day}"
        )
        raise Exception(f"Could not select data: {details}") from exception


def select_variable_day(
    dataset: xarray.Dataset,
    variable: Variable,
    lead_day: int,
) -> xarray.DataArray:
    time_name = StandardDimension.TIME.dimension_name_from_dataset_standard_names(dataset)
    try:
        var_data = get_variable(dataset, variable)
        
        # Vérifier si la dimension time existe et contient des valeurs
        if time_name and time_name in var_data.dims and var_data.sizes.get(time_name, 0) > 0:
            # Vérifier que lead_day est dans les limites
            if lead_day < var_data.sizes[time_name]:
                new_dataset = var_data.isel({time_name: lead_day})
            else:
                # lead_day trop grand, prendre le dernier temps disponible
                new_dataset = var_data.isel({time_name: -1})
        else:
            # Pas de dimension time ou dimension vide, retourner la variable telle quelle
            new_dataset = var_data
            
        return new_dataset
    except Exception as exception:
        # Essayer d'extraire les informations pour le message d'erreur
        try:
            start_datetime = datetime.fromisoformat(str(get_variable(dataset, variable)[0].values))
        except:
            start_datetime = "unknown"
            
        details = (
            f"start_datetime={start_datetime}, variable={variable.value},"
            + f" lead_day={lead_day}"
        )
        raise Exception(f"Could not select data: {details}") from exception


def get_length(obj):
    """Obtient la longueur en gérant différents cas"""
    if obj is None:
        return 0
    elif isinstance(obj, (list, tuple, str, dict, set)):
        return len(obj)
    elif isinstance(obj, np.ndarray):
        return obj.size if obj.ndim > 0 else 1
    elif isinstance(obj, (xarray.Dataset, xarray.DataArray)):
        # Gestion spécifique pour xarray
        try:
            return len(obj)
        except TypeError:
            # Si c'est un scalaire ou 0-dimensionnel
            return 1 if obj.ndim == 0 else obj.size
    elif hasattr(obj, '__len__'):
        try:
            return len(obj)
        except TypeError:
            # L'objet a __len__ mais len() échoue (ex: scalaire NumPy, etc.)
            return 1
    else:
        # Scalaire ou objet sans taille
        return 1
