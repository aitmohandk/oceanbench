# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from datetime import datetime
from enum import Enum
from typing import Optional
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
        new_dataset = get_variable(dataset, variable).isel({time_name: lead_day})
        return (
            new_dataset.sel({depth_name: depth_level.value})
            if depth_name in get_variable(dataset, variable).coords
            else new_dataset
        )
    except Exception as exception:
        start_datetime = datetime.fromisoformat(str(get_variable(dataset, variable)[0].values))
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
        new_dataset = get_variable(dataset, variable).isel({time_name: lead_day})
        return new_dataset
    except Exception as exception:
        start_datetime = datetime.fromisoformat(str(get_variable(dataset, variable)[0].values))
        details = (
            f"start_datetime={start_datetime}, variable={variable.value},"
            + f" lead_day={lead_day}"
        )
        raise Exception(f"Could not select data: {details}") from exception
