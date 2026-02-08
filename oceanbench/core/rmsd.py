# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from functools import partial
import os
from typing import List, Optional

from loguru import logger
import gc
import os

import numpy as np
import pandas
import psutil
import xarray

def log_mem_state(context):
    process = psutil.Process(os.getpid())
    logger.info(f"[MEM][rmsd] {context} | RAM used: {psutil.virtual_memory().used/1e9:.2f} GB | Process: {process.memory_info().rss/1e6:.2f} MB | Open files: {len(process.open_files())}")

# Lightweight, safe import of memory_profiler's decorator (no-op fallback)
try:  # pragma: no cover - profiling aid only
    from memory_profiler import profile  # type: ignore
except Exception:  # pragma: no cover
    def profile(func):
        return func

from oceanbench.core.distributed import DatasetProcessor
from oceanbench.core.dataset_utils import (
    Variable,
    Dimension,
    DepthLevel,
    get_variable,
    select_variable_day_and_depth,
    select_variable_day,
)
from oceanbench.core.lead_day_utils import lead_day_labels


VARIABLE_LABELS: dict[str, str] = {
    Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID: "surface height",
    Variable.SEA_WATER_POTENTIAL_TEMPERATURE: "temperature",
    Variable.SEA_WATER_SALINITY: "salinity",
    Variable.NORTHWARD_SEA_WATER_VELOCITY: "northward velocity",
    Variable.EASTWARD_SEA_WATER_VELOCITY: "eastward velocity",
    Variable.MIXED_LAYER_THICKNESS: "mixed layer thickness",
    Variable.GEOSTROPHIC_NORTHWARD_SEA_WATER_VELOCITY: "northward geostrophic velocity",
    Variable.GEOSTROPHIC_EASTWARD_SEA_WATER_VELOCITY: "eastward geostrophic velocity",
    Variable.UPWARD_SEA_WATER_VELOCITY: "upward velocity",
    Variable.MEAN_DYNAMIC_TOPOGRAPHY: "mean dynamic topography",
    Variable.SEA_SURFACE_HEIGHT_ABOVE_SEA_LEVEL: "height above sea level",
    Variable.SEA_SURFACE_TEMPERATURE: "sea surface temperature",
    Variable.SEA_SURFACE_SALINITY: "sea surface salinity",
}


DEPTH_LABELS: dict[DepthLevel, str] = {
    DepthLevel.SURFACE: "surface",
    DepthLevel.MINUS_50_METERS: "50m",
    DepthLevel.MINUS_200_METERS: "200m",
    DepthLevel.MINUS_550_METERS: "550m",
}


#@profile
def _rmsd(data, reference_data):
    """
    Compute RMSD between two arrays, avoiding NaNs.
    Version optimized for performance.
    Args:
        data: array-like (DataArray, Dataset, ndarray, dask array, etc.)
        reference_data: array-like (DataArray, Dataset, ndarray, dask array, etc.)
    Returns:
        float: RMSD value or np.nan if no valid data
    """

    # Direct conversion based on type
    def extract_values(obj):
        """Extract numpy values from an object in an optimized way."""
        obj_type = type(obj).__name__
        
        if obj_type in ('DataArray', 'Dataset'):
            return obj.values
        elif obj_type in ('dask.array.core.Array',):
            return obj.compute()
        elif obj_type in ('ndarray',):
            return obj
        else:
            # Fallback for other types
            return np.asarray(obj)
    
    # Extraction rapide des valeurs
    data_vals = extract_values(data)
    ref_vals = extract_values(reference_data)
    
    # Conversion en numpy float64
    data_flat = np.asarray(data_vals, dtype=np.float64).flatten()
    ref_flat = np.asarray(ref_vals, dtype=np.float64).flatten()
    
    # Equalize sizes
    min_size = min(data_flat.size, ref_flat.size)
    if data_flat.size != ref_flat.size:
        data_flat = data_flat[:min_size]
        ref_flat = ref_flat[:min_size]
    
    # Vectorized mask (faster than two separate conditions)
    valid = ~(np.isnan(data_flat) | np.isnan(ref_flat))
    
    if not valid.any():
        return np.nan
    
    # Vectorized RMSD
    diff = data_flat[valid] - ref_flat[valid]
    rmsd_value = np.sqrt(np.mean(diff * diff))  # Plus rapide que diff**2
    
    # CRITICAL: Force cleanup of large arrays to prevent memory leaks
    del data_flat, ref_flat, diff, valid, data_vals, ref_vals
    gc.collect()  # Force garbage collection immediately
    
    return rmsd_value



#@profile
def _get_rmsd(challenger_dataset, reference_dataset, variable, depth_level, lead_day):
    """
    Compute RMSD between two datasets for a given variable, depth level and forecast day.
    Uses an optimized RMSD version.
    Args:
        challenger_dataset: Challenger xarray Dataset
        reference_dataset: Reference xarray Dataset
        variable: Variable to evaluate
        depth_level: Depth level (or None for surface)
        lead_day: Forecast day (int)
    Returns:
        float: RMSD value
    """
    if depth_level:
        challenger_dataarray = select_variable_day_and_depth(challenger_dataset, variable, depth_level, lead_day)
        reference_dataarray = select_variable_day_and_depth(reference_dataset, variable, depth_level, lead_day)
    else:
        challenger_dataarray = select_variable_day(challenger_dataset, variable, lead_day)
        reference_dataarray = select_variable_day(reference_dataset, variable, lead_day)
    
    # No .compute() here - let _rmsd_optimized handle it
    rmsd_value = _rmsd(challenger_dataarray, reference_dataarray)
    
    # CRITICAL: Force cleanup of DataArrays
    del challenger_dataarray, reference_dataarray
    gc.collect()  # Force garbage collection immediately
    
    return rmsd_value


def get_lead_days_count(dataset: xarray.Dataset) -> int:
    # always 1 day long in the current datasets
    # forecasts are managed in dc-tools library
    return 1

#@profile
def _get_rmsd_for_all_lead_days(
    dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
    variable: Variable,
    depth_level: DepthLevel,
) -> list[float]:
    LEAD_DAYS_COUNT = get_lead_days_count(dataset)
    return list(
        map(
            partial(
                _get_rmsd,
                dataset,
                reference_dataset,
                variable,
                depth_level,
            ),
            range(LEAD_DAYS_COUNT),
        )
    )


#@profile
def _compute_rmsd(
    datasets: List[xarray.Dataset],
    reference_datasets: List[xarray.Dataset],
    variable: Variable,
    depth_level: DepthLevel,
) -> np.ndarray:

    all_rmsd = np.array(
        list(
            map(
                partial(
                    _get_rmsd_for_all_lead_days,
                    variable=variable,
                    depth_level=depth_level,
                ),
                datasets,
                reference_datasets,
            )
        )
    )
    try:
        result = all_rmsd.mean(axis=0)
    finally:
        # free large temporary array asap
        try:
            del all_rmsd
        except Exception:
            pass
        gc.collect()
    return result


def _variale_depth_label(dataset: xarray.Dataset, variable: Variable, depth_level: DepthLevel) -> str:
    if depth_level:
        return (
            f"{DEPTH_LABELS[depth_level]} {VARIABLE_LABELS[variable]}".capitalize()
            if _has_depths(dataset, variable)
            else f"surface {VARIABLE_LABELS[variable]}".capitalize()
            # else f"{DepthLevel.SURFACE} {VARIABLE_LABELS[variable]}"
        ).capitalize()
    else:
        # return f"{DepthLevel.SURFACE} {VARIABLE_LABELS[variable]}".capitalize()
        return f"{VARIABLE_LABELS[variable]}".capitalize()


def _has_depths_legacy(dataset: xarray.Dataset, variable: Variable) -> bool:
    if Dimension.DEPTH.dimension_name_from_dataset(get_variable(dataset, variable)) is None:
        return False
    else:
        return Dimension.DEPTH.dimension_name_from_dataset(dataset) in get_variable(dataset, variable).coords


def _has_depths(dataset: xarray.Dataset, variable: Variable) -> bool:
    """
    Check if a variable has a depth dimension.
    
    Args:
        dataset: xarray Dataset
        variable: Variable to test
        
    Returns:
        bool: True if the variable has a depth dimension
    """
    try:
        # Get the variable
        var_data = get_variable(dataset, variable)
        
        # List of possible names for the depth dimension
        depth_names = ['depth', 'z', 'lev', 'level', 'deptht', 'bottom']
        
        # Check if a depth dimension exists in the variable
        var_dims = list(var_data.dims)
        has_depth_dim = any(depth_name in var_dims for depth_name in depth_names)
        
        # Also check in the variable's coordinates
        var_coords = list(var_data.coords)
        has_depth_coord = any(depth_name in var_coords for depth_name in depth_names)
        
        return has_depth_dim or has_depth_coord
        
    except Exception as e:
        logger.debug(f"Error checking depth dimension for variable {variable}: {e}")
        return False

def _is_surface(depth_level: DepthLevel) -> bool:
    return depth_level == DepthLevel.SURFACE


def _variable_and_depth_combinations(
    ref_dataset: xarray.Dataset, 
    challenger_dataset: xarray.Dataset, 
    variables: list[Variable],
    depth_levels: Optional[List[DepthLevel]],
    depth_dim: str = 'depth',
) -> list[tuple[Variable, DepthLevel]]:
    """
    Generate all valid (variable, depth_level) combinations for a dataset.
    
    Args:
        ref_dataset: Reference xarray Dataset
        challenger_dataset: Challenger xarray Dataset
        variables: List of variables to evaluate
        depth_levels: List of depth levels (can be None)
    
    Returns:
        List of tuples (Variable, DepthLevel or None)
    """
    list_combs = []
    
    def _depth_level_exists_in_dataset(dataset: xarray.Dataset, variable: Variable, depth_level: DepthLevel) -> bool:
        """Check if a depth_level exists in the depth dimension of a variable in a dataset."""
        try:
            get_variable(dataset, variable)  # Validate variable exists

            # Check if the depth_level value exists in the coordinates
            depth_values = dataset[depth_dim].values
            depth_level_value = depth_level.value  # Supposant que DepthLevel a un attribut .value
            
            # Tolerance for floating-point comparisons
            tolerance = 1e-3
            return any(abs(float(dv) - float(depth_level_value)) < tolerance for dv in depth_values)
            
        except Exception as e:
            logger.debug(f"Error checking depth level {depth_level} for variable {variable}: {e}")
            return False
    
    if depth_levels is not None:
        # If depth levels are specified
        for variable in variables:
            if _has_depths(ref_dataset, variable) and _has_depths(challenger_dataset, variable):
                # Variable with depth: check that each depth_level exists in both datasets
                for depth_level in depth_levels:
                    if (_depth_level_exists_in_dataset(ref_dataset, variable, depth_level) and 
                        _depth_level_exists_in_dataset(challenger_dataset, variable, depth_level)):
                        list_combs.append((variable, depth_level))
            else:
                # Variable without depth: use None as depth_level
                list_combs.append((variable, None))
    else:
        # No depth levels specified: all variables with None
        for variable in variables:
            list_combs.append((variable, None))
    
    return list_combs


def rmsd_legacy(
    dataset_processor: DatasetProcessor,
    challenger_datasets: List[xarray.Dataset],
    reference_datasets: List[xarray.Dataset],
    variables: List[Variable],
    depth_levels: Optional[List[DepthLevel]] = DEPTH_LABELS,
) -> pandas.DataFrame:
    """Compute RMSD between challenger and reference datasets for given variables and depth levels.
    Args:
        dataset_processor: DatasetProcessor instance for distributed processing
        challenger_datasets: List of challenger datasets
        reference_datasets: List of reference datasets
        variables: List of variables to evaluate
        depth_levels: List of depth levels (or None)
    Returns:
        pandas.DataFrame: DataFrame of RMSD scores
    """

    all_combinations = _variable_and_depth_combinations(
        reference_datasets[0],
        challenger_datasets[0],
        variables,
        depth_levels,
    )

    if len(all_combinations) == 2:
        variable = all_combinations[0]
        depth_level = all_combinations[1]
        scores = {
            _variale_depth_label(challenger_datasets[0], variable, depth_level): list(
                _compute_rmsd(
                    challenger_datasets,
                    reference_datasets,
                    variable,
                    depth_level,
                )
            )
        }
    else:
        scores = {
            _variale_depth_label(challenger_datasets[0], variable, depth_level): list(
                _compute_rmsd(
                    challenger_datasets,
                    reference_datasets,
                    variable,
                    depth_level,
                )
            )
            for (variable, depth_level) in all_combinations
        }

    LEAD_DAYS_COUNT = get_lead_days_count(challenger_datasets[0])
    score_dataframe = pandas.DataFrame(scores)
    score_dataframe.index = lead_day_labels(1, LEAD_DAYS_COUNT)
    print(score_dataframe.to_markdown())
    return score_dataframe.T

def log_memory(fct):
    """Log memory usage of the current process."""
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1e6
    print(f"[{fct}] Memory usage: {mem_mb:.2f} MB")


#@profile
def rmsd(
    challenger_datasets: List[xarray.Dataset],
    reference_datasets: List[xarray.Dataset],
    variables: List[Variable],
    depth_levels: Optional[List[DepthLevel]] = DEPTH_LABELS,
) -> pandas.DataFrame:
    """Compute RMSD between challenger and reference datasets for given variables and depth levels.
    Args:
        dataset_processor: DatasetProcessor instance for distributed processing
        challenger_datasets: List of challenger datasets
        reference_datasets: List of reference datasets
        variables: List of variables to evaluate
        depth_levels: List of depth levels (or None)
    Returns:
        pandas.DataFrame: DataFrame of RMSD scores
    """
    # log_memory("Start rmsd")
    dataset_processor = None
    all_combinations = _variable_and_depth_combinations(
        reference_datasets[0],
        challenger_datasets[0],
        variables,
        depth_levels,
    )

    if len(all_combinations) == 2:
        variable = all_combinations[0]
        depth_level = all_combinations[1]
        
        # Submit the task to the Dask client
        if dataset_processor.client is not None:
            future = dataset_processor.client.submit(
                _compute_rmsd,
                challenger_datasets,
                reference_datasets,
                variable,
                depth_level,
            )
            rmsd_result = future.result()  # Wait for result
        else:
            # Fallback if no Dask client
            rmsd_result = _compute_rmsd(
                challenger_datasets,
                reference_datasets,
                variable,
                depth_level,
            )
        
        scores = {
            _variale_depth_label(challenger_datasets[0], variable, depth_level): list(rmsd_result)
        }
    else:
        # Parallel submission of all tasks
        if dataset_processor is not None and dataset_processor.client is not None:
            futures = []
            for (variable, depth_level) in all_combinations:
                future = dataset_processor.client.submit(
                    _compute_rmsd,
                    challenger_datasets,
                    reference_datasets,
                    variable,
                    depth_level,
                )
                futures.append((variable, depth_level, future))
            
            # Collect results
            scores = {}
            for variable, depth_level, future in futures:
                rmsd_result = future.result()  # Wait for result
                scores[_variale_depth_label(challenger_datasets[0], variable, depth_level)] = list(rmsd_result)
        else:
            # Fallback if no Dask client
            scores = {
                _variale_depth_label(challenger_datasets[0], variable, depth_level): list(
                    _compute_rmsd(
                        challenger_datasets,
                        reference_datasets,
                        variable,
                        depth_level,
                    )
                )
                for (variable, depth_level) in all_combinations
            }
    # return scores
    LEAD_DAYS_COUNT = get_lead_days_count(challenger_datasets[0])
    score_dataframe = pandas.DataFrame(scores)
    score_dataframe.index = lead_day_labels(1, LEAD_DAYS_COUNT)
    # print(score_dataframe.to_markdown())
    score_dataframe = score_dataframe.T
    # log_memory("End rmsd")
    return score_dataframe