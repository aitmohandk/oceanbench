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


# ── per-lat-bin helpers ────────────────────────────────────────────────────────
# ``bin_resolution`` parameter threaded from the YAML config.
_DEFAULT_BIN_RESOLUTION = 5

_LAT_BIN_EDGES = list(range(-90, 91, _DEFAULT_BIN_RESOLUTION))
_LON_BIN_EDGES = list(range(-180, 181, _DEFAULT_BIN_RESOLUTION))


def _make_bin_edges(resolution: int = _DEFAULT_BIN_RESOLUTION):
    """Return (lat_edges, lon_edges) lists for a given resolution in degrees."""
    return list(range(-90, 91, resolution)), list(range(-180, 181, resolution))


def _lat_bin_label(lat_min: float, lat_max: float) -> str:
    """Return a human-readable latitude band label, e.g. '10S-0' or '0-10N'."""
    def _side(v: int) -> str:
        if v < 0:
            return f"{abs(v)}S"
        elif v > 0:
            return f"{v}N"
        return "0"
    return f"{_side(int(lat_min))}-{_side(int(lat_max))}"


def _lon_bin_label(lon_min: float, lon_max: float) -> str:
    """Return a human-readable longitude band label, e.g. '30W-0' or '0-30E'."""
    def _side(v: int) -> str:
        if v < 0:
            return f"{abs(v)}W"
        elif v > 0:
            return f"{v}E"
        return "0"
    return f"{_side(int(lon_min))}-{_side(int(lon_max))}"


def _rmsd_per_lat_bin(challenger_da, reference_da, bin_resolution=None) -> list:
    """Compute RMSD per crossed (lat x lon) bin between two DataArrays.

    Fully vectorized O(N) implementation using np.bincount — no nested loops.

    Parameters
    ----------
    bin_resolution : int or None
        Spatial resolution in degrees.  ``None`` falls back to
        ``_DEFAULT_BIN_RESOLUTION`` (set from YAML config).

    Returns a list of dicts:
        [{"lat_bin": str, "lon_bin": str, "rmsd": float | None, "n_points": int}, ...]
    """
    if bin_resolution is None:
        bin_resolution = _DEFAULT_BIN_RESOLUTION
    lat_edges, lon_edges = _make_bin_edges(bin_resolution)
    # 1. Convert to numpy once
    chall_vals = np.asarray(challenger_da.values, dtype=np.float64)
    ref_vals = np.asarray(reference_da.values, dtype=np.float64)

    # 2. Find the latitude coordinate
    lat_coord = None
    for name in ("latitude", "lat", "y"):
        if hasattr(challenger_da, "coords") and name in challenger_da.coords:
            lat_coord = name
            break

    # 2b. Find the longitude coordinate
    lon_coord = None
    for name in ("longitude", "lon", "x"):
        if hasattr(challenger_da, "coords") and name in challenger_da.coords:
            lon_coord = name
            break

    if lat_coord is None:
        # No spatial coord — single global entry
        valid = ~(np.isnan(chall_vals) | np.isnan(ref_vals))
        n = int(valid.sum())
        if n == 0:
            return [{"lat_bin": "global", "lon_bin": "global", "rmsd": None, "n_points": 0}]
        diff = chall_vals.ravel()[valid.ravel()] - ref_vals.ravel()[valid.ravel()]
        return [{"lat_bin": "global", "lon_bin": "global", "rmsd": float(np.sqrt(np.mean(diff * diff))), "n_points": n}]

    # 3. Broadcast lat to the full data shape
    lat_1d = challenger_da.coords[lat_coord].values
    dims = list(challenger_da.dims)
    lat_dim_idx = dims.index(lat_coord)
    shape = chall_vals.shape

    reshape_lat = [1] * len(shape)
    reshape_lat[lat_dim_idx] = len(lat_1d)
    lat_bcast = np.broadcast_to(lat_1d.reshape(reshape_lat), shape)

    # 3b. Broadcast lon to the full data shape
    has_lon = lon_coord is not None and lon_coord in dims
    if has_lon:
        lon_1d = challenger_da.coords[lon_coord].values
        lon_dim_idx = dims.index(lon_coord)
        reshape_lon = [1] * len(shape)
        reshape_lon[lon_dim_idx] = len(lon_1d)
        lon_bcast = np.broadcast_to(lon_1d.reshape(reshape_lon), shape)

    # Flatten everything
    lat_flat = lat_bcast.ravel()
    chall_flat = chall_vals.ravel()
    ref_flat = ref_vals.ravel()
    valid = ~(np.isnan(chall_flat) | np.isnan(ref_flat))

    # 4. Assign bin indices (np.digitize, single pass)
    _lat_edges_arr = np.asarray(lat_edges, dtype=np.float64)
    n_lat_bins = len(_lat_edges_arr) - 1
    lat_bin_idx = np.digitize(lat_flat, _lat_edges_arr) - 1
    lat_bin_idx = np.clip(lat_bin_idx, 0, n_lat_bins - 1)

    if has_lon:
        _lon_edges_arr = np.asarray(lon_edges, dtype=np.float64)
        n_lon_bins = len(_lon_edges_arr) - 1
        lon_flat = lon_bcast.ravel()
        lon_bin_idx = np.digitize(lon_flat, _lon_edges_arr) - 1
        lon_bin_idx = np.clip(lon_bin_idx, 0, n_lon_bins - 1)
    else:
        n_lon_bins = 1
        lon_bin_idx = np.zeros_like(lat_bin_idx)

    # 5. Vectorised RMSD per bin via np.bincount (O(N))
    # Combined 1-D bin index for the (lat, lon) grid
    combined_idx = lat_bin_idx * n_lon_bins + lon_bin_idx
    n_total_bins = n_lat_bins * n_lon_bins

    # Restrict to valid (non-NaN) points
    v_idx = combined_idx[valid]
    v_diff_sq = (chall_flat[valid] - ref_flat[valid]) ** 2

    counts = np.bincount(v_idx, minlength=n_total_bins).astype(np.int64)
    sum_sq = np.bincount(v_idx, weights=v_diff_sq, minlength=n_total_bins)

    # Build output only for populated bins
    populated = np.nonzero(counts[:n_total_bins])[0]

    bins = []
    for flat_idx in populated:
        i_lat = int(flat_idx // n_lon_bins)
        i_lon = int(flat_idx % n_lon_bins)
        c = int(counts[flat_idx])
        rmsd_val = float(np.sqrt(sum_sq[flat_idx] / c))
        lat_min = lat_edges[i_lat]
        lat_max = lat_edges[i_lat + 1]
        if has_lon:
            lon_min = lon_edges[i_lon]
            lon_max = lon_edges[i_lon + 1]
            lon_label = _lon_bin_label(lon_min, lon_max)
        else:
            lon_label = "global"
        bins.append({
            "lat_bin": _lat_bin_label(lat_min, lat_max),
            "lon_bin": lon_label,
            "rmsd": rmsd_val,
            "n_points": c,
        })

    return bins


def _get_per_bins(challenger_dataset, reference_dataset, variable, depth_level, lead_day=0, bin_resolution=None):
    """Extract DataArrays for (variable, depth_level, lead_day) and compute per-lat-bin RMSD."""
    if depth_level:
        challenger_da = select_variable_day_and_depth(
            challenger_dataset, variable, depth_level, lead_day
        )
        reference_da = select_variable_day_and_depth(
            reference_dataset, variable, depth_level, lead_day
        )
    else:
        challenger_da = select_variable_day(challenger_dataset, variable, lead_day)
        reference_da = select_variable_day(reference_dataset, variable, lead_day)

    bins = _rmsd_per_lat_bin(challenger_da, reference_da, bin_resolution=bin_resolution)
    del challenger_da, reference_da
    gc.collect()
    return bins


def _get_rmsd_and_per_bins(challenger_dataset, reference_dataset, variable, depth_level, lead_day=0, bin_resolution=None):
    """Select DataArrays once and compute both RMSD value and per_bins in a single data load.

    Calling .compute() before any .values access ensures that dask arrays are
    materialised exactly once.  Without this, _rmsd() and _rmsd_per_lat_bin()
    would each trigger an independent dask graph traversal, doubling peak RAM.
    """
    if depth_level:
        chall_da = select_variable_day_and_depth(
            challenger_dataset, variable, depth_level, lead_day
        )
        ref_da = select_variable_day_and_depth(
            reference_dataset, variable, depth_level, lead_day
        )
    else:
        chall_da = select_variable_day(challenger_dataset, variable, lead_day)
        ref_da = select_variable_day(reference_dataset, variable, lead_day)

    # Materialise dask arrays ONCE so that _rmsd() and _rmsd_per_lat_bin()
    # operate on in-memory DataArrays without re-triggering the dask graph.
    try:
        chall_da = chall_da.compute()
    except AttributeError:
        pass  # already numpy-backed
    try:
        ref_da = ref_da.compute()
    except AttributeError:
        pass

    rmsd_value = _rmsd(chall_da, ref_da)
    per_bins = _rmsd_per_lat_bin(chall_da, ref_da, bin_resolution=bin_resolution)
    del chall_da, ref_da
    gc.collect()
    return rmsd_value, per_bins


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
    bin_resolution: Optional[int] = None,
) -> dict:
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
    all_combinations = _variable_and_depth_combinations(
        reference_datasets[0],
        challenger_datasets[0],
        variables,
        depth_levels,
    )

    scores = {}
    per_bins_by_var = {}
    for (variable, depth_level) in all_combinations:
        _label = _variale_depth_label(challenger_datasets[0], variable, depth_level)
        _LEAD_DAYS = get_lead_days_count(challenger_datasets[0])
        # Dataset[0], lead_day=0: load data once for both RMSD and per_bins
        try:
            _rmsd_ds0, _p_bins = _get_rmsd_and_per_bins(
                challenger_datasets[0], reference_datasets[0],
                variable, depth_level, lead_day=0,
                bin_resolution=bin_resolution,
            )
        except Exception as _pb_exc:
            logger.warning(f"per_bins computation failed for {_label}: {_pb_exc}")
            _rmsd_ds0 = _get_rmsd(
                challenger_datasets[0], reference_datasets[0], variable, depth_level, 0
            )
            _p_bins = []
        per_bins_by_var[_label] = _p_bins
        # Build RMSD across all datasets / lead days
        _rmsd_by_lead: list = [[] for _ in range(_LEAD_DAYS)]
        _rmsd_by_lead[0].append(_rmsd_ds0)
        for _ds_idx in range(1, len(challenger_datasets)):
            _rmsd_by_lead[0].append(
                _get_rmsd(
                    challenger_datasets[_ds_idx], reference_datasets[_ds_idx],
                    variable, depth_level, 0,
                )
            )
        for _lead_day in range(1, _LEAD_DAYS):
            for _ds_idx in range(len(challenger_datasets)):
                _rmsd_by_lead[_lead_day].append(
                    _get_rmsd(
                        challenger_datasets[_ds_idx], reference_datasets[_ds_idx],
                        variable, depth_level, _lead_day,
                    )
                )
        scores[_label] = [float(np.nanmean(vals)) for vals in _rmsd_by_lead]
    # return scores
    LEAD_DAYS_COUNT = get_lead_days_count(challenger_datasets[0])
    score_dataframe = pandas.DataFrame(scores)
    score_dataframe.index = lead_day_labels(1, LEAD_DAYS_COUNT)
    # print(score_dataframe.to_markdown())
    score_dataframe = score_dataframe.T
    # log_memory("End rmsd")
    return {"results": score_dataframe, "per_bins": per_bins_by_var}