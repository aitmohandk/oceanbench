import gc
import os
import traceback
import tracemalloc
import time

import pandas as pd
import xarray as xr
import numpy as np
from typing import Callable, Dict, List, Optional, Union, Generator

from concurrent.futures import ThreadPoolExecutor
import geopandas as gpd
from loguru import logger
import psutil
import pyinterp
import pyinterp.backends.xarray
from scipy.spatial import cKDTree
#import xesmf as xe
import xskillscore as xs
# from xskillscore import rmse, pearson_r, mae, crps_ensemble 

def log_mem_state(context):
    process = psutil.Process(os.getpid())
    logger.info(f"[MEM][class4_evaluator] {context} | RAM used: {psutil.virtual_memory().used/1e9:.2f} GB | Process: {process.memory_info().rss/1e6:.2f} MB | Open files: {len(process.open_files())}")

# Lightweight, safe import of memory_profiler's decorator (no-op fallback)
try:  # pragma: no cover - profiling aid only
    from memory_profiler import profile  # type: ignore
except Exception:  # pragma: no cover
    def profile(func):
        return func

# set of metrics from xskillscore library
# https://xskillscore.readthedocs.io/en/stable/api.html#module-xskillscore
XSKILL_METRICS = {
    "rmse": xs.rmse,
    "mae": xs.mae,
    "mse": xs.mse,
    "pearson_r": xs.pearson_r,
    "spearman_r": xs.spearman_r,
    "crps_ensemble": xs.crps_ensemble,
    "crps_gaussian": xs.crps_gaussian,
    "me": xs.me,
    "median_absolute_error": xs.median_absolute_error,
    "r2": xs.r2,
    "mape": xs.mape,
    "smape": xs.smape,
    "bias": xs.me,  # alias
}

# Default mapping: variable → QC field name + valid flags
DEFAULT_QC_MAPPING: Dict[str, Dict[str, List[int]]] = {
    "sea_surface_temperature": {"qc_variable": "quality_flag", "valid_flags": [0]},
    "sea_surface_salinity": {"qc_variable": "dqf", "valid_flags": [0, 1]},
    "sea_surface_height": {"qc_variable": "quality_level", "valid_flags": [1, 2]},
    "temperature": {"qc_variable": "qc_flag", "valid_flags": [0]},
    "salinity": {"qc_variable": "qc_flag", "valid_flags": [0]},
}


def log_memory(fct):
    """Log memory usage of the current process."""
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1e6
    print(f"[{fct}] Memory usage: {mem_mb:.2f} MB")



# ----------------------------------  SUPEROBS by model grid  -------------------------------------



#@profile
def stack_obs(ds_obs: xr.Dataset) -> xr.Dataset:
    """Ensure observations are stacked into a single n_points dimension."""
    if "n_points" not in ds_obs.dims:
        ds_obs = ds_obs.stack(n_points=tuple(d for d in ds_obs.dims if d not in ("time",)))
    return ds_obs


#@profile
def compute_grid_index(
    ds_obs: xr.Dataset,
    ds_model: xr.Dataset,
    lon_name: str = "lon",
    lat_name: str = "lat",
) -> xr.Dataset:
    """Assign grid cell indices (lat_idx, lon_idx) to observation points
    based on the nearest regular model grid cell.
    Works with either (time) or (n_points) as obs dimension.
    Args:
        ds_obs: Dataset of observations with 1D lat/lon coordinates.
        ds_model: Dataset of model with 1D lat/lon coordinates.
        lon_name: Name of the longitude coordinate.
        lat_name: Name of the latitude coordinate.
    Returns:
        xr.Dataset: Dataset with grid indices assigned.
    """
    
    # Warning: Using .values converts to numpy array and forces loading into memory
    # If arrays are very large (high-resolution satellite), this saturates RAM
    
    # Use Dask if available to stay lazy
    
    # Model (regular grid, usually small): safe to load
    lon_model = ds_model[lon_name].data
    lat_model = ds_model[lat_name].data
    
    # If dask array, compute because searchsorted needs precise edge values
    # But model grids are small (< a few MB), so this is OK
    if hasattr(lon_model, "compute"):
         lon_model = lon_model.compute()
    if hasattr(lat_model, "compute"):
         lat_model = lat_model.compute()
         
    # Obs: Potentially very large. Do not load if Dask.
    # Xarray allows using lazy arrays
    lon_obs = ds_obs[lon_name].data
    lat_obs = ds_obs[lat_name].data
    
    # Lazy method with map_blocks for searchsorted
    # Define a kernel function that takes a dask array block (obs lat/lon)
    # and returns the corresponding indices.
    
    def _searchsorted_kernel(x, bins):
        # x is a dask array chunk
        # bins is the edge array (model grid)
        idx = np.searchsorted(bins, x) - 1
        # clip to stay valid
        return np.clip(idx, 0, len(bins) - 1)

    # Create lazy indices
    if hasattr(lon_obs, "map_blocks"):
        # Dask array
        lon_idx = lon_obs.map_blocks(
            _searchsorted_kernel,
            bins=lon_model,
            dtype=np.int32
        )
        lat_idx = lat_obs.map_blocks(
            _searchsorted_kernel,
            bins=lat_model,
            dtype=np.int32
        )
    else:
        # Standard NumPy (eager fallback)
        lon_idx = np.searchsorted(lon_model, lon_obs) - 1
        lat_idx = np.searchsorted(lat_model, lat_obs) - 1
        lon_idx = np.clip(lon_idx, 0, len(lon_model) - 1)
        lat_idx = np.clip(lat_idx, 0, len(lat_model) - 1)


    # Auto-detect observation dimension
    obs_dims = ds_obs[lon_name].dims
    if len(obs_dims) != 1:
        raise ValueError(
            f"Expected 1D coordinates for obs lon/lat, got dims={obs_dims}"
        )
    obs_dim = obs_dims[0]

    return ds_obs.assign_coords(
        lon_idx=(obs_dim, lon_idx),
        lat_idx=(obs_dim, lat_idx),
    )

#@profile
def aggregate_superobs(
    subset_ds: xr.DataArray,
    var_name: str,
    reduce: str = "median",
    model: Optional[xr.Dataset] = None,
    lat_name: str = "lat",
    lon_name: str = "lon",
    min_count: int = 1,
    max_std: Optional[float] = None,
    weighting: Optional[str] = None,
) -> Optional[xr.Dataset]:
    """
    Aggregate observations into superobs per model grid cell.
    Args:
        subset_ds: DataArray of observations (with lat_idx, lon_idx coords).
        var_name: Name of the variable to aggregate.
        reduce: "mean" or "median".
        model: Model dataset to attach real coordinates.
        min_count: Minimum number of observations to keep a superobs.
        max_std: Standard deviation threshold to filter noisy superobs.
        weighting: None (default), "count" or "inv_var" (1/std²).

    Returns:
        result_ds: Dataset of aggregated superobs with diagnostics.
    """
    # Optional tracemalloc instrumentation
    _do_tracemalloc = os.environ.get("DC_TRACEMALLOC") == "1"
    if _do_tracemalloc:
        try:
            tracemalloc.start()
            _tracemalloc_before = tracemalloc.take_snapshot()
            _tracemalloc_t0 = time.time()
        except Exception:
            _do_tracemalloc = False

    # Check that index coordinates exist
    if "lat_idx" not in subset_ds.coords or "lon_idx" not in subset_ds.coords:
        raise ValueError("subset_ds must have 'lat_idx' and 'lon_idx' coordinates")
    
    lat_idx_coord = subset_ds["lat_idx"]
    lon_idx_coord = subset_ds["lon_idx"]
    
    # ------------------------------------------------------------------------
    # OPTIMIZATION: Eager loading (.values) here causes massive RAM spikes.
    # We must operate on lazy dask arrays or perform block-wise reductions.
    # ------------------------------------------------------------------------
    
    # If Dask, avoid .values.ravel() which loads everything into memory
    is_dask = hasattr(subset_ds.data, "dask") or hasattr(lat_idx_coord.data, "dask")

    if is_dask:
        # Smart fallback: For aggregation, we cannot easily do everything in pure lazy mode
        # (because GroupBy on irregular grids is complex).
        # BUT we can load only the necessary columns without unnecessary copies.
        
        # We only compute() what we need
        # tip: load() loads in place, compute() returns numpy
        lat_idx_flat = lat_idx_coord.compute().values.ravel()
        lon_idx_flat = lon_idx_coord.compute().values.ravel()
        
        # Data values
        if isinstance(subset_ds, xr.Dataset):
             data_flat = subset_ds[var_name].compute().values.ravel()
        else:
             data_flat = subset_ds.compute().values.ravel()
             
        # RAM risk still exists here during compute(), but we avoid hidden copies
        # If it still crashes, dask.dataframe.groupby should be used
        
    else:
        # Eager execution (original logic)
        lat_idx_flat = lat_idx_coord.values.ravel()
        lon_idx_flat = lon_idx_coord.values.ravel()
        
        # Data extraction
        if hasattr(subset_ds, 'values'):
            data_values = subset_ds.values
        else:
            # If it is a Dataset instead of a DataArray
            if var_name in subset_ds.data_vars:
                data_values = subset_ds[var_name].values
            else:
                raise ValueError(f"Variable '{var_name}' not found in dataset")
        
        data_flat = data_values.ravel()

    # Validate and correct lengths
    # Find the common (minimum) length
    min_length = min(len(lat_idx_flat), len(lon_idx_flat), len(data_flat))
    
    if len(set([len(lat_idx_flat), len(lon_idx_flat), len(data_flat)])) > 1:
        logger.warning(f"Arrays have different lengths. Truncating to {min_length}")
        
        # Truncate all arrays to the minimum length
        lat_idx_flat = lat_idx_flat[:min_length]
        lon_idx_flat = lon_idx_flat[:min_length]
        data_flat = data_flat[:min_length]

    if len(lat_idx_flat) == 0:
        logger.warning("Empty arrays after truncation")
        return None

    # Build the DataFrame
    try:
        df = pd.DataFrame({
            "lat_idx": lat_idx_flat,
            "lon_idx": lon_idx_flat,
            var_name: data_flat
        })
        
        logger.debug(f"DataFrame created successfully with shape: {df.shape}")
        
    except Exception as e:
        logger.error(f"Failed to create DataFrame: {e}")
        logger.debug(f"lat_idx_flat: {lat_idx_flat[:5]}...")
        logger.debug(f"lon_idx_flat: {lon_idx_flat[:5]}...")
        logger.debug(f"data_flat: {data_flat[:5]}...")
        return None
    
    # Remove rows with NaN values
    df = df.dropna(how="any")
    
    if df.empty:
        logger.warning("DataFrame is empty after removing NaN values")
        return None

    # Aggregation
    agg_main = agg_std = agg_count = grouped = None
    try:
        grouped = df.groupby(["lat_idx", "lon_idx"])
        
        # Reductions
        agg_main = getattr(grouped[[var_name]], reduce)()
        agg_std = grouped[[var_name]].std()
        agg_count = grouped[[var_name]].count()
        
        # Back to xarray 
        result_ds = agg_main.to_xarray()
        try:
            del agg_main  # Free immediately after use
        except Exception:
            pass
        
        result_ds[f"{var_name}_std"] = agg_std[var_name].to_xarray()
        try:
            del agg_std  # Free immediately after use
        except Exception:
            pass
        
        result_ds[f"{var_name}_count"] = agg_count[var_name].to_xarray()
        try:
            del agg_count  # Free immediately after use
        except Exception:
            pass
        
        logger.debug(f"Aggregation successful. Result shape: {dict(result_ds.dims)}")
        
    except Exception as e:
        logger.error(f"Aggregation failed: {e}")
        return None
    finally:
        # Explicit cleanup of temporaries and prompt GC
        try:
            del grouped, df
        except Exception:
            pass
        for _name in ('agg_main', 'agg_std', 'agg_count'):
            try:
                if _name in locals():
                    del locals()[_name]
            except Exception:
                pass
        try:
            import gc as _gc
            _gc.collect()
        except Exception:
            pass
        gc.collect()  # Double cleanup for safety

    # Filter superobs
    if min_count > 1:
        result_ds = result_ds.where(result_ds[f"{var_name}_count"] >= min_count, drop=True)
    if max_std is not None:
        result_ds = result_ds.where(result_ds[f"{var_name}_std"] <= max_std, drop=True)

    # Optional weighting
    if weighting == "count":
        weights = result_ds[f"{var_name}_count"]
    elif weighting == "inv_var":
        weights = 1.0 / (result_ds[f"{var_name}_std"]**2)
    else:
        weights = None

    if weights is not None:
        # Normalize weights
        weights = weights / weights.sum()
        # Apply weighted mean
        weighted_mean = (result_ds[var_name] * weights).sum()
        result_ds[f"{var_name}_weighted"] = weighted_mean

    # Attach model geographic coordinates
    if model is not None:
        try:
            lat_centers = model[lat_name].values
            lon_centers = model[lon_name].values

            lat_indices = result_ds.lat_idx.values
            lon_indices = result_ds.lon_idx.values

            valid_lat = (lat_indices >= 0) & (lat_indices < len(lat_centers))
            valid_lon = (lon_indices >= 0) & (lon_indices < len(lon_centers))

            if np.all(valid_lat) and np.all(valid_lon):
                result_ds = result_ds.assign_coords(
                    lat=("lat_idx", lat_centers[lat_indices]),
                    lon=("lon_idx", lon_centers[lon_indices]),
                )
                # logger.debug("Successfully assigned model coordinates")
            else:
                logger.warning("Some indices are out of bounds for model coordinates")
                
        except Exception as e:
            logger.error(f"Failed to assign model coordinates: {e}")

    # Final instrumentation & cleanup
    try:
        if _do_tracemalloc:
            try:
                _tracemalloc_after = tracemalloc.take_snapshot()
                stats = _tracemalloc_after.compare_to(_tracemalloc_before, 'lineno')[:20]
                out_dir = os.environ.get("DC_TRACEMALLOC_DIR", os.environ.get("DC_RESULTS_DIR", "results"))
                os.makedirs(out_dir, exist_ok=True)
                fname = os.path.join(out_dir, f"tracemalloc_aggregate_superobs_{int(time.time())}.txt")
                with open(fname, "w") as fh:
                    fh.write(f"Duration: {time.time() - _tracemalloc_t0}\n\n")
                    for s in stats:
                        fh.write(str(s) + "\n")
                tracemalloc.stop()
            except Exception:
                pass
    finally:
        gc.collect()
    return result_ds


#@profile
def make_superobs(
    ds_obs: xr.Dataset,
    ds_model: xr.Dataset,
    var_name: str,
    lon_name: str = "lon",
    lat_name: str = "lat",
    reduce: str = "mean",
) -> xr.Dataset:
    """Pipeline: stack -> index -> aggregate."""
    try:
        ds_obs = compute_grid_index(ds_obs, ds_model, lon_name, lat_name)
        return aggregate_superobs(ds_obs, var_name, reduce=reduce, model=ds_model)
    except Exception as e:
        logger.error(f"Error in make_superobs: {e}")
        traceback.print_exc()
        raise(e)


#@profile
def superobs_binning(
    obs: xr.Dataset,
    model: xr.Dataset,
    var: str,
    method: str = "mean",
    obs_lat_name: str = 'lat',
    obs_lon_name: str = 'lon',
    model_lat_name: str = 'lat',
    model_lon_name: str = 'lon',
) -> xr.Dataset:
    """Bin observations onto model grid before comparison.
    
    Args:
        obs: Observations dataset with coords (lat_idx, lon_idx, n_points).
        model: Model dataset with target grid.
        var: Variable name to regrid (e.g., 'ssh').
        method: Aggregation ('mean', 'median', 'count').
    """
    try:
        if var not in obs.data_vars:
            raise ValueError(
                f"Variable '{var}' not found in obs dataset. "
                f"Available variables: {list(obs.data_vars.keys())}"
            )
        return _superobs_binning_with_coords(obs, model, var, method, obs_lat_name, obs_lon_name, 
                                            model_lat_name, model_lon_name)

    except Exception as e:
        logger.error(f"Error in superobs_binning: {e}")
        traceback.print_exc()
        raise

#@profile
def _superobs_binning_with_coords(
    obs: xr.Dataset,
    model: xr.Dataset,
    var: str,
    method: str,
    obs_lat_name: str,
    obs_lon_name: str,
    model_lat_name: str,
    model_lon_name: str,
    time_name: str = "time",
    depth_name: str = "depth",
    time_freq: str = "1D",
) -> xr.Dataset:
    """Bin observations onto the model grid (lat/lon/time/depth)."""
    # Optional tracemalloc instrumentation
    _do_tracemalloc = os.environ.get("DC_TRACEMALLOC") == "1"
    if _do_tracemalloc:
        try:
            tracemalloc.start()
            _tracemalloc_before = tracemalloc.take_snapshot()
            _tracemalloc_t0 = time.time()
        except Exception:
            _do_tracemalloc = False


    def compute_edges(centers: np.ndarray) -> np.ndarray:
        step = np.diff(centers) / 2
        edges = np.zeros(len(centers) + 1)
        edges[1:-1] = centers[:-1] + step
        edges[0] = centers[0] - step[0]
        edges[-1] = centers[-1] + step[-1]
        return edges

    # Retrieve centers and compute edges
    lat_centers = model[model_lat_name].values
    lon_centers = model[model_lon_name].values
    lat_edges = compute_edges(lat_centers)
    lon_edges = compute_edges(lon_centers)

    depth_edges, depth_centers = None, None
    if depth_name in model.coords:
        depth_centers = model[depth_name].values
        depth_edges = compute_edges(depth_centers)

    vars_to_extract = [var, obs_lat_name, obs_lon_name]
    if time_name in obs:
        vars_to_extract.append(time_name)
    if depth_name in obs:
        vars_to_extract.append(depth_name)

    # OPTIMIZATION: Use the streaming generator to process chunks
    # df = xr_to_obs_dataframe(obs[vars_to_extract], include_geometry=False)
    chunk_generator = xr_to_obs_dataframe(obs[vars_to_extract], include_geometry=False, yield_chunks=True)

    # We need to aggregate stats across chunks: count, sum, sum_sq (for std)
    # The result will be accumulated in a list of partial groupbys or a single dataframe if small enough
    # Since the target grid (lat/lon bins) is fixed, the number of groups is bounded by the model grid size.
    
    # Storage for aggregation
    agg_sum = None
    agg_count = None
    agg_sum_sq = None # Only for std
    
    requires_std = method in ["mean", "median", "std"] # Median is not supported exactly in streaming, falling back to mean or approx?
    # Actually median streaming is hard. If method is median, we might have to accept higher memory or warn.
    # For now, let's treat 'var' aggregation using sum/count for mean.
    # Note: 'median' is default in signature. If user wants true median, we can't stream easily without approx.
    # However, superobs are usually mean. Let's support mean and count fully.
    
    if method == "median":
        logger.warning("Streaming superobs binning does not support exact 'median' without full memory. using 'mean' logic for streaming optimization.") 
        method = "mean"

    for i, df_chunk in enumerate(chunk_generator):
        # Spatial binning
        df_chunk["lat_bin"] = pd.cut(df_chunk[obs_lat_name], bins=lat_edges, labels=lat_centers, include_lowest=True)
        df_chunk["lon_bin"] = pd.cut(df_chunk[obs_lon_name], bins=lon_edges, labels=lon_centers, include_lowest=True)

        # Temporal binning
        if time_name in df_chunk.columns:
            df_chunk["time_bin"] = pd.to_datetime(df_chunk[time_name]).dt.floor(time_freq)
        else:
            df_chunk["time_bin"] = pd.Timestamp("1900-01-01")

        # Depth binning
        if depth_name in df_chunk.columns and depth_edges is not None:
            df_chunk["depth_bin"] = pd.cut(df_chunk[depth_name], bins=depth_edges, labels=depth_centers, include_lowest=True)
        else:
            df_chunk["depth_bin"] = -1

        # Drop NaN bins
        # df_chunk = df_chunk.dropna(subset=["lat_bin", "lon_bin", "time_bin"])

        # Groupby
        group_cols = ["lat_bin", "lon_bin", "time_bin", "depth_bin"]
        # Optim: Set observed=True to handle categoricals properly (pandas future warning)
        grouped = df_chunk.groupby(group_cols, dropna=True, observed=True)[var]
        
        # Calculate partial stats
        chunk_count = grouped.count()
        chunk_sum = grouped.sum()
        
        # Accumulate
        if agg_count is None:
            agg_count = chunk_count
            agg_sum = chunk_sum
        else:
            agg_count = agg_count.add(chunk_count, fill_value=0)
            agg_sum = agg_sum.add(chunk_sum, fill_value=0)
            
        if requires_std:
            # Var(X) = E[X^2] - (E[X])^2
            # We need Sum(X^2)
            # Use a temporary series for X^2
            # Re-grouping might be slow if we attach it to df, better to just group the series
            # But we need the index. 
            # Alternative: df_chunk['var_sq'] = df_chunk[var]**2; grouped_sq = ...
            chunk_sum_sq = df_chunk.assign(var_sq=df_chunk[var]**2).groupby(group_cols, dropna=True, observed=True)['var_sq'].sum()
            
            if agg_sum_sq is None:
                agg_sum_sq = chunk_sum_sq
            else:
                agg_sum_sq = agg_sum_sq.add(chunk_sum_sq, fill_value=0)
        
        # Cleanup chunk
        del df_chunk, grouped, chunk_count, chunk_sum
        if requires_std:
             del chunk_sum_sq
        gc.collect()

    # Final Aggregation
    if agg_count is None:
        # Empty result
        result_count = pd.Series(dtype=float)
        result_main = pd.Series(dtype=float)
        result_std = None
    else:
        result_count = agg_count
        
        if method == "count":
            result_main = result_count
            result_std = None
        elif method == "mean":
            # Avoid division by zero
            result_main = agg_sum / agg_count
            
            if requires_std:
                # Std = sqrt( Mean(X^2) - Mean(X)^2 )
                # Mean(X^2) = Sum_sq / Count
                mean_sq = agg_sum_sq / agg_count
                mean_squared = result_main ** 2
                var_val = mean_sq - mean_squared
                # Fix precision issues (negative variance)
                var_val[var_val < 0] = 0
                result_std = np.sqrt(var_val)
            else:
                result_std = pd.Series(0, index=result_main.index) # dummy
        else:
             # Fallback
             result_main = agg_sum / agg_count
             result_std = None

    # Conversion to Xarray
    # Ensure index names map to dims
    # The series index is MultiIndex with names lat_bin, lon_bin, time_bin, depth_bin
    
    # Helper to convert series to DataArray
    def to_da(series, name):
        if series.empty:
            # Return empty dataset with correct dims if possible, or just empty
            return xr.DataArray(name=name)
        return series.to_xarray().rename(name)

    obs_binned = to_da(result_main, f"{var}_binned").to_dataset()

    if result_std is not None:
        obs_binned[f"{var}_std"] = to_da(result_std, f"{var}_std")
        
    obs_binned[f"{var}_count"] = to_da(result_count, f"{var}_count")
    
    # Cleanup aggregators
    del agg_sum, agg_count, agg_sum_sq
    gc.collect()

    # Convert physical coordinates
    # ... (rest of the function is the same, operating on obs_binned which is small)
    if "lat_bin" in obs_binned.coords:
        obs_binned["lat_bin"] = obs_binned["lat_bin"].astype(float)
    if "lon_bin" in obs_binned.coords:
        obs_binned["lon_bin"] = obs_binned["lon_bin"].astype(float)
    if "depth_bin" in obs_binned.coords and depth_centers is not None:
        obs_binned["depth_bin"] = obs_binned["depth_bin"].astype(float)

    # Final instrumentation & cleanup
    try:
        if _do_tracemalloc:
            try:
                _tracemalloc_after = tracemalloc.take_snapshot()
                stats = _tracemalloc_after.compare_to(_tracemalloc_before, 'lineno')[:20]
                out_dir = os.environ.get("DC_TRACEMALLOC_DIR", os.environ.get("DC_RESULTS_DIR", "results"))
                os.makedirs(out_dir, exist_ok=True)
                fname = os.path.join(out_dir, f"tracemalloc_superobs_binning_{int(time.time())}.txt")
                with open(fname, "w") as fh:
                    fh.write(f"Duration: {time.time() - _tracemalloc_t0}\n\n")
                    for s in stats:
                        fh.write(str(s) + "\n")
                tracemalloc.stop()
            except Exception:
                pass
    finally:
        gc.collect()
    return obs_binned


#@profile
def add_model_values(
    superobs_df: gpd.GeoDataFrame,
    model_ds: xr.DataArray,  # xr.Dataset,
    var: str,
    time_mode: str = "nearest"
):
    """
    Aligne les superobs (déjà binnées) avec les valeurs modèle sur une grille régulière (lat, lon en 1D).
    
    Parameters
    ----------
    superobs_df : pandas.DataFrame
        Doit contenir au minimum les colonnes ["time_bin", "lat_bin", "lon_bin", f"{var}_obs"].
        lat_bin/lon_bin doivent être les centres des mailles.
    model_ds : xr.Dataset
        Dataset modèle avec dimensions (time, lat, lon).
    var : str
        Nom de la variable modèle à extraire (ex: "ssh").
    time_mode : {"nearest", "mean"}
        - "nearest" : prend le pas de temps modèle le plus proche de time_bin.
        - "mean" : moyenne du modèle sur l’intervalle [time_bin, time_bin+Δt] (si défini).
    
    Returns
    -------
    pandas.DataFrame
        Le DataFrame enrichi avec une colonne f"{var}_model".
    """
    df = superobs_df.copy()

    # Convert time_bin to datetime
    df["time_center"] = pd.to_datetime(df["time_bin"])

    # Create DataArrays to vectorize the selection
    times = xr.DataArray(df["time_center"].values, dims="points")
    lats = xr.DataArray(df["lat_bin"].values, dims="points")
    lons = xr.DataArray(df["lon_bin"].values, dims="points")

    if time_mode == "nearest":
        model_on_points = model_ds.sel(
            time=times, lat=lats, lon=lons, method="nearest"
        )
    elif time_mode == "mean":
        model_on_points = model_ds.sel(
            time=times, lat=lats, lon=lons, method="mean"
        )
    else:
        raise ValueError("time_mode must be 'nearest' or 'mean'.")

    df[f"{var}_model"] = model_on_points.values
    return df


# ----------------------------------  Model → obs interpolation  -------------------------------------

def interpolate_grid_to_track_pyinterp(
    ds_model: xr.Dataset,
    ds_obs: xr.Dataset,
    reduce_precision: bool = False,
) -> xr.Dataset:
    """
    Interpolate Model grid to Observation track using PyInterp (High Performance).
    Supports Pairwise (Grid-to-Track) interpolation.
    """
    import pyinterp
    import pyinterp.backends.xarray
    
    # 1. Detect Lat/Lon names in Obs
    lat_name_obs = None
    lon_name_obs = None
    for c in ["latitude", "lat", "LATITUDE", "nav_lat", "LAT", "NAVLAT"]:
        if c in ds_obs.coords or c in ds_obs.data_vars:
            lat_name_obs = c
            break
    for c in ["longitude", "lon", "LONGITUDE", "nav_lon", "LON", "NAVLON"]:
        if c in ds_obs.coords or c in ds_obs.data_vars:
            lon_name_obs = c
            break
            
    if not lat_name_obs or not lon_name_obs:
        logger.warning("Could not find lat/lon in observation data.")
        return ds_model

    # 2. Extract Target Coordinates
    # Use .data to keep laziness if dask array
    tgt_lat = ds_obs[lat_name_obs].data if hasattr(ds_obs[lat_name_obs], 'data') else ds_obs[lat_name_obs].values
    tgt_lon = ds_obs[lon_name_obs].data if hasattr(ds_obs[lon_name_obs], 'data') else ds_obs[lon_name_obs].values

    # Ensure numpy array for pyinterp (will trigger compute if dask)
    tgt_lat = np.asarray(tgt_lat)
    tgt_lon = np.asarray(tgt_lon)
    
    if reduce_precision:
        tgt_lat = tgt_lat.astype(np.float32)
        tgt_lon = tgt_lon.astype(np.float32)
        ds_model = ds_model.astype(np.float32)

    # Output setup for Pairwise
    if len(tgt_lat) != len(tgt_lon):
        raise ValueError(f"Pairwise interpolation requires lat/lon of same length. Got {len(tgt_lat)} vs {len(tgt_lon)}")
    
    x_target = tgt_lon
    y_target = tgt_lat
    output_dim_size = len(tgt_lat)
    out_coords = {"points": np.arange(output_dim_size)}

    # Define wrapper for ufunc
    def _pyinterp_wrapper(data, lat_src, lon_src, xt=x_target, yt=y_target):
        try:
            x_axis = pyinterp.Axis(lon_src)
            y_axis = pyinterp.Axis(lat_src)
            
            # Check shapes
            if data.shape == (len(lat_src), len(lon_src)):
                    grid_data = data.T # (lon, lat)
            elif data.shape == (len(lon_src), len(lat_src)):
                    grid_data = data
            else:
                    # Attempt reshape
                    grid_data = data.reshape(len(lon_src), len(lat_src))

            grid = pyinterp.Grid2D(x_axis, y_axis, grid_data)
            
            # Pairwise interpolation
            res = grid.bivariate(
                x=xt,
                y=yt,
                interpolator="bilinear"
            )
            return res
        except Exception as e:
            # logger.error(f"PyInterp error: {e}")
            raise e

    # Apply to all variables in model
    # We create a new dataset
    ds_out = xr.Dataset()
    ds_out.attrs = ds_model.attrs
    
    for var_name in ds_model.data_vars:
        da = ds_model[var_name]
        
        # Detect Lat/Lon dims in Model
        lat_dim = next((d for d in da.dims if d in ['latitude', 'lat', 'nav_lat']), None)
        lon_dim = next((d for d in da.dims if d in ['longitude', 'lon', 'nav_lon']), None)
        
        if not lat_dim or not lon_dim:
            continue

        src_lat = da[lat_dim].values
        src_lon = da[lon_dim].values

        res = xr.apply_ufunc(
            _pyinterp_wrapper,
            da,
            src_lat,
            src_lon,
            input_core_dims=[[lat_dim, lon_dim], [], []],
            output_core_dims=[['points']], 
            vectorize=True, 
            dask='parallelized',
            output_dtypes=[np.float32 if reduce_precision else np.float64],
            dask_gufunc_kwargs={'allow_rechunk': True, 'output_sizes': {'points': output_dim_size}}
        )
        ds_out[var_name] = res

    ds_out = ds_out.assign_coords(points=out_coords["points"])
    return ds_out

#@profile
def interpolate_model_on_obs(
    model_da: xr.DataArray, 
    obs_df: pd.DataFrame, 
    variable: str, 
    method: str = "pyinterp",
    cache: Optional[Dict] = None,
) -> pd.DataFrame:
    '''Optimized version without unnecessary copies, using interpolate_grid_to_track_pyinterp if possible.'''
    
    if method == "pyinterp":
        # Attempt to use the new fast method if data is in xarray format
        # Otherwise fall back to the legacy internal method
        try:
             # Convert obs_df to minimal Dataset for the new function
            ds_obs = xr.Dataset.from_dataframe(obs_df[['lat', 'lon']])
            # Create a model Dataset wrapper (the new function expects a dataset)
            ds_model = model_da.to_dataset(name=variable)
            
            ds_interp = interpolate_grid_to_track_pyinterp(ds_model, ds_obs)
            interp_vals = ds_interp[variable].values
        except Exception:
            # logger.warning(f"Fast path failed, falling back to legacy")
            interp_vals = interpolate_with_pyinterp(model_da, obs_df, cache=cache)
            
    elif method == "kdtree":
        interp_vals = interpolate_with_kdtree(model_da, obs_df, variable)
    else:
        raise ValueError(f"Unknown interpolation method: {method}")
    
    # Assign using .loc to be explicit
    obs_df = obs_df.copy() # Ensure we have a writable copy if it was a view
    obs_df.loc[:, f"{variable}_model"] = interp_vals
    
    # CRITICAL: Immediate cleanup of temporaries
    del interp_vals
    gc.collect()
    
    return obs_df

#@profile
def _nearest_index(values: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """
    Return index of nearest value in sorted `values` for each element in `targets`.
    Vectorized.
    """
    # values must be 1D sorted
    idx = np.searchsorted(values, targets)
    idx = np.clip(idx, 1, len(values)-1)
    left = values[idx - 1]
    right = values[idx]
    choose_left = np.abs(targets - left) <= np.abs(right - targets)
    return np.where(choose_left, idx - 1, idx)

#@profile
def interpolate_with_pyinterp(
    model_da: xr.DataArray,
    obs_df: pd.DataFrame,
    n_threads: int = 4,
    rtree_k: int = 4,
    cache: Optional[Dict] = None,
) -> np.ndarray:
    """
    Memory-efficient interpolation of model_da on obs_df.
    - builds Grid2D per (time_idx, depth_idx) pair and releases it immediately.
    - groups obs by (time_idx, depth_idx) using numpy for low overhead.
    - processes groups in batches to control peak memory.
    Parameters:
        model_da: xr.DataArray with dims lat, lon [, time, depth]
        obs_df: pandas DataFrame with columns 'lon','lat' and optional 'time','depth'
        n_threads: number of threads passed to pyinterp.bilinear; keep small to limit memory
        batch_size: process at most this many obs indices per slice (to limit intermediate arrays)
        rtree_k: k for IDW fallback
        cache: Optional dictionary to cache loaded model slices (Grid2D objects)
    Returns:
        interp_vals: numpy array of length len(obs_df)
    """
    dims = set(model_da.dims)
    has_time = "time" in dims
    has_depth = "depth" in dims

    # Pre-calculate horizontal dims to avoid doing it per slice
    _check_dims = dims - {"time", "depth"}
    horizontal_dims = ("x", "y") if ("x" in _check_dims) and ("y" in _check_dims) else ("lon", "lat")
    use_geodetic_dist = False if horizontal_dims == ("x", "y") else True

    # OPTIMIZATION: Do NOT copy the full dataframe, it might be Huge.
    # obs_df = obs_df.copy()  # keep caller df untouched
    n_obs = len(obs_df)
    interp_vals = np.full(n_obs, np.nan, dtype=float)

    # prepare time/depth arrays (numpy)
    time_vals = model_da.time.values if has_time else None
    depth_vals = model_da.depth.values if has_depth else None

    # compute nearest indices for each observation
    t_indices = None
    d_indices = None

    if has_time:
        # Check if column exists
        if "time" in obs_df.columns:
            # Create a temporary Series/Array for time conversion, do not modify DF
            # This avoids coping the whole DF just to cast one column
            if pd.api.types.is_datetime64_any_dtype(obs_df["time"]):
                 obs_time_vals = obs_df["time"].values
            else:
                 temp_times = pd.to_datetime(obs_df["time"])
                 obs_time_vals = temp_times.values.astype("datetime64[ns]")
                 del temp_times
            
            # nearest index vectorized
            t_indices = _nearest_index(time_vals.astype("datetime64[ns]"), obs_time_vals)
            del obs_time_vals # Free temp memory

    if has_depth:
        if "depth" in obs_df.columns:
            # Create temp array for depth
            obs_depth_vals = obs_df["depth"].astype(float).values
            d_indices = _nearest_index(depth_vals.astype(float), obs_depth_vals)
            del obs_depth_vals

    # Build keys array of shape (n_obs, 2) with -1 for None
    keys = np.zeros((n_obs, 2), dtype=np.int64)
    keys[:, 0] = t_indices if t_indices is not None else -1
    keys[:, 1] = d_indices if d_indices is not None else -1

    # Map unique keys -> indices in obs_df
    # We will iterate over unique keys (but in memory-friendly order)
    if n_obs == 0:
        return interp_vals
    # t=time, d=depth, i8=int64
    structured = keys.view([("t", "i8"), ("d", "i8")]) 
    uniq, inverse = np.unique(structured, return_inverse=True)
    # uniq is array of shape (n_unique,), inverse maps obs -> uniq_idx

    # produce groups: for each uniq_idx, list obs indices
    # To avoid huge lists, we'll process uniq groups in batches
    n_unique = len(uniq)

    global _cache_hits, _cache_misses
    _cache_hits = 0
    _cache_misses = 0

    # helper to process one group key
    def _process_key(uniq_idx: int) -> None:
        global _cache_hits, _cache_misses
        # find obs indices for this group
        obs_idx = np.nonzero(inverse == uniq_idx)[0]
        if obs_idx.size == 0:
            return
        # compute t_idx, d_idx from uniq
        t_idx = int(uniq[uniq_idx]["t"])
        d_idx = int(uniq[uniq_idx]["d"])
        t_idx = None if t_idx == -1 else t_idx
        d_idx = None if d_idx == -1 else d_idx

        # Check Cache
        cache_key = (t_idx, d_idx)
        grid = None
        if cache is not None:
            grid = cache.get(cache_key)

        if grid is not None:
            # Cache Hit
            _cache_hits += 1

            pts = obs_df.iloc[obs_idx][list(horizontal_dims)].to_numpy()
            vals = pyinterp.bivariate(
                grid,
                x = pts[:, 0],
                y = pts[:, 1],
                bounds_error=False,
                num_threads=max(1, n_threads))
            interp_vals[obs_idx] = vals
            del pts, vals
            return

        # Cache Miss
        _cache_misses += 1
        # if _cache_misses % 10 == 0:
        #     logger.debug(f"Cache Miss: key={cache_key}, cache_keys={list(cache.keys()) if cache else []}")
        # prepare model slice: use isel which returns a view
        da_slice = model_da
        if t_idx is not None:
            da_slice = da_slice.isel(time=t_idx)
        if d_idx is not None:
            da_slice = da_slice.isel(depth=d_idx)
        
        # Use outer scope horizontal_dims

        # Quick and dirty fix: if we're in the x/y case, drop all coords that are not x,y[,time,depth]
        # TODO: Figure out why latitude and longitude are here in the first place in TOPAZ data (DC3)
        if horizontal_dims == ("x", "y"):
            coords_to_drop = [c for c in ["latitude", "longitude"] if c in da_slice.coords]
            da_slice = da_slice.drop_vars(coords_to_drop)

        # Build Grid2D from this slice
        pts = None
        vals = None
        try:
            # Ensure da_slice is loaded to avoid lazy loading memory leaks
            da_slice = da_slice.load()
            grid = pyinterp.backends.xarray.Grid2D(da_slice, geodetic=use_geodetic_dist)
            
            # Save to cache if enabled
            if cache is not None:
                # Limit cache size to avoid memory leak / OOM
                # FIFO Cache of size 10
                if len(cache) >= 10:
                    cache.pop(next(iter(cache)))
                cache[cache_key] = grid

            # fetch obs points for this group - avoid creating unnecessary copies
            pts = obs_df.iloc[obs_idx][list(horizontal_dims)].to_numpy()
            
            # Call interpolator
            vals = pyinterp.bivariate(
                grid,
                x = pts[:, 0], # lon if geodetic
                y = pts[:, 1], # lat if geodetic
                bounds_error=False,
                num_threads=max(1, n_threads))
            interp_vals[obs_idx] = vals
            
        except Exception:
            # fallback: build local RTree from numpy arrays of slice (build minimal arrays)
            lon = lat = lon2d = lat2d = points = values = tree = vals_idw = None
            try:
                lon = da_slice[horizontal_dims[0]].values
                lat = da_slice[horizontal_dims[1]].values
                # make sure small temporary arrays only
                lon2d, lat2d = np.meshgrid(lon, lat)
                points = np.column_stack([lon2d.ravel(), lat2d.ravel()])
                values = da_slice.values.ravel()
                tree = pyinterp.RTree()
                tree.packing(points, values)
                if pts is None:
                    pts = obs_df.iloc[obs_idx][list(horizontal_dims)].to_numpy()
                vals_idw, _ = tree.inverse_distance_weighting(pts, k=rtree_k)
                interp_vals[obs_idx] = vals_idw
            except Exception:
                interp_vals[obs_idx] = np.nan
            finally:
                # Explicit cleanup of all fallback temporaries
                del lon, lat, lon2d, lat2d, points, values, tree, vals_idw
        finally:
            # CRITICAL: Explicit cleanup of all variables to prevent memory leaks
            # Close da_slice if it has a close method
            try:
                if hasattr(da_slice, 'close'):
                    da_slice.close()
            except Exception:
                pass
            del grid, pts, vals, da_slice
            # force cleanup after each slice processing
            gc.collect()

    # Process groups in small batches to limit memory peaks
    # Optionally use a tiny thread pool if CPU-bound but be careful with memory
    if n_threads is None or n_threads <= 1:
        for u in range(n_unique):
            _process_key(u)
    else:
        # process in chunks with a limited worker pool
        # we won't submit all tasks at once to avoid memory blowup
        chunk_size = max(1, min(64, n_unique))
        for start in range(0, n_unique, chunk_size):
            end = min(n_unique, start + chunk_size)
            with ThreadPoolExecutor(max_workers=n_threads) as ex:
                futures = [ex.submit(_process_key, u) for u in range(start, end)]
                for f in futures:
                    # we wait for completion and re-raise
                    f.result()
            # after each chunk, force collection and free futures
            try:
                del futures
            except Exception:
                pass
            gc.collect()

    # final cleanup of large helper arrays
    try:
        for _name in ('uniq', 'inverse', 'structured', 'keys', 't_indices', 'd_indices'):
            if _name in locals():
                try:
                    del locals()[_name]
                except Exception:
                    pass
    except Exception:
        pass
    
    
    try:
        gc.collect()
    except Exception:
        pass

    # logger.debug(f"Interpolation Cache: {_cache_hits} hits, {_cache_misses} misses. Cache Size: {len(cache) if cache is not None else 0}")
    return interp_vals


#@profile
def interpolate_with_kdtree(
    model_da: xr.DataArray,
    obs_df: pd.DataFrame,
    time_index: int = 0,
    obs_lon_col: str = "lon",
    obs_lat_col: str = "lat",
    obs_depth_col: str = "depth",
) -> np.ndarray:
    """
    Interpolate model values onto observation positions using KDTree.
    
    Args:
        model_da: Model DataArray
        obs_df: Observations DataFrame
        time_index: Index temporel (ignoré si pas de dimension time)
        obs_lon_col: Nom de la colonne longitude dans obs_df
        obs_lat_col: Nom de la colonne latitude dans obs_df  
        obs_depth_col: Nom de la colonne profondeur dans obs_df
        
    Returns:
        np.ndarray: Array of interpolated values (same length as obs_df)
    """
    
    # Input validation
    if not isinstance(model_da, xr.DataArray):
        raise TypeError(f"Expected xr.DataArray, got {type(model_da)}")

    # Initialize result
    n_obs = len(obs_df)
    result = np.full(n_obs, np.nan)  # Result array initialized to NaN
    
    # If time exists, select the time slice
    if "time" in model_da.dims:
        try:
            da = model_da.isel(time=time_index)
            logger.debug(f"Selected time index {time_index}")
        except Exception as e:
            logger.warning(f"Could not isel time={time_index}: {e}; trying to squeeze time.")
            da = model_da.squeeze("time", drop=True)
    else:
        da = model_da
    # Extraire les arrays lat/lon
    lat_vals = da.coords[obs_lat_col].values
    lon_vals = da.coords[obs_lon_col].values

    # Construire la grille spatiale
    if lat_vals.ndim == 1 and lon_vals.ndim == 1:
        lon_grid, lat_grid = np.meshgrid(lon_vals, lat_vals)
    elif lat_vals.ndim == 2 and lon_vals.ndim == 2:
        lat_grid = lat_vals
        lon_grid = lon_vals
    else:
        raise ValueError(f"Unsupported lat/lon shapes: lat {lat_vals.shape}, lon {lon_vals.shape}")

    # Flatten spatial coordinates and build the KDTree
    pts = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])  # [lat, lon] order
    tree = cKDTree(pts)
    logger.debug(f"KDTree built on {pts.shape[0]} spatial nodes.")

    # Check if the model has a depth dimension
    has_depth = "depth" in da.dims
    
    if not has_depth:
        # cas 2d : pas de dimension depth
        if obs_df.empty:
            return result
        
        # Ensure the data slice is 2D
        data2d = da.values
        if data2d.ndim != 2:
            data2d = np.squeeze(data2d)
        if data2d.shape != lat_grid.shape:
            if data2d.shape == (lat_grid.shape[1], lat_grid.shape[0]):
                data2d = data2d.T
            else:
                raise ValueError(f"Model 2D slice shape {data2d.shape} doesn't match lat/lon grid {lat_grid.shape}")

        # Prepare observation points
        obs_mask_valid = (~obs_df[obs_lat_col].isna()) & (~obs_df[obs_lon_col].isna())
        if not obs_mask_valid.any():
            return result
            
        obs_points = np.column_stack([
            obs_df.loc[obs_mask_valid, obs_lat_col].values, 
            obs_df.loc[obs_mask_valid, obs_lon_col].values
        ])

        # interpolation and assignment in the result array
        _, indices = tree.query(obs_points, k=1)
        vals = data2d.ravel()[indices]
        
        # Assign interpolated values to the correct positions
        result[obs_mask_valid] = vals
        
        return result

    # cas 3d : depth existe
    depth_vals = da.coords["depth"].values
    
    # Si les obs n'ont pas d'info de profondeur, utiliser la surface
    if obs_depth_col not in obs_df.columns:
        logger.warning("Observations have no depth column — using surface depth for all obs.")
        obs_depths = np.full(len(obs_df), depth_vals[0])
    else:
        obs_depths = obs_df[obs_depth_col].values

    # For each observation, find the nearest model depth index
    depth_idx_per_obs = np.full(n_obs, -1, dtype=int)
    valid_depth_mask = ~pd.isna(obs_depths)
    
    if valid_depth_mask.any():
        # Vectorized computation of the nearest depth index
        diffs = np.abs(depth_vals[:, None].astype(float) - obs_depths[valid_depth_mask][None, :].astype(float))
        nearest_idxs = np.argmin(diffs, axis=0)
        depth_idx_per_obs[valid_depth_mask] = nearest_idxs
    else:
        logger.warning("No valid observation depths found")
        return result

    # Process observations grouped by depth index
    unique_depth_idxs = np.unique(depth_idx_per_obs[depth_idx_per_obs >= 0])
    logger.debug(f"Found {len(unique_depth_idxs)} unique model depth levels to process")

    for d_idx in unique_depth_idxs:
        mask = depth_idx_per_obs == d_idx
        if not mask.any():
            continue

        # Take the 2D model slice at this depth
        try:
            slice_da = da.isel(depth=d_idx)
        except Exception:
            slice_da = da.sel(depth=depth_vals[d_idx], method="nearest")

        data2d = slice_da.values
        data2d = np.squeeze(data2d)
        if data2d.ndim != 2:
            raise ValueError(f"Model slice at depth index {d_idx} is not 2D: shape {data2d.shape}")

        # Align the shape with lat_grid/lon_grid
        if data2d.shape != lat_grid.shape:
            if data2d.shape == (lat_grid.shape[1], lat_grid.shape[0]):
                data2d = data2d.T
            else:
                try:
                    if obs_lat_col in slice_da.dims and obs_lon_col in slice_da.dims:
                        slice_da2 = slice_da.transpose(obs_lat_col, obs_lon_col)
                        data2d = slice_da2.values
                    else:
                        raise ValueError
                except Exception:
                    raise ValueError(f"Data shape mismatch at depth {d_idx}: data {data2d.shape}, grid {lat_grid.shape}")

        # Calculate observation points for this depth
        valid_mask = mask & (~obs_df[obs_lat_col].isna()) & (~obs_df[obs_lon_col].isna())
        if not valid_mask.any():
            continue
            
        obs_points = np.column_stack([
            obs_df.loc[valid_mask, obs_lat_col].values,
            obs_df.loc[valid_mask, obs_lon_col].values
        ])

        # interpolation
        _, indices = tree.query(obs_points, k=1)
        interpolated = data2d.ravel()[indices]

        # Assign results to the correct positions in the result array
        result[valid_mask] = interpolated

    return result



# ----------------------------------  Binning  -------------------------------------

#@profile
def apply_binning(
    df: pd.DataFrame,
    bin_specs: Optional[Dict[str, Union[int, list, str]]] = None
) -> pd.DataFrame:
    """
    Add binning columns (time_bin, lat_bin, lon_bin, depth_bin) according to bin_specs.
    Handles column or bin errors.

    Args:
        df (pd.DataFrame): DataFrame to bin.
        bin_specs (dict, optional): Dictionary {dim: bins}. By default, covers all 4 dimensions.

    Returns:
        pd.DataFrame: DataFrame with added bin columns.
    """

    # --- Import pandas at the top to avoid scope errors ---
    import pandas as pd

    # Default bin values
    default_bins = {
        "time": "1D",  # Daily temporal binning
        "lat": np.arange(-90, 91, 1),
        "lon": np.arange(-180, 180, 1),
        "depth": None,  # To be defined if the column exists
    }
    if bin_specs is None:
        bin_specs = default_bins.copy()
    else:
        # Fill in with default values if missing
        for k, v in default_bins.items():
            if k not in bin_specs:
                bin_specs[k] = v

    groupby = []
    for dim, bins in bin_specs.items():
        if dim not in df.columns:
            continue  # Ignore si la colonne n'existe pas
        try:
            if dim == "time":
                # Diagnostic log for time column
                try:
                    col = df[dim]
                    # logger.info(f"[apply_binning] time column: dtype={col.dtype}, min={col.min()}, max={col.max()}, n_unique={col.nunique()}")
                except Exception as diag_e:
                    logger.warning(f"[apply_binning] Could not log time column diagnostics: {diag_e}")
                # Optimisation: check if conversion is needed
                if not pd.api.types.is_datetime64_any_dtype(df[dim]):
                    df[dim] = pd.to_datetime(df[dim], errors="coerce")

                # --- Filtrage explicite des dates aberrantes ---
                # Fix for "ValueError: value too large" with extreme dates (e.g. 1677, 2262)
                # Filter to keep only data within a reasonable range (1900-2100)
                try:
                    min_valid = pd.Timestamp("1900-01-01")
                    max_valid = pd.Timestamp("2100-01-01")
                    # Ensure column is timezone-naive or compatible
                    if df[dim].dt.tz is not None:
                         min_valid = min_valid.tz_localize(df[dim].dt.tz)
                         max_valid = max_valid.tz_localize(df[dim].dt.tz)
                         
                    mask_valid = (df[dim] >= min_valid) & (df[dim] <= max_valid)
                    if (~mask_valid).any():
                        # logger.warning(f"[apply_binning] Dropping {(~mask_valid).sum()} rows with out-of-bounds dates")
                        df = df.loc[mask_valid].reset_index(drop=True)
                except Exception as e:
                    print(f"[apply_binning] Error filtering dates: {e}")


                if isinstance(bins, str):
                    df.loc[:, f"{dim}_bin"] = df[dim].dt.floor(bins)
                    groupby.append(f"{dim}_bin")
                else:
                    raise ValueError("For the 'time' dimension, bins must be a string (e.g., '1D').")
            else:
                # For spatial dimensions
                if bins is None:
                    # If no bins provided for depth, try to make auto bins
                    if dim == "depth":
                        unique_depths = np.unique(df[dim].dropna())
                        if len(unique_depths) > 1:
                            bins = unique_depths
                        else:
                            continue  # No binning possible
                    else:
                        continue
                if isinstance(bins, int):
                    df.loc[:, f"{dim}_bin"] = pd.qcut(df[dim], q=bins, duplicates="drop")
                    groupby.append(f"{dim}_bin")
                elif isinstance(bins, (list, np.ndarray)):
                    df.loc[:, f"{dim}_bin"] = pd.cut(df[dim], bins=bins, include_lowest=True)
                    groupby.append(f"{dim}_bin")
                else:
                    raise ValueError(f"Unsupported bin type for {dim}: {type(bins)}")
        except Exception as e:
            # Gather more diagnostic info
            col_info = ""
            if dim in df.columns:
                col = df[dim]
                col_info = (
                    f"type={type(col).__name__}, dtype={col.dtype}, "
                    f"shape={col.shape}, min={col.min() if hasattr(col, 'min') else 'N/A'}, "
                    f"max={col.max() if hasattr(col, 'max') else 'N/A'}, "
                    f"n_unique={col.nunique() if hasattr(col, 'nunique') else 'N/A'}, "
                    f"sample={col.head(5).tolist()}"
                )
            else:
                col_info = "(column missing from df)"
            print(f"[apply_binning] Error for dimension '{dim}': {e}\n  - bin_spec: {bins}\n  - column: {col_info}")
            continue

    return (df, groupby)

# ----------------------------------  Scores with xskillscore  -------------------------------------
#@profile
def compute_scores_xskillscore(
    df: pd.DataFrame,
    y_obs_col: str,
    y_pred_col: str,
    metrics: Optional[list] = None,
    weights: Optional[pd.Series] = None,
    groupby: Optional[list] = None,
    #binning: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Compute xskillscore (and custom) scores on a DataFrame, with binning and error handling.

    Args:
        df: DataFrame containing observation and prediction columns.
        y_obs_col: Name of the observations column.
        y_pred_col: Name of the predictions column.
        metrics: List of metrics to compute (str).
        weights: Optional weights (Series aligned with df).
        groupby: List of columns for groupby (e.g., ["lat_bin", "time_bin"]).
        custom_metrics: Dictionary {name: function(y_pred, y_obs)} to add custom scores.
        binning: Dictionary for binning (see apply_binning).

    Returns:
        DataFrame with computed scores.
    """
    try:
        all_results = {}
        df = df.dropna(subset=[y_obs_col, y_pred_col])

        if groupby is not None and len(groupby) > 0:
            grouped = df.groupby(groupby, observed=False)
        else:
            grouped = [(None, df)]

        # Compute scores
        bin_results = []
        for group_key, group_df in grouped:
            # Convert to DataArray for xskillscore
            y_obs = xr.DataArray(group_df[y_obs_col].values, dims="points")
            y_pred = xr.DataArray(group_df[y_pred_col].values, dims="points")
            group_result = {}
            for metric in metrics:
                if metric == "rmsd":   # Oceanbench compatibility
                    metric = "rmse"  
                metric_func = XSKILL_METRICS.get(metric)
                try:
                    # Handle weights if supported
                    if metric in [
                        "rmse", "mae", "mse", "me", "median_absolute_error",
                        "mean_squared_log_error", "r2", "mape", "smape",
                    ]:
                        if weights is not None:
                            score = metric_func(
                                y_pred, y_obs, dim="points",
                                weights=xr.DataArray(weights.values, dims="points"))
                        else:
                            score = metric_func(y_pred, y_obs, dim="points")
                    elif metric in ["pearson_r", "spearman_r"]:
                        score = metric_func(y_pred, y_obs, dim="points")
                    elif metric == "crps_ensemble":
                        # For crps_ensemble, y_pred must be (ensemble, points)
                        # Here, we assume y_pred is already in the correct shape
                        score = metric_func(y_pred, y_obs, dim="points")
                    else:
                        score = metric_func(y_pred, y_obs, dim="points")
                    # Extract the scalar value
                    if hasattr(score, "item"):
                        group_result[metric] = score.values  #.item()
                    else:
                        group_result[metric] = float(score)
                except Exception as e:
                    group_result[metric] = np.nan
                    print(f"[compute_scores_xskillscore] Error on '{metric}': {e}")
            # Add groupby keys if grouped
            if group_key is not None:
                if isinstance(group_key, tuple):
                    for k, v in zip(groupby, group_key):
                        group_result[k] = v
                else:
                    group_result[groupby[0]] = group_key
            bin_results.append(group_result)

        # 4. Score global (tous bins confondus)
        df_bins = pd.DataFrame(bin_results)
        metrics = [col for col in df_bins.columns if col not in ['lat_bin', 'lon_bin', 'time_bin', 'depth_bin']]
        global_stats = {}
        def to_float_scalar(x):
            try:
                if hasattr(x, "item"):
                    return float(x.item())
                return float(x)
            except Exception:
                return np.nan

        for metric in metrics:
            vals = df_bins[metric].apply(to_float_scalar)
            global_stats[f"{metric}_mean"] = vals.mean()
            global_stats[f"{metric}_median"] = vals.median()
            global_stats[f"{metric}_std"] = vals.std()

        all_results["per_bins"] = bin_results
        all_results["global"] = global_stats

        return all_results
    except Exception as e:
        print(f"Error in Compute_scores_xskillscore: {e}")
        traceback.print_exc()


# ----------------------------------  QC filtering  -------------------------------------
def filter_observations_by_qc(
    ds: xr.Dataset,
    qc_mappings: Dict[str, Dict[str, List[int]]] = DEFAULT_QC_MAPPING,
    drop: bool = True,
) -> xr.Dataset:
    """
    Filters observation dataset using QC flags.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing observation data.
    variable : str
        Name of the variable to check QC for (e.g., "sea_surface_temperature").
    qc_variable : str, optional
        Name of the QC flag variable. If not provided, inferred from DEFAULT_QC_MAPPING.
    valid_flags : list, optional
        Valid QC flag values. If not provided, inferred from DEFAULT_QC_MAPPING.
    drop : bool
        Whether to drop invalid observations (True) or mask them (False).
    fallback_to_any_qc : bool
        If True, will use any available QC variable if specific one is not found.

    Returns
    -------
    xr.Dataset
        Dataset with filtered or masked invalid observations.
    """

    if not qc_mappings or len(qc_mappings) == 0:
        raise ValueError("QC mappings must be provided and cannot be empty.")
    # Automatic determination if possible
    for variable in qc_mappings:
        if variable in ds:
            qc_mapping = qc_mappings[variable]
            qc_variable = qc_mapping.get("qc_variable")
            valid_flags = qc_mapping.get("valid_flags")

            if qc_variable is None or qc_variable not in ds:
                raise ValueError(f"QC variable could not be determined or found in dataset for '{variable}'.")

            qc = ds[qc_variable]
            if valid_flags is None:
                raise ValueError("No valid flags specified or found in DEFAULT_QC_MAPPING.")

            valid_mask = qc.isin(valid_flags)

            ds = ds.where(valid_mask, drop=drop)
    return ds


# ----------------------------------  OTHER  -------------------------------------
def apply_spatial_mask(df: pd.DataFrame, mask_fn: Callable[[pd.DataFrame], pd.Series]) -> pd.DataFrame:
    return df[mask_fn(df)]


#@profile
def xr_to_obs_dataframe(
    obj: xr.Dataset | xr.DataArray,
    include_geometry: bool = False,
    coord_like: Optional[List[str]] = ["lat", "lon", "depth", "time"],
    ocean_vars: Optional[List[str]] = None,
    yield_chunks: bool = False,
) -> pd.DataFrame | Generator[pd.DataFrame, None, None]:
    '''
    Version optimisée de xr_to_obs_dataframe qui minimise les copies mémoire.
    Traite les gros datasets par chunks pour éviter la saturation mémoire.
    yield_chunks: Si True, retourne un générateur de DataFrames (chunks) au lieu de concaténer.
    '''
    # Normalize to Dataset
    if isinstance(obj, xr.DataArray):
        ds = obj.to_dataset(name=obj.name or "value")
    else:
        ds = obj

    exclude_dims = {"n_points", "num_nadir", "num_points", "num_obs"}

    # Identify variables to keep (without creating copies)
    coord_vars = []
    for cname in coord_like:
        if cname in ds.dims and cname not in exclude_dims:
            coord_vars.append(cname)
        elif cname in ds.coords and cname not in exclude_dims:
            coord_vars.append(cname)
        elif cname in ds.data_vars and cname not in exclude_dims:
            coord_vars.append(cname)

    if ocean_vars is None:
        ocean_vars = [v for v in ds.data_vars
                     if v not in exclude_dims and v.lower() not in coord_like]

    keep_vars = list(dict.fromkeys(coord_vars + ocean_vars))

    # Subset without unnecessary copy
    subset = ds[keep_vars]

    # Check if chunks processing is needed (to avoid memory saturation)
    n_points_dim = None
    for dim in subset.dims:
        if dim in exclude_dims:
            n_points_dim = dim
            break

    chunk_size = 500000  # Process 500k points at a time to reduce overhead
    # Check if data is already in memory (not Dask arrays)
    is_dask_array = any(hasattr(subset[var].data, 'compute') for var in subset.data_vars)
    
    # CASE 1: Eager dataset or small dataset -> Return full DF (or yield one chunk)
    if not (n_points_dim is not None and subset.sizes.get(n_points_dim, 0) > chunk_size and is_dask_array):
        # Normal conversion for small datasets
        # Note: If yield_chunks is True, we must still respect it even for small datasets
        df = subset.to_dataframe().reset_index(drop=False)
        del subset
        
        if isinstance(obj, xr.DataArray) and obj.name is None:
            df.rename(columns={"value": "variable"}, inplace=True)
            
        if include_geometry and {'lon', 'lat'}.issubset(df.columns):
            geometry = gpd.points_from_xy(df.lon, df.lat)
            df = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326", copy=False)
            
        if yield_chunks:
            return iter([df])
        else:
            return df

    # CASE 2: Large Lazy Dataset -> Process chunks
    # We define a generator logic here
    def chunk_generator():
        import dask
        import gc
        
        n_points = subset.sizes[n_points_dim]
        
        for chunk_idx, start_idx in enumerate(range(0, n_points, chunk_size)):
            end_idx = min(start_idx + chunk_size, n_points)
            
            # Extract chunk slice (still lazy at this point)
            chunk = subset.isel({n_points_dim: slice(start_idx, end_idx)})

            chunk_df = None
            max_retries = 3
            
            for retry in range(max_retries):
                try:
                    # Use synchronous scheduler to avoid Dask worker issues
                    # We create a new context to ensure isolation
                    with dask.config.set(scheduler='synchronous'):
                        chunk_df = chunk.to_dataframe().reset_index(drop=False)
                    break
                except (KeyError, Exception) as e:
                    if retry < max_retries - 1:
                        logger.warning(f"Retry {retry + 1}/{max_retries} for chunk {start_idx}-{end_idx}: {e}")
                        gc.collect() 
                    else:
                        logger.error(f"Failed to convert chunk {start_idx}-{end_idx}. Skipping.")
                        chunk_df = None

            del chunk
            
            if chunk_df is not None:
                if isinstance(obj, xr.DataArray) and obj.name is None:
                    chunk_df.rename(columns={"value": "variable"}, inplace=True)

                if include_geometry and {'lon', 'lat'}.issubset(chunk_df.columns):
                    geometry_chunk = gpd.points_from_xy(chunk_df.lon, chunk_df.lat)
                    chunk_df = gpd.GeoDataFrame(chunk_df, geometry=geometry_chunk, crs="EPSG:4326", copy=False)

                yield chunk_df
                
                # Cleanup after yield
                del chunk_df
            
            gc.collect()

    # Determine return type based on yield_chunks
    if yield_chunks:
        # Return the generator directly
        return chunk_generator()
    else:
        # Consume generator and contact (Legacy behavior)
        chunks = list(chunk_generator())
        if chunks:
            df = pd.concat(chunks, ignore_index=True)
        else:
            df = pd.DataFrame() # Empty result
            
        del chunks
        del subset
        return df


# ------------------------------------- Format output -------------------------------------------

'''def format_class4_results(class4_results_df):
    """
    Extract Class4 results and format them like sample_results_grid.txt
    
    Args:
        class4_results_df: DataFrame with columns 'per_bins', 'global', 'variable'
    
    Returns:
        pd.DataFrame: Results formatted by depth and variable
    """
    
    # Mapping of target depths (like in sample_results_grid.txt)
    target_depths = {
        'Surface': (0, 50),      # Premier bin = Surface
        '50m': (45, 55),         # Autour de 50m
        '200m': (180, 220),      # Autour de 200m
        '550m': (500, 600)       # Autour de 550m
    }
    
    results = []
    
    for _, row in class4_results_df.iterrows():
        variable = row['variable']
        per_bins = row.get('per_bins', [])
        
        if not per_bins:  # No results for this variable
            continue
        
        # Check if depth bins exist
        has_depth_bins = any('depth_bin' in bin_data for bin_data in per_bins)
        
        if not has_depth_bins:
            metric_name = next(iter(per_bins)) 
            # No depth bins: all variables are "Surface"
            # Compute the mean of metrics over all bins
            rmse_values = [float(bin_data['rmse']) for bin_data in per_bins if 'rmse' in bin_data]
            if rmse_values:
                mean_rmse = np.mean(rmse_values)
                results.append({
                    'Metric': f"Surface {variable}",
                    'Value': mean_rmse
                })
        else:
            # Depth bins exist: normal processing
            depth_rmse = {}
            
            for bin_data in per_bins:
                if 'depth_bin' in bin_data and 'rmse' in bin_data:
                    depth_interval = bin_data.get('depth_bin', None)
                    rmse_value = float(bin_data['rmse'])
                    
                    # Associate with the corresponding target depth
                    depth_center = (depth_interval.left + depth_interval.right) / 2
                    
                    for depth_label, (min_depth, max_depth) in target_depths.items():
                        if min_depth <= depth_center <= max_depth:
                            if depth_label not in depth_rmse:
                                depth_rmse[depth_label] = []
                            depth_rmse[depth_label].append(rmse_value)
                            break
            
            # Calculate averages by depth for this variable
            for depth_label in target_depths.keys():
                if depth_label in depth_rmse:
                    mean_rmse = np.mean(depth_rmse[depth_label])
                    results.append({
                        'Variable': f"{depth_label} {variable}",
                        'Value': mean_rmse
                    })
    
    # Create the final DataFrame
    if results:
        df_results = pd.DataFrame(results)
        df_results = df_results.set_index('Metric')
        return df_results
    else:
        return pd.DataFrame()'''


#@profile
def format_class4_results(class4_results_df):
    """
    Extract Class4 results and format them like sample_results_grid.txt
    
    Args:
        class4_results_df: DataFrame with columns 'per_bins', 'global', 'variable'
    
    Returns:
        pd.DataFrame: Results formatted by depth and variable
    """
    
    # Mapping of target depths (like in sample_results_grid.txt)
    target_depths = {
        'Surface': (0, 50),      # Premier bin = Surface
        '50m': (45, 55),         # Autour de 50m
        '200m': (180, 220),      # Autour de 200m
        '550m': (500, 600)       # Autour de 550m
    }
    
    results = []
    
    for _, row in class4_results_df.iterrows():
        variable = row['variable']
        per_bins = row.get('per_bins', [])
        
        if not per_bins:  # No results for this variable
            continue
        
        # Check if depth bins exist
        has_depth_bins = any('depth_bin' in bin_data for bin_data in per_bins)
        
        if not has_depth_bins:
            # No depth bins: all variables are "Surface"
            # Accumulate all values for each metric
            metrics_values = {}
            
            for bin_data in per_bins:
                for metric_name, metric_value in bin_data.items():
                    if metric_name not in XSKILL_METRICS and metric_name != "count":
                        continue
                    
                    if metric_name not in metrics_values:
                        metrics_values[metric_name] = []
                    try:
                        metrics_values[metric_name].append(float(metric_value))
                    except (ValueError, TypeError):
                        pass
            
            # Compute the mean for each metric
            for metric_name, values in metrics_values.items():
                if values:
                    mean_value = np.mean(values)
                    results.append({
                        'Metric': metric_name,
                        'Variable': f"Surface {variable}",
                        'Value': mean_value
                    })
        else:
            # Depth bins exist: normal processing par profondeur
            # Organize by metric and by depth
            metrics_by_depth = {}
            
            for bin_data in per_bins:
                if 'depth_bin' not in bin_data:
                    continue
                    
                depth_interval = bin_data['depth_bin']
                
                # Calculer le centre de la profondeur
                depth_center = (depth_interval.left + depth_interval.right) / 2
                
                for metric_name, metric_value in bin_data.items():
                    if metric_name not in XSKILL_METRICS and metric_name != "count":
                        continue
                    
                    try:
                        val = float(metric_value)
                    except (ValueError, TypeError):
                        continue

                    # Associate with the corresponding target depth
                    for depth_label, (min_depth, max_depth) in target_depths.items():
                        if min_depth <= depth_center <= max_depth:
                            if metric_name not in metrics_by_depth:
                                metrics_by_depth[metric_name] = {}
                            if depth_label not in metrics_by_depth[metric_name]:
                                metrics_by_depth[metric_name][depth_label] = []
                            
                            metrics_by_depth[metric_name][depth_label].append(val)
                            break
            
            # Compute means by metric and by depth
            for metric_name, depth_values in metrics_by_depth.items():
                for depth_label, values in depth_values.items():
                    if values:
                        mean_value = np.mean(values)
                        results.append({
                            'Metric': metric_name,
                            'Variable': f"{depth_label} {variable}",
                            'Value': mean_value
                        })
    
    # Create the final DataFrame
    if results:
        df_results = pd.DataFrame(results)
        # df_results = df_results.set_index('Metric')
        return df_results
    else:
        return pd.DataFrame()


# ----------------------------------  Main Class4Evaluator  -------------------------------------

class Class4Evaluator:
    def __init__(
        self,
               metrics: List[str] = ["rmse"],
        interpolation_method: str = "pyinterp",
        delta_t: pd.Timedelta = pd.Timedelta("12h"),
        bin_specs: Optional[Dict[str, Union[int, List]]] = None,
        spatial_mask_fn: Optional[Callable[[pd.DataFrame], pd.Series]] = None,
        cache_dir: Optional[str] = None,
        apply_qc: bool = True,
        qc_mapping: Optional[Dict[str, Dict[str, List[int]]]] = None,
    ):
        """
        Parameters
        ----------
        metrics : list of str
            List of metrics to compute (e.g., ["rmse", "mae"]).
        interpolation_method : str
            Method to interpolate model onto observations ("pyinterp", "kdtree", "xesmf").
        delta_t : pd.Timedelta
            Maximum time difference allowed between model and observation for matching.
        bin_specs : dict, optional
            Binning specifications, e.g., {"time": "1D", "lat": 1, "lon": 1, "depth": None}.
        spatial_mask_fn : callable, optional
            Function that takes a DataFrame and returns a boolean Series to mask spatially.
        cache_dir : str, optional
            Directory to use for caching intermediate results.
        apply_qc : bool
            Whether to apply QC filtering on observations.
        qc_mapping : dict, optional
            Mapping for QC variables and valid flags. If None, uses DEFAULT_QC_MAPPING.
        """
        self.metrics = metrics
        self.interp_method = interpolation_method
        self.time_tol = delta_t
        self.bin_specs = bin_specs
        self.mask_fn = spatial_mask_fn
        self.cache_dir = cache_dir
        self.apply_qc = apply_qc
        self.qc_mapping = qc_mapping or DEFAULT_QC_MAPPING

    #@profile
    def run(
        self,
        model_ds: xr.Dataset,
        obs_ds: xr.Dataset,
        variables: List[str],
        ref_coords: dict,
        matching_type: str = "nearest",  # " "nearest" or "superobs"
    ) -> pd.DataFrame:
        """
        Main evaluation method.
        Parameters
        ----------
        model_ds : xr.Dataset
            Model dataset with dimensions (time, lat, lon) or (time, depth, lat, lon).
        obs_ds : xr.Dataset
            Observation dataset with dimensions (time, lat, lon) or (time, depth, lat, lon).
        variables : list of str
            List of variable names to evaluate (must be in both datasets).
        ref_coords : dict
            Reference coordinates for lat/lon (e.g., {"lat": "latitude", "lon": "longitude"}).
        matching_type : str
            Type of matching: "nearest" for nearest neighbor, "superobs" for superobservations.
        Returns
        -------
        pd.DataFrame
            DataFrame with computed scores for each variable and bin.
        """
        # log_memory("Class4Evaluator 1")
        all_scores = {}
        
        for var in variables:
            try:
                # logger.info(f"Processing variable: {var}")
                obs_da = obs_ds[var]
                model_da = model_ds[var]
                
                # Filter and apply QC if needed
                if self.apply_qc:
                    obs_da = filter_observations_by_qc(
                        ds=obs_da,
                        qc_mappings=self.qc_mapping,
                    )

                # Initialize grouping columns and observation/model columns
                groupby_cols = []
                obs_col = f"{var}_obs"
                model_col = f"{var}_model"

                # Interpolate model onto observations
                if matching_type == "nearest":
                    # Streaming implementation to avoid memory saturation
                    # We process data in chunks: Obs Chunk -> Bin -> Interpolate -> Accumulate Stats
                    
                    # 1. Initialize Stats Accumulators
                    # Key: Bin Tuple or Scalar, Value: {count, sum_sq, sum_abs, sum_err}
                    bin_stats = {}
                    groupby_cols = None
                    
                    # Caching for interpolation grid reuse across chunks
                    # Cleared per variable to avoid collision between variables (e.g. ssh vs sst)
                    interp_cache = {}

                    # 2. Iterate chunks
                    chunk_gen = xr_to_obs_dataframe(obs_da, include_geometry=False, yield_chunks=True)
                    
                    for i, chunk_df in enumerate(chunk_gen):
                        if chunk_df.empty:
                            continue
                        
                        # Create a copy to explicitly avoid SettingWithCopyWarning
                        chunk_df = chunk_df.copy()
                        
                        # Apply Binning
                        chunk_df, current_groupby = apply_binning(chunk_df, self.bin_specs)

                        if groupby_cols is None:
                            groupby_cols = current_groupby
                            
                        # Filter for variable and NaNs
                        if var not in chunk_df.columns:
                            # Try to find the data column (sometimes named "value" or "variable")
                            for col in ["value", "variable"]:
                                if col in chunk_df.columns:
                                    chunk_df = chunk_df.rename(columns={col: var})
                                    break
                        
                        if var not in chunk_df.columns:
                            continue 
                            
                        chunk_df = chunk_df.dropna(subset=[var]).copy()
                        if chunk_df.empty:
                            continue
                        
                        # Interpolate expects the variable column to exist in proper format
                        # And typically interpolate_model_on_obs uses just coords
                        
                        # Interpolate
                        chunk_df = interpolate_model_on_obs(
                            model_da, chunk_df, var, method=self.interp_method, cache=interp_cache
                        )
                        
                        # Limit memory usage by clearing cache if it grows too large
                        # Reduce to very small size (1-2) to prevent memory buildup
                        # Since we stream chronologically, we typically only need the current slice
                        if len(interp_cache) > 2: 
                            interp_cache.clear()
                        
                        # Rename obs column for consistency
                        chunk_df = chunk_df.rename(columns={var: f"{var}_obs"})
                        
                        # Filter NaNs in Model and Obs
                        chunk_df = chunk_df.dropna(subset=[f"{var}_model", f"{var}_obs"])
                        if chunk_df.empty:
                            continue
                        
                        # Calculate Errors
                        obs_val = chunk_df[f"{var}_obs"].values
                        mod_val = chunk_df[f"{var}_model"].values
                        diff = mod_val - obs_val

                        # logger.info(f"Variable {var} Chunk {i}: size={len(chunk_df)}, binning={dt_bin:.2f}s, interp={dt_interp:.2f}s, total={dt_total:.2f}s")
                        
                        # Aggregate
                        
                        # Aggregate
                        # Prepare Aggregation DF (minimized)
                        agg_cols = list(groupby_cols) if groupby_cols else []
                        
                        if agg_cols:
                            if 'time_bin' not in chunk_df.columns:
                                logger.error(f"Column 'time_bin' missing from DataFrame. Available columns: {list(chunk_df.columns)}")
                            df_agg = chunk_df[agg_cols].copy()
                            df_agg["sq_err"] = diff ** 2
                            df_agg["abs_err"] = np.abs(diff)
                            df_agg["err"] = diff
                            df_agg["count"] = 1
                            
                            # Groupby and Sum
                            grouped = df_agg.groupby(agg_cols, observed=True, dropna=True).sum(numeric_only=True)
                            
                            for name, row in grouped.iterrows():
                                if name not in bin_stats:
                                    bin_stats[name] = {"count":0.0, "sum_sq":0.0, "sum_abs":0.0, "sum_err":0.0}
                                
                                s = bin_stats[name]
                                s["count"] += row["count"]
                                s["sum_sq"] += row["sq_err"]
                                s["sum_abs"] += row["abs_err"]
                                s["sum_err"] += row["err"]
                                
                            del df_agg, grouped
                        else:
                            # Global bin (no bins defined)
                            name = "global"
                            if name not in bin_stats:
                                 bin_stats[name] = {"count":0.0, "sum_sq":0.0, "sum_abs":0.0, "sum_err":0.0}
                            s = bin_stats[name]
                            s["count"] += len(diff)
                            s["sum_sq"] += np.sum(diff**2)
                            s["sum_abs"] += np.sum(np.abs(diff))
                            s["sum_err"] += np.sum(diff)
                            
                        del chunk_df, diff, obs_val, mod_val
                        gc.collect()

                    # Explicitly clear cache after processing all chunks for this variable
                    if interp_cache:
                        interp_cache.clear()
                    del interp_cache

                    # 3. Post-Process Stats into `scores_result`
                    bin_results = []
                    
                    for name, s in bin_stats.items():
                        n = s["count"]
                        if n == 0:
                            continue
                        
                        res = {}
                        # Add Bin Labels
                        if groupby_cols:
                            if len(groupby_cols) == 1:
                                res[groupby_cols[0]] = name
                            else:
                                for i, col in enumerate(groupby_cols):
                                    res[col] = name[i]
                        
                        # Calc Metrics 
                        res["rmse"] = np.sqrt(s["sum_sq"] / n)
                        res["mse"] = s["sum_sq"] / n
                        res["mae"] = s["sum_abs"] / n
                        res["bias"] = s["sum_err"] / n
                        res["me"] = res["bias"]
                        res["count"] = n
                        
                        bin_results.append(res)
                        
                    # Aggregate Global Stats
                    global_stats = {}
                    if bin_results:
                        df_res = pd.DataFrame(bin_results)
                        metric_cols = ["rmse", "mse", "mae", "bias", "me"]
                        for m in metric_cols:
                            if m in df_res.columns:
                                global_stats[f"{m}_mean"] = df_res[m].mean()
                                # global_stats[f"{m}_median"] = df_res[m].median()
                                global_stats[f"{m}_std"] = df_res[m].std()
                                
                    scores_result = {
                        "per_bins": bin_results,
                        "global": global_stats
                    }
                    
                    del bin_stats
                    gc.collect()

                elif matching_type == "superobs":
                    # Bin observations into superobs
                    superobs = make_superobs(obs_da, model_da, var, reduce="mean")
                    # Bin observations onto the model grid
                    obs_binned = superobs_binning(superobs, model_da, var=var)

                    # Conversion
                    binned_df = xr_to_obs_dataframe(
                        obs_binned, include_geometry=False
                    )
                    
                    if f"{var}_binned" in binned_df.columns:
                        binned_df = binned_df.dropna(subset=[f"{var}_binned"])
                        binned_df = binned_df.rename(columns={f"{var}_binned": f"{var}_obs"})

                    # Add model values to the dataframe
                    final_df = add_model_values(binned_df, model_da, var=var)
                    
                    scores_result = compute_scores_xskillscore(
                        df=final_df,
                        y_obs_col=obs_col,
                        y_pred_col=model_col,
                        metrics=self.metrics,
                        weights=None,
                        groupby=groupby_cols,
                    )
                else:
                    raise ValueError(f"Unknown matching_type: {matching_type}")
                # log_memory("Class4Evaluator 6")

                # Convert the result to a DataFrame if not already the case
                if isinstance(scores_result, dict):
                    # If it is a dictionary, convert it to DataFrame
                    scores_df = pd.DataFrame([scores_result])
                    scores_df['variable'] = var
                elif isinstance(scores_result, pd.DataFrame):
                    # Already a DataFrame, add the variable column
                    scores_df = scores_result.copy()
                    scores_df['variable'] = var
                else:
                    # For other types, create a basic DataFrame
                    scores_df = pd.DataFrame({
                        'variable': [var],
                        'result': [scores_result]
                    })
                
                all_scores[var] = scores_df
                
                # CRITICAL: Clean up large DataFrames immediately after use
                try:
                    del scores_result
                    gc.collect()
                except Exception:
                    pass
                gc.collect()
                
            except Exception as e:
                logger.error(f"Error processing variable {var}: {e}")
                import traceback
                traceback.print_exc()
                # Create an error DataFrame for this variable
                error_df = pd.DataFrame({
                    'variable': [var],
                    'error': [str(e)],
                    'status': ['failed']
                })
                all_scores[var] = error_df
                continue
        
        # log_memory("Class4Evaluator 7")
        # Concatenate DataFrames instead of a dictionary
        if all_scores:
            # Extraire les DataFrames du dictionnaire
            dataframes_to_concat = list(all_scores.values())
            
            # Verify that all elements are DataFrames
            valid_dataframes = []
            for df in dataframes_to_concat:
                if isinstance(df, pd.DataFrame):
                    valid_dataframes.append(df)
                else:
                    logger.warning(f"Non-DataFrame result found: {type(df)}")
            
            if valid_dataframes:
                final_result = pd.concat(valid_dataframes, ignore_index=True)
                grid_results = format_class4_results(final_result)
                # log_memory("Class4Evaluator 8")
                return grid_results
            else:
                logger.warning("No valid DataFrames to concatenate")
                return pd.DataFrame()  # Empty DataFrame
        else:
            logger.warning("No scores computed for any variable")
            return pd.DataFrame()  # Empty DataFrame

