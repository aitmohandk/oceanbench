'''from loguru import logger
import numpy as np
from oceanbench.core.class4_metrics.binning import apply_binning
import xarray as xr
from scipy.spatial import cKDTree
from scipy.interpolate import griddata

from datetime import timedelta
from typing import Union, Optional'''




import geopandas as gpd
from memory_profiler import profile
import numpy as np
import pandas as pd
import xarray as xr
from typing import Union, Optional, Literal
from scipy.spatial import cKDTree
import pyinterp
import pyinterp.backends.xarray
#import xesmf as xe



'''
def match_model_to_obs(
    model: xr.Dataset,
    observations: xr.Dataset,
    variables: list[str],
    method: str = "kdtree",
    vertical_interp: str = "nearest",
    nb_points_interp: int = 1,
    search_radius: float = None,
    obs_lon_name: str = "lon",
    obs_lat_name: str = "lat",
    model_lon_name: str = "lon",
    model_lat_name: str = "lat",
    obs_depth_name: str = "depth",
    model_depth_name: str = "depth",
    vertical_tolerance: float = 0.5,
    allow_extrapolation: bool = False,
    mask: xr.DataArray = None,
    return_indices: bool = False,
    **kwargs
) -> dict:
    """
    Match/interpolate model outputs to observation points for CLASS4 metrics.

    Parameters
    ----------
    model : xr.Dataset
        Model dataset (gridded).
    observations : xr.Dataset
        Observations dataset (point or profile).
    variables : list of str
        Variables to match/interpolate.
    method : str
        Horizontal interpolation method: 'kdtree', 'griddata', or 'nearest'.
    vertical_interp : str
        Vertical interpolation: 'nearest', 'linear', or 'none'.
    nb_points_interp : int
        Number of neighbors for KDTree (1=nearest, >1=IDW).
    search_radius : float, optional
        Maximum search radius (degrees or meters, depending on context).
    obs_lon_name, obs_lat_name : str
        Names of longitude/latitude in obs.
    model_lon_name, model_lat_name : str
        Names of longitude/latitude in model.
    obs_depth_name, model_depth_name : str
        Names of depth in obs/model (for 3D).
    vertical_tolerance : float
        Max allowed vertical distance (for vertical matching).
    allow_extrapolation : bool
        Allow extrapolation outside model grid.
    mask : xr.DataArray, optional
        Optional mask to apply to model grid before matching.
    return_indices : bool
        If True, also return indices of matched model points.
    **kwargs : dict
        Other parameters for advanced use.

    Returns
    -------
    matched : dict
        Dict of {var: xr.DataArray} with model values interpolated at obs points.
        If return_indices=True, also returns {var: (values, indices)}.
    """
    matched = {}
    # Get obs positions
    obs_lon = observations[obs_lon_name].values
    obs_lat = observations[obs_lat_name].values
    obs_depth = observations[obs_depth_name].values if obs_depth_name in observations else None

    # Get model grid
    model_lon = model[model_lon_name].values
    model_lat = model[model_lat_name].values
    model_depth = model[model_depth_name].values if model_depth_name in model else None

    # Flatten model grid for KDTree/griddata
    if model_lon.ndim == 2 and model_lat.ndim == 2:
        points_model = np.column_stack([model_lon.flatten(), model_lat.flatten()])
    else:
        lon2d, lat2d = np.meshgrid(model_lon, model_lat)
        points_model = np.column_stack([lon2d.flatten(), lat2d.flatten()])

    points_obs = np.column_stack([obs_lon.flatten(), obs_lat.flatten()])

    # Optional: apply mask to model grid
    if mask is not None:
        mask_flat = mask.values.flatten()
        valid_idx = np.where(mask_flat)[0]
        points_model = points_model[valid_idx]

    # --- NEW: handle 1D observations (e.g. only 'time') ---
    if len(observations.dims) == 1 and "time" in observations.dims:
        for var in variables:
            # On suppose que le modèle et l'observation ont la même coordonnée "time"
            # ou que l'on veut juste extraire la série temporelle du modèle
            if "time" in model[var].dims:
                # Alignement automatique sur le temps
                matched_arr = model[var].sel(time=observations["time"])
                matched[var] = matched_arr
            else:
                raise ValueError(f"Variable {var} in model has no 'time' dimension.")
        return matched

    for var in variables:
        model_var = model[var].values
        # 2D or 3D?
        if model_var.ndim == 2 or (model_var.ndim == 3 and model_depth is None):
            # 2D matching
            if method == "kdtree":
                tree = cKDTree(points_model)
                dist, idx = tree.query(points_obs, k=nb_points_interp)
                if search_radius is not None:
                    mask_dist = dist <= search_radius
                    idx = np.where(mask_dist, idx, -1)
                if nb_points_interp == 1:
                    matched_vals = model_var.flatten()[idx]
                else:
                    # Inverse distance weighting (IDW)
                    weights = 1 / (dist + 1e-12)
                    weights /= weights.sum(axis=1, keepdims=True)
                    vals = model_var.flatten()[idx]
                    matched_vals = (vals * weights).sum(axis=1)
            elif method == "griddata":
                matched_vals = griddata(points_model, model_var.flatten(), points_obs, method="linear")
                if allow_extrapolation:
                    nan_mask = np.isnan(matched_vals)
                    matched_vals[nan_mask] = griddata(points_model, model_var.flatten(), points_obs[nan_mask], method="nearest")
            elif method == "nearest":
                tree = cKDTree(points_model)
                _, idx = tree.query(points_obs, k=1)
                matched_vals = model_var.flatten()[idx]
            else:
                raise ValueError(f"Unknown method: {method}")
            matched_arr = xr.DataArray(matched_vals.reshape(obs_lon.shape), dims=observations[var].dims, coords=observations[var].coords)
            matched[var] = (matched_arr, idx) if return_indices else matched_arr

        elif model_var.ndim == 3 and model_depth is not None and obs_depth is not None:
            # 3D matching: horizontal first, then vertical
            matched_vals = np.full(obs_lon.shape, np.nan)
            idx_horiz = np.full(obs_lon.shape, -1)
            for i, (lon, lat, depth) in enumerate(zip(obs_lon.flatten(), obs_lat.flatten(), obs_depth.flatten())):
                # Horizontal
                tree = cKDTree(points_model)
                _, idx = tree.query([[lon, lat]], k=nb_points_interp)
                idx = idx[0][0] if nb_points_interp == 1 else idx[0]
                # Get vertical profile at matched horizontal point
                if mask is not None:
                    idx_model = valid_idx[idx]
                else:
                    idx_model = idx
                # Find vertical index
                if model_depth.ndim == 1:
                    vert_prof = model_depth
                else:
                    vert_prof = model_depth[:, idx_model]
                if vertical_interp == "nearest":
                    vert_idx = np.argmin(np.abs(vert_prof - depth))
                    if np.abs(vert_prof[vert_idx] - depth) <= vertical_tolerance:
                        matched_vals.flat[i] = model_var[vert_idx, ...].flatten()[idx_model]
                        idx_horiz.flat[i] = idx_model
                elif vertical_interp == "linear":
                    # Find two nearest depths
                    diffs = vert_prof - depth
                    if np.any(diffs == 0):
                        matched_vals.flat[i] = model_var[diffs == 0, ...].flatten()[idx_model]
                        idx_horiz.flat[i] = idx_model
                    else:
                        above = np.where(diffs > 0)[0]
                        below = np.where(diffs < 0)[0]
                        if above.size and below.size:
                            i_above = above[0]
                            i_below = below[-1]
                            d1, d2 = vert_prof[i_below], vert_prof[i_above]
                            v1, v2 = model_var[i_below, ...].flatten()[idx_model], model_var[i_above, ...].flatten()[idx_model]
                            w = (depth - d1) / (d2 - d1)
                            matched_vals.flat[i] = v1 * (1 - w) + v2 * w
                            idx_horiz.flat[i] = idx_model
                else:
                    raise ValueError(f"Unknown vertical_interp: {vertical_interp}")
            matched_arr = xr.DataArray(matched_vals.reshape(obs_lon.shape), dims=observations[var].dims, coords=observations[var].coords)
            matched[var] = (matched_arr, idx_horiz) if return_indices else matched_arr
        else:
            raise ValueError(f"Unsupported variable shape for {var}: {model_var.shape}")

    return matched
'''

'''def match_model_to_obs(
    model: xr.Dataset,
    observations: xr.Dataset,
    variables: list[str],
    method: str = "kdtree",
    vertical_interp: str = "nearest",
    nb_points_interp: int = 1,
    search_radius: float = None,
    obs_lon_name: str = "lon",
    obs_lat_name: str = "lat",
    model_lon_name: str = "lon",
    model_lat_name: str = "lat",
    obs_depth_name: str = "depth",
    model_depth_name: str = "depth",
    vertical_tolerance: float = 0.5,
    allow_extrapolation: bool = False,
    mask: xr.DataArray = None,
    return_indices: bool = False,
    time_tolerance: float = None,  # en jours, secondes, ou np.timedelta64 selon vos données
    **kwargs
) -> dict:
    """
    Match/interpolate model outputs to observation points for CLASS4 metrics.

    Parameters
    ----------
    model : xr.Dataset
        Model dataset (gridded).
    observations : xr.Dataset
        Observations dataset (point or profile).
    variables : list of str
        Variables to match/interpolate.
    method : str
        Horizontal interpolation method: 'kdtree', 'griddata', or 'nearest'.
    vertical_interp : str
        Vertical interpolation: 'nearest', 'linear', or 'none'.
    nb_points_interp : int
        Number of neighbors for KDTree (1=nearest, >1=IDW).
    search_radius : float, optional
        Maximum search radius (degrees or meters, depending on context).
    obs_lon_name, obs_lat_name : str
        Names of longitude/latitude in obs.
    model_lon_name, model_lat_name : str
        Names of longitude/latitude in model.
    obs_depth_name, model_depth_name : str
        Names of depth in obs/model (for 3D).
    vertical_tolerance : float
        Max allowed vertical distance (for vertical matching).
    allow_extrapolation : bool
        Allow extrapolation outside model grid.
    mask : xr.DataArray, optional
        Optional mask to apply to model grid before matching.
    return_indices : bool
        If True, also return indices of matched model points.
    time_tolerance : float or np.timedelta64, optional
        Maximum allowed time difference for matching (for 1D or N-D time matching).
    **kwargs : dict
        Other parameters for advanced use.

    Returns
    -------
    matched : dict
        Dict of {var: xr.DataArray} with model values interpolated at obs points.
        If return_indices=True, also returns {var: (values, indices)}.
    """
    matched = {}

    # Handle 1D observations (only 'time')
    if len(observations.dims) == 1 and "time" in observations.dims:
        for var in variables:
            if "time" in model[var].dims:
                obs_times = observations["time"].values
                model_times = model["time"].values
                if time_tolerance is not None:
                    matched_indices = []
                    for t in obs_times:
                        dt = np.abs(model_times - t)
                        min_idx = np.argmin(dt)
                        # if (dt[min_idx].total_seconds() / 3600) <= time_tolerance:   # difference in hours
                        if (dt[min_idx].astype('timedelta64[s]').astype(float) / 3600) <= time_tolerance:
                            matched_indices.append(min_idx)
                        else:
                            matched_indices.append(None)
                    matched_arr = xr.full_like(observations[var], np.nan)
                    for i, idx in enumerate(matched_indices):
                        if idx is not None:
                            matched_arr[i] = model[var].isel(time=idx)
                    matched[var] = matched_arr
                else:
                    matched_arr = model[var].sel(time=observations["time"])
                    matched[var] = matched_arr
            else:
                raise ValueError(f"Variable {var} in model has no 'time' dimension.")
        return matched

    # Handle N-D observations with 'time' (e.g. (time, lat, lon), (time, lat, lon, depth))
    for var in variables:
        obs_var = observations[var]
        model_var = model[var]
        if "time" in obs_var.dims and "time" in model_var.dims and time_tolerance is not None:
            obs_times = obs_var["time"].values
            model_times = model_var["time"].values
            # Build index array for each obs time
            matched_indices = []
            for t in obs_times:
                dt = np.abs(model_times - t)
                min_idx = np.argmin(dt)
                # if (dt[min_idx].total_seconds() / 3600) <= time_tolerance:  # difference in hours
                if (dt[min_idx].astype('timedelta64[s]').astype(float) / 3600) <= time_tolerance:
                    matched_indices.append(min_idx)
                else:
                    matched_indices.append(None)
            # Now, for each obs point, select the model time if within tolerance, else NaN
            # We need to broadcast this selection to all spatial dims
            obs_shape = obs_var.shape
            matched_arr = xr.full_like(obs_var, np.nan)
            # Assume time is the first dim
            for i, idx in enumerate(matched_indices):
                if idx is not None:
                    # Select the model slice at time idx and assign to matched_arr at time i
                    matched_arr[i] = model_var.isel(time=idx).values
            matched[var] = matched_arr
            continue  # skip to next variable

        # --- Standard spatial/vertical matching (original code) ---
        # Get obs positions
        obs_lon = observations[obs_lon_name].values
        obs_lat = observations[obs_lat_name].values
        obs_depth = observations[obs_depth_name].values if obs_depth_name in observations else None

        # Get model grid
        model_lon = model[model_lon_name].values
        model_lat = model[model_lat_name].values
        model_depth = model[model_depth_name].values if model_depth_name in model else None

        # Flatten model grid for KDTree/griddata
        if model_lon.ndim == 2 and model_lat.ndim == 2:
            points_model = np.column_stack([model_lon.flatten(), model_lat.flatten()])
        else:
            lon2d, lat2d = np.meshgrid(model_lon, model_lat)
            points_model = np.column_stack([lon2d.flatten(), lat2d.flatten()])

        points_obs = np.column_stack([obs_lon.flatten(), obs_lat.flatten()])

        # Optional: apply mask to model grid
        if mask is not None:
            mask_flat = mask.values.flatten()
            valid_idx = np.where(mask_flat)[0]
            points_model = points_model[valid_idx]

        # 2D or 3D?
        model_var_np = model_var.values
        if model_var_np.ndim == 2 or (model_var_np.ndim == 3 and model_depth is None):
            # 2D matching
            if method == "kdtree":
                tree = cKDTree(points_model)
                dist, idx = tree.query(points_obs, k=nb_points_interp)
                if search_radius is not None:
                    mask_dist = dist <= search_radius
                    idx = np.where(mask_dist, idx, -1)
                if nb_points_interp == 1:
                    matched_vals = model_var_np.flatten()[idx]
                else:
                    # Inverse distance weighting (IDW)
                    weights = 1 / (dist + 1e-12)
                    weights /= weights.sum(axis=1, keepdims=True)
                    vals = model_var_np.flatten()[idx]
                    matched_vals = (vals * weights).sum(axis=1)
            elif method == "griddata":
                matched_vals = griddata(points_model, model_var_np.flatten(), points_obs, method="linear")
                if allow_extrapolation:
                    nan_mask = np.isnan(matched_vals)
                    matched_vals[nan_mask] = griddata(points_model, model_var_np.flatten(), points_obs[nan_mask], method="nearest")
            elif method == "nearest":
                tree = cKDTree(points_model)
                _, idx = tree.query(points_obs, k=1)
                matched_vals = model_var_np.flatten()[idx]
            else:
                raise ValueError(f"Unknown method: {method}")
            matched_arr = xr.DataArray(matched_vals.reshape(obs_lon.shape), dims=observations[var].dims, coords=observations[var].coords)
            matched[var] = (matched_arr, idx) if return_indices else matched_arr

        elif model_var_np.ndim == 3 and model_depth is not None and obs_depth is not None:
            # 3D matching: horizontal first, then vertical
            matched_vals = np.full(obs_lon.shape, np.nan)
            idx_horiz = np.full(obs_lon.shape, -1)
            for i, (lon, lat, depth) in enumerate(zip(obs_lon.flatten(), obs_lat.flatten(), obs_depth.flatten())):
                # Horizontal
                tree = cKDTree(points_model)
                _, idx = tree.query([[lon, lat]], k=nb_points_interp)
                idx = idx[0][0] if nb_points_interp == 1 else idx[0]
                # Get vertical profile at matched horizontal point
                if mask is not None:
                    idx_model = valid_idx[idx]
                else:
                    idx_model = idx
                # Find vertical index
                if model_depth.ndim == 1:
                    vert_prof = model_depth
                else:
                    vert_prof = model_depth[:, idx_model]
                if vertical_interp == "nearest":
                    vert_idx = np.argmin(np.abs(vert_prof - depth))
                    if np.abs(vert_prof[vert_idx] - depth) <= vertical_tolerance:
                        matched_vals.flat[i] = model_var_np[vert_idx, ...].flatten()[idx_model]
                        idx_horiz.flat[i] = idx_model
                elif vertical_interp == "linear":
                    # Find two nearest depths
                    diffs = vert_prof - depth
                    if np.any(diffs == 0):
                        matched_vals.flat[i] = model_var_np[diffs == 0, ...].flatten()[idx_model]
                        idx_horiz.flat[i] = idx_model
                    else:
                        above = np.where(diffs > 0)[0]
                        below = np.where(diffs < 0)[0]
                        if above.size and below.size:
                            i_above = above[0]
                            i_below = below[-1]
                            d1, d2 = vert_prof[i_below], vert_prof[i_above]
                            v1, v2 = model_var_np[i_below, ...].flatten()[idx_model], model_var_np[i_above, ...].flatten()[idx_model]
                            w = (depth - d1) / (d2 - d1)
                            matched_vals.flat[i] = v1 * (1 - w) + v2 * w
                            idx_horiz.flat[i] = idx_model
                else:
                    raise ValueError(f"Unknown vertical_interp: {vertical_interp}")
            matched_arr = xr.DataArray(matched_vals.reshape(obs_lon.shape), dims=observations[var].dims, coords=observations[var].coords)
            matched[var] = (matched_arr, idx_horiz) if return_indices else matched_arr
        else:
            raise ValueError(f"Unsupported variable shape for {var}: {model_var_np.shape}")

    return matched'''

'''def match_model_to_obs(
    model: xr.Dataset,
    obs: xr.Dataset,
    variables: list[str],
    coord_level: str,  # "grid_3d", "grid_2d", "point_3d", "point_2d"
    time_tolerance: float = 0.5,  # in days
    interp_method: str = "kdtree",  # "kdtree", "griddata", "nearest"
    mask: xr.DataArray = None,
    qc_names: list[str] = None,
    good_qc: int | list[int] = 0,
    obs_time_name: str = "time",
    model_time_name: str = "time",
    nb_points_interp: int = 1,
    search_radius: float = None,
    obs_lon_name: str = "lon",
    obs_lat_name: str = "lat",
    model_lon_name: str = "lon",
    model_lat_name: str = "lat",
    obs_depth_name: str = "depth",
    model_depth_name: str = "depth",
    vertical_interp: str = "nearest",  # "nearest", "linear", "none"
    vertical_tolerance: float = 0.5,
    allow_extrapolation: bool = False,
    return_indices: bool = False,
    **kwargs
) -> dict:
    """
    Match/interpolate model outputs to observation points, with configurable options.

    Parameters
    ----------
    ... (identique à la version précédente, voir plus haut)
    allow_extrapolation : bool
        If True, allow extrapolation outside model grid (for griddata).
    ...
    """
    matched = {}
    obs_times = obs[obs_time_name].values
    model_times = model[model_time_name].values

    obs_times64 = np.array(obs_times, dtype='datetime64[ns]')
    model_times64 = np.array(model_times, dtype='datetime64[ns]')


    # For each observation, find the closest model time within tolerance
    time_indices = []
    for t_obs in obs_times64:
        dt = np.abs(model_times64 - t_obs)
        min_dt = dt.min()
        idx = dt.argmin()
        tol_ns = np.timedelta64(int(time_tolerance * 24 * 3600 * 1e9), 'ns')
        if min_dt <= tol_ns:
            time_indices.append(idx)
        else:
            time_indices.append(None)

    # QC mask
    if qc_names is not None:
        qc_mask = np.ones(obs[variables[0]].shape, dtype=bool)
        for qc_var in qc_names:
            if qc_var in obs:
                qc = obs[qc_var].values
                if isinstance(good_qc, (list, tuple, np.ndarray)):
                    qc_mask &= np.isin(qc, good_qc)
                else:
                    qc_mask &= (qc == good_qc)
        obs_mask = qc_mask
    else:
        obs_mask = np.ones(obs[variables[0]].shape, dtype=bool)

    # Prepare model grid points
    model_lon = model[model_lon_name].values
    model_lat = model[model_lat_name].values
    if mask is not None:
        mask_flat = mask.values.flatten()
        valid_idx = np.where(mask_flat)[0]
        if model_lon.ndim == 2 and model_lat.ndim == 2:
            points_model = np.column_stack([model_lon.flatten()[valid_idx], model_lat.flatten()[valid_idx]])
        else:
            lon2d, lat2d = np.meshgrid(model_lon, model_lat)
            points_model = np.column_stack([lon2d.flatten()[valid_idx], lat2d.flatten()[valid_idx]])
    else:
        if model_lon.ndim == 2 and model_lat.ndim == 2:
            points_model = np.column_stack([model_lon.flatten(), model_lat.flatten()])
        else:
            lon2d, lat2d = np.meshgrid(model_lon, model_lat)
            points_model = np.column_stack([lon2d.flatten(), lat2d.flatten()])

    # Prepare obs positions (handle 1D and 2D cases)
    if coord_level in ["point_2d", "point_3d"]:
        obs_lon = obs[obs_lon_name].values
        obs_lat = obs[obs_lat_name].values
        if obs_lon.ndim == 1 and obs_lat.ndim == 1:
            points_obs = np.column_stack([obs_lon, obs_lat])
        else:
            points_obs = np.column_stack([obs_lon.flatten(), obs_lat.flatten()])
        obs_depth = obs[obs_depth_name].values if (coord_level == "point_3d" and obs_depth_name in obs) else None
    elif coord_level in ["grid_2d", "grid_3d"]:
        obs_lon = obs[obs_lon_name].values
        obs_lat = obs[obs_lat_name].values
        if obs_lon.ndim == 2 and obs_lat.ndim == 2:
            points_obs = np.column_stack([obs_lon.flatten(), obs_lat.flatten()])
        else:
            lon2d, lat2d = np.meshgrid(obs_lon, obs_lat)
            points_obs = np.column_stack([lon2d.flatten(), lat2d.flatten()])
        obs_depth = obs[obs_depth_name].values if (coord_level == "grid_3d" and obs_depth_name in obs) else None
    else:
        raise ValueError(f"Unknown coord_level: {coord_level}")

    for var in variables:
        # Handle 1D obs (e.g. only time)
        if obs[var].ndim == 1 and obs[var].dims == (obs_time_name,):
            obs_ssh = obs[var]  ### TODO: REMOVE THIS
            matched_vals = np.full(obs[var].shape, np.nan)
            matched_time_idx = np.full(obs[var].shape, -1)
            for i, idx_time in enumerate(time_indices):
                if idx_time is None or not obs_mask[i]:
                    continue
                model_slice = model.isel({model_time_name: idx_time})
                model_var = model_slice[var].values
                if model_var.ndim == 1:
                    matched_vals[i] = model_var[idx_time]
                    matched_time_idx[i] = idx_time
                else:
                    matched_vals[i] = np.nanmean(model_var)
                    matched_time_idx[i] = idx_time
            matched_arr = xr.DataArray(matched_vals, dims=obs[var].dims, coords=obs[var].coords)
            if return_indices:
                matched[var] = (matched_arr, matched_time_idx)
            else:
                matched[var] = matched_arr
            continue

        matched_vals = np.full(obs[var].shape, np.nan)
        matched_time_idx = np.full(obs[var].shape, -1)
        for i, idx_time in enumerate(time_indices):
            if idx_time is None or not obs_mask.flat[i]:
                continue
            model_slice = model.isel({model_time_name: idx_time})
            model_var = model_slice[var].values
            # 2D or 3D?
            if model_var.ndim == 2 or (model_var.ndim == 3 and model_depth_name not in model):
                # 2D matching
                if interp_method == "kdtree":
                    tree = cKDTree(points_model)
                    dist, idx = tree.query(points_obs[i], k=nb_points_interp)
                    if search_radius is not None and dist > search_radius:
                        continue
                    if mask is not None:
                        idx_model = valid_idx[idx]
                    else:
                        idx_model = idx
                    matched_vals.flat[i] = model_var.flatten()[idx_model]
                    matched_time_idx.flat[i] = idx_time
                elif interp_method == "griddata":
                    # griddata returns nan if outside convex hull unless allow_extrapolation=True
                    interp_val = griddata(
                        points_model,
                        model_var.flatten(),
                        points_obs[i],
                        method="linear",
                        fill_value=np.nan if not allow_extrapolation else None,
                        rescale=False,
                    )
                    if np.isnan(interp_val) and allow_extrapolation:
                        interp_val = griddata(
                            points_model,
                            model_var.flatten(),
                            points_obs[i],
                            method="nearest",
                            fill_value=np.nan,
                            rescale=False,
                        )
                    matched_vals.flat[i] = interp_val
                    matched_time_idx.flat[i] = idx_time
                elif interp_method == "nearest":
                    tree = cKDTree(points_model)
                    _, idx = tree.query(points_obs[i], k=1)
                    if mask is not None:
                        idx_model = valid_idx[idx]
                    else:
                        idx_model = idx
                    matched_vals.flat[i] = model_var.flatten()[idx_model]
                    matched_time_idx.flat[i] = idx_time
            elif model_var.ndim == 3 and model_depth_name in model and obs_depth is not None:
                # 3D matching: horizontal first, then vertical
                tree = cKDTree(points_model)
                dist, idx = tree.query(points_obs[i], k=nb_points_interp)
                if search_radius is not None and dist > search_radius:
                    continue
                if mask is not None:
                    idx_model = valid_idx[idx]
                else:
                    idx_model = idx
                model_depth = model[model_depth_name].values
                if model_depth.ndim == 1:
                    vert_prof = model_depth
                else:
                    vert_prof = model_depth[:, idx_model]
                obs_z = obs_depth.flat[i] if obs_depth is not None else 0
                if vertical_interp == "nearest":
                    vert_idx = np.argmin(np.abs(vert_prof - obs_z))
                    if np.abs(vert_prof[vert_idx] - obs_z) <= vertical_tolerance:
                        matched_vals.flat[i] = model_var[vert_idx, ...].flatten()[idx_model]
                        matched_time_idx.flat[i] = idx_time
                elif vertical_interp == "linear":
                    diffs = vert_prof - obs_z
                    if np.any(diffs == 0):
                        matched_vals.flat[i] = model_var[diffs == 0, ...].flatten()[idx_model]
                        matched_time_idx.flat[i] = idx_time
                    else:
                        above = np.where(diffs > 0)[0]
                        below = np.where(diffs < 0)[0]
                        if above.size and below.size:
                            i_above = above[0]
                            i_below = below[-1]
                            d1, d2 = vert_prof[i_below], vert_prof[i_above]
                            v1, v2 = model_var[i_below, ...].flatten()[idx_model], model_var[i_above, ...].flatten()[idx_model]
                            w = (obs_z - d1) / (d2 - d1)
                            matched_vals.flat[i] = v1 * (1 - w) + v2 * w
                            matched_time_idx.flat[i] = idx_time
        matched_arr = xr.DataArray(matched_vals.reshape(obs[var].shape), dims=obs[var].dims, coords=obs[var].coords)
        if return_indices:
            matched[var] = (matched_arr, matched_time_idx)
        else:
            matched[var] = matched_arr

    return matched'''


'''import numpy as np
import xarray as xr
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
import pandas as pd

def match_model_to_obs(
    model: xr.Dataset,
    observations: xr.Dataset,
    variables: list[str],
    coord_level: str,
    time_tolerance: float = 3600,  # en secondes
    interp_method: str = "kdtree",
    mask: xr.DataArray = None,
    qc_names: list[str] = None,
    good_qc: int | list[int] = 0,
    obs_time_name: str = "time",
    model_time_name: str = "time",
    nb_points_interp: int = 1,
    search_radius: float = None,
    obs_lon_name: str = "lon",
    obs_lat_name: str = "lat",
    model_lon_name: str = "lon",
    model_lat_name: str = "lat",
    obs_depth_name: str = "depth",
    model_depth_name: str = "depth",
    vertical_interp: str = "nearest",
    vertical_tolerance: float = 0.5,
    allow_extrapolation: bool = False,
    return_indices: bool = False,
    **kwargs
) -> dict:
    """
    Interpolates/matches model outputs to observation points for CLASS4 metrics.
    """
    matched = {}

    # 1. Préparation des coordonnées
    obs_times = pd.to_datetime(observations[obs_time_name].values)
    model_times = pd.to_datetime(model[model_time_name].values)
    obs_lon = observations[obs_lon_name].values
    obs_lat = observations[obs_lat_name].values
    obs_depth = observations[obs_depth_name].values if obs_depth_name in observations else None

    model_lon = model[model_lon_name].values
    model_lat = model[model_lat_name].values
    model_depth = model[model_depth_name].values if model_depth_name in model else None

    # 2. Matching temporel vectorisé (tolérance en secondes)
    tol = pd.Timedelta(seconds=time_tolerance*3600)
    model_time_idx = []
    for t_obs in obs_times:
        dt = np.abs(model_times - t_obs)
        idx = dt.argmin()
        min_dt = dt[idx]
        if min_dt <= tol:
            model_time_idx.append(idx)
        else:
            model_time_idx.append(None)

    # 3. Pour chaque variable, interpolation spatiale (et verticale si besoin)
    for var in variables:
        # Pour chaque observation, on extrait le modèle au temps le plus proche (si dans la tolérance)
        matched_vals = np.full(obs_lon.shape, np.nan)
        for i, idx_time in enumerate(model_time_idx):
            if idx_time is None:
                continue
            # Sélection du modèle à l'instant le plus proche
            model_slice = model.isel({model_time_name: idx_time})
            # Interpolation spatiale
            if model_lon.ndim == 2 and model_lat.ndim == 2:
                points_model = np.column_stack([model_lon.flatten(), model_lat.flatten()])
            else:
                lon2d, lat2d = np.meshgrid(model_lon, model_lat)
                points_model = np.column_stack([lon2d.flatten(), lat2d.flatten()])
            point_obs = np.array([[obs_lon.flat[i], obs_lat.flat[i]]])
            # Masquage éventuel
            if mask is not None:
                mask_flat = mask.values.flatten()
                valid_idx = np.where(mask_flat)[0]
                points_model = points_model[valid_idx]
                model_var = model_slice[var].values.flatten()[valid_idx]
            else:
                model_var = model_slice[var].values.flatten()
            # Interpolation
            if interp_method == "kdtree":
                tree = cKDTree(points_model)
                dist, idx = tree.query(point_obs, k=nb_points_interp)
                if search_radius is not None and dist[0][0] > search_radius:
                    continue

                if np.isscalar(model_var):
                    matched_val = model_var
                elif nb_points_interp == 1:
                    # idx peut être un scalaire ou un tableau
                    idx_val = idx[0][0] if isinstance(idx[0], (np.ndarray, list)) else idx[0]
                    matched_val = model_var[idx_val]
                else:
                    weights = 1 / (dist[0] + 1e-12)
                    weights /= weights.sum()
                    matched_val = (model_var[idx[0]] * weights).sum()
            elif interp_method == "griddata":
                matched_val = griddata(points_model, model_var, point_obs, method="linear")
                if allow_extrapolation and np.isnan(matched_val):
                    matched_val = griddata(points_model, model_var, point_obs, method="nearest")
                matched_val = matched_val[0]
            elif interp_method == "nearest":
                tree = cKDTree(points_model)
                _, idx = tree.query(point_obs, k=1)
                matched_val = model_var[idx[0][0]]
            else:
                raise ValueError(f"Unknown interp_method: {interp_method}")
            # Interpolation verticale si besoin
            if model_depth is not None and obs_depth is not None:
                obs_z = obs_depth.flat[i]
                if model_depth.ndim == 1:
                    vert_prof = model_depth
                else:
                    vert_prof = model_depth[:, idx[0][0]]
                vert_idx = np.argmin(np.abs(vert_prof - obs_z))
                if np.abs(vert_prof[vert_idx] - obs_z) <= vertical_tolerance:
                    matched_val = model_slice[var].isel({model_depth_name: vert_idx}).values.flatten()[idx[0][0]]
                else:
                    matched_val = np.nan
            matched_vals.flat[i] = matched_val

        # Création du DataArray aligné sur les observations
        matched[var] = xr.DataArray(
            matched_vals.reshape(obs_lon.shape),
            dims=observations[var].dims,
            coords=observations[var].coords,
            attrs=observations[var].attrs,
        )

    return matched'''

'''
import numpy as np
import xarray as xr
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
import pandas as pd
import dask.array as da

def match_model_to_obs(
    model: xr.Dataset,
    observations: xr.Dataset,
    variables: list[str],
    coord_level: str,
    time_tolerance: float = 3600,  # tolérance en secondes
    interp_method: str = "kdtree",
    mask: xr.DataArray = None,
    nb_points_interp: int = 1,
    search_radius: float = None,
    obs_lon_name: str = "lon",
    obs_lat_name: str = "lat",
    model_lon_name: str = "lon",
    model_lat_name: str = "lat",
    obs_depth_name: str = "depth",
    model_depth_name: str = "depth",
    vertical_interp: str = "nearest",
    vertical_tolerance: float = 0.5,
    allow_extrapolation: bool = False,
    return_indices: bool = False,
    parallel: bool = True,  # Active la parallélisation Dask
    chunk_size: int = 100_000,  # Taille des chunks pour Dask
    **kwargs
) -> dict:
    """
    Vectorized and optionally parallel matching/interpolation of model outputs to observation points.
    """
    matched = {}

    # 1. Préparation des coordonnées
    obs_times = pd.to_datetime(observations["time"].values)
    model_times = pd.to_datetime(model["time"].values)
    obs_lon = observations[obs_lon_name].values
    obs_lat = observations[obs_lat_name].values
    obs_depth = observations[obs_depth_name].values if obs_depth_name in observations else None

    model_lon = model[model_lon_name].values
    model_lat = model[model_lat_name].values
    model_depth = model[model_depth_name].values if model_depth_name in model else None

    # Convertir explicitement en numpy array
    obs_times_np = np.array(obs_times)
    model_times_np = np.array(model_times)

    # Matching temporel vectorisé (tolérance en secondes)
    tol = pd.Timedelta(seconds=time_tolerance*3600)
    dt_matrix = np.abs(model_times_np[None, :] - obs_times_np[:, None])
    idx_min = dt_matrix.argmin(axis=1)
    min_dt = np.take_along_axis(dt_matrix, idx_min[:, None], axis=1).flatten()
    model_time_idx = np.where(min_dt <= tol, idx_min, -1) 

    # 3. Préparation des points pour la spatialisation
    if model_lon.ndim == 2 and model_lat.ndim == 2:
        points_model = np.column_stack([model_lon.flatten(), model_lat.flatten()])
    else:
        lon2d, lat2d = np.meshgrid(model_lon, model_lat)
        points_model = np.column_stack([lon2d.flatten(), lat2d.flatten()])

    points_obs = np.column_stack([obs_lon.flatten(), obs_lat.flatten()])

    # Masquage éventuel
    if mask is not None:
        mask_flat = mask.values.flatten()
        valid_idx = np.where(mask_flat)[0]
        points_model = points_model[valid_idx]

    # 4. Vectorisation spatiale (et verticale)
    for var in variables:
        # Pour chaque observation, on extrait le modèle au temps le plus proche (si dans la tolérance)
        matched_vals = np.full(obs_lon.shape, np.nan)
        if parallel:
            # Utilisation de Dask pour paralléliser sur les observations
            obs_indices = np.arange(obs_lon.size)
            obs_indices_da = da.from_array(obs_indices, chunks=chunk_size)

            def process_obs_chunk(obs_idx_chunk):
                chunk_vals = np.full(obs_idx_chunk.shape, np.nan)
                for j, i in enumerate(obs_idx_chunk):
                    idx_time = model_time_idx[i]
                    if idx_time == -1:
                        continue
                    model_slice = model.isel(time=idx_time)
                    model_var = model_slice[var].values
                    # 2D matching
                    if interp_method == "kdtree":
                        tree = cKDTree(points_model)
                        dist, idx = tree.query(points_obs[i], k=nb_points_interp)
                        if search_radius is not None and dist > search_radius:
                            continue
                        if nb_points_interp == 1:
                            chunk_vals[j] = model_var.flatten()[idx]
                        else:
                            weights = 1 / (dist + 1e-12)
                            weights /= weights.sum()
                            chunk_vals[j] = (model_var.flatten()[idx] * weights).sum()
                    elif interp_method == "griddata":
                        chunk_vals[j] = griddata(points_model, model_var.flatten(), points_obs[i], method="linear")
                        if allow_extrapolation and np.isnan(chunk_vals[j]):
                            chunk_vals[j] = griddata(points_model, model_var.flatten(), points_obs[i], method="nearest")
                    elif interp_method == "nearest":
                        tree = cKDTree(points_model)
                        _, idx = tree.query(points_obs[i], k=1)
                        chunk_vals[j] = model_var.flatten()[idx]
                return chunk_vals

            matched_vals = obs_indices_da.map_blocks(process_obs_chunk, dtype=float).compute()
            matched_vals = matched_vals.reshape(obs_lon.shape)
        else:
            # Version vectorisée (sans Dask)
            for i in range(obs_lon.size):
                idx_time = model_time_idx[i]
                if idx_time == -1:
                    continue
                model_slice = model.isel(time=idx_time)
                model_var = model_slice[var].values
                # 2D matching
                if interp_method == "kdtree":
                    tree = cKDTree(points_model)
                    dist, idx = tree.query(points_obs[i], k=nb_points_interp)
                    if search_radius is not None and dist > search_radius:
                        continue
                    if nb_points_interp == 1:
                        matched_vals.flat[i] = model_var.flatten()[idx]
                    else:
                        weights = 1 / (dist + 1e-12)
                        weights /= weights.sum()
                        matched_vals.flat[i] = (model_var.flatten()[idx] * weights).sum()
                elif interp_method == "griddata":
                    matched_vals.flat[i] = griddata(points_model, model_var.flatten(), points_obs[i], method="linear")
                    if allow_extrapolation and np.isnan(matched_vals.flat[i]):
                        matched_vals.flat[i] = griddata(points_model, model_var.flatten(), points_obs[i], method="nearest")
                elif interp_method == "nearest":
                    tree = cKDTree(points_model)
                    _, idx = tree.query(points_obs[i], k=1)
                    matched_vals.flat[i] = model_var.flatten()[idx]
        matched_arr = xr.DataArray(
            matched_vals.reshape(obs_lon.shape),
            dims=observations[var].dims,
            coords=observations[var].coords,
            attrs=observations[var].attrs,
        )
        matched[var] = matched_arr

    return matched'''



'''

import pandas as pd
import numpy as np
import xarray as xr
from typing import Literal, Union
from scipy.spatial import cKDTree
import pyinterp
import pyinterp.backends.xarray
import xesmf as xe
import geopandas as gpd
import dask
from dask import delayed


def idw_interpolate(x, y, z, xi, yi, power=2):
    """Simple inverse distance weighting interpolation."""
    dist = np.sqrt((x[:, None] - xi[None, :])**2 + (y[:, None] - yi[None, :])**2)
    weights = 1 / (dist**power + 1e-12)
    weights /= weights.sum(axis=0)
    return np.dot(z, weights)


def match_model_to_obs(
    model_ds: xr.Dataset,
    obs_df: Union[pd.DataFrame, gpd.GeoDataFrame],
    varname: str,
    delta_t: pd.Timedelta = pd.Timedelta("12h"),
    method: Literal["pyinterp", "kdtree", "xesmf", "idw"] = "pyinterp",
    return_format: Literal["dataframe", "dataset", "geodataframe"] = "dataset",
    mask: Optional[Union[np.ndarray, pd.Series]] = None,
    delayed_mode: bool = False,
    binning: Optional[dict] = None,
) -> Union[pd.DataFrame, xr.Dataset, gpd.GeoDataFrame, dask.delayed]:
    """
    Vectorized, optionally parallel, co-localization between model and observations.

    Parameters
    ----------
    obs_df : DataFrame or GeoDataFrame
    model_ds : xr.Dataset
    varname : str
    delta_t : pd.Timedelta
    method : str
        Interpolation method: "pyinterp", "kdtree", "xesmf", "idw"
    return_format : str
        Output format: "dataframe", "dataset", or "geodataframe"
    mask : np.ndarray or pd.Series, optional
        Boolean mask applied to the final output
    delayed_mode : bool
        If True, returns a dask.delayed object instead of computing

    Returns
    -------
    Interpolated and compared data as specified by return_format
    """
    if isinstance(delta_t, (int, float)):
        delta_t = pd.Timedelta(hours=delta_t)
    if delayed_mode:
        return delayed(match_model_to_obs)(
            obs_df, model_ds, varname, delta_t, method, return_format, False
        )

    obs_df = obs_df.copy()
    model_times = pd.to_datetime(model_ds["time"].values)

    obs_df["matched_time"] = pd.NaT
    obs_df["model_value"] = np.nan
    obs_df["error"] = np.nan

    grouped = obs_df.groupby(obs_df["time"].dt.floor("1D"))

    for day, group in grouped:
        times = pd.to_datetime(group["time"])
        lats = group["lat"].to_numpy()
        lons = group["lon"].to_numpy()
        values = group[varname].to_numpy()

        matched_times = []
        for t in times:
            dt = abs(model_times - t)
            matched_times.append(model_times[np.argmin(dt)] if dt.min() <= delta_t else None)
        matched_times = np.array(matched_times)

        for unique_time in pd.unique(matched_times):
            if pd.isnull(unique_time):
                continue
            mask_t = matched_times == unique_time
            if not np.any(mask_t):
                continue

            sub_lats = lats[mask_t]
            sub_lons = lons[mask_t]
            sub_obs = values[mask_t]

            model_slice = model_ds.sel(time=unique_time, method="nearest")

            if method == "pyinterp":
                interp = pyinterp.backends.xarray.Grid2D(
                    longitude=model_slice["lon"],
                    latitude=model_slice["lat"],
                    values=model_slice[varname],
                )
                model_vals = interp.interpolate(sub_lons, sub_lats)

            elif method == "kdtree":
                lon2d, lat2d = np.meshgrid(model_slice["lon"].values, model_slice["lat"].values)
                points = np.column_stack([lon2d.ravel(), lat2d.ravel()])
                values_grid = model_slice[varname].values.ravel()
                tree = cKDTree(points)
                _, idx = tree.query(np.column_stack([sub_lons, sub_lats]))
                model_vals = values_grid[idx]

            elif method == "xesmf":
                regridder = xe.Regridder(
                    model_slice[varname].to_dataset(name=varname),
                    xr.Dataset({"lat": (["points"], sub_lats), "lon": (["points"], sub_lons)}),
                    method="bilinear",
                    periodic=False,
                    reuse_weights=True,
                )
                model_vals = regridder(model_slice[varname]).values

            elif method == "idw":
                lon2d, lat2d = np.meshgrid(model_slice["lon"].values, model_slice["lat"].values)
                model_vals = idw_interpolate(
                    lon2d.ravel(), lat2d.ravel(), model_slice[varname].values.ravel(),
                    sub_lons, sub_lats
                )

            else:
                raise ValueError(f"Unknown method: {method}")

            obs_df.loc[group.index[mask_t], "matched_time"] = unique_time
            obs_df.loc[group.index[mask_t], "model_value"] = model_vals
            obs_df.loc[group.index[mask_t], "error"] = model_vals - sub_obs

    obs_df = obs_df.dropna(subset=["model_value"])


    if mask is not None:
        obs_df = obs_df.loc[mask]


    if binning:
        obs_df = apply_binning(obs_df, binning)
        # Détecter les colonnes de bin et les dimensions standards
        bin_cols = [col for col in obs_df.columns if col.endswith("_bin")]
        rename_dict = {"time_bin": "time", "lat_bin": "lat", "lon_bin": "lon", "depth_bin": "depth"}

        # Renommer les colonnes de bin
        obs_df = obs_df.rename(columns={k: v for k, v in rename_dict.items() if k in obs_df.columns})


        # Définir l'index multi-dimensionnel
        index_cols = [v for k, v in rename_dict.items() if v in obs_df.columns]
        if index_cols:
            obs_df = obs_df.drop_duplicates(subset=index_cols)
            obs_df = obs_df.set_index(index_cols)


    if return_format == "dataframe":
        return obs_df
    elif return_format == "geodataframe":
        return gpd.GeoDataFrame(obs_df, geometry=gpd.points_from_xy(obs_df.lon, obs_df.lat), crs="EPSG:4326")
    elif return_format == "dataset":
        logger.debug(f"BINS: {obs_df}")  #.to_markdown()}")
        # Détecter toutes les colonnes de dimensions pertinentes
        possible_dims = ["time", "lat", "lon", "depth"]
        columns = obs_df.columns
        index_cols = [col for col in possible_dims if col in obs_df.columns]
        if index_cols:
            return xr.Dataset.from_dataframe(obs_df.set_index(index_cols))
        else:
            return xr.Dataset.from_dataframe(obs_df)
        # return xr.Dataset.from_dataframe(obs_df.set_index("time"))
    else:
        raise ValueError(f"Invalid return_format: {return_format}")'''


@profile
def match_model_to_obs(
    model_ds: xr.Dataset,
    obs_df: pd.DataFrame,
    varname: str,
    delta_t: pd.Timedelta = pd.Timedelta("12h"),
    method: Literal["pyinterp", "kdtree", "xesmf", "idw"] = "pyinterp",
    mask: Optional[Union[np.ndarray, pd.Series]] = None,
    return_format: Literal["dataframe", "dataset", "geodataframe"] = "dataframe",
) -> Union[pd.DataFrame, xr.Dataset, gpd.GeoDataFrame]:
    """
    Interpolate model values at observation points.
    Returns a DataFrame with obs, model_value, error, and all coordinates.
    """
    obs_df = obs_df.copy()
    model_times = pd.to_datetime(model_ds["time"].values)

    obs_df["matched_time"] = pd.NaT
    obs_df["model_value"] = np.nan
    obs_df["error"] = np.nan

    # Interpolation par jour (ou autre regroupement temporel)
    grouped = obs_df.groupby(obs_df["time"].dt.floor("1D"))

    for day, group in grouped:
        times = pd.to_datetime(group["time"])
        lats = group["lat"].to_numpy()
        lons = group["lon"].to_numpy()
        obs_vals = group[varname].to_numpy()

        # Associer chaque obs à la date modèle la plus proche
        matched_times = []
        for t in times:
            dt = abs(model_times - t)
            matched_times.append(model_times[np.argmin(dt)] if dt.min() <= delta_t else None)
        matched_times = np.array(matched_times)

        for unique_time in pd.unique(matched_times):
            if pd.isnull(unique_time):
                continue
            mask_t = matched_times == unique_time
            if not np.any(mask_t):
                continue

            sub_lats = lats[mask_t]
            sub_lons = lons[mask_t]
            sub_obs = obs_vals[mask_t]

            model_slice = model_ds.sel(time=unique_time, method="nearest")

            if method == "pyinterp":
                interp = pyinterp.backends.xarray.Grid2D(
                    longitude=model_slice["lon"],
                    latitude=model_slice["lat"],
                    values=model_slice[varname],
                )
                model_vals = interp.interpolate(sub_lons, sub_lats)
            elif method == "kdtree":
                lon2d, lat2d = np.meshgrid(model_slice["lon"].values, model_slice["lat"].values)
                points = np.column_stack([lon2d.ravel(), lat2d.ravel()])
                values_grid = model_slice[varname].values.ravel()
                tree = cKDTree(points)
                _, idx = tree.query(np.column_stack([sub_lons, sub_lats]))
                model_vals = values_grid[idx]
            #elif method == "xesmf":
            #    regridder = xe.Regridder(
            #        model_slice[varname].to_dataset(name=varname),
            #        xr.Dataset({"lat": (["points"], sub_lats), "lon": (["points"], sub_lons)}),
            #        method="bilinear",
            #        periodic=False,
            #        reuse_weights=True,
            #    )
            #    model_vals = regridder(model_slice[varname]).values
            else:
                raise ValueError(f"Unknown interpolation method: {method}")

            obs_df.loc[group.index[mask_t], "matched_time"] = unique_time
            obs_df.loc[group.index[mask_t], "model_value"] = model_vals
            obs_df.loc[group.index[mask_t], "error"] = model_vals - sub_obs

    # Nettoyage
    obs_df = obs_df.dropna(subset=["model_value", varname])
    if mask is not None:
        obs_df = obs_df.loc[mask]

    # Ajout explicite de la colonne d'observation sous un nom standard
    obs_df["obs_value"] = obs_df[varname]

    # Format de sortie
    if return_format == "dataframe":
        return obs_df
    elif return_format == "geodataframe":
        import geopandas as gpd
        return gpd.GeoDataFrame(
            obs_df, geometry=gpd.points_from_xy(obs_df.lon, obs_df.lat), crs="EPSG:4326"
        )
    elif return_format == "dataset":
        # Utiliser toutes les dimensions disponibles comme index
        index_cols = [col for col in ["time", "lat", "lon", "depth"] if col in obs_df.columns]
        if index_cols:
            obs_df = obs_df.drop_duplicates(subset=index_cols)
            obs_df = obs_df.set_index(index_cols)
        return xr.Dataset.from_dataframe(obs_df)
    else:
        raise ValueError(f"Invalid return_format: {return_format}")