
import gc
import os
import traceback

import pandas as pd
import xarray as xr
import numpy as np
import dask.array as da
from typing import Callable, Dict, List, Optional, Tuple, Union

import geopandas as gpd
from loguru import logger
import psutil
import pyinterp
from scipy.spatial import cKDTree
#import xesmf as xe
import xskillscore as xs
# from xskillscore import rmse, pearson_r, mae, crps_ensemble 

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

# Mapping par défaut : variable → nom du champ QC + flags valides
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



# ----------------------------------  SUPEROBS par maillage modèle  -------------------------------------



def stack_obs(ds_obs: xr.Dataset) -> xr.Dataset:
    """Ensure observations are stacked into a single n_points dimension."""
    if "n_points" not in ds_obs.dims:
        ds_obs = ds_obs.stack(n_points=tuple(d for d in ds_obs.dims if d not in ("time",)))
    return ds_obs


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
    lon_model = ds_model[lon_name].values
    lat_model = ds_model[lat_name].values
    lon = ds_obs[lon_name].values
    lat = ds_obs[lat_name].values

    # robust indexing using searchsorted
    lon_idx = np.searchsorted(lon_model, lon) - 1
    lat_idx = np.searchsorted(lat_model, lat) - 1

    # clip to avoid going outside bounds
    lon_idx = np.clip(lon_idx, 0, len(lon_model) - 1)
    lat_idx = np.clip(lat_idx, 0, len(lat_model) - 1)

    # ✅ détection auto de la dimension des obs
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
    Agrège les observations en superobs par maille modèle.
    Args:
        subset_ds: DataArray des observations (avec coords lat_idx, lon_idx).
        var_name: Nom de la variable à agréger.
        reduce: "mean" ou "median".
        model: Dataset modèle pour rattacher les coords réelles.
        min_count: Nombre minimum d'observations pour garder un superobs.
        max_std: Seuil d'écart-type pour filtrer les superobs bruités.
        weighting: None (par défaut), "count" ou "inv_var" (1/std²).

    Returns:
        result_ds: Dataset des superobs agrégés avec diagnostics.
    """
    # Vérifier que les coordonnées d'index existent
    if "lat_idx" not in subset_ds.coords or "lon_idx" not in subset_ds.coords:
        raise ValueError("subset_ds must have 'lat_idx' and 'lon_idx' coordinates")
    
    lat_idx_coord = subset_ds["lat_idx"]
    lon_idx_coord = subset_ds["lon_idx"]
    
    # Aplatir les coordonnées
    lat_idx_flat = lat_idx_coord.values.ravel()
    lon_idx_flat = lon_idx_coord.values.ravel()
    
    # extraction des données
    if hasattr(subset_ds, 'values'):
        data_values = subset_ds.values
    else:
        # Si c'est un Dataset au lieu d'un DataArray
        if var_name in subset_ds.data_vars:
            data_values = subset_ds[var_name].values
        else:
            raise ValueError(f"Variable '{var_name}' not found in dataset")
    
    # Aplatir les données
    data_flat = data_values.ravel()
    # validation et correction des longueurs
    # Trouver la longueur commune (minimum)
    min_length = min(len(lat_idx_flat), len(lon_idx_flat), len(data_flat))
    
    if len(set([len(lat_idx_flat), len(lon_idx_flat), len(data_flat)])) > 1:
        logger.warning(f"Arrays have different lengths. Truncating to {min_length}")
        
        # Tronquer tous les arrays à la longueur minimale
        lat_idx_flat = lat_idx_flat[:min_length]
        lon_idx_flat = lon_idx_flat[:min_length]
        data_flat = data_flat[:min_length]

    if len(lat_idx_flat) == 0:
        logger.warning("Empty arrays after truncation")
        return None

    # construction du DataFrame
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
    
    # Supprimer les lignes avec des NaN
    df = df.dropna(how="any")
    
    if df.empty:
        logger.warning("DataFrame is empty after removing NaN values")
        return None

    # Agrégation
    try:
        grouped = df.groupby(["lat_idx", "lon_idx"])
        
        # Réductions
        agg_main = getattr(grouped[[var_name]], reduce)()
        agg_std = grouped[[var_name]].std()
        agg_count = grouped[[var_name]].count()
        
        # Revenir en xarray
        result_ds = agg_main.to_xarray()
        result_ds[f"{var_name}_std"] = agg_std[var_name].to_xarray()
        result_ds[f"{var_name}_count"] = agg_count[var_name].to_xarray()
        
        logger.debug(f"Aggregation successful. Result shape: {dict(result_ds.dims)}")
        
    except Exception as e:
        logger.error(f"Aggregation failed: {e}")
        return None

    # Filtrage des superobs
    if min_count > 1:
        result_ds = result_ds.where(result_ds[f"{var_name}_count"] >= min_count, drop=True)
    if max_std is not None:
        result_ds = result_ds.where(result_ds[f"{var_name}_std"] <= max_std, drop=True)

    # pondération optionnelle
    if weighting == "count":
        weights = result_ds[f"{var_name}_count"]
    elif weighting == "inv_var":
        weights = 1.0 / (result_ds[f"{var_name}_std"]**2)
    else:
        weights = None

    if weights is not None:
        # Normaliser les poids
        weights = weights / weights.sum()
        # Appliquer une moyenne pondérée
        weighted_mean = (result_ds[var_name] * weights).sum()
        result_ds[f"{var_name}_weighted"] = weighted_mean

    # rattacher coordonnées géographiques du modèle
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

    return result_ds


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
        # ds_obs = stack_obs(ds_obs, lon_name, lat_name)
        ds_obs = compute_grid_index(ds_obs, ds_model, lon_name, lat_name)
        return aggregate_superobs(ds_obs, var_name, reduce=reduce, model=ds_model)
    except Exception as e:
        logger.error(f"Error in make_superobs: {e}")
        traceback.print_exc()
        raise(e)


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
    """Binning des observations sur la grille modèle (lat/lon/time/depth)."""

    def compute_edges(centers: np.ndarray) -> np.ndarray:
        step = np.diff(centers) / 2
        edges = np.zeros(len(centers) + 1)
        edges[1:-1] = centers[:-1] + step
        edges[0] = centers[0] - step[0]
        edges[-1] = centers[-1] + step[-1]
        return edges

    # Récupérer les centres et calculer les bords
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

    df = obs[vars_to_extract].to_dataframe().reset_index()

    # Binning spatial
    df["lat_bin"] = pd.cut(df[obs_lat_name], bins=lat_edges, labels=lat_centers, include_lowest=True)
    df["lon_bin"] = pd.cut(df[obs_lon_name], bins=lon_edges, labels=lon_centers, include_lowest=True)

    # Binning temporel
    if time_name in df.columns:
        df["time_bin"] = pd.to_datetime(df[time_name]).dt.floor(time_freq)
    else:
        df["time_bin"] = pd.Timestamp("1900-01-01")  # valeur sentinelle

    # Binning profondeur
    if depth_name in df.columns and depth_edges is not None:
        df["depth_bin"] = pd.cut(df[depth_name], bins=depth_edges, labels=depth_centers, include_lowest=True)
    else:
        df["depth_bin"] = -1

    # Groupby
    group_cols = ["lat_bin", "lon_bin", "time_bin", "depth_bin"]
    grouped = df.groupby(group_cols, dropna=True, observed=True)

    # Agrégation en superobs
    if method == "mean":
        result_main = grouped[var].mean()
        result_std = grouped[var].std()
        result_count = grouped[var].count()
    elif method == "median":
        result_main = grouped[var].median()
        result_std = grouped[var].std()
        result_count = grouped[var].count()
    elif method == "count":
        result_main = grouped[var].count()
        result_std = None
        result_count = result_main
    else:
        raise ValueError(f"Unsupported method: {method}")

    # Conversion
    obs_binned = result_main.to_xarray().rename(f"{var}_binned").to_dataset()

    if result_std is not None:
        obs_binned[f"{var}_std"] = result_std.to_xarray()
    obs_binned[f"{var}_count"] = result_count.to_xarray()

    # Convertir les coordonnées physiques
    if "lat_bin" in obs_binned.coords:
        obs_binned["lat_bin"] = obs_binned["lat_bin"].astype(float)
    if "lon_bin" in obs_binned.coords:
        obs_binned["lon_bin"] = obs_binned["lon_bin"].astype(float)
    if "depth_bin" in obs_binned.coords and depth_centers is not None:
        obs_binned["depth_bin"] = obs_binned["depth_bin"].astype(float)

    return obs_binned


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

    # Convertir time_bin en datetime
    df["time_center"] = pd.to_datetime(df["time_bin"])

    # Créer des DataArray pour vectoriser la sélection
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




'''def match_times(model_ds: xr.Dataset, obs_df: pd.DataFrame, tol: str = "12H") -> pd.DataFrame:
    """
    Associe chaque observation à l'instant modèle le plus proche dans une tolérance donnée.
    
    Args:
        model_ds: Dataset modèle avec une coordonnée "time".
        obs_df: DataFrame/GeoDataFrame contenant une colonne "time".
        tol: Tolérance temporelle (ex: "6H", "1D").

    Returns:
        DataFrame identique à obs_df mais avec une colonne supplémentaire "model_time".
    """
    import pandas as pd

    model_times = pd.to_datetime(model_ds.time.values)
    obs_times = pd.to_datetime(obs_df["time"].values)

    matched_model_times = []
    for t_obs in obs_times:
        # Trouver le temps modèle le plus proche
        idx = model_times.get_indexer([t_obs], method="nearest")[0]
        t_model = model_times[idx]

        # Vérifier la tolérance
        if abs(t_model - t_obs) <= pd.Timedelta(tol):
            matched_model_times.append(t_model)
        else:
            matched_model_times.append(pd.NaT)  # Pas de correspondance

    obs_df["model_time"] = matched_model_times

    # Supprimer les obs sans correspondance
    obs_df = obs_df.dropna(subset=["model_time"]).reset_index(drop=True)

    return obs_df'''



# ----------------------------------  Interpolation modèle → obs  -------------------------------------

def interpolate_model_on_obs(
        model_da: xr.DataArray, obs_df: pd.DataFrame, variable: str, method: str = "pyinterp"
    ) -> pd.DataFrame:
    """
    Interpole les valeurs du modèle sur les positions des observations.
    Args:
        model_da: DataArray modèle avec dimensions ('time', 'lat', 'lon') ou ('time', 'depth', 'lat', 'lon').
        obs_df: DataFrame/GeoDataFrame avec colonnes ['lon', 'lat'] et éventuellement ['depth'] et 'model_time'.
        variable: Nom de la variable interpolée (sert à nommer la colonne de sortie).
        method: Méthode d'interpolation ('pyinterp', 'kdtree', 'xesmf').
    Returns:
        DataFrame enrichi d'une colonne <variable>_model avec les valeurs interpolées.
    """
    method  = "pyinterp"
    obs_df[f"{variable}_model"] = np.nan
    if method == "pyinterp":
        interp_vals = interpolate_with_pyinterp(model_da, obs_df)
    elif method == "kdtree":
        interp_vals = interpolate_with_kdtree(model_da, obs_df, variable)
    #elif method == "xesmf":
    #    return interpolate_with_xesmf(model_ds, obs_df, variable)
    else:
        raise ValueError(f"Unknown interpolation method: {method}")
    obs_df[f"{variable}_model"] = interp_vals
    return obs_df


def interpolate_with_pyinterp(model_da: xr.DataArray, obs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Interpole un champ du modèle (2D lat/lon ou 3D lat/lon/depth) sur les observations.
    
    Args:
        model_da: DataArray avec dimensions ('lat', 'lon') ou ('depth', 'lat', 'lon').
        obs_df: DataFrame avec colonnes ['lon', 'lat'] et éventuellement ['depth'].
        variable: Nom de la variable interpolée (sert à nommer la colonne de sortie).
        
    Returns:
        obs_df enrichi d'une colonne <variable>_model avec les valeurs interpolées.
    """
    
    # cs 2d lat/lon
    if {"lat", "lon"}.issubset(set(model_da.dims)):
        lon = model_da.lon.values
        lat = model_da.lat.values
        lon2d, lat2d = np.meshgrid(lon, lat)
        points = np.column_stack([lon2d.ravel(), lat2d.ravel()])
        values = model_da.values.ravel()

        grid = pyinterp.RTree()
        grid.packing(points, values)

        obs_points = obs_df[["lon", "lat"]].values
        interp_vals, _ = grid.inverse_distance_weighting(obs_points, k=4)

    # cas 3d lon/lat/depth
    elif {"depth", "lat", "lon"}.issubset(set(model_da.dims)):
        lon = model_da.lon.values
        lat = model_da.lat.values
        depth = model_da.depth.values

        lon3d, lat3d, depth3d = np.meshgrid(lon, lat, depth, indexing="xy")
        points = np.column_stack([lon3d.ravel(), lat3d.ravel(), depth3d.ravel()])
        values = model_da.values.ravel()

        grid = pyinterp.RTree()
        grid.packing(points, values)

        if "depth" not in obs_df.columns:
            raise ValueError("Observations need a 'depth' column for 3D interpolation")

        obs_points = obs_df[["lon", "lat", "depth"]].values
        interp_vals, _ = grid.inverse_distance_weighting(obs_points, k=8)

    else:
        raise ValueError(f"Unsupported dimensions for model data: {model_da.dims}")

    return interp_vals


def _find_coord_name(da: xr.DataArray, candidates=("lat", "latitude", "nav_lat")) -> str:
    """Return first coord/dim name matching candidates (case-insensitive) or raise."""
    for name in list(da.coords) + list(da.dims):
        if name.lower() in {c.lower() for c in candidates}:
            return name
    raise ValueError(f"No coordinate found among candidates {candidates}")


def interpolate_with_kdtree(
    model_da: xr.DataArray,
    obs_df: pd.DataFrame,
    time_index: int = 0,
    obs_lon_col: str = "lon",
    obs_lat_col: str = "lat",
    obs_depth_col: str = "depth",
) -> np.ndarray:
    """
    Interpole valeurs modèle sur positions d'observations avec KDTree.
    
    Args:
        model_da: DataArray du modèle
        obs_df: DataFrame des observations
        time_index: Index temporel (ignoré si pas de dimension time)
        obs_lon_col: Nom de la colonne longitude dans obs_df
        obs_lat_col: Nom de la colonne latitude dans obs_df  
        obs_depth_col: Nom de la colonne profondeur dans obs_df
        
    Returns:
        np.ndarray: Array des valeurs interpolées (même longueur que obs_df)
    """
    
    # validation des entrées
    if not isinstance(model_da, xr.DataArray):
        raise TypeError(f"Expected xr.DataArray, got {type(model_da)}")

    # initialiser le résultat
    n_obs = len(obs_df)
    result = np.full(n_obs, np.nan)  # Array de résultats initialisé à NaN
    
    # Si time existe, sélectionner le slice temporel
    if "time" in model_da.dims:
        try:
            da = model_da.isel(time=time_index)
            logger.debug(f"Selected time index {time_index}")
        except Exception as e:
            logger.warning(f"Could not isel time={time_index}: {e}; trying to squeeze time.")
            da = model_da.squeeze("time", drop=True)
    else:
        da = model_da

    # Détecter les noms des coordonnées lat/lon
    lat_name = _find_coord_name(da, candidates=("lat", "latitude", "nav_lat", "y"))
    lon_name = _find_coord_name(da, candidates=("lon", "longitude", "nav_lon", "x"))

    # Extraire les arrays lat/lon
    lat_vals = da.coords[lat_name].values
    lon_vals = da.coords[lon_name].values

    # Construire la grille spatiale
    if lat_vals.ndim == 1 and lon_vals.ndim == 1:
        lon_grid, lat_grid = np.meshgrid(lon_vals, lat_vals)
    elif lat_vals.ndim == 2 and lon_vals.ndim == 2:
        lat_grid = lat_vals
        lon_grid = lon_vals
    else:
        raise ValueError(f"Unsupported lat/lon shapes: lat {lat_vals.shape}, lon {lon_vals.shape}")

    # Aplatir les coordonnées spatiales et construire le KDTree
    pts = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])  # [lat, lon] order
    tree = cKDTree(pts)
    logger.debug(f"KDTree built on {pts.shape[0]} spatial nodes.")

    # Vérifier si le modèle a une dimension depth
    has_depth = "depth" in da.dims
    
    if not has_depth:
        # cas 2d : pas de dimension depth
        if obs_df.empty:
            return result
        
        # S'assurer que la slice de données est 2D
        data2d = da.values
        if data2d.ndim != 2:
            data2d = np.squeeze(data2d)
        if data2d.shape != lat_grid.shape:
            if data2d.shape == (lat_grid.shape[1], lat_grid.shape[0]):
                data2d = data2d.T
            else:
                raise ValueError(f"Model 2D slice shape {data2d.shape} doesn't match lat/lon grid {lat_grid.shape}")

        # Préparer les points d'observation
        obs_mask_valid = (~obs_df[obs_lat_col].isna()) & (~obs_df[obs_lon_col].isna())
        if not obs_mask_valid.any():
            return result
            
        obs_points = np.column_stack([
            obs_df.loc[obs_mask_valid, obs_lat_col].values, 
            obs_df.loc[obs_mask_valid, obs_lon_col].values
        ])

        # interpolation et assignation dans le result array
        _, indices = tree.query(obs_points, k=1)
        vals = data2d.ravel()[indices]
        
        # Assigner les valeurs interpolées aux positions correctes
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

    # Pour chaque observation, trouver l'index de profondeur modèle le plus proche
    depth_idx_per_obs = np.full(n_obs, -1, dtype=int)
    valid_depth_mask = ~pd.isna(obs_depths)
    
    if valid_depth_mask.any():
        # Calcul vectorisé de l'index de profondeur le plus proche
        diffs = np.abs(depth_vals[:, None].astype(float) - obs_depths[valid_depth_mask][None, :].astype(float))
        nearest_idxs = np.argmin(diffs, axis=0)
        depth_idx_per_obs[valid_depth_mask] = nearest_idxs
    else:
        logger.warning("No valid observation depths found")
        return result

    # Traiter les observations groupées par index de profondeur
    unique_depth_idxs = np.unique(depth_idx_per_obs[depth_idx_per_obs >= 0])
    logger.debug(f"Found {len(unique_depth_idxs)} unique model depth levels to process")

    for d_idx in unique_depth_idxs:
        mask = depth_idx_per_obs == d_idx
        if not mask.any():
            continue

        # Prendre la slice 2D du modèle à cette profondeur
        try:
            slice_da = da.isel(depth=d_idx)
        except Exception:
            slice_da = da.sel(depth=depth_vals[d_idx], method="nearest")

        data2d = slice_da.values
        data2d = np.squeeze(data2d)
        if data2d.ndim != 2:
            raise ValueError(f"Model slice at depth index {d_idx} is not 2D: shape {data2d.shape}")

        # Aligner la forme avec lat_grid/lon_grid
        if data2d.shape != lat_grid.shape:
            if data2d.shape == (lat_grid.shape[1], lat_grid.shape[0]):
                data2d = data2d.T
            else:
                try:
                    if lat_name in slice_da.dims and lon_name in slice_da.dims:
                        slice_da2 = slice_da.transpose(lat_name, lon_name)
                        data2d = slice_da2.values
                    else:
                        raise ValueError
                except Exception:
                    raise ValueError(f"Data shape mismatch at depth {d_idx}: data {data2d.shape}, grid {lat_grid.shape}")

        # Calculer les points d'observation pour cette profondeur
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

        # Assigner les résultats aux positions correctes dans l'array de résultat
        result[valid_mask] = interpolated

    return result



# ----------------------------------  Binning  -------------------------------------

def apply_binning(
    df: pd.DataFrame,
    bin_specs: Optional[Dict[str, Union[int, list, str]]] = None
) -> pd.DataFrame:
    """
    Ajoute des colonnes de binning (time_bin, lat_bin, lon_bin, depth_bin) selon bin_specs.
    Gère les erreurs de colonne ou de bins.

    Args:
        df (pd.DataFrame): DataFrame à binner.
        bin_specs (dict, optionnel): Dictionnaire {dim: bins}. Par défaut, couvre les 4 dimensions.

    Returns:
        pd.DataFrame: DataFrame avec colonnes de bin ajoutées.
    """
    # Valeurs par défaut pour les bins
    default_bins = {
        "time": "1D",  # Binning temporel journalier
        "lat": np.arange(-90, 91, 1),
        "lon": np.arange(-180, 180, 1),
        "depth": None,  # À définir si la colonne existe
    }
    if bin_specs is None:
        bin_specs = default_bins.copy()
    else:
        # Compléter avec les valeurs par défaut si manquantes
        for k, v in default_bins.items():
            if k not in bin_specs:
                bin_specs[k] = v

    groupby = []
    for dim, bins in bin_specs.items():
        if dim not in df.columns:
            continue  # Ignore si la colonne n'existe pas
        try:
            if dim == "time":
                df[dim] = pd.to_datetime(df[dim], errors="coerce")
                if isinstance(bins, str):
                    df[f"{dim}_bin"] = df[dim].dt.floor(bins)
                    groupby.append(f"{dim}_bin")
                else:
                    raise ValueError(f"Pour la dimension 'time', bins doit être une string (ex: '1D').")
            else:
                # Pour les dimensions spatiales
                if bins is None:
                    # Si pas de bins fournis pour depth, essayer de faire des bins auto
                    if dim == "depth":
                        unique_depths = np.unique(df[dim].dropna())
                        if len(unique_depths) > 1:
                            bins = np.linspace(unique_depths.min(), unique_depths.max(), 10)
                        else:
                            continue  # Pas de binning possible
                    else:
                        continue
                if isinstance(bins, int):
                    df[f"{dim}_bin"] = pd.qcut(df[dim], q=bins, duplicates="drop")
                    groupby.append(f"{dim}_bin")
                elif isinstance(bins, (list, np.ndarray)):
                    df[f"{dim}_bin"] = pd.cut(df[dim], bins=bins, include_lowest=True)
                    groupby.append(f"{dim}_bin")
                else:
                    raise ValueError(f"Type de bins non supporté pour {dim}: {type(bins)}")
        except Exception as e:
            print(f"[apply_binning] Erreur pour la dimension '{dim}': {e}")
            continue

    return (df, groupby)

# ----------------------------------  Scores avec xskillscore  -------------------------------------
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
    Calcule les scores xskillscore (et custom) sur un DataFrame, avec gestion du binning et des erreurs.

    Args:
        df: DataFrame contenant les colonnes d'observation et de prédiction.
        y_obs_col: Nom de la colonne des observations.
        y_pred_col: Nom de la colonne des prédictions.
        metrics: Liste des métriques à calculer (str).
        weights: Poids optionnels (Series alignée sur df).
        groupby: Liste de colonnes pour groupby (ex: ["lat_bin", "time_bin"]).
        custom_metrics: Dictionnaire {nom: fonction(y_pred, y_obs)} pour ajouter des scores custom.
        binning: Dictionnaire pour binning (voir apply_binning).

    Returns:
        DataFrame avec les scores calculés.
    """
    #log_memory("START compute_scores_xskillscore")
    try:
        all_results = {}
        df = df.dropna(subset=[y_obs_col, y_pred_col])

        if groupby:
            grouped = df.groupby(groupby, observed=False)
        else:
            grouped = [(None, df)]

        # Calcul des scores
        bin_results = []
        for group_key, group_df in grouped:
            # Conversion en DataArray pour xskillscore
            y_true = xr.DataArray(group_df[y_obs_col].values, dims="points")
            y_pred = xr.DataArray(group_df[y_pred_col].values, dims="points")
            group_result = {}
            for metric in metrics:
                if metric == "rmsd":   # compatibilité Oceanbench
                    metric = "rmse"
                func = XSKILL_METRICS.get(metric)
                try:
                    # Gestion des poids si supporté
                    if metric in ["rmse", "mae", "mse", "me", "median_absolute_error", "mean_squared_log_error", "r2", "mape", "smape"]:
                        if weights is not None:
                            score = func(y_pred, y_true, dim="points", weights=xr.DataArray(weights.values, dims="points"))
                        else:
                            score = func(y_pred, y_true, dim="points")
                    elif metric in ["pearson_r", "spearman_r"]:
                        score = func(y_pred, y_true, dim="points")
                    elif metric == "crps_ensemble":
                        # Pour crps_ensemble, y_pred doit être (ensemble, points)
                        # Ici, on suppose que y_pred est déjà de la bonne forme
                        score = func(y_pred, y_true, dim="points")
                    else:
                        score = func(y_pred, y_true, dim="points")
                    # Extraire la valeur scalaire
                    if hasattr(score, "item"):
                        group_result[metric] = score.item()
                    else:
                        group_result[metric] = float(score)
                except Exception as e:
                    group_result[metric] = np.nan
                    print(f"[compute_scores_xskillscore] Erreur sur '{metric}': {e}")
            # Ajoute les clés de groupby si groupé
            if group_key is not None:
                if isinstance(group_key, tuple):
                    for k, v in zip(groupby, group_key):
                        group_result[k] = v
                else:
                    group_result[groupby[0]] = group_key
            bin_results.append(group_result)
        # all_results["by_bin"] = bin_results  # TODO : activate and process this
        # 4. Score global (tous bins confondus)
        y_true_all = xr.DataArray(df[y_obs_col].values, dims="points")
        y_pred_all = xr.DataArray(df[y_pred_col].values, dims="points")
        global_weights = None
        if weights is not None:
            global_weights = xr.DataArray(weights.loc[df.index].values, dims="points")
        global_result = {}
        for metric in metrics:
            if metric == "rmsd":   # compatibilité Oceanbench
                metric = "rmse"
            func = XSKILL_METRICS.get(metric)
            try:
                if func is not None:
                    if global_weights is not None and "weights" in func.__code__.co_varnames:
                        score = func(y_pred_all, y_true_all, dim="points", weights=global_weights)
                    else:
                        score = func(y_pred_all, y_true_all, dim="points")
                    global_result[metric] = float(score)
                else:
                    global_result[metric] = np.nan
            except Exception as e:
                global_result[metric] = np.nan
        #for k in groupby:
        #    global_result[k] = "ALL"
        # scores_df = pd.concat([scores_df, pd.DataFrame([global_result])], ignore_index=True)

        all_results["global"] = global_result
        #log_memory("END compute_scores_xskillscore")
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
    # Détermination automatique si possible
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


def xr_dataset_to_dataframe(
    ds: xr.Dataset, var: str,
    include_geometry: bool = True,
    data_type: str = "_obs",
) -> pd.DataFrame:
    """
    Converts an xarray.Dataset with observation points into a (Geo)DataFrame.

    Args:
        ds (xr.Dataset): Input dataset, with 'time', 'lat', 'lon' (and optionally 'depth').
        var (str): Name of the variable to extract as observation.
        include_geometry (bool): If True, returns a GeoDataFrame with geometry column.

    Returns:
        pd.DataFrame or gpd.GeoDataFrame: Flattened dataframe with all coordinates and variables.
    """
    #log_memory("START xr_dataset_to_dataframe") 
    # Convert to DataFrame and reset index
    df = ds.to_dataframe().reset_index()

    # Drop NaN rows for the variable, rename (var_obs or var_model)
    if f"{var}_binned" in df.columns:
        df = df.dropna(subset=[f"{var}_binned"])
        df = df.rename(columns={f"{var}_binned": f"{var}_{data_type}"})

    # Add geometry column if requested and coordinates are present
    if include_geometry and {'lon', 'lat'}.issubset(df.columns):
        df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")
    #log_memory("END xr_dataset_to_dataframe")
    return df


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
        all_scores = {}

        # Normalisation des systèmes de coordonnées
        logger.debug("Normalizing coordinate systems")
        #obs_ds, model_ds = detect_and_normalize_longitude_system(
        #    obs_ds, model_ds, "lon"
        #)
        
        for var in variables:
            try:
                logger.info(f"Processing variable: {var}")
                obs_da = obs_ds[var]
                model_da = model_ds[var]
                
                # Filtrer et appliquer QC si nécessaire
                if self.apply_qc:
                    obs_da = filter_observations_by_qc(
                        ds=obs_da,
                        qc_mappings=self.qc_mapping,
                    )

                # Initialisation des colonnes de regroupement et des colonnes d'observation/modèle
                groupby_cols = []  # Valeur par défaut
                obs_col = f"{var}_obs"
                model_col = f"{var}_model"

                # Interpolation du modèle sur les observations
                if matching_type == "nearest":
                    # Convertir les observations en DataFrame
                    obs_df = xr_dataset_to_dataframe(
                        obs_da, var, include_geometry=True, data_type="obs"
                    )
                    # Binning
                    obs_df, groupby_cols = apply_binning(obs_df, self.bin_specs)
                    if obs_df.empty:
                        logger.warning(f"Nb observations found for variable {var}")
                        continue

                    # Drop NaN rows for the variable
                    obs_df = obs_df.dropna(subset=[var])
                    obs_df = obs_df.rename(columns={var: f"{var}_obs"})
                    final_df = interpolate_model_on_obs(
                        model_da, obs_df, var, method=self.interp_method
                    )

                elif matching_type == "superobs":
                    # Binning des obs en superobs
                    superobs = make_superobs(obs_da, model_da, var, reduce="mean")
                    # Binning des obs sur la grille du modèle
                    obs_binned = superobs_binning(superobs, model_da, var=var)
                    binned_df = xr_dataset_to_dataframe(
                        obs_binned, var, include_geometry=True, data_type="obs"
                    )
                    # Ajout des valeurs modèle au dataframe
                    final_df = add_model_values(binned_df, model_da, var=var)
                else:
                    raise ValueError(f"Unknown matching_type: {matching_type}")

                # Calcul des scores pour cette variable
                scores_result = compute_scores_xskillscore(
                    df=final_df,
                    y_obs_col=obs_col,
                    y_pred_col=model_col,
                    metrics=self.metrics,
                    weights=None,
                    groupby=groupby_cols,
                )

                # Convertir le résultat en DataFrame si ce n'est pas déjà le cas
                if isinstance(scores_result, dict):
                    # Si c'est un dictionnaire, le convertir en DataFrame
                    scores_df = pd.DataFrame([scores_result])
                    scores_df['variable'] = var
                elif isinstance(scores_result, pd.DataFrame):
                    # Si c'est déjà un DataFrame, ajouter la colonne variable
                    scores_df = scores_result.copy()
                    scores_df['variable'] = var
                else:
                    # Pour d'autres types, créer un DataFrame basique
                    scores_df = pd.DataFrame({
                        'variable': [var],
                        'result': [scores_result]
                    })
                
                all_scores[var] = scores_df
                
            except Exception as e:
                logger.error(f"Error processing variable {var}: {e}")
                import traceback
                traceback.print_exc()
                # Créer un DataFrame d'erreur pour cette variable
                error_df = pd.DataFrame({
                    'variable': [var],
                    'error': [str(e)],
                    'status': ['failed']
                })
                all_scores[var] = error_df
                continue
        
        # Concaténer les DataFrames au lieu d'un dictionnaire
        if all_scores:
            # Extraire les DataFrames du dictionnaire
            dataframes_to_concat = list(all_scores.values())
            
            # Vérifier que tous les éléments sont des DataFrames
            valid_dataframes = []
            for df in dataframes_to_concat:
                if isinstance(df, pd.DataFrame):
                    valid_dataframes.append(df)
                else:
                    logger.warning(f"Non-DataFrame result found: {type(df)}")
            
            if valid_dataframes:
                return pd.concat(valid_dataframes, ignore_index=True)
            else:
                logger.warning("No valid DataFrames to concatenate")
                return pd.DataFrame()  # DataFrame vide
        else:
            logger.warning("No scores computed for any variable")
            return pd.DataFrame()  # DataFrame vide
