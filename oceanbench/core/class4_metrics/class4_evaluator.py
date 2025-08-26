
import gc
import os
import traceback

import pandas as pd
import xarray as xr
import numpy as np
import dask.array as da
from typing import Callable, Dict, List, Optional, Union

import geopandas as gpd
import psutil
import pyinterp
from scipy.spatial import cKDTree
#import xesmf as xe
import xskillscore as xs
# from xskillscore import rmse, pearson_r, mae, crps_ensemble 

def log_memory(fct):
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1e6
    print(f"[{fct}] Memory usage: {mem_mb:.2f} MB")

# -------------------- Temporal Matching --------------------
def match_times(model_ds: xr.Dataset, obs_df: pd.DataFrame, time_tol: pd.Timedelta) -> pd.DataFrame:
    """Matches observation times with model times using a tolerance."""
    model_times = model_ds.time.values
    matched_idx = []
    for i, obs_time in enumerate(obs_df["time"]):
        diffs = np.abs(model_times - np.datetime64(obs_time))
        min_diff_idx = np.argmin(diffs)
        if diffs[min_diff_idx] <= np.timedelta64(time_tol):
            matched_idx.append(min_diff_idx)
        else:
            matched_idx.append(None)
    obs_df["matched_model_time"] = [model_times[i] if i is not None else pd.NaT for i in matched_idx]
    return obs_df.dropna(subset=["matched_model_time"])


def interpolate_model_on_obs(model_ds: xr.Dataset, obs_df: pd.DataFrame, variable: str, method: str = "pyinterp") -> pd.DataFrame:
    if method == "pyinterp":
        return interpolate_with_pyinterp(model_ds, obs_df, variable)
    elif method == "kdtree":
        return interpolate_with_kdtree(model_ds, obs_df, variable)
    #elif method == "xesmf":
    #    return interpolate_with_xesmf(model_ds, obs_df, variable)
    else:
        raise ValueError(f"Unknown interpolation method: {method}")


def interpolate_with_pyinterp(model_ds: xr.Dataset, obs_df: pd.DataFrame, variable: str) -> pd.DataFrame:
    #log_memory("START interpolate_with_pyinterp")
    model_ds = model_ds.copy()

    obs_df = obs_df.copy()
    obs_df[f"{variable}_model"] = np.nan

    model_times = pd.to_datetime(model_ds.time.values)
    try:
        for t_model in np.unique(obs_df["matched_model_time"]):
            if pd.isnull(t_model):
                continue
            mask = obs_df["matched_model_time"] == t_model
            n_obs = mask.sum()
            if n_obs == 0:
                continue
            obs_lons = obs_df.loc[mask, "lon"].values
            obs_lats = obs_df.loc[mask, "lat"].values
            
            t_diffs = np.abs(model_times - np.datetime64(t_model))
            t_idx = int(np.argmin(t_diffs))
            da = model_ds.isel(time=t_idx).squeeze()
            # Si "depth" existe et est de taille 1, squeeze aussi
            if "depth" in da.dims and da.sizes["depth"] == 1:
                da = da.isel(depth=0).squeeze()
            if da.ndim != 2:
                raise ValueError(f"Expected 2D DataArray (lat, lon), got shape {da.shape}")
            if da.dims != ("lat", "lon"):
                da = da.transpose("lat", "lon")

            lon = da.lon.values
            lat = da.lat.values
            lon2d, lat2d = np.meshgrid(lon, lat)
            points = np.column_stack([lon2d.ravel(), lat2d.ravel()])
            values = da.values.ravel()

            grid = pyinterp.RTree()
            grid.packing(points, values)
            obs_points = np.column_stack([obs_lons, obs_lats])
            interp_vals, _ = grid.inverse_distance_weighting(
                obs_points, k=4
            )
            if len(interp_vals) != n_obs:
                raise ValueError(f"Interpolation output shape mismatch: {len(interp_vals)} vs {n_obs}")
            obs_df.loc[mask, f"{variable}_model"] = interp_vals
        #log_memory("END interpolate_with_pyinterp")

        return obs_df
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise


'''def interpolate_with_xesmf(model_ds: xr.Dataset, obs_df: gpd.GeoDataFrame, variable: str) -> np.ndarray:
    """Interpolate model values on observation points using xESMF."""
    if 'lat' not in model_ds or 'lon' not in model_ds:
        raise ValueError("Model dataset must have 'lat' and 'lon' coordinates.")

    model_ds = model_ds.copy()
    # Prepare the grid of model
    model_grid = {"lon": model_ds.lon.values, "lat": model_ds.lat.values}

    # Observation grid (irregular)
    obs_lon = np.array(obs_df.geometry.x)
    obs_lat = np.array(obs_df.geometry.y)
    obs_grid = {"lon": obs_lon, "lat": obs_lat}

    # xESMF expects 2D grids — broadcast the obs locations
    obs_lon2d, obs_lat2d = np.meshgrid(obs_lon, obs_lat)

    # Create regridder
    regridder = xe.Regridder(
        model_ds,
        xr.Dataset({"lat": (["y", "x"], obs_lat2d), "lon": (["y", "x"], obs_lon2d)}),
        method="bilinear",
        reuse_weights=False
    )

    # Regrid the variable
    interp = regridder(model_ds)

    # Extract interpolated values at obs points
    # This assumes interp has dimensions time, y, x
    result = interp.values.diagonal(axis1=1, axis2=2)
    return result.T  # shape: (n_obs, n_time) -> transpose if needed'''


def interpolate_with_kdtree(model_ds: xr.Dataset, obs_df: gpd.GeoDataFrame, variable: str) -> np.ndarray:
    """Interpolate model values at observation points using nearest neighbor (KDTree)."""
    lon = model_ds.lon.values
    lat = model_ds.lat.values

    if lon.ndim == 1 and lat.ndim == 1:
        lon2d, lat2d = np.meshgrid(lon, lat)
    else:
        lon2d, lat2d = lon, lat

    points_model = np.column_stack((lon2d.ravel(), lat2d.ravel()))
    tree = cKDTree(points_model)

    obs_points = np.column_stack((obs_df.geometry.x, obs_df.geometry.y))
    _, idx = tree.query(obs_points)

    # Get interpolated values for each time step
    n_obs = len(obs_points)
    n_time = len(model_ds.time)
    interp_values = np.empty((n_obs, n_time))

    for i, t in enumerate(model_ds.time):
        data = model_ds.sel(time=t).values.ravel()
        interp_values[:, i] = data[idx]

    return interp_values


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
    #log_memory("START apply_binning")

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
        # Complète avec les valeurs par défaut si manquantes
        for k, v in default_bins.items():
            if k not in bin_specs:
                bin_specs[k] = v

    df = df.copy()
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
    
    #log_memory("END apply_binning")
    return (df, groupby)

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
        import traceback
        print(f"[compute_scores_xskillscore] Exception: {e}")
        traceback.print_exc()

# Mapping par défaut : variable → nom du champ QC + flags valides
DEFAULT_QC_MAPPING: Dict[str, Dict[str, List[int]]] = {
    "sea_surface_temperature": {"qc_variable": "quality_flag", "valid_flags": [0]},
    "sea_surface_salinity": {"qc_variable": "dqf", "valid_flags": [0, 1]},
    "sea_surface_height": {"qc_variable": "quality_level", "valid_flags": [1, 2]},
    "temperature": {"qc_variable": "qc_flag", "valid_flags": [0]},
    "salinity": {"qc_variable": "qc_flag", "valid_flags": [0]},
}

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


def apply_spatial_mask(df: pd.DataFrame, mask_fn: Callable[[pd.DataFrame], pd.Series]) -> pd.DataFrame:
    return df[mask_fn(df)]


def xr_dataset_to_dataframe(ds: xr.Dataset, var: str, include_geometry: bool = True) -> pd.DataFrame:
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
    ds = ds.copy()
    df = ds.to_dataframe().reset_index()

    # Drop NaN rows for the variable
    if var not in df.columns:
        raise ValueError(f"Variable '{var}' not found in dataset columns: {df.columns.tolist()}")
    df = df.dropna(subset=[var])

    df = df.rename(columns={var: f"{var}_obs"})

    # Add geometry column if requested and coordinates are present
    if include_geometry and {'lon', 'lat'}.issubset(df.columns):
        df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")
    #log_memory("END xr_dataset_to_dataframe")
    return df


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
    ) -> pd.DataFrame:
        #log_memory("START run Class4Evaluator")
        all_scores = {}
        for var in variables:
            ds = model_ds[var].copy()
            if self.mask_fn:
                ds = apply_spatial_mask(ds, self.mask_fn)

            # Apply QC if needed
            if self.apply_qc:
                ds = filter_observations_by_qc(
                    ds=ds,
                    qc_mappings=self.qc_mapping,
                )
            obs_df = xr_dataset_to_dataframe(obs_ds, var, include_geometry=True)

            matched = match_times(ds, obs_df, self.time_tol)
            ds = interpolate_model_on_obs(ds, matched, var, method=self.interp_method)
            # Binning
            ds, groupby = apply_binning(ds, self.bin_specs)

            # Harmonisation des colonnes
            obs_col = f"{var}_obs"  # nom de la colonne d'observation pour cette variable
            model_col = f"{var}_model"  # nom de la colonne du modèle pour cette variable

            # Calcul des scores pour cette variable
            scores = compute_scores_xskillscore(
                ds,
                y_obs_col=obs_col,
                y_pred_col=model_col,
                metrics=self.metrics,
                groupby=groupby,
            )
            all_scores[var] = scores
            del matched
            gc.collect()
        #log_memory("END run Class4Evaluator")
        return all_scores
