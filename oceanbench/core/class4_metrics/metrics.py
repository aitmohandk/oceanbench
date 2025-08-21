from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr
import xskillscore as xs


from .binning import apply_binning

'''def compute_score(
    model: xr.DataArray,
    obs: xr.DataArray,
    list_scores: str | list[str] = "rmsd",
    mask: xr.DataArray = None,
    weights: np.ndarray = None,
    qc: np.ndarray = None,
    good_qc: int | list[int] = 0,
    reduction_dim: str | list[str] = None,
    baseline: xr.DataArray = None,
    depth: xr.DataArray = None,
    depth_bins: list[float] = None,
):
    """
    Compute selected CLASS4 metrics between model and obs, with full parameterization.

    Parameters
    ----------
    model : xr.DataArray
        Model values at obs points.
    obs : xr.DataArray
        Observed values.
    list_scores : str or list of str
        List of scores to compute: e.g. ['rmsd', 'bias', 'corr', ...] or "all" to compute all.
    mask : xr.DataArray, optional
        Boolean mask (True=valid).
    weights : np.ndarray or xr.DataArray, optional
        Weights for weighted metrics.
    qc : np.ndarray or xr.DataArray, optional
        QC flags for obs.
    good_qc : int or list, optional
        Value(s) of QC considered valid.
    reduction_dim : str or list, optional
        Dimension(s) for reduction.
    baseline : xr.DataArray, optional
        Baseline for skill scores.
    depth : xr.DataArray, optional
        Depth values for binning.
    depth_bins : list, optional
        Bin edges for depth binning.
    **kwargs : dict
        Other parameters for advanced use.

    Returns
    -------
    dict
        Dictionary of computed scores (only those in list_scores).
    """
    # Handle list_scores as str or list
    all_possible_scores = [
        "rmsd", "bias", "mae", "std_obs", "std_model", "corr", "n_obs",
        "skill_score", "anomaly_corr", "variance_explained", "rmsd_by_depth_bin"
    ]
    if isinstance(list_scores, str):
        if list_scores == "all":
            list_scores = all_possible_scores
        else:
            list_scores = [list_scores]

    if depth is None and "depth" in obs.coords:
        depth = obs["depth"]
    # 1. Apply mask and QC
    arr_model = model.values if isinstance(model, xr.DataArray) else model
    arr_obs = obs.values if isinstance(obs, xr.DataArray) else obs

    valid = np.ones(arr_obs.shape, dtype=bool)
    if mask is not None:
        valid &= mask.values if isinstance(mask, xr.DataArray) else mask
    if qc is not None:
        if isinstance(good_qc, (list, tuple, np.ndarray)):
            valid &= np.isin(qc, good_qc)
        else:
            valid &= (qc == good_qc)
    valid &= np.isfinite(arr_model) & np.isfinite(arr_obs)
    arr_model = np.where(valid, arr_model, np.nan)
    arr_obs = np.where(valid, arr_obs, np.nan)
    if baseline is not None:
        arr_baseline = baseline.values if isinstance(baseline, xr.DataArray) else baseline
        arr_baseline = np.where(valid, arr_baseline, np.nan)
    else:
        arr_baseline = None

    # 2. Binning by depth if requested
    if "rmsd_by_depth_bin" in list_scores and depth is not None and depth_bins is not None:
        arr_depth = depth.values if isinstance(depth, xr.DataArray) else depth
        rmsd_bins = []
        for i in range(len(depth_bins) - 1):
            in_bin = (arr_depth >= depth_bins[i]) & (arr_depth < depth_bins[i + 1]) & valid
            if np.any(in_bin):
                diff = arr_model[in_bin] - arr_obs[in_bin]
                rmsd_bins.append(np.sqrt(np.nanmean(diff**2)))
            else:
                rmsd_bins.append(np.nan)
        # Only add if requested
        if "rmsd_by_depth_bin" in list_scores:
            out = {"rmsd_by_depth_bin": np.array(rmsd_bins)}
            if len(list_scores) == 1:
                return out
        else:
            out = {}
    else:
        out = {}

    # 3. Weighted metrics
    def _nanmean(x, axis=None):
        return np.nanmean(x, axis=axis)

    def _nansum(x, axis=None):
        return np.nansum(x, axis=axis)

    def _nanstd(x, axis=None):
        return np.nanstd(x, axis=axis)

    def _pearson(x, y, axis=None):
        xm = np.nanmean(x, axis=axis)
        ym = np.nanmean(y, axis=axis)
        num = np.nanmean((x - xm) * (y - ym), axis=axis)
        den = np.nanstd(x, axis=axis) * np.nanstd(y, axis=axis)
        return num / den

    # 4. Compute metrics only if requested
    diff = arr_model - arr_obs
    abs_diff = np.abs(diff)
    sq_diff = diff ** 2

    # Déduction automatique de reduction_dim si besoin
    if reduction_dim is None:
        # On retire les dimensions de binning (ex: "depth") et on agrège sur le reste
        reduction_dim = [d for d in obs.dims if d not in ("depth", "region", "time")]
    # Convertir reduction_dim (liste de noms) en tuple d'indices pour numpy
    if isinstance(reduction_dim, str):
        axis = obs.get_axis_num(reduction_dim)
    elif isinstance(reduction_dim, (list, tuple)):
        axis = tuple(obs.get_axis_num(d) for d in reduction_dim)
    else:
        axis = reduction_dim  # None ou int

    if weights is not None:
        w = weights if isinstance(weights, np.ndarray) else weights.values
        if "rmsd" in list_scores:
            out["rmsd"] = np.sqrt(_nansum(w * sq_diff) / _nansum(w))
        if "bias" in list_scores:
            out["bias"] = _nansum(w * diff) / _nansum(w)
        if "mae" in list_scores:
            out["mae"] = _nansum(w * abs_diff) / _nansum(w)
    else:
        if "rmsd" in list_scores:
            out["rmsd"] = np.sqrt(_nanmean(sq_diff, axis=axis))
        if "bias" in list_scores:
            out["bias"] = _nanmean(diff, axis=axis)
        if "mae" in list_scores:
            out["mae"] = _nanmean(abs_diff, axis=axis)

    if "std_obs" in list_scores:
        out["std_obs"] = _nanstd(arr_obs, axis=axis)
    if "std_model" in list_scores:
        out["std_model"] = _nanstd(arr_model, axis=axis)
    if "corr" in list_scores:
        out["corr"] = _pearson(arr_model, arr_obs, axis=axis)
    if "n_obs" in list_scores:
        out["n_obs"] = np.sum(~np.isnan(arr_model) & ~np.isnan(arr_obs), axis=axis)

    if arr_baseline is not None:
        if "skill_score" in list_scores:
            mse_model = _nanmean((arr_model - arr_obs) ** 2, axis=axis)
            mse_base = _nanmean((arr_baseline - arr_obs) ** 2, axis=axis)
            out["skill_score"] = 1 - mse_model / mse_base
        if "anomaly_corr" in list_scores:
            out["anomaly_corr"] = _pearson(arr_model - arr_baseline, arr_obs - arr_baseline, axis=axis)
        if "variance_explained" in list_scores:
            out["variance_explained"] = 1 - _nanstd(arr_model - arr_obs, axis=axis) ** 2 / _nanstd(arr_obs, axis=axis) ** 2

    return out'''


def rmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return np.sqrt(np.nanmean((y_pred - y_true) ** 2))


def bias(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return np.nanmean(y_pred - y_true)

def mae(y_pred, y_true):
    return np.nanmean(np.abs(y_pred - y_true))

def std(y_pred, y_true):
    return np.nanstd(y_pred)

def correlation(y_pred, y_true):
    if len(y_pred) < 2:
        return np.nan
    return np.corrcoef(y_pred, y_true)[0, 1]

def scatter_index(y_pred, y_true):
    std_diff = np.nanstd(y_pred - y_true)
    mean_obs = np.nanmean(y_true)
    return std_diff / mean_obs if mean_obs != 0 else np.nan

def normalized_rmse(y_pred, y_true):
    rmse_val = rmse(y_pred, y_true)
    std_obs = np.nanstd(y_true)
    return rmse_val / std_obs if std_obs != 0 else np.nan


DEFAULT_SCORE_FUNCS: Dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "bias": bias,
    "rmse": rmse,
    "mae": mae,
    "std_pred": std,
    "correlation": correlation,
    "scatter_index": scatter_index,
    "nrmse": normalized_rmse,
}


def weighted_score(
    df: pd.DataFrame,
    score_col: str = "rmsd",
    lat_col: str = "lat",
    n_obs_col: str = "n_obs",
    custom_weight_col: str = "custom_weight",
    area_weight: bool = False,
    n_obs_weight: bool = False,
    custom_weight: bool = False,
    custom_weights: np.ndarray = None,
) -> float:
    """
    Calcule un score global pondéré à partir d'un DataFrame de scores par bin.

    Args:
        df (pd.DataFrame): DataFrame avec les scores par bin.
        score_col (str): Colonne du score à pondérer.
        lat_col (str): Colonne de latitude (pour pondération surfacique).
        n_obs_col (str): Colonne du nombre d'observations (pour pondération par n_obs).
        custom_weight_col (str): Colonne de poids personnalisés (si custom_weight=True).
        area_weight (bool): Active la pondération surfacique (cos(lat)).
        n_obs_weight (bool): Active la pondération par nombre d'observations.
        custom_weight (bool): Active la pondération personnalisée.
        custom_weights (np.ndarray): Tableau de poids personnalisés (optionnel).

    Returns:
        float: Score global pondéré.
    """
    weights = np.ones(len(df))

    if area_weight and lat_col in df.columns:
        # Pondération surfacique (cos(lat))
        weights = weights * np.abs(np.cos(np.deg2rad(df[lat_col].astype(float))))
    if n_obs_weight and n_obs_col in df.columns:
        weights = weights * df[n_obs_col].astype(float)
    if custom_weight:
        if custom_weights is not None:
            weights = weights * np.asarray(custom_weights)
        elif custom_weight_col in df.columns:
            weights = weights * df[custom_weight_col].astype(float)
        else:
            raise ValueError("Custom weights not provided or column missing.")

    # Nettoyage des poids et scores
    mask = np.isfinite(df[score_col]) & np.isfinite(weights)
    if not np.any(mask):
        return np.nan

    return np.average(df.loc[mask, score_col], weights=weights[mask])


'''# Pondération surfacique uniquement
global_rmsd = weighted_score(df, score_col="rmsd", lat_col="lat_bin", area_weight=True)

# Pondération par nombre d'observations uniquement
global_rmsd = weighted_score(df, score_col="rmsd", n_obs_col="n_obs", n_obs_weight=True)

# Pondération surfacique ET par nombre d'observations
global_rmsd = weighted_score(df, score_col="rmsd", lat_col="lat_bin", n_obs_col="n_obs", area_weight=True, n_obs_weight=True)

# Pondération personnalisée (colonne)
global_rmsd = weighted_score(df, score_col="rmsd", custom_weight_col="my_weights", custom_weight=True)

# Pondération personnalisée (tableau externe)
global_rmsd = weighted_score(df, score_col="rmsd", custom_weight=True, custom_weights=my_weights_array)
'''

'''def compute_scores_per_bin(
    df: pd.DataFrame,
    bin_cols: list,
    score_fns: list,
    model_col: str,
    obs_col: str,
) -> pd.DataFrame:
    """
    Applique plusieurs fonctions de score à chaque bin.
    Chaque fonction doit retourner un dict de scores.
    """
    def apply_scores(group):
        model = group[model_col].values
        obs = group[obs_col].values
        out = {}
        for fn in score_fns:
            out.update(fn(model, obs))
        return pd.Series(out)
    return df.groupby(bin_cols).apply(apply_scores).reset_index()
'''


def compute_scores_per_bin(
    df: pd.DataFrame,
    bin_cols: list,
    score_fns: list,
    model_col: str,
    obs_col: str,
    extra_args: dict = None,
) -> pd.DataFrame:
    """
    Applique plusieurs fonctions de score à chaque bin.
    Chaque fonction doit retourner un dict de scores.
    Ajoute n_obs (nombre d'observations par bin).
    """
    extra_args = extra_args or {}

    def apply_scores(group):
        model = group[model_col].values
        obs = group[obs_col].values
        out = {"n_obs": len(model)}
        for fn in score_fns:
            # Supporte les fonctions à 2 ou plusieurs arguments
            try:
                out.update(fn(model, obs, **extra_args))
            except TypeError:
                out.update(fn(model, obs))
        return pd.Series(out)

    # On évite les bins vides
    grouped = df.groupby(bin_cols)
    result = grouped.apply(apply_scores).reset_index()
    return result














XSKILL_METRICS = {
    "rmse": xs.rmse,
    "mae": xs.mae,
    "mse": xs.mse,
    "correlation": xs.pearson_r,
    "pearson_r": xs.pearson_r,
    "spearman_r": xs.spearman_r,
    # ajouter d’autres métriques au besoin
}


def compute_scores_xskillscore(
    df: pd.DataFrame,
    y_obs_col: str = "obs",
    y_pred_col: str = "pred",
    metrics: Optional[List[str]] = None,
    weights: Optional[pd.Series] = None,
    groupby: Optional[List[str]] = None,
    custom_metrics: Optional[Dict[str, Callable]] = None,
    binning: Optional[Dict[str, Union[str, List, np.ndarray]]] = None,
) -> pd.DataFrame:
    """
    Compute scores using xskillscore with optional weights, grouping, and binning.

    Parameters
    ----------
    df : pd.DataFrame
        Matched data containing observation and prediction columns.
    y_obs_col : str
        Name of observation column.
    y_pred_col : str
        Name of prediction column.
    metrics : list of str, optional
        List of metric names to compute.
    weights : pd.Series, optional
        Optional weights indexed like df (e.g. spatial or temporal weights).
    groupby : list of str, optional
        Columns to group by before computing scores.
    custom_metrics : dict, optional
        Custom metrics: {name: function(y_pred, y_obs, weights=None)}.
    binning : dict, optional
        Binning instructions, keys = columns, values = bin specs.
        Example:
        {
          "time": "1D",
          "depth": np.arange(0, 1000, 50),
          "lat": np.arange(-90, 90, 1),
          "lon": np.arange(-180, 180, 1)
        }

    Returns
    -------
    pd.DataFrame
        DataFrame with scores per group/bin or overall.
    """

    # Nettoyage des données
    valid_df = df.dropna(subset=[y_obs_col, y_pred_col]).copy()

    # Application du binning
    if binning:
        valid_df = apply_binning(valid_df, binning)
        # Ajout des colonnes binning au groupby
        groupby = (groupby or []) + [f"{col}_bin" for col in binning.keys()]

    # Préparation des poids
    if weights is not None:
        weights = weights.loc[valid_df.index]

    # Préparation des fonctions métriques
    metric_funcs = XSKILL_METRICS.copy()
    if custom_metrics:
        metric_funcs.update(custom_metrics)

    if metrics is None:
        metrics = list(metric_funcs.keys())

    def _calc_scores(group: pd.DataFrame) -> pd.Series:
        y_true = group[y_obs_col].values
        y_pred = group[y_pred_col].values
        w = None if weights is None else weights.loc[group.index].values

        results = {}
        for m in metrics:
            func = metric_funcs[m]
            try:
                score = func(y_true, y_pred, weights=w)
            except TypeError:
                # fallback si poids non supportés
                score = func(y_true, y_pred)
            results[m] = score
        return pd.Series(results)

    if groupby:
        return valid_df.groupby(groupby).apply(_calc_scores).reset_index()
    else:
        return _calc_scores(valid_df).to_frame().T