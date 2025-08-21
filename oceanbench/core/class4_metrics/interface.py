
from loguru import logger
import numpy as np
import pandas as pd
import xarray as xr


from memory_profiler import profile
from oceanbench.core.dataset_utils import Variable, get_variable
from oceanbench.core.rmsd import get_lead_days_count

from .observation_preprocesing import preprocess_observations
from .matching import match_model_to_obs
from .binning import apply_binning, aggregate_scores
from .metrics import compute_scores_per_bin, weighted_score, rmse, bias
from .utils import area_weights, apply_qc_mask



'''def _apply_multi_bin(data, multi_bin):
    """Apply multi-dimensional binning using xarray groupby_bins."""
    binned = data
    for dim, bins in multi_bin.items():
        binned = binned.groupby_bins(dim, bins).mean()
    return binned

def _aggregate_time(result, time_dim, agg="mean"):
    """Aggregate over the time dimension if present."""
    if isinstance(result, xr.DataArray) and time_dim in result.dims:
        if agg == "mean":
            return result.groupby(time_dim).mean()
        elif agg == "sum":
            return result.groupby(time_dim).sum()
        # Add more aggregation methods if needed
    return result'''
'''
def compute_class4_metrics(
    model: xr.Dataset,
    observations: xr.Dataset,
    variables: list,  # Accept both str and Variable
    pred_coords: dict,
    ref_coords: dict,
    interpolation_method: str = "kdtree",
    list_scores: list[str] = ["rmsd"],
    time_tolerance: float = 12.0,  # Tolerance in hours for time matching
    # depth_bins: list[float] = None,
    region_mask: xr.DataArray = None,
    region_ids: list[int] = None,
    qc_rules: dict = None,
    qc_names: list[str] = None,
    good_qc: int | list[int] = 0,
    weights: xr.DataArray = None,
    # leadtimes: list[int] = None,
    # baseline: dict = None,
    binning_params={
        "time": "1D",
        "lat": 1.0,
        "lon": 1.0
    },
    agg_dict = {
        "rmsd": ["mean", "std"],   # moyenne et écart-type du RMSD
        "bias": "median",
        "n_obs": ["sum", "count"], # somme et nombre de valeurs
    },
    # time_dim: str = "time",
    # depth_dim: str = "depth",
    # diagnostics: list[str] = None,
    # multi_bin: dict = None,
    # concat_times: bool = False,
    # **kwargs
) -> dict:
    """
    Compute CLASS4 metrics between a model and observation dataset, with support for advanced options.

    This function performs the following steps:
      1. Interpolates/matches model outputs to observation points (supports 2D/3D, kdtree/griddata, etc.).
      2. Applies QC masking, including multi-level/cross-source QC if specified.
      3. Optionally applies area or custom weights.
      4. Supports binning by depth, region, or multi-dimensional bins.
      5. Supports multi-leadtime evaluation (forecast, persistence, climatology, etc.).
      6. Supports diagnostics métiers spécifiques (e.g., Lagrangian).
      7. Can aggregate/concatenate results over a time dimension.

    Parameters
    ----------
    model : xr.Dataset
        Model dataset (gridded).
    observations : xr.Dataset
        Observations dataset (point or profile).
    variables : list of str
        Variables to compare.
    interpolation_method : str, optional
        Horizontal interpolation method: 'kdtree', 'griddata', or 'nearest'. Default is 'kdtree'.
    list_scores : list[str], optional
        Metric to compute: 'rmsd', 'bias', 'corr', etc. Default is 'rmsd'.
    depth_bins : list of float, optional
        Bin edges for vertical binning.
    region_mask : xr.DataArray, optional
        Region mask for regional binning.
    region_ids : list of int, optional
        List of region IDs for regional binning.
    qc_names : list of str, optional
        List of QC variable names to use for masking.
    good_qc : int or list of int, optional
        QC flag(s) considered as valid.
    weights : xr.DataArray, optional
        Weights for weighted metrics (e.g., area weights).
    leadtimes : list of int, optional
        List of leadtimes to evaluate (for multi-leadtime metrics).
    baseline : dict, optional
        Dictionary of baseline datasets (e.g., {"climatology": ds, "persistence": ds}).
    qc_rules : dict, optional
        Dictionary specifying QC rules per variable/source (for cross-QC).
    time_dim : str, optional
        Name of the time dimension for temporal aggregation.
    diagnostics : list of str, optional
        List of diagnostics métiers spécifiques to compute (e.g., ["lagrangian"]).
    multi_bin : dict, optional
        Dictionary of binning instructions for multi-dimensional binning (e.g., {"depth": [...], "region": [...]}).
    concat_times : bool, optional
        If True, aggregate/concatenate results over the time dimension.
    time_tolerance : float, optional
        Tolerance in hours for time matching.
    **kwargs : dict
        Additional arguments passed to lower-level functions.

    Returns
    -------
    dict
        Dictionary of computed metrics, with keys indicating variable, bin, region, leadtime, etc.
    """
    # Convert variables to Variable enum if needed
    # variables = [Variable[v] if isinstance(v, str) else v for v in variables]


    results = {}

    # model_lead = model.sel(leadtime=lead) if "leadtime" in model.dims else model
    # obs_lead = observations.sel(leadtime=lead) if "leadtime" in observations.dims else observations

    for var in variables:
        # Prétraitement
        obs_df = preprocess_observations(
            observations, varname=var, level=ref_coords.coord_level
        )

        # Matching + interpolation + binning

        ##  var_name = var.variable_name_from_dataset(model)
        var_name = var    # TODO : implement same approach as "standard" oceanbench
        matched_model = match_model_to_obs(
            obs_df=obs_df,
            model_ds=model,
            varname=var,
            delta_t=time_tolerance,
            method=interpolation_method,
            binning=binning_params,
            return_format="dataframe"
        )
        matched_model["obs_value"] = obs_df[var].values
        obs_var = observations[var_name]
        model_var = matched_model[var_name]

        # 3. QC multi-niveaux/croisé
        if qc_rules is not None and var_name in qc_rules:
            qc_mask = True
            for qc_name, good_vals in qc_rules[var_name].items():
                if qc_name in observations:
                    qc_mask = qc_mask & apply_qc_mask(obs_var, observations[qc_name], good_vals)
            obs_var = obs_var.where(qc_mask)
            model_var = model_var.where(qc_mask)
        elif qc_names is not None and any(qc in observations for qc in qc_names):
            qc_mask = True
            for qc in qc_names:
                if qc in observations:
                    qc_mask = qc_mask & apply_qc_mask(obs_var, observations[qc], good_qc)
            obs_var = obs_var.where(qc_mask)
            model_var = model_var.where(qc_mask)


        score_funcs = {"rmse": rmse}   #, "bias": lambda y, yhat: np.nanmean(yhat - y)}
        # 2. Scores multiples par bin

        possible_dims = ["time", "lat", "lon", "depth"]
        bin_cols = [col for col in possible_dims if col in obs_df.columns]
        # bin_cols = [col for col in matched_model.columns if col.endswith("_bin")]
        scores_df = compute_scores_per_bin(
            matched_model, bin_cols,
            score_funcs, 
            obs_col="obs_value",
            model_col="model_value"
        )
        # 4. Pondération
        if weights is not None:
            w = weights
        elif "lat" in obs_var.dims:
            w = area_weights(obs_var["lat"])
        else:
            w = None
   
        scores_df = weighted_score(scores_df, score_col="rmsd", lat_col="lat_bin", n_obs_col="n_obs", area_weight=True, n_obs_weight=True)

        # 3. Agrégation des scores
        agg_df = aggregate_scores(scores_df, bin_cols, agg_dict)

        # 3. Visualisation ou export
        scores_xr = scores_df.set_index(["time_bin", "lat_bin", "lon_bin"]).to_xarray()

        # 5. Binning multi-dimensionnel (toujours avant agrégation temporelle)
        if multi_bin is not None:
            binned = model_var - obs_var
            binned = _apply_multi_bin(binned, multi_bin)
            if concat_times and time_dim in binned.dims:
                binned = _aggregate_time(binned, time_dim)
            results[f"{var_name}_multi_bin_lead{lead}"] = binned
        elif depth_bins is not None and depth_dim in obs_var.dims:
            binned = bin_by_depth(
                data=model_var - obs_var,
                depth=obs_var[depth_dim],
                bins=depth_bins,
                agg="mean"
            )
            if concat_times and time_dim in binned.dims:
                binned = _aggregate_time(binned, time_dim)
            results[f"{var_name}_binned_lead{lead}"] = binned
        elif region_mask is not None and region_ids is not None:
            binned = bin_by_region(
                data=model_var - obs_var,
                region_mask=region_mask,
                region_ids=region_ids,
                agg="mean"
            )
            if concat_times and time_dim in binned.dims:
                binned = _aggregate_time(binned, time_dim)
            results[f"{var_name}_region_lead{lead}"] = binned
        else:
            base = {k: v[var_name].sel(leadtime=lead) for k, v in (baseline or {}).items()} if baseline else {}
            score = compute_score(
                model_var, obs_var, list_scores=list_scores, weights=w, baseline=base.get("climatology"), **kwargs
            )
            if concat_times and isinstance(score, xr.DataArray) and time_dim in score.dims:
                score = _aggregate_time(score, time_dim)
            results[f"{var_name}_lead{lead}"] = score

        if diagnostics is not None:
            for diag in diagnostics:
                diag_result = run_diagnostics(
                    diag, model_var, obs_var, lead=lead, **kwargs
                )
                if concat_times and isinstance(diag_result, xr.DataArray) and time_dim in diag_result.dims:
                    diag_result = _aggregate_time(diag_result, time_dim)
                results[f"{var_name}_{diag}_lead{lead}"] = diag_result

    return agg_df'''

def make_serializable(obj):
    import numpy as np
    import pandas as pd
    import xarray as xr

    if isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.tolist()
    if isinstance(obj, xr.Dataset):
        return {k: make_serializable(v) for k, v in obj.to_dict()["data_vars"].items()}
    if isinstance(obj, xr.DataArray):
        return obj.to_dict()
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    return str(obj)


@profile
def compute_class4_metrics(
    model: xr.Dataset,
    observations: xr.Dataset,
    variables: list,  # Accept both str and Variable
    pred_coords: dict,
    ref_coords: dict,
    interpolation_method: str = "kdtree",
    list_scores: list[str] = ["rmsd", "bias"],
    time_tolerance: float = 12.0,
    region_mask: xr.DataArray = None,
    region_ids: list[int] = None,
    qc_rules: dict = None,
    qc_names: list[str] = None,
    good_qc: int | list[int] = 0,
    weights: xr.DataArray = None,
    binning_params={
        "time": "1D",
        "lat": 1.0,
        "lon": 1.0
    },
    agg_dict = {
        "rmsd": ["mean", "std"],
        "bias": "median",
        "n_obs": ["sum", "count"],
    },
) -> dict:
    """
    Pipeline robuste pour le calcul des métriques de classe 4.
    """
    results = {}

    for var in variables:
        # 1. Prétraitement des observations
        obs_df = preprocess_observations(
            observations, varname=var, level=ref_coords.coord_level
        )
        obs_df = obs_df.reset_index(drop=True)
        obs_df["obs_idx"] = obs_df.index

        # 2. Matching/interpolation (la colonne "obs_value" est créée ici)
        matched_df = match_model_to_obs(
            obs_df=obs_df,
            model_ds=model,
            varname=var,
            delta_t=pd.Timedelta(f"{time_tolerance}h"),
            method=interpolation_method,
            return_format="dataframe"
        )

        # 3. (Supprimer toute ligne qui manipule "obs_value" ici !)

        # 4. Binning, scoring, etc.
        binned_df = apply_binning(matched_df, binning=binning_params, drop_original=True)

        # 3. Ajout explicite de la colonne d'observation (alignement par index)
        # On suppose que matched_df et obs_df sont alignés après dropna
        matched_df = matched_df.reset_index(drop=True)
        # obs_df = obs_df.loc[matched_df.index].reset_index(drop=True)
        # matched_df["obs_value"] = obs_df[var].values

        # 4. QC (optionnel)
        # (À faire sur obs_df AVANT matching si possible, sinon sur matched_df ici)
        # Ex: matched_df = matched_df[matched_df["quality_flag"] == 1]

        # 5. Binning (création des colonnes *_bin et suppression des originales)
        binned_df = apply_binning(matched_df, binning=binning_params, drop_original=True)

        # 6. Calcul des scores par bin
        bin_cols = [col for col in binned_df.columns if col.endswith("_bin")]
        score_fns = []
        if "rmsd" in list_scores:
            score_fns.append(lambda m, o: {"rmsd": rmse(o, m)})
        if "bias" in list_scores:
            score_fns.append(lambda m, o: {"bias": bias(o, m)})
        # Ajoute d'autres scores ici si besoin

        scores_df = compute_scores_per_bin(
            binned_df, bin_cols, score_fns,
            model_col="model_value", obs_col="obs_value"
        )
        logger.debug(f"\n\n\n BRUT SCORES: {scores_df}")

        # 7. Pondération (optionnelle)
        # Ex: pondération surfacique et par n_obs
        '''scores_df["weight"] = 1.0
        if "lat_bin" in scores_df.columns:
            scores_df["weight"] *= np.abs(np.cos(np.deg2rad(scores_df["lat_bin"])))
        if "n_obs" in scores_df.columns:
            scores_df["weight"] *= scores_df["n_obs"]

        # Score global pondéré (ex: RMSD global)
        if "rmsd" in scores_df.columns:
            global_rmsd = weighted_score(
                scores_df, score_col="rmsd", lat_col="lat_bin",
                n_obs_col="n_obs", area_weight=True, n_obs_weight=True
            )
            results[f"{var}_global_rmsd"] = global_rmsd'''

        # 8. Agrégation finale par bin
        agg_df = aggregate_scores(scores_df, bin_cols, agg_dict)

        # scores_only = {k: v for k, v in results.items()}  #  if "agg_df" in k or "score" in k}
        scores = make_serializable(agg_df)
        # logger.info(f"\n\n\n\n serializable_scores: {serializable_scores}")
        #logger.info(f"\n\n\n\n scores_only: {scores_only}")
        #logger.info(f"\n\n\n\n results: {results}")
        #logger.info(f"\n\n\n\n agg_df: {agg_df}")

        # 9. Export xarray (optionnel)
        try:
            # xr_ds = xr.Dataset.from_dataframe(agg_df.set_index(bin_cols))
            results[var] = scores
        except Exception as e:
            results[var] = None

    return results
