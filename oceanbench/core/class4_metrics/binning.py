
import itertools
from typing import Dict, List, Union

from loguru import logger
import numpy as np
import pandas as pd


DEFAULT_BINNING = {
    "time": "1D",
    "depth": np.arange(0, 1000, 50),
    "lat": np.arange(-90, 90, 1),
    "lon": np.arange(-180, 180, 1)
}

def apply_binning(
    df: pd.DataFrame,
    binning: Dict[str, Union[str, List, np.ndarray]] = DEFAULT_BINNING,
) -> pd.DataFrame:
    """
    Apply binning on specified columns of the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with columns to bin.
    binning : dict
        Dict with keys = column names, values = bin spec:
          - For 'time', a pandas offset alias string (e.g. '1D', '6H')
          - For numeric columns, a list or array of bin edges

    Returns
    -------
    pd.DataFrame
        DataFrame with new columns {col}_bin representing bins.
    """
    df_binned = df.copy()

    for col, bins in binning.items():
        if col not in df.columns:
            continue

        if col == "time":
            # For time, floor datetime to bin resolution
            df_binned[col] = pd.to_datetime(df_binned[col])
            df_binned[f"{col}_bin"] = df_binned[col].dt.floor(bins)

        else:
            # Numeric binning with pd.cut
            df_binned[f"{col}_bin"] = pd.cut(df_binned[col], bins=bins, include_lowest=True)

    return df_binned

def apply_binning_0(df: pd.DataFrame, binning: dict, drop_original: bool = True) -> pd.DataFrame:
    """
    Ajoute des colonnes de bin (ex: lat_bin, lon_bin, time_bin) sans agrégation.
    Si drop_original=True, supprime les anciennes colonnes de dimension.
    """
    df = df.copy()
    bin_created = []
    if "time" in binning:
        df["time_bin"] = df["time"].dt.floor(binning["time"])
        bin_created.append("time")
    if "lat" in binning:
        df["lat_bin"] = (df["lat"] // binning["lat"]) * binning["lat"]
        bin_created.append("lat")
    if "lon" in binning:
        df["lon_bin"] = (df["lon"] // binning["lon"]) * binning["lon"]
        bin_created.append("lon")
    if "depth" in binning and "depth" in df.columns:
        df["depth_bin"] = (df["depth"] // binning["depth"]) * binning["depth"]
        bin_created.append("depth")

    # Supprimer les anciennes colonnes de dimension si demandé
    if drop_original:
        df = df.drop(columns=[col for col in bin_created if col in df.columns])

    return df


def aggregate_scores(
    scores_df: pd.DataFrame,
    bin_cols: list,
    agg_dict: dict = None,
    flatten_columns: bool = True,
    dropna: bool = True,
    max_dim_comb: int = 2,  # jusqu'à quelles combinaisons de dimensions (1=par dim, 2=par paires, etc.)
) -> dict:
    """
    Agrège les scores par bin, et retourne aussi tous les scores globaux possibles
    (par dimension, global, etc.).

    Returns:
        dict: {
            "by_bin": DataFrame (scores par bin complet),
            "global": DataFrame (score global),
            "by_time_bin": DataFrame (score moyen par temps),
            "by_lat_bin": DataFrame (score moyen par latitude),
            "by_lon_bin": DataFrame (score moyen par longitude),
            ...
        }
    """
    # Colonnes à agréger
    if agg_dict is None:
        value_cols = [col for col in scores_df.columns if col not in bin_cols]
        agg_dict = {col: "mean" for col in value_cols}
    agg_dict = {k: v for k, v in agg_dict.items() if k in scores_df.columns}
    if not agg_dict:
        raise ValueError("Aucune colonne à agréger trouvée dans scores_df.")

    results = {}

    # 1. Scores par bin complet (toutes les dimensions)
    '''grouped = scores_df.groupby(bin_cols, dropna=dropna).agg(agg_dict).reset_index()
    if flatten_columns and isinstance(grouped.columns, pd.MultiIndex):
        grouped.columns = [
            "_".join([str(c) for c in col if c and c != ""]) for col in grouped.columns.values
        ]
    results["by_bin"] = grouped'''

    # 2. Score global (aucun groupby)
    global_df = scores_df.agg(agg_dict)
    if isinstance(global_df, pd.Series):
        global_df = global_df.to_frame().T
    results["global"] = global_df

    # 3. Scores par dimension individuelle
    for dim in bin_cols:
        grouped_dim = scores_df.groupby(dim, dropna=dropna).agg(agg_dict).reset_index()
        if flatten_columns and isinstance(grouped_dim.columns, pd.MultiIndex):
            grouped_dim.columns = [
                "_".join([str(c) for c in col if c and c != ""]) for col in grouped_dim.columns.values
            ]
        results[f"by_{dim}"] = grouped_dim

    # 4. (Optionnel) Scores par combinaison de dimensions (ex: lat_bin+lon_bin)
    '''for n in range(2, max_dim_comb+1):
        for dims in itertools.combinations(bin_cols, n):
            grouped_dims = scores_df.groupby(list(dims), dropna=dropna).agg(agg_dict).reset_index()
            if flatten_columns and isinstance(grouped_dims.columns, pd.MultiIndex):
                grouped_dims.columns = [
                    "_".join([str(c) for c in col if c and c != ""]) for col in grouped_dims.columns.values
                ]
            key = "by_" + "_".join(dims)
            results[key] = grouped_dims'''

    return results

def flatten_global_df(global_df: pd.DataFrame, agg_dict: dict) -> dict:
    """
    Transforme un DataFrame d'agrégation globale multi-indexé en un dict compact,
    ne gardant que les valeurs d'intérêt selon agg_dict et non-NaN.
    """
    # Si multi-index sur les lignes (méthodes d'agg), aplatir
    if isinstance(global_df.index, pd.MultiIndex) or (global_df.shape[0] > 1):
        # On suppose que l'index est le nom de l'agg (mean, std, etc.)
        result = {}
        for col, aggs in agg_dict.items():
            if isinstance(aggs, str):
                aggs = [aggs]
            for agg in aggs:
                try:
                    val = global_df.loc[agg, col]
                except KeyError:
                    continue
                key = f"{col}_{agg}"
                if pd.notnull(val):
                    result[key] = val
        return result
    else:
        # Cas DataFrame à une seule ligne, colonnes déjà aplaties
        row = global_df.iloc[0]
        expected_keys = []
        for col, aggs in agg_dict.items():
            if isinstance(aggs, str):
                expected_keys.append(f"{col}_{aggs}")
            else:
                for agg in aggs:
                    expected_keys.append(f"{col}_{agg}")
        return {k: row[k] for k in expected_keys if k in row and pd.notnull(row[k])}

def aggregate_scores(
    scores_df: pd.DataFrame,
    bin_cols: list,
    agg_dict: dict = None,
    flatten_columns: bool = True,
    dropna: bool = True,
    max_dim_comb: int = 2,
) -> dict:
    """
    Agrège les scores par bin, et retourne aussi tous les scores globaux possibles
    (par dimension, global, etc.), avec des clés explicites.
    """
    import itertools

    if agg_dict is None:
        value_cols = [col for col in scores_df.columns if col not in bin_cols]
        agg_dict = {col: "mean" for col in value_cols}
    agg_dict = {k: v for k, v in agg_dict.items() if k in scores_df.columns}
    if not agg_dict:
        raise ValueError("Aucune colonne à agréger trouvée dans scores_df.")

    results = {}

    # 1. Scores par bin complet (toutes les dimensions)
    '''grouped = scores_df.groupby(bin_cols, dropna=dropna).agg(agg_dict).reset_index()
    if flatten_columns and isinstance(grouped.columns, pd.MultiIndex):
        grouped.columns = [
            "_".join([str(c) for c in col if c and c != ""]) for col in grouped.columns.values
        ]
    results["by_bin"] = {
        "dims": bin_cols,
        "values": grouped
    }'''

    # 2. Score global (aucun groupby)
    '''global_df = scores_df.agg(agg_dict)
    if isinstance(global_df, pd.Series):
        global_df = global_df.to_frame().T
    if isinstance(global_df, pd.DataFrame) and isinstance(global_df.columns, pd.MultiIndex):
        # Aplatir les colonnes multi-index
        global_df.columns = [
            "_".join([str(c) for c in col if c and c != ""]) for col in global_df.columns.values
        ]
    # Prendre la première ligne comme dict (il n'y en a qu'une)
    global_dict = global_df.iloc[0].to_dict()'''

    logger.debug(f"\n\n\n AGG0 SCORES: {scores_df}")

    global_df = scores_df.agg(agg_dict, axis=0)

    if isinstance(global_df, pd.Series):
        global_df = global_df.to_frame().T

    if isinstance(global_df, pd.DataFrame) and isinstance(global_df.columns, pd.MultiIndex):
        global_df.columns = [
            "_".join([str(c) for c in col if c and c != ""]) for col in global_df.columns.values
        ]

    # Maintenant, global_df a une seule ligne avec toutes les valeurs
    #global_dict = global_df.iloc[0].to_dict()
    global_dict = flatten_global_df(global_df, agg_dict)
    results["global"] = {"dims": "all", "values": global_dict}

    # 3. Scores par dimension individuelle
    for dim in bin_cols:
        grouped_dim = scores_df.groupby(dim, dropna=dropna).agg(agg_dict).reset_index()
        if flatten_columns and isinstance(grouped_dim.columns, pd.MultiIndex):
            grouped_dim.columns = [
                "_".join([str(c) for c in col if c and c != ""]) for col in grouped_dim.columns.values
            ]
        # results[f"mean_over_{dim}"] = {
        results[f"{dim}"] = {
            "dims": [dim],
            "values": grouped_dim
        }

    # 4. Scores par combinaison de dimensions (ex: lat_bin+lon_bin)
    '''for n in range(2, max_dim_comb+1):
        for dims in itertools.combinations(bin_cols, n):
            grouped_dims = scores_df.groupby(list(dims), dropna=dropna).agg(agg_dict).reset_index()
            if flatten_columns and isinstance(grouped_dims.columns, pd.MultiIndex):
                grouped_dims.columns = [
                    "_".join([str(c) for c in col if c and c != ""]) for col in grouped_dims.columns.values
                ]
            key = "mean_over_" + "_".join(dims)
            results[key] = {
                "dims": list(dims),
                "values": grouped_dims
            }'''

    return results