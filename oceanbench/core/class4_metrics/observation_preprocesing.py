
from typing import Literal, Optional, Union
import pandas as pd
import numpy as np
import xarray as xr
import geopandas as gpd
from scipy.spatial import cKDTree
import pyinterp
import pyinterp.backends.xarray
# import xesmf as xe


# ------------------------------
# Prétraitement des observations
# ------------------------------
def preprocess_observations(
    obs: Union[pd.DataFrame, gpd.GeoDataFrame, xr.Dataset],
    varname: str,
    level: Literal["L1", "L2", "L3", "L4"]
) -> pd.DataFrame:
    """
    Normalize observation data to a flat DataFrame with columns: time, lat, lon, varname

    Parameters
    ----------
    obs : observation data (GeoDataFrame, DataFrame or xarray.Dataset)
    varname : str
        Variable to extract
    level : str
        Observation data level (L1, L2, L3...)

    Returns
    -------
    pd.DataFrame
    """
    # --- xarray.Dataset ---
    if isinstance(obs, xr.Dataset):
        # L3/L4: Gridded data (lat/lon as dims)
        if level in ["L3", "L4"]:
            df = obs[varname].to_dataframe().reset_index()
            return df.dropna(subset=["lat", "lon", "time", varname])

        # L2: Point observations (lat/lon/time as variables, not dims)
        if level == "L2":
            # Try to extract as columns
            cols = ["time", "lat", "lon", varname]
            # Try to get as variables (not dims)
            try:
                arrs = [obs[c].values for c in cols]
                df = pd.DataFrame({c: obs[c].values for c in cols})
                # QC mask if present
                if "quality_flag" in obs.variables:
                    qc = obs["quality_flag"].values
                    df = df[qc == 1]
                return df.dropna()
            except Exception:
                # Fallback: try to convert to DataFrame and filter columns
                df = obs.to_dataframe().reset_index()
                if all(c in df.columns for c in cols):
                    if "quality_flag" in df.columns:
                        df = df[df["quality_flag"] == 1]
                    return df[cols].dropna()
                else:
                    raise ValueError(f"Could not extract columns {cols} from xarray.Dataset for L2.")

        # L1: Raw data, not georeferenced
        if level == "L1":
            raise NotImplementedError("L1 raw data requires external georeferencing.")

    # --- DataFrame / GeoDataFrame ---
    if isinstance(obs, (pd.DataFrame, gpd.GeoDataFrame)):
        cols = ["time", "lat", "lon", varname]
        if not all(c in obs.columns for c in cols):
            raise ValueError(f"Missing columns in observation data: expected {cols}")
        df = obs[cols].dropna()
        if level == "L2" and "quality_flag" in obs.columns:
            df = df[obs["quality_flag"] == 1]
        return df

    # --- Other types or fallback ---
    raise ValueError(f"Unsupported combination of input type and level: {type(obs)} / {level}")

# -----------------------------------
# Interpolation vectorisée + matching
# -----------------------------------

def idw_interpolate(x, y, z, xi, yi, power=2):
    dist = np.sqrt((x[:, None] - xi[None, :])**2 + (y[:, None] - yi[None, :])**2)
    weights = 1 / (dist**power + 1e-12)
    weights /= weights.sum(axis=0)
    return np.dot(z, weights)


def match_and_interpolate_vectorized(
    obs_df: pd.DataFrame,
    model_ds: xr.Dataset,
    varname: str,
    delta_t: pd.Timedelta = pd.Timedelta("6h"),
    method: Literal["pyinterp", "kdtree", "xesmf", "idw"] = "pyinterp",
    binning: Optional[dict] = None,
    return_format: Literal["dataframe", "dataset", "geodataframe"] = "dataframe",
) -> Union[pd.DataFrame, xr.Dataset, gpd.GeoDataFrame]:
    """
    Match observations with model data in time, interpolate in space (vectorized), optionally bin.

    Parameters
    ----------
    obs_df : pd.DataFrame
        Cleaned observation table
    model_ds : xr.Dataset
        Model dataset
    varname : str
        Name of the variable to compare
    delta_t : pd.Timedelta
        Temporal tolerance
    method : str
        Interpolation method: pyinterp, kdtree, xesmf, idw
    binning : dict
        Optional dict for binning (e.g. {"time": "1D", "lat": 1.0, "lon": 1.0})
    return_format : str
        Output format: dataframe, dataset, or geodataframe

    Returns
    -------
    pd.DataFrame or xr.Dataset or gpd.GeoDataFrame
    """
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
            mask = matched_times == unique_time
            if not np.any(mask):
                continue

            sub_lats = lats[mask]
            sub_lons = lons[mask]
            sub_obs = values[mask]

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

            elif method == "idw":
                lon2d, lat2d = np.meshgrid(model_slice["lon"].values, model_slice["lat"].values)
                model_vals = idw_interpolate(
                    lon2d.ravel(), lat2d.ravel(), model_slice[varname].values.ravel(),
                    sub_lons, sub_lats
                )

            else:
                raise ValueError(f"Unknown interpolation method: {method}")

            obs_df.loc[group.index[mask], "matched_time"] = unique_time
            obs_df.loc[group.index[mask], "model_value"] = model_vals
            obs_df.loc[group.index[mask], "error"] = model_vals - sub_obs

    obs_df = obs_df.dropna(subset=["model_value"])

    if binning:
        obs_df = apply_binning(obs_df, binning)

    if return_format == "dataframe":
        return obs_df
    elif return_format == "geodataframe":
        return gpd.GeoDataFrame(
            obs_df, geometry=gpd.points_from_xy(obs_df.lon, obs_df.lat), crs="EPSG:4326"
        )
    elif return_format == "dataset":
        return xr.Dataset.from_dataframe(obs_df.set_index("time"))
    else:
        raise ValueError(f"Invalid return_format: {return_format}")


# --------------------
# Binning (optionnel)
# --------------------

def apply_binning(df: pd.DataFrame, binning: dict) -> pd.DataFrame:
    df = df.copy()
    if "time" in binning:
        df["time_bin"] = df["time"].dt.floor(binning["time"])
    if "lat" in binning:
        df["lat_bin"] = (df["lat"] // binning["lat"]) * binning["lat"]
    if "lon" in binning:
        df["lon_bin"] = (df["lon"] // binning["lon"]) * binning["lon"]
    if "depth" in binning and "depth" in df.columns:
        df["depth_bin"] = (df["depth"] // binning["depth"]) * binning["depth"]

    return (
        df.groupby([col for col in df.columns if col.endswith("_bin")])
        .agg({
            "model_value": "mean",
            "error": ["mean", "std", "count"]
        })
        .reset_index()
    )
