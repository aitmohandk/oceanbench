from typing import Callable, Union, List, Tuple, Optional
import pandas as pd
import geopandas as gpd
import numpy as np
import xarray as xr


'''class UnifiedObservationView:
    def __init__(
        self,
        source: Union[xr.Dataset, List[xr.Dataset], pd.DataFrame, gpd.GeoDataFrame],
        load_fn: Optional[Callable[[str], xr.Dataset]] = None
    ):
        """
        Parameters:
            source: either
                - one or more xarray Datasets (data already loaded)
                - a DataFrame/GeoDataFrame containing metadata, including file links
            load_fn: a callable that loads a dataset given a link (required if source is a DataFrame)
        """
        self.is_metadata = isinstance(source, (pd.DataFrame, gpd.GeoDataFrame))
        self.load_fn = load_fn

        if self.is_metadata:
            if self.load_fn is None:
                raise ValueError("A `load_fn(link: str)` must be provided when using metadata.")
            self.meta_df = source
        else:
            self.datasets = source if isinstance(source, list) else [source]


    def open_concat_in_time(self, time_interval: tuple) -> xr.Dataset:
        """
        Filtre les métadonnées selon l'intervalle de temps, ouvre les fichiers correspondants,
        puis concatène les datasets le long de la dimension 'time'.

        Parameters
        ----------
        time_interval : tuple
            (start_time, end_time) sous forme de pd.Timestamp ou de string compatible pandas.

        Returns
        -------
        xr.Dataset
            Dataset concaténé sur la dimension 'time'.
        """
        t0, t1 = time_interval
        # Filtrage des métadonnées
        filtered = self.meta_df[
            (self.meta_df["date_start"] <= t1) & (self.meta_df["date_end"] >= t0)
        ]
        if filtered.empty:
            raise ValueError("Aucune donnée dans l'intervalle de temps demandé.")

        # Ouverture des fichiers NetCDF/Zarr
        datasets = [self.load_fn(row["path"]) for _, row in filtered.iterrows()]

        # Concaténation sur la dimension 'time'
        combined = xr.concat(datasets, dim="time")
        # Optionnel : trier et supprimer les doublons temporels
        combined = combined.sortby("time")
        combined = combined.sel(time=slice(t0, t1))
        return combined

    def filter_by_time(self, time_range: Tuple[pd.Timestamp, pd.Timestamp]) -> List[xr.Dataset]:
        """
        Returns a list of datasets that fall within the time window.
        If source is metadata, loads only the required datasets.
        """
        t0, t1 = time_range

        if self.is_metadata:
            filtered = self.meta_df[
                (self.meta_df["date_start"] >= t0) & (self.meta_df["date_end"] <= t1)
            ]
            if filtered.empty:
                return []

            return [self.load_fn(row["link"]) for _, row in filtered.iterrows()]
        else:
            return [
                ds.sel(time=slice(t0, t1)) for ds in self.datasets
                if "time" in ds.dims or "time" in ds.coords
            ]

    def filter_by_time_and_region(
        self,
        time_range: Tuple[pd.Timestamp, pd.Timestamp],
        lon_bounds: Tuple[float, float],
        lat_bounds: Tuple[float, float]
    ) -> List[xr.Dataset]:
        """
        Filters by both time and bounding box [lon_min, lon_max], [lat_min, lat_max].
        Only applies to datasets that contain time and spatial coordinates.
        """
        t0, t1 = time_range
        lon_min, lon_max = lon_bounds
        lat_min, lat_max = lat_bounds

        if self.is_metadata:
            filtered = self.meta_df[
                (self.meta_df["date_start"] >= t0) & (self.meta_df["date_end"] <= t1) &
                (self.meta_df["lon"] >= lon_min) & (self.meta_df["lon"] <= lon_max) &
                (self.meta_df["lat"] >= lat_min) & (self.meta_df["lat"] <= lat_max)
            ]
            return [self.load_fn(row["link"]) for _, row in filtered.iterrows()]
        else:
            result = []
            for ds in self.datasets:
                if not all(k in ds.coords for k in ["lat", "lon", "time"]):
                    continue
                ds_subset = ds.sel(
                    time=slice(t0, t1),
                    longitude=slice(lon_min, lon_max),
                    latitude=slice(lat_min, lat_max)
                )
                result.append(ds_subset)
            return result'''


def apply_qc_mask(data: xr.DataArray, qc: xr.DataArray, good_qc: int | list[int] = 0) -> xr.DataArray:
    """
    Applique un masque QC à un DataArray d'observations ou de modèles.

    Parameters
    ----------
    data : xr.DataArray
        Données à masquer.
    qc : xr.DataArray
        Drapeaux de contrôle qualité (mêmes dimensions que data).
    good_qc : int ou list[int]
        Valeur(s) de QC considérées comme valides.

    Returns
    -------
    xr.DataArray
        DataArray masqué (valeurs invalides remplacées par NaN).
    """
    if isinstance(good_qc, (list, tuple, np.ndarray)):
        mask = np.isin(qc, good_qc)
    else:
        mask = (qc == good_qc)
    return data.where(mask)

def area_weights(lat: xr.DataArray) -> xr.DataArray:
    """
    Calcule les poids de surface pour une latitude (pour une moyenne pondérée).

    Parameters
    ----------
    lat : xr.DataArray
        Tableau des latitudes (en degrés).

    Returns
    -------
    xr.DataArray
        Poids de surface (mêmes dimensions que lat).
    """
    # Si lat est une coordonnée 1D, on la diffuse sur les autres dimensions si besoin
    weights = np.cos(np.deg2rad(lat))
    # Diffuser si lat n'est pas déjà aligné avec les autres dims
    if hasattr(lat, "dims") and len(lat.dims) == 1:
        for d in lat.dims:
            weights = xr.DataArray(weights, dims=lat.dims, coords={d: lat})
    return weights

def flatten_valid(data: xr.DataArray) -> np.ndarray:
    """
    Aplati un DataArray en 1D en ne gardant que les valeurs valides (non-NaN).

    Parameters
    ----------
    data : xr.DataArray

    Returns
    -------
    np.ndarray
        Tableau 1D des valeurs valides.
    """
    return data.values[np.isfinite(data.values)]