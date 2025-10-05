# -*- coding: utf-8 -*-
"""
Distributed processing utilities for OceanBench
"""
import logging
from xmlrpc import client

import numpy as np
import os
from pathlib import Path 
from tqdm import tqdm
import xarray as xr
import shutil
import tempfile
from typing import Any, Callable, Dict, List, Optional, Sequence

from dask.distributed import Client, LocalCluster, as_completed, progress

import dask
from loguru import logger


class DatasetProcessor:
    def __init__(
        self,
        client: Optional[Client] = None,
        distributed: bool = False,
        n_workers: int = 4,
        threads_per_worker: int = 1,
        memory_limit: str = "4GB",
    ):
        """
        Parameters
        ----------
        client: Optional[dask.distributed.Client]
            Si fourni, la classe réutilise ce client (ne le ferme pas).
        distributed: bool
            Si True et client None -> crée un LocalCluster + Client géré par la classe.
        n_workers, threads_per_worker: int
            Paramètres du LocalCluster si créé automatiquement.
        """
        self._owns_client = False
        self.client: Optional[Client] = None
        self.cluster = None

        # Création du répertoire temporaire
        self._temp_dir = tempfile.mkdtemp(prefix="apply_ufunc_executor_")
        logger.info(f"Created temporary directory: {self._temp_dir}")
        
        # Cache pour les fichiers temporaires
        self._temp_files_cache: List[str] = []

        if client is not None:
            self.client = client
            self._owns_client = False
        elif distributed:
            # Configuration client
            dask.config.set({
                'distributed.p2p.storage.disk': False,
                'distributed.scheduler.work-stealing': False,
                'distributed.scheduler.work-stealing-interval': '0s',
                'distributed.comm.compression': None,
                'distributed.comm.timeouts.tcp': '300s',
                'distributed.worker.memory.target': 0.7,
                'distributed.worker.memory.spill': 0.8,
                'array.chunk-size': '64MB',
                'distributed.worker.daemon': False,
                'distributed.admin.event-loop-monitor-interval': '8000ms',
            })

            self.cluster = LocalCluster(
                n_workers=n_workers,
                threads_per_worker=threads_per_worker,
                memory_limit=memory_limit,
                # direct_to_workers=True,
                local_directory=self._temp_dir,
                # protocol="tcp://",
                processes=True,
                #dashboard_address=None,  # Désactiver dashboard
                #silence_logs=True,
            )
            self.client = Client(self.cluster)
            self._owns_client = True
            logger.info(f"\n\n\n============= LINK TO DASHBOARD DASK : {self.client.dashboard_link} =============\n\n")
            dask.config.set({'logging': {'distributed.worker': 'WARNING'}})


    def add_temp_file(self, file_path: str) -> None:
        """
        Ajoute un fichier temporaire au cache pour nettoyage ultérieur.
        
        Args:
            file_path (str): Chemin vers le fichier temporaire à enregistrer
        """
        abs_path = str(Path(file_path).resolve())
        if abs_path not in self._temp_files_cache:
            self._temp_files_cache.append(abs_path)
            logger.debug(f"Added temp file to cache: {abs_path}")

    def create_temp_file(self, suffix: str = ".zarr", prefix: str = "temp_") -> str:
        """
        Crée un fichier temporaire dans le répertoire temporaire et l'ajoute au cache.
        
        Args:
            suffix (str): Extension du fichier (par défaut ".zarr")
            prefix (str): Préfixe du nom de fichier (par défaut "temp_")
            
        Returns:
            str: Chemin complet vers le fichier temporaire créé
        """
        import uuid
        
        # Génération d'un nom unique
        unique_id = uuid.uuid4().hex[:8]
        filename = f"{prefix}{unique_id}{suffix}"
        temp_file_path = Path(self._temp_dir) / filename
        
        # Ajouter au cache
        self.add_temp_file(str(temp_file_path))
        
        logger.debug(f"Created temp file: {temp_file_path}")
        return str(temp_file_path)

    def list_temp_files(self) -> List[str]:
        """
        Retourne la liste des fichiers temporaires dans le cache.
        
        Returns:
            List[str]: Liste des chemins des fichiers temporaires
        """
        return self._temp_files_cache.copy()

    def cleanup_temp_files(self) -> None:
        """
        Supprime tous les fichiers temporaires du cache et le répertoire temporaire.
        """
        # Supprimer les fichiers individuels du cache
        for file_path in self._temp_files_cache:
            try:
                path_obj = Path(file_path)
                if path_obj.exists():
                    if path_obj.is_file():
                        path_obj.unlink()
                        logger.debug(f"Deleted temp file: {file_path}")
                    #elif path_obj.is_dir():   # TODO : check temp file cleaning
                    #    shutil.rmtree(path_obj)
                    #    logger.debug(f"Deleted temp directory: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to delete temp file {file_path}: {e}")
        
        # Vider le cache
        self._temp_files_cache.clear()
        
        # Supprimer le répertoire temporaire principal   # TODO check this
        '''if self._temp_dir and Path(self._temp_dir).exists():
            try:
                shutil.rmtree(self._temp_dir)
                logger.info(f"Cleaned up temporary directory: {self._temp_dir}")
                self._temp_dir = None
            except Exception as e:
                logger.warning(f"Failed to cleanup temporary directory: {e}")'''


    # context manager support
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def close(self):
        """Ferme le client/cluster si la classe les a créés."""
        try:
            # Nettoyage des fichiers temporaires avant fermeture
            self.cleanup_temp_files()
            if self._owns_client and self.client is not None:
                self.client.close()
            if self.cluster is not None:
                self.cluster.close()
        except Exception:
            pass
        finally:
            self.client = None
            self.cluster = None
            self._owns_client = False

    def __del__(self):
        """
        Destructeur - nettoyage automatique à la destruction de l'instance.
        """
        try:
            self.cleanup_temp_files()
        except Exception:
            # Ignorer les erreurs dans le destructeur
            pass

    # -----------------------------
    # helpers
    # -----------------------------
    @staticmethod
    def _ensure_numpy(x):
        """Retourne un numpy array pour une entrée x (DataArray ou array-like)."""
        if isinstance(x, xr.DataArray):
            return np.asarray(x.values)
        return np.asarray(x)

    @staticmethod
    def _rechunk_for_apply(da: xr.DataArray, lat_name: str, lon_name: str):
        """
        Rechunk pour que lat/lon soient single-chunk (core dims) et
        time/depth chunk=1 pour que apply_ufunc fonctionne sur slices.
        """
        rechunk_spec = {}
        if lon_name in da.dims:
            rechunk_spec[lon_name] = -1
        if lat_name in da.dims:
            rechunk_spec[lat_name] = -1
        for d in ("time", "depth"):
            if d in da.dims:
                rechunk_spec[d] = 1
        if rechunk_spec:
            return da.chunk(rechunk_spec)
        return da

    @staticmethod
    def _select_depth_nearest(
        da: xr.DataArray,
        target_depth: float,
        depth_name: str = "depth",
        tol: float = 1e-3,
    ):
        """
        Sélectionne la tranche la plus proche de target_depth.
        Retourne (da_squeezed, actual_depth_scalar).
        Si la variable n'a pas de dimension depth -> (da, None).
        """
        if depth_name not in da.dims:
            return da, None

        # Récupérer toutes les profondeurs
        depth_vals = da.coords[depth_name].values.astype(float)

        # Trouver la valeur la plus proche
        idx = int(np.abs(depth_vals - target_depth).argmin())
        actual = float(depth_vals[idx])

        # Vérifier la tolérance
        if abs(actual - target_depth) > tol:
            raise ValueError(
                f"No depth within tol={tol} of {target_depth}; closest={actual}"
            )

        # Sélectionner et supprimer la dimension depth
        sel = da.isel({depth_name: idx}).squeeze(drop=True)

        return sel, actual

    @staticmethod
    def rechunk_for_zarr(ds: xr.Dataset, target_chunks: dict = None) -> xr.Dataset:
        """
        Rechunk an xarray Dataset to ensure Zarr compatibility.
        
        - Ensures that each dimension has uniform chunk sizes,
        except possibly for the last chunk (allowed by Zarr).
        - Allows user-specified chunk sizes via target_chunks.
        - Falls back to a safe default if not provided.

        Parameters
        ----------
        ds : xr.Dataset
            Input dataset (possibly already chunked).
        target_chunks : dict, optional
            Desired chunk sizes per dimension (e.g., {"latitude": 256, "longitude": 256}).
            If not provided, defaults are chosen automatically.

        Returns
        -------
        xr.Dataset
            Dataset rechunked for safe writing to Zarr.
        """

        # Dimensions of the dataset
        sizes = ds.sizes

        # Defaults: one chunk along time/depth, ~256 along lat/lon if available
        default_chunks = {}
        for dim, size in sizes.items():
            if dim.lower() in ("lat", "latitude", "lon", "longitude"):
                default_chunks[dim] = min(256, size)
            else:
                # for time, depth or others: single chunk
                default_chunks[dim] = -1

        # Merge user chunks if provided
        if target_chunks is not None:
            default_chunks.update(target_chunks)

        # Apply chunking
        ds_rechunked = ds.chunk(default_chunks)

        return ds_rechunked

    @staticmethod
    def rechunk_uniform(ds, chunksize=256):
        """Rechunk dataset to uniform chunk sizes for Zarr compatibility."""
        new_chunks = {}
        for dim, sizes in ds.chunks.items():
            # Si ce dim est déjà uniforme -> on garde
            if len(set(sizes)) == 1:
                new_chunks[dim] = sizes[0]
            else:
                # chunks irréguliers : on force à une taille fixe
                new_chunks[dim] = min(chunksize, max(sizes))
        return ds.chunk(new_chunks)

    @staticmethod
    def _uniform_rechunk_dataset_for_zarr(ds: xr.Dataset, chunk_spec: Dict[str, int]) -> xr.Dataset:
        """
        Rechunk dataset en tailles uniformes par dimension (utile pour Zarr).
        chunk_spec: dict dim->chunk_size
        """
        ds = ds.unify_chunks()
        spec = {}
        for dim, c in (chunk_spec or {}).items():
            if dim in ds.dims:
                spec[dim] = min(c, ds.sizes.get(dim, c))
        if spec:
            print(f"SPEC: {spec}")
            return ds.chunk(spec)
        return ds



    @staticmethod
    def run_parallel_tasks(
        da: xr.DataArray,
        callable_fct: Callable,
        lat_name: str,
        lon_name: str,
        lat_src: np.ndarray,
        lon_src: np.ndarray,
        tgt_lat: np.ndarray,
        tgt_lon: np.ndarray,
        safe_kwargs: Dict[str, Any],
        dask_gufunc_kwargs: Dict[str, Any],
    ) -> np.ndarray:
        """Applique une fonction sur un DataArray en parallèle."""
        
        def _wrapped(data2d, lat_src, lon_src, target_lat, target_lon, **kwargs):
            return callable_fct(
                data2d, lat_src, lon_src, target_lat, target_lon, **kwargs
            )

        return xr.apply_ufunc(
            _wrapped,
            da,
            input_core_dims=[[lat_name, lon_name]],
            output_core_dims=[["_lat_out", "_lon_out"]],
            dask="parallelized",
            vectorize=True,
            kwargs=dict(
                lat_src=lat_src,
                lon_src=lon_src,
                target_lat=tgt_lat,
                target_lon=tgt_lon,
                **safe_kwargs,
            ),
            output_dtypes=[da.dtype],
            keep_attrs=True,
            dask_gufunc_kwargs=dask_gufunc_kwargs,
        )

    def scatter_data(
        self, scatter_item: Any,
        broadcast_item: Optional[bool] = True,
    ):
        return self.client.scatter(scatter_item, broadcast=broadcast_item)

    def scatter_data_list(
        self, scatter_list: List[Any],
        broadcast_list: Optional[List[bool]] = None,
    ):
        scatter_items = []
        if scatter_list is not None:
            for scatter_item, broadcast_item in zip(scatter_list, broadcast_list):
                scatter_items.append(
                    self.client.scatter(scatter_item, broadcast=broadcast_item)
                )
        return scatter_items

    def compute_delayed_tasks(
        self, delayed_tasks: List[Any],
        sync: Optional[bool] = False,
    ) -> List[Any]:
        """Compute a list of delayed tasks in parallel on workers."""

        futures = self.client.compute(delayed_tasks, sync=sync)
        results = []
        for future in tqdm(as_completed(futures), total=len(futures)):
            results.append(future.result())
        #progress(futures)
        #results = self.client.gather(futures)
        return results


    @staticmethod
    def cleanup_worker_memory():
        """Fonction de nettoyage à exécuter sur chaque worker."""
        import gc
        gc.collect()

    def run_parallel_tasks_double(
        self, da1, da2, callable_fct,
        lat_name, lon_name,
        lat_src, lon_src,
        tgt_lat, tgt_lon,
        safe_kwargs, dask_gufunc_kwargs
    ):
        """Applique une fonction sur deux DataArrays en parallèle."""
        def _wrapped(data2d_1, data2d_2, lat_src, lon_src, tgt_lat, tgt_lon, **kwargs):
            return callable_fct(data2d_1, data2d_2, lat_src, lon_src, tgt_lat, tgt_lon, **kwargs)

        return xr.apply_ufunc(
            _wrapped, da1, da2,
            input_core_dims=[[lat_name, lon_name], [lat_name, lon_name]],
            output_core_dims=[["_lat_out", "_lon_out"]],
            vectorize=True, dask="parallelized",
            kwargs=dict(lat_src=lat_src, lon_src=lon_src, tgt_lat=tgt_lat, tgt_lon=tgt_lon, **safe_kwargs),
            **dask_gufunc_kwargs,
        )


    def _apply_over_time_depth(
        self,
        ds: xr.Dataset,
        var_names: Sequence[str],
        depth_name: str,
        lat_name: str,
        lon_name: str,
        lat_src: np.ndarray,
        lon_src: np.ndarray,
        tgt_lat: np.ndarray,
        tgt_lon: np.ndarray,
        callable_fct: Callable,
        callable_kwargs: Dict[str, Any],
        dask_gufunc_kwargs: Dict[str, Any],
        tol_depth: float = 1e-3,
        mode: str = "single",   # "single" | "double"
        ds2: Optional[xr.Dataset] = None,
    ) -> Dict[str, xr.DataArray]:
        """
        Generic loop over time/depth that applies a callable either on one or two datasets.
        Parameters
        ----------
        ds: xr.Dataset
            Dataset d'entrée (doit contenir lat/lon)
        var_names: Sequence[str]
            Liste des noms de variables à traiter.
        depth_name, lat_name, lon_name: str
            Noms des dimensions dans le Dataset.
        lat_src, lon_src: np.ndarray
            Coordonnées source (1D arrays).
        tgt_lat, tgt_lon: np.ndarray
            Coordonnées cibles (1D arrays).
        callable_fct: Callable
            Fonction à appliquer. Doit avoir la signature:
                fct(data2d, lat_src, lon_src, target_lat, target_lon, **kwargs)
            où data2d est une tranche 2D (lat, lon) de la variable.
        callable_kwargs: Dict[str, Any]
            Arguments supplémentaires à passer à callable_fct.
        dask_gufunc_kwargs: Dict[str, Any]
            Arguments supplémentaires pour dask.apply_ufunc.
        tol_depth: float
            Tolérance pour la sélection de profondeur.
        mode: str
            "single" pour une seule dataset, "double" pour deux datasets (ds2 requis).
        ds2: Optional[xr.Dataset]
            Second dataset si mode="double".
        Returns
        -------
        Dict[str, xr.DataArray]
            Dictionnaire des variables résultantes après application de la fonction.
        """
        out_vars = {}

        for var in var_names:
            if var not in ds:
                continue

            da = ds[var]
            if not {lat_name, lon_name}.issubset(da.dims):
                continue

            has_time = "time" in da.dims
            has_depth = depth_name in da.dims

            time_slices_out = []
            for t in (ds.time.values if has_time else [None]):
                depth_slices_out = []
                for z in (ds[depth_name].values if has_depth else [None]):
                    # Sélection
                    sel_kwargs = {}
                    if t is not None:
                        sel_kwargs["time"] = t
                    if z is not None:
                        sel_kwargs[depth_name] = z

                    da_sel = da.sel(**sel_kwargs, method="nearest")

                    if z is not None:
                        actual_depth = float(da_sel[depth_name].values)
                        if abs(actual_depth - float(z)) > tol_depth:
                            continue

                    # --- appliquer fonction ---
                    if mode == "single":
                        da_out = self.run_parallel_tasks(
                            da_sel, callable_fct,
                            lat_name, lon_name,
                            lat_src, lon_src,
                            tgt_lat, tgt_lon,
                            callable_kwargs, dask_gufunc_kwargs,
                        )
                    elif mode == "double":
                        if ds2 is None:
                            raise ValueError("ds2 required for double mode")
                        da2_sel = ds2[var].sel(**sel_kwargs, method="nearest")
                        da_out = self.run_parallel_tasks_double(
                            da_sel, da2_sel, callable_fct,
                            lat_name, lon_name,
                            lat_src, lon_src,
                            tgt_lat, tgt_lon,
                            callable_kwargs, dask_gufunc_kwargs,
                        )
                    else:
                        raise ValueError(f"Unknown mode {mode}")

                    # coords out
                    da_out = da_out.assign_coords({"_lat_out": tgt_lat, "_lon_out": tgt_lon})
                    da_out = da_out.rename({"_lat_out": lat_name, "_lon_out": lon_name})

                    if z is not None:
                        da_out = da_out.expand_dims({depth_name: [actual_depth]})
                    if t is not None:
                        da_out = da_out.expand_dims({"time": [t]})

                    depth_slices_out.append(da_out)

                if len(depth_slices_out) > 0:
                    depth_concat = xr.concat(depth_slices_out, dim=depth_name) if has_depth else depth_slices_out[0]
                    time_slices_out.append(depth_concat)

            if len(time_slices_out) > 0:
                var_out = xr.concat(time_slices_out, dim="time") if has_time else time_slices_out[0]
                out_vars[var] = var_out

        return out_vars


    def apply_single(
        self,
        ds: xr.Dataset,
        callable_fct: Callable,
        *,
        target_grid: Dict[str, Sequence],
        var_names: Optional[Sequence[str]] = None,
        depth_name: str = "depth",
        lat_name: str = "latitude",
        lon_name: str = "longitude",
        callable_kwargs: Optional[Dict[str, Any]] = None,
        dask_gufunc_kwargs: Optional[Dict[str, Any]] = None,
        tol_depth: float = 1e-3,
        output_mode: str = "zarr",  # 'zarr'|'lazy'|'inmemory'
        output_path: str = None,
        zarr_target_chunks: dict = None,
    ) -> xr.Dataset:
        """
        Applique la fonction callable_fct sur un Dataset en utilisant un maillage cible.
        Parameters
        ----------
        ds: xr.Dataset
            Dataset d'entrée (doit contenir lat/lon)
        callable_fct: Callable
            Fonction à appliquer. Doit avoir la signature:
                fct(data2d, lat_src, lon_src, target_lat, target_lon, **kwargs)
            où data2d est une tranche 2D (lat, lon) de la variable.
        target_grid: Dict[str, Sequence]
            Dictionnaire avec 'lat' et 'lon' comme clés et les valeurs cibles.
        var_names: Optional[Sequence[str]]
            Liste des noms de variables à traiter. Si None, toutes les variables avec lat/lon sont utilisées.
        depth_name, lat_name, lon_name: str
            Noms des dimensions dans le Dataset.
        callable_kwargs: Optional[Dict[str, Any]]
            Arguments supplémentaires à passer à callable_fct.
        dask_gufunc_kwargs: Optional[Dict[str, Any]]
            Arguments supplémentaires pour dask.apply_ufunc.
        tol_depth: float
            Tolérance pour la sélection de profondeur.
        output_mode: str
            Mode de sortie: 'zarr' pour écrire dans un fichier Zarr, 'lazy' pour un Dataset paresseux, 'inmemory' pour un Dataset en mémoire.
        output_path: str
            Chemin du fichier de sortie si output_mode est 'zarr'. Si None, un fichier temporaire est créé.
        zarr_target_chunks: dict
            Spécification des chunks pour l'écriture Zarr.
        Returns
        -------
        xr.Dataset
            Dataset résultant après application de la fonction.
        """
        # target grid
        tgt_lat = np.asarray(target_grid["lat"])
        tgt_lon = np.asarray(target_grid["lon"])

        # var selection
        if var_names is None:
            var_names = [v for v, da in ds.data_vars.items() if lat_name in da.dims and lon_name in da.dims]

        safe_kwargs = callable_kwargs or {}
        dask_gufunc_kwargs = dask_gufunc_kwargs or {
            "allow_rechunk": True,
            "output_sizes": {"_lat_out": len(tgt_lat), "_lon_out": len(tgt_lon)},
        }

        lat_src = np.asarray(ds[lat_name])
        lon_src = np.asarray(ds[lon_name])

        out_vars = self._apply_over_time_depth(
            ds, var_names, depth_name, lat_name, lon_name,
            lat_src, lon_src, tgt_lat, tgt_lon,
            callable_fct, safe_kwargs, dask_gufunc_kwargs,
            tol_depth=tol_depth, mode="single"
        )

        if len(out_vars) == 0:
            raise RuntimeError("No metrics computed.")

        ds_out = xr.Dataset(out_vars)

        # writing / returning according to mode
        if output_mode == "zarr":
            # Utiliser le système de fichiers temporaires
            if output_path is None:
                output_path = self.create_temp_file(suffix=".zarr", prefix="pairwise_")
                logger.info(f"Using temporary file: {output_path}")
            
            if zarr_target_chunks is None:
                zarr_target_chunks = {}
                for d in ("time", depth_name):
                    if d in ds_out.dims:
                        zarr_target_chunks[d] = 1
            ds_out = ds_out.chunk(zarr_target_chunks)
            outp = Path(output_path)
            if outp.exists():
                import shutil; shutil.rmtree(outp)
            ds_out.to_zarr(output_path, mode="w", consolidated=True)
            ds_out.close()
            ds_out = xr.open_zarr(output_path, chunks=zarr_target_chunks)

        elif output_mode == "lazy":
            pass # return ds_out

        elif output_mode == "inmemory":
            ds_out = ds_out.compute()

        else:
            raise ValueError(f"Unknown mode {output_mode}")

        return ds_out


    def apply_double(
        self,
        ds1: xr.Dataset,
        ds2: xr.Dataset,
        callable_fct: Callable,
        *,
        target_grid: Dict[str, Sequence],
        var_names: Optional[Sequence[str]] = None,
        depth_name: str = "depth",
        lat_name: str = "latitude",
        lon_name: str = "longitude",
        callable_kwargs: Optional[Dict[str, Any]] = None,
        dask_gufunc_kwargs: Optional[Dict[str, Any]] = None,
        tol_depth: float = 1e-3,
        output_mode: str = "zarr",  # 'zarr'|'lazy'|'inmemory'
        output_path: str = None,
        zarr_target_chunks: dict = None,
    ) -> xr.Dataset:
        """
        Applique la fonction callable_fct sur deux Datasets en utilisant un maillage cible.
        Parameters
        ----------
        ds1, ds2: xr.Dataset
            Datasets d'entrée (doivent contenir lat/lon)
        callable_fct: Callable
            Fonction à appliquer. Doit avoir la signature:
                fct(data2d_1, data2d_2, lat_src, lon_src, target_lat, target_lon, **kwargs)
            où data2d_1 et data2d_2 sont des tranches 2D (lat, lon) de la variable.
        target_grid: Dict[str, Sequence]
            Grille cible pour la rééchantillonnage (doit contenir 'lat' et 'lon').
        var_names: Optional[Sequence[str]]
            Noms des variables à traiter. Si None, toutes les variables 2D (lat, lon) seront utilisées.
        """
        tgt_lat = np.asarray(target_grid["lat"])
        tgt_lon = np.asarray(target_grid["lon"])

        if var_names is None:
            var_names = [v for v, da in ds1.data_vars.items() if lat_name in da.dims and lon_name in da.dims]

        safe_kwargs = callable_kwargs or {}
        dask_gufunc_kwargs = dask_gufunc_kwargs or {
            "allow_rechunk": True,
            "output_sizes": {"_lat_out": len(tgt_lat), "_lon_out": len(tgt_lon)},
        }

        lat_src = np.asarray(ds1[lat_name])
        lon_src = np.asarray(ds1[lon_name])

        out_vars = self._apply_over_time_depth(
            ds1, var_names, depth_name, lat_name, lon_name,
            lat_src, lon_src, tgt_lat, tgt_lon,
            callable_fct, safe_kwargs, dask_gufunc_kwargs,
            tol_depth=tol_depth, mode="double", ds2=ds2
        )

        if len(out_vars) == 0:
            raise RuntimeError("No metrics computed.")

        ds_out = xr.Dataset(out_vars)

        # writing / returning according to mode
        if output_mode == "zarr":
            # Utiliser le système de fichiers temporaires
            if output_path is None:
                output_path = self.create_temp_file(suffix=".zarr", prefix="pairwise_")
            
            if zarr_target_chunks is None:
                zarr_target_chunks = {}
                for d in ("time", depth_name):
                    if d in ds_out.dims:
                        zarr_target_chunks[d] = 1
            ds_out = ds_out.chunk(zarr_target_chunks)
            outp = Path(output_path)
            if outp.exists():
                shutil.rmtree(outp)
            logger.info(f"Saving dataset to temporary file: {output_path}")
            ds_out.to_zarr(output_path, mode="w", consolidated=True)
            ds_out.close()
            ds_out = xr.open_zarr(output_path, chunks=zarr_target_chunks)

        elif output_mode == "lazy":
            pass # return ds_out

        elif output_mode == "inmemory":
            ds_out = ds_out.compute()

        else:
            raise ValueError(f"Unknown mode {output_mode}")

        return ds_out


    def get_dataset_processor_workers(self, dataset_processor):
        """Retourne le nombre de workers du DatasetProcessor."""
        if hasattr(dataset_processor, 'client') and dataset_processor.client:
            try:
                workers_info = dataset_processor.client.scheduler_info()['workers']
                return len(workers_info)
            except:
                return 0
        return 0
