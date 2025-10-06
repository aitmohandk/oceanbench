# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from functools import partial
from typing import List, Optional

from loguru import logger
import numpy as np
import xarray
import pandas

from oceanbench.core.distributed import DatasetProcessor
from oceanbench.core.dataset_utils import (
    Variable,
    Dimension,
    DepthLevel,
    get_length,
    get_variable,
    select_variable_day_and_depth,
    select_variable_day,
)
from oceanbench.core.lead_day_utils import lead_day_labels


VARIABLE_LABELS: dict[str, str] = {
    Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID: "surface height",
    Variable.SEA_WATER_POTENTIAL_TEMPERATURE: "temperature",
    Variable.SEA_WATER_SALINITY: "salinity",
    Variable.NORTHWARD_SEA_WATER_VELOCITY: "northward velocity",
    Variable.EASTWARD_SEA_WATER_VELOCITY: "eastward velocity",
    Variable.MIXED_LAYER_THICKNESS: "mixed layer thickness",
    Variable.GEOSTROPHIC_NORTHWARD_SEA_WATER_VELOCITY: "northward geostrophic velocity",
    Variable.GEOSTROPHIC_EASTWARD_SEA_WATER_VELOCITY: "eastward geostrophic velocity",
    Variable.UPWARD_SEA_WATER_VELOCITY: "upward velocity",
    Variable.MEAN_DYNAMIC_TOPOGRAPHY: "mean dynamic topography",
    Variable.SEA_SURFACE_HEIGHT_ABOVE_SEA_LEVEL: "height above sea level",
    Variable.SEA_SURFACE_TEMPERATURE: "sea surface temperature",
    Variable.SEA_SURFACE_SALINITY: "sea surface salinity",
}


DEPTH_LABELS: dict[DepthLevel, str] = {
    DepthLevel.SURFACE: "surface",
    DepthLevel.MINUS_50_METERS: "50m",
    DepthLevel.MINUS_200_METERS: "200m",
    DepthLevel.MINUS_550_METERS: "550m",
}


def _rmsd(data, reference_data):
    """
    Calcul du RMSD entre deux arrays, en évitant les NaNs.
    Version optimisée pour la performance.
    Args:
        data: array-like (DataArray, Dataset, ndarray, dask array, etc.)
        reference_data: array-like (DataArray, Dataset, ndarray, dask array, etc.)
    Returns:
        float: RMSD value or np.nan if no valid data
    """
    
    # Conversion directe basée sur le type
    def extract_values(obj):
        """Extrait les valeurs numpy d'un objet de manière optimisée."""
        obj_type = type(obj).__name__
        
        if obj_type in ('DataArray', 'Dataset'):
            return obj.values
        elif obj_type in ('dask.array.core.Array',):
            return obj.compute()
        elif obj_type in ('ndarray',):
            return obj
        else:
            # Fallback pour autres types
            return np.asarray(obj)
    
    # Extraction rapide des valeurs
    data_vals = extract_values(data)
    ref_vals = extract_values(reference_data)
    
    # Conversion en numpy float64
    data_flat = np.asarray(data_vals, dtype=np.float64).flatten()
    ref_flat = np.asarray(ref_vals, dtype=np.float64).flatten()
    
    # Égalisation des tailles
    min_size = min(data_flat.size, ref_flat.size)
    if data_flat.size != ref_flat.size:
        data_flat = data_flat[:min_size]
        ref_flat = ref_flat[:min_size]
    
    # Masque vectorisé (plus rapide que two separate conditions)
    valid = ~(np.isnan(data_flat) | np.isnan(ref_flat))
    
    if not valid.any():
        return np.nan
    
    # RMSD vectorisé
    diff = data_flat[valid] - ref_flat[valid]
    return np.sqrt(np.mean(diff * diff))  # Plus rapide que diff**2



def _get_rmsd(challenger_dataset, reference_dataset, variable, depth_level, lead_day):
    """
    Calcule le RMSD entre deux datasets pour une variable, un niveau de profondeur et un jour de prévision donnés.
    Utilise une version optimisée de RMSD.
    Args:
        challenger_dataset: Dataset xarray du challenger
        reference_dataset: Dataset xarray de référence
        variable: Variable à évaluer
        depth_level: Niveau de profondeur (ou None pour surface)
        lead_day: Jour de prévision (int)
    Returns:
        float: RMSD value
    """
    if depth_level:
        challenger_dataarray = select_variable_day_and_depth(challenger_dataset, variable, depth_level, lead_day)
        reference_dataarray = select_variable_day_and_depth(reference_dataset, variable, depth_level, lead_day)
    else:
        challenger_dataarray = select_variable_day(challenger_dataset, variable, lead_day)
        reference_dataarray = select_variable_day(reference_dataset, variable, lead_day)
    
    # pas de .compute() ici - laisser _rmsd_optimized gérer
    return _rmsd(challenger_dataarray, reference_dataarray)


def get_lead_days_count(dataset: xarray.Dataset) -> int:
    # always 1 day long in the current datasets
    # forecasts are managed in dc-tools library
    return 1

def _get_rmsd_for_all_lead_days(
    dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
    variable: Variable,
    depth_level: DepthLevel,
) -> list[float]:
    LEAD_DAYS_COUNT = get_lead_days_count(dataset)
    return list(
        map(
            partial(
                _get_rmsd,
                dataset,
                reference_dataset,
                variable,
                depth_level,
            ),
            range(LEAD_DAYS_COUNT),
        )
    )


def _compute_rmsd(
    datasets: List[xarray.Dataset],
    reference_datasets: List[xarray.Dataset],
    variable: Variable,
    depth_level: DepthLevel,
) -> np.ndarray:

    all_rmsd = np.array(
        list(
            map(
                partial(
                    _get_rmsd_for_all_lead_days,
                    variable=variable,
                    depth_level=depth_level,
                ),
                datasets,
                reference_datasets,
            )
        )
    )
    return all_rmsd.mean(axis=0)


def _variale_depth_label(dataset: xarray.Dataset, variable: Variable, depth_level: DepthLevel) -> str:
    if depth_level:
        return (
            f"{DEPTH_LABELS[depth_level]} {VARIABLE_LABELS[variable]}".capitalize()
            if _has_depths(dataset, variable)
            else f"surface {VARIABLE_LABELS[variable]}".capitalize()
            # else f"{DepthLevel.SURFACE} {VARIABLE_LABELS[variable]}"
        ).capitalize()
    else:
        # return f"{DepthLevel.SURFACE} {VARIABLE_LABELS[variable]}".capitalize()
        return f"{VARIABLE_LABELS[variable]}".capitalize()


def _has_depths_legacy(dataset: xarray.Dataset, variable: Variable) -> bool:
    if Dimension.DEPTH.dimension_name_from_dataset(get_variable(dataset, variable)) is None:
        return False
    else:
        return Dimension.DEPTH.dimension_name_from_dataset(dataset) in get_variable(dataset, variable).coords


def _has_depths(dataset: xarray.Dataset, variable: Variable) -> bool:
    """
    Vérifie si une variable a une dimension de profondeur.
    
    Args:
        dataset: Dataset xarray
        variable: Variable à tester
        
    Returns:
        bool: True si la variable a une dimension depth
    """
    try:
        # Obtenir la variable
        var_data = get_variable(dataset, variable)
        
        # Liste des noms possibles pour la dimension depth
        depth_names = ['depth', 'z', 'lev', 'level', 'deptht', 'bottom']
        
        # Vérifier si une dimension depth existe dans la variable
        var_dims = list(var_data.dims)
        has_depth_dim = any(depth_name in var_dims for depth_name in depth_names)
        
        # Vérifier aussi dans les coordonnées de la variable
        var_coords = list(var_data.coords)
        has_depth_coord = any(depth_name in var_coords for depth_name in depth_names)
        
        return has_depth_dim or has_depth_coord
        
    except Exception as e:
        logger.debug(f"Error checking depth dimension for variable {variable}: {e}")
        return False

def _is_surface(depth_level: DepthLevel) -> bool:
    return depth_level == DepthLevel.SURFACE


def _variable_and_depth_combinations(
    ref_dataset: xarray.Dataset, 
    challenger_dataset: xarray.Dataset, 
    variables: list[Variable],
    depth_levels: Optional[List[DepthLevel]],
    depth_dim: str = 'depth',
) -> list[tuple[Variable, DepthLevel]]:
    """
    Génère toutes les combinaisons (variable, depth_level) valides pour un dataset.
    
    Args:
        ref_dataset: Dataset xarray de référence
        challenger_dataset: Dataset xarray challenger
        variables: Liste des variables à évaluer
        depth_levels: Liste des niveaux de profondeur (peut être None)
    
    Returns:
        Liste de tuples (Variable, DepthLevel ou None)
    """
    list_combs = []
    
    def _depth_level_exists_in_dataset(dataset: xarray.Dataset, variable: Variable, depth_level: DepthLevel) -> bool:
        """Vérifie si un depth_level existe dans la dimension depth d'une variable dans un dataset."""
        try:
            var_data = get_variable(dataset, variable)
                
            # Vérifier si la valeur du depth_level existe dans les coordonnées
            depth_values = dataset[depth_dim].values
            depth_level_value = depth_level.value  # Supposant que DepthLevel a un attribut .value
            
            # Tolérance pour les comparaisons de flottants
            tolerance = 1e-3
            return any(abs(float(dv) - float(depth_level_value)) < tolerance for dv in depth_values)
            
        except Exception as e:
            logger.debug(f"Error checking depth level {depth_level} for variable {variable}: {e}")
            return False
    
    if depth_levels is not None:
        # Si des niveaux de profondeur sont spécifiés
        for variable in variables:
            if _has_depths(ref_dataset, variable) and _has_depths(challenger_dataset, variable):
                # Variable avec profondeur : vérifier que chaque depth_level existe dans les deux datasets
                for depth_level in depth_levels:
                    if (_depth_level_exists_in_dataset(ref_dataset, variable, depth_level) and 
                        _depth_level_exists_in_dataset(challenger_dataset, variable, depth_level)):
                        list_combs.append((variable, depth_level))
            else:
                # Variable sans profondeur : utiliser None comme depth_level
                list_combs.append((variable, None))
    else:
        # Aucun niveau de profondeur spécifié : toutes les variables avec None
        for variable in variables:
            list_combs.append((variable, None))
    
    return list_combs


def rmsd_legacy(
    dataset_processor: DatasetProcessor,
    challenger_datasets: List[xarray.Dataset],
    reference_datasets: List[xarray.Dataset],
    variables: List[Variable],
    depth_levels: Optional[List[DepthLevel]] = DEPTH_LABELS,
) -> pandas.DataFrame:
    """ Calcule le RMSD entre des datasets challengers et de référence pour des variables et niveaux de profondeur donnés.
    Args:
        dataset_processor: Instance de DatasetProcessor pour le traitement distribué
        challenger_datasets: Liste des datasets challengers
        reference_datasets: Liste des datasets de référence
        variables: Liste des variables à évaluer
        depth_levels: Liste des niveaux de profondeur (ou None)
    Returns:
        pandas.DataFrame: DataFrame des scores RMSD
    """

    all_combinations = _variable_and_depth_combinations(
        reference_datasets[0],
        challenger_datasets[0],
        variables,
        depth_levels,
    )

    if len(all_combinations) == 2:
        variable = all_combinations[0]
        depth_level = all_combinations[1]
        scores = {
            _variale_depth_label(challenger_datasets[0], variable, depth_level): list(
                _compute_rmsd(
                    challenger_datasets,
                    reference_datasets,
                    variable,
                    depth_level,
                )
            )
        }
    else:
        scores = {
            _variale_depth_label(challenger_datasets[0], variable, depth_level): list(
                _compute_rmsd(
                    challenger_datasets,
                    reference_datasets,
                    variable,
                    depth_level,
                )
            )
            for (variable, depth_level) in all_combinations
        }

    LEAD_DAYS_COUNT = get_lead_days_count(challenger_datasets[0])
    score_dataframe = pandas.DataFrame(scores)
    score_dataframe.index = lead_day_labels(1, LEAD_DAYS_COUNT)
    print(score_dataframe.to_markdown())
    return score_dataframe.T



def rmsd(
    challenger_datasets: List[xarray.Dataset],
    reference_datasets: List[xarray.Dataset],
    variables: List[Variable],
    depth_levels: Optional[List[DepthLevel]] = DEPTH_LABELS,
) -> pandas.DataFrame:
    """ Calcule le RMSD entre des datasets challengers et de référence pour des variables et niveaux de profondeur donnés.
    Args:
        dataset_processor: Instance de DatasetProcessor pour le traitement distribué
        challenger_datasets: Liste des datasets challengers
        reference_datasets: Liste des datasets de référence
        variables: Liste des variables à évaluer
        depth_levels: Liste des niveaux de profondeur (ou None)
    Returns:
        pandas.DataFrame: DataFrame des scores RMSD
    """
    dataset_processor = None
    all_combinations = _variable_and_depth_combinations(
        reference_datasets[0],
        challenger_datasets[0],
        variables,
        depth_levels,
    )

    if len(all_combinations) == 2:
        variable = all_combinations[0]
        depth_level = all_combinations[1]
        
        # Soumission de la tâche au client Dask
        if dataset_processor.client is not None:
            future = dataset_processor.client.submit(
                _compute_rmsd,
                challenger_datasets,
                reference_datasets,
                variable,
                depth_level,
            )
            rmsd_result = future.result()  # Attendre le résultat
        else:
            # Fallback si pas de client Dask
            rmsd_result = _compute_rmsd(
                challenger_datasets,
                reference_datasets,
                variable,
                depth_level,
            )
        
        scores = {
            _variale_depth_label(challenger_datasets[0], variable, depth_level): list(rmsd_result)
        }
    else:
        # Soumission parallèle de toutes les tâches
        if dataset_processor is not None and dataset_processor.client is not None:
            futures = []
            for (variable, depth_level) in all_combinations:
                future = dataset_processor.client.submit(
                    _compute_rmsd,
                    challenger_datasets,
                    reference_datasets,
                    variable,
                    depth_level,
                )
                futures.append((variable, depth_level, future))
            
            # Collecte des résultats
            scores = {}
            for variable, depth_level, future in futures:
                rmsd_result = future.result()  # Attendre le résultat
                scores[_variale_depth_label(challenger_datasets[0], variable, depth_level)] = list(rmsd_result)
        else:
            # Fallback si pas de client Dask
            scores = {
                _variale_depth_label(challenger_datasets[0], variable, depth_level): list(
                    _compute_rmsd(
                        challenger_datasets,
                        reference_datasets,
                        variable,
                        depth_level,
                    )
                )
                for (variable, depth_level) in all_combinations
            }
    # return scores
    LEAD_DAYS_COUNT = get_lead_days_count(challenger_datasets[0])
    score_dataframe = pandas.DataFrame(scores)
    score_dataframe.index = lead_day_labels(1, LEAD_DAYS_COUNT)
    # print(score_dataframe.to_markdown())
    score_dataframe = score_dataframe.T
    return score_dataframe