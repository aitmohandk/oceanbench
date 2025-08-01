# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from datetime import datetime, timedelta
from typing import List
import xarray as xr
from xarray import Dataset
import copernicusmarine
import logging
from loguru import logger

logger = logging.getLogger("copernicusmarine")
logger.setLevel(level=logging.WARNING)


def _glorys_subset(start_datetime: datetime) -> Dataset:
    return copernicusmarine.open_dataset(
        dataset_id="cmems_mod_glo_phy_myint_0.083deg_P1D-m",
        dataset_version="202311",
        variables=["thetao", "zos", "uo", "vo", "so"],
        start_datetime=start_datetime,
        end_datetime=start_datetime + timedelta(days=10),
    )


def _to_1_4(glorys_dataset: Dataset) -> Dataset:
    initial_datetime = datetime.fromisoformat(str(glorys_dataset["time"][0].values))
    initial_datetime_string = initial_datetime.strftime("%Y%m%d")
    return xr.open_dataset(
        f"https://minio.dive.edito.eu/project-oceanbench/public/glorys14_full_2024/{initial_datetime_string}.zarr",
        engine="zarr",
    )


def _glorys_datasets(challenger_dataset: Dataset) -> Dataset:
    start_datetime = datetime.fromisoformat(str(challenger_dataset["time"][0].values))
    return _to_1_4(_glorys_subset(start_datetime))


def glorys_datasets(challenger_datasets: List[Dataset]) -> List[Dataset]:
    return list(map(_glorys_datasets, challenger_datasets))
