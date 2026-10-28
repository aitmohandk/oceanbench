"""Gridded RMSD: area-weighted, and never between fields of different shapes."""

import numpy as np
import pytest
import xarray as xr

from oceanbench.core.rmsd import _rmsd

LAT = np.array([0.0, 60.0])
LON = np.array([0.0, 1.0, 2.0])


def _field(values, dims=("lat", "lon")):
    return xr.DataArray(np.asarray(values, float), dims=dims, coords={"lat": LAT, "lon": LON})


def test_perfect_field_is_zero():
    f = _field(np.ones((2, 3)))
    assert _rmsd(f, f) == 0.0


def test_points_weigh_their_cell_area():
    truth = _field(np.zeros((2, 3)))
    # error 1 at the equator, 0 at 60N: weights 1 and 0.5
    pred = _field([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
    assert _rmsd(pred, truth) == pytest.approx(np.sqrt(1.0 / 1.5))
    assert _rmsd(pred, truth, area_weighted=False) == pytest.approx(np.sqrt(0.5))


def test_dimension_order_does_not_matter():
    truth = _field(np.arange(6.0).reshape(2, 3))
    pred = truth + 0.2
    assert _rmsd(pred.transpose("lon", "lat"), truth) == pytest.approx(0.2)


def test_different_shapes_are_an_error():
    with pytest.raises(ValueError, match="different shapes"):
        _rmsd(_field(np.zeros((2, 3))), xr.DataArray(np.zeros((2, 2)), dims=("lat", "lon")))


def test_nan_points_are_left_out():
    truth = _field([[0.0, np.nan, 0.0], [0.0, 0.0, 0.0]])
    pred = _field(np.full((2, 3), 0.5))
    assert _rmsd(pred, truth) == pytest.approx(0.5)
