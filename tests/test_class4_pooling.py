"""Class 4 scores over several bins are the scores of all their points at once."""

import numpy as np
import pandas as pd
import pytest

from oceanbench.core.class4_metrics.class4_evaluator import (
    _depth_label,
    format_class4_results,
    pool_bins,
)


def _bin(errors, **extra):
    e = np.asarray(errors, dtype=float)
    return {"rmse": float(np.sqrt((e**2).mean())), "mse": float((e**2).mean()),
            "mae": float(np.abs(e).mean()), "bias": float(e.mean()), "me": float(e.mean()),
            "count": float(e.size), **extra}


def _scores(df):
    return {(r.Metric, r.Variable): r.Value for r in df.itertuples()}


def test_pooled_scores_are_those_of_all_the_points():
    # one bin of 1 point with a large error, one of 999 points with a small one
    small, large = np.full(999, 0.1), np.array([10.0])
    pooled = pool_bins([_bin(large), _bin(small)])
    everything = np.concatenate([large, small])
    assert pooled["rmse"] == pytest.approx(np.sqrt((everything**2).mean()))
    assert pooled["mae"] == pytest.approx(np.abs(everything).mean())
    assert pooled["bias"] == pytest.approx(everything.mean())
    assert pooled["count"] == 1000
    # the mean of the two bins' RMSE, what was reported before, is 50 times larger
    assert (10.0 + 0.1) / 2 > 15 * pooled["rmse"]


def test_perfect_forecast_scores_zero_and_an_offset_is_bias_only():
    perfect = pool_bins([_bin(np.zeros(10)), _bin(np.zeros(5))])
    assert perfect["rmse"] == 0 and perfect["bias"] == 0 and perfect["crmse"] == 0
    offset = pool_bins([_bin(np.full(10, 0.3)), _bin(np.full(5, 0.3))])
    assert offset["rmse"] == pytest.approx(0.3)
    assert offset["bias"] == pytest.approx(0.3)
    assert offset["crmse"] == pytest.approx(0.0, abs=1e-6)   # sqrt of a difference of squares


def test_bins_weigh_their_point_weights_when_they_have_them():
    # cos(lat)-weighted bins: w_sum, not count, is what a bin weighs
    a, b = _bin([1.0], w_sum=0.5), _bin([3.0], w_sum=1.5)
    assert pool_bins([a, b])["bias"] == pytest.approx((0.5 * 1 + 1.5 * 3) / 2.0)


def test_depth_labels_are_the_gridded_levels():
    assert _depth_label(0.494025) == "Surface"
    assert _depth_label(47.37369) == "50m"
    assert _depth_label(pd.Interval(200.0, 250.0)) == "200m"
    assert _depth_label({"left": 500.0, "right": 600.0}) == "550m"
    # a deep level is reported at none of them; 50 m is no longer swallowed by "Surface"
    assert _depth_label(1245.291) is None
    assert _depth_label(10.0) is None
    # point observations at their own depth: drogued drifters, Argo parking-depth drift
    assert _depth_label(15.0) == "15m"
    assert _depth_label(1000.0) == "1000m"
    assert _depth_label(975.0) == "1000m" and _depth_label(900.0) is None


def test_format_reports_each_depth_separately():
    per_bins = ([_bin([0.1, 0.1], depth_bin=0.494025)] + [_bin([1.0], depth_bin=47.37369)]
                + [_bin([5.0], depth_bin=1245.291)])
    df = format_class4_results(pd.DataFrame([{"variable": "temperature", "per_bins": per_bins}]))
    scores = _scores(df)
    assert scores[("rmse", "Surface temperature")] == pytest.approx(0.1)
    assert scores[("count", "Surface temperature")] == 2
    assert scores[("rmse", "50m temperature")] == pytest.approx(1.0)
    assert not any("1245" in v for _, v in scores)


def test_format_without_depth_is_surface():
    df = format_class4_results(pd.DataFrame([{"variable": "ssh",
                                              "per_bins": [_bin([0.2]), _bin([0.2, 0.2, 0.2])]}]))
    scores = _scores(df)
    assert scores[("rmse", "Surface ssh")] == pytest.approx(0.2)
    assert scores[("count", "Surface ssh")] == 4


def test_ensemble_bins_pool_crps_spread_and_ssr():
    # bins scored from a 4-member ensemble: crps and the ensemble variance are weighted means
    a = _bin([0.1] * 3, crps=0.2, ens_var=0.04, n_members=4.0)
    b = _bin([0.3], crps=0.6, ens_var=0.16, n_members=4.0)
    pooled = pool_bins([a, b])
    assert pooled["crps"] == pytest.approx((3 * 0.2 + 0.6) / 4)
    spread = np.sqrt((3 * 0.04 + 0.16) / 4)
    assert pooled["spread"] == pytest.approx(spread)
    assert pooled["ssr"] == pytest.approx(np.sqrt(5 / 4) * spread / pooled["rmse"])
    # a deterministic forecast's bins have none of them
    assert "crps" not in pool_bins([_bin([0.1, 0.2])])
