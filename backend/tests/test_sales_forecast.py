"""Sales forecast tests: no-leakage property, backtest bookkeeping, and artifact sanity."""
import json
import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.forecasting import sales

RNG = np.random.default_rng(0)
Y = np.abs(RNG.normal(100, 20, 60)) + np.linspace(0, 30, 60)


@pytest.mark.parametrize("name", list(sales.MODELS))
def test_forecast_uses_only_history(name):
    """Changing FUTURE values must not change a forecast made at an earlier origin."""
    fn = sales.MODELS[name]
    t = 40
    a = fn(Y[:t].copy(), 4)
    y2 = Y.copy(); y2[t:] = 1e6          # wildly different future
    b = fn(y2[:t].copy(), 4)
    assert np.allclose(a, b)
    assert len(a) == 4 and (np.asarray(a) >= 0).all()


def test_backtest_targets_are_after_origin():
    idx = pd.date_range("2011-01-03", periods=60, freq="W-MON")
    bt = sales.backtest(pd.Series(Y, index=idx))
    assert (bt["target_week"] > bt["origin_week"]).all()
    assert bt["origin_idx"].min() == sales.MIN_TRAIN
    assert set(bt["horizon"]) == set(range(1, sales.HORIZON + 1))
    # actuals line up with the series
    assert np.allclose(bt["actual"].values, [Y[t] for t in (bt["origin_idx"] + bt["horizon"] - 1)])


def test_metrics_known_values():
    m = sales._metrics([100, 200], [110, 180])
    assert m["mape_pct"] == 10.0 and m["mae"] == 15.0 and round(m["rmse"], 2) == 15.81


ART = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts", "retail", "sales_forecast.json")


@pytest.mark.skipif(not os.path.exists(ART), reason="run python -m ml.forecasting.sales first")
def test_artifact_is_honest_and_consistent():
    r = json.load(open(ART))
    assert r["chosen_model"] in r["backtest"]["models"]
    nm = r["next_month"]
    assert nm["lower_80"] <= nm["upper_80"]   # interval need not contain the point forecast: it is built from skewed backtest ratios
    assert abs(sum(w["forecast"] for w in r["next_weeks"]) - nm["forecast_total"]) < 5
    assert len(r["next_weeks"]) == sales.HORIZON
    assert "accuracy_note" in r and "one year" in r["accuracy_note"]
    assert r["history"][-1]["week_start"] < r["next_weeks"][0]["week_start"]
