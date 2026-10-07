"""
Next-month sales forecast for the Online Retail dataset (the only dataset with real transaction dates).

SCOPE / HONESTY NOTES (also written into the output artifact):
  * Target = weekly revenue (Quantity x UnitPrice) over COMPLETE Mon-Sun weeks. "Next month" = the next
    4 weeks (28 days), reported as a total and week by week.
  * The mirror used here contains only transactions with a CustomerID, so this is revenue from identified
    customers, not total company sales.
  * Only ~52 weeks of history exist, i.e. ONE year. A model cannot learn yearly seasonality from one year,
    and the backtest window includes the Sep-Nov holiday ramp, so errors there are large and expected.
  * Models are deliberately simple (scikit-learn / numpy only): naive, 4-week moving average, simple
    exponential smoothing, linear trend, ridge regression on lags.
  * Backtest = rolling-origin (expanding window): at each origin the model sees ONLY data up to that
    origin, forecasts the next 4 weeks, and is scored on what actually happened. No future leakage.
  * Prediction intervals are empirical: quantiles of the backtest actual/forecast ratios of the chosen model.
    They assume future errors look like past errors, which is a weak assumption with ~25 origins.
  * The best model is chosen on the same backtest it is reported on, so its score is mildly optimistic.

Run from backend/:  python -m ml.forecasting.sales
"""
from __future__ import annotations
import json
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_PATH = os.path.join(BASE, "data", "retail", "online_retail_raw.csv")
OUT_PATH = os.path.join(BASE, "artifacts", "retail", "sales_forecast.json")

HORIZON = 4            # weeks ahead == "next month"
MIN_TRAIN = 26         # first backtest origin has at least half a year of history
INTERVAL = (0.10, 0.90)  # 80% interval


# ---------------------------------------------------------------- data
def weekly_revenue(path: str = RAW_PATH) -> tuple[pd.Series, dict]:
    df = pd.read_csv(path, usecols=["InvoiceDate", "Quantity", "UnitPrice"])
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df["rev"] = df["Quantity"] * df["UnitPrice"]
    first, last = df["InvoiceDate"].min().normalize(), df["InvoiceDate"].max().normalize()

    # W-MON bins closed/labelled on the left: each bin is Monday 00:00 .. Sunday 23:59, labelled by its Monday
    wk = df.set_index("InvoiceDate")["rev"].resample("W-MON", label="left", closed="left").sum()
    complete = [(s >= first) and (s + pd.Timedelta(days=6) <= last) for s in wk.index]
    series = wk[complete]
    series.index.name = "week_start"
    partial = wk[[not c for c in complete]]
    info = {"data_first_day": str(first.date()), "data_last_day": str(last.date()),
            "dropped_partial_weeks": {str(k.date()): round(float(v), 2) for k, v in partial.items()}}
    return series, info


# ---------------------------------------------------------------- models: each takes history (1-D array) and returns h forecasts
def f_naive(y, h):
    return np.repeat(y[-1], h)


def f_ma4(y, h):
    return np.repeat(np.mean(y[-4:]), h)


def f_ses(y, h):
    """Simple exponential smoothing; alpha chosen by one-step SSE on the history only."""
    best_a, best_sse = 0.3, np.inf
    for a in np.linspace(0.05, 0.95, 19):
        lvl, sse = y[0], 0.0
        for v in y[1:]:
            sse += (v - lvl) ** 2
            lvl = a * v + (1 - a) * lvl
        if sse < best_sse:
            best_a, best_sse = a, sse
    lvl = y[0]
    for v in y[1:]:
        lvl = best_a * v + (1 - best_a) * lvl
    return np.repeat(lvl, h)


def f_trend(y, h):
    """Linear trend on the last 26 weeks (avoids one old regime dominating)."""
    w = y[-26:]
    x = np.arange(len(w))
    b, a = np.polyfit(x, w, 1)
    return np.maximum(a + b * np.arange(len(w), len(w) + h), 0)


def _feat(hist):
    return [hist[-1], hist[-2], hist[-3], hist[-4], np.mean(hist[-4:]), np.mean(hist[-8:])]


def f_ridge(y, h):
    """Ridge on lags + rolling means (scaled by series mean), recursive multi-step."""
    scale = max(float(np.mean(y)), 1.0)
    z = y / scale
    X = np.array([_feat(z[:t]) for t in range(8, len(z))])
    t_ = z[8:]
    m = Ridge(alpha=1.0).fit(X, t_)
    hist = list(z)
    out = []
    for _ in range(h):
        p = float(m.predict([_feat(np.array(hist))])[0])
        p = max(p, 0.0)
        out.append(p)
        hist.append(p)
    return np.array(out) * scale


MODELS = {"naive": f_naive, "moving_avg_4w": f_ma4, "exp_smoothing": f_ses,
          "linear_trend_26w": f_trend, "ridge_lags": f_ridge}


# ---------------------------------------------------------------- evaluation
def backtest(series: pd.Series) -> pd.DataFrame:
    y = series.values.astype(float)
    rows = []
    for origin in range(MIN_TRAIN, len(y) - HORIZON + 1):
        hist = y[:origin]            # ONLY data before the origin
        for name, fn in MODELS.items():
            fc = fn(hist, HORIZON)
            for k in range(HORIZON):
                rows.append({"model": name, "origin_idx": origin, "origin_week": series.index[origin - 1],
                             "horizon": k + 1, "target_week": series.index[origin + k],
                             "actual": y[origin + k], "forecast": float(fc[k])})
    return pd.DataFrame(rows)


def _metrics(a, f):
    a, f = np.asarray(a, float), np.asarray(f, float)
    return {"mape_pct": round(float(np.mean(np.abs(a - f) / a) * 100), 2),
            "rmse": round(float(np.sqrt(np.mean((a - f) ** 2))), 2),
            "mae": round(float(np.mean(np.abs(a - f))), 2),
            "bias_pct": round(float(np.mean(f - a) / np.mean(a) * 100), 2)}


def summarise(bt: pd.DataFrame) -> dict:
    out = {}
    for name, g in bt.groupby("model"):
        tot = g.groupby("origin_idx").agg(actual=("actual", "sum"), forecast=("forecast", "sum"))
        out[name] = {"weekly_all_horizons": _metrics(g["actual"], g["forecast"]),
                     "weekly_by_horizon": {int(h): _metrics(gg["actual"], gg["forecast"]) for h, gg in g.groupby("horizon")},
                     "next_4_week_total": _metrics(tot["actual"], tot["forecast"]),
                     "n_origins": int(tot.shape[0])}
    return out


def intervals(bt: pd.DataFrame, model: str) -> dict:
    g = bt[bt["model"] == model]
    wk = {}
    for h, gg in g.groupby("horizon"):
        r = (gg["actual"] / gg["forecast"].clip(lower=1)).values
        wk[int(h)] = [float(np.quantile(r, INTERVAL[0])), float(np.quantile(r, INTERVAL[1]))]
    tot = g.groupby("origin_idx").agg(a=("actual", "sum"), f=("forecast", "sum"))
    r = (tot["a"] / tot["f"].clip(lower=1)).values
    total_ratio = [float(np.quantile(r, INTERVAL[0])), float(np.quantile(r, INTERVAL[1]))]
    cover = float(np.mean((r >= total_ratio[0]) & (r <= total_ratio[1])))
    return {"weekly_ratio_by_horizon": wk, "total_ratio": total_ratio, "total_backtest_coverage": round(cover, 3)}


# ---------------------------------------------------------------- main
def run() -> dict:
    series, info = weekly_revenue()
    bt = backtest(series)
    summary = summarise(bt)
    best = min(summary, key=lambda m: summary[m]["weekly_all_horizons"]["mape_pct"])
    iv = intervals(bt, best)

    y = series.values.astype(float)
    fc = MODELS[best](y, HORIZON)
    last_week = series.index[-1]
    fc_weeks = [last_week + pd.Timedelta(weeks=k + 1) for k in range(HORIZON)]
    fc_rows = []
    for k in range(HORIZON):
        lo, hi = iv["weekly_ratio_by_horizon"][k + 1]
        fc_rows.append({"week_start": str(fc_weeks[k].date()), "horizon": k + 1, "forecast": round(float(fc[k]), 2),
                        "lower_80": round(float(fc[k] * lo), 2), "upper_80": round(float(fc[k] * hi), 2)})
    total = float(fc.sum())
    next_month = {"from": str(fc_weeks[0].date()), "to": str((fc_weeks[-1] + pd.Timedelta(days=6)).date()),
                  "forecast_total": round(total, 2),
                  "lower_80": round(total * iv["total_ratio"][0], 2), "upper_80": round(total * iv["total_ratio"][1], 2)}

    # context: what the same 4 calendar-aligned weeks produced a year earlier (only 1 prior year exists)
    ly_start = fc_weeks[0] - pd.Timedelta(weeks=52)
    ly = series[(series.index >= ly_start) & (series.index < ly_start + pd.Timedelta(weeks=HORIZON))]
    if len(ly) == HORIZON:
        next_month["same_weeks_last_year"] = {"from": str(ly.index[0].date()), "total": round(float(ly.sum()), 2),
                                             "weekly": [round(float(v), 2) for v in ly.values],
                                             "forecast_vs_last_year_ratio": round(total / float(ly.sum()), 2),
                                             "note": "Weeks near Christmas collapse in the prior year (the business appears to pause); "
                                                     "no model here can learn that from a single year, so treat late-December weeks with caution."}

    # sanity check against the days we already have for the first forecast week (partial actuals)
    partial = info["dropped_partial_weeks"].get(str(fc_weeks[0].date()))
    partial_check = None
    if partial is not None:
        partial_check = {"week_start": str(fc_weeks[0].date()), "actual_so_far": partial,
                         "exceeds_upper_80_of_full_week": bool(partial > fc_rows[0]["upper_80"]),
                         "note": "Only part of this week exists in the data. If actual_so_far already exceeds the week's "
                                 "upper bound, the model is under-forecasting the holiday ramp."}

    # weekly actual vs one-step-ahead (h=1) backtest forecast of the chosen model, for the chart
    h1 = bt[(bt["model"] == best) & (bt["horizon"] == 1)].set_index("target_week")
    history = [{"week_start": str(d.date()), "actual": round(float(v), 2),
                "backtest_forecast_1w": round(float(h1.loc[d, "forecast"]), 2) if d in h1.index else None}
               for d, v in series.items()]
    # 4-week-ahead backtest (forecast made 4 weeks before) for the chart as well
    h4 = bt[(bt["model"] == best) & (bt["horizon"] == 4)].set_index("target_week")
    for row, (d, _) in zip(history, series.items()):
        row["backtest_forecast_4w"] = round(float(h4.loc[d, "forecast"]), 2) if d in h4.index else None

    monthly = (pd.read_csv(RAW_PATH, usecols=["InvoiceDate", "Quantity", "UnitPrice"])
               .assign(InvoiceDate=lambda d: pd.to_datetime(d["InvoiceDate"]))
               .query("Quantity > 0 and UnitPrice > 0"))
    monthly["rev"] = monthly["Quantity"] * monthly["UnitPrice"]
    mm = monthly.set_index("InvoiceDate")["rev"].resample("MS").sum()
    mm = mm[mm.index + pd.offsets.MonthEnd(0) <= pd.Timestamp(info["data_last_day"])]
    monthly_actuals = [{"month": str(d.date())[:7], "actual": round(float(v), 2)} for d, v in mm.items()]

    result = {
        "dataset": "retail", "currency": "GBP",
        "target": "weekly revenue (Quantity x UnitPrice), identified customers only",
        "horizon_weeks": HORIZON, "n_weeks_history": int(len(series)),
        "history_from": str(series.index[0].date()), "history_to": str(series.index[-1].date()),
        "chosen_model": best, "selection_rule": "lowest weekly MAPE pooled over horizons 1-4 in the rolling-origin backtest",
        "backtest": {"scheme": f"expanding window, first origin after {MIN_TRAIN} weeks, forecast {HORIZON} weeks ahead",
                     "models": summary},
        "interval": {"level": "80%", "method": "empirical quantiles of backtest actual/forecast ratios", **iv},
        "next_month": next_month, "next_weeks": fc_rows, "partial_actual_check": partial_check,
        "history": history, "monthly_actuals": monthly_actuals,
        "data_info": info,
        "accuracy_note": (
            "Only one year of weekly history exists, so yearly seasonality cannot be learned and the backtest window "
            "contains the Sep-Nov holiday ramp; expect under-forecasting there. Intervals are empirical and based on "
            f"~{summary[best]['n_origins']} backtest origins. Revenue covers identified customers only. "
            "The chosen model was selected on the same backtest it is scored on, so its error is mildly optimistic."),
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    json.dump(result, open(OUT_PATH, "w"), indent=2)
    return result


if __name__ == "__main__":
    r = run()
    print(f"history: {r['n_weeks_history']} weeks {r['history_from']} .. {r['history_to']}  dropped partial: {list(r['data_info']['dropped_partial_weeks'])}")
    for m, v in r["backtest"]["models"].items():
        print(f"{m:18s} weekly MAPE {v['weekly_all_horizons']['mape_pct']:6.2f}%  RMSE {v['weekly_all_horizons']['rmse']:10.0f}  "
              f"4w-total MAPE {v['next_4_week_total']['mape_pct']:6.2f}%  bias {v['weekly_all_horizons']['bias_pct']:+.1f}%")
    print("chosen:", r["chosen_model"], "| coverage of 80% interval on 4w total (in-sample):", r["interval"]["total_backtest_coverage"])
    nm = r["next_month"]; print(f"next month {nm['from']}..{nm['to']}: {nm['forecast_total']:,.0f}  [{nm['lower_80']:,.0f} - {nm['upper_80']:,.0f}]")
    print("partial check:", r["partial_actual_check"])
