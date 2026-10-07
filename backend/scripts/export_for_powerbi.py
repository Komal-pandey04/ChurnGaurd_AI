"""
Export ChurnGuard's real predictions and model results as a Power BI-ready star schema.

Run from backend/:   python scripts/export_for_powerbi.py
Writes CSVs to ../powerbi/data/ :

  dim_dataset.csv        one row per dataset (name, domain, currency, label caveat, headline metrics)
  fact_customers.csv     one row per customer across ALL datasets (common columns + revenue_at_risk)
  fact_shap.csv          top SHAP drivers per customer (dataset_key, customer_id, rank, feature, contribution)
  fact_model_metrics.csv one row per dataset x model (validation/test ROC-AUC, PR-AUC, etc.)
  fact_sales_weekly.csv  Retail weekly actual vs backtest vs next-4-week forecast (+80% interval)
  fact_sales_backtest.csv  Retail sales-forecast backtest metrics per model (MAPE / RMSE / bias)
  detail_<dataset>.csv   full per-dataset table with every dataset-specific column

Nothing is invented: every value comes from churnguard.db and artifacts/*.json produced by the pipeline.
"""
import json
import os
import sqlite3
import sys

import pandas as pd

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND)
from app.services import registry  # noqa: E402

DB = os.path.join(BACKEND, "data", "churnguard.db")
OUT = os.path.abspath(os.path.join(BACKEND, "..", "powerbi", "data"))
RISK_ORDER = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}


def _json(path):
    return json.load(open(path)) if os.path.exists(path) else None


def main():
    os.makedirs(OUT, exist_ok=True)
    conn = sqlite3.connect(DB)
    dims, facts, shaps, metric_rows = [], [], [], []

    for key, cfg in registry.DATASETS.items():
        df = pd.read_sql(f"SELECT * FROM {cfg['table']}", conn)
        id_col = cfg["id_col"]
        art = cfg["artifact_dir"]
        metrics = _json(os.path.join(art, "metrics.json"))
        prod = metrics["production_model"] if metrics else None

        # --- detail table (all columns) ---
        df.to_csv(os.path.join(OUT, f"detail_{key}.csv"), index=False)

        # --- common fact rows ---
        seg_col = cfg["segment_col"]
        fact = pd.DataFrame({
            "dataset_key": key,
            "customer_id": df[id_col].astype(str),
            "churn_flag": df["churn_flag"],
            "CLV": df["CLV"],
            "churn_probability": df["churn_probability"],
            "risk_level": df["risk_level"],
            "risk_rank": df["risk_level"].map(RISK_ORDER),
            "segment": df[seg_col].astype(str) if seg_col in df else "n/a",
        })
        fact["revenue_at_risk"] = (fact["CLV"] * fact["churn_probability"]).round(2)
        facts.append(fact)

        # --- SHAP drivers ---
        shap = _json(os.path.join(art, "shap_values.json")) or {}
        rows = [(key, str(cid), r, d["feature"], d["contribution"])
                for cid, drivers in shap.items() for r, d in enumerate(drivers, 1)]
        shaps.append(pd.DataFrame(rows, columns=["dataset_key", "customer_id", "driver_rank", "feature", "contribution"]))

        # --- model metrics ---
        if metrics:
            for model, m in metrics["models"].items():
                t, v = m.get("test", {}), m.get("validation", {})
                metric_rows.append({
                    "dataset_key": key, "model": model, "is_production": model == prod,
                    "val_roc_auc": v.get("roc_auc"), "val_pr_auc": v.get("pr_auc"),
                    "test_roc_auc": t.get("roc_auc"), "test_pr_auc": t.get("pr_auc"),
                    "test_f1": t.get("f1"), "test_precision": t.get("precision"), "test_recall": t.get("recall"),
                    "test_brier": t.get("brier_score"), "test_ece": t.get("ece"),
                })

        prod_test = (metrics["models"][prod]["test"] if metrics else {})
        dims.append({
            "dataset_key": key, "dataset_name": cfg["name"], "domain": cfg["domain"], "currency": cfg["currency"],
            "segment_label": cfg["segment_label"], "label_note": cfg["label_note"],
            "production_model": prod, "test_roc_auc": prod_test.get("roc_auc"), "test_pr_auc": prod_test.get("pr_auc"),
            "n_customers": len(df), "actual_churn_rate": round(float(df["churn_flag"].mean()), 4),
        })
        print(f"[{key}] {len(df)} customers, {len(rows)} SHAP rows")

    # --- sales forecast (Retail only): actual vs backtest vs forecast, one row per week ---
    sf_path = os.path.join(registry.get("retail")["artifact_dir"], "sales_forecast.json")
    sf = _json(sf_path)
    if sf:
        hist = pd.DataFrame(sf["history"]).assign(row_type="history")
        fut = pd.DataFrame(sf["next_weeks"]).rename(columns={"forecast": "forecast"}).assign(row_type="forecast")
        wk = pd.concat([hist, fut], ignore_index=True).rename(columns={
            "backtest_forecast_1w": "backtest_1w_ahead", "backtest_forecast_4w": "backtest_4w_ahead"})
        wk["dataset_key"] = "retail"
        wk.to_csv(os.path.join(OUT, "fact_sales_weekly.csv"), index=False)
        rows = []
        for model, m in sf["backtest"]["models"].items():
            rows.append({"model": model, "is_chosen": model == sf["chosen_model"],
                         "weekly_mape_pct": m["weekly_all_horizons"]["mape_pct"], "weekly_rmse": m["weekly_all_horizons"]["rmse"],
                         "weekly_bias_pct": m["weekly_all_horizons"]["bias_pct"],
                         "total_4w_mape_pct": m["next_4_week_total"]["mape_pct"], "total_4w_rmse": m["next_4_week_total"]["rmse"],
                         "n_origins": m["n_origins"]})
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "fact_sales_backtest.csv"), index=False)
        print(f"[retail] sales forecast rows: {len(wk)}")

    pd.DataFrame(dims).to_csv(os.path.join(OUT, "dim_dataset.csv"), index=False)
    pd.concat(facts).to_csv(os.path.join(OUT, "fact_customers.csv"), index=False)
    pd.concat(shaps).to_csv(os.path.join(OUT, "fact_shap.csv"), index=False)
    pd.DataFrame(metric_rows).to_csv(os.path.join(OUT, "fact_model_metrics.csv"), index=False)
    print(f"Wrote Power BI files to {OUT}")


if __name__ == "__main__":
    main()
