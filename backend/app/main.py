import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

from app.core.config import CORS_ORIGINS
from app.services import artifacts, customers as customer_service, explain
from app.services import counterfactual
from ml.explainability.retention_optimizer import evaluate_actions

app = FastAPI(
    title="ChurnGuard AI API",
    description="Customer intelligence & churn prevention platform -- all figures are computed "
                "from real training runs on the IBM Telco Customer Churn dataset (7,043 customers). "
                "No metric in this API is hardcoded.",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "ChurnGuard AI", "mode": "live", "dummy_data": False}


@app.get("/api/dashboard/summary")
def dashboard_summary():
    return customer_service.dashboard_summary()


@app.get("/api/customers")
def list_customers(
    risk_level: Optional[str] = Query(None, description="Comma-separated: Low,Medium,High,Critical"),
    contract: Optional[str] = None,
    min_tenure: Optional[int] = None,
    max_tenure: Optional[int] = None,
    min_monthly: Optional[float] = None,
    max_monthly: Optional[float] = None,
    internet_service: Optional[str] = None,
    search: Optional[str] = None,
    sort_by: str = "churn_probability",
    sort_dir: str = "desc",
    page: int = 1,
    page_size: int = 25,
):
    risk_levels = risk_level.split(",") if risk_level else None
    return customer_service.list_customers(
        risk_level=risk_levels, contract=contract, min_tenure=min_tenure, max_tenure=max_tenure,
        min_monthly=min_monthly, max_monthly=max_monthly, internet_service=internet_service,
        search=search, sort_by=sort_by, sort_dir=sort_dir, page=page, page_size=page_size,
    )


@app.get("/api/customers/{customer_id}")
def get_customer(customer_id: str):
    c = customer_service.get_customer(customer_id)
    if c is None:
        raise HTTPException(404, f"Customer {customer_id} not found")
    return c


@app.get("/api/customers/{customer_id}/risk")
def get_customer_risk(customer_id: str):
    c = customer_service.get_customer(customer_id)
    if c is None:
        raise HTTPException(404, f"Customer {customer_id} not found")
    return {
        "customer_id": customer_id,
        "churn_probability": c["churn_probability"],
        "risk_level": c["risk_level"],
        "churn_risk_6m": c["churn_risk_6m"],
        "clv": c["CLV"],
        "production_model": c["production_model"],
    }


@app.get("/api/customers/{customer_id}/explanation")
def get_customer_explanation(customer_id: str):
    exp = explain.get_explanation(customer_id)
    if exp is None:
        raise HTTPException(404, f"No explanation available for {customer_id}")
    return exp


@app.get("/api/customers/{customer_id}/survival")
def get_customer_survival(customer_id: str):
    surv = explain.get_survival(customer_id)
    if surv is None:
        raise HTTPException(404, f"No survival curve available for {customer_id}")
    return surv


@app.get("/api/customers/{customer_id}/counterfactual")
def get_customer_counterfactual(customer_id: str):
    result = counterfactual.generate_scenarios(customer_id)
    if result is None:
        raise HTTPException(404, f"Customer {customer_id} not found")
    return result


@app.get("/api/customers/{customer_id}/recommendations")
def get_customer_recommendations(customer_id: str):
    c = customer_service.get_customer(customer_id)
    if c is None:
        raise HTTPException(404, f"Customer {customer_id} not found")
    actions = evaluate_actions(c["churn_probability"], c["CLV"])
    return {
        "customer_id": customer_id,
        "mode": "simulation",
        "note": "Expected values use assumed retention-uplift benchmarks, not a causal model "
                "trained on this dataset (no treatment/control data is available). See "
                "ml/explainability/retention_optimizer.py.",
        "actions": actions,
    }


@app.get("/api/survival/population")
def survival_population():
    import json
    from app.core.config import ARTIFACT_DIR
    path = ARTIFACT_DIR / "survival_population.json"
    if not path.exists():
        raise HTTPException(404, "Survival model artifact not found -- run `python -m ml.models.survival`")
    return json.load(open(path))


@app.get("/api/models")
def list_models():
    m = artifacts.get_metrics()
    if m is None:
        raise HTTPException(404, "No trained models found -- run `python -m ml.models.train`")
    summary = []
    for name, data in m["models"].items():
        summary.append({
            "model": name,
            "is_production": name == m["production_model"],
            "validation": {
                "roc_auc": data["validation"]["roc_auc"],
                "pr_auc": data["validation"]["pr_auc"],
                "f1": data["validation"]["f1"],
                "brier_score": data["validation"]["brier_score"],
                "ece": data["validation"]["ece"],
            },
            "test": {
                "roc_auc": data["test"]["roc_auc"],
                "pr_auc": data["test"]["pr_auc"],
                "f1": data["test"]["f1"],
                "brier_score": data["test"]["brier_score"],
                "ece": data["test"]["ece"],
            },
            "train_time_seconds": data["train_time_seconds"],
            "inference_time_ms_per_sample": data["inference_time_ms_per_sample"],
        })
    return {
        "production_model": m["production_model"],
        "selection_reason": m["selection_reason"],
        "models": summary,
    }


@app.get("/api/models/{model_name}/metrics")
def get_model_metrics(model_name: str):
    m = artifacts.get_metrics()
    if m is None or model_name not in m["models"]:
        raise HTTPException(404, f"Model {model_name} not found")
    return m["models"][model_name]


@app.get("/api/models/production/calibration")
def get_production_calibration():
    cal = artifacts.get_calibrated_test_metrics()
    if cal is None:
        raise HTTPException(404, "Calibration metrics not found")
    return cal


@app.get("/api/data-quality")
def data_quality():
    dq = artifacts.get_data_quality()
    if dq is None:
        raise HTTPException(404, "Data quality report not found")
    return dq


@app.get("/api/leakage-audit")
def leakage_audit():
    audit = artifacts.get_leakage_audit()
    if audit is None:
        raise HTTPException(404, "Leakage audit not found")
    return audit


@app.get("/api/drift")
def drift():
    d = artifacts.get_drift_report()
    if d is None:
        raise HTTPException(404, "Drift report not found -- run training pipeline")
    return d


@app.get("/api/datasets")
def datasets():
    import json
    from app.core.config import ARTIFACT_DIR

    def _load_summary(key):
        path = ARTIFACT_DIR / key / "metrics.json"
        if not path.exists():
            return None
        m = json.load(open(path))
        prod = m["production_model"]
        return {
            "production_model": prod,
            "test_roc_auc": m["models"][prod]["test"]["roc_auc"],
            "test_pr_auc": m["models"][prod]["test"]["pr_auc"],
            "n_total": m["n_total"],
            "positive_rate": m["positive_rate"],
        }

    telco_summary = artifacts.get_metrics()
    from app.core.db import get_connection as _gc
    _c = _gc()
    _n, _pos = _c.execute("SELECT COUNT(*), AVG(churn_flag) FROM customers").fetchone()
    _c.close()
    telco = {
        "key": "telco",
        "name": "IBM Telco Customer Churn",
        "domain": "Telecom",
        "rows": _n,
        "features": 20,
        "temporal": False,
        "labels": True,
        "status": "active",
        "source": "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
        "notes": "Cross-sectional snapshot, no event timestamps. Used for classification, "
                 "survival (tenure as duration), and explainability in this build.",
    }
    if telco_summary:
        prod = telco_summary["production_model"]
        telco["result"] = {
            "production_model": prod,
            "test_roc_auc": telco_summary["models"][prod]["test"]["roc_auc"],
            "test_pr_auc": telco_summary["models"][prod]["test"]["pr_auc"],
            "n_total": _n,
            "positive_rate": round(_pos, 4),
        }

    bank = {
        "key": "bank", "name": "Bank Customer Churn", "domain": "Banking",
        "rows": 10000, "features": 11, "temporal": False, "labels": True, "status": "active",
        "source": "https://github.com/Aslm-Fawzy/Churn_Classification_for_Bank_Customers "
                   "(originally Kaggle: shrutimechlearn/churn-modelling)",
        "notes": "Cross-sectional snapshot. Real target: Exited.",
        "result": _load_summary("bank"),
    }
    iranian = {
        "key": "iranian", "name": "Iranian Telecom Churn", "domain": "Telecom",
        "rows": 3150, "features": 12, "temporal": False, "labels": True, "status": "active",
        "source": "UCI ML Repository dataset 563, loaded via the `survivalpredict` PyPI package's "
                   "bundled copy (Jafari-Marandi et al. 2020).",
        "notes": "Features aggregate the first 9 months; label reflects month 12. Very high AUC (~0.99) is NOT "
                 "a single-column leak (see ablation in Data Quality & Leakage), but usage near the label "
                 "window may reflect already-disengaged customers, so it is not comparable to Telco.",
        "result": _load_summary("iranian"),
    }
    retail = {
        "key": "retail", "name": "Online Retail (RFM-derived churn)", "domain": "Retail",
        "rows": 3363, "features": 14, "temporal": True, "labels": False, "status": "active",
        "source": "UCI ML Repository (Daqing Chen et al. 2012), mirrored at "
                   "github.com/eaintkyawthmu/UCI_Online_Retail_Dataset_Cleaned_Version",
        "notes": "IMPORTANT: this dataset has no provided churn label. We derived one with a "
                 "leakage-safe time split -- RFM features from transactions up to 2011-09-09, "
                 "churn label = no purchase in the following 3 months. This is a heuristic "
                 "label, not a ground-truth business outcome; treat accuracy claims accordingly.",
        "result": _load_summary("retail"),
    }
    return {"datasets": [telco, bank, iranian, retail]}


# ---- Dataset-scoped API (v3): one code path for every dataset in the registry ----
from app.services import dataset_service as ds


def _require(key: str, capability: str):
    if not ds.cfg(key)["capabilities"].get(capability):
        raise HTTPException(404, f"'{capability}' is not available for '{key}': the dataset lacks the data it needs "
                                 f"(see /api/datasets/{key}/schema -> capabilities).")


@app.get("/api/datasets/{key}/schema")
def ds_schema(key: str):
    return ds.schema(key)


@app.get("/api/datasets/{key}/summary")
def ds_summary(key: str):
    return ds.summary(key)


@app.get("/api/datasets/{key}/customers")
def ds_customers(key: str, risk_level: Optional[str] = None, segment: Optional[str] = None,
                 search: Optional[str] = None, sort_by: str = "churn_probability", sort_dir: str = "desc",
                 page: int = 1, page_size: int = 20):
    return ds.list_customers(key, risk_level, segment, search, sort_by, sort_dir, page, page_size)


@app.get("/api/datasets/{key}/customers/{cid}")
def ds_customer(key: str, cid: str):
    return ds.get_customer(key, cid)


@app.get("/api/datasets/{key}/customers/{cid}/explanation")
def ds_explanation(key: str, cid: str):
    return ds.explanation(key, cid)


@app.get("/api/datasets/{key}/customers/{cid}/recommendations")
def ds_recommendations(key: str, cid: str):
    return ds.recommendations(key, cid)


@app.get("/api/datasets/{key}/customers/{cid}/survival")
def ds_customer_survival(key: str, cid: str):
    _require(key, "survival")
    surv = explain.get_survival(cid)
    if surv is None:
        raise HTTPException(404, f"No survival curve for {cid}")
    return surv


@app.get("/api/datasets/{key}/customers/{cid}/counterfactual")
def ds_counterfactual(key: str, cid: str):
    _require(key, "counterfactual")
    res = counterfactual.generate_scenarios(cid)
    if res is None:
        raise HTTPException(404, f"Customer {cid} not found")
    return res


@app.get("/api/datasets/{key}/survival/population")
def ds_survival_population(key: str):
    _require(key, "survival")
    return survival_population()


@app.get("/api/datasets/{key}/sales-forecast")
def ds_sales_forecast(key: str):
    _require(key, "sales_forecast")
    import json
    path = ds.cfg(key)["artifact_dir"] / "sales_forecast.json"
    if not path.exists():
        raise HTTPException(404, "Sales forecast artifact not found -- run `python -m ml.forecasting.sales` from backend/")
    return json.load(open(path))


@app.get("/api/datasets/{key}/models")
def ds_models(key: str):
    return ds.models(key)


@app.get("/api/datasets/{key}/calibration")
def ds_calibration(key: str):
    return ds.calibration(key)


@app.get("/api/datasets/{key}/drift")
def ds_drift(key: str):
    return ds.drift(key)


@app.get("/api/datasets/{key}/leakage-audit")
def ds_leakage(key: str):
    return ds.leakage(key)


@app.get("/api/datasets/{key}/data-quality")
def ds_data_quality(key: str):
    return ds.data_quality(key)


@app.get("/api/datasets/{key}/experiments")
def ds_experiments(key: str):
    return ds.experiments(key)


@app.get("/api/experiments")
def experiments():
    m = artifacts.get_metrics()
    if m is None:
        raise HTTPException(404, "No experiments found")
    exps = []
    for i, (name, data) in enumerate(m["models"].items(), start=1):
        exps.append({
            "experiment_id": f"EXP-{i:03d}",
            "dataset": "IBM Telco Customer Churn",
            "model": name,
            "features": "20 leakage-cleared columns, one-hot encoded",
            "split": "Stratified random 60/20/20 (train/val/test) -- see split.py",
            "pr_auc": data["validation"]["pr_auc"],
            "roc_auc": data["validation"]["roc_auc"],
            "brier": data["validation"]["brier_score"],
            "status": "completed",
            "is_production": name == m["production_model"],
        })
    return {"experiments": exps}


@app.post("/api/predict")
def predict(payload: dict):
    """Score an arbitrary customer feature payload with the production model."""
    from app.core.config import MODEL_DIR
    import joblib
    from app.services.counterfactual import _row_to_feature_vector
    import json
    from app.core.config import ARTIFACT_DIR

    required = ["gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
                "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
                "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
                "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges"]
    missing = [f for f in required if f not in payload]
    if missing:
        raise HTTPException(400, f"Missing required fields: {missing}")

    model = joblib.load(MODEL_DIR / "production_calibrated.joblib")
    feature_columns = json.load(open(ARTIFACT_DIR / "feature_columns.json"))
    vec = _row_to_feature_vector(payload, feature_columns)
    prob = float(model.predict_proba(vec)[:, 1][0])
    risk_level = "Low" if prob < 0.25 else "Medium" if prob < 0.5 else "High" if prob < 0.75 else "Critical"
    return {"churn_probability": round(prob, 4), "risk_level": risk_level, "model": "production_calibrated"}


@app.post("/api/retention/optimize")
def retention_optimize(payload: dict):
    if "churn_probability" not in payload or "clv" not in payload:
        raise HTTPException(400, "Payload must include churn_probability and clv")
    actions = evaluate_actions(float(payload["churn_probability"]), float(payload["clv"]))
    return {"mode": "simulation", "actions": actions}
