# ChurnGuard in Power BI -- build guide (about 20 minutes)

Power BI Desktop is Windows-only. On Mac/Linux, use the web app or the React dashboard.

## 1. Generate the data
From `backend/`:  `python scripts/export_for_powerbi.py`
(The CSVs in `powerbi/data/` are already generated from the trained models; re-run after retraining.)

## 2. Load it
Follow `load_data.pq`: one `DataFolder` parameter, then 4 queries
(`fact_customers`, `dim_dataset`, `fact_shap`, `fact_model_metrics`, plus `fact_sales_weekly` and `fact_sales_backtest` for the sales forecast).
Easier: Get Data > Text/CSV for each file, then Transform data > set the column types.

## 3. Model
Follow the relationship notes at the top of `measures.dax`, then paste the measures
(Modeling > New measure). Create the What-If parameter "Risk Threshold" first, and rename
the generated measure to `Risk Threshold Value`.

## 4. Pages to build
| Page | Visuals |
|---|---|
| **Executive Overview** | Slicer: dataset_name. Cards: Customers, Actual Churn Rate, High Risk %, Revenue at Risk (single dataset). Donut: Customers by risk_level. Bar: Avg Predicted Churn Probability by segment. |
| **Who to Call First** | Table: customer_id, segment, CLV, churn_probability, risk_level, sorted by churn_probability desc, filtered to High/Critical. What-If slider + Customers Above Threshold + Revenue Protectable at Threshold. |
| **Why They Churn** | Bar: top 10 features by Avg Absolute SHAP (filter dataset). Matrix: feature x risk_level. |
| **Sales Forecast** | Line chart on `fact_sales_weekly[week_start]`: Weekly Actual Sales, Backtest Forecast 1w, Forecast Sales, with Forecast Lower 80 / Upper 80 as an error band (Format > Error bars, or draw as two light lines). Cards: Next 4 Weeks Forecast / Low / High, Backtest MAPE 1w, Backtest RMSE 1w. Table: `fact_sales_backtest` (all five models). Add a text box with the accuracy caveat. |
| **Model Report Card** | Table from fact_model_metrics (model, test ROC-AUC, PR-AUC, F1, Brier, ECE) with is_production highlighted. Cards: Production Model, ROC-AUC, Label Caveat. Calibration Gap by dataset. |

## 5. What to say honestly in the demo
- Every number comes from the trained pipeline; Power BI only visualises it.
- Iranian's ~0.99 AUC and Retail's derived label are not comparable to Telco/Bank (use the Label Caveat card).
- Revenue at Risk mixes currencies across datasets, so it is only shown for one dataset at a time.
- The sales forecast uses only one year of weekly Retail data, so it cannot learn yearly seasonality and under-forecasts the holiday ramp; read the Sales Forecast page's warning.
- Scoring, SHAP, survival and the retention optimizer live in the Python app, not in Power BI.
