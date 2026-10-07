// Power Query (M) -- paste into Home > Get Data > Blank Query > Advanced Editor, once per table.
// 1) Create a parameter first: Home > Transform data > Manage Parameters > New
//      Name: DataFolder   Type: Text   Current value: C:\path\to\churnguard\powerbi\data\
// 2) Create 4 queries named exactly: fact_customers, dim_dataset, fact_shap, fact_model_metrics
//    Each uses the same pattern below; change only FileName.

let
    FileName = "fact_customers.csv",            // <- change per query
    Source   = Csv.Document(File.Contents(DataFolder & FileName),
                 [Delimiter = ",", Encoding = 65001, QuoteStyle = QuoteStyle.Csv]),
    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars = true]),
    Typed    = Table.TransformColumnTypes(Promoted, {
                 {"churn_flag", Int64.Type}, {"CLV", type number},
                 {"churn_probability", type number}, {"risk_rank", Int64.Type},
                 {"revenue_at_risk", type number}})
    // For fact_shap use:  {"driver_rank", Int64.Type}, {"contribution", type number}
    // For dim_dataset use: {"test_roc_auc", type number}, {"test_pr_auc", type number},
    //                      {"n_customers", Int64.Type}, {"actual_churn_rate", type number}
    // For fact_model_metrics use number types for every val_*/test_* column.
in
    Typed
