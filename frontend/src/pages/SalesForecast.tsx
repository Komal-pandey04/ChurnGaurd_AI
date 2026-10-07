import React, { useEffect, useState } from "react";
import { ComposedChart, Line, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from "recharts";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import { Card, CardHeader, KpiCard, Loading, ErrorState } from "../components/ui";
import { useDataset } from "../context/DatasetContext";

const fmt = (n: number) => `£${Math.round(n).toLocaleString()}`;
const compact = (n: number) => (n >= 1e6 ? `£${(n / 1e6).toFixed(2)}M` : `£${Math.round(n / 1e3)}k`);

export default function SalesForecast() {
  const { dataset, schema } = useDataset();
  const [data, setData] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const supported = schema?.capabilities.sales_forecast;

  useEffect(() => {
    setData(null); setError(null);
    if (supported) api.ds.salesForecast(dataset).then(setData).catch((e) => setError(e.message));
  }, [dataset, supported]);

  if (!schema) return <Loading />;
  if (!supported) return (
    <div className="space-y-4">
      <h1 className="text-xl font-semibold text-slate-900">Sales Forecast</h1>
      <Card className="p-6 max-w-2xl">
        <div className="font-semibold text-slate-800">Not available for {schema.name}</div>
        <p className="text-sm text-slate-500 mt-2">A sales forecast needs dated transactions. Only the Online Retail dataset has them; this dataset is a customer snapshot, so we do not fabricate a sales history.</p>
        <Link to="/" className="inline-block mt-4 text-xs font-medium text-brand-600 hover:underline">← Back to Overview</Link>
      </Card>
    </div>);
  if (error) return <ErrorState message={error} />;
  if (!data) return <Loading />;

  const nm = data.next_month, chosen = data.backtest.models[data.chosen_model];
  const ly = nm.same_weeks_last_year;
  const chart = [
    ...data.history.map((h: any) => ({ week: h.week_start, actual: h.actual, back1: h.backtest_forecast_1w, back4: h.backtest_forecast_4w })),
    ...data.next_weeks.map((w: any) => ({ week: w.week_start, forecast: w.forecast, band: [w.lower_80, w.upper_80] })),
  ];
  const models = Object.entries(data.backtest.models).map(([name, m]: any) => ({
    name, mape: m.weekly_all_horizons.mape_pct, rmse: m.weekly_all_horizons.rmse, total: m.next_4_week_total.mape_pct, bias: m.weekly_all_horizons.bias_pct,
  }));

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold text-slate-900">Sales Forecast</h1>
        <p className="text-sm text-slate-400 mt-0.5">Next-month revenue for {schema.name}. {data.target}. Model: <b>{data.chosen_model.replace(/_/g, " ")}</b> ({data.selection_rule}).</p>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <KpiCard label={`Next 4 weeks (${nm.from} → ${nm.to})`} value={compact(nm.forecast_total)} sub={`80% range ${compact(nm.lower_80)} – ${compact(nm.upper_80)}`} />
        <KpiCard label="Backtest MAPE (weekly)" value={`${chosen.weekly_all_horizons.mape_pct}%`} sub={`RMSE ${fmt(chosen.weekly_all_horizons.rmse)}`} tone="warn" />
        <KpiCard label="Backtest MAPE (4-week total)" value={`${chosen.next_4_week_total.mape_pct}%`} sub={`${chosen.n_origins} rolling origins`} tone="warn" />
        <KpiCard label="Average bias" value={`${chosen.weekly_all_horizons.bias_pct}%`} sub="negative = under-forecasts" tone="bad" />
      </div>

      <Card className="p-4 border-amber-200 bg-amber-50">
        <div className="text-sm font-semibold text-amber-900">Read this before trusting the number</div>
        <ul className="text-sm text-amber-900 mt-2 list-disc pl-5 space-y-1">
          <li>{data.accuracy_note}</li>
          {ly && <li>The same 4 weeks last year totalled {fmt(ly.total)} (weekly: {ly.weekly.map((v: number) => compact(v)).join(", ")}). This forecast is {ly.forecast_vs_last_year_ratio}× that. {ly.note}</li>}
          {data.partial_actual_check?.exceeds_upper_80_of_full_week && <li>Check against data we already have: only part of the first forecast week exists, and it is already {fmt(data.partial_actual_check.actual_so_far)}, above that week's upper bound. The model under-forecasts the early-December ramp while probably over-forecasting late December.</li>}
        </ul>
      </Card>

      <Card className="p-4">
        <CardHeader title="Predicted vs actual weekly sales" subtitle="Actual revenue, the model's 1-week-ahead and 4-week-ahead backtest predictions, then the 4-week forecast with an 80% interval" />
        <ResponsiveContainer width="100%" height={340}>
          <ComposedChart data={chart}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis dataKey="week" tick={{ fontSize: 10 }} interval={5} />
            <YAxis tick={{ fontSize: 10 }} tickFormatter={(v) => compact(v)} />
            <Tooltip formatter={(v: any) => (Array.isArray(v) ? `${fmt(v[0])} – ${fmt(v[1])}` : fmt(v))} />
            <Legend />
            <Area dataKey="band" name="80% interval" stroke="none" fill="#6366f1" fillOpacity={0.18} />
            <Line dataKey="actual" name="Actual" stroke="#0f172a" strokeWidth={2} dot={false} />
            <Line dataKey="back1" name="Backtest (1 wk ahead)" stroke="#10b981" strokeWidth={1.5} dot={false} />
            <Line dataKey="back4" name="Backtest (4 wk ahead)" stroke="#f59e0b" strokeWidth={1.5} dot={false} strokeDasharray="4 3" />
            <Line dataKey="forecast" name="Forecast" stroke="#6366f1" strokeWidth={2.5} dot={{ r: 3 }} />
          </ComposedChart>
        </ResponsiveContainer>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <Card className="p-4">
          <CardHeader title="Next 4 weeks" subtitle="Week by week, with 80% interval" />
          <table className="w-full text-sm"><thead><tr className="text-left text-xs text-slate-400"><th className="py-1">Week of</th><th>Forecast</th><th>Low</th><th>High</th></tr></thead>
            <tbody>{data.next_weeks.map((w: any) => (<tr key={w.week_start} className="border-t border-slate-100"><td className="py-1.5">{w.week_start}</td><td>{fmt(w.forecast)}</td><td>{fmt(w.lower_80)}</td><td>{fmt(w.upper_80)}</td></tr>))}</tbody></table>
        </Card>
        <Card className="p-4">
          <CardHeader title="Backtest: every model, nothing hidden" subtitle="Rolling-origin holdout. Lower is better; all models under-forecast the holiday ramp" />
          <table className="w-full text-sm"><thead><tr className="text-left text-xs text-slate-400"><th className="py-1">Model</th><th>MAPE</th><th>RMSE</th><th>4-wk total MAPE</th><th>Bias</th></tr></thead>
            <tbody>{models.map((m) => (<tr key={m.name} className={`border-t border-slate-100 ${m.name === data.chosen_model ? "font-semibold bg-indigo-50" : ""}`}><td className="py-1.5">{m.name.replace(/_/g, " ")}</td><td>{m.mape}%</td><td>{fmt(m.rmse)}</td><td>{m.total}%</td><td>{m.bias}%</td></tr>))}</tbody></table>
        </Card>
      </div>

      <Card className="p-4">
        <CardHeader title="Monthly revenue (complete months)" subtitle="Context only; the forecast itself is built on weekly data" />
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={data.monthly_actuals}><CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" /><XAxis dataKey="month" tick={{ fontSize: 10 }} /><YAxis tick={{ fontSize: 10 }} tickFormatter={(v) => compact(v)} /><Tooltip formatter={(v: any) => fmt(v)} /><Bar dataKey="actual" fill="#64748b" radius={[3, 3, 0, 0]} /></BarChart>
        </ResponsiveContainer>
      </Card>
    </div>
  );
}
