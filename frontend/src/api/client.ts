const BASE = "/api";

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`GET ${path} failed: ${res.status} ${body}`);
  }
  return res.json();
}

async function post<T>(path: string, data: any): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`POST ${path} failed: ${res.status} ${body}`);
  }
  return res.json();
}

const qs = (params: Record<string, string | number | undefined>) => {
  const q = new URLSearchParams();
  Object.entries(params).forEach(([k, v]) => { if (v !== undefined && v !== "") q.set(k, String(v)); });
  return q.toString();
};
const enc = encodeURIComponent;

// Dataset-scoped endpoints: every page uses these, driven by the header dropdown.
const ds = {
  schema: (k: string) => get<any>(`/datasets/${k}/schema`),
  summary: (k: string) => get<any>(`/datasets/${k}/summary`),
  customers: (k: string, p: Record<string, string | number | undefined> = {}) => get<any>(`/datasets/${k}/customers?${qs(p)}`),
  customer: (k: string, id: string) => get<any>(`/datasets/${k}/customers/${enc(id)}`),
  explanation: (k: string, id: string) => get<any>(`/datasets/${k}/customers/${enc(id)}/explanation`),
  recommendations: (k: string, id: string) => get<any>(`/datasets/${k}/customers/${enc(id)}/recommendations`),
  survival: (k: string, id: string) => get<any>(`/datasets/${k}/customers/${enc(id)}/survival`),
  counterfactual: (k: string, id: string) => get<any>(`/datasets/${k}/customers/${enc(id)}/counterfactual`),
  salesForecast: (k: string) => get<any>(`/datasets/${k}/sales-forecast`),
  survivalPopulation: (k: string) => get<any>(`/datasets/${k}/survival/population`),
  models: (k: string) => get<any>(`/datasets/${k}/models`),
  calibration: (k: string) => get<any>(`/datasets/${k}/calibration`),
  drift: (k: string) => get<any>(`/datasets/${k}/drift`),
  leakage: (k: string) => get<any>(`/datasets/${k}/leakage-audit`),
  dataQuality: (k: string) => get<any>(`/datasets/${k}/data-quality`),
  experiments: (k: string) => get<any>(`/datasets/${k}/experiments`),
};

export const api = {
  ds,
  health: () => get<any>("/health"),
  dashboardSummary: () => get<any>("/dashboard/summary"),
  listCustomers: (params: Record<string, string | number | undefined>) => {
    const qs = new URLSearchParams();
    Object.entries(params).forEach(([k, v]) => {
      if (v !== undefined && v !== "") qs.set(k, String(v));
    });
    return get<any>(`/customers?${qs.toString()}`);
  },
  getCustomer: (id: string) => get<any>(`/customers/${id}`),
  getCustomerRisk: (id: string) => get<any>(`/customers/${id}/risk`),
  getCustomerExplanation: (id: string) => get<any>(`/customers/${id}/explanation`),
  getCustomerSurvival: (id: string) => get<any>(`/customers/${id}/survival`),
  getCustomerCounterfactual: (id: string) => get<any>(`/customers/${id}/counterfactual`),
  getCustomerRecommendations: (id: string) => get<any>(`/customers/${id}/recommendations`),
  getSurvivalPopulation: () => get<any>("/survival/population"),
  listModels: () => get<any>("/models"),
  getModelMetrics: (name: string) => get<any>(`/models/${name}/metrics`),
  getProductionCalibration: () => get<any>("/models/production/calibration"),
  getDataQuality: () => get<any>("/data-quality"),
  getLeakageAudit: () => get<any>("/leakage-audit"),
  getDrift: () => get<any>("/drift"),
  getDatasets: () => get<any>("/datasets"),
  getDatasetMetrics: (key: string) => get<any>(`/datasets/${key}/metrics`),
  getDatasetCustomers: (key: string, params: Record<string, string | number | undefined> = {}) => {
    const qs = new URLSearchParams();
    Object.entries(params).forEach(([k, v]) => {
      if (v !== undefined && v !== "") qs.set(k, String(v));
    });
    return get<any>(`/datasets/${key}/customers?${qs.toString()}`);
  },
  getDatasetLeakageAudit: (key: string) => get<any>(`/datasets/${key}/leakage-audit`),
  getDatasetDrift: (key: string) => get<any>(`/datasets/${key}/drift`),
  getExperiments: () => get<any>("/experiments"),
  predict: (payload: any) => post<any>("/predict", payload),
  optimizeRetention: (payload: any) => post<any>("/retention/optimize", payload),
};
