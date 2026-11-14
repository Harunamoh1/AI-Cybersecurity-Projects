/** Typed client for the PhishLens API (via the same-origin /api proxy). */

export type Verdict = "phishing" | "legitimate";
export type RiskLevel = "low" | "medium" | "high" | "critical";

export interface Contribution {
  feature: string;
  description: string;
  value: number | string;
  reference: number | string;
  impact: number;
}

export interface Indicator {
  token: string;
  kind: string;
  reason: string;
}

export interface Components {
  scheme: string | null;
  host: string;
  subdomain: string;
  domain: string;
  suffix: string;
  registered_domain: string;
  hosting_platform: string | null;
  port: number | null;
  path: string;
  query: string;
  is_ip: boolean;
}

export interface ScanResult {
  id?: string | null;
  created_at?: string | null;
  url: string;
  canonical_url: string;
  verdict: Verdict;
  risk_score: number;
  risk_level: RiskLevel;
  probability: number;
  threshold: number;
  components: Components;
  contributions: Contribution[];
  indicators: Indicator[];
  features: Record<string, number | string>;
  model: { name: string; version: string };
  latency_ms: number;
}

export interface HistoryItem {
  id: string;
  created_at: string;
  url: string;
  registered_domain: string | null;
  verdict: Verdict;
  risk_level: RiskLevel;
  risk_score: number;
}

export interface HistoryPage {
  total: number;
  limit: number;
  offset: number;
  items: HistoryItem[];
}

export interface Stats {
  total_scans: number;
  phishing: number;
  legitimate: number;
  detection_rate: number;
  average_risk_score: number;
  by_risk_level: Record<RiskLevel, number>;
  timeline: { date: string; phishing: number; legitimate: number }[];
  top_tlds: { tld: string; scans: number; phishing: number }[];
  top_indicators: { kind: string; count: number }[];
  score_histogram: { range: string; count: number }[];
}

export interface Metrics {
  n: number;
  threshold: number;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  roc_auc: number | null;
  pr_auc: number | null;
  brier: number;
  false_positive_rate: number | null;
  false_negative_rate: number | null;
  confusion_matrix: { tn: number; fp: number; fn: number; tp: number };
}

export interface ModelCard {
  name: string;
  version: string;
  trained_at: string;
  threshold: number;
  selection: { criterion: string; selected_model: string; decision_threshold: number; threshold_criterion: string };
  datasets: { name: string; url: string; sha256: string; license: string; citation: string }[];
  cleaning: Record<string, number>;
  splits: Record<string, { rows: number; phishing: number; legitimate: number; domains: number; by_source: Record<string, number> }>;
  split_strategy: string;
  models: Record<
    string,
    {
      explainable: boolean;
      input: string;
      fit_seconds: number;
      validation: Metrics;
      test: Metrics;
      test_at_tuned_threshold?: Metrics;
    }
  >;
  per_source_test: Record<string, Metrics>;
  experiments: Record<string, { description: string; overall: Metrics; per_source?: Record<string, Metrics> }>;
  curves: { roc: { x: number; y: number }[]; pr: { x: number; y: number }[] };
  score_distribution: { bin: number; legitimate: number; phishing: number }[];
  feature_importance: { feature: string; importance: number; std: number }[];
  environment: Record<string, string>;
}

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`/api/${path}`, {
      ...init,
      headers: { "content-type": "application/json", ...(init?.headers ?? {}) },
      cache: "no-store",
    });
  } catch {
    throw new ApiError("Cannot reach the detection service", 0);
  }
  const text = await res.text();
  let body: unknown = null;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    body = null;
  }
  if (!res.ok) {
    const detail = (body as { detail?: unknown } | null)?.detail;
    throw new ApiError(typeof detail === "string" ? detail : `Request failed (${res.status})`, res.status);
  }
  return body as T;
}

export const api = {
  scan: (url: string) => request<ScanResult>("v1/scan", { method: "POST", body: JSON.stringify({ url }) }),
  scanBatch: (urls: string[]) =>
    request<{ results: ScanResult[]; errors: { url: string; error: string }[] }>("v1/scan/batch", {
      method: "POST",
      body: JSON.stringify({ urls }),
    }),
  history: (params: { limit: number; offset: number; verdict?: string; q?: string }) => {
    const qs = new URLSearchParams({ limit: String(params.limit), offset: String(params.offset) });
    if (params.verdict) qs.set("verdict", params.verdict);
    if (params.q) qs.set("q", params.q);
    return request<HistoryPage>(`v1/history?${qs}`);
  },
  historyItem: (id: string) => request<ScanResult>(`v1/history/${encodeURIComponent(id)}`),
  deleteItem: (id: string) => request<{ deleted: boolean }>(`v1/history/${encodeURIComponent(id)}`, { method: "DELETE" }),
  clearHistory: () => request<{ deleted: number }>("v1/history", { method: "DELETE" }),
  stats: (days = 14) => request<Stats>(`v1/stats?days=${days}`),
  model: () => request<ModelCard>("v1/model"),
};
