/** Typed client for the SentinelAI API (through the same-origin /api proxy). */

export const CLASSES = ["BENIGN", "DoS", "DDoS", "PortScan", "BruteForce", "WebAttack", "Bot", "Infiltration"] as const;
export type ClassName = (typeof CLASSES)[number];
export type Severity = "none" | "low" | "medium" | "high" | "critical";

export interface Reason {
  feature: string;
  value: number;
  impact: number;
}

export interface DistributionShift {
  out_of_range_rate: number;
  max_feature_rate?: number;
  level?: "low" | "moderate" | "high";
  features: { feature: string; rate: number }[];
}

export interface GroundTruth {
  labelled_flows: number;
  unmapped_labels: number;
  accuracy: number;
  attack_recall: number | null;
  attack_precision: number | null;
  false_positive_rate: number;
  per_class_recall: Partial<Record<ClassName, number>>;
}

export interface AnalysisSummary {
  class_counts: Record<ClassName, number>;
  severity_counts: Record<Exclude<Severity, "none">, number>;
  confidence_bands: Partial<Record<"high" | "medium" | "low", number>>;
  mean_confidence: number;
  threats: number;
  flows_with_imputed_values: number;
  columns: { mapped: number; identifiers_dropped: string[]; unexpected_columns: string[]; label_column: string | null };
  distribution_shift: DistributionShift;
  ground_truth?: GroundTruth;
}

export interface Analysis {
  id: string;
  created_at: string;
  filename: string;
  source: "upload" | "sample" | "api";
  total_flows: number;
  threats: number;
  rejected_rows: number;
  duration_ms: number;
  model_version: string;
  summary: AnalysisSummary;
}

export interface FlowRecord {
  analysis_id: string;
  idx: number;
  created_at: string;
  predicted: ClassName;
  confidence: number;
  attack_prob: number;
  severity: Severity;
  true_label: ClassName | null;
  imputed: number;
  reasons: Reason[];
  filename: string;
}

export interface Page<T> {
  total: number;
  limit: number;
  offset: number;
  items: T[];
}

export interface ContributionItem {
  feature: string;
  label: string;
  description: string;
  unit: string;
  value: number;
  raw_value: number | null;
  benign_median: number | null;
  impact: number;
}

export interface Explanation {
  predicted_class: ClassName;
  confidence: number;
  attack_probability: number;
  probabilities: Record<ClassName, number>;
  base_value: number;
  logit: number;
  contributions: ContributionItem[];
  attack_evidence: ContributionItem[];
  other_features_impact: number;
}

export interface FlowDetail extends FlowRecord {
  explanation: Explanation;
  features: Record<string, number>;
}

export interface Stats {
  analyses: number;
  total_flows: number;
  threats: number;
  threat_rate: number;
  by_class: Partial<Record<ClassName, { count: number; mean_confidence: number }>>;
  by_severity: Partial<Record<Severity, number>>;
  timeline: { date: string; total: number; threats: number; by_class: Partial<Record<ClassName, number>> }[];
  threat_confidence_histogram: { bin: string; count: number }[];
  labelled_flows: number;
  labelled_agreement: number | null;
}

export interface Sample {
  name: string;
  title?: string;
  description?: string;
  size_bytes: number;
}

export interface PerClass {
  precision: number;
  recall: number;
  f1: number;
  support: number;
  recall_ci95: [number, number];
  roc_auc?: number;
}

export interface BinaryMetrics {
  precision: number;
  recall: number;
  f1: number;
  false_positive_rate: number;
  tp: number;
  fp: number;
  fn: number;
  tn: number;
  roc_auc?: number;
}

export interface SupervisedResult {
  display_name: string;
  train_seconds: number;
  inference_us_per_flow: number;
  model_size_bytes: number;
  validation: { macro_f1: number };
  test: {
    accuracy: number;
    macro_precision: number;
    macro_recall: number;
    macro_f1: number;
    weighted_f1: number;
    roc_auc_macro_ovr: number;
    per_class: Record<ClassName, PerClass>;
    confusion_matrix: number[][];
  };
  binary_test: BinaryMetrics;
  binary_roc: [number, number][];
  class_roc?: Partial<Record<ClassName, [number, number][]>>;
  best_iteration?: number;
  n_trees?: number;
}

export interface AnomalyResult {
  display_name: string;
  kind: "anomaly_detection";
  train_seconds: number;
  trained_on: string;
  inference_us_per_flow: number;
  model_size_bytes: number;
  threshold: { rule: string; value: number };
  binary_test: BinaryMetrics;
  binary_test_f1_threshold: BinaryMetrics & { threshold: number };
  binary_roc: [number, number][];
  per_class_detection: Partial<Record<ClassName, { detection_rate: number; support: number; ci95: [number, number] }>>;
}

export interface CalibrationReport {
  ece: number;
  bins: { bin: [number, number]; confidence: number; accuracy: number; count: number }[];
}

export interface ExperimentHead {
  macro_f1: number;
  attack_recall: number;
  false_positive_rate: number;
  per_class_recall: Partial<Record<ClassName, number>>;
}

export interface ModelInfo {
  name: string;
  version: string;
  trained_at: string;
  temperature: number;
  classes: { name: ClassName; label: string; severity: Severity; description: string }[];
  feature_count: number;
  deployed_model: string;
  selection: { rule: string; validation_ranking: { model: string; macro_f1: number }[] };
  dataset: {
    name: string;
    citation: string;
    archive: { url: string; sha256: string };
    cleaning: {
      raw_rows: number;
      clean_rows: number;
      duplicate_rows_removed: number;
      conflicting_rows_removed: number;
      conflicting_vectors: number;
      cells_nan: number;
      cells_inf: number;
      excluded_labels: Record<string, number>;
      class_counts: Record<ClassName, number>;
    };
    split_protocol: string;
    splits: Record<"train" | "val" | "test", Record<ClassName, number>>;
    class_weights: Record<ClassName, number>;
  };
  features: {
    selected: { name: string; label: string; label_2018: string; description: string; unit: string }[];
    dropped: Record<string, string>;
  };
  models: Record<string, SupervisedResult | AnomalyResult>;
  calibration: {
    method: string;
    temperature: number;
    test_nll_before: number;
    test_nll_after: number;
    before: CalibrationReport;
    after: CalibrationReport;
  };
  unseen_attack_probe: {
    note: string;
    flows?: number;
    xgboost_predictions?: Record<string, number>;
    xgboost_flagged_as_attack?: number;
    isolation_forest_flagged?: number;
  };
  experiments: {
    temporal_split?: {
      description: string;
      results: Record<string, { display_name: string; macro_f1?: number; accuracy?: number; binary: BinaryMetrics;
        per_class_recall?: Partial<Record<ClassName, number>> }>;
    };
    cross_dataset?: {
      description: string;
      dataset: { name: string; citation: string; sampling: string; splits: Record<string, Record<string, number>> };
      summary: Record<"test_2017" | "test_2018", Record<string, ExperimentHead>>;
      ablation_excluded_features: string[];
      A_2017_model_on_all_2018: { distribution_shift: DistributionShift };
    };
  };
  environment: Record<string, string | number>;
}

export interface GlobalExplainability {
  sample: { size: number; per_class_cap: number; seconds: number };
  units: string;
  importance: { feature: string; label: string; mean_abs_shap: number }[];
  per_class: Record<ClassName, { feature: string; mean_abs_shap: number }[]>;
  class_feature_matrix: { classes: ClassName[]; features: string[]; values: number[][] };
  beeswarm: { feature: string; mean_abs: number; points: [number, number, number][] }[];
  class_profiles: { features: string[]; medians: Record<ClassName, Record<string, number>> };
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
    res = await fetch(`/api/${path}`, { cache: "no-store", ...init });
  } catch {
    throw new ApiError("Cannot reach the detection service", 0);
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(detail, res.status);
  }
  return res.json() as Promise<T>;
}

function qs(params: Record<string, string | number | boolean | undefined | null>): string {
  const s = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) if (v !== undefined && v !== null && v !== "") s.set(k, String(v));
  const out = s.toString();
  return out ? `?${out}` : "";
}

export const api = {
  stats: (days = 14) => request<Stats>(`v1/stats${qs({ days })}`),
  model: () => request<ModelInfo>("v1/model"),
  explainability: () => request<GlobalExplainability>("v1/model/explainability"),
  samples: () => request<Sample[]>("v1/samples"),
  analyses: (limit = 20, offset = 0) => request<Page<Analysis>>(`v1/analyses${qs({ limit, offset })}`),
  analysis: (id: string) => request<Analysis>(`v1/analyses/${encodeURIComponent(id)}`),
  deleteAnalysis: (id: string) => request<{ deleted: boolean }>(`v1/analyses/${encodeURIComponent(id)}`, { method: "DELETE" }),
  flows: (id: string, p: { cls?: string; severity?: string; threats_only?: boolean; order?: string; limit?: number; offset?: number }) =>
    request<Page<FlowRecord>>(`v1/analyses/${encodeURIComponent(id)}/flows${qs(p)}`),
  flow: (id: string, idx: number) => request<FlowDetail>(`v1/analyses/${encodeURIComponent(id)}/flows/${idx}`),
  threats: (p: { cls?: string; severity?: string; min_confidence?: number; order?: string; limit?: number; offset?: number }) =>
    request<Page<FlowRecord>>(`v1/threats${qs(p)}`),
  upload: (file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    return request<{ analysis: Analysis; top_threats: FlowRecord[] }>("v1/analyses", { method: "POST", body: fd });
  },
  analyzeSample: (name: string) =>
    request<{ analysis: Analysis; top_threats: FlowRecord[] }>(`v1/samples/${encodeURIComponent(name)}/analyze`, { method: "POST" }),
};
