import type { RiskLevel } from "./api";

export const RISK_META: Record<RiskLevel, { label: string; color: string; text: string; bg: string; ring: string }> = {
  low: { label: "Low", color: "var(--risk-low)", text: "text-low", bg: "bg-low", ring: "ring-low/40" },
  medium: { label: "Medium", color: "var(--risk-medium)", text: "text-medium", bg: "bg-medium", ring: "ring-medium/40" },
  high: { label: "High", color: "var(--risk-high)", text: "text-high", bg: "bg-high", ring: "ring-high/40" },
  critical: { label: "Critical", color: "var(--risk-critical)", text: "text-critical", bg: "bg-critical", ring: "ring-critical/40" },
};

export const pct = (v: number | null | undefined, digits = 1) =>
  v === null || v === undefined ? "—" : `${(v * 100).toFixed(digits)}%`;

export const num = (v: number, digits = 0) =>
  v.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });

export function relativeTime(iso: string): string {
  const diff = (Date.now() - new Date(iso).getTime()) / 1000;
  if (diff < 60) return "just now";
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return new Date(iso).toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

export const MODEL_LABELS: Record<string, string> = {
  hist_gradient_boosting: "Histogram Gradient Boosting",
  random_forest: "Random Forest",
  logistic_regression: "Logistic Regression",
  char_ngram_logreg: "Char n-gram TF-IDF + LogReg",
};

export const INDICATOR_LABELS: Record<string, string> = {
  brand: "Brand impersonation",
  lookalike: "Look-alike domain",
  keyword: "Credential keywords",
  ip: "Raw IP host",
  obfuscation: "Obfuscation",
  shortener: "URL shortener",
  hosting: "Free hosting",
  structure: "Suspicious structure",
  payload: "Executable payload",
};

export function formatFeatureValue(v: number | string): string {
  if (typeof v === "string") return v;
  if (Number.isInteger(v)) return String(v);
  return v.toFixed(2);
}
