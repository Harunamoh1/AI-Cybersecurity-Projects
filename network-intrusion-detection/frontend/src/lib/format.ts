import type { ClassName, Severity } from "./api";

export const CLASS_META: Record<ClassName, { label: string; color: string; short: string }> = {
  BENIGN: { label: "Benign", short: "Benign", color: "var(--c-benign)" },
  DoS: { label: "Denial of Service", short: "DoS", color: "var(--c-dos)" },
  DDoS: { label: "Distributed DoS", short: "DDoS", color: "var(--c-ddos)" },
  PortScan: { label: "Port Scan", short: "PortScan", color: "var(--c-portscan)" },
  BruteForce: { label: "Brute Force", short: "Brute Force", color: "var(--c-bruteforce)" },
  WebAttack: { label: "Web Attack", short: "Web Attack", color: "var(--c-webattack)" },
  Bot: { label: "Botnet", short: "Botnet", color: "var(--c-bot)" },
  Infiltration: { label: "Infiltration", short: "Infiltration", color: "var(--c-infiltration)" },
};

export const ATTACK_CLASSES = ["DoS", "DDoS", "PortScan", "BruteForce", "WebAttack", "Bot", "Infiltration"] as const;

export const SEVERITY_META: Record<Severity, { label: string; color: string; rank: number }> = {
  critical: { label: "Critical", color: "var(--sev-critical)", rank: 4 },
  high: { label: "High", color: "var(--sev-high)", rank: 3 },
  medium: { label: "Medium", color: "var(--sev-medium)", rank: 2 },
  low: { label: "Low", color: "var(--sev-low)", rank: 1 },
  none: { label: "None", color: "var(--c-benign)", rank: 0 },
};

export const MODEL_ORDER = ["xgboost", "random_forest", "decision_tree", "logistic_regression", "isolation_forest"];

const nf = new Intl.NumberFormat("en-US");
export const int = (n: number | null | undefined) => (n == null ? "—" : nf.format(Math.round(n)));
export const pct = (n: number | null | undefined, digits = 1) => (n == null ? "—" : `${(n * 100).toFixed(digits)}%`);
export const dec = (n: number | null | undefined, digits = 3) => (n == null ? "—" : n.toFixed(digits));

export function compact(n: number): string {
  if (Math.abs(n) >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
  if (Math.abs(n) >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (Math.abs(n) >= 1e4) return `${(n / 1e3).toFixed(0)}k`;
  if (Math.abs(n) >= 1e3) return `${(n / 1e3).toFixed(1)}k`;
  return nf.format(n);
}

export function bytes(n: number): string {
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)} MB`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(0)} kB`;
  return `${n} B`;
}

/** Flow-feature values: CICFlowMeter times are microseconds. */
export function featureValue(v: number | null | undefined, unit = ""): string {
  if (v == null || Number.isNaN(v)) return "—";
  if (unit === "µs") {
    if (v >= 1e6) return `${(v / 1e6).toFixed(2)} s`;
    if (v >= 1e3) return `${(v / 1e3).toFixed(1)} ms`;
    return `${v.toFixed(0)} µs`;
  }
  const s = Math.abs(v) >= 1e4 ? compact(v) : Number.isInteger(v) ? nf.format(v) : v.toFixed(2);
  return unit && unit !== "µs" ? `${s} ${unit}` : s;
}

export function dateTime(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleString("en-GB", { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" });
}

export function relative(iso: string): string {
  const s = (Date.now() - new Date(iso).getTime()) / 1000;
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  return `${Math.floor(s / 86400)} d ago`;
}

export const featureLabel = (name: string, map?: Record<string, string>) =>
  map?.[name] ?? name.replace(/_/g, " ").replace(/\b(fwd|bwd|iat|psh|urg|ack|syn|fin|rst)\b/g, (m) => m.toUpperCase());
