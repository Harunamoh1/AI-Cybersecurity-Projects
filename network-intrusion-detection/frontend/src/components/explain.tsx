"use client";

import { useMemo, useState } from "react";
import type { ClassName, ContributionItem, Explanation, GlobalExplainability } from "@/lib/api";
import { CLASS_META, dec, featureLabel, featureValue, pct } from "@/lib/format";
import { BarList } from "./charts";
import { Legend } from "./ui";

const POS = "#e66767"; // diverging red: pushes toward the predicted class / toward "attack"
const NEG = "#3987e5"; // diverging blue: pushes away
const SEQ = ["#104281", "#184f95", "#1c5cab", "#256abf", "#2a78d6", "#3987e5", "#5598e7", "#6da7ec", "#86b6ef", "#9ec5f4", "#b7d3f6"];
const seq = (t: number) => SEQ[Math.max(0, Math.min(SEQ.length - 1, Math.round(t * (SEQ.length - 1))))];

/** Diverging horizontal bars of SHAP contributions (centered on zero). */
export function ContributionChart({ items, labels, positiveLabel, negativeLabel }: {
  items: ContributionItem[];
  labels?: Record<string, string>;
  positiveLabel: string;
  negativeLabel: string;
}) {
  const max = Math.max(1e-9, ...items.map((i) => Math.abs(i.impact)));
  return (
    <div>
      <div className="mb-3"><Legend items={[{ label: positiveLabel, color: POS }, ...(negativeLabel ? [{ label: negativeLabel, color: NEG }] : [])]} /></div>
      <ul className="space-y-1.5">
        {items.map((c) => {
          const w = (Math.abs(c.impact) / max) * 50;
          const pos = c.impact >= 0;
          return (
            <li key={c.feature} className="grid grid-cols-[minmax(0,15rem)_1fr_4.5rem] items-center gap-3 text-xs"
              title={`${c.description}. Value ${featureValue(c.value, c.unit)}${c.benign_median != null ? `, typical benign ${featureValue(c.benign_median, c.unit)}` : ""}. SHAP ${c.impact.toFixed(3)}`}>
              <div className="min-w-0">
                <div className="truncate text-ink">{labels?.[c.feature] ?? c.label}</div>
                <div className="truncate font-mono text-[11px] text-ink-3">
                  {featureValue(c.value, c.unit)}
                  {c.benign_median != null && <span> · benign median {featureValue(c.benign_median, c.unit)}</span>}
                </div>
              </div>
              <div className="relative h-4">
                <div className="absolute inset-y-0 left-1/2 w-px bg-axis" style={{ background: "var(--axis)" }} />
                <div className="absolute inset-y-0.5 rounded-[3px]"
                  style={{ background: pos ? POS : NEG, width: `${Math.max(w, 0.6)}%`, left: pos ? "50%" : `${50 - w}%` }} />
              </div>
              <div className="tnum text-right font-mono text-ink-2">{c.impact > 0 ? "+" : ""}{c.impact.toFixed(3)}</div>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

export function LocalExplanation({ e }: { e: Explanation }) {
  const meta = CLASS_META[e.predicted_class];
  const benign = e.predicted_class === "BENIGN";
  const probs = (Object.entries(e.probabilities) as [ClassName, number][]).sort((a, b) => b[1] - a[1]);
  return (
    <div className="grid gap-4 xl:grid-cols-3">
      <div className="space-y-4">
        <div className="rounded-lg border border-line bg-surface p-4">
          <div className="text-[11px] uppercase tracking-wider text-ink-3">Classification</div>
          <div className="mt-1.5 flex items-center gap-2 text-lg font-semibold text-ink">
            <span className="h-3 w-3 rounded-[3px]" style={{ background: meta.color }} />{meta.label}
          </div>
          <dl className="mt-3 space-y-1.5 text-[13px]">
            <div className="flex justify-between"><dt className="text-ink-2">Calibrated confidence</dt><dd className="tnum font-mono text-ink">{pct(e.confidence, 2)}</dd></div>
            <div className="flex justify-between"><dt className="text-ink-2">Attack probability</dt><dd className="tnum font-mono text-ink">{pct(e.attack_probability, 2)}</dd></div>
            <div className="flex justify-between"><dt className="text-ink-2">Logit = base + Σ SHAP</dt><dd className="tnum font-mono text-ink">{dec(e.base_value, 2)} → {dec(e.logit, 2)}</dd></div>
          </dl>
        </div>
        <div className="rounded-lg border border-line bg-surface p-4">
          <div className="mb-3 text-[13px] font-medium text-ink">Class probabilities</div>
          <BarList format={(n) => pct(n, 2)} items={probs.slice(0, 5).map(([c, p]) => ({ key: c, label: CLASS_META[c].short, value: p, color: CLASS_META[c].color }))} />
        </div>
      </div>
      <div className="space-y-4 xl:col-span-2">
        <div className="rounded-lg border border-line bg-surface p-4">
          <div className="text-[13px] font-medium text-ink">Why “{meta.short}”?</div>
          <p className="mb-4 mt-0.5 text-xs text-ink-3">
            Exact TreeSHAP contributions of the flow’s features to the {meta.short} logit. The remaining features add {e.other_features_impact >= 0 ? "+" : ""}{e.other_features_impact.toFixed(3)}.
          </p>
          <ContributionChart items={e.contributions} positiveLabel={`Pushes toward ${meta.short}`} negativeLabel={`Pushes away from ${meta.short}`} />
        </div>
        {!benign && e.attack_evidence.length > 0 && (
          <div className="rounded-lg border border-line bg-surface p-4">
            <div className="text-[13px] font-medium text-ink">What made it look malicious</div>
            <p className="mb-4 mt-0.5 text-xs text-ink-3">Features that lowered the BENIGN logit, regardless of attack type.</p>
            <ContributionChart items={e.attack_evidence.slice(0, 8)} positiveLabel="Evidence against benign" negativeLabel="" />
          </div>
        )}
      </div>
    </div>
  );
}

export function ImportanceChart({ g }: { g: GlobalExplainability }) {
  return (
    <BarList format={(n) => n.toFixed(3)}
      items={g.importance.slice(0, 15).map((d) => ({ key: d.feature, label: d.label, value: d.mean_abs_shap, color: "var(--accent)" }))} />
  );
}

export function ClassFeatureHeatmap({ g, labels }: { g: GlobalExplainability; labels: Record<string, string> }) {
  const { classes, features, values } = g.class_feature_matrix;
  const max = Math.max(...values.flat());
  const [hover, setHover] = useState<string | null>(null);
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[720px] border-separate" style={{ borderSpacing: 2 }}>
        <thead>
          <tr>
            <th />
            {classes.map((c) => <th key={c} className="px-1 pb-1 text-left text-[11px] font-medium text-ink-3">{CLASS_META[c].short}</th>)}
          </tr>
        </thead>
        <tbody>
          {features.map((f, j) => (
            <tr key={f}>
              <td className="whitespace-nowrap pr-3 text-xs text-ink-2">{labels[f] ?? featureLabel(f)}</td>
              {classes.map((c, i) => {
                const v = values[i][j];
                const t = Math.sqrt(v / max);
                const key = `${c}-${f}`;
                return (
                  <td key={c} onMouseEnter={() => setHover(key)} onMouseLeave={() => setHover(null)}
                    title={`${CLASS_META[c].short} · ${labels[f] ?? f}: mean |SHAP| ${v.toFixed(3)}`}
                    className="h-7 min-w-14 rounded-[3px] text-center font-mono text-[10px]"
                    style={{ background: v ? seq(t) : "var(--surface-2)", color: t > 0.6 ? "#0b0b0b" : "var(--text-2)",
                      outline: hover === key ? "1px solid var(--text)" : undefined }}>
                    {hover === key || t > 0.6 ? v.toFixed(2) : ""}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <div className="mt-3 flex items-center gap-2 text-[11px] text-ink-3">
        low
        <div className="h-2 w-40 rounded" style={{ background: `linear-gradient(to right, ${SEQ.join(",")})` }} />
        high mean |SHAP| (class’s own logit, flows of that class)
      </div>
    </div>
  );
}

/** SHAP beeswarm of "attack evidence" (−SHAP of BENIGN). Colour = feature value. */
export function Beeswarm({ g, labels }: { g: GlobalExplainability; labels: Record<string, string> }) {
  const rows = g.beeswarm.slice(0, 10);
  const [focus, setFocus] = useState<ClassName | "all">("all");
  const lim = useMemo(() => {
    const all = rows.flatMap((r) => r.points.map((p) => Math.abs(p[0])));
    all.sort((a, b) => a - b);
    return Math.max(0.5, all[Math.floor(all.length * 0.995)] ?? 1);
  }, [rows]);
  const W = 820, rowH = 30, left = 190, plotW = W - left - 10, H = rows.length * rowH + 26;
  const x = (v: number) => left + ((Math.max(-lim, Math.min(lim, v)) + lim) / (2 * lim)) * plotW;
  const ORDER = ["BENIGN", "DoS", "DDoS", "PortScan", "BruteForce", "WebAttack", "Bot", "Infiltration"] as ClassName[];
  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center gap-2 text-xs">
        <span className="text-ink-3">Highlight</span>
        {(["all", "DoS", "DDoS", "PortScan", "BruteForce", "WebAttack", "Bot", "Infiltration", "BENIGN"] as const).map((c) => (
          <button key={c} onClick={() => setFocus(c)} aria-pressed={focus === c}
            className={`rounded border px-2 py-0.5 ${focus === c ? "border-accent text-ink" : "border-line text-ink-3 hover:text-ink-2"}`}>
            {c === "all" ? "All flows" : CLASS_META[c].short}
          </button>
        ))}
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="SHAP beeswarm of attack evidence for the top features">
        <line x1={x(0)} x2={x(0)} y1={0} y2={rows.length * rowH} stroke="var(--axis)" />
        {rows.map((r, j) => {
          const cy = j * rowH + rowH / 2;
          return (
            <g key={r.feature}>
              <line x1={left} x2={W - 10} y1={cy} y2={cy} stroke="var(--grid)" />
              <text x={0} y={cy + 4} fontSize="12" fill="var(--text-2)">{(labels[r.feature] ?? featureLabel(r.feature)).slice(0, 26)}</text>
              {r.points.map((p, k) => {
                const cls = ORDER[p[2]];
                const dim = focus !== "all" && cls !== focus;
                const jitter = (((k * 9301 + 49297) % 233280) / 233280 - 0.5) * (rowH - 10);
                return (
                  <circle key={k} cx={x(p[0])} cy={cy + jitter} r={2.2} fill={seq(p[1])} opacity={dim ? 0.07 : 0.85}>
                    <title>{`${CLASS_META[cls].short} flow · SHAP ${p[0].toFixed(3)} · feature value ${(p[1] * 100).toFixed(0)}% of training range`}</title>
                  </circle>
                );
              })}
            </g>
          );
        })}
        <text x={left} y={H - 6} fontSize="11" fill="var(--text-3)">← evidence for benign</text>
        <text x={W - 10} y={H - 6} fontSize="11" fill="var(--text-3)" textAnchor="end">evidence for attack →</text>
      </svg>
      <div className="mt-2 flex items-center gap-2 text-[11px] text-ink-3">
        feature value: low
        <div className="h-2 w-32 rounded" style={{ background: `linear-gradient(to right, ${SEQ.join(",")})` }} />
        high (log-scaled within the training range)
      </div>
    </div>
  );
}
