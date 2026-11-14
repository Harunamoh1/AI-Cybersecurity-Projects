"use client";

import { useState } from "react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { TooltipBox } from "@/components/charts";
import { Card, ErrorState, Legend, Loading, PageHeader, Segmented, Stat, Td, Th, useApi } from "@/components/ui";
import { api, CLASSES, type AnomalyResult, type ClassName, type ModelInfo, type SupervisedResult } from "@/lib/api";
import { ATTACK_CLASSES, bytes, CLASS_META, dec, int, MODEL_ORDER, pct } from "@/lib/format";

const MODEL_COLORS: Record<string, string> = {
  xgboost: "#3987e5", random_forest: "#d95926", decision_tree: "#199e70", logistic_regression: "#c98500", isolation_forest: "#d55181",
};
const isSup = (r: SupervisedResult | AnomalyResult): r is SupervisedResult => "test" in r;

function Comparison({ m }: { m: ModelInfo }) {
  const val = Object.fromEntries(m.selection.validation_ranking.map((r) => [r.model, r.macro_f1]));
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[1080px]">
        <thead className="border-b border-line">
          <tr>
            <Th>Model</Th><Th className="text-right">Val macro-F1</Th><Th className="text-right">Macro-F1</Th><Th className="text-right">Accuracy</Th>
            <Th className="text-right">Macro P</Th><Th className="text-right">Macro R</Th><Th className="text-right">ROC-AUC (OvR)</Th>
            <Th className="text-right">Attack recall</Th><Th className="text-right">FPR</Th><Th className="text-right">Binary AUC</Th>
            <Th className="text-right">Train</Th><Th className="text-right">Inference</Th><Th className="text-right">Size</Th>
          </tr>
        </thead>
        <tbody className="divide-y divide-line">
          {MODEL_ORDER.filter((k) => m.models[k]).map((k) => {
            const r = m.models[k];
            const sup = isSup(r);
            return (
              <tr key={k} className={k === m.deployed_model ? "bg-surface-2" : undefined}>
                <Td className="text-ink">
                  <span className="mr-2 inline-block h-2 w-2 rounded-[2px]" style={{ background: MODEL_COLORS[k] }} />
                  {r.display_name}
                  {k === m.deployed_model && <span className="ml-2 rounded border border-accent/50 px-1 py-px text-[10px] uppercase text-accent">deployed</span>}
                  {!sup && <span className="ml-2 text-[10px] uppercase text-ink-3">unsupervised</span>}
                </Td>
                <Td className="tnum text-right font-mono">{sup ? dec(val[k], 4) : "—"}</Td>
                <Td className="tnum text-right font-mono text-ink">{sup ? dec(r.test.macro_f1, 4) : "—"}</Td>
                <Td className="tnum text-right font-mono">{sup ? pct(r.test.accuracy, 2) : "—"}</Td>
                <Td className="tnum text-right font-mono">{sup ? dec(r.test.macro_precision, 3) : "—"}</Td>
                <Td className="tnum text-right font-mono">{sup ? dec(r.test.macro_recall, 3) : "—"}</Td>
                <Td className="tnum text-right font-mono">{sup ? dec(r.test.roc_auc_macro_ovr, 4) : "—"}</Td>
                <Td className="tnum text-right font-mono">{pct(r.binary_test.recall, 2)}</Td>
                <Td className="tnum text-right font-mono">{pct(r.binary_test.false_positive_rate, 3)}</Td>
                <Td className="tnum text-right font-mono">{dec(r.binary_test.roc_auc, 4)}</Td>
                <Td className="tnum text-right font-mono">{r.train_seconds.toFixed(0)} s</Td>
                <Td className="tnum text-right font-mono">{r.inference_us_per_flow.toFixed(1)} µs</Td>
                <Td className="tnum text-right font-mono">{bytes(r.model_size_bytes)}</Td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function PerClass({ m }: { m: ModelInfo }) {
  const sup = MODEL_ORDER.filter((k) => m.models[k] && isSup(m.models[k])) as string[];
  const dep = m.models[m.deployed_model] as SupervisedResult;
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[860px]">
        <thead className="border-b border-line">
          <tr>
            <Th>Class</Th><Th className="text-right">Test support</Th>
            {sup.map((k) => <Th key={k} className="text-right">{m.models[k].display_name} F1</Th>)}
            <Th className="text-right">XGBoost recall (95% CI)</Th>
          </tr>
        </thead>
        <tbody className="divide-y divide-line">
          {CLASSES.map((c) => {
            const best = Math.max(...sup.map((k) => (m.models[k] as SupervisedResult).test.per_class[c].f1));
            const pc = dep.test.per_class[c];
            return (
              <tr key={c}>
                <Td><span className="flex items-center gap-2 text-ink"><span className="h-2 w-2 rounded-[2px]" style={{ background: CLASS_META[c].color }} />{CLASS_META[c].label}</span></Td>
                <Td className="tnum text-right font-mono">{int(pc.support)}</Td>
                {sup.map((k) => {
                  const f = (m.models[k] as SupervisedResult).test.per_class[c].f1;
                  return <Td key={k} className={`tnum text-right font-mono ${f === best ? "text-ink" : ""}`}>{f.toFixed(3)}</Td>;
                })}
                <Td className="tnum text-right font-mono">
                  {pct(pc.recall, 1)} <span className="text-ink-3">[{pct(pc.recall_ci95[0], 0)}–{pct(pc.recall_ci95[1], 0)}]</span>
                </Td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function Confusion({ m }: { m: ModelInfo }) {
  const sup = MODEL_ORDER.filter((k) => m.models[k] && isSup(m.models[k]));
  const [model, setModel] = useState(m.deployed_model);
  const cm = (m.models[model] as SupervisedResult).test.confusion_matrix;
  const [hover, setHover] = useState<string | null>(null);
  return (
    <div>
      <div className="mb-3">
        <Segmented label="Model" value={model} onChange={setModel} options={sup.map((k) => ({ value: k, label: m.models[k].display_name }))} />
      </div>
      <div className="overflow-x-auto">
        <table className="border-separate" style={{ borderSpacing: 2 }}>
          <thead>
            <tr>
              <th className="pr-2 text-right text-[10px] font-normal text-ink-3">true ↓ / predicted →</th>
              {CLASSES.map((c) => <th key={c} className="w-16 px-1 text-[11px] font-medium text-ink-3">{CLASS_META[c].short}</th>)}
            </tr>
          </thead>
          <tbody>
            {cm.map((row, i) => {
              const total = row.reduce((a, b) => a + b, 0);
              return (
                <tr key={i}>
                  <td className="whitespace-nowrap pr-2 text-right text-xs text-ink-2">{CLASS_META[CLASSES[i]].short}</td>
                  {row.map((v, j) => {
                    const share = total ? v / total : 0;
                    const diag = i === j;
                    const key = `${i}-${j}`;
                    const bg = v === 0 ? "var(--surface-2)" : diag
                      ? `color-mix(in srgb, #3987e5 ${15 + share * 85}%, var(--surface-2))`
                      : `color-mix(in srgb, #e66767 ${20 + Math.min(1, share * 8) * 80}%, var(--surface-2))`;
                    return (
                      <td key={j} onMouseEnter={() => setHover(key)} onMouseLeave={() => setHover(null)}
                        title={`True ${CLASSES[i]} → predicted ${CLASSES[j]}: ${v.toLocaleString()} (${pct(share, 2)} of row)`}
                        className="h-11 rounded-[3px] text-center font-mono text-[11px]"
                        style={{ background: bg, color: v === 0 ? "var(--text-3)" : "var(--text)", outline: hover === key ? "1px solid var(--text)" : undefined }}>
                        {v === 0 ? "·" : v >= 10000 ? `${(v / 1000).toFixed(0)}k` : v.toLocaleString()}
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="mt-3 text-xs text-ink-3">Blue: correct (shade = row recall). Red: errors (shade = share of the row). Hover a cell for exact values.</p>
    </div>
  );
}

function RocChart({ series }: { series: { key: string; label: string; color: string; points: [number, number][] }[] }) {
  const data = series.map((s) => ({ ...s, points: s.points.map(([x, y]) => ({ x: Math.max(x, 1e-6), y })) }));
  return (
    <div>
      <div className="h-72">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart margin={{ top: 6, right: 12, left: -10, bottom: 4 }}>
            <CartesianGrid />
            <XAxis type="number" dataKey="x" scale="log" domain={[1e-6, 1]} allowDataOverflow tickLine={false}
              ticks={[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]} tickFormatter={(v: number) => (v >= 0.01 ? `${v * 100}%` : `${(v * 100).toPrecision(1)}%`)} />
            <YAxis type="number" dataKey="y" domain={[0, 1]} tickFormatter={(v: number) => `${Math.round(v * 100)}%`} tickLine={false} />
            <Tooltip content={({ active, payload }) => active && payload?.length ? (
              <TooltipBox title={`FPR ${pct(payload[0].payload.x, 4)}`}
                rows={payload.map((p) => ({ label: String(p.name), value: pct(p.payload.y, 2), color: p.color as string }))} />
            ) : null} />
            {data.map((s) => (
              <Line key={s.key} data={s.points} dataKey="y" name={s.label} stroke={s.color} strokeWidth={2} dot={false} isAnimationActive={false} />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="mt-2"><Legend items={series.map((s) => ({ label: s.label, color: s.color }))} /></div>
      <p className="mt-2 text-xs text-ink-3">False-positive rate on a log axis: at IDS traffic volumes the region below 1% FPR is the one that matters.</p>
    </div>
  );
}

function Rocs({ m }: { m: ModelInfo }) {
  const [view, setView] = useState<"models" | "classes">("models");
  const dep = m.models[m.deployed_model] as SupervisedResult;
  const series = view === "models"
    ? MODEL_ORDER.filter((k) => m.models[k]).map((k) => ({ key: k, label: `${m.models[k].display_name} (AUC ${dec(m.models[k].binary_test.roc_auc, 4)})`, color: MODEL_COLORS[k], points: m.models[k].binary_roc }))
    : ATTACK_CLASSES.filter((c) => dep.class_roc?.[c]).map((c) => ({ key: c, label: `${CLASS_META[c].short} (AUC ${dec(dep.test.per_class[c].roc_auc, 4)})`, color: CLASS_META[c].color, points: dep.class_roc![c]! }));
  return (
    <div>
      <div className="mb-3"><Segmented label="ROC view" value={view} onChange={setView}
        options={[{ value: "models", label: "Attack vs benign — all models" }, { value: "classes", label: "Per class — XGBoost" }]} /></div>
      <RocChart series={series} />
    </div>
  );
}

function Calibration({ m }: { m: ModelInfo }) {
  const c = m.calibration;
  const toPts = (bins: typeof c.after.bins) => bins.filter((b) => b.count >= 20).map((b) => ({ x: b.confidence, y: b.accuracy, n: b.count }));
  const lines = [
    { key: "before", label: `Uncalibrated (ECE ${dec(c.before.ece, 4)})`, color: "#c98500", points: toPts(c.before.bins) },
    { key: "after", label: `Temperature-scaled (ECE ${dec(c.after.ece, 4)})`, color: "#3987e5", points: toPts(c.after.bins) },
  ];
  return (
    <div>
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart margin={{ top: 6, right: 12, left: -10, bottom: 4 }}>
            <CartesianGrid />
            <XAxis type="number" dataKey="x" domain={[0.1, 1]} tickFormatter={(v: number) => `${Math.round(v * 100)}%`} tickLine={false} />
            <YAxis type="number" dataKey="y" domain={[0, 1]} tickFormatter={(v: number) => `${Math.round(v * 100)}%`} tickLine={false} />
            <Line data={[{ x: 0.1, y: 0.1 }, { x: 1, y: 1 }]} dataKey="y" stroke="var(--axis)" strokeDasharray="4 4" dot={false} isAnimationActive={false} name="Perfect calibration" />
            {lines.map((l) => (
              <Line key={l.key} data={l.points} dataKey="y" name={l.label} stroke={l.color} strokeWidth={2} dot={{ r: 3, strokeWidth: 0, fill: l.color }} isAnimationActive={false} />
            ))}
            <Tooltip content={({ active, payload }) => active && payload?.length ? (
              <TooltipBox title={`Confidence ${pct(payload[0].payload.x, 1)}`}
                rows={payload.filter((p) => p.payload.n).map((p) => ({ label: `${String(p.name).split(" (")[0]} accuracy`, value: `${pct(p.payload.y, 1)} (n=${int(p.payload.n)})`, color: p.color as string }))} />
            ) : null} />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="mt-2"><Legend items={lines.map((l) => ({ label: l.label, color: l.color }))} /></div>
      <p className="mt-2 text-xs leading-relaxed text-ink-3">
        {c.method}. T = {dec(c.temperature, 3)}; test NLL {dec(c.test_nll_before, 5)} → {dec(c.test_nll_after, 5)}. Temperature scaling never changes the predicted class.
        Bins with fewer than 20 flows are hidden. Most flows sit in the top bin (confidence &gt; 93%).
      </p>
    </div>
  );
}

function Temporal({ m }: { m: ModelInfo }) {
  const t = m.experiments.temporal_split;
  if (!t) return <p className="text-sm text-ink-3">Not run.</p>;
  return (
    <div>
      <p className="mb-3 text-xs leading-relaxed text-ink-3">{t.description}</p>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[820px]">
          <thead className="border-b border-line">
            <tr><Th>Model</Th><Th className="text-right">Macro-F1 random split</Th><Th className="text-right">Macro-F1 temporal split</Th><Th className="text-right">Δ</Th>
              <Th className="text-right">Attack recall</Th><Th className="text-right">FPR</Th><Th className="text-right">Binary AUC</Th></tr>
          </thead>
          <tbody className="divide-y divide-line">
            {MODEL_ORDER.filter((k) => t.results[k]).map((k) => {
              const r = t.results[k];
              const p = m.models[k];
              const pf = isSup(p) ? p.test.macro_f1 : null;
              return (
                <tr key={k}>
                  <Td className="text-ink">{r.display_name}</Td>
                  <Td className="tnum text-right font-mono">{dec(pf, 4)}</Td>
                  <Td className="tnum text-right font-mono text-ink">{dec(r.macro_f1, 4)}</Td>
                  <Td className="tnum text-right font-mono">{pf != null && r.macro_f1 != null ? (r.macro_f1 - pf).toFixed(3) : "—"}</Td>
                  <Td className="tnum text-right font-mono">{pct(r.binary.recall, 2)}</Td>
                  <Td className="tnum text-right font-mono">{pct(r.binary.false_positive_rate, 3)}</Td>
                  <Td className="tnum text-right font-mono">{dec(r.binary.roc_auc, 4)}</Td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {t.results.xgboost?.per_class_recall && (
        <div className="mt-4 flex flex-wrap gap-2">
          {ATTACK_CLASSES.map((c) => (
            <div key={c} className="rounded border border-line px-2.5 py-1.5 text-xs">
              <div className="text-ink-3">{CLASS_META[c].short} recall (XGBoost)</div>
              <div className="tnum font-mono text-ink">
                {pct((m.models.xgboost as SupervisedResult).test.per_class[c].recall, 1)} → {pct(t.results.xgboost.per_class_recall![c], 1)}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

const TRAIN_LABEL: Record<string, string> = {
  trained_2017: "CIC-IDS2017 (deployed)", trained_2018: "CSE-CIC-IDS2018 only", trained_joint: "2017 + 2018 jointly",
  trained_2017_no_host_stack: "2017, without host TCP-stack features",
};

function CrossDataset({ m }: { m: ModelInfo }) {
  const x = m.experiments.cross_dataset;
  if (!x) return <p className="text-sm text-ink-3">Not run.</p>;
  const keys = Object.keys(x.summary.test_2018);
  const classes2018 = Object.keys(x.summary.test_2018.trained_2017.per_class_recall) as ClassName[];
  return (
    <div>
      <p className="mb-3 text-xs leading-relaxed text-ink-3">
        Does a model trained on 2017 traffic detect the same attack types captured a year later on a different network? 2018 flows: {x.dataset.sampling}; split chronologically within each day and class.
        Macro-F1 over the classes present (2018 has no port scans). Ablation removes: <span className="font-mono">{x.ablation_excluded_features.join(", ")}</span>.
      </p>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[860px]">
          <thead className="border-b border-line">
            <tr><Th>Trained on</Th><Th className="text-right">2017 test macro-F1</Th><Th className="text-right">2017 attack recall</Th>
              <Th className="text-right">2018 test macro-F1</Th><Th className="text-right">2018 attack recall</Th><Th className="text-right">2018 FPR</Th></tr>
          </thead>
          <tbody className="divide-y divide-line">
            {keys.map((k) => {
              const a = x.summary.test_2017[k], b = x.summary.test_2018[k];
              return (
                <tr key={k} className={k === "trained_2017" ? "bg-surface-2" : undefined}>
                  <Td className="text-ink">{TRAIN_LABEL[k] ?? k}</Td>
                  <Td className="tnum text-right font-mono">{dec(a.macro_f1, 4)}</Td>
                  <Td className="tnum text-right font-mono">{pct(a.attack_recall, 1)}</Td>
                  <Td className="tnum text-right font-mono text-ink">{dec(b.macro_f1, 4)}</Td>
                  <Td className="tnum text-right font-mono">{pct(b.attack_recall, 1)}</Td>
                  <Td className="tnum text-right font-mono">{pct(b.false_positive_rate, 2)}</Td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="mt-4 overflow-x-auto">
        <table className="w-full min-w-[720px]">
          <thead className="border-b border-line">
            <tr><Th>2018 per-class recall</Th>{classes2018.map((c) => <Th key={c} className="text-right">{CLASS_META[c].short}</Th>)}</tr>
          </thead>
          <tbody className="divide-y divide-line">
            {keys.map((k) => (
              <tr key={k}>
                <Td className="text-ink-2">{TRAIN_LABEL[k] ?? k}</Td>
                {classes2018.map((c) => <Td key={c} className="tnum text-right font-mono">{pct(x.summary.test_2018[k].per_class_recall[c], 1)}</Td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function Anomaly({ m }: { m: ModelInfo }) {
  const r = m.models.isolation_forest as AnomalyResult | undefined;
  if (!r) return null;
  const p = m.unseen_attack_probe;
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <div>
        <p className="text-xs leading-relaxed text-ink-3">Trained on {r.trained_on}. Threshold: {r.threshold.rule}.</p>
        <dl className="mt-3 grid grid-cols-2 gap-3 text-[13px]">
          {[["ROC-AUC", dec(r.binary_test.roc_auc, 4)], ["Recall at 1% FPR budget", pct(r.binary_test.recall, 2)],
            ["Recall at best-F1 threshold*", pct(r.binary_test_f1_threshold.recall, 2)], ["FPR at best-F1 threshold*", pct(r.binary_test_f1_threshold.false_positive_rate, 2)]].map(([k, v]) => (
            <div key={k} className="rounded border border-line px-3 py-2"><dt className="text-xs text-ink-3">{k}</dt><dd className="tnum font-mono text-ink">{v}</dd></div>
          ))}
        </dl>
        <p className="mt-2 text-[11px] text-ink-3">*Threshold chosen with validation labels — shown for context, not a fair unsupervised result.</p>
      </div>
      <div>
        <div className="mb-2 text-xs text-ink-3">Detection rate per attack class at the 1% FPR budget</div>
        <ul className="space-y-1.5 text-xs">
          {ATTACK_CLASSES.filter((c) => r.per_class_detection[c]).map((c) => {
            const d = r.per_class_detection[c]!;
            return (
              <li key={c} className="flex items-center justify-between">
                <span className="flex items-center gap-2 text-ink-2"><span className="h-2 w-2 rounded-[2px]" style={{ background: CLASS_META[c].color }} />{CLASS_META[c].short}</span>
                <span className="tnum font-mono text-ink">{pct(d.detection_rate, 1)} <span className="text-ink-3">of {int(d.support)}</span></span>
              </li>
            );
          })}
        </ul>
        {p.flows != null && (
          <p className="mt-3 rounded border border-line px-3 py-2 text-xs leading-relaxed text-ink-2">
            <span className="text-ink">Unseen-attack probe (Heartbleed, never in training):</span> XGBoost flagged {p.xgboost_flagged_as_attack}/{p.flows} flows as an attack
            ({Object.entries(p.xgboost_predictions ?? {}).map(([k, v]) => `${v} as ${k}`).join(", ")}); Isolation Forest flagged {p.isolation_forest_flagged}/{p.flows}.
          </p>
        )}
      </div>
    </div>
  );
}

function DataFacts({ m }: { m: ModelInfo }) {
  const c = m.dataset.cleaning;
  const t = m.dataset.splits;
  const sum = (o: Record<string, number>) => Object.values(o).reduce((a, b) => a + b, 0);
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <div className="text-[13px]">
        <dl className="divide-y divide-line">
          {[["Raw flows", int(c.raw_rows)], ["Exact duplicates removed", int(c.duplicate_rows_removed)],
            ["Contradictory flows removed", `${int(c.conflicting_rows_removed)} (${int(c.conflicting_vectors)} feature vectors with >1 label)`],
            ["Excluded labels", Object.entries(c.excluded_labels).map(([k, v]) => `${k} (${v})`).join(", ") || "—"],
            ["Clean flows", int(c.clean_rows)], ["Infinite / missing cells", `${int(c.cells_inf)} / ${int(c.cells_nan)}`],
            ["Split", m.dataset.split_protocol], ["Train / val / test", `${int(sum(t.train))} / ${int(sum(t.val))} / ${int(sum(t.test))}`]].map(([k, v]) => (
            <div key={k} className="flex justify-between gap-4 py-1.5"><dt className="text-ink-2">{k}</dt><dd className="tnum text-right font-mono text-xs text-ink">{v}</dd></div>
          ))}
        </dl>
        <p className="mt-3 text-[11px] leading-relaxed text-ink-3">{m.dataset.citation}</p>
      </div>
      <div className="text-xs">
        <div className="mb-2 text-ink-2">{m.features.selected.length} features selected of 77 · selection fitted on the training split only</div>
        <details>
          <summary className="cursor-pointer text-ink-2 hover:text-ink">Dropped features and reasons ({Object.keys(m.features.dropped).length})</summary>
          <ul className="mt-2 space-y-1 font-mono text-[11px] text-ink-3">
            {Object.entries(m.features.dropped).map(([f, r]) => <li key={f}><span className="text-ink-2">{f}</span> — {r}</li>)}
          </ul>
        </details>
        <details className="mt-2">
          <summary className="cursor-pointer text-ink-2 hover:text-ink">Class weights (√ inverse frequency, capped)</summary>
          <ul className="mt-2 grid grid-cols-2 gap-1 font-mono text-[11px] text-ink-3">
            {Object.entries(m.dataset.class_weights).map(([k, v]) => <li key={k}>{k}: {v}</li>)}
          </ul>
        </details>
      </div>
    </div>
  );
}

export default function ModelPage() {
  const { data: m, error, reload } = useApi(api.model);
  if (error) return <ErrorState message={error} onRetry={reload} />;
  if (!m) return <Loading />;
  const dep = m.models[m.deployed_model] as SupervisedResult;
  const t = m.experiments.temporal_split?.results.xgboost;
  const x = m.experiments.cross_dataset?.summary.test_2018.trained_2017;
  return (
    <>
      <PageHeader title="Model performance"
        description={<>All numbers below were produced by <span className="font-mono text-ink">python -m sentinelai.training.train</span> on the held-out test split and are reproducible from the pinned dataset (seed {String(m.environment.seed)}). Model selection, early stopping and calibration used only the validation split.</>} />

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Stat label="Macro-F1 · random split" value={dec(dep.test.macro_f1, 3)} sub={`accuracy ${pct(dep.test.accuracy, 2)}`} />
        <Stat label="Attack detection" value={pct(dep.binary_test.recall, 2)} sub={`FPR ${pct(dep.binary_test.false_positive_rate, 3)} · ${int(dep.binary_test.fp)} false alarms`} />
        <Stat label="Macro-F1 · temporal split" value={dec(t?.macro_f1, 3)} sub="tested on later traffic" />
        <Stat label="Macro-F1 · CIC-IDS2018" value={dec(x?.macro_f1, 3)} sub="different network, 2017 model" tone={x && x.macro_f1 < 0.5 ? "var(--sev-high)" : undefined} />
      </div>

      <Card title="Model comparison" subtitle="Primary protocol — stratified random split after de-duplication" className="mt-4" bodyClass="p-0">
        <Comparison m={m} />
        <p className="border-t border-line px-4 py-2.5 text-xs leading-relaxed text-ink-3">
          Selection rule: {m.selection.rule} Inference time is batch scoring of the test split on one machine ({String(m.environment.platform).split("-")[0]}, {String(m.environment.jobs)} threads).
        </p>
      </Card>

      <div className="mt-4 grid gap-4 xl:grid-cols-2">
        <Card title="Confusion matrix" subtitle="Test split"><Confusion m={m} /></Card>
        <Card title="ROC curves" subtitle="Test split"><Rocs m={m} /></Card>
      </div>

      <Card title="Per-class performance" subtitle="F1 per model; best per row in white. Wilson 95% intervals show how little the smallest classes can tell us." className="mt-4" bodyClass="p-0">
        <PerClass m={m} />
      </Card>

      <div className="mt-4 grid gap-4 xl:grid-cols-2">
        <Card title="Calibration" subtitle="Reliability of the deployed model’s confidence (test split)"><Calibration m={m} /></Card>
        <Card title="Anomaly detection baseline" subtitle="Isolation Forest"><Anomaly m={m} /></Card>
      </div>

      <Card title="Robustness: temporal split" subtitle="Same models, tested only on traffic recorded after the training traffic" className="mt-4"><Temporal m={m} /></Card>
      <Card title="Generalisation: CSE-CIC-IDS2018" subtitle="Cross-dataset experiments" className="mt-4"><CrossDataset m={m} /></Card>
      <Card title="Data and features" className="mt-4"><DataFacts m={m} /></Card>
    </>
  );
}
