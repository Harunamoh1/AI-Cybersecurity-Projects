"use client";

import clsx from "clsx";
import { AlertTriangle, Database, FlaskConical } from "lucide-react";
import { useEffect, useState } from "react";
import { Area, AreaChart, Bar, BarChart, CartesianGrid, Legend, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ChartTooltip, ErrorBanner, PageHeader, Panel, Skeleton, StatTile } from "@/components/ui";
import { api, type Metrics, type ModelCard } from "@/lib/api";
import { MODEL_LABELS, num, pct } from "@/lib/format";

const AXIS = { stroke: "var(--line-strong)", tick: { fill: "var(--text-3)", fontSize: 11, fontFamily: "var(--font-geist-mono)" }, tickLine: false };
const SOURCE_LABELS: Record<string, string> = {
  phiusiil: "PhiUSIIL (UCI)",
  hannousse: "Hannousse & Yahiouche (Mendeley)",
  commoncrawl_legit: "Legitimate deep URLs (Common Crawl × Tranco)",
};

function ConfusionMatrix({ m }: { m: Metrics }) {
  const { tn, fp, fn, tp } = m.confusion_matrix;
  const cells = [
    { label: "True negative", v: tn, row: tn + fp, good: true },
    { label: "False positive", v: fp, row: tn + fp, good: false },
    { label: "False negative", v: fn, row: fn + tp, good: false },
    { label: "True positive", v: tp, row: fn + tp, good: true },
  ];
  return (
    <div className="grid grid-cols-[auto_1fr_1fr] gap-1.5 text-center">
      <div />
      <div className="label pb-1">Pred. legit</div>
      <div className="label pb-1">Pred. phishing</div>
      {[0, 1].map((r) => (
        <div key={r} className="contents">
          <div className="label flex items-center pr-2 [writing-mode:vertical-rl] rotate-180 sm:[writing-mode:horizontal-tb] sm:rotate-0">{r === 0 ? "Legit" : "Phishing"}</div>
          {cells.slice(r * 2, r * 2 + 2).map((c) => {
            const share = c.v / c.row;
            return (
              <div
                key={c.label}
                className={clsx("rounded-lg border px-2 py-4", c.good ? "border-line-strong" : "border-line")}
                style={{ background: c.good ? `rgb(57 135 229 / ${0.06 + share * 0.3})` : `rgb(230 103 103 / ${0.04 + share * 0.6})` }}
              >
                <div className="font-mono text-xl font-medium tabular-nums text-ink">{num(c.v)}</div>
                <div className="mt-0.5 font-mono text-[11px] text-ink-2">{pct(share)}</div>
                <div className="label mt-1.5 !text-[9.5px]">{c.label}</div>
              </div>
            );
          })}
        </div>
      ))}
    </div>
  );
}

function MetricRow({ name, m, highlight }: { name: string; m: Metrics; highlight?: boolean }) {
  return (
    <tr className={clsx("border-b border-line/60 last:border-0", highlight && "bg-surface-2/60")}>
      <td className="px-5 py-2.5 text-[13px] text-ink">
        {name}
        {highlight && <span className="ml-2 rounded border border-line-strong px-1 py-0.5 font-mono text-[9.5px] uppercase tracking-wider text-ink-2">deployed</span>}
      </td>
      {[m.accuracy, m.precision, m.recall, m.f1, m.roc_auc, m.false_positive_rate].map((v, i) => (
        <td key={i} className="px-3 py-2.5 text-right font-mono text-[12.5px] tabular-nums text-ink-2">
          {v === null ? "—" : i === 4 ? v.toFixed(4) : pct(v)}
        </td>
      ))}
    </tr>
  );
}

const HEAD = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "FPR"];

export default function ModelPage() {
  const [card, setCard] = useState<ModelCard | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.model().then(setCard).catch((e) => setError(e.message));
  }, []);

  if (error) return <ErrorBanner message={error} />;
  if (!card)
    return (
      <div className="space-y-4">
        <Skeleton className="h-16 w-1/2" />
        <div className="grid gap-4 sm:grid-cols-5">{Array.from({ length: 5 }, (_, i) => <Skeleton key={i} className="h-24" />)}</div>
        <Skeleton className="h-72" />
      </div>
    );

  const deployed = card.models[card.name];
  const t = deployed.test_at_tuned_threshold ?? deployed.test;
  const roc = [{ x: 0, y: 0 }, ...card.curves.roc].map((p) => ({ fpr: p.x, tpr: p.y, chance: p.x }));
  const fi = card.feature_importance.slice(0, 12);
  const dist = card.score_distribution.map((d) => ({ ...d, label: d.bin.toFixed(2) }));

  return (
    <div>
      <PageHeader
        eyebrow="Model card"
        title={MODEL_LABELS[card.name] ?? card.name}
        description={
          <>
            Every number on this page was computed by the training pipeline on a held-out test split of{" "}
            <span className="font-mono text-ink">{num(card.splits.test.rows)}</span> URLs from{" "}
            <span className="font-mono text-ink">{num(card.splits.test.domains)}</span> domains. No test domain appears in training.
          </>
        }
        action={
          <div className="flex flex-wrap gap-2 font-mono text-[11px] text-ink-2">
            <span className="rounded-md border border-line bg-surface px-2 py-1">v{card.version}</span>
            <span className="rounded-md border border-line bg-surface px-2 py-1">threshold {card.threshold.toFixed(3)}</span>
            <span className="rounded-md border border-line bg-surface px-2 py-1">trained {card.trained_at.slice(0, 10)}</span>
          </div>
        }
      />

      <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-5">
        <StatTile label="Accuracy" value={pct(t.accuracy)} />
        <StatTile label="Precision" value={pct(t.precision)} sub="flagged URLs that were phishing" />
        <StatTile label="Recall" value={pct(t.recall)} sub="phishing URLs caught" />
        <StatTile label="F1-score" value={pct(t.f1)} />
        <StatTile label="ROC-AUC" value={t.roc_auc?.toFixed(4) ?? "—"} sub="threshold-independent" />
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-[1fr_1.2fr]">
        <Panel title="Confusion matrix" subtitle={`Test split at the tuned threshold (${t.threshold.toFixed(3)})`}>
          <ConfusionMatrix m={t} />
          <p className="mt-4 text-xs leading-relaxed text-ink-3">
            False-positive rate <span className="font-mono text-ink-2">{pct(t.false_positive_rate, 2)}</span> · false-negative rate{" "}
            <span className="font-mono text-ink-2">{pct(t.false_negative_rate, 2)}</span>. Percentages are per actual class (row).
          </p>
        </Panel>
        <Panel title="ROC curve" subtitle={`True-positive vs false-positive rate · AUC ${t.roc_auc?.toFixed(4)}`}>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={roc} margin={{ top: 4, right: 18, left: -12, bottom: 4 }}>
                <CartesianGrid stroke="var(--line)" />
                <XAxis dataKey="fpr" type="number" domain={[0, 1]} ticks={[0, 0.2, 0.4, 0.6, 0.8, 1]} {...AXIS} tickFormatter={(v: number) => v.toFixed(1)} label={{ value: "False positive rate", position: "insideBottom", offset: -2, fill: "var(--text-3)", fontSize: 11 }} height={36} />
                <YAxis type="number" domain={[0, 1]} ticks={[0, 0.2, 0.4, 0.6, 0.8, 1]} {...AXIS} tickFormatter={(v: number) => v.toFixed(1)} />
                <Tooltip content={<ChartTooltip formatter={(v) => Number(v).toFixed(3)} />} labelFormatter={(v) => `FPR ${Number(v).toFixed(3)}`} />
                <Area type="monotone" dataKey="tpr" name="True positive rate" stroke="var(--series-legit)" strokeWidth={2} fill="var(--series-legit)" fillOpacity={0.12} isAnimationActive={false} />
                <Line type="linear" dataKey="chance" name="Chance" stroke="var(--text-3)" strokeDasharray="4 4" dot={false} strokeWidth={1} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </Panel>
      </div>

      <Panel className="mt-4" title="Model comparison" subtitle="All candidates trained on the same split. Selection used validation ROC-AUC only; these are test results at a 0.5 threshold." bodyClassName="p-0">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[720px]">
            <thead>
              <tr className="border-b border-line">
                <th className="label px-5 py-2.5 text-left font-normal">Model</th>
                {HEAD.map((h) => <th key={h} className="label px-3 py-2.5 text-right font-normal">{h}</th>)}
              </tr>
            </thead>
            <tbody>
              {Object.entries(card.models).map(([name, m]) => (
                <MetricRow key={name} name={`${MODEL_LABELS[name] ?? name}${m.explainable ? "" : " (reference, opaque)"}`} m={m.test} highlight={name === card.name} />
              ))}
            </tbody>
          </table>
        </div>
      </Panel>

      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        <Panel title="Permutation feature importance" subtitle="Drop in validation ROC-AUC when a feature is shuffled">
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart layout="vertical" data={fi} margin={{ top: 0, right: 12, left: 0, bottom: 0 }}>
                <CartesianGrid horizontal={false} stroke="var(--line)" />
                <XAxis type="number" {...AXIS} tickFormatter={(v: number) => v.toFixed(3)} />
                <YAxis type="category" dataKey="feature" {...AXIS} axisLine={false} width={170} tick={{ ...AXIS.tick, fill: "var(--text-2)" }} />
                <Tooltip cursor={{ fill: "rgb(255 255 255 / 0.03)" }} content={<ChartTooltip formatter={(v) => Number(v).toFixed(4)} />} />
                <Bar dataKey="importance" name="Δ ROC-AUC" fill="var(--series-legit)" radius={[0, 4, 4, 0]} barSize={12} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Panel>
        <Panel title="Score distribution" subtitle="Test URLs per predicted-probability bin, by true class">
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={dist} margin={{ top: 4, right: 4, left: 0, bottom: 0 }}>
                <CartesianGrid vertical={false} stroke="var(--line)" />
                <XAxis dataKey="label" {...AXIS} interval={3} />
                <YAxis {...AXIS} axisLine={false} scale="sqrt" tickFormatter={(v: number) => (v >= 1000 ? `${(v / 1000).toFixed(0)}k` : String(v))} />
                <Tooltip cursor={{ fill: "rgb(255 255 255 / 0.03)" }} content={<ChartTooltip formatter={(v) => num(Number(v))} />} />
                <Legend iconType="square" iconSize={8} wrapperStyle={{ fontSize: 12 }} />
                <Bar dataKey="legitimate" name="Legitimate" fill="var(--series-legit)" radius={[3, 3, 0, 0]} />
                <Bar dataKey="phishing" name="Phishing" fill="var(--series-phish)" radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="mt-2 text-xs text-ink-3">Square-root y-axis so both tails stay visible.</p>
        </Panel>
      </div>

      <Panel className="mt-4" title="Robustness" subtitle="Where the model holds up and where it does not" bodyClassName="p-0">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[720px]">
            <thead>
              <tr className="border-b border-line">
                <th className="label px-5 py-2.5 text-left font-normal">Slice / experiment</th>
                {HEAD.map((h) => <th key={h} className="label px-3 py-2.5 text-right font-normal">{h}</th>)}
              </tr>
            </thead>
            <tbody>
              {Object.entries(card.per_source_test).map(([src, m]) => (
                <MetricRow key={src} name={`Test slice: ${SOURCE_LABELS[src] ?? src}`} m={m} />
              ))}
              {Object.entries(card.experiments).map(([k, e]) => (
                <MetricRow key={k} name={e.description} m={e.overall} />
              ))}
            </tbody>
          </table>
        </div>
        <p className="border-t border-line px-5 py-3 text-xs leading-relaxed text-ink-3">
          The legitimate-only slice has no phishing rows, so precision, recall and AUC do not apply there; its accuracy equals 1 − FPR.
          The cross-dataset rows show how much performance drops when the model sees a URL distribution it was not trained on.
        </p>
      </Panel>

      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        <Panel title={<span className="flex items-center gap-2"><Database className="h-4 w-4 text-ink-3" />Training data</span>}>
          <ul className="space-y-3">
            {card.datasets.map((d) => (
              <li key={d.name} className="text-[13px]">
                <div className="font-medium text-ink">{SOURCE_LABELS[d.name] ?? d.name}</div>
                <div className="mt-0.5 text-xs leading-relaxed text-ink-3">{d.citation}</div>
                <div className="mt-1 font-mono text-[10.5px] text-ink-3">
                  {d.license} · sha256 {d.sha256.slice(0, 16)}…
                </div>
              </li>
            ))}
          </ul>
          <dl className="mt-4 grid grid-cols-3 gap-3 border-t border-line pt-4">
            {(["train", "validation", "test"] as const).map((s) => (
              <div key={s}>
                <dt className="label">{s}</dt>
                <dd className="mt-1 font-mono text-sm tabular-nums">{num(card.splits[s].rows)}</dd>
                <dd className="font-mono text-[10.5px] text-ink-3">{num(card.splits[s].domains)} domains</dd>
              </div>
            ))}
          </dl>
          <p className="mt-3 text-xs text-ink-3">{card.split_strategy}.</p>
        </Panel>
        <Panel title={<span className="flex items-center gap-2"><AlertTriangle className="h-4 w-4 text-medium" />Known limitations</span>}>
          <ul className="space-y-2.5 text-[13px] leading-relaxed text-ink-2">
            <li>Lexical only: the page is never fetched, so a clean-looking URL on a compromised legitimate site will score low.</li>
            <li>Phishing on bare domains with no path (common in PhiUSIIL) is hard to tell apart from small legitimate sites, which limits recall.</li>
            <li>Legitimate URLs with long random tokens, many subdomains or uncommon TLDs can be false positives.</li>
            <li>The training data is a 2019–2026 snapshot. Attacker tactics drift, so the model needs periodic retraining.</li>
            <li>Explanations measure each feature on its own; correlated features can share or hide credit.</li>
          </ul>
          <div className="mt-4 flex items-center gap-2 rounded-lg border border-line bg-surface-2/60 px-3 py-2 text-xs text-ink-3">
            <FlaskConical className="h-3.5 w-3.5" />
            {card.environment && `scikit-learn ${card.environment.scikit_learn} · python ${card.environment.python}`}
          </div>
        </Panel>
      </div>
    </div>
  );
}
