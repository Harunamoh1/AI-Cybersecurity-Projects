"use client";

import { AlertTriangle, ArrowLeft, CheckCircle2, Info } from "lucide-react";
import Link from "next/link";
import { use, useState } from "react";
import { BarList } from "@/components/charts";
import { FlowTable } from "@/components/FlowTable";
import { Card, ErrorState, Loading, PageHeader, Pagination, Select, Stat, useApi } from "@/components/ui";
import { api, CLASSES, type Analysis, type ClassName } from "@/lib/api";
import { CLASS_META, dateTime, featureLabel, int, pct, SEVERITY_META } from "@/lib/format";

function ShiftNotice({ a }: { a: Analysis }) {
  const s = a.summary.distribution_shift;
  if (!s.level || s.level === "low") return null;
  const high = s.level === "high";
  return (
    <div className={`mb-4 flex gap-3 rounded-lg border px-4 py-3 text-sm ${high ? "border-high/50 bg-high/5" : "border-medium/40 bg-medium/5"}`} role="status">
      <AlertTriangle className={`mt-0.5 h-4 w-4 shrink-0 ${high ? "text-high" : "text-medium"}`} />
      <div className="text-ink-2">
        <span className="font-medium text-ink">{high ? "Strong distribution shift." : "Moderate distribution shift."}</span>{" "}
        {pct(s.out_of_range_rate, 2)} of feature values (up to {pct(s.max_feature_rate ?? 0, 1)} for{" "}
        <span className="font-mono">{featureLabel(s.features[0]?.feature ?? "")}</span>) fall outside the range seen in CIC-IDS2017 training data.
        This traffic likely comes from a different network or CICFlowMeter version; treat classifications with caution — cross-dataset
        evaluation shows accuracy drops sharply in this situation (see Model Performance).
      </div>
    </div>
  );
}

function GroundTruth({ a }: { a: Analysis }) {
  const g = a.summary.ground_truth;
  if (!g) return null;
  return (
    <Card title="Comparison with supplied labels" subtitle={`${int(g.labelled_flows)} flows carried a recognised Label value`}>
      <div className="grid grid-cols-2 gap-x-6 gap-y-2 text-[13px] sm:grid-cols-4">
        {[["Accuracy", pct(g.accuracy, 2)], ["Attack recall", pct(g.attack_recall, 2)], ["Attack precision", pct(g.attack_precision, 2)],
          ["False-positive rate", pct(g.false_positive_rate, 2)]].map(([k, v]) => (
          <div key={k}><div className="text-xs text-ink-3">{k}</div><div className="tnum font-mono text-ink">{v}</div></div>
        ))}
      </div>
      <div className="mt-4 grid grid-cols-2 gap-2 sm:grid-cols-4">
        {Object.entries(g.per_class_recall).map(([c, r]) => (
          <div key={c} className="rounded border border-line px-2.5 py-1.5 text-xs">
            <div className="flex items-center gap-1.5 text-ink-2">
              <span className="h-2 w-2 rounded-[2px]" style={{ background: CLASS_META[c as ClassName].color }} />{CLASS_META[c as ClassName].short}
            </div>
            <div className="tnum mt-0.5 font-mono text-ink">recall {pct(r, 1)}</div>
          </div>
        ))}
      </div>
    </Card>
  );
}

export default function AnalysisPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const [cls, setCls] = useState("");
  const [view, setView] = useState("threats");
  const [offset, setOffset] = useState(0);
  const analysis = useApi(() => api.analysis(id), [id]);
  const flows = useApi(() => api.flows(id, {
    cls: cls || undefined, threats_only: view === "threats", order: view === "threats" ? "severity" : "index", limit: 50, offset,
  }), [id, cls, view, offset]);

  if (analysis.error) return <ErrorState message={analysis.error} />;
  if (!analysis.data) return <Loading />;
  const a = analysis.data;
  const s = a.summary;
  const benign = s.class_counts.BENIGN ?? 0;

  return (
    <>
      <Link href="/analyze" className="mb-3 inline-flex items-center gap-1 text-xs text-ink-3 hover:text-ink"><ArrowLeft className="h-3 w-3" /> All analyses</Link>
      <PageHeader title={a.filename}
        description={<>Analysed {dateTime(a.created_at)} · {int(a.total_flows)} flows in {(a.duration_ms / 1000).toFixed(2)} s · model v{a.model_version}</>} />

      <ShiftNotice a={a} />

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-5">
        <Stat label="Flows" value={int(a.total_flows)} sub={a.rejected_rows ? `${int(a.rejected_rows)} rows rejected` : "no rows rejected"} />
        <Stat label="Threats" value={int(a.threats)} sub={`${pct(a.threats / Math.max(a.total_flows, 1))} of flows`} />
        <Stat label="Critical" value={int(s.severity_counts.critical)} tone={s.severity_counts.critical ? SEVERITY_META.critical.color : undefined}
          sub={`${int(s.severity_counts.high)} high · ${int(s.severity_counts.medium)} medium · ${int(s.severity_counts.low)} low`} />
        <Stat label="Mean confidence" value={pct(s.mean_confidence)} sub={`${int(s.confidence_bands.low ?? 0)} low-confidence flows`} />
        <Stat label="Imputed values" value={int(s.flows_with_imputed_values)} sub="flows with missing/invalid cells" />
      </div>

      <div className="mt-4 grid gap-4 xl:grid-cols-3">
        <Card title="Classification breakdown" subtitle={`${int(benign)} benign · ${int(a.threats)} threat flows`}>
          <BarList total={a.total_flows}
            items={CLASSES.map((c) => ({ key: c, label: CLASS_META[c].label, value: s.class_counts[c] ?? 0, color: CLASS_META[c].color }))
              .filter((i) => i.value > 0).sort((x, y) => y.value - x.value)} />
        </Card>
        <div className="space-y-4 xl:col-span-2">
          <GroundTruth a={a} />
          <Card title="Input report">
            <ul className="space-y-1.5 text-[13px] text-ink-2">
              <li className="flex gap-2"><CheckCircle2 className="mt-0.5 h-3.5 w-3.5 shrink-0 text-[var(--c-bot)]" />{s.columns.mapped} flow-feature columns recognised</li>
              {s.columns.identifiers_dropped.length > 0 && (
                <li className="flex gap-2"><Info className="mt-0.5 h-3.5 w-3.5 shrink-0 text-ink-3" />Dropped identifier columns (not stored): <span className="font-mono text-xs">{s.columns.identifiers_dropped.join(", ")}</span></li>
              )}
              {s.columns.unexpected_columns.length > 0 && (
                <li className="flex gap-2"><Info className="mt-0.5 h-3.5 w-3.5 shrink-0 text-ink-3" />Ignored unrecognised columns: <span className="font-mono text-xs">{s.columns.unexpected_columns.slice(0, 8).join(", ")}</span></li>
              )}
              <li className="flex gap-2"><Info className="mt-0.5 h-3.5 w-3.5 shrink-0 text-ink-3" />
                Distribution shift vs. training data: <span className="text-ink">{s.distribution_shift.level ?? "low"}</span>
                <span className="font-mono text-xs text-ink-3">({pct(s.distribution_shift.out_of_range_rate, 2)} of values out of range)</span>
              </li>
            </ul>
          </Card>
        </div>
      </div>

      <Card title="Flows" className="mt-4" bodyClass="p-0"
        actions={
          <div className="flex flex-wrap items-center gap-3">
            <Select label="Show" value={view} onChange={(v) => { setView(v); setOffset(0); }}>
              <option value="threats">Threats (by severity)</option>
              <option value="all">All flows (file order)</option>
            </Select>
            <Select label="Class" value={cls} onChange={(v) => { setCls(v); setOffset(0); }}>
              <option value="">All</option>
              {CLASSES.map((c) => <option key={c} value={c}>{CLASS_META[c].short}</option>)}
            </Select>
          </div>
        }>
        {flows.error ? <div className="p-4"><ErrorState message={flows.error} /></div>
          : !flows.data ? <div className="px-4"><Loading /></div>
          : flows.data.items.length === 0 ? <p className="p-4 text-sm text-ink-3">No flows match.</p>
          : (
            <>
              <FlowTable flows={flows.data.items} showTruth={!!s.ground_truth} />
              <Pagination total={flows.data.total} limit={50} offset={offset} onChange={setOffset} />
            </>
          )}
      </Card>
    </>
  );
}
