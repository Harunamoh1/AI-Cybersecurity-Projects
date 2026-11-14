"use client";

import Link from "next/link";
import { BarList, DetectionTrend } from "@/components/charts";
import { FlowTable } from "@/components/FlowTable";
import { Card, ClassBadge, Empty, ErrorState, Loading, PageHeader, SeverityGlyph, Stat, Td, Th, useApi } from "@/components/ui";
import { api, type SupervisedResult } from "@/lib/api";
import { ATTACK_CLASSES, CLASS_META, dec, int, pct, relative, SEVERITY_META } from "@/lib/format";

function ModelSummary() {
  const { data } = useApi(api.model);
  if (!data) return <Loading />;
  const m = data.models[data.deployed_model] as SupervisedResult;
  const rows = [
    ["Macro F1 (8 classes)", dec(m.test.macro_f1)],
    ["Accuracy", pct(m.test.accuracy, 2)],
    ["Attack recall", pct(m.binary_test.recall, 2)],
    ["False-positive rate", pct(m.binary_test.false_positive_rate, 3)],
    ["Calibration error (ECE)", dec(data.calibration.after.ece, 4)],
  ];
  return (
    <div>
      <dl className="divide-y divide-line">
        {rows.map(([k, v]) => (
          <div key={k} className="flex items-center justify-between py-2 text-[13px]">
            <dt className="text-ink-2">{k}</dt>
            <dd className="tnum font-mono text-ink">{v}</dd>
          </div>
        ))}
      </dl>
      <p className="mt-3 text-xs leading-relaxed text-ink-3">
        XGBoost, held-out CIC-IDS2017 test split ({int(Object.values(data.dataset.splits.test).reduce((a, b) => a + b, 0))} flows).{" "}
        <Link href="/model" className="text-accent hover:underline">Full evaluation</Link>
      </p>
    </div>
  );
}

export default function Overview() {
  const stats = useApi(() => api.stats(14));
  const recent = useApi(() => api.analyses(6));
  const threats = useApi(() => api.threats({ order: "severity", limit: 6 }));

  if (stats.error) return <ErrorState message={stats.error} onRetry={stats.reload} />;
  if (!stats.data) return <Loading />;
  const s = stats.data;
  const attackTotal = ATTACK_CLASSES.reduce((a, c) => a + (s.by_class[c]?.count ?? 0), 0);

  return (
    <>
      <PageHeader title="Security overview"
        description="Aggregated detections across every analysed traffic capture. Classifications come from the deployed XGBoost model; confidences are temperature-calibrated." />

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-5">
        <Stat label="Flows analysed" value={int(s.total_flows)} sub={`${int(s.analyses)} analyses`} />
        <Stat label="Threats detected" value={int(s.threats)} sub="flows classified as an attack" />
        <Stat label="Threat rate" value={pct(s.threat_rate)} sub="of analysed flows" />
        <Stat label="Critical alerts" value={int(s.by_severity.critical ?? 0)} tone={s.by_severity.critical ? "var(--sev-critical)" : undefined}
          sub={`${int(s.by_severity.high ?? 0)} high · ${int(s.by_severity.medium ?? 0)} medium`} />
        <Stat label="Label agreement" value={s.labelled_agreement == null ? "—" : pct(s.labelled_agreement)}
          sub={s.labelled_flows ? `on ${int(s.labelled_flows)} labelled flows` : "no labelled uploads yet"} />
      </div>

      {s.total_flows === 0 ? (
        <div className="mt-6">
          <Empty title="No traffic analysed yet">
            Upload a CICFlowMeter CSV or run one of the bundled held-out samples on the{" "}
            <Link href="/analyze" className="text-accent hover:underline">Traffic Analysis</Link> page.
          </Empty>
        </div>
      ) : (
        <>
          <div className="mt-4 grid gap-4 xl:grid-cols-3">
            <Card title="Detection trend" subtitle="Threat flows per day by attack class (last 14 days)" className="xl:col-span-2">
              <DetectionTrend data={s.timeline} />
            </Card>
            <Card title="Threat distribution" subtitle={`${int(attackTotal)} threat flows`}>
              {attackTotal ? (
                <BarList total={attackTotal}
                  items={ATTACK_CLASSES.map((c) => ({ key: c, label: CLASS_META[c].label, value: s.by_class[c]?.count ?? 0, color: CLASS_META[c].color }))
                    .filter((i) => i.value > 0).sort((a, b) => b.value - a.value)} />
              ) : <p className="text-sm text-ink-3">No threats detected.</p>}
            </Card>
          </div>

          <div className="mt-4 grid gap-4 xl:grid-cols-3">
            <Card title="Highest-severity detections" subtitle="Across all analyses" className="xl:col-span-2" bodyClass="p-0"
              actions={<Link href="/threats" className="text-xs text-accent hover:underline">All threats</Link>}>
              {threats.data?.items.length ? <FlowTable flows={threats.data.items} showSource /> : <div className="p-4 text-sm text-ink-3">No threats.</div>}
            </Card>
            <div className="space-y-4">
              <Card title="Severity">
                <ul className="space-y-2">
                  {(["critical", "high", "medium", "low"] as const).map((k) => (
                    <li key={k} className="flex items-center justify-between text-[13px]">
                      <span className="flex items-center gap-2 text-ink-2"><SeverityGlyph severity={k} />{SEVERITY_META[k].label}</span>
                      <span className="tnum font-mono text-ink">{int(s.by_severity[k] ?? 0)}</span>
                    </li>
                  ))}
                </ul>
              </Card>
              <Card title="Model performance" subtitle="Offline evaluation">
                <ModelSummary />
              </Card>
            </div>
          </div>

          <Card title="Recent analyses" className="mt-4" bodyClass="p-0"
            actions={<Link href="/analyze" className="text-xs text-accent hover:underline">Analyse traffic</Link>}>
            <div className="overflow-x-auto">
              <table className="w-full min-w-[640px]">
                <thead className="border-b border-line"><tr><Th>File</Th><Th>When</Th><Th className="text-right">Flows</Th><Th className="text-right">Threats</Th><Th>Dominant threat</Th></tr></thead>
                <tbody className="divide-y divide-line">
                  {recent.data?.items.map((a) => {
                    const top = ATTACK_CLASSES.map((c) => [c, a.summary.class_counts[c] ?? 0] as const).sort((x, y) => y[1] - x[1])[0];
                    return (
                      <tr key={a.id} className="hover:bg-surface-2">
                        <Td><Link href={`/analyze/${a.id}`} className="text-ink hover:text-accent">{a.filename}</Link></Td>
                        <Td>{relative(a.created_at)}</Td>
                        <Td className="tnum text-right font-mono">{int(a.total_flows)}</Td>
                        <Td className="tnum text-right font-mono">{int(a.threats)}</Td>
                        <Td>{top && top[1] > 0 ? <ClassBadge cls={top[0]} /> : <span className="text-ink-3">—</span>}</Td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}
    </>
  );
}
