"use client";

import { useState } from "react";
import { FlowTable } from "@/components/FlowTable";
import { Card, ClassBadge, Empty, ErrorState, Loading, PageHeader, Pagination, Select, SeverityGlyph, useApi } from "@/components/ui";
import { api } from "@/lib/api";
import { ATTACK_CLASSES, CLASS_META, int, pct, SEVERITY_META } from "@/lib/format";

const CLASS_INFO: Record<string, string> = {
  DoS: "Single-source floods and slow-rate exhaustion (Hulk, GoldenEye, Slowloris, SlowHTTPTest).",
  DDoS: "Multi-source volumetric HTTP flood (LOIC).",
  PortScan: "Reconnaissance probing many ports (Nmap scan types).",
  BruteForce: "Credential guessing against FTP and SSH (Patator).",
  WebAttack: "Web brute force, XSS and SQL injection.",
  Bot: "Command-and-control traffic of an infected host (Ares).",
  Infiltration: "Post-exploitation activity from an internal compromised host.",
};

export default function ThreatsPage() {
  const [cls, setCls] = useState("");
  const [severity, setSeverity] = useState("");
  const [minConf, setMinConf] = useState("");
  const [order, setOrder] = useState("recent");
  const [offset, setOffset] = useState(0);
  const stats = useApi(() => api.stats(30));
  const threats = useApi(() => api.threats({
    cls: cls || undefined, severity: severity || undefined, min_confidence: minConf ? Number(minConf) : undefined, order, limit: 50, offset,
  }), [cls, severity, minConf, order, offset]);
  const reset = (fn: (v: string) => void) => (v: string) => { fn(v); setOffset(0); };

  return (
    <>
      <PageHeader title="Threat intelligence"
        description="Every flow the model classified as malicious, across all analyses. Severity is the attack class's base impact, lowered one level when calibrated confidence is below 60%." />

      {stats.data && (
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
          {ATTACK_CLASSES.filter((c) => stats.data!.by_class[c]?.count).map((c) => {
            const b = stats.data!.by_class[c]!;
            return (
              <button key={c} onClick={() => reset(setCls)(cls === c ? "" : c)} aria-pressed={cls === c}
                className={`rounded-lg border bg-surface px-4 py-3 text-left transition-colors hover:bg-surface-2 ${cls === c ? "border-accent" : "border-line"}`}>
                <div className="flex items-center justify-between">
                  <ClassBadge cls={c} />
                  <span className="tnum font-mono text-sm text-ink">{int(b.count)}</span>
                </div>
                <p className="mt-1.5 text-xs leading-relaxed text-ink-3">{CLASS_INFO[c]}</p>
                <p className="tnum mt-1.5 font-mono text-[11px] text-ink-3">mean confidence {pct(b.mean_confidence)}</p>
              </button>
            );
          })}
        </div>
      )}

      <Card title="Suspicious flows" className="mt-4" bodyClass="p-0"
        subtitle={threats.data ? `${int(threats.data.total)} matching flows` : undefined}
        actions={
          <div className="flex flex-wrap items-center gap-3">
            <Select label="Class" value={cls} onChange={reset(setCls)}>
              <option value="">All</option>
              {ATTACK_CLASSES.map((c) => <option key={c} value={c}>{CLASS_META[c].short}</option>)}
            </Select>
            <Select label="Severity" value={severity} onChange={reset(setSeverity)}>
              <option value="">All</option>
              {(["critical", "high", "medium", "low"] as const).map((s) => <option key={s} value={s}>{SEVERITY_META[s].label}</option>)}
            </Select>
            <Select label="Min. confidence" value={minConf} onChange={reset(setMinConf)}>
              <option value="">Any</option>
              <option value="0.6">60%</option>
              <option value="0.9">90%</option>
              <option value="0.99">99%</option>
            </Select>
            <Select label="Sort" value={order} onChange={reset(setOrder)}>
              <option value="recent">Most recent</option>
              <option value="severity">Severity</option>
            </Select>
          </div>
        }>
        {threats.error ? <div className="p-4"><ErrorState message={threats.error} onRetry={threats.reload} /></div>
          : !threats.data ? <div className="px-4"><Loading /></div>
          : threats.data.items.length === 0 ? <div className="p-4"><Empty title="No suspicious flows">Nothing matches these filters.</Empty></div>
          : (
            <>
              <FlowTable flows={threats.data.items} showSource />
              <Pagination total={threats.data.total} limit={50} offset={offset} onChange={setOffset} />
            </>
          )}
      </Card>

      <p className="mt-4 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-ink-3">
        Severity scale:
        {(["critical", "high", "medium", "low"] as const).map((s) => (
          <span key={s} className="flex items-center gap-1.5"><SeverityGlyph severity={s} />{SEVERITY_META[s].label}</span>
        ))}
      </p>
    </>
  );
}
