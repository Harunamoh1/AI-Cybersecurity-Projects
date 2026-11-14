"use client";

import { BarChart3 } from "lucide-react";
import Link from "next/link";
import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ChartTooltip, EmptyState, ErrorBanner, PageHeader, Panel, Skeleton, StatTile } from "@/components/ui";
import { api, type RiskLevel, type Stats } from "@/lib/api";
import { INDICATOR_LABELS, RISK_META, num, pct } from "@/lib/format";

const AXIS = { stroke: "var(--line-strong)", tick: { fill: "var(--text-3)", fontSize: 11, fontFamily: "var(--font-geist-mono)" }, tickLine: false };
const RANGES = [7, 14, 30];

export default function AnalyticsPage() {
  const [stats, setStats] = useState<Stats | null>(null);
  const [days, setDays] = useState(14);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.stats(days).then(setStats).catch((e) => setError(e.message));
  }, [days]);

  const rangePicker = (
    <div className="inline-flex rounded-lg border border-line bg-surface p-0.5">
      {RANGES.map((d) => (
        <button
          key={d}
          onClick={() => setDays(d)}
          aria-pressed={days === d}
          className={`rounded-md px-3 py-1.5 font-mono text-xs transition-colors ${days === d ? "bg-surface-3 text-ink" : "text-ink-3 hover:text-ink-2"}`}
        >
          {d}d
        </button>
      ))}
    </div>
  );

  if (error) return <ErrorBanner message={error} />;

  return (
    <div>
      <PageHeader
        eyebrow="Threat overview"
        title="Analytics"
        description="Aggregates over every URL analysed in this workspace. Numbers update as new scans are recorded."
        action={rangePicker}
      />

      {!stats ? (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {Array.from({ length: 4 }, (_, i) => <Skeleton key={i} className="h-[104px]" />)}
        </div>
      ) : stats.total_scans === 0 ? (
        <div className="rounded-xl border border-line bg-surface">
          <EmptyState icon={BarChart3} title="No data yet">
            Run a few scans from the <Link href="/" className="text-ink underline">scanner</Link> and the dashboard will populate.
          </EmptyState>
        </div>
      ) : (
        <div className="space-y-4">
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <StatTile label="Total scans" value={num(stats.total_scans)} sub="all time" />
            <StatTile label="Flagged phishing" value={num(stats.phishing)} sub={`${num(stats.legitimate)} legitimate`} accent="var(--series-phish)" />
            <StatTile label="Flag rate" value={pct(stats.detection_rate)} sub="share of scans above threshold" />
            <StatTile label="Mean risk score" value={stats.average_risk_score.toFixed(1)} sub="0 – 100 scale" />
          </div>

          <div className="grid gap-4 lg:grid-cols-3">
            <Panel className="lg:col-span-2" title="Scan volume" subtitle={`Daily scans by verdict, last ${days} days`}>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={stats.timeline} margin={{ top: 4, right: 4, left: -18, bottom: 0 }}>
                    <CartesianGrid vertical={false} stroke="var(--line)" />
                    <XAxis dataKey="date" {...AXIS} tickFormatter={(d: string) => d.slice(5)} minTickGap={16} />
                    <YAxis {...AXIS} allowDecimals={false} axisLine={false} />
                    <Tooltip cursor={{ fill: "rgb(255 255 255 / 0.03)" }} content={<ChartTooltip />} />
                    <Legend iconType="square" iconSize={8} wrapperStyle={{ fontSize: 12, color: "var(--text-2)" }} />
                    <Bar dataKey="legitimate" name="Legitimate" stackId="v" fill="var(--series-legit)" stroke="var(--surface)" strokeWidth={2} maxBarSize={28} />
                    <Bar dataKey="phishing" name="Phishing" stackId="v" fill="var(--series-phish)" stroke="var(--surface)" strokeWidth={2} radius={[4, 4, 0, 0]} maxBarSize={28} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </Panel>

            <Panel title="Risk distribution" subtitle="Scans per risk band">
              <ul className="space-y-4">
                {(Object.keys(RISK_META) as RiskLevel[]).map((lvl) => {
                  const n = stats.by_risk_level[lvl] ?? 0;
                  const share = stats.total_scans ? n / stats.total_scans : 0;
                  return (
                    <li key={lvl}>
                      <div className="mb-1.5 flex items-baseline justify-between text-[13px]">
                        <span className="flex items-center gap-2 text-ink">
                          <span className="h-2 w-2 rounded-sm" style={{ background: RISK_META[lvl].color }} />
                          {RISK_META[lvl].label}
                        </span>
                        <span className="font-mono text-xs tabular-nums text-ink-2">
                          {num(n)} <span className="text-ink-3">· {pct(share, 0)}</span>
                        </span>
                      </div>
                      <div className="h-1.5 rounded-full bg-surface-2">
                        <div className="h-full rounded-full" style={{ width: `${share * 100}%`, background: RISK_META[lvl].color }} />
                      </div>
                    </li>
                  );
                })}
              </ul>
            </Panel>
          </div>

          <div className="grid gap-4 lg:grid-cols-3">
            <Panel title="Risk score histogram" subtitle="Number of scans per 10-point bin">
              <div className="h-56">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={stats.score_histogram} margin={{ top: 4, right: 4, left: -24, bottom: 0 }}>
                    <CartesianGrid vertical={false} stroke="var(--line)" />
                    <XAxis dataKey="range" {...AXIS} interval={1} tickFormatter={(r: string) => r.split("-")[0]} />
                    <YAxis {...AXIS} allowDecimals={false} axisLine={false} />
                    <Tooltip cursor={{ fill: "rgb(255 255 255 / 0.03)" }} content={<ChartTooltip />} />
                    <Bar dataKey="count" name="Scans" fill="var(--series-legit)" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </Panel>

            <Panel title="Indicators on flagged URLs" subtitle="Rule-based observations among phishing verdicts">
              {stats.top_indicators.length ? (
                <div className="h-56">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      layout="vertical"
                      data={stats.top_indicators.map((d) => ({ ...d, label: INDICATOR_LABELS[d.kind] ?? d.kind }))}
                      margin={{ top: 0, right: 12, left: 0, bottom: 0 }}
                    >
                      <XAxis type="number" hide allowDecimals={false} />
                      <YAxis type="category" dataKey="label" {...AXIS} axisLine={false} width={128} tick={{ ...AXIS.tick, fill: "var(--text-2)" }} />
                      <Tooltip cursor={{ fill: "rgb(255 255 255 / 0.03)" }} content={<ChartTooltip />} />
                      <Bar dataKey="count" name="Occurrences" fill="var(--series-phish)" radius={[0, 4, 4, 0]} barSize={14} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                <p className="text-sm text-ink-3">No phishing verdicts yet.</p>
              )}
            </Panel>

            <Panel title="Top-level domains" subtitle="Most-scanned TLDs and their flagged share" bodyClassName="p-0">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-line text-left">
                    <th className="label px-5 py-2.5 font-normal">TLD</th>
                    <th className="label px-3 py-2.5 text-right font-normal">Scans</th>
                    <th className="label px-5 py-2.5 font-normal">Flagged</th>
                  </tr>
                </thead>
                <tbody>
                  {stats.top_tlds.map((t) => (
                    <tr key={t.tld} className="border-b border-line/60 last:border-0">
                      <td className="px-5 py-2 font-mono text-[12.5px] text-ink">.{t.tld}</td>
                      <td className="px-3 py-2 text-right font-mono tabular-nums text-ink-2">{t.scans}</td>
                      <td className="px-5 py-2">
                        <div className="flex items-center gap-2">
                          <div className="h-1.5 flex-1 rounded-full bg-surface-2">
                            <div className="h-full rounded-full bg-phish" style={{ width: `${(t.phishing / t.scans) * 100}%` }} />
                          </div>
                          <span className="w-9 text-right font-mono text-[11px] tabular-nums text-ink-3">{pct(t.phishing / t.scans, 0)}</span>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Panel>
          </div>
        </div>
      )}
    </div>
  );
}
