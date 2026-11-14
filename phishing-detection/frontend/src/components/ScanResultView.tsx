"use client";

import clsx from "clsx";
import { ChevronDown, Fingerprint, ShieldCheck } from "lucide-react";
import { useState } from "react";
import type { ScanResult } from "@/lib/api";
import { INDICATOR_LABELS, RISK_META, formatFeatureValue } from "@/lib/format";
import { Contributions } from "./Contributions";
import { RiskGauge } from "./RiskGauge";
import { Panel, RiskBadge } from "./ui";
import { UrlAnatomy } from "./UrlAnatomy";

const SUMMARY: Record<ScanResult["risk_level"], string> = {
  low: "No meaningful phishing signal in the URL structure.",
  medium: "Some suspicious traits, but below the decision threshold. Verify the sender before trusting it.",
  high: "Above the decision threshold. Treat as phishing unless independently verified.",
  critical: "Strong phishing signal across multiple features. Do not open or enter credentials.",
};

export function ScanResultView({ result, compact = false }: { result: ScanResult; compact?: boolean }) {
  const [showFeatures, setShowFeatures] = useState(false);
  const meta = RISK_META[result.risk_level];
  const isPhish = result.verdict === "phishing";

  return (
    <div className="space-y-4">
      <div className={clsx("grid gap-4", !compact && "lg:grid-cols-[340px_1fr]")}>
        <section className="relative overflow-hidden rounded-xl border border-line bg-surface p-5">
          <span className="absolute inset-x-0 top-0 h-[2px]" style={{ background: meta.color }} />
          <div className="flex items-center justify-between">
            <span className="label">Risk assessment</span>
            <RiskBadge level={result.risk_level} />
          </div>
          <div className="mt-3 flex flex-col items-center">
            <RiskGauge probability={result.probability} threshold={result.threshold} level={result.risk_level} />
            <div className="mt-1 text-center">
              <div className="font-mono text-5xl font-medium tracking-tight tabular-nums text-ink">
                {Math.round(result.risk_score)}
                <span className="text-lg text-ink-3">/100</span>
              </div>
              <div className="mt-1.5 text-sm font-medium" style={{ color: meta.color }}>
                {isPhish ? "Likely phishing" : "Likely legitimate"}
              </div>
            </div>
          </div>
          <p className="mt-4 text-center text-[13px] leading-relaxed text-ink-2">{SUMMARY[result.risk_level]}</p>
          <dl className="mt-5 grid grid-cols-3 gap-2 border-t border-line pt-4 text-center">
            <div>
              <dt className="label">Probability</dt>
              <dd className="mt-1 font-mono text-sm tabular-nums">{(result.probability * 100).toFixed(1)}%</dd>
            </div>
            <div>
              <dt className="label">Threshold</dt>
              <dd className="mt-1 font-mono text-sm tabular-nums">{(result.threshold * 100).toFixed(1)}%</dd>
            </div>
            <div>
              <dt className="label">Latency</dt>
              <dd className="mt-1 font-mono text-sm tabular-nums">{result.latency_ms.toFixed(0)} ms</dd>
            </div>
          </dl>
        </section>

        <Panel title="URL anatomy" subtitle="Parsed with the Public Suffix List. Flagged tokens are underlined.">
          <UrlAnatomy components={result.components} indicators={result.indicators} />
          <div className="mt-4 rounded-lg border border-line bg-surface-2/50 px-3.5 py-2.5 font-mono text-[11.5px] text-ink-3">
            canonical form analysed by the model: <span className="break-all text-ink-2">{result.canonical_url}</span>
          </div>
        </Panel>
      </div>

      <div className={clsx("grid gap-4", !compact && "lg:grid-cols-[1.25fr_1fr]")}>
        <Panel title="Why this score" subtitle="Shift in log-odds of phishing if each feature took the value typical of legitimate URLs.">
          <Contributions items={result.contributions} />
        </Panel>
        <Panel title="Risk indicators" subtitle={`${result.indicators.length} observation${result.indicators.length === 1 ? "" : "s"} · context only, not part of the score`}>
          {result.indicators.length ? (
            <ul className="divide-y divide-line">
              {result.indicators.map((ind, i) => (
                <li key={i} className="flex gap-3 py-2.5 first:pt-0 last:pb-0">
                  <Fingerprint className="mt-0.5 h-4 w-4 shrink-0 text-ink-3" strokeWidth={1.75} />
                  <div className="min-w-0">
                    <div className="label !text-[10px]">{INDICATOR_LABELS[ind.kind] ?? ind.kind}</div>
                    <div className="text-[13px] leading-snug text-ink">{ind.reason}</div>
                  </div>
                </li>
              ))}
            </ul>
          ) : (
            <div className="flex items-center gap-3 text-sm text-ink-2">
              <ShieldCheck className="h-5 w-5 text-low" strokeWidth={1.75} />
              No rule-based red flags found in this URL.
            </div>
          )}
        </Panel>
      </div>

      <section className="rounded-xl border border-line bg-surface">
        <button
          onClick={() => setShowFeatures((s) => !s)}
          className="flex w-full items-center justify-between px-5 py-3.5 text-left"
          aria-expanded={showFeatures}
        >
          <span className="text-[13px] font-medium">
            Feature vector <span className="ml-2 font-mono text-xs text-ink-3">{Object.keys(result.features).length} lexical features</span>
          </span>
          <ChevronDown className={clsx("h-4 w-4 text-ink-3 transition-transform", showFeatures && "rotate-180")} />
        </button>
        {showFeatures && (
          <div className="grid grid-cols-1 gap-x-6 border-t border-line px-5 py-4 sm:grid-cols-2 lg:grid-cols-3">
            {Object.entries(result.features).map(([k, v]) => (
              <div key={k} className="flex justify-between gap-3 border-b border-line/60 py-1.5 font-mono text-[11.5px]">
                <span className="truncate text-ink-3">{k}</span>
                <span className="text-ink tabular-nums">{formatFeatureValue(v)}</span>
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}
