"use client";

import Link from "next/link";
import type { FlowRecord } from "@/lib/api";
import { dateTime, featureLabel, featureValue } from "@/lib/format";
import { ClassBadge, ConfidenceBar, SeverityBadge, Td, Th } from "./ui";

export function FlowTable({ flows, showSource = false, showTruth = false }: { flows: FlowRecord[]; showSource?: boolean; showTruth?: boolean }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[760px]">
        <thead className="border-b border-line">
          <tr>
            <Th>Flow</Th>
            {showSource && <Th>Source file</Th>}
            <Th>Classification</Th>
            <Th>Severity</Th>
            <Th>Confidence</Th>
            {showTruth && <Th>Label</Th>}
            <Th>Top reason</Th>
            <Th className="text-right"><span className="sr-only">Actions</span></Th>
          </tr>
        </thead>
        <tbody className="divide-y divide-line">
          {flows.map((f) => {
            const r = f.reasons[0];
            const mismatch = showTruth && f.true_label && f.true_label !== f.predicted;
            return (
              <tr key={`${f.analysis_id}-${f.idx}`} className="hover:bg-surface-2">
                <Td className="font-mono text-xs text-ink-3">
                  #{f.idx}
                  {showSource && <div className="mt-0.5 whitespace-nowrap font-sans text-[11px]">{dateTime(f.created_at)}</div>}
                </Td>
                {showSource && <Td className="max-w-48 truncate" >{f.filename}</Td>}
                <Td><ClassBadge cls={f.predicted} /></Td>
                <Td><SeverityBadge severity={f.severity} /></Td>
                <Td><ConfidenceBar value={f.confidence} /></Td>
                {showTruth && (
                  <Td className={mismatch ? "text-high" : "text-ink-3"}>
                    {f.true_label ?? "—"}
                    {mismatch && <span className="sr-only"> (disagrees with prediction)</span>}
                  </Td>
                )}
                <Td className="whitespace-nowrap text-xs">
                  {r ? (
                    <span>
                      <span className="text-ink">{featureLabel(r.feature)}</span>
                      <span className="ml-1.5 font-mono text-ink-3">{featureValue(r.value)}</span>
                    </span>
                  ) : <span className="text-ink-3">—</span>}
                </Td>
                <Td className="text-right">
                  <Link href={`/explain?a=${f.analysis_id}&i=${f.idx}`} className="text-xs font-medium text-accent hover:underline">
                    Explain
                  </Link>
                </Td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
