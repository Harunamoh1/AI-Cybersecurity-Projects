"use client";

import clsx from "clsx";
import { FileUp, Loader2, Play, Trash2 } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useRef, useState } from "react";
import { Button, Card, Empty, ErrorState, Loading, PageHeader, Pagination, Td, Th, useApi } from "@/components/ui";
import { api } from "@/lib/api";
import { bytes, dateTime, int, pct } from "@/lib/format";

const MAX_MB = 25;

export default function AnalyzePage() {
  const router = useRouter();
  const input = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [drag, setDrag] = useState(false);
  const [offset, setOffset] = useState(0);
  const samples = useApi(api.samples);
  const history = useApi(() => api.analyses(10, offset), [offset]);
  const features = useApi(() => fetch("/api/v1/model/features").then((r) => r.json() as Promise<{ required: { cic2017: string }[] }>));

  async function run(label: string, job: () => Promise<{ analysis: { id: string } }>) {
    setBusy(label);
    setError(null);
    try {
      const res = await job();
      router.push(`/analyze/${res.analysis.id}`);
    } catch (e) {
      setError((e as Error).message);
      setBusy(null);
    }
  }

  function onFile(file: File | undefined) {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".csv")) return setError("Only .csv files are accepted.");
    if (file.size > MAX_MB * 1024 * 1024) return setError(`File exceeds the ${MAX_MB} MB limit.`);
    run(file.name, () => api.upload(file));
  }

  async function remove(id: string) {
    await api.deleteAnalysis(id).catch((e) => setError(e.message));
    history.reload();
  }

  return (
    <>
      <PageHeader title="Traffic analysis"
        description="Upload a network-flow CSV exported by CICFlowMeter (CIC-IDS2017 or CSE-CIC-IDS2018 column names). Files are parsed as data only — never executed — and host identifiers such as IP addresses are discarded before storage." />

      <div className="grid gap-4 xl:grid-cols-5">
        <Card title="Upload flows" className="xl:col-span-3">
          <div
            onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
            onDragLeave={() => setDrag(false)}
            onDrop={(e) => { e.preventDefault(); setDrag(false); onFile(e.dataTransfer.files[0]); }}
            className={clsx("flex flex-col items-center justify-center rounded-lg border border-dashed px-6 py-10 text-center transition-colors",
              drag ? "border-accent bg-accent/5" : "border-line-strong")}>
            {busy ? (
              <div className="flex items-center gap-2 text-sm text-ink-2" role="status">
                <Loader2 className="h-4 w-4 animate-spin" /> Analysing <span className="font-mono text-ink">{busy}</span>…
              </div>
            ) : (
              <>
                <FileUp className="h-6 w-6 text-ink-3" aria-hidden />
                <p className="mt-3 text-sm text-ink">Drop a .csv file here</p>
                <p className="mt-1 text-xs text-ink-3">Up to {MAX_MB} MB and 100,000 flows per file</p>
                <Button variant="primary" className="mt-4" onClick={() => input.current?.click()}>Choose file</Button>
                <input ref={input} type="file" accept=".csv,text/csv" className="hidden"
                  onChange={(e) => { onFile(e.target.files?.[0]); e.target.value = ""; }} />
              </>
            )}
          </div>
          {error && <div className="mt-3"><ErrorState message={error} /></div>}
          <details className="mt-4 text-xs text-ink-3">
            <summary className="cursor-pointer text-ink-2 hover:text-ink">
              Required columns ({features.data?.required.length ?? "…"}) and input handling
            </summary>
            <ul className="mt-2 list-disc space-y-1 pl-5">
              <li>Extra columns are ignored and reported; identifier columns (Flow ID, IPs, ports, Timestamp) are dropped.</li>
              <li>A <span className="font-mono">Label</span> column, if present, is used only to compare predictions with ground truth.</li>
              <li>Missing, non-numeric or infinite values are imputed with training medians and counted; rows with more than half the features invalid are rejected.</li>
            </ul>
            {features.data && (
              <p className="mt-2 font-mono leading-relaxed text-ink-3">{features.data.required.map((f) => f.cic2017).join(" · ")}</p>
            )}
          </details>
        </Card>

        <Card title="Held-out samples" subtitle="Flows from the test split — never seen in training" className="xl:col-span-2" bodyClass="p-0">
          {samples.loading ? <div className="px-4"><Loading /></div> : (
            <ul className="divide-y divide-line">
              {samples.data?.map((s) => (
                <li key={s.name} className="flex items-start gap-3 px-4 py-3">
                  <div className="min-w-0 flex-1">
                    <div className="text-[13px] text-ink">{s.title ?? s.name}</div>
                    <div className="mt-0.5 text-xs leading-relaxed text-ink-3">{s.description}</div>
                    <div className="mt-1 flex gap-3 font-mono text-[11px] text-ink-3">
                      <a href={`/api/v1/samples/${s.name}`} className="hover:text-ink" download>{s.name}</a>
                      <span>{bytes(s.size_bytes)}</span>
                    </div>
                  </div>
                  <Button disabled={!!busy} onClick={() => run(s.name, () => api.analyzeSample(s.name))} aria-label={`Analyse ${s.name}`}>
                    <Play className="h-3 w-3" /> Analyse
                  </Button>
                </li>
              ))}
            </ul>
          )}
        </Card>
      </div>

      <Card title="Analysis history" className="mt-4" bodyClass="p-0">
        {history.error ? <div className="p-4"><ErrorState message={history.error} onRetry={history.reload} /></div>
          : !history.data ? <div className="px-4"><Loading /></div>
          : history.data.items.length === 0 ? <div className="p-4"><Empty title="No analyses yet">Your uploads will appear here.</Empty></div>
          : (
            <>
              <div className="overflow-x-auto">
                <table className="w-full min-w-[720px]">
                  <thead className="border-b border-line">
                    <tr><Th>File</Th><Th>Analysed</Th><Th className="text-right">Flows</Th><Th className="text-right">Threats</Th><Th className="text-right">Label agreement</Th><Th className="text-right">Time</Th><Th /></tr>
                  </thead>
                  <tbody className="divide-y divide-line">
                    {history.data.items.map((a) => (
                      <tr key={a.id} className="hover:bg-surface-2">
                        <Td>
                          <Link href={`/analyze/${a.id}`} className="text-ink hover:text-accent">{a.filename}</Link>
                          <span className="ml-2 rounded border border-line px-1 py-px text-[10px] uppercase text-ink-3">{a.source}</span>
                        </Td>
                        <Td>{dateTime(a.created_at)}</Td>
                        <Td className="tnum text-right font-mono">{int(a.total_flows)}</Td>
                        <Td className="tnum text-right font-mono">{int(a.threats)}</Td>
                        <Td className="tnum text-right font-mono">{a.summary.ground_truth ? pct(a.summary.ground_truth.accuracy) : "—"}</Td>
                        <Td className="tnum text-right font-mono">{(a.duration_ms / 1000).toFixed(2)} s</Td>
                        <Td className="text-right">
                          <Button variant="ghost" onClick={() => remove(a.id)} aria-label={`Delete analysis ${a.filename}`}>
                            <Trash2 className="h-3.5 w-3.5" />
                          </Button>
                        </Td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <Pagination total={history.data.total} limit={10} offset={offset} onChange={setOffset} />
            </>
          )}
      </Card>
    </>
  );
}
