"use client";

import clsx from "clsx";
import { ChevronLeft, ChevronRight, History as HistoryIcon, Loader2, Search, Trash2, X } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { ScanResultView } from "@/components/ScanResultView";
import { Button, EmptyState, ErrorBanner, PageHeader, RiskBadge, Skeleton, VerdictTag } from "@/components/ui";
import { api, type HistoryPage, type ScanResult } from "@/lib/api";
import { relativeTime } from "@/lib/format";

const PAGE_SIZE = 15;
const FILTERS = [
  { value: "", label: "All" },
  { value: "phishing", label: "Phishing" },
  { value: "legitimate", label: "Legitimate" },
];

export default function HistoryPage() {
  const [page, setPage] = useState<HistoryPage | null>(null);
  const [offset, setOffset] = useState(0);
  const [verdict, setVerdict] = useState("");
  const [query, setQuery] = useState("");
  const [debounced, setDebounced] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<ScanResult | null>(null);
  const [loadingDetail, setLoadingDetail] = useState<string | null>(null);

  useEffect(() => {
    const t = setTimeout(() => setDebounced(query.trim()), 250);
    return () => clearTimeout(t);
  }, [query]);

  const load = useCallback(() => {
    api
      .history({ limit: PAGE_SIZE, offset, verdict: verdict || undefined, q: debounced || undefined })
      .then((p) => {
        setPage(p);
        setError(null);
      })
      .catch((e) => setError(e.message));
  }, [offset, verdict, debounced]);

  useEffect(() => {
    load();
  }, [load]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setSelected(null);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  async function open(id: string) {
    setLoadingDetail(id);
    try {
      setSelected(await api.historyItem(id));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not load scan");
    } finally {
      setLoadingDetail(null);
    }
  }

  async function remove(id: string) {
    await api.deleteItem(id).catch((e) => setError(e.message));
    load();
  }

  async function clearAll() {
    if (!window.confirm("Delete the entire scan history? This cannot be undone.")) return;
    await api.clearHistory().catch((e) => setError(e.message));
    setOffset(0);
    load();
  }

  const total = page?.total ?? 0;
  const from = total ? offset + 1 : 0;
  const to = Math.min(offset + PAGE_SIZE, total);

  return (
    <div>
      <PageHeader
        eyebrow="Detection log"
        title="Scan history"
        description="Every analysis is stored with its full explanation. Select a row to review the evidence."
        action={
          total > 0 && (
            <Button variant="danger" onClick={clearAll}>
              <Trash2 className="h-4 w-4" /> Clear history
            </Button>
          )
        }
      />

      <div className="mb-4 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="relative w-full sm:max-w-sm">
          <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-ink-3" />
          <input
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setOffset(0);
            }}
            placeholder="Filter by URL or domain"
            aria-label="Filter by URL"
            maxLength={200}
            className="h-9 w-full rounded-lg border border-line bg-surface pl-9 pr-3 font-mono text-[12.5px] text-ink placeholder:text-ink-3 focus:border-ink-3 focus:outline-none"
          />
        </div>
        <div className="inline-flex rounded-lg border border-line bg-surface p-0.5">
          {FILTERS.map((f) => (
            <button
              key={f.value}
              onClick={() => {
                setVerdict(f.value);
                setOffset(0);
              }}
              aria-pressed={verdict === f.value}
              className={clsx(
                "rounded-md px-3 py-1.5 text-xs font-medium transition-colors",
                verdict === f.value ? "bg-surface-3 text-ink" : "text-ink-3 hover:text-ink-2",
              )}
            >
              {f.label}
            </button>
          ))}
        </div>
      </div>

      {error && <div className="mb-4"><ErrorBanner message={error} /></div>}

      <div className="overflow-hidden rounded-xl border border-line bg-surface">
        {!page ? (
          <div className="space-y-2 p-5">
            {Array.from({ length: 6 }, (_, i) => <Skeleton key={i} className="h-9" />)}
          </div>
        ) : page.items.length === 0 ? (
          <EmptyState icon={HistoryIcon} title={debounced || verdict ? "No matching scans" : "No scans yet"}>
            {debounced || verdict ? "Try a different filter." : "Results from the scanner are logged here automatically."}
          </EmptyState>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[640px] text-sm">
              <thead>
                <tr className="border-b border-line text-left">
                  <th className="label px-5 py-3 font-normal">URL</th>
                  <th className="label px-3 py-3 font-normal">Verdict</th>
                  <th className="label px-3 py-3 font-normal">Risk</th>
                  <th className="label px-3 py-3 text-right font-normal">Score</th>
                  <th className="label px-3 py-3 font-normal">When</th>
                  <th className="w-10 px-3 py-3"><span className="sr-only">Actions</span></th>
                </tr>
              </thead>
              <tbody>
                {page.items.map((r) => (
                  <tr key={r.id} className="group border-b border-line/60 last:border-0 hover:bg-surface-2/60">
                    <td className="max-w-0 px-5 py-2.5">
                      <button onClick={() => open(r.id)} className="flex w-full min-w-0 items-center gap-2 text-left">
                        {loadingDetail === r.id && <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin text-ink-3" />}
                        <span className="min-w-0">
                          <span className="block truncate font-mono text-[12.5px] text-ink group-hover:underline" title={r.url}>{r.url}</span>
                          <span className="block truncate font-mono text-[11px] text-ink-3">{r.registered_domain}</span>
                        </span>
                      </button>
                    </td>
                    <td className="px-3 py-2.5"><VerdictTag verdict={r.verdict} /></td>
                    <td className="px-3 py-2.5"><RiskBadge level={r.risk_level} /></td>
                    <td className="px-3 py-2.5 text-right font-mono tabular-nums text-ink">{Math.round(r.risk_score)}</td>
                    <td className="whitespace-nowrap px-3 py-2.5 font-mono text-[11.5px] text-ink-3" title={new Date(r.created_at).toLocaleString()}>
                      {relativeTime(r.created_at)}
                    </td>
                    <td className="px-3 py-2.5">
                      <button
                        onClick={() => remove(r.id)}
                        aria-label={`Delete scan of ${r.url}`}
                        className="rounded p-1 text-ink-3 opacity-0 transition hover:bg-surface-3 hover:text-critical focus:opacity-100 group-hover:opacity-100"
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        {page && total > 0 && (
          <div className="flex items-center justify-between border-t border-line px-5 py-3">
            <span className="font-mono text-[11.5px] text-ink-3">
              {from}–{to} of {total}
            </span>
            <div className="flex gap-1">
              <Button variant="ghost" className="h-8 px-2" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))} aria-label="Previous page">
                <ChevronLeft className="h-4 w-4" />
              </Button>
              <Button variant="ghost" className="h-8 px-2" disabled={to >= total} onClick={() => setOffset(offset + PAGE_SIZE)} aria-label="Next page">
                <ChevronRight className="h-4 w-4" />
              </Button>
            </div>
          </div>
        )}
      </div>

      {selected && (
        <div className="fixed inset-0 z-50 flex justify-end" role="dialog" aria-modal="true" aria-label="Scan details">
          <button className="absolute inset-0 bg-black/60 backdrop-blur-[2px]" aria-label="Close details" onClick={() => setSelected(null)} />
          <div className="relative h-full w-full max-w-3xl overflow-y-auto border-l border-line bg-bg">
            <div className="sticky top-0 z-10 flex items-center justify-between gap-3 border-b border-line bg-bg/90 px-5 py-3 backdrop-blur">
              <div className="min-w-0">
                <div className="label">Scan detail</div>
                <div className="truncate font-mono text-xs text-ink-2">{selected.created_at && new Date(selected.created_at).toLocaleString()}</div>
              </div>
              <button onClick={() => setSelected(null)} className="rounded-md p-1.5 text-ink-2 hover:bg-surface-2" aria-label="Close">
                <X className="h-4 w-4" />
              </button>
            </div>
            <div className="p-5">
              <ScanResultView result={selected} compact />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
