"use client";

import clsx from "clsx";
import { ArrowRight, CornerDownLeft, Layers, Link2, Loader2, ScanSearch } from "lucide-react";
import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import { ScanResultView } from "@/components/ScanResultView";
import { Button, ErrorBanner, PageHeader, Panel, RiskBadge, VerdictTag } from "@/components/ui";
import { api, type HistoryItem, type ScanResult } from "@/lib/api";
import { relativeTime } from "@/lib/format";

// Sample inputs to try. They are sent to the live model like any other URL;
// nothing about their result is precomputed.
const SAMPLES = [
  "https://github.com/scikit-learn/scikit-learn/issues",
  "https://en.wikipedia.org/wiki/Phishing",
  "http://paypal.com.account-verify.secure-login.xyz/webscr?cmd=_login",
  "https://microsoft365-auth-portal.web.app/owa/signin",
  "http://192.168.10.4:8080/banking/update.php",
];

type Mode = "single" | "batch";

export default function ScannerPage() {
  const [mode, setMode] = useState<Mode>("single");
  const [url, setUrl] = useState("");
  const [batch, setBatch] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ScanResult | null>(null);
  const [batchResult, setBatchResult] = useState<{ results: ScanResult[]; errors: { url: string; error: string }[] } | null>(null);
  const [recent, setRecent] = useState<HistoryItem[]>([]);
  const inputRef = useRef<HTMLInputElement>(null);

  const loadRecent = useCallback(() => {
    api.history({ limit: 6, offset: 0 }).then((p) => setRecent(p.items)).catch(() => setRecent([]));
  }, []);

  useEffect(() => {
    loadRecent();
    inputRef.current?.focus();
  }, [loadRecent]);

  async function runScan(target?: string) {
    const value = (target ?? url).trim();
    if (!value) {
      setError("Enter a URL to analyse.");
      return;
    }
    if (target) setUrl(target);
    setLoading(true);
    setError(null);
    try {
      setResult(await api.scan(value));
      loadRecent();
    } catch (e) {
      setResult(null);
      setError(e instanceof Error ? e.message : "Scan failed");
    } finally {
      setLoading(false);
    }
  }

  async function runBatch() {
    const urls = batch.split(/\r?\n/).map((s) => s.trim()).filter(Boolean);
    if (!urls.length) {
      setError("Paste one URL per line.");
      return;
    }
    if (urls.length > 100) {
      setError("Batch scans are limited to 100 URLs.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      setBatchResult(await api.scanBatch(urls));
      loadRecent();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Batch scan failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <PageHeader
        eyebrow="Threat analysis"
        title="Scan a URL"
        description="A gradient-boosted classifier trained on about 250k labelled URLs scores the link from its structure, explains which features drove the score, and records it in your history. The link itself is never opened."
      />

      <div className="mb-3 inline-flex rounded-lg border border-line bg-surface p-0.5" role="tablist">
        {(["single", "batch"] as Mode[]).map((m) => (
          <button
            key={m}
            role="tab"
            aria-selected={mode === m}
            onClick={() => {
              setMode(m);
              setError(null);
            }}
            className={clsx(
              "flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium transition-colors",
              mode === m ? "bg-surface-3 text-ink" : "text-ink-3 hover:text-ink-2",
            )}
          >
            {m === "single" ? <Link2 className="h-3.5 w-3.5" /> : <Layers className="h-3.5 w-3.5" />}
            {m === "single" ? "Single URL" : "Batch"}
          </button>
        ))}
      </div>

      {mode === "single" ? (
        <form
          onSubmit={(e) => {
            e.preventDefault();
            runScan();
          }}
          className="group relative overflow-hidden rounded-xl border border-line-strong bg-surface shadow-[0_0_0_4px_rgb(255_255_255/0.015)] transition-colors focus-within:border-ink-3"
        >
          <div className="flex flex-col gap-2 p-2 sm:flex-row sm:items-center">
            <div className="flex min-w-0 flex-1 items-center gap-3 px-3">
              <span className="font-mono text-sm text-ink-3" aria-hidden>
                ›
              </span>
              <label htmlFor="url" className="sr-only">
                URL to scan
              </label>
              <input
                ref={inputRef}
                id="url"
                type="text"
                inputMode="url"
                autoComplete="off"
                spellCheck={false}
                maxLength={2048}
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                placeholder="paste a suspicious link, e.g. secure-login.example.com/verify"
                className="h-11 w-full min-w-0 bg-transparent font-mono text-[14px] text-ink placeholder:text-ink-3 focus:outline-none"
              />
            </div>
            <Button type="submit" disabled={loading} className="h-11 shrink-0 px-5">
              {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <ScanSearch className="h-4 w-4" />}
              {loading ? "Analysing" : "Analyse"}
              {!loading && <CornerDownLeft className="hidden h-3.5 w-3.5 opacity-50 sm:block" />}
            </Button>
          </div>
          {loading && (
            <div className="absolute inset-x-0 bottom-0 h-px overflow-hidden">
              <div className="animate-sweep h-px w-1/4 bg-gradient-to-r from-transparent via-ink to-transparent" />
            </div>
          )}
        </form>
      ) : (
        <div className="rounded-xl border border-line-strong bg-surface p-2">
          <label htmlFor="batch" className="sr-only">
            URLs to scan, one per line
          </label>
          <textarea
            id="batch"
            value={batch}
            onChange={(e) => setBatch(e.target.value)}
            rows={6}
            spellCheck={false}
            placeholder={"one URL per line (max 100)\nhttps://example.com/login\nhttp://203.0.113.7/verify"}
            className="w-full resize-y bg-transparent px-3 py-2 font-mono text-[13px] text-ink placeholder:text-ink-3 focus:outline-none"
          />
          <div className="flex items-center justify-between border-t border-line px-2 pt-2">
            <span className="font-mono text-[11px] text-ink-3">{batch.split(/\r?\n/).filter((s) => s.trim()).length} URLs</span>
            <Button onClick={runBatch} disabled={loading}>
              {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Layers className="h-4 w-4" />}
              Analyse batch
            </Button>
          </div>
        </div>
      )}

      {mode === "single" && (
        <div className="mt-3 flex flex-wrap items-center gap-2">
          <span className="label mr-1">Try</span>
          {SAMPLES.map((s) => (
            <button
              key={s}
              onClick={() => runScan(s)}
              disabled={loading}
              className="max-w-full truncate rounded-md border border-line bg-surface px-2 py-1 font-mono text-[11px] text-ink-2 transition-colors hover:border-line-strong hover:text-ink sm:max-w-[300px]"
              title={s}
            >
              {s.replace(/^https?:\/\//, "")}
            </button>
          ))}
        </div>
      )}

      <div className="mt-6">{error && <ErrorBanner message={error} />}</div>

      {mode === "single" && result && (
        <div className="mt-2">
          <ScanResultView result={result} />
        </div>
      )}

      {mode === "batch" && batchResult && (
        <Panel
          className="mt-2"
          title="Batch results"
          subtitle={`${batchResult.results.length} analysed · ${batchResult.results.filter((r) => r.verdict === "phishing").length} flagged · ${batchResult.errors.length} rejected`}
          bodyClassName="p-0"
        >
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-line text-left">
                  <th className="label px-5 py-2.5 font-normal">URL</th>
                  <th className="label px-3 py-2.5 font-normal">Verdict</th>
                  <th className="label px-3 py-2.5 font-normal">Risk</th>
                  <th className="label px-5 py-2.5 text-right font-normal">Score</th>
                </tr>
              </thead>
              <tbody>
                {[...batchResult.results]
                  .sort((a, b) => b.risk_score - a.risk_score)
                  .map((r) => (
                    <tr key={r.id ?? r.url} className="border-b border-line/60 last:border-0">
                      <td className="max-w-[420px] truncate px-5 py-2.5 font-mono text-[12.5px] text-ink" title={r.url}>
                        {r.url}
                      </td>
                      <td className="px-3 py-2.5"><VerdictTag verdict={r.verdict} /></td>
                      <td className="px-3 py-2.5"><RiskBadge level={r.risk_level} /></td>
                      <td className="px-5 py-2.5 text-right font-mono tabular-nums">{Math.round(r.risk_score)}</td>
                    </tr>
                  ))}
                {batchResult.errors.map((e, i) => (
                  <tr key={`err-${i}`} className="border-b border-line/60 last:border-0">
                    <td className="max-w-[420px] truncate px-5 py-2.5 font-mono text-[12.5px] text-ink-3">{e.url}</td>
                    <td colSpan={3} className="px-3 py-2.5 text-xs text-critical">{e.error}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
      )}

      {!result && !batchResult && (
        <div className="mt-2 grid gap-4 lg:grid-cols-3">
          <Panel
            className="lg:col-span-2"
            title="Recent detections"
            action={
              <Link href="/history" className="flex items-center gap-1 text-xs text-ink-2 hover:text-ink">
                View all <ArrowRight className="h-3 w-3" />
              </Link>
            }
            bodyClassName="p-0"
          >
            {recent.length ? (
              <ul className="divide-y divide-line">
                {recent.map((r) => (
                  <li key={r.id}>
                    <button onClick={() => runScan(r.url)} className="flex w-full items-center gap-4 px-5 py-3 text-left hover:bg-surface-2/60">
                      <RiskBadge level={r.risk_level} />
                      <span className="min-w-0 flex-1 truncate font-mono text-[12.5px] text-ink-2">{r.url}</span>
                      <span className="hidden font-mono text-[11px] text-ink-3 sm:block">{relativeTime(r.created_at)}</span>
                    </button>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="px-5 py-8 text-center text-sm text-ink-3">No scans yet. Your analyses will appear here.</p>
            )}
          </Panel>
          <Panel title="How scoring works">
            <ol className="space-y-3 text-[13px] leading-relaxed text-ink-2">
              {[
                ["Parse", "The URL is canonicalised and split into subdomain, registered domain, path and query."],
                ["Extract", "36 lexical features: structure, entropy, brand and keyword signals, hosting type, TLD."],
                ["Score", "The gradient-boosted model outputs a phishing probability, then compares it with a threshold tuned on validation data."],
                ["Explain", "Each feature is reset to a typical legitimate value to measure how much it moved the score."],
              ].map(([t, d], i) => (
                <li key={t} className="flex gap-3">
                  <span className="mt-0.5 grid h-5 w-5 shrink-0 place-items-center rounded border border-line font-mono text-[10px] text-ink-3">{i + 1}</span>
                  <span>
                    <span className="font-medium text-ink">{t}.</span> {d}
                  </span>
                </li>
              ))}
            </ol>
          </Panel>
        </div>
      )}
    </div>
  );
}
