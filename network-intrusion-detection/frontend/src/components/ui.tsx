"use client";

import clsx from "clsx";
import { AlertTriangle, Loader2 } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import type { ClassName, Severity } from "@/lib/api";
import { CLASS_META, SEVERITY_META } from "@/lib/format";

/* ----------------------------------------------------------------- data hook */

export function useApi<T>(fn: () => Promise<T>, deps: unknown[] = []) {
  const key = JSON.stringify(deps);
  const fnRef = useRef(fn);
  useEffect(() => {
    fnRef.current = fn;
  });
  const [tick, setTick] = useState(0);
  const [state, setState] = useState<{ key: string | null; tick: number; data: T | null; error: string | null }>({
    key: null, tick: -1, data: null, error: null,
  });
  useEffect(() => {
    let alive = true;
    fnRef.current()
      .then((data) => alive && setState({ key, tick, data, error: null }))
      .catch((e: Error) => alive && setState((s) => ({ ...s, key, tick, error: e.message })));
    return () => {
      alive = false;
    };
  }, [key, tick]);
  const loading = state.key !== key || state.tick !== tick;
  return { data: state.data, error: loading ? null : state.error, loading, reload: () => setTick((t) => t + 1) };
}

/* ------------------------------------------------------------------- layout */

export function PageHeader({ title, description, actions }: { title: string; description?: React.ReactNode; actions?: React.ReactNode }) {
  return (
    <header className="mb-6 flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
      <div>
        <h1 className="text-xl font-semibold tracking-tight text-ink">{title}</h1>
        {description && <p className="mt-1 max-w-3xl text-sm text-ink-2">{description}</p>}
      </div>
      {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
    </header>
  );
}

export function Card({ title, subtitle, actions, children, className, bodyClass }: {
  title?: React.ReactNode;
  subtitle?: React.ReactNode;
  actions?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
  bodyClass?: string;
}) {
  return (
    <section className={clsx("rounded-lg border border-line bg-surface", className)}>
      {(title || actions) && (
        <div className="flex items-start justify-between gap-3 border-b border-line px-4 py-3">
          <div className="min-w-0">
            {title && <h2 className="text-[13px] font-medium text-ink">{title}</h2>}
            {subtitle && <p className="mt-0.5 text-xs text-ink-3">{subtitle}</p>}
          </div>
          {actions}
        </div>
      )}
      <div className={clsx("p-4", bodyClass)}>{children}</div>
    </section>
  );
}

export function Stat({ label, value, sub, tone }: { label: string; value: React.ReactNode; sub?: React.ReactNode; tone?: string }) {
  return (
    <div className="rounded-lg border border-line bg-surface px-4 py-3.5">
      <div className="text-[11px] font-medium uppercase tracking-wider text-ink-3">{label}</div>
      <div className="tnum mt-1.5 text-2xl font-semibold tracking-tight text-ink" style={tone ? { color: tone } : undefined}>
        {value}
      </div>
      {sub && <div className="mt-1 text-xs text-ink-3">{sub}</div>}
    </div>
  );
}

/* ------------------------------------------------------------------- badges */

export function ClassBadge({ cls, className }: { cls: ClassName; className?: string }) {
  const m = CLASS_META[cls];
  return (
    <span className={clsx("inline-flex items-center gap-1.5 whitespace-nowrap text-[12px] text-ink", className)}>
      <span className="h-2 w-2 shrink-0 rounded-[2px]" style={{ background: m.color }} aria-hidden />
      {m.short}
    </span>
  );
}

export function SeverityBadge({ severity }: { severity: Severity }) {
  const m = SEVERITY_META[severity];
  if (severity === "none") return <span className="text-[12px] text-ink-3">—</span>;
  return (
    <span
      className="inline-flex items-center gap-1.5 rounded border px-1.5 py-0.5 text-[11px] font-medium uppercase tracking-wide"
      style={{ borderColor: `color-mix(in srgb, ${m.color} 45%, transparent)`, color: m.color,
        background: `color-mix(in srgb, ${m.color} 10%, transparent)` }}
    >
      <SeverityGlyph severity={severity} />
      {m.label}
    </span>
  );
}

/** Shape-coded so severity never relies on colour alone. */
export function SeverityGlyph({ severity, size = 8 }: { severity: Severity; size?: number }) {
  const c = SEVERITY_META[severity].color;
  const bars = SEVERITY_META[severity].rank;
  return (
    <svg width={size + 4} height={size} viewBox="0 0 12 8" aria-hidden>
      {[0, 1, 2, 3].map((i) => (
        <rect key={i} x={i * 3} y={8 - (i + 1) * 2} width={2} height={(i + 1) * 2} rx={0.5}
          fill={i < bars ? c : "var(--line-strong)"} />
      ))}
    </svg>
  );
}

export function ConfidenceBar({ value, className }: { value: number; className?: string }) {
  return (
    <div className={clsx("flex items-center gap-2", className)}>
      <div className="h-1.5 w-16 overflow-hidden rounded-full bg-surface-3">
        <div className="h-full rounded-full bg-ink-2" style={{ width: `${Math.max(2, value * 100)}%` }} />
      </div>
      <span className="tnum font-mono text-[11px] text-ink-2">{(value * 100).toFixed(1)}%</span>
    </div>
  );
}

/* ------------------------------------------------------------------- states */

export function Loading({ label = "Loading…" }: { label?: string }) {
  return (
    <div className="flex items-center gap-2 py-10 text-sm text-ink-3" role="status">
      <Loader2 className="h-4 w-4 animate-spin" /> {label}
    </div>
  );
}

export function ErrorState({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="flex items-start gap-3 rounded-lg border border-critical/40 bg-critical/5 px-4 py-3 text-sm" role="alert">
      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-critical" />
      <div className="flex-1 text-ink-2">{message}</div>
      {onRetry && (
        <button onClick={onRetry} className="text-xs font-medium text-ink underline-offset-2 hover:underline">
          Retry
        </button>
      )}
    </div>
  );
}

export function Empty({ title, children }: { title: string; children?: React.ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center rounded-lg border border-dashed border-line-strong px-6 py-12 text-center">
      <div className="text-sm font-medium text-ink">{title}</div>
      {children && <div className="mt-1.5 max-w-md text-sm text-ink-3">{children}</div>}
    </div>
  );
}

/* ------------------------------------------------------------------ controls */

export function Segmented<T extends string>({ value, options, onChange, label }: {
  value: T;
  options: { value: T; label: string }[];
  onChange: (v: T) => void;
  label: string;
}) {
  return (
    <div role="radiogroup" aria-label={label} className="inline-flex rounded-md border border-line bg-surface-2 p-0.5">
      {options.map((o) => (
        <button key={o.value} role="radio" aria-checked={value === o.value} onClick={() => onChange(o.value)}
          className={clsx("rounded px-2.5 py-1 text-xs transition-colors",
            value === o.value ? "bg-surface-3 text-ink" : "text-ink-3 hover:text-ink-2")}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Select({ value, onChange, label, children }: {
  value: string;
  onChange: (v: string) => void;
  label: string;
  children: React.ReactNode;
}) {
  return (
    <label className="flex items-center gap-2 text-xs text-ink-3">
      {label}
      <select value={value} onChange={(e) => onChange(e.target.value)}
        className="rounded-md border border-line bg-surface-2 px-2 py-1.5 text-xs text-ink focus:border-accent focus:outline-none">
        {children}
      </select>
    </label>
  );
}

export function Button({ children, variant = "secondary", className, ...props }: React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "secondary" | "ghost" | "danger";
}) {
  return (
    <button {...props}
      className={clsx("inline-flex items-center justify-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-50",
        variant === "primary" && "bg-ink text-bg hover:bg-white",
        variant === "secondary" && "border border-line-strong bg-surface-2 text-ink hover:bg-surface-3",
        variant === "ghost" && "text-ink-2 hover:bg-surface-2 hover:text-ink",
        variant === "danger" && "border border-critical/40 text-critical hover:bg-critical/10",
        className)}>
      {children}
    </button>
  );
}

export function Pagination({ total, limit, offset, onChange }: { total: number; limit: number; offset: number; onChange: (o: number) => void }) {
  if (total <= limit) return null;
  const page = Math.floor(offset / limit) + 1;
  const pages = Math.ceil(total / limit);
  return (
    <div className="flex items-center justify-between border-t border-line px-4 py-2.5 text-xs text-ink-3">
      <span className="tnum">
        {offset + 1}–{Math.min(offset + limit, total)} of {total.toLocaleString()}
      </span>
      <div className="flex items-center gap-1">
        <Button variant="ghost" disabled={page <= 1} onClick={() => onChange(Math.max(0, offset - limit))}>Previous</Button>
        <span className="tnum px-2">{page} / {pages}</span>
        <Button variant="ghost" disabled={page >= pages} onClick={() => onChange(offset + limit)}>Next</Button>
      </div>
    </div>
  );
}

export function Legend({ items }: { items: { label: string; color: string }[] }) {
  return (
    <ul className="flex flex-wrap gap-x-4 gap-y-1.5 text-xs text-ink-2">
      {items.map((i) => (
        <li key={i.label} className="flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-[2px]" style={{ background: i.color }} aria-hidden />
          {i.label}
        </li>
      ))}
    </ul>
  );
}

export function Th({ children, className }: { children?: React.ReactNode; className?: string }) {
  return <th className={clsx("whitespace-nowrap px-4 py-2 text-left text-[11px] font-medium uppercase tracking-wider text-ink-3", className)}>{children}</th>;
}

export function Td({ children, className }: { children?: React.ReactNode; className?: string }) {
  return <td className={clsx("px-4 py-2.5 text-[13px] text-ink-2", className)}>{children}</td>;
}
