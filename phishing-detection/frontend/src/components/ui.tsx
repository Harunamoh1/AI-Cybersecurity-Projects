import clsx from "clsx";
import { AlertTriangle, CheckCircle2, OctagonAlert, ShieldAlert } from "lucide-react";
import type { RiskLevel } from "@/lib/api";
import { RISK_META } from "@/lib/format";

export function Panel({
  title,
  subtitle,
  action,
  children,
  className,
  bodyClassName,
}: {
  title?: React.ReactNode;
  subtitle?: React.ReactNode;
  action?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
  bodyClassName?: string;
}) {
  return (
    <section className={clsx("rounded-xl border border-line bg-surface", className)}>
      {(title || action) && (
        <header className="flex items-start justify-between gap-3 border-b border-line px-5 py-3.5">
          <div className="min-w-0">
            {title && <h2 className="text-[13px] font-medium text-ink">{title}</h2>}
            {subtitle && <p className="mt-0.5 text-xs text-ink-3">{subtitle}</p>}
          </div>
          {action}
        </header>
      )}
      <div className={clsx("p-5", bodyClassName)}>{children}</div>
    </section>
  );
}

export function PageHeader({ eyebrow, title, description, action }: { eyebrow: string; title: string; description?: React.ReactNode; action?: React.ReactNode }) {
  return (
    <div className="mb-6 flex flex-col gap-4 sm:mb-8 sm:flex-row sm:items-end sm:justify-between">
      <div>
        <div className="label mb-2">{eyebrow}</div>
        <h1 className="text-2xl font-semibold tracking-tight text-ink sm:text-[28px]">{title}</h1>
        {description && <p className="mt-2 max-w-2xl text-sm leading-relaxed text-ink-2">{description}</p>}
      </div>
      {action}
    </div>
  );
}

const RISK_ICON: Record<RiskLevel, typeof CheckCircle2> = {
  low: CheckCircle2,
  medium: AlertTriangle,
  high: ShieldAlert,
  critical: OctagonAlert,
};

export function RiskBadge({ level, size = "sm" }: { level: RiskLevel; size?: "sm" | "md" }) {
  const meta = RISK_META[level];
  const Icon = RISK_ICON[level];
  return (
    <span
      className={clsx(
        "inline-flex items-center gap-1.5 rounded-md border border-line bg-surface-2 font-mono uppercase tracking-wider text-ink",
        size === "sm" ? "px-1.5 py-0.5 text-[10.5px]" : "px-2 py-1 text-xs",
      )}
    >
      <Icon className={clsx(size === "sm" ? "h-3 w-3" : "h-3.5 w-3.5", meta.text)} strokeWidth={2.25} />
      {meta.label}
    </span>
  );
}

export function VerdictTag({ verdict }: { verdict: "phishing" | "legitimate" }) {
  return (
    <span className="inline-flex items-center gap-1.5 text-xs text-ink-2">
      <span className={clsx("h-1.5 w-1.5 rounded-full", verdict === "phishing" ? "bg-phish" : "bg-legit")} />
      {verdict === "phishing" ? "Phishing" : "Legitimate"}
    </span>
  );
}

export function StatTile({ label, value, sub, accent }: { label: string; value: React.ReactNode; sub?: React.ReactNode; accent?: string }) {
  return (
    <div className="relative overflow-hidden rounded-xl border border-line bg-surface px-5 py-4">
      {accent && <span className="absolute inset-x-0 top-0 h-px" style={{ background: accent }} />}
      <div className="label">{label}</div>
      <div className="mt-2 font-mono text-[26px] font-medium tracking-tight text-ink tabular-nums">{value}</div>
      {sub && <div className="mt-1 text-xs text-ink-3">{sub}</div>}
    </div>
  );
}

export function Button({
  variant = "primary",
  className,
  ...props
}: React.ButtonHTMLAttributes<HTMLButtonElement> & { variant?: "primary" | "secondary" | "ghost" | "danger" }) {
  return (
    <button
      {...props}
      className={clsx(
        "inline-flex items-center justify-center gap-2 rounded-lg px-3.5 py-2 text-sm font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-50",
        variant === "primary" && "bg-ink text-bg hover:bg-white",
        variant === "secondary" && "border border-line-strong bg-surface-2 text-ink hover:bg-surface-3",
        variant === "ghost" && "text-ink-2 hover:bg-surface-2 hover:text-ink",
        variant === "danger" && "border border-critical/40 text-critical hover:bg-critical/10",
        className,
      )}
    />
  );
}

export function EmptyState({ icon: Icon, title, children }: { icon: typeof CheckCircle2; title: string; children?: React.ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center px-6 py-14 text-center">
      <div className="mb-4 grid h-11 w-11 place-items-center rounded-xl border border-line bg-surface-2">
        <Icon className="h-5 w-5 text-ink-2" strokeWidth={1.75} />
      </div>
      <div className="text-sm font-medium text-ink">{title}</div>
      {children && <div className="mt-1.5 max-w-sm text-sm text-ink-3">{children}</div>}
    </div>
  );
}

export function ErrorBanner({ message }: { message: string }) {
  return (
    <div role="alert" className="flex items-start gap-2.5 rounded-lg border border-critical/30 bg-critical/[0.06] px-4 py-3 text-sm text-ink">
      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-critical" />
      <span>{message}</span>
    </div>
  );
}

export function Skeleton({ className }: { className?: string }) {
  return <div className={clsx("animate-pulse rounded-md bg-surface-2", className)} />;
}

export function ChartTooltip({
  active,
  payload,
  label,
  formatter,
}: {
  active?: boolean;
  payload?: { name?: string; value?: number | string; color?: string; payload?: unknown }[];
  label?: string | number;
  formatter?: (v: number | string, name?: string) => string;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-lg border border-line-strong bg-surface-2/95 px-3 py-2 text-xs shadow-xl backdrop-blur">
      {label !== undefined && label !== "" && <div className="mb-1 font-mono text-ink-3">{label}</div>}
      {payload.map((p, i) => (
        <div key={i} className="flex items-center gap-2 text-ink">
          <span className="h-2 w-2 rounded-sm" style={{ background: p.color }} />
          <span className="text-ink-2">{p.name}</span>
          <span className="ml-auto pl-3 font-mono tabular-nums">{formatter ? formatter(p.value ?? "", p.name) : p.value}</span>
        </div>
      ))}
    </div>
  );
}
