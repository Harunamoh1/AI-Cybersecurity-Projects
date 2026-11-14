"use client";

import clsx from "clsx";
import { Activity, BarChart3, BrainCircuit, History, Menu, ScanSearch, X } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { Logo } from "./Logo";

const NAV = [
  { href: "/", label: "Scanner", icon: ScanSearch, hint: "Analyse a URL" },
  { href: "/history", label: "History", icon: History, hint: "Past detections" },
  { href: "/analytics", label: "Analytics", icon: BarChart3, hint: "Threat overview" },
  { href: "/model", label: "Model", icon: BrainCircuit, hint: "Evaluation report" },
];

interface Health {
  status: string;
  model: string;
  model_version: string;
}

function useHealth() {
  const [health, setHealth] = useState<Health | null | "down">(null);
  useEffect(() => {
    let alive = true;
    const check = () =>
      fetch("/api/health", { cache: "no-store" })
        .then((r) => (r.ok ? r.json() : Promise.reject()))
        .then((h: Health) => alive && setHealth(h))
        .catch(() => alive && setHealth("down"));
    check();
    const id = setInterval(check, 30_000);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);
  return health;
}

function StatusPill() {
  const health = useHealth();
  const up = health && health !== "down";
  return (
    <div className="flex items-center gap-2 rounded-full border border-line bg-surface px-3 py-1.5 font-mono text-[11px] text-ink-2">
      <span className="relative flex h-2 w-2">
        {up && <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-low opacity-50" />}
        <span className={clsx("relative inline-flex h-2 w-2 rounded-full", up ? "bg-low" : health === "down" ? "bg-critical" : "bg-ink-3")} />
      </span>
      {up ? (
        <span>
          ENGINE ONLINE <span className="text-ink-3">· v{(health as Health).model_version}</span>
        </span>
      ) : health === "down" ? (
        <span>ENGINE OFFLINE</span>
      ) : (
        <span>CONNECTING…</span>
      )}
    </div>
  );
}

export function Shell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);
  const [lastPath, setLastPath] = useState(pathname);
  if (pathname !== lastPath) {
    setLastPath(pathname);
    setOpen(false);
  }

  const nav = (
    <nav className="flex flex-col gap-0.5" aria-label="Main">
      {NAV.map(({ href, label, icon: Icon, hint }) => {
        const active = href === "/" ? pathname === "/" : pathname.startsWith(href);
        return (
          <Link
            key={href}
            href={href}
            aria-current={active ? "page" : undefined}
            className={clsx(
              "group relative flex items-center gap-3 rounded-md px-3 py-2 text-sm transition-colors",
              active ? "bg-surface-2 text-ink" : "text-ink-2 hover:bg-surface hover:text-ink",
            )}
          >
            {active && <span className="absolute inset-y-1.5 left-0 w-0.5 rounded-full bg-ink" />}
            <Icon className="h-4 w-4 shrink-0" strokeWidth={1.75} />
            <span className="flex-1">{label}</span>
            <span className="hidden text-[11px] text-ink-3 xl:inline">{hint}</span>
          </Link>
        );
      })}
    </nav>
  );

  return (
    <div className="flex min-h-dvh">
      {/* Desktop sidebar */}
      <aside className="sticky top-0 hidden h-dvh w-64 shrink-0 flex-col border-r border-line bg-bg px-3 py-5 lg:flex">
        <Link href="/" className="mb-8 flex items-center gap-2.5 px-3">
          <Logo className="h-7 w-7 text-ink" />
          <div className="leading-tight">
            <div className="text-[15px] font-semibold tracking-tight">PhishLens</div>
            <div className="label !text-[9.5px]">URL threat analysis</div>
          </div>
        </Link>
        <div className="label mb-2 px-3">Workspace</div>
        {nav}
        <div className="mt-auto space-y-3 px-3">
          <div className="rounded-lg border border-line bg-surface p-3">
            <div className="mb-1.5 flex items-center gap-1.5 text-xs font-medium text-ink">
              <Activity className="h-3.5 w-3.5" strokeWidth={2} /> Passive analysis
            </div>
            <p className="text-[11.5px] leading-relaxed text-ink-3">
              URLs are scored from their structure only. PhishLens never visits the link.
            </p>
          </div>
        </div>
      </aside>

      {/* Mobile drawer */}
      {open && (
        <div className="fixed inset-0 z-40 lg:hidden">
          <button className="absolute inset-0 bg-black/60" aria-label="Close menu" onClick={() => setOpen(false)} />
          <aside className="absolute inset-y-0 left-0 w-72 border-r border-line bg-bg px-3 py-5">
            <div className="mb-6 flex items-center justify-between px-3">
              <div className="flex items-center gap-2.5">
                <Logo className="h-7 w-7 text-ink" />
                <span className="font-semibold">PhishLens</span>
              </div>
              <button onClick={() => setOpen(false)} aria-label="Close menu" className="text-ink-2">
                <X className="h-5 w-5" />
              </button>
            </div>
            {nav}
          </aside>
        </div>
      )}

      <div className="relative flex min-w-0 flex-1 flex-col">
        <div className="grid-bg pointer-events-none absolute inset-x-0 top-0 h-[420px]" />
        <header className="sticky top-0 z-30 flex h-14 items-center justify-between gap-3 border-b border-line bg-bg/80 px-4 backdrop-blur-md sm:px-6 lg:px-8">
          <div className="flex items-center gap-3">
            <button className="text-ink-2 lg:hidden" onClick={() => setOpen(true)} aria-label="Open menu">
              <Menu className="h-5 w-5" />
            </button>
            <Link href="/" className="flex items-center gap-2 lg:hidden">
              <Logo className="h-6 w-6 text-ink" />
              <span className="font-semibold">PhishLens</span>
            </Link>
            <div className="hidden font-mono text-[11px] text-ink-3 lg:block">
              {NAV.find((n) => (n.href === "/" ? pathname === "/" : pathname.startsWith(n.href)))?.label ?? ""}
              <span className="mx-2 text-line-strong">/</span>
              <span className="text-ink-2">console</span>
            </div>
          </div>
          <StatusPill />
        </header>
        <main className="relative mx-auto w-full max-w-[1280px] flex-1 px-4 py-6 sm:px-6 sm:py-8 lg:px-8">{children}</main>
        <footer className="border-t border-line px-4 py-4 text-center font-mono text-[11px] text-ink-3 sm:px-6 lg:px-8">
          PhishLens · ML decision support — always verify before acting on a verdict
        </footer>
      </div>
    </div>
  );
}
