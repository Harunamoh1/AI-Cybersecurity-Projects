"use client";

import clsx from "clsx";
import { Activity, BrainCircuit, FileSearch, LayoutDashboard, Menu, ShieldAlert, Sigma, X } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";

const NAV = [
  { href: "/", label: "Overview", icon: LayoutDashboard },
  { href: "/analyze", label: "Traffic Analysis", icon: FileSearch },
  { href: "/threats", label: "Threat Intelligence", icon: ShieldAlert },
  { href: "/explain", label: "Explainability", icon: Sigma },
  { href: "/model", label: "Model Performance", icon: BrainCircuit },
];

function Logo() {
  return (
    <div className="flex items-center gap-2.5">
      <svg width="22" height="22" viewBox="0 0 24 24" aria-hidden>
        <path d="M12 2 4 5v6c0 5 3.4 9.4 8 11 4.6-1.6 8-6 8-11V5l-8-3Z" fill="none" stroke="var(--text)" strokeWidth="1.6" />
        <path d="M7 12.5h2.4l1.4-3 2.4 6 1.4-3H17" fill="none" stroke="var(--accent)" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
      <div className="whitespace-nowrap leading-tight">
        <div className="text-[14px] font-semibold tracking-tight text-ink">SentinelAI</div>
        <div className="text-[10px] uppercase tracking-[0.14em] text-ink-3">Network IDS</div>
      </div>
    </div>
  );
}

function EngineStatus() {
  const [state, setState] = useState<{ ok: boolean; version?: string } | null>(null);
  useEffect(() => {
    let alive = true;
    const check = () =>
      fetch("/api/health", { cache: "no-store" })
        .then((r) => (r.ok ? r.json() : Promise.reject()))
        .then((h) => alive && setState({ ok: true, version: h.model_version }))
        .catch(() => alive && setState({ ok: false }));
    check();
    const id = setInterval(check, 30_000);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);
  return (
    <div className="flex items-center gap-2 font-mono text-[11px] text-ink-3">
      <Activity className={clsx("h-3.5 w-3.5", state?.ok ? "text-[var(--c-bot)]" : state ? "text-critical" : "text-ink-3")} aria-hidden />
      {state == null ? "Connecting…" : state.ok ? <>Online<span className="hidden sm:inline"> · model v{state.version}</span></> : "Offline"}
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
      {NAV.map(({ href, label, icon: Icon }) => {
        const active = href === "/" ? pathname === "/" : pathname.startsWith(href);
        return (
          <Link key={href} href={href} aria-current={active ? "page" : undefined}
            className={clsx("flex items-center gap-2.5 rounded-md px-2.5 py-2 text-[13px] transition-colors",
              active ? "bg-surface-3 text-ink" : "text-ink-2 hover:bg-surface-2 hover:text-ink")}>
            <Icon className={clsx("h-4 w-4", active ? "text-accent" : "text-ink-3")} aria-hidden />
            {label}
          </Link>
        );
      })}
    </nav>
  );

  return (
    <div className="flex min-h-dvh">
      <aside className="sticky top-0 hidden h-dvh w-60 shrink-0 flex-col border-r border-line bg-surface lg:flex">
        <div className="px-4 py-5"><Logo /></div>
        <div className="px-2">{nav}</div>
        <div className="mt-auto border-t border-line px-4 py-4 text-[11px] leading-relaxed text-ink-3">
          Trained on CIC-IDS2017 flow features.
          <br />
          Predictions are decision support, not ground truth.
        </div>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-20 flex h-12 items-center justify-between gap-4 border-b border-line bg-bg/90 px-4 backdrop-blur lg:px-8">
          <button className="rounded p-1 text-ink-2 lg:hidden" onClick={() => setOpen(true)} aria-label="Open navigation">
            <Menu className="h-5 w-5" />
          </button>
          <div className="lg:hidden"><Logo /></div>
          <div className="ml-auto"><EngineStatus /></div>
        </header>
        <main className="mx-auto w-full max-w-[1400px] flex-1 px-4 py-6 lg:px-8">{children}</main>
      </div>

      {open && (
        <div className="fixed inset-0 z-40 lg:hidden" role="dialog" aria-modal="true" aria-label="Navigation">
          <div className="absolute inset-0 bg-black/60" onClick={() => setOpen(false)} />
          <div className="absolute inset-y-0 left-0 w-64 border-r border-line bg-surface p-3">
            <div className="mb-4 flex items-center justify-between px-1 py-2">
              <Logo />
              <button onClick={() => setOpen(false)} aria-label="Close navigation" className="rounded p-1 text-ink-2">
                <X className="h-5 w-5" />
              </button>
            </div>
            {nav}
          </div>
        </div>
      )}
    </div>
  );
}
