"use client";

import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { ClassName } from "@/lib/api";
import { ATTACK_CLASSES, CLASS_META, compact, int } from "@/lib/format";
import { Legend } from "./ui";

export function TooltipBox({ title, rows }: { title?: string; rows: { label: string; value: string; color?: string }[] }) {
  return (
    <div className="min-w-40 rounded-md border border-line-strong bg-surface-2 px-3 py-2 text-xs shadow-xl">
      {title && <div className="mb-1.5 font-medium text-ink">{title}</div>}
      <div className="space-y-1">
        {rows.map((r) => (
          <div key={r.label} className="flex items-center justify-between gap-4">
            <span className="flex items-center gap-1.5 text-ink-2">
              {r.color && <span className="h-2 w-2 rounded-[2px]" style={{ background: r.color }} />}
              {r.label}
            </span>
            <span className="tnum font-mono text-ink">{r.value}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

/** Horizontal, direct-labelled bars: one per category, sorted by value. */
export function BarList({ items, total, format = int }: {
  items: { key: string; label: React.ReactNode; value: number; color: string }[];
  total?: number;
  format?: (n: number) => string;
}) {
  const max = Math.max(1, ...items.map((i) => i.value));
  return (
    <ul className="space-y-2.5">
      {items.map((i) => (
        <li key={i.key} className="group" title={`${i.key}: ${format(i.value)}`}>
          <div className="mb-1 flex items-center justify-between text-xs">
            <span className="text-ink-2">{i.label}</span>
            <span className="tnum font-mono text-ink">
              {format(i.value)}
              {total ? <span className="ml-2 text-ink-3">{((i.value / total) * 100).toFixed(1)}%</span> : null}
            </span>
          </div>
          <div className="h-2 overflow-hidden rounded-[3px] bg-surface-3">
            <div className="h-full rounded-[3px] transition-[width] group-hover:brightness-125"
              style={{ width: `${(i.value / max) * 100}%`, background: i.color, minWidth: i.value ? 3 : 0 }} />
          </div>
        </li>
      ))}
    </ul>
  );
}

interface TrendPoint {
  date: string;
  total: number;
  threats: number;
  by_class: Partial<Record<ClassName, number>>;
}

export function DetectionTrend({ data }: { data: TrendPoint[] }) {
  const rows = data.map((d) => ({ date: d.date.slice(5), ...Object.fromEntries(ATTACK_CLASSES.map((c) => [c, d.by_class[c] ?? 0])), benign: d.by_class.BENIGN ?? 0 }));
  const present = ATTACK_CLASSES.filter((c) => rows.some((r) => (r as Record<string, number | string>)[c]));
  return (
    <div>
      <div className="h-56">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={rows} margin={{ top: 4, right: 4, left: -8, bottom: 0 }} barCategoryGap="22%">
            <CartesianGrid vertical={false} />
            <XAxis dataKey="date" tickLine={false} axisLine={false} interval="preserveStartEnd" />
            <YAxis tickLine={false} axisLine={false} tickFormatter={compact} width={44} allowDecimals={false} />
            <Tooltip
              cursor={{ fill: "rgba(255,255,255,0.04)" }}
              content={({ active, payload, label }) => {
                if (!active || !payload?.length) return null;
                const p = payload[0].payload as Record<string, number>;
                const rs = present.filter((c) => p[c]).map((c) => ({ label: CLASS_META[c].short, value: int(p[c]), color: CLASS_META[c].color }));
                return <TooltipBox title={`${label} · ${int(present.reduce((s, c) => s + (p[c] ?? 0), 0))} threats`}
                  rows={[...rs, { label: "Benign flows", value: int(p.benign) }]} />;
              }}
            />
            {present.map((c, i) => (
              <Bar key={c} dataKey={c} stackId="t" fill={CLASS_META[c].color} stroke="var(--surface)" strokeWidth={1}
                radius={i === present.length - 1 ? [3, 3, 0, 0] : 0} maxBarSize={36} isAnimationActive={false} />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>
      {present.length > 0 && <div className="mt-3"><Legend items={present.map((c) => ({ label: CLASS_META[c].short, color: CLASS_META[c].color }))} /></div>}
    </div>
  );
}
