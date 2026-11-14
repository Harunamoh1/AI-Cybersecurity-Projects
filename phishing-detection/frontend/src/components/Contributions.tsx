import type { Contribution } from "@/lib/api";
import { formatFeatureValue } from "@/lib/format";

/**
 * Diverging bars: how far each feature shifted the model's log-odds of phishing
 * relative to a typical legitimate URL (occlusion explanation from the backend).
 * Log-odds are additive-ish and do not saturate, unlike probabilities.
 */
export function Contributions({ items }: { items: Contribution[] }) {
  if (!items.length) {
    return <p className="text-sm text-ink-3">Every feature sits at the value typical of legitimate URLs; nothing moved the score.</p>;
  }
  const max = Math.max(...items.map((c) => Math.abs(c.impact)), 0.5);
  return (
    <div>
      <div className="mb-3 flex items-center justify-between font-mono text-[10.5px] uppercase tracking-wider text-ink-3">
        <span className="flex items-center gap-1.5"><span className="h-2 w-2 rounded-sm bg-legit" />Toward legitimate</span>
        <span className="flex items-center gap-1.5">Toward phishing<span className="h-2 w-2 rounded-sm bg-phish" /></span>
      </div>
      <ul className="space-y-2.5">
        {items.map((c) => {
          const w = (Math.abs(c.impact) / max) * 50;
          const pos = c.impact > 0;
          return (
            <li key={c.feature} className="group">
              <div className="mb-1 flex items-baseline justify-between gap-3 text-[12.5px]">
                <span className="truncate text-ink">{c.description}</span>
                <span className="shrink-0 font-mono text-[11.5px] tabular-nums text-ink-2">
                  {pos ? "+" : "−"}
                  {Math.abs(c.impact).toFixed(2)}
                </span>
              </div>
              <div className="relative h-2 rounded-full bg-surface-2">
                <span className="absolute inset-y-[-3px] left-1/2 w-px bg-line-strong" />
                <span
                  className="absolute inset-y-0 rounded-full"
                  style={{
                    background: pos ? "var(--series-phish)" : "var(--series-legit)",
                    left: pos ? "50%" : `${50 - w}%`,
                    width: `${w}%`,
                  }}
                />
              </div>
              <div className="mt-1 font-mono text-[10.5px] text-ink-3">
                value {formatFeatureValue(c.value)} · typical legit {formatFeatureValue(c.reference)}
              </div>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
