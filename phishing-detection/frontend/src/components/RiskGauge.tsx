import type { RiskLevel } from "@/lib/api";
import { RISK_META } from "@/lib/format";

const CX = 120;
const CY = 118;
const R = 92;

function polar(v: number, r = R) {
  const a = Math.PI * (1 - v);
  return { x: CX + r * Math.cos(a), y: CY - r * Math.sin(a) };
}

function arc(from: number, to: number, r = R) {
  const s = polar(from, r);
  const e = polar(to, r);
  return `M ${s.x.toFixed(2)} ${s.y.toFixed(2)} A ${r} ${r} 0 0 1 ${e.x.toFixed(2)} ${e.y.toFixed(2)}`;
}

/** Semicircular instrument gauge. Bands are anchored on the model's tuned decision threshold. */
export function RiskGauge({ probability, threshold, level }: { probability: number; threshold: number; level: RiskLevel }) {
  const bands: { level: RiskLevel; from: number; to: number }[] = [
    { level: "low", from: 0, to: 0.5 * threshold },
    { level: "medium", from: 0.5 * threshold, to: threshold },
    { level: "high", from: threshold, to: threshold + (1 - threshold) / 2 },
    { level: "critical", from: threshold + (1 - threshold) / 2, to: 1 },
  ];
  const gap = 0.006;
  const needle = polar(Math.min(Math.max(probability, 0), 1), R - 20);
  const needleBase = polar(Math.min(Math.max(probability, 0), 1), R + 9);
  const th = polar(threshold, R + 12);
  const thIn = polar(threshold, R - 12);

  return (
    <svg viewBox="0 0 240 140" className="w-full max-w-[300px]" role="img" aria-label={`Risk score ${Math.round(probability * 100)} of 100, ${RISK_META[level].label} risk`}>
      {Array.from({ length: 11 }, (_, i) => {
        const a = polar(i / 10, R + 14);
        const b = polar(i / 10, R + (i % 5 === 0 ? 20 : 17));
        return <line key={i} x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke="var(--line-strong)" strokeWidth={1} />;
      })}
      <path d={arc(0, 1)} fill="none" stroke="var(--surface-3)" strokeWidth={12} />
      {bands.map((b) => (
        <path
          key={b.level}
          d={arc(b.from + (b.from > 0 ? gap : 0), b.to - (b.to < 1 ? gap : 0))}
          fill="none"
          stroke={RISK_META[b.level].color}
          strokeWidth={12}
          opacity={b.level === level ? 1 : 0.22}
        />
      ))}
      <line x1={thIn.x} y1={thIn.y} x2={th.x} y2={th.y} stroke="var(--text)" strokeWidth={1.5} strokeDasharray="2 2" />
      <line x1={CX} y1={CY} x2={needle.x} y2={needle.y} stroke="var(--text)" strokeWidth={2} strokeLinecap="round" />
      <circle cx={needleBase.x} cy={needleBase.y} r={4} fill="var(--text)" stroke="var(--surface)" strokeWidth={2} />
      <circle cx={CX} cy={CY} r={5} fill="var(--surface)" stroke="var(--text)" strokeWidth={2} />
      <text x={CX - R} y={CY + 16} textAnchor="middle" className="fill-[var(--text-3)] font-mono text-[9px]">0</text>
      <text x={CX + R} y={CY + 16} textAnchor="middle" className="fill-[var(--text-3)] font-mono text-[9px]">100</text>
    </svg>
  );
}
