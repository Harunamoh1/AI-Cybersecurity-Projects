export function Logo({ className = "" }: { className?: string }) {
  return (
    <svg viewBox="0 0 32 32" className={className} aria-hidden="true">
      <circle cx="14" cy="14" r="8" fill="none" stroke="currentColor" strokeWidth="2.5" />
      <path d="M6.5 14h15" stroke="currentColor" strokeWidth="1.25" strokeDasharray="1.5 2" opacity="0.6" />
      <path d="M20 20l6.5 6.5" stroke="var(--risk-critical)" strokeWidth="3" strokeLinecap="round" />
    </svg>
  );
}
