import clsx from "clsx";
import type { Components, Indicator } from "@/lib/api";

interface Segment {
  key: string;
  label: string;
  text: string;
  className: string;
}

/** Highlight indicator tokens inside a segment of the URL. */
function Highlighted({ text, tokens }: { text: string; tokens: string[] }) {
  const usable = tokens.filter((t) => t.length >= 2 && text.toLowerCase().includes(t.toLowerCase()));
  if (!usable.length) return <>{text}</>;
  const escaped = usable.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  const parts = text.split(new RegExp(`(${escaped.join("|")})`, "gi"));
  return (
    <>
      {parts.map((p, i) =>
        usable.some((t) => t.toLowerCase() === p.toLowerCase()) ? (
          <mark key={i} className="rounded-[3px] bg-critical/20 px-0.5 text-ink underline decoration-critical decoration-2 underline-offset-[3px]">
            {p}
          </mark>
        ) : (
          <span key={i}>{p}</span>
        ),
      )}
    </>
  );
}

export function UrlAnatomy({ components, indicators }: { components: Components; indicators: Indicator[] }) {
  const c = components;
  const domainPart = c.is_ip ? c.host : [c.domain, c.suffix].filter(Boolean).join(".");
  const segments: Segment[] = [
    { key: "scheme", label: "Scheme", text: c.scheme ? `${c.scheme}://` : "", className: "text-ink-3" },
    { key: "subdomain", label: "Subdomain", text: c.subdomain ? `${c.subdomain}.` : "", className: "text-ink-2" },
    { key: "domain", label: c.is_ip ? "IP host" : "Registered domain", text: domainPart, className: "font-semibold text-ink" },
    { key: "port", label: "Port", text: c.port ? `:${c.port}` : "", className: "text-high" },
    { key: "path", label: "Path", text: c.path, className: "text-ink-2" },
    { key: "query", label: "Query", text: c.query ? `?${c.query}` : "", className: "text-ink-3" },
  ].filter((s) => s.text);
  const tokens = indicators.map((i) => i.token);

  return (
    <div>
      <div className="overflow-x-auto rounded-lg border border-line bg-bg px-4 py-3.5">
        <code className="block whitespace-pre-wrap break-all font-mono text-[13.5px] leading-7">
          {segments.map((s) => (
            <span key={s.key} className={clsx(s.className, s.key === "domain" && "border-b border-dashed border-ink-3 pb-0.5")}>
              <Highlighted text={s.text} tokens={tokens} />
            </span>
          ))}
        </code>
      </div>
      <dl className="mt-3 grid grid-cols-2 gap-x-4 gap-y-2 sm:grid-cols-3">
        {segments
          .filter((s) => s.key !== "scheme")
          .map((s) => (
            <div key={s.key} className="min-w-0">
              <dt className="label">{s.label}</dt>
              <dd className="truncate font-mono text-xs text-ink-2" title={s.text}>
                {s.text}
              </dd>
            </div>
          ))}
        {c.hosting_platform && (
          <div className="min-w-0">
            <dt className="label">Hosting platform</dt>
            <dd className="truncate font-mono text-xs text-ink-2">{c.hosting_platform}</dd>
          </div>
        )}
      </dl>
    </div>
  );
}
