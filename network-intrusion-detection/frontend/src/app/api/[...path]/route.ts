import { NextRequest } from "next/server";

/**
 * Same-origin proxy to the FastAPI backend.
 *
 * - The browser only talks to the Next.js origin; the backend is not exposed.
 * - Only an allow-list of API prefixes is forwarded.
 * - If SENTINEL_API_KEY is set, it is added server-side, so the key never
 *   reaches the browser.
 */
const BACKEND_URL = () => process.env.BACKEND_URL ?? "http://localhost:8000";
const ALLOWED_PREFIXES = ["health", "v1/predict", "v1/analyses", "v1/samples", "v1/threats", "v1/stats", "v1/model"];
const MAX_BODY = 26 * 1024 * 1024;

async function proxy(req: NextRequest, ctx: { params: Promise<{ path: string[] }> }) {
  const { path } = await ctx.params;
  if (path.some((p) => p === ".." || p === "." || p.includes("/") || p.includes("\\"))) {
    return Response.json({ detail: "Not found" }, { status: 404 });
  }
  const joined = path.map(encodeURIComponent).join("/");
  if (!ALLOWED_PREFIXES.some((p) => joined === p || joined.startsWith(p + "/"))) {
    return Response.json({ detail: "Not found" }, { status: 404 });
  }
  const length = Number(req.headers.get("content-length") ?? 0);
  if (length > MAX_BODY) {
    return Response.json({ detail: "Upload too large" }, { status: 413 });
  }
  const base = joined === "health" ? `${BACKEND_URL()}/health` : `${BACKEND_URL()}/api/${joined}`;
  const headers: Record<string, string> = {
    "x-real-ip": req.headers.get("x-forwarded-for")?.split(",")[0]?.trim() ?? "",
  };
  const ctype = req.headers.get("content-type");
  if (ctype) headers["content-type"] = ctype;
  if (process.env.SENTINEL_API_KEY) headers["x-api-key"] = process.env.SENTINEL_API_KEY;
  const init: RequestInit = { method: req.method, headers, cache: "no-store" };
  if (req.method !== "GET" && req.method !== "HEAD") init.body = await req.arrayBuffer();
  try {
    const res = await fetch(`${base}${req.nextUrl.search}`, init);
    const out = new Headers({ "content-type": res.headers.get("content-type") ?? "application/json" });
    const disposition = res.headers.get("content-disposition");
    if (disposition) out.set("content-disposition", disposition);
    return new Response(await res.arrayBuffer(), { status: res.status, headers: out });
  } catch {
    return Response.json({ detail: "Detection service unavailable" }, { status: 503 });
  }
}

export const GET = proxy;
export const POST = proxy;
export const DELETE = proxy;
export const dynamic = "force-dynamic";
