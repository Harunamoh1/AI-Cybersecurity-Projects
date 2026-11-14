import { NextRequest } from "next/server";

/**
 * Same-origin proxy to the FastAPI backend.
 *
 * The browser only ever talks to the Next.js origin, so the backend needs no
 * public exposure or permissive CORS. BACKEND_URL is read at request time,
 * which lets the same Docker image run against any backend address.
 */
const BACKEND_URL = () => process.env.BACKEND_URL ?? "http://localhost:8000";
const ALLOWED_PREFIXES = ["health", "v1/scan", "v1/history", "v1/stats", "v1/model"];

async function proxy(req: NextRequest, ctx: { params: Promise<{ path: string[] }> }) {
  const { path } = await ctx.params;
  const joined = path.map(encodeURIComponent).join("/");
  if (!ALLOWED_PREFIXES.some((p) => joined === p || joined.startsWith(p + "/"))) {
    return Response.json({ detail: "Not found" }, { status: 404 });
  }
  const base = joined === "health" ? `${BACKEND_URL()}/health` : `${BACKEND_URL()}/api/${joined}`;
  const target = `${base}${req.nextUrl.search}`;
  const init: RequestInit = {
    method: req.method,
    headers: {
      "content-type": req.headers.get("content-type") ?? "application/json",
      "x-real-ip": req.headers.get("x-forwarded-for")?.split(",")[0]?.trim() ?? "",
    },
    cache: "no-store",
  };
  if (req.method !== "GET" && req.method !== "HEAD") init.body = await req.text();
  try {
    const res = await fetch(target, init);
    const body = await res.text();
    return new Response(body, {
      status: res.status,
      headers: { "content-type": res.headers.get("content-type") ?? "application/json" },
    });
  } catch {
    return Response.json({ detail: "Detection service unavailable" }, { status: 503 });
  }
}

export const GET = proxy;
export const POST = proxy;
export const DELETE = proxy;
export const dynamic = "force-dynamic";
