import { validate } from "./schema.mjs";
import { aggregate } from "./aggregate.mjs";

const MAX_BODY = 2048;
const json = (status, body, extra = {}) =>
  new Response(body === null ? null : JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json", "access-control-allow-origin": "*", ...extra },
  });

// Fixed window per key, in memory. Tests and local runs use it; the Worker uses Cloudflare's binding instead.
export class MemoryLimiter {
  constructor(limit = 20, windowMs = 3600_000, now = Date.now) {
    Object.assign(this, { limit, windowMs, now, hits: new Map() });
  }

  async allow(key) {
    const t = this.now();
    for (const [k, v] of this.hits) if (t - v.start >= this.windowMs) this.hits.delete(k);
    const cur = this.hits.get(key) ?? { start: t, count: 0 };
    cur.count += 1;
    this.hits.set(key, cur);
    return cur.count <= this.limit;
  }
}

// The caller address is read only to pick a rate limit bucket. It is never stored, logged or passed on.
const caller = (req) => req.headers.get("cf-connecting-ip") || req.headers.get("x-real-ip") || "unknown";

export async function handle(req, { store, limiter, now = () => new Date() }) {
  const path = new URL(req.url).pathname;
  if (path === "/v1/stats") {
    if (req.method === "OPTIONS") return json(204, null, { "access-control-allow-methods": "GET" });
    if (req.method !== "GET") return json(405, { error: "method" });
    return json(200, aggregate(await store.groups()), { "cache-control": "public, max-age=300" });
  }
  if (path !== "/v1/report") return json(404, { error: "not found" });
  if (req.method !== "POST") return json(405, { error: "method" });
  if (!(req.headers.get("content-type") || "").startsWith("application/json")) return json(415, { error: "type" });
  if (Number(req.headers.get("content-length") || 0) > MAX_BODY) return json(413, { error: "size" });
  if (!(await limiter.allow(caller(req)))) return json(429, { error: "rate" }, { "retry-after": "3600" });
  const raw = await req.text();
  if (raw.length > MAX_BODY) return json(413, { error: "size" });
  let report = null;
  try { report = validate(JSON.parse(raw)); } catch { /* falls through to the rejection */ }
  if (!report) return json(400, { error: "schema" });
  await store.add(report, now().toISOString().slice(0, 7));
  return json(204, null);
}
