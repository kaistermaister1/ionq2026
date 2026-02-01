export default async function handler(req, res) {
  // Vercel serverless function proxy to bypass browser CORS.
  // Usage:
  //   /api/proxy?path=/v1/graph&base=https%3A%2F%2Fdemo-entanglement-distillation-qfhvrahfcq-uc.a.run.app

  // CORS
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET,POST,OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");
  if (req.method === "OPTIONS") {
    res.status(204).end();
    return;
  }

  const path = typeof req.query.path === "string" ? req.query.path : null;
  if (!path || !path.startsWith("/")) {
    res.status(400).json({ ok: false, error: { code: "BAD_REQUEST", message: "Missing ?path=/v1/..." } });
    return;
  }

  const defaultUpstream = "https://demo-entanglement-distillation-qfhvrahfcq-uc.a.run.app";
  const upstreamBase =
    (typeof req.query.base === "string" && req.query.base) ||
    process.env.UPSTREAM_BASE_URL ||
    defaultUpstream;

  // Minimal safety: only allow https upstream.
  if (!upstreamBase.startsWith("https://")) {
    res.status(400).json({ ok: false, error: { code: "BAD_REQUEST", message: "Upstream base must start with https://" } });
    return;
  }

  const url = upstreamBase.replace(/\/$/, "") + path;

  const headers = {
    "Content-Type": "application/json",
  };
  if (req.headers.authorization) headers["Authorization"] = req.headers.authorization;

  const init = {
    method: req.method,
    headers,
  };

  if (req.method !== "GET" && req.method !== "HEAD") {
    // Vercel parses JSON bodies for us
    init.body = JSON.stringify(req.body ?? {});
  }

  let upstreamResp;
  try {
    upstreamResp = await fetch(url, init);
  } catch (e) {
    res.status(502).json({ ok: false, error: { code: "UPSTREAM_ERROR", message: String(e?.message || e) } });
    return;
  }

  const text = await upstreamResp.text();
  res.status(upstreamResp.status);
  res.setHeader("Content-Type", upstreamResp.headers.get("content-type") || "application/json");
  res.send(text);
}

