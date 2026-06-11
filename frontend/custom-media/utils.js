// Small, dependency-free helpers: routing checks, HTML escaping, the JSON API
// client, and numeric/array utilities used across the overlay.

export function isMediaPath() {
  return (window.location.pathname.replace(/\/+$/, "") || "/") === "/media";
}

export function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

export async function api(name, payload = {}) {
  const response = await fetch(`/api/${name}`, {
    method: "POST",
    cache: "no-store",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const json = await response.json();
  if (!response.ok || json.error) {
    throw new Error(json.error || `Request failed for ${name}`);
  }
  return json.data;
}

export function pageContent() {
  return document.querySelector(".page-content");
}

export function distinct(items, field) {
  return [...new Set(items.map((item) => item[field]).filter((value) => value != null && value !== ""))];
}

export function numericInput(value, fallback) {
  if (value === "" || value == null) return fallback;
  const num = Number(value);
  return Number.isFinite(num) ? num : fallback;
}

export function clampNumber(value, lo, hi) {
  return Math.min(hi, Math.max(lo, value));
}

// Parse a step-filter expression into a predicate over integer step values.
//
// Grammar: comma-separated terms, each matching a step value (terms OR together):
//   N        exact step N
//   A:B      inclusive range A..B (any available step in [A, B])
//   A:S:B    arithmetic stride — A, A+S, A+2S, … up to B (value-based membership)
// Any bound may be omitted for open-ended forms: ":B", "A:", "A:S:", ":S:B".
// Whitespace is ignored.
//
// Examples: "1" → every step · "0:40:200" → 0,40,80,…,200 ·
//           "0,1,5,10:50,50:20:200" → those exacts/ranges/strides combined.
//
// Returns { empty, valid, match }:
//   empty — no usable expression (caller shows everything)
//   valid — every term parsed; when false the caller should show everything
//   match — (step:number) => boolean
export function parseStepFilter(expr) {
  const text = String(expr ?? "").trim();
  const showAll = { empty: true, valid: true, match: () => true };
  if (!text) return showAll;

  // "" → null (open bound); a malformed token → undefined (parse error).
  const bound = (token) => {
    const trimmed = token.trim();
    if (trimmed === "") return null;
    return /^[-+]?\d+$/.test(trimmed) ? Number(trimmed) : undefined;
  };
  const invalid = { empty: false, valid: false, match: () => true };

  const predicates = [];
  for (const raw of text.split(",")) {
    const term = raw.trim();
    if (term === "") continue;
    const parts = term.split(":");
    if (parts.length === 1) {
      const n = bound(parts[0]);
      if (n == null) return invalid; // empty or malformed single value
      predicates.push((step) => step === n);
    } else if (parts.length === 2) {
      const lo = bound(parts[0]);
      const hi = bound(parts[1]);
      if (lo === undefined || hi === undefined) return invalid;
      const min = lo ?? -Infinity;
      const max = hi ?? Infinity;
      predicates.push((step) => step >= min && step <= max);
    } else if (parts.length === 3) {
      const start = bound(parts[0]);
      const stride = bound(parts[1]);
      const stop = bound(parts[2]);
      if (start === undefined || stride === undefined || stop === undefined) return invalid;
      if (stride == null || stride <= 0) return invalid;
      const base = start ?? 0;
      const min = start ?? -Infinity;
      const max = stop ?? Infinity;
      predicates.push(
        (step) => step >= min && step <= max && Number.isInteger((step - base) / stride),
      );
    } else {
      return invalid;
    }
  }
  if (predicates.length === 0) return showAll;
  return { empty: false, valid: true, match: (step) => predicates.some((p) => p(step)) };
}
