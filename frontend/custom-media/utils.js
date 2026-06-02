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
