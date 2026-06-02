// Concurrency-capped, viewport-aware image loader.
//
// Native `loading="lazy"` lets the browser fire an unbounded number of /file
// requests as soon as images are near the viewport. On a large gallery (a single
// project can hold thousands of media items) that saturates the ~6 HTTP/1.1
// connections the page shares with Trackio's data API — so switching to the
// Metrics tab starves its `/api/*` requests and charts fill in very slowly.
//
// Instead we load only on-screen images and never keep more than MAX_CONCURRENT
// requests in flight, always leaving connections free for the rest of the app.
// Gallery <img> tags carry their URL in `data-fc-src` (no `src`) until loaded.

const MAX_CONCURRENT = 4;

let observer = null;
const pending = new Set(); // visible images waiting for a free slot
const inFlight = new Set(); // images currently downloading

function ensureObserver() {
  if (!observer) {
    observer = new IntersectionObserver(onIntersect, { rootMargin: "200px 0px" });
  }
  return observer;
}

function onIntersect(entries) {
  for (const entry of entries) {
    if (entry.isIntersecting && entry.target.dataset.fcSrc) {
      pending.add(entry.target);
    } else {
      pending.delete(entry.target);
    }
  }
  pump();
}

function pump() {
  for (const img of pending) {
    if (inFlight.size >= MAX_CONCURRENT) break;
    pending.delete(img);
    startLoad(img);
  }
}

function startLoad(img) {
  const src = img.dataset.fcSrc;
  if (!src) return;
  inFlight.add(img);
  const done = () => {
    img.removeEventListener("load", done);
    img.removeEventListener("error", done);
    inFlight.delete(img);
    observer?.unobserve(img);
    pump();
  };
  img.addEventListener("load", done);
  img.addEventListener("error", done);
  delete img.dataset.fcSrc;
  img.src = src;
}

// Start managing every not-yet-loaded image found under `host`.
export function observeImages(host) {
  const obs = ensureObserver();
  host.querySelectorAll("img[data-fc-src]").forEach((img) => obs.observe(img));
}

// Stop watching, abort in-flight downloads, and release their connections.
// Called whenever the gallery is rebuilt or the overlay is torn down.
export function resetImageLoader() {
  observer?.disconnect();
  for (const img of inFlight) {
    img.removeAttribute("src"); // aborts the request, freeing the connection
  }
  pending.clear();
  inFlight.clear();
}
