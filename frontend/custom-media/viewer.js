// Full-screen lightbox: zoom (wheel / double-click), pan (drag), and keyboard
// navigation across the currently rendered gallery items.

import { state } from "./state.js";
import { clampNumber } from "./utils.js";
import { mediaUrl } from "./media-data.js";

const MIN_SCALE = 1;
const MAX_SCALE = 16;

const viewer = {
  root: null,
  img: null,
  stage: null,
  caption: null,
  counter: null,
  items: [],
  index: 0,
  scale: 1,
  tx: 0,
  ty: 0,
  dragging: false,
  moved: false,
  startX: 0,
  startY: 0,
};

function ensureViewer() {
  if (viewer.root) return viewer.root;
  const root = document.createElement("div");
  root.className = "fc-viewer";
  root.setAttribute("hidden", "");
  root.innerHTML = `
    <button class="fc-viewer-btn fc-viewer-close" data-fc-viewer="close" title="Close (Esc)" aria-label="Close">&times;</button>
    <button class="fc-viewer-btn fc-viewer-prev" data-fc-viewer="prev" title="Previous (←)" aria-label="Previous">&#8249;</button>
    <button class="fc-viewer-btn fc-viewer-next" data-fc-viewer="next" title="Next (→)" aria-label="Next">&#8250;</button>
    <div class="fc-viewer-stage" data-fc-viewer="stage">
      <img class="fc-viewer-img" alt="" draggable="false">
    </div>
    <div class="fc-viewer-bar">
      <span class="fc-viewer-counter"></span>
      <span class="fc-viewer-caption"></span>
      <span class="fc-viewer-hint">scroll to zoom · drag to pan · double-click to reset · ← → to switch · Esc to close</span>
    </div>
  `;
  document.body.appendChild(root);
  viewer.root = root;
  viewer.img = root.querySelector(".fc-viewer-img");
  viewer.stage = root.querySelector(".fc-viewer-stage");
  viewer.caption = root.querySelector(".fc-viewer-caption");
  viewer.counter = root.querySelector(".fc-viewer-counter");

  root.addEventListener("click", (event) => {
    const action = event.target?.dataset?.fcViewer;
    if (action === "close") closeViewer();
    else if (action === "prev") viewerNav(-1);
    else if (action === "next") viewerNav(1);
  });
  viewer.stage.addEventListener("click", (event) => {
    if (event.target === viewer.img) return;
    if (viewer.moved) {
      viewer.moved = false;
      return;
    }
    closeViewer();
  });
  viewer.stage.addEventListener("wheel", onViewerWheel, { passive: false });
  viewer.stage.addEventListener("pointerdown", onViewerPointerDown);
  window.addEventListener("pointermove", onViewerPointerMove);
  window.addEventListener("pointerup", onViewerPointerUp);
  viewer.img.addEventListener("dblclick", onViewerDblClick);
  window.addEventListener("keydown", onViewerKey);
  return root;
}

function viewerOpen() {
  return viewer.root != null && !viewer.root.hasAttribute("hidden");
}

export function openViewer(index) {
  ensureViewer();
  viewer.items = (state.viewerItems || []).slice();
  if (!viewer.items.length) return;
  viewer.index = clampNumber(index, 0, viewer.items.length - 1);
  loadViewerImage();
  viewer.root.removeAttribute("hidden");
  document.body.classList.add("fc-viewer-open");
}

function closeViewer() {
  if (!viewer.root) return;
  viewer.root.setAttribute("hidden", "");
  document.body.classList.remove("fc-viewer-open");
  viewer.dragging = false;
  viewer.moved = false;
  viewer.img.removeAttribute("src");
}

function loadViewerImage() {
  const item = viewer.items[viewer.index];
  if (!item) return;
  viewer.img.src = mediaUrl(item.file_path);
  viewer.img.alt = item.caption || item.imageId || "";
  viewer.counter.textContent = `${viewer.index + 1} / ${viewer.items.length}`;
  const bits = [item.imageId, item.runName, `step ${item.step}`];
  if (item.caption) bits.push(item.caption);
  viewer.caption.textContent = bits.filter((bit) => bit != null && bit !== "").join("  ·  ");
  resetTransform();
  applyTransform();
  updateNavButtons();
}

function updateNavButtons() {
  const prev = viewer.root.querySelector(".fc-viewer-prev");
  const next = viewer.root.querySelector(".fc-viewer-next");
  if (prev) prev.disabled = viewer.index <= 0;
  if (next) next.disabled = viewer.index >= viewer.items.length - 1;
}

function viewerNav(delta) {
  if (!viewer.items.length) return;
  const next = clampNumber(viewer.index + delta, 0, viewer.items.length - 1);
  if (next === viewer.index) return;
  viewer.index = next;
  loadViewerImage();
}

function resetTransform() {
  viewer.scale = 1;
  viewer.tx = 0;
  viewer.ty = 0;
}

function applyTransform() {
  viewer.img.style.transform = `translate(${viewer.tx}px, ${viewer.ty}px) scale(${viewer.scale})`;
  viewer.img.style.cursor = viewer.scale > 1 ? "grab" : "zoom-in";
}

function zoomAt(clientX, clientY, factor) {
  const rect = viewer.stage.getBoundingClientRect();
  const cx = rect.left + rect.width / 2;
  const cy = rect.top + rect.height / 2;
  const newScale = clampNumber(viewer.scale * factor, MIN_SCALE, MAX_SCALE);
  const ratio = newScale / viewer.scale;
  const sx = clientX - cx;
  const sy = clientY - cy;
  viewer.tx = sx * (1 - ratio) + ratio * viewer.tx;
  viewer.ty = sy * (1 - ratio) + ratio * viewer.ty;
  viewer.scale = newScale;
  if (viewer.scale <= MIN_SCALE) {
    viewer.tx = 0;
    viewer.ty = 0;
  }
  applyTransform();
}

function onViewerWheel(event) {
  event.preventDefault();
  const factor = Math.exp(-event.deltaY * 0.0015);
  zoomAt(event.clientX, event.clientY, factor);
}

function onViewerDblClick(event) {
  event.preventDefault();
  if (viewer.scale > MIN_SCALE) {
    resetTransform();
    applyTransform();
  } else {
    zoomAt(event.clientX, event.clientY, 2.5);
  }
}

function onViewerPointerDown(event) {
  if (event.button !== 0) return;
  if (event.target !== viewer.img) return;
  viewer.dragging = true;
  viewer.moved = false;
  viewer.startX = event.clientX - viewer.tx;
  viewer.startY = event.clientY - viewer.ty;
  viewer.img.style.cursor = "grabbing";
}

function onViewerPointerMove(event) {
  if (!viewer.dragging) return;
  viewer.tx = event.clientX - viewer.startX;
  viewer.ty = event.clientY - viewer.startY;
  viewer.moved = true;
  applyTransform();
}

function onViewerPointerUp() {
  if (!viewer.dragging) return;
  viewer.dragging = false;
  viewer.img.style.cursor = viewer.scale > 1 ? "grab" : "zoom-in";
}

function onViewerKey(event) {
  if (!viewerOpen()) return;
  if (event.key === "Escape") closeViewer();
  else if (event.key === "ArrowLeft") viewerNav(-1);
  else if (event.key === "ArrowRight") viewerNav(1);
  else if (event.key === "+" || event.key === "=") zoomAt(window.innerWidth / 2, window.innerHeight / 2, 1.25);
  else if (event.key === "-" || event.key === "_") zoomAt(window.innerWidth / 2, window.innerHeight / 2, 1 / 1.25);
  else return;
  event.preventDefault();
}
