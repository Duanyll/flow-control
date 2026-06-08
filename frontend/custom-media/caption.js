// Prompt/caption handling: render captions with a uniform line clamp so cards
// stay compact, then reveal the full prompt in a modal when the text overflows.

import { escapeHtml } from "./utils.js";

export function captionText(item) {
  const value = item?.caption;
  return value == null ? "" : String(value).trim();
}

export function distinctCaptions(items) {
  return [...new Set(items.map(captionText).filter(Boolean))];
}

// Captions always render as a plain block clamped to a few lines by CSS. The
// full text lives in the DOM (so the modal can read it back); overflowing
// captions get flagged by `markClampedCaptions` once laid out.
export function captionBlock(text, { group = false } = {}) {
  if (!text) return "";
  const cls = group ? "fc-caption fc-caption-group" : "fc-caption";
  return `<div class="${cls}" data-fc-caption>${escapeHtml(text)}</div>`;
}

// After the gallery HTML is in the DOM, flag captions whose text is taller than
// the clamp so they get the "click to expand" affordance and the modal handler.
export function markClampedCaptions(root) {
  root.querySelectorAll(".fc-caption[data-fc-caption]").forEach((el) => {
    const clipped = el.scrollHeight - el.clientHeight > 1;
    el.classList.toggle("fc-caption-clip", clipped);
    if (clipped) {
      el.title = "Click to view the full prompt";
    } else {
      el.removeAttribute("title");
    }
  });
}

let modal = null;

function ensureModal() {
  if (modal) return modal;
  const root = document.createElement("div");
  root.className = "fc-caption-modal";
  root.setAttribute("hidden", "");
  root.innerHTML = `
    <div class="fc-caption-modal-panel" role="dialog" aria-modal="true" aria-label="Prompt">
      <button type="button" class="fc-caption-modal-close" data-fc-caption-modal="close" title="Close (Esc)" aria-label="Close">&times;</button>
      <div class="fc-caption-modal-body"></div>
    </div>
  `;
  document.body.appendChild(root);
  root.addEventListener("click", (event) => {
    if (event.target === root || event.target?.dataset?.fcCaptionModal === "close") {
      closeCaptionModal();
    }
  });
  window.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !root.hasAttribute("hidden")) {
      event.preventDefault();
      closeCaptionModal();
    }
  });
  modal = { root, body: root.querySelector(".fc-caption-modal-body") };
  return modal;
}

export function openCaptionModal(text) {
  const { root, body } = ensureModal();
  body.textContent = text;
  root.removeAttribute("hidden");
  document.body.classList.add("fc-caption-modal-open");
}

function closeCaptionModal() {
  if (!modal) return;
  modal.root.setAttribute("hidden", "");
  document.body.classList.remove("fc-caption-modal-open");
}
