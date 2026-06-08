// Gallery rendering: filtering items by the control panel, grouping them by
// image-id or step, building cards/captions/chips, and wiring the control inputs.

import { state } from "./state.js";
import { distinct, escapeHtml, numericInput } from "./utils.js";
import { mediaUrl } from "./media-data.js";
import { observeImages, resetImageLoader } from "./image-loader.js";
import { openViewer } from "./viewer.js";
import {
  captionBlock,
  captionText,
  distinctCaptions,
  markClampedCaptions,
  openCaptionModal,
} from "./caption.js";

function imageIdMatches(imageId) {
  const query = state.controls.imageIdQuery.trim();
  if (!query) return true;
  if (state.controls.imageIdMatch === "regex") {
    try {
      return new RegExp(query, "i").test(imageId);
    } catch {
      return false;
    }
  }
  return imageId.toLowerCase().includes(query.toLowerCase());
}

function filteredItems() {
  const allSteps = state.items.map((item) => item.step).filter(Number.isFinite);
  const fallbackMin = allSteps.length ? Math.min(...allSteps) : 0;
  const fallbackMax = allSteps.length ? Math.max(...allSteps) : 0;
  const minStep = numericInput(state.controls.minStep, fallbackMin);
  const maxStep = numericInput(state.controls.maxStep, fallbackMax);
  const search = state.controls.search.trim().toLowerCase();
  return state.items.filter((item) => {
    if (state.controls.mode === "image" && !imageIdMatches(item.imageId)) return false;
    if (Number.isFinite(item.step) && item.step < minStep) return false;
    if (Number.isFinite(item.step) && item.step > maxStep) return false;
    if (search) {
      const haystack = [item.imageId, item.caption, item.runName, item.step].join(" ").toLowerCase();
      if (!haystack.includes(search)) return false;
    }
    return true;
  });
}

function groupItems(items, keyFn) {
  const groups = new Map();
  for (const item of items) {
    const key = keyFn(item);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(item);
  }
  return groups;
}

function fieldVaries(items, fn) {
  if (items.length <= 1) return false;
  const first = fn(items[0]);
  return items.some((item) => fn(item) !== first);
}

function constChips(items, which) {
  const first = items[0];
  const chips = [];
  if (which.includes("imageId") && !fieldVaries(items, (i) => i.imageId)) {
    chips.push(`<span class="fc-chip" title="${escapeHtml(first.imageId)}">${escapeHtml(first.imageId)}</span>`);
  }
  if (which.includes("run") && !fieldVaries(items, (i) => i.runId)) {
    chips.push(`<span class="fc-chip"><span class="fc-run-dot" style="background:${first.runColor}"></span>${escapeHtml(first.runName)}</span>`);
  }
  if (which.includes("step") && !fieldVaries(items, (i) => String(i.step))) {
    chips.push(`<span class="fc-chip">step ${escapeHtml(first.step)}</span>`);
  }
  return chips.join("");
}

function cardHtml(item, index, fields) {
  const parts = [];
  if (fields.label) {
    parts.push(
      `<button type="button" class="fc-media-label fc-jump" data-fc-jump="image" data-fc-value="${escapeHtml(item.imageId)}" title="Show “${escapeHtml(item.imageId)}” across steps">${escapeHtml(item.imageId)}</button>`,
    );
  }
  if (fields.run) {
    parts.push(`<div class="fc-media-mini-row"><span class="fc-run-dot" style="background:${item.runColor}"></span>${escapeHtml(item.runName)}</div>`);
  }
  if (fields.step) {
    parts.push(
      `<button type="button" class="fc-media-step fc-jump" data-fc-jump="step" data-fc-value="${escapeHtml(item.step)}" title="Show all images at step ${escapeHtml(item.step)}">step ${escapeHtml(item.step)} ↗</button>`,
    );
  }
  if (fields.caption) {
    parts.push(captionBlock(captionText(item)));
  }
  const meta = parts.join("");
  // `data-fc-src` (not `src`) defers the fetch to the concurrency-capped loader
  // in image-loader.js, so the gallery never monopolizes the connection pool.
  return `
    <article class="fc-media-card">
      <a class="fc-media-img-wrap" href="${mediaUrl(item.file_path)}" data-fc-index="${index}" target="_blank" rel="noopener noreferrer">
        <img data-fc-src="${mediaUrl(item.file_path)}" alt="${escapeHtml(captionText(item) || item.imageId)}" decoding="async">
      </a>
      ${meta ? `<div class="fc-media-card-meta">${meta}</div>` : ""}
    </article>
  `;
}

export function renderControls(host) {
  const steps = distinct(state.items, "step").filter(Number.isFinite).sort((left, right) => left - right);
  const defaultMin = steps.length ? steps[0] : 0;
  const defaultMax = steps.length ? steps[steps.length - 1] : 0;
  host.querySelector(".fc-media-controls").innerHTML = `
    <label class="fc-control-field">
      <span>View</span>
      <select data-fc-control="mode">
        <option value="image" ${state.controls.mode === "image" ? "selected" : ""}>Image ID over steps</option>
        <option value="step" ${state.controls.mode === "step" ? "selected" : ""}>Step range grid</option>
      </select>
    </label>
    <label class="fc-control-field fc-control-wide">
      <span>Image ID search</span>
      <input data-fc-control="imageIdQuery" type="search" placeholder="e.g. validation/0 or validation/[0-9]+" value="${escapeHtml(state.controls.imageIdQuery)}" ${state.controls.mode !== "image" ? "disabled" : ""}>
    </label>
    <label class="fc-control-field fc-control-compact">
      <span>Match</span>
      <select data-fc-control="imageIdMatch" ${state.controls.mode !== "image" ? "disabled" : ""}>
        <option value="substring" ${state.controls.imageIdMatch === "substring" ? "selected" : ""}>Substring</option>
        <option value="regex" ${state.controls.imageIdMatch === "regex" ? "selected" : ""}>Regex</option>
      </select>
    </label>
    <label class="fc-control-field">
      <span>Min step</span>
      <input data-fc-control="minStep" type="number" step="1" min="${defaultMin}" max="${defaultMax}" placeholder="${defaultMin}" value="${escapeHtml(state.controls.minStep)}">
    </label>
    <label class="fc-control-field">
      <span>Max step</span>
      <input data-fc-control="maxStep" type="number" step="1" min="${defaultMin}" max="${defaultMax}" placeholder="${defaultMax}" value="${escapeHtml(state.controls.maxStep)}">
    </label>
    <label class="fc-control-field fc-control-wide">
      <span>Search</span>
      <input data-fc-control="search" type="search" placeholder="caption, run, image id" value="${escapeHtml(state.controls.search)}">
    </label>
    <div class="fc-button-row">
      <button class="fc-secondary-button" type="button" data-fc-control="expandAll">Expand all</button>
      <button class="fc-secondary-button" type="button" data-fc-control="collapseAll">Collapse all</button>
      <button class="fc-secondary-button" type="button" data-fc-control="reset">Reset</button>
    </div>
  `;
}

function setAllGroupsOpen(host, open) {
  host.querySelectorAll(".fc-media-group").forEach((details) => {
    details.open = open;
  });
}

function scheduleGalleryRender(host, delay = 320) {
  window.clearTimeout(state.galleryTimer);
  state.galleryTimer = window.setTimeout(() => renderGallery(host), delay);
}

export function renderGallery(host) {
  const filtered = filteredItems();
  const subtitle = host.querySelector(".fc-media-subtitle");
  const gallery = host.querySelector(".fc-media-gallery");
  resetImageLoader(); // abort loads from the previous render before rebuilding
  state.viewerItems = [];
  const card = (item, fields) => {
    const index = state.viewerItems.length;
    state.viewerItems.push(item);
    return cardHtml(item, index, fields);
  };
  gallery.onclick = (event) => {
    const jump = event.target?.closest?.("[data-fc-jump]");
    if (jump) {
      event.preventDefault();
      focusAndSwitch(host, jump.dataset.fcJump, jump.dataset.fcValue);
      return;
    }
    const caption = event.target?.closest?.(".fc-caption-clip");
    if (caption) {
      event.preventDefault();
      openCaptionModal(caption.textContent);
      return;
    }
    const wrap = event.target?.closest?.(".fc-media-img-wrap");
    if (!wrap) return;
    if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    const index = Number(wrap.dataset.fcIndex);
    if (Number.isFinite(index)) openViewer(index);
  };
  const imageIdCount = distinct(state.items, "imageId").length;
  const stepCount = distinct(state.items, "step").length;
  subtitle.textContent = `${filtered.length}/${state.items.length} images across ${imageIdCount} image IDs and ${stepCount} steps for ${state.selectedRunNames.length} selected run${state.selectedRunNames.length === 1 ? "" : "s"}.`;
  if (!state.items.length) {
    gallery.innerHTML = '<div class="fc-empty-panel">No Trackio images found in the selected runs.</div>';
    return;
  }
  if (state.controls.mode === "step") {
    const stepGroups = groupItems(filtered, (item) => item.step);
    gallery.innerHTML = [...stepGroups.entries()]
      .sort(([a], [b]) => Number(a) - Number(b))
      .map(([step, stepItems]) => {
        const captions = distinctCaptions(stepItems);
        const groupCaption = captions.length === 1 ? captionBlock(captions[0], { group: true }) : "";
        const fields = {
          label: fieldVaries(stepItems, (i) => i.imageId),
          run: fieldVaries(stepItems, (i) => i.runId),
          step: false,
          caption: captions.length > 1,
        };
        const chips = constChips(stepItems, ["imageId", "run"]);
        const body = stepItems.map((item) => card(item, fields)).join("");
        return `
          <details class="fc-media-group" open data-fc-group="${escapeHtml(step)}">
            <summary><span class="fc-group-head"><span class="fc-group-title">step ${escapeHtml(step)}</span>${chips}</span><span class="fc-group-count">${stepItems.length} image${stepItems.length === 1 ? "" : "s"}</span></summary>
            ${groupCaption}
            <div class="fc-media-grid">${body}</div>
          </details>
        `;
      })
      .join("") || '<div class="fc-empty-panel">No images match this step range.</div>';
    markClampedCaptions(gallery);
    applyFocus(gallery);
    observeImages(gallery);
    return;
  }
  const imageGroups = groupItems(filtered, (item) => item.imageId);
  gallery.innerHTML = [...imageGroups.entries()]
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([imageId, imageItems]) => {
      const captions = distinctCaptions(imageItems);
      const groupCaption = captions.length === 1 ? captionBlock(captions[0], { group: true }) : "";
      const captionPerCard = captions.length > 1;
      const runGroups = groupItems(imageItems, (item) => item.runId);
      const lanes = [...runGroups.values()].map((runItems) => {
        const first = runItems[0];
        const fields = { label: false, run: false, step: true, caption: captionPerCard };
        return `
          <section class="fc-media-run-lane">
            <h3><span class="fc-run-dot" style="background:${first.runColor}"></span>${escapeHtml(first.runName)} <small>${runItems.length} frame${runItems.length === 1 ? "" : "s"}</small></h3>
            <div class="fc-media-strip">${runItems.map((item) => card(item, fields)).join("")}</div>
          </section>
        `;
      }).join("");
      return `
        <details class="fc-media-group" open data-fc-group="${escapeHtml(imageId)}">
          <summary><span class="fc-group-head"><span class="fc-group-title" title="${escapeHtml(imageId)}">${escapeHtml(imageId)}</span></span><span class="fc-group-count">${imageItems.length} image${imageItems.length === 1 ? "" : "s"}</span></summary>
          ${groupCaption}
          ${lanes}
        </details>
      `;
    })
    .join("") || '<div class="fc-empty-panel">No images match this image-id/step slice.</div>';
  markClampedCaptions(gallery);
  applyFocus(gallery);
  observeImages(gallery);
}

function focusAndSwitch(host, mode, value) {
  if (mode !== "step" && mode !== "image") return;
  state.controls.mode = mode;
  state.focus = String(value ?? "");
  renderControls(host);
  bindControls(host);
  renderGallery(host);
}

function applyFocus(gallery) {
  const target = state.focus;
  state.focus = null; // one-shot: only the explicit jump collapses other groups
  if (target == null) return;
  let matched = null;
  gallery.querySelectorAll(".fc-media-group").forEach((details) => {
    const isMatch = details.getAttribute("data-fc-group") === target;
    details.open = isMatch;
    if (isMatch) matched = details;
  });
  if (matched) {
    matched.classList.add("fc-group-focused");
    window.setTimeout(() => matched.classList.remove("fc-group-focused"), 1600);
    matched.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

export function bindControls(host) {
  const controls = host.querySelector(".fc-media-controls");
  controls.oninput = (event) => {
    const key = event.target?.dataset?.fcControl;
    if (!key || !(key in state.controls)) return;
    state.controls[key] = event.target.value;
    const delay = key === "imageIdQuery" || key === "search" ? 450 : 300;
    scheduleGalleryRender(host, delay);
  };
  controls.onchange = (event) => {
    const key = event.target?.dataset?.fcControl;
    if (!key || !(key in state.controls)) return;
    state.controls[key] = event.target.value;
    if (key === "mode") {
      renderControls(host);
      bindControls(host);
    }
    scheduleGalleryRender(host, key === "imageIdMatch" ? 120 : 260);
  };
  controls.onclick = (event) => {
    const action = event.target?.dataset?.fcControl;
    if (action === "expandAll") {
      setAllGroupsOpen(host, true);
    } else if (action === "collapseAll") {
      setAllGroupsOpen(host, false);
    } else if (action === "reset") {
      state.controls.imageIdQuery = "";
      state.controls.imageIdMatch = "substring";
      state.controls.minStep = "";
      state.controls.maxStep = "";
      state.controls.search = "";
      renderControls(host);
      bindControls(host);
      renderGallery(host);
    }
  };
}
