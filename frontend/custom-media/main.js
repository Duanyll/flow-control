// Entry point for the custom Media overlay. Mounts our gallery on top of
// Trackio's native "/media" page, keeps it in sync with sidebar/route changes,
// and tears itself down when the user navigates away.

import { state, VERSION } from "./state.js";
import { escapeHtml, isMediaPath, pageContent } from "./utils.js";
import { currentProject, selectedRunNamesFromSidebar } from "./trackio-dom.js";
import { extractMediaItems, loadLogs, loadRuns, loadSettings } from "./media-data.js";
import { bindControls, renderControls, renderGallery } from "./gallery.js";
import { resetImageLoader } from "./image-loader.js";

// Stable, collision-free key identifying "what should currently be rendered".
function renderKeyFor(project, selectedNames) {
  return JSON.stringify([project, selectedNames]);
}

function debounceSchedule(delay = 120) {
  window.clearTimeout(state.scheduleTimer);
  state.scheduleTimer = window.setTimeout(renderIfNeeded, delay);
}

function shell(project) {
  return `
    <header class="fc-media-header">
      <p class="fc-eyebrow">Flow Control Dashboard</p>
      <h1>Media</h1>
      <p class="fc-media-subtitle">Loading Trackio media for ${escapeHtml(project)}.</p>
    </header>
    <div class="fc-media-controls"></div>
    <div class="fc-media-gallery"><div class="fc-empty-panel">Loading media...</div></div>
  `;
}

function ensureOverlayHost(content) {
  content.classList.add("fc-media-active");
  let host = content.querySelector(":scope > .fc-custom-media-root");
  if (!host) {
    host = document.createElement("div");
    host.className = "fc-custom-media-root";
    host.dataset.fcVersion = VERSION;
    content.appendChild(host);
  }
  return host;
}

function teardownOverlay() {
  // Restore the native Trackio page: drop our overlay and unhide its content.
  // We never clobber `.page-content` itself, so Svelte's render anchor survives
  // and other tabs keep working.
  resetImageLoader(); // free any connections still held by gallery image loads
  document
    .querySelectorAll(".page-content.fc-media-active")
    .forEach((node) => node.classList.remove("fc-media-active"));
  document
    .querySelectorAll(".fc-custom-media-root")
    .forEach((node) => node.remove());
  state.renderKey = "";
  state.loadingKey = "";
}

async function renderIfNeeded() {
  const content = pageContent();
  if (!isMediaPath()) {
    teardownOverlay();
    return;
  }
  if (!content) return;
  const project = currentProject();
  const selectedNames = selectedRunNamesFromSidebar();
  const key = renderKeyFor(project, selectedNames);
  if (!project) return;
  if (state.renderKey === key && content.querySelector(":scope > .fc-custom-media-root")) return;
  if (state.loadingKey === key) return;
  state.loadingKey = key;
  const host = ensureOverlayHost(content);
  host.innerHTML = shell(project);
  try {
    await loadSettings();
    const runs = await loadRuns(project, selectedNames);
    state.selectedRunNames = runs.map((run) => run.name);
    if (!runs.length) {
      host.querySelector(".fc-media-subtitle").textContent = "Select one or more runs in the sidebar.";
      host.querySelector(".fc-media-controls").innerHTML = "";
      host.querySelector(".fc-media-gallery").innerHTML = '<div class="fc-empty-panel">No runs selected.</div>';
    } else {
      const logs = await loadLogs(project, runs);
      state.items = extractMediaItems(logs);
      renderControls(host);
      bindControls(host);
      renderGallery(host);
    }
    state.renderKey = key;
  } catch (error) {
    console.error("Flow Control media overlay failed:", error);
    host.querySelector(".fc-media-subtitle").textContent = "Could not load Trackio media.";
    host.querySelector(".fc-media-gallery").innerHTML = `<div class="fc-empty-panel">Media loading failed: ${escapeHtml(error.message || error)}</div>`;
  } finally {
    state.loadingKey = "";
  }
}

function install() {
  const observer = new MutationObserver(() => debounceSchedule());
  observer.observe(document.body, { childList: true, subtree: true, attributes: true, attributeFilter: ["checked", "value", "class"] });
  window.addEventListener("popstate", () => debounceSchedule(0));
  window.addEventListener("pushstate", () => debounceSchedule(0));
  document.addEventListener("change", () => {
    state.renderKey = "";
    debounceSchedule(0);
  }, true);
  setInterval(() => debounceSchedule(), 1000);
  debounceSchedule(500);
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", install, { once: true });
} else {
  install();
}
