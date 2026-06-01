(() => {
  const VERSION = "flow-control-custom-media-1";
  const MEDIA_TYPES = new Set(["trackio.image", "trackio.table"]);
  const MAX_LOG_POINTS = 10000;
  const state = {
    mediaDir: "",
    runs: [],
    selectedRunNames: [],
    items: [],
    controls: {
      mode: "image",
      imageIdQuery: "",
      imageIdMatch: "substring",
      minStep: "",
      maxStep: "",
      search: "",
    },
    cache: new Map(),
    renderKey: "",
    loadingKey: "",
    scheduleTimer: null,
    galleryTimer: null,
  };

  function isMediaPath() {
    return (window.location.pathname.replace(/\/+$/, "") || "/") === "/media";
  }

  function escapeHtml(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
  }

  async function api(name, payload = {}) {
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

  function debounceSchedule(delay = 120) {
    window.clearTimeout(state.scheduleTimer);
    state.scheduleTimer = window.setTimeout(renderIfNeeded, delay);
  }

  function pageContent() {
    return document.querySelector(".page-content");
  }

  function currentProject() {
    const locked = document.querySelector(".locked-project");
    if (locked?.textContent?.trim() && locked.textContent.trim() !== "-") {
      return locked.textContent.trim();
    }
    const projectInput = document.querySelector('input[aria-label="Project"]');
    return projectInput?.value?.trim() || new URLSearchParams(window.location.search).get("project") || "";
  }

  function selectedRunNamesFromSidebar() {
    const names = [];
    document.querySelectorAll(".sidebar .checkbox-item").forEach((label) => {
      const input = label.querySelector('input[type="checkbox"]');
      const nameEl = label.querySelector(".run-name");
      if (input?.checked && nameEl) {
        const name = nameEl.getAttribute("title") || nameEl.textContent;
        if (name?.trim()) names.push(name.trim());
      }
    });
    return [...new Set(names)];
  }

  function runKey(run) {
    return run.id || run.name;
  }

  function colorForIndex(index) {
    const colors = ["#A8769B", "#E89957", "#3B82F6", "#10B981", "#EF4444", "#8B5CF6", "#14B8A6", "#F59E0B"];
    return colors[((index % colors.length) + colors.length) % colors.length];
  }

  function mediaUrl(path) {
    if (!path) return "";
    const pathString = String(path);
    const resolved = pathString.startsWith("/") || /^[A-Za-z]:[\\/]/.test(pathString)
      ? pathString
      : state.mediaDir
        ? `${state.mediaDir.replace(/\/$/, "")}/${pathString}`
        : pathString;
    return `/file?path=${encodeURIComponent(resolved)}`;
  }

  function mediaIdFromKey(key) {
    const parts = String(key || "").split("/");
    return parts.length ? parts[parts.length - 1] : String(key || "");
  }

  function isMediaObject(value) {
    return value && typeof value === "object" && MEDIA_TYPES.has(value._type);
  }

  function extractMediaItems(runLogs) {
    const items = [];
    for (const { run, logs, color } of runLogs) {
      logs.forEach((log, logIndex) => {
        const step = Number.isFinite(log.step) ? log.step : logIndex;
        for (const [key, value] of Object.entries(log)) {
          if (!isMediaObject(value)) continue;
          const base = {
            key,
            imageId: key,
            shortId: mediaIdFromKey(key),
            step,
            runName: run.name || "Unnamed run",
            runId: runKey(run),
            runColor: color,
            ...value,
          };
          if (value._type === "trackio.image") {
            items.push(base);
          } else if (value._type === "trackio.table" && Array.isArray(value._value)) {
            value._value.forEach((row, rowIndex) => {
              Object.entries(row || {}).forEach(([column, cell]) => {
                const cells = Array.isArray(cell) ? cell : [cell];
                cells.forEach((nested, nestedIndex) => {
                  if (nested?._type === "trackio.image") {
                    items.push({
                      ...base,
                      ...nested,
                      key: `${key}/${column}/${rowIndex}/${nestedIndex}`,
                      imageId: `${key}/${column}/${rowIndex}/${nestedIndex}`,
                      shortId: `${column}:${rowIndex}`,
                      tableKey: key,
                    });
                  }
                });
              });
            });
          }
        }
      });
    }
    return items.sort((left, right) => {
      if (left.step !== right.step) return left.step - right.step;
      if (left.imageId !== right.imageId) return left.imageId.localeCompare(right.imageId);
      return left.runName.localeCompare(right.runName);
    });
  }

  async function loadSettings() {
    if (state.mediaDir) return;
    try {
      const settings = await api("get_settings");
      state.mediaDir = settings?.media_dir || "";
    } catch {
      state.mediaDir = "";
    }
  }

  async function loadRuns(project, selectedNames) {
    const allRuns = await api("get_runs_for_project", { project });
    const selected = selectedNames.length
      ? allRuns.filter((run) => selectedNames.includes(run.name))
      : [];
    state.runs = allRuns;
    return selected;
  }

  async function loadLogs(project, runs) {
    const missing = runs.filter((run) => !state.cache.has(`${project}\u0000${runKey(run)}`));
    if (missing.length) {
      const batch = await api("get_logs_batch", {
        project,
        runs: missing.map((run) => ({ run: run.name, run_id: run.id })),
        max_points: MAX_LOG_POINTS,
      });
      for (const entry of batch) {
        const run = missing.find((candidate) => candidate.id === entry.run_id || candidate.name === entry.run);
        if (run) state.cache.set(`${project}\u0000${runKey(run)}`, entry.logs || []);
      }
    }
    return runs.map((run, index) => ({
      run,
      color: colorForIndex(state.runs.findIndex((candidate) => runKey(candidate) === runKey(run)) || index),
      logs: state.cache.get(`${project}\u0000${runKey(run)}`) || [],
    }));
  }

  function distinct(items, field) {
    return [...new Set(items.map((item) => item[field]).filter((value) => value != null && value !== ""))];
  }

  function numericInput(value, fallback) {
    if (value === "" || value == null) return fallback;
    const num = Number(value);
    return Number.isFinite(num) ? num : fallback;
  }

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

  function card(item) {
    return `
      <article class="fc-media-card">
        <a class="fc-media-img-wrap" href="${mediaUrl(item.file_path)}" target="_blank" rel="noopener noreferrer">
          <img src="${mediaUrl(item.file_path)}" alt="${escapeHtml(item.caption || item.imageId)}" loading="lazy">
        </a>
        <div class="fc-media-card-meta">
          <div class="fc-media-label" title="${escapeHtml(item.imageId)}">${escapeHtml(item.imageId)}</div>
          <div class="fc-media-mini-row"><span class="fc-run-dot" style="background:${item.runColor}"></span>${escapeHtml(item.runName)}</div>
          <div>step ${escapeHtml(item.step)}</div>
          ${item.caption ? `<div class="fc-caption">${escapeHtml(item.caption)}</div>` : ""}
        </div>
      </article>
    `;
  }

  function renderControls(host) {
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

  function renderGallery(host) {
    const filtered = filteredItems();
    const subtitle = host.querySelector(".fc-media-subtitle");
    const gallery = host.querySelector(".fc-media-gallery");
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
        .map(([step, stepItems]) => `
          <details class="fc-media-group" open>
            <summary><span>step ${escapeHtml(step)}</span><span>${stepItems.length} image${stepItems.length === 1 ? "" : "s"}</span></summary>
            <div class="fc-media-grid">${stepItems.map(card).join("")}</div>
          </details>
        `)
        .join("") || '<div class="fc-empty-panel">No images match this step range.</div>';
      return;
    }
    const imageGroups = groupItems(filtered, (item) => item.imageId);
    gallery.innerHTML = [...imageGroups.entries()]
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([imageId, imageItems]) => {
        const runGroups = groupItems(imageItems, (item) => item.runId);
        const lanes = [...runGroups.values()].map((runItems) => {
          const first = runItems[0];
          return `
            <section class="fc-media-run-lane">
              <h3><span class="fc-run-dot" style="background:${first.runColor}"></span>${escapeHtml(first.runName)} <small>${runItems.length} frame${runItems.length === 1 ? "" : "s"}</small></h3>
              <div class="fc-media-strip">${runItems.map(card).join("")}</div>
            </section>
          `;
        }).join("");
        return `
          <details class="fc-media-group" open>
            <summary><span>${escapeHtml(imageId)}</span><span>${imageItems.length} image${imageItems.length === 1 ? "" : "s"}</span></summary>
            ${lanes}
          </details>
        `;
      })
      .join("") || '<div class="fc-empty-panel">No images match this image-id/step slice.</div>';
  }

  function bindControls(host) {
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

  function shell(project) {
    return `
      <div class="fc-custom-media-root" data-fc-version="${VERSION}">
        <header class="fc-media-header">
          <p class="fc-eyebrow">Flow Control Dashboard</p>
          <h1>Media</h1>
          <p class="fc-media-subtitle">Loading Trackio media for ${escapeHtml(project)}.</p>
        </header>
        <div class="fc-media-controls"></div>
        <div class="fc-media-gallery"><div class="fc-empty-panel">Loading media...</div></div>
      </div>
    `;
  }

  async function renderIfNeeded() {
    if (!isMediaPath()) return;
    const content = pageContent();
    if (!content) return;
    const project = currentProject();
    const selectedNames = selectedRunNamesFromSidebar();
    const key = `${project}\u0000${selectedNames.join("\u0001")}`;
    if (!project) return;
    if (state.renderKey === key && content.querySelector(".fc-custom-media-root")) return;
    if (state.loadingKey === key) return;
    state.loadingKey = key;
    content.innerHTML = shell(project);
    const host = content.querySelector(".fc-custom-media-root");
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
})();
