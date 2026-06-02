// Data layer: turning Trackio run logs into flat media items, resolving media
// file URLs, and the API calls that load settings, runs, and batched logs.

import { MAX_LOG_POINTS, MEDIA_TYPES, state } from "./state.js";
import { api } from "./utils.js";
import { sidebarRunColors } from "./trackio-dom.js";

export function runKey(run) {
  return run.id || run.name;
}

// Stable, collision-free key for the per-(project, run) log cache.
function cacheKey(project, run) {
  return JSON.stringify([project, runKey(run)]);
}

export function colorForIndex(index) {
  const colors = ["#A8769B", "#E89957", "#3B82F6", "#10B981", "#EF4444", "#8B5CF6", "#14B8A6", "#F59E0B"];
  return colors[((index % colors.length) + colors.length) % colors.length];
}

export function mediaUrl(path) {
  if (!path) return "";
  const pathString = String(path);
  const resolved = pathString.startsWith("/") || /^[A-Za-z]:[\\/]/.test(pathString)
    ? pathString
    : state.mediaDir
      ? `${state.mediaDir.replace(/\/$/, "")}/${pathString}`
      : pathString;
  return `/file?path=${encodeURIComponent(resolved)}`;
}

export function mediaIdFromKey(key) {
  const parts = String(key || "").split("/");
  return parts.length ? parts[parts.length - 1] : String(key || "");
}

export function isMediaObject(value) {
  return value && typeof value === "object" && MEDIA_TYPES.has(value._type);
}

export function extractMediaItems(runLogs) {
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

export async function loadSettings() {
  if (state.mediaDir) return;
  try {
    const settings = await api("get_settings");
    state.mediaDir = settings?.media_dir || "";
  } catch {
    state.mediaDir = "";
  }
}

export async function loadRuns(project, selectedNames) {
  const allRuns = await api("get_runs_for_project", { project });
  const selected = selectedNames.length
    ? allRuns.filter((run) => selectedNames.includes(run.name))
    : [];
  state.runs = allRuns;
  return selected;
}

export async function loadLogs(project, runs) {
  const missing = runs.filter((run) => !state.cache.has(cacheKey(project, run)));
  if (missing.length) {
    const batch = await api("get_logs_batch", {
      project,
      runs: missing.map((run) => ({ run: run.name, run_id: run.id })),
      max_points: MAX_LOG_POINTS,
    });
    for (const entry of batch) {
      const run = missing.find((candidate) => candidate.id === entry.run_id || candidate.name === entry.run);
      if (run) state.cache.set(cacheKey(project, run), entry.logs || []);
    }
  }
  const sidebarColors = sidebarRunColors();
  return runs.map((run, index) => {
    const fullIndex = state.runs.findIndex((candidate) => runKey(candidate) === runKey(run));
    const color = sidebarColors.get(run.name) || colorForIndex(fullIndex >= 0 ? fullIndex : index);
    return {
      run,
      color,
      logs: state.cache.get(cacheKey(project, run)) || [],
    };
  });
}
