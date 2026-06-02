// Readers that scrape the host Trackio page (project name, selected runs, and
// the colors Trackio assigns each run in the sidebar). These are the only places
// that depend on Trackio's native DOM structure.

export function currentProject() {
  const locked = document.querySelector(".locked-project");
  if (locked?.textContent?.trim() && locked.textContent.trim() !== "-") {
    return locked.textContent.trim();
  }
  const projectInput = document.querySelector('input[aria-label="Project"]');
  return projectInput?.value?.trim() || new URLSearchParams(window.location.search).get("project") || "";
}

export function selectedRunNamesFromSidebar() {
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

export function sidebarRunColors() {
  const map = new Map();
  document.querySelectorAll(".sidebar .checkbox-item").forEach((label) => {
    const nameEl = label.querySelector(".run-name");
    const dot = label.querySelector(".color-dot");
    if (!nameEl || !dot) return;
    const name = (nameEl.getAttribute("title") || nameEl.textContent || "").trim();
    if (!name) return;
    const color =
      dot.style.backgroundColor ||
      window.getComputedStyle(dot).backgroundColor ||
      "";
    if (color && color !== "rgba(0, 0, 0, 0)" && color !== "transparent") {
      map.set(name, color);
    }
  });
  return map;
}
