// Shared constants and the single mutable UI-state object for the custom Media
// overlay. Every other module imports `state` by reference and mutates its
// fields in place (the object identity never changes).

export const VERSION = "flow-control-custom-media-1";
export const MEDIA_TYPES = new Set(["trackio.image", "trackio.table"]);
export const MAX_LOG_POINTS = 10000;

export const state = {
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
    stepFilter: "",
    search: "",
  },
  cache: new Map(),
  viewerItems: [],
  focus: null,
  renderKey: "",
  loadingKey: "",
  scheduleTimer: null,
  galleryTimer: null,
};
