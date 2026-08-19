/* Ported from ui_helpers.py - these are functional/data colors, not part of
   the structural theme, and must stay exactly in sync with the Python
   originals (same RGB anchors, same pitch-type hexes). */
const GRAD_BLUE = [37, 99, 235];
const GRAD_GREY = [156, 163, 175];
const GRAD_RED = [220, 38, 38];

function lerp(c1, c2, t) {
  return c1.map((v, i) => Math.round(v + (c2[i] - v) * t));
}

function gradientColor(pct) {
  pct = Math.max(0, Math.min(100, pct));
  const [r, g, b] = pct <= 50
    ? lerp(GRAD_BLUE, GRAD_GREY, pct / 50)
    : lerp(GRAD_GREY, GRAD_RED, (pct - 50) / 50);
  return `rgb(${r},${g},${b})`;
}

function hexToRgba(hex, alpha) {
  hex = hex.replace("#", "");
  const r = parseInt(hex.slice(0, 2), 16);
  const g = parseInt(hex.slice(2, 4), 16);
  const b = parseInt(hex.slice(4, 6), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

const PITCH_COLORS = {
  FF: "#d62728", CU: "#1f77b4", FC: "#8c564b", SI: "#ff7f0e",
  CH: "#2ca02c", FS: "#40E0D0", ST: "#b35900", SL: "#e6c200",
};
const DEFAULT_PITCH_COLOR = "#999999";

function pitchColor(pitchType) {
  return PITCH_COLORS[pitchType] || DEFAULT_PITCH_COLOR;
}
