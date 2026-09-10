/* Ported from ui_helpers.py - these are functional/data colors, not part of
   the structural theme, and must stay exactly in sync with the Python
   originals (same RGB anchors, same pitch-type hexes). */
const GRAD_BLUE = [37, 99, 235];
const GRAD_GREY = [156, 163, 175];
const GRAD_RED = [220, 38, 38];

function lerp(c1, c2, t) {
  return c1.map((v, i) => Math.round(v + (c2[i] - v) * t));
}

// Generic percentile color used everywhere a percentile bar/badge shows up
// (player page, compare, radar): blue (bad/low) - grey (average, middle) -
// red (good/high). `alpha` (0-1) gives a faint tint for backgrounds without
// needing a second hexToRgba round-trip; omit it for a solid color (badge
// text/border).
function gradientColor(pct, alpha) {
  pct = Math.max(0, Math.min(100, pct));
  const [r, g, b] = pct <= 50
    ? lerp(GRAD_BLUE, GRAD_GREY, pct / 50)
    : lerp(GRAD_GREY, GRAD_RED, (pct - 50) / 50);
  return alpha !== undefined ? `rgba(${r},${g},${b},${alpha})` : `rgb(${r},${g},${b})`;
}

// Stuff+/Deception+/Tunnel+/Location+ "plus stat" scale - a deliberately
// different color language from the generic percentile gradient above,
// reserved for these four model badges/breakdown tables. A vivid gold-
// midpoint version of this was tried and reverted per direction (didn't
// like it) - back to the original red(bad)-grey(mid)-green(good) scale.
const GRAD_RED_STUFF = [220, 38, 38];
const GRAD_GREY_STUFF = [156, 163, 175];
const GRAD_GREEN_STUFF = [26, 127, 55];
function stuffColor(pct, alpha) {
  pct = Math.max(0, Math.min(100, pct));
  const [r, g, b] = pct <= 50
    ? lerp(GRAD_RED_STUFF, GRAD_GREY_STUFF, pct / 50)
    : lerp(GRAD_GREY_STUFF, GRAD_GREEN_STUFF, (pct - 50) / 50);
  return alpha !== undefined ? `rgba(${r},${g},${b},${alpha})` : `rgb(${r},${g},${b})`;
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

// Real Statcast pitch_type codes -> full names, for tables that show a
// pitch type as a bare 2-letter code (e.g. the Pitch Table/Performance vs
// Pitch Type table) - not every code has its own PITCH_COLORS entry above,
// but every real Statcast code gets a real name here.
const PITCH_FULL_NAMES = {
  FF: "Four-Seam Fastball", SI: "Sinker", FC: "Cutter", FA: "Fastball",
  CH: "Changeup", FS: "Splitter", FO: "Forkball", SC: "Screwball",
  CU: "Curveball", KC: "Knuckle Curve", CS: "Slow Curve",
  SL: "Slider", ST: "Sweeper", SV: "Slurve",
  KN: "Knuckleball", EP: "Eephus", PO: "Pitchout", IN: "Intentional Ball",
};

function pitchFullName(pitchType) {
  const full = PITCH_FULL_NAMES[pitchType];
  return full ? `${full} (${pitchType})` : pitchType;
}
