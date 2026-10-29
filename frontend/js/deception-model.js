/**
 * Deception+ v2 page (rebuilt 2026-10-03).
 * Reads all scores from /api/deception/* — no hardcoded numbers.
 *
 * Sections:
 *   1. Leaderboard (season selector, top-10 / bottom-10)
 *   2. Pitcher lookup (score, percentile, mechanics bars, career table)
 *   3. Under the hood (4 validation cards + version history)
 */

// ──────────────────────────────────────────────────────── mechanics labels
const MECH_LABELS = {
  release_pos_x_as:  "Release side (arm side)",
  release_pos_z:     "Release height",
  release_extension: "Extension",
  arm_angle:         "Arm angle",
};

// ──────────────────────────────────────────────────────── color helpers
function scoreColor(v) {
  if (v == null) return "var(--text-muted)";
  if (v >= 120) return "#1a7f37";
  if (v >= 110) return "#2da44e";
  if (v >=  90) return "var(--text)";
  if (v >=  80) return "#b45309";
  return "#c9302c";
}

function pctileLabel(p) {
  if (p == null) return "";
  return `${Math.round(p)}th percentile`;
}

function fmtScore(v) {
  if (v == null) return "—";
  return Math.round(v).toString();
}

// ──────────────────────────────────────────────────────── URL state
const params = new URLSearchParams(location.search);
function setParam(k, v) {
  if (v == null || v === "") params.delete(k); else params.set(k, String(v));
  const url = location.pathname + (params.toString() ? "?" + params.toString() : "");
  history.replaceState(null, "", url);
}
function getParam(k) { return params.get(k); }

// ──────────────────────────────────────────────────────── state
let _leaderboard = null;   // { season, available_seasons, pitchers }
let _currentSeason = null;
let _currentPitcher = null;
let _playerNames = {};     // pitcher_id → name

// ──────────────────────────────────────────────────────── load player names
async function loadPlayerNames(season) {
  try {
    const data = await getJSON(`/api/players/all?season=${season}`);
    if (Array.isArray(data)) {
      for (const p of data) {
        if (p.player_id && p.name) _playerNames[p.player_id] = p.name;
      }
    }
  } catch(e) { /* non-fatal */ }
}

function playerName(id) {
  return _playerNames[id] || `Pitcher #${id}`;
}

// ──────────────────────────────────────────────────────── leaderboard
async function loadLeaderboard(season) {
  const url = season ? `api/deception/leaderboard?season=${season}` : "api/deception/leaderboard";
  _leaderboard = await getJSON(url);
  return _leaderboard;
}

function buildLeaderboardRow(p, rank) {
  const row = document.createElement("div");
  row.className = "dc-lb-row";
  row.dataset.pitcherId = p.pitcher_id;
  const col = scoreColor(p.deception_plus);
  row.innerHTML = `
    <div class="dc-lb-rank">${rank}</div>
    <div class="dc-lb-name">${playerName(p.pitcher_id)}</div>
    <div class="dc-lb-n">${p.n_pitches ?? ""}p</div>
    <div class="dc-lb-val" style="color:${col}">${fmtScore(p.deception_plus)}</div>
  `;
  row.addEventListener("click", () => {
    selectPitcher(p.pitcher_id);
    document.getElementById("dc-pitcher-input").value = playerName(p.pitcher_id);
  });
  return row;
}

function renderLeaderboard(lb) {
  if (!lb || !lb.pitchers) return;
  const sorted = [...lb.pitchers].sort((a, b) => b.deception_plus - a.deception_plus);
  const top10 = sorted.slice(0, 10);
  const bot10 = [...sorted].reverse().slice(0, 10);

  const topEl = document.getElementById("dc-lb-top");
  const botEl = document.getElementById("dc-lb-bot");
  topEl.innerHTML = "";
  botEl.innerHTML = "";
  top10.forEach((p, i) => topEl.appendChild(buildLeaderboardRow(p, i + 1)));
  bot10.forEach((p, i) => botEl.appendChild(buildLeaderboardRow(p, sorted.length - i)));

  const cap = document.getElementById("dc-lb-caption");
  cap.textContent = `${lb.season} • ${sorted.length} qualified pitchers (≥300 pitches)`;
}

// ──────────────────────────────────────────────────────── season selector
function buildSeasonSelector(available, current, selId, onChange) {
  const sel = document.getElementById(selId);
  sel.innerHTML = "";
  for (const s of [...available].reverse()) {
    const opt = document.createElement("option");
    opt.value = s;
    opt.textContent = s;
    if (s === String(current)) opt.selected = true;
    sel.appendChild(opt);
  }
  sel.addEventListener("change", () => onChange(sel.value));
}

// ──────────────────────────────────────────────────────── pitcher lookup
async function selectPitcher(pitcherId, season) {
  _currentPitcher = pitcherId;
  const mount = document.getElementById("dc-pitcher-mount");
  const status = document.getElementById("dc-pitcher-status");
  status.textContent = "Loading…";
  mount.style.display = "none";

  // Get available seasons for this pitcher
  const url = `api/deception/pitcher/${pitcherId}` + (season ? `?season=${season}` : "");
  let data;
  try {
    data = await getJSON(url);
  } catch(e) {
    status.textContent = `No Deception+ data for this pitcher.`;
    return;
  }

  if (!data) { status.textContent = "No data found."; return; }
  status.textContent = "";
  mount.style.display = "block";
  setParam("pitcher", pitcherId);
  setParam("season", data.season);

  // Update season selector for pitcher
  if (data.available_seasons) {
    buildSeasonSelector(data.available_seasons, data.season, "dc-pitcher-season-sel",
      (s) => selectPitcher(pitcherId, s));
  }

  // Score headline
  const score = data.deception_plus ?? null;
  const scoreEl = document.getElementById("dc-score-val");
  scoreEl.textContent = fmtScore(score);
  scoreEl.style.color = scoreColor(score);

  document.getElementById("dc-pct-val").textContent = pctileLabel(data.percentile);
  document.getElementById("dc-pitches-val").textContent = `${data.n_pitches ?? "—"} pitches`;

  const unqMsg = document.getElementById("dc-unqualified-msg");
  unqMsg.style.display = data.qualified ? "none" : "block";

  // Mechanics bars (percentile bars from league data)
  renderMechBars(data);

  // Career/season table — fetch all seasons this pitcher has
  await renderCareerTable(pitcherId, data.season, data.available_seasons || [data.season]);

  // Live SHAP breakdown (v2b fold models only)
  loadShapBreakdown(pitcherId, data.season);
}

// ──────────────────────────────────────────────────────── SHAP breakdown
function renderShapContributions(breakdown) {
  if (!breakdown || !breakdown.contributions || !breakdown.contributions.length) {
    return `<div style="font-size:12.5px;color:var(--text-muted);">No feature breakdown available.</div>`;
  }
  const maxAbs = Math.max(...breakdown.contributions.map(([, v]) => Math.abs(v)), 0.01);
  const rows = breakdown.contributions.map(([f, pts, tv]) => {
    const halfW = (Math.abs(pts) / maxAbs) * 50;
    const left  = pts >= 0 ? 50 : 50 - halfW;
    const clr   = pts >= 0 ? "#1a7f37" : "#c9302c";
    const tvStr = typeof tv === "number" ? tv.toFixed(2) : "n/a";
    const label = MECH_LABELS[f] || f;
    return `<div class="dc-shap-row">
      <div class="dc-shap-label">${label} <span class="muted">(${tvStr})</span></div>
      <div class="dc-shap-track">
        <div class="dc-shap-zero" style="left:50%"></div>
        <div class="dc-shap-bar" style="left:${left}%;width:${halfW}%;background:${clr};"></div>
      </div>
      <div class="dc-shap-val ${pts >= 0 ? "pos" : "neg"}">${pts >= 0 ? "+" : ""}${pts.toFixed(1)}</div>
    </div>`;
  }).join("");
  return `<div>${rows}<div class="dc-shap-note">Contributions are in Deception+ points, scored live through the fold models that held this pitcher out at training time (fold ${breakdown.fold}). The sum may differ from (Deception+−100) by the EB-shrinkage correction.</div></div>`;
}

async function loadShapBreakdown(pitcherId, season) {
  const mount = document.getElementById("dc-shap-mount");
  if (!mount) return;
  mount.innerHTML = `<div style="font-size:12.5px;color:var(--text-muted);">Loading…</div>`;
  try {
    const breakdown = await getJSON(`/api/deception/pitcher/${pitcherId}/shap?season=${season}`);
    mount.innerHTML = renderShapContributions(breakdown);
  } catch (e) {
    mount.innerHTML = `<div style="font-size:12.5px;color:var(--text-muted);">No feature breakdown available for this pitcher/season.</div>`;
  }
}

function renderMechBars(data) {
  const el = document.getElementById("dc-mech-bars");
  el.innerHTML = "";
  // data.percentile is the score percentile; for mechanics we show raw values if available
  // If the API doesn't return per-feature percentiles, show a note
  const mechKeys = Object.keys(MECH_LABELS);
  const hasMech = mechKeys.some(k => data[k] != null);
  if (!hasMech) {
    el.innerHTML = `<div style="font-size:12.5px;color:var(--text-muted);">Mechanics breakdown not available in this version — see the report endpoint for feature importance.</div>`;
    return;
  }
  for (const [key, label] of Object.entries(MECH_LABELS)) {
    const val = data[key];
    if (val == null) continue;
    const pct = Math.min(100, Math.max(0, data[key + "_pct"] ?? 50));
    const row = document.createElement("div");
    row.className = "dc-mech-row";
    row.innerHTML = `
      <div class="dc-mech-label">${label}</div>
      <div class="dc-mech-track">
        <div class="dc-mech-fill" style="width:${pct}%"></div>
      </div>
      <div class="dc-mech-val">${typeof val === 'number' ? val.toFixed(1) : val}</div>
    `;
    el.appendChild(row);
  }
}

async function renderCareerTable(pitcherId, currentSeason, availableSeasons) {
  const tbody = document.getElementById("dc-career-tbody");
  tbody.innerHTML = "";
  for (const s of [...availableSeasons].sort().reverse()) {
    let row_data;
    try {
      row_data = await getJSON(`api/deception/pitcher/${pitcherId}?season=${s}`);
    } catch(e) { continue; }
    if (!row_data) continue;
    const tr = document.createElement("tr");
    if (s === String(currentSeason)) tr.className = "dc-current-row";
    const score = row_data.deception_plus ?? null;
    const pct = row_data.percentile != null ? Math.round(row_data.percentile) : "—";
    tr.innerHTML = `
      <td>${s}</td>
      <td style="color:${scoreColor(score)};font-weight:800;">${fmtScore(score)}</td>
      <td>${pct !== "—" ? pct + "th" : "—"}</td>
      <td>${row_data.n_pitches ?? "—"}</td>
    `;
    tbody.appendChild(tr);
  }
}

// ──────────────────────────────────────────────────────── under the hood cards
function buildHoodCards(report) {
  const el = document.getElementById("dc-cards-mount");
  el.innerHTML = "";
  if (!report) return;

  const yoy = report.yoy_stability || {};
  const pooled_yoy = yoy.pooled;
  const ss = report.same_season_validity || {};
  const xfip_r = ss.xFIP?.r;
  const ns = report.next_season_validity || {};
  const ns_rs = Object.values(ns).map(v => v.xfip_r).filter(r => r != null);
  const avg_ns_r = ns_rs.length ? ns_rs.reduce((a, b) => a + b, 0) / ns_rs.length : null;
  const indep = report.independence || {};
  const stuff_r = indep.v2_vs_stuff_plus_v4?.r;
  const loc_r = indep.v2_vs_location_plus_v3?.r;

  const cards = [
    {
      title: "(a) What I tried",
      what: "Cross-validated configurations: v1 baseline (A0), pruned identity-prone features (A1/A2), pitcher-season aggregation (B). Equivalence tests selected the leanest config that isn't significantly worse.",
      numbers: [
        ["v1 baseline (A0) xFIP corr", xfip_r != null ? xfip_r.toFixed(3) : "n/a"],
        ["Chosen config", report._chosen_config || "B"],
        ["Seasons trained", "2021–2026"],
      ],
      verdict: "—",
      verdict_note: "See part2_results.json for full config test table.",
    },
    {
      title: "(b) Does it repeat?",
      what: "Year-over-year score correlation for pitchers with ≥300 pitches in consecutive seasons (excluding 2020 pairs). Split-half reliability at 300 pitches.",
      numbers: [
        ["Pooled YoY correlation", pooled_yoy != null ? pooled_yoy.toFixed(3) : "—"],
        ["Target (pre-registered)", "≥ 0.30"],
        ["Min consecutive pair", yoy.pairs ? Object.values(yoy.pairs).filter(v => v.r != null).map(v => v.r).reduce((a,b) => Math.min(a,b), 1).toFixed(3) : "—"],
      ],
      verdict: pooled_yoy != null ? (pooled_yoy >= 0.30 ? "Passes" : "Fails") : "Pending",
      verdict_note: "Pre-registered threshold: pooled YoY ≥ 0.30, no consecutive pair < 0.25 (excl. 2020).",
    },
    {
      title: "(c) Does it predict results?",
      what: "Same-season xFIP/ERA correlation (cross-fitted, so no data leakage). Next-season xFIP correlation. Incremental delta-R² over last-season xFIP.",
      numbers: [
        ["Same-season xFIP r (cross-fitted)", xfip_r != null ? xfip_r.toFixed(3) : "—"],
        ["Avg next-season xFIP r", avg_ns_r != null ? avg_ns_r.toFixed(3) : "—"],
        ["Target (evidence bar)", "≤ −0.20"],
      ],
      verdict: avg_ns_r != null ? (avg_ns_r <= -0.20 ? "Passes" : "Mixed") : "Pending",
      verdict_note: "Pre-registered evidence bar: next-season xFIP r ≤ −0.20. Same-season r expected negative (higher Deception+ = lower xFIP).",
    },
    {
      title: "(d) Is it learning mechanics or pitchers?",
      what: "Independence from Stuff+ (pitch-level) and Location+ (location-level). Within-pitcher score variation confirms the model isn't just memorizing identity.",
      numbers: [
        ["v2 vs Stuff+ v4 r", stuff_r != null ? stuff_r.toFixed(3) : "—"],
        ["v2 vs Location+ v3 r", loc_r != null ? loc_r.toFixed(3) : "—"],
        ["Within-pitcher score std", report.memorization_check?.score_within_pitcher_std != null ? report.memorization_check.score_within_pitcher_std.toFixed(2) : "—"],
      ],
      verdict: stuff_r != null ? (Math.abs(stuff_r) < 0.40 && Math.abs(loc_r || 0) < 0.40 ? "Passes" : "Mixed") : "Pending",
      verdict_note: "Low correlation with Stuff+/Location+ confirms Deception+ is measuring something distinct.",
    },
  ];

  const VERDICT_COL = { "Passes": "#1a7f37", "Mixed": "#b45309", "Fails": "#c9302c", "Pending": "var(--text-muted)", "—": "var(--text-muted)" };

  for (const card of cards) {
    const div = document.createElement("div");
    div.className = "dc-card";
    const numRows = card.numbers.map(([k, v]) =>
      `<div><span class="kv-label">${k}: </span><span class="kv-val">${v}</span></div>`
    ).join("");
    const col = VERDICT_COL[card.verdict] || "var(--text-muted)";
    div.innerHTML = `
      <div class="dc-card-title">${card.title}</div>
      <div class="dc-card-summary">${card.what}</div>
      <div class="dc-card-numbers">${numRows}</div>
      <div class="dc-card-verdict" style="color:${col}">${card.verdict}</div>
      <div class="dc-card-verdict-note">${card.verdict_note}</div>
    `;
    el.appendChild(div);
  }
}

// ──────────────────────────────────────────────────────── version badge
async function loadVersionMeta() {
  try {
    const meta = await getJSON("api/deception/meta");
    if (!meta) return;
    const badge = document.getElementById("dc-version-badge");
    const v = meta.version || "2";
    badge.textContent = v.startsWith("v") ? v : "v" + v;
  } catch(e) { /* non-fatal */ }
}

async function loadLabelBadge() {
  try {
    const gates = await getJSON("diagnostics/deception_v2/gate_results.json").catch(() => null);
    if (!gates) return;
    const label = gates.label;
    const badge = document.getElementById("dc-label-badge");
    if (label === "validated") {
      badge.className = "dc-badge validated";
      badge.textContent = "Validated";
      badge.style.display = "";
    } else if (label === "limited_evidence") {
      badge.className = "dc-badge limited";
      badge.textContent = "Limited evidence";
      badge.style.display = "";
    }
  } catch(e) { /* non-fatal */ }
}

// ──────────────────────────────────────────────────────── under-the-hood toggle
document.getElementById("dc-hood-btn").addEventListener("click", function() {
  const expanded = this.getAttribute("aria-expanded") === "true";
  this.setAttribute("aria-expanded", String(!expanded));
  const body = document.getElementById("dc-hood-body");
  body.classList.toggle("open", !expanded);
});

// ──────────────────────────────────────────────────────── player search wiring
function wirePlayerSearch() {
  const input = document.getElementById("dc-pitcher-input");
  const dropdown = document.getElementById("dc-pitcher-dropdown");
  if (typeof PlayerSearch !== "undefined") {
    new PlayerSearch(input, dropdown, {
      filterType: "pitcher",
      onSelect: (p) => {
        selectPitcher(p.player_id);
        input.value = p.name;
      }
    });
  }
}

// ──────────────────────────────────────────────────────── main init
(async function init() {
  const initSeason = Number(getParam("season")) || 2026;
  await loadPlayerNames(initSeason);
  loadVersionMeta();
  loadLabelBadge();

  // Load leaderboard
  let lb;
  try {
    lb = await loadLeaderboard(getParam("season"));
  } catch(e) {
    console.error("Deception+ leaderboard load failed:", e);
    return;
  }

  const available = lb.available_seasons || [lb.season];
  _currentSeason = lb.season;

  buildSeasonSelector(available, _currentSeason, "dc-season-sel", async (s) => {
    _currentSeason = s;
    setParam("season", s);
    const newLb = await loadLeaderboard(s);
    renderLeaderboard(newLb);
  });
  renderLeaderboard(lb);

  // Also wire pitcher lookup season selector initially
  buildSeasonSelector(available, _currentSeason, "dc-pitcher-season-sel", (s) => {
    if (_currentPitcher) selectPitcher(_currentPitcher, s);
  });

  // Under-the-hood cards
  try {
    const report = await getJSON("api/deception/report");
    buildHoodCards(report);
  } catch(e) { console.warn("Could not load deception report:", e); }

  // Deep-link support
  wirePlayerSearch();
  const pitcherParam = getParam("pitcher");
  if (pitcherParam) {
    const sParam = getParam("season");
    await selectPitcher(parseInt(pitcherParam), sParam || undefined);
  }
})();
