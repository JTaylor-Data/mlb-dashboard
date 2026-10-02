// Trend Viewer v5 — multi-series, display cards, compatible-unit gating

const CHART_W = 720, CHART_H = 300;
const PAD = { top: 16, right: 28, bottom: 50, left: 60 };
const INNER_W = CHART_W - PAD.left - PAD.right;
const INNER_H = CHART_H - PAD.top - PAD.bottom;

const LINE_COLORS = ["#2d6a4f","#1f77b4","#e6a020","#c9302c","#8c564b","#a040c0","#17becf","#e377c2"];
const MONTH_ABBR = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sept","Oct","Nov","Dec"];

function fmtDate(iso) {
  // "2026-09-06" → "Sept 6"
  const parts = iso.split("-");
  const m = parseInt(parts[1], 10) - 1;
  const d = parseInt(parts[2], 10);
  return `${MONTH_ABBR[m]} ${d}`;
}

let state = {
  role: "pitcher",
  season: "2026",
  activeCat: "Physical",
  horizon: "days365",   // e.g. "games5", "days180", "pa50", "career"
  showLgAvg: true,
  playerId: null,
  playerName: null,
  data: null,
  lgAvgs: null,
  pitchType: null,
  activeDisplays: [], // [{key, label, unit, decimals, color}]
  compatibleUnit: null,
};

// ── player search ─────────────────────────────────────────────────────────

let _trendsSearch = null;

async function initTrendsSearch(role) {
  if (_trendsSearch) _trendsSearch.destroy();
  const inputEl = document.getElementById("player-input");
  const dropdownEl = document.getElementById("player-datalist-dropdown");
  _trendsSearch = createPlayerSearch(inputEl, dropdownEl, (p) => {
    state.playerId = p.player_id;
    state.playerName = p.name;
    state.role = p.role || role;
    document.querySelectorAll(".pill-btn[data-role]").forEach(b =>
      b.classList.toggle("active", b.dataset.role === state.role));
    state.activeDisplays = [];
    state.compatibleUnit = null;
    render();
  }, { roleFilter: role });
  const season = state.season === "all" ? "2026" : state.season;
  await _trendsSearch.load(season);
}

// ── horizon filter ────────────────────────────────────────────────────────

// horizon codes: "games5", "days180", "days365", "days1095", "career",
//                "pa50", "pa150", "pa250"
function filterByHorizon(games, horizon) {
  if (!games || !games.length || !horizon || horizon === "career") return games;

  // PA-based (batters): keep last N plate appearances by going backwards
  const paMatch = horizon.match(/^pa(\d+)$/);
  if (paMatch) {
    const target = parseInt(paMatch[1], 10);
    let accum = 0;
    let cutIdx = 0;
    for (let i = games.length - 1; i >= 0; i--) {
      accum += games[i].n_pa || 0;
      if (accum >= target) { cutIdx = i; break; }
    }
    const filtered = games.slice(cutIdx);
    return filtered.length >= 2 ? filtered : games;
  }

  // Game-count based (pitchers): last N games
  const gamesMatch = horizon.match(/^games(\d+)$/);
  if (gamesMatch) {
    const n = parseInt(gamesMatch[1], 10);
    const filtered = games.slice(-n);
    return filtered.length >= 1 ? filtered : games;
  }

  // Days-based
  const daysMatch = horizon.match(/^days(\d+)$/);
  if (daysMatch) {
    const days = parseInt(daysMatch[1], 10);
    const cutoff = new Date();
    cutoff.setDate(cutoff.getDate() - days);
    const cutStr = cutoff.toISOString().slice(0, 10);
    const filtered = games.filter(g => g.game_date >= cutStr);
    return filtered.length >= 2 ? filtered : games;
  }

  return games;
}

// ── chart rendering ───────────────────────────────────────────────────────

function buildChart(allGames, displays, horizon, lgAvgs, showLgAvg) {
  const games = filterByHorizon(allGames, horizon);
  if (!games || !games.length) {
    return `<div class="trend-empty">No data for this player / period.</div>`;
  }
  if (!displays.length) {
    return `<div class="trend-empty">Select a metric to view.</div>`;
  }

  // Collect all values for Y scale
  let allVals = [];
  for (const d of displays) {
    const vs = games.map(g => g[d.key]).filter(v => v != null);
    allVals = allVals.concat(vs);
    if (showLgAvg && lgAvgs && lgAvgs[d.key] != null) allVals.push(lgAvgs[d.key]);
  }
  if (!allVals.length) {
    return `<div class="trend-empty">No data in this period.</div>`;
  }

  let yMin = Math.min(...allVals), yMax = Math.max(...allVals);
  const span = yMax - yMin;
  const yPad = span * 0.15 || Math.abs(yMax) * 0.05 || 1;
  yMin -= yPad; yMax += yPad;

  const xStep = games.length > 1 ? INNER_W / (games.length - 1) : INNER_W;
  function xPos(i) { return PAD.left + i * xStep; }
  function yPos(v) { return PAD.top + INNER_H - ((v - yMin) / (yMax - yMin)) * INNER_H; }

  // Y ticks
  const yTicks = [];
  for (let t = 0; t <= 4; t++) {
    const v = yMin + (yMax - yMin) * (t / 4);
    yTicks.push({ v, y: yPos(v) });
  }

  // X ticks — density scales with horizon breadth
  const xTicks = [];
  const maxTicks = horizon === "career" || horizon === "days1095" ? 6
    : horizon === "days365" ? 8
    : horizon === "days180" ? 8
    : games.length <= 25 ? games.length : 10;
  if (games.length <= maxTicks) {
    games.forEach((g, i) => xTicks.push({ i, label: fmtDate(g.game_date) }));
  } else {
    const step = Math.ceil(games.length / maxTicks);
    for (let i = 0; i < games.length; i += step) {
      xTicks.push({ i, label: fmtDate(games[i].game_date) });
    }
  }

  const decimals = displays[0]?.decimals ?? 1;
  const unit = displays[0]?.unit ?? "";

  function fmtVal(v, dec) {
    if (v == null) return "—";
    return unit === "%" ? v.toFixed(dec) + "%" : v.toFixed(dec) + (unit ? " " + unit : "");
  }

  const yTickEls = yTicks.map(t => {
    const label = fmtVal(t.v, decimals);
    return `<text x="${PAD.left - 6}" y="${t.y + 4}" text-anchor="end" font-size="10" fill="var(--text-muted)">${label}</text>
      <line x1="${PAD.left}" y1="${t.y}" x2="${PAD.left + INNER_W}" y2="${t.y}" stroke="var(--border)" stroke-width="0.5"/>`;
  }).join("");

  const xTickEls = xTicks.map(t => {
    const x = xPos(t.i);
    return `<text x="${x}" y="${PAD.top + INNER_H + 15}" text-anchor="middle" font-size="9" fill="var(--text-muted)" transform="rotate(-30,${x},${PAD.top + INNER_H + 15})">${t.label}</text>`;
  }).join("");

  // Build line path
  function linePath(vals) {
    let d = "", prev = true;
    vals.forEach((v, i) => {
      if (v == null) { prev = true; return; }
      const x = xPos(i), y = yPos(v);
      d += prev ? `M${x},${y}` : ` L${x},${y}`;
      prev = false;
    });
    return d;
  }

  // Series lines
  const seriesEls = displays.map(disp => {
    const vals = games.map(g => g[disp.key]);
    const path = linePath(vals);
    return `<path d="${path}" fill="none" stroke="${disp.color}" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>`;
  }).join("");

  // League avg lines
  let lgAvgEls = "";
  if (showLgAvg && lgAvgs) {
    for (const disp of displays) {
      const lgv = lgAvgs[disp.key];
      if (lgv == null) continue;
      const ly = yPos(lgv);
      lgAvgEls += `<line x1="${PAD.left}" y1="${ly}" x2="${PAD.left + INNER_W}" y2="${ly}"
        stroke="#f09a00" stroke-width="1.5" stroke-dasharray="6,4" opacity="0.85"/>`;
    }
  }

  // Hover rects for first series only (tooltip shows all)
  const hovRects = games.map((g, i) => {
    const parts = displays.map(d => {
      const v = g[d.key];
      return v != null ? `${d.label}: ${fmtVal(v, d.decimals)}` : null;
    }).filter(Boolean).join(" | ");
    if (!parts) return "";
    const x = xPos(i);
    return `<rect class="tr-hr" x="${x - Math.max(xStep/2, 4)}" y="${PAD.top}" width="${Math.max(xStep, 8)}" height="${INNER_H}"
      fill="transparent" data-cx="${x}" data-cy="${yPos(displays[0] && g[displays[0].key] != null ? g[displays[0].key] : (yMin + yMax)/2)}" data-label="${fmtDate(g.game_date)}: ${parts}"></rect>`;
  }).join("");

  return `
    <svg class="trend-svg" viewBox="0 0 ${CHART_W} ${CHART_H}" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="xMidYMid meet">
      ${yTickEls}
      <line x1="${PAD.left}" y1="${PAD.top + INNER_H}" x2="${PAD.left + INNER_W}" y2="${PAD.top + INNER_H}" stroke="var(--border)" stroke-width="1"/>
      ${xTickEls}
      ${lgAvgEls}
      ${seriesEls}
      ${hovRects}
      <g id="tr-tt" style="display:none;">
        <rect id="tr-tt-bg" rx="4" fill="var(--surface)" stroke="var(--border)" stroke-width="1"/>
        <text id="tr-tt-txt" font-size="11" fill="var(--text)"></text>
      </g>
    </svg>`;
}

function initHover() {
  const svg = document.querySelector(".trend-svg");
  if (!svg) return;
  const ttg = document.getElementById("tr-tt");
  const ttbg = document.getElementById("tr-tt-bg");
  const tttxt = document.getElementById("tr-tt-txt");
  if (!ttg || !ttbg || !tttxt) return;

  svg.querySelectorAll(".tr-hr").forEach(rect => {
    rect.addEventListener("mouseenter", () => {
      const label = rect.dataset.label;
      const cx = parseFloat(rect.dataset.cx);
      const cy = parseFloat(rect.dataset.cy);
      tttxt.textContent = label;
      const tw = label.length * 6.2 + 14;
      ttbg.setAttribute("width", tw);
      ttbg.setAttribute("height", 20);
      let tx = cx + 8, ty = cy - 28;
      if (tx + tw > CHART_W - PAD.right) tx = cx - tw - 8;
      if (ty < PAD.top) ty = cy + 8;
      ttbg.setAttribute("x", tx); ttbg.setAttribute("y", ty);
      tttxt.setAttribute("x", tx + 6); tttxt.setAttribute("y", ty + 14);
      ttg.style.display = "";
    });
    rect.addEventListener("mouseleave", () => { ttg.style.display = "none"; });
  });
}

// ── metric grid ───────────────────────────────────────────────────────────

function getMetricsForCat(cat) {
  return (window._trendsMetrics || []).filter(m => m.category === cat);
}

function buildMetricGrid() {
  const metrics = getMetricsForCat(state.activeCat);
  const grid = document.getElementById("metric-grid");

  if (state.activeCat === "Pitch Type") {
    renderPitchTypeSelector();
  } else {
    document.getElementById("pitch-type-selector").style.display = "none";
  }

  if (!metrics.length) {
    grid.innerHTML = `<div style="font-size:12px;color:var(--text-muted);padding:4px 0;">No metrics in this category.</div>`;
    return;
  }

  grid.innerHTML = metrics.map(m => {
    const isActive = state.activeDisplays.some(d => d.key === m.key);
    const isIncompat = state.compatibleUnit !== null && m.unit !== state.compatibleUnit && !isActive;
    return `<button class="metric-btn${isActive ? " active" : ""}${isIncompat ? " incompatible" : ""}" data-key="${m.key}" data-unit="${m.unit}">${m.label}</button>`;
  }).join("");

  grid.querySelectorAll(".metric-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      if (btn.classList.contains("incompatible")) return;
      toggleDisplay(btn.dataset.key);
    });
  });
}

function renderPitchTypeSelector() {
  const sel = document.getElementById("pitch-type-selector");
  const ptypes = (state.data && state.data.pitch_types) || [];
  if (!ptypes.length) {
    sel.style.display = "none";
    return;
  }
  sel.style.display = "";
  sel.innerHTML = `<div style="font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:0.6px;color:var(--text-muted);margin-bottom:4px;">Pitch Type</div>
    <div class="pitch-type-row">` +
    ptypes.map(pt => `<button class="pt-filter-btn${state.pitchType === pt ? " active" : ""}" data-pt="${pt}">${pt}</button>`).join("") +
    `</div>`;

  sel.querySelectorAll(".pt-filter-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      state.pitchType = state.pitchType === btn.dataset.pt ? null : btn.dataset.pt;
      state.activeDisplays = [];
      state.compatibleUnit = null;
      sel.querySelectorAll(".pt-filter-btn").forEach(b => b.classList.toggle("active", b === btn && state.pitchType !== null));
      render();
    });
  });
}

// ── display card management ───────────────────────────────────────────────

function toggleDisplay(key) {
  const existing = state.activeDisplays.findIndex(d => d.key === key);
  if (existing >= 0) {
    // Remove it
    state.activeDisplays.splice(existing, 1);
    if (!state.activeDisplays.length) state.compatibleUnit = null;
    buildMetricGrid();
    renderChart();
    return;
  }

  const meta = (window._trendsMetrics || []).find(m => m.key === key);
  if (!meta) return;

  // Check compatibility
  if (state.compatibleUnit !== null && meta.unit !== state.compatibleUnit) return;

  const color = LINE_COLORS[state.activeDisplays.length % LINE_COLORS.length];
  state.activeDisplays.push({ key: meta.key, label: meta.label, unit: meta.unit, decimals: meta.decimals ?? 1, color });
  state.compatibleUnit = meta.unit;

  buildMetricGrid();
  renderChart();
}

function removeDisplay(key) {
  const idx = state.activeDisplays.findIndex(d => d.key === key);
  if (idx < 0) return;
  state.activeDisplays.splice(idx, 1);
  if (!state.activeDisplays.length) state.compatibleUnit = null;
  buildMetricGrid();
  renderChart();
}

// ── main render ───────────────────────────────────────────────────────────

async function render() {
  if (!state.playerId) return;
  const chartMount = document.getElementById("chart-mount");
  chartMount.innerHTML = `<div class="trend-empty">Loading…</div>`;
  clearActiveChips();

  try {
    let url = `/api/players/${state.playerId}/trends?role=${state.role}&season=${state.season}`;
    if (state.pitchType && state.activeCat === "Pitch Type") {
      url += `&pitch_type=${encodeURIComponent(state.pitchType)}`;
    }
    const data = await getJSON(url);
    state.data = data;
    window._trendsMetrics = data.metrics;

    // Reset active displays if they reference stale metrics
    state.activeDisplays = state.activeDisplays.filter(d =>
      data.metrics.some(m => m.key === d.key));
    if (!state.activeDisplays.length) state.compatibleUnit = null;

    // Ensure active category has metrics
    const catMetrics = data.metrics.filter(m => m.category === state.activeCat);
    if (!catMetrics.length) {
      state.activeCat = data.metrics[0]?.category || "Physical";
      document.querySelectorAll(".cat-tab").forEach(t =>
        t.classList.toggle("active", t.dataset.cat === state.activeCat));
    }

    await fetchLgAvg();
    buildMetricGrid();
    renderChart();
  } catch (err) {
    chartMount.innerHTML = `<div class="trend-empty">Error: ${err.message}</div>`;
  }
}

async function fetchLgAvg() {
  try {
    state.lgAvgs = await getJSON(
      `/api/trends/league-averages?role=${state.role}&season=${state.season === "all" ? "2026" : state.season}`
    );
  } catch {
    state.lgAvgs = null;
  }
}

function clearActiveChips() {
  document.getElementById("active-chips").innerHTML = "";
}

function renderChart() {
  if (!state.data) return;
  const chartMount = document.getElementById("chart-mount");
  const legendMount = document.getElementById("legend-mount");
  const chipsEl = document.getElementById("active-chips");
  const titleEl = document.getElementById("chart-title");

  // Chips
  chipsEl.innerHTML = state.activeDisplays.map(d =>
    `<span class="active-chip" style="background:${d.color};">${d.label}<span class="active-chip-x" data-key="${d.key}">×</span></span>`
  ).join("");
  chipsEl.querySelectorAll(".active-chip-x").forEach(x => {
    x.addEventListener("click", () => removeDisplay(x.dataset.key));
  });

  // Title
  if (state.activeDisplays.length === 1) {
    titleEl.textContent = `${state.playerName || "Player"} — ${state.activeDisplays[0].label}`;
  } else if (state.activeDisplays.length > 1) {
    titleEl.textContent = `${state.playerName || "Player"} — ${state.activeDisplays.map(d => d.label).join(", ")}`;
  } else {
    titleEl.textContent = state.playerName || "";
  }

  if (!state.activeDisplays.length) {
    chartMount.innerHTML = `<div class="trend-empty">Select a metric to view.</div>`;
    legendMount.innerHTML = "";
    return;
  }

  chartMount.innerHTML = buildChart(state.data.games, state.activeDisplays, state.horizon, state.lgAvgs, state.showLgAvg);

  // Legend
  let legendHtml = state.activeDisplays.map(d =>
    `<div class="legend-item"><div class="legend-swatch" style="background:${d.color};"></div><span class="legend-label">${d.label}</span></div>`
  ).join("");

  if (state.showLgAvg && state.lgAvgs) {
    const hasLgAvg = state.activeDisplays.some(d => state.lgAvgs[d.key] != null);
    if (hasLgAvg) {
      legendHtml += `<div class="legend-item"><div class="legend-swatch" style="background:#f09a00; border-top: 2px dashed #f09a00;"></div><span class="legend-label-lgavg">League avg</span></div>`;
    }
  }
  legendMount.innerHTML = legendHtml;

  // lg avg toggle label
  const lgLabel = document.getElementById("lg-avg-label");
  if (lgLabel) {
    const hasAny = state.showLgAvg && state.lgAvgs && state.activeDisplays.some(d => state.lgAvgs[d.key] != null);
    lgLabel.classList.toggle("has-value", hasAny);
  }

  initHover();
}

// ── controls wiring ───────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", async () => {
  state.season = "2026";
  state.horizon = "days365";
  document.getElementById("lg-avg-toggle").checked = true;

  // Role toggle
  document.querySelectorAll(".pill-btn[data-role]").forEach(btn => {
    btn.addEventListener("click", async () => {
      if (state.role === btn.dataset.role) return;
      state.role = btn.dataset.role;
      state.activeCat = "Physical";
      state.playerId = null; state.playerName = null; state.data = null;
      state.activeDisplays = []; state.compatibleUnit = null;
      document.getElementById("player-input").value = "";
      document.getElementById("chart-mount").innerHTML = `<div class="trend-empty">Select a player and metric to view trends.</div>`;
      document.getElementById("metric-grid").innerHTML = "";
      document.getElementById("chart-title").textContent = "";
      document.getElementById("legend-mount").innerHTML = "";
      document.getElementById("active-chips").innerHTML = "";
      document.querySelectorAll(".pill-btn[data-role]").forEach(b =>
        b.classList.toggle("active", b === btn));
      document.querySelectorAll(".cat-tab").forEach(t =>
        t.classList.toggle("active", t.dataset.cat === "Physical"));
      syncHorizonGroup();
      await initTrendsSearch(state.role);
    });
  });

  syncHorizonGroup(); // set correct horizon group on load

  // Year select
  document.getElementById("year-select").addEventListener("change", async (e) => {
    state.season = e.target.value;
    state.activeDisplays = []; state.compatibleUnit = null;
    if (state.playerId) await render();
  });

  // Horizon toggle — shared handler for both pitcher and batter groups
  function wireHorizonButtons() {
    document.querySelectorAll(".pill-btn[data-horizon]").forEach(btn => {
      btn.addEventListener("click", () => {
        state.horizon = btn.dataset.horizon;
        document.querySelectorAll(".pill-btn[data-horizon]").forEach(b =>
          b.classList.toggle("active", b === btn && b.closest(".horizon-group").style.display !== "none"));
        renderChart();
      });
    });
  }
  wireHorizonButtons();

  // Show the right horizon group for the current role and reset active button
  function syncHorizonGroup() {
    const isPitcher = state.role === "pitcher";
    const pitcherGroup = document.getElementById("horizon-pitcher");
    const batterGroup  = document.getElementById("horizon-batter");
    if (pitcherGroup) pitcherGroup.style.display = isPitcher ? "" : "none";
    if (batterGroup)  batterGroup.style.display  = isPitcher ? "none" : "";
    // default horizon per role
    state.horizon = isPitcher ? "days365" : "pa150";
    document.querySelectorAll(".pill-btn[data-horizon]").forEach(b => {
      const inActiveGroup = !b.closest(".horizon-group") || b.closest(".horizon-group").style.display !== "none";
      b.classList.toggle("active", inActiveGroup && b.dataset.horizon === state.horizon);
    });
  }

  // Category tabs
  document.querySelectorAll(".cat-tab").forEach(tab => {
    tab.addEventListener("click", () => {
      if (state.activeCat !== tab.dataset.cat) {
        state.activeCat = tab.dataset.cat;
        state.activeDisplays = []; state.compatibleUnit = null;
        if (state.activeCat !== "Pitch Type") state.pitchType = null;
        document.querySelectorAll(".cat-tab").forEach(t =>
          t.classList.toggle("active", t === tab));
        buildMetricGrid();
        renderChart();
      }
    });
  });

  // League avg toggle
  document.getElementById("lg-avg-toggle").addEventListener("change", async (e) => {
    state.showLgAvg = e.target.checked;
    if (state.showLgAvg && !state.lgAvgs && state.playerId) {
      await fetchLgAvg();
    }
    renderChart();
  });

  // Seed from PlayerContext
  const ctx = typeof PlayerContext !== "undefined" && PlayerContext.getCurrent();
  if (ctx && ctx.player_id) {
    state.playerId = ctx.player_id;
    state.playerName = ctx.name;
    state.role = ctx.role || "pitcher";
    document.querySelectorAll(".pill-btn[data-role]").forEach(b =>
      b.classList.toggle("active", b.dataset.role === state.role));
  }

  await initTrendsSearch(state.role);

  if (ctx && ctx.player_id && _trendsSearch) {
    _trendsSearch.selectDefault(p => p.player_id === ctx.player_id);
  }
});
