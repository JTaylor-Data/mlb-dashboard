let state = { mode: "pitcher", split: "ALL", playerId: null, season: 2026 };

const splitToggle = document.getElementById("split-toggle");
const splitLabel = document.getElementById("split-label");
const seasonSelect = createSeasonSelect(document.getElementById("season-select"), async (season) => {
  state.season = season;
  await playerSearch.load(state.season);
  await loadPlayer();
});

function setActiveButton(group, value) {
  [...group.querySelectorAll("button")].forEach((b) => {
    b.classList.toggle("active", b.dataset.value === value);
  });
}

function applyRoleChrome() {
  splitLabel.textContent = state.mode === "pitcher" ? "Opponent bats" : "Opponent pitcher throws";
  document.getElementById("right-panel-title").textContent = state.mode === "pitcher" ? "Movement Plot" : "Swing & Field";
  document.getElementById("table-title").textContent = state.mode === "pitcher" ? "Pitch Table" : "Performance vs. Pitch Type";
  // Percentiles and Swing & Field are meant to read as a matched pair for
  // batters (equal width); the pitcher's Movement Plot genuinely needs the
  // extra room for its scatter plot, so that pairing keeps the wider ratio.
  document.getElementById("player-layout").classList.toggle("equal-cols", state.mode !== "pitcher");
}

splitToggle.addEventListener("click", async (e) => {
  const btn = e.target.closest("button");
  if (!btn || btn.dataset.value === state.split) return;
  state.split = btn.dataset.value;
  setActiveButton(splitToggle, state.split);
  await loadPlayer();
});

const playerSearch = createPlayerSearch(
  document.getElementById("player-search-input"),
  document.getElementById("player-search-dropdown"),
  async (p) => {
    state.playerId = p.player_id;
    state.mode = p.role;
    applyRoleChrome();
    await loadPlayer();
  }
);

function renderPlayerCard(data) {
  const logo = data.team_id ? `https://www.mlbstatic.com/team-logos/${data.team_id}.svg` : "";
  const statBlocks = data.card
    .map((c) => `<div class="stat-block"><div class="stat-block-label">${c.label}</div><div class="stat-block-value">${c.value}</div></div>`)
    .join("");

  let badgeHtml = "";
  if (data.badge) {
    if (data.badge.kind === "stuff_plus") {
      const color = data.badge.value !== null ? gradientColor(Math.min(100, Math.max(0, (data.badge.value - 70) / 60 * 100))) : "var(--text-muted)";
      badgeHtml = `<div class="badge" style="border-color:${color};"><div class="badge-label">STUFF+</div><div class="badge-value" style="color:${color};">${data.badge.value_str}</div></div>`;
    } else if (data.badge.kind === "archetype") {
      badgeHtml = `<div class="badge" title="${data.badge.description}"><div class="badge-label">ARCHETYPE</div><div class="badge-value">${data.badge.value}</div></div>`;
    }
  }

  const handednessHtml = data.handedness
    ? `<div class="player-card-handedness">${data.handedness}</div>`
    : "";

  document.getElementById("player-card-mount").innerHTML = `
    <div class="card player-card">
      <div class="player-card-id">
        ${logo ? `<img src="${logo}" onerror="this.style.display='none'" />` : ""}
        <div class="player-card-name">${data.name}</div>
        ${handednessHtml}
      </div>
      <div class="player-card-stats">${statBlocks}</div>
      <div>${badgeHtml}</div>
    </div>
  `;
}

// Batter panel: original blue(low)-grey(50th)-red(high) Savant-style
// gradient bars, unchanged.
function renderGradientPercentiles(groups) {
  let html = "";
  for (const g of groups) {
    html += `<div class="pct-group-header">${g.title}</div>`;
    for (const m of g.metrics) {
      const color = m.pct !== null ? gradientColor(m.pct) : "#ccc";
      const width = m.pct !== null ? m.pct : 0;
      const pctLabel = m.pct !== null ? m.pct.toFixed(0) : "-";
      html += `
        <div class="pct-row">
          <div class="pct-row-top"><span>${m.label}</span><span>${m.value_str}&nbsp;&nbsp;(${pctLabel})</span></div>
          <div class="pct-track"><div class="pct-fill" style="width:${width}%; background:${color};"></div></div>
        </div>
      `;
    }
  }
  return `<div class="card pct-card">${html}</div>`;
}

// Pitcher panel: boxed Results/Traits sections, green(>=50th)/red(<50th)
// bars with a percentile badge riding the end of the fill.
const RT_GOOD = "#1a7f37";
const RT_BAD = "#c9302c";

function renderResultsTraitsPercentiles(groups) {
  const sections = groups
    .map((g) => {
      const rows = g.metrics
        .map((m) => {
          const pct = m.pct !== null ? m.pct : 0;
          const color = pct >= 50 ? RT_GOOD : RT_BAD;
          const pctLabel = m.pct !== null ? m.pct.toFixed(0) : "-";
          return `
            <div class="rt-row">
              <div class="rt-label">${m.label}</div>
              <div class="rt-track">
                <div class="rt-fill" style="width:${pct}%; background:${color};"></div>
                <div class="rt-badge" style="left:${pct}%; background:${color};">${pctLabel}</div>
              </div>
              <div class="rt-value">${m.value_str}</div>
            </div>
          `;
        })
        .join("");
      return `<div><div class="rt-section-title">${g.title}</div>${rows}</div>`;
    })
    .join("");
  return `<div class="card rt-panel">${sections}</div>`;
}

function renderPercentiles(groups) {
  if (!groups || !groups.length) {
    document.getElementById("percentiles-mount").innerHTML = `<div class="card pct-card"><div class="empty-state">Not enough pitches/plate appearances to qualify in this split.</div></div>`;
    return;
  }
  document.getElementById("percentiles-mount").innerHTML = state.mode === "pitcher"
    ? renderResultsTraitsPercentiles(groups)
    : renderGradientPercentiles(groups);
}

function buildMovementPlotSVG(points, armAngle) {
  const size = 520, pad = 44;
  const plotSize = size - pad * 2;
  const domain = 25;

  const toSvgX = (hb) => pad + ((hb + domain) / (2 * domain)) * plotSize;
  const toSvgY = (ivb) => pad + ((domain - ivb) / (2 * domain)) * plotSize;

  let armLine = "";
  if (armAngle !== null && armAngle !== undefined) {
    const rad = (armAngle * Math.PI) / 180;
    const r = domain;
    const x1 = toSvgX(-r * Math.cos(rad)), y1 = toSvgY(-r * Math.sin(rad));
    const x2 = toSvgX(r * Math.cos(rad)), y2 = toSvgY(r * Math.sin(rad));
    const labelAnchor = x2 > toSvgX(0) ? "start" : "end";
    const labelDx = x2 > toSvgX(0) ? 6 : -6;
    armLine = `
      <line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="#aaa" stroke-width="3" stroke-dasharray="6,4"/>
      <text x="${x2 + labelDx}" y="${y2 - 4}" font-size="12" font-weight="700" fill="#888" text-anchor="${labelAnchor}">${armAngle.toFixed(0)}&#176; arm angle</text>
    `;
  }

  const circles = points
    .map((p) => {
      const color = pitchColor(p.pitch_type);
      return `<circle cx="${toSvgX(p.hb)}" cy="${toSvgY(p.ivb)}" r="3.6" fill="${color}" fill-opacity="0.9" stroke="rgba(0,0,0,0.45)" stroke-width="0.75"/>`;
    })
    .join("");

  const present = [...new Set(points.map((p) => p.pitch_type))];
  const legend = present
    .map((pt, i) => `
      <g transform="translate(${pad + i * 46}, ${size - 12})">
        <circle cx="0" cy="0" r="5" fill="${pitchColor(pt)}" />
        <text x="9" y="4" font-size="12" fill="var(--text)">${pt}</text>
      </g>
    `)
    .join("");

  return `
    <svg viewBox="0 0 ${size} ${size + 26}" width="100%" height="100%" preserveAspectRatio="xMidYMid meet" style="display:block;">
      <rect x="0" y="0" width="${size}" height="${size + 26}" fill="transparent"/>
      <line x1="${toSvgX(-domain)}" y1="${toSvgY(0)}" x2="${toSvgX(domain)}" y2="${toSvgY(0)}" stroke="#999" stroke-width="1"/>
      <line x1="${toSvgX(0)}" y1="${toSvgY(-domain)}" x2="${toSvgX(0)}" y2="${toSvgY(domain)}" stroke="#999" stroke-width="1"/>
      ${armLine}
      ${circles}
      <text x="${size / 2}" y="${size + 20}" font-size="12" text-anchor="middle" fill="var(--text-muted)">Horizontal Break (in)</text>
      <text x="12" y="${size / 2}" font-size="12" text-anchor="middle" fill="var(--text-muted)" transform="rotate(-90 12 ${size / 2})">Induced Vertical Break (in)</text>
      ${legend}
    </svg>
  `;
}

// Savant-style "vs LHH / Pitch / vs RHH" usage split - fills the blank
// space left below the short IVB/HB summary table in the movement plot
// card. Bar widths are pixel values (not %-of-container) so a lone
// off-speed pitch's tiny bar doesn't get stretched to look meaningful.
const PITCH_USAGE_MAX_PX = 120;

function buildPitchUsageWidget(usage) {
  if (!usage || !usage.length) return "";
  const rows = usage
    .map((u) => {
      const color = pitchColor(u.pitch_type);
      const lPx = (u.vs_l / 100) * PITCH_USAGE_MAX_PX;
      const rPx = (u.vs_r / 100) * PITCH_USAGE_MAX_PX;
      return `
        <div class="pitch-usage-row">
          <div class="pitch-usage-bar-l">
            <div class="pitch-usage-fill" style="width:${lPx.toFixed(0)}px; background:${color};">${u.vs_l.toFixed(0)}%</div>
          </div>
          <div class="pitch-usage-pill" style="background:${color};">${u.pitch_type}</div>
          <div class="pitch-usage-bar-r">
            <div class="pitch-usage-fill" style="width:${rPx.toFixed(0)}px; background:${color};">${u.vs_r.toFixed(0)}%</div>
          </div>
        </div>
      `;
    })
    .join("");
  return `
    <div class="pitch-usage-widget">
      <div class="pitch-usage-title">Pitch Usage</div>
      <div class="pitch-usage-header"><span>vs LHH</span><span>Pitch</span><span>vs RHH</span></div>
      ${rows}
    </div>
  `;
}

function buildMovementSummaryTable(summary) {
  const rows = summary
    .map((s) => `
      <tr>
        <td class="pitch-type-cell" style="background:${pitchColor(s.pitch_type)};">${s.pitch_type}</td>
        <td>${s.ivb.toFixed(1)}</td>
        <td>${s.hb.toFixed(1)}</td>
      </tr>
    `)
    .join("");
  return `
    <table class="data-table movement-summary-table">
      <thead><tr><th>Pitch</th><th>IVB</th><th>HB</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

// Real spray chart, Savant-style: every individual batted ball plotted at
// its actual field-relative landing spot (converted from Statcast's
// hc_x/hc_y in player_page.py's field_zone_pcts), colored by whether it
// went for a hit, PLUS the same explicit LF/CF/RF percentage labels the
// simplified wedge version used to show - not a replacement for the
// numbers, an upgrade of what's drawn underneath them. Sits at the TOP of
// the Swing & Field card.
function buildFieldSprayChart(lfPct, cfPct, rfPct, locations) {
  const size = 400;
  const homeX = size / 2, homeY = 372;
  const scale = 1.55;
  const fenceR = 195 * scale;

  const toScreen = (x, y) => [homeX + x * scale, homeY - y * scale];
  const onCircle = (deg, r) => {
    const rad = (deg * Math.PI) / 180;
    return [homeX + Math.sin(rad) * r, homeY - Math.cos(rad) * r];
  };

  const [flx, fly] = onCircle(-45, fenceR);
  const [frx, fry] = onCircle(45, fenceR);
  const [dlx, dly] = onCircle(-15, fenceR);
  const [drx, dry] = onCircle(15, fenceR);
  let fenceArc = "";
  for (let d = -45; d <= 45; d += 5) {
    const [ax, ay] = onCircle(d, fenceR);
    fenceArc += ` L ${ax.toFixed(1)},${ay.toFixed(1)}`;
  }

  const dots = (locations || [])
    .map((p) => {
      const [sx, sy] = toScreen(p.x, p.y);
      const color = p.hit ? "#dc2626" : "#3b6ea5";
      return `<circle cx="${sx.toFixed(1)}" cy="${sy.toFixed(1)}" r="3" fill="${color}" fill-opacity="0.55" stroke="${color}" stroke-width="0.6"/>`;
    })
    .join("");

  const labelAt = (deg, text) => {
    const [lx, ly] = onCircle(deg, fenceR * 0.68);
    return `<text x="${lx.toFixed(1)}" y="${ly.toFixed(1)}" font-size="17" fill="white" text-anchor="middle" font-weight="bold" style="paint-order:stroke; stroke:#00000066; stroke-width:3px;">${text}</text>`;
  };

  return `
    <svg viewBox="0 0 ${size} ${size}" width="100%" style="border-radius:8px; max-width:520px; background:#bfe3f5;">
      <path d="M ${homeX},${homeY} L ${flx.toFixed(1)},${fly.toFixed(1)}${fenceArc} L ${frx.toFixed(1)},${fry.toFixed(1)} Z" fill="#5b8c3a"/>
      <line x1="${homeX}" y1="${homeY}" x2="${flx.toFixed(1)}" y2="${fly.toFixed(1)}" stroke="#fff" stroke-width="2"/>
      <line x1="${homeX}" y1="${homeY}" x2="${frx.toFixed(1)}" y2="${fry.toFixed(1)}" stroke="#fff" stroke-width="2"/>
      <line x1="${homeX}" y1="${homeY}" x2="${dlx.toFixed(1)}" y2="${dly.toFixed(1)}" stroke="#ffffff99" stroke-width="1" stroke-dasharray="4,3"/>
      <line x1="${homeX}" y1="${homeY}" x2="${drx.toFixed(1)}" y2="${dry.toFixed(1)}" stroke="#ffffff99" stroke-width="1" stroke-dasharray="4,3"/>
      ${dots}
      ${labelAt(-30, `LF ${lfPct.toFixed(0)}%`)}
      ${labelAt(0, `CF ${cfPct.toFixed(0)}%`)}
      ${labelAt(30, `RF ${rfPct.toFixed(0)}%`)}
      <path d="M ${homeX - 13},${homeY} L ${homeX},${homeY - 13} L ${homeX + 13},${homeY} L ${homeX},${homeY + 7} Z" fill="#e8e2d0" stroke="#3a5f27" stroke-width="1.5"/>
    </svg>
  `;
}

// --- Swing gauges (Attack Angle / Attack Direction / Swing Path Tilt / Bat
// Speed) - small SVG diagrams pairing this player's real value against the
// league average, in the same visual language as the reference card.
//
// Attack Angle and Swing Path Tilt measure genuinely different things (the
// bat's angle at the INSTANT of contact vs. the tilt of the swing's whole
// arc/plane through the zone) but a plain rotating needle made both gauges
// look like the same idea. Fixed by giving each concept its own visual
// grammar: Attack Angle keeps a needle FROM A FIXED PIVOT with a shaded
// wedge sweeping out the angle (a point-in-time reading), while Swing Path
// Tilt is drawn as a fading, widening TRAIL along a curve (the same
// ghost-trail motif this page already uses for "a path traveled," not a
// single instant).
function fmtDeg(v) {
  return v === null || v === undefined ? "-" : `${v.toFixed(0)}°`;
}
function fmtMph(v) {
  return v === null || v === undefined ? "-" : `${v.toFixed(1)} mph`;
}

// --- Archetypes: a small, fun, honest label per gauge based on how far this
// player's value sits from the league average - not another data source,
// just a plain-language read of the same number already shown below it.
function archetypeChip(label) {
  return label ? `<div class="gauge-archetype">${label}</div>` : "";
}
function batSpeedArchetype(value, avg) {
  if (value == null || avg == null) return "";
  const diff = value - avg;
  if (diff >= 3) return "Fast Swinger";
  if (diff <= -3) return "No Pop";
  return "Steady Hands";
}
function attackAngleArchetype(value, avg) {
  if (value == null || avg == null) return "";
  const diff = value - avg;
  if (diff >= 6) return "Scoopy";
  if (diff <= -6) return "Groundball Merchant";
  return "Line-Drive";
}
function attackDirectionArchetype(value) {
  if (value == null) return "";
  if (value <= -10) return "Pull-Side Merchant";
  if (value >= 10) return "Oppo Pop";
  return "Spray Hitter";
}
function swingTiltArchetype(value, avg) {
  if (value == null || avg == null) return "";
  const diff = value - avg;
  if (diff <= -8) return "Flat Swinger";
  if (diff >= 8) return "Steep Swinger";
  return "Textbook Plane";
}

// Deliberately NOT a needle/dial (that read as "the same idea" as Swing
// Path Tilt's curve). Instead: a fixed ball at a fixed point, and the BAT
// drawn as a solid bar physically crossing through it, tilted by the angle
// - "the bat's angle at the instant it collides with the ball," a impact
// snapshot rather than any kind of rotating gauge. GROUND/LOFT labels replace
// the old horizontal reference line's ambiguity about what "0" means.
function buildAttackAngleGauge(value, leagueAvg) {
  const v = Math.max(-30, Math.min(45, value ?? 0));
  const contactX = 68, contactY = 50, halfLen = 36;
  const rad = (-v * Math.PI) / 180;
  const dx = halfLen * Math.cos(rad), dy = halfLen * Math.sin(rad);
  const x1 = contactX - dx, y1 = contactY + dy;
  const x2 = contactX + dx, y2 = contactY - dy;
  return `
    <div class="gauge-card">
      ${archetypeChip(attackAngleArchetype(value, leagueAvg))}
      <div class="gauge-title">Attack Angle</div>
      <div class="gauge-body">
        <svg viewBox="0 0 140 70" width="100%" style="max-width:220px;">
          <text x="4" y="12" font-size="9" fill="var(--text-muted)" font-weight="700">LOFT</text>
          <line x1="15" y1="50" x2="125" y2="50" stroke="var(--border)" stroke-width="1.5" stroke-dasharray="3,3"/>
          <text x="4" y="64" font-size="9" fill="var(--text-muted)" font-weight="700">GROUND</text>
          <line x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}" stroke="#8b5a2b" stroke-width="7" stroke-linecap="round"/>
          <circle cx="${contactX}" cy="${contactY}" r="6.5" fill="#fff" stroke="#dc2626" stroke-width="2.5"/>
        </svg>
        <div class="gauge-value">${fmtDeg(value)}</div>
        <div class="gauge-avg">MLB avg: ${fmtDeg(leagueAvg)}</div>
      </div>
    </div>
  `;
}

function buildAttackDirectionGauge(value, leagueAvg) {
  const v = Math.max(-45, Math.min(45, value ?? 0));
  const frac = (v + 45) / 90; // 0=full pull, 1=full oppo
  const cx = 15 + frac * 110;
  return `
    <div class="gauge-card">
      ${archetypeChip(attackDirectionArchetype(value))}
      <div class="gauge-title">Attack Direction</div>
      <div class="gauge-body">
        <svg viewBox="0 0 140 60" width="100%" style="max-width:220px;">
          <text x="4" y="14" font-size="9" fill="var(--forest-light)" font-weight="700">PULL</text>
          <text x="112" y="14" font-size="9" fill="var(--text-muted)" font-weight="700">OPPO</text>
          <line x1="15" y1="34" x2="125" y2="34" stroke="var(--border)" stroke-width="2"/>
          <line x1="70" y1="28" x2="70" y2="40" stroke="var(--border)" stroke-width="1.5"/>
          <circle cx="${cx.toFixed(1)}" cy="34" r="5" fill="#dc2626"/>
        </svg>
        <div class="gauge-value">${fmtDeg(value)}</div>
        <div class="gauge-avg">MLB avg: ${fmtDeg(leagueAvg)}</div>
      </div>
    </div>
  `;
}

function buildSwingTiltGauge(value, leagueAvg) {
  const v = Math.max(0, Math.min(90, value ?? 30));
  const startX = 15, startY = 60;
  const contactX = 100, contactY = startY - (v / 90) * 46;
  const ctrlX = startX + (contactX - startX) * 0.65, ctrlY = startY;
  const bezier = (t) => [
    (1 - t) * (1 - t) * startX + 2 * (1 - t) * t * ctrlX + t * t * contactX,
    (1 - t) * (1 - t) * startY + 2 * (1 - t) * t * ctrlY + t * t * contactY,
  ];
  const steps = 14;
  let trail = "";
  let prev = bezier(0);
  for (let i = 1; i <= steps; i++) {
    const t = i / steps;
    const point = bezier(t);
    const opacity = 0.15 + 0.7 * Math.pow(t, 1.3);
    const width = 2 + 3.2 * t;
    trail += `<line x1="${prev[0].toFixed(1)}" y1="${prev[1].toFixed(1)}" x2="${point[0].toFixed(1)}" y2="${point[1].toFixed(1)}" stroke="#e6c200" stroke-width="${width.toFixed(1)}" stroke-linecap="round" opacity="${opacity.toFixed(2)}"/>`;
    prev = point;
  }
  return `
    <div class="gauge-card">
      ${archetypeChip(swingTiltArchetype(value, leagueAvg))}
      <div class="gauge-title">Swing Path Tilt</div>
      <div class="gauge-body">
        <svg viewBox="0 0 140 70" width="100%" style="max-width:220px;">
          <line x1="15" y1="60" x2="120" y2="60" stroke="var(--border)" stroke-width="1.5" stroke-dasharray="3,3"/>
          ${trail}
          <circle cx="${contactX}" cy="${contactY.toFixed(1)}" r="4.5" fill="#fff" stroke="#e6c200" stroke-width="2"/>
          <text x="122" y="64" font-size="9" fill="var(--text-muted)">FLAT</text>
          <text x="10" y="12" font-size="9" fill="var(--text-muted)">STEEP</text>
        </svg>
        <div class="gauge-value">${fmtDeg(value)}</div>
        <div class="gauge-avg">MLB avg: ${fmtDeg(leagueAvg)}</div>
      </div>
    </div>
  `;
}

// Bat Speed's dial stays exactly as it was - only the archetype chip and
// the gauge-body centering wrapper (shared with the other three gauges) are
// new here.
function buildBatSpeedGauge(value, leagueAvg) {
  const v = Math.max(0, Math.min(90, value ?? 0));
  const startAngle = 180, endAngle = 0;
  const frac = v / 90;
  const angle = startAngle + (endAngle - startAngle) * frac;
  const rad = (angle * Math.PI) / 180;
  const cx = 70, cy = 58, r = 44;
  const nx = cx + r * Math.cos(rad), ny = cy - r * Math.sin(rad);
  const arcPath = (a1, a2, color) => {
    const r1 = (a1 * Math.PI) / 180, r2 = (a2 * Math.PI) / 180;
    const x1 = cx + r * Math.cos(r1), y1 = cy - r * Math.sin(r1);
    const x2 = cx + r * Math.cos(r2), y2 = cy - r * Math.sin(r2);
    return `<path d="M ${x1.toFixed(1)},${y1.toFixed(1)} A ${r},${r} 0 0 1 ${x2.toFixed(1)},${y2.toFixed(1)}" stroke="${color}" stroke-width="8" fill="none"/>`;
  };
  return `
    <div class="gauge-card">
      ${archetypeChip(batSpeedArchetype(value, leagueAvg))}
      <div class="gauge-title">Bat Speed</div>
      <div class="gauge-body">
        <svg viewBox="0 0 140 64" width="100%" style="max-width:220px;">
          ${arcPath(180, 100, "#9ca3af")}
          ${arcPath(100, 60, "#e6c200")}
          ${arcPath(60, 0, "#dc2626")}
          <line x1="${cx}" y1="${cy}" x2="${nx.toFixed(1)}" y2="${ny.toFixed(1)}" stroke="var(--forest-dark)" stroke-width="2.5"/>
          <circle cx="${cx}" cy="${cy}" r="3.5" fill="var(--forest-dark)"/>
        </svg>
        <div class="gauge-value">${fmtMph(value)}</div>
        <div class="gauge-avg">MLB avg: ${fmtMph(leagueAvg)}</div>
      </div>
    </div>
  `;
}

function renderRightPanel(data) {
  const mount = document.getElementById("right-panel-mount");

  if (state.mode === "pitcher") {
    const mv = data.movement_plot;
    if (!mv.points.length) {
      mount.innerHTML = `<div class="empty-state">No pitch data for this selection.</div>`;
      return;
    }
    mount.innerHTML = `
      <div class="card movement-plot-card">
        <div class="movement-plot-row">
          <div class="movement-plot-svg-wrap">${buildMovementPlotSVG(mv.points, mv.arm_angle)}</div>
          <div class="movement-plot-side-col">
            ${buildMovementSummaryTable(mv.summary)}
            ${buildPitchUsageWidget(data.pitch_usage_by_stand)}
          </div>
        </div>
      </div>
      <div class="movement-plot-caption">Dashed line = this pitcher's average arm angle (labeled). Arm-side-adjusted break, matching Baseball Savant - positive HB is arm side for either hand. Only the 50% of pitches closest to each pitch type's own average are plotted; the table and usage split on the right use every pitch.</div>
    `;
  } else {
    const sf = data.swing_field;
    const sw = data.swing_3d;
    const lg = sw.league_avg;
    mount.innerHTML = `
      <div class="card swing-field-card">
        ${buildFieldSprayChart(sf.lf, sf.cf, sf.rf, sf.locations)}
        <div class="spray-chart-caption">Every tracked batted ball this season, plotted at its real landing spot (<span style="color:#dc2626; font-weight:700;">red = hit</span>, <span style="color:#3b6ea5; font-weight:700;">blue = out</span>). Dashed lines mark the LF/CF/RF split the percentages are based on.</div>
        <div class="gauge-grid">
          ${buildAttackAngleGauge(sw.attack_angle, lg.attack_angle)}
          ${buildAttackDirectionGauge(sw.attack_direction, lg.attack_direction)}
          ${buildSwingTiltGauge(sw.swing_path_tilt, lg.swing_path_tilt)}
          ${buildBatSpeedGauge(sw.bat_speed, lg.bat_speed)}
        </div>
      </div>
    `;
  }
}

function renderTable(tableData, caption) {
  document.getElementById("table-caption").textContent = caption || "";
  const mount = document.getElementById("table-mount");
  if (!tableData || !tableData.rows.length) {
    mount.innerHTML = `<div class="empty-state">No data for this selection.</div>`;
    return;
  }
  const headerCells = tableData.columns.map((c) => `<th>${c.label}</th>`).join("");
  const bodyRows = tableData.rows
    .map((row) => {
      const color = pitchColor(row.pitch_type);
      const tint = hexToRgba(color, 0.12);
      const cells = row.cells.map((c) => `<td style="background:${tint};">${c.value}</td>`).join("");
      return `<tr><td class="pitch-type-cell" style="background:${color};">${row.pitch_type}</td>${cells}</tr>`;
    })
    .join("");

  let leagueRow = "";
  if (tableData.summary_row) {
    const cells = tableData.summary_row.map((c) => `<td>${c.value}</td>`).join("");
    leagueRow = `<tr class="league-avg-row"><td class="pitch-type-cell" style="background:var(--text-muted);">${tableData.summary_label || "Summary"}</td>${cells}</tr>`;
  }

  mount.innerHTML = `
    <table class="data-table">
      <thead><tr><th>Pitch</th>${headerCells}</tr></thead>
      <tbody>${bodyRows}${leagueRow}</tbody>
    </table>
  `;
}

// Rapid mode/player/split switching can fire several loadPlayer() calls
// before earlier ones resolve; without this guard, a slower older response
// can land after a newer one and overwrite the UI with stale data (or a
// stale playerId/mode pairing that doesn't match the currently rendered
// panel titles). Only the most recently issued request is allowed to render.
let loadPlayerRequestId = 0;

async function loadPlayer() {
  if (state.playerId === null || state.playerId === "") return;
  const requestId = ++loadPlayerRequestId;
  const requestMode = state.mode;
  const data = await getJSON(`/api/player/${state.playerId}?role=${state.mode}&split=${state.split}&season=${state.season}`);
  if (requestId !== loadPlayerRequestId || requestMode !== state.mode) return;
  renderPlayerCard(data);
  renderPercentiles(data.percentiles);
  renderRightPanel(data);
  if (state.mode === "pitcher") {
    renderTable(data.pitch_table, "");
    renderStuffBreakdown(data.stuff_breakdown);
  } else {
    renderTable(data.pitch_type_table, "Pitch types with 10 or fewer pitches seen are excluded.");
    document.getElementById("stuff-breakdown-mount").innerHTML = "";
  }
}

function renderStuffBreakdown(breakdown) {
  const mount = document.getElementById("stuff-breakdown-mount");
  if (!breakdown) {
    mount.innerHTML = "";
    return;
  }
  const headerCells = breakdown.columns.map((c) => `<th>${c}</th>`).join("");
  const bodyRows = breakdown.rows
    .map((row) => {
      const cells = breakdown.columns
        .map((col) => {
          const v = row.values[col];
          if (v === null || v === undefined) return `<td>-</td>`;
          const color = gradientColor(Math.min(100, Math.max(0, ((v - 70) / 60) * 100)));
          return `<td style="color:${color}; font-weight:700;">${v.toFixed(0)}</td>`;
        })
        .join("");
      return `<tr><td class="stuff-row-label">${row.label}</td>${cells}</tr>`;
    })
    .join("");
  mount.innerHTML = `
    <div class="card stuff-breakdown-card">
      <h3 style="margin:0 0 10px;">Stuff+ by Pitch &amp; Handedness</h3>
      <table class="data-table stuff-breakdown-table">
        <thead><tr><th></th>${headerCells}</tr></thead>
        <tbody>${bodyRows}</tbody>
      </table>
    </div>
  `;
}

window.onDataRefresh = async () => {
  await playerSearch.load(state.season);
  await loadPlayer();
};

(async function init() {
  state.season = await seasonSelect.load();
  await playerSearch.load(state.season);
  const defaultPlayer = playerSearch.selectDefault((p) => p.role === "pitcher");
  if (defaultPlayer) {
    state.playerId = defaultPlayer.player_id;
    state.mode = defaultPlayer.role;
    applyRoleChrome();
    await loadPlayer();
  }
})();
