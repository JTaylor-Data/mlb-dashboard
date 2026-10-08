// heatmapSplit: the Pitch Locations card's OWN vs-RHH/vs-LHH selection -
// independent of the page's main split toggle (which still has an ALL
// option, meaningless for a location heatmap), but the main toggle
// OVERRIDES it whenever it's set to a real L/R value (see splitToggle's
// click handler below) - the card's own toggle can then diverge again
// until the next time the main toggle changes.
let state = { mode: "pitcher", split: "ALL", heatmapSplit: "R", playerId: null, season: 2026 };

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
  // Both roles now get their own visual in the left-column slot (Pitch
  // Locations for pitchers, Hit Zones for batters) with Percentiles always
  // living in its own full-width horizontal row below, matching the
  // pitcher-style 3-group grid for both roles instead of batters keeping a
  // separate narrow stacked layout.
  document.getElementById("left-panel-title").textContent = state.mode === "pitcher" ? "Pitch Locations" : "Hit Zones";
}

splitToggle.addEventListener("click", async (e) => {
  const btn = e.target.closest("button");
  if (!btn || btn.dataset.value === state.split) return;
  state.split = btn.dataset.value;
  setActiveButton(splitToggle, state.split);
  // The main toggle overrides the Pitch Locations card's own local
  // selection whenever it's a real split - ALL has no meaning for a
  // location heatmap, so it's left alone in that case (see state comment).
  if (state.split === "L" || state.split === "R") {
    state.heatmapSplit = state.split;
  }
  await loadPlayer();
});

const playerSearch = createPlayerSearch(
  document.getElementById("player-search-input"),
  document.getElementById("player-search-dropdown"),
  async (p) => {
    state.playerId = p.player_id;
    state.mode = p.role;
    // If this player wasn't active in the current season, auto-switch to
    // their most recent active season so the page loads with real data.
    if (p.latest_season && p.latest_season !== state.season) {
      seasonSelect.set(p.latest_season);
      state.season = p.latest_season;
    }
    applyRoleChrome();
    await loadPlayer();
  }
);

function renderPlayerCard(data) {
  const logo = data.team_id ? `https://www.mlbstatic.com/team-logos/${data.team_id}.svg` : "";
  const statBlocks = data.card
    .map((c) => `<div class="stat-block"><div class="stat-block-label">${c.label}</div><div class="stat-block-value">${c.value}</div></div>`)
    .join("");

  // Archetype and Stuff+ share the top-right slot (.player-card-stuff) -
  // moved back here per direction, after a brief stint next to the name.
  // A player only ever has one badge kind active (pitchers get stuff_plus,
  // batters get archetype), so they never collide.
  let archetypeHtml = "";
  let stuffHtml = "";
  if (data.badge && data.badge.kind === "archetype") {
    archetypeHtml = `<div class="badge badge-inline" title="${data.badge.description}"><div class="badge-label">ARCHETYPE</div><div class="badge-value">${data.badge.value}</div></div>`;
  }
  // Collapsed the old per-pitch-type Stuff+ breakdown table out of this
  // banner slot entirely (still viewable on the Stuff+ model page itself,
  // one click away via this badge) - the freed space instead holds
  // Deception+ and Tunnel+ as their own clickable badges, same visual
  // language as Stuff+, so all three "investigate further" links sit
  // together. plusBadge() shares Stuff+'s own 70-130 "plus stat" color
  // scale (100 = average) since Deception+/Tunnel+ follow the same
  // convention.
  const plusBadge = (label, badge, href, title) => {
    if (!badge) return "";
    const pct = badge.value !== null && badge.value !== undefined ? Math.min(100, Math.max(0, (badge.value - 70) / 60 * 100)) : 50;
    const color = badge.value !== null && badge.value !== undefined ? stuffColor(pct) : "var(--text-muted)";
    // Bumped from a 0.14 background tint to 0.32 (and a thicker 3px border)
    // per direction - the red/green good-bad read was too subtle at a
    // glance before.
    const bg = badge.value !== null && badge.value !== undefined ? stuffColor(pct, 0.32) : "transparent";
    return `
      <a class="badge badge-inline badge-headline" href="${href}" style="border-color:${color}; border-width:3px; background:${bg};" title="${title}">
        <div class="badge-label">${label}</div><div class="badge-value" style="color:${color};">${badge.value_str}</div>
      </a>
    `;
  };

  // --- Scorecard-strip layout for the four model plaques (Stuff+/
  // Deception+/Tunnel+/Location+) - a single horizontal strip with large
  // numbers and a thin colored left-edge accent, swapped in as the default
  // per direction in place of the four separate bordered boxes. The boxed
  // version (plusBadge above) is left intact and easy to swap back to - see
  // the comment at the call site below.
  const plusBadgeAlt = (label, badge) => {
    if (!badge) return "";
    const pct = badge.value !== null && badge.value !== undefined ? Math.min(100, Math.max(0, (badge.value - 70) / 60 * 100)) : 50;
    const color = badge.value !== null && badge.value !== undefined ? stuffColor(pct) : "var(--text-muted)";
    return `
      <div class="plus-scorecard-item" style="border-left-color:${color};">
        <div class="plus-scorecard-value" style="color:${color};">${badge.value_str}</div>
        <div class="plus-scorecard-label">${label}</div>
      </div>
    `;
  };
  function buildPlusPlaquesAlt(cardData, badgeFn) {
    if (!cardData.badge || cardData.badge.kind !== "stuff_plus") return "";
    return `
      <div class="plus-scorecard-row">
        ${badgeFn("STUFF+", cardData.badge)}
        ${badgeFn("DECEPTION+", cardData.deception_badge)}
        ${badgeFn("TUNNEL+", cardData.tunneling_badge)}
        ${badgeFn("LOCATION+", cardData.location_badge)}
      </div>
    `;
  }
  // Scorecard-strip layout, now the default per direction. The original
  // four-boxes layout (plusBadge, with its own per-badge model-page link)
  // is still defined above and easy to restore - swap this line back for:
  // stuffHtml = `<div class="stuff-plus-block">${plusBadge(...)}...</div>`
  if (data.badge && data.badge.kind === "stuff_plus") {
    stuffHtml = buildPlusPlaquesAlt(data, plusBadgeAlt);
  }

  let handednessHtml = "";
  if (data.role === "batter" && data.bats) {
    // Two-line "BATS/THROWS" + "R/R" for batters (a fielder's throwing hand
    // is real, distinct info a single "Bats R" line dropped entirely) -
    // pitchers keep their existing single "RHP"/"LHP" line, unambiguous
    // on its own.
    handednessHtml = `
      <div class="player-card-handedness player-card-handedness-stacked">
        <div class="handedness-label">BATS/THROWS</div>
        <div class="handedness-value">${data.bats}/${data.throws || "-"}</div>
      </div>
    `;
  } else if (data.handedness) {
    handednessHtml = `<div class="player-card-handedness">${data.handedness}</div>`;
  }
  const ilHtml = buildIlBadge(data.il_status);

  const positionHtml = data.position ? `<div class="player-card-position">${data.position}</div>` : "";
  document.getElementById("player-card-mount").innerHTML = `
    <div class="card player-card">
      <div class="player-card-id">
        ${logo ? `<img src="${logo}" onerror="this.style.display='none'" />` : ""}
        <div class="player-card-name">${data.name}</div>
        ${positionHtml}
        ${handednessHtml}
        ${ilHtml}
      </div>
      <div class="player-card-stats">${statBlocks}</div>
      <div class="player-card-stuff">${stuffHtml}${archetypeHtml}</div>
    </div>
  `;
  const ilBtn = document.getElementById("il-badge-btn");
  if (ilBtn) {
    ilBtn.addEventListener("click", () => {
      const detail = document.getElementById("il-detail");
      detail.style.display = detail.style.display === "none" ? "block" : "none";
    });
  }
}

function fmtDate(iso) {
  if (!iso) return "Unknown";
  const d = new Date(`${iso}T00:00:00`);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
}

// Real IL badge - il_status comes straight from il_status (see
// pulls/pull_injuries.py): start_date and injury are real, parsed from the
// player's own MLB Stats API transaction history; min_return_date is a
// real COMPUTED lower bound (start_date + that IL type's minimum stint
// length, a hard roster rule) - there's no official "expected return date"
// published anywhere, so this is deliberately labeled "Eligible to return,"
// not "Expected return."
function buildIlBadge(il) {
  if (!il) return "";
  const label = (il.il_label || "").replace("Injured ", "").trim() || "IL";
  return `
    <div class="il-badge-wrap">
      <button class="il-badge" id="il-badge-btn" type="button">${label}</button>
      <div class="il-detail" id="il-detail" style="display:none;">
        <div><span class="il-detail-label">Injury:</span> ${il.injury || "Not specified"}</div>
        <div><span class="il-detail-label">Placed on IL:</span> ${fmtDate(il.start_date)}</div>
        <div><span class="il-detail-label">Stint:</span> ${label}</div>
        <div><span class="il-detail-label">Eligible to return:</span> ${fmtDate(il.min_return_date)}</div>
      </div>
    </div>
  `;
}

// One shared boxed layout (label, track+fill, and a percentile badge
// CENTERED on the end of the fill) for both roles - previously batters got
// a structurally different, badge-less layout (.pct-* - plain text, no
// centered dot) while pitchers got this one (.rt-*), which is why batter
// percentile numbers never looked centered in anything: there was no dot to
// center them in. Both roles now share the same blue-grey-red continuous
// gradient (gradientColor, blue=bad/low, red=good/high) - previously
// pitchers used a hard >=50th green/<50th red THRESHOLD (no grey at all,
// and no sense of HOW far above/below average); unified per direction to
// one consistent color language across the whole site. Stuff+'s own
// red-grey-green scale (stuffColor) is intentionally kept separate.
const pctColor = (pct) => gradientColor(pct);

// Unqualified players (didn't clear the real qualifying threshold - see
// api/routers/players.py's player_percentiles_allow_unqualified) still get
// real bars, ranked against the real qualified population, but shaded/
// muted rather than shown identically to a qualified player's - matching
// Baseball Savant's own convention, rather than hiding them outright.
function renderPercentileGroups(groups, colorFn, layout) {
  const anyUnqualified = groups.some((g) => g.metrics.some((m) => m.qualified === false));
  const sections = groups
    .map((g) => {
      const rows = g.metrics
        .map((m) => {
          const pct = m.pct !== null ? m.pct : 0;
          const color = m.pct !== null ? colorFn(pct) : "#ccc";
          const pctLabel = m.pct !== null ? m.pct.toFixed(0) : "-";
          const noteIcon = m.note ? `<span class="rt-note-icon" title="${m.note}">&#9432;</span>` : "";
          const rowCls = m.qualified === false ? "rt-row rt-row-unqualified" : "rt-row";
          const unqTitle = m.qualified === false ? ` title="Doesn't meet the qualifying threshold this season/split - shown against the real qualified population, shaded to distinguish it from a qualified player's percentile."` : "";
          return `
            <div class="${rowCls}"${unqTitle}>
              <div class="rt-label">${m.label}${noteIcon}</div>
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
  const cls = layout === "grid" ? "rt-panel rt-panel-grid" : "rt-panel";
  return `<div class="card ${cls}" data-any-unqualified="${anyUnqualified ? '1' : '0'}">${sections}</div>`;
}

function renderPercentiles(groups) {
  const mount = document.getElementById("pitcher-percentiles-mount");
  const section = document.getElementById("pitcher-percentiles-section");
  const heading = section ? section.querySelector("h2") : null;

  // Clear any prior inline notes
  if (heading) {
    heading.querySelectorAll(".pct-no-data-note, .pct-unqualified-note").forEach(el => el.remove());
  }

  if (!groups || !groups.length) {
    mount.innerHTML = "";
    if (heading) {
      const note = document.createElement("span");
      note.className = "pct-no-data-note";
      note.textContent = " — insufficient data for this split";
      heading.appendChild(note);
    }
    return;
  }

  mount.innerHTML = renderPercentileGroups(groups, pctColor, "grid");

  // Move the unqualified note inline to the heading (no box inside the card)
  const card = mount.querySelector("[data-any-unqualified]");
  if (heading && card && card.dataset.anyUnqualified === "1") {
    const note = document.createElement("span");
    note.className = "pct-unqualified-note";
    note.textContent = " — unqualified; ranked vs. qualified population, shaded";
    heading.appendChild(note);
  }
}

// Arm-side-adjusted HB (positive = arm side, for EITHER hand, per Savant's
// own convention - see caption) makes a RHP and a LHP plot identically,
// which is exactly the "looks the same" complaint. Mirroring HB back to an
// absolute/catcher's-eye view for lefties specifically (negate hb here, and
// only here - the raw data/summary table stay arm-side-adjusted, matching
// Savant) is what actually makes them look different, the way a real
// scatter of pitch shapes would if you were standing behind the plate.
function buildMovementPlotSVG(points, armAngle, throws, peerMovement) {
  const size = 440, pad = 40;
  const plotSize = size - pad * 2;
  const domain = 25;
  const mirror = throws === "L" ? -1 : 1;

  const toSvgX = (hb) => pad + ((hb + domain) / (2 * domain)) * plotSize;
  const toSvgY = (ivb) => pad + ((domain - ivb) / (2 * domain)) * plotSize;

  let armLine = "";
  if (armAngle !== null && armAngle !== undefined) {
    // A normal arm angle is 0 (sidearm) to 90 (pure overhand) - both cos and
    // sin are non-negative in that range, so the line only ever needs to run
    // from the origin OUT to that one point - the old version also drew the
    // mirrored ray into the opposite (meaningless) quadrant.
    const rad = (armAngle * Math.PI) / 180;
    const r = domain;
    const x1 = toSvgX(0), y1 = toSvgY(0);
    const x2 = toSvgX(mirror * r * Math.cos(rad)), y2 = toSvgY(r * Math.sin(rad));
    const labelAnchor = x2 > x1 ? "start" : "end";
    const labelDx = x2 > x1 ? 6 : -6;
    armLine = `
      <line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="#aaa" stroke-width="3" stroke-dasharray="6,4"/>
      <text x="${x2 + labelDx}" y="${y2 - 4}" font-size="12" font-weight="700" fill="#888" text-anchor="${labelAnchor}">${armAngle.toFixed(0)}&#176;</text>
    `;
  }

  const circles = points
    .map((p) => {
      const color = pitchColor(p.pitch_type);
      return `<circle cx="${toSvgX(mirror * p.hb)}" cy="${toSvgY(p.ivb)}" r="3.6" fill="${color}" fill-opacity="0.9" stroke="rgba(0,0,0,0.45)" stroke-width="0.75"/>`;
    })
    .join("");

  // One big translucent "average movement" circle per pitch type, drawn
  // UNDER the individual dots - centered on the REAL LEAGUE-WIDE average
  // movement for that pitch type among OTHER pitchers whose own arm angle
  // is within +/-3deg of this pitcher's (peerMovement, from
  // similar_arm_angle_movement() - a real delivery-comparable peer group,
  // not this pitcher's own self-referential mean, which is what this used
  // to show). Falls back to this pitcher's own average for a pitch type
  // with no qualifying peer data (rare - only if nobody else in the league
  // throws it from a similar slot). Radius still scaled to THIS pitcher's
  // own point spread (a tight pitch gets a small circle, a slider with more
  // break-to-break variance a bigger one) - the peer group only moves the
  // circle's CENTER, not its size. Sized up ~30% from the original per
  // direction.
  const byType = {};
  points.forEach((p) => (byType[p.pitch_type] ||= []).push(p));
  const pxPerInch = plotSize / (2 * domain);
  const avgCircles = Object.entries(byType)
    .map(([pt, pts]) => {
      const n = pts.length;
      const ownHb = pts.reduce((s, p) => s + p.hb, 0) / n;
      const ownIvb = pts.reduce((s, p) => s + p.ivb, 0) / n;
      const peer = peerMovement && peerMovement[pt];
      const mHb = peer ? peer.hb : ownHb;
      const mIvb = peer ? peer.ivb : ownIvb;
      const varHb = pts.reduce((s, p) => s + (p.hb - ownHb) ** 2, 0) / n;
      const varIvb = pts.reduce((s, p) => s + (p.ivb - ownIvb) ** 2, 0) / n;
      const spreadIn = Math.sqrt((varHb + varIvb) / 2);
      const r = Math.max(18.2, Math.min(62.4, spreadIn * pxPerInch * 1.82));
      const color = pitchColor(pt);
      const title = peer
        ? `<title>League avg for pitchers with a similar arm angle (${peer.n_pitchers} pitchers, ${peer.n_pitches} pitches)</title>`
        : "";
      return `<circle cx="${toSvgX(mirror * mHb).toFixed(1)}" cy="${toSvgY(mIvb).toFixed(1)}" r="${r.toFixed(1)}" fill="${color}" fill-opacity="0.16" stroke="${color}" stroke-opacity="0.4" stroke-width="1.5">${title}</circle>`;
    })
    .join("");

  // Numeric ticks every 10in across the domain on both axes, replacing the
  // old bottom pitch-type legend (removed - the summary table/usage widget
  // in the side column already identify each pitch type by color).
  const TICK_STEP = 10;
  let ticks = "";
  for (let v = -domain; v <= domain; v += TICK_STEP) {
    if (v === 0) continue;
    const tx = toSvgX(mirror * v), ty = toSvgY(v);
    ticks += `
      <line x1="${tx}" y1="${toSvgY(0) - 4}" x2="${tx}" y2="${toSvgY(0) + 4}" stroke="#999" stroke-width="1"/>
      <text x="${tx}" y="${toSvgY(0) + 16}" font-size="10" text-anchor="middle" fill="var(--text-muted)">${v}</text>
      <line x1="${toSvgX(0) - 4}" y1="${ty}" x2="${toSvgX(0) + 4}" y2="${ty}" stroke="#999" stroke-width="1"/>
      <text x="${toSvgX(0) - 8}" y="${ty + 3}" font-size="10" text-anchor="end" fill="var(--text-muted)">${v}</text>
    `;
  }

  // Bottom margin tightened (was size+26, a 60px+ gap below the plot's own
  // bottom edge for one line of axis-title text) - moved the "Horizontal
  // Break" title up closer to the plot instead, freeing real vertical
  // footprint for the Similar Players section below this card's row (see
  // renderRightPanel) without touching the card's own overall height.
  const bottomMargin = 16;
  return `
    <svg viewBox="0 0 ${size} ${size + bottomMargin}" width="100%" height="100%" preserveAspectRatio="xMidYMid meet" style="display:block;">
      <rect x="0" y="0" width="${size}" height="${size + bottomMargin}" fill="transparent"/>
      <line x1="${toSvgX(-domain)}" y1="${toSvgY(0)}" x2="${toSvgX(domain)}" y2="${toSvgY(0)}" stroke="#999" stroke-width="1"/>
      <line x1="${toSvgX(0)}" y1="${toSvgY(-domain)}" x2="${toSvgX(0)}" y2="${toSvgY(domain)}" stroke="#999" stroke-width="1"/>
      ${ticks}
      ${armLine}
      ${avgCircles}
      ${circles}
      <text x="${size / 2}" y="${size + 11}" font-size="12" text-anchor="middle" fill="var(--text-muted)">Horizontal Break (in)</text>
      <text x="12" y="${size / 2}" font-size="12" text-anchor="middle" fill="var(--text-muted)" transform="rotate(-90 12 ${size / 2})">Induced Vertical Break (in)</text>
    </svg>
  `;
}

// Rubber-position indicator: a simple mound-rubber bar viewed from the
// catcher/umpire's perspective, with the "foot" highlighted at his real
// average release position (raw, field-relative release_pos_x - NOT
// arm-side-adjusted, since "which side of the rubber" is a fixed physical
// fact shared by every pitcher regardless of throwing hand - some pitchers
// deliberately work from the extreme edge of the rubber for a better angle
// against a given batter hand, independent of which arm they throw with).
//
// Sign convention CORRECTED after a real bug (found via the Heatmaps page -
// see heatmaps.js for the full derivation): NEGATIVE plate_x/release_pos_x
// is the 3B side, POSITIVE is the 1B side - the opposite of what was first
// assumed here from a misread of the Statcast docs. Confirmed against real
// data, not just documentation: RHP sinkers (textbook "tail in on a RHH's
// hands") average plate_x=-0.17 vs RHH, RHP sliders (textbook "run away
// from a same-side hitter") average plate_x=+0.52 - only consistent if
// negative is the 3B/RHH side. Matches the canonical PITCHf/x definition
// too (origin at the catcher facing the pitcher, negative = catcher's left
// = 3B side). Clamped to a generous +/-3ft range since release point (arm
// reach/extension) spans wider than actual foot placement on the 24in
// rubber - a rough, honest indicator, not a precise foot-plant measurement.
const RUBBER_RANGE_FT = 3;

// A real cleat photo (frontend/img/cleat.png - cropped/recolored from the
// user-supplied source image) planted on the rubber, positioned by real
// release_pos_x - the metric is center-of-rubber to center-of-foot, so the
// cleat's own visual center is what lands at that offset, not an edge or a
// separate marker.
//
// The rubber is drawn at its REAL width (24in = 2ft), not stretched to fill
// the whole plottable range - it's a distinct board sitting in the middle of
// a wider dirt strip that spans the full +/-RUBBER_RANGE_FT the foot can
// travel across. A rule of the position is that only the pitcher's TOE has
// to contact the rubber; the center of the foot (what release_pos_x tracks)
// can legitimately sit outside the rubber's own 24in width entirely - so
// the foot icon is free to visually slide past the rubber's edges onto the
// dirt on either side, rather than being clamped to always look like it's
// centered on the rubber board itself (the previous version scaled the
// whole +/-3ft range to the rubber's own drawn width, which made the foot
// look glued to the rubber no matter how extreme the real value was).
const RUBBER_REAL_WIDTH_FT = 2; // 24 inches, official MLB rubber width

function buildRubberIconSVG(releasePosX, throws) {
  if (releasePosX === null || releasePosX === undefined) return "";
  const v = Math.max(-RUBBER_RANGE_FT, Math.min(RUBBER_RANGE_FT, releasePosX));
  // Bigger now that it has this side column to itself (pitch usage moved to
  // the Pitch Locations card instead) - pxPerFt up from 18 to 30. The dirt
  // background is gone (a purely decorative rect, not load-bearing), but
  // the underlying math is unchanged: footX is still positioned on the
  // exact same real-feet scale as the rubber board itself, still free to
  // sit outside the board's own edges when release_pos_x is extreme - that
  // accuracy was never about the dirt, just about not stretching the whole
  // +/-3ft range to fit the rubber's own 24in width.
  const pxPerFt = 30;
  const rangeW = 2 * RUBBER_RANGE_FT * pxPerFt;
  const vbW = rangeW + 20, rangeX = (vbW - rangeW) / 2;
  const rubberW = RUBBER_REAL_WIDTH_FT * pxPerFt, rubberX = rangeX + (rangeW - rubberW) / 2;
  const rubberY = 85, rubberH = 13;
  const footX = rangeX + ((v + RUBBER_RANGE_FT) / (2 * RUBBER_RANGE_FT)) * rangeW;
  let sideLabel;
  if (releasePosX < -1.5) sideLabel = "Extreme 3B";
  else if (releasePosX < -0.3) sideLabel = "3B side";
  else if (releasePosX <= 0.3) sideLabel = "Neutral";
  else if (releasePosX <= 1.5) sideLabel = "1B side";
  else sideLabel = "Extreme 1B";
  const imgW = 56, imgH = imgW * (172 / 360);
  return `
    <div class="rubber-widget">
      <div class="pitch-usage-title">Rubber Position${throws ? ` (${throws}HP)` : ""}</div>
      <svg viewBox="0 0 ${vbW.toFixed(0)} 110" width="100%" style="max-width:240px; display:block; margin:0 auto;">
        <rect x="${rubberX.toFixed(1)}" y="${rubberY}" width="${rubberW.toFixed(1)}" height="${rubberH}" rx="2" fill="#e8e2d0" stroke="#8b7355" stroke-width="2"/>
        <rect x="${(footX - 13).toFixed(1)}" y="${rubberY}" width="26" height="${rubberH}" rx="3" fill="#dc2626" fill-opacity="0.3"/>
        <image href="img/cleat.png" x="${(footX - imgW / 2).toFixed(1)}" y="${(rubberY + rubberH * 0.4 - imgH).toFixed(1)}" width="${imgW.toFixed(1)}" height="${imgH.toFixed(1)}" preserveAspectRatio="xMidYMid meet"/>
      </svg>
      <div class="gauge-avg" style="text-align:center; margin-top:2px;">${sideLabel}</div>
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

// Real spray chart, Savant-style, HITS ONLY (outs are not plotted - this
// chart is about where the ball actually lands when he connects, not a
// full batted-ball scatter). Every hit plotted at its real field-relative
// landing spot and real distance (NOT wall-clamped - a HR by definition
// cleared the fence somewhere, and a real double can legitimately travel
// farther than this player's own shortest homer, so both are free to
// render past the drawn wall line), colored by real hit type (single/
// double/triple orange/blue/yellow, HR pink - real events string from the
// backend, not a distance guess). Every dot's ANGLE is still defensively
// clamped to +/-45 degrees so it's always within/on the foul lines, which
// are themselves drawn out to the farthest real dot rather than stopping
// at the fence corner. A thick brown wall is drawn at the edge of the field.
// The pull/oppo side the batter favors is shaded red (the ONE highlighted
// region - the field is otherwise a single uniform green, no other fill
// color variation), opacity scaled by how pronounced the lean is
// (highlight_side/highlight_opacity come from player_page.py's
// field_zone_pcts - the SAME hand-adjusted angle that drives the Attack
// Direction gauge's dot/label, so this highlight can never visually
// disagree with that gauge). Separately, whichever third has the most
// real raw batted-ball distribution (plain lf/cf/rf%, can legitimately be
// CF) gets its own bold gold OUTLINE on top (a stroke, not a fill, so it
// doesn't add a second field color). Sits at the TOP of the Swing & Field
// card, nudged down a few px from the card's own top edge.
function buildFieldSprayChart(lfPct, cfPct, rfPct, locations, highlightSide, highlightOpacity) {
  const baseSize = 440; // widened from 400 per direction
  const homeY = 372;
  const scale = 1.55;
  // Wall placed at this player's own CLOSEST real home run - every one of
  // his real homers is then guaranteed to sit on or beyond the fence by
  // construction (not just "most," which a fixed leaguewide radius could
  // never promise for every individual player). Falls back to the old
  // fixed leaguewide calibration (125, see git history for how that number
  // was derived) only for the rare case of a player with zero real HRs
  // this season, since there's nothing of his own to anchor to.
  const hrDists = (locations || [])
    .filter((p) => p.is_hr)
    .map((p) => Math.hypot(p.x, p.y));
  const fenceR = (hrDists.length ? Math.min(...hrDists) : 125) * scale;

  // Dots are no longer wall-clamped (see the dots map below) - a real HR by
  // definition cleared the fence SOMEWHERE, and a real double can
  // legitimately have traveled farther than this player's own SHORTEST
  // homer (all fenceR really represents), so both now render at their real,
  // technical distance even past the drawn wall. That means the canvas has
  // to size itself off the farthest real dot, not just fenceR, or a deep
  // shot would get clipped outside the SVG's own viewBox - `size`/`homeX`
  // below only grow past the widened baseline when a real dot actually
  // needs the room.
  const allDists = (locations || []).map((p) => Math.hypot(p.x, p.y) * scale);
  const maxDist = Math.max(fenceR, ...allDists, 0);
  const halfWidthNeeded = maxDist * Math.sin(Math.PI / 4) + 24;
  const size = Math.max(baseSize, halfWidthNeeded * 2);
  const homeX = size / 2;

  const onCircle = (deg, r) => {
    const rad = (deg * Math.PI) / 180;
    return [homeX + Math.sin(rad) * r, homeY - Math.cos(rad) * r];
  };

  const [flx, fly] = onCircle(-45, fenceR);
  const [frx, fry] = onCircle(45, fenceR);
  const [dlx, dly] = onCircle(-15, fenceR);
  const [drx, dry] = onCircle(15, fenceR);
  // Foul lines are drawn out to the farthest real dot (not just the fence
  // corners) so a HR/double plotted past the wall still visually sits
  // inside them, per direction ("all dots within or on the foul lines").
  const foulLineR = maxDist + 15;
  const [foulLx, foulLy] = onCircle(-45, foulLineR);
  const [foulRx, foulRy] = onCircle(45, foulLineR);
  let fenceArc = "";
  for (let d = -45; d <= 45; d += 5) {
    const [ax, ay] = onCircle(d, fenceR);
    fenceArc += ` L ${ax.toFixed(1)},${ay.toFixed(1)}`;
  }

  // ONE highlighted wedge only (the favored pull/oppo lean side) - the
  // field is otherwise a single uniform color per direction, so the
  // unfavored/opposite-side wedge (previously drawn in blue) was removed
  // rather than leaving two colored regions on the field.
  const wedgeArc = (side) => {
    const [d0x, d0y] = side === "lf" ? [flx, fly] : [drx, dry];
    const [d1x, d1y] = side === "lf" ? [dlx, dly] : [frx, fry];
    const from = side === "lf" ? -45 : 15;
    let arc = "";
    for (let d = from; d <= from + 30; d += 5) {
      const [ax, ay] = onCircle(d, fenceR);
      arc += ` L ${ax.toFixed(1)},${ay.toFixed(1)}`;
    }
    return `<path d="M ${homeX},${homeY} L ${d0x.toFixed(1)},${d0y.toFixed(1)}${arc} L ${d1x.toFixed(1)},${d1y.toFixed(1)} Z" fill="#dc2626" fill-opacity="${(0.12 + 0.38 * (highlightOpacity || 0)).toFixed(2)}"/>`;
  };
  let highlightWedge = "";
  if (highlightSide === "lf" || highlightSide === "rf") highlightWedge += wedgeArc(highlightSide);

  // Whichever third actually has the most real batted-ball distribution
  // (plain lf/cf/rf%, not the hand-adjusted pull/oppo angle above - can
  // legitimately be CF, which the pull/oppo wedges above never highlight)
  // gets a bold gold OUTLINE, not a fill - drawn on top of everything else
  // so it stays legible regardless of what fill color already sits under it.
  const distMax = Math.max(lfPct, cfPct, rfPct);
  const distSide = distMax === lfPct ? "lf" : distMax === rfPct ? "rf" : "cf";
  const distRange = { lf: [-45, -15], cf: [-15, 15], rf: [15, 45] }[distSide];
  let distArc = "";
  for (let d = distRange[0]; d <= distRange[1]; d += 5) {
    const [ax, ay] = onCircle(d, fenceR);
    distArc += ` L ${ax.toFixed(1)},${ay.toFixed(1)}`;
  }
  const [dm0x, dm0y] = onCircle(distRange[0], fenceR);
  const [dm1x, dm1y] = onCircle(distRange[1], fenceR);
  const distOutline = `<path d="M ${homeX},${homeY} L ${dm0x.toFixed(1)},${dm0y.toFixed(1)}${distArc} L ${dm1x.toFixed(1)},${dm1y.toFixed(1)} Z" fill="none" stroke="#e6c200" stroke-width="3.5" stroke-linejoin="round"/>`;

  // Singles orange, doubles blue, triples yellow, HR unchanged (pink -
  // doesn't collide with any of the three new colors) - per direction.
  const HIT_COLORS = { single: "#f97316", double: "#2563eb", triple: "#eab308" };
  // Real, technical distance - NOT wall-clamped (reverted per direction: a
  // HR by definition cleared the fence somewhere, and a real double can
  // legitimately travel farther than this player's own shortest homer, so
  // both should be free to render past the drawn wall line instead of
  // being pulled in to sit on it). The only clamp left is angular, not
  // radial: every dot's real angle is capped to +/-45 degrees so it's
  // guaranteed to land within or on the foul lines - real batted-ball hit
  // locations should already satisfy this by construction (a "hit" can't
  // be a foul ball), so this is a defensive safety net, not a correction
  // expected to actually trigger.
  const dots = (locations || [])
    .map((p) => {
      const dist = Math.hypot(p.x, p.y);
      if (dist === 0) return "";
      const angle = Math.max(-45, Math.min(45, (Math.atan2(p.x, p.y) * 180) / Math.PI));
      const [sx, sy] = onCircle(angle, dist * scale);
      if (p.is_hr) {
        return `<circle cx="${sx.toFixed(1)}" cy="${sy.toFixed(1)}" r="5" fill="#ec4899" stroke="#fff" stroke-width="1.5"/>`;
      }
      const color = HIT_COLORS[p.event] || HIT_COLORS.single;
      return `<circle cx="${sx.toFixed(1)}" cy="${sy.toFixed(1)}" r="3.5" fill="${color}" fill-opacity="0.85" stroke="#fff" stroke-width="1"/>`;
    })
    .join("");

  const labelAt = (deg, text) => {
    const [lx, ly] = onCircle(deg, fenceR * 0.66);
    return `<text x="${lx.toFixed(1)}" y="${ly.toFixed(1)}" font-size="17" fill="white" text-anchor="middle" font-weight="bold" style="paint-order:stroke; stroke:#00000066; stroke-width:3px;">${text}</text>`;
  };

  // The field/dots only ever occupy the bottom ~half of this canvas (home
  // plate sits at y=372, and even the shallowest realistic fence apex
  // rarely reaches above y~150) - the rest was pure blank sky, real
  // "negative space" being collapsed per direction. Cropped dynamically off
  // maxDist (the farthest real dot, not just fenceR - see above) so a deep
  // shot never gets clipped above the visible area either.
  const cropTop = Math.max(0, homeY - maxDist - 24);
  const cropH = size - cropTop;

  return `
    <div style="display:flex; justify-content:center; margin-top:14px;">
    <svg viewBox="0 ${cropTop.toFixed(1)} ${size} ${cropH.toFixed(1)}" width="100%" style="max-width:480px; background:transparent;">
      <path d="M ${homeX},${homeY} L ${flx.toFixed(1)},${fly.toFixed(1)}${fenceArc} L ${frx.toFixed(1)},${fry.toFixed(1)} Z" fill="#5b8c3a"/>
      ${highlightWedge}
      <line x1="${homeX}" y1="${homeY}" x2="${foulLx.toFixed(1)}" y2="${foulLy.toFixed(1)}" stroke="#fff" stroke-width="2"/>
      <line x1="${homeX}" y1="${homeY}" x2="${foulRx.toFixed(1)}" y2="${foulRy.toFixed(1)}" stroke="#fff" stroke-width="2"/>
      <line x1="${homeX}" y1="${homeY}" x2="${dlx.toFixed(1)}" y2="${dly.toFixed(1)}" stroke="#ffffff99" stroke-width="1" stroke-dasharray="4,3"/>
      <line x1="${homeX}" y1="${homeY}" x2="${drx.toFixed(1)}" y2="${dry.toFixed(1)}" stroke="#ffffff99" stroke-width="1" stroke-dasharray="4,3"/>
      <path d="M ${flx.toFixed(1)},${fly.toFixed(1)}${fenceArc}" fill="none" stroke="#5c4a3a" stroke-width="5" stroke-linecap="round"/>
      <path d="M ${flx.toFixed(1)},${fly.toFixed(1)}${fenceArc}" fill="none" stroke="#8b7355" stroke-width="2" stroke-linecap="round"/>
      ${dots}
      ${distOutline}
      ${labelAt(-30, `LF ${lfPct.toFixed(0)}%`)}
      ${labelAt(0, `CF ${cfPct.toFixed(0)}%`)}
      ${labelAt(30, `RF ${rfPct.toFixed(0)}%`)}
      <path d="M ${homeX - 13},${homeY} L ${homeX},${homeY - 13} L ${homeX + 13},${homeY} L ${homeX},${homeY + 7} Z" fill="#e8e2d0" stroke="#3a5f27" stroke-width="1.5"/>
    </svg>
    </div>
  `;
}

// --- Swing gauges (Attack Angle / Attack Direction / Swing Length / Swing
// Tilt) - small SVG diagrams pairing this player's real value against the
// league average, in the same visual language as the reference card.
//
// Attack Angle and Swing Tilt measure genuinely different things (the
// bat's angle at the INSTANT of contact vs. the tilt of the swing's whole
// arc/plane through the zone) but a plain rotating needle made both gauges
// look like the same idea. Fixed by giving each concept its own visual
// grammar: Attack Angle draws a rigid bat crossing a FIXED PIVOT point (a
// point-in-time reading), while Swing Tilt is drawn as a fading, widening
// TRAIL along a curve (the same ghost-trail motif this page already uses
// for "a path traveled," not a
// single instant).
function fmtDeg(v) {
  return v === null || v === undefined ? "-" : `${v.toFixed(0)}°`;
}
// --- Archetypes: a small, fun, honest label per gauge based on how far this
// player's value sits from the league average - not another data source,
// just a plain-language read of the same number already shown below it.
function archetypeChip(label) {
  return label ? `<div class="gauge-archetype">${label}</div>` : "";
}
// All four archetype functions below share one cutoff convention (0-30th
// percentile = low-end label, 70th-100th = high-end label, everything
// between = the middle label) - purely a function of the metric's own
// percentile, nothing else (no raw-value thresholds, no vs-league-average
// gap). Real MLB values cluster tightly enough on most of these that a
// fixed-unit threshold rarely triggered for anyone; percentile always
// spreads players across all three buckets regardless of how tight the
// underlying raw-value distribution is.
function attackAngleArchetype(pct) {
  if (pct == null) return "";
  if (pct <= 30) return "Groundball Merchant";
  if (pct >= 70) return "Scoopy";
  return "Line Drive";
}
function attackDirectionArchetype(pct) {
  // attack_direction's percentile is a plain rank (None polarity: no
  // inherent better/worse), so low percentile = most-negative = most pull,
  // high = most-positive = most oppo, by construction.
  if (pct == null) return "";
  if (pct <= 30) return "Pull-Side Pop";
  if (pct >= 70) return "Gets It Deep";
  return "Spray Hitter";
}
function swingLengthArchetype(pct) {
  if (pct == null) return "";
  if (pct <= 30) return "Handsy";
  if (pct >= 70) return "Long Swinger";
  return "Textbook";
}

// Deliberately NOT a needle/dial (that read as "the same idea" as Swing
// Path Tilt's curve). Instead: a fixed ball at a fixed point, and the BAT
// drawn as a solid bar physically crossing through it, tilted by the angle
// - "the bat's angle at the instant it collides with the ball," a impact
// snapshot rather than any kind of rotating gauge. GROUND/LOFT labels replace
// the old horizontal reference line's ambiguity about what "0" means.
// All four swing gauges below share the same SVG viewBox (140x90) and the
// same top/bottom label y-positions (12 / 84) so their graphics line up
// row-to-row in the 2x2 grid instead of each card's content sitting at a
// different height because its own SVG had a different aspect ratio.
const GAUGE_VB = "0 0 140 90";
const GAUGE_TOP_Y = 12, GAUGE_BOTTOM_Y = 84, GAUGE_MID_Y = 45;

function buildAttackAngleGauge(value, leagueAvg, pct) {
  const v = Math.max(-30, Math.min(45, value ?? 0));
  const contactX = 68, contactY = GAUGE_MID_Y, halfLen = 26;
  // SVG y grows DOWNWARD, so a positive (loft) angle needs a NEGATIVE
  // screen slope (right end higher = smaller y). halfLen kept short enough
  // (26) that even the max +/-45deg angle can't reach the label rows above/
  // below (worst case dy = 26*sin(45)=18.4, well clear of GAUGE_TOP_Y/
  // GAUGE_BOTTOM_Y) - a real, previously-shipped overlap bug at the old
  // halfLen=36 on a shorter viewBox.
  const rad = (v * Math.PI) / 180;
  const dx = halfLen * Math.cos(rad), dy = halfLen * Math.sin(rad);
  const x1 = contactX - dx, y1 = contactY + dy;
  const x2 = contactX + dx, y2 = contactY - dy;
  return `
    <div class="gauge-card">
      ${archetypeChip(attackAngleArchetype(pct))}
      <div class="gauge-title">Attack Angle</div>
      <div class="gauge-body">
        <svg viewBox="${GAUGE_VB}" width="100%" style="max-width:220px;">
          <text x="4" y="${GAUGE_TOP_Y}" font-size="9" fill="var(--text-muted)" font-weight="700">LOFT</text>
          <line x1="15" y1="${GAUGE_MID_Y}" x2="125" y2="${GAUGE_MID_Y}" stroke="var(--border)" stroke-width="1.5" stroke-dasharray="3,3"/>
          <text x="4" y="${GAUGE_BOTTOM_Y}" font-size="9" fill="var(--text-muted)" font-weight="700">GROUND</text>
          <line x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}" stroke="#8b5a2b" stroke-width="7" stroke-linecap="round"/>
          <circle cx="${contactX}" cy="${contactY}" r="6.5" fill="#fff" stroke="#dc2626" stroke-width="2.5"/>
        </svg>
        <div class="gauge-value">${fmtDeg(value)}</div>
        <div class="gauge-avg">MLB avg: ${fmtDeg(leagueAvg)}${pct != null ? ` &middot; ${fmtPct(pct)}` : ""}</div>
      </div>
    </div>
  `;
}

function buildAttackDirectionGauge(value, leagueAvg, pct) {
  const v = Math.max(-45, Math.min(45, value ?? 0));
  const frac = (v + 45) / 90; // 0=full pull, 1=full oppo
  const cx = 15 + frac * 110;
  return `
    <div class="gauge-card">
      ${archetypeChip(attackDirectionArchetype(pct))}
      <div class="gauge-title">Attack Direction</div>
      <div class="gauge-body">
        <svg viewBox="${GAUGE_VB}" width="100%" style="max-width:220px;">
          <text x="4" y="${GAUGE_TOP_Y}" font-size="9" fill="var(--forest-light)" font-weight="700">PULL</text>
          <text x="112" y="${GAUGE_TOP_Y}" font-size="9" fill="var(--text-muted)" font-weight="700">OPPO</text>
          <line x1="15" y1="${GAUGE_MID_Y}" x2="125" y2="${GAUGE_MID_Y}" stroke="var(--border)" stroke-width="2"/>
          <line x1="70" y1="${GAUGE_MID_Y - 6}" x2="70" y2="${GAUGE_MID_Y + 6}" stroke="var(--border)" stroke-width="1.5"/>
          <circle cx="${cx.toFixed(1)}" cy="${GAUGE_MID_Y}" r="5" fill="#dc2626"/>
        </svg>
        <div class="gauge-value">${fmtDeg(value)}</div>
        <div class="gauge-avg">MLB avg: ${fmtDeg(leagueAvg)}${pct != null ? ` &middot; ${fmtPct(pct)}` : ""}</div>
      </div>
    </div>
  `;
}

function fmtFt(v) {
  return v === null || v === undefined ? "-" : `${v.toFixed(1)} ft`;
}
function ordinal(n) {
  const rem100 = n % 100;
  if (rem100 >= 11 && rem100 <= 13) return `${n}th`;
  return `${n}${["th", "st", "nd", "rd"][n % 10] || "th"}`;
}
function fmtPct(v) {
  return v === null || v === undefined ? "" : `${ordinal(Math.round(v))} percentile`;
}

// Swing Length replaces Swing Path Tilt (see player_page.py comments) - a
// simple horizontal bar filling to this swing's length against a fixed
// reasonable range, same "gauge-card" visual language as the others.
const SWING_LENGTH_MIN = 5.5, SWING_LENGTH_MAX = 8.5;

function buildSwingLengthGauge(value, leagueAvg, pct) {
  // Slider fill is driven by PERCENTILE (50th = half full), not the raw
  // value against a fixed range - same reasoning as Bat Speed's needle:
  // real swing lengths cluster tightly, so a raw-value fill barely moves
  // for most players. Falls back to the value-based fill only if this
  // player has no percentile (shouldn't normally happen).
  const frac = pct != null
    ? Math.max(0, Math.min(100, pct)) / 100
    : (Math.max(SWING_LENGTH_MIN, Math.min(SWING_LENGTH_MAX, value ?? SWING_LENGTH_MIN)) - SWING_LENGTH_MIN) / (SWING_LENGTH_MAX - SWING_LENGTH_MIN);
  const barX = 15, barW = 110, barY = GAUGE_MID_Y - 7;
  const fillW = frac * barW;
  return `
    <div class="gauge-card">
      ${archetypeChip(swingLengthArchetype(pct))}
      <div class="gauge-title">Swing Length</div>
      <div class="gauge-body">
        <svg viewBox="${GAUGE_VB}" width="100%" style="max-width:220px;">
          <text x="4" y="${GAUGE_TOP_Y}" font-size="9" fill="var(--text-muted)" font-weight="700">SHORT</text>
          <text x="98" y="${GAUGE_TOP_Y}" font-size="9" fill="var(--text-muted)" font-weight="700">LONG</text>
          <rect x="${barX}" y="${barY}" width="${barW}" height="14" rx="7" fill="var(--surface)" stroke="var(--border)"/>
          <rect x="${barX}" y="${barY}" width="${fillW.toFixed(1)}" height="14" rx="7" fill="#8b5a2b"/>
          <circle cx="${(barX + fillW).toFixed(1)}" cy="${GAUGE_MID_Y}" r="7" fill="#fff" stroke="#8b5a2b" stroke-width="2.5"/>
        </svg>
        <div class="gauge-value">${fmtFt(value)}</div>
        <div class="gauge-avg">MLB avg: ${fmtFt(leagueAvg)}${pct != null ? ` &middot; ${fmtPct(pct)}` : ""}</div>
      </div>
    </div>
  `;
}

function swingTiltArchetype(pct) {
  if (pct == null) return "";
  if (pct <= 30) return "Worm Burner";
  if (pct >= 70) return "Golfer";
  return "Good Tilt";
}

// Swing Tilt replaces Bat Speed here (bat speed is still shown elsewhere -
// this slot is for the shape of the swing plane, not its velocity). Drawn
// as a fading comet trail sweeping from the hands (pivot) out toward the
// bat head - "a path traveled through the zone," not a single instant
// (that's Attack Angle's job: a rigid bat crossing a fixed point). Same
// angle convention as the fixed Attack Angle gauge (positive = steeper,
// right end rises, SVG y grows downward so sin(v) is subtracted).
// A real bat silhouette (handle + tapered barrel, not just a straight
// line/trail) pivoting from a fixed "hands" point near the top-left, tilted
// by the swing plane's real angle - FLAT keeps the barrel level/pointing
// right, STEEP droops it DOWNWARD (a steeper swing plane cuts down through
// the zone more, not up - previously this rotated the bat upward for
// steeper values, backwards from the real mechanic). Distinct from Attack
// Angle's short rigid bat-crossing-a-ball (a point-in-time contact reading)
// by being a longer, full-bat shape swept from a fixed pivot - "the plane
// the whole swing travels through," not an instant.
function buildSwingTiltGauge(value, leagueAvg, pct) {
  const v = Math.max(0, Math.min(55, value ?? 0));
  const pivotX = 18, pivotY = GAUGE_TOP_Y + 10;
  // Two dashed reference rays FROM the pivot (the white dot), not a single
  // fixed line independent of it - FLAT is a true 0deg horizontal through
  // the dot, STEEP is drawn at the real league-average tilt angle (same
  // rotation convention as the bat below: positive degrees droops the ray
  // downward), so the two rays double as "what flat looks like" and "what
  // average looks like" for comparing this player's own bat angle against.
  const REF_LEN = 80;
  const flatX2 = pivotX + REF_LEN, flatY2 = pivotY;
  const lgRad = ((leagueAvg ?? 0) * Math.PI) / 180;
  const lgX2 = pivotX + REF_LEN * Math.cos(lgRad);
  const lgY2 = pivotY + REF_LEN * Math.sin(lgRad);
  return `
    <div class="gauge-card">
      ${archetypeChip(swingTiltArchetype(pct))}
      <div class="gauge-title">Swing Tilt</div>
      <div class="gauge-body">
        <svg viewBox="${GAUGE_VB}" width="100%" style="max-width:220px;">
          <line x1="${pivotX}" y1="${pivotY}" x2="${flatX2.toFixed(1)}" y2="${flatY2.toFixed(1)}" stroke="var(--border)" stroke-width="1.5" stroke-dasharray="3,3"/>
          <text x="${(flatX2 - 26).toFixed(1)}" y="${(flatY2 - 5).toFixed(1)}" font-size="9" fill="var(--text-muted)" font-weight="700">FLAT</text>
          <line x1="${pivotX}" y1="${pivotY}" x2="${lgX2.toFixed(1)}" y2="${lgY2.toFixed(1)}" stroke="var(--border)" stroke-width="1.5" stroke-dasharray="3,3"/>
          <text x="${(lgX2 - 10).toFixed(1)}" y="${(lgY2 + 13).toFixed(1)}" font-size="9" fill="var(--text-muted)" font-weight="700">STEEP</text>
          <g transform="rotate(${v.toFixed(1)} ${pivotX} ${pivotY})">
            <rect x="${pivotX}" y="${pivotY - 3}" width="36" height="6" rx="3" fill="#8b5a2b"/>
            <rect x="${pivotX + 30}" y="${pivotY - 7.5}" width="36" height="15" rx="7.5" fill="#8b5a2b"/>
          </g>
          <circle cx="${pivotX}" cy="${pivotY}" r="4" fill="#fff" stroke="#8b5a2b" stroke-width="2"/>
        </svg>
        <div class="gauge-value">${fmtDeg(value)}</div>
        <div class="gauge-avg">MLB avg: ${fmtDeg(leagueAvg)}${pct != null ? ` &middot; ${fmtPct(pct)}` : ""}</div>
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
    const rp = data.rubber_position || {};
    mount.innerHTML = `
      <div class="card movement-plot-card">
        <div class="movement-plot-row">
          <div class="movement-plot-svg-wrap">${buildMovementPlotSVG(mv.points, mv.arm_angle, rp.throws, mv.peer_avg_movement)}</div>
          <div class="movement-plot-side-col">
            ${buildMovementSummaryTable(mv.summary)}
            ${buildRubberIconSVG(rp.release_pos_x, rp.throws)}
            ${data.babip_against ? `<div class="babip-widget"><div class="pitch-usage-title">BABIP Against</div><div class="gauge-value" style="text-align:center;">${data.babip_against.value_str}</div></div>` : ""}
          </div>
        </div>
        <div id="similar-players-mount"></div>
      </div>
      <div class="movement-plot-caption">Dashed line = this pitcher's average arm angle (labeled). Shaded circles = the league-wide average movement for that pitch among OTHER pitchers with a similar (within 3&#176;) arm angle - hover for the pitcher count. Arm-side-adjusted break, matching Baseball Savant - positive HB is arm side for either hand. Only the 50% of pitches closest to each pitch type's own average are plotted; the table on the right uses every pitch. Fixed across the L/R toggle (a pitch's movement doesn't depend on who's in the box) - still updates with season. Pitch usage by batter side is shown on the Pitch Locations card.</div>
    `;
  } else {
    const sf = data.swing_field;
    const sw = data.swing_3d;
    const lg = sw.league_avg;
    mount.innerHTML = `
      <div class="card swing-field-card">
        ${buildFieldSprayChart(sf.lf, sf.cf, sf.rf, sf.locations, sf.highlight_side, sf.highlight_opacity)}
        <div class="gauge-grid">
          ${buildAttackAngleGauge(sw.attack_angle, lg.attack_angle, sw.attack_angle_pct)}
          ${buildAttackDirectionGauge(sw.attack_direction, lg.attack_direction, sw.attack_direction_pct)}
          ${buildSwingLengthGauge(sw.swing_length, lg.swing_length, sw.swing_length_pct)}
          ${buildSwingTiltGauge(sw.swing_path_tilt, lg.swing_path_tilt, sw.swing_path_tilt_pct)}
        </div>
        <div id="similar-players-mount"></div>
      </div>
    `;
  }
}

// Similar Players - shared by both roles (Movement Plot card for pitchers,
// Swing & Field card for batters share the same #similar-players-mount id,
// mutually exclusive since only one role renders at a time). Fetched
// separately from the main player payload (its own endpoint, its own
// season-spanning pool - see api/routers/similar_players.py) rather than
// blocking the rest of the card on it; renders into the placeholder div
// left inside renderRightPanel's per-role markup above.
async function renderSimilarPlayers() {
  const mount = document.getElementById("similar-players-mount");
  if (!mount) return;
  const role = state.mode === "pitcher" ? "pitcher" : "batter";
  // Doesn't vary with split at all (no split param in the URL) - cached
  // separately from the main player payload (keyed without split) so
  // toggling ALL/L/R never re-fetches an answer that can't have changed.
  const cacheKey = `${state.playerId}|${role}|${state.season}`;
  let d = similarPlayersCache.get(cacheKey);
  if (!d) {
    d = await getJSON(`/api/player/${state.playerId}/similar?role=${role}&season=${state.season}`);
    similarPlayersCache.set(cacheKey, d);
  }
  if (!document.getElementById("similar-players-mount")) return;
  if (!d.qualified || !d.comps.length) {
    mount.innerHTML = `
      <div class="sp-title">SIMILAR PLAYERS</div>
      <div class="sp-empty">Not enough qualifying playing time this season for a real comp.</div>
    `;
    return;
  }
  const cards = d.comps
    .map((c) => {
      const teamLogo = c.team_id ? `<img class="sp-team-logo" src="https://www.mlbstatic.com/team-logos/${c.team_id}.svg" onerror="this.style.visibility='hidden'" />` : "";
      const headshot = c.player_id
        ? `<img class="sp-headshot" src="https://img.mlbstatic.com/mlb-photos/image/upload/d_people:generic:headshot:67:current.png/w_120,q_auto:best/v1/people/${c.player_id}/headshot/67/current" alt="${c.name || ""}" />`
        : "";
      const url = `index.html?player=${c.player_id}&season=${c.season}`;
      return `
        <div class="sp-comp" role="button" tabindex="0" onclick="location.href='${url}'" onkeydown="if(event.key==='Enter'||event.key===' ')location.href='${url}'">
          ${headshot}
          ${teamLogo}
          <a class="sp-comp-name" href="${url}" onclick="event.stopPropagation()">${c.name || "Unknown"}</a>
          <div class="sp-comp-season">${c.season}</div>
        </div>
      `;
    })
    .join("");
  mount.innerHTML = `
    <div class="sp-title">SIMILAR PLAYERS <span class="sp-subtitle">closest career comps, any season</span></div>
    <div class="sp-row">${cards}</div>
  `;
}

// Last 5 games - real, live box-score stats (RBI/SB/ER aren't derivable
// from Statcast alone), pitchers: starts only, batters: any game played -
// see analytics/player_page.py's player_last_n_games(). Sits below Pitch
// Locations (pitchers) / the heat-zone card (batters), same card width,
// filling flex-1 down to the Movement/Swing card's own bottom edge.
function last5RowHtml(g, role) {
  const s = g.stats || {};
  const cells = role === "pitcher"
    ? [["IP", s.ip ?? "-"], ["H", s.h ?? 0], ["K", s.k ?? 0], ["R", s.r ?? 0], ["ER", s.er ?? 0]]
    : [["AB", s.ab ?? "-"], ["H", s.h ?? 0], ["HR", s.hr ?? 0], ["RBI", s.rbi ?? 0], ["SB", s.sb ?? 0]];
  const walkoff = g.walkoff ? `<span class="walkoff-badge">WALK-OFF</span>` : "";
  // For pitchers, make the date a link to the outing breakdown
  const dateContent = (role === "pitcher" && g.game_id && state.playerId)
    ? `<a class="last5-outing-link" href="outings.html?player_id=${state.playerId}&game_id=${g.game_id}&season=${state.season}">${g.date}</a>`
    : g.date;
  return `
    <div class="last5-row${g.top_performer ? " last5-row-top-performer" : ""}">
      <div class="last5-meta">
        <div class="last5-date">${dateContent}${walkoff}</div>
        <div class="last5-opp">${g.opponent || ""}</div>
      </div>
      <div class="last5-cells">
        ${cells.map(([label, value]) => `<div class="stat-cell"><div class="stat-cell-value">${value}</div><div class="stat-cell-label">${label}</div></div>`).join("")}
      </div>
    </div>
  `;
}

function ilGapRowHtml(il) {
  const label = (il.il_label || "").replace("Injured ", "").trim() || "IL";
  const fmtD = (d) => d ? new Date(d + "T12:00:00").toLocaleDateString("en-US", { month: "short", day: "numeric" }) : "";
  const dates = il.min_return_date
    ? `${fmtD(il.start_date)} · eligible ${fmtD(il.min_return_date)}`
    : fmtD(il.start_date);
  return `
    <div class="last5-il-gap">
      <span class="last5-il-label">${label}</span>
      <span class="last5-il-dates">${dates}</span>
      ${il.injury ? `<span class="last5-il-injury">${il.injury}</span>` : ""}
    </div>`;
}

function buildLast5Card(games, role, ilStatus) {
  if (!games || !games.length) {
    return `<div class="card"><div class="empty-state">No recent games found.</div></div>`;
  }
  // Backend returns most-recent-first (see player_last_n_games) - reversed
  // here so the card reads chronologically top-to-bottom, oldest at top,
  // most recent at the bottom.
  const chronological = [...games].reverse();

  if (!ilStatus || !ilStatus.start_date) {
    return `<div class="card last5-card">${chronological.map((g) => last5RowHtml(g, role)).join("")}</div>`;
  }

  // Insert IL gap card at the correct chronological position.
  let rows = "";
  let inserted = false;
  for (let i = 0; i < chronological.length; i++) {
    const g = chronological[i];
    rows += last5RowHtml(g, role);
    const next = chronological[i + 1];
    if (!inserted && next && g.date < ilStatus.start_date && next.date >= ilStatus.start_date) {
      rows += ilGapRowHtml(ilStatus);
      inserted = true;
    }
  }
  // IL started after the last game we have (current/ongoing stint)
  if (!inserted) {
    const last = chronological[chronological.length - 1];
    if (last && last.date < ilStatus.start_date) rows += ilGapRowHtml(ilStatus);
  }
  return `<div class="card last5-card">${rows}</div>`;
}

// Hit Zones - real xwOBA per 3x3 strike-zone cell, toggleable by pitch
// type, replacing the batter's old percentile left-panel slot (percentiles
// moved to the shared full-width row below, matching pitchers). PITCHER'S
// VIEW now (was Catcher's View - switched per direction, to match the
// Pitch Locations card's own convention rather than deliberately diverge
// from it). The backend's zone_idx (row*3+col) is still computed from raw,
// unmirrored plate_x (see api/routers/hitzones.py) - the mirror is applied
// PURELY at render time here, by flipping which screen column each col
// lands in (2-col instead of col), same "mirror only in toSvgX" pattern
// buildPitchLocationsSVG already uses, so the backend's zone numbering
// never has to change. Opponent-hand split comes from the page's own main
// ALL/L/R toggle, not a separate local one. Silhouette flanks BOTH sides of
// the zone, same structural layout as the Pitch Locations card's two
// batter boxes (just the same player mirrored on each side here, not two
// different batters) - zone drawn at a fixed, roomy box rather than its
// real (very narrow) aspect ratio so per-cell numbers stay legible, shrunk
// moderately from an earlier oversized version and vertically centered on
// the silhouette's own hip height (roughly 55% of body height up from the
// feet - a standard anthropometric approximation, not this specific image's
// measured pixels) rather than the whole card's midpoint, so the zone sits
// at a visually plausible height relative to the batter it's drawn beside.
const HITZONE_W = 460, HITZONE_H = 320;
const HITZONE_IMG_ASPECT = 501 / 1262;
// Shrunk ~30% both dimensions per direction (was 150x180) - label font
// sizes below are scaled down by the same 0.7 factor to match.
const HITZONE_ZONE_W = 150, HITZONE_ZONE_H = 180;
const HITZONE_HIP_FRAC = 0.55;
// Very slight downward nudge per direction - everything else about the
// zone's position/size is unchanged from before the (reverted) shrink.
const HITZONE_ZONE_Y_SHIFT = 6;
let currentHitZonePitchType = "ALL";

function xwobaColor(xwoba) {
  if (xwoba === null || xwoba === undefined) return "var(--surface)";
  // .200-.500 xwOBA mapped to the full blue(bad)-grey-red(good) range - a
  // realistic span for a per-zone (often small-sample) split rather than a
  // season aggregate, which would compress almost everyone into one end.
  const pct = Math.max(0, Math.min(100, ((xwoba - 0.2) / (0.5 - 0.2)) * 100));
  return gradientColor(pct, 0.8);
}

function buildHitZonesSVG(d, bats) {
  const groundY = HITZONE_H - 15;
  const boxW = 120, boxH = HITZONE_H - 40;
  const imgH = boxH * 0.92, imgW = imgH * HITZONE_IMG_ASPECT;
  const leftBoxX0 = 10, rightBoxX1 = HITZONE_W - 10;

  const zoneX0 = HITZONE_W / 2 - HITZONE_ZONE_W / 2;
  const hipY = groundY - HITZONE_HIP_FRAC * imgH;
  const zoneTopY = hipY - HITZONE_ZONE_H / 2 + HITZONE_ZONE_Y_SHIFT;
  const cellW = HITZONE_ZONE_W / 3, cellH = HITZONE_ZONE_H / 3;

  const cells = (d.cells[currentHitZonePitchType] || []);
  const cellsHtml = cells
    .map((c) => {
      const row = Math.floor(c.zone / 3), rawCol = c.zone % 3;
      const col = 2 - rawCol; // pitcher's view mirror - see module comment above
      const x = zoneX0 + col * cellW, y = zoneTopY + row * cellH;
      const label = c.xwoba !== null ? c.xwoba.toFixed(3).replace(/^0/, "") : "-";
      return `
        <rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${cellW.toFixed(1)}" height="${cellH.toFixed(1)}"
              fill="${xwobaColor(c.xwoba)}" stroke="var(--surface-alt)" stroke-width="2.5" />
        <text x="${(x + cellW / 2).toFixed(1)}" y="${(y + cellH / 2 + 6).toFixed(1)}" font-size="19" font-weight="800"
              text-anchor="middle" fill="var(--forest-dark)">${label}</text>
        <text x="${(x + cellW / 2).toFixed(1)}" y="${(y + cellH / 2 + 24).toFixed(1)}" font-size="10.5"
              text-anchor="middle" fill="var(--text-muted)">${c.n} P</text>
      `;
    })
    .join("");

  // Only ONE silhouette should ever be visible - the batter's own real
  // hand - not the same image mirrored onto both sides. Same swapped-file
  // correction and left/right-by-hand convention the Pitch Locations chart
  // already uses (RHH box = right box on screen under Pitcher's View), so
  // the invisible box is opacity 0 exactly like that chart's inactive side,
  // not a second copy of the same real person.
  const lhhFile = "batter_rhh.png", rhhFile = "batter_lhh.png";
  // Dashed batter's-box outline, same visual treatment as the Pitch
  // Locations card's own batterBox() - was missing here entirely, drawn
  // regardless of which side is active (matching that card's convention of
  // showing both boxes, only the silhouette image itself toggling opacity).
  const boxTop = groundY - boxH;
  const silhouette = (x0, x1, file, active) => {
    const imgX = (x0 + x1) / 2 - imgW / 2, imgY = groundY - imgH;
    return `<image href="img/${file}" x="${imgX.toFixed(1)}" y="${imgY.toFixed(1)}" width="${imgW.toFixed(1)}" height="${imgH.toFixed(1)}" opacity="${active ? 1 : 0}" preserveAspectRatio="xMidYMax meet" />`;
  };
  return `
    <svg viewBox="0 0 ${HITZONE_W} ${HITZONE_H}" width="100%" style="max-width:500px; display:block; margin:0 auto;">
      <rect x="0" y="0" width="${HITZONE_W}" height="${HITZONE_H}" fill="var(--surface-alt)" />
      ${silhouette(leftBoxX0, leftBoxX0 + boxW, lhhFile, bats === "L")}
      ${silhouette(rightBoxX1 - boxW, rightBoxX1, rhhFile, bats !== "L")}
      <rect x="${zoneX0.toFixed(1)}" y="${zoneTopY.toFixed(1)}" width="${HITZONE_ZONE_W}" height="${HITZONE_ZONE_H}"
            fill="none" stroke="var(--forest-dark)" stroke-width="2.5" />
      ${cellsHtml}
    </svg>
  `;
}

async function renderHitZones() {
  const mount = document.getElementById("left-panel-mount");
  const cacheKey = `${state.playerId}|${state.season}|${state.split}`;
  let d = hitzoneCache.get(cacheKey);
  if (!d) {
    d = await getJSON(`/api/hitzones/${state.playerId}?season=${state.season}&split=${state.split}`);
    hitzoneCache.set(cacheKey, d);
  }
  const bats = (currentPlayerBats || "R");
  if (!d.has_data) {
    mount.innerHTML = `<div class="card"><div class="empty-state">No qualifying at-bats for this selection.</div></div>`;
    return;
  }
  if (!d.cells[currentHitZonePitchType]) currentHitZonePitchType = "ALL";
  // Order: fastballs → offspeed → breaking; filter out <1% of total pitches
  const PITCH_DISPLAY_ORDER = ["FF","SI","FA","FC","FT","CH","FS","SL","CU","ST","KC","SV","CS","SP"];
  const totalPitches = (d.cells["ALL"] || []).reduce((s, c) => s + (c.n || 0), 0);
  const filteredTypes = d.pitch_types.filter(pt => {
    const ptTotal = (d.cells[pt] || []).reduce((s, c) => s + (c.n || 0), 0);
    return totalPitches === 0 || ptTotal / totalPitches >= 0.01;
  }).sort((a, b) => {
    const ia = PITCH_DISPLAY_ORDER.indexOf(a), ib = PITCH_DISPLAY_ORDER.indexOf(b);
    return (ia < 0 ? 999 : ia) - (ib < 0 ? 999 : ib);
  });
  const pitchTypes = ["ALL", ...filteredTypes];
  // Centered (not the shared left-panel header's default flex-end) and
  // each pitch-type button tinted with a translucent wash of that pitch's
  // own color (pitchColor(), same palette the movement plot/pitch usage
  // bar use) - "ALL" gets no tint, it isn't one pitch type.
  const toggleHtml = `
    <div class="toggle-group toggle-group-sm" id="hitzone-pt-toggle">
      ${pitchTypes.map((pt) => {
        const active = pt === currentHitZonePitchType;
        const style = pt === "ALL" ? "" : `style="background:${hexToRgba(pitchColor(pt), active ? 0.85 : 0.22)}; border-color:${pitchColor(pt)}; color:${active ? "#fff" : "var(--text)"};"`;
        return `<button data-value="${pt}" class="${active ? "active" : ""}" ${style}>${pt}</button>`;
      }).join("")}
    </div>
  `;
  mount.innerHTML = `
    <div class="card pitch-locations-card">
      <div class="pitch-locations-header hitzone-header-centered">${toggleHtml}</div>
      <div class="heatmap-subtitle">Pitcher's View</div>
      <div class="heatmap-wrap">${buildHitZonesSVG(d, bats)}</div>
      <div class="heatmap-caption">xwOBA by zone (real qualifying PAs only) &middot; zone drawn at a fixed, larger box for readability, not its real narrow aspect ratio</div>
    </div>
  `;
  document.getElementById("hitzone-pt-toggle").addEventListener("click", (e) => {
    const btn = e.target.closest("button");
    if (!btn || btn.dataset.value === currentHitZonePitchType) return;
    currentHitZonePitchType = btn.dataset.value;
    renderHitZones();
  });
}

// Pitch-type table column groups - "real" (sample-size) vs "outcome"
// (quality/physical) vs "percent" (rate) stats - shared by BOTH the
// pitcher table (usage_pct/n_pitches | velo/spin/ivb/hb | zone_pct/
// chase_pct/whiff_pct/k_pct/stuff_plus) and the batter table (n_pitches |
// OPS/wOBA/xwOBA | K_pct/whiff_pct/barrel_pct) - a union of both tables'
// group-START column keys, since this one renderer draws both. A faint
// left border on whichever column key starts a new group, not a whole
// separate divider element, keeps real column widths/alignment untouched.
const TABLE_GROUP_START_KEYS = new Set(["velo", "zone_pct", "OPS", "K_pct"]);

function renderTable(tableData, caption) {
  document.getElementById("table-caption").textContent = caption || "";
  const mount = document.getElementById("table-mount");
  if (!tableData || !tableData.rows.length) {
    mount.innerHTML = `<div class="empty-state">No data for this selection.</div>`;
    return;
  }
  const groupCls = (key) => (TABLE_GROUP_START_KEYS.has(key) ? " col-group-start" : "");
  const headerCells = tableData.columns.map((c) => `<th class="${groupCls(c.key).trim()}">${c.label}</th>`).join("");
  const bodyRows = tableData.rows
    .map((row) => {
      const color = pitchColor(row.pitch_type);
      const tint = hexToRgba(color, 0.12);
      const cells = row.cells.map((c) => `<td class="${groupCls(c.key).trim()}" style="background:${tint};">${c.value}</td>`).join("");
      return `<tr><td class="pitch-type-cell" style="background:${color};">${pitchFullName(row.pitch_type)}</td>${cells}</tr>`;
    })
    .join("");

  let leagueRow = "";
  if (tableData.summary_row) {
    const cells = tableData.summary_row.map((c) => `<td class="${groupCls(c.key).trim()}">${c.value}</td>`).join("");
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

// In-memory cache of full player payloads, keyed by playerId|mode|split|
// season - GET /api/player/{id} and its own /progress variant do the exact
// same server-side computation and return the exact same shape (the
// streaming version just adds checkpoint events along the way), so a
// response fetched silently in the background is just as usable as one the
// user waited on. Cleared on "Refresh data" (window.onDataRefresh below),
// since the underlying data can genuinely change then - never cleared just
// from browsing, a session's worth of players/splits is a small amount of
// data to hold onto and the whole point is to not re-pay these fetches.
const playerDataCache = new Map();
const playerDataInFlight = new Map(); // same key -> in-progress prefetch Promise, so rapid split toggling can't fire duplicate background requests for the same combination

// Similar Players never varies with split (see renderSimilarPlayers) -
// keyed without it so an ALL/L/R toggle can never re-fetch the same
// answer. Pitch Locations (heatmaps) / Hit Zones DO vary with split -
// cached per (playerId, season, split) same as the main payload, and
// warmed by the same background prefetch.
const similarPlayersCache = new Map();
const heatmapCache = new Map();
const hitzoneCache = new Map();

function playerCacheKey(playerId, mode, split, season) {
  return `${playerId}|${mode}|${split}|${season}`;
}

// Silently fetches and caches whichever of ALL/L/R aren't already cached or
// mid-fetch for this exact player/mode/season - fired after every real
// load completes so that by the time someone actually clicks the other
// split buttons, the answer is usually already sitting in memory instead
// of a live wait. Deliberately uses the plain (non-streaming) endpoint -
// nothing is watching a progress bar for a prefetch, so there's no reason
// to pay the NDJSON parsing overhead for checkpoint events nobody sees.
function prefetchOtherSplits(playerId, mode, season) {
  for (const split of ["ALL", "L", "R"]) {
    const key = playerCacheKey(playerId, mode, split, season);
    if (playerDataCache.has(key) || playerDataInFlight.has(key)) continue;
    const promise = getJSON(`/api/player/${playerId}?role=${mode}&split=${split}&season=${season}`)
      .then((data) => {
        playerDataCache.set(key, data);
      })
      .catch(() => {
        // Silent - a failed background prefetch just means that split falls
        // back to a normal live fetch (with its own error handling) if the
        // user actually clicks it, same as before this existed.
      })
      .finally(() => {
        playerDataInFlight.delete(key);
      });
    playerDataInFlight.set(key, promise);
  }

  // Pitch Locations (pitchers) / Hit Zones (batters) are the other two
  // per-request fetches loadPlayer() always makes - warm those for the
  // other splits too, so the left-panel card is ALSO instant when the
  // cached player payload above renders, not just the stat cards around
  // it. Heatmaps only ever take a real hand (L/R, never ALL - see
  // heatmapSplit in the state comment at the top of this file); Hit
  // Zones take the full ALL/L/R range, matching the main toggle exactly.
  if (mode === "pitcher") {
    for (const split of ["L", "R"]) {
      const key = `${playerId}|${season}|${split}`;
      if (heatmapCache.has(key)) continue;
      getJSON(`/api/heatmaps/${playerId}?season=${season}&split=${split}`)
        .then((d) => heatmapCache.set(key, d))
        .catch(() => {});
    }
  } else {
    for (const split of ["ALL", "L", "R"]) {
      const key = `${playerId}|${season}|${split}`;
      if (hitzoneCache.has(key)) continue;
      getJSON(`/api/hitzones/${playerId}?season=${season}&split=${split}`)
        .then((d) => hitzoneCache.set(key, d))
        .catch(() => {});
    }
  }
}

// Pitch Locations - integrated into the player page's left-column slot for
// pitchers (was its own separate heatmaps.html page; see MEMORY for the
// two-round orientation-bug history behind this chart's geometry). One dot
// per pitch TYPE at its own season-average (plate_x, plate_z), radius from
// its own spread. Colored via the shared pitchColor() palette so a pitch
// type reads consistently against the movement plot elsewhere on this page.
// Z_MAX_FT deliberately tight (was 6.3, full batter-height headroom left
// over from when a standing silhouette lived in this chart) - nothing
// renders above roughly the zone top plus a small margin any more, so that
// headroom was pure dead space. W/H are separate (not one square SIZE) so
// the card can be short without also cramping the two batter boxes
// horizontally.
const HEATMAP_X_RANGE_FT = 5.3, HEATMAP_Z_MIN_FT = 0, HEATMAP_Z_MAX_FT = 4.3;
const HEATMAP_W = 460, HEATMAP_H = 310, HEATMAP_PAD_X = 30, HEATMAP_PAD_TOP = 10, HEATMAP_PAD_BOTTOM = 24;
const HEATMAP_DOT_R_MIN = 5, HEATMAP_DOT_R_MAX = 13;
const HEATMAP_ABS_SPREAD_MIN_FT = 0.85, HEATMAP_ABS_SPREAD_MAX_FT = 1.5;
const HEATMAP_BOX_INNER_FT = 1.3, HEATMAP_BOX_OUTER_FT = 4.1;
const HEATMAP_TOTAL_W = HEATMAP_W;

function buildPitchLocationsSVG(d, split, usage) {
  const plotW = HEATMAP_W - HEATMAP_PAD_X * 2;
  const plotH = HEATMAP_H - HEATMAP_PAD_TOP - HEATMAP_PAD_BOTTOM;
  const padX = HEATMAP_PAD_X;
  // X is MIRRORED (negated) - pitcher-facing chart, and pitcher's view is
  // the mirror image of catcher's view. RHH (3B side, negative plate_x)
  // renders on the pitcher's RIGHT - confirmed directly, not re-derived
  // from scratch (see MEMORY for the real data + user-confirmed geometry
  // behind this).
  const toSvgX = (x) => padX + ((-x + HEATMAP_X_RANGE_FT) / (2 * HEATMAP_X_RANGE_FT)) * plotW;
  const toSvgZ = (z) => HEATMAP_PAD_TOP + plotH - ((z - HEATMAP_Z_MIN_FT) / (HEATMAP_Z_MAX_FT - HEATMAP_Z_MIN_FT)) * plotH;
  const groundY = toSvgZ(HEATMAP_Z_MIN_FT);

  const zxA = toSvgX(-d.plate_width_ft / 2), zxB = toSvgX(d.plate_width_ft / 2);
  const zx0 = Math.min(zxA, zxB), zx1 = Math.max(zxA, zxB);
  const zz0 = toSvgZ(d.sz_bot), zz1 = toSvgZ(d.sz_top);

  const rhhA = toSvgX(-HEATMAP_BOX_OUTER_FT), rhhB = toSvgX(-HEATMAP_BOX_INNER_FT);
  const rhhX0 = Math.min(rhhA, rhhB), rhhX1 = Math.max(rhhA, rhhB);
  const lhhA = toSvgX(HEATMAP_BOX_INNER_FT), lhhB = toSvgX(HEATMAP_BOX_OUTER_FT);
  const lhhX0 = Math.min(lhhA, lhhB), lhhX1 = Math.max(lhhA, lhhB);
  const BATTER_IMG_ASPECT = 501 / 1262;
  // Fit to the dashed batter box (not blown up past it to force an exact
  // knee/zone-bottom pixel match - tried that, it made the silhouette
  // dramatically bigger than the box, which looked wrong) - sized up only
  // slightly from the original (1.0x the box height, was 0.92x) and
  // anchored with its FEET on the real ground (groundY, z=0), not
  // repositioned based on the zone. Because the box itself already runs
  // from just above the real zone top down to the real ground, a figure
  // that fills it feet-to-head naturally has its knee fall in roughly the
  // right neighborhood of the zone's real bottom edge without needing to
  // override the box's own size to force an exact match.
  const boxTop = zz1 - 6;
  const boxH = groundY - boxTop;
  const imgH = boxH * 1.0;
  const imgW = imgH * BATTER_IMG_ASPECT;

  // Zone (and every pitch dot, which is positioned relative to it) shifted
  // down as a whole so the zone's real bottom edge (zz0/sz_bot) lands
  // exactly on THIS silhouette's own knee - the silhouette itself is
  // untouched (still fit-to-box, feet on the ground, per the fix above).
  // A pure vertical SHIFT (not a rescale) so the zone's real height and
  // every dot's real position relative to IT are preserved exactly -
  // this only changes where on the card that real data is drawn, not the
  // data itself (the "in zone"/chase-etc. classifications are already
  // computed server-side from the raw numbers, unaffected by this). Knee
  // height reuses the exact same KNEE_FRAC_FROM_FEET/imgH the silhouette
  // above is built from, so the two stay consistent if imgH ever changes.
  const KNEE_FRAC_FROM_FEET = 0.26;
  const kneeY = groundY - KNEE_FRAC_FROM_FEET * imgH;
  const zoneShift = kneeY - zz0;
  const zoneDrawZ0 = zz0 + zoneShift, zoneDrawZ1 = zz1 + zoneShift;

  const batterBox = (x0, x1, label, active) => {
    const imgX = (x0 + x1) / 2 - imgW / 2;
    const imgY = groundY - imgH;
    // Swapped, matching the same real-image-content correction already
    // applied to tunnel3d.js/pitch3d's texture maps - the file named
    // "_rhh" actually holds the pose that's visually correct for LHH
    // (and vice versa), confirmed by the user directly after two
    // earlier rounds of getting this backwards.
    const file = label === "RHH" ? "batter_lhh.png" : "batter_rhh.png";
    const inner = `<image href="img/${file}" x="${imgX.toFixed(1)}" y="${imgY.toFixed(1)}" width="${imgW.toFixed(1)}" height="${imgH.toFixed(1)}" opacity="${active ? 1 : 0}" preserveAspectRatio="xMidYMax meet" />`;
    return inner;
  };
  const showRHH = split !== "L";

  // Pitch usage vs the CURRENT split, drawn as a rounded pill-shaped
  // stacked bar centered INSIDE the opposite/empty batter box (the one
  // whose silhouette is invisible for this split) rather than in a lane
  // outside the boxes - that box is otherwise just dead space. A clipPath
  // rounds the stacked segments together (a plain per-segment rx would
  // leave hard corners at internal segment boundaries), then a separate
  // stroked rounded-rect on top draws the white border requested.
  const usageBar = (x0, x1) => {
    if (!usage || !usage.length) return "";
    const boxTop = zz1 - 6;
    const barW = 42;
    const barX = (x0 + x1) / 2 - barW / 2;
    const barTop = boxTop + 10, barBottom = groundY - 6, barH = barBottom - barTop;
    const rx = barW / 2;
    const key = showRHH ? "vs_r" : "vs_l";
    let y = barBottom;
    const segs = usage
      .map((u) => {
        const val = u[key] || 0;
        const h = Math.max(0, (val / 100) * barH);
        const segY = y - h;
        y = segY;
        // Borderless (no stroke outline) and bigger, per direction - was a
        // dark stroke outline at 9.5px.
        const label = h > 13
          ? `<text x="${(barX + barW / 2).toFixed(1)}" y="${(segY + h / 2 + 4).toFixed(1)}" font-size="12.5" font-weight="800" text-anchor="middle" fill="#fff">${val.toFixed(0)}%</text>`
          : "";
        return `<rect x="${barX.toFixed(1)}" y="${segY.toFixed(1)}" width="${barW}" height="${h.toFixed(1)}" fill="${pitchColor(u.pitch_type)}"><title>${u.pitch_type} ${val.toFixed(0)}%</title></rect>${label}`;
      })
      .join("");
    const clipId = `usage-clip-${showRHH ? "r" : "l"}`;
    // "USAGE" label above the bar/box - same side as the bar itself, so it
    // moves with it when the active batter hand toggles.
    const usageLabel = `<text x="${(barX + barW / 2).toFixed(1)}" y="${(boxTop - 8).toFixed(1)}" font-size="10" font-weight="800" letter-spacing="0.6" text-anchor="middle" fill="var(--text-muted)">USAGE</text>`;
    return `
      ${usageLabel}
      <clipPath id="${clipId}"><rect x="${barX.toFixed(1)}" y="${barTop.toFixed(1)}" width="${barW}" height="${barH.toFixed(1)}" rx="${rx.toFixed(1)}"/></clipPath>
      <g clip-path="url(#${clipId})">
        <rect x="${barX.toFixed(1)}" y="${barTop.toFixed(1)}" width="${barW}" height="${barH.toFixed(1)}" fill="var(--surface)"/>
        ${segs}
      </g>
      <rect x="${barX.toFixed(1)}" y="${barTop.toFixed(1)}" width="${barW}" height="${barH.toFixed(1)}" rx="${rx.toFixed(1)}" fill="none" stroke="#fff" stroke-width="2.5"/>
    `;
  };
  const usageBarsHtml = showRHH ? usageBar(lhhX0, lhhX1) : usageBar(rhhX0, rhhX1);

  // Don't draw an average-location dot for a pitch type thrown less than 5%
  // of the time to the currently-active batter hand - a location built from
  // a handful of pitches isn't meaningfully "his location for this pitch
  // against this hand." Only excludes when usage data positively says
  // <5%; a pitch type with no usage entry at all still renders (better to
  // show it than silently drop real location data over a data gap).
  const usageKey = showRHH ? "vs_r" : "vs_l";
  const usageByType = Object.fromEntries((usage || []).map((u) => [u.pitch_type, u]));
  const qualifyingPitches = d.pitches.filter((p) => {
    const u = usageByType[p.pitch_type];
    return !u || (u[usageKey] ?? 0) >= 5;
  });

  const dots = [...qualifyingPitches].sort((a, b) => b.spread - a.spread)
    .map((p) => {
      const cx = toSvgX(p.mean_x), cz = toSvgZ(p.mean_z) + zoneShift;
      // Sized by league-wide command percentile for THIS pitch type when
      // there's enough of a league sample to rank it (command_pct is null
      // below MIN_PITCHES_FOR_COMMAND_PCT server-side) - smallest dot =
      // best-commanded (100th percentile) version of that pitch in the
      // league, biggest = worst (0th). Falls back to the pitcher's own raw
      // spread, scaled against a fixed absolute range, only when a real
      // league percentile isn't available for that type.
      let r;
      if (p.command_pct != null) {
        const frac = 1 - p.command_pct / 100;
        r = HEATMAP_DOT_R_MIN + (HEATMAP_DOT_R_MAX - HEATMAP_DOT_R_MIN) * frac;
      } else {
        const frac = (p.spread - HEATMAP_ABS_SPREAD_MIN_FT) / (HEATMAP_ABS_SPREAD_MAX_FT - HEATMAP_ABS_SPREAD_MIN_FT);
        r = HEATMAP_DOT_R_MIN + (HEATMAP_DOT_R_MAX - HEATMAP_DOT_R_MIN) * Math.max(0, Math.min(1, frac));
      }
      const color = pitchColor(p.pitch_type);
      const title = p.command_pct != null
        ? `<title>${p.pitch_type} command: ${ordinal(Math.round(p.command_pct))} percentile league-wide (this pitch type)</title>`
        : "";
      return `
        <circle cx="${cx.toFixed(1)}" cy="${cz.toFixed(1)}" r="${r.toFixed(1)}" fill="${color}" stroke="#fff" stroke-width="1">${title}</circle>
        <text x="${cx.toFixed(1)}" y="${(cz - r - 5).toFixed(1)}" font-size="11" font-weight="700" text-anchor="middle" fill="${color}" style="paint-order:stroke; stroke:var(--surface-alt); stroke-width:3px;">${p.pitch_type}</text>
      `;
    })
    .join("");

  return `
    <svg viewBox="0 0 ${HEATMAP_TOTAL_W} ${HEATMAP_H}" width="100%" style="max-width:500px; display:block; margin:0 auto;">
      <rect x="0" y="0" width="${HEATMAP_TOTAL_W}" height="${HEATMAP_H}" fill="var(--surface-alt)" />
      ${usageBarsHtml}
      ${batterBox(lhhX0, lhhX1, "LHH", !showRHH)}
      ${batterBox(rhhX0, rhhX1, "RHH", showRHH)}
      <rect x="${zx0.toFixed(1)}" y="${zoneDrawZ1.toFixed(1)}" width="${(zx1 - zx0).toFixed(1)}" height="${(zoneDrawZ0 - zoneDrawZ1).toFixed(1)}"
            fill="none" stroke="var(--forest-dark)" stroke-width="2" />
      <line x1="${padX}" y1="${groundY}" x2="${HEATMAP_TOTAL_W - padX}" y2="${groundY}" stroke="var(--border)" stroke-width="1.5" />
      ${dots}
    </svg>
  `;
}

let currentPitchUsage = null; // vs_l/vs_r per pitch type, set in loadPlayer() - read here since this
                               // function re-renders on its own local toggles without a fresh /api/player fetch
let currentPlayerBats = null; // set in loadPlayer(), read by renderHitZones() on its own pitch-type toggle re-renders

async function renderPitchLocations() {
  const mount = document.getElementById("left-panel-mount");
  const cacheKey = `${state.playerId}|${state.season}|${state.heatmapSplit}`;
  let d = heatmapCache.get(cacheKey);
  if (!d) {
    d = await getJSON(`/api/heatmaps/${state.playerId}?season=${state.season}&split=${state.heatmapSplit}`);
    heatmapCache.set(cacheKey, d);
  }
  // Card-local toggle - independent of the page's main split toggle, but
  // the main toggle overrides it when set to a real L/R value (see
  // splitToggle's click handler). Re-bound after every render since the
  // buttons themselves get replaced.
  const toggleHtml = `
    <div class="toggle-group toggle-group-sm" id="heatmap-split-toggle">
      <button data-value="R" class="${state.heatmapSplit === "R" ? "active" : ""}">vs RHH</button>
      <button data-value="L" class="${state.heatmapSplit === "L" ? "active" : ""}">vs LHH</button>
    </div>
  `;
  mount.innerHTML = `
    <div class="card pitch-locations-card">
      <div class="pitch-locations-header">
        ${toggleHtml}
      </div>
      ${d.has_data ? `
        <div class="heatmap-subtitle">Pitcher View</div>
        <div class="heatmap-wrap">${buildPitchLocationsSVG(d, state.heatmapSplit, currentPitchUsage)}</div>
        <div class="heatmap-caption">${d.n_pitches} pitches - one dot per pitch type at its season-average location. Dot size = league-wide command percentile for that pitch type (smallest = best-commanded); falls back to this pitcher's own spread when the league sample's too thin to rank. Pitch types thrown under 5% of the time to the active batter hand aren't shown.</div>
      ` : `<div class="empty-state">No pitch-location data for this selection.</div>`}
    </div>
  `;
  document.getElementById("heatmap-split-toggle").addEventListener("click", async (e) => {
    const btn = e.target.closest("button");
    if (!btn || btn.dataset.value === state.heatmapSplit) return;
    state.heatmapSplit = btn.dataset.value;
    await renderPitchLocations();
  });
}

// Loading bar tied to GET /api/player/{id}/progress's real checkpoints
// (see api/routers/players.py's _build_player_steps) - not a fake timer.
function setLoadProgress(active, pct, label) {
  const el = document.getElementById("load-progress");
  if (!el) return;
  el.classList.toggle("active", active);
  document.getElementById("load-progress-fill").style.width = `${pct}%`;
  document.getElementById("load-progress-label").textContent = label || "";
}

async function loadPlayer() {
  if (state.playerId === null || state.playerId === "") return;
  const requestId = ++loadPlayerRequestId;
  const requestMode = state.mode;
  const cacheKey = playerCacheKey(state.playerId, state.mode, state.split, state.season);

  let data = playerDataCache.get(cacheKey);
  if (!data) {
    // Not cached (and not already being silently prefetched - if it is,
    // just fall through to a normal live fetch rather than waiting on a
    // background request with no progress reporting of its own) - show the
    // real progress bar and fetch live, same as before this cache existed.
    setLoadProgress(true, 0, "Loading player info...");
    try {
      data = await getJSONStreamed(
        `/api/player/${state.playerId}/progress?role=${state.mode}&split=${state.split}&season=${state.season}`,
        (evt) => {
          if (requestId !== loadPlayerRequestId) return;
          setLoadProgress(true, (evt.index / evt.total) * 100, `${evt.step}... (${evt.index}/${evt.total})`);
        }
      );
    } finally {
      if (requestId === loadPlayerRequestId) setLoadProgress(false, 100, "");
    }
    playerDataCache.set(cacheKey, data);
  }
  if (requestId !== loadPlayerRequestId || requestMode !== state.mode) return;
  // Silently warm the OTHER splits for this same player/season in the
  // background now that we know this one's real - by the time someone
  // actually clicks ALL/L/R, it's usually already sitting in memory.
  prefetchOtherSplits(state.playerId, state.mode, state.season);
  renderPlayerCard(data);
  renderPercentiles(data.percentiles);
  renderRightPanel(data);
  document.getElementById("last-5-title").textContent = state.mode === "pitcher" ? "Last 5 Outings" : "Last 5 Games";
  document.getElementById("last-5-mount").innerHTML = buildLast5Card(data.last_5_games, state.mode, data.il_status);
  if (state.mode === "pitcher") {
    currentPitchUsage = data.pitch_usage_by_stand;
    renderTable(data.pitch_table, "");
    await renderPitchLocations();
    await renderSimilarPlayers();
    // Part D: show outing breakdown for pitchers
    const outingSection = document.getElementById("outing-section");
    if (outingSection) {
      outingSection.style.display = "";
      if (typeof window.loadOutings === "function") {
        window.loadOutings(state.playerId, state.season);
      }
    }
  } else {
    currentPlayerBats = data.swing_3d ? data.swing_3d.bats : null;
    renderTable(data.pitch_type_table, "Pitch types with 10 or fewer pitches seen are excluded.");
    await renderHitZones();
    await renderSimilarPlayers();
    // Hide outing section for batters
    const outingSection = document.getElementById("outing-section");
    if (outingSection) outingSection.style.display = "none";
    await renderRollingXwoba(state.playerId, state.season);
  }
}

function _monotoneCubicPath(xs, ys) {
  // Fritsch-Carlson monotone cubic spline → SVG path string
  const n = xs.length;
  if (n < 2) return "";
  const d = [], m = new Array(n);
  for (let i = 0; i < n - 1; i++) d[i] = (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i]);
  m[0] = d[0];
  m[n - 1] = d[n - 2];
  for (let i = 1; i < n - 1; i++) {
    if (d[i - 1] * d[i] <= 0) { m[i] = 0; continue; }
    m[i] = (d[i - 1] + d[i]) / 2;
  }
  // Limit slopes for monotonicity
  for (let i = 0; i < n - 1; i++) {
    if (Math.abs(d[i]) < 1e-12) { m[i] = m[i + 1] = 0; continue; }
    const a = m[i] / d[i], b = m[i + 1] / d[i];
    const t = a * a + b * b;
    if (t > 9) { const s = 3 / Math.sqrt(t); m[i] = s * a * d[i]; m[i + 1] = s * b * d[i]; }
  }
  let path = `M${xs[0].toFixed(1)},${ys[0].toFixed(1)}`;
  for (let i = 0; i < n - 1; i++) {
    const h = xs[i + 1] - xs[i];
    const cp1x = xs[i] + h / 3, cp1y = ys[i] + m[i] * h / 3;
    const cp2x = xs[i + 1] - h / 3, cp2y = ys[i + 1] - m[i + 1] * h / 3;
    path += `C${cp1x.toFixed(1)},${cp1y.toFixed(1)} ${cp2x.toFixed(1)},${cp2y.toFixed(1)} ${xs[i + 1].toFixed(1)},${ys[i + 1].toFixed(1)}`;
  }
  return path;
}

// ── xwOBA chart constants ─────────────────────────────────────────────────
const XWOBA_FULL_INTENSITY = 0.080; // ±0.080 from league avg = full color
// Color encoding: above avg → dark red; below avg → saturated blue; avg → gray
// All three are WCAG AA on white/light card backgrounds.
function _xwobaStrokeColor(val, lgAvg) {
  const diff = val - lgAvg;
  const t = Math.max(-1, Math.min(1, diff / XWOBA_FULL_INTENSITY));
  if (t >= 0) {
    // gray(0) → dark red(1)
    const r = Math.round(130 + 75 * t);       // 130 → 205
    const g = Math.round(130 - 110 * t);      // 130 → 20
    const b = Math.round(130 - 110 * t);      // 130 → 20
    return `rgb(${r},${g},${b})`;
  } else {
    // gray(0) → deep blue(-1)
    const r = Math.round(130 + 93 * t);       // 130 → 37
    const g = Math.round(130 + 55 * t);       // 130 → 75
    const b = Math.round(130 - 115 * t);      // 130 → 245
    return `rgb(${r},${g},${b})`;
  }
}

// Centered rolling mean: window of `k` centered on point i.
// Edge points use whatever values are available.
function _centeredRolling(vals, k) {
  const n = vals.length, half = Math.floor(k / 2);
  return vals.map((_, i) => {
    const lo = Math.max(0, i - half), hi = Math.min(n, i + half + 1);
    const slice = vals.slice(lo, hi).filter(v => v != null && !isNaN(v));
    return slice.length ? slice.reduce((a, b) => a + b, 0) / slice.length : null;
  });
}

function _clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

// Format a date string "YYYY-MM-DD" → "Mon D" e.g. "May 28"
function _fmtChartDate(dateStr) {
  if (!dateStr) return "";
  const d = new Date(dateStr + "T00:00:00");
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}
function _fmtChartDateFull(dateStr) {
  if (!dateStr) return "";
  const d = new Date(dateStr + "T00:00:00");
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
}

function _buildXwobaChart(allPts, windowKey, lgAvg) {
  // Determine horizon slice and smooth window
  let pts;
  let smoothK;
  if (windowKey === "YR") {
    pts = allPts.slice();
    smoothK = 25;
  } else {
    const N = parseInt(windowKey, 10);
    pts = allPts.slice(-N);
    smoothK = _clamp(Math.round(N / 10), 5, 25);
  }

  if (!pts.length) return "";

  const rawVals = pts.map(p => p.xwoba);
  const smoothed = _centeredRolling(rawVals, smoothK);
  const validSmoothed = smoothed.filter(v => v != null);
  if (!validSmoothed.length) return "";

  const W = 540, H = 150, PAD = { top: 14, right: 20, bottom: 22, left: 44 };
  const iW = W - PAD.left - PAD.right;
  const iH = H - PAD.top - PAD.bottom;

  const allY = [...validSmoothed, lgAvg];
  const yMin = Math.min(0.150, Math.min(...allY) - 0.015);
  const yMax = Math.max(0.500, Math.max(...allY) + 0.015);

  const xScale = i => PAD.left + (i / (pts.length - 1 || 1)) * iW;
  const yScale = v => PAD.top + (1 - (v - yMin) / (yMax - yMin)) * iH;

  // Y grid and ticks
  let grid = "";
  for (let v = Math.ceil(yMin / 0.05) * 0.05; v <= yMax + 0.001; v += 0.05) {
    const vr = parseFloat(v.toFixed(3));
    const y = yScale(vr).toFixed(1);
    grid += `<line x1="${PAD.left}" y1="${y}" x2="${PAD.left + iW}" y2="${y}" stroke="var(--border)" stroke-width="0.7" stroke-dasharray="2,3"/>`;
    grid += `<text x="${PAD.left - 4}" y="${y}" text-anchor="end" dominant-baseline="middle" font-size="9.5" font-family="var(--font-data,monospace)" fill="var(--text-muted)">.${String(Math.round(vr * 1000)).padStart(3, "0")}</text>`;
  }

  // League-average dashed reference line
  const lgY = yScale(lgAvg).toFixed(1);
  const lgLine = `<line x1="${PAD.left}" y1="${lgY}" x2="${PAD.left + iW}" y2="${lgY}" stroke="#7c5c2e" stroke-width="1.5" stroke-dasharray="5,3" opacity="0.75"/>`;

  // Gradient stroke: draw colored segments between consecutive smoothed points
  let lineSegs = "";
  const validIdx = smoothed.map((v, i) => v != null ? i : -1).filter(i => i >= 0);
  for (let j = 0; j < validIdx.length - 1; j++) {
    const i0 = validIdx[j], i1 = validIdx[j + 1];
    if (i1 - i0 > 1) continue; // gap in data — skip
    const x0 = xScale(i0).toFixed(1), y0 = yScale(smoothed[i0]).toFixed(1);
    const x1 = xScale(i1).toFixed(1), y1 = yScale(smoothed[i1]).toFixed(1);
    const midVal = (smoothed[i0] + smoothed[i1]) / 2;
    const color = _xwobaStrokeColor(midVal, lgAvg);
    lineSegs += `<line x1="${x0}" y1="${y0}" x2="${x1}" y2="${y1}" stroke="${color}" stroke-width="3" stroke-linecap="round"/>`;
  }

  // Also draw a smoother path on top (transparent — for tooltip region); keep colored segments for look
  // X-axis date labels at first and last point
  const firstDate = pts[0]?.date || "";
  const lastDate = pts[pts.length - 1]?.date || "";
  const x0Label = PAD.left.toFixed(1);
  const x1Label = (PAD.left + iW).toFixed(1);
  const yLbl = (H - 4).toFixed(1);
  const dateLabels = `
    <text x="${x0Label}" y="${yLbl}" text-anchor="start" font-size="9.5" font-family="var(--font-data,monospace)" fill="var(--text-muted)">${_fmtChartDate(firstDate)}</text>
    <text x="${x1Label}" y="${yLbl}" text-anchor="end" font-size="9.5" font-family="var(--font-data,monospace)" fill="var(--text-muted)">${_fmtChartDate(lastDate)}</text>`;

  return `<svg viewBox="0 0 ${W} ${H}" style="width:100%;display:block;">${grid}${lgLine}${lineSegs}${dateLabels}</svg>`;
}

async function renderRollingXwoba(playerId, season) {
  const section = document.getElementById("rolling-xwoba-section");
  const mount = document.getElementById("rolling-xwoba-mount");
  const captionEl = document.getElementById("rolling-xwoba-caption");
  const toggleContainer = document.getElementById("rolling-xwoba-toggle");
  if (!section || !mount) return;

  let result;
  try {
    result = await getJSON(`/api/players/${playerId}/rolling-xwoba?season=${season}`);
  } catch (e) {
    section.style.display = "none";
    return;
  }

  const allPts = result.points || [];
  const lgAvg = result.league_avg ?? 0.320;

  if (allPts.length < 10) {
    section.style.display = "none";
    return;
  }

  section.style.display = "";

  const WINDOWS = [
    { key: "50",  label: "50 PA"  },
    { key: "100", label: "100 PA" },
    { key: "250", label: "250 PA" },
    { key: "YR",  label: "YR"     },
  ];

  let activeWindow = "100";

  function _windowPts(key) {
    if (key === "YR") return allPts;
    const N = parseInt(key, 10);
    return allPts.slice(-N);
  }

  function _horizonMean(key) {
    const pts = _windowPts(key);
    const vals = pts.map(p => p.xwoba).filter(v => v != null && !isNaN(v));
    if (!vals.length) return null;
    return vals.reduce((a, b) => a + b, 0) / vals.length;
  }

  function renderToggle() {
    if (!toggleContainer) return;
    toggleContainer.innerHTML = WINDOWS.map(w => {
      const active = w.key === activeWindow;
      return `<button class="xwoba-win-btn" data-win="${w.key}" aria-pressed="${active}">${w.label}</button>`;
    }).join("");
    toggleContainer.querySelectorAll("button").forEach(btn => {
      btn.addEventListener("click", () => {
        activeWindow = btn.dataset.win;
        renderToggle();
        renderChart();
      });
      btn.addEventListener("keydown", e => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          btn.click();
        }
      });
    });
  }

  function renderChart() {
    const pts = _windowPts(activeWindow);
    const actualN = pts.length;
    const requestedN = activeWindow === "YR" ? null : parseInt(activeWindow, 10);
    mount.innerHTML = _buildXwobaChart(allPts, activeWindow, lgAvg);

    // Caption: horizon mean xwOBA and league average
    const mean = _horizonMean(activeWindow);
    let note = "";
    if (requestedN !== null && actualN < requestedN) {
      note = ` &nbsp;·&nbsp; <em>only ${actualN} qualifying PA this season</em>`;
    }
    const firstDate = pts[0]?.date, lastDate = pts[pts.length - 1]?.date;
    const dateRange = firstDate ? `${_fmtChartDateFull(firstDate)} – ${_fmtChartDateFull(lastDate)}` : "";
    if (captionEl) {
      captionEl.innerHTML = mean != null
        ? `Horizon xwOBA: <strong>${mean.toFixed(3)}</strong> &nbsp;·&nbsp; Lg avg: <strong>${lgAvg.toFixed(3)}</strong> &nbsp;·&nbsp; ${dateRange}${note}`
        : dateRange + note;
    }
  }

  renderToggle();
  renderChart();
}

// Fire-and-forget prefetches for the two most-visited adjacent pages so
// they render instantly when the user navigates there.
function prefetchAdjacentPages(meta) {
  if (typeof prefetchInBackground !== "function") return;
  const date = meta && meta.latest_game_date;
  if (date) {
    prefetchInBackground(`/api/top-performers?date=${date}`);
    prefetchInBackground(`/api/probable-pitchers?date=${date}`);
  }
}

window.onDataRefresh = async () => {
  playerDataCache.clear();
  similarPlayersCache.clear();
  heatmapCache.clear();
  hitzoneCache.clear();
  await playerSearch.load(state.season);
  await loadPlayer();
};

// No remembered player (a real page reload) - default to today's actual
// #1 top performer instead of whoever happens to sort first alphabetically,
// so the page opens on someone real and relevant rather than an arbitrary
// name. Falls back to the old alphabetical-first-pitcher default if the
// top-performers fetch fails or the day has no scored entries yet.
async function defaultToTopPerformer() {
  try {
    const meta = await getJSON("/api/meta");
    const perf = await getJSON(`/api/top-performers?date=${meta.latest_game_date}`);
    const top = perf.entries && perf.entries[0];
    if (top) return playerSearch.selectDefault((p) => p.player_id === top.player_id);
  } catch (err) {
    // fall through to the alphabetical fallback below
  }
  return playerSearch.selectDefault((p) => p.role === "pitcher");
}

(async function init() {
  state.season = await seasonSelect.load();
  await playerSearch.load(state.season);

  // URL params win over everything — lets external links (probable-pitchers,
  // top performers, etc.) navigate directly to a specific player by ID.
  const urlParams = new URLSearchParams(window.location.search);
  const urlPlayerId = urlParams.get("player_id");
  const urlRole = urlParams.get("role");
  if (urlPlayerId) {
    const pid = Number(urlPlayerId);
    const fromSearch = playerSearch.selectDefault((p) => p.player_id === pid);
    const chosen = fromSearch || { player_id: pid, role: urlRole || "pitcher", name: "" };
    if (fromSearch) playerSearch.setValue(fromSearch);
    state.playerId = chosen.player_id;
    state.mode = chosen.role || urlRole || "pitcher";
    applyRoleChrome();
    await loadPlayer();
    return;
  }

  // A player selected on another tab (player/splits/stuff-model/
  // tunneling-model/deception-model all share this) wins over the default -
  // resets to empty on an actual page reload, see player-context.js.
  const remembered = PlayerContext.getCurrent();
  const chosen = remembered
    ? playerSearch.selectDefault((p) => p.player_id === remembered.player_id) || remembered
    : await defaultToTopPerformer();
  if (chosen) {
    if (remembered) playerSearch.setValue(chosen);
    state.playerId = chosen.player_id;
    state.mode = chosen.role;
    applyRoleChrome();
    await loadPlayer();
  }
  // After the main player renders, prefetch probable-pitchers and top-performers
  // in the background so those pages load instantly when navigated to.
  getJSON("/api/meta").then(prefetchAdjacentPages).catch(() => {});
})();
