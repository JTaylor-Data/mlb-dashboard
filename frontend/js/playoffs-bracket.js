// Full visual postseason bracket - both leagues at once, NL flowing left-
// to-right, AL mirrored flowing right-to-left. The two sides stay
// genuinely SEPARATE all the way to the World Series - each league's own
// champion slot sits on its own side of a center trophy, meeting only
// there; the real WS winner (once known) gets called out in its own box
// ABOVE the trophy rather than replacing either side's slot. Seeds 1 AND 2
// are both byes into the Division Series under the real 2022+ format (only
// seeds 3-6 play in the Wild Card round). Drawn directly on the page
// background (no card), with thin brown connector lines joining each pair
// of boxes to the box they feed into - drawn as an SVG overlay, measured
// against the ACTUAL rendered box positions (getBoundingClientRect) after
// layout, not guessed coordinates, so it stays correct at any screen width.
//
// Real results (see /api/playoffs/bracket - analytics/playoffs.py's
// unified_bracket) fill in whichever rounds have actually been played (real
// series record, real winner highlighted); everything past that stays an
// open TBD box - never a guessed winner.
const state = { season: 2026 };
const CONNECTOR_COLOR = "#6b4423"; // --bark

const seasonSelect = createSeasonSelect(document.getElementById("season-select"), async (season) => {
  state.season = season;
  await loadBracket();
});

function logoImg(teamId, size) {
  return `<img class="team-logo" src="https://www.mlbstatic.com/team-logos/${teamId}.svg" onerror="this.style.display='none'" style="width:${size}px; height:${size}px;" />`;
}

// Logo only, no team name text underneath - per direction, so the box can
// shrink to just what the logo needs and the logo itself can grow. The
// real team name is still there as a title attribute (hover tooltip).
function slotHtml(id, team, isWinner, size) {
  const cls = `bracket-slot${team ? "" : " empty"}${isWinner ? " winner" : ""}`;
  const body = team ? logoImg(team.team_id, size) : `<div class="bracket-slot-tbd">TBD</div>`;
  return `<div class="${cls}" id="${id}" title="${team ? team.name : "TBD"}">${body}</div>`;
}

function recordHtml(record) {
  return record
    ? `<div class="bracket-record">${record}</div>`
    : `<div class="bracket-record" style="visibility:hidden">0-0</div>`;
}

// One matchup: a stacked pair of slots (team_a on top, team_b below), plus
// the series record between them once real games exist.
function pairHtml(idPrefix, matchup) {
  const a = matchup && matchup.team_a;
  const b = matchup && matchup.team_b;
  const winner = matchup && matchup.winner_team_id;
  return `
    <div class="bracket-pair">
      ${slotHtml(`${idPrefix}-a`, a, winner && a && winner === a.team_id, 64)}
      ${recordHtml(matchup && matchup.record)}
      ${slotHtml(`${idPrefix}-b`, b, winner && b && winner === b.team_id, 64)}
    </div>
  `;
}

// Bold, simple gold cup silhouette - the earlier "real Commissioner's
// Trophy" attempt (globe + 6 flag pennants) read as a muddy blob at this
// size; a clean cup shape with a star accent is unmistakably "trophy" at
// a glance, which matters more here than literal accuracy to the real
// trophy's unusual flags-on-a-globe design.
function trophySvg() {
  return `
    <svg class="bracket-trophy" viewBox="0 0 64 64" width="70" height="70">
      <defs>
        <linearGradient id="trophyGold" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="#ffe9a8"/>
          <stop offset="50%" stop-color="#d4af37"/>
          <stop offset="100%" stop-color="#a8790a"/>
        </linearGradient>
      </defs>
      <path d="M18 6 H46 V19 C46 32 39 41 32 41 C25 41 18 32 18 19 Z" fill="url(#trophyGold)" stroke="#6b4a03" stroke-width="2.2" stroke-linejoin="round"/>
      <path d="M18 10 C7 10 5 23 14 29 C16.5 30.7 18 30.5 18 30.5" fill="none" stroke="#6b4a03" stroke-width="3.2" stroke-linecap="round"/>
      <path d="M46 10 C57 10 59 23 50 29 C47.5 30.7 46 30.5 46 30.5" fill="none" stroke="#6b4a03" stroke-width="3.2" stroke-linecap="round"/>
      <path d="M32 13.5 L34.2 18.6 L39.8 19 L35.5 22.6 L36.9 28 L32 25 L27.1 28 L28.5 22.6 L24.2 19 L29.8 18.6 Z" fill="#fff6dc"/>
      <rect x="28" y="41" width="8" height="10" fill="url(#trophyGold)" stroke="#6b4a03" stroke-width="2.2"/>
      <path d="M19 51 H45 L41 59 H23 Z" fill="url(#trophyGold)" stroke="#6b4a03" stroke-width="2.2" stroke-linejoin="round"/>
      <rect x="14" y="59" width="36" height="5.5" rx="2" fill="#6b4a03"/>
    </svg>
  `;
}

function leagueColumnsHtml(bracket, leaguePrefix, leagueLabel) {
  const wc = bracket.wildcard || [];
  const ds = bracket.division || [];
  const cs = (bracket.championship && bracket.championship[0]) || null;

  const wcCol = `
    <div class="bracket-col">
      <div class="bracket-col-label">Wild Card</div>
      <div class="bracket-col-body">
        ${pairHtml(`${leaguePrefix}-wc-0`, wc[0])}
        ${wc.length > 1 ? pairHtml(`${leaguePrefix}-wc-1`, wc[1]) : ""}
      </div>
    </div>
  `;
  const dsCol = `
    <div class="bracket-col">
      <div class="bracket-col-label">Division Series</div>
      <div class="bracket-col-body">
        ${pairHtml(`${leaguePrefix}-ds-0`, ds[0])}
        ${ds.length > 1 ? pairHtml(`${leaguePrefix}-ds-1`, ds[1]) : ""}
      </div>
    </div>
  `;
  const csCol = `
    <div class="bracket-col-standalone">
      <div class="bracket-col-label">${leagueLabel}CS</div>
      <div class="bracket-col-standalone-body">
        ${pairHtml(`${leaguePrefix}-cs-0`, cs)}
      </div>
    </div>
  `;
  return { wcCol, dsCol, csCol };
}

// The two league champion slots stay on their OWN side (NL left of center,
// AL right) all the way to the World Series - they only "meet" via the
// trophy sitting between them, with the real winner (once known) called
// out in its own box ABOVE the trophy rather than replacing either side -
// per direction, a deliberate redesign of the earlier stacked-pair version.
function worldSeriesHtml(ws) {
  const nlTeam = ws && ws.team_a;
  const alTeam = ws && ws.team_b;
  const winnerId = ws && ws.winner_team_id;
  const winnerTeam = winnerId ? (nlTeam && nlTeam.team_id === winnerId ? nlTeam : alTeam) : null;

  const nlSlotCol = `
    <div class="bracket-col-standalone bracket-ws-col">
      <div class="bracket-col-label">NL Champion</div>
      <div class="bracket-col-standalone-body">
        ${slotHtml("nl-ws-slot", nlTeam, !!(winnerId && nlTeam && nlTeam.team_id === winnerId), 60)}
      </div>
    </div>
  `;
  const alSlotCol = `
    <div class="bracket-col-standalone bracket-ws-col">
      <div class="bracket-col-label">AL Champion</div>
      <div class="bracket-col-standalone-body">
        ${slotHtml("al-ws-slot", alTeam, !!(winnerId && alTeam && alTeam.team_id === winnerId), 60)}
      </div>
    </div>
  `;
  // No text header (nothing to align to another column's label), but the
  // SAME invisible top spacer as the labeled columns, so the trophy still
  // centers against the exact same remaining height they do - otherwise it
  // would sit slightly higher than a labeled column's own centered content.
  const centerCol = `
    <div class="bracket-col-standalone bracket-ws-center-outer">
      <div class="bracket-col-label-spacer"></div>
      <div class="bracket-col-standalone-body bracket-ws-center-col">
        <div class="bracket-ws-trophy-block">
          <div class="bracket-ws-winner-slot" id="ws-winner-slot">
            ${winnerTeam ? slotHtml("ws-winner", winnerTeam, true, 68) : '<div class="bracket-slot empty bracket-ws-winner-placeholder">Champion</div>'}
          </div>
          ${trophySvg()}
          <div class="bracket-ws-label">World Series</div>
          ${recordHtml(ws && ws.record)}
        </div>
      </div>
    </div>
  `;
  return { nlSlotCol, centerCol, alSlotCol };
}

// Rendered into its own mount, alongside (not above) the season selector -
// per direction, sharing that row instead of costing a separate one, so
// the freed vertical space goes into bigger boxes below.
function titleHtml(data) {
  const title = data.postseason_started ? "Playoff Alignment" : "Current Playoff Alignment";
  return `
    <div class="bracket-title-row">
      <div class="bracket-title">${title}</div>
      ${data.is_real ? '<div class="bracket-real-badge">Real results</div>' : ""}
    </div>
  `;
}

function bracketHtml(data) {
  const al = leagueColumnsHtml(data.al, "al", "AL");
  const nl = leagueColumnsHtml(data.nl, "nl", "NL");
  const ws = worldSeriesHtml(data.world_series);
  return `
    <div class="bracket-wrap" id="bracket-wrap">
      <svg id="bracket-svg" class="bracket-svg"></svg>
      <div class="bracket-half">
        <div class="bracket-league-title">National League</div>
        <div class="bracket-half-cols">
          ${nl.wcCol}
          ${nl.dsCol}
        </div>
      </div>
      ${nl.csCol}
      ${ws.nlSlotCol}
      ${ws.centerCol}
      ${ws.alSlotCol}
      ${al.csCol}
      <div class="bracket-half">
        <div class="bracket-league-title">American League</div>
        <div class="bracket-half-cols">
          ${al.dsCol}
          ${al.wcCol}
        </div>
      </div>
    </div>
    <div class="muted" style="font-size:11px; margin-top:10px;">
      Seeds 1 and 2 (division winners) are both byes straight into the Division Series. Seeds 3-6 play the Wild
      Card round first.${data.is_real ? " Highlighted logos are real series winners; open boxes are rounds that haven't been played yet." : " Every round past Wild Card is shown open (TBD) - who advances isn't projected, only who's currently positioned to play."}
    </div>
  `;
}

// Draws a brown elbow connector from two source boxes (a pair) into one
// destination box - measured against real rendered positions (not guessed),
// so it holds up regardless of box size/wrapping. `wrapRect` is the
// wrap container's own rect (coordinates below are relative to it).
function connectPair(svg, wrapRect, idA, idB, destId) {
  const a = document.getElementById(idA);
  const b = document.getElementById(idB);
  const dest = document.getElementById(destId);
  if (!a || !b || !dest) return;
  const ra = a.getBoundingClientRect(), rb = b.getBoundingClientRect(), rd = dest.getBoundingClientRect();
  const ax = ra.right - wrapRect.left, ay = ra.top + ra.height / 2 - wrapRect.top;
  const bx = rb.right - wrapRect.left, by = rb.top + rb.height / 2 - wrapRect.top;
  const dxLeft = rd.left - wrapRect.left, dy = rd.top + rd.height / 2 - wrapRect.top;
  const elbowX = ax + (dxLeft - ax) / 2;
  const path = `M ${ax} ${ay} H ${elbowX} M ${bx} ${by} H ${elbowX} M ${elbowX} ${ay} V ${by} M ${elbowX} ${dy} H ${dxLeft}`;
  const el = document.createElementNS("http://www.w3.org/2000/svg", "path");
  el.setAttribute("d", path);
  el.setAttribute("stroke", CONNECTOR_COLOR);
  el.setAttribute("stroke-width", "2");
  el.setAttribute("fill", "none");
  svg.appendChild(el);
}

// Same shape, but the AL side flows right-to-left (mirrors the source/dest
// left/right edges used for the connector points).
function connectPairReversed(svg, wrapRect, idA, idB, destId) {
  const a = document.getElementById(idA);
  const b = document.getElementById(idB);
  const dest = document.getElementById(destId);
  if (!a || !b || !dest) return;
  const ra = a.getBoundingClientRect(), rb = b.getBoundingClientRect(), rd = dest.getBoundingClientRect();
  const ax = ra.left - wrapRect.left, ay = ra.top + ra.height / 2 - wrapRect.top;
  const bx = rb.left - wrapRect.left, by = rb.top + rb.height / 2 - wrapRect.top;
  const dxRight = rd.right - wrapRect.left, dy = rd.top + rd.height / 2 - wrapRect.top;
  const elbowX = ax - (ax - dxRight) / 2;
  const path = `M ${ax} ${ay} H ${elbowX} M ${bx} ${by} H ${elbowX} M ${elbowX} ${ay} V ${by} M ${elbowX} ${dy} H ${dxRight}`;
  const el = document.createElementNS("http://www.w3.org/2000/svg", "path");
  el.setAttribute("d", path);
  el.setAttribute("stroke", CONNECTOR_COLOR);
  el.setAttribute("stroke-width", "2");
  el.setAttribute("fill", "none");
  svg.appendChild(el);
}

// Two side-by-side boxes (the NL and AL champion slots) converging UP into
// one box above them (the World Series winner slot) - the two league sides
// stay visually separate, meeting only at this single point, per direction.
function connectConverge(svg, wrapRect, idA, idB, destId) {
  const a = document.getElementById(idA);
  const b = document.getElementById(idB);
  const dest = document.getElementById(destId);
  if (!a || !b || !dest) return;
  const ra = a.getBoundingClientRect(), rb = b.getBoundingClientRect(), rd = dest.getBoundingClientRect();
  const ax = ra.left + ra.width / 2 - wrapRect.left, ay = ra.top - wrapRect.top;
  const bx = rb.left + rb.width / 2 - wrapRect.left, by = rb.top - wrapRect.top;
  const dx = rd.left + rd.width / 2 - wrapRect.left, dBottom = rd.bottom - wrapRect.top;
  const midY = Math.min(ay, by) - 16;
  const path = `M ${ax} ${ay} V ${midY} M ${bx} ${by} V ${midY} M ${ax} ${midY} H ${bx} M ${dx} ${midY} V ${dBottom}`;
  const el = document.createElementNS("http://www.w3.org/2000/svg", "path");
  el.setAttribute("d", path);
  el.setAttribute("stroke", CONNECTOR_COLOR);
  el.setAttribute("stroke-width", "2");
  el.setAttribute("fill", "none");
  svg.appendChild(el);
}

function drawConnectors() {
  const wrap = document.getElementById("bracket-wrap");
  const svg = document.getElementById("bracket-svg");
  if (!wrap || !svg) return;
  const wrapRect = wrap.getBoundingClientRect();
  svg.setAttribute("width", wrapRect.width);
  svg.setAttribute("height", wrapRect.height);
  svg.innerHTML = "";

  // NL: left-to-right
  connectPair(svg, wrapRect, "nl-wc-0-a", "nl-wc-0-b", "nl-ds-0-b");
  connectPair(svg, wrapRect, "nl-wc-1-a", "nl-wc-1-b", "nl-ds-1-b");
  connectPair(svg, wrapRect, "nl-ds-0-a", "nl-ds-0-b", "nl-cs-0-a");
  connectPair(svg, wrapRect, "nl-ds-1-a", "nl-ds-1-b", "nl-cs-0-b");
  connectPair(svg, wrapRect, "nl-cs-0-a", "nl-cs-0-b", "nl-ws-slot");

  // AL: right-to-left (mirrored)
  connectPairReversed(svg, wrapRect, "al-wc-0-a", "al-wc-0-b", "al-ds-0-b");
  connectPairReversed(svg, wrapRect, "al-wc-1-a", "al-wc-1-b", "al-ds-1-b");
  connectPairReversed(svg, wrapRect, "al-ds-0-a", "al-ds-0-b", "al-cs-0-a");
  connectPairReversed(svg, wrapRect, "al-ds-1-a", "al-ds-1-b", "al-cs-0-b");
  connectPairReversed(svg, wrapRect, "al-cs-0-a", "al-cs-0-b", "al-ws-slot");

  // Both league champion slots converge up into the winner slot above the trophy
  connectConverge(svg, wrapRect, "nl-ws-slot", "al-ws-slot", "ws-winner-slot");
}

async function loadBracket() {
  setPageLoading(true);
  try {
    const data = await getJSON(`/api/playoffs/bracket?season=${state.season}`);
    document.getElementById("bracket-title-mount").innerHTML = titleHtml(data);
    document.getElementById("bracket-mount").innerHTML = bracketHtml(data);
    // setTimeout, not requestAnimationFrame - rAF callbacks are throttled
    // or fully suspended in a backgrounded/non-visible tab (confirmed live:
    // 0 connector lines drew until manually re-triggered), so a page that
    // loads while backgrounded would silently never get its lines. A short
    // timeout still fires either way, giving the browser time to commit the
    // new DOM/layout first so getBoundingClientRect reads real positions.
    // Re-drawn again once every logo image has actually loaded (their real
    // decoded size can shift box heights slightly from the pre-load guess).
    setTimeout(drawConnectors, 50);
    [...document.querySelectorAll("#bracket-wrap img")].forEach((img) => {
      if (img.complete) return;
      img.addEventListener("load", drawConnectors, { once: true });
    });
  } finally {
    setPageLoading(false);
  }
}

window.addEventListener("resize", () => {
  clearTimeout(window._bracketResizeTimer);
  window._bracketResizeTimer = setTimeout(drawConnectors, 150);
});

window.onDataRefresh = loadBracket;

(async function init() {
  state.season = await seasonSelect.load();
  await loadBracket();
})();
