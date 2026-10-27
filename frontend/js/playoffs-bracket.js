const CONNECTOR_COLOR = "#6b4423";
const state = { season: 2026 };

const seasonSelect = createSeasonSelect(document.getElementById("season-select"), async (season) => {
  state.season = season;
  await loadBracket();
});

function logoImg(teamId, size) {
  return `<img src="https://www.mlbstatic.com/team-logos/${teamId}.svg" style="width:${size}px;height:${size}px;" onerror="this.style.display='none'">`;
}

// Single team slot (WC/DS individual seeds, or champion/WS).
function teamSlot(id, team, isWinner, size) {
  const cls = `bk-slot${team ? "" : " empty"}${isWinner ? " winner" : ""}`;
  const inner = team ? logoImg(team.team_id, size) : `<span class="bk-tbd">TBD</span>`;
  return `<div class="${cls}" id="${id}" title="${team ? team.name : "TBD"}">${inner}</div>`;
}

// Single matchup-result slot (NLCS, Champion).
// Shows winner logo, a vs-chip if in progress, or TBD.
function matchupSlot(id, matchup, size) {
  const sz = size || 62;
  const winner = matchup && matchup.winner_team_id;
  const a = matchup && matchup.team_a;
  const b = matchup && matchup.team_b;

  let inner, cls;
  if (winner) {
    const winTeam = a && a.team_id === winner ? a : b;
    inner = logoImg(winTeam.team_id, sz);
    cls = "bk-slot winner";
  } else if (a && b) {
    inner = `<div class="bk-vs-chip">${logoImg(a.team_id, 34)}<span class="bk-vs">vs</span>${logoImg(b.team_id, 34)}</div>`;
    cls = "bk-slot";
  } else {
    inner = `<span class="bk-tbd">TBD</span>`;
    cls = "bk-slot empty";
  }
  const title = winner
    ? (a && a.team_id === winner ? a.name : (b ? b.name : "TBD"))
    : (a && b ? `${a.name} vs ${b.name}` : "TBD");
  return `<div class="${cls}" id="${id}" title="${title}">${inner}</div>`;
}

function recordBadge(r) {
  return r
    ? `<div class="bk-record">${r}</div>`
    : `<div class="bk-record" style="visibility:hidden">0-0</div>`;
}

// Two-team pair (WC or DS matchup showing both teams).
// slot-a = top (higher seed / bye), slot-b = bottom (lower seed / WC winner).
function teamPair(prefix, matchup) {
  const a = matchup && matchup.team_a;
  const b = matchup && matchup.team_b;
  const w = matchup && matchup.winner_team_id;
  return `<div class="bk-pair">
    ${teamSlot(`${prefix}-a`, a, !!(w && a && w === a.team_id), 62)}
    ${recordBadge(matchup && matchup.record)}
    ${teamSlot(`${prefix}-b`, b, !!(w && b && w === b.team_id), 62)}
  </div>`;
}

function trophy() {
  return `<svg class="bk-trophy" viewBox="0 0 64 64" width="72" height="72">
    <defs>
      <linearGradient id="tg" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="#ffe9a8"/>
        <stop offset="50%" stop-color="#d4af37"/>
        <stop offset="100%" stop-color="#a8790a"/>
      </linearGradient>
    </defs>
    <path d="M18 6H46V19C46 32 39 41 32 41C25 41 18 32 18 19Z" fill="url(#tg)" stroke="#6b4a03" stroke-width="2.2" stroke-linejoin="round"/>
    <path d="M18 10C7 10 5 23 14 29C16.5 30.7 18 30.5 18 30.5" fill="none" stroke="#6b4a03" stroke-width="3.2" stroke-linecap="round"/>
    <path d="M46 10C57 10 59 23 50 29C47.5 30.7 46 30.5 46 30.5" fill="none" stroke="#6b4a03" stroke-width="3.2" stroke-linecap="round"/>
    <path d="M32 13.5L34.2 18.6L39.8 19L35.5 22.6L36.9 28L32 25L27.1 28L28.5 22.6L24.2 19L29.8 18.6Z" fill="#fff6dc"/>
    <rect x="28" y="41" width="8" height="10" fill="url(#tg)" stroke="#6b4a03" stroke-width="2.2"/>
    <path d="M19 51H45L41 59H23Z" fill="url(#tg)" stroke="#6b4a03" stroke-width="2.2" stroke-linejoin="round"/>
    <rect x="14" y="59" width="36" height="5.5" rx="2" fill="#6b4a03"/>
  </svg>`;
}

function col(label, bodyHtml, extraClass) {
  return `<div class="bk-col${extraClass ? " " + extraClass : ""}">
    <div class="bk-lbl">${label}</div>
    <div class="bk-body">${bodyHtml}</div>
  </div>`;
}

// Build a "one side of the championship" slot from a championship matchup.
// side='a' = top team (DS-0 winner), side='b' = bottom team (DS-1 winner).
function csSlot(id, champ, side) {
  if (!champ) return teamSlot(id, null, false, 62);
  const team = side === "a" ? champ.team_a : champ.team_b;
  if (!team) return teamSlot(id, null, false, 62);
  const win = champ.winner_team_id;
  const isWinner = !!(win && team.team_id === win);
  return teamSlot(id, team, isWinner, 62);
}

// Championship pair: both slots + record badge between them (mirrors teamPair).
function csPair(prefix, champ) {
  return `<div class="bk-pair">
    ${csSlot(`${prefix}-0`, champ, "a")}
    ${recordBadge(champ && champ.record)}
    ${csSlot(`${prefix}-1`, champ, "b")}
  </div>`;
}

function bracketHtml(data) {
  const nl = data.nl, al = data.al;
  const ws = data.world_series;
  const nlTeam = ws && ws.team_a, alTeam = ws && ws.team_b;
  const winnerId = ws && ws.winner_team_id;
  const winnerTeam = winnerId
    ? (nlTeam && nlTeam.team_id === winnerId ? nlTeam : alTeam)
    : null;

  const nlWC = nl.wildcard || [], nlDS = nl.division || [];
  const alWC = al.wildcard || [], alDS = al.division || [];
  const nlCS = (nl.championship || [])[0] || null;
  const alCS = (al.championship || [])[0] || null;

  return `
    <div class="bk-wrap" id="bk-wrap">
      <svg id="bk-svg" class="bk-svg"></svg>

      <div class="bk-half">
        <div class="bk-league-lbl">National League</div>
        <div class="bk-half-row">
          ${col("Wild Card",
            teamPair("nl-wc-0", nlWC[0]) + teamPair("nl-wc-1", nlWC[1] || null),
            "bk-col-wc")}
          ${col("Division Series",
            teamPair("nl-ds-0", nlDS[0]) + teamPair("nl-ds-1", nlDS[1] || null))}
        </div>
      </div>

      ${col("NLCS", csPair("nl-cs", nlCS), "bk-col-mid")}
      ${col("NL Champion",
        matchupSlot("nl-champ", nlTeam ? { team_a: nlTeam, winner_team_id: winnerId ? nlTeam.team_id : null } : null),
        "bk-col-mid")}

      <div class="bk-col bk-col-ws">
        <div class="bk-lbl" style="visibility:hidden">WS</div>
        <div class="bk-body bk-ws-body">
          <div class="bk-ws-block">
            <div class="bk-ws-winner" id="ws-winner">
              ${winnerTeam
                ? teamSlot("ws-winner-slot", winnerTeam, true, 68)
                : `<div class="bk-slot empty bk-champ-ph">Champion</div>`}
            </div>
            ${trophy()}
            <div class="bk-ws-label">World Series</div>
            ${recordBadge(ws && ws.record)}
          </div>
        </div>
      </div>

      ${col("AL Champion",
        matchupSlot("al-champ", alTeam ? { team_a: alTeam, winner_team_id: winnerId ? alTeam.team_id : null } : null),
        "bk-col-mid")}
      ${col("ALCS", csPair("al-cs", alCS), "bk-col-mid")}

      <div class="bk-half">
        <div class="bk-league-lbl" style="text-align:right">American League</div>
        <div class="bk-half-row">
          ${col("Division Series",
            teamPair("al-ds-0", alDS[0]) + teamPair("al-ds-1", alDS[1] || null))}
          ${col("Wild Card",
            teamPair("al-wc-0", alWC[0]) + teamPair("al-wc-1", alWC[1] || null),
            "bk-col-wc")}
        </div>
      </div>
    </div>
    <div class="muted" style="font-size:11px;margin-top:10px;">
      Seeds 1 and 2 (division winners) are byes to the Division Series. Seeds 3–6 play the Wild Card round first.${
        data.is_real ? " Highlighted logos are real series winners; open boxes are rounds not yet played." : ""
      }
    </div>`;
}

// Elbow from two source slots to a destination, branching at their MIDPOINT.
// Used for DS-pair → NLCS-slot, and NLCS-slot → Champion.
function drawElbow(svg, wr, idA, idB, destId, reversed) {
  const a = document.getElementById(idA);
  const b = document.getElementById(idB);
  const d = document.getElementById(destId);
  if (!a || !b || !d) return;
  const ra = a.getBoundingClientRect(), rb = b.getBoundingClientRect(), rd = d.getBoundingClientRect();

  const ax = (reversed ? ra.left : ra.right) - wr.left;
  const bx = (reversed ? rb.left : rb.right) - wr.left;
  const ay = ra.top + ra.height / 2 - wr.top;
  const by = rb.top + rb.height / 2 - wr.top;
  const dx = (reversed ? rd.right : rd.left) - wr.left;
  const dy = rd.top + rd.height / 2 - wr.top;
  const midY = (ay + by) / 2;
  const ex = reversed ? ax - (ax - dx) / 2 : ax + (dx - ax) / 2;

  // Branch at midpoint; step to dest center if there's a height offset.
  const path = `M${ax} ${ay}H${ex} M${bx} ${by}H${ex} M${ex} ${ay}V${by} M${ex} ${midY}V${dy}H${dx}`;
  const el = document.createElementNS("http://www.w3.org/2000/svg", "path");
  el.setAttribute("d", path);
  el.setAttribute("stroke", CONNECTOR_COLOR);
  el.setAttribute("stroke-width", "2");
  el.setAttribute("fill", "none");
  svg.appendChild(el);
}

// Elbow from WC pair (A=top/loser, B=bottom/winner) to a DS destination slot,
// with the outgoing horizontal branch at B's height (the winner's position).
// Since WC-B and DS-dest are at the same height (same pair height, same body),
// this produces a clean horizontal connector.
function drawElbowWinner(svg, wr, idA, idB, destId, reversed) {
  const a = document.getElementById(idA);
  const b = document.getElementById(idB);
  const d = document.getElementById(destId);
  if (!a || !b || !d) return;
  const ra = a.getBoundingClientRect(), rb = b.getBoundingClientRect(), rd = d.getBoundingClientRect();

  const ax = (reversed ? ra.left : ra.right) - wr.left;
  const bx = (reversed ? rb.left : rb.right) - wr.left;
  const ay = ra.top + ra.height / 2 - wr.top;
  const by = rb.top + rb.height / 2 - wr.top;
  const dx = (reversed ? rd.right : rd.left) - wr.left;
  const ex = reversed ? ax - (ax - dx) / 2 : ax + (dx - ax) / 2;

  // Branch at slot-B height (winner), not at pair midpoint.
  const path = `M${ax} ${ay}H${ex} M${bx} ${by}H${ex} M${ex} ${ay}V${by} M${ex} ${by}H${dx}`;
  const el = document.createElementNS("http://www.w3.org/2000/svg", "path");
  el.setAttribute("d", path);
  el.setAttribute("stroke", CONNECTOR_COLOR);
  el.setAttribute("stroke-width", "2");
  el.setAttribute("fill", "none");
  svg.appendChild(el);
}

// NL/AL champion slots converging upward into the WS winner slot above the trophy.
function drawConverge(svg, wr, idA, idB, destId) {
  const a = document.getElementById(idA);
  const b = document.getElementById(idB);
  const d = document.getElementById(destId);
  if (!a || !b || !d) return;
  const ra = a.getBoundingClientRect(), rb = b.getBoundingClientRect(), rd = d.getBoundingClientRect();
  const ax = ra.left + ra.width / 2 - wr.left;
  const bx = rb.left + rb.width / 2 - wr.left;
  const ay = ra.top - wr.top;
  const by = rb.top - wr.top;
  const dx = rd.left + rd.width / 2 - wr.left;
  const dy = rd.bottom - wr.top;
  const midY = Math.min(ay, by) - 16;
  const path = `M${ax} ${ay}V${midY} M${bx} ${by}V${midY} M${ax} ${midY}H${bx} M${dx} ${midY}V${dy}`;
  const el = document.createElementNS("http://www.w3.org/2000/svg", "path");
  el.setAttribute("d", path);
  el.setAttribute("stroke", CONNECTOR_COLOR);
  el.setAttribute("stroke-width", "2");
  el.setAttribute("fill", "none");
  svg.appendChild(el);
}

function drawConnectors() {
  const wrap = document.getElementById("bk-wrap");
  const svg = document.getElementById("bk-svg");
  if (!wrap || !svg) return;
  const wr = wrap.getBoundingClientRect();
  svg.setAttribute("width", wr.width);
  svg.setAttribute("height", wr.height);
  svg.innerHTML = "";

  // NL: left to right
  // WC pair → DS: branch at winner (slot-b) height
  drawElbowWinner(svg, wr, "nl-wc-0-a", "nl-wc-0-b", "nl-ds-0-b", false);
  drawElbowWinner(svg, wr, "nl-wc-1-a", "nl-wc-1-b", "nl-ds-1-b", false);
  // DS pair → NLCS slot: branch at DS pair midpoint (= NLCS slot center)
  drawElbow(svg, wr, "nl-ds-0-a", "nl-ds-0-b", "nl-cs-0", false);
  drawElbow(svg, wr, "nl-ds-1-a", "nl-ds-1-b", "nl-cs-1", false);
  // NLCS slots → NL Champion: midpoint of both slots = champion center
  drawElbow(svg, wr, "nl-cs-0", "nl-cs-1", "nl-champ", false);

  // AL: right to left (mirrored)
  drawElbowWinner(svg, wr, "al-wc-0-a", "al-wc-0-b", "al-ds-0-b", true);
  drawElbowWinner(svg, wr, "al-wc-1-a", "al-wc-1-b", "al-ds-1-b", true);
  drawElbow(svg, wr, "al-ds-0-a", "al-ds-0-b", "al-cs-0", true);
  drawElbow(svg, wr, "al-ds-1-a", "al-ds-1-b", "al-cs-1", true);
  drawElbow(svg, wr, "al-cs-0", "al-cs-1", "al-champ", true);

  // Both champions converge into WS winner slot above the trophy
  drawConverge(svg, wr, "nl-champ", "al-champ", "ws-winner");
}

async function loadBracket() {
  setPageLoading(true);
  try {
    const data = await getJSON(`/api/playoffs/bracket?season=${state.season}`);
    const title = data.postseason_started ? "Playoff Alignment" : "Current Playoff Alignment";
    document.getElementById("bracket-title-mount").innerHTML = `
      <div style="display:flex;align-items:center;gap:12px;">
        <div style="font-size:24px;font-weight:800;color:var(--forest-dark)">${title}</div>
        ${data.is_real ? '<div class="bracket-real-badge">Real results</div>' : ""}
      </div>`;
    document.getElementById("bracket-mount").innerHTML = bracketHtml(data);
    setTimeout(drawConnectors, 50);
    [...document.querySelectorAll("#bk-wrap img")].forEach((img) => {
      if (!img.complete) img.addEventListener("load", drawConnectors, { once: true });
    });
  } finally {
    setPageLoading(false);
  }
}

window.addEventListener("resize", () => {
  clearTimeout(window._bkTimer);
  window._bkTimer = setTimeout(drawConnectors, 150);
});

window.onDataRefresh = loadBracket;

(async function () {
  state.season = await seasonSelect.load();
  await loadBracket();
})();
