// Rosters page - a row of clickable team logos (all 12 teams currently in
// bracket position, both leagues) instead of a dropdown, then that team's
// projected lineup/bench/rotation/bullpen. All highlighting is WAR-based,
// per direction: top 3 lineup bats, the best rotation arm, and the best 2
// bullpen arms (each ranked within their own group) get row-highlighted.
const state = { season: 2026, teamId: null };
let allTeams = [];

const seasonSelect = createSeasonSelect(document.getElementById("season-select"), async (season) => {
  state.season = season;
  await loadTeams();
});

function logoImg(teamId, size) {
  return `<img class="team-logo" src="https://www.mlbstatic.com/team-logos/${teamId}.svg" onerror="this.style.display='none'" style="width:${size}px; height:${size}px;" />`;
}

// Indices (not team_ids) of the top `n` entries by `key`, ranked `dir`
// ("asc" = lowest wins, e.g. ERA; "desc" = highest wins, e.g. WAR) - null/
// undefined values are excluded from ranking entirely rather than sorting
// to one end, same convention team_card.py's _rank() already uses.
function topIndices(players, key, n, dir) {
  const ranked = players
    .map((p, i) => ({ i, v: p[key] }))
    .filter((x) => x.v !== null && x.v !== undefined)
    .sort((a, b) => (dir === "asc" ? a.v - b.v : b.v - a.v));
  return new Set(ranked.slice(0, n).map((x) => x.i));
}

function batterRowHtml(b, isTop) {
  if (!b || b.player_id == null) return `<tr><td class="pos-cell">${b ? b.position : ""}</td><td colspan="6" class="muted">-</td></tr>`;
  return `
    <tr class="${isTop ? "roster-highlight" : ""}">
      <td class="pos-cell">${b.position}</td>
      <td class="team-cell">${b.name}</td>
      <td>${b.pa}</td>
      <td>${b.hr}</td>
      <td>${b.ops != null ? b.ops.toFixed(3) : "-"}</td>
      <td>${b.woba != null ? b.woba.toFixed(3) : "-"}</td>
      <td>${b.war != null ? b.war.toFixed(1) : "-"}</td>
    </tr>
  `;
}

function pitcherRowHtml(p, gamesCol, isTop) {
  return `
    <tr class="${isTop ? "roster-highlight" : ""}">
      <td class="team-cell">${p.name}</td>
      <td>${p.ip}</td>
      <td>${gamesCol === "gs" ? p.games_started : p.games_pitched}</td>
      <td>${p.era != null ? p.era.toFixed(2) : "-"}</td>
      <td>${p.k_pct != null ? p.k_pct.toFixed(1) + "%" : "-"}</td>
      <td>${p.bb_pct != null ? p.bb_pct.toFixed(1) + "%" : "-"}</td>
      <td>${p.war != null ? p.war.toFixed(1) : "-"}</td>
    </tr>
  `;
}

function rosterHtml(data) {
  const label = data.is_real_postseason_roster ? "Playoff Roster" : "Projected Playoff Roster";
  const note = data.is_real_postseason_roster
    ? "Real postseason games are underway for this team - this list is still built from real current-season usage patterns (this project has no source for the actual announced postseason active roster), so treat it as a best-estimate, not official."
    : "First-pass estimate from real current-season usage (PA for batters, starts/appearances for pitchers) among players who've actually played for this team this season and aren't on the IL - not a simulation of real 26-man roster rules.";

  const topBatters = topIndices(data.lineup, "war", 3, "desc");
  const topRotation = topIndices(data.rotation, "war", 1, "desc");
  const topBullpen = topIndices(data.bullpen, "war", 2, "desc");

  return `
    <div class="card">
      <div class="roster-title">${label}</div>
      <div class="muted" style="font-size:11px; margin-bottom:14px;">${note}</div>

      <div class="roster-section-title">Lineup <span class="muted" style="font-weight:400;">- top 3 bats by WAR highlighted</span></div>
      <table class="data-table" style="margin-bottom:18px;">
        <thead><tr><th>Pos</th><th>Player</th><th>PA</th><th>HR</th><th>OPS</th><th>wOBA</th><th>WAR</th></tr></thead>
        <tbody>${data.lineup.map((b, i) => batterRowHtml(b, topBatters.has(i))).join("")}</tbody>
      </table>

      <div class="roster-section-title">Bench</div>
      <table class="data-table" style="margin-bottom:18px;">
        <thead><tr><th>Pos</th><th>Player</th><th>PA</th><th>HR</th><th>OPS</th><th>wOBA</th><th>WAR</th></tr></thead>
        <tbody>${data.bench.length ? data.bench.map((b) => batterRowHtml(b, false)).join("") : '<tr><td colspan="7" class="muted">No bench candidates found.</td></tr>'}</tbody>
      </table>

      <div class="roster-section-title">Rotation (Top 3) <span class="muted" style="font-weight:400;">- best WAR highlighted</span></div>
      <table class="data-table" style="margin-bottom:18px;">
        <thead><tr><th>Player</th><th>IP</th><th>GS</th><th>ERA</th><th>K%</th><th>BB%</th><th>WAR</th></tr></thead>
        <tbody>${data.rotation.length ? data.rotation.map((p, i) => pitcherRowHtml(p, "gs", topRotation.has(i))).join("") : '<tr><td colspan="7" class="muted">No starters found.</td></tr>'}</tbody>
      </table>

      <div class="roster-section-title">Bullpen <span class="muted" style="font-weight:400;">- top 2 by WAR highlighted</span></div>
      <table class="data-table">
        <thead><tr><th>Player</th><th>IP</th><th>APP</th><th>ERA</th><th>K%</th><th>BB%</th><th>WAR</th></tr></thead>
        <tbody>${data.bullpen.length ? data.bullpen.map((p, i) => pitcherRowHtml(p, "app", topBullpen.has(i))).join("") : '<tr><td colspan="7" class="muted">No relievers found.</td></tr>'}</tbody>
      </table>
    </div>
  `;
}

function teamPickerHtml() {
  return `
    <div class="roster-team-picker">
      ${allTeams
        .map((t) => `<button class="roster-team-btn ${t.team_id === state.teamId ? "active" : ""}" data-team-id="${t.team_id}" title="${t.name}">${logoImg(t.team_id, 34)}</button>`)
        .join("")}
    </div>
  `;
}

async function loadTeams() {
  setPageLoading(true);
  try {
    const [al, nl] = await Promise.all([
      getJSON(`/api/playoffs/teams?league=AL&season=${state.season}`),
      getJSON(`/api/playoffs/teams?league=NL&season=${state.season}`),
    ]);
    allTeams = [...al.teams, ...nl.teams];
    if (!allTeams.length) {
      document.getElementById("team-picker-mount").innerHTML = "";
      document.getElementById("roster-mount").innerHTML = `<div class="empty-state">No projected playoff teams yet.</div>`;
      return;
    }
    if (state.teamId === null || !allTeams.some((t) => t.team_id === state.teamId)) {
      state.teamId = allTeams[0].team_id;
    }
    document.getElementById("team-picker-mount").innerHTML = teamPickerHtml();
    document.getElementById("team-picker-mount").addEventListener("click", async (e) => {
      const btn = e.target.closest(".roster-team-btn");
      if (!btn) return;
      state.teamId = Number(btn.dataset.teamId);
      [...document.querySelectorAll(".roster-team-btn")].forEach((b) => b.classList.toggle("active", Number(b.dataset.teamId) === state.teamId));
      await loadRoster();
    });
    await loadRoster();
  } finally {
    setPageLoading(false);
  }
}

async function loadRoster() {
  const mount = document.getElementById("roster-mount");
  mount.innerHTML = `<div class="empty-state">Loading roster...</div>`;
  const data = await getJSON(`/api/playoffs/roster?team_id=${state.teamId}&season=${state.season}`);
  mount.innerHTML = rosterHtml(data);
}

window.onDataRefresh = loadTeams;

(async function init() {
  state.season = await seasonSelect.load();
  await loadTeams();
})();
