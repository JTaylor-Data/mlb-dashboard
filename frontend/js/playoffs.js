// WC Race page (playoffs.html) - Division Leaders + the wild card race for
// the selected league. Bracket/Rosters are now their own real pages
// (playoffs-bracket.html/playoffs-roster.js) under the same subnav group -
// see nav.js's PLAYOFFS_GROUP.
const state = { season: 2026, league: "AL" };

const seasonSelect = createSeasonSelect(document.getElementById("season-select"), async (season) => {
  state.season = season;
  await loadAll();
});

document.getElementById("league-toggle").addEventListener("click", async (e) => {
  const btn = e.target.closest("button");
  if (!btn || btn.dataset.value === state.league) return;
  state.league = btn.dataset.value;
  [...document.getElementById("league-toggle").querySelectorAll("button")].forEach((b) => b.classList.toggle("active", b.dataset.value === state.league));
  await loadAll();
});

function teamLogoHtml(teamId, size) {
  return `<img class="team-logo" src="https://www.mlbstatic.com/team-logos/${teamId}.svg" onerror="this.style.display='none'" style="${size ? `width:${size}px; height:${size}px;` : ""}" />`;
}
function teamNameHtml(t) {
  return `<span class="team-card-link" data-teamcard-id="${t.team_id}">${t.name}</span>`;
}

// --- Division Leaders -------------------------------------------------

function divisionLeadersHtml(leaders) {
  const byLeague = { AL: leaders.filter((l) => l.league === "AL"), NL: leaders.filter((l) => l.league === "NL") };
  const col = (league) => `
    <div class="dl-col">
      <div class="dl-col-title">${league}</div>
      ${byLeague[league]
        .map((l) => `
          <div class="dl-row">
            ${teamLogoHtml(l.team_id, 30)}
            <div class="dl-info">
              <div class="dl-division">${l.division}</div>
              <div class="dl-name">${teamNameHtml(l)}</div>
            </div>
            <div class="dl-record">${l.wins}-${l.losses}</div>
            <div class="dl-lead">${l.lead != null ? (l.lead === 0 ? "Tied" : `+${l.lead}`) : "-"}</div>
          </div>
        `)
        .join("")}
    </div>
  `;
  return `<div class="card dl-card"><div class="dl-title">Division Leaders</div><div class="dl-grid">${col("AL")}${col("NL")}</div></div>`;
}

async function loadDivisionLeaders() {
  const data = await getJSON(`/api/playoffs/division-leaders?season=${state.season}`);
  document.getElementById("division-leaders-mount").innerHTML = divisionLeadersHtml(data.leaders);
}

// --- WC Race ------------------------------------------------------------

function wildcardHtml(race) {
  if (!race.length) return `<div class="empty-state">No wild card race data for this selection.</div>`;
  const rows = race
    .map((t) => {
      const rdColor = t.run_diff > 0 ? "#1a7f37" : t.run_diff < 0 ? "#c9302c" : "var(--text-muted)";
      return `
        <tr class="${t.rank === 3 ? "wc-cutline" : ""}">
          <td class="wc-rank">${t.rank}</td>
          <td>${teamLogoHtml(t.team_id, 26)}</td>
          <td class="team-cell">${teamNameHtml(t)}</td>
          <td>${t.wins}</td>
          <td>${t.losses}</td>
          <td>${t.win_pct.toFixed(3)}</td>
          <td style="color:${rdColor}; font-weight:700;">${t.run_diff > 0 ? "+" : ""}${t.run_diff}</td>
          <td>${t.games_back != null ? t.games_back.toFixed(1) : "-"}</td>
          <td>${t.in_wc_spot ? '<span class="wc-in">IN</span>' : ""}</td>
        </tr>
      `;
    })
    .join("");
  return `
    <div class="card wc-card">
      <table class="data-table wc-table">
        <thead><tr><th>#</th><th></th><th>Team</th><th>W</th><th>L</th><th>Win%</th><th>Run Diff</th><th>GB</th><th></th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
      <div class="muted" style="font-size:11px; margin-top:10px;">Division leaders excluded (they can't also hold a wild card spot). Top 3 = current wild card field. GB is games back of the last wild card spot - teams already in are "-".</div>
    </div>
  `;
}

async function loadWildcard() {
  const data = await getJSON(`/api/playoffs/wildcard?league=${state.league}&season=${state.season}`);
  document.getElementById("wildcard-mount").innerHTML = wildcardHtml(data.race);
}

async function loadAll() {
  setPageLoading(true);
  try {
    await Promise.all([loadDivisionLeaders(), loadWildcard()]);
  } finally {
    setPageLoading(false);
  }
}

window.onDataRefresh = loadAll;

(async function init() {
  state.season = await seasonSelect.load();
  await loadAll();
})();
