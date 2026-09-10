let currentLeague = "AL";
const leagueToggle = document.getElementById("league-toggle");
const seasonSelect = createSeasonSelect(document.getElementById("season-select"), async (season) => {
  state.season = season;
  await loadStandings();
});
const state = { season: 2026 };

leagueToggle.addEventListener("click", async (e) => {
  const btn = e.target.closest("button");
  if (!btn || btn.dataset.value === currentLeague) return;
  currentLeague = btn.dataset.value;
  [...leagueToggle.querySelectorAll("button")].forEach((b) => b.classList.toggle("active", b.dataset.value === currentLeague));
  await loadStandings();
});

function divisionTableHtml(division) {
  const rows = division.teams
    .map((t) => {
      const logo = `https://www.mlbstatic.com/team-logos/${t.team_id}.svg`;
      const crown = t.is_best_in_league ? " &#128081;" : "";
      const rdStr = t.run_diff > 0 ? `+${t.run_diff}` : `${t.run_diff}`;
      const rdColor = t.run_diff > 0 ? "#1a7f37" : t.run_diff < 0 ? "#c9302c" : "var(--text-muted)";
      return `
        <tr class="${t.is_best_in_league ? "best-record" : ""}">
          <td><img class="team-logo" src="${logo}" onerror="this.style.display='none'" /></td>
          <td class="team-cell"><span class="team-card-link" data-teamcard-id="${t.team_id}">${t.name}</span>${crown}</td>
          <td>${t.wins}</td>
          <td>${t.losses}</td>
          <td>${t.win_pct.toFixed(3)}</td>
          <td style="color:${rdColor}; font-weight:600;">${rdStr}</td>
          <td>${t.one_run_record}</td>
          <td>${t.last_10}</td>
          <td>${t.last_30}</td>
        </tr>
      `;
    })
    .join("");

  return `
    <h2>${division.name}</h2>
    <table class="data-table division-table" style="margin-bottom:22px;">
      <thead>
        <tr>
          <th></th><th>Team</th><th>W</th><th>L</th><th>Win%</th><th>Run Diff</th><th>1-Run</th><th>Last 10</th><th>Last 30</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

async function loadStandings() {
  setPageLoading(true);
  try {
    const data = await getJSON(`/api/standings?league=${currentLeague}&season=${state.season}`);
    document.getElementById("divisions-mount").innerHTML = data.divisions.map(divisionTableHtml).join("");
  } finally {
    setPageLoading(false);
  }
}

window.onDataRefresh = loadStandings;

(async function init() {
  state.season = await seasonSelect.load();
  await loadStandings();
})();
