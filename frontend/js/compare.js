let state = { split: "ALL", playerIdA: null, playerIdB: null, season: 2026 };

const splitToggle = document.getElementById("split-toggle");
const seasonSelect = createSeasonSelect(document.getElementById("season-select"), async (season) => {
  state.season = season;
  await loadPlayerLists();
  await loadCompare();
});

splitToggle.addEventListener("click", async (e) => {
  const btn = e.target.closest("button");
  if (!btn || btn.dataset.value === state.split) return;
  state.split = btn.dataset.value;
  [...splitToggle.querySelectorAll("button")].forEach((b) => b.classList.toggle("active", b.dataset.value === state.split));
  await loadCompare();
});

const playerSearchA = createPlayerSearch(
  document.getElementById("player-search-input-a"),
  document.getElementById("player-search-dropdown-a"),
  async (p) => { state.playerIdA = p.player_id; await loadCompare(); },
  { roleFilter: "batter" }
);
const playerSearchB = createPlayerSearch(
  document.getElementById("player-search-input-b"),
  document.getElementById("player-search-dropdown-b"),
  async (p) => { state.playerIdB = p.player_id; await loadCompare(); },
  { roleFilter: "batter" }
);

async function loadPlayerLists() {
  await playerSearchA.load(state.season);
  await playerSearchB.load(state.season);
  const a = playerSearchA.selectDefault();
  const b = playerSearchB.selectDefault((p) => p.player_id !== a?.player_id);
  if (a) state.playerIdA = a.player_id;
  if (b) state.playerIdB = b.player_id;
}

const RT_GOOD = "#1a7f37";
const RT_BAD = "#c9302c";

function barRowsHtml(rows) {
  return rows
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
}

function playerPanelHtml(player) {
  const logo = player.team_id ? `https://www.mlbstatic.com/team-logos/${player.team_id}.svg` : "";
  return `
    <div class="compare-player-header">
      ${logo ? `<img src="${logo}" onerror="this.style.display='none'" />` : ""}
      <div class="name">${player.name}</div>
    </div>
    <div class="card rt-panel">
      <div>
        <div class="rt-section-title">Swing Decisions</div>
        ${barRowsHtml(player.swing_decisions)}
      </div>
      <div>
        <div class="rt-section-title">Batted Ball Outcomes</div>
        ${barRowsHtml(player.batted_ball_outcomes)}
      </div>
    </div>
  `;
}

let loadCompareRequestId = 0;
async function loadCompare() {
  if (state.playerIdA === null || state.playerIdB === null) return;
  const requestId = ++loadCompareRequestId;
  const data = await getJSON(`/api/compare?player_id_a=${state.playerIdA}&player_id_b=${state.playerIdB}&split=${state.split}&season=${state.season}`);
  if (requestId !== loadCompareRequestId) return;
  document.getElementById("panel-a").innerHTML = playerPanelHtml(data.players[0]);
  document.getElementById("panel-b").innerHTML = playerPanelHtml(data.players[1]);
}

window.onDataRefresh = async () => {
  await loadPlayerLists();
  await loadCompare();
};

(async function init() {
  state.season = await seasonSelect.load();
  await loadPlayerLists();
  await loadCompare();
})();
