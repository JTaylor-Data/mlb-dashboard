let state = { role: "batter", split: "ALL", playerIdA: null, playerIdB: null, season: 2026 };

const splitToggle = document.getElementById("split-toggle");
const roleToggle = document.getElementById("role-toggle");
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

// roleFilter is baked in at createPlayerSearch() construction time (not a
// live-updatable option), so switching Batters<->Pitchers rebuilds both
// search widgets from scratch rather than trying to mutate one in place.
// destroy() the previous instances first - real, confirmed bug: without
// this, every role toggle stacked a fresh set of input/focus/blur/keydown
// listeners (plus a whole new closure's worth of matches/allPlayers state)
// on the SAME input/dropdown DOM nodes, since they're never replaced, only
// re-wrapped. After a couple of toggles this meant every keystroke ran
// render() multiple times and Enter fired select()/onSelect (a real
// loadCompare() + PlayerContext write) once per stacked instance,
// including old instances still holding a stale roleFilter.
let playerSearchA, playerSearchB;
function buildPlayerSearches() {
  if (playerSearchA) playerSearchA.destroy();
  if (playerSearchB) playerSearchB.destroy();
  const placeholder = state.role === "pitcher" ? "Search a pitcher..." : "Search a hitter...";
  const inputA = document.getElementById("player-search-input-a");
  const inputB = document.getElementById("player-search-input-b");
  inputA.placeholder = placeholder;
  inputB.placeholder = placeholder;
  inputA.value = "";
  inputB.value = "";
  playerSearchA = createPlayerSearch(
    inputA, document.getElementById("player-search-dropdown-a"),
    async (p) => { state.playerIdA = p.player_id; await loadCompare(); },
    { roleFilter: state.role }
  );
  playerSearchB = createPlayerSearch(
    inputB, document.getElementById("player-search-dropdown-b"),
    async (p) => { state.playerIdB = p.player_id; await loadCompare(); },
    { roleFilter: state.role }
  );
}
buildPlayerSearches();

document.getElementById("compare-subtitle").textContent = state.role === "pitcher"
  ? "Pitcher vs. pitcher, the same percentile metrics from the player page. Percentiles recompute with the split toggle."
  : "Batter vs. batter, swing decisions and batted-ball outcomes. Percentiles recompute with the split toggle.";
document.getElementById("split-label").textContent = state.role === "pitcher" ? "Split (opponent bats)" : "Split (opposing pitcher throws)";

roleToggle.addEventListener("click", async (e) => {
  const btn = e.target.closest("button");
  if (!btn || btn.dataset.value === state.role) return;
  state.role = btn.dataset.value;
  [...roleToggle.querySelectorAll("button")].forEach((b) => b.classList.toggle("active", b.dataset.value === state.role));
  document.getElementById("compare-subtitle").textContent = state.role === "pitcher"
    ? "Pitcher vs. pitcher, the same percentile metrics from the player page. Percentiles recompute with the split toggle."
    : "Batter vs. batter, swing decisions and batted-ball outcomes. Percentiles recompute with the split toggle.";
  document.getElementById("split-label").textContent = state.role === "pitcher" ? "Split (opponent bats)" : "Split (opposing pitcher throws)";
  state.playerIdA = null;
  state.playerIdB = null;
  buildPlayerSearches();
  await loadPlayerLists();
  await loadCompare();
});

async function loadPlayerLists() {
  await playerSearchA.load(state.season);
  await playerSearchB.load(state.season);
  const a = playerSearchA.selectDefault();
  const b = playerSearchB.selectDefault((p) => p.player_id !== a?.player_id);
  if (a) state.playerIdA = a.player_id;
  if (b) state.playerIdB = b.player_id;
}

function barRowsHtml(rows) {
  return rows
    .map((m) => {
      const pct = m.pct !== null ? m.pct : 0;
      const color = gradientColor(pct);
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
  const groupsHtml = player.groups
    .map((g) => `
      <div>
        <div class="rt-section-title">${g.title}</div>
        ${barRowsHtml(g.metrics)}
      </div>
    `)
    .join("");
  return `
    <div class="compare-player-header">
      ${logo ? `<img src="${logo}" onerror="this.style.display='none'" />` : ""}
      <div class="name">${player.name}</div>
    </div>
    <div class="card rt-panel">
      ${groupsHtml}
    </div>
  `;
}

let loadCompareRequestId = 0;
async function loadCompare() {
  if (state.playerIdA === null || state.playerIdB === null) return;
  const requestId = ++loadCompareRequestId;
  setPageLoading(true);
  let data;
  try {
    data = await getJSON(`/api/compare?player_id_a=${state.playerIdA}&player_id_b=${state.playerIdB}&role=${state.role}&split=${state.split}&season=${state.season}`);
  } catch (err) {
    if (requestId !== loadCompareRequestId) return; // a newer call superseded this one - its own request/render will win, not an error to surface
    throw err;
  } finally {
    if (requestId === loadCompareRequestId) setPageLoading(false); // only the latest in-flight request controls the bar
  }
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
