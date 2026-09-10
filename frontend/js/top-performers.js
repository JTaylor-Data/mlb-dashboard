const dateInput = document.getElementById("date-input");

const RANK_BG = { 1: "#e3c48c", 2: "#ecd7ab", 3: "#f3e6cb" };

// Every row - batter or pitcher - renders exactly 6 stat cells sharing the
// same .performer-row grid-template-columns (32px 46px 240px repeat(6,1fr)
// 84px, defined once in theme.css) so column N lands at the same x
// regardless of role. Each cell is a real <div>, never an empty string, so
// every column track stays reserved.
function statCells(entry) {
  const s = entry.stats || {};
  const defs = entry.role === "batter"
    ? [["AB", s.ab ?? "-"], ["H", s.h ?? "-"], ["HR", s.hr ?? 0], ["RBI", s.rbi ?? 0], ["SB", s.sb ?? 0], ["OPS", s.season_ops ?? "-"]]
    : [["IP", s.ip ?? "-"], ["H", s.h ?? 0], ["K", s.k ?? 0], ["R", s.r ?? 0], ["ER", s.er ?? 0], ["ERA", s.season_era ?? "-"]];
  return defs
    .map(([label, value]) => `<div class="stat-cell"><div class="stat-cell-value">${value}</div><div class="stat-cell-label">${label}</div></div>`)
    .join("");
}

function performerRowHtml(rank, entry) {
  const logo = entry.team_id ? `https://www.mlbstatic.com/team-logos/${entry.team_id}.svg` : "";
  const walkoff = entry.walkoff ? `<span class="walkoff-badge">WALK-OFF</span>` : "";
  const bg = RANK_BG[rank] || (rank % 2 ? "#fff" : "#fafafa");
  const roleTag = entry.role === "pitcher" ? "PITCHER" : "BATTER";

  return `
    <div class="performer-row" style="background:${bg};">
      <div class="performer-rank">${rank}</div>
      <img class="team-logo" src="${logo}" onerror="this.style.visibility='hidden'" />
      <div>
        <a class="performer-name performer-name-link" href="index.html"
           data-player-id="${entry.player_id}" data-name="${entry.name}" data-role="${entry.role}">${entry.name}</a>
        <div class="performer-sub">${roleTag} &nbsp;&#183;&nbsp; ${entry.opponent}${walkoff}${entry.role === "pitcher" && entry.game_id ? `&nbsp;<a class="outing-btn" href="outings.html?player_id=${entry.player_id}&game_id=${entry.game_id}&season=2026">outing ↗</a>` : ""}</div>
      </div>
      ${statCells(entry)}
      <div class="wpa-cell">
        <div class="stat-cell-label">WPA</div>
        <div class="stat-cell-value">${entry.wpa.toFixed(3)}</div>
      </div>
    </div>
  `;
}

// Small callout listing every walk-off from that night, independent of the
// leaderboard below it - a walk-off's day score may or may not also crack
// the top 10 (both are fine, not mutually exclusive).
function walkoffPanelHtml(walkoffs) {
  if (!walkoffs.length) return "";
  const items = walkoffs
    .map((w) => {
      const logo = w.team_id ? `https://www.mlbstatic.com/team-logos/${w.team_id}.svg` : "";
      const s = w.stats || {};
      return `
        <div class="walkoff-panel-item">
          ${logo ? `<img src="${logo}" onerror="this.style.display='none'" />` : ""}
          <span class="name">${w.name}</span>
          <span class="line">${s.h ?? 0}-${s.ab ?? 0}, ${s.hr ?? 0} HR, ${s.rbi ?? 0} RBI</span>
        </div>
      `;
    })
    .join("");
  return `<div class="walkoff-panel"><span class="walkoff-panel-title">Walk-Offs</span>${items}</div>`;
}

let activePerformerRole = "pitcher";
let lastPerformerData = null;

function renderPerformers(entries) {
  const mount = document.getElementById("performers-mount");
  if (!entries || !entries.length) {
    mount.innerHTML = `<div class="empty-state">No games found.</div>`;
    return;
  }
  const filtered = entries.filter(e => e.role === activePerformerRole).slice(0, 10);
  const toggle = `
    <div class="performer-role-toggle">
      <button class="${activePerformerRole === "pitcher" ? "active" : ""}" data-role="pitcher">Pitchers</button>
      <button class="${activePerformerRole === "batter" ? "active" : ""}" data-role="batter">Batters</button>
    </div>`;
  const rows = filtered.length
    ? `<div class="performers-list">${filtered.map((e, i) => performerRowHtml(i + 1, e)).join("")}</div>`
    : `<div class="empty-state">No ${activePerformerRole}s for this date.</div>`;
  mount.innerHTML = toggle + rows;
  mount.querySelectorAll(".performer-role-toggle button").forEach(btn => {
    btn.addEventListener("click", () => {
      activePerformerRole = btn.dataset.role;
      renderPerformers(lastPerformerData);
    });
  });
}

async function loadTopPerformers() {
  const day = dateInput.value;
  const walkoffMount = document.getElementById("walkoff-mount");
  setPageLoading(true);
  try {
    const data = await getJSON(`/api/top-performers?date=${day}`);
    walkoffMount.innerHTML = walkoffPanelHtml(data.walkoffs || []);
    lastPerformerData = data.entries || [];
    renderPerformers(lastPerformerData);
  } finally {
    setPageLoading(false);
  }
}

dateInput.addEventListener("change", loadTopPerformers);
window.onDataRefresh = loadTopPerformers;

// Clicking a name takes you to the Player tab already loaded on that
// person - PlayerContext is the same cross-page mechanism the player/
// radar/splits/model pages already read on their own init.
document.addEventListener("click", (e) => {
  const link = e.target.closest(".performer-name-link");
  if (!link) return;
  PlayerContext.setCurrent({
    player_id: Number(link.dataset.playerId), name: link.dataset.name, role: link.dataset.role,
  });
  PlayerContext.recordRecent({
    player_id: Number(link.dataset.playerId), name: link.dataset.name, role: link.dataset.role,
  });
});

(async function init() {
  const meta = await getJSON("/api/meta");
  dateInput.value = meta.latest_game_date;
  await loadTopPerformers();
})();
