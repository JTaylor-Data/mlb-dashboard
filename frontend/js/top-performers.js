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
        <div class="performer-name">${entry.name}</div>
        <div class="performer-sub">${roleTag} &nbsp;&#183;&nbsp; ${entry.opponent}${walkoff}</div>
      </div>
      ${statCells(entry)}
      <div class="wpa-cell">
        <div class="stat-cell-label">WPA</div>
        <div class="stat-cell-value">${entry.wpa.toFixed(3)}</div>
      </div>
    </div>
  `;
}

async function loadTopPerformers() {
  const day = dateInput.value;
  const mount = document.getElementById("performers-mount");
  const data = await getJSON(`/api/top-performers?date=${day}`);
  if (!data.entries.length) {
    mount.innerHTML = `<div class="empty-state">No games found for ${day}.</div>`;
    return;
  }
  mount.innerHTML = `<div class="performers-list">${data.entries.map((e, i) => performerRowHtml(i + 1, e)).join("")}</div>`;
}

dateInput.addEventListener("change", loadTopPerformers);
window.onDataRefresh = loadTopPerformers;

(async function init() {
  const meta = await getJSON("/api/meta");
  dateInput.value = meta.latest_game_date;
  await loadTopPerformers();
})();
