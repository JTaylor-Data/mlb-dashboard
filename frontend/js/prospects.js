const state = { view: "league", teamId: null, position: "", prospects: [], allProspects: [], selectedMlbId: null };

const viewToggle = document.getElementById("view-toggle");
const teamSelectWrap = document.getElementById("team-select-wrap");
const teamSelect = document.getElementById("team-select");
const positionFilterWrap = document.getElementById("position-filter-wrap");
const positionFilter = document.getElementById("position-filter");
const dateWrap = document.getElementById("date-wrap");
const dateInput = document.getElementById("performers-date-input");
const searchWrap = document.getElementById("search-wrap");
const searchInput = document.getElementById("prospect-search-input");
const searchDropdown = document.getElementById("prospect-search-dropdown");
const prospectsMount = document.getElementById("prospects-mount");
const detailMount = document.getElementById("detail-mount");

const PERFORMER_RANK_BG = { 1: "#e3c48c", 2: "#ecd7ab", 3: "#f3e6cb" };

function formatOPS(v) {
  if (v == null) return "-";
  const s = v.toFixed(3);
  return v < 1 ? s.replace(/^0\./, ".") : s;
}
function formatPct(v) {
  return v == null ? "-" : `${v.toFixed(1)}%`;
}
function foldText(s) {
  return s.normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase();
}

/* ---------- League Wide / Team Specific player list ---------- */

function statCells(p) {
  const hand = p.is_pitcher ? p.throws : p.bats;
  const statLabel = p.is_pitcher ? "ERA" : "OPS";
  const statValue = p.is_pitcher ? (p.era != null ? p.era.toFixed(2) : "-") : formatOPS(p.ops);
  const defs = [
    [p.level || "-", "Level"],
    [hand || "-", "Hand"],
    [p.position || "-", "Pos"],
    [p.gp ?? "-", "GP"],
    [formatPct(p.k_pct), "K%"],
    [formatPct(p.bb_pct), "BB%"],
    [statValue, statLabel],
  ];
  return defs
    .map(([v, l]) => `<div class="prospect-cell"><div class="prospect-cell-value">${v}</div><div class="prospect-cell-label">${l}</div></div>`)
    .join("");
}

function subLineHtml(p) {
  if (state.view === "league") {
    const logo = `https://www.mlbstatic.com/team-logos/${p.team_id}.svg`;
    return `<img class="team-logo" src="${logo}" onerror="this.style.visibility='hidden'" /> <span class="team-card-link" data-teamcard-id="${p.team_id}">${p.team_name || ""}</span> &nbsp;&#183;&nbsp; ETA ${p.eta ?? "-"}`;
  }
  return `Age ${p.age ?? "-"} &nbsp;&#183;&nbsp; ETA ${p.eta ?? "-"}`;
}

function plaqueHtml(p) {
  if (p.top100_rank == null) return `<div></div>`;
  return `
    <div class="top100-plaque">
      <div class="top100-plaque-label">Top 100</div>
      <div class="top100-plaque-value">#${p.top100_rank}</div>
    </div>
  `;
}

function prospectRowHtml(rank, p) {
  const classes = ["prospect-row"];
  if (p.top100_rank != null) classes.push("top100");
  if (p.mlb_id === state.selectedMlbId) classes.push("selected");
  return `
    <div class="${classes.join(" ")}" data-mlb-id="${p.mlb_id}">
      <div class="prospect-rank">${rank}</div>
      <img class="prospect-headshot" src="${p.headshot_url}" onerror="this.style.visibility='hidden'" />
      <div class="prospect-name-block">
        <div class="prospect-name">${p.name}</div>
        <div class="prospect-sub">${subLineHtml(p)}</div>
      </div>
      ${statCells(p)}
      ${plaqueHtml(p)}
    </div>
  `;
}

// "SS" also matches a multi-position tag like "SS/2B" or "3B/SS" - a
// prospect listed either way is still a shortstop for filtering purposes.
// "INF" is its own distinct, unspecified-infield tag (Pipeline uses it for
// prospects not pinned to one infield spot), not a stand-in for any of
// C/1B/2B/3B/SS individually.
function matchesPosition(p, token) {
  if (!token) return true;
  const pos = p.position || "";
  return token === "INF" ? pos === "INF" : pos.split("/").includes(token);
}

function renderList(prospects, rankKey) {
  state.allProspects = prospects;
  state.rankKey = rankKey;
  const filtered = prospects.filter((p) => matchesPosition(p, state.position));
  state.prospects = filtered;
  if (!filtered.length) {
    prospectsMount.innerHTML = `<div class="empty-state">No prospects found${state.position ? ` at ${state.position}` : ""}.</div>`;
    return;
  }
  prospectsMount.innerHTML = `<div class="prospects-list">${filtered.map((p) => prospectRowHtml(p[rankKey], p)).join("")}</div>`;
}

async function loadTeamOptions() {
  const data = await getJSON("/api/prospects/teams");
  teamSelect.innerHTML = data.teams.map((t) => `<option value="${t.team_id}">${t.name}</option>`).join("");
  if (state.teamId == null && data.teams.length) state.teamId = data.teams[0].team_id;
  if (state.teamId != null) teamSelect.value = state.teamId;
}

async function loadLeagueWide() {
  const data = await getJSON("/api/prospects/top100");
  renderList(data.prospects, "top100_rank");
}

async function loadTeamSpecific() {
  if (state.teamId == null) return;
  const data = await getJSON(`/api/prospects/team?team_id=${state.teamId}`);
  renderList(data.prospects, "org_rank");
}

/* ---------- Farm Rankings ---------- */

async function loadFarmRankings() {
  state.prospects = [];
  const data = await getJSON("/api/prospects/farm-rankings");
  const rows = data.teams
    .map((t) => {
      const logo = `https://www.mlbstatic.com/team-logos/${t.team_id}.svg`;
      const chips = t.top100_prospects
        .map((p) => `<span class="farm-chip">#${p.rank} ${p.name}</span>`)
        .join("");
      return `
        <tr class="farm-row ${t.farm_rank === 1 ? "farm-best" : ""}" data-team-id="${t.team_id}">
          <td class="center-col"><span class="farm-rank-badge">${t.farm_rank}</span></td>
          <td><img class="team-logo" src="${logo}" onerror="this.style.visibility='hidden'" /></td>
          <td class="farm-team-cell"><span class="team-card-link" data-teamcard-id="${t.team_id}">${t.team_name}</span></td>
          <td class="center-col farm-score-cell">${t.score}</td>
          <td class="center-col">${t.top100_count}</td>
          <td class="center-col">${t.best_rank ?? "-"}</td>
          <td><div class="farm-chips-cell">${chips || '<span class="muted">&#8212;</span>'}</div></td>
        </tr>
      `;
    })
    .join("");
  prospectsMount.innerHTML = `
    <div class="farm-table-wrap">
      <table class="farm-table">
        <thead>
          <tr>
            <th class="center-col">Rank</th><th></th><th>Org</th>
            <th class="center-col">Score</th><th class="center-col"># Top 100</th>
            <th class="center-col">Best Rank</th><th>Top 100 Prospects</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
  `;
  detailMount.innerHTML = "";
}

/* ---------- Top Performers (points-based, since MiLB games have no WPA) ---------- */

function prospectPerformerStatCells(entry) {
  const l = entry.line;
  const defs = entry.is_pitcher
    ? [["IP", l.ip], ["K", l.k], ["H", l.h], ["ER", l.er], ["BB", l.bb], ["W", l.w]]
    : [["AB", l.ab], ["H", l.h], ["HR", l.hr], ["RBI", l.rbi], ["SB", l.sb], ["R", l.r]];
  return defs
    .map(([label, value]) => `<div class="stat-cell"><div class="stat-cell-value">${value}</div><div class="stat-cell-label">${label}</div></div>`)
    .join("");
}

function prospectPerformerRowHtml(rank, entry) {
  const bg = PERFORMER_RANK_BG[rank] || (rank % 2 ? "#fff" : "#fbf7ef");
  const roleTag = entry.is_pitcher ? "PITCHER" : "BATTER";
  const logo = entry.team_id ? `https://www.mlbstatic.com/team-logos/${entry.team_id}.svg` : "";
  const seasonStatLabel = entry.is_pitcher ? "ERA" : "OPS";
  const seasonStatValue = entry.is_pitcher
    ? (entry.season_era != null ? entry.season_era.toFixed(2) : "-")
    : formatOPS(entry.season_ops);
  return `
    <div class="performer-row prospect-row" data-mlb-id="${entry.mlb_id}" style="background:${bg}; cursor:pointer;">
      <div class="performer-rank">${rank}</div>
      <img class="prospect-headshot" style="width:42px; height:42px;" src="${entry.headshot_url}" onerror="this.style.visibility='hidden'" />
      <div>
        <div class="performer-name">${entry.name}</div>
        <div class="performer-sub">
          <img class="team-logo" src="${logo}" onerror="this.style.visibility='hidden'" />
          ${roleTag} &nbsp;&#183;&nbsp; ${entry.level || "-"} &nbsp;&#183;&nbsp; ${entry.opponent}
        </div>
      </div>
      ${prospectPerformerStatCells(entry)}
      <div class="wpa-cell">
        <div class="stat-cell-label">${seasonStatLabel}</div>
        <div class="stat-cell-value">${seasonStatValue}</div>
      </div>
    </div>
  `;
}

async function loadProspectTopPerformers() {
  const day = dateInput.value;
  const data = await getJSON(`/api/prospects/top-performers?date=${day}`);
  // Populated (was always []) so selectProspect()'s own state.prospects
  // lookup succeeds when a performer row is clicked - the click delegation
  // in prospectsMount's listener below is role-agnostic (.prospect-row +
  // data-mlb-id, same as every other view), it just didn't have anything
  // to find before. These entries are missing team_name/position (fields
  // scoutingCardHtml also reads) since /api/prospects/top-performers
  // doesn't carry them - the header renders with a blank team name/
  // position for a performer-view selection rather than fabricating them.
  state.prospects = data.entries;
  if (!data.entries.length) {
    prospectsMount.innerHTML = `<div class="empty-state">No qualifying prospect performances found for ${day}.</div>`;
    detailMount.innerHTML = "";
    return;
  }
  prospectsMount.innerHTML = `<div class="performers-list">${data.entries.map((e, i) => prospectPerformerRowHtml(i + 1, e)).join("")}</div>`;
  detailMount.innerHTML = "";
}

/* ---------- Scouting detail card ---------- */

function gradeRowHtml(label, value) {
  const pct = Math.max(0, Math.min(100, ((value - 20) / 60) * 100));
  const color = pct >= 50 ? "#1a7f37" : "#c9302c";
  return `
    <div class="grade-row">
      <div class="grade-label">${label}</div>
      <div class="grade-track">
        <div class="grade-fill" style="width:${pct}%; background:${color};"></div>
        <div class="grade-badge" style="left:${pct}%; background:${color};">${value}</div>
      </div>
    </div>
  `;
}

function scoutingCardHtml(p, data) {
  const logo = `https://www.mlbstatic.com/team-logos/${p.team_id}.svg`;
  // One close button, in the shared header every entry point (league/Top
  // 100 list, team-specific list, farm rankings, search, performers) all
  // render through - closeProspectDetail() just re-clears detailMount,
  // same effect the existing view-switch/reload clears already had, now
  // also reachable without switching views.
  const closeBtn = `<button class="scouting-close-btn" type="button" onclick="closeProspectDetail()" aria-label="Close">&times;</button>`;
  const header = `
    <div class="scouting-card-header">
      ${closeBtn}
      <img class="scouting-headshot" src="${p.headshot_url}" onerror="this.style.visibility='hidden'" />
      <div>
        <div class="prospect-name">${p.name}</div>
        <div class="prospect-sub"><img class="team-logo" src="${logo}" onerror="this.style.visibility='hidden'" /> <span class="team-card-link" data-teamcard-id="${p.team_id}">${p.team_name || ""}</span> &nbsp;&#183;&nbsp; ${p.position || ""}</div>
      </div>
    </div>
  `;

  if (!data.has_report) {
    return `<div class="card scouting-card">${header}<div class="empty-state" style="margin-top:14px;">No scouting report available yet for this player.</div></div>`;
  }

  const overall = data.grades.Overall ?? data.overall_grade;
  const gradeRows = Object.entries(data.grades)
    .filter(([k]) => k !== "Overall")
    .map(([k, v]) => gradeRowHtml(k, v))
    .join("");
  const report = (data.report_text || "")
    .split("\n\n")
    .filter(Boolean)
    .map((para) => `<p>${para}</p>`)
    .join("");
  const video = data.video_url
    ? `<a href="${data.video_url}" target="_blank" rel="noreferrer" class="scouting-video-link">Watch scouting video &rarr;</a>`
    : "";

  return `
    <div class="card scouting-card">
      ${header}
      <div class="scouting-overall">
        <div class="badge"><div class="badge-label">Overall (20-80)</div><div class="badge-value">${overall ?? "-"}</div></div>
      </div>
      <div class="grade-panel">${gradeRows}</div>
      <div class="scouting-report-text">${report}</div>
      ${video}
      <div class="muted" style="font-size:11px; margin-top:10px;">Report: ${data.report_label} &middot; MLB Pipeline scouting write-up via Baseball Savant</div>
    </div>
  `;
}

function closeProspectDetail() {
  detailMount.innerHTML = "";
  state.selectedMlbId = null;
  [...document.querySelectorAll(".prospect-row.selected")].forEach((r) => r.classList.remove("selected"));
}

async function selectProspect(mlbId) {
  state.selectedMlbId = mlbId;
  [...document.querySelectorAll(".prospect-row")].forEach((r) =>
    r.classList.toggle("selected", Number(r.dataset.mlbId) === mlbId)
  );

  const p = state.prospects.find((x) => x.mlb_id === mlbId);
  if (!p) return;
  detailMount.innerHTML = `<div class="card scouting-card"><div class="empty-state">Loading scouting report...</div></div>`;
  const data = await getJSON(`/api/prospects/${mlbId}/scouting`);
  detailMount.innerHTML = scoutingCardHtml(p, data);
}

/* ---------- Search (scoped to whatever list is currently loaded) ---------- */

function renderSearchMatches(query) {
  const q = foldText(query.trim());
  if (!q) {
    searchDropdown.style.display = "none";
    return;
  }
  const matches = state.prospects.filter((p) => foldText(p.name).includes(q)).slice(0, 20);
  if (!matches.length) {
    searchDropdown.innerHTML = `<div class="player-search-empty">No matches on this list</div>`;
    searchDropdown.style.display = "block";
    return;
  }
  searchDropdown.innerHTML = matches
    .map((p) => {
      const tag = p.top100_rank != null
        ? `Top 100 #${p.top100_rank}`
        : state.view === "league" ? (p.team_name || "") : `#${p.org_rank}`;
      return `
        <div class="player-search-item" data-mlb-id="${p.mlb_id}">
          <span>${p.name}</span>
          <span class="player-search-role">${tag}</span>
        </div>
      `;
    })
    .join("");
  searchDropdown.style.display = "block";
}

function goToProspect(mlbId) {
  searchInput.value = "";
  searchDropdown.style.display = "none";
  selectProspect(mlbId);
  const row = document.querySelector(`.prospect-row[data-mlb-id="${mlbId}"]`);
  if (row) row.scrollIntoView({ behavior: "smooth", block: "center" });
}

searchInput.addEventListener("input", () => renderSearchMatches(searchInput.value));
searchInput.addEventListener("focus", () => renderSearchMatches(searchInput.value));
searchInput.addEventListener("blur", () => setTimeout(() => { searchDropdown.style.display = "none"; }, 150));
searchDropdown.addEventListener("mousedown", (e) => {
  const item = e.target.closest(".player-search-item");
  if (!item) return;
  e.preventDefault();
  goToProspect(Number(item.dataset.mlbId));
});

/* ---------- View switching ---------- */

async function switchToTeamView(teamId) {
  state.view = "team";
  state.teamId = teamId;
  [...viewToggle.querySelectorAll("button")].forEach((b) => b.classList.toggle("active", b.dataset.value === "team"));
  teamSelectWrap.style.display = "block";
  searchWrap.style.display = "block";
  // Was missing - the viewToggle click handler shows this for "team" view
  // (line ~380) but this alternate entry point (clicking a farm-rankings
  // row) never did, so a click from Farm Rankings landed on Team Specific
  // with the position filter dropdown stuck hidden until the user manually
  // clicked a top-nav view button.
  positionFilterWrap.style.display = "block";
  teamSelect.value = teamId;
  await loadProspects();
}

prospectsMount.addEventListener("click", (e) => {
  const prospectRow = e.target.closest(".prospect-row");
  if (prospectRow) {
    selectProspect(Number(prospectRow.dataset.mlbId));
    return;
  }
  const farmRow = e.target.closest(".farm-row");
  if (farmRow) switchToTeamView(Number(farmRow.dataset.teamId));
});

async function loadProspects() {
  prospectsMount.innerHTML = `<div class="empty-state">Loading...</div>`;
  if (state.view === "league") await loadLeagueWide();
  else if (state.view === "team") await loadTeamSpecific();
  else if (state.view === "performers") await loadProspectTopPerformers();
  else await loadFarmRankings();
}

viewToggle.addEventListener("click", async (e) => {
  const btn = e.target.closest("button");
  if (!btn || btn.dataset.value === state.view) return;
  state.view = btn.dataset.value;
  state.selectedMlbId = null;
  detailMount.innerHTML = "";
  [...viewToggle.querySelectorAll("button")].forEach((b) => b.classList.toggle("active", b.dataset.value === state.view));
  teamSelectWrap.style.display = state.view === "team" ? "block" : "none";
  positionFilterWrap.style.display = (state.view === "league" || state.view === "team") ? "block" : "none";
  dateWrap.style.display = state.view === "performers" ? "block" : "none";
  searchWrap.style.display = (state.view === "farm" || state.view === "performers") ? "none" : "block";
  await loadProspects();
});

teamSelect.addEventListener("change", async () => {
  state.teamId = Number(teamSelect.value);
  state.selectedMlbId = null;
  detailMount.innerHTML = "";
  await loadProspects();
});

positionFilter.addEventListener("change", () => {
  state.position = positionFilter.value;
  state.selectedMlbId = null;
  detailMount.innerHTML = "";
  renderList(state.allProspects, state.rankKey);
});

dateInput.addEventListener("change", loadProspects);

window.onDataRefresh = loadProspects;

(async function init() {
  await loadTeamOptions();
  const meta = await getJSON("/api/meta");
  if (meta.latest_game_date) dateInput.value = meta.latest_game_date;
  await loadProspects();
})();
