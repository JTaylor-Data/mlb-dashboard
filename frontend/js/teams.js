// Teams page — list of all 30 teams with aggregate model scores,
// click through to roster / depth chart / team stats.

let allTeams = [];
let currentLeague = "ALL";
let currentSeason = 2026;
let sortKey = "name";
let sortAsc = true;

function scoreClass(v) {
  if (v == null) return "score-na";
  if (v >= 105) return "score-good";
  if (v <= 95) return "score-bad";
  return "";
}

function fmtScore(v) {
  if (v == null) return '<span class="score-na">—</span>';
  const cls = scoreClass(v);
  return `<span class="model-score ${cls}">${v.toFixed(1)}</span>`;
}

function logoUrl(teamId) {
  return `https://www.mlbstatic.com/team-logos/${teamId}.svg`;
}

// ---------------------------------------------------------------------------
// Team list rendering
// ---------------------------------------------------------------------------

function sortedTeams() {
  const filtered = currentLeague === "ALL"
    ? allTeams
    : allTeams.filter(t => t.league && t.league.startsWith(currentLeague));

  const dir = sortAsc ? 1 : -1;
  return [...filtered].sort((a, b) => {
    const av = a[sortKey], bv = b[sortKey];
    if (av == null && bv == null) return 0;
    if (av == null) return 1;
    if (bv == null) return -1;
    if (typeof av === "string") return dir * av.localeCompare(bv);
    return dir * (av - bv);
  });
}

function renderTeamList() {
  const mount = document.getElementById("teams-list-mount");
  if (!mount) return;

  const teams = sortedTeams();
  if (!teams.length) {
    mount.innerHTML = `<div class="empty-state">No teams found. Make sure the teams table is populated (run pulls/pull_teams.py).</div>`;
    return;
  }

  const cols = [
    { key: "name",          label: "Team",         align: "left" },
    { key: "stuff_plus",    label: "Stuff+",        align: "right" },
    { key: "tunneling_plus",label: "Tunneling+",    align: "right" },
    { key: "location_plus", label: "Location+",     align: "right" },
    { key: "deception_plus",label: "Deception+",    align: "right" },
  ];

  const thCells = cols.map(c => {
    const active = c.key === sortKey ? `sorted${sortAsc ? "" : " asc"}` : "";
    return `<th class="${active}" data-key="${c.key}" style="text-align:${c.align}">${c.label}</th>`;
  }).join("");

  // Group by division
  const divOrder = [
    "AL East","AL Central","AL West",
    "NL East","NL Central","NL West"
  ];
  const byDiv = {};
  for (const t of teams) {
    const d = t.division || "Unknown";
    if (!byDiv[d]) byDiv[d] = [];
    byDiv[d].push(t);
  }

  const divKeys = divOrder.filter(d => byDiv[d]) // show in standard order
    .concat(Object.keys(byDiv).filter(d => !divOrder.includes(d)));

  let html = `<table class="teams-table data-table">
    <thead><tr>${thCells}</tr></thead>
    <tbody>`;

  for (const div of divKeys) {
    if ((currentLeague !== "ALL") && !div.startsWith(currentLeague)) continue;
    const divTeams = byDiv[div] || [];
    if (!divTeams.length) continue;
    html += `<tr><td colspan="${cols.length}" style="background:var(--surface-alt);padding:4px 10px;">
      <span class="division-title" style="font-size:10.5px;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;color:var(--text-muted);">${div}</span>
    </td></tr>`;
    for (const t of divTeams) {
      html += `<tr data-team-id="${t.team_id}" class="team-row">
        <td>
          <div class="team-row-name">
            <img src="${logoUrl(t.team_id)}" alt="" onerror="this.style.visibility='hidden'" />
            <span>${t.name}</span>
            <span style="font-size:11px;color:var(--text-muted);margin-left:2px;">${t.abbreviation || ""}</span>
          </div>
        </td>
        <td>${fmtScore(t.stuff_plus)}</td>
        <td>${fmtScore(t.tunneling_plus)}</td>
        <td>${fmtScore(t.location_plus)}</td>
        <td>${fmtScore(t.deception_plus)}</td>
      </tr>`;
    }
  }

  html += `</tbody></table>
  <div style="font-size:11px;color:var(--text-muted);margin-top:8px;">
    Model scores are team averages across qualified pitchers. Deception+ uses 2025 data (single-season model).
    Click any team row to view roster and stats.
  </div>`;

  mount.innerHTML = html;

  // Sort click
  mount.querySelectorAll("th[data-key]").forEach(th => {
    th.addEventListener("click", () => {
      const k = th.dataset.key;
      if (sortKey === k) sortAsc = !sortAsc;
      else { sortKey = k; sortAsc = k === "name"; }
      renderTeamList();
    });
  });

  // Row click -> detail
  mount.querySelectorAll("tr.team-row").forEach(row => {
    row.addEventListener("click", () => openTeamDetail(Number(row.dataset.teamId)));
  });
}

// ---------------------------------------------------------------------------
// Team detail panel
// ---------------------------------------------------------------------------

let detailState = { tab: "roster" };

function openTeamDetail(teamId) {
  const team = allTeams.find(t => t.team_id === teamId);
  if (!team) return;

  document.getElementById("teams-list-mount").style.display = "none";
  const panel = document.getElementById("team-detail");
  panel.classList.add("open");

  panel.innerHTML = `
    <button class="detail-back" id="detail-back-btn">Back to Teams</button>
    <div class="detail-header">
      <img src="${logoUrl(teamId)}" alt="${team.name}" onerror="this.style.visibility='hidden'" />
      <div>
        <div class="detail-name">${team.name}</div>
        <div class="detail-sub">${team.division || ""}</div>
      </div>
    </div>
    <div class="model-scores-row">
      ${modelBadge("Stuff+", team.stuff_plus)}
      ${modelBadge("Tunneling+", team.tunneling_plus)}
      ${modelBadge("Location+", team.location_plus)}
      ${modelBadge("Deception+", team.deception_plus)}
    </div>
    <a class="playoff-link" href="playoffs-roster.html">View Projected Playoff Roster →</a>
    <div class="detail-tabs" id="detail-tabs">
      <button class="detail-tab active" data-tab="roster">Active Roster</button>
      <button class="detail-tab" data-tab="depth">Depth Chart</button>
      <button class="detail-tab" data-tab="stats">Team Stats</button>
    </div>
    <div id="tab-roster" class="tab-pane active"><div class="empty-state">Loading...</div></div>
    <div id="tab-depth" class="tab-pane"><div class="empty-state">Loading...</div></div>
    <div id="tab-stats" class="tab-pane"><div class="empty-state">Loading...</div></div>
  `;

  document.getElementById("detail-back-btn").addEventListener("click", closeTeamDetail);

  document.getElementById("detail-tabs").addEventListener("click", e => {
    const btn = e.target.closest(".detail-tab");
    if (!btn) return;
    document.querySelectorAll(".detail-tab").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    const pane = document.getElementById(`tab-${btn.dataset.tab}`);
    if (pane) pane.classList.add("active");
  });

  // Load all three tabs in parallel
  loadRoster(teamId, "active", "tab-roster");
  loadRoster(teamId, "depthChart", "tab-depth");
  loadStats(teamId, currentSeason);
}

function modelBadge(label, value) {
  const cls = scoreClass(value);
  const val = value != null ? `<div class="model-badge-value ${cls}">${value.toFixed(1)}</div>` : `<div class="model-badge-value score-na">—</div>`;
  return `<div class="model-badge">${val}<div class="model-badge-label">${label}</div></div>`;
}

function closeTeamDetail() {
  document.getElementById("team-detail").classList.remove("open");
  document.getElementById("team-detail").innerHTML = "";
  document.getElementById("teams-list-mount").style.display = "";
}

async function loadRoster(teamId, rosterType, paneId) {
  const pane = document.getElementById(paneId);
  if (!pane) return;
  try {
    const data = await getJSON(`/api/teams/${teamId}/roster?roster_type=${rosterType}`);
    if (!data.length) {
      pane.innerHTML = `<div class="empty-state">No roster data cached. Run pulls/refresh_mlb_stats.py.</div>`;
      return;
    }
    const pulledAt = data[0].pulled_at ? `<p class="pulled-at-note">Pulled: ${data[0].pulled_at.slice(0,19)} UTC</p>` : "";
    // Group by position
    const byPos = {};
    for (const p of data) {
      const pos = p.position || "?";
      if (!byPos[pos]) byPos[pos] = [];
      byPos[pos].push(p);
    }
    let rows = "";
    for (const pos of Object.keys(byPos).sort()) {
      for (const p of byPos[pos]) {
        const pid = p.player_id;
        const nameCell = pid
          ? `<a class="roster-player-link" href="index.html?player_id=${pid}&role=pitcher">${p.player_name || "—"}</a>`
          : (p.player_name || "—");
        const ss = p.season_stats;
        const ip  = ss?.ip  != null ? ss.ip  : "—";
        const era = ss?.era != null ? ss.era.toFixed(2) : "—";
        const k   = ss?.k   != null ? ss.k   : "—";
        const bb  = ss?.bb  != null ? ss.bb  : "—";
        rows += `<tr>
          <td>${p.jersey_number || "—"}</td>
          <td>${nameCell}</td>
          <td>${pos}</td>
          <td class="roster-num">${ip}</td>
          <td class="roster-num">${era}</td>
          <td class="roster-num">${k}</td>
          <td class="roster-num">${bb}</td>
        </tr>`;
      }
    }
    pane.innerHTML = `
      <div class="roster-table-wrap">
        <table class="roster-table data-table">
          <thead><tr><th>#</th><th>Player</th><th>Pos</th><th>IP</th><th>ERA</th><th>K</th><th>BB</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
      ${pulledAt}`;
  } catch (err) {
    pane.innerHTML = `<div class="empty-state">Could not load roster: ${err.message}</div>`;
  }
}

async function loadStats(teamId, season) {
  const pane = document.getElementById("tab-stats");
  if (!pane) return;
  try {
    const d = await getJSON(`/api/teams/${teamId}/stats?season=${season}`);
    let html = "";

    if (d.official) {
      const o = d.official;
      html += `<h3 style="margin-top:0">Official MLB Stats (${o.snapshot_season || season})</h3>
        <div class="stats-grid">
          ${statBox("OPS", o.ops != null ? o.ops.toFixed(3) : "—")}
          ${statBox("ERA", o.era != null ? o.era.toFixed(2) : "—")}
          ${statBox("Bullpen ERA", o.bullpen_era != null ? o.bullpen_era.toFixed(2) : "—")}
          ${statBox("FIP", o.fip != null ? o.fip.toFixed(2) : "—")}
          ${statBox("OPS (L30)", o.ops_30d != null ? o.ops_30d.toFixed(3) : "—")}
          ${statBox("ERA (L30)", o.era_30d != null ? o.era_30d.toFixed(2) : "—")}
        </div>
        ${o.pulled_at ? `<p class="pulled-at-note">Snapshot pulled: ${o.pulled_at.slice(0,19)} UTC</p>` : ""}
      `;
    } else {
      html += `<p class="empty-state" style="margin:0 0 12px">No official stats snapshot. Run pulls/pull_team_stats.py.</p>`;
    }

    if (d.statcast_pitching && d.statcast_pitching.total_pitches) {
      const sp = d.statcast_pitching;
      html += `<h3 style="margin-top:16px">Statcast Pitching (${season}, from pitch data)</h3>
        <div class="stats-grid">
          ${statBox("Pitchers", sp.n_pitchers || "—")}
          ${statBox("Total Pitches", sp.total_pitches ? sp.total_pitches.toLocaleString() : "—")}
          ${statBox("Avg Velo", sp.avg_velo != null ? sp.avg_velo.toFixed(1) + " mph" : "—")}
          ${statBox("Avg Spin", sp.avg_spin != null ? Math.round(sp.avg_spin) + " rpm" : "—")}
          ${statBox("Whiff Rate", sp.whiff_rate != null ? (sp.whiff_rate * 100).toFixed(1) + "%" : "—")}
        </div>`;
    }

    if (!html) html = `<div class="empty-state">No stats available.</div>`;
    pane.innerHTML = html;
  } catch (err) {
    pane.innerHTML = `<div class="empty-state">Could not load stats: ${err.message}</div>`;
  }
}

function statBox(label, value) {
  return `<div class="stat-box"><div class="stat-box-label">${label}</div><div class="stat-box-value">${value}</div></div>`;
}

// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------

async function loadTeams(season) {
  setPageLoading(true);
  try {
    allTeams = await getJSON(`/api/teams?season=${season}`);
  } catch (err) {
    document.getElementById("teams-list-mount").innerHTML =
      `<div class="empty-state">Failed to load teams: ${err.message}</div>`;
    return;
  } finally {
    setPageLoading(false);
  }
  renderTeamList();
}

document.addEventListener("DOMContentLoaded", () => {
  // League toggle
  document.getElementById("league-toggle").addEventListener("click", e => {
    const btn = e.target.closest("button[data-value]");
    if (!btn) return;
    document.querySelectorAll("#league-toggle button").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    currentLeague = btn.dataset.value;
    renderTeamList();
  });

  window.onSeasonChange = (season) => {
    currentSeason = season;
    loadTeams(season);
  };

  loadTeams(currentSeason);
});
