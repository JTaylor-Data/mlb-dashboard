// Splits/Radar are grouped under Player and Stuff+/Tunneling+/Deception+
// under Models via a secondary sub-nav strip (see initSubNav below) rather
// than their own top-level entries - real pages, not a single-page JS
// merge (each already has its own independent, working player-search/
// season-select wiring; merging them into one script would collide on
// shared element IDs and global state). PLAYER_GROUP/MODELS_GROUP double
// as the sub-nav's own item lists so there's one source of truth for each
// group's member pages.
const PLAYER_GROUP = [
  { href: "index.html", label: "Overview" },
  { href: "splits.html", label: "Splits" },
  { href: "outings.html", label: "Outing" },
  { href: "trends.html", label: "Trends" },
  { href: "career.html", label: "Career" },
  { href: "compare.html", label: "Compare" },
];
const MODELS_GROUP = [
  { href: "about.html", label: "About" },
  { href: "stuff-model.html", label: "Stuff+" },
  { href: "tunneling-model.html", label: "Tunneling+" },
  { href: "tunnel3d.html", label: "Tunneling+ 3D" },
  { href: "deception-model.html", label: "Deception+" },
  { href: "location-model.html", label: "Location+" },
];
const PLAYOFFS_GROUP = [
  { href: "playoffs-bracket.html", label: "Bracket" },
  { href: "playoffs.html", label: "WC Race" },
  { href: "playoffs-roster.html", label: "Rosters" },
];

const GAMES_GROUP = [
  { href: "probable-pitchers.html", label: "Probable Pitchers" },
];

const NAV_LINKS = [
  { href: "probable-pitchers.html", label: "Games", group: GAMES_GROUP },
  { href: "index.html", label: "Player", group: PLAYER_GROUP },
  { href: "standings.html", label: "Standings" },
  { href: "news.html", label: "News" },
  { href: "top-performers.html", label: "Top Performers" },
  { href: "playoffs-bracket.html", label: "Playoffs", group: PLAYOFFS_GROUP },
  { href: "prospects.html", label: "Prospects" },
  { href: "about.html", label: "About", group: MODELS_GROUP },
];

function initNav() {
  const mount = document.getElementById("nav");
  if (!mount) return;

  const current = location.pathname.split("/").pop() || "index.html";
  const isActive = (l) => l.href === current || (l.group || []).some((g) => g.href === current);
  const links = NAV_LINKS.map(
    (l) => `<a href="${l.href}" class="${isActive(l) ? "active" : ""}">${l.label}</a>`
  ).join("");

  mount.innerHTML = `
    <div class="nav-inner">
      <div class="nav-brand">MLB Dashboard</div>
      <div class="nav-links">${links}</div>
      <div class="nav-meta">
        <span id="nav-data-as-of">Data as of &mdash;</span>
        <button class="nav-refresh" id="nav-refresh-btn" type="button">Refresh data</button>
      </div>
    </div>
  `;

  getJSON("/api/meta")
    .then((meta) => {
      const el = document.getElementById("nav-data-as-of");
      if (el && meta.latest_game_date) el.textContent = `Data as of ${meta.latest_game_date}`;
    })
    .catch(() => {});

  const btn = document.getElementById("nav-refresh-btn");
  btn.addEventListener("click", async () => {
    // Previously this was try/finally with NO catch - if anything in the
    // chain threw (a dropped request, a bad response, whatever), the button
    // silently reset to normal with zero indication anything had gone
    // wrong. From the outside that looks exactly like "I clicked it and
    // nothing happened" - which is exactly what got reported. Every path
    // below now either succeeds visibly or fails visibly; nothing is
    // silently swallowed.
    btn.disabled = true;
    btn.textContent = "Starting...";
    try {
      const { started } = await postJSON("/api/admin/pull-data");
      if (!started) {
        btn.textContent = "Already running - watching it...";
      }
      const finalStatus = await pollPullStatus(btn);
      if (finalStatus.state === "error") {
        btn.textContent = "Failed - click to retry";
        console.error("Refresh failed:", finalStatus.detail);
        return; // leave the failure message up; don't pretend it succeeded
      }
      const meta = await getJSON("/api/meta");
      const el = document.getElementById("nav-data-as-of");
      if (el && meta.latest_game_date) el.textContent = `Data as of ${meta.latest_game_date}`;
      if (typeof window.onDataRefresh === "function") {
        await window.onDataRefresh();
      } else {
        location.reload();
        return; // page is navigating away - don't touch btn below
      }
      btn.textContent = "Refresh data";
    } catch (err) {
      console.error("Refresh data button failed:", err);
      btn.textContent = "Error - click to retry";
    } finally {
      btn.disabled = false;
    }
  });
}

async function pollPullStatus(btn) {
  // Actually pulls new Statcast/schedule data (not just an app-cache clear)
  // - this can take anywhere from a few seconds to a couple minutes
  // depending on how many days behind the DB is, so poll rather than block.
  // Returns the final status object (caller decides what "error" means for
  // the button) instead of swallowing it into a generic "Done".
  const maxAttempts = 400; // ~10 minutes at 1.5s/poll - a hard ceiling so a
                            // stuck server-side state can never hang this
                            // forever with no way out for the user.
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    const status = await getJSON("/api/admin/pull-status");
    if (status.state === "running") {
      btn.textContent = status.detail || "Pulling...";
    } else {
      return status;
    }
    await new Promise((r) => setTimeout(r, 1500));
  }
  return { state: "error", detail: "Timed out waiting for the refresh to finish - check the server log." };
}

// Renders the second-level tab strip on any page that belongs to a
// PLAYER_GROUP/MODELS_GROUP/PLAYOFFS_GROUP page (mounts into #subnav if
// present - pages outside a group simply don't have that div, so this
// silently no-ops). Real navigation between real pages, not a JS view-swap
// - see the PLAYER_GROUP/MODELS_GROUP comment above for why.
function initSubNav() {
  const mount = document.getElementById("subnav");
  if (!mount) return;
  const current = location.pathname.split("/").pop() || "index.html";
  const group = [PLAYER_GROUP, MODELS_GROUP, PLAYOFFS_GROUP, GAMES_GROUP].find((g) => g.some((l) => l.href === current));
  if (!group) return;
  const links = group
    .map((l) => `<a href="${l.href}" class="${l.href === current ? "active" : ""}">${l.label}</a>`)
    .join("");
  mount.innerHTML = `<div class="subnav-inner">${links}</div>`;
}

// Team card popup - click ANY team logo anywhere on the site. Wired here
// (not a per-page script) specifically so it works everywhere without
// touching every page that happens to render a team logo: event
// delegation on document.body catches a click on any <img> whose src
// matches the mlbstatic team-logos URL pattern every page already uses
// (standings, prospects, top performers, the player card, ...), and the
// team_id is parsed straight out of that URL - no new data-* attribute
// needed on any existing logo <img>.
const TEAM_LOGO_ID_RE = /team-logos\/(\d+)\.svg/;

function teamCardPanelHtml() {
  return `
    <div id="team-card-backdrop"></div>
    <div id="team-card-panel">
      <button id="team-card-close" type="button" aria-label="Close">&times;</button>
      <div id="team-card-body"></div>
    </div>
  `;
}

// Dashes for anything unreported (no value, no rank) - never a fallback
// string like "season only", matching the site's convention elsewhere of
// "-" for missing data rather than explanatory filler text.
//
// `metricKey` (when given - only for metrics team_metric_rankings() in
// analytics/team_card.py's RANKED_METRICS actually supports) adds a small
// arrow that expands a full 30-team ranked list fetched on demand from
// /api/team/metric-rankings/{metric} - click again to collapse. Fetched
// once per metric per card open (cached in expandCache below), not
// re-fetched on every toggle.
function metricRowHtml(label, m, fmt, metricKey, season) {
  if (!m) return "";
  const rankColor = m.rank == null ? "var(--text-muted)" : m.rank <= 10 ? "#1a7f37" : m.rank >= 21 ? "#c9302c" : "var(--text-muted)";
  const rolling = m.rolling_30d != null
    ? `<span class="tc-rolling">${fmt(m.rolling_30d)} <span class="tc-rolling-label">L30</span></span>`
    : `<span class="tc-rolling tc-rolling-na">-</span>`;
  const valueStr = m.value != null ? fmt(m.value) : "-";
  const rankStr = m.rank != null ? `#${m.rank} <span class="tc-of">of ${m.n_teams}</span>` : "-";
  const fmtName = Object.keys(FMT_BY_NAME).find((k) => FMT_BY_NAME[k] === fmt) || "count";
  const expandBtn = metricKey
    ? `<button class="tc-expand-btn" type="button" data-metric="${metricKey}" data-fmt="${fmtName}" data-season="${season}" aria-label="Show all teams">&#9662;</button>`
    : "";
  return `
    <div class="tc-metric-row">
      <div class="tc-metric-label">${expandBtn}${label}</div>
      <div class="tc-metric-value">${valueStr}</div>
      <div class="tc-metric-rank" style="color:${rankColor};">${rankStr}</div>
      ${rolling}
    </div>
    ${metricKey ? `<div class="tc-expand-body" id="tc-expand-${metricKey}" style="display:none;"></div>` : ""}
  `;
}

const fmtRate = (v) => (v < 10 ? v.toFixed(3).replace(/^0\./, ".") : v.toFixed(2));
// ERA/FIP's own standard 2-decimal convention ("3.45"), distinct from
// fmtRate above (built for sub-1 rate stats like OPS - "3.45" was
// previously going through that one too and rendering as "3.450", a real
// 3-vs-2-decimal mismatch against how ERA/FIP are normally written).
const fmtEra = (v) => v.toFixed(2);
const fmtRuns = (v) => (v >= 0 ? "+" : "") + v.toFixed(1);
const fmtPct1 = (v) => (v * 100).toFixed(1) + "%";
const fmtWar = (v) => (v >= 0 ? "+" : "") + v.toFixed(1);
const fmtCount = (v) => String(Math.round(v));
const fmtParkFactor = (v) => v.toFixed(1);
const FMT_BY_NAME = { rate: fmtRate, era: fmtEra, runs: fmtRuns, war: fmtWar, pct1: fmtPct1, count: fmtCount, park: fmtParkFactor };

function renderTeamCard(data) {
  const logo = `https://www.mlbstatic.com/team-logos/${data.team_id}.svg`;
  const sos = data.metrics.remaining_sos;
  const sosNote = sos && sos.n_games_remaining != null
    ? `<div class="tc-seq-note" style="margin-top:2px;">${sos.n_games_remaining} games remaining &middot; rank 30 = hardest schedule &middot; updates as opponents' records do</div>`
    : "";
  document.getElementById("team-card-body").innerHTML = `
    <div class="tc-header">
      <img class="tc-logo" src="${logo}" onerror="this.style.visibility='hidden'" />
      <div class="tc-name">${data.name}</div>
    </div>
    <div class="tc-synopsis">${data.synopsis || ""}</div>
    <div class="tc-metrics">
      <div class="tc-metrics-header">
        <div></div><div>${data.season}</div><div>Rank</div><div>Trailing 30d</div>
      </div>
      ${metricRowHtml("OPS", data.metrics.ops, fmtRate, "ops", data.season)}
      ${metricRowHtml("ERA", data.metrics.era, fmtEra, "era", data.season)}
      ${metricRowHtml("Bullpen ERA", data.metrics.bullpen_era, fmtEra, "bullpen_era", data.season)}
      ${metricRowHtml("FIP", data.metrics.fip, fmtEra, "fip", data.season)}
      ${metricRowHtml("Baserunning", data.metrics.baserunning_value, fmtRuns, "baserunning_value", data.season)}
      ${metricRowHtml("Team WAR", data.metrics.war, fmtWar, "war", data.season)}
      ${metricRowHtml("Players on IL", data.metrics.il_count, fmtCount, "il_count", data.season)}
      ${metricRowHtml("Park Factor", data.metrics.park_factor, fmtParkFactor, "park_factor", data.season)}
      ${metricRowHtml("Remaining SOS", sos, fmtPct1)}
    </div>
    ${sosNote}
  `;
}

// One fetch per metric per card-open, cached here so re-expanding the same
// metric (or a different team's card, since this is keyed by metric+season
// only) doesn't re-hit the endpoint.
const metricRankingsCache = {};

async function toggleMetricExpand(btn) {
  const metric = btn.dataset.metric;
  const fmt = FMT_BY_NAME[btn.dataset.fmt] || fmtCount;
  const body = document.getElementById(`tc-expand-${metric}`);
  if (!body) return;
  const isOpen = body.style.display !== "none";
  if (isOpen) {
    body.style.display = "none";
    btn.classList.remove("open");
    return;
  }
  btn.classList.add("open");
  body.style.display = "block";
  const season = btn.dataset.season || 2026;
  const cacheKey = `${metric}-${season}`;
  if (!metricRankingsCache[cacheKey]) {
    body.innerHTML = `<div class="tc-expand-loading">Loading...</div>`;
    try {
      metricRankingsCache[cacheKey] = await getJSON(`/api/team/metric-rankings/${metric}?season=${season}`);
    } catch (err) {
      body.innerHTML = `<div class="tc-expand-loading">Couldn't load full list.</div>`;
      return;
    }
  }
  const teams = metricRankingsCache[cacheKey].teams;
  body.innerHTML = teams.length
    ? teams.map((t) => `
        <div class="tc-expand-row">
          <div class="tc-expand-rank">#${t.rank}</div>
          <img class="tc-expand-logo" src="https://www.mlbstatic.com/team-logos/${t.team_id}.svg" onerror="this.style.visibility='hidden'" />
          <div class="tc-expand-team">${t.abbreviation}</div>
          <div class="tc-expand-value">${fmt(t.value)}</div>
        </div>
      `).join("")
    : `<div class="tc-expand-loading">No data.</div>`;
}

let teamCardRequestId = 0;
async function openTeamCard(teamId) {
  const panel = document.getElementById("team-card-panel");
  const backdrop = document.getElementById("team-card-backdrop");
  panel.classList.add("open");
  backdrop.classList.add("open");
  document.getElementById("team-card-body").innerHTML = `<div class="empty-state">Loading...</div>`;
  const requestId = ++teamCardRequestId;
  try {
    const data = await getJSON(`/api/team/${teamId}/card`);
    if (requestId !== teamCardRequestId) return; // a later click won the race
    renderTeamCard(data);
  } catch (err) {
    if (requestId !== teamCardRequestId) return;
    document.getElementById("team-card-body").innerHTML =
      `<div class="empty-state">No stats snapshot for this team yet - run pull_team_stats.py.</div>`;
  }
}

function closeTeamCard() {
  document.getElementById("team-card-panel").classList.remove("open");
  document.getElementById("team-card-backdrop").classList.remove("open");
}

function initTeamCard() {
  document.body.insertAdjacentHTML("beforeend", teamCardPanelHtml());
  document.getElementById("team-card-close").addEventListener("click", closeTeamCard);
  document.getElementById("team-card-backdrop").addEventListener("click", closeTeamCard);
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeTeamCard(); });
  // Delegated from the body mount (not each button) since renderTeamCard()
  // replaces #team-card-body's innerHTML wholesale on every open/re-render -
  // a listener on an individual button would be destroyed with it.
  document.getElementById("team-card-body").addEventListener("click", (e) => {
    const btn = e.target.closest(".tc-expand-btn");
    if (btn) toggleMetricExpand(btn);
  });
  document.body.addEventListener("click", (e) => {
    // Deliberately narrow - ONLY the logo image itself, or an element
    // explicitly marked .team-card-link (the team NAME text, added at each
    // call site that shows a team name - see standings.js/prospects.js).
    // A previous version walked up to 6 ancestors looking for anything that
    // merely CONTAINED a logo, which made entire unrelated rows (e.g. every
    // top-performers row, every farm-rankings row) into accidental
    // team-card triggers - real reported annoyance ("it keeps popping up
    // when I don't want it"), especially since some of those rows already
    // have their OWN, different click behavior (prospects.js's farm-row ->
    // switchToTeamView, top-performers' player-name link).
    if (e.target.closest("a, button, input, select")) return;
    const logo = e.target.closest('img.team-logo, img[src*="team-logos"]');
    const nameLink = e.target.closest(".team-card-link");
    const el = logo || nameLink;
    if (!el) return;
    let teamId = null;
    if (logo) {
      const m = (logo.src || "").match(TEAM_LOGO_ID_RE);
      if (m) teamId = Number(m[1]);
    } else if (nameLink.dataset.teamcardId) {
      teamId = Number(nameLink.dataset.teamcardId);
    }
    if (teamId === null || Number.isNaN(teamId)) return;
    e.preventDefault();
    openTeamCard(teamId);
  });
}

document.addEventListener("DOMContentLoaded", () => {
  initNav();
  initSubNav();
  initTeamCard();
});
