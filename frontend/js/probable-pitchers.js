// Probable Pitchers — full-width one-game-per-row layout

// Convert an ET time string ("7:05 PM") + game date ("2026-09-03") to the
// viewer's local time with timezone abbreviation, e.g. "7:05 PM EDT" when
// already in Eastern, or "6:05 PM CDT" when in Central.
// Falls back to "7:05 PM ET" on any parse failure (malformed string, old
// Safari that can't parse the ISO offset form).
function formatGameTime(gameDate, timeStr) {
  if (!timeStr) return null;
  return timeStr; // already includes "ET" suffix from mlb_stats_api
}

function logoUrl(teamId) {
  return `https://www.mlbstatic.com/team-logos/${teamId}.svg`;
}

function headshotUrl(playerId) {
  return `https://img.mlbstatic.com/mlb-photos/image/upload/d_people:generic:headshot:67:current.png/w_213,q_auto:best/v1/people/${playerId}/headshot/67/current`;
}


function scoreClass(v) {
  if (v == null) return "";
  return v >= 105 ? "good" : v <= 95 ? "bad" : "";
}

function scoreBlock(label, value) {
  if (value == null) return "";
  const cls = scoreClass(value);
  return `<div class="score-block ${cls}">
    <div class="score-block-label">${label}</div>
    <div class="score-block-value">${value.toFixed(0)}</div>
  </div>`;
}

function statBlock(label, value) {
  if (value == null) return "";
  return `<div class="stat-block">
    <div class="stat-block-label">${label}</div>
    <div class="stat-block-value">${value}</div>
  </div>`;
}

function recentSummaryHtml(r) {
  if (!r || (Array.isArray(r) && !r.length)) {
    return `<div class="recent-empty">No starts this season</div>`;
  }
  let games, ip, era, k, bb, hr;
  if (Array.isArray(r)) {
    games = r.length;
    k = r.reduce((s, g) => s + (g.k || 0), 0);
    bb = r.reduce((s, g) => s + (g.bb || 0), 0);
    hr = r.reduce((s, g) => s + (g.hr || 0), 0);
    ip = null; era = null;
  } else {
    games = r.games; ip = r.ip; era = r.era != null ? r.era.toFixed(2) : null;
    k = r.k; bb = r.bb; hr = r.hr;
  }
  return `<div class="recent-row">
    <span class="recent-label">LAST ${games} STARTS</span>
    <div class="recent-stats">
      ${statBlock("IP", ip != null ? ip : "—")}
      ${statBlock("ERA", era != null ? era : "—")}
      ${statBlock("K", k)}
      ${statBlock("BB", bb)}
      ${statBlock("HR", hr != null ? hr : "—")}
    </div>
  </div>`;
}

// Score a pitcher for "Matchup of the Day" (lower ERA + more K + fewer BB = higher)
function pitcherMatchupScore(pitcher) {
  const r = pitcher?.recent_starts;
  if (!r) return -Infinity;
  const stats = Array.isArray(r)
    ? { k: r.reduce((s,g)=>s+(g.k||0),0), bb: r.reduce((s,g)=>s+(g.bb||0),0), era: null }
    : r;
  let score = 0;
  if (stats.era != null) score += (5 - stats.era) * 3;  // lower ERA better
  if (stats.k != null)  score += stats.k * 0.5;
  if (stats.bb != null) score -= stats.bb * 1.5;
  return score;
}

function findMatchupOfDay(games) {
  // Pick game where sum of both pitchers' scores is highest
  let best = null, bestScore = -Infinity;
  for (const g of games) {
    const s = pitcherMatchupScore(g.away_probable) + pitcherMatchupScore(g.home_probable);
    if (s > bestScore && g.away_probable && g.home_probable) {
      bestScore = s;
      best = g;
    }
  }
  return best;
}

function pitcherCardHtml(pitcher, abbr, gamePk, gameDate, compact = false) {
  if (!pitcher) {
    return `<div class="pitcher-card tbd-card">
      <div class="pitcher-role-label">${abbr} STARTER</div>
      <div class="tbd-name">TBD</div>
    </div>`;
  }
  const nameLink = pitcher.player_id
    ? `<a class="pitcher-name-link" href="index.html?player_id=${pitcher.player_id}&role=pitcher">${pitcher.name || "Unknown"}</a>`
    : `<span class="pitcher-name-link no-link">${pitcher.name || "Unknown"}</span>`;
  const s = pitcher.scores || {};
  const scores = [
    scoreBlock("STUFF+", s.stuff_plus),
    scoreBlock("TUN+", s.tunneling_plus),
    scoreBlock("LOC+", s.location_plus),
    scoreBlock("DEC+", s.deception_plus),
  ].filter(Boolean).join("");
  const photo = pitcher.player_id
    ? `<img class="pitcher-headshot" src="${headshotUrl(pitcher.player_id)}" alt="${pitcher.name}" onerror="this.style.display='none'" loading="lazy" />`
    : `<div class="pitcher-headshot-placeholder"></div>`;
  return `<div class="pitcher-card">
    <div class="pitcher-role-label">${abbr} STARTER</div>
    <div class="pitcher-card-inner">
      ${photo}
      <div class="pitcher-card-info">
        <div class="pitcher-name-row">${nameLink}</div>
        ${scores ? `<div class="pitcher-scores-row">${scores}</div>` : ""}
        ${recentSummaryHtml(pitcher.recent_starts)}
      </div>
    </div>
  </div>`;
}

function matchupOfDayHtml(game, dateStr) {
  if (!game) return "";
  const inner = gameCardInnerHtml(game, dateStr);
  return `<div class="motd-card">
    <div class="motd-header">
      <span class="motd-crown">★</span>
      <span class="motd-title">MATCHUP OF THE DAY</span>
    </div>
    <div class="game-card-inner">${inner}</div>
  </div>`;
}

function gameCardInnerHtml(game, gameDate) {
  const home = game.home_team || {}, away = game.away_team || {};
  const homeId = home.team_id, awayId = away.team_id;
  const timeRow = game.game_time_et
    ? `<div class="game-time">${formatGameTime(gameDate, game.game_time_et)}</div>`
    : (game.status ? `<div class="game-time">${game.status}</div>` : "");

  return `
    <div class="matchup-col">
      <div class="matchup-logos-row">
        <div class="team-block">
          <img class="team-logo-lg" src="${logoUrl(awayId)}" alt="${away.abbreviation}" onerror="this.style.visibility='hidden'" />
        </div>
        <div class="at-sign">@</div>
        <div class="team-block">
          <img class="team-logo-lg" src="${logoUrl(homeId)}" alt="${home.abbreviation}" onerror="this.style.visibility='hidden'" />
        </div>
      </div>
      <div class="matchup-abbrs-row">
        <div class="team-abbr">${away.abbreviation || "???"}</div>
        <div class="team-abbr-spacer"></div>
        <div class="team-abbr">${home.abbreviation || "???"}</div>
      </div>
      <div class="matchup-cities-row">
        <div class="team-city">${(away.name||"").replace(away.abbreviation||"###","").trim()}</div>
        <div class="team-city-spacer"></div>
        <div class="team-city">${(home.name||"").replace(home.abbreviation||"###","").trim()}</div>
      </div>
      ${timeRow ? `<div class="matchup-time-row">${timeRow}</div>` : ""}
    </div>

    <div class="venue-col">
      ${game.venue_id ? `<img class="venue-photo" src="/img/stadiums/${game.venue_id}.jpg" alt="${game.venue}" onerror="this.style.display='none'" loading="lazy" />` : ""}
      <div class="venue-name-label">${game.venue || ""}</div>
    </div>

    <div class="pitchers-col">
      ${pitcherCardHtml(game.away_probable, away.abbreviation || "???", game.game_pk, gameDate)}
      <div class="pitchers-divider"></div>
      ${pitcherCardHtml(game.home_probable, home.abbreviation || "???", game.game_pk, gameDate)}
    </div>`;
}

function gameCardHtml(game, dateStr) {
  return `<div class="game-card"><div class="game-card-inner">${gameCardInnerHtml(game, dateStr)}</div></div>`;
}

function dateOffsetStr(base, daysAhead) {
  const d = new Date(base + "T12:00:00Z");
  d.setUTCDate(d.getUTCDate() + daysAhead);
  return d.toISOString().slice(0, 10);
}

function formatDateLabel(dateStr) {
  const d = new Date(dateStr + "T12:00:00Z");
  return d.toLocaleDateString("en-US", { weekday: "long", month: "long", day: "numeric", timeZone: "UTC" });
}

async function loadGames(startDate, windowDays = 7) {
  setPageLoading(true);
  const mount = document.getElementById("games-mount");
  const motdMount = document.getElementById("motd-mount");
  const note = document.getElementById("freshness-note");
  try {
    // Fetch all days in parallel
    const dates = Array.from({ length: windowDays }, (_, i) => dateOffsetStr(startDate, i));
    const results = await Promise.all(dates.map(d =>
      getJSON(`/api/probable-pitchers?date=${d}&season=2026`).catch(() => null)
    ));

    const latestPull = results.map(r => r && r.pulled_at).filter(Boolean).sort().pop();
    note.textContent = latestPull
      ? `Data pulled: ${latestPull.slice(0, 16).replace("T", " ")} UTC`
      : "";

    // MOTD from the first day that has games
    const firstResult = results.find(r => r && r.games && r.games.length);
    const motd = firstResult ? findMatchupOfDay(firstResult.games) : null;
    motdMount.innerHTML = motd ? matchupOfDayHtml(motd, dates[results.indexOf(firstResult)]) : "";

    let html = "";
    let anyGames = false;
    for (let i = 0; i < dates.length; i++) {
      const data = results[i];
      if (!data || !data.games || !data.games.length) continue;
      anyGames = true;
      const dateLabel = formatDateLabel(dates[i]);
      const dayGames = motd && i === results.indexOf(firstResult)
        ? data.games.filter(g => g.game_pk !== motd.game_pk)
        : data.games;
      html += `<div class="date-section-label">${dateLabel}</div>`;
      html += dayGames.map(g => gameCardHtml(g, dates[i])).join("");
    }
    if (!anyGames) {
      mount.innerHTML = `<div class="no-games">No upcoming games found in the next ${windowDays} days.</div>`;
    } else {
      mount.innerHTML = html;
    }
  } catch (err) {
    mount.innerHTML = `<div class="empty-state">Error: ${err.message}</div>`;
  } finally {
    setPageLoading(false);
  }
}

document.addEventListener("DOMContentLoaded", () => {
  const today = new Date().toISOString().slice(0, 10);
  loadGames(today);
});
