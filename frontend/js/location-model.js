/*
 * Location+ v3 model-internals page.
 *
 * Architecture: five LightGBM classifiers (xSwing, xCalledStrike, xWhiff,
 * xFoul, BIP×EV) + COUNT-SPECIFIC run-value table → per-pitch xRV →
 * pitcher-season plus-scale (100 = league mean, 10 pts = 1 SD).
 *
 * Key v3 changes vs v1/v2:
 *  - Count-specific run-value table (7.8x variation in called-strike value
 *    across counts; v2 defect correction).
 *  - S1 baseline adjustment adopted (R2_mix=0.338 ≥ 0.10).
 *  - No shrinkage (split-half r_SB at 300 = 0.815 ≥ 0.70 threshold).
 *  - 2020 excluded from training; scored by averaging 2019/2021 predictions.
 *  - Qualification threshold raised to 300 pitches (111 for 2020).
 *
 * API responses:
 *  GET /api/location/report?season=N  → { scoring_config, gates, validation, leaderboard }
 *  GET /api/location/pitcher/:id?season=N → { score, n_pitches, percentile, ... }
 *  GET /api/location/meta → { version, features, scoring_config, ... }
 */

// ──────────────────────────────────────────────────────────── leaderboard (same UI as Stuff+)
function renderLbList(mountId, entries, onRowClick) {
  const el = document.getElementById(mountId);
  if (!el) return;
  if (!entries || !entries.length) {
    el.innerHTML = `<div class="muted" style="font-size:13px;">No qualified pitchers for this season.</div>`;
    return;
  }
  el.innerHTML = entries.map((e, i) => {
    const color = e.score >= 100 ? "#1a7f37" : "#c9302c";
    const team  = e.team_abbrev ? `<span class="muted"> (${e.team_abbrev})</span>` : "";
    const flag  = e.season_flag ? ` <span class="lc-asterisk" title="${e.season_flag}">*</span>` : "";
    return `<div class="lc-lb-row" tabindex="0" role="button"
        aria-label="${e.name}, Location+ ${e.score}"
        data-pitcher-id="${e.pitcher_id}" data-name="${e.name || ""}">
      <div class="lc-lb-rank">${i + 1}</div>
      <div class="lc-lb-name">${e.name || "—"}${team}${flag}</div>
      <div class="lc-lb-n">${(e.total_pitches || 0).toLocaleString()} pitches</div>
      <div class="lc-lb-val" style="color:${color}">${e.score.toFixed(0)}</div>
    </div>`;
  }).join("");

  el.querySelectorAll(".lc-lb-row").forEach(row => {
    const handler = () => {
      const pid  = Number(row.dataset.pitcherId);
      const name = row.dataset.name;
      if (onRowClick) onRowClick(pid, name);
    };
    row.addEventListener("click", handler);
    row.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") handler(); });
  });
}

async function loadOverallLeaderboard(season, onRowClick) {
  const captionEl = document.getElementById("lc-lb-caption");
  try {
    const lb = await getJSON(`/api/location/leaderboard?season=${season}&n=10`);
    renderLbList("lc-lb-top", lb.top || [], onRowClick);
    renderLbList("lc-lb-bottom", lb.bottom || [], onRowClick);
    const min = lb.min_pitches || 300;
    if (captionEl) captionEl.textContent = `Minimum ${min} pitches to qualify. * = 2020, a 60-game season.`;
  } catch (err) {
    document.getElementById("lc-lb-top").innerHTML = `<div class="muted">Failed to load.</div>`;
    document.getElementById("lc-lb-bottom").innerHTML = "";
  }
}

function renderPitcherDetail(detail, name, season) {
  const mount = document.getElementById("lc-pitcher-mount");
  if (!mount) return;
  const pct = detail.percentile !== undefined
    ? ` — ${detail.percentile.toFixed(0)}th percentile` : "";
  const flag = detail.season_flag ? ` <span class="lc-note">(${detail.season_flag})</span>` : "";
  const note = detail.shap_note
    ? `<div class="lc-note" style="margin-top:8px;">${detail.shap_note}</div>` : "";
  mount.innerHTML = `
    <div class="lc-arsenal-head">
      <b>${name || `Pitcher #${detail.pitcher_id || ""}`}</b> — Location+ <b style="color:var(--forest-dark);">${typeof detail.score === "number" ? detail.score.toFixed(0) : detail.score}</b>${pct}
      (${(detail.n_pitches || 0).toLocaleString()} pitches, ${season}${flag})
    </div>
    ${note}
  `;
}

async function loadPitcherDetail(playerId, name, season) {
  const status = document.getElementById("lc-status");
  const loadingBar = document.getElementById("lc-loading");
  const mount = document.getElementById("lc-pitcher-mount");
  if (status) status.textContent = "";
  if (loadingBar) loadingBar.classList.add("active");
  if (mount) mount.innerHTML = "";
  try {
    const detail = await getJSON(`/api/location/pitcher/${playerId}?season=${season}`);
    renderPitcherDetail(detail, name, season);
  } catch (err) {
    if (status) status.textContent = "No Location+ data for this pitcher/season (needs 300+ pitches; 111 for 2020).";
  } finally {
    if (loadingBar) loadingBar.classList.remove("active");
  }
}

async function init() {
  let selectedPlayer = null;

  const seasonSelect = createSeasonSelect(document.getElementById("season-select"), () => {
    if (selectedPlayer) loadPitcherDetail(selectedPlayer.player_id, selectedPlayer.name, seasonSelect.value);
  });
  const defaultSeason = await seasonSelect.load();

  const search = createPlayerSearch(
    document.getElementById("player-search-input"),
    document.getElementById("player-search-dropdown"),
    (player) => {
      selectedPlayer = player;
      loadPitcherDetail(player.player_id, player.name, seasonSelect.value);
    },
    { roleFilter: "pitcher" }
  );
  await search.load(defaultSeason);

  // ---- Leaderboard (own season selector, independent of pitcher lookup)
  function onLbRowClick(pitcherId, name) {
    selectedPlayer = { player_id: pitcherId, name };
    search.setValue(selectedPlayer);
    seasonSelect.set(Number(lbSeasonSelect.value));
    loadPitcherDetail(pitcherId, name, Number(lbSeasonSelect.value));
    document.getElementById("lc-lookup-card").scrollIntoView({ behavior: "smooth" });
  }
  const lbSeasonSelect = createSeasonSelect(document.getElementById("lc-lb-season"), (s) => {
    loadOverallLeaderboard(s, onLbRowClick);
  });
  const lbDefaultSeason = await lbSeasonSelect.load();
  await loadOverallLeaderboard(lbDefaultSeason, onLbRowClick);

  const params = new URLSearchParams(location.search);
  const urlPitcherId = params.get("pitcher_id");
  const remembered = PlayerContext.getCurrent();
  if (urlPitcherId) {
    selectedPlayer = { player_id: Number(urlPitcherId), name: "" };
    loadPitcherDetail(selectedPlayer.player_id, "", Number(params.get("season")) || defaultSeason);
  } else if (remembered && remembered.role === "pitcher") {
    selectedPlayer = remembered;
    search.setValue(remembered);
    loadPitcherDetail(remembered.player_id, remembered.name, defaultSeason);
  }
}

document.addEventListener("DOMContentLoaded", init);
