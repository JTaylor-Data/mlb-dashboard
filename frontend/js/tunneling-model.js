/*
 * Tunneling+ model-internals page: a hand-built, pair-type-normalized
 * geometric composite (analytics/tunneling_pair_composite.py) - NOT a
 * trained model, so there is no SHAP, no CV, no lag slots here. Renders a
 * live, season-driven leaderboard and a per-pitcher per-pitch-type-pair
 * breakdown from /api/tunneling/v3/*, which is the SAME composite (and, via
 * the server's cached pair table, the same underlying computation) that
 * feeds the pair comparison table on each pitcher's own Overview page.
 */
function leaderboardRows(rows) {
  return rows
    .map((r) => `<tr><td>${r.name || "#" + r.pitcher_id}${r.team_abbrev ? ` <span class="muted">(${r.team_abbrev})</span>` : ""}</td><td>${r.score.toFixed(1)}</td><td>${r.total_pitches.toLocaleString()}</td></tr>`)
    .join("");
}

function renderLeaderboard(report) {
  const lb = report.leaderboard;
  const header = `<thead><tr><th>Pitcher</th><th>Tunneling+</th><th>Pitches</th></tr></thead>`;
  document.getElementById("tn-top-table").innerHTML = header + `<tbody>${leaderboardRows(lb.top)}</tbody>`;
  document.getElementById("tn-bottom-table").innerHTML = header + `<tbody>${leaderboardRows(lb.bottom)}</tbody>`;
}

function renderPitcherDetail(detail, name, season) {
  const mount = document.getElementById("tn-pitcher-mount");
  const rows = detail.pairs.map((p) => `<tr>
    <td>${p.type_a}/${p.type_b}</td>
    <td>${p.pct_a !== undefined ? p.pct_a.toFixed(1) + "%" : p.n_a}</td>
    <td>${p.pct_b !== undefined ? p.pct_b.toFixed(1) + "%" : p.n_b}</td>
    <td>${p.commit_distance.toFixed(1)}"</td>
    <td>${p.plate_distance.toFixed(1)}"</td>
    <td>${p.divergence.toFixed(1)}"</td>
    <td>${p.shape_break_plus.toFixed(1)}</td>
    <td class="${p.composite_score >= 100 ? "plus-pos" : "plus-neg"}">${p.composite_score.toFixed(1)}</td>
  </tr>`).join("");
  mount.innerHTML = `
    <div class="tn-arsenal-head">
      <b>${name}</b> - Tunneling+ <b style="color:var(--forest-dark);">${detail.tunneling_plus.toFixed(0)}</b>
      (${detail.n_pairs} qualifying pair${detail.n_pairs === 1 ? "" : "s"} of ${(detail.total_pitches || 0).toLocaleString()} pitches, ${season})
    </div>
    <table class="data-table tn-table" id="tn-pitcher-pairs-table">
      <thead><tr>
        <th>Pair</th><th>Usage A</th><th>Usage B</th>
        <th>Commit dist.</th><th>Plate dist.</th><th>Divergence</th><th>Shape break</th><th>Composite</th>
      </tr></thead>
      <tbody>${rows}</tbody>
    </table>
    <div class="tn-note">Distances are the full pairwise mean (inches) between every real pitch of type A and every real pitch of type B at that point in the flight path, not centroid-to-centroid. Composite = pair-type-normalized z-score of (-commit distance, + divergence, + shape break), averaged 1-to-3. The headline Tunneling+ above is the usage-weighted mean of these pair composites.</div>
  `;
}

async function loadPitcherDetail(playerId, name, season) {
  const status = document.getElementById("tn-status");
  const mount = document.getElementById("tn-pitcher-mount");
  status.textContent = "Loading...";
  mount.innerHTML = "";
  try {
    const detail = await getJSON(`/api/tunneling/v3/pitcher/${playerId}?season=${season}`);
    status.textContent = "";
    renderPitcherDetail(detail, name, season);
  } catch (err) {
    status.textContent = "No Tunneling+ data for this pitcher/season (needs 2+ qualifying pitch types, 100+ pitches each, and a qualifying commit point).";
  }
}

async function loadReport(season) {
  const report = await getJSON(`/api/tunneling/v3/report?season=${season}`);
  renderLeaderboard(report);
  return report;
}

async function init() {
  let selectedPlayer = null;

  const seasonSelect = createSeasonSelect(document.getElementById("season-select"), () => {
    loadReport(seasonSelect.value);
    if (selectedPlayer) loadPitcherDetail(selectedPlayer.player_id, selectedPlayer.name, seasonSelect.value);
  });
  const defaultSeason = await seasonSelect.load();
  await loadReport(defaultSeason);

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

  const params = new URLSearchParams(location.search);
  const urlPitcherId = params.get("pitcher_id");
  const remembered = PlayerContext.getCurrent();
  if (urlPitcherId) {
    selectedPlayer = { player_id: Number(urlPitcherId), name: "" };
    loadPitcherDetail(selectedPlayer.player_id, selectedPlayer.name, Number(params.get("season")) || seasonSelect.value);
  } else if (remembered && remembered.role === "pitcher") {
    selectedPlayer = remembered;
    search.setValue(remembered);
    loadPitcherDetail(remembered.player_id, remembered.name, seasonSelect.value);
  }
}

document.addEventListener("DOMContentLoaded", init);
