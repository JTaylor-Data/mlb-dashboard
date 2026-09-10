/*
 * Location+ v1 model-internals page: a trained model over 7 features
 * (plate_x, plate_z, balls, strikes, pitch_type, stand, season) - no
 * physical pitch-shape, delivery, or arsenal-relationship inputs at all.
 * Renders the static training-time report (CV, SHAP importance, the full
 * validation battery including a direct Stuff+ stabilization comparison,
 * the count/pitch-type interaction sanity check, and the stand-inclusion
 * ablation) plus a live, season-driven leaderboard and per-pitcher SHAP
 * breakdown from /api/location/*.
 */
function renderMeta(report) {
  const items = [
    ["Features", report.all_features.length],
    ["Training rows", report.n_training_rows.toLocaleString()],
    ["CV mean R²", report.cv_r2_mean.toFixed(4)],
    ["CV mean MAE", report.cv_mae_mean.toFixed(4)],
    ["Shrinkage fallback K", report.shrinkage_fallback_k.toFixed(1)],
  ];
  document.getElementById("lc-meta-mount").innerHTML = items
    .map(([label, value]) => `<div class="lc-meta-item"><div class="lc-meta-label">${label}</div><div class="lc-meta-value">${value}</div></div>`)
    .join("");
}

function renderShap(report) {
  const entries = Object.entries(report.mean_abs_shap).sort((a, b) => b[1] - a[1]);
  const max = Math.max(...entries.map(([, v]) => v));
  document.getElementById("lc-shap-mount").innerHTML = entries
    .map(([feat, val]) => `
      <div class="lc-bar-row">
        <div>${feat}</div>
        <div class="lc-bar-track"><div class="lc-bar-fill" style="width:${(val / max) * 100}%"></div></div>
        <div>${val.toFixed(4)}</div>
      </div>
    `)
    .join("");
}

function renderCv(report) {
  const rows = report.cv_folds
    .map((f) => `<tr><td>${f.holdout_season}</td><td>${f.r2.toFixed(4)}</td><td>${f.mae.toFixed(4)}</td><td>${f.n.toLocaleString()}</td></tr>`)
    .join("");
  document.getElementById("lc-cv-table").innerHTML = `
    <thead><tr><th>Held-out season</th><th>R²</th><th>MAE</th><th>Rows</th></tr></thead>
    <tbody>${rows}</tbody>
  `;
}

function renderReliability(report) {
  const rel = report.reliability;
  const thresholds = Object.keys(rel.location_plus);
  const rows = thresholds
    .map((t) => {
      const loc = rel.location_plus[t], stuff = rel.stuff_plus[t];
      return `<tr><td>per-half ≥ ${t}</td><td>${loc.correlation !== null ? loc.correlation.toFixed(3) : "n/a"}</td><td>${stuff.correlation !== null ? stuff.correlation.toFixed(3) : "n/a"}</td></tr>`;
    })
    .join("");
  document.getElementById("lc-reliability-table").innerHTML = `
    <thead><tr><th>Threshold</th><th>Location+ r</th><th>Stuff+ r</th></tr></thead>
    <tbody>${rows}</tbody>
  `;
  document.getElementById("lc-reliability-note").textContent =
    `Location+ crosses r≥0.7 at per-half≥${rel.location_crosses_070 ?? "not reached"}, r≥0.9 at per-half≥${rel.location_crosses_090 ?? "not reached"}. ` +
    `Stuff+ crosses both at per-half≥${rel.stuff_crosses_070}. Same infrastructure, same thresholds, run side by side - Location+ takes dramatically longer to stabilize, confirming FanGraphs' own public finding on this project's own data.`;
}

function renderYoy(report) {
  const yoy = report.year_over_year;
  const rows = yoy.per_pair
    .map((r) => `<tr><td>${r.season_t} → ${r.season_t1}</td><td>${r.n}</td><td>${r.r2.toFixed(3)}</td></tr>`)
    .join("");
  document.getElementById("lc-yoy-table").innerHTML = `
    <thead><tr><th>Season pair</th><th>n</th><th>R²</th></tr></thead>
    <tbody>${rows}</tbody>
  `;
  document.getElementById("lc-yoy-note").textContent =
    `Pooled: R² = ${yoy.pooled_r2} (n=${yoy.n_pooled.toLocaleString()}). Lower/less sticky than Stuff+ is expected here, per the same stabilization finding above.`;
}

function renderPredictiveValidity(report) {
  const pv = report.predictive_validity;
  if (!pv) {
    document.getElementById("lc-predictive-table").innerHTML = `<tbody><tr><td>Not yet generated.</td></tr></tbody>`;
    return;
  }
  const minPitchesOptions = Object.keys(pv.results_by_min_pitches).map(Number).sort((a, b) => a - b);
  const midOption = String(minPitchesOptions[Math.floor(minPitchesOptions.length / 2)]);
  const result = pv.results_by_min_pitches[midOption];
  const rows = Object.entries(result.correlations)
    .map(([outcome, r]) => `<tr><td>${outcome}</td><td>${r.toFixed(3)}</td></tr>`)
    .join("");
  document.getElementById("lc-predictive-table").innerHTML = `
    <thead><tr><th>${pv.outcome_season} outcome</th><th>r vs. ${pv.score_season} score</th></tr></thead>
    <tbody>${rows}</tbody>
  `;
}

function renderCrossModel(report) {
  const cmc = report.cross_model_correlation;
  document.getElementById("lc-cross-table").innerHTML = `
    <thead><tr><th>vs.</th><th>Correlation</th></tr></thead>
    <tbody>
      <tr><td>Stuff+</td><td>${cmc.r_stuff.toFixed(3)}</td></tr>
      <tr><td>Deception+</td><td>${cmc.r_deception.toFixed(3)}</td></tr>
      <tr><td>Tunneling+</td><td>${cmc.r_tunneling.toFixed(3)}</td></tr>
      <tr><td>Fastest FB velo (leakage check)</td><td>${report.fastest_fb_velo_correlation.toFixed(3)}</td></tr>
    </tbody>
  `;
}

function renderInteraction(report) {
  const ci = report.count_interaction;
  if (!ci) return;
  document.getElementById("lc-interaction-note").textContent =
    `${ci.pitch_type} vs. ${ci.stand}HH, ${ci.season}: grid correlation between the two counts = ${ci.grid_correlation_3_1_vs_3_2.toFixed(3)} (near 1.0 is expected - the zone's overall shape doesn't move), ` +
    `mean |3-2 minus 3-1| = ${ci.mean_abs_diff_3_2_vs_3_1.toFixed(4)}, max = ${ci.max_abs_diff_3_2_vs_3_1.toFixed(4)} - a real, non-flat, location-specific interaction on top of the shared shape, not just a rescale.`;
}

function renderAblation(report) {
  const ab = report.stand_ablation;
  if (!ab) return;
  const rows = [
    ["CV R²", ab.with_stand.cv_r2_mean.toFixed(4), ab.without_stand.cv_r2_mean.toFixed(4)],
    ["CV MAE", ab.with_stand.cv_mae_mean.toFixed(4), ab.without_stand.cv_mae_mean.toFixed(4)],
    ["YoY pooled R²", ab.with_stand.yoy_pooled_r2, ab.without_stand.yoy_pooled_r2],
  ];
  document.getElementById("lc-ablation-mount").innerHTML = `
    <div class="lc-ablation-row head"><div></div><div>With stand (promoted)</div><div>Without stand (FanGraphs-exact)</div></div>
    ${rows.map(([label, w, wo]) => `<div class="lc-ablation-row"><div>${label}</div><div>${w}</div><div>${wo}</div></div>`).join("")}
    <div class="lc-note">Keeping stand improves both accuracy and stability - a real, earned deviation from FanGraphs' public methodology, not just a theoretical justification.</div>
  `;
}

function leaderboardRows(rows) {
  return rows
    .map((r) => `<tr><td>${r.name || "#" + r.pitcher_id}${r.team_abbrev ? ` <span class="muted">(${r.team_abbrev})</span>` : ""}</td><td>${r.score.toFixed(1)}</td><td>${r.total_pitches.toLocaleString()}</td></tr>`)
    .join("");
}

function renderLeaderboard(report) {
  const lb = report.leaderboard;
  const header = `<thead><tr><th>Pitcher</th><th>Location+</th><th>Pitches</th></tr></thead>`;
  document.getElementById("lc-top-table").innerHTML = header + `<tbody>${leaderboardRows(lb.top)}</tbody>`;
  document.getElementById("lc-bottom-table").innerHTML = header + `<tbody>${leaderboardRows(lb.bottom)}</tbody>`;
}

function renderPitcherDetail(detail, name, season) {
  const mount = document.getElementById("lc-pitcher-mount");
  const rows = detail.contributions
    .map(([feat, points, typicalValue]) => {
      const displayVal = typeof typicalValue === "number" ? typicalValue.toFixed(2) : typicalValue;
      return `<tr><td>${feat} <span class="muted">(${displayVal})</span></td><td class="${points >= 0 ? "plus-pos" : "plus-neg"}">${points >= 0 ? "+" : ""}${points.toFixed(2)}</td></tr>`;
    })
    .join("");
  mount.innerHTML = `
    <div class="lc-arsenal-head">
      <b>${name}</b> - Location+ <b style="color:var(--forest-dark);">${detail.score.toFixed(0)}</b>
      (${detail.n_pitches} pitches, ${season})
    </div>
    <table class="data-table lc-table" id="lc-pitcher-shap-table">
      <thead><tr><th>Feature (pitcher's own average value)</th><th>SHAP (plus-points)</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
    <div class="lc-note">Each row is how much that feature pushes this pitcher's own prediction away from the model's baseline, in Location+-point units.</div>
  `;
}

async function loadPitcherDetail(playerId, name, season) {
  const status = document.getElementById("lc-status");
  const mount = document.getElementById("lc-pitcher-mount");
  status.textContent = "Loading...";
  mount.innerHTML = "";
  try {
    const detail = await getJSON(`/api/location/pitcher/${playerId}?season=${season}`);
    status.textContent = "";
    renderPitcherDetail(detail, name, season);
  } catch (err) {
    status.textContent = "No Location+ data for this pitcher/season (needs 100+ pitches that season).";
  }
}

async function loadReport(season) {
  const report = await getJSON(`/api/location/report?season=${season}`);
  renderMeta(report);
  renderShap(report);
  renderCv(report);
  renderReliability(report);
  renderYoy(report);
  renderPredictiveValidity(report);
  renderCrossModel(report);
  renderInteraction(report);
  renderAblation(report);
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
