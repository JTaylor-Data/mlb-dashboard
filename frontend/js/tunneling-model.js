/*
 * Tunneling+ v3 model-internals page: a single trained model (no hand-built
 * composite, no per-pitch-type-pair table - see the page's own blurb) using
 * real within-at-bat lag features (up to 6 pitches back). Renders the
 * static training-time report (CV, lag-slot SHAP importance, the
 * validation battery) plus a live, season-driven leaderboard and
 * per-pitcher lag-slot breakdown from /api/tunneling/v3/*.
 */
function renderMeta(report) {
  const items = [
    ["Features", report.all_features.length],
    ["Training rows", report.n_training_rows.toLocaleString()],
    ["CV mean R²", report.cv_r2_mean.toFixed(5)],
    ["CV mean MAE", report.cv_mae_mean.toFixed(4)],
    ["Max lag (pitches back)", 6],
    ["Commit point basis", "v2, 200ms pre-arrival"],
  ];
  document.getElementById("tn-meta-mount").innerHTML = items
    .map(([label, value]) => `<div class="tn-meta-item"><div class="tn-meta-label">${label}</div><div class="tn-meta-value">${value}</div></div>`)
    .join("");
}

function renderLagShap(report) {
  const entries = Object.entries(report.lag_slot_shap_summary);
  const max = Math.max(...entries.map(([, v]) => v.distance_group_total + v.pitch_type));
  document.getElementById("tn-lagshap-mount").innerHTML = entries
    .map(([lag, v]) => {
      const total = v.distance_group_total + v.pitch_type;
      const lagN = lag.replace("lag", "");
      return `
        <div class="tn-bar-row">
          <div>${lagN} back</div>
          <div class="tn-bar-track"><div class="tn-bar-fill" style="width:${(total / max) * 100}%"></div></div>
          <div>${total.toFixed(4)}</div>
        </div>
      `;
    })
    .join("");
}

function renderCv(report) {
  const rows = report.cv_folds
    .map((f) => `<tr><td>${f.holdout_season}</td><td>${f.r2.toFixed(5)}</td><td>${f.mae.toFixed(4)}</td><td>${f.n.toLocaleString()}</td></tr>`)
    .join("");
  document.getElementById("tn-cv-table").innerHTML = `
    <thead><tr><th>Held-out season</th><th>R²</th><th>MAE</th><th>Rows</th></tr></thead>
    <tbody>${rows}</tbody>
  `;
}

function renderSequential(report) {
  const seq = report.sequential_same_at_bat_test;
  const order = [["low_pred", "Best predicted"], ["mid_pred", "Mid"], ["high_pred", "Worst predicted"]];
  const max = Math.max(...order.map(([k]) => seq.whiff_rate_by_predicted_tercile[k].whiff_rate));
  document.getElementById("tn-seq-mount").innerHTML = order
    .map(([key, label]) => {
      const r = seq.whiff_rate_by_predicted_tercile[key];
      return `
        <div class="tn-seq-bar-row">
          <div>${label}</div>
          <div class="tn-seq-track"><div class="tn-seq-fill" style="width:${(r.whiff_rate / max) * 100}%"></div></div>
          <div>${(r.whiff_rate * 100).toFixed(1)}% (n=${r.n.toLocaleString()})</div>
        </div>
      `;
    })
    .join("");
  document.getElementById("tn-seq-note").textContent =
    `Best-vs-worst-predicted-tercile z-score: ${seq.good_minus_bad_z_score.toFixed(1)}. Buckets the model's own final prediction into terciles across every real training row and checks real whiff rate on that same pitch - not directly comparable to v1/v2's pair-level z-scores (see the report JSON's own methodology_note for exactly why), but a real, honest check that the model's full prediction - lag context and all - tracks whether hitters actually miss.`;
}

function renderPredictiveValidity(report) {
  const pv = report.predictive_validity;
  if (!pv) {
    document.getElementById("tn-predictive-table").innerHTML = `<tbody><tr><td>Not yet generated.</td></tr></tbody>`;
    return;
  }
  const minPitchesOptions = Object.keys(pv.results_by_min_pitches).map(Number).sort((a, b) => a - b);
  const midOption = String(minPitchesOptions[Math.floor(minPitchesOptions.length / 2)]);
  const result = pv.results_by_min_pitches[midOption];
  const rows = Object.entries(result.correlations)
    .map(([outcome, r]) => `<tr><td>${outcome}</td><td>${r.toFixed(3)}</td></tr>`)
    .join("");
  document.getElementById("tn-predictive-table").innerHTML = `
    <thead><tr><th>${pv.outcome_season} outcome</th><th>r vs. ${pv.score_season} score</th></tr></thead>
    <tbody>${rows}</tbody>
  `;
}

function renderYoy(report) {
  const yoy = report.year_over_year;
  const rows = yoy.per_pair
    .map((r) => `<tr><td>${r.season_t} → ${r.season_t1}</td><td>${r.n}</td><td>${r.r2.toFixed(3)}</td></tr>`)
    .join("");
  document.getElementById("tn-yoy-table").innerHTML = `
    <thead><tr><th>Season pair</th><th>n</th><th>R²</th></tr></thead>
    <tbody>${rows}</tbody>
  `;
  document.getElementById("tn-yoy-note").textContent =
    `Pooled across every season pair: R² = ${yoy.pooled_r2} (n=${yoy.n_pooled.toLocaleString()}). Is a pitcher's tunneling score this year predictive of next year's, or mostly noise?`;
}

function renderCrossModel(report) {
  const cmc = report.cross_model_correlation;
  document.getElementById("tn-cross-table").innerHTML = `
    <thead><tr><th>vs.</th><th>Correlation</th></tr></thead>
    <tbody>
      <tr><td>Stuff+</td><td>${cmc.r_stuff.toFixed(3)}</td></tr>
      <tr><td>Deception+</td><td>${cmc.r_deception.toFixed(3)}</td></tr>
      <tr><td>Fastest FB velo</td><td>${report.fastest_fb_velo_correlation.toFixed(3)}</td></tr>
    </tbody>
  `;
}

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

const LAG_SLOT_LABELS = {
  commit_distance: "Commit-point distance",
  plate_distance: "Plate distance",
  shape_break: "Shape break",
  pitch_type: "Pitch type",
};

function renderPitcherDetail(detail, name, season) {
  const mount = document.getElementById("tn-pitcher-mount");
  const rows = [];
  detail.lag_slot_breakdown.forEach((slot) => {
    ["commit_distance", "plate_distance", "shape_break", "pitch_type"].forEach((key) => {
      const points = slot[key];
      rows.push(`<tr><td>${slot.lag} back - ${LAG_SLOT_LABELS[key]}</td><td class="${points >= 0 ? "plus-pos" : "plus-neg"}">${points >= 0 ? "+" : ""}${points.toFixed(2)}</td></tr>`);
    });
  });
  mount.innerHTML = `
    <div class="tn-arsenal-head">
      <b>${name}</b> - Tunneling+ <b style="color:var(--forest-dark);">${detail.score.toFixed(0)}</b>
      (${detail.n_pitches} scored pitches of ${detail.raw_pitches} thrown, ${season})
    </div>
    <table class="data-table tn-table" id="tn-pitcher-shap-table">
      <thead><tr><th>Lag slot - component</th><th>SHAP (plus-points)</th></tr></thead>
      <tbody>${rows.join("")}</tbody>
    </table>
    <div class="tn-note">Each row is how much that lag slot's component pushes this pitcher's own score away from the model's baseline, in Tunneling+-point units - not the same as the headline score being a sum of these, but the biggest movers explain why this pitcher scores where they do. v3 has no per-pitch-type-pair table like v1 did - there's no such quantity in this architecture.</div>
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
    status.textContent = "No Tunneling+ v3 data for this pitcher/season (needs 100+ pitches that season and a qualifying commit point).";
  }
}

async function loadReport(season) {
  const report = await getJSON(`/api/tunneling/v3/report?season=${season}`);
  renderMeta(report);
  renderLagShap(report);
  renderCv(report);
  renderSequential(report);
  renderPredictiveValidity(report);
  renderYoy(report);
  renderCrossModel(report);
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
