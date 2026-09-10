/*
 * Stuff+ model internals page (v3 architecture): renders the offline
 * training report (CV performance, global feature importance), a
 * top/bottom Stuff+ leaderboard, and, on picking a pitcher, fetches a
 * live per-pitch-type SHAP breakdown so every rating on the site can be
 * traced back to specific inputs rather than trusted blind.
 */
const FEATURE_LABELS = {
  release_speed: "Velo",
  effective_speed: "Perceived Velo",
  release_spin_rate: "Spin Rate",
  spin_axis: "Spin Axis",
  hb: "Horizontal Break",
  ivb: "Induced Vert. Break",
  vaa_residual: "VAA vs. Expected",
  axis_differential: "Axis Differential (SSW proxy)",
  velo_sep_vs_primary_fb: "Velo Sep. vs. Primary FB",
  hb_sep_vs_primary_fb: "H-Break Sep. vs. Primary FB",
  ivb_sep_vs_primary_fb: "V-Break Sep. vs. Primary FB",
  pitch_type: "Pitch Type",
  p_throws: "Pitcher Hand",
  stand: "Batter Stand",
  season: "Season",
};

function featureLabel(f) {
  return FEATURE_LABELS[f] || f;
}

function renderCvTable(report) {
  const table = document.getElementById("sm-cv-table");
  const rows = report.cv_folds
    .map((f) => `<tr><td>${f.holdout_season}</td><td>${f.r2.toFixed(4)}</td><td>${f.mae.toFixed(4)}</td><td>${f.n_pitches.toLocaleString()}</td></tr>`)
    .join("");
  table.innerHTML = `
    <thead><tr><th>Held-out season</th><th>R&sup2;</th><th>MAE</th><th>Pitches</th></tr></thead>
    <tbody>${rows}</tbody>
  `;
  document.getElementById("sm-cv-summary").textContent =
    `Mean across folds: R^2 = ${report.cv_r2_mean.toFixed(4)}, MAE = ${report.cv_mae_mean.toFixed(4)}. ` +
    `Trained on seasons ${report.cv_folds.map((f) => f.holdout_season)[0] ? "2021-2026 (2026 partial)" : ""}.`;
}

function renderImportance(report) {
  const mount = document.getElementById("sm-importance-mount");
  const entries = Object.entries(report.mean_abs_shap).sort((a, b) => b[1] - a[1]);
  const max = Math.max(...entries.map((e) => e[1]));
  mount.innerHTML = entries
    .map(([feature, value]) => `
      <div class="sm-imp-row">
        <div class="sm-imp-label">${featureLabel(feature)}</div>
        <div class="sm-imp-track"><div class="sm-imp-fill" style="width:${(value / max) * 100}%"></div></div>
        <div class="sm-imp-val">${value.toFixed(4)}</div>
      </div>
    `)
    .join("");
}

function renderPitcherBreakdown(breakdown) {
  const mount = document.getElementById("sm-pitcher-mount");
  const pitchTypes = Object.entries(breakdown).sort((a, b) => b[1].n_pitches - a[1].n_pitches);

  mount.innerHTML = pitchTypes
    .map(([pitchType, data]) => {
      const maxAbs = Math.max(...data.contributions.map(([, v]) => Math.abs(v)), 0.01);
      // Named to avoid shadowing the shared stuffColor() gradient function
      // from gradient.js - this is just a flat >=100 threshold color, not
      // that function.
      const plusColor = data.stuff_plus >= 100 ? "#1a7f37" : "#c9302c";
      const rows = data.contributions
        .map(([feature, points, typicalValue]) => {
          const halfWidth = (Math.abs(points) / maxAbs) * 50;
          const left = points >= 0 ? 50 : 50 - halfWidth;
          const color = points >= 0 ? "#1a7f37" : "#c9302c";
          // null means the feature is absent for every pitch in this slice
          // (e.g. a fastball-separation feature on the pitcher's own
          // primary-fastball pitch type - see _typical_value in
          // analytics/stuff_model.py) - "n/a", not the literal word "null".
          const displayVal = typeof typicalValue === "number" ? typicalValue.toFixed(2) : (typicalValue === null ? "n/a" : typicalValue);
          return `
            <div class="shap-row">
              <div class="shap-label">${featureLabel(feature)} <span class="muted">(${displayVal})</span></div>
              <div class="shap-track">
                <div class="shap-zero" style="left:50%"></div>
                <div class="shap-bar" style="left:${left}%; width:${halfWidth}%; background:${color};"></div>
              </div>
              <div class="shap-val ${points >= 0 ? "pos" : "neg"}">${points >= 0 ? "+" : ""}${points.toFixed(1)}</div>
            </div>
          `;
        })
        .join("");
      return `
        <div class="sm-pitch-card">
          <div class="sm-pitch-head">
            <div class="sm-pitch-name">${pitchType}</div>
            <div class="sm-pitch-meta">${data.n_pitches} pitches</div>
            <div class="sm-pitch-stuff" style="color:${plusColor}">${data.stuff_plus.toFixed(0)}</div>
          </div>
          ${rows}
        </div>
      `;
    })
    .join("");
}

function renderShrinkage(shrinkage) {
  const mount = document.getElementById("sm-shrinkage-mount");
  const entries = Object.entries(shrinkage.k_by_pitch_type).sort((a, b) => a[1] - b[1]);
  const max = Math.max(...entries.map((e) => e[1]), shrinkage.fallback_k);
  mount.innerHTML = entries
    .map(([pitchType, k]) => `
      <div class="sm-shrink-row">
        <div class="sm-shrink-label">${pitchType}</div>
        <div class="sm-shrink-track"><div class="sm-shrink-fill" style="width:${(k / max) * 100}%"></div></div>
        <div class="sm-shrink-val">K=${k.toFixed(1)}</div>
      </div>
    `)
    .join("");
  document.getElementById("sm-shrinkage-fallback").textContent =
    `Fallback K for pitch types too rare to estimate their own: ${shrinkage.fallback_k.toFixed(1)}.`;
}

function renderPredictiveValidity(pv) {
  document.getElementById("sm-pv-blurb").textContent =
    `Each pitcher's ${pv.score_season} Stuff+ (the live v3 model, scored by a version that never saw ` +
    `${pv.score_season} - honest out-of-sample) vs. their REAL ${pv.outcome_season} results.`;

  const minPitchesOptions = Object.keys(pv.results_by_min_pitches).map(Number).sort((a, b) => a - b);
  const midOption = String(minPitchesOptions[Math.floor(minPitchesOptions.length / 2)]);
  const result = pv.results_by_min_pitches[midOption];

  const rows = Object.entries(result.correlations)
    .map(([outcome, r]) => `<tr><td>${outcome}</td><td>${r.toFixed(3)}</td></tr>`)
    .join("");

  const mount = document.getElementById("sm-pv-mount");
  mount.innerHTML = `
    <table class="data-table sm-pv-table">
      <thead><tr><th>${pv.outcome_season} outcome</th><th>r vs. ${pv.score_season} score</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
    <div style="margin-top:8px; font-size:12px; color:var(--text-muted);">
      min ${midOption} ${pv.score_season} pitches, n=${result.n_qualified} qualified pitchers.
    </div>
  `;
}

function renderLeaderboardList(mountId, entries) {
  const mount = document.getElementById(mountId);
  if (!entries.length) {
    mount.innerHTML = `<div class="muted" style="font-size:13px;">No qualified pitchers for this season.</div>`;
    return;
  }
  mount.innerHTML = entries
    .map((e, i) => {
      const color = e.stuff_plus >= 100 ? "#1a7f37" : "#c9302c";
      const team = e.team_abbrev ? ` <span class="muted">(${e.team_abbrev})</span>` : "";
      return `
        <div class="sm-lb-row">
          <div class="sm-lb-rank">${i + 1}</div>
          <div class="sm-lb-name">${e.name}${team}</div>
          <div class="sm-lb-n muted">${e.n_pitches.toLocaleString()} pitches</div>
          <div class="sm-lb-val" style="color:${color}">${e.stuff_plus.toFixed(0)}</div>
        </div>
      `;
    })
    .join("");
}

async function loadLeaderboard(season) {
  try {
    const lb = await getJSON(`/api/stuff-model/leaderboard?season=${season}`);
    renderLeaderboardList("sm-lb-top-mount", lb.top);
    renderLeaderboardList("sm-lb-bottom-mount", lb.bottom);
  } catch (err) {
    document.getElementById("sm-lb-top-mount").innerHTML = `<div class="muted" style="font-size:13px;">Failed to load.</div>`;
    document.getElementById("sm-lb-bottom-mount").innerHTML = "";
  }
}

async function loadPitcherBreakdown(playerId, season) {
  const status = document.getElementById("sm-status");
  const mount = document.getElementById("sm-pitcher-mount");
  status.textContent = "Loading...";
  mount.innerHTML = "";
  try {
    const breakdown = await getJSON(`/api/stuff-model/pitcher/${playerId}?season=${season}`);
    status.textContent = "";
    renderPitcherBreakdown(breakdown);
  } catch (err) {
    status.textContent = "No data for this pitcher/season.";
  }
}

async function init() {
  const report = await getJSON("/api/stuff-model/report");
  renderCvTable(report);
  renderImportance(report);
  if (report.shrinkage) renderShrinkage(report.shrinkage);

  try {
    const pv = await getJSON("/api/stuff-model/predictive-validity");
    renderPredictiveValidity(pv);
  } catch (err) {
    document.getElementById("sm-pv-blurb").textContent =
      "Not yet generated - run python validate_predictive_power.py.";
  }

  let selectedPlayer = null;

  const seasonSelect = createSeasonSelect(document.getElementById("season-select"), () => {
    loadLeaderboard(seasonSelect.value);
    if (selectedPlayer) loadPitcherBreakdown(selectedPlayer.player_id, seasonSelect.value);
  });
  const defaultSeason = await seasonSelect.load();
  loadLeaderboard(defaultSeason);

  const search = createPlayerSearch(
    document.getElementById("player-search-input"),
    document.getElementById("player-search-dropdown"),
    (player) => {
      selectedPlayer = player;
      loadPitcherBreakdown(player.player_id, seasonSelect.value);
    },
    { roleFilter: "pitcher" }
  );
  await search.load(defaultSeason);

  const params = new URLSearchParams(location.search);
  const urlPitcherId = params.get("pitcher_id");
  const remembered = PlayerContext.getCurrent();
  if (urlPitcherId) {
    selectedPlayer = { player_id: Number(urlPitcherId), name: "", role: "pitcher" };
    loadPitcherBreakdown(selectedPlayer.player_id, Number(params.get("season")) || seasonSelect.value);
  } else if (remembered && remembered.role === "pitcher") {
    selectedPlayer = remembered;
    search.setValue(remembered);
    loadPitcherBreakdown(remembered.player_id, seasonSelect.value);
  }
}

document.addEventListener("DOMContentLoaded", init);
