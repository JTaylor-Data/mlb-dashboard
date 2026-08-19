/*
 * Deception+ v1 model internals page: proof-of-concept, single-season
 * (no season selector yet - see train_deception_model.py). Renders the
 * offline report (CV metrics, SHAP importance, split-half reliability,
 * outcome validity, worked examples) and a plain-language verdict on
 * whether the model is actually useful.
 */
function renderVerdict(report) {
  const v = report.verdict;
  const el = document.getElementById("dc-verdict");
  const isUseful = v.split_half_correlation_100plus !== null && v.split_half_correlation_100plus > 0.5
    && v.n_outcomes_with_meaningful_partial_correlation >= 2;
  el.className = "dc-verdict " + (isUseful ? "useful" : "not-useful");
  el.innerHTML = `
    <b>${v.headline}</b><br/>
    Split-half reliability (100+ pitch pitchers): <b>${v.split_half_correlation_100plus !== null ? v.split_half_correlation_100plus.toFixed(3) : "n/a"}</b>.
    Outcomes with a meaningful partial correlation beyond Stuff+: <b>${v.n_outcomes_with_meaningful_partial_correlation} / 5</b>.
  `;
}

function renderMeta(report) {
  const items = [
    ["Target", "delta_run_exp"],
    ["Report season", report.report_season],
    ["CV mean R²", report.cv_r2_mean.toFixed(5)],
    ["CV mean MAE", report.cv_mae_mean.toFixed(4)],
  ];
  document.getElementById("dc-meta-mount").innerHTML = items
    .map(([label, value]) => `<div class="dc-meta-item"><div class="dc-meta-label">${label}</div><div class="dc-meta-value">${value}</div></div>`)
    .join("");
  document.getElementById("dc-excluded-note").textContent = report.excluded_features_note;
}

function renderReliability(report) {
  const rows = Object.entries(report.split_half_reliability)
    .map(([minPitches, r]) => `<tr><td>min ${minPitches} pitches</td><td>${r.n}</td><td>${r.correlation !== null ? r.correlation.toFixed(3) : "n/a"}</td></tr>`)
    .join("");
  document.getElementById("dc-reliability-table").innerHTML = `
    <thead><tr><th>Cutoff</th><th>n pitchers</th><th>Correlation</th></tr></thead>
    <tbody>${rows}</tbody>
  `;
}

function renderValidity(report) {
  const v = report.outcome_validity;
  const rows = Object.entries(v.outcomes)
    .map(([outcome, r]) => `<tr><td>${outcome}</td><td>${r.raw_correlation.toFixed(3)}</td><td>${r.partial_correlation_stuff_removed.toFixed(3)}</td></tr>`)
    .join("");
  document.getElementById("dc-validity-table").innerHTML = `
    <thead><tr><th>${report.report_season} outcome</th><th>Raw r</th><th>Partial r (Stuff+ removed)</th></tr></thead>
    <tbody>${rows}</tbody>
  `;
  document.getElementById("dc-validity-note").textContent =
    `n=${v.n_qualified} qualified pitchers (300+ pitches). Correlation with overall Stuff+ itself: ${v.correlation_with_overall_stuff.toFixed(3)}. ` +
    `The partial column removes what Stuff+ already explains before correlating - a meaningfully positive number there means the delivery features add real, independent signal.`;
}

function renderShap(report) {
  const entries = Object.entries(report.mean_abs_shap).sort((a, b) => b[1] - a[1]);
  const max = Math.max(...entries.map(([, v]) => v));
  document.getElementById("dc-shap-mount").innerHTML = entries
    .map(([feat, val]) => `
      <div class="dc-bar-row">
        <div>${feat}</div>
        <div class="dc-bar-track"><div class="dc-bar-fill" style="width:${(val / max) * 100}%"></div></div>
        <div>${val.toFixed(4)}</div>
      </div>
    `)
    .join("");
}

function exampleRows(rows) {
  return rows
    .map((r) => `<tr><td>${r.name || "#" + r.pitcher_id}</td><td>${r.deception_plus.toFixed(0)}</td><td>${r.total_pitches}</td></tr>`)
    .join("");
}

function renderExamples(report) {
  const ex = report.worked_examples;
  const header = `<thead><tr><th>Pitcher</th><th>Deception+</th><th>Pitches</th></tr></thead>`;
  document.getElementById("dc-top-table").innerHTML = header + `<tbody>${exampleRows(ex.highest_deception)}</tbody>`;
  document.getElementById("dc-bottom-table").innerHTML = header + `<tbody>${exampleRows(ex.lowest_deception)}</tbody>`;
}

function renderCV(report) {
  const rows = report.cv_folds
    .map((f) => `<tr><td>${f.holdout_season}</td><td>${f.r2.toFixed(5)}</td><td>${f.mae.toFixed(4)}</td><td>${f.n_pitches.toLocaleString()}</td></tr>`)
    .join("");
  document.getElementById("dc-cv-table").innerHTML = `
    <thead><tr><th>Held-out season</th><th>R²</th><th>MAE</th><th>Pitches</th></tr></thead>
    <tbody>${rows}</tbody>
  `;
}

function renderPitcherDetail(detail) {
  const mount = document.getElementById("dc-pitcher-mount");
  const rows = detail.contributions
    .map(([feat, points, meanVal]) => {
      const displayVal = typeof meanVal === "number" ? meanVal.toFixed(2) : meanVal;
      return `<tr><td>${feat}</td><td>${displayVal}</td><td class="${points >= 0 ? 'plus-pos' : 'plus-neg'}">${points >= 0 ? '+' : ''}${points.toFixed(1)}</td></tr>`;
    })
    .join("");
  mount.innerHTML = `
    <div class="dc-arsenal-head">
      <b>${detail.name}</b> - Deception+ <b style="color:var(--bark-dark);">${detail.deception_plus.toFixed(0)}</b>
      (${detail.n_pitches} pitches, ${detail.report_season})
    </div>
    <table class="data-table dc-table" id="dc-pitcher-shap-table">
      <thead><tr><th>Feature</th><th>Pitcher's value</th><th>SHAP (plus-points)</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
    <div class="dc-note">Each row is how much that feature pushes THIS pitcher's prediction away from the model's baseline, in Deception+-point units - not the same as the headline score being a sum of these (see the report's shrinkage/aggregation notes), but the biggest movers here explain why this pitcher scores where they do.</div>
  `;
}

async function loadPitcherDetail(playerId, name) {
  const status = document.getElementById("dc-pitcher-status");
  const mount = document.getElementById("dc-pitcher-mount");
  status.textContent = "Loading...";
  mount.innerHTML = "";
  try {
    const detail = await getJSON(`/api/deception/pitcher/${playerId}`);
    detail.name = name;
    detail.report_season = window.__dcReportSeason;
    status.textContent = "";
    renderPitcherDetail(detail);
  } catch (err) {
    status.textContent = "No Deception+ data for this pitcher (needs pitches in the report season).";
  }
}

async function init() {
  const report = await getJSON("/api/deception/report");
  window.__dcReportSeason = report.report_season;
  renderVerdict(report);
  renderMeta(report);
  renderReliability(report);
  renderValidity(report);
  renderShap(report);
  renderExamples(report);
  renderCV(report);

  const search = createPlayerSearch(
    document.getElementById("player-search-input"),
    document.getElementById("player-search-dropdown"),
    (player) => loadPitcherDetail(player.player_id, player.name),
    { roleFilter: "pitcher" }
  );
  await search.load(report.report_season);
}

document.addEventListener("DOMContentLoaded", init);
