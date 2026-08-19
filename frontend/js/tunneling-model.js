/*
 * Tunneling+ v1 model internals page: renders the offline report (methodology,
 * validation results, leaderboards) and, on picking a pitcher, a live
 * pair-by-pair breakdown of their arsenal's tunnel geometry (compact table,
 * every qualifying pair visible at once - no per-pair bar charts). Every
 * report is precomputed per season by build_tunneling_report.py, so
 * changing the season reloads the entire page's data, not just the
 * pitcher panel.
 */
function renderMeta(report) {
  const m = report.methodology;
  const items = [
    ["Decision point", `${m.decision_point_feet_from_plate} ft from plate`],
    ["Min pitches/type", m.min_pitches_per_type],
    ["Distance floor", `${m.distance_floor_inches}"`],
    ["Pitchers scored", m.n_pitchers_scored],
  ];
  document.getElementById("tn-meta-mount").innerHTML = items
    .map(([label, value]) => `<div class="tn-meta-item"><div class="tn-meta-label">${label}</div><div class="tn-meta-value">${value}</div></div>`)
    .join("");
}

function renderReliability(report) {
  const rows = Object.entries(report.split_half_reliability)
    .map(([minPairs, r]) => `<tr><td>min ${minPairs} pair(s)</td><td>${r.n}</td><td>${r.correlation.toFixed(3)}</td></tr>`)
    .join("");
  document.getElementById("tn-reliability-table").innerHTML = `
    <thead><tr><th>Cutoff</th><th>n pitchers</th><th>Correlation</th></tr></thead>
    <tbody>${rows}</tbody>
  `;
}

function renderValidity(report) {
  const v = report.outcome_validity;
  const rows = Object.entries(v.outcomes)
    .map(([outcome, r]) => `<tr><td>${outcome}</td><td>${r.raw_correlation.toFixed(3)}</td><td>${r.partial_correlation_stuff_removed.toFixed(3)}</td></tr>`)
    .join("");
  document.getElementById("tn-validity-table").innerHTML = `
    <thead><tr><th>${report.season} outcome</th><th>Raw r</th><th>Partial r (Stuff+ removed)</th></tr></thead>
    <tbody>${rows}</tbody>
  `;
  document.getElementById("tn-validity-note").textContent =
    `n=${v.n_qualified} qualified pitchers (300+ pitches). Correlation with overall Stuff+ itself: ${v.correlation_with_overall_stuff.toFixed(3)} - ` +
    `modest, so this isn't just re-deriving Stuff+ under a new name. The partial column removes what Stuff+ already explains before correlating - ` +
    `a positive number there means tunneling adds real, independent signal.`;
}

function renderSequential(report) {
  const seq = report.sequential_same_at_bat_test;
  const terciles = ["low", "mid", "high"];
  const max = Math.max(...terciles.map((t) => seq.whiff_rate_by_tunnel_ratio_tercile[t].whiff_rate));
  const rows = terciles
    .map((t) => {
      const d = seq.whiff_rate_by_tunnel_ratio_tercile[t];
      const pct = d.whiff_rate * 100;
      return `
        <div class="tn-seq-bar-row">
          <div>${t}</div>
          <div class="tn-seq-track"><div class="tn-seq-fill" style="width:${(d.whiff_rate / max) * 100}%"></div></div>
          <div>${pct.toFixed(1)}%</div>
        </div>
      `;
    })
    .join("");
  document.getElementById("tn-seq-mount").innerHTML = rows;
  document.getElementById("tn-seq-note").textContent =
    `Whiff rate on the 2nd pitch of a same-at-bat, different-pitch-type sequence, bucketed by that specific pair's tunneling+ ` +
    `(n=${seq.n_with_known_pair_ratio.toLocaleString()} sequences). Low-to-high z-score: ${seq.low_vs_high_z_score.toFixed(2)} ` +
    `(>1.96 is the usual "statistically significant" bar - this tests the exact mechanism, not just a league-wide aggregate).`;
}

function exampleRows(rows) {
  return rows
    .map((r) => `<tr><td>${r.name || "#" + r.pitcher_id}</td><td>${r.tunneling_plus.toFixed(0)}</td><td>${r.n_pairs}</td><td>${r.total_pitches}</td></tr>`)
    .join("");
}

function renderExamples(report) {
  const ex = report.worked_examples;
  const header = `<thead><tr><th>Pitcher</th><th>Tunneling+</th><th>Pairs</th><th>Pitches</th></tr></thead>`;
  document.getElementById("tn-top-table").innerHTML = header + `<tbody>${exampleRows(ex.highest_tunneling)}</tbody>`;
  document.getElementById("tn-bottom-table").innerHTML = header + `<tbody>${exampleRows(ex.lowest_tunneling)}</tbody>`;
}

async function loadReport(season) {
  const report = await getJSON(`/api/tunneling/report?season=${season}`);
  renderMeta(report);
  renderReliability(report);
  renderValidity(report);
  renderSequential(report);
  renderExamples(report);
}

function renderPitcherDetail(detail) {
  const mount = document.getElementById("tn-pitcher-mount");
  const maxPlus = Math.max(...detail.pairs.map((p) => p.tunneling_plus), 100);

  const rows = detail.pairs
    .sort((a, b) => b.tunneling_plus - a.tunneling_plus)
    .map((p) => `
      <tr class="${p.is_anchor ? 'tn-anchor-row' : ''}">
        <td>${p.type_a} &harr; ${p.type_b}</td>
        <td>${p.n_a} / ${p.n_b}</td>
        <td>${p.release_distance.toFixed(1)}"</td>
        <td>${p.tunnel_distance.toFixed(1)}"</td>
        <td>${p.plate_distance.toFixed(1)}"</td>
        <td>${p.release_similarity_plus.toFixed(0)}</td>
        <td>${p.decision_similarity_plus.toFixed(0)}</td>
        <td>${p.plate_separation_plus.toFixed(0)}</td>
        <td class="${p.tunneling_plus === maxPlus ? 'plus-best' : ''}">${p.tunneling_plus.toFixed(0)}</td>
      </tr>
    `)
    .join("");

  mount.innerHTML = `
    <div class="tn-arsenal-head">
      <b>${detail.name}</b> - Tunneling+ <b style="color:var(--bark-dark);">${detail.tunneling_plus.toFixed(0)}</b>,
      release consistency <b>${detail.release_consistency.toFixed(1)}"</b>
    </div>
    <table class="data-table tn-table tn-pair-table">
      <thead>
        <tr>
          <th>Pair</th><th>N (a/b)</th><th>Release</th><th>Decision</th><th>Plate</th>
          <th>Release Sim+</th><th>Decision Sim+</th><th>Plate Sep+</th><th>Tunneling+</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
    <div class="tn-seq-note">Tunneling+ is the geometric mean of Decision Sim+ (close at the decision point) and Plate Sep+ (far apart at the plate) - both 100=league average. Release Sim+ is shown for context but isn't part of the score - it measurably weakened the model's validated predictive power when tested as a third factor. Shaded rows involve the four-seam or sinker and count double toward the headline score above.</div>
  `;
}

async function loadPitcherDetail(playerId, season) {
  const status = document.getElementById("tn-status");
  const mount = document.getElementById("tn-pitcher-mount");
  status.textContent = "Loading...";
  mount.innerHTML = "";
  try {
    const detail = await getJSON(`/api/tunneling/pitcher/${playerId}?season=${season}`);
    status.textContent = "";
    renderPitcherDetail(detail);
  } catch (err) {
    status.textContent = "No tunneling detail for this pitcher (needs 2+ qualifying pitch types, 300+ pitches).";
  }
}

let selectedPlayer = null;

async function init() {
  const seasonSelect = createSeasonSelect(document.getElementById("season-select"), (season) => {
    loadReport(season);
    if (selectedPlayer) {
      loadPitcherDetail(selectedPlayer.player_id, season);
    } else {
      document.getElementById("tn-pitcher-mount").innerHTML = "";
      document.getElementById("tn-status").textContent = "";
    }
  });
  const defaultSeason = await seasonSelect.load();
  await loadReport(defaultSeason);

  const search = createPlayerSearch(
    document.getElementById("player-search-input"),
    document.getElementById("player-search-dropdown"),
    (player) => {
      selectedPlayer = player;
      loadPitcherDetail(player.player_id, seasonSelect.value);
    },
    { roleFilter: "pitcher" }
  );
  await search.load(defaultSeason);
}

document.addEventListener("DOMContentLoaded", init);
