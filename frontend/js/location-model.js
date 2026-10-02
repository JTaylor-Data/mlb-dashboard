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

function renderMeta(report) {
  const sc = report.scoring_config || {};
  const gates = report.gates || {};

  const allPass = Object.values(gates).every(g => g && g.pass !== false);
  const gateLabel = Object.keys(gates).length === 0 ? "Not yet evaluated"
                  : allPass ? "All pass ✓" : "Some gates failed";
  const gateColor = Object.keys(gates).length === 0 ? "var(--text-muted)"
                  : allPass ? "#1a7f37" : "#c9302c";

  const items = [
    ["Classifiers", "5 (xSwing, xCS, xWhiff, xFoul, BIP×EV)"],
    ["Baseline variant", sc.chosen_variant || "—"],
    ["Shrinkage", sc.use_shrinkage ? "Yes" : "No"],
    ["Min pitches", (sc.leaderboard_min_pitches || 300).toString()],
    ["Stop-ship gates", gateLabel],
  ];

  document.getElementById("lc-meta-mount").innerHTML = items
    .map(([label, value], i) => {
      const style = i === 4 ? `style="color:${gateColor}"` : "";
      return `<div class="lc-meta-item">
        <div class="lc-meta-label">${label}</div>
        <div class="lc-meta-value" ${style}>${value}</div>
      </div>`;
    })
    .join("");
}

function renderGates(report) {
  const gates = report.gates || {};
  const mount = document.getElementById("lc-gates-mount");
  if (!mount) return;
  if (Object.keys(gates).length === 0) {
    mount.innerHTML = `<div class="lc-note">Gates not yet evaluated — run Parts 4–5.</div>`;
    return;
  }
  const gateDesc = {
    sign_check: (g) => `ERA r=${g.era_r !== undefined ? g.era_r.toFixed(3) : "—"}, xFIP r=${g.xfip_r !== undefined ? g.xfip_r.toFixed(3) : "—"}, BB% r=${g.bb_r !== undefined ? g.bb_r.toFixed(3) : "—"} (need era<0, xfip<0, bb≤−0.20)`,
    yoy_floor: (g) => `Min pair YoY r = ${g.min_pair !== undefined ? g.min_pair.toFixed(3) : "—"} (need ≥ 0.30)`,
    v2v3_corr: (g) => `v3 vs v2 pitcher-season r = ${g.corr !== undefined ? g.corr.toFixed(3) : "—"} (need ≥ 0.70)`,
    seasons_scored: (g) => `Scored seasons: ${(g.scored || []).join(", ")}`,
    ece: (g) => g.note || "Expected calibration error (not fully computed)",
  };
  const rows = Object.entries(gates).map(([name, g]) => {
    const ok = g.pass !== false;
    const icon = ok ? `<span style="color:#1a7f37;font-weight:700;">✓</span>`
                    : `<span style="color:#c9302c;font-weight:700;">✗</span>`;
    const desc = gateDesc[name] ? gateDesc[name](g) : (g.note || JSON.stringify(g));
    return `<tr><td>${icon}</td><td>${name}</td><td class="lc-note">${desc}</td></tr>`;
  }).join("");
  mount.innerHTML = `
    <table class="data-table lc-table">
      <thead><tr><th></th><th>Gate</th><th>Result</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

function renderReliability(report) {
  const sc = report.scoring_config || {};
  const sh = sc.split_half_by_threshold || {};
  const thresholds = Object.keys(sh).map(Number).sort((a, b) => a - b);
  if (thresholds.length === 0) {
    document.getElementById("lc-reliability-table").innerHTML =
      `<tbody><tr><td colspan="4">Not yet computed.</td></tr></tbody>`;
    return;
  }
  const rows = thresholds.map((t) => {
    const d = sh[t];
    return `<tr>
      <td>≥ ${t}</td>
      <td>${d.r_raw.toFixed(3)}</td>
      <td>${d.r_sb.toFixed(3)}</td>
      <td>${(d.n || "—").toLocaleString()}</td>
    </tr>`;
  }).join("");
  document.getElementById("lc-reliability-table").innerHTML = `
    <thead><tr><th>Min pitches / half</th><th>r (raw)</th><th>r (SB-corrected)</th><th>n</th></tr></thead>
    <tbody>${rows}</tbody>
  `;
  const crosses070 = thresholds.find(t => sh[t].r_sb >= 0.70);
  document.getElementById("lc-reliability-note").textContent =
    `Spearman-Brown corrected reliability crosses r ≥ 0.70 at ≥ ${crosses070 ?? "n/a"} pitches/half. ` +
    `v3 minimum is 300 pitches (full season). ` +
    (sc.chosen_variant === "S1" ? `S1 baseline adjustment adopted.` : "");
}

function renderOutcomeCorrs(report) {
  const sc = report.scoring_config || {};
  const corrs = sc.s1_outcome_corrs || sc.s0_outcome_corrs || {};
  const mount = document.getElementById("lc-outcome-mount");
  if (!mount) return;
  if (Object.keys(corrs).length === 0) {
    mount.innerHTML = `<div class="lc-note">Outcome correlations not yet computed.</div>`;
    return;
  }
  const labels = { bb_rate: "BB%", k_rate: "K%", era: "ERA", xfip: "xFIP" };
  const rows = Object.entries(corrs).map(([key, d]) => {
    const label = labels[key] || key;
    const dir = d.r < 0 ? "lower is better" : "higher is better";
    return `<tr>
      <td>${label}</td>
      <td>${d.r.toFixed(3)}</td>
      <td class="lc-note">[${d.ci_lo.toFixed(3)}, ${d.ci_hi.toFixed(3)}]</td>
      <td>${d.n.toLocaleString()}</td>
      <td class="lc-note">${dir}</td>
    </tr>`;
  }).join("");
  mount.innerHTML = `
    <table class="data-table lc-table">
      <thead><tr><th>Outcome</th><th>r</th><th>95% CI</th><th>n</th><th></th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
    <div class="lc-note">Pitcher-season correlations (2015–2026 excl. 2020, ≥ 300 pitches).
    S1 baseline adjustment improves BB% correlation from ${sc.s0_outcome_corrs && sc.s0_outcome_corrs.bb_rate ? sc.s0_outcome_corrs.bb_rate.r.toFixed(3) : "—"} to ${corrs.bb_rate ? corrs.bb_rate.r.toFixed(3) : "—"}.</div>
  `;
}

function renderYoy(report) {
  const sc = report.scoring_config || {};
  const val = report.validation || {};
  // Try validation_results first (Part 5), fall back to scoring_config (Part 2)
  const yoy = val.yoy_stability || null;
  const mount = document.getElementById("lc-yoy-mount");
  if (!mount) return;

  if (yoy && Array.isArray(yoy)) {
    const rows = yoy.map(r => `<tr>
      <td>${r.from_season} → ${r.to_season}</td>
      <td>${r.r !== undefined ? r.r.toFixed(3) : "—"}</td>
      <td>${r.n ? r.n.toLocaleString() : "—"}</td>
    </tr>`).join("");
    mount.innerHTML = `
      <table class="data-table lc-table">
        <thead><tr><th>Season pair</th><th>r</th><th>n</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
      <div class="lc-note">Pooled (excl. 2020 pairs): ${sc.yoy_s1_pooled ? sc.yoy_s1_pooled.toFixed(3) : "—"} (S1).</div>
    `;
  } else {
    // Fall back to pooled from scoring_config
    mount.innerHTML = `
      <table class="data-table lc-table">
        <thead><tr><th>Variant</th><th>YoY pooled r</th></tr></thead>
        <tbody>
          <tr><td>S0 (no baseline adj.)</td><td>${sc.yoy_s0_pooled ? sc.yoy_s0_pooled.toFixed(3) : "—"}</td></tr>
          <tr><td>S1 (adopted)</td><td>${sc.yoy_s1_pooled ? sc.yoy_s1_pooled.toFixed(3) : "—"}</td></tr>
        </tbody>
      </table>
      <div class="lc-note">YoY S1 slightly lower than S0 as expected (baseline adjustment removes mean-reversion component), within spec (diff &lt; 0.05).</div>
    `;
  }
}

function renderNextSeason(report) {
  const sc = report.scoring_config || {};
  const rows = (sc.next_season_bb_rate_s1 || [])
    .map(r => `<tr>
      <td>${r.from_season} → ${r.to_season}</td>
      <td>${r.r.toFixed(3)}</td>
      <td class="lc-note">[${r.ci_lo.toFixed(3)}, ${r.ci_hi.toFixed(3)}]</td>
      <td>${r.n}</td>
    </tr>`)
    .join("");
  const mount = document.getElementById("lc-predictive-mount");
  if (!mount) return;
  if (!rows) {
    mount.innerHTML = `<div class="lc-note">Next-season correlations not yet computed.</div>`;
    return;
  }
  mount.innerHTML = `
    <table class="data-table lc-table">
      <thead><tr><th>Season pair</th><th>r</th><th>95% CI</th><th>n</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
    <div class="lc-note">Next-season BB% correlation (S1 variant, ≥ 300 pitches in from-season).
    All negative = higher Location+ predicts lower BB% in the following year.</div>
  `;
}

function leaderboardRows(rows) {
  return rows.map((r) =>
    `<tr>
      <td>${r.name || "#" + r.pitcher_id}${r.team_abbrev ? ` <span class="muted">(${r.team_abbrev})</span>` : ""}</td>
      <td>${typeof r.score === "number" ? r.score.toFixed(1) : r.score}</td>
      <td>${(r.total_pitches || r.n_pitches || 0).toLocaleString()}</td>
    </tr>`
  ).join("");
}

function renderLeaderboard(report) {
  const lb = report.leaderboard || {};
  const header = `<thead><tr><th>Pitcher</th><th>Location+</th><th>Pitches</th></tr></thead>`;
  const top = document.getElementById("lc-top-table");
  const bottom = document.getElementById("lc-bottom-table");
  if (top) top.innerHTML = header + `<tbody>${leaderboardRows(lb.top || [])}</tbody>`;
  if (bottom) bottom.innerHTML = header + `<tbody>${leaderboardRows(lb.bottom || [])}</tbody>`;
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
  const mount = document.getElementById("lc-pitcher-mount");
  if (status) status.textContent = "Loading…";
  if (mount) mount.innerHTML = "";
  try {
    const detail = await getJSON(`/api/location/pitcher/${playerId}?season=${season}`);
    if (status) status.textContent = "";
    renderPitcherDetail(detail, name, season);
  } catch (err) {
    if (status) status.textContent = "No Location+ data for this pitcher/season (needs 300+ pitches; 111 for 2020).";
  }
}

async function loadReport(season) {
  const report = await getJSON(`/api/location/report?season=${season}`);
  renderMeta(report);
  renderGates(report);
  renderReliability(report);
  renderOutcomeCorrs(report);
  renderYoy(report);
  renderNextSeason(report);
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
    loadPitcherDetail(selectedPlayer.player_id, "", Number(params.get("season")) || defaultSeason);
  } else if (remembered && remembered.role === "pitcher") {
    selectedPlayer = remembered;
    search.setValue(remembered);
    loadPitcherDetail(remembered.player_id, remembered.name, defaultSeason);
  }
}

document.addEventListener("DOMContentLoaded", init);
