// Career page — season-by-season stats for any player (v4)

let careerState = {
  playerId: null,
  playerName: null,
  role: null,
  data: null,          // last fetched API response
  selectedSeasons: new Set(), // selected season years
};

let _careerSearch = null;

// ── Formatting helpers ──────────────────────────────────────────
function fmt3(v) {
  if (v == null) return "—";
  const n = parseFloat(v);
  if (isNaN(n)) return "—";
  return n.toFixed(3).replace(/^0\./, ".");
}
function fmt1(v, suffix = "") {
  if (v == null) return "—";
  const n = parseFloat(v);
  if (isNaN(n)) return "—";
  return n.toFixed(1) + suffix;
}
function fmtInt(v) {
  if (v == null) return "—";
  const n = parseFloat(v);
  if (isNaN(n)) return "—";
  return Math.round(n).toString();
}
function fmtModel(v) {
  if (v == null) return null;
  return Math.round(parseFloat(v));
}

function fmtCell(row, col) {
  const v = row[col.key];
  const top10 = row.top10 && row.top10[col.key];
  const boldCls = top10 ? " ct-bold-top10" : "";

  if (col.key === "team") {
    const tid = row.team_id;
    const abbr = row.team_abbr || "—";
    if (!tid) return `<td class="ct-left"><span class="ct-team"><span class="ct-team-abbr">${abbr}</span></span></td>`;
    return `<td class="ct-left">
      <span class="ct-team">
        <img class="ct-team-logo" src="https://www.mlbstatic.com/team-logos/${tid}.svg"
             onerror="this.style.display='none'" alt="">
        <span class="ct-team-abbr">${abbr}</span>
      </span>
    </td>`;
  }

  if (col.key === "primary_position") {
    return `<td class="ct-pos">${v || "—"}</td>`;
  }

  if (col.key === "awards") {
    const txt = v || "—";
    const cls = txt.includes("AS") ? " ct-award-as" : "";
    return `<td class="ct-awards${cls}">${txt}</td>`;
  }

  if (v == null || v === "") return `<td class="ct-dash">—</td>`;

  if (["stuff_plus","tunneling_plus","location_plus","deception_plus"].includes(col.key)) {
    const n = parseFloat(v);
    const cls = n >= 110 ? "ct-good" : n <= 90 ? "ct-bad" : "";
    return `<td class="ct-score ${cls}${boldCls}">${Math.round(n)}</td>`;
  }
  if (["ops","woba","xwoba"].includes(col.key)) {
    return `<td class="${boldCls}">${fmt3(v)}</td>`;
  }
  if (["k_pct","bb_pct","whiff_pct","hard_hit_pct","zone_pct"].includes(col.key)) {
    return `<td class="${boldCls}">${fmt1(v, "%")}</td>`;
  }
  if (col.key === "xfip" || col.key === "era") {
    return `<td class="${boldCls}">${fmt1(v, "").length > 0 ? parseFloat(v).toFixed(2) : "—"}</td>`;
  }
  if (col.key === "avg_ev") return `<td class="${boldCls}">${fmt1(v)}</td>`;
  if (col.key === "wrc_plus") return `<td class="ct-score${boldCls}">${fmtInt(v)}</td>`;
  if (col.key === "war") return `<td class="${boldCls}">${fmt1(v)}</td>`;

  return `<td class="${boldCls}">${v}</td>`;
}

// ── Per-season average summary row ─────────────────────────────
function computePer162Row(rows, cols, role) {
  if (!rows.length) return null;
  const N = rows.length;

  const avg = (key) => {
    const vals = rows.map(r => parseFloat(r[key])).filter(v => !isNaN(v));
    return vals.length ? vals.reduce((s, v) => s + v, 0) / vals.length : null;
  };
  const wavg = (key, weightKey) => {
    let num = 0, den = 0;
    rows.forEach(r => {
      const w = parseFloat(r[weightKey]) || 0;
      const v = parseFloat(r[key]);
      if (!isNaN(v)) { num += v * w; den += w; }
    });
    return den ? num / den : null;
  };
  const sum = (key) => rows.reduce((s, r) => s + (parseFloat(r[key]) || 0), 0);

  if (role === "pitcher") {
    const totalIP = rows.reduce((s, r) => s + (r.ip_dec || 0), 0);
    const totalRA = sum("runs_allowed");
    const avgEra  = totalIP > 0 ? (totalRA / totalIP * 9).toFixed(2) : null;
    const xfip_w  = wavg("xfip", "ip_dec");

    const cell = {};
    cols.forEach(c => cell[c.key] = null);
    Object.assign(cell, {
      _label:         "Avg/Season",
      gs:             Math.round(avg("gs")),
      ip:             fmt1(totalIP / N),
      k:              Math.round(avg("k")),
      bb:             Math.round(avg("bb")),
      hr:             Math.round(avg("hr")),
      era:            avgEra,
      xfip:           xfip_w != null ? xfip_w.toFixed(2) : null,
      k_pct:          wavg("k_pct", "gs"),
      bb_pct:         wavg("bb_pct", "gs"),
      whiff_pct:      wavg("whiff_pct", "gs"),
      zone_pct:       wavg("zone_pct", "gs"),
      stuff_plus:     avg("stuff_plus") != null ? Math.round(avg("stuff_plus")) : null,
      tunneling_plus: avg("tunneling_plus") != null ? Math.round(avg("tunneling_plus")) : null,
      location_plus:  avg("location_plus") != null ? Math.round(avg("location_plus")) : null,
      deception_plus: avg("deception_plus") != null ? Math.round(avg("deception_plus")) : null,
      war:            fmt1(avg("war")),
    });
    return cell;
  } else {
    const cell = {};
    cols.forEach(c => cell[c.key] = null);
    Object.assign(cell, {
      _label:       "Avg/Season",
      gp:           Math.round(avg("gp")),
      pa:           Math.round(avg("pa")),
      hr:           Math.round(avg("hr")),
      k:            Math.round(avg("k")),
      bb:           Math.round(avg("bb")),
      ops:          wavg("ops","pa") != null ? wavg("ops","pa").toFixed(3).replace(/^0\./, ".") : null,
      woba:         wavg("woba","pa") != null ? wavg("woba","pa").toFixed(3).replace(/^0\./, ".") : null,
      xwoba:        wavg("xwoba","pa") != null ? wavg("xwoba","pa").toFixed(3).replace(/^0\./, ".") : null,
      wrc_plus:     wavg("wrc_plus","pa") != null ? Math.round(wavg("wrc_plus","pa")) : null,
      avg_ev:       wavg("avg_ev","pa") != null ? fmt1(wavg("avg_ev","pa")) : null,
      hard_hit_pct: wavg("hard_hit_pct","pa"),
      k_pct:        wavg("k_pct","pa"),
      bb_pct:       wavg("bb_pct","pa"),
      whiff_pct:    wavg("whiff_pct","pa"),
      war:          fmt1(avg("war")),
    });
    return cell;
  }
}

function renderSummaryRow(summaryCell, cols) {
  if (!summaryCell) return "";
  const tds = cols.map(c => {
    const gcls = c._groupStart ? " ct-group-start" : "";
    if (c.key === "season") return `<td class="ct-left">${summaryCell._label || "Avg/Season"}</td>`;
    if (c.key === "team" || c.key === "awards" || c.key === "primary_position") return `<td></td>`;
    const v = summaryCell[c.key];
    if (v == null) return `<td class="ct-dash"></td>`;
    if (["k_pct","bb_pct","whiff_pct","hard_hit_pct","zone_pct"].includes(c.key)) {
      return `<td class="${gcls}">${fmt1(v, "%")}</td>`;
    }
    return `<td class="${gcls}">${v}</td>`;
  }).join("");
  return `<tr class="ct-summary-row">${tds}</tr>`;
}

// ── Table builder ───────────────────────────────────────────────
function buildCareerTable(data) {
  if (!data.rows || !data.rows.length) {
    return `<div class="empty-state">No data found for this player.</div>`;
  }

  const cols = data.columns;
  const groups = data.column_groups || [];

  // Annotate each column with group-start flag
  const groupOrder = groups.map(g => g.group);
  let lastGroup = null;
  cols.forEach(c => {
    c._groupStart = c.group !== lastGroup && lastGroup !== null && groupOrder.indexOf(c.group) > 0;
    if (c.group !== "id") lastGroup = c.group;
    if (c.group === "id" && lastGroup === null) lastGroup = c.group;
  });
  // Also mark first non-id group
  let firstNonId = cols.find(c => c.group !== "id");
  if (firstNonId) firstNonId._groupStart = true;

  // Group header row
  const groupCells = [];
  const groupSpan = {};
  cols.forEach(c => {
    const g = c.group;
    groupSpan[g] = (groupSpan[g] || 0) + 1;
  });
  let processedGroups = new Set();
  cols.forEach(c => {
    const g = c.group;
    if (!processedGroups.has(g)) {
      const grpMeta = groups.find(gm => gm.group === g);
      const label = (grpMeta && grpMeta.label) ? grpMeta.label : "";
      const gcls = c._groupStart ? " ct-group-start" : "";
      groupCells.push(`<th colspan="${groupSpan[g]}" class="${gcls}">${label}</th>`);
      processedGroups.add(g);
    }
  });

  // Column header row
  const colHeaders = cols.map(c => {
    const gcls = c._groupStart ? " ct-group-start" : "";
    const leftCls = (c.key === "season") ? " ct-left" : "";
    return `<th class="${gcls}${leftCls}">${c.label}</th>`;
  }).join("");

  const thead = `<thead>
    <tr class="ct-group-row">${groupCells.join("")}</tr>
    <tr class="ct-col-row">${colHeaders}</tr>
  </thead>`;

  const tbody_rows = data.rows.map((row, i) => {
    const sel = careerState.selectedSeasons.has(row.season) ? " ct-selected" : "";
    const tds = cols.map(c => {
      const gcls = c._groupStart ? " ct-group-start" : "";
      const cell = fmtCell(row, c);
      // inject group-start class into the td
      return cell.replace(/^<td/, `<td data-g="${gcls}"`).replace(`data-g="${gcls}"`, gcls ? `class="${gcls.trim()}"` : "");
    }).join("");
    return `<tr class="ct-row${sel}" data-season="${row.season}" data-idx="${i}">${tds}</tr>`;
  }).join("");

  // Per-162 summary row
  const summaryCell = computePer162Row(data.rows, cols, data.role);
  const summaryRow = renderSummaryRow(summaryCell, cols);

  return `<div class="ct-scroll"><table class="career-table">
    ${thead}
    <tbody>${tbody_rows}${summaryRow}</tbody>
  </table></div>`;
}

// ── Career aggregate summary box ───────────────────────────────
function updateSummaryBox(data, name, selectedSeasons) {
  const el = document.getElementById("career-summary");
  if (!data || !data.rows.length) { el.style.display = "none"; return; }
  el.style.display = "";

  const rows = selectedSeasons.size
    ? data.rows.filter(r => selectedSeasons.has(r.season))
    : data.rows;

  document.getElementById("cs-name").textContent = name || "";

  const note = selectedSeasons.size
    ? `${selectedSeasons.size} season${selectedSeasons.size > 1 ? "s" : ""} selected — totals and career avgs`
    : "Career totals — click rows to select specific seasons";
  document.getElementById("cs-note").textContent = note;

  const sum = (key) => rows.reduce((s, r) => s + (parseFloat(r[key]) || 0), 0);
  const wavg = (key, wkey) => {
    let n = 0, d = 0;
    rows.forEach(r => { const w = parseFloat(r[wkey]) || 0; if (r[key] != null) { n += r[key] * w; d += w; } });
    return d ? n / d : null;
  };

  let stats = [];
  if (data.role === "pitcher") {
    const totalGS  = sum("gs");
    const totalK   = sum("k");
    const totalBB  = sum("bb");
    const totalHR  = sum("hr");
    const totalIP  = rows.reduce((s, r) => s + (r.ip_dec || 0), 0);
    const totalRA  = sum("runs_allowed");
    const era      = totalIP > 0 ? (totalRA / totalIP * 9).toFixed(2) : "—";
    const xfip_w   = wavg("xfip", "ip_dec");
    const war_sum  = sum("war");

    stats = [
      { label: "G",      value: Math.round(totalGS) },
      { label: "IP",     value: fmt1(totalIP) },
      { label: "K",      value: Math.round(totalK) },
      { label: "BB",     value: Math.round(totalBB) },
      { label: "HR",     value: Math.round(totalHR) },
      { label: "ERA*",   value: era },
      { label: "xFIP",   value: xfip_w != null ? xfip_w.toFixed(2) : "—" },
      { label: "WAR",    value: fmt1(war_sum) },
      { label: "K%",     value: wavg("k_pct","gs") != null ? fmt1(wavg("k_pct","gs"), "%") : "—" },
    ];
  } else {
    const totalG   = sum("gp");
    const totalPA  = sum("pa");
    const totalHR  = sum("hr");
    const totalK   = sum("k");
    const totalBB  = sum("bb");
    const war_sum  = sum("war");

    stats = [
      { label: "G",      value: Math.round(totalG) },
      { label: "PA",     value: Math.round(totalPA) },
      { label: "HR",     value: Math.round(totalHR) },
      { label: "K",      value: Math.round(totalK) },
      { label: "BB",     value: Math.round(totalBB) },
      { label: "OPS",    value: wavg("ops","pa") != null ? fmt3(wavg("ops","pa")) : "—" },
      { label: "wOBA",   value: wavg("woba","pa") != null ? fmt3(wavg("woba","pa")) : "—" },
      { label: "wRC+",   value: wavg("wrc_plus","pa") != null ? Math.round(wavg("wrc_plus","pa")) : "—" },
      { label: "WAR",    value: fmt1(war_sum) },
    ];
  }

  document.getElementById("cs-stats").innerHTML = stats.map(s =>
    `<div class="cs-stat">
      <div class="cs-label">${s.label}</div>
      <div class="cs-value">${s.value ?? "—"}</div>
    </div>`
  ).join("");
}

// ── Row selection ───────────────────────────────────────────────
function bindRowSelection() {
  document.querySelectorAll(".career-table tbody tr.ct-row").forEach(tr => {
    tr.addEventListener("click", () => {
      const season = parseInt(tr.dataset.season, 10);
      if (careerState.selectedSeasons.has(season)) {
        careerState.selectedSeasons.delete(season);
        tr.classList.remove("ct-selected");
      } else {
        careerState.selectedSeasons.add(season);
        tr.classList.add("ct-selected");
      }
      updateSummaryBox(careerState.data, careerState.playerName, careerState.selectedSeasons);
    });
  });
}

// ── Main render ─────────────────────────────────────────────────
async function renderCareer() {
  if (!careerState.playerId) return;
  const mount = document.getElementById("career-mount");
  const title = document.getElementById("career-title");
  mount.innerHTML = `<div class="empty-state">Loading…</div>`;
  title.textContent = "";
  careerState.selectedSeasons.clear();

  try {
    const data = await getJSON(`/api/players/${careerState.playerId}/career`);
    careerState.data = data;
    careerState.role = data.role;

    const roleLabel = data.role === "pitcher" ? "Pitcher" : "Batter";
    title.innerHTML = `${careerState.playerName || "Player"} — Career Stats
      <span class="career-sel-note">Click rows to select seasons</span>`;

    mount.innerHTML = buildCareerTable(data);
    document.getElementById("career-note").style.display = "";
    bindRowSelection();
    updateSummaryBox(data, careerState.playerName, careerState.selectedSeasons);
  } catch (err) {
    mount.innerHTML = `<div class="empty-state">Error: ${err.message}</div>`;
  }
}

// ── Init ─────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", async () => {
  const inputEl    = document.getElementById("career-player-input");
  const dropdownEl = document.getElementById("career-player-dropdown");

  _careerSearch = createPlayerSearch(inputEl, dropdownEl, (p) => {
    careerState.playerId   = p.player_id;
    careerState.playerName = p.name;
    renderCareer();
  });  // No roleFilter → all players

  const meta = await getJSON("/api/meta").catch(() => null);
  const season = (meta && meta.season) || 2026;
  await _careerSearch.load(season);

  // Seed from PlayerContext if available
  if (typeof PlayerContext !== "undefined") {
    const cur = PlayerContext.getCurrent();
    if (cur && cur.player_id) {
      careerState.playerId   = cur.player_id;
      careerState.playerName = cur.name;
      await _careerSearch.load(season);
      if (_careerSearch) {
        _careerSearch.selectDefault(p => p.player_id === cur.player_id);
      }
      renderCareer();
    }
  }
});
