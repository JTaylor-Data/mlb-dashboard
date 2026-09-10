// Outing Breakdown — standalone page (outings.html)

(function () {

  // ── Pitch colours ─────────────────────────────────────────────────────────
  function pitchColor(pt) {
    if (typeof window.pitchColor === "function") return window.pitchColor(pt);
    const MAP = {
      FF:"#e85252", SI:"#e07020", FC:"#b060d0", SL:"#e6a020",
      ST:"#f0c040", CU:"#3090e0", CH:"#40b060", FS:"#50b0b0",
      KC:"#4070d0", SV:"#c050a0", FA:"#e85252",
    };
    return MAP[pt] || "#888";
  }

  // ── Format helpers ────────────────────────────────────────────────────────
  function fmt(v, dec = 1) { return v != null ? (+v).toFixed(dec) : "—"; }
  function fmtPct(v)       { return v != null ? `${(+v).toFixed(1)}%` : "—"; }

  // Delta arrow: positive = up (green), negative = down (red).
  // Threshold 0.05 keeps near-zero deltas from showing arrows.
  function deltaHtml(d) {
    if (d == null) return `<span class="delta delta-zero">—</span>`;
    const abs = Math.abs(d);
    if (abs < 0.05) return `<span class="delta delta-zero">±0</span>`;
    const cls = d > 0 ? "delta-up" : "delta-down";
    const arrow = d > 0 ? "▲" : "▼";
    const disp = abs < 1 ? abs.toFixed(1) : abs.toFixed(0);
    return `<span class="delta ${cls}">${arrow}${disp}</span>`;
  }

  // pfx_x / pfx_z come from the DB in feet; convert to inches for display.
  function toIn(v) { return v != null ? +(v * 12).toFixed(1) : null; }
  // Drop with gravity (inches): gravity contribution ≈ 11.4 in; IVB removes some of that.
  // drop = 11.4 - ivb_in   (positive = net downward, large for curveballs)
  function toDrop(ivb_ft) { return ivb_ft != null ? +(11.4 - ivb_ft * 12).toFixed(1) : null; }

  // ── URL params ────────────────────────────────────────────────────────────
  function urlParam(k) { return new URLSearchParams(window.location.search).get(k); }

  // ── State ─────────────────────────────────────────────────────────────────
  let playerId   = null;
  let season     = 2026;
  let activeSide = "main";
  let lastData   = null;

  // ── DOM refs ──────────────────────────────────────────────────────────────
  const selGroup     = () => document.getElementById("outing-select-group");
  const viewGroup    = () => document.getElementById("outing-viewing-group");
  const sel          = () => document.getElementById("outing-select");
  const body         = () => document.getElementById("outing-body");
  const toggleMain   = () => document.getElementById("toggle-main-pitcher");
  const toggleOpp    = () => document.getElementById("toggle-opp-pitcher");

  // ── Player search ─────────────────────────────────────────────────────────
  async function initSearch() {
    const inputEl    = document.getElementById("outing-player-search");
    const dropdownEl = document.getElementById("outing-player-dropdown");
    const search = createPlayerSearch(inputEl, dropdownEl, async (p) => {
      playerId = p.player_id;
      await loadOutingList();
    }, { roleFilter: "pitcher" });
    await search.load(season);

    const pid = urlParam("player_id");
    if (pid) {
      playerId = Number(pid);
      const p = search.selectDefault ? search.selectDefault((pl) => pl.player_id === Number(pid)) : null;
      if (p) { inputEl.value = p.name || ""; }
      await loadOutingList();
    } else {
      // Auto-load from PlayerContext (when navigated from Player tab)
      const ctx = typeof PlayerContext !== "undefined" && PlayerContext.getCurrent();
      if (ctx && ctx.role === "pitcher") {
        playerId = ctx.player_id;
        const p = search.selectDefault ? search.selectDefault((pl) => pl.player_id === ctx.player_id) : null;
        if (p) { inputEl.value = p.name || ""; }
        await loadOutingList();
      }
    }
  }

  // ── Load outing list for current playerId ────────────────────────────────
  async function loadOutingList() {
    if (!playerId) return;
    const s = sel();
    s.innerHTML = `<option value="">Loading…</option>`;
    selGroup().style.display = "";
    viewGroup().style.display = "none";
    body().innerHTML = "";
    s.onchange = () => { if (s.value) renderOuting(Number(s.value)); };

    try {
      const outings = await getJSON(`/api/players/${playerId}/outings?season=${season}`);
      if (!outings.length) {
        s.innerHTML = `<option value="">No outings found</option>`;
        return;
      }
      s.innerHTML = outings
        .map(o => `<option value="${o.game_id}">${o.label}</option>`)
        .join("");

      const targetGameId = urlParam("game_id");
      const match = targetGameId && outings.find(o => String(o.game_id) === String(targetGameId));
      s.value = match ? match.game_id : outings[0].game_id;
      await renderOuting(Number(s.value));
    } catch (err) {
      s.innerHTML = `<option value="">Error: ${err.message}</option>`;
    }
  }

  // ── Render one outing ─────────────────────────────────────────────────────
  async function renderOuting(gameId) {
    activeSide = "main";
    body().innerHTML = `<div class="empty-state">Loading…</div>`;
    viewGroup().style.display = "none";
    setPageLoading(true);
    try {
      const data = await getJSON(
        `/api/players/${playerId}/outing?game_id=${gameId}&season=${season}`
      );
      lastData = data;

      if (!data.pitch_types || !data.pitch_types.length) {
        body().innerHTML = `<div class="empty-state">No pitch data for this outing.</div>`;
        return;
      }

      setupViewingToggle(data);
      body().innerHTML = buildOutingHtml(data);
    } catch (err) {
      body().innerHTML = `<div class="empty-state">Error: ${err.message}</div>`;
    } finally {
      setPageLoading(false);
    }
  }

  // ── Viewing toggle ────────────────────────────────────────────────────────
  function setupViewingToggle(data) {
    const main = toggleMain();
    const opp  = toggleOpp();
    const grp  = viewGroup();

    main.textContent = data.player_name || "This Pitcher";
    activeSide = "main";
    main.classList.add("active");
    opp.classList.remove("active");

    if (data.opp_pitcher) {
      opp.textContent = data.opp_pitcher.name;
      opp.style.display = "";
      grp.style.display = "";

      main.onclick = () => {
        if (activeSide === "main") return;
        activeSide = "main";
        main.classList.add("active");
        opp.classList.remove("active");
        body().innerHTML = buildOutingHtml(lastData);
      };

      opp.onclick = async () => {
        if (activeSide === "opp") return;
        activeSide = "opp";
        opp.classList.add("active");
        main.classList.remove("active");
        body().innerHTML = `<div class="empty-state">Loading…</div>`;
        setPageLoading(true);
        try {
          const oppData = await getJSON(
            `/api/players/${data.opp_pitcher.player_id}/outing?game_id=${data.game_id}&season=${season}`
          );
          body().innerHTML = buildOutingHtml(oppData);
        } catch (err) {
          body().innerHTML = `<div class="empty-state">Error: ${err.message}</div>`;
        } finally {
          setPageLoading(false);
        }
      };
    } else {
      opp.style.display = "none";
      grp.style.display = "none";
    }
  }

  // ── Build HTML ────────────────────────────────────────────────────────────
  function buildOutingHtml(data) {
    return buildOutingHero(data)
         + buildUsageVeloStrikesTable(data.pitch_types, data.stat_line)
         + buildMovementSpinTable(data.pitch_types)
         + buildAtBatSequences(data);
  }

  // ── Hero header ───────────────────────────────────────────────────────────
  function buildOutingHero(data) {
    const pid  = data.player_id;
    const tid  = data.player_team_id;
    const headshot = pid
      ? `<img class="outing-headshot" loading="lazy"
             src="https://img.mlbstatic.com/mlb-photos/image/upload/d_people:generic:headshot:67:current.png/w_213,q_auto:best/v1/people/${pid}/headshot/67/current"
             onerror="this.style.display='none'" />`
      : "";
    const teamLogo = tid
      ? `<img class="outing-team-logo"
             src="https://www.mlbstatic.com/team-logos/${tid}.svg"
             onerror="this.style.display='none'" />`
      : "";
    const away = data.away_team || "", home = data.home_team || "";
    const meta = data.game_date ? `${data.game_date} · ${away} @ ${home}` : "";

    const avgVeloBox = data.avg_velo != null ? heroBox("Avg Velo", `${data.avg_velo.toFixed(1)} mph`) : "";
    const maxVeloBox = data.max_velo != null ? heroBox("Max Velo", `${data.max_velo.toFixed(1)} mph`) : "";
    const avgEvBox   = data.avg_ev   != null ? heroBox("Avg EV",   `${data.avg_ev.toFixed(1)} mph`)   : "";
    const maxEvBox   = data.max_ev   != null ? heroBox("Max EV",   `${data.max_ev.toFixed(1)} mph`)   : "";

    return `<div class="outing-hero">
      <div class="outing-hero-left">
        ${headshot}
        ${teamLogo}
        <div>
          <div class="outing-player-name">${data.player_name || ""}</div>
          <div class="outing-meta">${meta}</div>
        </div>
      </div>
      <div class="outing-hero-right">${avgVeloBox}${maxVeloBox}${avgEvBox}${maxEvBox}</div>
    </div>`;
  }

  function heroBox(label, val) {
    return `<div class="stat-box">
      <div class="stat-box-label">${label}</div>
      <div class="stat-box-value" style="font-size:16px;">${val}</div>
    </div>`;
  }

  // ── Table 1: Usage, Velocity & Strikes ────────────────────────────────────
  function buildUsageVeloStrikesTable(pts, sl) {
    const totalN      = pts.reduce((s, p) => s + p.count, 0);
    const totalRHH    = pts.reduce((s, p) => s + (p.rhh_count || 0), 0);
    const totalLHH    = pts.reduce((s, p) => s + (p.lhh_count || 0), 0);
    const totalSwings = pts.reduce((s, p) => s + (p.swings  || 0), 0);
    const totalCS     = pts.reduce((s, p) => s + (p.called_strikes || 0), 0);
    const totalWhiffs = sl?.total_whiffs ?? pts.reduce((s, p) => s + (p.whiffs || 0), 0);
    const totalK      = sl?.k ?? pts.reduce((s, p) => s + (p.k_count || 0), 0);

    const rows = pts.map(pt => {
      const color = pitchColor(pt.pitch_type);
      const whiff     = pt.swings  ? `${pt.whiffs ?? 0}/${pt.swings} (${((pt.whiffs ?? 0) / pt.swings * 100).toFixed(1)}%)` : "—";
      const csw       = (pt.called_strikes || 0) + (pt.whiffs || 0);
      const cswCombined = `${csw}/${pt.count} (${(csw / pt.count * 100).toFixed(1)}%)`;
      const rhh = pt.rhh_count != null ? `${pt.rhh_count} (${(pt.rhh_count / pt.count * 100).toFixed(0)}%)` : "—";
      const lhh = pt.lhh_count != null ? `${pt.lhh_count} (${(pt.lhh_count / pt.count * 100).toFixed(0)}%)` : "—";
      return `<tr>
        <td><span class="pt-chip" style="color:${color};">${pt.pitch_type}</span></td>
        <td>${pt.count}</td>
        <td>${pt.usage_pct != null ? pt.usage_pct.toFixed(0)+"%" : "—"}</td>
        <td>${pt.yr_usage_pct != null ? pt.yr_usage_pct.toFixed(0)+"%" : "—"}</td>
        <td>${rhh}</td>
        <td>${lhh}</td>
        <td>${fmt(pt.avg_velo)}</td>
        <td>${deltaHtml(pt.velo_delta)}</td>
        <td>${pt.k_count ?? 0}</td>
        <td>${pt.swings ?? 0}</td>
        <td>${pt.called_strikes ?? 0}</td>
        <td>${whiff}</td>
        <td>${cswCombined}</td>
      </tr>`;
    }).join("");

    // Weighted avg velocity for ALL row
    const veloValid = pts.filter(p => p.avg_velo != null);
    const allVelo = veloValid.length
      ? veloValid.reduce((s, p) => s + p.avg_velo * p.count, 0) / veloValid.reduce((s, p) => s + p.count, 0)
      : null;
    const allWhiff     = totalSwings ? `${totalWhiffs}/${totalSwings} (${(totalWhiffs / totalSwings * 100).toFixed(1)}%)` : "—";
    const allCSW       = totalCS + totalWhiffs;
    const allCSWCombined = `${allCSW}/${totalN} (${(allCSW / totalN * 100).toFixed(1)}%)`;
    const allRHH = `${totalRHH} (${totalN ? (totalRHH / totalN * 100).toFixed(0) : 0}%)`;
    const allLHH = `${totalLHH} (${totalN ? (totalLHH / totalN * 100).toFixed(0) : 0}%)`;

    const allRow = `<tr class="outing-all-row">
      <td><strong>ALL</strong></td>
      <td>${totalN}</td><td>100%</td><td>—</td>
      <td>${allRHH}</td><td>${allLHH}</td>
      <td>${allVelo != null ? allVelo.toFixed(1) : "—"}</td><td>—</td>
      <td>${totalK}</td><td>${totalSwings}</td><td>${totalCS}</td>
      <td>${allWhiff}</td>
      <td>${allCSWCombined}</td>
    </tr>`;

    return `<div class="panel-card" style="margin-bottom:18px;">
      <div class="panel-title">Usage, Velocity &amp; Strikes</div>
      <div style="overflow-x:auto;">
        <table class="outing-table">
          <thead><tr>
            <th style="text-align:left;">Pitch</th>
            <th>n</th><th>%</th><th>Yr%</th>
            <th>vs RHH</th><th>vs LHH</th>
            <th>Velo</th><th title="vs. this pitcher's season avg">±Velo</th>
            <th>K</th><th>SW</th><th>CS</th>
            <th>Whiff (W%)</th>
            <th>CS+W (CSW%)</th>
          </tr></thead>
          <tbody>${rows}${allRow}</tbody>
        </table>
      </div>
    </div>`;
  }

  // ── Table 2: Movement & Spin ───────────────────────────────────────────────
  function buildMovementSpinTable(pts) {
    function wtAvg(getter) {
      const valid = pts.filter(p => getter(p) != null);
      if (!valid.length) return null;
      const w = valid.reduce((s, p) => s + p.count, 0);
      return valid.reduce((s, p) => s + getter(p) * p.count, 0) / w;
    }

    const rows = pts.map(pt => {
      const color      = pitchColor(pt.pitch_type);
      const avgDrop    = toDrop(pt.avg_ivb);
      // Drop increases when IVB decreases (and vice versa): Δdrop = -Δivb_in
      const dropDelta  = pt.ivb_delta != null ? +(-pt.ivb_delta * 12).toFixed(1) : null;
      const avgIvbIn   = toIn(pt.avg_ivb);
      const ivbDeltaIn = toIn(pt.ivb_delta);
      const avgHbIn    = toIn(pt.avg_hb);
      const hbDeltaIn  = toIn(pt.hb_delta);
      const spin       = pt.avg_spin != null ? Math.round(pt.avg_spin) : null;
      return `<tr>
        <td><span class="pt-chip" style="color:${color};">${pt.pitch_type}</span></td>
        <td>${avgDrop ?? "—"}</td>
        <td>${deltaHtml(dropDelta)}</td>
        <td>${avgIvbIn ?? "—"}</td>
        <td>${deltaHtml(ivbDeltaIn)}</td>
        <td>${avgHbIn ?? "—"}</td>
        <td>${deltaHtml(hbDeltaIn)}</td>
        <td>${spin ?? "—"}</td>
        <td>${deltaHtml(pt.spin_delta)}</td>
      </tr>`;
    }).join("");

    const allIvb  = wtAvg(p => p.avg_ivb);
    const allHb   = wtAvg(p => p.avg_hb);
    const allSpin = wtAvg(p => p.avg_spin);

    const allRow = `<tr class="outing-all-row">
      <td><strong>ALL</strong></td>
      <td>${toDrop(allIvb) ?? "—"}</td><td>—</td>
      <td>${toIn(allIvb)   ?? "—"}</td><td>—</td>
      <td>${toIn(allHb)    ?? "—"}</td><td>—</td>
      <td>${allSpin != null ? Math.round(allSpin) : "—"}</td><td>—</td>
    </tr>`;

    return `<div class="panel-card">
      <div class="panel-title">Movement &amp; Spin</div>
      <div style="overflow-x:auto;">
        <table class="outing-table">
          <thead><tr>
            <th style="text-align:left;">Pitch</th>
            <th>Drop (in)</th><th title="vs. season avg">±Drop</th>
            <th>IVB (in)</th><th title="vs. season avg">±IVB</th>
            <th>HB (in)</th><th title="vs. season avg">±HB</th>
            <th>Spin (rpm)</th><th title="vs. season avg">±Spin</th>
          </tr></thead>
          <tbody>${rows}${allRow}</tbody>
        </table>
      </div>
    </div>`;
  }

  // ── At-bat sequence chart ─────────────────────────────────────────────────
  const RESULT_CLASS = {
    "K":"res-k","KDP":"res-k","BB":"res-bb","IBB":"res-bb","HBP":"res-bb",
    "HR":"res-hr","1B":"res-hit","2B":"res-hit","3B":"res-hit",
    "OUT":"res-out","DP":"res-out","FC":"res-out","SF":"res-out","SH":"res-out",
  };
  const DESC_LABEL = {
    "called_strike":"CS","swinging_strike":"SS","swinging_strike_blocked":"SS",
    "foul":"F","foul_tip":"FT","foul_bunt":"FB",
    "ball":"B","blocked_ball":"B","pitchout":"PO",
    "hit_by_pitch":"HBP",
    "hit_into_play":"HIP","hit_into_play_no_out":"HIP","hit_into_play_score":"HIP",
  };

  function buildAtBatSequences(data) {
    const seqs = data.sequences;
    if (!seqs || !seqs.length) return "";

    // Group by inning
    const byInning = {};
    seqs.forEach(ab => {
      if (!byInning[ab.inn]) byInning[ab.inn] = [];
      byInning[ab.inn].push(ab);
    });

    const innings = Object.keys(byInning).sort((a, b) => +a - +b);

    const inningSections = innings.map(inn => {
      const abs = byInning[inn];
      const rows = abs.map(ab => {
        const dots = ab.pitches.map((p, i) => {
          const col = pitchColor(p.pt);
          const lbl = DESC_LABEL[p.d] || "•";
          const tip = `${p.pt}${p.v ? " " + p.v.toFixed(1) : ""} — ${p.d || "?"}`;
          const isLast = i === ab.pitches.length - 1;
          return `<span class="seq-dot${isLast ? " seq-dot-last" : ""}" style="--dc:${col}" title="${tip}">
            <span class="seq-dot-circle" style="background:${col}"></span>
            <span class="seq-dot-lbl">${lbl}</span>
          </span>`;
        }).join("");
        const resCls = RESULT_CLASS[ab.result] || "res-out";
        const handIcon = ab.stand === "L" ? "L" : "R";
        return `<div class="seq-row">
          <span class="seq-hand ${ab.stand === "L" ? "seq-hand-l" : ""}">${handIcon}</span>
          <span class="seq-dots">${dots}</span>
          <span class="seq-result ${resCls}">${ab.result || "?"}</span>
        </div>`;
      }).join("");
      return `<div class="seq-inning">
        <div class="seq-inning-label">INN ${inn}</div>
        <div class="seq-abs">${rows}</div>
      </div>`;
    }).join("");

    // Build pitch type legend from types used in this outing
    const usedTypes = [...new Set(seqs.flatMap(ab => ab.pitches.map(p => p.pt)))];
    const legend = usedTypes.map(pt =>
      `<span class="seq-legend-item"><span class="seq-legend-dot" style="background:${pitchColor(pt)}"></span>${pt}</span>`
    ).join("");

    return `<div class="panel-card" style="margin-top:18px;">
      <div class="panel-title" style="margin-bottom:10px;">Pitch Sequence by At-Bat</div>
      <div class="seq-legend">${legend}</div>
      <div class="seq-grid">${inningSections}</div>
    </div>`;
  }

  // ── Init ──────────────────────────────────────────────────────────────────
  document.addEventListener("DOMContentLoaded", initSearch);

  window.loadOutings = function () {};

})();
