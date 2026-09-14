// Career page — season-by-season stats for any player

let careerState = {
  role: "pitcher",
  playerId: null,
  playerName: null,
};

let _careerSearch = null;

function fmtCell(v, col) {
  if (v == null || v === "") return `<td class="ct-dash">—</td>`;
  if (["stuff_plus","tunneling_plus","location_plus","deception_plus"].includes(col)) {
    const n = parseFloat(v);
    const cls = n >= 110 ? "ct-good" : n <= 90 ? "ct-bad" : "";
    return `<td class="ct-score ${cls}">${Math.round(n)}</td>`;
  }
  if (["ops","woba","xwoba"].includes(col)) {
    const n = parseFloat(v);
    if (isNaN(n)) return `<td class="ct-dash">—</td>`;
    return `<td>${n.toFixed(3).replace(/^0\./, ".")}</td>`;
  }
  if (["avg","obp","slg"].includes(col)) {
    const n = parseFloat(v);
    if (isNaN(n)) return `<td class="ct-dash">—</td>`;
    return `<td>${n.toFixed(3).replace(/^0\./, ".")}</td>`;
  }
  if (["k_pct","bb_pct","whiff_pct","hard_hit_pct"].includes(col)) {
    const n = parseFloat(v);
    if (isNaN(n)) return `<td class="ct-dash">—</td>`;
    return `<td>${n.toFixed(1)}%</td>`;
  }
  if (["xfip"].includes(col)) {
    const n = parseFloat(v);
    if (isNaN(n)) return `<td class="ct-dash">—</td>`;
    return `<td>${n.toFixed(2)}</td>`;
  }
  if (["avg_ev"].includes(col)) {
    const n = parseFloat(v);
    if (isNaN(n)) return `<td class="ct-dash">—</td>`;
    return `<td>${n.toFixed(1)}</td>`;
  }
  return `<td>${v}</td>`;
}

function buildCareerTable(data) {
  if (!data.rows || !data.rows.length) {
    return `<div class="empty-state">No data found for this player.</div>`;
  }
  const cols = data.columns;
  const thead = `<thead><tr>${cols.map(c => `<th>${c.label}</th>`).join("")}</tr></thead>`;
  const tbody = `<tbody>${data.rows.map(row => `
    <tr class="${row.season % 2 === 0 ? "ct-even" : ""}">
      ${cols.map(c => fmtCell(row[c.key], c.key)).join("")}
    </tr>`).join("")}
  </tbody>`;
  return `<div class="ct-scroll"><table class="career-table">${thead}${tbody}</table></div>`;
}

async function renderCareer() {
  if (!careerState.playerId) return;
  const mount = document.getElementById("career-mount");
  const title = document.getElementById("career-title");
  mount.innerHTML = `<div class="empty-state">Loading…</div>`;
  title.textContent = "";
  try {
    const data = await getJSON(
      `/api/players/${careerState.playerId}/career?role=${careerState.role}`
    );
    title.textContent = `${careerState.playerName || "Player"} — Career Stats`;
    mount.innerHTML = buildCareerTable(data);
  } catch (err) {
    mount.innerHTML = `<div class="empty-state">Error: ${err.message}</div>`;
  }
}

async function initCareerSearch(role) {
  if (_careerSearch) _careerSearch.destroy();
  const inputEl = document.getElementById("career-player-input");
  const dropdownEl = document.getElementById("career-player-dropdown");
  _careerSearch = createPlayerSearch(inputEl, dropdownEl, (p) => {
    careerState.playerId = p.player_id;
    careerState.playerName = p.name;
    careerState.role = p.role || role;
    renderCareer();
  }, { roleFilter: role === "pitcher" ? "pitcher" : "batter" });

  const meta = await getJSON("/api/meta").catch(() => null);
  const season = (meta && meta.season) || 2026;
  await _careerSearch.load(season);
}

document.addEventListener("DOMContentLoaded", async () => {
  // Role toggle
  document.querySelectorAll(".career-role-btn").forEach(btn => {
    btn.addEventListener("click", async () => {
      careerState.role = btn.dataset.role;
      careerState.playerId = null;
      careerState.playerName = null;
      document.getElementById("career-player-input").value = "";
      document.getElementById("career-mount").innerHTML = "";
      document.getElementById("career-title").textContent = "";
      document.querySelectorAll(".career-role-btn").forEach(b =>
        b.classList.toggle("active", b === btn));
      await initCareerSearch(careerState.role);
    });
  });

  await initCareerSearch(careerState.role);

  // Seed from PlayerContext if available
  if (typeof PlayerContext !== "undefined") {
    const cur = PlayerContext.getCurrent();
    if (cur && cur.player_id) {
      careerState.playerId = cur.player_id;
      careerState.playerName = cur.name;
      careerState.role = cur.role || "pitcher";
      document.querySelectorAll(".career-role-btn").forEach(b =>
        b.classList.toggle("active", b.dataset.role === careerState.role));
      // Re-init search for the correct role, then seed input
      await initCareerSearch(careerState.role);
      if (_careerSearch) {
        _careerSearch.selectDefault(p => p.player_id === cur.player_id);
      }
    }
  }
});
