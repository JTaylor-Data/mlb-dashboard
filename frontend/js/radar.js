let state = { mode: "pitcher", split: "ALL", playerId: null, season: 2026 };

const splitToggle = document.getElementById("split-toggle");
const splitLabel = document.getElementById("split-label");
const seasonSelect = createSeasonSelect(document.getElementById("season-select"), async (season) => {
  state.season = season;
  await playerSearch.load(state.season);
  await loadRadar();
});

function setActiveButton(group, value) {
  [...group.querySelectorAll("button")].forEach((b) => b.classList.toggle("active", b.dataset.value === value));
}

splitToggle.addEventListener("click", async (e) => {
  const btn = e.target.closest("button");
  if (!btn || btn.dataset.value === state.split) return;
  state.split = btn.dataset.value;
  setActiveButton(splitToggle, state.split);
  await loadRadar();
});

const playerSearch = createPlayerSearch(
  document.getElementById("player-search-input"),
  document.getElementById("player-search-dropdown"),
  async (p) => {
    state.playerId = p.player_id;
    state.mode = p.role;
    splitLabel.textContent = state.mode === "pitcher" ? "Opponent bats" : "Opponent pitcher throws";
    await loadRadar();
  }
);

const RADAR_FILL = "rgba(27, 67, 50, 0.35)";
const RADAR_STROKE = "#1b4332";

function buildRadarSVG(axes) {
  const size = 440, cx = size / 2, cy = size / 2 + 6, r = 150;
  const n = axes.length;
  const angleFor = (i) => -Math.PI / 2 + (i * 2 * Math.PI) / n;
  const pointFor = (i, value) => {
    const radius = (Math.max(0, Math.min(100, value)) / 100) * r;
    const a = angleFor(i);
    return [cx + radius * Math.cos(a), cy + radius * Math.sin(a)];
  };

  const rings = [25, 50, 75, 100]
    .map((ringPct) => {
      const pts = axes.map((_, i) => pointFor(i, ringPct).join(",")).join(" ");
      return `<polygon points="${pts}" fill="none" stroke="var(--border)" stroke-width="1"/>`;
    })
    .join("");

  const spokes = axes
    .map((_, i) => {
      const [x, y] = pointFor(i, 100);
      return `<line x1="${cx}" y1="${cy}" x2="${x}" y2="${y}" stroke="var(--border)" stroke-width="1"/>`;
    })
    .join("");

  const dataPts = axes.map((a, i) => pointFor(i, a.pct !== null ? a.pct : 0));
  const dataPolygon = `<polygon points="${dataPts.map((p) => p.join(",")).join(" ")}" fill="${RADAR_FILL}" stroke="${RADAR_STROKE}" stroke-width="2.5"/>`;
  const dataDots = dataPts.map(([x, y]) => `<circle cx="${x}" cy="${y}" r="4" fill="${RADAR_STROKE}"/>`).join("");

  const labels = axes
    .map((a, i) => {
      const [lx, ly] = pointFor(i, 122);
      const anchor = Math.abs(Math.cos(angleFor(i))) < 0.2 ? "middle" : (Math.cos(angleFor(i)) > 0 ? "start" : "end");
      const pctStr = a.pct !== null ? `${a.pct.toFixed(0)}th` : "-";
      return `
        <text x="${lx}" y="${ly - 6}" font-size="13" font-weight="700" text-anchor="${anchor}" fill="var(--forest-dark)">${pctStr}</text>
        <text x="${lx}" y="${ly + 10}" font-size="12" text-anchor="${anchor}" fill="var(--text-muted)">${a.label}</text>
      `;
    })
    .join("");

  return `
    <svg viewBox="0 0 ${size} ${size + 20}" width="100%" style="max-width:520px; display:block; margin:0 auto;">
      ${rings}
      ${spokes}
      ${dataPolygon}
      ${dataDots}
      ${labels}
    </svg>
  `;
}

function renderRadarAxisList(axes) {
  const rows = axes
    .map((a) => {
      const color = a.pct !== null ? gradientColor(a.pct) : "#999";
      const pctStr = a.pct !== null ? a.pct.toFixed(0) : "-";
      return `
        <div class="radar-axis-row">
          <span class="label">${a.label}</span>
          <span class="val">${a.value_str}<span class="pct-chip" style="background:${color};">${pctStr}</span></span>
        </div>
      `;
    })
    .join("");
  document.getElementById("radar-axis-mount").innerHTML = rows;
}

let loadRadarRequestId = 0;
async function loadRadar() {
  if (state.playerId === null || state.playerId === "") return;
  const requestId = ++loadRadarRequestId;
  const requestMode = state.mode;
  setPageLoading(true);
  let data;
  try {
    data = await getJSON(`/api/radar/${state.playerId}?role=${state.mode}&split=${state.split}&season=${state.season}`);
  } finally {
    if (requestId === loadRadarRequestId) setPageLoading(false); // only the latest in-flight request controls the bar
  }
  if (requestId !== loadRadarRequestId || requestMode !== state.mode) return;
  document.getElementById("radar-chart-mount").innerHTML = `
    <h3 style="margin-top:0;">${data.name}</h3>
    ${buildRadarSVG(data.axes)}
  `;
  renderRadarAxisList(data.axes);
}

window.onDataRefresh = async () => {
  await playerSearch.load(state.season);
  await loadRadar();
};

(async function init() {
  state.season = await seasonSelect.load();
  await playerSearch.load(state.season);
  const remembered = PlayerContext.getCurrent();
  const chosen = remembered
    ? playerSearch.selectDefault((p) => p.player_id === remembered.player_id) || remembered
    : playerSearch.selectDefault((p) => p.role === "pitcher");
  if (chosen) {
    if (remembered) playerSearch.setValue(chosen);
    state.playerId = chosen.player_id;
    state.mode = chosen.role;
    splitLabel.textContent = state.mode === "pitcher" ? "Opponent bats" : "Opponent pitcher throws";
    await loadRadar();
  }
})();
