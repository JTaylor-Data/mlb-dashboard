const state = { mode: "pitcher", playerId: null, playerName: "", season: 2026 };

const splitBySelect = document.getElementById("split-by-select");
const pitchTypeSelect = document.getElementById("pitch-type-select");
const seasonSelect = createSeasonSelect(document.getElementById("season-select"), async (season) => {
  state.season = season;
  await playerSearch.load(state.season);
  await loadPitchTypes();
  await loadSplits();
});

const SPLIT_OPTIONS = {
  pitcher: ["home_away", "month", "half", "stand"],
  batter: ["home_away", "month", "half", "p_throws", "archetype"],
};
const SPLIT_BY_LABELS = {
  home_away: "Home / Away", month: "Month", half: "1st / 2nd Half",
  stand: "Batter Hand", p_throws: "Pitcher Hand", archetype: "Opponent Archetype",
};

const COLUMN_LABELS = {
  split_value: "Split", n_pa: "PA", n_pitches: "Pitches",
  OPS: "OPS", wOBA: "wOBA", xwOBA: "xwOBA", K_pct: "K%",
  swing_pct: "Swing%", whiff_pct: "Whiff%", chase_pct: "Chase%", barrel_pct: "Barrel%",
};

function columnLabel(col) {
  if (COLUMN_LABELS[col]) return COLUMN_LABELS[col];
  const base = col.replace(/_overall$/, "");
  return (COLUMN_LABELS[base] || base) + (col.endsWith("_overall") ? " (season)" : "");
}

const playerSearch = createPlayerSearch(
  document.getElementById("player-search-input"),
  document.getElementById("player-search-dropdown"),
  async (p) => {
    state.playerId = p.player_id;
    state.mode = p.role;
    state.playerName = p.name;
    populateSplitBy();
    await loadPitchTypes();
    await loadSplits();
  }
);

splitBySelect.addEventListener("change", loadSplits);
pitchTypeSelect.addEventListener("change", loadSplits);

function populateSplitBy() {
  splitBySelect.innerHTML = SPLIT_OPTIONS[state.mode]
    .map((s) => `<option value="${s}">${SPLIT_BY_LABELS[s] || s}</option>`)
    .join("");
}

async function loadPitchTypes() {
  if (state.playerId === null) return;
  const types = await getJSON(`/api/splits/pitch-types?role=${state.mode}&player_id=${state.playerId}&season=${state.season}`);
  pitchTypeSelect.innerHTML = ["All", ...types].map((t) => `<option value="${t}">${t}</option>`).join("");
}

async function loadSplits() {
  if (state.playerId === null) return;
  const splitBy = splitBySelect.value;
  const pitchType = pitchTypeSelect.value;
  document.getElementById("results-title").textContent = `${state.playerName} — ${SPLIT_BY_LABELS[splitBy] || splitBy}`;

  const data = await getJSON(`/api/splits?role=${state.mode}&player_id=${state.playerId}&split_by=${splitBy}&pitch_type=${encodeURIComponent(pitchType)}&season=${state.season}`);
  const mount = document.getElementById("table-mount");
  if (!data.rows.length) {
    mount.innerHTML = `<div class="empty-state">No data for this selection.</div>`;
    return;
  }
  const header = data.columns.map((c) => `<th>${columnLabel(c)}</th>`).join("");
  const rows = data.rows
    .map((row) => {
      const cells = data.columns
        .map((c) => `<td class="${c.endsWith('_overall') ? 'overall-col' : ''}">${row[c]}</td>`)
        .join("");
      return `<tr>${cells}</tr>`;
    })
    .join("");
  mount.innerHTML = `<table class="splits-table"><thead><tr>${header}</tr></thead><tbody>${rows}</tbody></table>`;
}

window.onDataRefresh = async () => {
  await playerSearch.load(state.season);
  await loadPitchTypes();
  await loadSplits();
};

(async function init() {
  populateSplitBy();
  state.season = await seasonSelect.load();
  await playerSearch.load(state.season);
  const defaultPlayer = playerSearch.selectDefault((p) => p.role === "pitcher");
  if (defaultPlayer) {
    state.playerId = defaultPlayer.player_id;
    state.mode = defaultPlayer.role;
    state.playerName = defaultPlayer.name;
    populateSplitBy();
    await loadPitchTypes();
    await loadSplits();
  }
})();
