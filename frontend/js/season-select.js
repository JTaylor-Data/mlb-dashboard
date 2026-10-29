/*
 * Shared season <select>, used by every page with season-scoped data
 * (Player, Standings, Splits, Radar, Compare). Populates from /api/meta's
 * available_seasons - whatever's actually been pulled into the DB, not a
 * hardcoded year list, so a page never offers a season with no data behind
 * it. Defaults to the most recent season (index 0, since available_seasons
 * comes back sorted descending).
 */
function createSeasonSelect(selectEl, onChange) {
  let _allSeasons = [];
  selectEl.addEventListener("change", () => onChange(Number(selectEl.value)));
  return {
    async load() {
      const meta = await getJSON("/api/meta");
      _allSeasons = meta.available_seasons && meta.available_seasons.length ? meta.available_seasons : [2026];
      selectEl.innerHTML = _allSeasons.map((s) => `<option value="${s}">${s}</option>`).join("");
      return _allSeasons[0];
    },
    // Restrict the dropdown to a specific set of seasons (items 1 & 2).
    // Keeps the current value if it exists in the new list; otherwise
    // switches to the first (most recent) season in the filtered set.
    filter(seasons) {
      if (!seasons || !seasons.length) return;
      const cur = Number(selectEl.value);
      selectEl.innerHTML = seasons.map((s) => `<option value="${s}">${s}</option>`).join("");
      if (seasons.includes(cur)) {
        selectEl.value = String(cur);
      }
    },
    // Restore the full global season list (e.g. when the player is cleared).
    resetFilter() {
      selectEl.innerHTML = _allSeasons.map((s) => `<option value="${s}">${s}</option>`).join("");
    },
    get value() {
      return Number(selectEl.value);
    },
    set(season) {
      selectEl.value = String(season);
    },
  };
}
