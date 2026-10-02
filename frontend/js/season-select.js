/*
 * Shared season <select>, used by every page with season-scoped data
 * (Player, Standings, Splits, Radar, Compare). Populates from /api/meta's
 * available_seasons - whatever's actually been pulled into the DB, not a
 * hardcoded year list, so a page never offers a season with no data behind
 * it. Defaults to the most recent season (index 0, since available_seasons
 * comes back sorted descending).
 */
function createSeasonSelect(selectEl, onChange) {
  selectEl.addEventListener("change", () => onChange(Number(selectEl.value)));
  return {
    async load() {
      const meta = await getJSON("/api/meta");
      const seasons = meta.available_seasons && meta.available_seasons.length ? meta.available_seasons : [2026];
      selectEl.innerHTML = seasons.map((s) => `<option value="${s}">${s}</option>`).join("");
      return seasons[0];
    },
    get value() {
      return Number(selectEl.value);
    },
    set(season) {
      selectEl.value = String(season);
    },
  };
}
