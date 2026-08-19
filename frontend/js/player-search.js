/*
 * Reusable searchable/scrollable player picker - replaces the old
 * Pitcher/Batter toggle + giant <select> pattern. Fetches the merged
 * (player, role) list once, filters client-side as you type (fast enough
 * for ~1-2K rows), shows a scrollable dropdown, and reports {player_id,
 * name, team_id, role} back to the caller on selection - the caller no
 * longer manages a separate "mode" state, the selected player carries
 * their own role.
 */
function createPlayerSearch(inputEl, dropdownEl, onSelect, { roleFilter } = {}) {
  let allPlayers = [];
  let matches = [];
  let activeIndex = -1;

  // Accent-fold so typing "Gimenez" or "Munoz" (no diacritics, the common
  // case) still matches "Giménez" / "Muñoz".
  const DIACRITICS_RE = new RegExp("[̀-ͯ]", "g");
  function fold(s) {
    return s.normalize("NFD").replace(DIACRITICS_RE, "").toLowerCase();
  }

  function roleTag(role) {
    return role === "pitcher" ? "P" : "Hitter";
  }

  function render(query) {
    const q = fold(query.trim());
    matches = q
      ? allPlayers.filter((p) => fold(p.name).includes(q)).slice(0, 40)
      : allPlayers.slice(0, 40);

    if (!matches.length) {
      dropdownEl.innerHTML = `<div class="player-search-empty">No matches</div>`;
      dropdownEl.style.display = "block";
      return;
    }

    dropdownEl.innerHTML = matches
      .map((p, i) => `
        <div class="player-search-item" data-index="${i}">
          <span>${p.name}</span>
          <span class="player-search-role">${roleTag(p.role)}</span>
        </div>
      `)
      .join("");
    dropdownEl.style.display = "block";
    activeIndex = -1;

    [...dropdownEl.querySelectorAll(".player-search-item")].forEach((el) => {
      el.addEventListener("mousedown", (e) => {
        e.preventDefault();
        select(matches[Number(el.dataset.index)]);
      });
    });
  }

  function updateActive() {
    const items = [...dropdownEl.querySelectorAll(".player-search-item")];
    items.forEach((el, i) => el.classList.toggle("active", i === activeIndex));
    if (activeIndex >= 0) items[activeIndex].scrollIntoView({ block: "nearest" });
  }

  function select(p) {
    inputEl.value = `${p.name} (${roleTag(p.role)})`;
    dropdownEl.style.display = "none";
    onSelect(p);
  }

  inputEl.addEventListener("input", () => render(inputEl.value));
  inputEl.addEventListener("focus", () => render(inputEl.value));
  inputEl.addEventListener("blur", () => setTimeout(() => { dropdownEl.style.display = "none"; }, 150));
  inputEl.addEventListener("keydown", (e) => {
    if (dropdownEl.style.display !== "block" || !matches.length) return;
    if (e.key === "ArrowDown") {
      e.preventDefault();
      activeIndex = Math.min(activeIndex + 1, matches.length - 1);
      updateActive();
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      activeIndex = Math.max(activeIndex - 1, 0);
      updateActive();
    } else if (e.key === "Enter") {
      e.preventDefault();
      if (activeIndex >= 0) select(matches[activeIndex]);
    } else if (e.key === "Escape") {
      dropdownEl.style.display = "none";
    }
  });

  return {
    async load(season) {
      const url = season ? `/api/players/all?season=${season}` : "/api/players/all";
      const players = await getJSON(url);
      allPlayers = roleFilter ? players.filter((p) => p.role === roleFilter) : players;
    },
    selectDefault(predicate) {
      const found = predicate ? allPlayers.find(predicate) : allPlayers[0];
      if (found) select(found);
      return found;
    },
    setValue(p) {
      inputEl.value = `${p.name} (${roleTag(p.role)})`;
    },
  };
}
