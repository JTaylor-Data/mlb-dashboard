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

  // Pitchers: handedness + SP/RP (e.g. "R-SP", "L-RP" - unclassified
  // pitchers, usually too few appearances to call either way, default to
  // RP same as the Last 5 Outings card's own fallback). Batters: LHH/RHH,
  // matching the same "handedness box" label used elsewhere on this site;
  // switch hitters show "SHH" (bats='S').
  function roleTag(p) {
    if (p.role === "pitcher") return `${p.throws || "?"}-${p.pitcher_role || "RP"}`;
    return `${p.bats || "?"}HH`;
  }

  function render(query) {
    const q = fold(query.trim());
    let heading = "";

    if (!q) {
      // Empty box: show recent searches instead of dumping the first 40
      // players in the list, which was never a meaningful default order.
      const recent = (typeof PlayerContext !== "undefined" ? PlayerContext.getRecent() : [])
        .filter((p) => !roleFilter || p.role === roleFilter);
      matches = recent;
      heading = "Recent Searches";
      if (!matches.length) {
        dropdownEl.innerHTML = `<div class="player-search-empty">No recent searches</div>`;
        dropdownEl.style.display = "block";
        return;
      }
    } else {
      // p._folded is precomputed once in load() below, not re-folded
      // (normalize + regex replace) on every single keystroke against the
      // full ~1-2K player list.
      matches = allPlayers.filter((p) => p._folded.includes(q)).slice(0, 40);
      if (!matches.length) {
        dropdownEl.innerHTML = `<div class="player-search-empty">No matches</div>`;
        dropdownEl.style.display = "block";
        return;
      }
    }

    const heading_html = heading ? `<div class="player-search-heading">${heading}</div>` : "";
    dropdownEl.innerHTML = heading_html + matches
      .map((p, i) => `
        <div class="player-search-item" data-index="${i}">
          <span>${p.name}</span>
          <span class="player-search-role">${roleTag(p)}</span>
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
    inputEl.value = `${p.name} (${roleTag(p)})`;
    dropdownEl.style.display = "none";
    if (typeof PlayerContext !== "undefined") {
      PlayerContext.recordRecent(p);
      PlayerContext.setCurrent(p);
    }
    onSelect(p);
  }

  // Named (not inline-arrow) so destroy() below can actually remove them -
  // an anonymous listener can never be un-registered.
  const onInput = () => render(inputEl.value);
  const onFocus = () => render(inputEl.value);
  const onBlur = () => setTimeout(() => { dropdownEl.style.display = "none"; }, 150);
  const onKeydown = (e) => {
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
  };
  inputEl.addEventListener("input", onInput);
  inputEl.addEventListener("focus", onFocus);
  inputEl.addEventListener("blur", onBlur);
  inputEl.addEventListener("keydown", onKeydown);

  return {
    async load(_season) {
      const resp = await fetch("/api/players/all");
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const players = await resp.json();
      const filtered = roleFilter ? players.filter((p) => p.role === roleFilter) : players;
      allPlayers = filtered.map((p) => ({ ...p, _folded: fold(p.name) }));
    },
    selectDefault(predicate) {
      const found = predicate ? allPlayers.find(predicate) : allPlayers[0];
      if (found) select(found);
      return found;
    },
    setValue(p) {
      inputEl.value = `${p.name} (${roleTag(p)})`;
    },
    // Call before re-creating a search instance on the SAME input/dropdown
    // elements (e.g. compare.js rebuilding on a role toggle) - without this,
    // every rebuild stacked another full set of listeners + closure state on
    // the same DOM nodes (a real, confirmed bug: every keystroke fired
    // render() once per stacked instance, and Enter fired select()/onSelect
    // once per instance too, each with its own stale `matches`).
    destroy() {
      inputEl.removeEventListener("input", onInput);
      inputEl.removeEventListener("focus", onFocus);
      inputEl.removeEventListener("blur", onBlur);
      inputEl.removeEventListener("keydown", onKeydown);
    },
  };
}
