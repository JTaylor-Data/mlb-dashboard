/*
 * Shared cross-page player selection + recent searches. Loaded before
 * player-search.js on every page that has a player search box.
 *
 * "Current player" persists across navigation (player/radar/splits/
 * stuff-model/tunneling-model/deception-model/tunnel3d all read and write
 * the same sessionStorage key) so switching tabs to analyze the same guy
 * doesn't require re-searching every time - but resets to empty on an
 * actual page reload (F5/refresh), detected via the Navigation Timing API,
 * not just closing the tab. sessionStorage alone doesn't do this (it
 * survives reloads by default) - the explicit reload check below is what
 * makes it behave the way it was asked to.
 *
 * "Recent searches" (up to 5, most-recent-first, deduped by player_id) is a
 * separate, longer-lived list in localStorage - it's meant to survive
 * reloads and new sessions, unlike "current player."
 */
const PlayerContext = (() => {
  const CURRENT_KEY = "mlb_current_player";
  const RECENT_KEY = "mlb_recent_searches";
  const MAX_RECENT = 5;

  try {
    const navEntry = performance.getEntriesByType("navigation")[0];
    if (navEntry && navEntry.type === "reload") {
      sessionStorage.removeItem(CURRENT_KEY);
    }
  } catch (err) {
    // Navigation Timing unsupported - current-player just behaves like a
    // normal session value in that case, not worth failing over.
  }

  function getCurrent() {
    try {
      const raw = sessionStorage.getItem(CURRENT_KEY);
      return raw ? JSON.parse(raw) : null;
    } catch (err) {
      return null;
    }
  }

  function setCurrent(player) {
    try {
      sessionStorage.setItem(CURRENT_KEY, JSON.stringify(player));
    } catch (err) {
      // ignore (private browsing / storage disabled)
    }
  }

  function getRecent() {
    try {
      const raw = localStorage.getItem(RECENT_KEY);
      return raw ? JSON.parse(raw) : [];
    } catch (err) {
      return [];
    }
  }

  function recordRecent(player) {
    try {
      const list = getRecent().filter((p) => p.player_id !== player.player_id);
      list.unshift(player);
      localStorage.setItem(RECENT_KEY, JSON.stringify(list.slice(0, MAX_RECENT)));
    } catch (err) {
      // ignore
    }
  }

  return { getCurrent, setCurrent, getRecent, recordRecent };
})();
