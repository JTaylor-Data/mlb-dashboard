// Shared loading-bar toggle - pages that fetch data without any other
// loading indicator wrap their initial fetch(es) with this, targeting a
// `<div class="loading-bar" id="page-loading"><div class="loading-bar-fill">
// </div></div>` placed near their main content mount. Same visual pattern
// as the player page's own checkpoint-driven bar (see index.html's
// .load-progress), just a plain indeterminate animation instead of real
// checkpoints, for pages whose backend is a single opaque fetch.
// The bar is shown whenever a page holds it on (setPageLoading) or any getJSON
// fetch is still in flight, so every data-fetching page that has the element
// shows it for its whole load - including the first /api/meta call, which used
// to run before a page turned the bar on.
let _barManual = false;
let _barInflight = 0;

function _syncLoadingBar() {
  const el = document.getElementById("page-loading");
  if (el) el.classList.toggle("active", _barManual || _barInflight > 0);
}

function setPageLoading(active) {
  _barManual = active;
  _syncLoadingBar();
}

// Request cancelling. A page calls beginLoad() at the start of every
// user-driven load (a new player, season, split or tab); that aborts whatever
// the PREVIOUS load still has in flight, so clicking around quickly can't stack
// up requests the browser and server then have to work through one after
// another. An aborted request never settles, so the superseded load simply
// stops (no error banner, no stale render, the newer load owns the page).
// Shared lookups that must outlive a click (the season list, the player search
// index, /api/meta) pass { persist: true } and are never aborted. A caller can
// also hand getJSON its own { signal }.
let _loadController = null;

function beginLoad() {
  if (_loadController) _loadController.abort();
  _loadController = new AbortController();
  return _loadController.signal;
}

function _requestSignal(opts) {
  if (opts && opts.persist) return undefined;
  return (opts && opts.signal) || (_loadController && _loadController.signal) || undefined;
}

function _superseded(err, opts) {
  return err && err.name === "AbortError" && !(opts && opts.persist);
}

async function getJSON(url, opts) {
  _barInflight++;
  _syncLoadingBar();
  try {
    const signal = _requestSignal(opts);
    const resp = await fetch(url, signal ? { signal } : undefined);
    if (!resp.ok) {
      throw new Error(`${url} -> HTTP ${resp.status}`);
    }
    return await resp.json();
  } catch (err) {
    if (_superseded(err, opts)) return new Promise(() => {});
    throw err;
  } finally {
    _barInflight--;
    _syncLoadingBar();
  }
}

// /api/meta is asked for by the nav, the season picker and several pages on
// every load; share one request per page view. Never aborted.
let _metaPromise = null;
function getMeta() {
  if (!_metaPromise) {
    _metaPromise = getJSON("/api/meta", { persist: true }).catch((err) => {
      _metaPromise = null;
      throw err;
    });
  }
  return _metaPromise;
}

// The season pages should open on: the newest one that has pitch data, which
// flips to the new year by itself once its first pitches are ingested.
async function getLatestSeason() {
  const meta = await getMeta();
  return (meta.available_seasons && meta.available_seasons[0]) || meta.current_season;
}

// Prefetch cache: background fetches fired on player-page load so probable-
// pitchers and top-performers render instantly when navigated to. Each
// page checks window._prefetchCache[url] on init and uses it if present.
window._prefetchCache = window._prefetchCache || {};

function prefetchInBackground(url) {
  if (window._prefetchCache[url]) return; // already fetched
  window._prefetchCache[url] = "pending";
  fetch(url).then(r => r.ok ? r.json() : null).then(data => {
    if (data) window._prefetchCache[url] = data;
    else delete window._prefetchCache[url];
  }).catch(() => { delete window._prefetchCache[url]; });
}

async function postJSON(url) {
  const resp = await fetch(url, { method: "POST" });
  if (!resp.ok) {
    throw new Error(`${url} -> HTTP ${resp.status}`);
  }
  return resp.json();
}

// Reads a newline-delimited-JSON stream (one {"type":"progress",...} line
// per real server-side checkpoint, then one {"type":"done","data":...}
// line) - for endpoints built to report real progress mid-request rather
// than arriving as a single opaque wait. onProgress is called with each
// progress event as it arrives; the final done event's `data` is returned.
async function getJSONStreamed(url, onProgress, opts) {
  try {
    return await _readStream(url, onProgress, _requestSignal(opts));
  } catch (err) {
    if (_superseded(err, opts)) return new Promise(() => {});
    throw err;
  }
}

async function _readStream(url, onProgress, signal) {
  const resp = await fetch(url, signal ? { signal } : undefined);
  if (!resp.ok) {
    throw new Error(`${url} -> HTTP ${resp.status}`);
  }
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  let result = null;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, idx);
      buf = buf.slice(idx + 1);
      if (!line.trim()) continue;
      const evt = JSON.parse(line);
      if (evt.type === "progress") {
        onProgress && onProgress(evt);
      } else if (evt.type === "done") {
        result = evt.data;
      }
    }
  }
  return result;
}
