// Shared loading-bar toggle - pages that fetch data without any other
// loading indicator wrap their initial fetch(es) with this, targeting a
// `<div class="loading-bar" id="page-loading"><div class="loading-bar-fill">
// </div></div>` placed near their main content mount. Same visual pattern
// as the player page's own checkpoint-driven bar (see index.html's
// .load-progress), just a plain indeterminate animation instead of real
// checkpoints, for pages whose backend is a single opaque fetch.
function setPageLoading(active) {
  const el = document.getElementById("page-loading");
  if (el) el.classList.toggle("active", active);
}

async function getJSON(url) {
  const resp = await fetch(url);
  if (!resp.ok) {
    throw new Error(`${url} -> HTTP ${resp.status}`);
  }
  return resp.json();
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
async function getJSONStreamed(url, onProgress) {
  const resp = await fetch(url);
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
