// MLB News page — priority-sorted with descriptions

const PRIORITY_LABEL = {
  2: { label: "Breaking", cls: "priority-high" },
  1: { label: "", cls: "" },
  0: { label: "", cls: "priority-low" },
};

async function loadNews() {
  setPageLoading(true);
  const mount = document.getElementById("news-mount");
  const note = document.getElementById("freshness-note");
  try {
    const data = await getJSON("/api/news?limit=15");
    if (!data.items || !data.items.length) {
      mount.innerHTML = `<div class="no-news">
        No news items yet. Click "Refresh News" to fetch from MLB.com, or run
        <code>python pulls/refresh_mlb_stats.py</code> from the server.
      </div>`;
      return;
    }
    const latest = data.items[0].fetched_at;
    if (latest) note.textContent = `Last fetched: ${latest.slice(0, 19)} UTC`;

    function itemBody(item) {
      const desc    = (item.description || "").trim();
      const summary = (item.summary    || "").trim();
      return item.summarized && summary && summary !== item.headline
        ? summary
        : (desc && desc !== item.headline ? desc : "");
    }

    function metaHtml(item) {
      return `<div class="news-meta">
        <span>${item.pub_date ? formatDate(item.pub_date) : ""}</span>
        <span class="news-source-badge">MLB.com</span>
      </div>`;
    }

    const [hero, ...rest] = data.items;

    const heroBody = itemBody(hero);
    const heroHtml = `<div class="news-hero">
      <div class="news-eyebrow">Top Story</div>
      <a class="news-headline-link" href="${hero.source_url}" target="_blank" rel="noopener">${decodeHtml(hero.headline)}</a>
      ${heroBody ? `<div class="news-body">${decodeHtml(heroBody)}</div>` : ""}
      ${metaHtml(hero)}
    </div>`;

    const gridHtml = rest.map(item => {
      const p = item.priority ?? 1;
      const { label: prioLabel, cls: prioCls } = PRIORITY_LABEL[p] || { label: "", cls: "" };
      const priorityBadge = prioLabel
        ? `<span class="news-priority-badge ${prioCls}">${prioLabel}</span>`
        : "";
      const body = itemBody(item);
      return `<div class="news-item ${prioCls}">
        <div class="news-headline-row">
          ${priorityBadge}
          <a class="news-headline-link" href="${item.source_url}" target="_blank" rel="noopener">${decodeHtml(item.headline)}</a>
        </div>
        ${body ? `<div class="news-body">${decodeHtml(body)}</div>` : ""}
        ${metaHtml(item)}
      </div>`;
    }).join("");

    mount.innerHTML = heroHtml + `<div class="news-grid">${gridHtml}</div>`;
  } catch (err) {
    mount.innerHTML = `<div class="empty-state">Failed to load news: ${err.message}</div>`;
  } finally {
    setPageLoading(false);
  }
}

function decodeHtml(s) {
  if (!s) return "";
  const txt = document.createElement("textarea");
  txt.innerHTML = s;
  return txt.value.replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function formatDate(dateStr) {
  try {
    return new Date(dateStr).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
  } catch (_) { return dateStr; }
}

document.addEventListener("DOMContentLoaded", () => {
  const btn = document.getElementById("refresh-btn");
  btn.addEventListener("click", async () => {
    btn.disabled = true; btn.textContent = "Refreshing...";
    try {
      const result = await postJSON("/api/news/refresh");
      btn.textContent = result.status === "ok"
        ? `Done — ${result.new_items} new`
        : `Error: ${result.detail || "unknown"}`;
      if (result.status === "ok") await loadNews();
    } catch (err) {
      btn.textContent = `Error: ${err.message}`;
    } finally {
      btn.disabled = false;
      setTimeout(() => { btn.textContent = "Refresh News"; }, 3000);
    }
  });
  loadNews();
});
