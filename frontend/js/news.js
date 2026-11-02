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
        No news items yet. News updates automatically with the daily data job.
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
        <span>${item.pub_date ? escHtml(formatDate(item.pub_date)) : ""}</span>
        <span class="news-source-badge">MLB.com</span>
      </div>`;
    }

    const [hero, ...rest] = data.items;

    const heroBody = itemBody(hero);
    const heroHtml = `<div class="news-hero">
      <div class="news-eyebrow">Top Story</div>
      <a class="news-headline-link" href="${safeUrl(hero.source_url)}" target="_blank" rel="noopener">${decodeHtml(hero.headline)}</a>
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
          <a class="news-headline-link" href="${safeUrl(item.source_url)}" target="_blank" rel="noopener">${decodeHtml(item.headline)}</a>
        </div>
        ${body ? `<div class="news-body">${decodeHtml(body)}</div>` : ""}
        ${metaHtml(item)}
      </div>`;
    }).join("");

    mount.innerHTML = heroHtml + `<div class="news-grid">${gridHtml}</div>`;
  } catch (err) {
    mount.innerHTML = `<div class="empty-state">Failed to load news: ${escHtml(err.message)}</div>`;
  } finally {
    setPageLoading(false);
  }
}

function escHtml(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

// RSS text can contain HTML entities; decode them once for display, then
// escape so the result is safe to drop into innerHTML.
function decodeHtml(s) {
  if (!s) return "";
  const txt = document.createElement("textarea");
  txt.innerHTML = s;
  return escHtml(txt.value);
}

// Only http(s) links from the feed; anything else (javascript:, data:) is dropped.
function safeUrl(u) {
  return /^https?:\/\//i.test(u || "") ? escHtml(u) : "#";
}

function formatDate(dateStr) {
  try {
    return new Date(dateStr).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
  } catch (_) { return dateStr; }
}

document.addEventListener("DOMContentLoaded", () => {
  loadNews();
});
