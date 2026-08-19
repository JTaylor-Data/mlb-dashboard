const NAV_LINKS = [
  { href: "index.html", label: "Player" },
  { href: "standings.html", label: "Standings" },
  { href: "top-performers.html", label: "Top Performers" },
  { href: "prospects.html", label: "Prospects" },
  { href: "splits.html", label: "Splits" },
  { href: "radar.html", label: "Radar" },
  { href: "compare.html", label: "Compare" },
  { href: "stuff-model.html", label: "Stuff+ Model" },
  { href: "tunneling-model.html", label: "Tunneling+" },
  { href: "deception-model.html", label: "Deception+" },
];

function initNav() {
  const mount = document.getElementById("nav");
  if (!mount) return;

  const current = location.pathname.split("/").pop() || "index.html";
  const links = NAV_LINKS.map(
    (l) => `<a href="${l.href}" class="${l.href === current ? "active" : ""}">${l.label}</a>`
  ).join("");

  mount.innerHTML = `
    <div class="nav-inner">
      <div class="nav-brand">MLB Dashboard</div>
      <div class="nav-links">${links}</div>
      <div class="nav-meta">
        <span id="nav-data-as-of">Data as of &mdash;</span>
        <button class="nav-refresh" id="nav-refresh-btn" type="button">Refresh data</button>
      </div>
    </div>
  `;

  getJSON("/api/meta")
    .then((meta) => {
      const el = document.getElementById("nav-data-as-of");
      if (el && meta.latest_game_date) el.textContent = `Data as of ${meta.latest_game_date}`;
    })
    .catch(() => {});

  const btn = document.getElementById("nav-refresh-btn");
  btn.addEventListener("click", async () => {
    // Previously this was try/finally with NO catch - if anything in the
    // chain threw (a dropped request, a bad response, whatever), the button
    // silently reset to normal with zero indication anything had gone
    // wrong. From the outside that looks exactly like "I clicked it and
    // nothing happened" - which is exactly what got reported. Every path
    // below now either succeeds visibly or fails visibly; nothing is
    // silently swallowed.
    btn.disabled = true;
    btn.textContent = "Starting...";
    try {
      const { started } = await postJSON("/api/admin/pull-data");
      if (!started) {
        btn.textContent = "Already running - watching it...";
      }
      const finalStatus = await pollPullStatus(btn);
      if (finalStatus.state === "error") {
        btn.textContent = "Failed - click to retry";
        console.error("Refresh failed:", finalStatus.detail);
        return; // leave the failure message up; don't pretend it succeeded
      }
      const meta = await getJSON("/api/meta");
      const el = document.getElementById("nav-data-as-of");
      if (el && meta.latest_game_date) el.textContent = `Data as of ${meta.latest_game_date}`;
      if (typeof window.onDataRefresh === "function") {
        await window.onDataRefresh();
      } else {
        location.reload();
        return; // page is navigating away - don't touch btn below
      }
      btn.textContent = "Refresh data";
    } catch (err) {
      console.error("Refresh data button failed:", err);
      btn.textContent = "Error - click to retry";
    } finally {
      btn.disabled = false;
    }
  });
}

async function pollPullStatus(btn) {
  // Actually pulls new Statcast/schedule data (not just an app-cache clear)
  // - this can take anywhere from a few seconds to a couple minutes
  // depending on how many days behind the DB is, so poll rather than block.
  // Returns the final status object (caller decides what "error" means for
  // the button) instead of swallowing it into a generic "Done".
  const maxAttempts = 400; // ~10 minutes at 1.5s/poll - a hard ceiling so a
                            // stuck server-side state can never hang this
                            // forever with no way out for the user.
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    const status = await getJSON("/api/admin/pull-status");
    if (status.state === "running") {
      btn.textContent = status.detail || "Pulling...";
    } else {
      return status;
    }
    await new Promise((r) => setTimeout(r, 1500));
  }
  return { state: "error", detail: "Timed out waiting for the refresh to finish - check the server log." };
}

document.addEventListener("DOMContentLoaded", initNav);
