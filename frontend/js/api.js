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
