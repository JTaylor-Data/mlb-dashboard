# Behavior Verification Checklist — Post perf-cleanup

Generated: 2026-10-02

## How to use
Run through each item manually after the server has been restarted with
the new code. Mark [ ] → [x] for pass, or [F] with a note for fail.

---

## F3: Rolling xwOBA NaN fix

- [ ] Load Kyle Schwarber (656941) for 2026 — rolling xwOBA section appears
- [ ] Chart renders with smooth curve
- [ ] 50 PA / 100 PA / 250 PA buttons switch the line correctly
- [ ] No console errors for any player in 2026

---

## F4: Similar player endpoints

- [ ] `/api/player/645261/similar?season=2026&role=pitcher` returns 200
- [ ] `/api/player/656941/similar?season=2026&role=batter` returns 200

---

## F1: Heatmaps cache

- [ ] Load Alcantara (645261) pitcher page — heatmap renders at any speed
- [ ] Reload immediately — heatmap renders faster (cache hit)
- [ ] Switch L/R split — still fast on reload (cached per split)

---

## F2: Career N+1 fix

- [ ] `/api/players/645261/career` cold load completes in < 10s
- [ ] Reload same endpoint — near-instant (cached)

---

## F5: Server-side search filtering

- [ ] `/api/players/all?q=kersh` returns only Kershaw (not all 14K players)
- [ ] `/api/players/all?q=` (empty) returns all players
- [ ] Search box in UI still works — type "Kershaw", player appears

---

## Opt-4/5/6: SHAP caching

- [ ] Deception+ page (deception-model.html) for any pitcher loads correctly
- [ ] Stuff+ model page for pitcher loads and shows SHAP breakdown
- [ ] Tunneling+ model page shows v3 breakdown
- [ ] Location+ model page shows breakdown

---

## Regression: Plus-scale scores unchanged

- [ ] Stuff+ score for Alcantara 2026 matches pre-perf-cleanup value
- [ ] Tunneling+ v3 score matches
- [ ] Deception+ score matches
- [ ] Location+ score matches
- [ ] Run compare_golden.py — only registered intentional changes (search, xwoba)

---

## Frontend: xwOBA chart visual

- [ ] Chart renders with smooth curve (not jagged polyline)
- [ ] "PA N" labels appear at start and end of x-axis
- [ ] Legend shows correctly (Player / Lg Avg)
- [ ] All three window buttons (50 PA, 100 PA, 250 PA) work
- [ ] Active button is highlighted

---

## General regression

- [ ] Pitcher player page (Alcantara 2026) renders completely
- [ ] Batter player page (Schwarber 2026) renders completely
- [ ] Career page loads for both roles
- [ ] Standings page loads
- [ ] Playoffs bracket loads
