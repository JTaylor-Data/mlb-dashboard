# PROMOTION CHECKLIST — Stuff+ v4

**Status**: CANDIDATE ONLY. Do not execute until owner approves.  
**Branch**: `stuff-v4`  
**Artifacts**: `models/stuff_plus_v4_lgbm.txt`, `models/stuff_plus_v4_lgbm_meta.json`, `diagnostics/stuff_v4/stuff_v4_scores.parquet`

---

## 0. Pre-promotion verification

- [ ] `models/stuff_plus_v4_lgbm.txt` exists and SHA-256 hash matches `stuff_plus_v4_lgbm_meta.json → model_hash_sha256`
- [ ] `diagnostics/stuff_v4/stuff_v4_scores.parquet` exists with all 11 seasons (2016–2026)
- [ ] `post-stuff-v4` tag is pushed to `origin`
- [ ] All Part 4 validation tables are present in `diagnostics/stuff_v4/part4_validation.json`
- [ ] Owner has reviewed `Methodology/STUFF_V4_REPORT.md` and signed off on feature changes

---

## 1. Update model file references

**File**: `analytics/stuff_model.py` (lines 66–67)  
Change:
```python
_model = lgb.Booster(model_file=str(MODELS_DIR / "stuff_plus_v3_lgbm.txt"))
_meta = json.loads((MODELS_DIR / "stuff_plus_v3_lgbm_meta.json").read_text())
```
to:
```python
_model = lgb.Booster(model_file=str(MODELS_DIR / "stuff_plus_v4_lgbm.txt"))
_meta = json.loads((MODELS_DIR / "stuff_plus_v4_lgbm_meta.json").read_text())
```

---

## 2. Update feature pipeline (if features changed)

**File**: `analytics/stuff_features.py`

If `drop_season=True` was adopted (V1 passes):
- Remove `"season"` from `CATEGORICAL_FEATURES`
- Update `ALL_FEATURES` accordingly

If `drop_effective_speed=True` was adopted (V2 passes):
- Remove `"effective_speed"` from `CORE_NUMERIC_FEATURES`
- Update `NUMERIC_FEATURES`, `ALL_FEATURES`

If `include_haa=True` was adopted:
- Add `add_haa_residual()` call in `engineer_features()`
- Add `haa_residual` to `OPTIONAL_NUMERIC_FEATURES`
- Ensure `haa_coefs` are loaded from meta and passed through

**Verify**: `BANNED_IDENTIFIER_COLS` and `assert_no_identifiers()` still cover all forbidden features.

---

## 3. Update RAW_QUERY_COLUMNS

**File**: `analytics/stuff_features.py`

If HAA was adopted, add to `RAW_QUERY_COLUMNS`:
- `"vx0"`, `"ax"`, `"release_pos_x"` (if not already present)

---

## 4. Cache and API

- [ ] Call `sm_runtime.clear_cache()` after any live code change (or restart the API process)
- [ ] Verify `_load()` singleton picks up the new model file
- [ ] Check `_prepare_features()` in `stuff_model.py` builds the FB reference table correctly with the new feature set
- [ ] If HAA adopted: ensure `_prepare_features()` also computes `haa_residual` using the stored `haa_coefs` from meta

---

## 5. Season range update

- [ ] Update any UI references from "2016–2025" to "2016–2026" 
- [ ] Confirm 2020 shows the "60-game season" tooltip/asterisk on all leaderboard views (search for `season_flag` usage in frontend code)
- [ ] Qualification threshold for 2020: scale by factor 0.37 (60/162) — verify this is already applied or add it

---

## 6. Leaderboard/score table update

The `pitcher_model_scores` table in the DB currently holds v3 scores. Options:
1. **Recalculate and overwrite**: Run a backfill script to replace `stuff_plus` column for all seasons 2016–2026 from `stuff_v4_scores.parquet`
2. **Add a new column**: Add `stuff_plus_v4` column to `pitcher_model_scores`

Recommended: Option 1 (single source of truth, existing API endpoints continue to work unchanged).

Script outline:
```python
import pandas as pd, sqlite3
scores = pd.read_parquet("diagnostics/stuff_v4/stuff_v4_scores.parquet")
# Aggregate to pitcher-season (usage-weighted)
ps = (scores.groupby(["pitcher_id","season"])
      .apply(lambda x: pd.Series({"stuff_plus": np.average(x["stuff_plus_v4"], weights=x["n_pitches"])}))
      .reset_index())
conn = sqlite3.connect("mlb_data.db")
# Update each row
for _, row in ps.iterrows():
    conn.execute("UPDATE pitcher_model_scores SET stuff_plus=? WHERE pitcher_id=? AND season=?",
                 (row["stuff_plus"], row["pitcher_id"], row["season"]))
conn.commit(); conn.close()
```

---

## 7. Past-season score disclosure

Add to the "About" / methodology page or a tooltip:
> **Displayed past-season scores (2016–2026) were computed with a single model trained on all seasons 2016–2026 (excluding 2020). Validation figures in the methodology report are in-sample (descriptive) or from a pitcher-holdout ablation subsample (labeled "conservative"), not true out-of-sample historical forecasts.**

---

## 8. 2020 asterisk

- [ ] Anywhere a 2020 Stuff+ score appears, display: "⚠ 60-game season"
- [ ] In validation tables, 2020 is flagged `season_flag = "60-game season"`
- [ ] Do NOT exclude 2020 from the leaderboard — just flag it

---

## 9. Git finalization (after all above)

```bash
git add analytics/stuff_features.py analytics/stuff_model.py
# any other changed files
git commit -m "feat(stuff): promote Stuff+ v4 model to production"
git push origin stuff-v4
# Then merge stuff-v4 -> main (owner decision)
```

---

## 10. Post-promotion smoke test

- [ ] Hit `/api/stuff-plus/2025` and verify response has pitcher data
- [ ] Hit `/api/stuff-plus/2020` and verify 2020 data returns with `season_flag`
- [ ] Hit `/api/stuff-plus/2016` and verify historical data
- [ ] Compare top-10 leaderboard 2025 vs v3 — expected high correlation (>0.97)
- [ ] Check SHAP breakdown endpoint still works: `/api/stuff-plus/shap/2025/{pitcher_id}`

---

*This checklist was generated automatically as part of the Stuff+ v4 training run (2026-10-02). Do not execute until owner explicitly approves promotion.*
