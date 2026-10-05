"""
Location+ v3 — runtime scoring module.

Architecture: five LightGBM classifiers (xSwing, xCalledStrike, xWhiff,
xFoul, BIP×EV) combined with a COUNT-SPECIFIC run-value table to produce
per-pitch expected run value (xRV). Pitcher-season aggregation → plus-scale.

Key changes from v2:
  - Run-value table is now count-specific (balls × strikes per event type).
    This was a defect-correction: v2 used single pooled means, missing a
    ~7.8x variation in called-strike value between 0-0 and 3-2 counts.
  - 2020 excluded from training; scored by averaging 2019 and 2021 predictions.
  - Qualification threshold raised to 300 pitches (2020: ~111).
  - Scoring config (S1 baseline adjustment and shrinkage) determined by
    pre-registered adoption criteria in diagnostics/location_v3/spec_lock.json.

What it measures: where pitches ended up and how valuable those locations
were given count, pitch type, batter handedness. Does NOT measure pitch
shape, delivery, or sequencing.

Scale: 100 = league mean, 10 points = 1 SD. A pitcher at 110 has location
one SD better than average. Same plus-scale convention as Stuff+ v4,
Tunneling+ v2, Deception+ v1.
"""
import hashlib
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from analytics.location_features import to_plus_scale
from analytics.stuff_features import merge_pitch_type, exclude_junk_pitch_types
from config.qualifiers import MODEL_EXCLUDED

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DIAG_DIR = Path(__file__).resolve().parent.parent / "diagnostics" / "location_v3"
META_PATH = MODELS_DIR / "location_plus_v3_meta.json"
RV_PATH = DIAG_DIR / "count_rv_table.json"

# Fallback to v2 if v3 models not yet trained
V2_META_PATH = MODELS_DIR / "location_plus_v2_meta.json"

MIN_PITCHES = 300
MIN_PITCHES_2020 = 111
_TTL = 600
_cache = {}

_QUERY_COLS = ["pitcher_id", "plate_x", "plate_z", "balls", "strikes",
               "pitch_type", "stand", "season"]

LOCATION_ACTIVE_VERSION = "v3" if META_PATH.exists() else "v2"


def _meta():
    if "meta" not in _cache:
        if not META_PATH.exists():
            _cache["meta"] = None
            return None
        _cache["meta"] = json.loads(META_PATH.read_text())
    return _cache["meta"]


def _rv():
    if "rv" not in _cache:
        if not RV_PATH.exists():
            _cache["rv"] = None
            _cache["bip_classes"] = []
            _cache["rv_arrays"] = None
            return None, [], None
        d = json.loads(RV_PATH.read_text())
        rv_count = d["rv_count"]
        bip_classes = d["bip_classes"]
        # Precompute numpy arrays for fast vectorized lookup
        rv_arrays, rv_hb, bip_rv_matrix = _build_rv_arrays(rv_count, bip_classes)
        _cache["rv"] = rv_count
        _cache["bip_classes"] = bip_classes
        _cache["rv_arrays"] = rv_arrays
        _cache["rv_hb"] = rv_hb
        _cache["bip_rv_matrix"] = bip_rv_matrix
    return _cache["rv"], _cache["bip_classes"], _cache.get("rv_arrays")


def _build_rv_arrays(rv_count, bip_classes):
    MAX_BALLS, MAX_STRIKES = 4, 3
    def build_count_matrix(event):
        mat = np.zeros((MAX_BALLS, MAX_STRIKES), dtype=np.float64)
        all_mean = rv_count.get(f"{event}__all", 0.0)
        for b in range(MAX_BALLS):
            for s in range(MAX_STRIKES):
                mat[b, s] = rv_count.get(f"{event}__{b}_{s}", all_mean)
        return mat
    rv_arrays = {
        "whiff": build_count_matrix("whiff"),
        "foul": build_count_matrix("foul"),
        "called_strike": build_count_matrix("called_strike"),
        "ball": build_count_matrix("ball"),
    }
    rv_hb = rv_count.get("hbp__all", 0.36)
    bip_rv_matrix = np.zeros((len(bip_classes), MAX_BALLS, MAX_STRIKES), dtype=np.float64)
    for i, cls in enumerate(bip_classes):
        all_mean = rv_count.get(f"bip__{cls}__all", 0.0)
        for b in range(MAX_BALLS):
            for s in range(MAX_STRIKES):
                bip_rv_matrix[i, b, s] = rv_count.get(f"bip__{cls}__{b}_{s}", all_mean)
    return rv_arrays, rv_hb, bip_rv_matrix


def _models():
    if "m" not in _cache:
        if not META_PATH.exists():
            # Fall back to v2
            from analytics.location_v2_model import _models as v2_models
            _cache["m"] = v2_models()
            _cache["model_version"] = "v2"
            return _cache["m"]
        _cache["m"] = {
            n: lgb.Booster(model_file=str(MODELS_DIR / f"location_plus_v3_{n}.txt"))
            for n in ("xswing", "xcs", "xwhiff", "xfoul", "bip")
        }
        _cache["model_version"] = "v3"
    return _cache["m"]


def _cat_levels(conn):
    if "cat_levels" not in _cache:
        meta = _meta()
        if meta and "all_cats" in meta:
            _cache["cat_levels"] = meta["all_cats"]
        else:
            # Build from DB
            raw = pd.read_sql("""
                SELECT DISTINCT pitch_type, stand, season FROM pitches
                WHERE season BETWEEN 2015 AND 2026
            """, conn)
            raw["pitch_type"] = raw["pitch_type"].replace({"KC": "CU"})
            raw = raw[~raw["pitch_type"].isin(MODEL_EXCLUDED)]
            raw["season"] = raw["season"].astype(str)
            _cache["cat_levels"] = {
                "pitch_type": sorted(raw["pitch_type"].dropna().unique().tolist()),
                "stand": sorted(raw["stand"].dropna().unique().tolist()),
                "season": sorted(raw["season"].dropna().unique().tolist()),
            }
    return _cache["cat_levels"]


def _prep(df, cat_levels, feat_cols=None, cat_cols=None):
    meta = _meta()
    if feat_cols is None:
        feat_cols = meta["feat_cols"] if meta else [
            "plate_x", "plate_z", "balls", "strikes", "pitch_type", "stand", "season"]
    if cat_cols is None:
        cat_cols = meta["cat_cols"] if meta else ["pitch_type", "stand", "season"]
    df = df.copy()
    df = merge_pitch_type(df)
    df = exclude_junk_pitch_types(df)
    df["season"] = df["season"].astype(str)
    for col in ["plate_x", "plate_z", "balls", "strikes"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["plate_x", "plate_z", "balls", "strikes", "pitch_type", "stand"])
    for col in cat_cols:
        levels = cat_levels.get(col, sorted(df[col].dropna().astype(str).unique().tolist()))
        df[col] = pd.Categorical(
            df[col].astype(str).where(df[col].astype(str).isin(levels), other=np.nan),
            categories=levels,
        )
    return df


def _combine_xrv(df, models, rv_count, rv_arrays, rv_hb, bip_classes, bip_rv_matrix,
                  hbp_rate, feat_cols):
    """Vectorized xRV computation using count-specific run-value table."""
    X = df[feat_cols]
    p_sw = models["xswing"].predict(X)
    p_cs = models["xcs"].predict(X)
    p_wh = models["xwhiff"].predict(X)
    p_fo = models["xfoul"].predict(X)
    p_bip = models["bip"].predict(X)  # [n, n_classes]

    balls = df["balls"].clip(0, 3).astype(int).values
    strikes = df["strikes"].clip(0, 2).astype(int).values

    rv_wh = rv_arrays["whiff"][balls, strikes]
    rv_fo = rv_arrays["foul"][balls, strikes]
    rv_cs = rv_arrays["called_strike"][balls, strikes]
    rv_ba = rv_arrays["ball"][balls, strikes]

    bip_rvs = np.einsum("ni,in->n", p_bip, bip_rv_matrix[:, balls, strikes])
    xrv_contact = p_fo * rv_fo + (1.0 - p_fo) * bip_rvs
    xrv_swing = p_wh * rv_wh + (1.0 - p_wh) * xrv_contact
    p_ball = np.clip(1.0 - p_cs - hbp_rate, 0.0, 1.0)
    xrv_no_swing = p_cs * rv_cs + hbp_rate * rv_hb + p_ball * rv_ba
    return p_sw * xrv_swing + (1.0 - p_sw) * xrv_no_swing


def _raw_predictions(conn, season):
    key = f"raw_v3_{season}"
    entry = _cache.get(key)
    if entry is not None and (time.monotonic() - entry[1]) < _TTL:
        return entry[0]

    meta = _meta()
    feat_cols = meta["feat_cols"] if meta else [
        "plate_x", "plate_z", "balls", "strikes", "pitch_type", "stand", "season"]
    hbp_rate = meta.get("hbp_rate_no_swing", 0.00514) if meta else 0.00514

    from config.qualifiers import regular_season_filter
    df = pd.read_sql(
        f"SELECT {', '.join(_QUERY_COLS)} FROM pitches "
        f"WHERE season = ? AND delta_run_exp IS NOT NULL {regular_season_filter()}",
        conn, params=(season,),
    )
    if df.empty:
        out = pd.DataFrame(columns=["pitcher_id", "xrv"])
        _cache[key] = (out, time.monotonic())
        return out

    # Handle 2020: average predictions using season=2019 and season=2021
    cat_lv = _cat_levels(conn)
    rv_count, bip_classes, rv_arrays = _rv()
    if rv_arrays is None:
        # Fall back to v2
        from analytics.location_v2_model import _raw_predictions as v2_raw
        return v2_raw(conn, season)

    rv_hb = _cache.get("rv_hb", 0.36)
    bip_rv_matrix = _cache.get("bip_rv_matrix")

    if season == 2020:
        out_parts = []
        for fake_season in ["2019", "2021"]:
            df_s = df.copy()
            df_s["season"] = fake_season
            df_s_prep = _prep(df_s, cat_lv, feat_cols, meta.get("cat_cols") if meta else None)
            xrv = _combine_xrv(df_s_prep, _models(), rv_count, rv_arrays, rv_hb,
                                bip_classes, bip_rv_matrix, hbp_rate, feat_cols)
            out_parts.append(xrv)
        xrv_arr = (out_parts[0] + out_parts[1]) / 2
        df_prep = _prep(df, cat_lv, feat_cols, meta.get("cat_cols") if meta else None)
        out = df_prep[["pitcher_id"]].copy()
        out["xrv"] = xrv_arr
    else:
        df_prep = _prep(df, cat_lv, feat_cols, meta.get("cat_cols") if meta else None)
        xrv_arr = _combine_xrv(df_prep, _models(), rv_count, rv_arrays, rv_hb,
                                bip_classes, bip_rv_matrix, hbp_rate, feat_cols)
        out = df_prep[["pitcher_id"]].copy()
        out["xrv"] = xrv_arr

    _cache[key] = (out, time.monotonic())
    return out


def raw_predictions_if_cached(season):
    """Return cached raw predictions if available and fresh, else None. Non-blocking.
    Mirrors the same helper in location_model.py used by career.py."""
    key = f"raw_v3_{season}"
    entry = _cache.get(key)
    if entry is not None and (time.monotonic() - entry[1]) < _TTL:
        return entry[0]
    return None


def clear_cache():
    _cache.clear()


def _min_pitches(season):
    return MIN_PITCHES_2020 if season == 2020 else MIN_PITCHES


def leaderboard(conn, season, min_pitches=None, top_n=15):
    if min_pitches is None:
        min_pitches = _min_pitches(season)
    league = _raw_predictions(conn, season)
    if league.empty:
        return {"n_qualified": 0, "top": [], "bottom": []}

    g = league.groupby("pitcher_id").agg(
        mean_xrv=("xrv", "mean"), n_pitches=("xrv", "size")
    ).reset_index()
    q = g[g["n_pitches"] >= min_pitches].copy()
    if q.empty or q["mean_xrv"].std() == 0:
        return {"n_qualified": 0, "top": [], "bottom": []}

    lm = q["mean_xrv"].mean()
    ls = q["mean_xrv"].std()
    q["score"] = to_plus_scale(q["mean_xrv"], lm, ls)

    ids = q["pitcher_id"].astype(int).tolist()
    names = pd.read_sql(
        f"SELECT p.player_id AS pitcher_id, p.name, t.abbreviation AS team_abbrev "
        f"FROM players p LEFT JOIN teams t ON p.team_id = t.team_id "
        f"WHERE p.player_id IN ({','.join('?'*len(ids))})",
        conn, params=ids,
    )
    merged = q.merge(names, on="pitcher_id", how="left")
    season_flag = "60-game season" if season == 2020 else ""

    def _rows(d):
        return [
            dict(pitcher_id=int(r.pitcher_id), name=r.name,
                 team_abbrev=r.team_abbrev, total_pitches=int(r.n_pitches),
                 score=round(float(r.score), 1),
                 season_flag=season_flag)
            for r in d.itertuples()
        ]
    top = merged.sort_values("score", ascending=False).head(top_n)
    bottom = merged.sort_values("score", ascending=True).head(top_n)
    return {"n_qualified": int(len(q)), "top": _rows(top), "bottom": _rows(bottom),
            "model_version": LOCATION_ACTIVE_VERSION}


def pitcher_score(conn, season, pitcher_id, league_raw=None, min_pitches=None):
    if min_pitches is None:
        min_pitches = _min_pitches(season)
    if league_raw is None:
        league_raw = _raw_predictions(conn, season)
    if league_raw.empty:
        return None
    g = league_raw.groupby("pitcher_id").agg(
        mean_xrv=("xrv", "mean"), n=("xrv", "size")
    ).reset_index()
    qualified = g[g["n"] >= min_pitches]
    row = qualified[qualified["pitcher_id"] == pitcher_id]
    if row.empty or qualified["mean_xrv"].std() == 0:
        return None
    return float(to_plus_scale(
        row["mean_xrv"].iloc[0],
        qualified["mean_xrv"].mean(),
        qualified["mean_xrv"].std(),
    ))


def pitcher_breakdown(conn, season, pitcher_id, league_raw=None, min_pitches=None):
    """Per-pitcher Location+ v3 score with event component breakdown."""
    if min_pitches is None:
        min_pitches = _min_pitches(season)
    if league_raw is None:
        league_raw = _raw_predictions(conn, season)
    if league_raw.empty:
        return None

    g = league_raw.groupby("pitcher_id").agg(
        mean_xrv=("xrv", "mean"), n_pitches=("xrv", "size")
    ).reset_index()
    qualified = g[g["n_pitches"] >= min_pitches]
    row = qualified[qualified["pitcher_id"] == pitcher_id]
    if row.empty or qualified["mean_xrv"].std() == 0:
        return None

    lm = qualified["mean_xrv"].mean()
    ls = qualified["mean_xrv"].std()
    score = float(to_plus_scale(row["mean_xrv"].iloc[0], lm, ls))
    percentile = float((qualified["mean_xrv"] < row["mean_xrv"].iloc[0]).mean() * 100)

    meta = _meta()
    return dict(
        n_pitches=int(row["n_pitches"].iloc[0]),
        score=round(score, 1),
        percentile=round(percentile, 1),
        season_flag="60-game season" if season == 2020 else "",
        model_version=LOCATION_ACTIVE_VERSION,
        model_hash=meta.get("model_hashes", {}) if meta else {},
        contributions=[],  # event-component decomposition not yet implemented in runtime
        shap_note="Location+ v3 decomposes into event components (swing, CS, whiff, foul, BIP); SHAP breakdown spans five classifiers and is not available in real-time",
    )


def model_meta():
    """Return model metadata for /api/location-model/meta endpoint."""
    meta = _meta()
    if meta is None:
        return {"version": "v2", "status": "v3 not yet trained", "active": "v2"}
    return {
        "version": "v3",
        "active": LOCATION_ACTIVE_VERSION,
        "training_seasons": meta.get("train_seasons", []),
        "scored_seasons": meta.get("scored_seasons", []),
        "features": meta.get("feat_cols", []),
        "model_hashes": meta.get("model_hashes", {}),
        "trained_at": meta.get("trained_at", ""),
        "scoring_config": meta.get("scoring_config", {}),
        "min_pitches": MIN_PITCHES,
        "min_pitches_2020": MIN_PITCHES_2020,
        "rv_table": "count_specific_v3",
        "status": "validated" if (DIAG_DIR / "gate_results.json").exists() and
                  json.loads((DIAG_DIR / "gate_results.json").read_text()).get("sign_check", {}).get("pass")
                  else "provisional",
    }
