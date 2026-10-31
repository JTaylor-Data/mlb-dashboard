"""
Location+ v2 - runtime scoring. Five-classifier decomposed-event architecture:
  xSwing%, xCS%, xWhiff%, xFoul%, BIP×EV → fixed run-value table → per-pitch xRV
  → pitcher-season aggregation → plus-scale.

Replaces location_model.py (v1 single regressor) as the live scorer for the
/api/location/* routes. The v1 sign-flip issue (ERA r = −0.257, xFIP r = −0.507)
and stability collapse (YoY r → 0.009 by 2024→2025) are both resolved in v2.

See docs/methodology/in_progress/LOCATION_PLUS_V2_METHODOLOGY_IN_PROGRESS.html.
"""
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from analytics.location_features import to_plus_scale
from analytics.stuff_features import merge_pitch_type, exclude_junk_pitch_types

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
V2_ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "training" / "v2_artifacts"
META_PATH = MODELS_DIR / "location_plus_v2_meta.json"
RV_PATH = V2_ARTIFACTS_DIR / "run_value_table.json"

MIN_PITCHES = 100
_TTL = 600
_cache = {}

_QUERY_COLS = ["pitcher_id", "plate_x", "plate_z", "balls", "strikes",
               "pitch_type", "stand", "season"]


def _meta():
    if "meta" not in _cache:
        _cache["meta"] = json.loads(META_PATH.read_text())
    return _cache["meta"]


def _rv():
    if "rv" not in _cache:
        d = json.loads(RV_PATH.read_text())
        _cache["rv"] = d["rv_table"]
        _cache["bip_classes"] = d["bip_classes"]
    return _cache["rv"], _cache["bip_classes"]


def _models():
    if "m" not in _cache:
        _cache["m"] = {
            n: lgb.Booster(model_file=str(MODELS_DIR / f"location_plus_v2_{n}.txt"))
            for n in ("xswing", "xcs", "xwhiff", "xfoul", "bip")
        }
    return _cache["m"]


def _cat_levels(conn):
    """Derive category levels from training range (2015–2026), sorted — same
    ordering the training code used (sorted unique values). Cached per process."""
    if "cat_levels" not in _cache:
        raw = pd.read_sql("""
            SELECT DISTINCT pitch_type, stand, season FROM pitches
            WHERE season BETWEEN 2015 AND 2026
        """, conn)
        raw["pitch_type"] = raw["pitch_type"].replace({"KC": "CU"})
        from config.qualifiers import MODEL_EXCLUDED
        raw = raw[~raw["pitch_type"].isin(MODEL_EXCLUDED)]
        raw["season"] = raw["season"].astype(str)
        _cache["cat_levels"] = {
            "pitch_type": sorted(raw["pitch_type"].dropna().unique().tolist()),
            "stand": sorted(raw["stand"].dropna().unique().tolist()),
            "season": sorted(raw["season"].dropna().unique().tolist()),
        }
    return _cache["cat_levels"]


def _prep(df, cat_levels):
    meta = _meta()
    df = df.copy()
    df = merge_pitch_type(df)
    df = exclude_junk_pitch_types(df)
    df["season"] = df["season"].astype(str)
    for col in ["plate_x", "plate_z", "balls", "strikes"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["plate_x", "plate_z", "balls", "strikes", "pitch_type", "stand"])
    for col in meta["cat_cols"]:
        levels = cat_levels.get(col, sorted(df[col].dropna().astype(str).unique().tolist()))
        df[col] = pd.Categorical(
            df[col].astype(str).where(df[col].astype(str).isin(levels), other=np.nan),
            categories=levels,
        )
    return df


def _combine_xrv(df, models, rv, bip_classes, hbp_rate):
    meta = _meta()
    X = df[meta["feat_cols"]]
    m = models
    p_sw = m["xswing"].predict(X)       # P(swing)
    p_cs = m["xcs"].predict(X)          # P(called_strike | no_swing) — applied to all pitches
    p_wh = m["xwhiff"].predict(X)       # P(whiff | swing)
    p_fo = m["xfoul"].predict(X)        # P(foul | contact)
    p_bip = m["bip"].predict(X)         # [n, n_classes] BIP distribution

    bip_rvs = np.array([rv.get(c, 0.0) for c in bip_classes])
    xrv_bip = p_bip @ bip_rvs
    xrv_contact = p_fo * rv["foul"] + (1.0 - p_fo) * xrv_bip
    xrv_swing = p_wh * rv["whiff"] + (1.0 - p_wh) * xrv_contact

    p_ball = np.clip(1.0 - p_cs - hbp_rate, 0.0, 1.0)
    xrv_no_swing = (p_cs * rv["called_strike"]
                    + hbp_rate * rv["hbp"]
                    + p_ball * rv["ball"])

    return p_sw * xrv_swing + (1.0 - p_sw) * xrv_no_swing


def _raw_predictions(conn, season):
    key = f"raw_{season}"
    entry = _cache.get(key)
    if entry is not None and (time.monotonic() - entry[1]) < _TTL:
        return entry[0]

    df = pd.read_sql(
        f"SELECT {', '.join(_QUERY_COLS)} FROM pitches "
        "WHERE season = ? AND delta_run_exp IS NOT NULL",
        conn, params=(season,),
    )

    if df.empty:
        out = pd.DataFrame(columns=["pitcher_id", "xrv"])
        _cache[key] = (out, time.monotonic())
        return out

    cat_lv = _cat_levels(conn)
    df = _prep(df, cat_lv)
    rv, bip_classes = _rv()
    hbp_rate = _meta().get("hbp_rate_no_swing", 0.0051)
    xrv_arr = _combine_xrv(df, _models(), rv, bip_classes, hbp_rate)

    out = df[["pitcher_id"]].copy()
    out["xrv"] = xrv_arr
    _cache[key] = (out, time.monotonic())
    return out


def clear_cache():
    _cache.clear()


def leaderboard(conn, season, min_pitches=MIN_PITCHES, top_n=15):
    league = _raw_predictions(conn, season)
    if league.empty:
        return {"n_qualified": 0, "top": [], "bottom": []}

    g = league.groupby("pitcher_id").agg(
        mean_xrv=("xrv", "mean"), n_pitches=("xrv", "size")
    ).reset_index()
    q = g[g["n_pitches"] >= min_pitches].copy()
    if q.empty or q["mean_xrv"].std() == 0:
        return {"n_qualified": 0, "top": [], "bottom": []}

    league_mean = q["mean_xrv"].mean()
    league_std = q["mean_xrv"].std()
    q["score"] = to_plus_scale(q["mean_xrv"], league_mean, league_std)

    ids = q["pitcher_id"].astype(int).tolist()
    names = pd.read_sql(
        f"SELECT p.player_id AS pitcher_id, p.name, t.abbreviation AS team_abbrev "
        f"FROM players p LEFT JOIN teams t ON p.team_id = t.team_id "
        f"WHERE p.player_id IN ({','.join('?'*len(ids))})",
        conn, params=ids,
    )
    merged = q.merge(names, on="pitcher_id", how="left")

    def _rows(d):
        return [
            dict(pitcher_id=int(r.pitcher_id), name=r.name,
                 team_abbrev=r.team_abbrev, total_pitches=int(r.n_pitches),
                 score=round(float(r.score), 1))
            for r in d.itertuples()
        ]
    top = merged.sort_values("score", ascending=False).head(top_n)
    bottom = merged.sort_values("score", ascending=True).head(top_n)
    return {"n_qualified": int(len(q)), "top": _rows(top), "bottom": _rows(bottom)}


def pitcher_score(conn, season, pitcher_id, league_raw=None, min_pitches=MIN_PITCHES):
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


def pitcher_breakdown(conn, season, pitcher_id, league_raw, min_pitches=MIN_PITCHES):
    """Per-pitcher Location+ v2 score. SHAP breakdown is not available for the
    decomposed-classifier architecture (contributions span five classifiers +
    a run-value table, not one interpretable model)."""
    if league_raw.empty:
        return None
    g = league_raw.groupby("pitcher_id").agg(
        mean_xrv=("xrv", "mean"), n_pitches=("xrv", "size")
    ).reset_index()
    qualified = g[g["n_pitches"] >= min_pitches]
    row = qualified[qualified["pitcher_id"] == pitcher_id]
    if row.empty or qualified["mean_xrv"].std() == 0:
        return None
    league_mean = qualified["mean_xrv"].mean()
    league_std = qualified["mean_xrv"].std()
    score = float(to_plus_scale(row["mean_xrv"].iloc[0], league_mean, league_std))
    return dict(
        n_pitches=int(row["n_pitches"].iloc[0]),
        score=score,
        contributions=[],
        model_version="v2",
        shap_note="SHAP breakdown not available for v2 classifier architecture",
    )
