"""
Tunneling+ v2 - runtime scoring. Five-classifier decomposed-event architecture
applied to lag-context features (28 total: 6 lag slots × 4 features + 4 current
pitch categoricals):
  xSwing%, xCS%, xWhiff%, xFoul%, BIP×EV → fixed run-value table → per-pitch xRV
  → pitcher-season aggregation → plus-scale.

Replaces tunneling_v3_model.py as the live scorer for /api/tunneling/v3/* routes.
The v3 sign-flip (ERA r = −0.218, xFIP r = −0.348) is resolved in v2.

See docs/methodology/in_progress/TUNNELING_PLUS_V2_METHODOLOGY_IN_PROGRESS.html.
"""
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from analytics.location_features import to_plus_scale as _to_plus_scale
from analytics.tunneling_features import (
    lag_feature_columns, load_pitches, sequential_lag_features, MAX_LAG,
)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
V2_ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "training" / "v2_artifacts"
META_PATH = MODELS_DIR / "tunneling_plus_v2_meta.json"
RV_PATH = V2_ARTIFACTS_DIR / "run_value_table.json"
COMMIT_TABLE_PATH = (
    Path(__file__).resolve().parent.parent
    / "project_artifacts" / "training_artifacts"
    / "tunneling_v2_artifacts" / "200ms" / "pitcher_season_commit_point.csv"
)

MIN_SEASON_PITCHES = 100
_TTL = 600
_cache = {}

LAG_COLS = lag_feature_columns()
NUMERIC_LAG_COLS = [c for c in LAG_COLS if not c.endswith("_pitch_type")]
CATEGORICAL_LAG_COLS = [c for c in LAG_COLS if c.endswith("_pitch_type")]
CURRENT_CATEGORICAL = ["pitch_type", "p_throws", "stand", "season"]
ALL_CATEGORICAL = CATEGORICAL_LAG_COLS + CURRENT_CATEGORICAL
ALL_FEATURES = NUMERIC_LAG_COLS + ALL_CATEGORICAL


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
            n: lgb.Booster(model_file=str(MODELS_DIR / f"tunneling_plus_v2_{n}.txt"))
            for n in ("xswing", "xcs", "xwhiff", "xfoul", "bip")
        }
    return _cache["m"]


def _load_commit_table():
    if "commit_table" not in _cache:
        _cache["commit_table"] = pd.read_csv(COMMIT_TABLE_PATH)
    return _cache["commit_table"]


def _cat_levels(conn):
    """Derive category levels from training range (2015–2026), sorted — same
    ordering the training code used. Lag pitch_type columns share the same
    levels as pitch_type. Cached per process."""
    if "cat_levels" not in _cache:
        raw = pd.read_sql("""
            SELECT DISTINCT pitch_type, p_throws, stand, season FROM pitches
            WHERE season BETWEEN 2015 AND 2026
        """, conn)
        raw["pitch_type"] = raw["pitch_type"].replace({"KC": "CU"})
        raw = raw[~raw["pitch_type"].isin({"PO", "UN", "EP"})]
        raw["season"] = raw["season"].astype(str)
        pt_levels = sorted(raw["pitch_type"].dropna().astype(str).unique().tolist())
        # nan string for missing lag slots
        pt_levels_with_nan = pt_levels  # LightGBM handles true NaN natively
        lv = {
            "pitch_type": pt_levels,
            "p_throws": sorted(raw["p_throws"].dropna().astype(str).unique().tolist()),
            "stand": sorted(raw["stand"].dropna().astype(str).unique().tolist()),
            "season": sorted(raw["season"].dropna().unique().tolist()),
        }
        # Lag pitch_type columns share the same levels
        for col in CATEGORICAL_LAG_COLS:
            lv[col] = pt_levels_with_nan
        _cache["cat_levels"] = lv
    return _cache["cat_levels"]


def _stringify_keep_na(series):
    is_na = series.isna()
    return pd.Series(np.where(is_na, np.nan, series.astype(str)), index=series.index)


def _prepare_features(df, cat_levels):
    feat = sequential_lag_features(df, _load_commit_table(), max_lag=MAX_LAG)
    feat["season"] = feat["season"].astype(str)
    for col in ALL_CATEGORICAL:
        feat[col] = _stringify_keep_na(feat[col])
        levels = cat_levels.get(col)
        if levels:
            feat[col] = pd.Categorical(feat[col], categories=levels)
        else:
            feat[col] = pd.Categorical(feat[col])
    return feat


def _combine_xrv(df, models, rv, bip_classes, hbp_rate):
    meta = _meta()
    X = df[meta["feat_cols"]]
    m = models
    p_sw = m["xswing"].predict(X)
    p_cs = m["xcs"].predict(X)
    p_wh = m["xwhiff"].predict(X)
    p_fo = m["xfoul"].predict(X)
    p_bip = m["bip"].predict(X)  # [n, n_classes]

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

    df = load_pitches(conn, season)
    totals = df.groupby("pitcher_id").size()
    qualifying_ids = totals[totals >= MIN_SEASON_PITCHES].index
    df = df[df["pitcher_id"].isin(qualifying_ids)]

    if df.empty:
        out = pd.DataFrame(columns=["pitcher_id", "xrv"])
        _cache[key] = (out, time.monotonic())
        return out

    cat_lv = _cat_levels(conn)
    feat = _prepare_features(df, cat_lv)
    rv, bip_classes = _rv()
    hbp_rate = _meta().get("hbp_rate_no_swing", 0.0051)
    xrv_arr = _combine_xrv(feat, _models(), rv, bip_classes, hbp_rate)

    out = feat[["pitcher_id"]].copy()
    out["xrv"] = xrv_arr
    _cache[key] = (out, time.monotonic())
    return out


def raw_predictions_if_cached(season):
    """Return cached raw predictions if available and fresh, else None. Non-blocking.
    Mirrors the same helper in tunneling_v3_model.py used by career.py."""
    key = f"raw_{season}"
    entry = _cache.get(key)
    if entry is not None and (time.monotonic() - entry[1]) < _TTL:
        return entry[0]
    return None


def clear_cache():
    _cache.clear()


def leaderboard(conn, season, top_n=15):
    league = _raw_predictions(conn, season)
    if league.empty or league["xrv"].std() == 0:
        return {"n_qualified": 0, "top": [], "bottom": []}

    g = league.groupby("pitcher_id").agg(
        mean_xrv=("xrv", "mean"), n_pitches=("xrv", "size")
    ).reset_index()
    if g.empty or g["mean_xrv"].std() == 0:
        return {"n_qualified": 0, "top": [], "bottom": []}

    league_mean = g["mean_xrv"].mean()
    league_std = g["mean_xrv"].std()
    g["score"] = _to_plus_scale(g["mean_xrv"], league_mean, league_std)

    ids = g["pitcher_id"].astype(int).tolist()
    names = pd.read_sql(
        f"SELECT p.player_id AS pitcher_id, p.name, t.abbreviation AS team_abbrev "
        f"FROM players p LEFT JOIN teams t ON p.team_id = t.team_id "
        f"WHERE p.player_id IN ({','.join('?'*len(ids))})",
        conn, params=ids,
    )
    merged = g.merge(names, on="pitcher_id", how="left")

    def _rows(d):
        return [
            dict(pitcher_id=int(r.pitcher_id), name=r.name,
                 team_abbrev=r.team_abbrev, total_pitches=int(r.n_pitches),
                 score=round(float(r.score), 1))
            for r in d.itertuples()
        ]
    top = merged.sort_values("score", ascending=False).head(top_n)
    bottom = merged.sort_values("score", ascending=True).head(top_n)
    return {"n_qualified": int(len(g)), "top": _rows(top), "bottom": _rows(bottom)}


def pitcher_score(conn, season, pitcher_id, league_raw=None):
    if league_raw is None:
        league_raw = _raw_predictions(conn, season)
    if league_raw.empty or league_raw["xrv"].std() == 0:
        return None
    row = league_raw[league_raw["pitcher_id"] == pitcher_id]
    if row.empty:
        return None
    mean, std = league_raw["xrv"].mean(), league_raw["xrv"].std()
    return float(_to_plus_scale(row["xrv"].mean(), mean, std))


def pitcher_breakdown(conn, season, pitcher_id, league_raw):
    """Per-pitcher Tunneling+ v2 score. SHAP breakdown is not available for
    the decomposed-classifier architecture."""
    if league_raw.empty or league_raw["xrv"].std() == 0:
        return None
    g = league_raw.groupby("pitcher_id").agg(
        mean_xrv=("xrv", "mean"), n_pitches=("xrv", "size")
    ).reset_index()
    row = g[g["pitcher_id"] == pitcher_id]
    if row.empty:
        return None
    mean, std = g["mean_xrv"].mean(), g["mean_xrv"].std()
    if std == 0:
        return None
    score = float(_to_plus_scale(row["mean_xrv"].iloc[0], mean, std))
    return dict(
        n_pitches=int(row["n_pitches"].iloc[0]),
        score=score,
        lag_slot_contributions=[],
        model_version="v2",
        shap_note="SHAP breakdown not available for v2 classifier architecture",
    )
