"""
Pitcher- and batter-side splits engines, built entirely on top of `pitches`
(joined against `players`/`games` where needed) - no new tables, per the
project's single-source-of-truth rule.
"""
import numpy as np
import pandas as pd

from analytics.metrics import compute_barrel, classify_pitcher_archetype
from config.qualifiers import EXCLUDED_PITCH_TYPES, MIN_PITCH_TYPE_USAGE_FRACTION

# Actual All-Star Game date per season (the 1st/2nd-half boundary) - each
# season's own break, not just 2026's, now that multiple seasons are loaded.
ALL_STAR_DATES = {
    2021: "2021-07-13", 2022: "2022-07-19", 2023: "2023-07-11",
    2024: "2024-07-16", 2025: "2025-07-15", 2026: "2026-07-14",
}

# Statcast `events` classification, used to build OPS/wOBA-adjacent counting
# stats from PA-ending rows (events IS NOT NULL).
HIT_BASES = {"single": 1, "double": 2, "triple": 3, "home_run": 4}
WALK_EVENTS = {"walk", "intent_walk"}
HBP_EVENTS = {"hit_by_pitch"}
SAC_FLY_EVENTS = {"sac_fly", "sac_fly_double_play"}
SAC_BUNT_EVENTS = {"sac_bunt", "sac_bunt_double_play"}
STRIKEOUT_EVENTS = {"strikeout", "strikeout_double_play"}
NOT_AB_EVENTS = WALK_EVENTS | HBP_EVENTS | SAC_FLY_EVENTS | SAC_BUNT_EVENTS | {"catcher_interf"}

SWING_DESCRIPTIONS = {
    "swinging_strike", "swinging_strike_blocked", "foul", "foul_tip",
    "hit_into_play", "foul_bunt", "missed_bunt",
}
WHIFF_DESCRIPTIONS = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}
# Rulebook "strike" - called strikes plus every swing outcome (whiffs, fouls,
# balls in play) - i.e. everything except a ball/HBP/pitchout.
STRIKE_DESCRIPTIONS = SWING_DESCRIPTIONS | {"called_strike"}
IN_ZONE_CODES = set(range(1, 10))   # Statcast zones 1-9
OUT_ZONE_CODES = {11, 12, 13, 14}


def _load_pitches(conn, season, entity_col, entity_id=None, pitch_type=None):
    where, params = ["season = ?"], [season]
    if entity_id is not None:
        where.append(f"{entity_col} = ?")
        params.append(entity_id)
    if pitch_type is not None:
        where.append("pitch_type = ?")
        params.append(pitch_type)
    where_sql = "WHERE " + " AND ".join(where)
    return pd.read_sql(f"SELECT * FROM pitches {where_sql}", conn, params=params)


def _add_common_columns(df, season):
    df = df.copy()
    df["month"] = pd.to_datetime(df["game_date"]).dt.strftime("%Y-%m")
    if season not in ALL_STAR_DATES:
        import warnings
        warnings.warn(f"ALL_STAR_DATES has no entry for season {season}; using {season}-07-14 as fallback", stacklevel=2)
    all_star_date = ALL_STAR_DATES.get(season, f"{season}-07-14")
    df["half"] = np.where(df["game_date"] < all_star_date, "1H", "2H")
    df["is_swing"] = df["description"].isin(SWING_DESCRIPTIONS)
    df["is_whiff"] = df["description"].isin(WHIFF_DESCRIPTIONS)
    df["in_zone"] = df["zone"].isin(IN_ZONE_CODES)
    df["out_zone"] = df["zone"].isin(OUT_ZONE_CODES)
    return df


def _pitch_rate_metrics(df):
    """swing%, whiff%, chase% over a slice of pitch-level rows."""
    n = len(df)
    swings = df["is_swing"].sum()
    swing_pct = swings / n if n else np.nan
    whiff_pct = df["is_whiff"].sum() / swings if swings else np.nan
    out_zone_pitches = df["out_zone"].sum()
    chase_pct = (df["is_swing"] & df["out_zone"]).sum() / out_zone_pitches if out_zone_pitches else np.nan
    return swing_pct, whiff_pct, chase_pct


def _barrel_rate(pa_df):
    """Barrels / batted ball events (rows with a measured launch_speed)."""
    batted = pa_df[pa_df["launch_speed"].notna()]
    if len(batted) == 0:
        return np.nan
    barrels = compute_barrel(batted["launch_speed"], batted["launch_angle"]).sum()
    return barrels / len(batted)


def _pa_outcome_metrics(pa_df):
    """OPS, wOBA, xwOBA, K%, BABIP from a slice of PA-ending rows (events
    not null). BABIP = (H - HR) / (AB - K - HR + SF), the standard formula -
    balls actually put in play that fell for hits, excluding the outcomes
    (strikeouts, homers) that never involve a fielder at all."""
    n_pa = len(pa_df)
    if n_pa == 0:
        return dict(OPS=np.nan, wOBA=np.nan, xwOBA=np.nan, K_pct=np.nan, babip=np.nan, n_pa=0)

    events = pa_df["events"]
    bases = events.map(HIT_BASES).fillna(0)
    hits = (bases > 0).sum()
    tb = bases.sum()
    bb = events.isin(WALK_EVENTS).sum()
    hbp = events.isin(HBP_EVENTS).sum()
    sf = events.isin(SAC_FLY_EVENTS).sum()
    not_ab = events.isin(NOT_AB_EVENTS).sum()
    ab = n_pa - not_ab
    k = events.isin(STRIKEOUT_EVENTS).sum()
    hr = (events == "home_run").sum()

    obp_den = ab + bb + hbp + sf
    obp = (hits + bb + hbp) / obp_den if obp_den else np.nan
    slg = tb / ab if ab else np.nan
    ops = obp + slg if pd.notna(obp) and pd.notna(slg) else np.nan

    babip_den = ab - k - hr + sf
    babip = (hits - hr) / babip_den if babip_den > 0 else np.nan

    qualifying = pa_df["woba_denom"] == 1
    denom = qualifying.sum()
    woba = pa_df.loc[qualifying, "woba_value"].sum() / denom if denom else np.nan
    xwoba = pa_df.loc[qualifying, "estimated_woba_using_speedangle"].sum() / denom if denom else np.nan

    return dict(OPS=ops, wOBA=woba, xwOBA=xwoba, K_pct=k / n_pa, babip=babip, n_pa=n_pa)


def compute_pitcher_splits(conn, season, split_by, pitcher_id=None, pitch_type=None):
    """
    Opponent-performance splits from the pitcher's side, filterable by pitcher
    and/or pitch_type. split_by is one of: 'home_away', 'month', 'half',
    'stand' (L/R of the opposing batter).

    Reports OPS, wOBA, xwOBA, swing%, whiff%, chase%, K% per split value (from
    the opponent's perspective - lower is better for the pitcher), plus the
    pitcher's own overall average for the same pitch_type filter as a
    `_overall` companion column for each metric (headline vs. baseline).
    """
    df = _load_pitches(conn, season, "pitcher_id", pitcher_id, pitch_type)
    if df.empty:
        return pd.DataFrame()
    df = _add_common_columns(df, season)

    if split_by == "home_away":
        df["split_value"] = np.where(df["inning_topbot"] == "Top", "Home", "Away")
    elif split_by == "stand":
        df["split_value"] = df["stand"]
    elif split_by in ("month", "half"):
        df["split_value"] = df[split_by]
    elif split_by == "pitch_type":
        df["split_value"] = df["pitch_type"].fillna("UN")
        # Drop excluded pitch types (UN, PO, EP, FA) and low-usage types (< 1% of pitches)
        df = df[~df["split_value"].isin(EXCLUDED_PITCH_TYPES)]
        total_pitches = len(df)
        if total_pitches > 0:
            type_counts = df["split_value"].value_counts()
            kept = type_counts[type_counts / total_pitches >= MIN_PITCH_TYPE_USAGE_FRACTION].index
            df = df[df["split_value"].isin(kept)]
    else:
        raise ValueError(f"Unknown split_by: {split_by}")

    rows = []
    for value, group in df.groupby("split_value"):
        row = _pa_outcome_metrics(group[group["events"].notna()])
        swing_pct, whiff_pct, chase_pct = _pitch_rate_metrics(group)
        row.update(split_value=value, n_pitches=len(group),
                    swing_pct=swing_pct, whiff_pct=whiff_pct, chase_pct=chase_pct)
        rows.append(row)

    overall = _pa_outcome_metrics(df[df["events"].notna()])
    overall["swing_pct"], overall["whiff_pct"], overall["chase_pct"] = _pitch_rate_metrics(df)

    out = pd.DataFrame(rows).set_index("split_value")
    cols = ["n_pa", "n_pitches", "OPS", "wOBA", "xwOBA", "swing_pct", "whiff_pct", "chase_pct", "K_pct"]
    for col in ["OPS", "wOBA", "xwOBA", "swing_pct", "whiff_pct", "chase_pct", "K_pct"]:
        out[f"{col}_overall"] = overall[col]
        cols.append(f"{col}_overall")
    return out[cols]


def compute_batter_splits(conn, season, split_by, batter_id=None, pitch_type=None,
                           pitcher_archetype_threshold=0.6, archetype_df=None):
    """
    Batter-performance splits, filterable by batter and/or pitch_type (the
    type of pitch the batter saw - independent of 'archetype', which still
    reflects the opposing pitcher's whole arsenal regardless of this filter).
    split_by is one of: 'home_away', 'month', 'half', 'p_throws' (L/R of the
    opposing pitcher), 'archetype' (opposing pitcher's East-West/North-South
    arsenal shape - see metrics.classify_pitcher_archetype; pitchers without a
    one-sided arsenal are excluded from this split only, per design).

    `archetype_df`: optional pre-fetched result of classify_pitcher_archetype
    (a real, uncached league-wide pitch scan - see that function's own
    module) - api/routers/splits.py fetches it through a @ttl_cache()'d
    wrapper rather than every single archetype-split request re-scanning
    the whole season's pitches from scratch; omit to compute it fresh here.

    Reports OPS, wOBA, xwOBA, K%, whiff%, Barrel% per split value, plus the
    batter's own overall average (same pitch_type filter) as a `_overall`
    companion column per metric.
    """
    df = _load_pitches(conn, season, "batter_id", batter_id, pitch_type)
    if df.empty:
        return pd.DataFrame()
    df = _add_common_columns(df, season)

    if split_by == "archetype":
        arch = archetype_df if archetype_df is not None else classify_pitcher_archetype(conn, season, threshold=pitcher_archetype_threshold)
        df = df.merge(arch[["archetype"]], left_on="pitcher_id", right_index=True, how="inner")
        df["split_value"] = df["archetype"]
    elif split_by == "home_away":
        df["split_value"] = np.where(df["inning_topbot"] == "Bot", "Home", "Away")
    elif split_by == "p_throws":
        df["split_value"] = df["p_throws"]
    elif split_by in ("month", "half"):
        df["split_value"] = df[split_by]
    elif split_by == "pitch_type":
        df["split_value"] = df["pitch_type"].fillna("UN")
        # Drop excluded pitch types (UN, PO, EP, FA) and low-usage types (< 1% of pitches)
        df = df[~df["split_value"].isin(EXCLUDED_PITCH_TYPES)]
        total_pitches = len(df)
        if total_pitches > 0:
            type_counts = df["split_value"].value_counts()
            kept = type_counts[type_counts / total_pitches >= MIN_PITCH_TYPE_USAGE_FRACTION].index
            df = df[df["split_value"].isin(kept)]
    else:
        raise ValueError(f"Unknown split_by: {split_by}")

    rows = []
    for value, group in df.groupby("split_value"):
        pa_group = group[group["events"].notna()]
        row = _pa_outcome_metrics(pa_group)
        _, whiff_pct, _ = _pitch_rate_metrics(group)
        row.update(split_value=value, n_pitches=len(group),
                    whiff_pct=whiff_pct, barrel_pct=_barrel_rate(pa_group))
        rows.append(row)

    overall_pa = df[df["events"].notna()]
    overall = _pa_outcome_metrics(overall_pa)
    _, overall["whiff_pct"], _ = _pitch_rate_metrics(df)
    overall["barrel_pct"] = _barrel_rate(overall_pa)

    out = pd.DataFrame(rows).set_index("split_value")
    cols = ["n_pa", "n_pitches", "OPS", "wOBA", "xwOBA", "K_pct", "whiff_pct", "barrel_pct"]
    for col in ["OPS", "wOBA", "xwOBA", "K_pct", "whiff_pct", "barrel_pct"]:
        out[f"{col}_overall"] = overall[col]
        cols.append(f"{col}_overall")
    return out[cols]
