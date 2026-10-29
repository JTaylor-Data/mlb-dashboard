"""
/api/players and /api/player/{id} - the individual player page. Assembles
exactly what pages/1_Player_Page.py rendered, by calling the same analytics
modules it did, just packaged as JSON instead of Streamlit widgets/HTML.
"""
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from analytics.archetypes import ARCHETYPE_DESCRIPTIONS, classify_batter_archetype
from analytics.boxscore import fetch_boxscore, pitcher_daily_stats, season_era
from analytics.metrics import classify_pitcher_roles
from analytics.percentiles import (
    BATTER_POLARITY, PITCHER_POLARITY, batter_percentile_metrics,
    pitcher_percentile_metrics, player_percentiles, player_percentiles_allow_unqualified,
)
from analytics.player_page import (
    all_pitcher_arm_angles, batter_counting_stats, batter_pitch_type_table, batter_rolling_xwoba,
    batter_wrc_plus, field_zone_pcts, league_avg_xwoba, league_average_batter_row, league_average_swing_profile,
    league_hr_per_fb, movement_plot_data, pitch_usage_by_stand, pitcher_arm_angle,
    pitcher_approach_angle_percentile, pitcher_counting_stats, pitcher_league_avg_row, pitcher_pitch_table, pitcher_rubber_position,
    player_last_n_games, similar_arm_angle_movement, swing_profile,
)
from analytics.stuff_model import compute_stuff_plus, compute_stuff_plus_by_stand, overall_stuff_plus
from analytics.wpa import find_walkoff_batter

from api.cache import ttl_cache, ttl_cache_all_args
from api.db import get_conn
from api.presentation import (
    BATTER_FMT, BATTER_GROUPS, BATTER_LABELS, BATTER_TABLE_COLUMNS,
    PITCHER_FMT, PITCHER_GROUPS, PITCHER_LABELS, PITCHER_TABLE_COLUMNS,
    PITCHER_TABLE_FMT, format_row, format_value,
)
from api.routers.deception_report import _bracket as _deception_bracket
from api.routers.tunneling_report import _v3_raw_predictions_table
from analytics.tunneling_v3_model import pitcher_score as tunneling_v3_pitcher_score
from api.routers.location_report import _raw_predictions_table as _location_raw_predictions_table
from analytics.location_v3_model import pitcher_score as location_v3_pitcher_score
from api.routers.top_performers import _leaderboard
from analytics.tunneling_pair_composite import season_all_pairs, pitcher_pairs_from_season

router = APIRouter(prefix="/api")


def _real_sabermetric(conn, season, player_id, col):
    """Real season-total value from player_advanced_stats (see
    pulls/pull_sabermetrics.py), or None if this player isn't covered.
    Season-total only - no opposing-hand split exists for wRC+/xFIP, so
    callers only use this on the unsplit ALL view."""
    row = conn.execute(
        f"SELECT {col} FROM player_advanced_stats WHERE player_id = ? AND season = ?",
        (player_id, season),
    ).fetchone()
    return row[0] if row is not None else None


def _clean(v):
    """NaN/NaT -> None (and numpy scalars -> native python) so FastAPI's
    JSON encoder doesn't choke."""
    if v is None:
        return None
    if isinstance(v, (list, dict, str)):
        return v
    if pd.isna(v):
        return None
    return v.item() if hasattr(v, "item") else v


@ttl_cache()
def _pitcher_pool(conn, season, stand_filter):
    from analytics.percentile_cache import get_or_build_pool
    from config.model_policy import CURRENT_SEASON
    if season < CURRENT_SEASON:
        # Complete season — serve from disk-persisted pool (build + persist on first miss).
        return get_or_build_pool(conn, season, "pitcher", stand_filter)
    return pitcher_percentile_metrics(conn, season, stand_filter=stand_filter)


@ttl_cache()
def _batter_pool(conn, season, throws_filter):
    from analytics.percentile_cache import get_or_build_pool
    from config.model_policy import CURRENT_SEASON
    if season < CURRENT_SEASON:
        # Complete season — serve from disk-persisted pool (build + persist on first miss).
        return get_or_build_pool(conn, season, "batter", throws_filter)
    return batter_percentile_metrics(conn, season, throws_filter=throws_filter)


# Threshold-free (min_bf=0/min_pa=0) versions of the pools above - ONLY
# ever built when the currently-viewed player isn't in the real qualified
# pool (see player_percentiles_allow_unqualified's own docstring for why a
# second, unfiltered pool is needed at all: to look up an unqualified
# player's own raw value without ever letting their row into the
# population the percentile is RANKED against). Cached the same way, so a
# second unqualified player viewed in the same session doesn't re-pay this.
# Unqualified pools are NOT disk-persisted (they're used rarely, built fast
# once the qualified pool is warm, and their use cases don't justify the
# 6x disk footprint for 12 extra files per season).
@ttl_cache()
def _pitcher_pool_unqualified(conn, season, stand_filter):
    return pitcher_percentile_metrics(conn, season, min_bf=0, stand_filter=stand_filter)


@ttl_cache()
def _batter_pool_unqualified(conn, season, throws_filter):
    return batter_percentile_metrics(conn, season, min_pa=0, throws_filter=throws_filter)


@ttl_cache()
def _stuff_plus_table(conn, season):
    return compute_stuff_plus(conn, season)


@ttl_cache()
def _stuff_plus_by_stand_table(conn, season):
    return compute_stuff_plus_by_stand(conn, season)


def _stuff_breakdown(stuff_all, stuff_by_stand, overall_stuff, pitcher_id, pitch_types_ordered):
    """Grid data for the Stuff+ breakdown: columns = ALL + each pitch type
    (usage order), rows = ALL / vs RHH / vs LHH. Mirrors the layout of the
    per-pitch-type/handedness Stuff+ card the user referenced."""
    if pitcher_id not in stuff_all.index.get_level_values("pitcher_id"):
        return None

    all_rows = stuff_all.xs(pitcher_id, level="pitcher_id")
    stand_rows = None
    if pitcher_id in stuff_by_stand.index.get_level_values("pitcher_id"):
        stand_rows = stuff_by_stand.xs(pitcher_id, level="pitcher_id")

    columns = ["ALL"] + [pt for pt in pitch_types_ordered if pt in all_rows.index]

    def cell(col, row_stand):
        if col == "ALL":
            if row_stand == "ALL":
                return overall_stuff
            if stand_rows is None:
                return None
            sub = stand_rows[stand_rows.index.get_level_values("stand") == row_stand]
            if sub.empty:
                return None
            return float(np.average(sub["stuff_plus"], weights=sub["n_pitches"]))
        if row_stand == "ALL":
            return float(all_rows.loc[col, "stuff_plus"]) if col in all_rows.index else None
        if stand_rows is None or (col, row_stand) not in stand_rows.index:
            return None
        return float(stand_rows.loc[(col, row_stand), "stuff_plus"])

    rows = []
    for row_stand, row_label in [("ALL", "ALL"), ("R", "vs RHH"), ("L", "vs LHH")]:
        values = {}
        for col in columns:
            v = cell(col, row_stand)
            values[col] = round(v, 0) if _clean(v) is not None else None
        rows.append({"label": row_label, "values": values})

    return {"columns": columns, "rows": rows}


@ttl_cache()
def _pitcher_roles(conn, season):
    return classify_pitcher_roles(conn, season)


@ttl_cache()
def _league_hr_fb(conn, season):
    return league_hr_per_fb(conn, season)


@ttl_cache()
def _league_avg_batter_row(conn, season, throws_filter):
    return league_average_batter_row(conn, season, throws_filter)


# Mirrors _league_avg_batter_row above - pitcher_league_avg_row does a full
# SELECT * FROM pitches WHERE season=? (every column, the whole season) to
# build the pitch-table summary row, but was missing the same @ttl_cache()
# wrapper its batter equivalent already had - a real asymmetry bug, not a
# deliberate choice (nothing about this one is more per-player than the
# batter version). Every pitcher page load was re-scanning the full season
# table live; now it's a cache hit after the first pitcher/split combo.
@ttl_cache()
def _pitcher_league_avg_row(conn, season, stand_filter):
    return pitcher_league_avg_row(conn, season, stand_filter)


@ttl_cache()
def _league_swing_profile(conn, season):
    return league_average_swing_profile(conn, season)


# Split-INDEPENDENT per-player data (movement, arm angle, pitch usage,
# rubber position, unsplit swing profile) - each individually ~0.2-0.25s
# (measured directly), and every one of these gets called on EVERY request
# regardless of which split is selected (movement/arm-angle/usage/rubber
# deliberately ignore split_filter entirely; swing_profile is always called
# with split=None). Without caching, toggling ALL/L/R on the same player
# re-runs all ~5 of these unchanged queries every time - a real, measured
# ~1s of pure waste per toggle. Keyed by (player_id, season), same TTL-cache
# pattern already used for the league-wide pools above; safe to cache since
# none of these DataFrames/dicts are mutated by their callers (all read-only
# downstream in get_player()).
@ttl_cache()
def _movement_plot_data(conn, season, player_id):
    return movement_plot_data(conn, season, player_id, None)


# Real cost here: scans EVERY pitcher's season arm-angle average, then every
# pitch from whichever peer group falls out of that - genuinely comparable
# to a league-wide pool build, not a cheap per-player lookup, so this one
# gets cached same as the pools above (the other per-player functions in
# this section are cheap single-pitcher-filtered queries and intentionally
# NOT cached, per api/cache.py's own docstring).
@ttl_cache()
def _all_pitcher_arm_angles(conn, season):
    # Season-wide, not per-pitcher - see all_pitcher_arm_angles' docstring
    # for why this was split out and cached separately from the per-pitcher
    # function below (was a real, confirmed ~20s cost paid again for every
    # NEW pitcher a user looked at).
    return all_pitcher_arm_angles(conn, season)


@ttl_cache()
def _similar_arm_angle_movement(conn, season, player_id):
    return similar_arm_angle_movement(conn, season, player_id, arm_angles=_all_pitcher_arm_angles(conn, season))


@ttl_cache()
def _pitcher_arm_angle(conn, season, player_id):
    return pitcher_arm_angle(conn, season, player_id)


@ttl_cache()
def _pitch_usage_by_stand(conn, season, player_id):
    return pitch_usage_by_stand(conn, season, player_id)


@ttl_cache()
def _pitcher_rubber_position(conn, season, player_id):
    return pitcher_rubber_position(conn, season, player_id)


@ttl_cache()
def _pitcher_approach_angle_percentile(conn, season, player_id, throws):
    return pitcher_approach_angle_percentile(conn, season, player_id, throws)


@ttl_cache()
def _swing_profile_unsplit(conn, season, player_id):
    return swing_profile(conn, season, player_id, None)


# field_zone_pcts IS split-aware (varies with split_filter, unlike the
# five above) but still worth caching per (player_id, season, split) - a
# user re-toggling back to a split they already viewed this session
# (ALL -> L -> ALL) shouldn't re-pay the query.
@ttl_cache()
def _field_zone_pcts(conn, season, player_id, split_filter):
    return field_zone_pcts(conn, season, player_id, split_filter)


@ttl_cache_all_args()
def _season_era(player_id, season):
    try:
        return season_era(player_id, season)
    except Exception:
        # Live external call - a network hiccup shouldn't 500 the whole
        # player page, just fall back to "-" for the ERA card entry.
        return None


def _leaderboard_for_date(day_str):
    """Runs on a worker thread (see _last_5_games below) - opens its OWN
    sqlite3 connection rather than sharing the caller's, since sqlite3
    connections aren't safe to use across threads. _leaderboard itself
    already parallelizes its own internal boxscore fetches; this just
    stops _last_5_games from ALSO paying for up to 5 of those (one per
    game, one real date each) strictly one after another."""
    import sqlite3
    from api.db import DB_PATH
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    try:
        return _leaderboard(conn, day_str)
    finally:
        conn.close()


@ttl_cache()
def _last_5_games(conn, season, player_id, role):
    try:
        games = player_last_n_games(conn, season, player_id, role, 5)
    except Exception:
        return []

    # Each game's "was this player a top performer that day" check needs
    # that DAY's full leaderboard - up to 5 of these, one per game, each
    # independently ~1-1.5s (confirmed directly). Was a plain sequential
    # loop (~5-7s for a player whose last 5 games span 5 different, never-
    # before-viewed dates); running them concurrently cuts that to roughly
    # the slowest single date instead of the sum of all five.
    dates = [g["date"] for g in games]
    boards = {}
    if dates:
        with ThreadPoolExecutor(max_workers=min(len(dates), 5)) as pool:
            futures = {pool.submit(_leaderboard_for_date, d): d for d in dates}
            for future in futures:
                d = futures[future]
                try:
                    boards[d] = future.result()
                except Exception:
                    boards[d] = None

    for g in games:
        game_id = g.get("game_id")
        g["walkoff"] = role == "batter" and game_id is not None and find_walkoff_batter(conn, game_id) == player_id
        board = boards.get(g["date"])
        g["top_performer"] = bool(board) and any(e["player_id"] == player_id for e in board["entries"])
    return games


@ttl_cache()
def _deception_plus_badge(conn, player_id):
    """Deception+ has no season selector yet (see deception_report.py's
    module docstring) - always REPORT_SEASON (2025) regardless of the
    player page's own season, so a pitcher with no 2025 innings (e.g. a
    2026 rookie) genuinely has no score here. Fails soft to None (the
    frontend renders that as N/A, same as the existing Stuff+ badge does
    for a NaN) rather than erroring the whole player page. Cached (was not
    before) - this runs a real SHAP explainer call per pitcher, not free,
    and the underlying league table it reads from is already cached, so
    there's no reason the per-pitcher breakdown wasn't too."""
    try:
        result = _deception_bracket(player_id, "2025")
        return result.get("deception_plus") if result else None
    except Exception:
        return None


@ttl_cache()
def _tunneling_plus_badge(conn, season, player_id):
    """Tunneling+ v3 (the standard, promoted version) - live-scored, same
    try/except-wrapped pattern as _deception_plus_badge above, since this
    runs a real SHAP-adjacent per-pitcher scoring pass rather than reading
    a static file. v1's own static per-season report/pitcher-detail JSON
    (api/routers/tunneling_report.py's v1 endpoints) stays untouched for
    the separate 3D trajectory viewer, which still depends on its exact
    pair-shaped contract."""
    try:
        league = _v3_raw_predictions_table(conn, season)
        if league.empty:
            return None
        return tunneling_v3_pitcher_score(conn, season, player_id, league)
    except Exception:
        return None


@ttl_cache()
def _location_plus_badge(conn, season, player_id):
    """Location+ v3 badge scorer - real data from analytics.location_v3_model,
    min-300-pitches qualifying threshold applied (same as the v3 leaderboard)."""
    try:
        league = _location_raw_predictions_table(conn, season)
        if league.empty:
            return None
        return location_v3_pitcher_score(conn, season, player_id, league)
    except Exception:
        return None


@ttl_cache()
def _tunneling_pair_season(conn, season):
    """Precomputes all qualifying pitcher-season pairs with composite scores
    for one season. Expensive (~minutes), cached at the season grain so every
    subsequent pitcher page hit in the same session is a fast filter."""
    try:
        return season_all_pairs(conn, season)
    except Exception:
        return None


def _il_status(conn, player_id):
    """Current IL entry for this player, if any - see pulls/pull_injuries.py
    for how start_date/injury are real (parsed from the transactions feed)
    and min_return_date is a real computed lower bound, not a guess.

    Returns None when SHOW_IL_PLAQUES is False (postseason / offseason) so the
    backend never fetches injury data outside the regular season and the frontend
    receives null, rendering no IL badge.
    """
    from config.season_phase import SHOW_IL_PLAQUES
    if not SHOW_IL_PLAQUES:
        return None
    row = conn.execute(
        "SELECT il_type, il_label, start_date, injury, min_return_date FROM il_status WHERE player_id = ?",
        (player_id,),
    ).fetchone()
    if row is None:
        return None
    il_type, il_label, start_date, injury, min_return_date = row
    return {
        "il_type": il_type, "il_label": il_label, "start_date": start_date,
        "injury": injury, "min_return_date": min_return_date,
    }


def _norm_split(split):
    return None if split in (None, "ALL", "") else split


def _percentile_groups(pct_dict, labels, fmt_map, groups, qualified=True):
    """`qualified`: whether the player these percentiles were computed for
    cleared the real qualifying threshold (see
    player_percentiles_allow_unqualified) - carried through onto every
    row so the frontend can shade/mute an unqualified player's bars,
    matching Baseball Savant's own convention, rather than either hiding
    them entirely or showing them identically to a qualified player's."""
    out = []
    for title, metrics in groups:
        present = [m for m in metrics if m in pct_dict]
        if not present:
            continue
        rows = []
        for m in present:
            raw, p = pct_dict[m]
            rows.append({
                "key": m,
                "label": labels.get(m, m),
                "raw": _clean(raw),
                "pct": round(float(p), 1) if _clean(p) is not None else None,
                "value_str": format_value(raw, fmt_map.get(m, "{:.1f}")),
                "qualified": qualified,
            })
        out.append({"title": title, "metrics": rows})
    return out


def _table_to_rows(df, columns, fmt_map, league_avg_dict=None):
    # Use index-to-dict iteration — df is indexed by pitch_type (set_index called upstream)
    rows = [
        {
            "pitch_type": pitch_type,
            "cells": [{"key": k, "label": label, "value": format_row(row, fmt_map).get(k, "-")} for k, label in columns],
        }
        for pitch_type, row in df.to_dict("index").items()
    ]
    league_row = None
    if league_avg_dict is not None:
        formatted = format_row(league_avg_dict, fmt_map)
        league_row = [{"key": k, "label": label, "value": formatted.get(k, "-")} for k, label in columns]
    return rows, league_row


@ttl_cache()
def _all_players(conn, season):
    # One row per (player, role) across ALL seasons so inactive players still
    # appear in search. `latest_season` lets the frontend auto-select the
    # player's most recent active season instead of defaulting to the current
    # one (where they may have no data). The pitcher_role classification uses
    # the requested season for active players; inactive ones fall back to
    # their own latest season.
    # A player is a real pitcher only if their listed position says so (P or
    # the two-way tag TWP) AND they've pitched in >= 3 career games - the
    # games-pitched threshold alone isn't enough: 120 position players
    # (catchers, infielders, outfielders) have mopped up 3+ blowout games
    # over a career without ever being pitchers, e.g. Jeff Mathis, Miguel
    # Rojas, Josh Harrison. Gating on position fixes that; Ohtani (TWP, 114
    # games pitched) still clears it.
    df = pd.read_sql("""
        SELECT pl.player_id, pl.name, pl.team_id, pl.bats, pl.throws,
               'pitcher' AS role, p.latest_season
        FROM players pl
        JOIN (
            SELECT pitcher_id, MAX(season) AS latest_season
            FROM pitches
            GROUP BY pitcher_id
            HAVING COUNT(DISTINCT game_id) >= 3
        ) p ON pl.player_id = p.pitcher_id
        WHERE pl.position IN ('P', 'TWP')
        UNION
        SELECT pl.player_id, pl.name, pl.team_id, pl.bats, pl.throws,
               'batter' AS role, b.latest_season
        FROM players pl
        JOIN (SELECT batter_id, MAX(season) AS latest_season FROM pitches GROUP BY batter_id) b
          ON pl.player_id = b.batter_id
        WHERE pl.player_id NOT IN (
            SELECT pl2.player_id
            FROM players pl2
            JOIN (
                SELECT pitcher_id FROM pitches
                GROUP BY pitcher_id
                HAVING COUNT(DISTINCT game_id) >= 3
            ) p2 ON pl2.player_id = p2.pitcher_id
            WHERE pl2.position IN ('P', 'TWP')
        )
        ORDER BY name
    """, conn)
    pitcher_roles = _pitcher_roles(conn, season)
    df = df.merge(
        pitcher_roles[["role"]].rename(columns={"role": "pitcher_role"}),
        left_on="player_id", right_index=True, how="left",
    )
    df.loc[df["role"] != "pitcher", "pitcher_role"] = None
    return df


@router.get("/players/all")
def list_all_players(
    season: int = Query(2026),
    q: str = Query(default=""),
    conn=Depends(get_conn),
):
    records = _all_players(conn, season).to_dict(orient="records")
    for r in records:
        if "pitcher_role" in r:
            r["pitcher_role"] = _clean(r["pitcher_role"])
    if q:
        q_lower = q.lower()
        records = [r for r in records if q_lower in str(r.get("name", "")).lower()]
    return records


# Player-page checkpoint labels, in the exact order _build_player_steps
# below yields them - the frontend's loading bar reads step INDEX against
# this same count (both roles yield exactly PLAYER_LOAD_STEPS steps, see
# the "step" yields below), not the label text, so this list only needs to
# stay in sync for humans reading the two side by side.
PLAYER_LOAD_STEPS = 6


def _build_player_steps(conn, player_id, role, split, season):
    """Same computation get_player() used to do inline, restructured as a
    generator that yields ("step", label) after each major phase completes,
    then a final ("done", payload) - lets the plain JSON endpoint and the
    checkpoint-streaming endpoint below share one code path (so they can't
    drift out of sync) while giving the streaming one real progress points
    tied to actual work finishing, not a fake timer. Every phase boundary
    here is a genuine one already implicit in the original function's own
    blank-line groupings - no new work was reordered or split up to make
    this happen."""
    player_row = conn.execute(
        "SELECT name, team_id, bats, throws, position FROM players WHERE player_id = ?", (player_id,)
    ).fetchone()
    if player_row is None:
        raise HTTPException(status_code=404, detail="Player not found")
    name, team_id, bats, throws, position = player_row
    split_filter = _norm_split(split)

    base = {"player_id": player_id, "name": name, "team_id": team_id, "position": position, "role": role, "split": split, "season": season}
    base["last_5_games"] = _last_5_games(conn, season, player_id, role)
    base["il_status"] = _il_status(conn, player_id)
    if role == "pitcher":
        # SP/RP, not L/RHP - handedness is dropped here per direction (it's
        # already unambiguous from the movement plot's own throws-based HB
        # sign convention), reusing the same classify_pitcher_roles() pool
        # the games-started lookup below already needs.
        pitcher_roles = _pitcher_roles(conn, season)
        base["handedness"] = pitcher_roles.loc[player_id, "role"] if player_id in pitcher_roles.index else None
    else:
        base["handedness"] = f"Bats {bats}" if bats else None
        base["bats"], base["throws"] = bats, throws
    yield ("step", "Loading player info")

    if role == "pitcher":
        pool = _pitcher_pool(conn, season, split_filter)
        stuff = _stuff_plus_table(conn, season)
        roles = pitcher_roles
        gs = int(roles.loc[player_id, "games_started"]) if player_id in roles.index else None
        lg_hrfb = _league_hr_fb(conn, season)
        yield ("step", "Loading league pools")

        stats = pitcher_counting_stats(conn, season, player_id, split_filter, games_started=gs, league_hr_fb_rate=lg_hrfb)
        overall_stuff = overall_stuff_plus(stuff, player_id)
        era = _season_era(player_id, season)

        # Real xFIP (MLB Stats API sabermetrics, see pulls/pull_sabermetrics.py)
        # has no opposing-batter-hand split, same reasoning as ERA above - use
        # it on the unsplit ALL view, fall back to this project's own
        # pitch-level xFIP approximation (and label it as such) on L/R.
        is_all_split = not split_filter or split_filter == "ALL"
        real_xfip = _real_sabermetric(conn, season, player_id, "xfip") if is_all_split else None
        xfip_label = "xFIP" if real_xfip is not None else "xFIP (approx)"
        xfip_value = real_xfip if real_xfip is not None else stats["xfip"]
        real_war = _real_sabermetric(conn, season, player_id, "war_pitching")

        base["card"] = [
            {"label": "IP", "value": stats["ip"]},
            {"label": "K", "value": stats["k"]},
            {"label": "GS", "value": stats["gs"] if stats["gs"] is not None else "-"},
            {"label": "OPS against", "value": format_value(stats["OPS"], "{:.3f}")},
            {"label": "ERA (season)", "value": era if era is not None else "-"},
            {"label": xfip_label, "value": format_value(xfip_value, "{:.2f}")},
            {"label": "bWAR", "value": format_value(real_war, "{:.1f}")},
        ]
        base["badge"] = {
            "kind": "stuff_plus",
            "value": _clean(overall_stuff),
            "value_str": format_value(overall_stuff, "{:.0f}") if _clean(overall_stuff) is not None else "N/A",
        }
        yield ("step", "Computing stats")

        deception_plus = _deception_plus_badge(conn, player_id)
        tunneling_plus = _tunneling_plus_badge(conn, season, player_id)
        location_plus = _location_plus_badge(conn, season, player_id)
        base["deception_badge"] = {
            "value": _clean(deception_plus),
            "value_str": format_value(deception_plus, "{:.0f}") if deception_plus is not None else "N/A",
            "season": 2025,
        }
        base["tunneling_badge"] = {
            "value": _clean(tunneling_plus),
            "value_str": format_value(tunneling_plus, "{:.0f}") if tunneling_plus is not None else "N/A",
            "season": season,
        }
        base["location_badge"] = {
            "value": _clean(location_plus),
            "value_str": format_value(location_plus, "{:.0f}") if location_plus is not None else "N/A",
            "season": season,
        }

        # Unqualified players still get real percentile bars (Baseball
        # Savant's own convention), ranked against the QUALIFIED population
        # only, shaded on the frontend via the qualified flag - see
        # player_percentiles_allow_unqualified's docstring for exactly how
        # the ranking population is kept clean of the unqualified player's
        # own row. The threshold-free pool is only ever built (a real,
        # second league-wide query) when actually needed.
        unqualified_pool = None if player_id in pool.index else _pitcher_pool_unqualified(conn, season, split_filter)
        pct, pct_qualified = player_percentiles_allow_unqualified(pool, player_id, PITCHER_POLARITY, unqualified_pool)
        base["percentiles"] = _percentile_groups(pct, PITCHER_LABELS, PITCHER_FMT, PITCHER_GROUPS, qualified=pct_qualified)
        # FB VAA's percentile itself is never flipped for a sinkerballer -
        # it stays a plain, consistent "how flat is this fastball vs. the
        # league" rank for every pitcher - but a low percentile means
        # something different (good, not bad) for a pitcher whose PRIMARY
        # fastball is a sinker, so attach an explanatory note rather than
        # silently inverting the number.
        if player_id in pool.index and pool.loc[player_id].get("fb_vaa_primary_type") == "SI":
            for group in base["percentiles"]:
                for m in group["metrics"]:
                    if m["key"] == "fb_vaa":
                        m["note"] = "Steeper is better for sinkerballers - this pitcher's primary fastball is a sinker, so a LOW percentile here (steep) is a good sign, not a bad one."
        yield ("step", "Computing percentiles")

        # Movement is a property of the pitch itself, not who's in the box -
        # deliberately NOT filtered by split_filter (unlike everything else
        # on this page) so the plot stays static across the L/R toggle;
        # season is still a real axis (a real arsenal can differ year to
        # year) and is not held fixed here.
        mv = _movement_plot_data(conn, season, player_id)
        arm_angle = _pitcher_arm_angle(conn, season, player_id)
        base["pitch_usage_by_stand"] = _pitch_usage_by_stand(conn, season, player_id)
        aa_pct = _pitcher_approach_angle_percentile(conn, season, player_id, throws)
        base["rubber_position"] = {
            "release_pos_x": _clean(_pitcher_rubber_position(conn, season, player_id)),
            "throws": throws,
            "angle_deg": _clean(aa_pct["angle_deg"]) if aa_pct else None,
            "percentile": _clean(aa_pct["percentile"]) if aa_pct else None,
        }
        base["babip_against"] = {
            "value": _clean(stats["babip"]),
            "value_str": format_value(stats["babip"], "{:.3f}"),
        }


        summary_rows = []
        filtered = mv
        if not mv.empty:
            closest_frames = []
            for pt, group in mv.groupby("pitch_type"):
                avg_hb, avg_ivb = group["HB"].mean(), group["IVB"].mean()
                summary_rows.append({
                    "pitch_type": pt, "hb": _clean(avg_hb), "ivb": _clean(avg_ivb), "n_pitches": len(group),
                })
                # Only plot the half of points closest to this pitch type's
                # own average shape - decluttering, the summary stats above
                # are still computed from every pitch, not just the kept half.
                dist = ((group["HB"] - avg_hb) ** 2 + (group["IVB"] - avg_ivb) ** 2) ** 0.5
                keep_n = max(1, len(group) // 2)
                closest_frames.append(group.loc[dist.sort_values().index[:keep_n]])
            filtered = pd.concat(closest_frames)
            summary_rows.sort(key=lambda r: -r["n_pitches"])

        peer_movement = _similar_arm_angle_movement(conn, season, player_id)
        base["movement_plot"] = {
            "arm_angle": _clean(arm_angle),
            "points": [
                {"pitch_type": r["pitch_type"], "hb": _clean(r["HB"]), "ivb": _clean(r["IVB"])}
                for r in filtered.to_dict("records")
            ] if not mv.empty else [],
            "summary": summary_rows,
            # Real league-wide average movement per pitch type, among OTHER
            # pitchers whose own season-average arm angle is within +/-3deg
            # of this pitcher's - powers the movement plot's "average
            # movement" reference circles (a delivery-comparable peer
            # group's average, not this pitcher's own self-referential mean).
            "peer_avg_movement": {
                pt: {"hb": _clean(v["hb"]), "ivb": _clean(v["ivb"]),
                     "n_pitchers": int(v["n_pitchers"]), "n_pitches": int(v["n_pitches"])}
                for pt, v in peer_movement.items()
            },
        }
        yield ("step", "Computing movement profile")

        table = pitcher_pitch_table(conn, season, player_id, split_filter, stuff)

        stuff_by_stand = _stuff_plus_by_stand_table(conn, season)
        pitch_types_ordered = list(table.index) if not table.empty else []
        base["stuff_breakdown"] = _stuff_breakdown(stuff, stuff_by_stand, overall_stuff, player_id, pitch_types_ordered)

        if not table.empty:
            table = table.sort_values("usage_pct", ascending=False)
            total_row = _pitcher_league_avg_row(conn, season, split_filter)
            rows, total_row_cells = _table_to_rows(table, PITCHER_TABLE_COLUMNS, PITCHER_TABLE_FMT, total_row)
        else:
            rows, total_row_cells = [], None
        base["pitch_table"] = {"columns": [{"key": k, "label": v} for k, v in PITCHER_TABLE_COLUMNS],
                                "rows": rows, "summary_row": total_row_cells, "summary_label": "League Avg"}
        yield ("step", "Building pitch table")

    else:
        pool = _batter_pool(conn, season, split_filter)
        yield ("step", "Loading league pools")

        stats = batter_counting_stats(conn, season, player_id, split_filter)

        # Real wRC+ (MLB Stats API sabermetrics, see pulls/pull_sabermetrics.py)
        # has no opposing-pitcher-hand split - use it on the unsplit ALL view,
        # fall back to this project's own wOBA-based approximation on L/R.
        is_all_split = not split_filter or split_filter == "ALL"
        real_wrc = _real_sabermetric(conn, season, player_id, "wrc_plus") if is_all_split else None
        wrc_label = "wRC+" if real_wrc is not None else "wRC+ (approx)"
        wrc_value = real_wrc if real_wrc is not None else batter_wrc_plus(conn, season, player_id, split_filter)
        real_war = _real_sabermetric(conn, season, player_id, "war_batting")

        base["card"] = [
            {"label": "PA", "value": stats["n_pa"]},
            {"label": "HR", "value": stats["hr"]},
            {"label": "OPS", "value": format_value(stats["OPS"], "{:.3f}")},
            {"label": wrc_label, "value": format_value(wrc_value, "{:.0f}")},
            {"label": "bWAR", "value": format_value(real_war, "{:.1f}")},
        ]

        all_pool = _batter_pool(conn, season, None)
        archetype_pct = player_percentiles(all_pool, player_id, BATTER_POLARITY)
        archetype = classify_batter_archetype(archetype_pct) if archetype_pct else None
        base["badge"] = {
            "kind": "archetype",
            "value": archetype,
            "description": ARCHETYPE_DESCRIPTIONS.get(archetype, "") if archetype else "",
        } if archetype else None
        yield ("step", "Computing stats")

        # Same unqualified-but-shaded treatment as the pitcher side above.
        unqualified_pool = None if player_id in pool.index else _batter_pool_unqualified(conn, season, split_filter)
        pct, pct_qualified = player_percentiles_allow_unqualified(pool, player_id, BATTER_POLARITY, unqualified_pool)
        base["percentiles"] = _percentile_groups(pct, BATTER_LABELS, BATTER_FMT, BATTER_GROUPS, qualified=pct_qualified)
        yield ("step", "Computing percentiles")

        zones = _field_zone_pcts(conn, season, player_id, split_filter)
        base["swing_field"] = {
            "lf": _clean(zones["lf"]), "cf": _clean(zones["cf"]), "rf": _clean(zones["rf"]),
            "locations": zones["locations"],
            "highlight_side": zones["highlight_side"],
            "highlight_opacity": _clean(zones["highlight_opacity"]),
        }

        # Swing description cards (Attack Angle/Direction/Length/Tilt) are
        # deliberately NOT split-aware, per direction - a swing doesn't
        # change based on who's selected as the opposing hand toggle, same
        # reasoning the pitcher side's movement plot already uses. profile
        # is loaded unsplit (None), and percentiles/archetypes come from
        # `archetype_pct` (the ALL-split pool computed above for the
        # archetype badge) instead of the split-aware `pct`, so nothing
        # here moves when the page's L/R toggle changes.
        profile = _swing_profile_unsplit(conn, season, player_id)
        league_swing = _league_swing_profile(conn, season)
        bat_speed_pct = archetype_pct.get("bat_speed", (None, None))[1]
        swing_length_pct = archetype_pct.get("swing_length", (None, None))[1]
        attack_angle_pct = archetype_pct.get("attack_angle", (None, None))[1]
        # Attack Direction's raw value+percentile both come from the same
        # pool now (real hit-outcome landing angle, hits-only, hand-adjusted
        # - identical formula to field_zone_pcts' own pull_angle, just
        # computed unsplit here) instead of a separate zones-based lookup.
        attack_direction_value, attack_direction_pct = archetype_pct.get("attack_direction", (None, None))
        swing_tilt_value, swing_tilt_pct = archetype_pct.get("swing_path_tilt", (None, None))
        # League averages for these two also come from the same unsplit
        # all_pool (a plain column mean across every qualifying batter),
        # not a separate helper - all_pool already has both columns now.
        league_attack_direction = all_pool["attack_direction"].mean() if "attack_direction" in all_pool else None
        league_swing_tilt = all_pool["swing_path_tilt"].mean() if "swing_path_tilt" in all_pool else None
        base["swing_3d"] = {
            "bats": bats,
            "bat_speed": _clean(profile.get("bat_speed")),
            "bat_speed_pct": _clean(bat_speed_pct),
            "attack_angle": _clean(profile.get("attack_angle")),
            "attack_angle_pct": _clean(attack_angle_pct),
            "attack_direction": _clean(attack_direction_value),
            "attack_direction_pct": _clean(attack_direction_pct),
            "swing_length": _clean(profile.get("swing_length")),
            "swing_length_pct": _clean(swing_length_pct),
            "swing_path_tilt": _clean(swing_tilt_value),
            "swing_path_tilt_pct": _clean(swing_tilt_pct),
            "league_avg": {
                "bat_speed": _clean(league_swing.get("bat_speed")),
                "attack_angle": _clean(league_swing.get("attack_angle")),
                "attack_direction": _clean(league_attack_direction),
                "swing_length": _clean(league_swing.get("swing_length")),
                "swing_path_tilt": _clean(league_swing_tilt),
            },
        }
        yield ("step", "Computing swing profile")

        table = batter_pitch_type_table(conn, season, player_id, split_filter)
        if not table.empty:
            lg_row = _league_avg_batter_row(conn, season, split_filter)
            rows, league_row = _table_to_rows(table, BATTER_TABLE_COLUMNS, PITCHER_TABLE_FMT | {
                "OPS": "{:.3f}", "wOBA": "{:.3f}", "xwOBA": "{:.3f}", "K_pct": "{:.1%}",
                "whiff_pct": "{:.1%}", "barrel_pct": "{:.1%}", "pct_faced": "{:.1%}",
            }, lg_row)
        else:
            rows, league_row = [], None
        base["pitch_type_table"] = {"columns": [{"key": k, "label": v} for k, v in BATTER_TABLE_COLUMNS],
                                     "rows": rows, "summary_row": league_row, "summary_label": "League Avg"}
        yield ("step", "Building pitch table")

    yield ("done", base)


@router.get("/player/{player_id}")
def get_player(
    player_id: int,
    role: str = Query("pitcher", pattern="^(pitcher|batter)$"),
    split: str = Query("ALL", pattern="^(ALL|L|R)$"),
    season: int = Query(2026),
    conn=Depends(get_conn),
):
    for kind, payload in _build_player_steps(conn, player_id, role, split, season):
        if kind == "done":
            return payload
    raise HTTPException(status_code=500, detail="Player payload generator produced no result")


@router.get("/player/{player_id}/progress")
def get_player_progress(
    player_id: int,
    role: str = Query("pitcher", pattern="^(pitcher|batter)$"),
    split: str = Query("ALL", pattern="^(ALL|L|R)$"),
    season: int = Query(2026),
):
    """Newline-delimited-JSON version of GET /player/{id} for the Player
    page's own loading bar: one line per real checkpoint as
    _build_player_steps finishes it (`{"type":"progress","step":...,
    "index":N,"total":PLAYER_LOAD_STEPS}`), then one final line with the
    complete payload (`{"type":"done","data":{...}}`) - same computation,
    same result, just observable mid-flight instead of arriving as a single
    opaque wait. Opens its OWN sqlite connection (rather than the usual
    Depends(get_conn)) since that dependency's cleanup timing relative to a
    StreamingResponse's body being fully sent isn't guaranteed the way a
    plain return value's is - this sidesteps the question entirely."""
    import sqlite3
    from api.db import DB_PATH

    # Checked BEFORE the StreamingResponse is constructed - a StreamingResponse
    # commits to its 200 status as soon as it starts sending, before it pulls
    # even the first item from the body generator, so a 404 raised from
    # inside gen() below would arrive as a broken 200 response instead of a
    # clean 404. This mirrors _build_player_steps' own not-found check so a
    # bad player_id fails the same way on both endpoints.
    precheck_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    try:
        exists = precheck_conn.execute(
            "SELECT 1 FROM players WHERE player_id = ?", (player_id,)
        ).fetchone()
    finally:
        precheck_conn.close()
    if exists is None:
        raise HTTPException(status_code=404, detail="Player not found")

    def gen():
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA cache_size = -32000")
        conn.execute("PRAGMA temp_store = MEMORY")
        conn.execute("PRAGMA mmap_size = 268435456")
        try:
            index = 0
            for kind, payload in _build_player_steps(conn, player_id, role, split, season):
                if kind == "step":
                    index += 1
                    yield json.dumps({"type": "progress", "step": payload, "index": index, "total": PLAYER_LOAD_STEPS}) + "\n"
                else:
                    yield json.dumps({"type": "done", "data": payload}) + "\n"
        finally:
            conn.close()

    return StreamingResponse(gen(), media_type="application/x-ndjson")


# ---------------------------------------------------------------------------
# Part C2: Tunneling pair composite (separate endpoint — slow first call)
# ---------------------------------------------------------------------------

@router.get("/player/{player_id}/tunneling-pairs")
def get_tunneling_pairs(
    player_id: int,
    season: int = Query(2026),
    conn=Depends(get_conn),
):
    """Per-pitch-type-pair tunneling composite scores for one pitcher-season.
    Returned as a separate endpoint (not baked into /player/{id}) because
    the first season call builds all ~2700 pairs (~minutes); subsequent calls
    are cached. The frontend fetches this after the main player payload lands."""
    all_season_pairs = _tunneling_pair_season(conn, season)
    if all_season_pairs is None or all_season_pairs.empty:
        return []
    total_row = conn.execute(
        "SELECT COUNT(*) FROM pitches WHERE pitcher_id = ? AND season = ?",
        (player_id, season),
    ).fetchone()
    total_pitches = total_row[0] if total_row else None
    return pitcher_pairs_from_season(all_season_pairs, player_id, total_pitches)


# ---------------------------------------------------------------------------
# Part C2: Player season list (seasons with pitch data + team per season)
# ---------------------------------------------------------------------------

@router.get("/players/{player_id}/seasons")
def get_player_seasons(player_id: int, conn=Depends(get_conn)):
    """Seasons this player has pitch data in, most recent first.
    Each entry includes the team_id they were on that season (most-used team).
    Team derivation differs by role:
      - pitcher in Top inning  → pitching for home team
      - pitcher in Bot inning  → pitching for away team
      - batter  in Top inning  → batting for away team
      - batter  in Bot inning  → batting for home team
    We run separate subqueries for each role and aggregate before picking the
    plurality team per season so two-way players also work correctly."""
    rows = conn.execute(
        """
        SELECT season, team_id, SUM(n) AS total FROM (
            SELECT p.season,
                   CASE p.inning_topbot WHEN 'Top' THEN g.home_team ELSE g.away_team END AS team_id,
                   COUNT(*) AS n
            FROM pitches p JOIN games g ON p.game_id = g.game_id
            WHERE p.pitcher_id = ?
            GROUP BY p.season, team_id
            UNION ALL
            SELECT p.season,
                   CASE p.inning_topbot WHEN 'Top' THEN g.away_team ELSE g.home_team END AS team_id,
                   COUNT(*) AS n
            FROM pitches p JOIN games g ON p.game_id = g.game_id
            WHERE p.batter_id = ?
            GROUP BY p.season, team_id
        )
        GROUP BY season, team_id
        ORDER BY season DESC, total DESC
        """,
        (player_id, player_id),
    ).fetchall()

    seen = {}
    for season, team_id, _ in rows:
        if season not in seen:
            seen[season] = int(team_id) if team_id else None

    return [{"season": s, "team_id": t} for s, t in seen.items()]


# ---------------------------------------------------------------------------
# Part D: Outing breakdown endpoints (pitchers only)
# ---------------------------------------------------------------------------

@router.get("/players/{player_id}/outings")
def get_pitcher_outings(
    player_id: int,
    season: int = Query(2026),
    conn=Depends(get_conn),
):
    """
    List of games (outings) a pitcher appeared in during the given season,
    ordered most recent first. Pitchers only — batters should use the
    per-game stats on their own profile.
    """
    rows = conn.execute(
        """
        SELECT g.game_id, g.game_date,
               COALESCE(ht.abbreviation, CAST(g.home_team AS TEXT)) AS home_abbr,
               COALESCE(at.abbreviation, CAST(g.away_team AS TEXT)) AS away_abbr,
               COUNT(*) AS n_pitches
        FROM pitches p
        JOIN games g ON p.game_id = g.game_id
        LEFT JOIN teams ht ON CAST(g.home_team AS INTEGER) = ht.team_id
        LEFT JOIN teams at ON CAST(g.away_team AS INTEGER) = at.team_id
        WHERE p.pitcher_id = ? AND p.season = ?
        GROUP BY g.game_id
        ORDER BY g.game_date DESC
        """,
        (player_id, season),
    ).fetchall()

    if not rows:
        return []

    return [
        {
            "game_id": r[0],
            "game_date": r[1],
            "home_team": r[2],
            "away_team": r[3],
            "n_pitches": r[4],
            "label": f"{r[1]} — {r[3]} @ {r[2]} ({r[4]}p)",
        }
        for r in rows
    ]


@router.get("/players/{player_id}/outing")
def get_pitcher_outing(
    player_id: int,
    game_id: int = Query(...),
    season: int = Query(2026),
    conn=Depends(get_conn),
):
    """
    Full pitch-by-pitch breakdown for one outing plus per-pitch-type
    aggregate stats. Includes Stuff+/Deception+ mean per pitch type where
    available from the live model scoring.
    """
    # ---- raw pitches for this outing ----
    pitches_df = pd.read_sql(
        """
        SELECT pitch_type, release_speed, release_spin_rate,
               pfx_x, pfx_z, release_extension,
               description, events, plate_x, plate_z,
               balls, strikes, stand, launch_speed,
               COALESCE(sz_top, 3.5) as sz_top,
               COALESCE(sz_bot, 1.5) as sz_bot,
               at_bat_number, pitch_number,
               COALESCE(inning, 1) as inning
        FROM pitches
        WHERE pitcher_id = ? AND game_id = ? AND season = ?
        ORDER BY at_bat_number, pitch_number
        """,
        conn,
        params=(player_id, game_id, season),
    )

    if pitches_df.empty:
        return {"game_id": game_id, "pitch_types": [], "pitches": []}

    # ---- per-pitch-type stats ----
    _OUTING_SWINGS = {"swinging_strike", "swinging_strike_blocked", "foul", "foul_tip",
                      "hit_into_play", "hit_into_play_no_out", "hit_into_play_score"}
    _OUTING_WHIFFS = {"swinging_strike", "swinging_strike_blocked"}
    _OUTING_STRIKE_CALLS = {"called_strike", "swinging_strike", "swinging_strike_blocked",
                            "foul_tip", "foul", "foul_bunt"}

    d = pitches_df["description"]
    pitches_df["_swing"] = d.isin(_OUTING_SWINGS)
    pitches_df["_whiff"] = d.isin(_OUTING_WHIFFS)
    pitches_df["_called_strike"] = d == "called_strike"
    pitches_df["_in_zone"] = (
        pitches_df["plate_x"].between(-0.8333, 0.8333) &
        (pitches_df["plate_z"] >= pitches_df["sz_bot"]) &
        (pitches_df["plate_z"] <= pitches_df["sz_top"])
    )
    pitches_df["_first_pitch"] = (pitches_df["balls"] == 0) & (pitches_df["strikes"] == 0)
    pitches_df["_fp_strike"] = pitches_df["_first_pitch"] & d.isin(_OUTING_STRIKE_CALLS)

    # Season averages + counts in one query (for delta arrows and yr_usage_pct)
    season_agg_df = pd.read_sql(
        """SELECT pitch_type, COUNT(*) as n_pitches,
                  AVG(release_speed) as avg_velo, AVG(pfx_x) as avg_hb,
                  AVG(pfx_z) as avg_ivb, AVG(release_spin_rate) as avg_spin
           FROM pitches WHERE pitcher_id = ? AND season = ? GROUP BY pitch_type""",
        conn, params=(player_id, season),
    )
    season_total_all = int(season_agg_df["n_pitches"].sum())
    season_type_counts = dict(zip(season_agg_df["pitch_type"], season_agg_df["n_pitches"].astype(int)))
    season_avg = {row.pitch_type: row for _, row in season_agg_df.iterrows()}

    # Outing-level aggregates
    _ev_col = pitches_df["launch_speed"].dropna()
    max_ev = round(float(_ev_col.max()), 1) if not _ev_col.empty else None
    avg_ev = round(float(_ev_col.mean()), 1) if not _ev_col.empty else None
    _velo_col = pitches_df["release_speed"].dropna()
    max_velo = round(float(_velo_col.max()), 1) if not _velo_col.empty else None
    avg_velo = round(float(_velo_col.mean()), 1) if not _velo_col.empty else None

    _p_row = conn.execute(
        "SELECT team_id, name FROM players WHERE player_id = ?", (player_id,)
    ).fetchone()
    player_team_id = _p_row[0] if _p_row else None
    _player_name_cached = _p_row[1] if _p_row else None

    # Stat line for the outing
    outs = int(pitches_df["events"].isin([
        "field_out","force_out","grounded_into_double_play","double_play",
        "strikeout","strikeout_double_play","sac_fly","sac_bunt",
        "fielders_choice_out","other_out","triple_play",
    ]).sum())
    # Try to get official R/ER/ERA from the MLB Stats API boxscore
    _bx_r, _bx_er, _bx_era = None, None, None
    try:
        _bx = fetch_boxscore(game_id)
        _bx_stats = pitcher_daily_stats(_bx, player_id)
        if _bx_stats:
            _bx_r = _bx_stats.get("r")
            _bx_er = _bx_stats.get("er")
            _raw_era = _bx_stats.get("season_era")
            if _raw_era not in (None, "-", "-.--"):
                try:
                    _bx_era = round(float(_raw_era), 2)
                except (ValueError, TypeError):
                    pass
    except Exception:
        pass

    stat_line = {
        "ip": round(outs / 3, 1),
        "k": int(pitches_df["events"].isin(["strikeout","strikeout_double_play"]).sum()),
        "bb": int(pitches_df["events"].isin(["walk","intent_walk"]).sum()),
        "h": int(pitches_df["events"].isin(["single","double","triple","home_run"]).sum()),
        "hr": int(pitches_df["events"].isin(["home_run"]).sum()),
        "r": _bx_r,
        "er": _bx_er,
        "era": _bx_era,
        "total_pitches": len(pitches_df),
        "total_whiffs": int(pitches_df["_whiff"].sum()),
        "zone_pct": round(pitches_df["_in_zone"].mean() * 100, 1),
        "first_pitch_strike_pct": round(
            pitches_df.loc[pitches_df["_first_pitch"], "_fp_strike"].mean() * 100, 1
        ) if pitches_df["_first_pitch"].any() else None,
    }

    stuff_season = None
    try:
        stuff_season = _stuff_plus_table(conn, season)
    except Exception:
        pass

    summaries = []
    total_pitches = len(pitches_df)
    for pt, grp in pitches_df.groupby("pitch_type"):
        n = len(grp)
        swings = int(grp["_swing"].sum())
        whiffs = int(grp["_whiff"].sum())
        whiff_rate = whiffs / swings if swings > 0 else None

        def avg_col(col):
            s = grp[col].dropna()
            return round(float(s.mean()), 2) if not s.empty else None

        velo = avg_col("release_speed")
        hb = avg_col("pfx_x")
        ivb = avg_col("pfx_z")
        spin = round(float(grp["release_spin_rate"].dropna().mean()), 0) if not grp["release_spin_rate"].dropna().empty else None

        sa = season_avg.get(pt)
        def delta(val, avg_val):
            if val is None or avg_val is None or (hasattr(avg_val, 'item') and pd.isna(avg_val)):
                return None
            return round(float(val) - float(avg_val), 1)

        fp_pitches = grp[grp["_first_pitch"]]
        fp_strike_pct = round(fp_pitches["_fp_strike"].mean() * 100, 1) if not fp_pitches.empty else None
        zone_pct = round(grp["_in_zone"].mean() * 100, 1)

        pt_stuff = None
        if stuff_season is not None:
            try:
                idx = (player_id, pt)
                if idx in stuff_season.index:
                    pt_stuff = round(float(stuff_season.loc[idx, "stuff_plus"]), 0)
            except Exception:
                pass

        # New fields for the detailed outing breakdown table
        called_strikes_n = int(grp["_called_strike"].sum())
        k_count = int(grp["events"].isin(["strikeout", "strikeout_double_play"]).sum())
        rhh_n = int((grp["stand"] == "R").sum())
        lhh_n = int((grp["stand"] == "L").sum())
        yr_count = season_type_counts.get(pt, 0)
        yr_usage_pct = round(yr_count / season_total_all * 100, 1) if season_total_all > 0 else None
        _sa_spin = float(sa.avg_spin) if sa is not None and not pd.isna(sa.avg_spin) else None
        spin_delta_val = delta(spin, _sa_spin)

        summaries.append({
            "pitch_type": pt,
            "count": n,
            "usage_pct": round(n / total_pitches * 100, 1),
            "yr_usage_pct": yr_usage_pct,
            "rhh_count": rhh_n,
            "rhh_pct": round(rhh_n / total_pitches * 100, 1),
            "lhh_count": lhh_n,
            "lhh_pct": round(lhh_n / total_pitches * 100, 1),
            "avg_velo": velo,
            "velo_delta": delta(velo, sa.avg_velo if sa is not None else None),
            "avg_spin": spin,
            "spin_delta": spin_delta_val,
            "avg_hb": hb,
            "hb_delta": delta(hb, sa.avg_hb if sa is not None else None),
            "avg_ivb": ivb,
            "ivb_delta": delta(ivb, sa.avg_ivb if sa is not None else None),
            "swings": swings,
            "called_strikes": called_strikes_n,
            "k_count": k_count,
            "whiffs": whiffs,
            "whiff_rate": round(whiff_rate * 100, 1) if whiff_rate is not None else None,
            "zone_pct": zone_pct,
            "first_pitch_strike_pct": fp_strike_pct,
            "stuff_plus": pt_stuff,
        })

    summaries.sort(key=lambda x: -x["count"])

    game_row = conn.execute(
        """SELECT g.game_date,
                  COALESCE(ht.abbreviation, CAST(g.home_team AS TEXT)) AS home_abbr,
                  COALESCE(at.abbreviation, CAST(g.away_team AS TEXT)) AS away_abbr
           FROM games g
           LEFT JOIN teams ht ON CAST(g.home_team AS INTEGER) = ht.team_id
           LEFT JOIN teams at ON CAST(g.away_team AS INTEGER) = at.team_id
           WHERE g.game_id = ?""",
        (game_id,),
    ).fetchone()

    home_abbr = game_row[1] if game_row else None
    away_abbr = game_row[2] if game_row else None

    # Find the opposing pitcher (the other pitcher who started that game)
    opp_row = conn.execute(
        """SELECT p.pitcher_id, pl.name
           FROM pitches p
           LEFT JOIN players pl ON pl.player_id = p.pitcher_id
           WHERE p.game_id = ? AND p.season = ? AND p.pitcher_id != ?
             AND p.at_bat_number = 1
           GROUP BY p.pitcher_id
           LIMIT 1""",
        (game_id, season, player_id),
    ).fetchone()
    opp_pitcher = {"player_id": int(opp_row[0]), "name": opp_row[1] or "Opponent"} if opp_row else None

    player_name_row = (_player_name_cached,)

    # ---- at-bat sequences for pitch sequence chart ----
    RESULT_LABELS = {
        "strikeout": "K", "strikeout_double_play": "KDP",
        "walk": "BB", "intent_walk": "IBB",
        "single": "1B", "double": "2B", "triple": "3B", "home_run": "HR",
        "field_out": "OUT", "force_out": "FC", "grounded_into_double_play": "DP",
        "double_play": "DP", "sac_fly": "SF", "sac_bunt": "SH",
        "fielders_choice_out": "FC", "fielders_choice": "FC",
        "hit_by_pitch": "HBP", "catcher_interf": "CI",
        "other_out": "OUT", "triple_play": "TP",
    }
    sequences = []
    if "at_bat_number" in pitches_df.columns:
        for ab_num, ab_grp in pitches_df.sort_values("pitch_number").groupby("at_bat_number"):
            ab_pitches = [
                {
                    "pt": row["pitch_type"] if pd.notna(row["pitch_type"]) else "??",
                    "v": round(float(row["release_speed"]), 1) if pd.notna(row["release_speed"]) else None,
                    "d": row["description"] if pd.notna(row["description"]) else "",
                    "b": int(row["balls"]) if pd.notna(row["balls"]) else 0,
                    "s": int(row["strikes"]) if pd.notna(row["strikes"]) else 0,
                }
                for row in ab_grp.to_dict("records")
            ]
            ev_series = ab_grp["events"].dropna()
            raw_result = ev_series.iloc[-1] if not ev_series.empty else None
            result_label = RESULT_LABELS.get(raw_result, raw_result.upper()[:3] if raw_result else "?")
            sequences.append({
                "ab": int(ab_num),
                "inn": int(ab_grp["inning"].iloc[0]),
                "stand": ab_grp["stand"].iloc[0] if pd.notna(ab_grp["stand"].iloc[0]) else "R",
                "pitches": ab_pitches,
                "result": result_label,
                "raw_result": raw_result,
            })

    return {
        "game_id": game_id,
        "game_date": game_row[0] if game_row else None,
        "home_team": home_abbr,
        "away_team": away_abbr,
        "player_id": player_id,
        "player_name": player_name_row[0] if player_name_row else None,
        "player_team_id": player_team_id,
        "opp_pitcher": opp_pitcher,
        "max_ev": max_ev,
        "avg_ev": avg_ev,
        "max_velo": max_velo,
        "avg_velo": avg_velo,
        "stat_line": stat_line,
        "pitch_types": summaries,
        "sequences": sequences,
    }



@router.get("/players/{player_id}/rolling-xwoba")
def get_rolling_xwoba(
    player_id: int,
    season: int = Query(2026),
    conn=Depends(get_conn),
):
    """Raw per-qualifying-PA xwOBA for a batter; client computes horizons and smoothing."""
    points = batter_rolling_xwoba(conn, season, player_id)
    lg_avg = league_avg_xwoba(conn, season)
    return {"points": points, "league_avg": lg_avg}
