"""
Season-by-season career stats for the Career page.
GET /api/players/{player_id}/career  (role auto-detected from players table)

Changes from v1:
- Per-season team lookup via game joins (inning_topbot logic)
- ERA computed as RA/9 (runs allowed / IP * 9) from score data
- WAR from player_advanced_stats
- Top-10 league flags for key counting stats
- wOBA/xwOBA adjacent in columns
- Column group metadata for vertical dividers
- Awards placeholder column
"""
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, Query

from analytics.player_page import (
    _add_flags, OUTS_PER_EVENT, innings_pitched,
    compute_xfip, league_hr_per_fb,
)
from analytics.splits import STRIKEOUT_EVENTS, WALK_EVENTS, IN_ZONE_CODES
from api.cache import ttl_cache
from api.db import get_conn
from analytics.stuff_model import compute_stuff_plus_if_cached, overall_stuff_plus
from analytics.tunneling_v3_model import raw_predictions_if_cached as _tun_raw_if_cached
from analytics.location_model import raw_predictions_if_cached as _loc_raw_if_cached
from api.routers.deception_report import (
    REPORT_SEASON as DEC_SEASON,
    _pitcher_breakdown_cached as _dec_breakdown_cached,
)

router = APIRouter(prefix="/api")


@ttl_cache()
def _league_hr_fb(conn, season):
    return league_hr_per_fb(conn, season)


def _cf(v, decimals=2):
    if v is None:
        return None
    try:
        f = float(v)
        return None if np.isnan(f) else round(f, decimals)
    except (TypeError, ValueError):
        return None


def _per_season_teams_bulk(conn, player_id, is_pitcher):
    """One query returning {season: team_id} for all of a player's seasons."""
    player_col = "pitcher_id" if is_pitcher else "batter_id"
    if is_pitcher:
        team_expr = "CASE WHEN p.inning_topbot = 'Top' THEN CAST(g.home_team AS INTEGER) ELSE CAST(g.away_team AS INTEGER) END"
    else:
        team_expr = "CASE WHEN p.inning_topbot = 'Top' THEN CAST(g.away_team AS INTEGER) ELSE CAST(g.home_team AS INTEGER) END"

    rows = conn.execute(f"""
        SELECT p.season, {team_expr} AS team_id, COUNT(*) AS cnt
        FROM pitches p JOIN games g ON p.game_id = g.game_id
        WHERE p.{player_col} = ?
        GROUP BY p.season, team_id
    """, (player_id,)).fetchall()

    by_season = {}
    for season, team_id, cnt in rows:
        s = int(season)
        if s not in by_season or cnt > by_season[s][1]:
            by_season[s] = (int(team_id) if team_id else None, cnt)
    return {s: v[0] for s, v in by_season.items()}


def _team_info(conn, team_id):
    if not team_id:
        return {"abbr": "—", "team_id": None}
    r = conn.execute("SELECT abbreviation FROM teams WHERE team_id = ?", (team_id,)).fetchone()
    return {"abbr": r[0] if r else str(team_id), "team_id": team_id}


def _player_awards(conn, player_id):
    """Returns {season: 'AS, CY'} from player_awards table."""
    try:
        rows = conn.execute(
            "SELECT season, award FROM player_awards WHERE player_id = ? ORDER BY season, award",
            (player_id,),
        ).fetchall()
    except Exception:
        return {}
    by_season = {}
    for season, award in rows:
        s = int(season)
        by_season.setdefault(s, []).append(award)
    # Deduplicate awards within a season (e.g. two All-Star entries → one "AS")
    # Prioritize in display order: AS last so it appends nicely after major awards
    ORDER = ["MVP", "CY", "ROY", "SS", "GG", "AS"]
    def _sort_key(a):
        return ORDER.index(a) if a in ORDER else len(ORDER)
    return {s: ", ".join(sorted(set(awds), key=_sort_key)) for s, awds in by_season.items()}


def _pitcher_scores_from_db(conn, pitcher_id):
    """Read all pre-computed model scores for a pitcher from pitcher_model_scores.
    Returns {season: {stuff_plus, tunneling_plus, location_plus}} — instant, no LightGBM."""
    try:
        rows = conn.execute("""
            SELECT season, stuff_plus, tunneling_plus, location_plus
            FROM pitcher_model_scores WHERE pitcher_id = ?
        """, (pitcher_id,)).fetchall()
        return {int(r[0]): {"stuff_plus": r[1], "tunneling_plus": r[2], "location_plus": r[3]}
                for r in rows}
    except Exception:
        return {}


def _pitcher_scores_live(pitcher_id, season):
    """Cache-only live lookup — only used for the current (incomplete) season.
    Returns None for each model if not yet cached (avoids blocking on LightGBM)."""
    stuff_plus = None
    try:
        tbl = compute_stuff_plus_if_cached(season)
        if tbl is not None and pitcher_id in tbl.index.get_level_values("pitcher_id"):
            stuff_plus = _cf(overall_stuff_plus(tbl, pitcher_id), 1)
    except Exception:
        pass

    tunneling_plus = None
    try:
        raw = _tun_raw_if_cached(season)
        if raw is not None and not raw.empty:
            pp = raw[raw["pitcher_id"] == pitcher_id]["pred"]
            if len(pp) > 0:
                m, s = raw["pred"].mean(), raw["pred"].std()
                if s > 0:
                    tunneling_plus = _cf((pp.mean() - m) / s * 15 + 100, 1)
    except Exception:
        pass

    location_plus = None
    try:
        raw = _loc_raw_if_cached(season)
        if raw is not None and not raw.empty:
            pp = raw[raw["pitcher_id"] == pitcher_id]["pred"]
            if len(pp) > 0:
                m, s = raw["pred"].mean(), raw["pred"].std()
                if s > 0:
                    location_plus = _cf((pp.mean() - m) / s * 15 + 100, 1)
    except Exception:
        pass

    return {"stuff_plus": stuff_plus, "tunneling_plus": tunneling_plus, "location_plus": location_plus}


def _top10_rank_pitcher(conn, season, metric_val, metric):
    """Return rank (1-based) of pitcher for a counting stat in the league that season."""
    if metric_val is None:
        return None
    try:
        if metric == "k":
            events_in = "('strikeout','strikeout_double_play')"
            r = conn.execute(f"""
                SELECT COUNT(DISTINCT pitcher_id) + 1 FROM (
                    SELECT pitcher_id, COUNT(*) as v FROM pitches
                    WHERE season = ? AND events IN {events_in}
                    GROUP BY pitcher_id HAVING v > ?
                )
            """, (season, int(metric_val))).fetchone()
        elif metric == "era":
            # Lower is better for ERA — rank ascending
            r = None
        else:
            return None
        return int(r[0]) if r else None
    except Exception:
        return None


def _top10_rank_batter(conn, season, metric_val, metric):
    if metric_val is None:
        return None
    try:
        if metric == "hr":
            r = conn.execute("""
                SELECT COUNT(DISTINCT batter_id) + 1 FROM (
                    SELECT batter_id, COUNT(*) as v FROM pitches
                    WHERE season = ? AND events = 'home_run'
                    GROUP BY batter_id HAVING v > ?
                )
            """, (season, int(metric_val))).fetchone()
            return int(r[0]) if r else None
    except Exception:
        pass
    return None


@router.get("/players/{player_id}/career")
def get_player_career(
    player_id: int,
    conn=Depends(get_conn),
):
    # Auto-detect role from players table
    p_row = conn.execute(
        "SELECT position FROM players WHERE player_id = ?", (player_id,)
    ).fetchone()
    position = p_row[0] if p_row else None
    PITCHER_POSITIONS = {"P", "SP", "RP", "CL"}
    role = "pitcher" if position in PITCHER_POSITIONS else "batter"
    if role == "pitcher":
        return _pitcher_career(conn, player_id)
    else:
        return _batter_career(conn, player_id)


@ttl_cache()
def _pitcher_career(conn, player_id):
    df = pd.read_sql("""
        SELECT season, game_id, inning_topbot,
               p_throws, pitch_type, pfx_z,
               release_speed, release_spin_rate,
               description, zone, events, bb_type,
               bat_score, post_bat_score
        FROM pitches
        WHERE pitcher_id = ?
        ORDER BY season, game_id
    """, conn, params=[player_id])

    if df.empty:
        return {"rows": [], "columns": [], "column_groups": [], "role": "pitcher"}

    df = _add_flags(df)

    season_teams = _per_season_teams_bulk(conn, player_id, True)
    db_scores = _pitcher_scores_from_db(conn, player_id)
    awards_by_season = _player_awards(conn, player_id)

    rows = []
    for season, grp in df.groupby("season", sort=True):
        team_id = season_teams.get(int(season))
        team = _team_info(conn, team_id)

        pa = grp[grp["events"].notna()]
        ip_str, _ = innings_pitched(pa)
        k   = int(pa["events"].isin(STRIKEOUT_EVENTS).sum())
        bb  = int(pa["events"].isin(WALK_EVENTS).sum())
        hr  = int((pa["events"] == "home_run").sum())
        n_pa = len(pa)
        swings = int(grp["is_swing"].sum())
        whiffs = int(grp["is_whiff"].sum())
        in_z  = int(grp["zone"].isin(IN_ZONE_CODES).sum())
        outs  = pa["events"].map(OUTS_PER_EVENT).fillna(0).sum()
        ip_dec = outs / 3

        # RA/9: runs allowed from score changes on events
        runs_allowed = int(
            grp.loc[grp["events"].notna(), "post_bat_score"]
            .sub(grp.loc[grp["events"].notna(), "bat_score"])
            .clip(lower=0).sum()
        )
        era = _cf(runs_allowed / ip_dec * 9, 2) if ip_dec > 0 else None

        xfip = None
        if ip_dec > 0:
            try:
                lg_hr_fb = _league_hr_fb(conn, int(season))
                xfip = _cf(compute_xfip(pa, lg_hr_fb), 2)
            except Exception:
                pass

        # WAR + ERA from DB (prefer DB ERA over RA/9 when available)
        adv = conn.execute(
            "SELECT fwar_pitching, war_pitching, war, era FROM player_advanced_stats WHERE player_id = ? AND season = ?",
            (player_id, int(season)),
        ).fetchone()
        war = None
        if adv:
            war = _cf(adv[0] or adv[1] or adv[2], 1)
            if adv[3] is not None:
                era = _cf(adv[3], 2)

        # Top-10 rank flags
        k_rank  = _top10_rank_pitcher(conn, int(season), k, "k")

        scores = db_scores.get(int(season)) or _pitcher_scores_live(player_id, int(season))

        # Deception+ for current season only
        dec_plus = None
        if int(season) == DEC_SEASON:
            try:
                bd = _dec_breakdown_cached(conn, DEC_SEASON, player_id)
                dec_plus = _cf(bd.get("deception_plus"), 1) if bd else None
            except Exception:
                pass

        # Pitchers are always "P"
        primary_pos = "P"

        row = {
            "season":           int(season),
            "team_abbr":        team["abbr"],
            "team_id":          team["team_id"],
            "gs":               grp["game_id"].nunique(),
            "ip":               ip_str,
            "ip_dec":           round(ip_dec, 1),
            "k":                k,
            "bb":               bb,
            "hr":               hr,
            "era":              era,
            "runs_allowed":     runs_allowed,
            "xfip":             xfip,
            "k_pct":            _cf(k / n_pa * 100, 1) if n_pa else None,
            "bb_pct":           _cf(bb / n_pa * 100, 1) if n_pa else None,
            "whiff_pct":        _cf(whiffs / swings * 100, 1) if swings else None,
            "zone_pct":         _cf(in_z / len(grp) * 100, 1),
            "war":              war,
            "deception_plus":   dec_plus,
            "primary_position": primary_pos,
            "awards":           awards_by_season.get(int(season)),
            "top10":            {"k": k_rank is not None and k_rank <= 10},
        }
        row.update(scores)
        rows.append(row)

    columns = [
        {"key": "season",         "label": "Season",  "group": "id"},
        {"key": "team",           "label": "Team",    "group": "team"},
        {"key": "gs",             "label": "G",       "group": "trad"},
        {"key": "ip",             "label": "IP",      "group": "trad"},
        {"key": "k",              "label": "K",       "group": "trad"},
        {"key": "bb",             "label": "BB",      "group": "trad"},
        {"key": "hr",             "label": "HR",      "group": "trad"},
        {"key": "era",            "label": "ERA*",    "group": "trad"},
        {"key": "k_pct",          "label": "K%",      "group": "rate"},
        {"key": "bb_pct",         "label": "BB%",     "group": "rate"},
        {"key": "whiff_pct",      "label": "Whiff%",  "group": "rate"},
        {"key": "zone_pct",       "label": "Zone%",   "group": "rate"},
        {"key": "xfip",           "label": "xFIP",    "group": "adv"},
        {"key": "stuff_plus",     "label": "Stuff+",  "group": "model"},
        {"key": "tunneling_plus", "label": "Tun+",    "group": "model"},
        {"key": "location_plus",  "label": "Loc+",    "group": "model"},
        {"key": "deception_plus", "label": "Dec+",    "group": "model"},
        {"key": "war",              "label": "WAR",   "group": "war"},
        {"key": "primary_position", "label": "Pos",   "group": "pos"},
        {"key": "awards",           "label": "Awards","group": "awards"},
    ]

    column_groups = [
        {"label": None,          "group": "id"},
        {"label": None,          "group": "team"},
        {"label": "Traditional", "group": "trad"},
        {"label": "Rate",        "group": "rate"},
        {"label": "Advanced",    "group": "adv"},
        {"label": "Models",      "group": "model"},
        {"label": None,          "group": "war"},
        {"label": None,          "group": "pos"},
        {"label": None,          "group": "awards"},
    ]

    return {"rows": rows, "columns": columns, "column_groups": column_groups, "role": "pitcher"}


@ttl_cache()
def _batter_career(conn, player_id):
    from analytics.splits import _pa_outcome_metrics
    df = pd.read_sql("""
        SELECT season, game_id,
               launch_speed, launch_angle,
               estimated_woba_using_speedangle,
               woba_value, woba_denom,
               description, zone, events, inning_topbot
        FROM pitches
        WHERE batter_id = ?
        ORDER BY season, game_id
    """, conn, params=[player_id])

    if df.empty:
        return {"rows": [], "columns": [], "column_groups": [], "role": "batter"}

    df = _add_flags(df)

    season_teams = _per_season_teams_bulk(conn, player_id, False)
    awards_by_season = _player_awards(conn, player_id)
    rows = []
    for season, grp in df.groupby("season", sort=True):
        team_id = season_teams.get(int(season))
        team = _team_info(conn, team_id)

        pa = grp[grp["events"].notna()]
        k  = int(pa["events"].isin(STRIKEOUT_EVENTS).sum())
        bb = int(pa["events"].isin(WALK_EVENTS).sum())
        hr = int((pa["events"] == "home_run").sum())
        n_pa = len(pa)
        ev = grp["launch_speed"].dropna()
        n_ev = len(ev)
        hard = int((ev >= 95).sum())
        xwoba_vals = grp["estimated_woba_using_speedangle"].dropna()
        swings = int(grp["is_swing"].sum())
        whiffs = int(grp["is_whiff"].sum())
        outcomes = _pa_outcome_metrics(pa)

        adv = conn.execute(
            "SELECT wrc_plus, fwar_batting, war_batting, war, primary_position FROM player_advanced_stats WHERE player_id = ? AND season = ?",
            (player_id, int(season)),
        ).fetchone()
        wrc_plus = None
        war = None
        primary_pos = None
        if adv:
            wrc_plus = round(float(adv[0])) if adv[0] is not None else None
            war = _cf(adv[1] or adv[2] or adv[3], 1)
            primary_pos = adv[4]

        hr_rank = _top10_rank_batter(conn, int(season), hr, "hr")

        row = {
            "season":       int(season),
            "team_abbr":    team["abbr"],
            "team_id":      team["team_id"],
            "gp":           grp["game_id"].nunique(),
            "pa":           n_pa,
            "hr":           hr,
            "k":            k,
            "bb":           bb,
            "ops":          _cf(outcomes["OPS"], 3),
            "woba":         _cf(outcomes["wOBA"], 3),
            "xwoba":        _cf(xwoba_vals.mean(), 3) if len(xwoba_vals) else None,
            "wrc_plus":     wrc_plus,
            "avg_ev":       _cf(ev.mean(), 1) if n_ev else None,
            "hard_hit_pct": _cf(hard / n_ev * 100, 1) if n_ev else None,
            "k_pct":        _cf(k / n_pa * 100, 1) if n_pa else None,
            "bb_pct":       _cf(bb / n_pa * 100, 1) if n_pa else None,
            "whiff_pct":    _cf(whiffs / swings * 100, 1) if swings else None,
            "war":              war,
            "primary_position": primary_pos,
            "awards":           awards_by_season.get(int(season)),
            "top10":            {"hr": hr_rank is not None and hr_rank <= 10},
        }
        rows.append(row)

    columns = [
        {"key": "season",       "label": "Season",    "group": "id"},
        {"key": "team",         "label": "Team",      "group": "team"},
        {"key": "gp",           "label": "G",         "group": "trad"},
        {"key": "pa",           "label": "PA",        "group": "trad"},
        {"key": "hr",           "label": "HR",        "group": "trad"},
        {"key": "k",            "label": "K",         "group": "trad"},
        {"key": "bb",           "label": "BB",        "group": "trad"},
        {"key": "ops",          "label": "OPS",       "group": "quality"},
        {"key": "woba",         "label": "wOBA",      "group": "quality"},
        {"key": "xwoba",        "label": "xwOBA",     "group": "quality"},
        {"key": "wrc_plus",     "label": "wRC+",      "group": "quality"},
        {"key": "avg_ev",       "label": "Avg EV",    "group": "contact"},
        {"key": "hard_hit_pct", "label": "Hard-Hit%", "group": "contact"},
        {"key": "k_pct",        "label": "K%",        "group": "disc"},
        {"key": "bb_pct",       "label": "BB%",       "group": "disc"},
        {"key": "whiff_pct",    "label": "Whiff%",    "group": "disc"},
        {"key": "war",              "label": "WAR",     "group": "war"},
        {"key": "primary_position", "label": "Pos",     "group": "pos"},
        {"key": "awards",           "label": "Awards",  "group": "awards"},
    ]

    column_groups = [
        {"label": None,        "group": "id"},
        {"label": None,        "group": "team"},
        {"label": "Counting",  "group": "trad"},
        {"label": "Quality",   "group": "quality"},
        {"label": "Contact",   "group": "contact"},
        {"label": "Plate Disc","group": "disc"},
        {"label": None,        "group": "war"},
        {"label": None,        "group": "pos"},
        {"label": None,        "group": "awards"},
    ]

    return {"rows": rows, "columns": columns, "column_groups": column_groups, "role": "batter"}
