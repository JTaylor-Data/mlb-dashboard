"""
Derived pitch-level metrics computed on top of the `pitches` table.
Nothing here is persisted as a new table - these are queries/functions that
attach computed columns to a DataFrame when loaded, per the project's
single-source-of-truth rule for pitch-level data.
"""
import numpy as np
import pandas as pd

# Statcast trajectory constants (FanGraphs VAA primer):
# https://blogs.fangraphs.com/a-visualized-primer-on-vertical-approach-angle-vaa/
VAA_Y0 = 50.0     # release-side measurement plane, feet from home plate
VAA_YF = 17 / 12  # front edge of home plate, feet from home plate


def compute_vaa(vy0, vz0, ay, az):
    """
    Vertical Approach Angle, in degrees, per the FanGraphs formula:
        vy_f = -sqrt(vy0^2 - 2*ay*(y0 - yf))
        t    = (vy_f - vy0) / ay
        vz_f = vz0 + az*t
        VAA  = -atan(vz_f / vy_f) * (180/pi)
    Accepts scalars or array-likes (pandas Series / numpy arrays); NaN in,
    NaN out.
    """
    vy0, vz0, ay, az = (np.asarray(x, dtype=float) for x in (vy0, vz0, ay, az))
    with np.errstate(invalid="ignore"):
        vy_f = -np.sqrt(vy0**2 - 2 * ay * (VAA_Y0 - VAA_YF))
        t = (vy_f - vy0) / ay
        vz_f = vz0 + az * t
        vaa = -np.degrees(np.arctan(vz_f / vy_f))
    return vaa


def classify_pitcher_roles(conn, season, min_games_started=3, min_start_pct=0.5):
    """
    Classifies every pitcher who has thrown a pitch in `pitches` (for the
    given season) as 'SP' or 'RP'. There's no role field anywhere in the
    source data (the Stats API's primaryPosition is just 'P' for everyone),
    so this is derived purely from usage: within each game, the two pitchers
    with the earliest first at_bat_number (one per side) are that game's
    starters. A pitcher is 'SP' if that happened in at least
    `min_games_started` of their games and at least `min_start_pct` of their
    total appearances; otherwise 'RP'.

    Known limitation: this can't distinguish a true starter from an "opener"
    (a reliever used to throw the first inning ahead of the real bulk
    pitcher) - both look identical from at-bat ordering alone.

    Returns a DataFrame indexed by pitcher_id with columns: games_pitched,
    games_started, start_pct, role.
    """
    first_appearance = pd.read_sql("""
        SELECT game_id, pitcher_id, MIN(at_bat_number) AS first_ab
        FROM pitches
        WHERE season = ?
        GROUP BY game_id, pitcher_id
    """, conn, params=(season,))
    first_appearance["rank_in_game"] = (
        first_appearance.groupby("game_id")["first_ab"].rank(method="first")
    )
    first_appearance["is_start"] = first_appearance["rank_in_game"] <= 2

    role = first_appearance.groupby("pitcher_id").agg(
        games_pitched=("game_id", "nunique"),
        games_started=("is_start", "sum"),
    )
    role["start_pct"] = role["games_started"] / role["games_pitched"]
    role["role"] = np.where(
        (role["games_started"] >= min_games_started) & (role["start_pct"] >= min_start_pct),
        "SP", "RP",
    )
    return role


def compute_barrel(launch_speed, launch_angle):
    """
    Barrel classification (1/0), per the community-standard reverse-engineered
    approximation of Statcast's proprietary formula (Bill Petti's code_barrel()
    in the baseballr package - https://billpetti.github.io/baseballr/reference/code_barrel.html).
    Statcast has never published an exact closed-form formula, only anchor
    examples (98mph->26-30deg, 116mph->8-50deg); this approximation matches
    those almost exactly, with ~1deg imprecision at the extreme low-velocity
    edge. NaN in (e.g. non-batted-ball pitches) -> 0, not a barrel.
    """
    ev = np.asarray(launch_speed, dtype=float)
    la = np.asarray(launch_angle, dtype=float)
    with np.errstate(invalid="ignore"):
        is_barrel = (
            (la <= 50)
            & (ev >= 97)
            & ((ev * 1.5 - la) >= 117)
            & ((ev + la) >= 123)
        )
    return np.where(np.isnan(ev) | np.isnan(la), False, is_barrel)


def classify_pitch_movement(pfx_x, pfx_z):
    """
    Per-pitch movement axis: 'horizontal' if the horizontal break magnitude
    exceeds the vertical break magnitude, else 'vertical'. Used to build each
    pitcher's East-West / North-South arsenal profile from measured movement
    rather than a hand-maintained pitch-type name list.
    """
    pfx_x, pfx_z = np.asarray(pfx_x, dtype=float), np.asarray(pfx_z, dtype=float)
    return np.where(np.abs(pfx_x) > np.abs(pfx_z), "horizontal", "vertical")


def classify_pitcher_archetype(conn, season, threshold=0.6):
    """
    Labels each pitcher 'East-West' or 'North-South' based on their season-long
    arsenal shape: the share of their pitches that are horizontal-movement-
    dominant (see classify_pitch_movement). A pitcher is 'East-West' if that
    share exceeds `threshold`, 'North-South' if the vertical-dominant share
    exceeds `threshold`; pitchers in between are dropped entirely (excluded)
    so this split only ever measures genuinely one-sided arsenals.

    Returns a DataFrame indexed by pitcher_id with columns: total_pitches,
    pct_horizontal, archetype (only rows that got classified are included).
    """
    df = pd.read_sql("SELECT pitcher_id, pfx_x, pfx_z FROM pitches "
                      "WHERE season = ? AND pfx_x IS NOT NULL AND pfx_z IS NOT NULL", conn, params=(season,))
    df["movement_axis"] = classify_pitch_movement(df["pfx_x"], df["pfx_z"])

    agg = df.groupby("pitcher_id").agg(
        total_pitches=("movement_axis", "size"),
        pct_horizontal=("movement_axis", lambda s: (s == "horizontal").mean()),
    )
    agg["archetype"] = np.select(
        [agg["pct_horizontal"] > threshold, agg["pct_horizontal"] < (1 - threshold)],
        ["East-West", "North-South"],
        default=None,
    )
    return agg[agg["archetype"].notna()]


# wRC+ approximation. Real wRC+ needs park factors and FanGraphs' published
# wOBA_scale / league-runs-per-PA constants for the season - we don't have a
# FanGraphs pull (deferred, out of scope for now). This approximates using:
#   - league wOBA: computed live from our own Statcast data this season (real)
#   - wOBA_scale, lg R/PA: standard historical constants that don't swing much
#     year to year, but are NOT self-derived from our data like league wOBA is
#   - no park factor adjustment at all
# Treat it as directionally useful, not authoritative.
WOBA_SCALE_APPROX = 1.20
LG_R_PER_PA_APPROX = 0.115


def league_average_woba(conn, season):
    """Season-wide wOBA across all qualifying PAs (woba_denom = 1)."""
    row = conn.execute(
        "SELECT SUM(woba_value), SUM(woba_denom) FROM pitches WHERE season = ? AND woba_denom = 1",
        (season,),
    ).fetchone()
    total_woba, total_denom = row
    return total_woba / total_denom if total_denom else np.nan


def compute_wrc_plus(woba, league_woba):
    """
    wRC+ from a wOBA value (scalar or array-like) given the league wOBA.
    See module-level comment above for what's approximated vs. real here.
    """
    woba = np.asarray(woba, dtype=float)
    wrc_per_pa = ((woba - league_woba) / WOBA_SCALE_APPROX) + LG_R_PER_PA_APPROX
    return (wrc_per_pa / LG_R_PER_PA_APPROX) * 100
