"""
Rule-based batter archetype classifier. Percentile-threshold rules,
evaluated in priority order (most specific/rare combos first), applied to
the same percentile pool the player page already computes - so "high"
always means "high percentile," already polarity-corrected (see
percentiles.BATTER_POLARITY) except avg_launch_angle, which has no inherent
direction.

This is a v1: hand-picked thresholds, not fit/validated against anything.
Treat the labels as a fun, directionally-sane summary, not a rigorous
scouting grade.
"""
import pandas as pd

from percentiles import batter_percentile_metrics, player_percentiles, BATTER_POLARITY

HIGH = 65
VERY_HIGH = 80
LOW = 35
VERY_LOW = 20

ARCHETYPE_DESCRIPTIONS = {
    "5-Tool": "Elite (65th+ percentile) in power, hit tool, speed, and defense all at once - the rare complete player.",
    "Glove Guy": "Defense (OAA) in the 80th+ percentile carrying a below-average bat (xwOBA under 40th percentile).",
    "Three True Outcomes": "Strikes out a lot, walks a lot, and hits for real power - outcomes concentrated in K/BB/HR.",
    "Masher": "Power over everything - elite barrel rate and exit velo, nothing else required.",
    "Burner": "Elite speed (85th+) carrying a light bat (power 40th percentile or below).",
    "Waiter": "Lives on the walk - patient, rarely strikes out, doesn't hit for much power.",
    "Pest": "No power, no strikeouts, high expected average - the profile that's annoying to face.",
    "Steady Eddie": "Doesn't clear any specific archetype's bar - a generally competent, unremarkable everyday profile.",
}


def _p(pct, metric, default=50.0):
    return pct.get(metric, (None, default))[1]


def classify_batter_archetype(pct):
    """pct: output of percentiles.player_percentiles() for one batter.
    Returns an archetype name (always returns something - 'Steady Eddie' is
    the fallback for anyone who doesn't clear a specific bar)."""
    power = max(_p(pct, "xslg"), _p(pct, "barrel_pct"))
    hit = _p(pct, "xba")
    speed = _p(pct, "sprint_speed")
    defense = _p(pct, "oaa")
    bb = _p(pct, "bb_pct")
    contact = _p(pct, "k_pct")            # already inverted: high = low K%
    zone_contact = _p(pct, "zone_whiff_pct")  # inverted: high = rarely misses in zone
    ev = _p(pct, "avg_exit_velo")
    xwoba = _p(pct, "xwoba")

    tool_scores = [power, hit, speed, defense]
    if sum(1 for s in tool_scores if s >= HIGH) >= 4 and min(tool_scores) >= 55:
        return "5-Tool"

    if defense >= VERY_HIGH and xwoba <= LOW:
        return "Glove Guy"

    if contact <= LOW and bb >= HIGH and power >= HIGH:
        return "Three True Outcomes"

    if power >= VERY_HIGH and ev >= HIGH + 10:
        return "Masher"

    if speed >= VERY_HIGH and power <= 40:
        return "Burner"

    if bb >= 70 and contact >= 60 and power <= 50:
        return "Waiter"

    if power <= LOW and ev <= 40 and contact >= HIGH and zone_contact >= 60 and hit >= 60:
        return "Pest"

    return "Steady Eddie"


def classify_all_batters(conn, season, throws_filter="ALL"):
    """Returns a DataFrame indexed by batter_id with columns: name, archetype,
    plus the raw metrics used to decide, for every qualified batter."""
    pool = batter_percentile_metrics(conn, season, throws_filter=throws_filter)
    players = pd.read_sql("SELECT player_id, name FROM players", conn).set_index("player_id")

    rows = []
    for bid in pool.index:
        pct = player_percentiles(pool, bid, BATTER_POLARITY)
        archetype = classify_batter_archetype(pct)
        rows.append(dict(
            batter_id=bid,
            name=players["name"].get(bid, "?"),
            archetype=archetype,
            power_pct=max(_p(pct, "xslg"), _p(pct, "barrel_pct")),
            hit_pct=_p(pct, "xba"),
            speed_pct=_p(pct, "sprint_speed"),
            defense_pct=_p(pct, "oaa"),
            bb_pct=_p(pct, "bb_pct"),
            contact_pct=_p(pct, "k_pct"),
        ))
    return pd.DataFrame(rows).set_index("batter_id")
