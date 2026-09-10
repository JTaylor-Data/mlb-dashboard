"""
Player-season similarity ("comps") - nearest neighbors in PERCENTILE space
across a fixed, hand-picked set of traits, computed within each metric's
own season (so a 2021 EV percentile and a 2025 EV percentile both mean
"how good relative to THAT year's league," even if the raw scale drifted a
little year to year - juiced ball, sticky-stuff crackdown, etc.), then
averaged as a mean absolute percentile difference across the metric set to
rank every other player-season by how close it sits in that shared
percentile space. Deliberately simple and fully explainable, not a
machine-learned embedding - the same idea Baseball Savant's own Percentile
Rankings page uses for one stat at a time, just extended into a distance
metric across several of them at once.

## Pitcher comps: handedness lock (hard rule, not a preference)

A pitcher's similarity comps must always share his own throwing hand
(p_throws) - a percentile-close lefty is never offered as a comp for a
righty, regardless of distance. Enforced in find_comps() by restricting the
candidate pool to matching p_throws BEFORE ranking, not by penalizing a
mismatch in the distance metric (a soft penalty could still let an
opposite-hand pitcher win if everything else lined up closely enough; a
hard pool restriction can't).

## Pitcher similarity metric set (exactly these seven, per direct spec)

GB%, Avg EV, K%, Whiff%, O-Swing% (chase_pct), FB Velo, Arm Angle. Every
other metric previously in this set (Barrel%, BB%) has been removed.

## Arsenal-similarity adjustment - EXPLORATORY, FIRST ATTEMPT, NOT FINALIZED

Per direct instruction to add *some* reasonable arsenal-overlap adjustment
without a single obviously-correct method being specified, this
implementation: for each pitcher-season, builds the SET of pitch types
thrown at least 10% of the time that season (ARSENAL_MIN_USAGE, same
"real, not incidental" bar the rest of this project uses for pitch-type
qualification elsewhere - e.g. the movement plot's own usage-based
filtering), then computes the Jaccard distance (1 - overlap/union) between
the query pitcher's arsenal set and each candidate's, scaled to the same
0-100 range the percentile-distance metric already uses (100 = zero pitch
types in common, 0 = identical arsenals). This is ADDED to the mean
percentile difference with a modest blend weight (ARSENAL_BLEND_WEIGHT =
0.3, i.e. the percentile-similarity term still dominates the ranking) -
NOT a hard filter/exclusion, so a genuinely great percentile comp with a
somewhat different arsenal can still surface, just ranked slightly lower
than an equally-close comp who also throws a similar mix. A candidate with
no qualifying arsenal data gets NO penalty (treated as neutral, not
punished for a data gap). This is a first pass, not a validated
methodology - reasonable alternatives (a hard top-N-pitch-type match
requirement, a usage-weighted cosine similarity instead of a binary-set
Jaccard measure, a different blend weight) were not tested against this
one; flagged for review/sign-off, easy to revert by setting
ARSENAL_BLEND_WEIGHT back to 0.0.
"""
import pandas as pd

# Chosen per direction. Pitcher metrics are all "against" (opponent
# outcomes) except velo/arm_angle, matching how pitcher_percentile_metrics
# already defines them - no new formulas, just a curated subset of columns
# that function already computes.
HITTER_SIMILARITY_METRICS = [
    "avg_exit_velo", "barrel_pct", "k_pct", "bb_pct", "chase_pct", "whiff_pct", "sprint_speed", "oaa",
]
# Exactly these seven, per direct spec - see module docstring.
PITCHER_SIMILARITY_METRICS = [
    "gb_pct", "avg_exit_velo", "k_pct", "whiff_pct", "chase_pct", "velo", "arm_angle",
]

SIMILARITY_METRIC_LABELS = {
    "avg_exit_velo": "Avg EV", "barrel_pct": "Barrel%", "k_pct": "K%", "bb_pct": "BB%",
    "chase_pct": "O-Swing%", "whiff_pct": "Whiff%", "sprint_speed": "Sprint Speed",
    "oaa": "OAA", "gb_pct": "GB%", "velo": "FB Velo", "arm_angle": "Arm Angle",
}

ARSENAL_MIN_USAGE = 0.10  # a pitch type needs >= 10% season usage to count toward a pitcher's arsenal set - see module docstring
ARSENAL_BLEND_WEIGHT = 0.3  # how much the arsenal-overlap penalty (0-100 scale) counts toward the final distance, alongside the percentile-difference term (also 0-100 scale) - see module docstring


def build_similarity_pool(season_pools, metrics):
    """season_pools: {season: pool_df}, one already-qualified percentile
    pool per season (e.g. from pitcher_percentile_metrics/
    batter_percentile_metrics at the unsplit/ALL view - a career trait
    profile shouldn't depend on the L/R split toggle). Returns one
    DataFrame indexed by (player_id, season) with each of `metrics`
    replaced by its PERCENTILE RANK (0-100) within that season's own pool -
    polarity-agnostic on purpose (a comp is about how similar two profiles
    are, not who's better, so there's no PITCHER_POLARITY/BATTER_POLARITY
    inversion here, unlike the main percentile-badge system). Rows missing
    ANY of the requested metrics are dropped entirely - a comp built off a
    partial trait vector isn't comparable to one built off the full set."""
    frames = []
    for season, pool_df in season_pools.items():
        present = [m for m in metrics if m in pool_df.columns]
        if len(present) < len(metrics):
            continue
        pct = pool_df[metrics].rank(pct=True) * 100
        pct = pct.dropna(how="any").copy()
        if pct.empty:
            continue
        pct["season"] = season
        pct.index.name = "player_id"
        frames.append(pct.reset_index())
    if not frames:
        return pd.DataFrame(columns=["player_id", "season"] + metrics).set_index(["player_id", "season"])
    out = pd.concat(frames, ignore_index=True)
    return out.set_index(["player_id", "season"])


def build_pitcher_arsenal_pool(conn, seasons):
    """Per (pitcher_id, season): the SET of pitch types thrown at least
    ARSENAL_MIN_USAGE of the time that season, post PO/UN/EP exclusion
    (same junk-pitch-type set Stuff+ uses) and with KC merged into CU (same
    convention as the rest of this page). EXPLORATORY - see module
    docstring for exactly how/why this feeds find_comps()."""
    from analytics.stuff_features import EXCLUDED_PITCH_TYPES

    placeholders = ",".join("?" * len(seasons))
    df = pd.read_sql(f"""
        SELECT pitcher_id, season, pitch_type, COUNT(*) AS n
        FROM pitches WHERE season IN ({placeholders}) AND pitch_type IS NOT NULL
        GROUP BY pitcher_id, season, pitch_type
    """, conn, params=list(seasons))
    if df.empty:
        return pd.Series(dtype=object)
    df["pitch_type"] = df["pitch_type"].replace({"KC": "CU"})
    df = df[~df["pitch_type"].isin(EXCLUDED_PITCH_TYPES)]
    totals = df.groupby(["pitcher_id", "season"])["n"].transform("sum")
    df["pct"] = df["n"] / totals
    qualifying = df[df["pct"] >= ARSENAL_MIN_USAGE]
    return qualifying.groupby(["pitcher_id", "season"])["pitch_type"].apply(set)


def _arsenal_distance(target_set, other_set):
    """Jaccard distance (0 = identical arsenals, 100 = zero overlap),
    scaled to the same 0-100 range the percentile-difference metric uses."""
    if not target_set or not other_set:
        return 0.0  # missing arsenal data for either side - no penalty, don't punish a data gap
    union = target_set | other_set
    overlap = len(target_set & other_set) / len(union) if union else 1.0
    return (1 - overlap) * 100


def find_comps(pool, player_id, season, metrics, top_n=3, p_throws_by_player=None, arsenal_pool=None):
    """Nearest player-seasons to (player_id, season) in `pool` (as built by
    build_similarity_pool), ranked by mean absolute percentile difference
    across `metrics` (lower = more similar), optionally blended with an
    arsenal-overlap penalty (see ARSENAL_BLEND_WEIGHT / module docstring -
    EXPLORATORY). Excludes every season of the SAME player - a player isn't
    a meaningful "comp" for himself, even in a different year. Returns []
    if the query player-season itself didn't qualify for the pool (missed
    the qualifying threshold, or missing one of the required metrics that
    season).

    `p_throws_by_player`: optional {player_id: 'L'/'R'} map - when
    provided (pitcher comps only; hitters have no equivalent hard-lock
    concept in this build), candidates are restricted to the SAME
    p_throws as the query pitcher BEFORE ranking - a hard pool
    restriction, not a soft distance penalty, so an opposite-hand pitcher
    can never win regardless of how close his percentile profile is.
    `arsenal_pool`: optional build_pitcher_arsenal_pool() output.
    """
    key = (player_id, season)
    if key not in pool.index:
        return []
    target = pool.loc[key, metrics]
    others = pool[pool.index.get_level_values("player_id") != player_id]
    if p_throws_by_player is not None:
        my_hand = p_throws_by_player.get(player_id)
        if my_hand:
            other_ids = others.index.get_level_values("player_id")
            same_hand_mask = [p_throws_by_player.get(pid) == my_hand for pid in other_ids]
            others = others[same_hand_mask]
    if others.empty:
        return []

    diffs = (others[metrics] - target).abs().mean(axis=1)
    if arsenal_pool is not None and key in arsenal_pool.index:
        target_arsenal = arsenal_pool.loc[key]
        penalty = pd.Series(
            {idx: _arsenal_distance(target_arsenal, arsenal_pool.get(idx, set())) for idx in others.index},
            index=others.index,
        )
        diffs = diffs + ARSENAL_BLEND_WEIGHT * penalty

    nearest = diffs.nsmallest(top_n)
    out = []
    for (pid, yr), dist in nearest.items():
        out.append({
            "player_id": int(pid),
            "season": int(yr),
            "distance": float(dist),
            "percentiles": {m: round(float(others.loc[(pid, yr), m]), 1) for m in metrics},
        })
    return out
