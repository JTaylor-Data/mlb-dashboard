"""
Tunneling+ v2 - feature engineering. A genuine methodology revision of v1
(analytics/tunneling_model.py, UNTOUCHED by this file - v1's live artifacts
and scoring stay exactly as they are until a separate promotion decision).

## What changed from v1, and why

1. **Pitcher-specific commit point, not a fixed 23.5 ft.** v1 used one
   constant decision-point distance for every pitch thrown by every
   pitcher - a real, documented limitation (see tunneling_model.py's own
   docstring). v2 instead finds, per pitcher-season, that pitcher's own
   fastest fastball-family pitch (FF/SI/FC, whichever has the highest mean
   effective_speed) and computes where THAT pitch actually was 150ms
   before it arrived at the plate, using each qualifying pitch's own real
   trajectory - not a velocity-adjusted formula, the pitch's own physics.
   Every pitch that pitcher throws (any type) is then evaluated at THAT
   pitcher's own commit_point_distance. See build_commit_point_table().
2. **Junk pitch types excluded** (PO/UN/EP) from eligible pairs - same
   EXCLUDED_PITCH_TYPES set Stuff+ uses (imported, not duplicated, so the
   two models can't silently drift on what counts as "not a real pitch").
   KN stays eligible - a real competitive pitch type, not junk data, same
   reasoning as Stuff+.
3. **New distance-based feature, shape_break_plus** - captures a pair that
   "pops apart early, tracks back late" (or the reverse) relative to a
   straight-line interpolation between release and the commit point - see
   add_pair_features() below.
4. **New pitcher-season features**: release_consistency_score (identical
   construction to v1's own _release_consistency, tested for composite
   inclusion here rather than only displayed), arm_angle_consistency_score
   (same construction as Deception+'s arm_angle_std - std of arm_angle
   across the pitcher-season's own pitches, computed separately here, not
   imported, so Tunneling+ doesn't take a runtime dependency on
   Deception+'s internals for its own scoring), and arsenal_entropy
   (Shannon entropy over post-exclusion pitch-type usage shares).

## Reused from v1 unchanged

position_at_y() and _pairwise_mean_dist() (imported from
analytics.tunneling_model) - the underlying trajectory physics and the
full-pairwise-mean-distance methodology are NOT part of what this revision
changes (see tunneling_model.py's own docstring for why pairwise, not
centroid-to-centroid, is already the validated right call). Re-deriving
them here would risk a silent drift between v1 and v2 on math that isn't
actually in scope for this revision.
"""
import numpy as np
import pandas as pd

from analytics.stuff_features import EXCLUDED_PITCH_TYPES
from analytics.tunneling_model import Y0, PLATE_Y, position_at_y, _pairwise_mean_dist

FB_TYPES = ["FF", "SI", "FC"]  # fastball-family shapes eligible as the
    # pitcher's "fastest fastball" reference - same candidate set Stuff+
    # v3 uses for its own primary-fastball classification, but the
    # SELECTION criterion here is highest velocity, not highest usage -
    # a genuinely different question ("which pitch sets the hitter's
    # timing clock" vs. "which pitch is thrown most").

COMMIT_TIME_OFFSET_S = 0.150  # seconds before arrival at the plate - the
    # hitter reaction-time criterion Part B is built around, replacing
    # v1's fixed 23.5ft-from-plate constant with a fixed TIME-before-
    # arrival constant applied to each pitcher's own fastest real pitch.

MIN_PITCHES_PER_TYPE = 100  # matches v1's default; Part A.2 tests 200 too.
DISTANCE_FLOOR = 1.8  # inches - identical to v1's DISTANCE_FLOOR, same
    # reasoning (a divisor floor so near-zero separation doesn't produce
    # an explosive similarity score).

ANCHOR_TYPES = {"FF", "SI"}  # unchanged from v1.

RAW_QUERY_COLUMNS = [
    "pitcher_id", "pitch_type", "stand", "p_throws", "season", "at_bat_number",
    "pitch_number", "game_id", "release_pos_x", "release_pos_z", "vx0", "vy0", "vz0",
    "ax", "ay", "az", "plate_x", "plate_z", "description", "events",
    "effective_speed", "arm_angle", "delta_run_exp",
]  # delta_run_exp only needed by Part E's trained-model target - NOT
   # filtered to non-null here (unlike Stuff+'s convention) because Parts
   # A-D need every real pitch's trajectory regardless of whether it has a
   # scored run-value outcome; Part E's own training-data prep does its
   # own dropna on this column instead, keeping load_pitches() correct for
   # every caller.


def load_pitches(conn, season, pitcher_id=None):
    """Same shape as v1's load_pitcher_pitches, plus effective_speed (for
    fastest-fastball classification) and arm_angle (for
    arm_angle_consistency_score) - and the PO/UN/EP exclusion applied here,
    at load time, so every downstream v2 function only ever sees eligible
    pitch types (Part A.1)."""
    where = "WHERE season = ?"
    params = [season]
    if pitcher_id is not None:
        where += " AND pitcher_id = ?"
        params.append(pitcher_id)
    df = pd.read_sql(f"""
        SELECT {', '.join(RAW_QUERY_COLUMNS)}
        FROM pitches
        {where}
    """, conn, params=params)
    df["pitch_type"] = df["pitch_type"].replace({"KC": "CU"})
    df = df[~df["pitch_type"].isin(EXCLUDED_PITCH_TYPES)]
    required = ["release_pos_x", "release_pos_z", "vx0", "vy0", "vz0", "ax", "ay", "az", "plate_x", "plate_z"]
    df = df.dropna(subset=required + ["pitch_type"])
    return df


# --------------------------------------------------------------------- #
# Part B - pitcher-specific commit point
# --------------------------------------------------------------------- #

def _time_to_y(vy0, ay, target_y):
    """Time (seconds) for a pitch to travel from Y0 (measurement plane) to
    target_y, from the same constant-acceleration solve position_at_y()
    uses (v1/metrics.py's compute_vaa) - here returning t itself rather
    than the (x, z) position at that t."""
    vy0, ay = np.asarray(vy0, dtype=float), np.asarray(ay, dtype=float)
    with np.errstate(invalid="ignore"):
        vy_f = -np.sqrt(vy0 ** 2 - 2 * ay * (Y0 - target_y))
        t = (vy_f - vy0) / ay
    return t


def _y_at_time(vy0, ay, t):
    """Forward kinematic y-position (distance from the Y0 measurement
    plane, feet) at a given time t - the inverse operation of
    _time_to_y()/position_at_y() (those solve for t given a target
    position; this solves for position given a target t)."""
    return Y0 + vy0 * t + 0.5 * ay * t ** 2


def commit_point_distance_per_pitch(vy0, ay, time_offset_s=COMMIT_TIME_OFFSET_S):
    """Distance from the plate (feet), for EACH individual pitch, at
    `time_offset_s` before that specific pitch's own arrival - uses that
    pitch's own vy0/ay, not a league-average or pitcher-average velocity,
    so two pitches of the same nominal type still get individually
    correct answers if their velocity varies start to finish. NaN in,
    NaN out (a pitch with an unsolvable trajectory, e.g. ay >= 0, returns
    NaN via the same sqrt-of-negative mechanism position_at_y already
    relies on elsewhere in this codebase).

    Note on direction, confirmed empirically not assumed: this grows WITH
    velocity (harder throwers commit FARTHER from the plate, not closer) -
    a fixed TIME-before-arrival criterion converts to distance roughly as
    time_offset * velocity, so a faster pitch has covered more ground by
    that fixed time and is proportionally farther out. Intuitive once
    stated (more ground covered per unit time), but confirmed against
    real data (r=0.964 between fastest_fb_velo and commit_point_distance)
    rather than assumed from the formula alone - see build_tunneling_v2_
    dataset.py's Part B.3 output."""
    t_plate = _time_to_y(vy0, ay, PLATE_Y)
    t_commit = t_plate - time_offset_s
    y_commit = _y_at_time(vy0, ay, t_commit)
    return y_commit - PLATE_Y


def build_commit_point_table(df, time_offset_s=COMMIT_TIME_OFFSET_S):
    """Part B.1/B.2: one row per (pitcher_id, season) - fastest_fb_type
    (whichever of FF/SI/FC has the highest mean effective_speed that
    pitcher-season - a velocity question, deliberately NOT the usage
    question Stuff+ v3's own primary-fastball classification answers),
    fastest_fb_velo (that type's own mean effective_speed), and
    commit_point_distance (mean, across every real pitch of that type, of
    each pitch's own distance-from-plate `time_offset_s` before its own
    arrival - defaults to COMMIT_TIME_OFFSET_S/150ms, but callers compare
    150ms vs. 200ms by passing this explicitly). A pitcher-season with
    none of FF/SI/FC gets no row at all - real, small edge case, same as
    Stuff+ v3's fastball-reference table.

    `df` must already have pitch_type KC->CU merged (done in load_pitches)
    and needs effective_speed/vy0/ay - the raw query columns already
    include all three.
    """
    fb = df[df["pitch_type"].isin(FB_TYPES)].dropna(subset=["effective_speed", "vy0", "ay"])
    profile = fb.groupby(["pitcher_id", "season", "pitch_type"]).agg(
        n_pitches=("pitch_type", "size"),
        fastest_fb_velo=("effective_speed", "mean"),
    ).reset_index()

    idx = profile.groupby(["pitcher_id", "season"])["fastest_fb_velo"].idxmax()
    fastest = profile.loc[idx, ["pitcher_id", "season", "pitch_type", "fastest_fb_velo"]].rename(
        columns={"pitch_type": "fastest_fb_type"}
    )

    # Per-pitch commit-point distance, restricted to each pitcher-season's
    # OWN fastest_fb_type pitches, then averaged to one number.
    merged = fb.merge(fastest[["pitcher_id", "season", "fastest_fb_type"]], on=["pitcher_id", "season"], how="inner")
    merged = merged[merged["pitch_type"] == merged["fastest_fb_type"]]
    merged["commit_point_distance"] = commit_point_distance_per_pitch(
        merged["vy0"].to_numpy(dtype=float), merged["ay"].to_numpy(dtype=float), time_offset_s=time_offset_s
    )
    merged = merged.dropna(subset=["commit_point_distance"])
    commit_dist = merged.groupby(["pitcher_id", "season"])["commit_point_distance"].mean().reset_index()

    out = fastest.merge(commit_dist, on=["pitcher_id", "season"], how="inner")
    return out[["pitcher_id", "season", "fastest_fb_type", "fastest_fb_velo", "commit_point_distance"]]


def build_fixed_commit_point_table(df, distance_ft):
    """Alternative to build_commit_point_table(): a FIXED distance-from-
    plate for every pitcher-season (closer to v1's own single-constant
    23.5ft approach, just at a different, caller-chosen distance and
    labeled per-pitcher for comparability) - no per-pitch trajectory
    solve needed, since the commit point doesn't depend on velocity in
    this variant at all. Still requires a real fastest_fb_type
    classification (reuses the exact same velocity-based selection as
    the time-based version) so every commit-point variant carries the
    same fastest_fb_type/fastest_fb_velo columns for apples-to-apples
    comparison - only commit_point_distance itself differs."""
    fb = df[df["pitch_type"].isin(FB_TYPES)].dropna(subset=["effective_speed"])
    profile = fb.groupby(["pitcher_id", "season", "pitch_type"]).agg(
        fastest_fb_velo=("effective_speed", "mean"),
    ).reset_index()
    idx = profile.groupby(["pitcher_id", "season"])["fastest_fb_velo"].idxmax()
    fastest = profile.loc[idx, ["pitcher_id", "season", "pitch_type", "fastest_fb_velo"]].rename(
        columns={"pitch_type": "fastest_fb_type"}
    ).reset_index(drop=True)
    fastest["commit_point_distance"] = float(distance_ft)
    return fastest[["pitcher_id", "season", "fastest_fb_type", "fastest_fb_velo", "commit_point_distance"]]


# --------------------------------------------------------------------- #
# Part C - pair-level distance features at the pitcher-specific commit point
# --------------------------------------------------------------------- #

def pitch_type_centroids(df):
    """Per (pitcher_id, season, pitch_type) pitch count - grouped by
    season too, not just pitcher_id, since this dataset spans multiple
    seasons (unlike v1's own pitch_type_centroids, which only ever saw one
    season's df at a time and could get away with a pitcher_id-only
    group). Pooling across seasons here would silently corrupt every
    downstream qualification decision - confirmed as a real bug during
    initial testing (a KeyError surfaced it, but the qualification-set
    version would have failed silently instead)."""
    return df.groupby(["pitcher_id", "season", "pitch_type"]).agg(n_pitches=("pitch_type", "size")).reset_index()


def pairwise_distance_features(df, centroids, commit_table, min_pitches_per_type=MIN_PITCHES_PER_TYPE):
    """One row per (pitcher_id, type_a, type_b), both clearing
    min_pitches_per_type - release_distance (at Y0, i.e. the release
    plane), mid_distance (actual, at the geometric midpoint-by-feet
    between release and this pitcher's own commit point),
    commit_distance (at this pitcher's own commit_point_distance - the
    input to tunnel_spread_plus), and plate_distance (at PLATE_Y). A
    pitcher-season with no commit_point_distance (no qualifying FF/SI/FC)
    is skipped entirely - there's no principled commit point to evaluate
    at."""
    commit_lookup = commit_table.set_index(["pitcher_id", "season"])["commit_point_distance"].to_dict()
    qualifying_index = set(map(
        tuple, centroids[centroids["n_pitches"] >= min_pitches_per_type][["pitcher_id", "season", "pitch_type"]].to_numpy()
    ))
    rows = []
    for (pitcher_id, season), group in df.groupby(["pitcher_id", "season"]):
        commit_dist = commit_lookup.get((pitcher_id, season))
        if commit_dist is None:
            continue
        commit_y = PLATE_Y + commit_dist
        mid_y = (Y0 + commit_y) / 2

        types = sorted(t for t in group["pitch_type"].unique() if (pitcher_id, season, t) in qualifying_index)
        if len(types) < 2:
            continue
        subs = {t: group[group["pitch_type"] == t].copy() for t in types}
        for t, sub in subs.items():
            sub["mid_x"], sub["mid_z"] = position_at_y(
                sub["release_pos_x"], sub["release_pos_z"], sub["vx0"], sub["vy0"], sub["vz0"],
                sub["ax"], sub["ay"], sub["az"], mid_y,
            )
            sub["commit_x"], sub["commit_z"] = position_at_y(
                sub["release_pos_x"], sub["release_pos_z"], sub["vx0"], sub["vy0"], sub["vz0"],
                sub["ax"], sub["ay"], sub["az"], commit_y,
            )
            subs[t] = sub
        total_pitches = sum(len(subs[t]) for t in types)

        for i in range(len(types)):
            for j in range(i + 1, len(types)):
                a, b = subs[types[i]], subs[types[j]]
                release_d = _pairwise_mean_dist(a, b, "release_pos_x", "release_pos_z")
                mid_d = _pairwise_mean_dist(a, b, "mid_x", "mid_z")
                commit_d = _pairwise_mean_dist(a, b, "commit_x", "commit_z")
                plate_d = _pairwise_mean_dist(a, b, "plate_x", "plate_z")
                usage_a, usage_b = len(a) / total_pitches, len(b) / total_pitches
                rows.append(dict(
                    pitcher_id=pitcher_id, season=season, type_a=types[i], type_b=types[j],
                    n_a=len(a), n_b=len(b),
                    release_distance=release_d, mid_distance=mid_d,
                    commit_distance=commit_d, plate_distance=plate_d,
                    usage_a=usage_a, usage_b=usage_b,
                    pair_weight=float(np.sqrt(usage_a * usage_b)),
                ))
    return pd.DataFrame(rows)


def league_pair_baselines(pairs):
    avg_release = float(np.average(pairs["release_distance"], weights=pairs["pair_weight"]))
    avg_commit = float(np.average(pairs["commit_distance"], weights=pairs["pair_weight"]))
    avg_plate = float(np.average(pairs["plate_distance"], weights=pairs["pair_weight"]))
    return avg_release, avg_commit, avg_plate


def add_pair_features(pairs, avg_release=None, avg_commit=None, avg_plate=None):
    """Adds release_similarity_plus, tunnel_spread_plus (was
    decision_similarity_plus - same ratio logic, now at the pitcher's own
    commit point), late_divergence_plus (was plate_separation_plus,
    unchanged), and shape_break_plus (new - see module docstring). Does
    NOT add a combined score - that's Part D's job, several candidate
    ways of combining these, tested side by side."""
    if avg_release is None or avg_commit is None or avg_plate is None:
        avg_release, avg_commit, avg_plate = league_pair_baselines(pairs)
    pairs = pairs.copy()
    pairs["release_similarity_plus"] = 100 * avg_release / pairs["release_distance"].clip(lower=DISTANCE_FLOOR)
    pairs["tunnel_spread_plus"] = 100 * avg_commit / pairs["commit_distance"].clip(lower=DISTANCE_FLOOR)
    pairs["late_divergence_plus"] = 100 * pairs["plate_distance"] / avg_plate

    # shape_break_plus: actual midpoint separation vs. what a straight-line
    # interpolation between release and the commit point would predict.
    # mid_y is EXACTLY halfway between Y0 (release plane) and commit_y by
    # construction (see pairwise_distance_features), so the linear
    # interpolation is just the arithmetic mean of the two endpoint
    # distances - not 100-scaled yet, a raw signed inch value (Part C.3).
    pairs["expected_mid_distance"] = (pairs["release_distance"] + pairs["commit_distance"]) / 2
    pairs["actual_mid_distance"] = pairs["mid_distance"]
    pairs["shape_break_plus"] = pairs["actual_mid_distance"] - pairs["expected_mid_distance"]
    return pairs


# --------------------------------------------------------------------- #
# Part C - pitcher-season aggregate features
# --------------------------------------------------------------------- #

def pitcher_season_consistency_features(df, centroids, min_pitches_per_type=MIN_PITCHES_PER_TYPE):
    """release_consistency_score (identical construction to v1's own
    _release_consistency - RMS distance of each pitch's release point
    from the pitcher-season's own mean release point, inches),
    arm_angle_consistency_score (std of arm_angle across the pitcher-
    season's own pitches, same construction as Deception+'s
    arm_angle_std, computed independently here rather than imported), and
    arsenal_entropy (Shannon entropy, natural log/nats, over post-
    exclusion pitch-type usage shares) - restricted to qualifying pitch
    types only (same MIN_PITCHES_PER_TYPE gate the pair scoring uses), so
    a single show-me pitch can't distort either the release centroid or
    the entropy figure."""
    qualifying = centroids[centroids["n_pitches"] >= min_pitches_per_type]
    valid_index = qualifying.set_index(["pitcher_id", "season", "pitch_type"]).index
    df = df[df.set_index(["pitcher_id", "season", "pitch_type"]).index.isin(valid_index)].copy()

    g = df.groupby(["pitcher_id", "season"])
    centers = g[["release_pos_x", "release_pos_z"]].transform("mean")
    sq_dist = (df["release_pos_x"] - centers["release_pos_x"]) ** 2 + (df["release_pos_z"] - centers["release_pos_z"]) ** 2
    release_consistency = (sq_dist.groupby([df["pitcher_id"], df["season"]]).mean() ** 0.5) * 12  # feet -> inches

    arm_angle_consistency = g["arm_angle"].std()

    usage = qualifying.copy()
    totals = usage.groupby(["pitcher_id", "season"])["n_pitches"].transform("sum")
    usage["p"] = usage["n_pitches"] / totals
    usage["term"] = -usage["p"] * np.log(usage["p"])
    entropy = usage.groupby(["pitcher_id", "season"])["term"].sum()

    out = pd.DataFrame({
        "release_consistency_score": release_consistency,
        "arm_angle_consistency_score": arm_angle_consistency,
    })
    out = out.join(entropy.rename("arsenal_entropy"), how="outer")
    return out.reset_index()


# --------------------------------------------------------------------- #
# Part E - two grains for the trained model
# --------------------------------------------------------------------- #

def arsenal_aggregate_features(scored_pairs):
    """Part E, arsenal-aggregate grain: for each (pitcher_id, season,
    pitch_type), the usage-weighted average of tunnel_spread_plus/
    late_divergence_plus/shape_break_plus/release_similarity_plus across
    every qualifying pair involving that type - "how this pitch type
    tunnels with the REST of this pitcher's arsenal, on average." Meant
    to be joined onto every individual pitch of that type (same join
    pattern Stuff+ v3's separation features already use, joined by
    pitcher_id/season/pitch_type). Reuses `pair_weight` (already the
    hand-built composite's own aggregation weight), so this grain is
    genuinely "the hand-built version's own pair numbers, at the
    individual-pitch grain" rather than a fresh weighting scheme."""
    feature_cols = ["tunnel_spread_plus", "late_divergence_plus", "shape_break_plus", "release_similarity_plus"]
    a = scored_pairs[["pitcher_id", "season", "type_a", "pair_weight"] + feature_cols].rename(columns={"type_a": "pitch_type"})
    b = scored_pairs[["pitcher_id", "season", "type_b", "pair_weight"] + feature_cols].rename(columns={"type_b": "pitch_type"})
    stacked = pd.concat([a, b], ignore_index=True)

    def _wavg(g):
        return pd.Series({c: np.average(g[c], weights=g["pair_weight"]) for c in feature_cols})

    out = stacked.groupby(["pitcher_id", "season", "pitch_type"]).apply(_wavg, include_groups=False).reset_index()
    return out.rename(columns={c: f"arsenal_{c}" for c in feature_cols})


def sequential_pair_features(df, commit_table):
    """Part E, sequential/real-pair grain: for each pitch with a real,
    DIFFERENT-type pitch immediately preceding it in the SAME at-bat, the
    actual distance (inches) between THAT SPECIFIC pair of individual
    pitches (not an aggregate over every pitch of each type) at release,
    the geometric midpoint, this pitcher-season's own commit point, and
    the plate - plus a per-pitch shape_break value. Pitches with no
    qualifying preceding pitch (the first pitch of a PA, or a repeated
    same-type pitch immediately before) are simply absent from the
    output - excluded, not imputed, per direct instruction. Raw distances
    are returned (not +-scaled) - meant as trained-model INPUT features,
    where the model finds its own functional form rather than being hand-
    fed a pre-imposed linear "+" transform."""
    commit_lookup = commit_table.set_index(["pitcher_id", "season"])["commit_point_distance"].to_dict()
    df = df.copy()
    idx = df.set_index(["pitcher_id", "season"]).index
    df["commit_point_distance"] = [commit_lookup.get(k) for k in idx]
    df = df.dropna(subset=["commit_point_distance"])
    commit_y = PLATE_Y + df["commit_point_distance"]
    mid_y = (Y0 + commit_y) / 2

    df["mid_x"], df["mid_z"] = position_at_y(
        df["release_pos_x"], df["release_pos_z"], df["vx0"], df["vy0"], df["vz0"],
        df["ax"], df["ay"], df["az"], mid_y,
    )
    df["commit_x"], df["commit_z"] = position_at_y(
        df["release_pos_x"], df["release_pos_z"], df["vx0"], df["vy0"], df["vz0"],
        df["ax"], df["ay"], df["az"], commit_y,
    )

    df_sorted = df.sort_values(["game_id", "at_bat_number", "pitch_number"]).reset_index(drop=True)
    group_cols = ["game_id", "at_bat_number"]
    shift_cols = ["pitch_type", "release_pos_x", "release_pos_z", "mid_x", "mid_z", "commit_x", "commit_z", "plate_x", "plate_z"]
    prev = df_sorted.groupby(group_cols)[shift_cols].shift(1).add_prefix("prev_")
    combined = pd.concat([df_sorted, prev], axis=1)
    combined = combined.dropna(subset=["prev_pitch_type"])
    combined = combined[combined["pitch_type"] != combined["prev_pitch_type"]].copy()

    def _dist(x1, z1, x2, z2):
        return np.hypot((x1 - x2) * 12, (z1 - z2) * 12)  # feet -> inches

    combined["seq_release_distance"] = _dist(combined["release_pos_x"], combined["release_pos_z"], combined["prev_release_pos_x"], combined["prev_release_pos_z"])
    combined["seq_mid_distance"] = _dist(combined["mid_x"], combined["mid_z"], combined["prev_mid_x"], combined["prev_mid_z"])
    combined["seq_commit_distance"] = _dist(combined["commit_x"], combined["commit_z"], combined["prev_commit_x"], combined["prev_commit_z"])
    combined["seq_plate_distance"] = _dist(combined["plate_x"], combined["plate_z"], combined["prev_plate_x"], combined["prev_plate_z"])
    combined["seq_expected_mid_distance"] = (combined["seq_release_distance"] + combined["seq_commit_distance"]) / 2
    combined["seq_shape_break"] = combined["seq_mid_distance"] - combined["seq_expected_mid_distance"]
    return combined


# --------------------------------------------------------------------- #
# Part A/B (v3) - real within-at-bat pairing, fixed lag slots
# --------------------------------------------------------------------- #
# Supersedes sequential_pair_features() above (v2's sequential grain,
# lag_1-only, same-type-prior-pitch excluded) - kept in place unmodified
# as v2's own artifact-generating code, not reused by v3. v3 is a genuine
# redesign of the pairing logic, not a refinement of it: every pitch from
# the 2nd-onward in its at-bat pairs against EVERY real prior pitch up to
# MAX_LAG back (not just the immediately-preceding one), and - unlike v2 -
# a same-type prior pitch is NOT excluded (a pitcher's 2nd straight
# fastball at lag_2 is real information the model should see, not a
# case to throw out).

MAX_LAG = 6
LAG_POSITION_COLS = ["release_pos_x", "release_pos_z", "mid_x", "mid_z", "commit_x", "commit_z", "plate_x", "plate_z", "pitch_type"]


def sequential_lag_features(df, commit_table, max_lag=MAX_LAG):
    """One row per real pitch from the 2nd-onward in its at-bat (the 1st
    pitch of every at-bat has no lag_1 and is excluded entirely - not
    imputed). For each lag N in 1..max_lag, adds lagN_commit_distance,
    lagN_plate_distance, lagN_shape_break (all real inches, computed
    against that SPECIFIC prior pitch's own trajectory - not an aggregate,
    not filtered by whether it's the same or a different pitch type), and
    lagN_pitch_type (categorical - what type was thrown N pitches back).
    A lag slot beyond how many real prior pitches exist in that at-bat
    (e.g. lag_4 for the 2nd pitch of an at-bat) is true NaN across all 4
    of that slot's columns - not zero-filled, not duplicated from another
    slot. 6 slots x 4 columns = 24 lag-derived columns.

    mid_x/z and commit_x/z are computed ONCE per pitch (not once per lag
    pair) using this PITCHER-SEASON's own commit_point_distance from
    `commit_table` - safe to do before the lag shift since every pitch in
    a real at-bat shares the same pitcher, so a single pitcher-season's
    commit point never varies mid-at-bat."""
    commit_lookup = commit_table.set_index(["pitcher_id", "season"])["commit_point_distance"].to_dict()
    df = df.copy()
    idx = df.set_index(["pitcher_id", "season"]).index
    df["commit_point_distance"] = [commit_lookup.get(k) for k in idx]
    df = df.dropna(subset=["commit_point_distance"])
    commit_y = PLATE_Y + df["commit_point_distance"]
    mid_y = (Y0 + commit_y) / 2

    df["mid_x"], df["mid_z"] = position_at_y(
        df["release_pos_x"], df["release_pos_z"], df["vx0"], df["vy0"], df["vz0"],
        df["ax"], df["ay"], df["az"], mid_y,
    )
    df["commit_x"], df["commit_z"] = position_at_y(
        df["release_pos_x"], df["release_pos_z"], df["vx0"], df["vy0"], df["vz0"],
        df["ax"], df["ay"], df["az"], commit_y,
    )

    df_sorted = df.sort_values(["game_id", "at_bat_number", "pitch_number"]).reset_index(drop=True)
    g = df_sorted.groupby(["game_id", "at_bat_number"])

    lag_blocks = []
    for lag in range(1, max_lag + 1):
        lag_blocks.append(g[LAG_POSITION_COLS].shift(lag).add_prefix(f"lag{lag}_"))
    combined = pd.concat([df_sorted] + lag_blocks, axis=1)

    # 1st pitch of every at-bat has no lag_1 at all - excluded entirely,
    # not imputed. Every other row (2nd-onward) is kept regardless of
    # whether deeper lag slots exist for it.
    combined = combined.dropna(subset=["lag1_pitch_type"]).copy()

    def _dist(x1, z1, x2, z2):
        return np.hypot((x1 - x2) * 12, (z1 - z2) * 12)  # feet -> inches

    internal_cols = []
    for lag in range(1, max_lag + 1):
        p = f"lag{lag}_"
        release_d = _dist(combined["release_pos_x"], combined["release_pos_z"], combined[p + "release_pos_x"], combined[p + "release_pos_z"])
        commit_d = _dist(combined["commit_x"], combined["commit_z"], combined[p + "commit_x"], combined[p + "commit_z"])
        mid_d = _dist(combined["mid_x"], combined["mid_z"], combined[p + "mid_x"], combined[p + "mid_z"])
        plate_d = _dist(combined["plate_x"], combined["plate_z"], combined[p + "plate_x"], combined[p + "plate_z"])
        expected_mid = (release_d + commit_d) / 2
        combined[f"{p}commit_distance"] = commit_d
        combined[f"{p}plate_distance"] = plate_d
        combined[f"{p}shape_break"] = mid_d - expected_mid
        # lagN_pitch_type already exists (from the shift) under this exact name.
        internal_cols += [p + "release_pos_x", p + "release_pos_z", p + "mid_x", p + "mid_z", p + "commit_x", p + "commit_z", p + "plate_x", p + "plate_z"]

    combined = combined.drop(columns=internal_cols)
    return combined


def lag_feature_columns(max_lag=MAX_LAG):
    """The 24 (at max_lag=6) lag-derived column names, in slot order -
    same list Part D's training script uses as its feature set, kept
    here as the single source of truth so the dataset-build script and
    the training script can't drift on naming."""
    cols = []
    for lag in range(1, max_lag + 1):
        cols += [f"lag{lag}_commit_distance", f"lag{lag}_plate_distance", f"lag{lag}_shape_break", f"lag{lag}_pitch_type"]
    return cols
