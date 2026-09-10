"""
Tunneling+ v1 - a first-pass, self-directed model of pitch-pair tunneling.

Formerly named "Deception+" - renamed because "deception" as a word covers
several distinct mechanical ideas (tunneling, release-point consistency,
spin mirroring, effective-velocity masking), and what this model actually
measures is specifically ONE of those: tunneling. Keeping the old name would
have kept implying it was the general-purpose "deception" model when it
isn't; renamed to say exactly what it measures instead.

No external spec was given for this one (unlike Stuff+ v2, which was built
to explicit direction) - the mechanism, the math, and the validation plan
below are all my own call. This docstring plus the accompanying
generated PDF (see build_report() at the bottom) are the record of that
reasoning, kept deliberately verbose since the whole point is that it should
be auditable afterward.

## What tunneling means, and why it was picked

Two pitch types from the same pitcher "tunnel" well if they travel through
nearly the same point in space at the moment a hitter has to commit to
swinging (the "decision point," well before the plate), but end up in very
different locations by the time they reach the plate. A hitter who can't
tell two pitches apart at the decision point is reacting to incomplete
information - that gap between "how similar they looked when he had to
decide" and "how different they actually were" IS the tunnel effect,
definitionally. This is the most commonly cited, most mechanistic idea in
public pitching research under the "deception" umbrella, and it's fully
computable from Statcast's own trajectory physics - no proxy needed.

## Units

Release/tunnel/plate distances are reported in **inches** - this is a
pitch-shape/movement-scale quantity (typically a few inches to a couple
feet), same convention as IVB/HB elsewhere on this site, and far more
readable than fractional feet. The decision point itself
(`DECISION_POINT_Y`) is a *location along the pitch's flight path*, not a
movement magnitude - kept in feet, same convention public research and this
site's own extension-style numbers use.

## Distances are full pairwise averages, not centroid-to-centroid

Every distance (release/tunnel/plate) is the mean straight-line distance
between EVERY individual pitch of type A and EVERY individual pitch of type
B (`_pairwise_mean_dist()`) - not the distance between the two types'
average positions. This matters more than it sounds: averaging each type's
position down to one centroid FIRST, then measuring one distance between
those two points, quietly erases real pitch-to-pitch command/movement
variance within each type. Checked directly on real data (Tarik Skubal's
CH/FF pair): the two types' centroids are 5.3" apart at the decision point,
but each type's own individual pitches vary by 4-5" (std dev) around their
own centroid - so the "typical" real changeup and "typical" real fastball
are actually 9.7" apart at that moment, not 5.3". Centroid-distance was
systematically UNDERSTATING true separation for any pitcher with real
command variance, i.e. every real pitcher. The full pairwise version is
maybe 15 seconds slower per season, leaguewide - not a meaningful cost for
a materially more honest number.

## The score: two independent components, combined by geometric mean

v1 originally scored each pair as a single ratio, `plate_distance /
tunnel_distance`. That coupled the two halves of the idea together by
division: a pair's plate separation only counted in proportion to how
small its tunnel separation already was, so a pair with a genuinely large
plate separation (literally the point of throwing two different pitch
types) could still score below average just because its tunnel separation
was also moderately large - real plate separation was being diluted, not
credited. Caught via a real example (Tarik Skubal's CH<->FF: the single
largest plate separation in his arsenal scored a below-average score under
the ratio, purely because its tunnel separation was also middling).

Fixed by splitting into two independently-normalized components, each
scaled 100=league-average:

- **decision_similarity_plus** = 100 * league_avg_tunnel_distance /
  max(tunnel_distance, FLOOR) - rewards being CLOSE at the decision point,
  on its own terms.
- **plate_separation_plus** = 100 * plate_distance / league_avg_plate_distance
  - rewards being FAR APART at the plate, on its own terms, regardless of
  how the tunnel separation happens to look.
- **tunneling_plus** = the geometric mean of the two. Requires BOTH to be
  reasonably good to produce a high combined score - a pair can't be
  "rescued" to a great score by acing just one - preserving the "close,
  THEN suddenly different" AND-logic that's the actual definition of
  tunneling, without the original ratio's harsh division-based coupling.

**A third component, release_similarity_plus, was tried and rejected.**
The theory was sound (a pair could tip itself off at release even if its
later trajectory converges), and it's still computed and shown per-pair -
but folding it into a 3-way geometric mean was tested against the full
validation battery and made the model measurably WORSE, not better: the
sequential same-at-bat test - the single most rigorous, mechanism-specific
check available - dropped from z=9.06 (2-way, see below) to z=1.60, below
the conventional significance bar. Real hitters' whiff rates on the second
pitch of a sequence just don't track release-point similarity the way they
track decision-point similarity and plate separation; diluting a strong,
validated signal with a much weaker one made the composite worse. Kept as
a displayed, unscored diagnostic instead - same role release_consistency
already plays.

Separately, distances (release/tunnel/plate) were switched from
centroid-to-centroid to the full pairwise mean between every individual
pitch of type A and every individual pitch of type B - see "Distances are
full pairwise averages" above. Unlike the release addition, this change
was a clear, unambiguous improvement across every validation check
(sequential test z: 5.17 -> 9.06; every outcome correlation increased).

Every scoring change here is a genuinely different function from what came
before, not a rescaling - pair rankings can and do change with each one.
Every validation check (split-half reliability, outcome correlation, the
sequential same-at-bat test) is re-run against the current version rather
than assumed to still hold from an earlier one; see build_report()'s
output.

## Fastball-anchor weighting in the pitcher-level composite

A hitter builds their read primarily against a pitcher's fastball (or, for a
sinkerballer/east-west arsenal, the sinker) - how well an offspeed pitch
tunnels with FF or SI matters more to that read than how two offspeed
pitches tunnel with each other. Per explicit user direction, pairs
involving FF or SI count double (`ANCHOR_WEIGHT_BONUS`), per unit of usage,
toward a pitcher's headline `tunneling_plus` - see `composite_weight_fn()`.
This ONLY affects the pitcher-level aggregate; `league_pair_baselines()`
(what "average" means for each of the three components) stays pure
usage-weighted, since that's describing the league's actual pair geometry,
not a personal-preference weighting. A pair's own tunneling_plus (shown
per-pair in the inspector) is unaffected either way - only how much each
pair counts toward the pitcher's single headline number changes.

## Why tunneling is computable at all from this data

Statcast fits every pitch to a constant-acceleration trajectory: velocity
(vx0, vy0, vz0) and acceleration (ax, ay, az) are given at a fixed
measurement plane, and this codebase already uses that exact model
elsewhere (metrics.py's compute_vaa, for Vertical Approach Angle) to solve
for the ball's state at a different y-plane than where it was measured.
This file generalizes that same solve from "z only, at the plate" to
"x AND z, at any distance from home plate" - which is exactly what's needed
to find where a pitch actually was in 3D space at a decision point instead
of only at release or only at the plate.
"""
import numpy as np
import pandas as pd

# Same measurement-plane constant metrics.py's compute_vaa uses (y=50ft is
# where Statcast's vx0/vy0/vz0 are actually measured) - treating
# release_pos_x/release_pos_z as the position AT that plane is the same
# simplification compute_vaa already makes (release truly happens a few
# feet closer, around y=54-55, but this keeps the model consistent with
# the rest of the codebase rather than introducing a second convention).
Y0 = 50.0
PLATE_Y = 17 / 12

# How far from home plate a hitter has effectively committed to a swing
# decision, IN FEET (a flight-path location, not a movement magnitude - see
# module docstring "Units"). Public pitch-tunneling research (Driveline's
# original public work on tunnel points, among others) generally cites a
# range of roughly 20-24 feet from the plate for when a hitter's "point of
# no return" is, depending on pitch velocity. 23.5 ft is a single fixed
# value inside that range - deliberately simple (one number for every
# pitch, not a velocity-adjusted one) for a v1. A velocity-adjusted
# decision point (faster pitches -> decision point slightly further from
# the plate, since less flight time remains) is a documented, real
# limitation - see build_report() - not something this version accounts for.
DECISION_POINT_Y = 23.5

MIN_PITCHES_PER_TYPE = 100  # a pitch type needs this many pitches (this
    # season, this pitcher) before it's trusted enough to include in any
    # pairwise tunnel comparison - a 4-pitch show-me slider's centroid/
    # pairwise-average is mostly noise, and noisy inputs inflate apparent
    # distances in both directions unpredictably.
DISTANCE_FLOOR = 1.8  # inches (= 0.15 ft); a divisor floor for ANY of the
    # three distance components, so a near-zero separation (e.g. two pitch
    # types with nearly identical release points) doesn't produce an
    # explosive, meaningless similarity score from dividing by ~0.

ANCHOR_TYPES = {"FF", "SI"}  # four-seam and sinker - the two "primary"
    # fastball shapes a hitter's read is built against. See module
    # docstring "Fastball-anchor weighting."
ANCHOR_WEIGHT_BONUS = 2.0  # a pair involving FF or SI counts this many
    # times its usage weight toward the pitcher-level composite score.


def position_at_y(release_x, release_z, vx0, vy0, vz0, ax, ay, az, target_y):
    """
    Where a pitch actually was (x, z) when it crossed a given y-plane
    (distance from home plate), computed from Statcast's constant-
    acceleration trajectory fit - the same quadratic solve metrics.py's
    compute_vaa uses for z only, generalized to x too and to an arbitrary
    target_y instead of only the plate. Returns feet (Statcast's native
    unit) - conversion to inches happens in _pairwise_mean_dist(), not
    here. NaN in, NaN out.
    """
    release_x, release_z, vx0, vy0, vz0, ax, ay, az = (
        np.asarray(v, dtype=float) for v in (release_x, release_z, vx0, vy0, vz0, ax, ay, az)
    )
    with np.errstate(invalid="ignore"):
        vy_f = -np.sqrt(vy0**2 - 2 * ay * (Y0 - target_y))
        t = (vy_f - vy0) / ay
        x = release_x + vx0 * t + 0.5 * ax * t**2
        z = release_z + vz0 * t + 0.5 * az * t**2
    return x, z


def load_pitcher_pitches(conn, season, pitcher_id=None):
    """Raw pitch rows with everything the model needs. `pitcher_id=None`
    loads the whole season (used for the league-wide build); a single id
    is used for on-demand per-pitcher inspection."""
    where = "WHERE season = ?"
    params = [season]
    if pitcher_id is not None:
        where += " AND pitcher_id = ?"
        params.append(pitcher_id)
    df = pd.read_sql(f"""
        SELECT pitcher_id, pitch_type, stand, p_throws, at_bat_number, pitch_number, game_id,
               release_pos_x, release_pos_z, vx0, vy0, vz0, ax, ay, az, plate_x, plate_z,
               description, events
        FROM pitches
        {where}
    """, conn, params=params)
    df["pitch_type"] = df["pitch_type"].replace({"KC": "CU"})  # same convention as stuff_features.py
    required = ["release_pos_x", "release_pos_z", "vx0", "vy0", "vz0", "ax", "ay", "az", "plate_x", "plate_z"]
    df = df.dropna(subset=required + ["pitch_type"])
    dx, dz = position_at_y(
        df["release_pos_x"], df["release_pos_z"], df["vx0"], df["vy0"], df["vz0"],
        df["ax"], df["ay"], df["az"], DECISION_POINT_Y,
    )
    df["decision_x"], df["decision_z"] = dx, dz
    df = df.dropna(subset=["decision_x", "decision_z"])
    return df


def pitch_type_centroids(df):
    """Per (pitcher_id, pitch_type): usage count, used to decide which
    types qualify (MIN_PITCHES_PER_TYPE) and their usage share for
    pair_weight. Pairwise distances themselves come from the raw per-pitch
    data (see pairwise_tunnel_scores), not from these centroid positions -
    kept here only for the count."""
    return df.groupby(["pitcher_id", "pitch_type"]).agg(
        n_pitches=("pitch_type", "size"),
    ).reset_index()


def _pairwise_mean_dist(a_df, b_df, colx, colz):
    """Full pairwise mean straight-line distance (inches) between every
    individual pitch in a_df and every individual pitch in b_df, at the
    given (colx, colz) point in the trajectory - NOT a distance between
    centroids (see module docstring). Vectorized via broadcasting; cheap
    even at full-season pitch counts (~1M pairwise combos in ~0.03s for a
    high-volume pitcher, ~15s for the whole league)."""
    ax = a_df[colx].to_numpy(dtype=float)[:, None] * 12
    az = a_df[colz].to_numpy(dtype=float)[:, None] * 12
    bx = b_df[colx].to_numpy(dtype=float)[None, :] * 12
    bz = b_df[colz].to_numpy(dtype=float)[None, :] * 12
    return float(np.hypot(ax - bx, az - bz).mean())


def pairwise_tunnel_scores(df, centroids):
    """
    Every within-pitcher pair of pitch types (both clearing
    MIN_PITCHES_PER_TYPE), with:
      - release_distance, tunnel_distance, plate_distance: full pairwise
        mean distance (inches) between every real pitch of type A and
        every real pitch of type B at release / the decision point / the
        plate respectively (see module docstring "Distances are full
        pairwise averages").
      - pair_weight: geometric mean of the two types' usage share within
        this pitcher's (qualifying) arsenal - a pair only matters to a
        hitter's actual experience in proportion to how often BOTH pitches
        actually show up.
    Raw distances only - `add_tunneling_plus()` turns these into the
    scored components (release_similarity_plus, decision_similarity_plus,
    plate_separation_plus, tunneling_plus), which needs league-wide
    baselines the caller supplies (see module docstring).
    One row per (pitcher_id, type_a, type_b) with type_a < type_b
    alphabetically (each unordered pair appears once).
    """
    qualifying_index = set(map(
        tuple, centroids[centroids["n_pitches"] >= MIN_PITCHES_PER_TYPE][["pitcher_id", "pitch_type"]].to_numpy()
    ))
    rows = []
    for pitcher_id, group in df.groupby("pitcher_id"):
        types = sorted(t for t in group["pitch_type"].unique() if (pitcher_id, t) in qualifying_index)
        if len(types) < 2:
            continue
        subs = {t: group[group["pitch_type"] == t] for t in types}
        total_pitches = sum(len(subs[t]) for t in types)
        for i in range(len(types)):
            for j in range(i + 1, len(types)):
                a, b = subs[types[i]], subs[types[j]]
                release_d = _pairwise_mean_dist(a, b, "release_pos_x", "release_pos_z")
                tunnel_d = _pairwise_mean_dist(a, b, "decision_x", "decision_z")
                plate_d = _pairwise_mean_dist(a, b, "plate_x", "plate_z")
                usage_a, usage_b = len(a) / total_pitches, len(b) / total_pitches
                rows.append(dict(
                    pitcher_id=pitcher_id, type_a=types[i], type_b=types[j],
                    n_a=len(a), n_b=len(b),
                    release_distance=release_d, tunnel_distance=tunnel_d, plate_distance=plate_d,
                    pair_weight=float(np.sqrt(usage_a * usage_b)),
                ))
    return pd.DataFrame(rows)


def league_pair_baselines(pairs):
    """Usage-weighted league-average release/tunnel/plate distance - the
    three baselines release_similarity_plus/decision_similarity_plus/
    plate_separation_plus are each normalized against (see module
    docstring). Computed once per season's pairs so pair-level and
    pitcher-level "+" numbers stay on the same scale."""
    avg_release = float(np.average(pairs["release_distance"], weights=pairs["pair_weight"]))
    avg_tunnel = float(np.average(pairs["tunnel_distance"], weights=pairs["pair_weight"]))
    avg_plate = float(np.average(pairs["plate_distance"], weights=pairs["pair_weight"]))
    return avg_release, avg_tunnel, avg_plate


def add_tunneling_plus(pairs, avg_release=None, avg_tunnel=None, avg_plate=None):
    """Adds release_similarity_plus, decision_similarity_plus,
    plate_separation_plus, and tunneling_plus to a copy of `pairs`.
    release_similarity_plus is computed and shown for context, but is NOT
    part of tunneling_plus - it was tested as a third geometric-mean
    component and measurably hurt validated predictive power (see module
    docstring). tunneling_plus is the geometric mean of decision_similarity_plus
    and plate_separation_plus only. Baselines default to being computed
    fresh from `pairs` itself; pass explicit ones to score a subset (e.g.
    one pitcher) against the full league's baseline instead of its own."""
    if avg_release is None or avg_tunnel is None or avg_plate is None:
        avg_release, avg_tunnel, avg_plate = league_pair_baselines(pairs)
    pairs = pairs.copy()
    pairs["release_similarity_plus"] = 100 * avg_release / pairs["release_distance"].clip(lower=DISTANCE_FLOOR)
    pairs["decision_similarity_plus"] = 100 * avg_tunnel / pairs["tunnel_distance"].clip(lower=DISTANCE_FLOOR)
    pairs["plate_separation_plus"] = 100 * pairs["plate_distance"] / avg_plate
    pairs["tunneling_plus"] = np.sqrt(pairs["decision_similarity_plus"] * pairs["plate_separation_plus"])
    return pairs


def composite_weight_fn(pairs):
    """Per-pair weight used ONLY when aggregating a pitcher's individual
    pairs into their headline tunneling_plus - pair_weight (usage-based),
    boosted by ANCHOR_WEIGHT_BONUS for any pair involving FF or SI (see
    module docstring "Fastball-anchor weighting"). Does not affect a pair's
    own tunneling_plus, and is NOT used for league_pair_baselines()."""
    is_anchor = pairs["type_a"].isin(ANCHOR_TYPES) | pairs["type_b"].isin(ANCHOR_TYPES)
    return pairs["pair_weight"] * is_anchor.map({True: ANCHOR_WEIGHT_BONUS, False: 1.0})


def pitcher_tunneling_scores(conn, season, df=None, pairs=None):
    """
    League-wide, one row per qualifying pitcher (needs at least 2 pitch
    types clearing MIN_PITCHES_PER_TYPE, i.e. at least one real pair to
    score):
      - tunneling_plus: composite-weighted average of each pair's own
        tunneling_plus (the geometric mean of decision_similarity_plus and
        plate_separation_plus - see module docstring), with FF/SI-involving
        pairs counted double (see "Fastball-anchor weighting") - THE
        headline number shown on the site. 100 = league average.
      - release_consistency: usage-weighted RMS distance (inches) of each
        pitch's release point from the pitcher's own overall weighted-mean
        release point, across the WHOLE arsenal, not just pairs - lower
        means "releases every pitch from about the same spot." A
        different, simpler, whole-arsenal diagnostic from the pairwise
        release_similarity_plus (computed and shown per-pair but not
        scored - see module docstring), kept separate.
      - n_pairs, n_pitch_types, total_pitches for context.

    `df`/`pairs` are optional pre-computed inputs (pairs already run through
    add_tunneling_plus) - a caller that also needs the raw pitches or pairs
    for other checks (build_tunneling_report.py's split-half/sequential/
    detail-export checks all need the same data) can compute them once and
    pass both through here instead of this function repeating the same
    full-season load and O(n^2) pairwise-distance pass on its own. Omit
    either (or both) for the standalone, self-contained behavior.
    """
    if df is None:
        df = load_pitcher_pitches(conn, season)
    centroids = pitch_type_centroids(df)
    if pairs is None:
        pairs = pairwise_tunnel_scores(df, centroids)
        if not pairs.empty:
            pairs = add_tunneling_plus(pairs)

    if pairs.empty:
        return pd.DataFrame(columns=[
            "tunneling_plus", "release_consistency", "n_pairs", "n_pitch_types", "total_pitches",
        ])

    pairs = pairs.copy()
    pairs["composite_weight"] = composite_weight_fn(pairs)
    pairs["weighted_plus"] = pairs["tunneling_plus"] * pairs["composite_weight"]
    agg = pairs.groupby("pitcher_id").agg(
        _weight_sum=("composite_weight", "sum"),
        _weighted_plus_sum=("weighted_plus", "sum"),
        n_pairs=("pitcher_id", "size"),
    )
    agg["tunneling_plus"] = agg["_weighted_plus_sum"] / agg["_weight_sum"]

    qualifying = centroids[centroids["n_pitches"] >= MIN_PITCHES_PER_TYPE]
    type_counts = qualifying.groupby("pitcher_id")["pitch_type"].nunique().rename("n_pitch_types")
    total_pitches = qualifying.groupby("pitcher_id")["n_pitches"].sum().rename("total_pitches")

    rc = _release_consistency(df, qualifying)

    out = agg[["tunneling_plus", "n_pairs"]].join(type_counts).join(total_pitches).join(rc)
    return out.dropna(subset=["tunneling_plus"])


def _release_consistency(df, qualifying_centroids):
    """Usage-weighted RMS distance (inches) of each individual pitch's
    release point from its pitcher's own overall release centroid,
    restricted to qualifying pitch types (same MIN_PITCHES_PER_TYPE filter
    as the pair scoring, so a single show-me pitch thrown from a wildly
    different slot can't dominate the number)."""
    valid_types = qualifying_centroids.set_index(["pitcher_id", "pitch_type"]).index
    df = df[df.set_index(["pitcher_id", "pitch_type"]).index.isin(valid_types)]
    centers = df.groupby("pitcher_id")[["release_pos_x", "release_pos_z"]].transform("mean")
    sq_dist = (df["release_pos_x"] - centers["release_pos_x"]) ** 2 + (df["release_pos_z"] - centers["release_pos_z"]) ** 2
    out = (sq_dist.groupby(df["pitcher_id"]).mean() ** 0.5) * 12  # feet -> inches
    return out.rename("release_consistency")
