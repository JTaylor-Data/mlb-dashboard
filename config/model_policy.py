"""
config/model_policy.py — Single source of truth for per-model scoring policy.

Contains minimum qualifying pitch counts, available seasons, and default
display seasons for every live model. Values here reproduce existing behavior
exactly (not changing any threshold or logic), so `perf/compare_golden.py`
should detect no API response change after this file is added.

Usage:
    from config.model_policy import STUFF_PLUS, LOCATION_PLUS, TUNNELING_PLUS, DECEPTION_PLUS

Each constant is a dict with:
  - min_pitches_scoring      : int or None (None = no explicit threshold)
  - min_pitches_leaderboard  : int or None
  - available_seasons        : list[int]
  - default_season           : int
  - reason                   : one-sentence plain-language explanation
  - notes                    : str with caveats / pointers to related decisions
"""

STUFF_PLUS = {
    "min_pitches_scoring": None,
    "min_pitches_leaderboard": 200,
    # 2015 excluded: spin_axis is 0% populated that year (no data to train or score).
    "available_seasons": list(range(2016, 2027)),
    "default_season": 2026,
    "reason": (
        "Scores every pitch's physical shape (velocity, movement, spin) — "
        "no explicit per-pitcher minimum because empirical-Bayes shrinkage handles "
        "small samples; the leaderboard filters to 200-pitch pitchers."
    ),
    "notes": (
        "Leaderboard min_pitches=200 lives in api/routers/stuff_model_report.py:30 "
        "(LEADERBOARD_MIN_PITCHES). Shrinkage K estimated fresh each season, "
        "range 0.4–2.8 pitches. Unseen season (e.g. 2027) uses LightGBM "
        "missing-category branch; annual retrain recommended (see NEEDS_DECISION.md ND-01)."
    ),
}

LOCATION_PLUS = {
    # MIN_PITCHES = 100 in analytics/location_v2_model.py:28
    "min_pitches_scoring": 100,
    "min_pitches_leaderboard": 100,
    "available_seasons": list(range(2015, 2027)),
    "default_season": 2026,
    "reason": (
        "Grades where pitches land relative to the zone; 100-pitch minimum "
        "ensures the average plate position is stable before a score is reported."
    ),
    "notes": (
        "Threshold enforced in analytics/location_v2_model.py leaderboard() and "
        "pitcher_score(). Excludes PO/UN/EP pitch types; maps KC→CU. "
        "Season categorical capped at 2015–2026 by _cat_levels() query; "
        "2027 season value coerced to NaN."
    ),
}

TUNNELING_PLUS = {
    # MIN_SEASON_PITCHES = 100 in analytics/tunneling_v2_model.py:36
    "min_pitches_scoring": 100,
    "min_pitches_leaderboard": 100,
    "available_seasons": list(range(2015, 2027)),
    "default_season": 2026,
    "reason": (
        "Grades how similar pitches look to each other using lag-context features; "
        "100 raw pitches (before lag-pair exclusion) required before scoring."
    ),
    "notes": (
        "Threshold applied in analytics/tunneling_v2_model.py:_raw_predictions() before "
        "the lag-feature pipeline. A pitcher also needs a commit-point entry "
        "(qualifying FF/SI/FC pitch) to appear in the commit table; without one they "
        "produce no rows. Live scorer is tunneling_v2_model.py (promoted 2026-09-21); "
        "router URLs still say /v3/."
    ),
}

DECEPTION_PLUS = {
    # No explicit count check in analytics/deception_model.py (see NEEDS_DECISION.md ND-03)
    "min_pitches_scoring": None,
    "min_pitches_leaderboard": None,
    # arm_angle available from ~2021; effective scoring range is 2021–2026 even though
    # the model meta lists 2015–2026 as trained seasons (see NEEDS_DECISION.md ND-02)
    "available_seasons": list(range(2021, 2027)),
    "default_season": 2026,
    "reason": (
        "Grades pitcher body and release mechanics; limited to seasons where "
        "arm_angle is populated in Statcast (introduced mid-2020, reliable from 2021)."
    ),
    "notes": (
        "No integer min_pitches check; returns None if feat.empty after dropna. "
        "Requires pitcher_bio record (height_in, weight_lb). REPORT_SEASON = 2026 "
        "hardcoded in api/routers/deception_report.py:25 — no season selector yet. "
        "See NEEDS_DECISION.md ND-02 and ND-03."
    ),
}

ALL_MODELS = {
    "stuff_plus": STUFF_PLUS,
    "location_plus": LOCATION_PLUS,
    "tunneling_plus": TUNNELING_PLUS,
    "deception_plus": DECEPTION_PLUS,
}
