"""Pull Air% follows Savant's reading: air = fly ball + line drive + popup, pulled = more than
15 true degrees toward the pull side of dead center, denominator = all batted balls."""
import numpy as np
import pandas as pd

from analytics.percentiles import _SPRAY_X0, _SPRAY_Y0, _add_shared_flags, _rate_metrics_pool


def _ball(stand, deg_toward_pull, bb_type, dist=200.0):
    """One batted ball hit `deg_toward_pull` true degrees to the pull side."""
    sign = -1 if stand == "R" else 1  # a right-hander pulls toward negative x (left field)
    a = np.radians(deg_toward_pull)
    return dict(batter_id=1, stand=stand, bb_type=bb_type,
                hc_x=_SPRAY_X0 + sign * np.sin(a) * dist, hc_y=_SPRAY_Y0 - np.cos(a) * dist)


def _pull_air(balls):
    df = pd.DataFrame(balls)
    for col, val in dict(description="hit_into_play", zone=5, events="single", launch_speed=95.0,
                         launch_angle=20.0).items():
        df[col] = val
    df = _add_shared_flags(df)
    return float(_rate_metrics_pool(df, "batter_id").loc[1, "pull_air_pct"])


def test_cutoff_is_fifteen_degrees_toward_the_pull_side():
    assert _pull_air([_ball("R", 20, "fly_ball")]) == 1.0
    assert _pull_air([_ball("R", 10, "fly_ball")]) == 0.0      # inside the middle third
    assert _pull_air([_ball("R", -20, "fly_ball")]) == 0.0     # opposite field
    assert _pull_air([_ball("L", 20, "line_drive")]) == 1.0    # lefties pull the other way


def test_popups_count_as_air_and_grounders_do_not():
    assert _pull_air([_ball("R", 30, "popup")]) == 1.0
    assert _pull_air([_ball("R", 30, "ground_ball")]) == 0.0


def test_denominator_is_all_batted_balls():
    balls = [_ball("R", 30, "fly_ball"), _ball("R", 30, "ground_ball"), _ball("R", -30, "line_drive"),
             _ball("R", 5, "popup")]
    assert _pull_air(balls) == 0.25
