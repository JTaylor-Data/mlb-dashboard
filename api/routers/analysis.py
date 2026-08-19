"""
/api/radar and /api/compare - two new views the user asked for on top of the
same percentile pools the player page already computes: a 6-axis radar
chart (pitcher or batter, same metric set, direction per-metric already
encoded in PITCHER_POLARITY/BATTER_POLARITY) and a batter-vs-batter
swing-decisions/batted-ball-outcomes comparison.
"""
import sys
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from percentiles import BATTER_POLARITY, PITCHER_POLARITY, player_percentiles

from api.db import get_conn
from api.presentation import (
    BATTER_FMT, BATTER_LABELS, COMPARE_BATTED_BALL_METRICS, COMPARE_SWING_METRICS,
    PITCHER_FMT, PITCHER_LABELS, RADAR_METRICS, format_value,
)
from api.routers.players import _batter_pool, _clean, _norm_split, _pitcher_pool

router = APIRouter(prefix="/api")


def _player_name(conn, player_id):
    row = conn.execute("SELECT name FROM players WHERE player_id = ?", (player_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Player not found")
    return row[0]


@router.get("/radar/{player_id}")
def get_radar(
    player_id: int,
    role: str = Query("pitcher", pattern="^(pitcher|batter)$"),
    split: str = Query("ALL", pattern="^(ALL|L|R)$"),
    season: int = Query(2026),
    conn=Depends(get_conn),
):
    name = _player_name(conn, player_id)
    split_filter = _norm_split(split)

    if role == "pitcher":
        pool = _pitcher_pool(conn, season, split_filter)
        polarity, labels, fmt = PITCHER_POLARITY, PITCHER_LABELS, PITCHER_FMT
    else:
        pool = _batter_pool(conn, season, split_filter)
        polarity, labels, fmt = BATTER_POLARITY, BATTER_LABELS, BATTER_FMT

    pct = player_percentiles(pool, player_id, polarity)
    axes = []
    for key, label in RADAR_METRICS:
        if key in pct:
            raw, p = pct[key]
            axes.append({
                "key": key, "label": labels.get(key, key),
                "raw": _clean(raw), "pct": round(float(p), 1),
                "value_str": format_value(raw, fmt.get(key, "{:.2f}")),
            })
        else:
            axes.append({"key": key, "label": labels.get(key, key), "raw": None, "pct": None, "value_str": "-"})

    return {"player_id": player_id, "name": name, "role": role, "split": split, "axes": axes}


def _metric_rows(pct, metrics):
    rows = []
    for key, label in metrics:
        if key in pct:
            raw, p = pct[key]
            rows.append({"key": key, "label": label, "raw": _clean(raw), "pct": round(float(p), 1),
                         "value_str": format_value(raw, BATTER_FMT.get(key, "{:.1%}"))})
        else:
            rows.append({"key": key, "label": label, "raw": None, "pct": None, "value_str": "-"})
    return rows


def _compare_metric_rows(pct):
    return _metric_rows(pct, COMPARE_SWING_METRICS), _metric_rows(pct, COMPARE_BATTED_BALL_METRICS)


@router.get("/compare")
def compare_players(
    player_id_a: int = Query(...),
    player_id_b: int = Query(...),
    split: str = Query("ALL", pattern="^(ALL|L|R)$"),
    season: int = Query(2026),
    conn=Depends(get_conn),
):
    split_filter = _norm_split(split)
    pool = _batter_pool(conn, season, split_filter)

    result = []
    for player_id in (player_id_a, player_id_b):
        name = _player_name(conn, player_id)
        team_id = conn.execute("SELECT team_id FROM players WHERE player_id = ?", (player_id,)).fetchone()[0]
        pct = player_percentiles(pool, player_id, BATTER_POLARITY)
        swing_rows, batted_rows = _compare_metric_rows(pct)
        result.append({
            "player_id": player_id, "name": name, "team_id": team_id,
            "swing_decisions": swing_rows, "batted_ball_outcomes": batted_rows,
        })

    return {"split": split, "players": result}
