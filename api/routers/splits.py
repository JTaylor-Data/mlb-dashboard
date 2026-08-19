"""
/api/splits - thin wrap of splits.py, parity with dashboard.py's
rudimentary splits explorer (mode/player/split-by/pitch-type -> a table).
"""
import sys
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends, Query

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from splits import compute_batter_splits, compute_pitcher_splits

from api.db import get_conn

router = APIRouter(prefix="/api")


@router.get("/splits/players")
def splits_players(role: str = Query("pitcher", pattern="^(pitcher|batter)$"),
                    season: int = Query(2026), conn=Depends(get_conn)):
    id_col = "pitcher_id" if role == "pitcher" else "batter_id"
    df = pd.read_sql(f"""
        SELECT DISTINCT pl.player_id, pl.name
        FROM players pl JOIN pitches p ON pl.player_id = p.{id_col}
        WHERE p.season = ?
        ORDER BY pl.name
    """, conn, params=(season,))
    return df.to_dict(orient="records")


@router.get("/splits/pitch-types")
def splits_pitch_types(
    role: str = Query("pitcher", pattern="^(pitcher|batter)$"),
    player_id: int = Query(...),
    season: int = Query(2026),
    conn=Depends(get_conn),
):
    id_col = "pitcher_id" if role == "pitcher" else "batter_id"
    df = pd.read_sql(
        f"SELECT DISTINCT pitch_type FROM pitches WHERE season = ? AND {id_col} = ? AND pitch_type IS NOT NULL ORDER BY pitch_type",
        conn, params=(season, player_id),
    )
    return df["pitch_type"].tolist()


@router.get("/splits")
def get_splits(
    role: str = Query("pitcher", pattern="^(pitcher|batter)$"),
    player_id: int = Query(...),
    split_by: str = Query(...),
    pitch_type: str | None = Query(None),
    season: int = Query(2026),
    conn=Depends(get_conn),
):
    pitch_type_arg = None if not pitch_type or pitch_type == "All" else pitch_type
    if role == "pitcher":
        df = compute_pitcher_splits(conn, season, split_by, pitcher_id=player_id, pitch_type=pitch_type_arg)
    else:
        df = compute_batter_splits(conn, season, split_by, batter_id=player_id, pitch_type=pitch_type_arg)

    if df.empty:
        return {"columns": [], "rows": []}

    df = df.reset_index().rename(columns={"index": "split_value"})
    pct_cols = [c for c in df.columns if c not in ("split_value", "n_pa", "n_pitches")]
    columns = ["split_value", "n_pa", "n_pitches"] + pct_cols
    rows = []
    for _, row in df.iterrows():
        cells = {}
        for c in columns:
            v = row[c]
            if c in pct_cols and pd.notna(v):
                cells[c] = f"{v:.3f}"
            elif pd.isna(v):
                cells[c] = "-"
            else:
                cells[c] = str(v)
        rows.append(cells)
    return {"columns": columns, "rows": rows}
