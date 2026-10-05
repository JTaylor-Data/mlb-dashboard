"""
Combined test: 5 Houck (RHP) + 5 Sale (LHP) pitches.

Validates:
  Task 1 — throw_side from Statcast p_throws; stride_norm_mirrored sign:
            crossfire should be negative for both RHP and LHP.
  Task 2 — detect_rubber_ankle_box success rate vs. chalk-circle fallback;
            per-pitch rubber position and detection method.

Reports a per-pitch table and aggregate success rates.
"""
import csv
import io
import os
import sys
import time

import cv2
import numpy as np
import requests
from bs4 import BeautifulSoup

SCRATCH = os.path.dirname(os.path.abspath(__file__))
D4 = os.path.join(SCRATCH, "deception_v4")
sys.path.insert(0, D4)

from d4_extract import load_models
from d4_pipeline import process_one

# ── Pitchers ─────────────────────────────────────────────────────────────────
PITCHERS = [
    {"name": "houck", "player_id": 657006, "p_throws": "R",
     "date_gt": "2025-04-01", "date_lt": "2025-09-01"},
    {"name": "sale",  "player_id": 519242, "p_throws": "L",
     "date_gt": "2025-04-01", "date_lt": "2025-09-01"},
]
TARGET_EACH = 5
MAX_TRIES_EACH = 30

OUT_DIR   = os.path.join(SCRATCH, "deception_v4", "out_houck_sale_test")
VIDEO_DIR = os.path.join(OUT_DIR, "videos")
LM_DIR    = os.path.join(OUT_DIR, "landmarks")

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0"}
VIDEO_HEADERS = {**HEADERS, "Referer": "https://baseballsavant.mlb.com/"}

STATCAST_URL = (
    "https://baseballsavant.mlb.com/statcast_search/csv"
    "?all=true&type=details&player_type=pitcher"
    "&pitchers_lookup%5B%5D={player_id}"
    "&game_date_gt={date_gt}&game_date_lt={date_lt}"
    "&sort_col=pitches&sort_order=desc&limit=50"
)
GAME_FEED_URL    = "https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
SAVANT_VIDEO_URL = "https://baseballsavant.mlb.com/sporty-videos?playId={}"


def statcast_pitches(player_id, date_gt, date_lt):
    url = STATCAST_URL.format(player_id=player_id, date_gt=date_gt, date_lt=date_lt)
    r = requests.get(url, timeout=30, headers=HEADERS)
    r.raise_for_status()
    return list(csv.DictReader(io.StringIO(r.content.decode("utf-8-sig"))))


def get_play_id(game_pk, ab_num, pitch_num):
    r = requests.get(GAME_FEED_URL.format(game_pk=game_pk), timeout=30, headers=HEADERS)
    r.raise_for_status()
    data = r.json()
    target_ab = int(ab_num) - 1
    for play in data.get("liveData", {}).get("plays", {}).get("allPlays", []):
        if play.get("atBatIndex") == target_ab:
            for evt in play.get("playEvents", []):
                if evt.get("isPitch") and evt.get("pitchNumber") == int(pitch_num):
                    return evt.get("playId")
    return None


def download_video(play_id, out_path):
    r = requests.get(SAVANT_VIDEO_URL.format(play_id), timeout=30, headers=HEADERS)
    r.raise_for_status()
    soup = BeautifulSoup(r.content, "html.parser")
    box = soup.find("div", class_="video-box")
    if not box: return False, "no video-box"
    vid = box.find("video")
    if not vid: return False, "no video tag"
    src = vid.find("source", type="video/mp4")
    if not src: return False, "no mp4 source"
    vr = requests.get(src["src"], stream=True, timeout=60, headers=VIDEO_HEADERS)
    vr.raise_for_status()
    with open(out_path, "wb") as f:
        for chunk in vr.iter_content(chunk_size=8192):
            f.write(chunk)
    return True, None


def fmt(v, d=3):
    if v is None: return "None"
    if isinstance(v, float): return f"{v:.{d}f}"
    return str(v)


def run_pitcher(pitcher_cfg, models):
    name       = pitcher_cfg["name"]
    player_id  = pitcher_cfg["player_id"]
    p_throws   = pitcher_cfg["p_throws"]
    date_gt    = pitcher_cfg["date_gt"]
    date_lt    = pitcher_cfg["date_lt"]

    print(f"\n{'='*65}")
    print(f"PITCHER: {name.upper()}  (id={player_id}, p_throws={p_throws})")
    print(f"{'='*65}")

    rows = statcast_pitches(player_id, date_gt, date_lt)
    print(f"  {len(rows)} Statcast rows fetched")
    if not rows:
        print("  No rows found — skipping pitcher.")
        return []

    results = []
    tried   = 0

    for row in rows[:MAX_TRIES_EACH]:
        if len(results) >= TARGET_EACH:
            break
        tried += 1
        game_pk   = row.get("game_pk", "").strip()
        ab_num    = row.get("at_bat_number", "").strip()
        pitch_num = row.get("pitch_number", "").strip()
        pitch_type = row.get("pitch_type", "?").strip()
        # Confirm p_throws from row (use configured default if field absent)
        throws_row = row.get("p_throws", p_throws).strip() or p_throws
        if not all([game_pk, ab_num, pitch_num]):
            continue

        tag = f"{name}_{tried:02d}"
        print(f"\n  [{len(results)+1}/{TARGET_EACH}] {tag}  "
              f"type={pitch_type}  gm={game_pk} ab={ab_num} p={pitch_num}  "
              f"p_throws={throws_row}", flush=True)

        try:
            play_id = get_play_id(game_pk, ab_num, pitch_num)
        except Exception as e:
            print(f"    feed error: {e}"); time.sleep(0.5); continue
        if not play_id:
            print("    no playId"); time.sleep(0.5); continue

        vid_path = os.path.join(VIDEO_DIR, f"{tag}.mp4")
        try:
            ok_dl, err_dl = download_video(play_id, vid_path)
        except Exception as e:
            print(f"    download error: {e}"); time.sleep(0.5); continue
        if not ok_dl:
            print(f"    download failed: {err_dl}"); time.sleep(0.5); continue

        sz = os.path.getsize(vid_path)
        if sz < 100_000:
            print(f"    tiny file ({sz}B) — skip")
            os.remove(vid_path); time.sleep(0.5); continue
        print(f"    {sz//1024}KB", flush=True)

        try:
            res = process_one(vid_path, models, tag,
                              keep_video=False, landmarks_dir=LM_DIR,
                              throws=throws_row)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"    pipeline error: {e}")
            if os.path.exists(vid_path): os.remove(vid_path)
            time.sleep(0.5); continue

        if not res["ok"]:
            print(f"    FAILED: {res.get('error')}")
            time.sleep(0.5); continue

        entry = {
            "pitcher":              name,
            "p_throws":             throws_row,
            "tag":                  tag,
            "pitch_type":           pitch_type,
            "stride_dir":           res.get("stride_direction"),
            "stride_norm":          res.get("stride_norm"),
            "stride_norm_mirrored": res.get("stride_norm_mirrored"),
            "rubber_pos":           res.get("rubber_position"),
            "rubber_method":        res.get("rubber_method"),
            "method":               res.get("method"),
        }
        results.append(entry)
        print(f"    OK  stride_dir={entry['stride_dir']}  "
              f"stride_norm={fmt(entry['stride_norm'])}  "
              f"mirrored={fmt(entry['stride_norm_mirrored'])}  "
              f"rubber_pos={fmt(entry['rubber_pos'])}  "
              f"rubber_method={entry['rubber_method']}", flush=True)
        time.sleep(0.5)

    return results


def print_table(all_results):
    print(f"\n\n{'='*90}")
    print("PER-PITCH RESULTS TABLE")
    print(f"{'='*90}")
    hdr = (f"{'pitcher':<8}  {'throws':<6}  {'tag':<12}  {'type':<4}  "
           f"{'stride_dir':<10}  {'stride_norm':>11}  {'mirrored':>9}  "
           f"{'rubber_pos':>10}  {'rubber_method':<14}  {'pipe_method'}")
    print(hdr)
    print("-" * len(hdr))
    for e in all_results:
        print(f"{e['pitcher']:<8}  {e['p_throws']:<6}  {e['tag']:<12}  "
              f"{e['pitch_type']:<4}  "
              f"{str(e['stride_dir'] or 'n/a'):<10}  "
              f"{fmt(e['stride_norm']):>11}  "
              f"{fmt(e['stride_norm_mirrored']):>9}  "
              f"{fmt(e['rubber_pos']):>10}  "
              f"{str(e['rubber_method'] or 'None'):<14}  "
              f"{e['method']}")
    print("-" * len(hdr))


def print_rubber_summary(all_results):
    print("\n── RUBBER DETECTION SUMMARY ──────────────────────────────────────")
    methods = [e["rubber_method"] for e in all_results]
    n_total  = len(methods)
    n_ankle  = sum(1 for m in methods if m == "ankle_box")
    n_chalk  = sum(1 for m in methods if m == "chalk")
    n_dirt   = sum(1 for m in methods if m == "not_dirt")
    n_none   = sum(1 for m in methods if m is None)
    print(f"  Total accepted pitches : {n_total}")
    print(f"  ankle_box succeeded    : {n_ankle}/{n_total}  ({100*n_ankle/max(1,n_total):.0f}%)")
    print(f"  chalk fallback used    : {n_chalk}/{n_total}")
    print(f"  not_dirt fallback used : {n_dirt}/{n_total}")
    print(f"  rubber detection None  : {n_none}/{n_total}")

    # Per-pitcher breakdown
    for pitcher in set(e["pitcher"] for e in all_results):
        pe = [e for e in all_results if e["pitcher"] == pitcher]
        pm = [e["rubber_method"] for e in pe]
        pa = sum(1 for m in pm if m == "ankle_box")
        print(f"    {pitcher:<8}: ankle_box {pa}/{len(pe)}")


def print_stride_summary(all_results):
    print("\n── STRIDE_NORM_MIRRORED SUMMARY (negative = crossfire) ───────────")
    for pitcher in sorted(set(e["pitcher"] for e in all_results)):
        pe = [e for e in all_results if e["pitcher"] == pitcher]
        print(f"  {pitcher.upper()} (p_throws={pe[0]['p_throws']}):")
        for e in pe:
            m = e["stride_norm_mirrored"]
            sign_ok = (m is not None and m < 0) if e["stride_dir"] == "CROSSFIRE" else True
            note = "" if e["stride_dir"] != "CROSSFIRE" else ("✓" if sign_ok else "✗ WRONG SIGN")
            print(f"    {e['tag']}: dir={str(e['stride_dir'] or 'n/a'):<10}  "
                  f"raw={fmt(e['stride_norm'])}  mirrored={fmt(m)}  {note}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(VIDEO_DIR, exist_ok=True)
    os.makedirs(LM_DIR, exist_ok=True)

    print("Loading models...", flush=True)
    models = load_models()
    print("Models loaded.\n", flush=True)

    all_results = []
    for p_cfg in PITCHERS:
        results = run_pitcher(p_cfg, models)
        all_results.extend(results)

    print_table(all_results)
    print_rubber_summary(all_results)
    print_stride_summary(all_results)

    import json
    out_json = os.path.join(OUT_DIR, "results_houck_sale.json")
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved -> {out_json}")


if __name__ == "__main__":
    main()
