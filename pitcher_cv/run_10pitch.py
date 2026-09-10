"""
10-pitch consistency test for Deception+ pipeline.
Pitcher: Framber Valdez (664285, LHP, Astros) — high leg kick, slightly unorthodox delivery.

Fetches up to 25 Statcast pitches, runs full pipeline on each with retry-on-fail logic,
collects 10 successful results, reports all individual stat values + medians.
"""
import csv
import io
import json
import os
import statistics
import sys
import time

import requests
from bs4 import BeautifulSoup

SCRATCH = os.path.dirname(os.path.abspath(__file__))
D4 = os.path.join(SCRATCH, "deception_v4")
sys.path.insert(0, D4)

from d4_pipeline import process_one
from d4_phases import PHASE_KEYS

# --- pitcher ---
PITCHER_NAME = "valdez"
PITCHER_ID   = 664285   # Framber Valdez, LHP, Astros

TARGET_OK    = 10       # stop collecting after this many successful pitches
MAX_TRIES    = 25       # maximum candidate pitches to attempt

# --- output dirs ---
OUT_DIR    = os.path.join(SCRATCH, "deception_v4", "out_10pitch")
VIDEOS_DIR = os.path.join(OUT_DIR, "videos")
LM_DIR     = os.path.join(OUT_DIR, "landmarks")

# --- network ---
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0"}
VIDEO_HEADERS = {**HEADERS, "Referer": "https://baseballsavant.mlb.com/"}
STATCAST_URL = (
    "https://baseballsavant.mlb.com/statcast_search/csv"
    "?all=true&type=details&player_type=pitcher"
    "&pitchers_lookup%5B%5D={player_id}"
    "&game_date_gt={start_dt}&game_date_lt={end_dt}"
    "&sort_col=pitches&sort_order=desc&limit=50"
)
GAME_FEED_URL  = "https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
SAVANT_VIDEO_URL = "https://baseballsavant.mlb.com/sporty-videos?playId={}"


def statcast_pitches(player_id, start_dt="2025-04-01", end_dt="2025-08-01"):
    url = STATCAST_URL.format(player_id=player_id, start_dt=start_dt, end_dt=end_dt)
    r = requests.get(url, timeout=30, headers=HEADERS)
    r.raise_for_status()
    text = r.content.decode("utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(text)))
    return rows


def get_play_id(game_pk, at_bat_number, pitch_number):
    r = requests.get(GAME_FEED_URL.format(game_pk=game_pk), timeout=30, headers=HEADERS)
    r.raise_for_status()
    data = r.json()
    target_ab = int(at_bat_number) - 1
    for play in data.get("liveData", {}).get("plays", {}).get("allPlays", []):
        if play.get("atBatIndex") == target_ab:
            for evt in play.get("playEvents", []):
                if evt.get("isPitch") and evt.get("pitchNumber") == int(pitch_number):
                    return evt.get("playId")
    return None


def download_video(play_id, out_path):
    r = requests.get(SAVANT_VIDEO_URL.format(play_id), timeout=30, headers=HEADERS)
    r.raise_for_status()
    soup = BeautifulSoup(r.content, "html.parser")
    box = soup.find("div", class_="video-box")
    if not box:
        return False, "no video-box"
    vid = box.find("video")
    if not vid:
        return False, "no video tag"
    src = vid.find("source", type="video/mp4")
    if not src:
        return False, "no mp4 source"
    vr = requests.get(src["src"], stream=True, timeout=60, headers=VIDEO_HEADERS)
    vr.raise_for_status()
    with open(out_path, "wb") as f:
        for chunk in vr.iter_content(chunk_size=8192):
            f.write(chunk)
    return True, None


def run_pipeline_on_video(video_path, tag, models):
    from d4_pipeline import process_one
    res = process_one(
        video_path, models, tag,
        keep_video=False,
        landmarks_dir=LM_DIR,
    )
    return res


def median_or_none(vals):
    clean = [v for v in vals if v is not None]
    if not clean:
        return None
    return statistics.median(clean)


def mode_str(vals):
    clean = [v for v in vals if v is not None]
    if not clean:
        return None
    return max(set(clean), key=clean.count)


def fmt(v, decimals=3):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(VIDEOS_DIR, exist_ok=True)
    os.makedirs(LM_DIR, exist_ok=True)
    frames_dir = os.path.join(LM_DIR, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    print(f"Loading models (YOLO + pose)...", flush=True)
    from d4_extract import load_models
    models = load_models()
    print("Models loaded.\n", flush=True)

    print(f"Fetching Statcast candidates for {PITCHER_NAME} (player_id={PITCHER_ID})...")
    rows = statcast_pitches(PITCHER_ID)
    if not rows:
        print("No Statcast rows found — check date range or player_id")
        return

    # Deduplicate by game+AB+pitch (avoid reprocessing the same event)
    seen = set()
    candidates = []
    for row in rows:
        key = (row.get("game_pk",""), row.get("at_bat_number",""), row.get("pitch_number",""))
        if all(key) and key not in seen:
            seen.add(key)
            candidates.append(row)
    print(f"  {len(candidates)} unique pitch candidates found\n")

    results = []
    tried = 0

    for row in candidates[:MAX_TRIES]:
        if len(results) >= TARGET_OK:
            break
        tried += 1
        game_pk  = row["game_pk"].strip()
        ab_num   = row["at_bat_number"].strip()
        pitch_num = row["pitch_number"].strip()
        pitch_type = row.get("pitch_type", "?").strip()
        pitch_n  = len(results) + 1

        tag = f"{PITCHER_NAME}_{tried:02d}"
        print(f"[{tried}/{MAX_TRIES}] {tag}  game={game_pk} ab={ab_num} p={pitch_num} type={pitch_type}", flush=True)

        # -- get play_id --
        try:
            play_id = get_play_id(game_pk, ab_num, pitch_num)
        except Exception as e:
            print(f"  feed lookup error: {e}")
            time.sleep(0.5)
            continue
        if not play_id:
            print("  no playId found")
            time.sleep(0.5)
            continue

        # -- download video --
        vpath = os.path.join(VIDEOS_DIR, f"{tag}.mp4")
        try:
            ok, err = download_video(play_id, vpath)
        except Exception as e:
            print(f"  download error: {e}")
            time.sleep(0.5)
            continue
        if not ok:
            print(f"  download failed: {err}")
            time.sleep(0.5)
            continue

        sz = os.path.getsize(vpath)
        if sz < 100_000:
            print(f"  tiny file ({sz} bytes) — likely CDN error page, skip")
            os.remove(vpath)
            time.sleep(0.5)
            continue
        print(f"  {sz//1024}KB downloaded ({play_id})", flush=True)

        # -- run pipeline --
        try:
            res = process_one(vpath, models, tag, keep_video=False, landmarks_dir=LM_DIR,
                              throws=row.get("p_throws", "R"))
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  pipeline error: {e}")
            if os.path.exists(vpath):
                os.remove(vpath)
            time.sleep(0.5)
            continue

        if not res["ok"]:
            reason = res.get("error", "unknown")
            print(f"  FAILED: {reason}")
            time.sleep(0.5)
            continue

        # -- success --
        pt  = res["phase_times"]
        p2r = (pt["release"] - pt["foot_plant"]) if (pt.get("release") and pt.get("foot_plant")) else None
        delivery = (pt["release"] - pt["leg_lift_start"]) if (pt.get("release") and pt.get("leg_lift_start")) else None
        p2p = (pt["foot_plant"] - pt["leg_lift_peak"]) if (pt.get("foot_plant") and pt.get("leg_lift_peak")) else None

        entry = {
            "pitch_n":       len(results) + 1,
            "tag":           tag,
            "pitch_type":    pitch_type,
            "plant_t":       pt.get("foot_plant"),
            "release_t":     pt.get("release"),
            "p2r":           p2r,
            "delivery_s":    delivery,
            "peak_to_plant": p2p,
            "stride_dir":    res.get("stride_direction"),
            "stride_norm":   res.get("stride_norm"),
            "rubber_pos":    res.get("rubber_position"),
            "method":        res.get("method"),
        }
        results.append(entry)

        print(
            f"  OK #{len(results):2d}  plant={fmt(pt.get('foot_plant'))}s  "
            f"rel={fmt(pt.get('release'))}s  p2r={fmt(p2r)}  "
            f"del={fmt(delivery)}  stride={res.get('stride_direction','n/a')} ({fmt(res.get('stride_norm'), 2)})"
        )
        time.sleep(0.5)

    # --- report ---
    print(f"\n{'='*75}")
    print(f"RESULTS: {len(results)}/{TARGET_OK} successful pitches for {PITCHER_NAME.upper()}")
    print(f"{'='*75}")

    # Column headers
    cols = ["#", "type", "plant_t", "release_t", "p2r", "delivery", "p2p", "stride", "stride_norm", "rubber"]
    hdr  = f"{'#':>3}  {'type':>4}  {'plant_t':>8}  {'release_t':>9}  {'p2r':>6}  {'delivery':>8}  {'p2p':>6}  {'stride':>9}  {'stride_norm':>11}  {'rubber':>7}"
    print(hdr)
    print("-" * len(hdr))

    for e in results:
        print(
            f"{e['pitch_n']:>3}  {e['pitch_type']:>4}  "
            f"{fmt(e['plant_t']):>8}  {fmt(e['release_t']):>9}  "
            f"{fmt(e['p2r']):>6}  {fmt(e['delivery_s']):>8}  "
            f"{fmt(e['peak_to_plant']):>6}  "
            f"{str(e['stride_dir'] or 'n/a'):>9}  "
            f"{fmt(e['stride_norm'], 2):>11}  "
            f"{fmt(e['rubber_pos'], 2):>7}"
        )

    print("-" * len(hdr))
    # Medians
    med_plant    = median_or_none([e["plant_t"] for e in results])
    med_release  = median_or_none([e["release_t"] for e in results])
    med_p2r      = median_or_none([e["p2r"] for e in results])
    med_delivery = median_or_none([e["delivery_s"] for e in results])
    med_p2p      = median_or_none([e["peak_to_plant"] for e in results])
    med_stride_n = median_or_none([e["stride_norm"] for e in results])
    mode_stride  = mode_str([e["stride_dir"] for e in results])
    med_rubber   = median_or_none([e["rubber_pos"] for e in results])

    print(
        f"{'MED':>3}  {'---':>4}  "
        f"{fmt(med_plant):>8}  {fmt(med_release):>9}  "
        f"{fmt(med_p2r):>6}  {fmt(med_delivery):>8}  "
        f"{fmt(med_p2p):>6}  "
        f"{str(mode_stride or 'n/a'):>9}  "
        f"{fmt(med_stride_n, 2):>11}  "
        f"{fmt(med_rubber, 2):>7}"
    )
    print(f"{'='*75}")

    # Save JSON
    out_json = os.path.join(OUT_DIR, "results_10pitch.json")
    with open(out_json, "w") as f:
        json.dump({
            "pitcher": PITCHER_NAME,
            "player_id": PITCHER_ID,
            "n_ok": len(results),
            "n_tried": tried,
            "results": results,
            "medians": {
                "plant_t":       med_plant,
                "release_t":     med_release,
                "p2r":           med_p2r,
                "delivery_s":    med_delivery,
                "peak_to_plant": med_p2p,
                "stride_norm":   med_stride_n,
                "stride_dir":    mode_stride,
                "rubber_pos":    med_rubber,
            }
        }, f, indent=2)
    print(f"\nSaved -> {out_json}")
    return results


if __name__ == "__main__":
    main()
