"""
10-pitch crossfire pitcher test.

Target: Tanner Houck (RHP, Boston Red Sox) — publicly known crossfire delivery,
strides toward the 3B side as a right-hander.

Reports:
  - All individual pitch values
  - Median + IQR across 10 pitches
  - Per-section timing breakdown (median across pitches)
  - Phase screenshot grid for first successful pitch
"""
import csv, io, json, os, sys, time
import cv2, numpy as np, requests
from bs4 import BeautifulSoup

SCRATCH = os.path.dirname(os.path.abspath(__file__))
D4      = os.path.join(SCRATCH, "deception_v4")
sys.path.insert(0, D4)

from d4_extract  import load_models
from d4_pipeline import process_one

PITCHER_NAME = "houck"
PITCHER_ID   = 657006   # Tanner Houck, RHP

TARGET_PITCHES = 5
MAX_CANDIDATES = 20

OUT_DIR     = os.path.join(SCRATCH, "deception_v4", "out_houck_test")
VIDEO_DIR   = os.path.join(OUT_DIR, "videos")
LM_DIR      = os.path.join(OUT_DIR, "landmarks")
GRID_PATH   = os.path.join(OUT_DIR, "phase_grid.png")
RESULTS_JSON= os.path.join(OUT_DIR, "results.json")

HEADERS       = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0"}
VIDEO_HEADERS = {**HEADERS, "Referer": "https://baseballsavant.mlb.com/"}
STATCAST_URL  = (
    "https://baseballsavant.mlb.com/statcast_search/csv"
    "?all=true&type=details&player_type=pitcher"
    "&pitchers_lookup%5B%5D={player_id}"
    "&game_date_gt=2025-04-01&game_date_lt=2025-09-01"
    "&sort_col=pitches&sort_order=desc&limit={limit}"
)
GAME_FEED_URL    = "https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
SAVANT_VIDEO_URL = "https://baseballsavant.mlb.com/sporty-videos?playId={}"

# ── COCO skeleton for phase grid overlay ────────────────────────────────────
SKELETON = [
    ("l_shoulder", "r_shoulder"), ("l_hip", "r_hip"),
    ("l_shoulder", "l_hip"),      ("r_shoulder", "r_hip"),
    ("l_shoulder", "l_elbow"),    ("l_elbow", "l_wrist"),
    ("r_shoulder", "r_elbow"),    ("r_elbow", "r_wrist"),
    ("l_hip", "l_knee"),          ("l_knee", "l_ankle"),
    ("r_hip", "r_knee"),          ("r_knee", "r_ankle"),
    ("nose", "l_shoulder"),       ("nose", "r_shoulder"),
]

PHASE_LABELS = {
    "coming_set":     "COMING SET",
    "leg_lift_start": "LIFT START",
    "leg_lift_peak":  "LIFT PEAK",
    "foot_plant":     "FOOT PLANT",
    "release":        "RELEASE",
}
PHASE_ORDER = list(PHASE_LABELS.keys())

PHASE_COLOR = {   # BGR
    "coming_set":     (180, 180, 60),
    "leg_lift_start": (60, 200, 60),
    "leg_lift_peak":  (0, 200, 255),
    "foot_plant":     (0, 100, 255),
    "release":        (0, 0, 240),
}


# ── Statcast / video fetch helpers ──────────────────────────────────────────

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
        for chunk in vr.iter_content(chunk_size=8192): f.write(chunk)
    return True, None


# ── Phase grid builder ───────────────────────────────────────────────────────

def _draw_skeleton_on_crop(crop, lms, offset_x, offset_y, color):
    def pt(n):
        return (int(lms[n][0]) - offset_x, int(lms[n][1]) - offset_y)
    for a, b in SKELETON:
        if a in lms and b in lms:
            cv2.line(crop, pt(a), pt(b), color, 2)
    for n in lms:
        cv2.circle(crop, pt(n), 4, (255, 255, 255), -1)
        cv2.circle(crop, pt(n), 4, color, 1)


def build_phase_grid(frames_dir, tag, out_path):
    """
    Reads individual phase PNGs saved by d4_pipeline._save_phase_frames and
    stitches them into a single labeled horizontal grid.
    """
    panels = []
    TARGET_H = 320

    for key in PHASE_ORDER:
        path = os.path.join(frames_dir, f"{tag}_{key}.png")
        if not os.path.exists(path):
            continue
        img = cv2.imread(path)
        if img is None:
            continue
        # Resize to TARGET_H, maintain aspect
        scale = TARGET_H / img.shape[0]
        img = cv2.resize(img, (max(1, int(img.shape[1] * scale)), TARGET_H),
                         interpolation=cv2.INTER_CUBIC)
        # Add label bar at top
        bar_h = 32
        bar = np.zeros((bar_h, img.shape[1], 3), dtype=np.uint8)
        col = PHASE_COLOR.get(key, (200, 200, 200))
        bar[:] = col
        label = PHASE_LABELS.get(key, key.upper())
        cv2.putText(bar, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 3)
        cv2.putText(bar, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1)
        panels.append(np.vstack([bar, img]))

    if not panels:
        print("  No phase panels found for grid.")
        return

    # Pad all panels to same width before hstacking
    max_h = max(p.shape[0] for p in panels)
    padded = []
    for p in panels:
        if p.shape[0] < max_h:
            pad = np.zeros((max_h - p.shape[0], p.shape[1], 3), dtype=np.uint8)
            p = np.vstack([p, pad])
        padded.append(p)

    # 2-px white separator between panels
    sep = np.full((max_h, 2, 3), 255, dtype=np.uint8)
    grid_parts = []
    for i, p in enumerate(padded):
        grid_parts.append(p)
        if i < len(padded) - 1:
            grid_parts.append(sep)

    grid = np.hstack(grid_parts)
    cv2.imwrite(out_path, grid)
    print(f"  Phase grid saved → {out_path}")


# ── Statistics helpers ───────────────────────────────────────────────────────

def iqr(vals):
    arr = np.array([v for v in vals if v is not None], dtype=float)
    if len(arr) < 2:
        return float("nan")
    return float(np.percentile(arr, 75) - np.percentile(arr, 25))


def median(vals):
    arr = np.array([v for v in vals if v is not None], dtype=float)
    if len(arr) == 0:
        return float("nan")
    return float(np.median(arr))


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(VIDEO_DIR, exist_ok=True)

    print(f"Fetching {PITCHER_NAME} (id={PITCHER_ID}) Statcast rows…", flush=True)
    url = STATCAST_URL.format(player_id=PITCHER_ID, limit=MAX_CANDIDATES)
    r   = requests.get(url, timeout=30, headers=HEADERS)
    r.raise_for_status()
    rows = list(csv.DictReader(io.StringIO(r.content.decode("utf-8-sig"))))
    print(f"  {len(rows)} candidate rows")

    if not rows:
        print("No rows returned — check player_id or date range.")
        return

    print("Loading models…", flush=True)
    models = load_models()

    results        = []
    all_timings    = []
    grid_done      = False
    candidates_tried = 0

    for row in rows:
        if len(results) >= TARGET_PITCHES:
            break
        candidates_tried += 1

        game_pk   = row.get("game_pk", "").strip()
        ab_num    = row.get("at_bat_number", "").strip()
        pitch_num = row.get("pitch_number", "").strip()
        pitch_type= row.get("pitch_type", "??").strip()
        if not all([game_pk, ab_num, pitch_num]):
            continue

        tag       = f"{PITCHER_NAME}_{game_pk}_{ab_num}_{pitch_num}"
        vid_path  = os.path.join(VIDEO_DIR, f"{tag}.mp4")

        print(f"\n[{len(results)+1}/{TARGET_PITCHES}] {pitch_type} "
              f"gm={game_pk} ab={ab_num} p={pitch_num}", flush=True)

        # Download video
        play_id = get_play_id(game_pk, ab_num, pitch_num)
        if not play_id:
            print("  no play_id, skip")
            continue
        ok_dl, err_dl = download_video(play_id, vid_path)
        if not ok_dl:
            print(f"  download failed: {err_dl}")
            continue
        sz = os.path.getsize(vid_path)
        if sz < 100_000:
            print(f"  tiny file ({sz}B), skip")
            os.remove(vid_path)
            continue
        print(f"  {sz//1024} KB downloaded", flush=True)

        # Decide whether to keep video for grid generation
        need_grid   = not grid_done
        lm_dir_here = LM_DIR if need_grid else None

        t0  = time.time()
        res = process_one(vid_path, models, tag,
                          keep_video=need_grid,
                          landmarks_dir=lm_dir_here,
                          throws=row.get("p_throws", "R"))
        elapsed = time.time() - t0

        ok     = res.get("ok", False)
        timings= res.get("timings", {})
        pt     = res.get("phase_times", {})
        print(f"  ok={ok}  method={res.get('method')}  "
              f"pose_hit={res.get('pose_hit')}  {elapsed:.1f}s", flush=True)

        if not ok:
            print(f"  reason: {res.get('error')}")
            # Cleanup video if it wasn't already removed
            if os.path.exists(vid_path):
                os.remove(vid_path)
            continue

        # Build phase grid from first successful pitch
        if need_grid:
            frames_dir = os.path.join(LM_DIR, "frames")
            build_phase_grid(frames_dir, tag, GRID_PATH)
            grid_done = True
            if os.path.exists(vid_path):
                os.remove(vid_path)

        # Collect metrics
        def safe(k): return pt.get(k)
        p2r = None
        if safe("foot_plant") is not None and safe("release") is not None:
            p2r = safe("release") - safe("foot_plant")
        delivery_s = None
        if safe("leg_lift_start") is not None and safe("release") is not None:
            delivery_s = safe("release") - safe("leg_lift_start")
        peak_to_plant = None
        if safe("leg_lift_peak") is not None and safe("foot_plant") is not None:
            peak_to_plant = safe("foot_plant") - safe("leg_lift_peak")

        entry = {
            "tag":                  tag,
            "pitch_type":           pitch_type,
            "p_throws":             row.get("p_throws", "?"),
            "plant_t":              safe("foot_plant"),
            "release_t":            safe("release"),
            "p2r":                  p2r,
            "delivery_s":           delivery_s,
            "peak_to_plant":        peak_to_plant,
            "stride_dir":           res.get("stride_direction"),
            "stride_norm":          res.get("stride_norm"),
            "stride_norm_mirrored": res.get("stride_norm_mirrored"),
            "rubber_pos":           res.get("rubber_position"),
            "rubber_method":        res.get("rubber_method"),
            "wall_s":               round(elapsed, 1),
        }
        results.append(entry)
        all_timings.append(timings)

        # Print individual result
        def _fmt(v, decimals=3):
            return f"{v:.{decimals}f}" if v is not None else "None"
        print(f"  p_throws={entry['p_throws']}  "
              f"stride_norm={_fmt(entry['stride_norm'])}  "
              f"stride_norm_mirrored={_fmt(entry['stride_norm_mirrored'])}  "
              f"stride_dir={entry['stride_dir']}  "
              f"rubber_pos={_fmt(entry['rubber_pos'])}  "
              f"rubber_method={entry['rubber_method']}", flush=True)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"RESULTS: {len(results)}/{candidates_tried} pitches succeeded")
    print("="*60)

    if not results:
        print("No successful pitches — cannot aggregate.")
        return

    METRICS = ["plant_t", "release_t", "p2r", "delivery_s",
               "peak_to_plant", "stride_norm", "rubber_pos"]
    print(f"\n{'Metric':<18}  {'Median':>9}  {'IQR':>9}  {'N':>4}")
    print("-" * 46)
    for m in METRICS:
        vals = [r.get(m) for r in results]
        med  = median(vals)
        iq   = iqr(vals)
        n    = sum(1 for v in vals if v is not None)
        print(f"  {m:<16}  {med:>9.3f}  {iq:>9.3f}  {n:>4}")

    # ── Timing breakdown ──────────────────────────────────────────────────────
    print("\n── Per-section timings (median across pitches, seconds) ──")
    all_keys = sorted({k for t in all_timings for k in t})
    timing_rows = []
    for k in all_keys:
        vals = [t[k] for t in all_timings if k in t]
        timing_rows.append((k, median(vals), len(vals)))
    timing_rows.sort(key=lambda x: -x[1])  # slowest first
    total_med = sum(v for _, v, _ in timing_rows if not np.isnan(v))
    for k, med, n in timing_rows:
        pct = 100 * med / total_med if total_med > 0 else 0
        print(f"  {k:<35}  {med:>6.2f}s  ({pct:>4.1f}%)  n={n}")
    print(f"  {'TOTAL (sum of medians)':<35}  {total_med:>6.2f}s")
    wall_med = median([r["wall_s"] for r in results])
    print(f"  {'WALL TIME median':<35}  {wall_med:>6.1f}s")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    with open(RESULTS_JSON, "w") as f:
        json.dump({"results": results, "timings_per_pitch": all_timings}, f, indent=2)
    print(f"\nResults saved → {RESULTS_JSON}")
    print(f"Phase grid    → {GRID_PATH}")


if __name__ == "__main__":
    main()
