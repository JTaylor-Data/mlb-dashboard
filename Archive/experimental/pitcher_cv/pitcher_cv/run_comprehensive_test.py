"""
Comprehensive 3-pitcher deception_v4 pipeline test.

Pitchers:
  1. Tanner Houck      (657006, RHP, starter,  BOS / Fenway Park)
  2. Andres Munoz      (669474, RHP, reliever,  SEA / T-Mobile Park)
  3. Freddy Peralta    (642232, RHP, starter,  MIL / American Family Field)

Rubber anchor (confirmed from d4_pipeline.py line ~178):
  anchor_t = times_yolo.get("leg_lift_peak") or times_yolo.get("coming_set")
  => leg_lift_peak is PRIMARY; coming_set is FALLBACK.

_stride_norm_mirrored: returns raw (no negation), confirmed in d4_phases.py.
"""

import csv, io, json, os, sys, time, shutil
import cv2, numpy as np, requests
from bs4 import BeautifulSoup

SCRATCH = os.path.dirname(os.path.abspath(__file__))
D4      = os.path.join(SCRATCH, "deception_v4")
sys.path.insert(0, D4)

from d4_extract  import load_models
from d4_pipeline import process_one

# ── Configuration ────────────────────────────────────────────────────────────

PITCHERS = [
    {
        "name":           "munoz",
        "display":        "Andres Munoz (RHP, reliever)",
        "player_id":      662253,   # confirmed: Andrés Muñoz, SEA closer
        "throws":         "R",
        "park_id":        "SEA",   # T-Mobile Park
        "fastball_types": ["FF", "SI", "FC"],
        "role":           "reliever",
    },
    {
        "name":           "kirby",
        "display":        "George Kirby (RHP, starter)",
        "player_id":      669923,   # confirmed: George Kirby, SEA starter
        "throws":         "R",
        "park_id":        "SEA",   # T-Mobile Park
        "fastball_types": ["FF", "SI", "FC"],
        "role":           "starter",
    },
    {
        "name":           "valdez",
        "display":        "Framber Valdez (LHP, starter)",
        "player_id":      664285,   # confirmed: Framber Valdez, HOU starter
        "throws":         "L",
        "park_id":        "HOU",   # Daikin Park (formerly Minute Maid)
        "fastball_types": ["FF", "SI", "FC"],
        "role":           "starter",
    },
]

MAX_CANDIDATES  = 15   # hard budget per pitcher, no replacement
FETCH_LIMIT     = 500  # how many Statcast rows to fetch before filtering

OUT_BASE        = os.path.join(SCRATCH, "out_comprehensive_test")

HEADERS         = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0"}
VIDEO_HEADERS   = {**HEADERS, "Referer": "https://baseballsavant.mlb.com/"}

STATCAST_URL    = (
    "https://baseballsavant.mlb.com/statcast_search/csv"
    "?all=true&type=details&player_type=pitcher"
    "&pitchers_lookup%5B%5D={player_id}"
    "&game_date_gt=2024-03-01&game_date_lt=2025-11-01"
    "&sort_col=pitches&sort_order=desc&limit={limit}"
)
GAME_FEED_URL    = "https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
SAVANT_VIDEO_URL = "https://baseballsavant.mlb.com/sporty-videos?playId={}"

# COCO-style skeleton used for frame overlay
SKELETON = [
    ("l_shoulder", "r_shoulder"), ("l_hip", "r_hip"),
    ("l_shoulder", "l_hip"),      ("r_shoulder", "r_hip"),
    ("l_shoulder", "l_elbow"),    ("l_elbow", "l_wrist"),
    ("r_shoulder", "r_elbow"),    ("r_elbow", "r_wrist"),
    ("l_hip", "l_knee"),          ("l_knee", "l_ankle"),
    ("r_hip", "r_knee"),          ("r_knee", "r_ankle"),
    ("nose", "l_shoulder"),       ("nose", "r_shoulder"),
]

PHASE_ORDER = ["coming_set", "leg_lift_start", "leg_lift_peak", "foot_plant", "release"]

# ── Network helpers ──────────────────────────────────────────────────────────

def get_play_id(game_pk, ab_num, pitch_num):
    try:
        r = requests.get(GAME_FEED_URL.format(game_pk=game_pk), timeout=30, headers=HEADERS)
        r.raise_for_status()
        data = r.json()
        target_ab = int(ab_num) - 1
        for play in data.get("liveData", {}).get("plays", {}).get("allPlays", []):
            if play.get("atBatIndex") == target_ab:
                for evt in play.get("playEvents", []):
                    if evt.get("isPitch") and evt.get("pitchNumber") == int(pitch_num):
                        return evt.get("playId")
    except Exception as e:
        return None
    return None


def download_video(play_id, out_path):
    try:
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
    except Exception as e:
        return False, str(e)


# ── Frame-by-frame audit ─────────────────────────────────────────────────────

def save_frame_audit(video_path, data, phase_times_dict, pitch_out_dir):
    """
    For every tracked frame in data["frames"], seek to that frame in the
    video, draw crop_box + skeleton + phase label, and save as PNG.
    """
    frames_list = data.get("frames", [])
    if not frames_list:
        print("    [audit] no frames in data, skipping", flush=True)
        return 0

    os.makedirs(pitch_out_dir, exist_ok=True)

    # Build a lookup: for each frame, which phase is nearest?
    phase_t_items = [(k, v) for k, v in phase_times_dict.items() if v is not None]

    cap = cv2.VideoCapture(video_path)
    saved = 0

    for i, rec in enumerate(frames_list):
        frame_idx = rec.get("frame")
        if frame_idx is None:
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, full = cap.read()
        if not ok:
            continue

        t_frame = rec.get("t", 0.0)
        crop_box = rec.get("crop_box")
        lms = rec.get("landmarks") or {}

        # Draw crop_box
        if crop_box:
            x0, y0, x1, y1 = [int(v) for v in crop_box]
            cv2.rectangle(full, (x0, y0), (x1, y1), (0, 255, 0), 2)

        # Draw skeleton
        def pt(n):
            return (int(lms[n][0]), int(lms[n][1]))
        for a, b in SKELETON:
            if a in lms and b in lms:
                cv2.line(full, pt(a), pt(b), (0, 230, 255), 2)
        for n in lms:
            cv2.circle(full, pt(n), 4, (0, 0, 255), -1)
            cv2.circle(full, pt(n), 3, (255, 255, 255), 1)

        # Nearest phase label
        nearest_phase = ""
        if phase_t_items:
            dists = [(abs(t_frame - tv), k) for k, tv in phase_t_items]
            dists.sort()
            nearest_phase = dists[0][1].upper().replace("_", " ")

        # Overlay text
        label_t = f"t={t_frame:.3f}s"
        cv2.putText(full, label_t, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(full, label_t, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        if nearest_phase:
            cv2.putText(full, nearest_phase, (6, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
            cv2.putText(full, nearest_phase, (6, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        out_png = os.path.join(pitch_out_dir, f"frame_{i:04d}.png")
        cv2.imwrite(out_png, full)
        saved += 1

    cap.release()
    print(f"    [audit] saved {saved} frame PNGs → {pitch_out_dir}", flush=True)
    return saved


# ── Stabilization table ──────────────────────────────────────────────────────

def stabilization_table(pitcher_name, results):
    """Print running median+IQR at n=3,5,7,10,final."""
    metrics = ["plant_t", "release_t", "p2r", "delivery_s", "peak_to_plant", "stride_norm"]
    ok_results = [r for r in results if r.get("ok")]
    if not ok_results:
        print(f"  No successful pitches for {pitcher_name}")
        return

    print(f"\nPitcher: {pitcher_name}")
    for metric in metrics:
        vals = [r.get(metric) for r in ok_results]
        checkpoints = [n for n in [3, 5, 7, 10] if n <= len(vals)]
        checkpoints_plus_final = list(dict.fromkeys(checkpoints + [len(vals)]))
        print(f"  Metric: {metric}")
        for n in checkpoints_plus_final:
            subset = [v for v in vals[:n] if v is not None]
            if not subset:
                print(f"    n={n}: no valid values")
                continue
            arr = np.array(subset, dtype=float)
            med = float(np.median(arr))
            if len(arr) >= 2:
                iqr_val = float(np.percentile(arr, 75) - np.percentile(arr, 25))
            else:
                iqr_val = float("nan")
            label = f"n={n}" if n < len(vals) else "final"
            print(f"    {label}: median={med:.3f}  IQR={iqr_val:.3f}")


# ── Statistics helpers ───────────────────────────────────────────────────────

def _median(vals):
    arr = np.array([v for v in vals if v is not None], dtype=float)
    return float(np.median(arr)) if len(arr) > 0 else float("nan")

def _iqr(vals):
    arr = np.array([v for v in vals if v is not None], dtype=float)
    if len(arr) < 2: return float("nan")
    return float(np.percentile(arr, 75) - np.percentile(arr, 25))


# ── Per-pitcher processing ───────────────────────────────────────────────────

def process_pitcher(pitcher_cfg, models):
    name       = pitcher_cfg["name"]
    player_id  = pitcher_cfg["player_id"]
    park_id    = pitcher_cfg["park_id"]
    fb_types   = pitcher_cfg["fastball_types"]
    throws     = pitcher_cfg["throws"]

    pitcher_out = os.path.join(OUT_BASE, name)
    lm_dir      = os.path.join(pitcher_out, "landmarks")
    video_dir   = os.path.join(pitcher_out, "videos")
    os.makedirs(pitcher_out, exist_ok=True)
    os.makedirs(lm_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)

    print(f"\n{'='*60}", flush=True)
    print(f"PITCHER: {pitcher_cfg['display']}", flush=True)
    print(f"  park_id={park_id}  fastball_types={fb_types}", flush=True)
    print(f"  Rubber anchor: leg_lift_peak (primary) or coming_set (fallback)", flush=True)

    # Fetch Statcast rows
    print(f"  Fetching Statcast CSV (player_id={player_id}) …", flush=True)
    url = STATCAST_URL.format(player_id=player_id, limit=FETCH_LIMIT)
    try:
        r = requests.get(url, timeout=60, headers=HEADERS)
        r.raise_for_status()
        all_rows = list(csv.DictReader(io.StringIO(r.content.decode("utf-8-sig"))))
    except Exception as e:
        print(f"  ERROR fetching Statcast CSV: {e}", flush=True)
        return {"pitcher": name, "results": [], "error": str(e)}

    print(f"  Total rows fetched: {len(all_rows)}", flush=True)

    # Filter: fastball type AND home_team
    candidates = [
        row for row in all_rows
        if row.get("pitch_type", "").strip() in fb_types
        and row.get("home_team", "").strip().upper() == park_id.upper()
    ]
    print(f"  Matching candidates (FB at {park_id}): {len(candidates)}", flush=True)

    # Hard budget: first MAX_CANDIDATES only
    candidates = candidates[:MAX_CANDIDATES]
    print(f"  Will attempt: {len(candidates)} pitches", flush=True)

    results = []

    for n, row in enumerate(candidates):
        game_pk   = row.get("game_pk", "").strip()
        ab_num    = row.get("at_bat_number", "").strip()
        pitch_num = row.get("pitch_number", "").strip()
        pitch_type= row.get("pitch_type", "??").strip()
        p_throws  = row.get("p_throws", throws).strip() or throws

        if not all([game_pk, ab_num, pitch_num]):
            results.append({"n": n+1, "ok": False, "error": "missing game/ab/pitch keys"})
            continue

        tag      = f"{name}_{game_pk}_{ab_num}_{pitch_num}"
        vid_path = os.path.join(video_dir, f"{tag}.mp4")
        pitch_frame_dir = os.path.join(pitcher_out, f"pitch_{n+1:02d}")

        print(f"\n  [{n+1}/{len(candidates)}] {pitch_type} "
              f"gm={game_pk} ab={ab_num} p={pitch_num}  tag={tag}", flush=True)

        # Get play_id
        play_id = get_play_id(game_pk, ab_num, pitch_num)
        if not play_id:
            print(f"    no play_id, skip", flush=True)
            results.append({"n": n+1, "tag": tag, "ok": False, "error": "no play_id"})
            continue

        # Download video
        ok_dl, err_dl = download_video(play_id, vid_path)
        if not ok_dl:
            print(f"    download failed: {err_dl}", flush=True)
            results.append({"n": n+1, "tag": tag, "ok": False, "error": f"download: {err_dl}"})
            continue

        sz = os.path.getsize(vid_path)
        if sz < 100_000:
            print(f"    tiny file ({sz}B), skip", flush=True)
            os.remove(vid_path)
            results.append({"n": n+1, "tag": tag, "ok": False, "error": f"tiny file {sz}B"})
            continue
        print(f"    {sz//1024} KB downloaded", flush=True)

        # Run pipeline — keep_video=True so we can do frame audit
        t0  = time.time()
        res = process_one(vid_path, models, tag,
                          keep_video=True,
                          landmarks_dir=lm_dir,
                          throws=p_throws)
        elapsed = time.time() - t0

        ok     = res.get("ok", False)
        pt     = res.get("phase_times", {})
        print(f"    ok={ok}  method={res.get('method')}  "
              f"pose_hit={res.get('pose_hit')}  {elapsed:.1f}s", flush=True)

        if not ok:
            print(f"    reason: {res.get('error')}", flush=True)

        # Frame-by-frame audit (regardless of ok, if video is on disk)
        frame_count = 0
        if os.path.exists(vid_path):
            # Load YOLO landmark data from the process_one result
            yolo_json = os.path.join(lm_dir, f"{tag}_yolo.json")
            audit_data = None
            if os.path.exists(yolo_json):
                with open(yolo_json) as f:
                    audit_data = json.load(f)
            elif ok:
                # process_one only saves landmarks_dir when ok AND landmarks_dir set
                pass

            if audit_data is not None:
                frame_count = save_frame_audit(vid_path, audit_data, pt, pitch_frame_dir)
            else:
                print(f"    [audit] no YOLO json found at {yolo_json}", flush=True)

            # Copy phase_grid if it was saved
            frames_dir_lm = os.path.join(lm_dir, "frames")
            phase_grid_src = None
            for ext_tag in [tag]:
                # The pipeline saves individual phase PNGs named {tag}_{phase}.png
                # build_phase_grid from run_crossfire_test.py is a separate step
                # Here we just note what was saved
                pass

            # Delete video after audit
            os.remove(vid_path)
            print(f"    video deleted after audit", flush=True)

        # Derive compound metrics
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
            "n":              n + 1,
            "tag":            tag,
            "ok":             ok,
            "pitch_type":     pitch_type,
            "p_throws":       p_throws,
            "method":         res.get("method"),
            "error":          res.get("error"),
            "plant_t":        safe("foot_plant"),
            "release_t":      safe("release"),
            "p2r":            p2r,
            "delivery_s":     delivery_s,
            "peak_to_plant":  peak_to_plant,
            "stride_norm":    res.get("stride_norm"),
            "stride_norm_mirrored": res.get("stride_norm_mirrored"),
            "rubber_pos":     res.get("rubber_position"),
            "rubber_method":  res.get("rubber_method"),
            "wall_s":         round(elapsed, 1),
            "frame_pngs_saved": frame_count,
        }
        results.append(entry)

        def _fmt(v, d=3):
            return f"{v:.{d}f}" if isinstance(v, float) else str(v)
        print(f"    stride_norm={_fmt(entry['stride_norm'])}  "
              f"rubber_pos={_fmt(entry['rubber_pos'])}  "
              f"rubber_method={entry['rubber_method']}  "
              f"frame_pngs={frame_count}", flush=True)

    # Save per-pitcher results.json
    out_json = os.path.join(pitcher_out, "results.json")
    with open(out_json, "w") as f:
        json.dump({"pitcher": pitcher_cfg["display"], "results": results}, f, indent=2)
    print(f"\n  results.json → {out_json}", flush=True)

    # Print stabilization table
    stabilization_table(pitcher_cfg["display"], results)

    n_ok  = sum(1 for r in results if r.get("ok"))
    n_att = len(results)
    print(f"\n  Summary: {n_ok}/{n_att} pitches succeeded "
          f"({100*n_ok//n_att if n_att else 0}%)", flush=True)

    return {"pitcher": pitcher_cfg["display"], "results": results}


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60, flush=True)
    print("Comprehensive deception_v4 pipeline test", flush=True)
    print("Rubber anchor: leg_lift_peak (primary) | coming_set (fallback)", flush=True)
    print("_stride_norm_mirrored: raw (no negation)", flush=True)
    print("=" * 60, flush=True)

    os.makedirs(OUT_BASE, exist_ok=True)

    print("\nLoading models…", flush=True)
    models = load_models()
    print("Models loaded.", flush=True)

    all_pitcher_results = []

    for pitcher_cfg in PITCHERS:
        pr = process_pitcher(pitcher_cfg, models)
        all_pitcher_results.append(pr)

    # Save combined summary
    summary_path = os.path.join(OUT_BASE, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_pitcher_results, f, indent=2)
    print(f"\nSummary saved → {summary_path}", flush=True)

    # Final report
    print("\n" + "=" * 60, flush=True)
    print("FINAL SUMMARY", flush=True)
    print("=" * 60, flush=True)
    for pr in all_pitcher_results:
        results = pr.get("results", [])
        n_ok  = sum(1 for r in results if r.get("ok"))
        n_att = len(results)
        pct   = f"{100*n_ok//n_att}%" if n_att else "N/A"
        print(f"  {pr['pitcher']}: {n_ok}/{n_att} ({pct})", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
