"""
Test rubber detection v2 on a single Chris Sale pitch.
Fetches one video, extracts coming_set frame, runs detect_rubber, saves annotated image.
"""
import csv, io, json, os, sys, time
import cv2, numpy as np, requests
from bs4 import BeautifulSoup

SCRATCH = os.path.dirname(os.path.abspath(__file__))
D4      = os.path.join(SCRATCH, "deception_v4")
sys.path.insert(0, D4)

from d4_extract import load_models, extract_yolo
from d4_phases  import Signals, detect_phases, phase_times
from d4_rubber  import detect_rubber

PITCHER_NAME = "sale"
PITCHER_ID   = 519242   # Chris Sale, LHP

OUT_DIR    = os.path.join(SCRATCH, "deception_v4", "out_rubber_test")
VIDEO_PATH = os.path.join(OUT_DIR, "sale_test.mp4")

HEADERS       = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0"}
VIDEO_HEADERS = {**HEADERS, "Referer": "https://baseballsavant.mlb.com/"}
STATCAST_URL  = (
    "https://baseballsavant.mlb.com/statcast_search/csv"
    "?all=true&type=details&player_type=pitcher"
    "&pitchers_lookup%5B%5D={player_id}"
    "&game_date_gt=2025-04-01&game_date_lt=2025-08-01"
    "&pitch_type=FF"
    "&sort_col=pitches&sort_order=desc&limit=20"
)
GAME_FEED_URL    = "https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
SAVANT_VIDEO_URL = "https://baseballsavant.mlb.com/sporty-videos?playId={}"


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


def annotate_frame(frame_bgr, result, pivot_xy, scan_y_top, scan_y_bot, body_scale):
    """Draw mound-center rubber detection overlay on a full-resolution frame."""
    out = frame_bgr.copy()
    ax, ay = int(pivot_xy[0]), int(pivot_xy[1])
    scan_mid = (scan_y_top + scan_y_bot) // 2

    # Scan band
    cv2.rectangle(out, (0, scan_y_top), (frame_bgr.shape[1], scan_y_bot), (200, 200, 0), 1)

    # Ankle marker
    cv2.circle(out, (ax, ay), 9, (0, 230, 255), -1)
    cv2.circle(out, (ax, ay), 9, (0, 0, 0), 1)

    if result:
        ml  = int(result["mound_left_x"])
        mr  = int(result["mound_right_x"])
        mc  = int(result["mound_center_x"])
        pos = result["position"]

        # Mound left edge (green vertical bar)
        cv2.line(out, (ml, scan_y_top), (ml, scan_y_bot), (0, 255, 60), 3)
        cv2.putText(out, "L", (ml + 4, scan_y_top + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 60), 2)

        # Mound right edge (green vertical bar)
        cv2.line(out, (mr, scan_y_top), (mr, scan_y_bot), (0, 255, 60), 3)
        cv2.putText(out, "R", (mr - 16, scan_y_top + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 60), 2)

        # Mound center (white dashed line)
        for y in range(scan_y_top, scan_y_bot, 8):
            cv2.line(out, (mc, y), (mc, min(y+4, scan_y_bot)), (255, 255, 255), 2)

        # Ankle-to-center offset line
        cv2.line(out, (ax, scan_mid), (mc, scan_mid), (0, 180, 255), 2)
        cv2.arrowedLine(out, (ax, scan_mid), (mc, scan_mid), (0, 180, 255), 2, tipLength=0.15)

        side = "3B SIDE" if pos > 0.10 else "1B SIDE" if pos < -0.10 else "CENTER"  # pos>0 = right of center = 3B
        label = f"pos={pos:+.2f}  ({side})"
        lx = min(ml, ax) - 5
        cv2.putText(out, label, (20, scan_y_top - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(out, label, (20, scan_y_top - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 80), 2)

        wlabel = (f"mound_w={result['mound_width_px']:.0f}px  "
                  f"rubber_half_est={result['rubber_half_w_est']:.0f}px  "
                  f"body_scale={body_scale:.0f}px")
        cv2.putText(out, wlabel, (20, scan_y_bot + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(out, wlabel, (20, scan_y_bot + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 255, 200), 1)
    else:
        cv2.putText(out, "MOUND EDGES NOT FOUND", (10, scan_y_top - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(out, "MOUND EDGES NOT FOUND", (10, scan_y_top - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 255), 2)

    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- fetch Statcast rows ---
    print(f"Fetching Sale (id={PITCHER_ID}) Statcast rows...", flush=True)
    url = STATCAST_URL.format(player_id=PITCHER_ID)
    r   = requests.get(url, timeout=30, headers=HEADERS)
    r.raise_for_status()
    rows = list(csv.DictReader(io.StringIO(r.content.decode("utf-8-sig"))))
    print(f"  {len(rows)} rows")

    video_path = None
    for row in rows[:10]:
        game_pk   = row.get("game_pk","").strip()
        ab_num    = row.get("at_bat_number","").strip()
        pitch_num = row.get("pitch_number","").strip()
        if not all([game_pk, ab_num, pitch_num]): continue
        play_id = get_play_id(game_pk, ab_num, pitch_num)
        if not play_id: continue
        print(f"  play_id={play_id}", flush=True)
        ok, err = download_video(play_id, VIDEO_PATH)
        if not ok:
            print(f"  download failed: {err}")
            continue
        sz = os.path.getsize(VIDEO_PATH)
        if sz < 100_000:
            print(f"  tiny file {sz}B, skip")
            os.remove(VIDEO_PATH)
            continue
        print(f"  {sz//1024}KB downloaded")
        video_path = VIDEO_PATH
        break

    if not video_path:
        print("Could not download any video for Sale")
        return

    # --- run YOLO extraction ---
    print("Loading models...", flush=True)
    models = load_models()
    print("Extracting landmarks...", flush=True)
    data, err = extract_yolo(video_path, models)
    if data is None:
        print(f"Extraction failed: {err}")
        return

    sg = Signals(data)
    ph, _ = detect_phases(sg)
    times = phase_times(sg, ph)
    print(f"  coming_set={times.get('coming_set'):.3f}s" if times.get('coming_set') else "  coming_set=None")

    if ph is None or times.get("coming_set") is None:
        print("No phases detected — using first frame for rubber test")
        cs_t = 0.0
    else:
        cs_t = times["coming_set"]

    # --- find coming_set frame record ---
    frames = data.get("frames", [])
    cs_idx = int(np.argmin([abs(f["t"] - cs_t) for f in frames]))
    cs_rec = frames[cs_idx]

    pivot_side = "r" if (ph and ph.get("lift_side") == "l") else "l"
    pivot_key  = f"{pivot_side}_ankle"
    lms = cs_rec.get("landmarks") or {}

    if pivot_key not in lms:
        print(f"No {pivot_key} landmark in coming_set frame — trying r_ankle")
        pivot_key = "r_ankle" if pivot_key == "l_ankle" else "l_ankle"
    if pivot_key not in lms:
        print("No ankle landmark at all — abort")
        return

    pivot_xy = lms[pivot_key][:2]
    print(f"  pivot ankle ({pivot_side}): {pivot_xy}", flush=True)

    # --- read full frame ---
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, cs_rec["frame"])
    ok_f, full_frame = cap.read()
    cap.release()
    if not ok_f:
        print("Could not read frame from video")
        return

    h, w = full_frame.shape[:2]
    print(f"  full frame: {w}x{h}", flush=True)

    # --- run rubber detection ---
    result = detect_rubber(full_frame, pivot_xy, sg.body_scale)
    print(f"\nRubber detection result: {result}", flush=True)

    # Show scan band used for detection
    from d4_rubber import SCAN_ABOVE_MAX, SCAN_BELOW_MAX
    scan_y_top = max(0, int(pivot_xy[1]) - SCAN_ABOVE_MAX)
    scan_y_bot = min(h, int(pivot_xy[1]) + SCAN_BELOW_MAX)

    # --- annotate + save ---
    annotated = annotate_frame(full_frame, result, pivot_xy,
                               scan_y_top, scan_y_bot, sg.body_scale)

    # Also save a zoomed crop around the rubber area for close inspection
    zoom_y0 = max(0, scan_y_top - 80)
    zoom_y1 = min(h, scan_y_bot + 60)
    zoom_x0 = max(0, int(pivot_xy[0]) - 200)
    zoom_x1 = min(w, int(pivot_xy[0]) + 200)
    zoomed  = annotated[zoom_y0:zoom_y1, zoom_x0:zoom_x1]
    if zoomed.shape[0] > 10 and zoomed.shape[1] > 10:
        zoom_up = cv2.resize(zoomed, (zoomed.shape[1]*3, zoomed.shape[0]*3),
                             interpolation=cv2.INTER_CUBIC)
        zoom_path = os.path.join(OUT_DIR, "sale_rubber_zoom.png")
        cv2.imwrite(zoom_path, zoom_up)
        print(f"Saved zoom -> {zoom_path}")

    full_path = os.path.join(OUT_DIR, "sale_rubber_full.png")
    cv2.imwrite(full_path, annotated)
    print(f"Saved full -> {full_path}")

    # Save debug: the grass mask across the full frame width at scan level
    roi = full_frame[scan_y_top:scan_y_bot, :].astype(float)
    from d4_rubber import CHALK_MIN, CHALK_MAX_SAT
    B2, G2, R2 = roi[:,:,0].astype(float), roi[:,:,1].astype(float), roi[:,:,2].astype(float)
    min_c = np.minimum(np.minimum(R2, G2), B2)
    max_c = np.maximum(np.maximum(R2, G2), B2)
    grass_mask = ((min_c >= CHALK_MIN) & ((max_c - min_c) <= CHALK_MAX_SAT)).astype(np.uint8)*255
    roi = roi.astype(np.uint8)
    gm_big = cv2.resize(grass_mask, (roi.shape[1], roi.shape[0]),
                        interpolation=cv2.INTER_NEAREST)
    debug = np.vstack([roi, cv2.cvtColor(gm_big, cv2.COLOR_GRAY2BGR)])
    debug_big = cv2.resize(debug, (debug.shape[1]*2, debug.shape[0]*4),
                           interpolation=cv2.INTER_NEAREST)
    cv2.putText(debug_big, "original scan band", (5, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
    cv2.putText(debug_big, "GRASS mask (green columns)", (5, roi.shape[0]*4+18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
    dbg_path = os.path.join(OUT_DIR, "sale_rubber_debug.png")
    cv2.imwrite(dbg_path, debug_big)
    print(f"Saved debug masks -> {dbg_path}")

    if result:
        pos = result["position"]
        label = "3B SIDE" if pos > 0.10 else "1B SIDE" if pos < -0.10 else "CENTER"  # pos>0 = right = 3B
        print(f"\n=== RESULT: position={pos:+.2f} ({label}) ===")
        print(f"    mound_width={result['mound_width_px']:.0f}px  rubber_half_est={result['rubber_half_w_est']:.0f}px  body_scale={sg.body_scale:.0f}px")
    else:
        print("\n=== RESULT: NOT DETECTED ===")
        print("    Check sale_rubber_debug.png to see white/dirt masks")

    # Delete video
    os.remove(video_path)
    print("\nVideo deleted.")


if __name__ == "__main__":
    main()
