"""
Deception+ pipeline, EXTRACTION layer - clean rewrite (v4).

YOLO-pose is the PRIMARY (and normally only) pose source now, not MediaPipe -
MediaPipe is kept only as an escalation path (see extract_lite_escalation),
invoked by d4_pipeline.py when the YOLO-only result fails check_results().
The YOLO bbox detector (phc_detector, pitcher identity) is unrelated to pose
model choice and runs in both paths regardless - it is what fixed real
identity-tracking bugs (skeleton locking onto the batter) that no pose model
addresses.

Everything model-heavy runs BATCHED where it can be (bbox detection already
was; pose extraction now is too) and nothing here re-decodes a frame it
already has in memory - a video is read exactly once for the primary pass.
"""
import math
import os

import cv2
import numpy as np
from ultralytics import YOLO

from d4_timing import Stopwatch

SCRATCH_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MP_LITE_MODEL_PATH = os.path.join(SCRATCH_PARENT, "pose_landmarker_lite.task")
YOLO_POSE_PATH = os.path.join(SCRATCH_PARENT, "yolo11m-pose.pt")

YOLO_IMGSZ_DET = 384      # phc_detector, bbox only - 4x faster than 640, same accuracy (measured)
YOLO_IMGSZ_POSE = 640
DET_BATCH = 8
POSE_BATCH = 8
UPSCALE = 2.5             # crop upscale before pose - small pitcher in a wide broadcast frame

CUT_SAMPLE_STRIDE = 3
CUT_ABS_MIN = 12.0
CUT_K = 5.0
MIN_SHOT_S = 1.5
SHOT_PROBE_FRAMES = 8
SHOT_MIN_HIT_RATE = 0.60   # raised from 0.50 — clips below this drifted to wrong person

CLUSTER_DIST_FRAC = 1.5
FINE_STRIDE = 2
ANCHOR_REJECT_FRAC = 0.45
CROP_PAD_FRAC = 0.20
TRACK_MAX_JUMP_FRAC = 0.45
TRACK_SMOOTH = 5

CUTOFF_SAMPLES = 22
CUTOFF_PAD_FRAC = 0.40
CUTOFF_BUFFER_S = 0.50
CUTOFF_MIN_RISE_PX = 30
CUTOFF_MIN_SHOT_FRAC = 0.35

# COCO-17 keypoint order -> this project's landmark names. No toe/foot in
# COCO (unlike MediaPipe's 31/32) - Signals/plant-detection handle a missing
# "foot" key gracefully (see d4_phases.py's has_foot check).
COCO_NAMES = {0: "nose", 5: "l_shoulder", 6: "r_shoulder", 7: "l_elbow", 8: "r_elbow",
              9: "l_wrist", 10: "r_wrist", 11: "l_hip", 12: "r_hip",
              13: "l_knee", 14: "r_knee", 15: "l_ankle", 16: "r_ankle"}

MP_LM_NAMES = {0: "nose", 11: "l_shoulder", 12: "r_shoulder", 13: "l_elbow", 14: "r_elbow",
               15: "l_wrist", 16: "r_wrist", 23: "l_hip", 24: "r_hip",
               25: "l_knee", 26: "r_knee", 27: "l_ankle", 28: "r_ankle",
               31: "l_foot", 32: "r_foot"}


# --------------------------------------------------------------------------
# models - each loaded exactly ONCE by the caller (d4_run.py) and reused
# across every pitch in a batch, never reloaded per-pitch.
# --------------------------------------------------------------------------

def load_models():
    from baseballcv.functions import LoadTools
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    det = YOLO(LoadTools().load_model("phc_detector"))
    pose = YOLO(YOLO_POSE_PATH)
    mp_options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MP_LITE_MODEL_PATH),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=3, min_pose_detection_confidence=0.20,
        min_pose_presence_confidence=0.20, output_segmentation_masks=False,
    )
    mp_lite = mp_vision.PoseLandmarker.create_from_options(mp_options)
    return {"det": det, "pose": pose, "mp_lite": mp_lite, "mp": mp}


# --------------------------------------------------------------------------
# frame reading
# --------------------------------------------------------------------------

def read_frames(video_path, start, end, stride):
    """One seek + sequential decode - repeated cap.set() seeking measured
    much slower on H.264 (re-decodes from the previous keyframe each time)."""
    cap = cv2.VideoCapture(video_path)
    if start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    out, idx = [], start
    while idx <= end:
        ok, frame = cap.read()
        if not ok:
            break
        if (idx - start) % stride == 0:
            out.append((idx, frame))
        idx += 1
    cap.release()
    return out


def video_meta(video_path):
    cap = cv2.VideoCapture(video_path)
    meta = {"fps": cap.get(cv2.CAP_PROP_FPS), "n_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "w": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "h": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    cap.release()
    return meta


# --------------------------------------------------------------------------
# shot segmentation + selection (bbox only, no pose model involved)
# --------------------------------------------------------------------------

def detect_shots(video_path, n_frames, fps):
    sampled = read_frames(video_path, 0, n_frames - 1, CUT_SAMPLE_STRIDE)
    if len(sampled) < 3:
        return [(0, n_frames - 1)]
    smalls = [(idx, cv2.cvtColor(cv2.resize(f, (64, 36)), cv2.COLOR_BGR2GRAY).astype(np.float32))
              for idx, f in sampled]
    diffs = [(i1, float(np.mean(np.abs(a - b)))) for (_, a), (i1, b) in zip(smalls, smalls[1:])]
    if not diffs:
        return [(0, n_frames - 1)]
    med = float(np.median([d for _, d in diffs]))
    thresh = max(CUT_ABS_MIN, CUT_K * med)
    cut_at = [idx for idx, d in diffs if d > thresh]
    bounds, prev = [], 0
    for c in cut_at:
        if c - prev >= 2:
            bounds.append((prev, c - 1))
        prev = c
    bounds.append((prev, n_frames - 1))
    return [b for b in bounds if b[1] > b[0]]


def pitcher_boxes_all(det_model, indexed_frames):
    out = {}
    for i in range(0, len(indexed_frames), DET_BATCH):
        chunk = indexed_frames[i:i + DET_BATCH]
        results = det_model.predict([f for _, f in chunk], verbose=False, imgsz=YOLO_IMGSZ_DET)
        for (idx, _), result in zip(chunk, results):
            dets = []
            for box in result.boxes:
                if det_model.names[int(box.cls[0])].lower() != "pitcher":
                    continue
                dets.append((float(box.conf[0]), [float(v) for v in box.xyxy[0].tolist()]))
            out[idx] = sorted(dets, key=lambda d: -d[0])
    return out


def pitcher_boxes(det_model, indexed_frames):
    return {idx: (dets[0][1] if dets else None)
            for idx, dets in pitcher_boxes_all(det_model, indexed_frames).items()}


def pick_pitch_shot(video_path, shots, det_model, fps):
    diags, qualifying = [], []
    for (s0, s1) in shots:
        dur = (s1 - s0 + 1) / fps
        if dur < MIN_SHOT_S:
            diags.append({"shot": [s0, s1], "verdict": "too short"})
            continue
        stride = max(1, (s1 - s0) // SHOT_PROBE_FRAMES)
        probe = read_frames(video_path, s0, s1, stride)
        boxes = pitcher_boxes(det_model, probe)
        rate = sum(1 for b in boxes.values() if b is not None) / max(1, len(probe))
        ok = rate >= SHOT_MIN_HIT_RATE
        diags.append({"shot": [s0, s1], "hit_rate": round(rate, 2),
                      "verdict": "pitcher shot" if ok else "no consistent pitcher"})
        if ok:
            qualifying.append((s0, s1))
    if not qualifying:
        return None, diags
    return qualifying[0], diags


# --------------------------------------------------------------------------
# YOLO-pose - low-level batched predict, shared by the follow-through probe
# (small, sparse batches) and the main per-frame extraction (larger batches).
# --------------------------------------------------------------------------

def _crop_upscale(frame, box):
    bx0, by0, bx1, by1 = box
    sub = frame[by0:by1, bx0:bx1]
    ch, cw = sub.shape[:2]
    if ch < 4 or cw < 4:
        return None
    return cv2.resize(sub, (int(cw * UPSCALE), int(ch * UPSCALE)), interpolation=cv2.INTER_CUBIC)


def _yolo_pose_predict(pose_model, crop_imgs):
    if not crop_imgs:
        return []
    results = pose_model.predict(crop_imgs, verbose=False, imgsz=YOLO_IMGSZ_POSE)
    return [r.keypoints.data.cpu().numpy() if r.keypoints is not None and len(r.keypoints) else None
            for r in results]


def _select_and_convert(kps_batch, bx0, by0, anchor_xy, max_dist):
    """kps_batch: (n_people, 17, 3) in upscaled-crop-local pixel coords, or
    None. Picks whichever detected person is nearest ANCHOR_XY (original-
    frame coords) and returns named landmarks - or None if nobody is close
    enough (a padded crop can contain a second person; ranking alone is not
    enough when only one candidate is proposed and it's the wrong person)."""
    if kps_batch is None or len(kps_batch) == 0:
        return None
    ax, ay = anchor_xy
    best, best_d = None, None
    for kp in kps_batch:
        cx = bx0 + float(np.mean(kp[:, 0])) / UPSCALE
        cy = by0 + float(np.mean(kp[:, 1])) / UPSCALE
        d = math.hypot(cx - ax, cy - ay)
        if best_d is None or d < best_d:
            best, best_d = kp, d
    if best_d is None or best_d > max_dist:
        return None
    return {name: [round(bx0 + float(best[i][0]) / UPSCALE, 2),
                   round(by0 + float(best[i][1]) / UPSCALE, 2), round(float(best[i][2]), 3)]
            for i, name in COCO_NAMES.items()}


def _first_notable_rise(y, frame_idxs, min_rise_px):
    baseline = float(np.median(y))
    idxs = np.where(y <= baseline - min_rise_px)[0]
    if len(idxs) == 0:
        return None
    return float(frame_idxs[int(idxs[0])])


def find_follow_through_cutoff(video_path, det_model, pose_model, shot, fps):
    """Sparse probe: the PIVOT leg stays planted through the whole delivery
    and only rises once it becomes the free trailing leg AFTER release - its
    first notable rise reliably falls shortly after the pitch is over, so the
    expensive full-per-frame pass below can stop there instead of running to
    the end of the shot (which can include several seconds of dead broadcast
    time). Side-agnostic: whichever leg's first rise comes LATER is trusted
    to be the pivot (the lift leg's own pre-plant kick always rises first)."""
    s0, s1 = shot
    stride = max(1, (s1 - s0) // CUTOFF_SAMPLES)
    sampled = read_frames(video_path, s0, s1, stride)
    if len(sampled) < 6:
        return s1
    boxes = pitcher_boxes(det_model, sampled)

    frames_found, ly, ry = [], [], []
    for idx, frame in sampled:
        b = boxes.get(idx)
        if b is None:
            continue
        bw, bh = b[2] - b[0], b[3] - b[1]
        box = (max(0, int(b[0] - bw * CUTOFF_PAD_FRAC)), max(0, int(b[1] - bh * CUTOFF_PAD_FRAC)),
              int(b[2] + bw * CUTOFF_PAD_FRAC), int(b[3] + bh * CUTOFF_PAD_FRAC))
        center = ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)
        crop = _crop_upscale(frame, box)
        if crop is None:
            continue
        kps = _yolo_pose_predict(pose_model, [crop])[0]
        lm = _select_and_convert(kps, box[0], box[1], center, max(bw, bh) * 1.5)
        if lm is None or "l_knee" not in lm or "r_knee" not in lm:
            continue
        frames_found.append(idx); ly.append(lm["l_knee"][1]); ry.append(lm["r_knee"][1])

    if len(frames_found) < 6:
        return s1
    frames_found = np.array(frames_found, dtype=float)
    lt = _first_notable_rise(np.array(ly), frames_found, CUTOFF_MIN_RISE_PX)
    rt = _first_notable_rise(np.array(ry), frames_found, CUTOFF_MIN_RISE_PX)
    candidates = [t for t in (lt, rt) if t is not None]
    if not candidates:
        return s1
    pivot_frame = max(candidates)
    cutoff = int(pivot_frame + CUTOFF_BUFFER_S * fps)
    min_cutoff = s0 + int(CUTOFF_MIN_SHOT_FRAC * (s1 - s0))
    return int(max(min_cutoff, min(s1, cutoff)))


# --------------------------------------------------------------------------
# per-frame pitcher tracking (pure math, no model call)
# --------------------------------------------------------------------------

def _cluster_by_center(items, dist_frac=CLUSTER_DIST_FRAC):
    if not items:
        return []
    centers = [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for _, b in items]
    sizes = [max(b[2] - b[0], b[3] - b[1]) for _, b in items]
    thresh = dist_frac * float(np.median(sizes))
    clusters = []
    for i, c in enumerate(centers):
        best, best_d = None, None
        for cl in clusters:
            cx = sum(centers[j][0] for j in cl) / len(cl)
            cy = sum(centers[j][1] for j in cl) / len(cl)
            d = math.hypot(c[0] - cx, c[1] - cy)
            if d <= thresh and (best_d is None or d < best_d):
                best, best_d = cl, d
        if best is None:
            clusters.append([i])
        else:
            best.append(i)
    clusters.sort(key=len, reverse=True)
    return [[items[i] for i in cl] for cl in clusters]


def _median_smooth(a, win):
    if win < 3 or len(a) < win:
        return a
    pad = win // 2
    p = np.pad(a, (pad, pad), mode="edge")
    return np.array([np.median(p[i:i + win]) for i in range(len(a))])


def track_pitcher(dets_by_frame, order):
    """Identity held by continuity (seed on the dominant early cluster, then
    nearest-to-last-accepted each step) - a single locked box + sparse
    anchor was confirmed to lose the pitcher exactly at release on fast
    motion. Crop SIZE is fixed for the whole clip (median of detections);
    only its CENTRE tracks, median-smoothed to strip detector jitter."""
    sizes = [(b[2] - b[0], b[3] - b[1]) for f in order for _, b in dets_by_frame.get(f, [])]
    if not sizes:
        return None, None, {"error": "no pitcher detections in shot"}
    med_w = float(np.median([s[0] for s in sizes]))
    med_h = float(np.median([s[1] for s in sizes]))

    seed_frames = order[:max(4, len(order) // 4)]
    seed_items = [(f, b) for f in seed_frames for _, b in dets_by_frame.get(f, [])]
    clusters = _cluster_by_center(seed_items)
    if not clusters:
        return None, None, {"error": "no detections to seed the tracker"}
    seed = clusters[0]
    cur = (float(np.median([(b[0] + b[2]) / 2 for _, b in seed])),
           float(np.median([(b[1] + b[3]) / 2 for _, b in seed])))

    max_jump = TRACK_MAX_JUMP_FRAC * med_h
    centers, n_tracked = {}, 0
    for f in order:
        best, best_d = None, None
        for _, b in dets_by_frame.get(f, []):
            c = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
            d = math.hypot(c[0] - cur[0], c[1] - cur[1])
            if d <= max_jump and (best_d is None or d < best_d):
                best, best_d = c, d
        if best is not None:
            cur = best; centers[f] = best; n_tracked += 1

    if n_tracked < 8:
        return None, None, {"error": f"pitcher track too sparse ({n_tracked} frames)"}
    known = sorted(centers)
    xs = np.interp(order, known, [centers[f][0] for f in known])
    ys = np.interp(order, known, [centers[f][1] for f in known])
    xs, ys = _median_smooth(xs, TRACK_SMOOTH), _median_smooth(ys, TRACK_SMOOTH)
    # sized off the LARGEST detected extent (p95), not the median - the
    # throwing arm extends well outside the standing footprint at release.
    big_w = float(np.percentile([s[0] for s in sizes], 95))
    big_h = float(np.percentile([s[1] for s in sizes], 95))
    return ({f: (float(xs[i]), float(ys[i])) for i, f in enumerate(order)},
            (med_w, med_h, big_w, big_h), {"tracked": n_tracked})


# --------------------------------------------------------------------------
# main per-frame pose extraction - YOLO, batched, single pass
# --------------------------------------------------------------------------

def pose_extraction_yolo(indexed, centers, bw, bh, w, h, med_h, pose_model, fps, batch_size=POSE_BATCH):
    max_dist = ANCHOR_REJECT_FRAC * med_h
    boxes = {}
    for idx, _ in indexed:
        c = centers[idx]
        x0 = min(max(0.0, c[0] - bw / 2), w - bw)
        y0 = min(max(0.0, c[1] - bh / 2), h - bh)
        boxes[idx] = (int(x0), int(y0), int(x0 + bw), int(y0 + bh))

    frames_out = []
    for i in range(0, len(indexed), batch_size):
        chunk = indexed[i:i + batch_size]
        crops, slots = [], []
        for idx, frame in chunk:
            crop = _crop_upscale(frame, boxes[idx])
            slots.append((idx, len(crops) if crop is not None else None))
            if crop is not None:
                crops.append(crop)
        preds = _yolo_pose_predict(pose_model, crops)
        for idx, ci in slots:
            box = boxes[idx]
            lm = _select_and_convert(preds[ci], box[0], box[1], centers[idx], max_dist) if ci is not None else None
            frames_out.append({"frame": idx, "t": round(idx / fps, 4), "landmarks": lm,
                               "crop_box": list(box), "anchor": [round(centers[idx][0], 1), round(centers[idx][1], 1)]})
    return frames_out


def extract_yolo(video_path, models):
    """Full YOLO-primary extraction for one clip. Returns (data, err). One
    video decode pass for the main window (read_frames), one batched bbox
    pass, one batched pose pass - nothing here re-reads a frame it already
    has, and no localization window narrower than the whole selected shot is
    used (an earlier version's 'guess the leg-lift peak, window around it'
    step was itself the single biggest source of wrong results)."""
    sw = Stopwatch()
    meta = video_meta(video_path)
    fps, n_frames, w, h = meta["fps"], meta["n_frames"], meta["w"], meta["h"]
    if not fps or n_frames < 10:
        return None, "unreadable or too-short video"
    sw.lap("video_meta")

    # Fast motion scan — no ML, narrows the fine-stride window before YOLO runs.
    from d4_crop import find_delivery_window
    crop_s0, crop_s1 = find_delivery_window(video_path)
    sw.lap("delivery_crop")

    shots = detect_shots(video_path, n_frames, fps)
    sw.lap("shot_segmentation")
    shot, shot_diags = pick_pitch_shot(video_path, shots, models["det"], fps)
    sw.lap("shot_selection_probe")
    if shot is None:
        return None, f"no shot contains a consistently-detected pitcher (shots={shot_diags})"

    s0, s1_full = shot
    s1 = find_follow_through_cutoff(video_path, models["det"], models["pose"], (s0, s1_full), fps)
    sw.lap("followthrough_cutoff_probe")

    # Clamp fine-stride window to the motion-detected delivery window.
    # Fall back to the full shot if the crop is too aggressive (< 60 frames).
    s0_fine = max(s0, crop_s0)
    s1_fine = min(s1, crop_s1)
    if s1_fine - s0_fine < 60:
        s0_fine, s1_fine = s0, s1

    indexed = read_frames(video_path, s0_fine, s1_fine, FINE_STRIDE)
    sw.lap("frame_decode")
    if len(indexed) < 12:
        return None, "selected shot too short after striding"

    order = [idx for idx, _ in indexed]
    dets = pitcher_boxes_all(models["det"], indexed)
    sw.lap("pitcher_detection_yolo")
    centers, dims, tdiag = track_pitcher(dets, order)
    sw.lap("pitcher_tracking")
    if centers is None:
        return None, tdiag["error"]
    med_w, med_h, big_w, big_h = dims
    bw = min(w, big_w * (1 + 2 * CROP_PAD_FRAC))
    bh = min(h, big_h * (1 + 2 * CROP_PAD_FRAC))

    frames_out = pose_extraction_yolo(indexed, centers, bw, bh, w, h, med_h, models["pose"], fps)
    sw.lap("pose_extraction_yolo")
    n_hit = sum(1 for f in frames_out if f["landmarks"] is not None)
    if n_hit < 12:
        return None, f"too few frames with a usable pose ({n_hit} of {len(frames_out)})"

    data = {"fps": fps, "n_frames": n_frames, "w": w, "h": h, "frames": frames_out,
            "shot": [s0, s1], "track": tdiag, "crop_size": [round(bw), round(bh)],
            "pose_hit": n_hit, "n_processed": len(frames_out), "stride": FINE_STRIDE,
            "timings": sw.laps}
    return data, None


# --------------------------------------------------------------------------
# MediaPipe LITE - escalation-only path. Reuses the crop_box/anchor already
# tracked by the YOLO pass (no re-detection, re-tracking, or re-shot-
# selection) - only a fresh pose pass over the exact frames already chosen.
# --------------------------------------------------------------------------

def _pose_in_box_mediapipe(pose_detector, mp, frame, box, anchor_xy, max_dist):
    bx0, by0, bx1, by1 = box
    sub = frame[by0:by1, bx0:bx1]
    ch, cw = sub.shape[:2]
    if ch < 4 or cw < 4:
        return None
    up = cv2.resize(sub, (int(cw * UPSCALE), int(ch * UPSCALE)), interpolation=cv2.INTER_CUBIC)
    uh, uw = up.shape[:2]
    result = pose_detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(up, cv2.COLOR_BGR2RGB)))
    if not result.pose_landmarks:
        return None
    ax, ay = anchor_xy
    best, best_d = None, None
    for lm in result.pose_landmarks:
        cx = bx0 + (sum(lm[i].x for i in MP_LM_NAMES) / len(MP_LM_NAMES)) * uw / UPSCALE
        cy = by0 + (sum(lm[i].y for i in MP_LM_NAMES) / len(MP_LM_NAMES)) * uh / UPSCALE
        d = math.hypot(cx - ax, cy - ay)
        if best_d is None or d < best_d:
            best, best_d = lm, d
    if best_d is None or best_d > max_dist:
        return None
    return {name: [round(bx0 + (best[i].x * uw) / UPSCALE, 2), round(by0 + (best[i].y * uh) / UPSCALE, 2),
                   round(best[i].visibility, 3)] for i, name in MP_LM_NAMES.items()}


def extract_lite_escalation(video_path, yolo_data, models):
    """Re-run pose using MediaPipe LITE over the exact same frames the YOLO pass
    already processed. Reads the video SEQUENTIALLY (one cap.set then forward
    reads) instead of seeking to every frame individually — random seeks into
    H.264 each decode from the nearest keyframe and are 10-30x slower than
    sequential decode. The crop window is inherited from the YOLO frame list:
    we start at the first YOLO frame index and stop at the last, so no
    out-of-window frames are ever processed."""
    sw = Stopwatch()

    # Build a lookup: frame_index -> yolo record.  Sorted order is guaranteed
    # since YOLO frames are appended in read_frames() sequential order.
    rec_by_frame = {rec["frame"]: rec for rec in yolo_data["frames"]}
    frames_out = []

    if rec_by_frame:
        start_f = min(rec_by_frame)
        end_f   = max(rec_by_frame)

        cap = cv2.VideoCapture(video_path)
        if start_f > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

        idx = start_f
        while idx <= end_f:
            ok, full = cap.read()
            if not ok:
                break
            if idx in rec_by_frame:
                rec = rec_by_frame[idx]
                box, anchor = tuple(rec["crop_box"]), tuple(rec["anchor"])
                max_dist = ANCHOR_REJECT_FRAC * max(box[3] - box[1], box[2] - box[0])
                lm = _pose_in_box_mediapipe(models["mp_lite"], models["mp"], full, box, anchor, max_dist)
                frames_out.append({"frame": rec["frame"], "t": rec["t"], "landmarks": lm,
                                   "crop_box": rec["crop_box"], "anchor": rec["anchor"]})
            idx += 1
        cap.release()

    sw.lap("pose_extraction_lite_escalation")
    return {"fps": yolo_data["fps"], "n_frames": yolo_data["n_frames"], "w": yolo_data["w"],
            "h": yolo_data["h"], "stride": yolo_data["stride"], "frames": frames_out, "timings": sw.laps}
