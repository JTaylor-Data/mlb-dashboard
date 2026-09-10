"""
Deception+ pipeline, ORCHESTRATION - clean rewrite (v4).

Per pitch: YOLO-only extraction + phases -> check_results. If that passes,
done - this is meant to be the common case, and the only case that pays for
just one pose model. If it fails (missing phase, bad ordering, or a sanity
bound), escalate: reuse the SAME tracked crop_box/anchor already computed
(no re-detection/re-tracking) and run MediaPipe LITE over those same frames,
then average whichever phase values each model found and re-run
check_results on the average. If that still fails, reject the pitch.

The video is deleted exactly once, after this whole attempt (accept or
reject) is finished - never mid-attempt, since escalation needs the file
still on disk to re-read frames.
"""
import os
import traceback

import cv2
import numpy as np

from d4_extract import extract_yolo, extract_lite_escalation
from d4_phases import Signals, detect_phases, phase_times, check_results, PHASE_KEYS, _stride_direction, _stride_norm, _stride_norm_mirrored
from d4_rubber import detect_rubber, detect_rubber_ankle_box
from d4_timing import Stopwatch

SKELETON = [
    ("l_shoulder", "l_hip"), ("r_shoulder", "r_hip"),
    ("l_hip", "l_knee"), ("r_hip", "r_knee"),
    ("l_knee", "l_ankle"), ("r_knee", "r_ankle"),
    ("l_shoulder", "r_shoulder"), ("l_hip", "r_hip"),
    ("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"),
    ("r_shoulder", "r_elbow"), ("r_elbow", "r_wrist"),
]


def _save_phase_frames(video_path, lm_data, phase_t, out_dir, tag):
    """Save one cropped frame PNG per phase key, with skeleton overlay."""
    frames = lm_data.get("frames", [])
    if not frames:
        return

    phase_keys = ["coming_set", "leg_lift_start", "leg_lift_peak", "foot_plant", "release"]
    # Build t -> frame_idx mapping
    ts = [f["t"] for f in frames]

    cap = cv2.VideoCapture(video_path)
    for key in phase_keys:
        t = phase_t.get(key)
        if t is None:
            continue
        idx = int(np.argmin([abs(tt - t) for tt in ts]))
        rec = frames[idx]
        cap.set(cv2.CAP_PROP_POS_FRAMES, rec["frame"])
        ok, full = cap.read()
        if not ok:
            continue
        x0, y0, x1, y1 = rec["crop_box"]
        lms = rec.get("landmarks") or {}
        if lms:
            lx0 = min(v[0] for v in lms.values()); lx1 = max(v[0] for v in lms.values())
            ly0 = min(v[1] for v in lms.values()); ly1 = max(v[1] for v in lms.values())
            x0 = int(max(0, min(x0, lx0 - 20))); y0 = int(max(0, min(y0, ly0 - 20)))
            x1 = int(min(full.shape[1], max(x1, lx1 + 20)))
            y1 = int(min(full.shape[0], max(y1, ly1 + 20)))
        crop = full[y0:y1, x0:x1].copy()
        if lms:
            def pt(n):
                return (int(lms[n][0] - x0), int(lms[n][1] - y0))
            for a, b in SKELETON:
                if a in lms and b in lms:
                    cv2.line(crop, pt(a), pt(b), (0, 230, 255), 2)
            for n in lms:
                cv2.circle(crop, pt(n), 3, (0, 0, 255), -1)
        label = f"t={rec['t']:.3f}s"
        cv2.putText(crop, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4)
        cv2.putText(crop, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # Upscale narrow crops for visibility
        if crop.shape[1] < 300:
            s = 300 / crop.shape[1]
            crop = cv2.resize(crop, (300, int(crop.shape[0] * s)), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(out_dir, f"{tag}_{key}.png"), crop)
    cap.release()


def average_times(a, b):
    out = {}
    for k in PHASE_KEYS:
        vals = [v for v in (a.get(k), b.get(k)) if v is not None]
        out[k] = (sum(vals) / len(vals)) if vals else None
    return out


def process_one(video_path, models, tag, keep_video=False, landmarks_dir=None, throws="R"):
    """Returns a result dict, always - failures included, with a reason.
    res['timings'] is the full stage breakdown for this one pitch.
    landmarks_dir, if given, saves the winning model's landmarks JSON (the
    durable artifact - trivial to re-derive metrics from, unlike the video).

    Parameters
    ----------
    throws : str, "R" or "L" — Statcast p_throws for this pitcher. Passed
             through to detect_phases() so throw_side is set from the
             authoritative Statcast field rather than pose inference.
    """
    res = {"tag": tag, "ok": False}

    data, err = extract_yolo(video_path, models)
    if data is None:
        res["stage"], res["error"] = "extract", err
        res["timings"] = {}
        if not keep_video and os.path.exists(video_path):
            os.remove(video_path)
        return res
    timings = dict(data["timings"])

    # A fresh Stopwatch here (not one spanning back through extract_yolo)
    # is the whole point - laps must start AFTER the thing they're not
    # supposed to measure, or they silently absorb its wall time.
    sw = Stopwatch()
    sg = Signals(data)
    sw.lap("signals_yolo")
    ph = None
    times_yolo = {k: None for k in PHASE_KEYS}
    if sg.ok:
        ph, _ = detect_phases(sg, throws=throws)
        sw.lap("phase_detection_yolo")
        times_yolo = phase_times(sg, ph)
    timings.update(sw.laps)   # capture before `sw` is replaced below

    ok, reason = check_results(times_yolo)
    method = "yolo_only"
    lite_data = None

    if not ok:
        lite_data = extract_lite_escalation(video_path, data, models)
        timings.update({f"escalation_{k}": v for k, v in lite_data["timings"].items()})
        sw = Stopwatch()   # same reason - extract_lite_escalation's own cost
        sg2 = Signals(lite_data)
        sw.lap("signals_lite")
        times_lite = {k: None for k in PHASE_KEYS}
        if sg2.ok:
            ph2, _ = detect_phases(sg2, throws=throws)
            sw.lap("phase_detection_lite")
            times_lite = phase_times(sg2, ph2)

        avg = average_times(times_yolo, times_lite)
        ok2, reason2 = check_results(avg)
        sw.lap("check_results")
        timings.update(sw.laps)
        if ok2:
            times_yolo, ok, reason, method = avg, True, None, "yolo+lite_avg"
        else:
            reason = f"failed after escalation: yolo=({reason}) lite+avg=({reason2})"
            method = "rejected"

    if landmarks_dir:
        os.makedirs(landmarks_dir, exist_ok=True)
        import json
        with open(os.path.join(landmarks_dir, f"{tag}_yolo.json"), "w") as f:
            json.dump(data, f)
        if lite_data is not None:
            with open(os.path.join(landmarks_dir, f"{tag}_lite.json"), "w") as f:
                json.dump(lite_data, f)

    if landmarks_dir and ok:
        frames_dir = os.path.join(landmarks_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        _save_phase_frames(video_path, data, times_yolo, frames_dir, tag)

    # Always compute rubber + stride for accepted pitches (video still on disk here).
    # Anchor frame: prefer leg_lift_peak (pivot foot firmly planted, full-resolution
    # frame, after the pitcher has settled onto the rubber) with coming_set as
    # fallback. coming_set fails when the Savant clip starts mid-at-bat and never
    # contains a true set position; leg_lift_peak is almost always present.
    if ok:
        rubber_pos = None
        anchor_t = times_yolo.get("leg_lift_peak") or times_yolo.get("coming_set")
        if anchor_t is not None and ph is not None:
            try:
                frames_list = data.get("frames", [])
                anc_idx = int(np.argmin([abs(f["t"] - anchor_t) for f in frames_list]))
                anc_rec = frames_list[anc_idx]
                pivot_side = "r" if ph["lift_side"] == "l" else "l"
                pivot_key = f"{pivot_side}_ankle"
                lms = anc_rec.get("landmarks") or {}
                if pivot_key in lms:
                    # Full-resolution frame from the original mp4 (NOT d4_crop's
                    # 160x90 thumbnails).  detect_rubber needs full pixel data for
                    # the chalk-circle scan.
                    cap = cv2.VideoCapture(video_path)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, anc_rec["frame"])
                    ok_f, full_frame = cap.read()
                    cap.release()
                    if ok_f:
                        # Try ankle_box first (more precise when rubber pixels
                        # are visible below the pivot ankle), fall back to
                        # chalk-circle/not-dirt method on failure.
                        result = detect_rubber_ankle_box(full_frame, lms[pivot_key][:2], sg.body_scale)
                        if result is None:
                            result = detect_rubber(full_frame, lms[pivot_key][:2], sg.body_scale)
                        if result:
                            rubber_pos = result["position"]
                            res["rubber_method"] = result.get("method")
            except Exception:
                pass
        res["rubber_position"] = rubber_pos
        res["stride_direction"] = _stride_direction(sg, ph)
        res["stride_norm"] = _stride_norm(sg, ph)
        res["stride_norm_mirrored"] = _stride_norm_mirrored(sg, ph)

    if not keep_video and os.path.exists(video_path):
        os.remove(video_path)

    res["ok"] = ok
    res["method"] = method
    res["phase_times"] = times_yolo
    if not ok:
        res["stage"], res["error"] = "check_results", reason
    res["pose_hit"] = f"{data['pose_hit']}/{data['n_processed']}"
    res["timings"] = timings
    res["seconds"] = round(sum(res["timings"].values()), 1)
    return res
