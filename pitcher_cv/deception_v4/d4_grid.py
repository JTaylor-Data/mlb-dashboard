"""Builds a 5-row x 5-phase-column screenshot grid from a v4 run's saved
landmarks + the (kept) source videos, for visual spot-checking."""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from d4_run import TEST5

CELL_W, CELL_H = 340, 410
HEADER_H = 42
LABEL_W = 150
COLS = ["coming_set", "leg_lift_start", "leg_lift_peak", "foot_plant", "release"]
BG = (26, 26, 26)
FG = (240, 240, 240)
FAIL = (70, 70, 190)
SKELETON = [("l_shoulder", "l_hip"), ("r_shoulder", "r_hip"), ("l_hip", "l_knee"),
            ("r_hip", "r_knee"), ("l_knee", "l_ankle"), ("r_knee", "r_ankle"),
            ("l_shoulder", "r_shoulder"), ("l_hip", "r_hip"),
            ("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"),
            ("r_shoulder", "r_elbow"), ("r_elbow", "r_wrist")]


def _text(img, s, org, scale=0.5, color=FG, thick=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)


def _fit(img, w, h):
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    r = cv2.resize(img, (max(1, int(iw * s)), max(1, int(ih * s))), interpolation=cv2.INTER_AREA)
    out = np.full((h, w, 3), BG, np.uint8)
    y0, x0 = (h - r.shape[0]) // 2, (w - r.shape[1]) // 2
    out[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
    return out


def _draw_skeleton(img, lm, off):
    ox, oy = off
    def pt(n):
        return (int(lm[n][0] - ox), int(lm[n][1] - oy))
    for a, b in SKELETON:
        if a in lm and b in lm:
            cv2.line(img, pt(a), pt(b), (0, 230, 255), 2)
    for n in lm:
        cv2.circle(img, pt(n), 3, (0, 0, 255), -1)


def _upscale(img, min_w=560):
    if img.shape[1] < min_w:
        s = min_w / img.shape[1]
        img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)), interpolation=cv2.INTER_CUBIC)
    return img


def nearest_frame_idx(frames, t):
    if t is None:
        return None
    return int(np.argmin([abs(f["t"] - t) for f in frames]))


def cell_for(video_path, frames, idx):
    if idx is None:
        cell = np.full((CELL_H - 8, CELL_W - 8, 3), BG, np.uint8)
        _text(cell, "NOT FOUND", (12, 28), 0.5, FAIL)
        return cell
    rec = frames[idx]
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, rec["frame"])
    ok, full = cap.read()
    cap.release()
    if not ok:
        cell = np.full((CELL_H - 8, CELL_W - 8, 3), BG, np.uint8)
        _text(cell, "read failed", (12, 28), 0.5, FAIL)
        return cell
    x0, y0, x1, y1 = rec["crop_box"]
    lms = rec.get("landmarks")
    if lms:
        lx0 = min(v[0] for v in lms.values()); lx1 = max(v[0] for v in lms.values())
        ly0 = min(v[1] for v in lms.values()); ly1 = max(v[1] for v in lms.values())
        x0 = int(max(0, min(x0, lx0 - 25))); y0 = int(max(0, min(y0, ly0 - 25)))
        x1 = int(min(full.shape[1], max(x1, lx1 + 25)))
        y1 = int(min(full.shape[0], max(y1, ly1 + 25)))
    crop = full[y0:y1, x0:x1].copy()
    if lms:
        _draw_skeleton(crop, lms, (x0, y0))
    _label = f"t={rec['t']:.3f}s"
    cv2.putText(crop, _label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 4)
    cv2.putText(crop, _label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
    crop = _upscale(crop)
    return _fit(crop, CELL_W - 8, CELL_H - 8)


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "out"
    videos_dir = sys.argv[2] if len(sys.argv) > 2 else "test5_videos"
    results = json.load(open(os.path.join(out_dir, "results.json")))["results"]
    by_tag = {r["tag"]: r for r in results}

    tags = [t for t in TEST5 if t in by_tag and by_tag[t]["ok"]]
    W = LABEL_W + CELL_W * len(COLS)
    H = HEADER_H + CELL_H * len(tags)
    canvas = np.full((H, W, 3), BG, np.uint8)
    _text(canvas, "deception_v4 - YOLO-primary, escalate-on-fail (5 canonical pitchers)", (12, 28), 0.6, FG, 2)
    for c, name in enumerate(COLS):
        _text(canvas, name, (LABEL_W + c * CELL_W + 10, 28), 0.55, FG, 1)

    for r, tag in enumerate(tags):
        res = by_tag[tag]
        lm_path = os.path.join(out_dir, "landmarks", f"{tag}_yolo.json")
        frames = json.load(open(lm_path))["frames"]
        video_path = os.path.join(videos_dir, f"{tag}_0.mp4")
        y = HEADER_H + r * CELL_H
        cv2.line(canvas, (0, y), (W, y), (70, 70, 70), 1)
        _text(canvas, f"{tag} [{res['method']}]", (10, y + 26), 0.5, FG, 1)
        pt = res["phase_times"]
        line = 46
        for k in COLS:
            v = f"{pt[k]:.3f}" if pt[k] is not None else "n/a"
            _text(canvas, f"{k[:5]}={v}", (10, y + line), 0.34, FG)
            line += 15
        for c, k in enumerate(COLS):
            x = LABEL_W + c * CELL_W
            idx = nearest_frame_idx(frames, pt[k])
            cell = cell_for(video_path, frames, idx)
            canvas[y + 4:y + 4 + cell.shape[0], x + 4:x + 4 + cell.shape[1]] = cell

    out_path = os.path.join(out_dir, "grid_v4_test5.png")
    cv2.imwrite(out_path, canvas)
    print(f"saved {out_path} ({W}x{H})")


if __name__ == "__main__":
    main()
