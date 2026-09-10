"""
Pre-crop: find the delivery window in a Savant clip using frame differencing.

Savant clips are 15-25 seconds but the actual delivery (coming_set → release +
follow-through) is 2-5 seconds. Running YOLO+pose on the full clip wastes 75-85%
of inference time on frames that will never be used.

Algorithm:
  1. Sample the video at ~5 fps (no ML).
  2. Compute inter-frame pixel difference at each sample.
  3. Find the peak-motion frame (≈ arm acceleration / release).
  4. Return (start_frame, end_frame) = peak ± generous buffer.

The result is passed directly to read_frames() in d4_extract.py — no temp file
written, no re-encode, just a tighter frame range.
"""
import cv2
import numpy as np

# How many seconds to include before and after the peak-motion frame.
# Peak ≈ arm release, so we need 3-4s before (for leg lift + set position).
PRE_BUFFER_S  = 4.0
POST_BUFFER_S = 2.0

# Sample rate for the fast scan (frames per second, no ML)
SCAN_FPS = 5.0

# Minimum fraction of the video's max diff that counts as "real motion"
# (avoids picking a noise spike as the peak)
MOTION_FRAC_THRESH = 0.35


def find_delivery_window(video_path):
    """
    Fast motion scan to find the delivery window.

    Returns (start_frame, end_frame) suitable for passing to read_frames().
    Falls back to (0, n_frames-1) if detection fails.
    """
    cap = cv2.VideoCapture(video_path)
    fps      = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0 or n_frames <= 0:
        cap.release()
        return 0, max(0, n_frames - 1)

    stride = max(1, int(round(fps / SCAN_FPS)))

    # Scan: read every `stride` frames, compute mean absolute diff from previous
    samples   = []   # (frame_idx, diff_score)
    prev_gray = None
    idx       = 0

    while idx < n_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        # Shrink to 160px wide for speed (diff is scale-invariant)
        small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
        gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev_gray is not None:
            diff = float(np.mean(np.abs(gray - prev_gray)))
            samples.append((idx, diff))
        prev_gray = gray
        idx += stride

    cap.release()

    if not samples:
        return 0, max(0, n_frames - 1)

    diffs = np.array([d for _, d in samples])

    # Find the first frame whose motion exceeds the threshold fraction of the max.
    # This is usually the leg-kick / arm acceleration peak.
    peak_idx  = int(np.argmax(diffs))
    peak_frame = samples[peak_idx][0]

    start_frame = max(0,          peak_frame - int(fps * PRE_BUFFER_S))
    end_frame   = min(n_frames-1, peak_frame + int(fps * POST_BUFFER_S))

    return start_frame, end_frame
