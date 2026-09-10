"""
Rubber position detection for deception_v4.

Measures where the pitcher's pivot foot sits on the rubber, expressed as a
fraction of the rubber's half-width:
  0.0  = foot at exact center
 +1.0  = foot at third-base edge  (behind-HP camera: 3B = right side of image)
 -1.0  = foot at first-base edge

Algorithm (widest-dirt-circle method):
  1. Detect the white chalk circle that marks the mound boundary.
  2. Scan the ankle-relative vertical band; per column, measure what fraction
     of rows are chalk (bright + low-saturation white).
  3. Find the leftmost chalk column in the left half and rightmost chalk column
     in the right half — these are the two extremes of the mound circle.
  4. Their midpoint = mound center = rubber center (the rubber sits at the
     geometric center of the mound circle).
  5. Fallback: if chalk detection fails (park without visible chalk or bad angle),
     use "not-dirt" column signal — dirt is brown/tan (R >> G), everything else
     is the boundary.
  6. position = (ankle_x - mound_center_x) / rubber_half_width_est.

Why chalk works: the chalk circle is drawn centered on the rubber. Its leftmost
and rightmost points are equidistant from center by construction. Their average
is the rubber x-coordinate regardless of ankle vertical position.
"""
import cv2
import numpy as np
from scipy.ndimage import uniform_filter1d

# Vertical scan range relative to the pivot ankle (pixels).
# The chalk circle is visible near ankle height.
SCAN_ABOVE_MAX   = 120    # px above ankle to include in scan
SCAN_BELOW_MAX   = 80     # px below ankle to include in scan

# Chalk / white boundary detection.
# Chalk is bright (all channels high) and roughly white (low saturation).
CHALK_MIN        = 160    # all BGR channels must exceed this
CHALK_MAX_SAT    = 65     # max(B,G,R) - min(B,G,R) must be below this
CHALK_SMOOTH_K   = 5      # column-smoothing kernel size
CHALK_COL_FRAC   = 0.06   # min fraction of rows in a col that must be chalk

# Not-dirt fallback (brown/tan = R clearly dominates G).
DIRT_R_OVER_G    = 10     # "dirt" = R - G >= this
DIRT_MIN_R       = 60     # must be bright enough to be real surface
DIRT_COL_FRAC    = 0.40   # min not-dirt fraction for a column to count
DIRT_SMOOTH_K    = 15     # column-smoothing kernel for not-dirt signal

# Minimum run of qualifying columns to count as a real edge
GRASS_RUN_MIN    = 15

# Frame-edge guard: mound boundary must be this far from frame border
EDGE_MARGIN      = 40

# Plausible mound width bounds (pixels, for 1280-wide broadcast frame).
# Real mound ~18 ft dia; anything outside this range is a false detection.
MIN_MOUND_W      = 80
MAX_MOUND_W      = 700

# Rubber half-width estimate: rubber is 24 in, body ~72 in → half = 12/72
RUBBER_HALF_FRAC = 12.0 / 72.0


def _chalk_col_signal(roi_bgr):
    """
    Returns a 1-D bool array (width,) where True = column is chalky/white.
    Uses bright + low-saturation threshold — matches mound chalk line in
    broadcast video across different parks and lighting.
    """
    roi = roi_bgr.astype(float)
    B, G, R = roi[:, :, 0], roi[:, :, 1], roi[:, :, 2]
    min_c = np.minimum(np.minimum(R, G), B)
    max_c = np.maximum(np.maximum(R, G), B)
    is_chalk = (min_c >= CHALK_MIN) & ((max_c - min_c) <= CHALK_MAX_SAT)
    col_frac = uniform_filter1d(is_chalk.mean(axis=0).astype(float), size=CHALK_SMOOTH_K)
    return col_frac >= CHALK_COL_FRAC


def _not_dirt_col_signal(roi_bgr):
    """
    Returns a 1-D bool array (width,) where True = column is NOT mound dirt.
    Dirt is brown/tan: R clearly dominates G.
    """
    roi = roi_bgr.astype(float)
    B, G, R = roi[:, :, 0], roi[:, :, 1], roi[:, :, 2]
    is_dirt = (R - G >= DIRT_R_OVER_G) & (R >= DIRT_MIN_R)
    not_dirt_frac = uniform_filter1d((~is_dirt).mean(axis=0).astype(float), size=DIRT_SMOOTH_K)
    return not_dirt_frac >= DIRT_COL_FRAC


def _find_run_left(col_signal, min_run):
    """Rightmost col of the leftmost qualifying run from the left."""
    run = start = 0
    for x in range(len(col_signal)):
        if col_signal[x]:
            if run == 0: start = x
            run += 1
        else:
            if run >= min_run:
                return start + run - 1
            run = 0
    return None


def _find_run_right(col_signal, min_run):
    """Leftmost col of the leftmost qualifying run from the right."""
    run = start = 0
    for x in range(len(col_signal) - 1, -1, -1):
        if col_signal[x]:
            if run == 0: start = x
            run += 1
        else:
            if run >= min_run:
                return start - run + 1
            run = 0
    return None


def detect_rubber(frame_bgr, pivot_ankle_xy, body_scale):
    """
    Detect pitching rubber position via widest-dirt-circle method.

    Finds the leftmost and rightmost chalk marks of the mound boundary circle
    in a vertical band around the pivot ankle; their midpoint is the rubber
    center. Falls back to a not-dirt gap scan if chalk is not visible.

    Parameters
    ----------
    frame_bgr      : np.ndarray (H, W, 3) BGR
    pivot_ankle_xy : (x, y) pixel coords of the pivot ankle
    body_scale     : float, estimated pixel height of pitcher body

    Returns
    -------
    dict with mound geometry and rubber position, or None if detection fails.
    """
    h, w = frame_bgr.shape[:2]
    ax = float(pivot_ankle_xy[0])
    ay = float(pivot_ankle_xy[1])

    y_top = int(max(0,   ay - SCAN_ABOVE_MAX))
    y_bot = int(min(h-1, ay + SCAN_BELOW_MAX))
    if y_bot - y_top < 10:
        return None

    roi = frame_bgr[y_top:y_bot+1, :]
    half = w // 2

    # ── Method 1: chalk circle boundary ──────────────────────────────────────
    is_chalk = _chalk_col_signal(roi)

    left_chalk  = np.where(is_chalk[:half])[0]
    right_chalk = np.where(is_chalk[half:])[0]

    method = None
    left_x = right_x = None

    if left_chalk.size >= 1 and right_chalk.size >= 1:
        lx = int(left_chalk[0])           # leftmost chalk col in left half
        rx = int(half + right_chalk[-1])   # rightmost chalk col in right half
        width = rx - lx
        if MIN_MOUND_W <= width <= MAX_MOUND_W and lx >= EDGE_MARGIN and rx <= w - EDGE_MARGIN:
            left_x, right_x, method = lx, rx, "chalk"

    # ── Method 2: not-dirt gap (fallback) ────────────────────────────────────
    if method is None:
        is_nd = _not_dirt_col_signal(roi)
        lx2 = _find_run_left(is_nd, GRASS_RUN_MIN)
        rx2 = _find_run_right(is_nd, GRASS_RUN_MIN)
        if lx2 is not None and rx2 is not None and lx2 < rx2:
            width2 = rx2 - lx2
            if MIN_MOUND_W <= width2 <= MAX_MOUND_W and lx2 >= EDGE_MARGIN and rx2 <= w - EDGE_MARGIN:
                left_x, right_x, method = lx2, rx2, "not_dirt"

    if left_x is None:
        return None

    mound_center  = (left_x + right_x) / 2.0
    rubber_half_w = body_scale * RUBBER_HALF_FRAC

    # Behind-home-plate camera: pitcher faces the camera, so their LEFT (3B)
    # appears on the RIGHT side of the image (higher x).
    #   positive = ankle right of center = 3B side
    #   negative = ankle left of center  = 1B side
    position = (ax - mound_center) / rubber_half_w

    return {
        "mound_left_x":      float(left_x),
        "mound_right_x":     float(right_x),
        "mound_center_x":    mound_center,
        "mound_width_px":    float(right_x - left_x),
        "rubber_half_w_est": rubber_half_w,
        "method":            method,
        "position":          float(np.clip(position, -2.0, 2.0)),
    }


def detect_rubber_ankle_box(frame_bgr, pivot_ankle_xy, body_scale):
    """
    Detect pitching rubber position using a tight box below the pivot ankle.

    The rubber is a white rectangle (6" wide x 24" tall from pitcher's
    perspective, or 24" wide x 6" tall in plan view). From behind-home-plate
    camera, the rubber appears directly below (higher image y) the pivot ankle.

    A pixel is classified as rubber if all of R, G, B >= 200 AND
    max(R,G,B) - min(R,G,B) <= 40 (bright, near-neutral white rubber material
    — stricter than chalk to reject chalk-dust halos).

    Sign convention (matches detect_rubber):
      positive = ankle right of rubber center = 3B side
      negative = ankle left  of rubber center = 1B side

    Parameters
    ----------
    frame_bgr      : np.ndarray (H, W, 3) BGR
    pivot_ankle_xy : (x, y) pixel coords of the pivot ankle
    body_scale     : float, estimated pixel height of pitcher body (pixels/72in)

    Returns
    -------
    dict with keys {"position", "method", "mound_x", "mound_w"}, or None on
    failure (not enough rubber pixels, or detected width out of plausible range).
    """
    h, w = frame_bgr.shape[:2]
    ax = float(pivot_ankle_xy[0])
    ay = float(pivot_ankle_xy[1])

    # Rubber is 24" wide; body_scale is pixels per 72" (body height).
    # half-width estimate = 12/72 * body_scale (same as RUBBER_HALF_FRAC).
    rubber_half_frac = 12.0 / 72.0
    rubber_half_px = body_scale * rubber_half_frac

    # Search box: centered on ankle x, starting just above ankle y and
    # extending downward by enough to cover the 6" rubber height + margin.
    box_w = int(round(body_scale * 24.0 / 72.0 * 2))   # full rubber width
    box_h = int(round(body_scale * 8.0 / 72.0))         # rubber 6" + 2" margin

    # Slight upward offset so the box starts just above the ankle contact point
    y_top = int(ay - body_scale * 2.0 / 72.0)
    y_bot = y_top + box_h

    x_left  = int(ax - box_w / 2)
    x_right = x_left + box_w

    # Clamp to frame
    x_left  = max(0, x_left)
    x_right = min(w, x_right)
    y_top   = max(0, y_top)
    y_bot   = min(h, y_bot)

    if (x_right - x_left) < 4 or (y_bot - y_top) < 2:
        return None

    roi = frame_bgr[y_top:y_bot, x_left:x_right].astype(float)
    B, G, R = roi[:, :, 0], roi[:, :, 1], roi[:, :, 2]
    min_c = np.minimum(np.minimum(R, G), B)
    max_c = np.maximum(np.maximum(R, G), B)
    is_rubber = (min_c >= 200) & ((max_c - min_c) <= 40)

    total_rubber_px = int(is_rubber.sum())
    if total_rubber_px < 20:
        return None

    # Find leftmost and rightmost column containing at least one rubber pixel
    col_has_rubber = is_rubber.any(axis=0)
    rubber_cols = np.where(col_has_rubber)[0]
    if len(rubber_cols) == 0:
        return None

    left_col  = int(rubber_cols[0])
    right_col = int(rubber_cols[-1])
    detected_w = right_col - left_col

    # Validate detected width: must be at least half expected rubber width and
    # not wider than 40" worth (rejects full-row false positives such as bright
    # chalk lines or grass glare spanning the entire box).
    min_w = body_scale * 12.0 / 72.0
    max_w = body_scale * 40.0 / 72.0
    if detected_w < min_w or detected_w > max_w:
        return None

    # Convert column indices back to full-frame x coords
    rubber_left_x  = x_left + left_col
    rubber_right_x = x_left + right_col
    rubber_center_x = (rubber_left_x + rubber_right_x) / 2.0

    # Position: positive = ankle to the right of rubber center = 3B side
    position = (ax - rubber_center_x) / rubber_half_px

    return {
        "position": float(np.clip(position, -2.0, 2.0)),
        "method":   "ankle_box",
        "mound_x":  int(round(rubber_center_x)),
        "mound_w":  int(round(detected_w)),
    }
