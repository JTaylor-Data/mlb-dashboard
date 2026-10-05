"""
Deception+ pipeline, PHASE DETECTION - clean rewrite (v4). Current, validated
methodology only - no superseded candidate rules kept around for reference
(that history lives in deception_v3/dv3_phases.py's docstring, not here).

Phases are found BACKWARD FROM RELEASE: release is a violent, unmistakable,
once-per-clip arm-speed spike - the strongest real signal in the whole
delivery - and every earlier phase is bounded by the one after it, so a
search can never wander into unrelated post-pitch activity.

    release  <- strongest per-arm elbow+wrist speed peak that yields a
                complete, physically sane delivery
    peak     <- interior local max of lift-knee height, bounded before release
    start    <- walking back from peak to the knee-height baseline
    set      <- start of the last real quiet stretch in both-ankle speed
                before the lift
    plant    <- first frame the lift ankle reaches/passes the pivot ankle's
                height after peak, rejecting any crossing caused by a
                landmark jump rather than real motion (ankle_cross_pivot)

check_results() is the single gate for BOTH temporal ordering and the
physical-plausibility bounds, and is model-agnostic - it runs identically on
a single model's phase times or on an average of two models' times, which is
what lets d4_pipeline.py escalate cleanly.
"""
import numpy as np
from scipy.signal import find_peaks

SMOOTH_WIN = 5
LR_PAIRS = ["shoulder", "elbow", "wrist", "hip", "knee", "ankle", "foot"]

VIS_MIN = 0.35
JUMP_CAP = {"wrist": 0.40, "elbow": 0.30, "foot": 0.25, "ankle": 0.22,
            "knee": 0.18, "hip": 0.15, "shoulder": 0.15, "nose": 0.20}

PEAK_LOOKBACK_S = 1.20
PEAK_MIN_LEAD_S = 0.22
START_LOOKBACK_S = 0.90
QUIET_SET_FRAC = 0.05

ANKLE_JUMP_REJECT_FRAC = 0.05
ANKLE_JUMP_SETTLE_S = 0.10
ANKLE_LANDED_TOL_FRAC = 0.15
MIN_LIFT_RISE_FRAC = 0.06

# --- Plant re-diagnosis (this session) ---
# Real root cause, confirmed against real per-frame velocity data on 4
# labeled pitchers (skenes/skubal/wheeler/rogers), NOT the mechanism
# originally assumed going into this session (see session notes) - the
# ankle-crosses-pivot-height position test alone is real but noisy (the
# crossing flickers on/off frame to frame - confirmed directly), and worse,
# the settle check that was supposed to confirm a real landing only looked
# at VERTICAL ankle velocity. That's the exact "mid-air dead point" trap
# this project's own deception_v3 history already flagged and never fixed:
# a stride's dead point can sit near-zero in vy alone while the ankle is
# still moving fast HORIZONTALLY (confirmed directly - skubal's rejected
# plant candidate had vy=9.1px/frame, looking "quiet" under the old
# vertical-only jump gate, but real 2D speed was 41.6px/frame, driven
# almost entirely by vx=40.6 - obviously still in mid-stride, not planted).
# Fix: gate on 2D speed (hypot(vx, vy)), not vy alone.
PLANT_QUIET_SPEED_FRAC = 0.05    # 2D speed ceiling for "settled", as a fraction of body_scale.
                                  # Raised from an initial 0.045 - that value put a real, validated
                                  # plant frame's own trailing blip (10.7px/frame) a hairline
                                  # (0.05px/frame) over the ceiling, which combined with an earlier
                                  # blip to exceed the one-soft-violation allowance and wrongly reject
                                  # it. 0.05 still rejects every real violation found in this session's
                                  # labeled set by a wide margin (smallest confirmed-bad reading was
                                  # skubal's 28.8px/frame, ~2.4x this ceiling at his body scale).
PLANT_SOFT_VIOLATION_MULT = 3.0  # a single frame up to this multiple of the ceiling is tolerated
                                  # (real pose-tracking blips happen inside genuine plateaus - confirmed
                                  # on wheeler's independently-validated plant frame, which has exactly
                                  # one such blip 1 frame after touchdown) - but never at the candidate
                                  # frame itself, and never more than one frame in the window.
PLANT_POST_SETTLE_FRAMES = 4     # ~130ms at ~30fps - covers most of a real plant-to-release gap
                                  # (pre-motion check anchors back to leg_lift_peak instead of a fixed
                                  # frame count - see _plant_via_ankle_cross_pivot's docstring for why)

MAX_RELEASE_CANDIDATES = 10
RELEASE_DOMINANCE = 1.8
RELEASE_DECAY_FRAC = 0.25

PHASE_KEYS = ("coming_set", "leg_lift_start", "leg_lift_peak", "foot_plant", "release")

# --- Stride Direction (Part B) ---
# Categorize the pitcher's stride direction (angle of front foot's approach
# relative to the mound's centerline). Three categories:
# - CROSSFIRE: ankle strides toward the hitter's box (closing off the zone)
# - STRAIGHT: minimal lateral stride
# - OPEN: ankle strides away from the hitter's box (opening up)
# Measured as the lift-ankle's x-position at foot_plant relative to the pivot
# ankle's x-position, normalized by body_scale. Positive = first-base side
# (crossfire for RHP, open for LHP). Pivot-relative removes camera-perspective
# bias: in a catcher-facing clip, "forward" is depth (invisible in 2D), so
# the old arctan2(dx,dy) returned ~75-85° even for straight strides.
# Thresholds calibrated on 8-pitcher sample; needs Sale/Fisher to validate.
STRIDE_CROSSFIRE_NORM = 0.10    # fraction of body_scale, lift-ankle past pivot toward first-base side
STRIDE_OPEN_NORM = -0.10        # fraction of body_scale, lift-ankle past pivot toward third-base side

# Physical sanity bounds - reject nonsense, not "typical". delivery_time_s
# gates whether a release candidate is accepted at all inside detect_phases;
# plant_to_release_s/peak_to_plant_s are re-checked here in check_results,
# model-agnostically, against whatever phase TIMES (single-model or
# averaged) are on the table.
SANE = {
    "delivery_time_s": (0.15, 2.80),      # lift_start→release; wide for high-kick LHP
    "plant_to_release_s": (0.03, 0.80),   # plant→release; was 0.40, some arms take longer
    "peak_to_plant_s": (0.03, 1.80),      # peak→plant; extra room for pausing deliveries
}


def smooth(a, win=SMOOTH_WIN):
    a = np.asarray(a, dtype=float)
    if len(a) < win:
        return a
    pad = win // 2
    return np.convolve(np.pad(a, (pad, pad), mode="edge"), np.ones(win) / win, mode="valid")[:len(a)]


def _interp_nans(a):
    a = a.copy()
    ok = ~np.isnan(a)
    if ok.sum() < 2:
        return a
    idx = np.arange(len(a))
    a[~ok] = np.interp(idx[~ok], idx[ok], a[ok])
    return a


class Signals:
    """Clean, gated, side-consistent signals built from one extraction JSON
    (YOLO or MediaPipe LITE - same shape either way, see d4_extract.py)."""

    def __init__(self, data):
        self.data = data
        self.frames = data["frames"]
        self.fps = data["fps"]
        self.t = np.array([f["t"] for f in self.frames], dtype=float)
        self.ok = False
        self.fail = None
        self._build()

    @property
    def rate(self):
        # true sample rate of the ARRAY (stride-advanced), never self.fps
        if len(self.t) < 2:
            return self.fps
        dt = float(np.median(np.diff(self.t)))
        return 1.0 / dt if dt > 0 else self.fps

    def n_idx(self, seconds):
        return max(1, int(round(seconds * self.rate)))

    def _raw(self, name, comp):
        out = np.full(len(self.frames), np.nan)
        for i, f in enumerate(self.frames):
            lm = f.get("landmarks")
            if lm and name in lm:
                out[i] = lm[name][comp]
        return out

    def _build(self):
        names = ["nose"] + [f"{s}_{p}" for p in LR_PAIRS for s in ("l", "r")]
        self.x = {n: self._raw(n, 0) for n in names}
        self.y = {n: self._raw(n, 1) for n in names}
        self.vis = {n: self._raw(n, 2) for n in names}

        self.has_pose = ~np.isnan(self.y["l_knee"])
        if self.has_pose.mean() < 0.5:
            self.fail = f"pose coverage too low ({self.has_pose.mean():.0%})"
            return

        scale = self._rough_body_scale()
        if scale is None or scale < 20:
            self.fail = "could not establish a body scale"
            return
        self.body_scale = scale

        self._gate_quality(scale)
        self._fix_lr_continuity()
        self.has_pose = (~np.isnan(self.y["l_knee"]) & ~np.isnan(self.y["r_knee"])
                         & ~np.isnan(self.y["l_ankle"]) & ~np.isnan(self.y["r_ankle"]))
        for n in names:
            self.x[n] = _interp_nans(self.x[n])
            self.y[n] = _interp_nans(self.y[n])

        # Body-relative, not raw image pixels: a striding pitcher translates
        # up the frame by a large fraction of his own height, which would
        # otherwise swamp the genuine leg lift.
        hip_mid_y = (self.y["l_hip"] + self.y["r_hip"]) / 2.0
        self.knee_rel = {s: smooth(hip_mid_y - self.y[f"{s}_knee"]) for s in ("l", "r")}
        self.ankle_sp = {s: smooth(self._speed(f"{s}_ankle")) for s in ("l", "r")}
        self.arm_sp = {s: smooth(self._speed(f"{s}_elbow") + self._speed(f"{s}_wrist")) for s in ("l", "r")}
        self.both_ankle_sp = self.ankle_sp["l"] + self.ankle_sp["r"]
        self.ok = True

    def _rough_body_scale(self):
        vals = []
        for i in range(len(self.frames)):
            ny = self.y["nose"][i]
            ay = np.nanmax([self.y["l_ankle"][i], self.y["r_ankle"][i]])
            if not np.isnan(ny) and not np.isnan(ay) and ay > ny:
                vals.append(ay - ny)
        return float(np.median(vals)) if vals else None

    def _gate_quality(self, scale):
        """Drop landmarks that are low-confidence or moved further between
        samples than a joint physically can (both -> NaN, interpolated
        later). The per-sample cap scales with the gap since the last
        accepted sample - without that, one rejection goes stale and cascades
        into rejecting every later sample too."""
        self.n_gated = 0
        for n in self.x:
            joint = n.split("_", 1)[-1] if "_" in n else n
            cap = JUMP_CAP.get(joint, 0.2) * scale
            bad_vis = self.vis[n] < VIS_MIN
            self.x[n][bad_vis] = np.nan
            self.y[n][bad_vis] = np.nan
            self.n_gated += int(bad_vis.sum())
            last, last_i = None, None
            for i in range(len(self.x[n])):
                if np.isnan(self.x[n][i]):
                    continue
                p = np.array([self.x[n][i], self.y[n][i]])
                if last is not None:
                    gap = max(1, i - last_i)
                    if np.linalg.norm(p - last) > cap * gap:
                        self.x[n][i] = np.nan
                        self.y[n][i] = np.nan
                        self.n_gated += 1
                        continue
                last, last_i = p, i

    def _fix_lr_continuity(self):
        """MediaPipe/YOLO assign left/right per frame with no temporal
        identity - legs crossing mid-delivery flips it, silently corrupting
        any cross-frame comparison. Greedy fix: at each frame keep-or-swap
        ALL paired landmarks together, whichever is more continuous with the
        previous accepted frame. Runs pre-interpolation so filled gaps can't
        vote."""
        n = len(self.t)
        raw = {s: {p: np.stack([self.x[f"{s}_{p}"], self.y[f"{s}_{p}"]], axis=1) for p in LR_PAIRS}
               for s in ("l", "r")}
        last = {s: {p: (None if np.isnan(raw[s][p][0]).any() else raw[s][p][0]) for p in LR_PAIRS}
                for s in ("l", "r")}
        swap = np.zeros(n, dtype=bool)
        for i in range(1, n):
            keep = swp = 0.0
            any_valid = False
            for p in LR_PAIRS:
                cl, cr = raw["l"][p][i], raw["r"][p][i]
                pl, pr = last["l"][p], last["r"][p]
                if pl is not None and not np.isnan(cl).any():
                    keep += np.sum((cl - pl) ** 2); any_valid = True
                if pr is not None and not np.isnan(cr).any():
                    keep += np.sum((cr - pr) ** 2); any_valid = True
                if pl is not None and not np.isnan(cr).any():
                    swp += np.sum((cr - pl) ** 2)
                if pr is not None and not np.isnan(cl).any():
                    swp += np.sum((cl - pr) ** 2)
            do = any_valid and swp < keep
            swap[i] = do
            for p in LR_PAIRS:
                nl = raw["r"][p][i] if do else raw["l"][p][i]
                nr = raw["l"][p][i] if do else raw["r"][p][i]
                if not np.isnan(nl).any():
                    last["l"][p] = nl
                if not np.isnan(nr).any():
                    last["r"][p] = nr
        if swap.any():
            for p in LR_PAIRS:
                nl = np.where(swap[:, None], raw["r"][p], raw["l"][p])
                nr = np.where(swap[:, None], raw["l"][p], raw["r"][p])
                self.x[f"l_{p}"], self.y[f"l_{p}"] = nl[:, 0], nl[:, 1]
                self.x[f"r_{p}"], self.y[f"r_{p}"] = nr[:, 0], nr[:, 1]

    def _speed(self, name):
        x, y = _interp_nans(self.x[name]), _interp_nans(self.y[name])
        return np.hypot(np.diff(x, prepend=x[0]), np.diff(y, prepend=y[0]))


def _release_candidates(sg):
    """Candidate TIMES are the union of real local peaks in EACH ARM'S OWN
    speed series independently, not peaks of a merged max(l,r) signal - a
    frame-by-frame merge lets the GLOVE arm dominate right around real
    release (a real, common mechanical event), hiding the true throwing-arm
    peak from ever becoming a candidate at all."""
    times = set()
    for side in ("l", "r"):
        arm = sg.arm_sp[side]
        med_side = float(np.median(arm))
        if arm.max() <= 0:
            continue
        idxs, _ = find_peaks(arm, prominence=max(1e-6, 0.15 * (arm.max() - arm.min())))
        for i in list(idxs) + [int(np.argmax(arm))]:
            if arm[i] >= RELEASE_DOMINANCE * med_side:
                times.add(int(i))
    combined = np.maximum(sg.arm_sp["l"], sg.arm_sp["r"])
    med = float(np.median(combined))
    if not times:
        return [], med, combined
    cands = sorted(times, key=lambda i: -combined[i])
    return cands[:MAX_RELEASE_CANDIDATES], med, combined


def side_at_release(sg, rel, half_window_s=0.05):
    w = sg.n_idx(half_window_s)
    lo, hi = max(0, rel - w), min(len(sg.t), rel + w + 1)
    l = float(np.mean(sg.arm_sp["l"][lo:hi]))
    r = float(np.mean(sg.arm_sp["r"][lo:hi]))
    return ("l", l, r) if l >= r else ("r", r, l)


def _ankle_landed_ok(sg, cset, release, lift):
    """Is the lift ankle already plausibly down at this candidate release?
    Closes an arm-cocking-phase false positive (real arm speed, ball still
    in hand, front foot still airborne)."""
    lo = max(0, cset - 5)
    baseline = float(np.median(sg.y[f"{lift}_ankle"][lo:cset + 1]))
    at_release = float(sg.y[f"{lift}_ankle"][release])
    deficit_frac = (baseline - at_release) / sg.body_scale
    return deficit_frac <= ANKLE_LANDED_TOL_FRAC


def _plant_via_ankle_cross_pivot(sg, peak, bound, lift, pivot):
    """foot_plant = first frame after peak that is BOTH (a) at/past the
    pivot ankle's height (image y, down=+ - a real, necessary physical
    condition: the lift foot can't be down before it's at least level with
    the standing pivot foot) AND (b) the START of a genuine, sustained 2D-
    speed plateau - confirming an actual stop, not just a coincidental
    height crossing mid-swing.

    2D SPEED, NOT VERTICAL-ONLY: gating on vertical ankle velocity alone
    (the original v3/v4 approach) cannot tell a real landing apart from a
    stride's mid-air "dead point," which sits near-zero in vy while still
    moving fast horizontally - a known trap flagged in this project's
    deception_v3 history and never actually fixed there. Confirmed directly
    this session against real per-frame data: a wrongly-accepted plant
    candidate had vy=9.1px/frame ("quiet" by the old check) but 2D speed
    41.6px/frame (vx=40.6 - plainly still mid-stride). See PLANT_* constants
    above for the full reasoning and real numbers behind each threshold.

    A single post-candidate frame up to PLANT_SOFT_VIOLATION_MULT x the
    settle ceiling is tolerated (real tracking blips occur inside genuine
    plateaus - confirmed on an independently-validated plant frame that has
    exactly one such blip immediately after touchdown), but never at the
    candidate frame itself and never more than one frame in the window.

    PRE-CHECK is anchored back to `peak` (the top of the leg lift), not a
    small fixed window - confirmed necessary against real data: the actual
    downswing can finish several frames before the ankle reaches the pivot
    height and settles (a real final-approach coast, not stillness), so a
    tight few-frame pre-window wrongly flagged an independently-validated
    real plant as "not a real transition" and pushed the result onto a
    later, wrong frame. Anchoring to `peak` keeps the spirit of the check
    (this candidate isn't sitting inside a plateau that predates the lift
    even starting) without being fooled by that normal pre-landing coast."""
    la, pa = sg.y[f"{lift}_ankle"], sg.y[f"{pivot}_ankle"]
    lax = sg.x[f"{lift}_ankle"]
    vx = np.diff(lax, prepend=lax[0])
    vy = np.diff(la, prepend=la[0])
    speed = np.hypot(vx, vy)
    ceiling = PLANT_QUIET_SPEED_FRAC * sg.body_scale
    soft_cap = PLANT_SOFT_VIOLATION_MULT * ceiling
    bound = min(bound, len(la))
    n = len(la)

    def settled_start(i):
        if speed[i] > ceiling:
            return False  # candidate frame itself must be strictly quiet, no exception
        if not np.any(speed[peak:i] > ceiling):
            return False  # not a real transition - candidate is already deep in a plateau
        post_hi = min(n, i + PLANT_POST_SETTLE_FRAMES)
        soft_used = False
        for j in range(i + 1, post_hi):
            if speed[j] <= ceiling:
                continue
            if speed[j] <= soft_cap and not soft_used:
                soft_used = True
                continue
            return False
        return True

    last_reason = "lift ankle never reaches pivot-ankle height before the search bound"
    for i in range(peak, bound):
        if la[i] < pa[i]:
            continue
        last_reason = f"crossing at t={sg.t[i]:.3f}s rejected - not the start of a real settle"
        if settled_start(i):
            return i, None
    return None, last_reason


def _release_via_arm_decay(sg, plant, throw, window_s=None, decay_frac=RELEASE_DECAY_FRAC):
    """Release is not the frame of peak arm speed - the ball is genuinely
    gone once speed has fallen off the peak by a bit. Find the peak within
    the window (bounded after plant), walk forward to the first point that
    has decayed decay_frac off it - that's release."""
    if window_s is None:
        window_s = SANE["plant_to_release_s"][1]
    arm = sg.arm_sp[throw]
    lo = plant + 1
    hi = min(len(arm), plant + sg.n_idx(window_s))
    if hi <= lo:
        return None, "no room after plant for a release search"
    seg = arm[lo:hi]
    pk_local = int(np.argmax(seg))
    thresh = float(seg[pk_local]) * (1 - decay_frac)
    for i in range(pk_local, len(seg)):
        if seg[i] <= thresh:
            return lo + i, None
    return lo + len(seg) - 1, None


def _find_coming_set(sg, start):
    """coming_set = start of the LAST real, sustained quiet stretch in
    both-ankle speed before the lift - not the single quietest frame, and
    not "last moving sample + 1" (collapses to `start` on a smooth ramp-up,
    reporting a false hold_time_s=0). No time cap: searches to the start of
    the processed shot (the shot boundary is the real constraint, not an
    arbitrary duration)."""
    both = sg.both_ankle_sp[:start]
    valid = sg.has_pose[:start]
    n = len(both)
    if n == 0 or not valid.any():
        return 0, True
    thresh = QUIET_SET_FRAC * sg.body_scale
    quiet = valid & (both <= thresh)
    min_run = max(2, sg.n_idx(0.08))
    runs, k = [], 0
    while k < n:
        if quiet[k]:
            j = k
            while j + 1 < n and quiet[j + 1]:
                j += 1
            if j - k + 1 >= min_run:
                runs.append((k, j))
            k = j + 1
        else:
            k += 1
    if not runs:
        return 0, True
    last_start, _ = runs[-1]
    return last_start, (last_start == 0)


def _phases_from_release(sg, release, lift, throw):
    """Full phase set backward from one candidate release. Plant detection
    is fully decoupled from whether this release/peak/start/set candidate
    gets accepted - only after that chain is locked in does plant get one
    attempt; if not found, foot_plant comes back None (an honest gap, not a
    reason to discard an otherwise-valid candidate here - check_results, run
    afterward by the pipeline, is what actually enforces completeness)."""
    kh = sg.knee_rel[lift]

    hi = release - sg.n_idx(PEAK_MIN_LEAD_S)
    lo = max(0, release - sg.n_idx(PEAK_LOOKBACK_S))
    if hi - lo < 2:
        return None, "no room before release for a leg-lift peak"
    seg_kh = kh[lo:hi]
    pk_idxs, pk_props = find_peaks(seg_kh, prominence=max(1e-6, 0.10 * float(np.ptp(seg_kh))))
    if len(pk_idxs):
        peak = lo + int(pk_idxs[int(np.argmax(pk_props["prominences"]))])
    else:
        cand = int(np.argmax(seg_kh))
        if cand in (0, len(seg_kh) - 1):
            return None, "no interior leg-lift peak before release"
        peak = lo + cand

    s_lo = max(0, peak - sg.n_idx(START_LOOKBACK_S))
    if peak - s_lo < 2:
        return None, "not enough samples before the leg-lift peak"
    base = float(np.min(kh[s_lo:peak + 1]))
    rise = float(kh[peak] - base)
    if rise < MIN_LIFT_RISE_FRAC * sg.body_scale:
        return None, f"lift-knee rise {rise:.1f}px too small - no real leg lift"
    onset = base + 0.15 * rise
    start = peak
    while start > s_lo and kh[start] > onset:
        start -= 1

    if start < 1:
        return None, "clip does not start early enough to see a set position"
    cset, cset_clipped = _find_coming_set(sg, start)

    if not (cset <= start < peak < release):
        return None, f"phases out of order: {[cset, start, peak, release]}"
    t = sg.t
    delivery_time_s = float(t[release] - t[start])
    dlo, dhi = SANE["delivery_time_s"]
    if not (dlo <= delivery_time_s <= dhi):
        return None, f"delivery_time_s={delivery_time_s:.3f}s outside {dlo}-{dhi}s"

    if not _ankle_landed_ok(sg, cset, release, lift):
        return None, "front foot not plausibly down yet at this release candidate"

    release_anchor = release
    plant_bound = release_anchor + sg.n_idx(SANE["plant_to_release_s"][1])
    plant, plant_err = _plant_via_ankle_cross_pivot(sg, peak, plant_bound, lift, throw)
    if plant is not None:
        release, release_err = _release_via_arm_decay(sg, plant, throw)
        if release is None:
            release, release_err = release_anchor, f"refinement failed ({release_err}) - kept anchor"
        if release <= plant:
            # Anchor-fallback path isn't guaranteed to land after plant - most
            # often because plant itself landed AT the release anchor with zero
            # frames of runway after it (a real edge case, confirmed on a real
            # abbreviated/submarine delivery whose clip ends exactly at release -
            # not a detection failure). Previously this silently discarded plant
            # entirely, surfacing as an uninformative "missing phases" reason.
            # Keep both values and let check_results reject with the real,
            # specific reason (ordering or a sub-0.01s plant_to_release_s) - same
            # outcome (correctly rejected), but debuggable.
            release = release_anchor
    else:
        release, release_err = release_anchor, None

    return {"coming_set": cset, "leg_lift_start": start, "leg_lift_peak": peak,
            "foot_plant": plant, "release": release, "_rise": rise,
            "_release_anchor": release_anchor, "_release_err": release_err,
            "_cset_clipped": cset_clipped}, None


def detect_phases(sg, throws="R"):
    """Every release candidate tried against both side assignments; a
    candidate/side combo survives only if it produces a complete, sane
    delivery. Side is decided by whichever assignment gives the bigger
    knee-height rise (the real lift leg rises far more than the pivot leg's
    follow-through). The winning candidate is the EARLIEST survivor, not the
    strongest spike (post-pitch follow-through is often crisper/untouched by
    motion blur and would otherwise win over the real, earlier release).

    Parameters
    ----------
    sg     : Signals
    throws : str, "R" or "L" — Statcast p_throws field for this pitcher.
             Used to set ph["throw_side"] ("r" or "l") directly from the
             authoritative Statcast source rather than inferring it from
             which arm's speed dominates at release (which is a camera-view
             artifact that produces wrong results on standard broadcast clips).
    """
    cands, med, combined = _release_candidates(sg)
    if not cands:
        return None, f"no arm-speed spike dominant enough to be a release (max {combined.max():.1f} vs median {med:.1f})"

    tried, passed = [], []
    for rel in cands:
        sp_throw, sp_other = side_at_release(sg, rel)[1:]
        best = None
        for throw in ("l", "r"):
            lift = "r" if throw == "l" else "l"
            ph, err = _phases_from_release(sg, rel, lift, throw)
            if ph is not None and (best is None or ph["_rise"] > best[1]["_rise"]):
                best = (rel, ph, throw, lift)
        if best is not None:
            passed.append(best)
    if not passed:
        return None, "no release candidate produced a sane delivery"

    passed.sort(key=lambda p: p[0])
    rel, ph, throw, lift = passed[0]
    # throw_side comes from Statcast p_throws, not pose inference.
    # Pose inference assigns throw_side based on which arm speed dominates
    # at release in the broadcast camera view — a camera-convention artifact
    # that gives the same result (often "l") for both LHP and RHP pitchers.
    throw_side_statcast = throws.lower() if throws else "r"
    ph.update({"lift_side": lift, "pivot_side": throw, "throw_side": throw_side_statcast})
    for k in ("_release_anchor", "_release_err", "_cset_clipped", "_rise"):
        ph.pop(k, None)
    return ph, None


def _stride_direction(sg, ph):
    """Compute stride direction (crossfire/straight/open).

    Measures lift-ankle x at foot_plant relative to pivot-ankle x at foot_plant,
    normalized by body_scale. This is camera-perspective-stable: in a catcher-
    facing clip the 'forward' direction is depth and disappears in 2D, so angle-
    based methods produce 75-85° even for straight strides. The pivot-relative
    lateral offset isolates only the true crossfire/open component.

    Sign: dx_norm positive = lift-ankle is to the RIGHT of pivot-ankle in image.
    For RHP (left foot strides): right = first-base side = CROSSFIRE.
    For LHP (right foot strides): right = third-base side = OPEN.
    """
    if ph is None or ph.get("foot_plant") is None:
        return None
    lift_side = ph["lift_side"]
    pivot_side = "r" if lift_side == "l" else "l"
    lift_ankle = f"{lift_side}_ankle"
    pivot_ankle = f"{pivot_side}_ankle"
    plant_i = ph["foot_plant"]

    dx_norm = (sg.x[lift_ankle][plant_i] - sg.x[pivot_ankle][plant_i]) / sg.body_scale

    throw = ph["throw_side"]
    if throw == "r":
        # positive dx_norm = first-base side = CROSSFIRE
        if dx_norm >= STRIDE_CROSSFIRE_NORM:
            return "CROSSFIRE"
        elif dx_norm <= STRIDE_OPEN_NORM:
            return "OPEN"
        else:
            return "STRAIGHT"
    else:
        # LHP: positive dx_norm = third-base side = OPEN
        if dx_norm <= -STRIDE_CROSSFIRE_NORM:
            return "CROSSFIRE"
        elif dx_norm >= -STRIDE_OPEN_NORM:
            return "OPEN"
        else:
            return "STRAIGHT"


def _stride_norm(sg, ph):
    """Return the raw normalized pivot-relative x offset for stride direction.
    Positive = first-base side (right in image), negative = third-base side.

    Sign is camera-relative, NOT handedness-relative. Callers that want a
    single sign convention regardless of handedness should use
    _stride_norm_mirrored() instead."""
    if ph is None or ph.get("foot_plant") is None:
        return None
    lift_side = ph["lift_side"]
    pivot_side = "r" if lift_side == "l" else "l"
    plant_i = ph["foot_plant"]
    return float((sg.x[f"{lift_side}_ankle"][plant_i] - sg.x[f"{pivot_side}_ankle"][plant_i]) / sg.body_scale)


def _stride_norm_mirrored(sg, ph):
    """Stride norm with a unified sign convention: negative = crossfire, positive = open.

    Empirically confirmed on Sale (LHP, 3 crossfire pitches) and Houck (RHP,
    2 crossfire pitches): raw stride_norm is already negative for crossfire
    and positive for open for BOTH handedness in behind-home-plate broadcast
    clips. No per-handedness negation is needed — the camera geometry and
    lift_side assignment conspire to produce the same sign for the same
    delivery type regardless of actual throwing hand."""
    raw = _stride_norm(sg, ph)
    return raw


def phase_times(sg, ph):
    if ph is None:
        return {k: None for k in PHASE_KEYS}
    return {k: (float(sg.t[ph[k]]) if ph[k] is not None else None) for k in PHASE_KEYS}


def check_results(times):
    """Single gate for completeness + temporal ordering + physical-
    plausibility bounds. Model-agnostic - runs identically whether `times`
    came from one model's detect_phases() or an average of two. Returns
    (ok: bool, reason: str|None)."""
    missing = [k for k in PHASE_KEYS if times.get(k) is None]
    if missing:
        return False, f"missing phases: {missing}"
    vals = [times[k] for k in PHASE_KEYS]
    if not all(vals[i] < vals[i + 1] for i in range(len(vals) - 1)):
        return False, f"phases out of order: {times}"

    delivery = times["release"] - times["leg_lift_start"]
    peak_to_plant = times["foot_plant"] - times["leg_lift_peak"]
    plant_to_release = times["release"] - times["foot_plant"]
    dlo, dhi = SANE["delivery_time_s"]
    plo, phi = SANE["peak_to_plant_s"]
    rlo, rhi = SANE["plant_to_release_s"]
    if not (dlo <= delivery <= dhi):
        return False, f"delivery_time_s={delivery:.3f}s outside {dlo}-{dhi}s"
    if not (plo <= peak_to_plant <= phi):
        return False, f"peak_to_plant_s={peak_to_plant:.3f}s outside {plo}-{phi}s"
    if not (rlo <= plant_to_release <= rhi):
        return False, f"plant_to_release_s={plant_to_release:.3f}s outside {rlo}-{rhi}s"
    return True, None
