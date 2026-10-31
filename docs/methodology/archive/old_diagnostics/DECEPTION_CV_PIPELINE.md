# Deception+ v4 — Pipeline Handoff
_Last updated: 2026-08-28_

---

## What this project is

A computer-vision pipeline that extracts pitcher delivery biomechanics from
MLB Savant broadcast clips. Given a video URL, it outputs phase timestamps,
stride direction, and rubber position for one pitch — eventually fed into a
Deception+ model as features.

---

## Repository layout

```
mlb_pitcher_isolation_investigation/
├── deception_v4/               ← active pipeline (everything else is old)
│   ├── d4_crop.py              ← fast pre-crop (motion scan, no ML)
│   ├── d4_extract.py           ← YOLO detection + pose extraction
│   ├── d4_phases.py            ← phase detection + stride metrics
│   ├── d4_pipeline.py          ← orchestration (YOLO-only → escalation)
│   ├── d4_rubber.py            ← rubber position detection (chalk circle)
│   └── d4_timing.py            ← Stopwatch utility
├── run_10pitch.py              ← 10-pitch test on Framber Valdez (LHP)
├── run_crossfire_test.py       ← 10-pitch test on Tanner Houck (RHP crossfire)
├── test_rubber_sale.py         ← single-pitch rubber detection validation
├── models/                     ← YOLO weights (gitignored)
├── yolo11m-pose.pt             ← YOLO pose model
├── pose_landmarker_lite.task   ← MediaPipe fallback model
└── bcv_venv311/                ← Python 3.11 venv (gitignored)
```

**Run everything with**: `bcv_venv311/Scripts/python.exe <script>.py`

---

## Pipeline — step by step

### 1. Video fetch
`run_*.py` → Statcast CSV → MLB Stats API game feed → `playId` UUID →
`baseballsavant.mlb.com/sporty-videos?playId=...` → mp4 download (requires
`Referer: https://baseballsavant.mlb.com/` header).

### 2. Pre-crop (`d4_crop.py`)
Before any ML runs, sample the video at 5 fps using frame differencing
(160×90 thumbnails). Find the peak-motion frame (≈ arm acceleration / release)
and return a window of ±4s pre / ±2s post. Falls back to full video if no
clear peak found. **This cut `frame_decode` from ~8s to ~0.28s (median).**

### 3. Shot detection + pitcher selection (`d4_extract.py`)
Coarse stride scan finds camera-cut boundaries. For each shot, probe 8 frames
with the phc_detector YOLO model; accept shot only if the pitcher appears in
≥60% of probed frames (`SHOT_MIN_HIT_RATE=0.60`, raised from 0.50 to fix
skeleton drift to wrong person in Skenes clips). Follow-through cutoff probe
trims the shot end.

### 4. YOLO pose extraction (`d4_extract.py`)
Run phc_detector (bbox) then YOLO11m-pose (keypoints) in batches of 8 frames
on the pre-cropped shot window. 17 COCO keypoints → named dict
(`l_ankle`, `r_shoulder`, etc.). **Biggest bottleneck: 23.8s median (47%
of wall time).**

### 5. Phase detection (`d4_phases.py`)
Works **backward from release** (the strongest, most unambiguous signal):

| Phase | Detection method |
|-------|-----------------|
| **release** | Peak of combined elbow+wrist 2D speed |
| **leg_lift_peak** | Interior local max of lift-knee height, bounded before release |
| **leg_lift_start** | Walk back from peak to knee-height baseline |
| **coming_set** | Last quiet stretch in both-ankle speed before the lift |
| **foot_plant** | First frame lift ankle reaches/passes pivot ankle height, confirmed by 2D speed below 5% body_scale (not just vertical speed — that was the v3 bug) |

SANE bounds (sanity, not "typical"):
- `delivery_time_s`: 0.15–2.80s (wide for high-kick LHP like Valdez)
- `plant_to_release_s`: 0.03–0.80s
- `peak_to_plant_s`: 0.03–1.80s

### 6. Escalation (`d4_pipeline.py`)
If YOLO-only fails `check_results()`, re-run the same tracked crop boxes
through MediaPipe LITE (no re-detection). Average the two models' phase times
and re-check. If still failing → reject. Escalation adds ~5s when it fires
but the full-failure path (pose_hit fine but phases all missing) still burns
~35–50s per pitch.

### 7. Rubber position (`d4_rubber.py`)
Called on the **coming_set** frame at full video resolution.

**Algorithm — chalk circle method (primary):**
1. Scan ±120px / +80px around the pivot ankle.
2. Detect bright + low-saturation pixels (chalk/white): min(BGR) ≥ 160,
   max−min ≤ 65.
3. Find leftmost chalk column in the left half, rightmost in right half.
4. Midpoint = mound center = rubber center (the chalk circle is drawn centered
   on the rubber).
5. Validate: width 80–700px, both edges ≥ 40px from frame border.

**Fallback — not-dirt gap:**
If no chalk found: identify "dirt" columns (R−G ≥ 10 and R ≥ 60), find the
gap between the outer not-dirt runs on each side.

**Sign convention (behind-home-plate camera):**
Pitcher faces the camera, so their left (3B side) is on the **right** of the
image (higher x).
- `position > 0` → ankle right of center → **3B side**
- `position < 0` → ankle left of center → **1B side**
- `±1.0` ≈ at the rubber edge; scale is `body_scale × 12/72`

**Known limitation:** coming_set is not detected in many Savant clips (the
clip starts mid-at-bat rather than from the set). When coming_set is None,
rubber_pos is None. Needs a fallback anchor (e.g., foot_plant frame).

### 8. Stride metrics (`d4_phases.py`)
- `stride_direction`: sign of `(lift_ankle_x_at_plant − pivot_ankle_x) /
  body_scale`. Positive = stride toward 1B side; negative = toward 3B side.
  For RHP: negative = crossfire, positive = open. For LHP: reversed.
- `stride_norm`: same value (unnormalized fraction of body_scale).

---

## Aggregation strategy

Multiple pitches per pitcher are aggregated with **median + IQR** (not
mean/STD). Rationale: pose estimates are noisy; median is robust to the
same outliers that motivated choosing it over mean; IQR is the natural
spread measure alongside a median.

---

## Validated results

### Framber Valdez (LHP, high leg kick) — `run_10pitch.py`
- 10 pitches processed
- High leg kick correctly captured (wide SANE bounds needed for `peak_to_plant`)
- LHP crossfire shows as `stride_norm < 0`

### Tanner Houck (RHP, crossfire) — `run_crossfire_test.py`, 2026-08-28
- 10/38 candidates succeeded (Savant clips often batter-focused or wrong angle)
- **stride_norm median = -0.178** (IQR 0.183) — consistently crossfire ✓
- rubber_pos = all None (coming_set not detected in these clips)

| Metric | Median | IQR |
|--------|--------|-----|
| plant_t | 2.830s | 0.190 |
| release_t | 3.032s | 0.138 |
| p2r | 0.268s | 0.151 |
| delivery_s | 1.057s | 0.102 |
| peak_to_plant | 0.512s | 0.176 |
| stride_norm | **-0.178** | 0.183 |

### Chris Sale (LHP) — `test_rubber_sale.py`, chalk method
- `position = +0.96 (3B SIDE)` — Sale works the 3B side, confirmed ✓
- method: chalk

---

## Timing profile (Houck run, median per pitch, n=10)

| Stage | Median | Share |
|-------|--------|-------|
| pose_extraction_yolo | 23.8s | **47%** |
| followthrough_cutoff_probe | 9.1s | 18% |
| pitcher_detection_yolo | 8.0s | 16% |
| shot_selection_probe | 3.0s | 6% |
| delivery_crop | 1.1s | 2% |
| shot_segmentation | 0.6s | 1% |
| frame_decode | **0.3s** | <1% ← pre-crop working |
| **WALL TIME** | **45.6s** | |

The pre-crop reduced `frame_decode` to essentially zero. The remaining
bottleneck is YOLO inference on ~60 frames (pose) + the two probe passes
(cutoff, shot selection). Those require batched GPU inference to significantly
improve — CPU-bound on this machine.

---

## Known issues / open TODOs

| Priority | Issue | Notes |
|----------|-------|-------|
| HIGH | rubber_pos always None | coming_set not in most Savant clips; needs fallback to foot_plant frame as rubber anchor |
| HIGH | ~75% pitch rejection rate | Clips that fail all-phases detection still burn 35–50s. Filter by pitch_type or add a fast pre-screen before escalation |
| MED | stride_norm convention | Positive = 1B side for both handedness; caller must flip sign for LHP to get "crossfire" in a handedness-agnostic feature |
| MED | Phase grid labels | Phase grid is built from coming_set but that's often missing; grid may only show 4 of 5 phases |
| LOW | rubber_pos magnitude | Sale result (+0.96) is plausible; some clips may drift higher (~1.5) because chalk detection hits inner arc. Not validated on more than 1 pitcher |
| LOW | Escalation path slow | MediaPipe escalation ignores pre-crop benefit; runs on full shot window even when crop was available |

---

## How to run a new pitcher

```python
# In run_crossfire_test.py (or duplicate it):
PITCHER_NAME = "lastname"
PITCHER_ID   = 123456   # MLBAM player ID from Baseball Savant
```

Then:
```
bcv_venv311/Scripts/python.exe run_crossfire_test.py
```

Outputs go to `deception_v4/out_crossfire_test/`:
- `results.json` — per-pitch values + timings
- `phase_grid.png` — storyboard of first successful pitch
- `landmarks/` — YOLO JSON + per-phase frame PNGs

---

## Key constants to know

| Constant | File | Value | Why |
|----------|------|-------|-----|
| `SHOT_MIN_HIT_RATE` | d4_extract.py | 0.60 | Raised from 0.50 to fix Skenes wrong-skeleton bug |
| `FINE_STRIDE` | d4_extract.py | 2 | Every other frame for pose |
| `SANE["delivery_time_s"]` | d4_phases.py | (0.15, 2.80) | Wide for high-kick LHP |
| `PRE_BUFFER_S` | d4_crop.py | 4.0s | Enough for set + full leg kick |
| `CHALK_MIN` | d4_rubber.py | 160 | All BGR channels must exceed this |
| `RUBBER_HALF_FRAC` | d4_rubber.py | 12/72 | Rubber 12" / body 72" |
