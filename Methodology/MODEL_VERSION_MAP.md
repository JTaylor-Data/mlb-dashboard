# Model Version Map — 2026-10-04

## Active Models

### Stuff+ v4

| Field | Value |
|---|---|
| Active version | v4 |
| Model hash (SHA-256 short) | 0addb8e9 |
| Full hash | 0addb8e96ea438ec196b3addc19fd350f1d76e241a9291d3719773f2b9766c28 |
| Trained at | 2026-10-02 14:54:17 |
| Training seasons | 2016-2019, 2021-2026 (10 seasons; 2020 excluded from training) |
| Scored seasons | 2016-2026 (11 seasons; 2020 scored by model despite not trained on it) |
| Status | provisional (2026 in-season) |
| Branch built on | stuff-v4 → stuff-v4-live → closeout-blend |
| Tags | post-stuff-v4, post-stuff-v4-live, post-closeout-blend |
| Retired versions | v1, v2, v3 (not in models/ directory) |
| Notes | LightGBM pitch-level model, empirical-Bayes shrinkage, 200-pitch leaderboard min |

### Location+ v3

| Field | Value |
|---|---|
| Active version | v3 |
| Model hash | see models/location_plus_v3_meta.json |
| Trained at | 2026-10-02 20:11:27 |
| Training seasons | 2015-2019, 2021-2025 (2020 excluded; scored by 2019/2021 average) |
| Scored seasons | 2015-2026 (12 seasons) |
| Status | live |
| Branch built on | location-v3 → closeout-blend |
| Tags | post-location-v3, post-closeout-blend |
| Retired versions | v1 (location_plus_v1_*.txt), v2 (location_plus_v2_*.txt) — model files kept per Rule 3 |
| Notes | 5-component LightGBM ensemble (xcs/xswing/xwhiff/xfoul/bip), 300-pitch minimum |

### Deception+ v2b

| Field | Value |
|---|---|
| Active version | v2b |
| Model files | models/deception_plus_v2b_fold[0-4]_rep[0-2].txt (15 fold files) |
| Scores parquet | diagnostics/deception_v2/deception_v2b_scores.parquet |
| Trained at | 2026-09-15 22:34:59 |
| Features | release_pos_x_as, release_pos_z, release_extension, arm_angle |
| Training seasons | 2015-2026 (but arm_angle only reliable 2021+) |
| Scored seasons | 2021-2026 (arm_angle required; earlier seasons not scored) |
| Status | limited_evidence |
| Branch built on | deception-v2 → closeout-blend |
| Tags | post-deception-v2, post-deception-v2-closeout, post-closeout-blend |
| Retired versions | v1 (removed), v2 (interim — models/deception_plus_v2_*.txt kept per Rule 3) |
| Config | DECEPTION_ACTIVE_VERSION = "v2b" in config/model_policy.py |
| Notes | season dropped from v2→v2b (diff=-0.0012, below 0.03 margin = inconclusive_means_drop). |
|       | `/api/deception/meta` returns 404 — v2b uses different meta format. `/api/deception/report` works. |

### Tunneling+

Left as is; needs a full rethinking; not maintained.

Router URLs say `/v3/` but live scorer is `analytics/tunneling_v2_model.py` (promoted 2026-09-21).
Model files: `models/tunneling_v3.txt`, `models/tunneling_plus_v2_*.txt`.
Not touched during this merge. Byte-identity verified.

---

## Retired / Superseded Versions on Disk

| File(s) | Reason kept |
|---|---|
| models/deception_plus_v2_*.txt | Rule 3 (never discard model artifacts); v2 interim before season-drop |
| models/location_plus_v1_*.txt, v2_*.txt | Rule 3; v1/v2 superseded by v3 |
| models/tunneling_v1_pitchers_*.json | Rule 3; v1 pre-model lookup tables |
| models/tunneling_plus_v2_*.txt | Rule 3; v2 pre-v3 model |
