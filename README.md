# The Whole Nine
**The Whole Nine... Innings. Every pitcher and hitter metric you actually check, none of the fluff.**


<table>
<tr>
<td><img width="524" height="468" alt="Miz Dash Photo" src="https://github.com/user-attachments/assets/08c277ad-1381-4b97-b2d8-fd9a344b2975" />
<td><img width="464" height="467" alt="Yordan Dash Photo" src="https://github.com/user-attachments/assets/3f3749bd-a672-43b9-977c-3e85500896f3" />
</tr>
</table>

## What it Does

1. Player pages for pitchers including average location by pitch, percentile rankings, counting stats, movement plot, approach angles, last 5 outings, and results by pitch type.
2. Player pages for hitters including heat zones, spray charts, swing graphics, counting stats, last 5 games, and results by pitch type.
3. Four proprietary ML pitcher grading models examining command, stuff, deception, and pitch tunnelling
4. Season-by-Season and Career views available as well as trend charts over any time period
5. Live playoff bracket and playoff roster projections as well as WC race tracking
6. A player comparison tool
7. A probable pitchers page showing the next day's games
8. An outing tool allowing in-depth analysis of a single pitcher outing
9. Prospect scores, rankings, and analysis league-wide and team specific

## Fun Case Studies 
1. Stuff+ flagged Aaron Ashby as the best Curveball in baseball (Stuff+: 124) before his historic scoreless playoff run
2. Stuff+ flagged Tatsuya Imai as a bottom 4 pitcher (91 Stuff+). Despite a 78th percentile VAA Imai's intriguing fastball got crushed (.815 OPS) due to below average total movement, poor location+ (92) and because it's his only reliable zone pitch. He struggled to miss any bats with his 4S-FB.
3. Jeff Hoffman ran a near 4.00 ERA despite a 98th percentile K rate (over 100 K's). Hoffman struggled with command (96 Location+) and Holding Runners this year (1st percentile). In addition, he ran a .352 BABIP across 65IP which should be expected to mean revert next year. While his BABIP suggests he was getting extremely unlucky, I'd argue hitters earned a lot of their success by sitting on his (theoretically) elite tailing fastball in-zone and managing to barrel the ball for hard contact. 
4. 2026 Michael Harris was one of the best young players in baseball despite struggling in 2024 & 2025. However, there don't seem to be many changes in underlying profile YOY for MH-II. He remained consistently bottom of the league in chase% (1st percentile) and BB% (2nd Percentile), relying on infrequent high quality contact instead. His 2026 results seem driven by a modest increase in Pull Air% (~5pp) improving quality of contact metrics, in 2027 he can be expected to return to career norms (<10th percentile) Pull Air % and his offensive production will likely follow.

## The Models

1. **Stuff+:** a lightGBM model trained on pitch level data from 2015 - 2026 (excluding 2020). Feature set includes effective speed, release_spin_rate, spin_axis, hb, ivb, vaa_residual, and seperation characteristics from each major fastball as well as a handful of categorical league context variables.

2. **Location +:** a series of lightGBM models trained to predict at-bat outcomes (whiff, CS, foul, swing, or BIP) it then multiplies these weights by historical run values in each context to determine the value of a certain pitch in a certain location. Aggregate this score to the player-year and you have a Location+ score.

3. **Deception+:** a lightGBM model trained on pitch level data from 2021-2025 (arm angle not available pre 2021). The training features include release position in the x-y-z space, arm angle, and extension. This model was deliberately restricted to delivery only characteristics so as not to overlap with the other models in the grading framework.

4. **Tunneling+:** Takes 3 points in pitch flight and averages each pitch pair distance at each point. Each average distance is z-scored, and placed onto a "+" scale. These 3 scores are then equal-weighted averaged into the overall tunnelling pair composite score. 

## Architecture

**Backend:** FastAPI with 21 routers each serving one of the clickable dashboard pages, mounted onto one larger application.

**Frontend:** Javascript served as static files through the same FastAPI, allows both backend and frontend to originate from the same place for simplicity.

**Data:** SQLite database, local in dev, and a mounted persistent file for the live site. TTL caching is employed for heavy computations like yearly percentage pools and model scores

**Models:** 3 LightGBM models and a hand-built composite model with their own feature sets, governance rules, and qualifying validation statistics

**Pipeline:** sheduled pull scripts (pulls/, orchestrated by api/pull.py) sourcing from Savant and MLB StatsAPI, feeds the SQLite database

**Deployment:** Dockerized file deployed to Render (render.yaml) with a persistent disk database and an auto generated admin token. 


## Limitations

- Deception+ feature set is relatively thin due to extremely limited pitcher delivery metrics available publicly. This mean that ERA and xFIP correlations appear relatively weak and the model struggles to indepedently predict future results for a player
-  Multiple iterations of the tunneling model agree broadly on the worst tunnelers in the league, but disagree starkly on the best. Suggests that there might be a larger floor for "Poor tunnelling" than there is a ceiling for "Elite tunnelling" and limits the potential signal capturable by ANY model
- Certain large computations (percentile pools, SHAP contributions, Model scores) take significant time to pre-load and cache, so requests made to a cold server can be slow to process. 

## Future Work

1. Computer Vision (YOLO) delivery mechanics pipeline in development to gather additional data for Deception+ inputs
2. 3D Tunnelling+ visualizer program in development. Showing all pitches, color-coded, with trajectories to visualize pitch divergence on the path to the plate, at commit point, and final destinations
3.  Tunnelling+ expansion into a broader "Pitch-type optimization" model giving suggested optimal pitches and locations based on game context, at-bat situation, pitcher/hitter characteristics, and previous pitch outcomes (early/late/over/under). 

## Repo Structure
```
mlb-dashboard/
├── analytics/             [core model + data logic]
├── api/                   [FastAPI backend, routers]
├── config/                [model policy, thresholds]
├── data/                  [local data, DB files]
├── deploy/                [deployment config]
├── diagnostics/           [live validation artifacts]
├── docs/                  [methodology + architecture docs]
├── frontend/              [vanilla JS/HTML site]
├── models/                [trained model artifacts]
├── parquets/              [cached score data]
├── pulls/                 [data-pull scripts]
├── tests/                 [test suite]
├── tools/                 [maintenance scripts]
├── training/              [model training artifacts]
├── Dockerfile             [container build]
├── README.md              [this file]
├── pytest.ini             [test config]
├── render.yaml            [Render deploy config]
├── requirements.txt       [pinned dependencies]
├── schema.sql             [database schema]
└── start.sh               [container start script]
```

## About

This project is part of an ongoing portfolio exploring data analytics, machine learning, statistics, and my passion for the game of baseball. I'm a third-year Economics student at the University of Toronto (Focus in DA, Statistics Minor). Feel free to reach out if you'd like to talk shop or have feedback on the analysis.

josh.scott.tay@gmail.com | [LinkedIN](https://www.linkedin.com/in/joshua-taylor-86450a32a/)
