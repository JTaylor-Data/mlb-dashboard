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

## The Models

1. **Stuff+:** a lightGBM model trained on pitch level data from 2015 - 2026 (excluding 2020). Feature set includes effective speed, release_spin_rate, spin_axis, hb, ivb, vaa_residual, and seperation characteristics from each major fastball as well as a handful of categorical league context variables.

2. **Location +:** a series of lightGBM models trained to predict at-bat outcomes (whiff, CS, foul, swing, or BIP) it then multiplies these weights by historical run values in each context to determine the value of a certain pitch in a certain location. Aggregate this score to the player-year and you have a Location+ score.

3. **Deception+:** a lightGBM model trained on pitch level data from 2021-2025 (arm angle not available pre 2021). The training features include release position in the x-y-z space, arm angle, and extension. This model was deliberately restricted to delivery only characteristics so as not to overlap with the other models in the grading framework.

4. **Tunneling+:** Takes 3 points in pitch flight and averages each pitch pair distance at each point. Each average distance is z-scored, and placed onto a "+" scale. These 3 scores are then equal-weighted averaged into the overall tunnelling pair composite score. 

## Architecture

**Backend:**


## Fun Facts 



## Limitations/Future Work



## If you want to run locally



## About

