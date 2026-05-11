# Evaluation Report — Generative AI-Based Robotic Kitting

Aggregate statistics generated from `eval.db`. Drop these tables into the *Results & Discussion* chapter as-is; cross-reference each section with the corresponding raw `*.csv` file for the source data.

## 1. Perception accuracy (vs USD ground truth)

Sample size: **32 scan(s)** logged in `perception_events.csv`.

| Metric | N | Mean | Median | Stdev | Min | Max |
|---|---|---|---|---|---|---|
| Precision | 32 | 0.969 | 1.000 | 0.177 | 0.000 | 1.000 |
| Recall | 32 | 0.547 | 0.500 | 0.195 | 0.000 | 1.000 |
| F1 | 32 | 0.688 | 0.667 | 0.168 | 0.000 | 1.000 |
| Label accuracy | 32 | 0.969 | 1.000 | 0.177 | 0.000 | 1.000 |
| Mean IoU | 32 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Grounding error (mm) | 32 | 64.4 | 66.2 | 14.1 | 0.0 | 74.8 |
| VLM latency (s) | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |


## 2. Grasping success and alignment

Sample size: **50 pick attempt(s)** logged in `pick_events.csv`. Pick success rate: **68.0 %** (34 of 50). Tip-sensor contact-stop fired in **0 (0.0 %)** of attempts.

| Metric | N | Mean | Median | Stdev | Min | Max |
|---|---|---|---|---|---|---|
| End-of-descent XY drift (mm) | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| End-of-descent Z drift (mm) | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Pick cycle time (s) | 50 | 133.47 | 158.52 | 70.53 | 30.62 | 333.63 |


## 3. Place verification (Camera_Kit VLM)

Sample size: **34 place attempt(s)** (0 verified by Camera_Kit VLM). Verified-success rate: **0.0 %** (0 of 0 verified). Overall (treating un-verified as success): **0.0 %**.

| Metric | N | Mean | Median | Stdev | Min | Max |
|---|---|---|---|---|---|---|
| VLM confidence | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Place cycle time (s) | 34 | 57.95 | 57.38 | 3.49 | 48.71 | 62.06 |


## 4. End-to-end task performance

Sample size: **30 operator command(s)**. End-to-end success rate: **66.7 %** (20 of 30 fully completed).

| Metric | N | Mean | Median | Stdev | Min | Max |
|---|---|---|---|---|---|---|
| Per-task completion ratio | 30 | 0.354 | 0.000 | 0.449 | 0.000 | 1.000 |
| End-to-end time (s) | 30 | 415.6 | 338.9 | 204.3 | 127.3 | 943.5 |


## 5. Failure-mode breakdown

Of **16 failed pick(s)**, the breakdown by failure cause is:

| Error code | Count | Fraction |
|---|---|---|
| no_part_visible_wrist | 15 | 93.8 % |
| descend_failed | 1 | 6.2 % |

Error codes correspond to specific abort points in the pick pipeline:
- `mechanical_close_failed` — gripper closed but the finger angle did not reach target. Indicates the descent stopped against the bin floor / divider before the part.
- `empty_gripper_vlm` — post-grasp wrist VLM confirmed empty fingers (high-confidence visual signal).
- `descend_failed` — Cartesian descent IK rejected the target pose. Indicates the planned XY is unreachable.
- `no_part_in_depth` — the depth-analysis VLM check (disabled by default) rejected the approach.
- `place_not_verified` — Camera_Kit VLM did not see the part inside the kitting tray after place.



## 6. Raw CSV exports

All per-event raw data has been exported alongside this report — drop into Excel / pandas for deeper analysis or thesis-figure plots:

| File | Rows |
|---|---|
| perception_events.csv | 32 |
| pick_events.csv | 50 |
| place_events.csv | 34 |
| task_events.csv | 30 |
| failure_modes.csv | 2 |

