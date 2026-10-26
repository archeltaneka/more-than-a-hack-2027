# More Than a Hack 2027 — Phase 1: Issues #1–#3 TL;DR

**Scope:** Barcelona vs Athletic Club possession inference; public 5-minute ground truth only for scored diagnostics. Full-match analyses are descriptive, not ground-truth evaluations.

## Issue #1 — Baseline possession-boundary diagnostics

**Question:** Where does the official `baseline_smooth50` fail, especially at possession boundaries?

**Findings and evidence**
- Public `spellF1 = 0.325581`, verified against the official evaluator at all five temporal-IoU thresholds; at IoU 0.5, **7 TP / 19 FP / 10 FN**.
- **17** ground-truth possessions versus **26** predicted possessions. **5/17** true possessions overlap multiple same-team predicted spells (candidate fragmentation); **3** predicted spells overlap multiple same-team true spells (candidate merges).
- **164** ground-truth live frames were incorrectly predicted `D`; **all 164 lacked ball XY**. Across all **6,031** live frames, **391** lacked ball XY.
- The baseline also showed missed or shifted transitions. Boundary offsets and overlap flags are diagnostic, not proof of causal mechanisms.

**Steps taken**
1. Reproduced the baseline and official public scoring.
2. Implemented interval matching, split/merge candidates, transition diagnostics, and GT-versus-prediction timelines.
3. Overlaid ball visibility and selected missing-ball handling as a hypothesis for deeper investigation.

**Outcome:** A reproducible error-analysis pipeline, with a clear lead on missingness but insufficient evidence to explain all fragmentation.

## Issue #2 — Tracking gaps and game-state reliability

**Question:** Does a missing ball mean dead ball, and can geometry or Smart Tagging distinguish stoppages?

**Findings and evidence**
- Public window: **6** missing-ball intervals. **391 live frames** had missing ball XY, while **975 genuinely dead frames** still had visible ball XY.
- **2 false-stoppage intervals / 164 frames** versus **6 missed-stoppage intervals / 1,155 frames**. All **4** true `D` intervals had some visible-ball observations; **2** were entirely visible.
- **4/5** candidate-split true possessions had **no missing-ball frames**. Thus occlusion cannot explain most split candidates.
- Median nearest tracked-player distance: **1.64 m** (`H`), **1.67 m** (`A`), **2.89 m** (`D`), computed on finite geometry only; distributions overlap.
- Automatic set-piece phases occurred within ±5 seconds of **3/6** missed-stoppage intervals. Useful context, but incomplete coverage and uncertain precision.
- Within published gameplay halves, there were **150** missing-ball gaps (median **10.72 s**), underscoring the importance of gap handling at match scale.

**Steps taken**
1. Characterized missing-ball intervals and true live/dead composition in the public window.
2. Audited player–ball geometry, gap endpoints, nearby Smart Tagging, and examples of false and missed stoppages.
3. Extracted full-match gap distributions without claiming full-match accuracy.
4. Formed three hypotheses: separate visibility from game state; use temporal team-control evidence; use set-piece tags as a noisy restart prior.

**Outcome:** Rejected the simplistic `missing ball = dead ball` assumption. Identified **dead-ball detection even with a visible ball** and **unstable team-control inference** as separate problems.
