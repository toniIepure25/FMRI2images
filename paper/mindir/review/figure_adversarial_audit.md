# Phase 11 — Figure adversarial audit

For each proposed main figure: exact claim supported? essential? could it mislead? participant-level points?
does it hide N=8? could schedules look like large N? are oracle/ceiling defined? development vs held-out clear?

## Global requirements (apply to every panel)
- **Show participant-level points (all 8) wherever scientifically useful**, with the group median overlaid.
- **Never** plot hundreds of schedules/folds in a way that visually implies a large sample. Where schedule-level
  quantities are shown, aggregate to one point per participant first, or clearly mark schedules as within-
  participant repeated measures (not independent N).
- Define **oracle** (native oracle) and **ceiling** (R111 full-resource) in every caption that uses TOTAL/FCF.
- Mark **development-cohort** on every data figure; mark held-out vs training where relevant.
- Cite the source artifact + commit SHA in each caption (from `evidence_table.csv`).

## Panel-by-panel
- **F1 (conceptual).** No data → no misleading-N risk. Essential (frames the decomposition). Ensure it is
  clearly labeled "schematic, not data."
- **F2 (perception ceiling vs native oracle; O2.5/O2.6).** Show 8 participant points per ROI, not just bars.
  Define ceiling/oracle. Risk: bar-only plots hide N — require dots.
- **F3 (subject-specificity; O2.10/O2.11/O2.13).** Essential. Show donor-transfer per participant vs null; show
  prediction failure per participant. **Caption must say "subject-specific under the tested representation and
  transfer procedure,"** not "unique geometry." Risk of overinterpretation → caption guard.
- **F4 (M×T calibration grid; O2.9/O2.12).** The grid is the paper's most attackable figure. Highlight M4T2 /
  8-observations, but the caption **must** carry the bounded wording (smallest prospectively tested common
  burden under the frozen estimator; not a universal minimum) and reference F5 (fragility). Show participant
  spread for the M4T2 cell, not only the median.
- **F5 (schedule fragility + Q_OUT; O2.14/O2.15).** **Highest misleading-N risk.** Panel A: plot the coverage
  distribution as one value per participant (8 points), not one point per schedule. Panel B: plot participant
  median ρ (8 points) with the participant as the unit; if individual schedules are shown, mark them explicitly
  as within-participant repeated measures. Caption: "participant is the inferential unit; schedules are not."
- **F6 (mechanistic constraints; X1–X4).** Must be headed **"Exploratory."** Show 8 participant points per
  panel. Panel D must show both the positive (ANISO 8/8) and the negative (MODE_GAP→margin, opposite sign) so
  the figure cannot be read as "drift explains failure."

## Verdict
No figure needs new inference. Two figures (F4, F5) carry real misleading-N / overclaim risk and require the
bounded captions and participant-level rendering above. Recommend adding a one-line "how to read N=8" note to
the figure legend of F4/F5.
