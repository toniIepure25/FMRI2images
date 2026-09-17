# Engineering pilot execution protocol — DRAFT (≤2 participants; excluded from confirmatory N)

> Engineering only. Pilot participants are **permanently excluded** from the confirmatory replication and **never pooled**. Pilot neural outcomes may **not** tune M, T, D, r, the estimator, the Q_MON formula, thresholds, or recovery criteria.

**Purpose (engineering only):** verify task comprehension, scanner trigger synchronization, scanner timing, event logging, image presentation, response box, preprocessing, registration, ROI creation, and finite Q_MON — before confirmatory acquisition.

**Procedure:** run the frozen perception + imagery tasks on ≤2 pilot volunteers under approved ethics/site permissions; produce BIDS + events; execute the frozen preprocessing and Stage-A software (calibration + Q_MON) with the held-out access guard active.

**Go/No-Go (engineering):** see `pilot_go_no_go.json` — GO requires all expected files produced, trigger alignment correct, event onset mapping verified, no missing stimulus IDs, exact repeat indexing, perception-response derivation runs, ROI mapping runs, Stage-A executes, Q_MON finite, and the held-out guard demonstrably blocks outcome access. **No scientific metric is a Go criterion.**

**Status:** NOT_RUN (no scanner / no acquisition authorization).
