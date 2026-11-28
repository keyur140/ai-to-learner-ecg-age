# Protocol

*AI-to-learner knowledge transfer study (ECG age), version 1.0.1 (preregistration frozen 2026-11-27). Author: KEYUR SHAILESHKUMAR PATEL (ORCID https://orcid.org/0009-0008-5261-043X). Conducted independently; not an official Brunel University London project.*

The exact order of work, from an empty folder to a reported result. Each step names the command, what it produces, and what must be true before going on. The preregistration (`prereg/preregistration.md`) says *what* is tested; this file says *how* the study is carried out.

## Phase 0: before preregistration (synthetic validation done for v0.2.0; real-data pilot done for v0.3.0)

| Step | Command | Produces | Go on only if |
| --- | --- | --- | --- |
| 0.1 | `pip install -r requirements.lock` (or `docker build`) | exact environment | install succeeds |
| 0.2 | `pytest -q` | unit tests, smoke run twice, byte-identical check | all tests pass |
| 0.3 | `python run_study.py smoke` | a full run on fake data | 0 self-checks fail |
| 0.4 | `python run_study.py validate` | `validation/` with VALIDATION_REPORT.pdf | every world meets its pre-stated expectations, or each miss is explained in ISSUES.md |

Steps 0.1 to 0.4 open no PTB-XL file.

**Step 0.5, exploratory pilot on real data (decision D7, which replaced D3 on 2026-11-07).**

| Step | Command | Produces | Rules |
| --- | --- | --- | --- |
| 0.5a | `python run_study.py check-data` | PTB-XL downloaded, unpacked and checked against PhysioNet's SHA256SUMS | status must be `ok` |
| 0.5b | `python run_study.py run --pilot 30` | a run folder labelled EXPLORATORY PILOT | Folds 1-9 only. Fold 10 and the held-out device are sealed by the code; self-check C15 must show 0 sealed ECGs used. |
| 0.5c | Write up what was seen | `pilot/EXPLORATORY_PILOT_REPORT.md`, `prereg/PILOT_DISCLOSURE.md` | Every problem goes into ISSUES.md; every design change is a dated decision of the author in DECISIONS.md. |
| 0.5d | Rerun 0.2 and 0.3 after any code change | tests and smoke run | all tests pass, 0 self-checks fail |

The pilot was run from 2026-11-07 to 2026-11-12. Its results are exploratory and are never combined with the confirmatory run. The confirmatory test data (fold 10, the held-out device) have still not been opened.

**After any design change that follows from the pilot, the synthetic validation (0.4) is repeated on the final code before freezing**, because the validation report attached to the preregistration must describe the code that is frozen.

## Phase 1: freeze and register

| Step | Command | Produces | Go on only if |
| --- | --- | --- | --- |
| 1.1 | Fill in the [bracketed] items in the preregistration | final text | author has read every section |
| 1.2 | `python scripts/pin_models.py` | exact model commits in `configs/study.yaml` | all 3 models resolve (HF token, licences accepted) |
| 1.3 | `python run_study.py freeze` | `prereg/LOCK.json` + 4 printed fingerprints | - |
| 1.4 | Zip the whole folder; record its SHA-256 | package zip | - |
| 1.5 | Submit on OSF: preregistration text, LOCK.json, fingerprints, validation report, package zip | OSF registration with time stamp | registration is public or embargoed with a time stamp |

## Phase 2: confirmatory run

| Step | Command | Produces | Notes |
| --- | --- | --- | --- |
| 2.1 | `python run_study.py run` | `runs/<id>_study/` | Checks the lock first; downloads PTB-XL; checks it against PhysioNet SHA256SUMS; stops on any mismatch |
| 2.2 | (if interrupted) `python run_study.py run --resume runs/<id>` | same folder | same seeds; finished learners are never re-drawn |
| 2.3 | `python run_study.py verify runs/<id>` | ok / list of changed files | must say ok |
| 2.4 | Read Section 3 of REPORT.pdf (self-checks) | - | any FAIL goes to ISSUES.md before any number is quoted |
| 2.5 | Read the preconditions (teacher AUROC, pre-test, gate) | - | a failed precondition is reported as such, not as evidence about transfer |

The study runs in this order inside one command: cohort and splits, teacher, cues and lessons, saliency, Experiment B (4 groups x 1,002 nets), Experiment A main run (gate and probe per model, then 4 groups x 1,002 learners), Experiment A blinded run (gate per model, 4 x 1,002), analyses, self-checks, reports, manifest and SHA-256 list.

## Phase 3: reporting

| Step | What | Rule |
| --- | --- | --- |
| 3.1 | Upload the run folder (or its SHA256SUMS.txt and manifest.json) to OSF | hashes must match the run |
| 3.2 | Write up H1 per experiment exactly as the report states it | supported only if p < 0.05 and difference of at least 3 points |
| 3.3 | Report H2-H7 with Holm p-values; every null with its MDE | no new hypotheses presented as confirmatory |
| 3.4 | List every change since freezing in DEVIATIONS.md | with reason and whether results had been seen |
| 3.5 | Anything not in the preregistration is labelled exploratory | - |

## Who does what

The author makes every design and reporting decision (see DECISIONS.md). Software does the computing; no human labels or judges any learner answer. An AI assistant helped write the code and drafts under the author's direction; this is stated on every document.
