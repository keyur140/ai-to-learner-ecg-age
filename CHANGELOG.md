# Changelog

*AI-to-learner knowledge transfer study (ECG age). Author: KEYUR SHAILESHKUMAR PATEL (ORCID https://orcid.org/0009-0008-5261-043X).*

## 1.0.1 (2026-11-27): one outdated phrase corrected, re-frozen

- P25: section 5 of the preregistration, "Data exclusion", still said "the gate and collapse rules", although since D15 (version 0.4.1) the flat rule decides which models count, as sections 4 and 5 already say. Changed to "the gate and flat rules". No rule, number or code changed. Found while mapping the plan onto the OSF form, before OSF submission and before any confirmatory data were seen. The author chose to fix it and re-freeze (D20).
- `prereg/LOCK.json` rewritten with `python run_study.py freeze --force`. The 1.0.0 lock is kept in `freeze_audit/LOCK_v1.0.0_superseded.json`.

## 1.0.0 (2026-11-27): preregistration frozen

- The author confirmed `prereg/PILOT_DISCLOSURE.md` and froze the preregistration (decision card, 2026-11-27). The version line now reads "Version 1.0.0, frozen on 2026-11-27, before any confirmatory data were seen"; no other wording changed from the 0.5.2 draft.
- `prereg/PILOT_DISCLOSURE.md` is now kept inside `prereg/` in the working package, so it is hashed into `prereg/LOCK.json` together with the plan.
- Learner models pinned to exact Hugging Face commits in `configs/study.yaml` (`experiment_llm.revisions`).
- New checks before freezing: every fixed number and rule in the plan compared with the config and code (70 of 70 agree).
- `prereg/LOCK.json` written by `python run_study.py freeze`. No confirmatory data (fold 10, held-out device) have been opened.

## 0.5.2 (2026-11-20)

- P24: language-model results are saved at most every 30 seconds (and after the last learner) instead of after every learner. The synthetic validation had become very slow because the whole file was rewritten after each fast mock learner. A real model still saves after every learner.
- The synthetic validation was repeated on this code (run `20261120T132404Z_validation`, 49 minutes): 26 of 31 expectations met, as before, all 16 applicable self-checks pass in every world, and two runs from scratch are byte-identical. Section 7 of the preregistration draft now reports these numbers. Report and summary in `validation/`; the full item-level files are in a separate zip.

## 0.5.1 (2026-11-20)

- D17: the confirmatory run stays in bfloat16 with the float32 answer read-out. No code change.
- D18 (issue P20): the prior-knowledge probe asks each question twice with the answer letters swapped; a direction counts as known only if both answers agree. New fields in `gate_llm.json`: `probe_known_direction`, `probe_p_higher_A_first`, `probe_p_higher_B_first`. The report also shows how many measurements a model answered inconsistently. Preregistration section 4 updated (the author chose this option on 2026-11-20).
- Tests: 12 (new: the probe is consistent only when both orderings agree).

## 0.5.0 (2026-11-15): preregistration draft brought up to date

- `prereg/preregistration.md` now carries the wording approved by the author on 2026-11-15 (items A1 to A11 of `pilot/PREREG_CHANGES_PROPOSED.md`): existing-data statement, five teachers and cue consensus, neutral placebo, self-checks C1-C17, bias-corrected primary score, flat rule, Experiment B secondary, and a new section 9. It is a **draft, not frozen**. One sentence was added to section 9 beyond the preview: the list of decisions D12 to D16.
- D16: 24 practice cases are kept in Experiment A. No code change.
- No change to the study code since 0.4.1.

Still to do before freezing: the author's choices on float32 runs (P23), the probe (P20) and the Hugging Face token; pin the model commits; repeat the synthetic validation on the final code and replace the numbers in section 7; fill in the bracketed items.

## 0.4.1 (2026-11-14): three more decisions after the pilot

- D13: Experiment B is **secondary** (`experiment_nets.role: secondary`); it runs as designed and the report says so.
- D14: H1 (AI cues beat feedback-only practice by at least 3 points) stays the primary hypothesis. No code change.
- D15: the flat rule built in 0.4.0 is confirmed in place of the collapse rule.

## 0.4.0 (2026-11-14): first changes after the pilot, before preregistration

Design (decided by the author, see DECISIONS.md):
- **Bias-corrected accuracy is the primary score of Experiment A** (D12; issue P19): each learner's items on a test are split at that learner's own median P(older); ties get half credit. New config value `experiment_llm.primary_score`. The raw score and each learner's AUROC are saved and reported as robustness. Experiment B is unchanged.

Proposed defaults that follow from D12 (await the author's confirmation, see `pilot/PREREG_CHANGES_PROPOSED.md`):
- The collapse flag is still reported but no longer removes a model (`collapse_excludes_model: false`). A **flat rule** replaces it: a model does not count if more than half of its learners have more than half of their post-test items tied at the median (`flat_tie_share`, `flat_model_fraction`).

Fixes and improvements:
- P23: the two answer letters are read in float32 from the last hidden state (`experiment_llm.answer_precision: float32`), because bfloat16 logits gave only about 13 distinct probabilities per 40-item test and many ties.
- The prompt cache is kept between a learner's tests (the lesson is read once, not three times): about a quarter less time per taught learner. `AI2HUMAN_REUSE_CACHE=0` switches it off.
- Learner score columns are always recomputed from the item file at the end of a run. Self-check C6 recounts both scores by hand; C12 also checks which models count.

Exploratory follow-up on folds 1-9 (`pilot/PILOT_FOLLOWUP.md`): the pilot re-scored with the package's own analysis; a practice-size sweep for Experiment B (P16).

Not yet done: `prereg/preregistration.md` still has the 0.2.0 wording; the synthetic validation has not been repeated on this code.

Tests: 11 pass (new: the bias-corrected score ignores letter bias).

## 0.3.0 (2026-11-12): changes from the exploratory real-data pilot, before preregistration

An exploratory pilot was run on real data before preregistering (decision D7, replacing D3): PTB-XL folds 1-9 only; fold 10 and the held-out device stayed sealed. Report: `pilot/EXPLORATORY_PILOT_REPORT.md`. What the preregistration must say about it: `prereg/PILOT_DISCLOSURE.md`.

Design (decided by the author, see DECISIONS.md):
- **Cue consensus across five teachers** (D9): five teachers are trained (seeds master seed + 1000k); the cue selection runs on each one's guesses; a feature is kept if at least 3 of 5 chose it. Teacher 0 is the teacher used everywhere else. New config values `cues.n_teacher_seeds`, `cues.consensus_min`. Self-check **C16** recounts the votes.
- **Neutral placebo** (D10): the scrambled lesson is re-drawn with a fixed seed until its rules, as a tally, score 45-55% on the cue fold. New config values `cues.scrambled_band`, `cues.scrambled_max_draws`. Self-check **C17** recomputes the score.
- New command option `run --pilot N`: protected exploratory pilot (seals the test fold and held-out device; self-check **C15** confirms no sealed ECG was used).

Fixes of problems found in the pilot (ISSUES.md):
- P11: input scaling summed in 64-bit, so a fresh read and a cached read give the same teacher. Before, the same seed could give different teachers and different cues.
- P12: `check-data` downloads and unpacks PTB-XL if needed.
- P13: language-model results are saved after every learner.
- P17, P18: long prompts no longer run out of memory: the shared prompt is fed in chunks, only the last position's logits are computed, and cache copies are capped by a memory budget (`AI2HUMAN_PREFILL_CHUNK`, `AI2HUMAN_KV_BUDGET_GB`).
- P21: README run times are measured, not guessed.

Documentation only: P22 (in pilot mode the seal is applied after the feature table is loaded; stated exactly in the disclosure). `prereg/preregistration.md` itself is **not yet changed**: the new wording for D7, D9 and D10 is previewed in `pilot/PREREG_CHANGES_PROPOSED.md` for the author's approval.

Open after the pilot (author's decisions before freezing): outcome measure and collapse rule for Experiment A (P19), Experiment B's ceiling (P16), the prior-knowledge probe (P20).

Tests: 10 pass (two new: scaling does not depend on memory layout; placebo is neutral on the cue fold).

## 0.2.0 (2026-11-03): "bulletproof" upgrade, before preregistration

Design (decided by the author, see DECISIONS.md):
- New secondary group **scrambled_cues** (placebo lesson, same format, features the AI did not pick, random directions) and hypothesis **H7**: real cues beat scrambled cues (D1).
- **Known-answer gate** (40 items, 85%) and **answer-collapse rule** for language-model learners; failing models are reported in full but do not count for the hypotheses (D2).
- **Blinded-labels run** at full size (1,002 per group): measurement names become "Measure 1".."Measure 18", no units (D4).
- **Prior-knowledge probe**: each model is asked, before any lesson, how each measurement changes with age.
- **Synthetic validation** (6 fake worlds with known answers + statistics simulations at the planned size + reproducibility and tamper tests), run before the preregistration (D3).

Fixes of problems found in v0.1 (ISSUES.md):
- P1: roles are now assigned by **patient** (no patient in two roles; validation patients chosen by a fixed SHA-256 rule; other-device ECGs of transfer patients set aside).
- P2: each learner's tests use **different patients** (no patient twice within a learner's practice or tests).
- P3: Experiment B's no-training nets now practise on the same ECGs with **shuffled labels** (they had zero variance before).
- P4: learners share a pool of test ECGs, so an **item-level two-way cluster-robust** test (learner and ECG) is reported as a robustness check.
- Answer letters are now **exactly balanced** within each model and group (were a coin flip per learner) (P5).
- Learner models are loaded at their **pinned commits** (P6).
- run.log is finished before it is hashed, so `verify` no longer flags it (P8).
- The 3-point bar for H1 was confirmed by the author after seeing its power curve (D6).

Integrity and reporting:
- PTB-XL is checked against PhysioNet's own **SHA256SUMS** and published record and patient counts; a mismatch stops the run.
- **Self-checks C1-C14** recompute key results a second, independent way and appear at the top of every report.
- Every null result shows its **minimum detectable effect**.
- New commands: `smoke`, `validate`, `freeze`, `run` (refuses without the lock unless `--exploratory`), `verify` (re-hash a run), `check-data`, `report`.
- Reports in three formats from one source: **REPORT.md, REPORT.html, REPORT.pdf**, with "The short answer", plain-words boxes, checks, glossary and common questions.
- AI-use statement and licence line on every report.
- `requirements.lock`, `Dockerfile`, `scripts/pin_models.py` for exact environments and model commits.

## 0.1.0 (2026-10-26)

First complete pipeline: PTB-XL cohort, ResNet teacher, stability-selection cues, saliency check, Experiment A (3 language-model families) and Experiment B (small nets), three groups, 1,002 learners per group, preregistered Welch tests with Holm correction, SHA-256 manifests.
