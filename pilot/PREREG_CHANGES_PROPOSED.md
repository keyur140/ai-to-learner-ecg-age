# Changes to the preregistration text after the pilot (approved by the author on 2026-11-15)

*AI-to-learner knowledge transfer study (ECG age). Author: KEYUR SHAILESHKUMAR PATEL (ORCID https://orcid.org/0009-0008-5261-043X). Conducted independently; not an official Brunel University London project.*

**Status: APPLIED to `prereg/preregistration.md` in version 0.5.0 (a draft, not frozen), after the author approved items A1 to A11 on 2026-11-15.** Two things differ from this preview: the version line reads 0.5.0, not 0.3.0, and one sentence listing decisions D12 to D16 was added to the new section 9. This file is kept as the record of old and new wording. This preview was drafted with help from an AI assistant (Claude, by Anthropic); the choices behind it (D7, D9, D10, D12 to D16) are the author's, and the wording is for the author to accept, change or reject.

This file sits in `pilot/`, not in `prereg/`, so it is not hashed into the preregistration lock.

Part A lists changes that follow from decisions the author has already taken. Part B lists places that depend on decisions still open; no wording is proposed for those yet.

---

## Part A. Changes that follow from decisions already taken

### A1. Header (line 4)

Old: `Version 0.2.0 (2026-11-03).`

New: `Version 0.3.0 ([date of freezing]).`

### A2. Section 3, "Existing data" (the bracketed statement)

Old:

> **[Confirm: "As of the date of this preregistration, the author has not analysed the relationship between age and any ECG feature in PTB-XL." The synthetic validation (section 7) did not open any PTB-XL file.]**

New:

> As of the date of this preregistration, the author has run one exploratory pilot on PTB-XL folds 1-9 (described in `prereg/PILOT_DISCLOSURE.md` and reported in full in `pilot/EXPLORATORY_PILOT_REPORT.md`). In it the author saw how the teacher's guesses relate to the 18 panel measurements and saw exploratory group results for 30 learners per group. The author has not opened, analysed or seen any summary of the confirmatory test data: PTB-XL fold 10 and all ECGs from the held-out recording device. The synthetic validation (section 7) opened no PTB-XL file.

Reason: decision D7. The old sentence is no longer true for folds 1-9.

### A3. Section 4, "AI teacher"

Add after the existing sentence:

> Five teachers are trained in the same way with seeds 20261026 + 1000k (k = 0 to 4). Teacher 0 is "the teacher": its accuracy is the precondition in section 5, and its guesses give the clearest cases, the saliency check, and each cue's direction and cut-point. Teachers 1 to 4 are used only for the cue consensus below.

Reason: decision D9.

### A4. Section 4, "Cue extraction (fixed rule)"

Old:

> On the cue fold only, using the teacher's guesses (p > 0.5), not the true ages: stability selection with L1 logistic regression over the panel (200 bootstraps, C grid 0.005-1.0). The sparsest C whose stable set (selection frequency 0.6 or more) has at least 5 features is used; at most 8 cues are kept. Each cue becomes a rule: its direction comes from an L2 refit; its cut-point is the midpoint of the medians of the two teacher-guessed groups. Fallback, if no C reaches 5: the top 5 by frequency at the largest C (reported).

New:

> On the cue fold only, using the teachers' guesses (p > 0.5), not the true ages. For each of the five teachers separately: stability selection with L1 logistic regression over the panel (200 bootstraps, C grid 0.005-1.0); the sparsest C whose stable set (selection frequency 0.6 or more) has at least 5 features gives that teacher's list (fallback, if no C reaches 5: the top 5 by frequency at the largest C). **Consensus:** a measurement becomes a cue if at least 3 of the 5 teachers have it on their list. Cues are ordered by number of votes, then by mean selection frequency; at most 8 are kept. Fallback, if fewer than 5 measurements get 3 votes: the list is topped up to 5 in that same order (reported). Each cue becomes a rule: its direction comes from an L2 refit on teacher 0's guesses; its cut-point is the midpoint of the medians of teacher 0's two guessed groups. For each cue the report shows how many teachers chose it and how many agree on its direction.

Reason: decision D9 (pilot issues P11, P14: the one-teacher rule gave 5 cues in one run and 8 in the next).

### A5. Section 4, "Scrambled cues (placebo, fixed rule)"

Old:

> Same number of rules, same ranks and weights, same wording. Features: ones the AI did not select, in a random order from seed 20261033 (master seed + 7); topped up with selected features only if the panel runs out. Direction: random. Cut-point: the feature's cue-fold median.

New:

> Same number of rules, same ranks and weights, same wording. Features: ones the AI did not select, in a random order from seed 20261033 (master seed + 7); topped up with selected features only if the panel runs out. Direction: random. Cut-point: the feature's cue-fold median. **The placebo must be neutral:** with that same seed, the draw is repeated until the placebo rules, applied as a simple tally, are right between 45% and 55% of the time against the true age groups of the cue fold. The first draw that passes is used, and the number of draws is reported. If none of 1,000 draws passes, the draw closest to 50% is used and this is reported as a failed self-check (C17). Test data are never used for this.

Reason: decision D10 (pilot issue P15: the first draw scored 36%, which could teach the opposite of the truth and make H7 too easy).

### A6. Section 5, "Self-checks"

Old: `(C1-C14: patient overlap, ... teacher weights hash)`

New: `(C1-C17: patient overlap, teacher and cue data roles, re-drawing learners from their seeds, test items, accuracies from item files, a hand-written Welch test, Holm via statsmodels, cue-rule accuracy from saved files, leak scan, letter balance, gate decisions, scrambled-feature overlap, teacher weights hash for all five teachers, no sealed ECG used (pilot mode only), cue consensus votes recounted, placebo score recomputed)`

### A7. Section 7, synthetic validation

Add at the start of the section:

> The validation below was run on software version 0.2.0. **[Before freezing: repeat it on the final code and replace the numbers below. The cue rule and the placebo rule changed in 0.3.0, so the old validation does not cover them.]**

Keep "No PTB-XL file was opened" for the synthetic validation (it is still true), but change "(decision D3)" to "(the validation itself; the later real-data pilot is disclosed in `prereg/PILOT_DISCLOSURE.md`)".

### A9. Primary score of Experiment A (decision D12, 2026-11-14)

Section 4, "Measured variables", first bullet. Old:

> Accuracy per learner per test: share of items correct, threshold P(older) > 0.5. A: P(older) from the model's next-token probabilities for the two answer letters (renormalised); B: the sigmoid output.

New:

> - **Experiment A, primary score: bias-corrected accuracy** per learner per test. P(older) for each item comes from the model's next-token probabilities for the two answer letters (renormalised; the two letter logits are read in float32 from the model's last hidden state). The learner's items on that test are split at the learner's own median P(older): items above the median count as the answer "older", items below as "younger", items exactly at the median get half credit. The score is the mean credit. Every test is half older and half younger, so chance is 50% whichever letter a learner favours. Wherever sections 1 and 5 say "accuracy" for Experiment A, they mean this score.
> - Experiment B: accuracy with threshold P(older) > 0.5 on the sigmoid output, as before.
> - Reported beside it for A (robustness, not used for claims): raw accuracy (threshold 0.5), AUROC of P(older) per learner, share of items tied at the median.

Reason: decision D12 (pilot issues P19, P23). This is a change of the primary outcome made after seeing pilot results; it is listed as such in `prereg/PILOT_DISCLOSURE.md` section 5.

### A10. Which models count (decision D15, 2026-11-14)

Section 4, "Answer-collapse rule". Old:

> A learner is flagged as collapsed on a test if 95% or more of its answers are the same class. A model is uninformative if more than half of its learners are collapsed on the post-test.

New:

> **Flat rule.** A learner is flat on a test if more than half of its items tie at its own median P(older), so that they cannot be ranked. A model is uninformative if more than half of its learners are flat on the post-test. **Answer-collapse flag (reported only).** A learner is flagged as collapsed on a test if 95% or more of its answers (threshold 0.5) are the same class; the share is reported for every model but does not remove a model, because the primary score does not depend on which letter a learner favours.

Section 5, "Which learners count": replace "are not uninformative by the collapse rule" with "are not uninformative by the flat rule".

Why a rule is still needed: a model that gives the same probability to every case passes the gate but carries no information; it would add learners fixed at exactly 50%. Other options: no rule at all (gate only), or keep the old collapse rule (which would remove Qwen2.5-3B on the pilot's evidence).

### A11. Experiment B becomes secondary (decision D13, 2026-11-14)

Section 1, "Hypotheses". Old: "Stated separately for Experiment A (language-model learners) and Experiment B (neural-net learners)."

New:

> Stated separately for Experiment A (language-model learners) and Experiment B (neural-net learners). **Experiment A is the primary experiment. Experiment B is secondary:** it is run and reported exactly as designed, with the same tests, but its H1 is not a primary claim. In the exploratory pilot on folds 1-9 the cues did not beat feedback-only practice in Experiment B at any practice size from 4 to 128 ECGs, and scrambled cues did as well as the real ones, so no difference is expected there. This expectation is stated in advance; a difference in either direction will be reported as found.

Section 5, "H1": add "H1 in Experiment B is reported as a secondary result."

### A8. New section 9, "Changes from version 0.2"

> ## 9. Changes from version 0.2 (made after the exploratory pilot on folds 1-9, before any confirmatory data were seen)
>
> An exploratory real-data pilot was run on PTB-XL folds 1-9 (D7, replacing D3); fold 10 and the held-out device were sealed. Changes made because of it: cue consensus across five teachers (D9; P11, P14); neutral placebo (D10; P15); input scaling summed in 64-bit so that the same seed gives the same teacher (P11); results saved after every learner (P13); bounded memory for long prompts (P17, P18); self-checks C15-C17. What the author saw in the pilot, and why the confirmatory test is still a fair test, is in `prereg/PILOT_DISCLOSURE.md`, which is part of this preregistration. See CHANGELOG.md, DECISIONS.md and ISSUES.md.

Also in section 8, change "synthetic validation only, no real-data pilot (D3)" to "synthetic validation only, no real-data pilot (D3; replaced by D7 in version 0.3)".

---

## Part B. Places that depend on decisions still open (no wording proposed yet)

| Place in the preregistration | Why it may change | Open point |
| --- | --- | --- |
| Section 2 "Blinding" / config `dtype` | The models' 16-bit format adds rounding noise (up to 0.06 in probability on the same prompt) | P23: float32 runs on a GPU |
| Section 4 "Prior-knowledge probe" | The probe answered "higher in older adults" for 17 of 18 measurements | P20: probe wording, or drop it |
| Section 6 "Learner models are pinned" | Llama-3.2-3B and Gemma-2-2B need a Hugging Face token; none was set in the pilot | Token and model commits |

Any change made in Part B is made with knowledge of the pilot results and must be listed in `prereg/PILOT_DISCLOSURE.md` section 5 with its date and reason.
