# Preregistration: Can learners acquire a perceptual skill they never had from AI-derived cues?

*A computational, fully open-source test of AI-to-learner knowledge transfer on ECG age.*
Version 1.0.1, frozen on 2026-11-27, before any confirmatory data were seen. (Version 1.0.1 corrects one outdated phrase of 1.0.0 in section 5, "Data exclusion"; no rule changed. See CHANGELOG.md and ISSUES.md P25.) Format follows the OSF Preregistration template.

**Author.** KEYUR SHAILESHKUMAR PATEL (ORCID: https://orcid.org/0009-0008-5261-043X).
1) MSc Artificial Intelligence student, Department of Computer Science, Brunel University London, UK;
2) Independent researcher.
Conducted independently; not an official Brunel University London project.

**Use of AI tools.** The research questions, hypotheses and design choices are the author's. The code and draft documents were written with help from an AI assistant (Claude, by Anthropic) under the author's direction and checked by the author. Separately, the study uses AI models as its objects of study: the AI teacher (a neural network trained here) and the learners (open language models and small neural networks).

**Licence.** Text and figures CC BY 4.0; code MIT.

---

## 1. Study information

**Title.** Can learners acquire a perceptual skill they never had from AI-derived cues? A preregistered computational study of AI-to-learner knowledge transfer on ECG age.

**Description.** Deep networks can tell a person's age group from a resting ECG far better than people can. We ask whether what such a network uses can be extracted as a short list of plain cues and taught to learners who start near chance, so that they improve more than with practice alone. No human participants are involved. Learners are (A) open-source language models taught in words and (B) small neural networks taught with data. Each learner is one independent "participant" with its own seed, practice ECGs and test ECGs.

**Hypotheses.** Stated separately for Experiment A (language-model learners) and Experiment B (neural-net learners). **Experiment A is the primary experiment. Experiment B is secondary:** it is run and reported exactly as designed, with the same tests, but its H1 is not a primary claim. In the exploratory pilot on folds 1-9 the cues did not beat feedback-only practice in Experiment B at any practice size from 4 to 128 ECGs, and scrambled cues did as well as the real ones, so no difference is expected there. This expectation is stated in advance; a difference in either direction will be reported as found. All are directional (one-sided).

- **H1 (primary, confirmatory).** On the post-test with new ECGs, mean accuracy is higher with AI-cue training than with feedback-only practice, by at least 3 accuracy points.
- H2 (secondary). Post-test: AI-cue training > no training.
- H3 (secondary). Post-test: feedback-only > no training.
- H4 (secondary, retention). Delayed test after an interference task: AI-cue training > feedback-only.
- H5 (secondary, transfer). ECGs from a held-out recording device: AI-cue training > feedback-only.
- H6 (secondary). Gain (post minus pre): AI-cue training > feedback-only.
- **H7 (secondary, new in v0.2).** Post-test: AI-cue training > scrambled-cue training (a placebo lesson in the same format). This asks whether the AI's *content* matters, not just having a lesson.

## 2. Design plan

**Study type.** Experiment: randomised, between-learner, four groups, with repeated tests within learner.

**Blinding.** Learners are machines, so blinding of participants does not apply. The analysis is fixed in code, frozen with SHA-256 before the confirmatory run, and run automatically. In Experiment A a second, **blinded-labels run** hides what the measurements are (names become "Measure 1" to "Measure 18", no units), so a model cannot use what it already knows about ECGs.

**Groups.**

| Group | Experiment A (language models) | Experiment B (neural nets) |
| --- | --- | --- |
| no_training | Neutral lesson describing the measurements, length-matched in words to the cue lesson. No examples. | Practises on the same 64 ECGs and steps as the others, but with the labels shuffled (no real feedback). |
| feedback_only | Neutral lesson + 30 practice cases with correct answers (24 + 6 extra). | Trained on 64 labelled practice ECGs (raw 12-lead signal), 300 steps. |
| cue_training | AI-cue lesson + the AI's 6 most confident ("clearest") cases + the same 24 practice cases. | Same 64 ECGs and steps, plus an added loss (weight 1.0) for predicting the standardised cue values. |
| scrambled_cues | Placebo lesson in the same format (see 4) + 6 random cue-fold cases shown as "clearest" + the same 24 practice cases. | Same as cue_training, but the added loss predicts the scrambled cue values. |

Every taught group sees the same number of example cases, so differences come from the lesson, not from the amount of practice.

**Tests within each learner.** Pre (before teaching), post (primary), delayed (after interference), transfer (held-out device). Each test is balanced 50/50 younger/older.

**Randomisation.** Each learner's random stream is `numpy.random.SeedSequence([20261026, experiment, group, learner_index])`, with experiment codes llm = 1, nets = 2, llm_blinded = 3. The stream fixes the learner's practice ECGs, test ECGs and case order. In Experiment A, learners are assigned to language models by learner index (index mod 3), so each model gets 334 learners per group, and the answer letter for "older" alternates (A, B, A, ...) within each model and group, so letters are exactly balanced.

## 3. Sampling plan

**Existing data.** PTB-XL v1.0.3 (21,799 ECGs from 18,869 patients; Wagner et al., 2020) and PTB-XL+ v1.0.1 (12SL features). Both public, CC BY 4.0. As of the date of this preregistration, the author has run one exploratory pilot on PTB-XL folds 1-9 (described in `prereg/PILOT_DISCLOSURE.md` and reported in full in `pilot/EXPLORATORY_PILOT_REPORT.md`). In it the author saw how the teacher's guesses relate to the 18 panel measurements and saw exploratory group results for 30 learners per group. The author has not opened, analysed or seen any summary of the confirmatory test data: PTB-XL fold 10 and all ECGs from the held-out recording device. The synthetic validation (section 7) opened no PTB-XL file.

**Data checks before use.** Every downloaded file is checked against PhysioNet's own `SHA256SUMS.txt`; any mismatch stops the run. The record and patient counts are compared with the published 21,799 and 18,869 and shown in the report.

**Data collection procedure.**
- Labels: younger = age 39 or under; older = age 65 or over. Ages 40-64 are excluded. PTB-XL codes ages over 89 as 300; these are recoded to 90 (older).
- ECGs missing more than 2 of the 18 panel features are excluded; remaining gaps are filled with the teacher-training median.
- Roles are assigned **by patient**, so no patient appears in two roles:
  - teacher training: folds 1-8, except validation patients;
  - teacher validation (early stopping): about 10% of the patients in folds 1-8, chosen by a fixed rule (SHA-256 of "20261026:val:patient_id" below 0.10);
  - cue extraction: fold 9;
  - learner tests: fold 10;
  - transfer: every ECG from the recording device with the 2nd-most eligible ECGs. If a patient has any ECG on that device, their ECGs from other devices are set aside and used nowhere.
- Learner practice ECGs come from the teacher-training role (true labels shown).
- Within a learner, every practice and test set uses at most one ECG per patient, and the pre, post and delayed tests use different patients.

**Sample size.** 1,002 learners per group per experiment: Experiment A main run 4,008, Experiment A blinded run 4,008, Experiment B 4,008. 1,002 so that each of the 3 language models gets exactly 334 learners per group.

**Sample size rationale.** With 1,002 per group, a one-sided Welch test at alpha 0.05 has 80% power for d = 0.11 and 90% for d = 0.13. Because such a large sample makes tiny differences significant, H1 also requires a minimum meaningful difference of 3 points (section 5). The synthetic validation (section 7) shows what this rule means in practice. With learner spread like the validation's informative world (SD 6.2 points, 40 items each), a true effect of 3 points passes H1 about half the time (49%), 3.5 points 97%, and 4 points or more almost always, while a true null passes 0 times in 4,000 simulated studies. With a larger spread (SD 13 points, seen in the quick dry run) 4 points passes about 94%. The author confirmed the 3-point bar after seeing this (decision D6).

**Stopping rule.** Fixed N. No interim looks at outcomes. An interrupted run is resumed with `--resume`, which continues the same seeds and never re-draws finished learners.

## 4. Variables

**AI teacher.** 1D ResNet (64 channels, 4 blocks) on raw 100 Hz 12-lead ECGs from the teacher-training role; early stopping on validation AUROC (patience 6, max 40 epochs). Five teachers are trained in the same way with seeds 20261026 + 1000k (k = 0 to 4). Teacher 0 is "the teacher": its accuracy is the precondition in section 5, and its guesses give the clearest cases, the saliency check, and each cue's direction and cut-point. Teachers 1 to 4 are used only for the cue consensus below.

**Measurement panel (fixed in advance).** Heart rate, PR interval, QRS duration, QT, QTc (Bazett), P/QRS/T axes, P-wave height and duration (II), R-wave height (aVL, V5), S-wave depth (V1), T-wave height (II, V1, V5), ST level (V5), Q-wave depth (III). Every learner in every group sees exactly these 18 numbers for every ECG.

**Cue extraction (fixed rule).** On the cue fold only, using the teachers' guesses (p > 0.5), not the true ages. For each of the five teachers separately: stability selection with L1 logistic regression over the panel (200 bootstraps, C grid 0.005-1.0); the sparsest C whose stable set (selection frequency 0.6 or more) has at least 5 features gives that teacher's list (fallback, if no C reaches 5: the top 5 by frequency at the largest C). **Consensus:** a measurement becomes a cue if at least 3 of the 5 teachers have it on their list. Cues are ordered by number of votes, then by mean selection frequency; at most 8 are kept. Fallback, if fewer than 5 measurements get 3 votes: the list is topped up to 5 in that same order (reported). Each cue becomes a rule: its direction comes from an L2 refit on teacher 0's guesses; its cut-point is the midpoint of the medians of teacher 0's two guessed groups. For each cue the report shows how many teachers chose it and how many agree on its direction.

**Scrambled cues (placebo, fixed rule).** Same number of rules, same ranks and weights, same wording. Features: ones the AI did not select, in a random order from seed 20261033 (master seed + 7); topped up with selected features only if the panel runs out. Direction: random. Cut-point: the feature's cue-fold median. **The placebo must be neutral:** with that same seed, the draw is repeated until the placebo rules, applied as a simple tally, are right between 45% and 55% of the time against the true age groups of the cue fold. The first draw that passes is used, and the number of draws is reported. If none of 1,000 draws passes, the draw closest to 50% is used and this is reported as a failed self-check (C17). Test data are never used for this.

**Saliency check (descriptive).** Integrated gradients on 400 cue-fold ECGs; saliency density in the P, QRS and ST-T segments compared with the cue weight per segment. Reported, not tested.

**Interference before the delayed test.** A: about 1,500 tokens of an unrelated weather task with feedback, placed between the lesson and the delayed test. B: the network is trained on sex classification from 256 other ECGs for 300 steps, then re-tested on age with its original age head.

**Known-answer gate (Experiment A).** Before a model's learners count, the model sees 40 teacher-training ECGs and one explicit rule ("older exactly when heart rate is above [its training median]"), half the items with each letter mapping. It must answer at least 85% (34 of 40) by the rule. The gate is run separately for the main and blinded runs.

**Which models count (Experiment A).** **Flat rule.** A learner is flat on a test if more than half of its items tie at its own median P(older), so that they cannot be ranked. A model is uninformative if more than half of its learners are flat on the post-test. **Answer-collapse flag (reported only).** A learner is flagged as collapsed on a test if 95% or more of its answers (threshold 0.5) are the same class; the share is reported for every model but does not remove a model, because the primary score does not depend on which letter a learner favours.

**Prior-knowledge probe (Experiment A, main run).** Before any lesson, each model is asked for each of the 18 measurements whether it is usually higher or lower in older adults. Each question is asked twice, once with "higher" as answer A and once with "higher" as answer B. A direction counts as known only if both answers agree; otherwise it is counted as inconsistent. The report counts how many of the AI's cue directions each model already "knew", and how many measurements it answered inconsistently. (Decision D18: in the pilot a single ordering gave "higher" for 17 of 18 measurements, a letter bias rather than knowledge; issue P20.)

**Prompt leak scan.** Every case text and lesson is scanned for patient IDs, sex, device, site and dates. Any hit is reported as a failed self-check.

**Measured variables.**
- **Experiment A, primary score: bias-corrected accuracy** per learner per test. P(older) for each item comes from the model's next-token probabilities for the two answer letters (renormalised; the two letter logits are read in float32 from the model's last hidden state). The learner's items on that test are split at the learner's own median P(older): items above the median count as the answer "older", items below as "younger", items exactly at the median get half credit. The score is the mean credit. Every test is half older and half younger, so chance is 50% whichever letter a learner favours. Wherever sections 1 and 5 say "accuracy" for Experiment A, they mean this score.
- Experiment B: accuracy with threshold P(older) > 0.5 on the sigmoid output, as before.
- Reported beside it for A (robustness, not used for claims): raw accuracy (threshold 0.5), AUROC of P(older) per learner, share of items tied at the median.
- Mean P(correct) per learner per test (robustness).
- A only: mean P(older) per learner (response bias), majority-answer share (collapse).

**Item counts.** A: 40 items per test. B: 200 items per test. The transfer test uses as many items as the device pool allows, up to that number, balanced.

## 5. Analysis plan

**Which learners count.** Experiment B: all. Experiment A: learners of models that pass the gate and are not uninformative by the flat rule (decided per run). Every model's results are reported in full either way, and H1 is also reported with all models included (sensitivity). If no model is informative, Experiment A gives no answer to H1, and the report says so.

**H1.** One-sided Welch two-sample t-test on learner-level post-test accuracy, cue_training vs feedback_only, alpha = 0.05, separately in each experiment and run. No correction across experiments: they test separate claims. Report the difference, a 10,000-sample bootstrap 95% CI, Cohen's d, and the minimum detectable effect (80% power). H1 in Experiment B is reported as a secondary result.

**Inference criteria for H1.** Supported only if both hold: p < 0.05, and the observed difference is at least **3 accuracy points**. A significant but smaller difference is reported as "detectable but below the meaningful threshold". If H1 is not supported, a TOST equivalence test (margin 3 points) says whether the result is a true null (equivalent) or inconclusive.

**Which run is primary in Experiment A.** The main run (real measurement names) is primary. The blinded run is a secondary check, analysed identically. If the main run supports H1 but the blinded run clearly does not, the report says the effect may come from what models already knew.

**Secondary H2-H7.** Same test. Holm correction across H2-H7 within each experiment and run.

**Preconditions (stated in advance).**
1. Teacher test AUROC of 0.80 or more. If not, the teacher is not an expert and the transfer question cannot be answered.
2. Mean pre-test accuracy of 0.65 or less. If higher, the result is "improving an existing skill", not "a new skill".
3. Cue fidelity: agreement between the cue rules and the teacher on fold 10 is reported.

**Robustness (reported, not used for claims).** ANCOVA of post accuracy on group and pre-test accuracy (HC3 errors); H1 on mean P(correct); A: H1 adjusted for model, per model, and with all models; item-level linear probability model with standard errors clustered by learner and by ECG (Cameron, Gelbach and Miller, 2011); response bias and collapse shares.

**Transfer fraction (descriptive).** For each group: (post accuracy - 0.5) / (teacher test accuracy - 0.5).

**Self-checks.** Every run recomputes key results a second, independent way (C1-C17: patient overlap, teacher and cue data roles, re-drawing learners from their seeds, test items, accuracies from item files, a hand-written Welch test, Holm via statsmodels, cue-rule accuracy from saved files, leak scan, letter balance, gate decisions, scrambled-feature overlap, teacher weights hash for all five teachers, no sealed ECG used (pilot mode only), cue consensus votes recounted, placebo score recomputed). A failed check is shown at the top of the report and must be explained in ISSUES.md before any result is quoted.

**Data exclusion.** No learners are excluded, except as defined by the gate and flat rules above (Experiment A, model level), and those are still reported.

**Missing data.** None expected. A learner that fails to run is rerun with the same seed; if impossible, it is listed in the report.

**Exploratory analyses (labelled as exploratory).** Item-level mixed models; which cues each model follows; cue ablations; other learner models.

## 6. Other

**Reproducibility.** `python run_study.py freeze` hashes this document, `configs/study.yaml` and all code into `prereg/LOCK.json`; the four printed fingerprints are posted on OSF with this preregistration. `python run_study.py run` refuses to run if anything changed, unless `--allow-deviation "reason"` is given, which writes the reason into the report. Learner models are pinned to exact Hugging Face commits (`scripts/pin_models.py`) before freezing. Every run writes `manifest.json` (configuration, code hashes, environment, package versions, stage timings) and `SHA256SUMS.txt` over every output; `python run_study.py verify runs/<id>` re-checks them. Exact package versions: `requirements.lock`; container: `Dockerfile`.

**Software and data licences.** PTB-XL and PTB-XL+: CC BY 4.0. Learner models: their own licences (Qwen2.5-3B: Qwen Research Licence; Llama 3.2: Llama 3.2 Community Licence; Gemma 2: Gemma Terms of Use).

**Limits stated in advance.** Machine learners are not people; a positive result shows that the cues can be used by a learner that lacked the skill, not that humans will learn it. PTB-XL comes from one country and one hospital network. Cues describe what this teacher uses, which may include dataset quirks. The language models may already know some ECG facts; the probe and the blinded run measure this.

**Planned follow-up.** A study with people using the same cues, tests and analysis plan.

## 7. Synthetic validation done before this preregistration

Before freezing, the full pipeline was run on made-up ECG-like data with scripted mock learners, in six worlds where the right answer was known in advance (five planned, one added after a dry run of version 0.2.0). Earlier dry runs found one software fault (ISSUES.md P8, fixed) and one mis-set expectation (P9). The validation was first run on version 0.2.0 (run `20261103T220829Z_validation`) and **repeated on the final code** after the changes that followed the pilot (cue consensus, neutral placebo, bias-corrected score, flat rule, two-way probe); the numbers below are from the repeat. No PTB-XL file was opened (the validation itself; the later real-data pilot is disclosed in `prereg/PILOT_DISCLOSURE.md`). Full report: `validation/VALIDATION_REPORT.pdf`, with every number in `validation_summary.json` and fingerprints in `SHA256SUMS.txt`.

Validation run `20261120T132404Z_validation` (software 0.5.2, master seed 20261026, 49 minutes on CPU). Language-model worlds used the full 1,002 learners per group in both the main and blinded runs, scored with the bias-corrected score; neural-net worlds used 30 per group.

| World | What it was built to show | Expected outcomes met | Key numbers |
| --- | --- | --- | --- |
| W1 Informative | Pipeline finds a planted lesson effect | 5 of 7 | H7 +24.8 points (supported); H1 +1.8 points (main), +1.7 (blinded): p < 0.001 but below the 3-point bar |
| W1b Strong teaching (added after the dry run) | A larger planted effect | 2 of 4 | H7 +31.5 points (supported); H1 +1.0 points (main), +0.5 (blinded): significant, below the bar |
| W2 Null learners | No false win when learners learn nothing | 5 of 5 | H1 +0.2 points, p = 0.32; H7 +0.1, not supported |
| W3 Prior knowledge | Probe and blinded run catch memorised facts | 4 of 5 | Probe (asked both ways): 6 of 6 cue directions known by every model; pre-test 84.2% (main) vs 49.8% (blinded); blinded H1 +1.5 points, below the bar |
| W4 Broken models | Gate and flat rule remove bad models | 6 of 6 | Random model gate 47% (fail); model that always answers "older": gate 100%, but 100% of its learners flat (and collapsed), so it does not count; only the good model counted |
| W5 No signal | Preconditions catch a teacher with nothing to find | 4 of 4 | Teacher AUROC 0.496 (precondition fails); H1 not supported in either experiment |

All 16 self-checks that apply outside pilot mode (C1-C14, C16, C17) passed in every world. Overall, 26 of 31 pre-stated expectations were met, as in the first validation. All 5 misses have the same cause: the planted lesson effect in the mock worlds came out at 0.5 to 1.8 points, below the 3-point bar, and the H1 rule correctly called it "significant but below the meaningful threshold" (ISSUES.md P9, P10). The expectations are kept as written. That large effects are found is shown by H7 (+25 to +32 points, supported in every world whose learners read lessons).

Statistics at the planned size (4,000 simulated studies per point unless stated; learner spread taken from W1 under the bias-corrected score, SD 6.3 points):

- H1 false-positive rate with no true effect: 0.00% (Welch test alone: 5.2%).
- Chance that H1 says yes: 49% at a true 3-point effect, 97% at 3.5 points, more than 99.9% at 4 points or more.
- Holm family-wise error across H2-H7 when nothing works: 3.75%.
- Learners sharing test ECGs (300 simulated studies, 300 learners per group, 400-ECG pool): false-positive rate 5.0% for the learner-level Welch test and 5.3% for the item-level two-way cluster test.
- Known-answer gate: a model that truly follows the rule 90% of the time passes 90% of the time; at 70%, it passes 2%.
- Collapse flag (now reported only): a learner leaning 70% toward one answer is flagged 0.01% of the time; at 95%, 68%.
- Reproducibility: two runs from scratch gave byte-identical results in all 41 compared files. Changing one byte in a results file was caught by `verify`, and restoring it passed again.

These results test the software and the statistics only. They are not evidence about real ECGs, real models or people.

## 8. Changes from version 0.1 (made before any confirmatory data were seen)

Patient-level roles (P1, P2); shuffled-label practice for Experiment B's no-training nets (P3); item-level two-way cluster robustness (P4); exactly balanced answer letters (P5); pinned model commits (P6); scrambled-cue group and H7 (D1); known-answer gate and collapse rule (D2); synthetic validation only, no real-data pilot (D3; replaced by D7 in version 0.3); blinded-labels run at full size (D4); PhysioNet checksum check; self-checks C1-C14; minimum detectable effects next to nulls. See CHANGELOG.md, DECISIONS.md and ISSUES.md.

## 9. Changes from version 0.2 (made after the exploratory pilot on folds 1-9, before any confirmatory data were seen)

An exploratory real-data pilot was run on PTB-XL folds 1-9 (D7, replacing D3); fold 10 and the held-out device were sealed. Changes made because of it: cue consensus across five teachers (D9; P11, P14); neutral placebo (D10; P15); input scaling summed in 64-bit so that the same seed gives the same teacher (P11); results saved after every learner (P13); bounded memory for long prompts (P17, P18); self-checks C15-C17. What the author saw in the pilot, and why the confirmatory test is still a fair test, is in `prereg/PILOT_DISCLOSURE.md`, which is part of this preregistration. See CHANGELOG.md, DECISIONS.md and ISSUES.md.

Decisions taken after the pilot, each with knowledge of its results: bias-corrected accuracy as the primary score of Experiment A (D12; P19, P23); Experiment B secondary (D13; P16); H1 kept as the primary hypothesis (D14); flat rule in place of the collapse rule (D15); 24 practice cases kept (D16); 16-bit models with a float32 answer read-out (D17); probe asked both ways (D18; P20). Each is listed with its date in `prereg/PILOT_DISCLOSURE.md` section 5.
