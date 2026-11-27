# Disclosure of the exploratory real-data pilot

*AI-to-learner knowledge transfer study (ECG age). Author: KEYUR SHAILESHKUMAR PATEL (ORCID https://orcid.org/0009-0008-5261-043X). Conducted independently; not an official Brunel University London project.*

*Part of the preregistration: this file is hashed into `prereg/LOCK.json` together with `preregistration.md`. Drafted with help from an AI assistant (Claude, by Anthropic) under the author's direction; checked and confirmed by the author before freezing (confirmed on 2026-11-27).*

## 1. What happened

Version 0.2.0 planned no real-data run before preregistration (decision D3). On 2026-11-07 the author decided to run an exploratory pilot on real data first, in a protected form (decision D7). It ran from 2026-11-07 to 2026-11-12. This file states what was looked at, what was not, what was seen, and what was changed because of it, so that readers can judge the confirmatory study with that in mind.

## 2. Data that were used, and data that were not

| Data | Role in the pilot | Role in the confirmatory study |
| --- | --- | --- |
| PTB-XL folds 1-7 | Teacher training and validation; learners' practice ECGs | Teacher training (with fold 8) |
| PTB-XL fold 8 | Cue extraction | Teacher training |
| PTB-XL fold 9 | Learners' tests (pre, post, delayed) | Cue extraction |
| Device CS-12 (largest device after sealing) | Learners' transfer test | Part of folds 1-9 as above |
| **PTB-XL fold 10** | **Sealed. Never opened.** | **Learners' tests: the data H1 is tested on** |
| **Device AT-6 C 5.5** (2nd-largest overall) | **Sealed. Never opened.** | **Transfer test** |

The seal is enforced in code (`run --pilot`): 3,646 ECGs of 3,347 patients were removed before anything else ran, and self-check C15 counts sealed ECGs in the run (0 found). Patients on the sealed device were removed with all their ECGs.

**What the code touched in the sealed data, stated exactly (P22).** The seal is applied in the cohort step, after three things that run over the whole dataset, sealed rows included: (1) every downloaded file is hashed and compared with PhysioNet's checksums (bytes only, nothing is parsed); (2) the PTB-XL+ feature table is loaded whole, and for each amplitude measurement the code takes the median absolute value over all rows to decide whether the file stores millivolts or microvolts; (3) to pick the sealed device by the preregistered rule ('2nd-largest'), the code counts eligible ECGs per recording device, and 'eligible' uses each ECG's age group and how many of its 18 panel values are missing. None of these relates age to any ECG feature, none is printed per ECG, and no ECG signal of a sealed ECG was read (signals are loaded only for the kept rows). After the seal, sealed rows are dropped from every table and are not written to any output file.

## 3. What the author has now seen

Before this pilot the author had not analysed how age relates to any ECG feature in PTB-XL. That is no longer true for folds 1-9. The author has seen:

- that the teacher separates the two age groups very well on fold 9 (AUROC 0.985), and that a logistic regression on the 18 panel measurements also does (AUROC 0.947);
- which measurements the teacher's guesses load on in fold 8, and in which direction: PR interval (longer in older), T-wave height in lead II (lower), R-wave height in aVL (higher), QRS axis (lower), QTc (longer), S-wave depth in V1 (higher); and that a tally of these six rules is right about 82.0% of the time on fold 9;
- Experiment B on fold 9, 30 nets per group: after training, feedback only 80.9%, AI cues 80.8%, scrambled 80.4%, no training 48.2%;
- Experiment A with one of the three planned models (Qwen2.5-3B), 30 learners per group, main run (120 learners) and blinded run (120 learners). By the planned accuracy score: feedback only 53.9%, AI cues 52.5%, scrambled 52.4%, no training 48.6%; the model answered almost only one letter and was marked uninformative by the collapse rule. By a bias-corrected score chosen after seeing this (each learner's items split at its own median probability): feedback only 69.2%, AI cues 65.3%, scrambled 58.7%, no training 49.9% (main); feedback only 66.0%, AI cues 66.6%, scrambled 63.0%, no training 44.7% (blinded).

- A follow-up on 2026-11-14 (`pilot/PILOT_FOLLOWUP.md`): Experiment B on fold 9 with 4, 8, 16, 32, 64 and 128 practice ECGs, 30 nets per group each. At no size did the AI's cues beat feedback only (differences -4.6 to +1.2 points), and the scrambled cues did as well as the real ones.

In short: **in the pilot the AI's cues did not beat feedback-only practice** (the H1 comparison) in either experiment, and by the bias-corrected score they beat the placebo in the main run of Experiment A (+6.6 points) but not clearly in the blinded run (+3.6 points). The author knows this when writing the final preregistration.

The author has not seen: any ECG, feature or result from fold 10 or from device AT-6 C 5.5; any result from Llama-3.2-3B or Gemma-2-2B; any result at the planned sample size.

## 4. Replacement for the 'existing data' statement

The bracketed statement in section 3 of the preregistration is replaced by:

> As of the date of this preregistration, the author has run one exploratory pilot on PTB-XL folds 1-9 (described in `prereg/PILOT_DISCLOSURE.md` and reported in full in `pilot/EXPLORATORY_PILOT_REPORT.md`). In it the author saw how the teacher's guesses relate to the 18 panel measurements and saw exploratory group results for 30 learners per group. The author has not opened, analysed or seen any summary of the confirmatory test data: PTB-XL fold 10 and all ECGs from the held-out recording device. The synthetic validation (section 7) opened no PTB-XL file.

## 5. Changes made because of the pilot

Each was decided by the author on the date shown, after seeing the pilot result named, and before any confirmatory data were seen.

| Ref | Date | Change | Seen in the pilot | Could it favour the hypotheses? |
| --- | --- | --- | --- | --- |
| D9 | 2026-11-07 | Cue consensus across five teacher seeds (keep a cue if at least 3 of 5 teachers choose it) | Two runs gave 5 and then 8 cues (P11, P14) | Changes which cues are taught. Does not use outcomes. |
| D10 | 2026-11-08 | Placebo lesson re-drawn (fixed seed) until its rules score 45-55% on the cue fold. (Built into the v0.3 pilot run on 2026-11-07 as the assistant's recommended option; the rule was therefore fixed in code before any v0.3 learner ran. The author confirmed it on 2026-11-08; by then the v0.3 main run of Experiment A had finished, so the confirmation was not blind to those results.) | The placebo scored 36% as a tally (P15) | Makes H7 harder to pass, not easier: a misleading placebo would have inflated H7. |
| D12 | 2026-11-14 | Primary score of Experiment A changed from raw accuracy to bias-corrected accuracy (each learner's items split at its own median probability). | Qwen2.5-3B answered almost only one letter; raw accuracy was near 50% in every group while the probabilities still ranked the ECGs (P19) | **Yes, it could.** This is a change of the primary outcome made after seeing pilot outcomes, including the group means under both scores. It does not favour one group over another by construction (it is applied identically to every learner and chance stays 50%), but it was chosen knowing that it shows group differences in folds 1-9. |
| D13 | 2026-11-14 | Experiment B becomes secondary; it runs exactly as designed | Cues did not beat feedback in Experiment B at any practice size from 4 to 128 ECGs (P16) | Removes a primary test that the pilot says would fail. It does not change any test or make any claim easier to support; the expected null is stated in advance. |
| D14 | 2026-11-14 | No change: H1 (cues beat feedback-only practice by at least 3 points) stays primary in Experiment A | Cues did not beat feedback in the pilot; they beat the placebo in the main run | No. The author kept the original question although the pilot suggests it may fail. |
| D15 | 2026-11-14 | Flat rule replaces the collapse rule for deciding which models count | The collapse rule would have removed Qwen2.5-3B (P19) | It keeps a model the old rule would remove. Chosen knowing that this model shows group differences under the new score in folds 1-9. |
| D16 | 2026-11-14 | No change: 24 practice cases are kept in Experiment A | 24 cases alone taught Qwen2.5-3B as much as the cues did; prompts of 9,000 to 12,000 tokens | No. The author declined to change the practice size or to look at more pilot data. |
| D17 | 2026-11-20 | No change: the confirmatory run stays in bfloat16 with the float32 answer read-out | Rounding noise up to 0.06 in probability inside the 16-bit model (P23) | No. The noise is random and the same for every group. |
| D18 | 2026-11-20 | The prior-knowledge probe asks each question twice with the letters swapped | The probe said 'higher' for 17 of 18 measurements (P20) | No. The probe is descriptive only. |
| Fix P23 | 2026-11-14 | Answer letters read in float32 | About 13 distinct probabilities per 40-item test; tie rates differed by group | Measurement precision. Reduces ties, which pulled scores towards 50% unevenly across groups. |
| Fix P11 | 2026-11-07 | Input scaling summed in 64-bit | Same seed gave different teachers | Reproducibility only. |
| Fixes P13, P17, P18 | 2026-11-07/08 | Save after every learner; bounded memory for long prompts | Lost results; out-of-memory kills | Same computation, different batching. Probabilities can differ in the second or third decimal (bfloat16). |

No other design choice was changed after the pilot. On 2026-11-27, before freezing, the three learner models were pinned to the exact Hugging Face commits current on that day (Qwen2.5-3B-Instruct `aa8e7253`, Llama-3.2-3B-Instruct `0cb88a4f`, Gemma-2-2b-it `299a8560`; full hashes in `configs/study.yaml`), as the plan required; this needs no judgment and uses no data. The Hugging Face token needed to download Llama and Gemma will be added before the confirmatory run; it changes nothing in the plan.

## 6. Why the confirmatory test is still a fair test, and what risk remains

- The data H1 is tested on (fold 10, and the held-out device for H5) were never opened, so no choice could have been tuned to them.
- The confirmatory study retrains everything from scratch with its own split (teacher on folds 1-8, cues from fold 9); no teacher, cue list or lesson from the pilot is reused.
- The confirmatory learners draw their practice ECGs from folds 1-8, which the pilot also used. Practice ECGs are shown with their true labels in every taught group, so this cannot favour one group.

Risks that remain, stated plainly:

- The author chose design details knowing the pilot's results on folds 1-9. Folds 1-9 and fold 10 come from the same hospital network and the same population, so a design tuned on one tends to behave similarly on the other. This is the usual cost of piloting; the protection is that every such choice is listed in section 5 with its date, and that the analysis is frozen before fold 10 is opened.
- The pilot used one language model. Design choices that suit Qwen2.5-3B may not suit the other two.
- The bias-corrected score in section 3 was invented after seeing the planned score fail. If it becomes part of the confirmatory analysis, that is a choice made with knowledge of pilot outcomes, and it is listed as such.

## 7. Where everything is

`pilot/EXPLORATORY_PILOT_REPORT.md` (and `.pdf`): full report. `pilot/v0.3_run/`, `pilot/v0.2_run1/`, `pilot/v0.2_run2/`: run outputs with logs, manifests and SHA-256 sums. `pilot/PILOT_FOLLOWUP.md`: follow-up of 2026-11-14. `ISSUES.md` P11-P23. `DECISIONS.md` D7-D18. `CHANGELOG.md` 0.3.0 to 0.5.1.
