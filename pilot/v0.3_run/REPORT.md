# Can learners acquire a perceptual skill they never had from AI-derived cues? (ECG age)

**Run report - EXPLORATORY PILOT (protected: confirmatory test fold and held-out device sealed)**

**KEYUR SHAILESHKUMAR PATEL** · ORCID https://orcid.org/0009-0008-5261-043X  
MSc Artificial Intelligence student, Department of Computer Science, Brunel University London, UK; Independent researcher.  
*Conducted independently; not an official Brunel University London project.*

Run: 20261107T164810Z_study_pilot  |  Software version 0.3.0  
Preregistration: NOT LOCKED: exploratory run, not confirmatory  
Use of AI tools. The research questions, hypotheses and design choices are the author's. The code and draft documents were written with help from an AI assistant (Claude, by Anthropic) under the author's direction and checked by the author. Separately, the study uses AI models as its objects of study: the AI teacher (a neural network trained here) and the learners (open language models and small neural networks).  
Licence: text and figures CC BY 4.0; code MIT.  


## 1. The short answer

- Experiment A: H1 could not be tested (no language model passed the known-answer gate and collapse rule).
- Experiment A (blinded run): H1 could not be tested (no language model passed the known-answer gate and collapse rule).
- Experiment B: right after training, AI-cue learners got 80.8% right; feedback-only learners got 80.9% (difference -0.1 points, 95% range -1.7 points to +1.6 points; one-sided p = 0.523). H1: **NOT SUPPORTED**.
- Experiment B: real cues vs scrambled cues: 80.8% vs 80.4% (difference +0.4 points; Holm p = 0.641). H7: **not supported**.
- Self-checks: 17 passed, 0 failed, 0 not applicable (Section 3).

> **What this means in plain words.** Experiment A gives no answer, because no language model passed the known-answer gate and collapse rule. Experiment A (blinded run) gives no answer, because no language model passed the known-answer gate and collapse rule. In Experiment B, learners taught the AI's cues did not do clearly better than learners who only practised with feedback. The test could have found a true difference of about 2.2 points or more most of the time. The equivalence test also says any difference is smaller than 3 points.


## 2. What this report is, and what it is not

This run tests the preregistered hypotheses on PTB-XL. The plan, code and settings were frozen before it ran; their fingerprints are in Section 3 and must match the ones on OSF.

It is a fully computational study: the 'learners' are machine learners (language models and small neural networks), not people. It asks whether a skill an AI found can be handed on in a teachable form, which is a first step before any study with people. It does not show that people would learn the same way.


## 3. Can we trust the run? (checks)

Each check recomputes something a second, independent way. A FAIL is shown here and must be explained in ISSUES.md before any number from this run is quoted.

| # | Check | Result | Detail |
| --- | --- | --- | --- |
| C1 | No patient appears in two roles | pass | 0 shared; 69 ECGs set aside |
| C2 | Teacher saw only its training and validation ECGs | pass | train roles ['teacher_train'], early-stopping roles ['teacher_val'] |
| C3 | Cues built only from the cue fold, from the teacher's guesses | pass | 740 ECGs, roles ['cue'] |
| C4 | Learner sets come from the right roles, balanced, one ECG per patient | pass | 60 learners re-drawn from their seeds; problems: none |
| C5 | Seeds reproduce exactly the test items each learner saw | pass | 0 mismatched test sets |
| C6 | Learner accuracies recomputed from the item files | pass | 960 values, largest difference 0.00e+00 |
| C7 | H1 p-values recomputed with a hand-written Welch test | pass | 1 experiments, largest difference 0.0e+00 |
| C8 | Holm adjustment recomputed with statsmodels | pass | 1 families, largest difference 0.0e+00 |
| C9 | Cue-rule accuracy recomputed from the saved files | pass | 0.8197 vs 0.8197 |
| C10 | Prompt leak scan: no IDs, sex, device, site or dates in learner text | pass | 0 flagged case lines, 0 flagged lesson texts |
| C11 | Answer letters balanced: in every group and model, 'older = A' for half the learners (within one) | pass | 8 group-model cells, largest imbalance 0.0 learners |
| C12 | Known-answer gate decisions follow the preregistered rule | pass | 2 model decisions |
| C13 | Scrambled lesson uses features the AI did not pick | pass | 0 shared features; scrambled-rule accuracy on test 0.466 |
| C14 | Teacher weights match their recorded SHA-256 | pass | c512582aaf85fc8e... (+4 consensus teachers) |
| C16 | Cue consensus: every cue was chosen by at least 3 of 5 teachers (recounted) | pass | votes PR_Int_Global 5, T_Amp_II 5, R_Amp_aVL 5, QT_IntBazett_Global 3, S_Amp_V1 3, R_AxisFrontal_Global 5 |
| C15 | Protected pilot: no ECG from the sealed test fold or the sealed device was used | pass | 3646 ECGs of 3347 patients sealed; 0 found in the run |
| C17 | Placebo is neutral: scrambled rules score 45%-55% on the cue fold (recomputed) | pass | 0.462 (draw 5); on the learners' test pool 0.466 |


### Fingerprints (SHA-256)

| Item | SHA-256 |
| --- | --- |
| Preregistration (prereg/*.md) | 42feaab76270ca69b0edfbaa2342c6bb79b46ffbcf5606ac98271829743d8ef9 |
| Config (configs/study.yaml) | bee9450aa0b40160ea16a59c7a4784ba5d0bc141b9b31ba6d5370bc4d5cf726f |
| Code (src/*.py) | 28035bdcf007c72948620cc0e1132dd35b2ce4b3ca3db517bf08f99253740d2b |


### Deviations from the preregistration

- Command-line overrides: only=None, backend=None, models=['Qwen/Qwen2.5-3B-Instruct']
- Learner models were not pinned to exact Hugging Face commits (run scripts/pin_models.py before freezing).


## 4. Data and splits

Source: PTB-XL 1.0.3 + PTB-XL+ 1.0.1. 21799 ECGs, 12336 in the two age groups (under 40 or 65 and over), 8689 with the measurements, from 7327 patients. Roles are given by patient, so no patient is in two roles; 69 ECGs were set aside for this. Held-out device: CS-12.

| Role | ECGs | Older (65+) | Used for |
| --- | --- | --- | --- |
| teacher_train | 4624 | 3199 | teaching the AI; learners' practice cases |
| teacher_val | 527 | 352 | stopping the AI's training |
| cue | 740 | 518 | finding the cues |
| test | 610 | 404 | learners' pre, post and delayed tests |
| transfer | 2119 | 1921 | learners' new-device test |
| unused | 69 | 67 | set aside (patient overlap) |

Data integrity: PhysioNet checksum file present: True; status ok; 87203 files checked. Records 21799 and patients 18869 match the published counts: True.


## 5. The AI teacher

A 1D ResNet read the raw 12-lead ECG and learned to tell younger from older adults. 'Measurements only' is a simple model that sees only the 18 measurements the learners see: it shows how much is possible from those numbers alone.

| Split | AI AUROC | AI accuracy | Measurements-only AUROC | Measurements-only accuracy |
| --- | --- | --- | --- | --- |
| teacher_val | 0.979 | 92.2% | 0.901 | 86.0% |
| cue | 0.978 | 92.4% | 0.913 | 87.3% |
| test | 0.985 | 92.6% | 0.947 | 88.4% |
| transfer | 0.956 | 85.7% | 0.880 | 89.1% |

![Validation AUROC by epoch.](figures/teacher_training.png)
*Validation AUROC by epoch.*


## 6. The cues taken from the AI

On the cue fold only, a sparse model was fitted to the AI's own guesses (not the true ages); the measurements it kept became plain rules. This is the lesson the AI-cue group reads:

```
LESSON: How an expert AI tells younger (under 40) from older (65+) adults on an ECG.
The AI relies on these measurements, most important first:
1. PR interval: in older adults it tends to be higher (cut-point about 158 ms; lower points to younger).
2. T-wave height, lead II: in older adults it tends to be lower (cut-point about 0.22 mV; higher points to younger).
3. R-wave height, lead aVL: in older adults it tends to be higher (cut-point about 0.41 mV; lower points to younger).
4. QTc (Bazett): in older adults it tends to be higher (cut-point about 428 ms; lower points to younger).
5. S-wave depth, lead V1: in older adults it tends to be higher (cut-point about -0.78 mV; lower points to younger).
6. QRS axis: in older adults it tends to be lower (cut-point about 33 deg; higher points to younger).
No single cue is decisive. Count how many cues point to 'older', giving more weight to the cues near the top of the list, and decide by the balance.
```

![Weight of each cue in the lesson.](figures/cues.png)
*Weight of each cue in the lesson.*

The placebo group reads this lesson instead (same format, measurements the AI did not pick, random directions):

```
LESSON: How an expert AI tells younger (under 40) from older (65+) adults on an ECG.
The AI relies on these measurements, most important first:
1. P-wave axis: in older adults it tends to be lower (cut-point about 51 deg; higher points to younger).
2. T-wave axis: in older adults it tends to be higher (cut-point about 49 deg; lower points to younger).
3. T-wave height, lead V1: in older adults it tends to be lower (cut-point about 0.00 mV; higher points to younger).
4. Heart rate: in older adults it tends to be lower (cut-point about 72 bpm; higher points to younger).
5. P-wave height, lead II: in older adults it tends to be lower (cut-point about 0.10 mV; higher points to younger).
6. T-wave height, lead V5: in older adults it tends to be lower (cut-point about 0.22 mV; higher points to younger).
No single cue is decisive. Count how many cues point to 'older', giving more weight to the cues near the top of the list, and decide by the balance.
```

| Split | Surrogate agrees with AI | Rules agree with AI | Rules right (true age) | Scrambled rules right |
| --- | --- | --- | --- | --- |
| cue | 83.4% | 79.3% | 77.4% | 46.2% |
| test | 83.8% | 81.5% | 82.0% | 46.6% |
| transfer | 80.0% | 76.0% | 77.3% | 39.9% |


### Saliency check (does the AI look where the cues say?)

| ECG segment | Share of AI attention | Share of time | Density ratio |
| --- | --- | --- | --- |
| P | 13.3% | 19.9% | 0.67 |
| QRS | 45.3% | 17.2% | 2.64 |
| ST-T | 30.8% | 42.4% | 0.73 |

Densest attention: QRS. Segment holding most cue weight: QRS. Agree: True.


## 7. Experiment A: language-model learners, taught in words


### Which learner models count (known-answer gate and collapse rule)

Before a model's learners count, it must follow one explicit rule on 40 cases (at least 85% right), and fewer than half of its learners may give the same answer to 95% or more of the post-test cases. Models that fail are reported but do not count.

| Model | Gate accuracy | Gate | Learners collapsed (post) | Counts for H1 |
| --- | --- | --- | --- | --- |
| Qwen/Qwen2.5-3B-Instruct | 90.0% | pass | 69.2% | no |

Prior-knowledge probe: before any lesson, each model was asked, for each measurement, whether it is usually higher in older adults. The table counts how many of the AI's cue directions a model already 'knew'. High numbers mean the lesson may teach little that is new; the blinded run is the check.

| Model | Cue directions already known |
| --- | --- |
| Qwen/Qwen2.5-3B-Instruct | 4 of 6 |


### Results by group

Learners counted: 0 ().

| Group | n | Pre | Post | Delayed | New device |
| --- | --- | --- | --- | --- | --- |
| No training | 30 | 50.1% [49.9%, 50.2%] | 48.6% [47.5%, 49.7%] | 50.0% [50.0%, 50.0%] | 48.4% [46.8%, 50.0%] |
| Feedback only | 30 | 50.2% [50.0%, 50.5%] | 53.9% [51.8%, 56.1%] | 59.4% [55.7%, 63.1%] | 51.7% [50.2%, 53.1%] |
| AI-cue training | 30 | 50.4% [50.1%, 50.8%] | 52.5% [50.7%, 54.3%] | 55.2% [52.9%, 57.4%] | 52.4% [51.3%, 53.6%] |
| Scrambled cues (placebo) | 30 | 50.4% [50.1%, 50.8%] | 52.4% [50.4%, 54.4%] | 52.3% [50.2%, 54.5%] | 50.8% [49.6%, 51.9%] |

![Mean accuracy per group and test, with 95% confidence intervals.](figures/accuracy_llm.png)
*Mean accuracy per group and test, with 95% confidence intervals.*


### Hypotheses

Differences are in percentage points of accuracy. MDE = the smallest true difference this test would find 80% of the time. H1 needs p < 0.05 AND a difference of at least 3 points. H2-H7 are Holm-corrected.

|  | Claim | Diff | 95% CI | d | p | p (Holm) | MDE | Supported |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H1 (primary) | AI-cue training beats feedback only, right after training |  |  |  |  |  |  | skipped: no language model passed the known-answer gate and collapse rule |

> **What this means in plain words.** Experiment A gives no answer, because no language model passed the known-answer gate and collapse rule.


## 8. Experiment A, blinded-labels run (measurement names hidden)


### Which learner models count (known-answer gate and collapse rule)

Before a model's learners count, it must follow one explicit rule on 40 cases (at least 85% right), and fewer than half of its learners may give the same answer to 95% or more of the post-test cases. Models that fail are reported but do not count.

| Model | Gate accuracy | Gate | Learners collapsed (post) | Counts for H1 |
| --- | --- | --- | --- | --- |
| Qwen/Qwen2.5-3B-Instruct | 85.0% | pass | 78.3% | no |


### Results by group

Learners counted: 0 ().

| Group | n | Pre | Post | Delayed | New device |
| --- | --- | --- | --- | --- | --- |
| No training | 30 | 49.3% [48.0%, 50.7%] | 48.5% [47.5%, 49.5%] | 49.8% [49.4%, 50.1%] | 49.8% [49.0%, 50.7%] |
| Feedback only | 30 | 47.0% [45.5%, 48.5%] | 54.3% [51.7%, 57.0%] | 55.9% [53.0%, 58.9%] | 53.0% [50.9%, 55.1%] |
| AI-cue training | 30 | 47.2% [45.6%, 48.7%] | 52.1% [50.7%, 53.4%] | 53.4% [51.6%, 55.2%] | 51.1% [50.4%, 51.8%] |
| Scrambled cues (placebo) | 30 | 47.6% [46.2%, 49.0%] | 50.9% [50.2%, 51.7%] | 51.3% [50.1%, 52.6%] | 52.2% [50.4%, 54.0%] |

![Mean accuracy per group and test, with 95% confidence intervals.](figures/accuracy_llm_blinded.png)
*Mean accuracy per group and test, with 95% confidence intervals.*


### Hypotheses

Differences are in percentage points of accuracy. MDE = the smallest true difference this test would find 80% of the time. H1 needs p < 0.05 AND a difference of at least 3 points. H2-H7 are Holm-corrected.

|  | Claim | Diff | 95% CI | d | p | p (Holm) | MDE | Supported |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H1 (primary) | AI-cue training beats feedback only, right after training |  |  |  |  |  |  | skipped: no language model passed the known-answer gate and collapse rule |

> **What this means in plain words.** Experiment A (blinded run) gives no answer, because no language model passed the known-answer gate and collapse rule.


## 9. Experiment B: small neural-net learners, taught with data


### Results by group

Learners counted: 120 (No training 30, Feedback only 30, AI-cue training 30, Scrambled cues (placebo) 30).

| Group | n | Pre | Post | Delayed | New device |
| --- | --- | --- | --- | --- | --- |
| No training | 30 | 49.9% [49.8%, 50.0%] | 48.2% [45.6%, 50.8%] | 49.6% [47.4%, 51.8%] | 49.8% [47.5%, 52.1%] |
| Feedback only | 30 | 50.0% [50.0%, 50.0%] | 80.9% [79.7%, 82.1%] | 53.2% [50.7%, 55.8%] | 71.5% [70.1%, 72.9%] |
| AI-cue training | 30 | 49.9% [49.8%, 50.1%] | 80.8% [79.6%, 82.0%] | 58.8% [56.3%, 61.3%] | 72.6% [71.5%, 73.8%] |
| Scrambled cues (placebo) | 30 | 50.0% [49.9%, 50.0%] | 80.4% [79.0%, 81.8%] | 61.0% [58.3%, 63.7%] | 71.7% [70.2%, 73.2%] |

![Mean accuracy per group and test, with 95% confidence intervals.](figures/accuracy_nets.png)
*Mean accuracy per group and test, with 95% confidence intervals.*


### Hypotheses

Differences are in percentage points of accuracy. MDE = the smallest true difference this test would find 80% of the time. H1 needs p < 0.05 AND a difference of at least 3 points. H2-H7 are Holm-corrected.

|  | Claim | Diff | 95% CI | d | p | p (Holm) | MDE | Supported |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H1 (primary) | AI-cue training beats feedback only, right after training | -0.1 | [-1.7, +1.6] | -0.02 | 0.523 | - | 2.2 | no |
| H2 | AI-cue training beats no training, right after training | +32.7 | [+29.9, +35.5] | 5.78 | < 0.001 | < 0.001 | 3.7 | yes |
| H3 | Feedback only beats no training, right after training | +32.7 | [+29.9, +35.5] | 5.79 | < 0.001 | < 0.001 | 3.7 | yes |
| H4 | AI-cue training beats feedback only after the interference task (delayed) | +5.6 | [+2.1, +9.2] | 0.79 | 0.002 | 0.007 | 4.6 | yes |
| H5 | AI-cue training beats feedback only on the held-out device (transfer) | +1.1 | [-0.7, +2.8] | 0.31 | 0.119 | 0.356 | 2.3 | no |
| H6 | AI-cue learners improve more from pre- to post-test than feedback-only learners | +0.0 | [-1.7, +1.7] | 0.00 | 0.500 | 0.641 | 2.2 | no |
| H7 | The AI's real cues beat scrambled cues in the same format (placebo control) | +0.4 | [-1.3, +2.2] | 0.12 | 0.320 | 0.641 | 2.3 | no |

> **What this means in plain words.** In Experiment B, learners taught the AI's cues did not do clearly better than learners who only practised with feedback. The test could have found a true difference of about 2.2 points or more most of the time. The equivalence test also says any difference is smaller than 3 points.


### Preregistered preconditions

| Check | Value |
| --- | --- |
| teacher_test_auroc | 0.985 |
| teacher_is_expert (AUROC >= 0.80) | True |
| pretest_mean | 0.499 |
| new_skill (pretest mean <= 0.65) | True |


### Robustness checks (reported, not used for the verdict)

- Adjusting for each learner's pre-test score (ANCOVA): cue effect -0.0 points, two-sided p = 0.980.
- Same test on mean probability of the right answer: difference -0.2 points, p = 0.611.
- Share of learners whose answers collapsed to one class: Pre-test 98.3%, Post-test (primary) 0.0%, Delayed 9.2%, New device 0.0%.
- Transfer fraction (share of the AI teacher's advantage over chance reached at post-test): No training -0.04, Feedback only 0.72, AI-cue training 0.72, Scrambled cues (placebo) 0.71.


## 10. Problems found

- No self-check failed in this run. Known design limits are listed in ISSUES.md and in Section 11.


## 11. What we can and cannot say


### We can say

- Whether, in this setting, machine learners taught the AI's cues beat machine learners given only feedback (H1), and whether the AI's real cues beat a placebo lesson in the same format (H7).


### We cannot say

- That people would learn the same way. All learners here are machines.
- That the cues are how the AI 'really' works. They are a simple summary that agrees with the AI most of the time (Section 6).
- Anything clinical. Telling younger from older ECGs is a test task, not a diagnosis.


## 12. Words used

| Word | Meaning |
| --- | --- |
| AI teacher | The neural network trained here to tell younger from older adults from the raw ECG. |
| Cue | A plain rule taken from the AI, e.g. 'heart rate tends to be lower in older adults'. |
| Learner | One machine 'participant': one language-model session or one small neural network, with its own cases. |
| Feedback only | Learners practise on cases and are told the right answer, but get no rules. |
| Scrambled cues (placebo) | A lesson that looks the same but uses measurements the AI did not pick, with random directions. |
| Known-answer gate | A test that a model can follow one simple written rule. Models that cannot are not counted. |
| Collapse | When a learner gives the same answer to nearly every case. Its accuracy then says nothing. |
| AUROC | How well a model ranks older above younger; 0.5 is guessing, 1.0 is perfect. |
| p-value | How surprising a result would be if there were no real difference. Small = unlikely to be luck. |
| Holm correction | A rule that makes p-values stricter when several hypotheses are tested together. |
| MDE | Minimum detectable effect: the smallest true difference the test would find 80% of the time. |
| Equivalence test (TOST) | A test of whether a difference is smaller than a set margin, so 'no difference' can be claimed. |
| Preregistration | Publishing the plan before running the test, so it cannot be changed after. |
| SHA-256 | A file fingerprint. Change one letter and the fingerprint changes. |


## 13. How to check this report

- Master seed: 20261026. Every learner's cases and letters come from its own seed, so any learner can be re-made.
- Re-hash every output: python run_study.py verify runs/20261107T164810Z_study_pilot
- Compare the three fingerprints in Section 3 with prereg/LOCK.json and with the OSF registration.
- All learner-level results are in learners_*.csv; every single answer is in items_*.csv; checks.json holds the self-checks.


## 14. Common questions

| Question | Answer |
| --- | --- |
| Did an AI do the research? | No. The idea, hypotheses and every design choice are the author's. An AI assistant helped write code and drafts under the author's direction. Separately, AI models are what the study tests. |
| Why machines and not people? | It is cheaper, fully repeatable and can use 1,002 learners per group. If cues do not transfer even to machines, a study with people is unlikely to work; if they do, it is the next step. |
| Why a placebo lesson? | A lesson might help just because it is a lesson. The placebo has the same format and length, so only the AI's content differs. |
| Why hide the measurement names in one run? | Language models may already know how ECGs change with age. With names hidden, they can only use what the lesson tells them. |
| What if no model passes the gate? | Then Experiment A gives no answer for H1, and the report says so. The rule was fixed in advance. |
