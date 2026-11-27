# Exploratory pilot on real data: report

*AI-to-learner knowledge transfer study (ECG age).*

**Author:** KEYUR SHAILESHKUMAR PATEL · ORCID [0009-0008-5261-043X](https://orcid.org/0009-0008-5261-043X)  
MSc Artificial Intelligence student, Department of Computer Science, Brunel University London, UK · Independent researcher

*Conducted independently; not an official Brunel University London project.*

**Use of AI tools.** The research questions, hypotheses and design choices are the author's. The code and this draft were written with help from an AI assistant (Claude, by Anthropic) under the author's direction and checked by the author. Separately, the study uses AI models as its objects of study.  
**Licence.** Text and figures CC BY 4.0; code MIT.

> **Status: EXPLORATORY. Nothing here tests a hypothesis.** This pilot was run before preregistration, on purpose, to find real-data problems. It used PTB-XL folds 1-9 only. Fold 10 and the held-out recording device (AT-6 C 5.5) were sealed and were never opened (decision D7). Every number below can change in the confirmatory study.

## 1. The short answer

1. **The data and the AI teacher are fine.** PTB-XL matched PhysioNet's checksums (87,203 files). The teacher reached AUROC 0.985 on the pilot's test fold (precondition: at least 0.80).
2. **The pilot found a reproducibility bug** (P11): the same seed gave different teachers and different cue lists (5 cues, then 8). It is fixed in version 0.3.
3. **The cue list was fragile, and is now stable** (P14, decision D9). With five teachers voting, four cues are unanimous (PR interval, T-wave height in lead II, R-wave height in aVL, QRS axis) and two more pass 3 of 5 (QTc, S-wave depth in V1). As a simple tally these six rules are right 82.0% of the time.
4. **The placebo lesson was misleading, and is now neutral** (P15, decision D10): it scored 36% on its own; the fixed re-draw rule brings it to 46-47%.
5. **Experiment B (small nets) hits a ceiling.** Practice alone reaches 80.9%; adding the AI's cues gives 80.8%. As designed it is unlikely to show a cue benefit (P16, open decision).
6. **Experiment A (Qwen2.5-3B) is dominated by letter bias.** By the planned accuracy score every group is near chance (48.6% to 53.9%) and the preregistered collapse rule marks the model uninformative. But the model's probabilities do rank the ECGs: with a bias-corrected score (defined after seeing the data, so exploratory) the groups separate: no training 49.9%, scrambled 58.7%, AI cues 65.3%, feedback only 69.2% (P19, open decision).
7. **The language-model code could not run long prompts** without running out of memory (P17, P18). Fixed in 0.3.
8. **In this pilot the AI's cues did not beat practice with feedback**, in either experiment, by any score. By the bias-corrected score they beat the placebo in the main run of Experiment A (+6.6 points); in the blinded run the gap was smaller (+3.6 points) and not clear. This is one small model and 30 learners per group, so it is a warning for the design, not a result.

## 2. What was run

| Run | Code version | What | Status |
| --- | --- | --- | --- |
| v0.2 pilot, run 1 | 0.2.0 | Teacher, cues, Experiment B (30 nets per group) | Complete. First run on this machine (signals read fresh from the WFDB files). |
| v0.2 pilot, run 2 | 0.2.0 | Teacher, cues, then Experiment A | Teacher and cues complete. Showed the reproducibility bug. Experiment A stopped at 20 learners (machine restart, results not saved: P13). |
| Consensus check | 0.3.0 (in development) | Five teachers and cue consensus only | Complete. Same cues as the later full run. |
| v0.3 pilot | 0.3.0 | Five teachers, cues, Experiment B (30 per group), Experiment A main and blinded (Qwen2.5-3B, 30 per group) | Complete: Experiment B, Experiment A main (120 learners) and blinded (120 learners). |

All runs used master seed 20261026, the protected pilot split (teacher folds 1-7, cue fold 8, test fold 9; transfer device CS-12, the largest remaining device), CPU only (4 cores, no GPU), and `python run_study.py run --pilot 30`.

The v0.3 run was interrupted several times (machine restarts and two out-of-memory kills) and resumed each time. Learners are seeded one by one, so a resumed learner gets exactly the prompts it would have got. Two memory fixes (P17, P18) were made between resumes; they change how the same computation is batched, not what is computed. Learners finished before and after each fix are in the same files. For the last resumes a small helper script (`pilot/resume_llm_only.py`) continued the language-model learners from the saved cues and settings without retraining the five teachers; a final normal `--resume` then retrained the teachers, checked that they match the saved ones (self-check C14), and wrote the self-checks and the run report. See Section 9.

**Models.** Llama-3.2-3B and Gemma-2-2B are licence-gated and no Hugging Face token was set, so the pilot used Qwen2.5-3B-Instruct only (decision D8; commit `aa8e72537993ba99e69dfaafa59ed015b17504d1`, the head of the repository on 2026-11-07; not pinned in the config).

## 3. Data and cohort (pilot split)

| Role | ECGs | Older (65+) | Younger (under 40) |
| --- | --- | --- | --- |
| teacher train | 4624 | 3199 | 1425 |
| teacher val | 527 | 352 | 175 |
| cue | 740 | 518 | 222 |
| test | 610 | 404 | 206 |
| transfer | 2119 | 1921 | 198 |
| unused | 69 | 67 | 2 |

21,799 records, 12,336 in the two age groups. **Sealed and never opened: 3,646 ECGs of 3,347 patients** (fold 10 and device AT-6 C 5.5). 8,689 ECGs of 7,327 patients remained. No patient is in two roles (self-check C1).

Note for the design: the pilot's transfer device (CS-12) is 91% older adults, so learners' transfer tests draw their younger half from a small pool (198 ECGs).

## 4. The AI teacher

| Teacher | Best epoch | AUROC, test fold | Accuracy |
| --- | --- | --- | --- |
| v0.2 run 1 (fresh read) | 18 | 0.987 | 91.3% |
| v0.2 run 2 (cached read) | 10 | 0.989 | 92.8% |
| v0.3 primary teacher | 10 | 0.985 | 92.6% |
| v0.3 consensus teacher 1 | 21 | 0.983 |  |
| v0.3 consensus teacher 2 | 10 | 0.979 |  |
| v0.3 consensus teacher 3 | 11 | 0.983 |  |
| v0.3 consensus teacher 4 | 12 | 0.989 |  |

A plain logistic regression on the 18 panel measurements reaches AUROC 0.947 (88.4%): most of the age signal the teacher uses is already in the numbers learners see. On the transfer device the primary teacher scores AUROC 0.956.

**The reproducibility bug (P11).** Runs 1 and 2 used the same code, seed and data, and should have been identical. They were not: the per-lead scaling was summed in 32-bit floats over about 5 million values, and the result depended on whether the signals had just been read from disk or loaded from the cache (standard deviations 0.5% apart). Version 0.3 sums in 64-bit; a fresh read and a cached read now agree bit for bit, and the v0.3 run reproduced the separate consensus check exactly (same cues, same teacher AUROC).

## 5. The cues

| Measurement | v0.2 run 1 (5 cues) | v0.2 run 2 (8 cues) | v0.3: teachers that chose it | v0.3 cue |
| --- | --- | --- | --- | --- |
| T-wave height, lead II | yes | yes | 5 of 5 | yes |
| QRS axis | yes | yes | 5 of 5 | yes |
| R-wave height, lead aVL | yes | yes | 5 of 5 | yes |
| PR interval | yes | yes | 5 of 5 | yes |
| S-wave depth, lead V1 | - | yes | 3 of 5 | yes |
| QTc (Bazett) | - | yes | 3 of 5 | yes |
| T-wave height, lead V1 | - | yes | 2 of 5 | - |
| T-wave height, lead V5 | yes | - | 1 of 5 | - |
| ST level, lead V5 | - | yes | 1 of 5 | - |

**Why the list swung (P14).** The rule takes the sparsest setting with at least 5 stable features. In run 1 that setting gave exactly 5; with a barely different teacher it gave 4, so the rule moved to a looser setting and took 8. **Decision D9:** train five teachers (seeds 0-4), run the same selection on each, keep features chosen by at least 3. All five teachers agree on the direction of every kept cue.

**The v0.3 lesson, as learners saw it:**

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

| On the pilot test fold | Agrees with true age group | Agrees with the teacher |
| --- | --- | --- |
| AI cue rules, applied as a tally | 82.0% | 81.5% |
| Surrogate model on the 6 cues | 86.2% | 83.8% |
| Scrambled (placebo) rules, as a tally | 46.6% | - |
| Teacher itself | 92.6% | 100% |

**The placebo (P15, decision D10).** With the v0.3 cues the first placebo draw scored 36% as a tally: its random directions mostly pointed the wrong way, so it could have taught the opposite and inflated H7. The fixed rule now re-draws (seeded) until the tally is within 45-55% on the cue fold. Here: draw 5, 46.2% on the cue fold, 46.6% on the test fold.

## 6. Experiment B: small neural nets (30 per group)

**v0.3 (consensus cues, neutral placebo):**

| Group | Learners | Before | Right after | After interference | New device |
| --- | --- | --- | --- | --- | --- |
| No training | 30 | 49.9% | 48.2% | 49.6% | 49.8% |
| Feedback only | 30 | 50.0% | 80.9% | 53.2% | 71.5% |
| AI cues | 30 | 49.9% | 80.8% | 58.8% | 72.6% |
| Scrambled cues (placebo) | 30 | 50.0% | 80.4% | 61.0% | 71.7% |

**v0.2 run 1 (5 cues, old placebo):**

| Group | Learners | Before | Right after | After interference | New device |
| --- | --- | --- | --- | --- | --- |
| No training | 30 | 49.9% | 48.3% | 49.5% | 49.6% |
| Feedback only | 30 | 50.0% | 80.8% | 53.3% | 71.4% |
| AI cues | 30 | 49.9% | 81.0% | 60.4% | 72.3% |
| Scrambled cues (placebo) | 30 | 50.0% | 80.7% | 56.3% | 72.7% |

| Comparison (right after training) | Learners | Difference (points) | 95% range | Two-sided p |
| --- | --- | --- | --- | --- |
| AI cues minus feedback only (the H1 comparison) | 30 vs 30 | -0.1 | -1.8 to +1.7 | 0.954 |
| AI cues minus scrambled cues (the H7 comparison) | 30 vs 30 | +0.4 | -1.4 to +2.3 | 0.641 |
| AI cues minus no training | 30 vs 30 | +32.7 | +29.7 to +35.6 | <0.001 |
| Feedback only minus no training | 30 vs 30 | +32.7 | +29.8 to +35.7 | <0.001 |

**Reading.** Sixty-four labelled ECGs are enough for a small net to reach about 81% by itself; the extra cue-prediction loss adds nothing right after training. After the interference task the cue and scrambled groups keep more than feedback only, by a similar amount, so that difference comes from the extra training signal, not from the AI's content. This is the ceiling already seen in synthetic world W1 (issues P9, P10), now on real data (P16).

## 7. Experiment A: language-model learners (Qwen2.5-3B, 30 per group)

### Main run (real measurement names): 120 learners

Known-answer gate: 90.0% (bar 85%): passed. Share of learners giving nearly one answer on the post-test: 69.2%; by the preregistered rule (over half) the model is **uninformative**.

**(a) Planned score: accuracy (answer = the letter with the higher probability).**

| Group | Learners | Before | Right after | After interference | New device |
| --- | --- | --- | --- | --- | --- |
| No training | 30 | 50.1% | 48.6% | 50.0% | 48.4% |
| Feedback only | 30 | 50.2% | 53.9% | 59.4% | 51.7% |
| AI cues | 30 | 50.4% | 52.5% | 55.2% | 52.4% |
| Scrambled cues (placebo) | 30 | 50.4% | 52.4% | 52.3% | 50.8% |

Share of items on which a learner gave its more frequent answer (50% = balanced, 100% = one answer only):

| Group | Learners | Before | Right after | After interference | New device |
| --- | --- | --- | --- | --- | --- |
| No training | 30 | 99.9% | 89.6% | 100.0% | 89.2% |
| Feedback only | 30 | 99.8% | 94.1% | 84.9% | 94.2% |
| AI cues | 30 | 99.6% | 96.3% | 89.3% | 96.3% |
| Scrambled cues (placebo) | 30 | 99.6% | 92.9% | 88.0% | 91.6% |

**(b) Exploratory, defined after seeing the data: bias-corrected accuracy.** Each learner's items are split at that learner's own median probability of 'older' on that test (tests are half older, half younger, so chance is 50%).

| Group | Learners | Before | Right after | After interference | New device |
| --- | --- | --- | --- | --- | --- |
| No training | 30 | 56.8% | 49.9% | 48.0% | 44.5% |
| Feedback only | 30 | 55.1% | 69.2% | 68.6% | 60.6% |
| AI cues | 30 | 54.6% | 65.3% | 58.8% | 62.0% |
| Scrambled cues (placebo) | 30 | 53.2% | 58.7% | 55.5% | 56.3% |

| Comparison (right after training) | Learners | Difference (points) | 95% range | Two-sided p |
| --- | --- | --- | --- | --- |
| AI cues minus feedback only (the H1 comparison) | 30 vs 30 | -3.9 | -7.8 to +0.0 | 0.052 |
| AI cues minus scrambled cues (the H7 comparison) | 30 vs 30 | +6.6 | +1.6 to +11.7 | 0.011 |
| AI cues minus no training | 30 vs 30 | +15.4 | +11.3 to +19.4 | <0.001 |
| Feedback only minus no training | 30 vs 30 | +19.3 | +15.5 to +23.0 | <0.001 |

**(c) Exploratory: AUROC of the learner's probability of 'older'** (0.50 = no information, 1.00 = perfect ranking).

| Group | Learners | Before | Right after | After interference | New device |
| --- | --- | --- | --- | --- | --- |
| No training | 30 | 0.569 | 0.470 | 0.456 | 0.422 |
| Feedback only | 30 | 0.559 | 0.739 | 0.728 | 0.644 |
| AI cues | 30 | 0.560 | 0.713 | 0.624 | 0.668 |
| Scrambled cues (placebo) | 30 | 0.541 | 0.621 | 0.570 | 0.587 |

### Blinded run (names replaced by 'Measure 1..18'): 120 learners

Known-answer gate: 85.0% (bar 85%): passed. Share of learners giving nearly one answer on the post-test: 78.3%; by the preregistered rule (over half) the model is **uninformative**.

**(a) Planned score: accuracy (answer = the letter with the higher probability).**

| Group | Learners | Before | Right after | After interference | New device |
| --- | --- | --- | --- | --- | --- |
| No training | 30 | 49.3% | 48.5% | 49.8% | 49.8% |
| Feedback only | 30 | 47.0% | 54.3% | 55.9% | 53.0% |
| AI cues | 30 | 47.2% | 52.1% | 53.4% | 51.1% |
| Scrambled cues (placebo) | 30 | 47.6% | 50.9% | 51.3% | 52.2% |

Share of items on which a learner gave its more frequent answer (50% = balanced, 100% = one answer only):

| Group | Learners | Before | Right after | After interference | New device |
| --- | --- | --- | --- | --- | --- |
| No training | 30 | 93.5% | 96.3% | 99.8% | 95.5% |
| Feedback only | 30 | 94.0% | 94.5% | 87.6% | 94.7% |
| AI cues | 30 | 93.8% | 97.9% | 95.3% | 98.4% |
| Scrambled cues (placebo) | 30 | 94.1% | 98.6% | 96.5% | 97.7% |

**(b) Exploratory, defined after seeing the data: bias-corrected accuracy.** Each learner's items are split at that learner's own median probability of 'older' on that test (tests are half older, half younger, so chance is 50%).

| Group | Learners | Before | Right after | After interference | New device |
| --- | --- | --- | --- | --- | --- |
| No training | 30 | 41.7% | 44.7% | 46.0% | 46.5% |
| Feedback only | 30 | 42.5% | 66.0% | 64.7% | 61.7% |
| AI cues | 30 | 37.9% | 66.6% | 66.9% | 64.2% |
| Scrambled cues (placebo) | 30 | 42.1% | 63.0% | 61.0% | 59.9% |

| Comparison (right after training) | Learners | Difference (points) | 95% range | Two-sided p |
| --- | --- | --- | --- | --- |
| AI cues minus feedback only (the H1 comparison) | 30 vs 30 | +0.7 | -3.8 to +5.1 | 0.765 |
| AI cues minus scrambled cues (the H7 comparison) | 30 vs 30 | +3.6 | -0.5 to +7.8 | 0.087 |
| AI cues minus no training | 30 vs 30 | +22.0 | +17.7 to +26.2 | <0.001 |
| Feedback only minus no training | 30 vs 30 | +21.3 | +16.9 to +25.6 | <0.001 |

**(c) Exploratory: AUROC of the learner's probability of 'older'** (0.50 = no information, 1.00 = perfect ranking).

| Group | Learners | Before | Right after | After interference | New device |
| --- | --- | --- | --- | --- | --- |
| No training | 30 | 0.387 | 0.428 | 0.437 | 0.444 |
| Feedback only | 30 | 0.395 | 0.727 | 0.706 | 0.672 |
| AI cues | 30 | 0.344 | 0.733 | 0.727 | 0.689 |
| Scrambled cues (placebo) | 30 | 0.395 | 0.673 | 0.671 | 0.640 |

**Reading.**

- Qwen passes the short known-answer gate, so it can apply one written rule. On the real task it answers almost only 'A' or only 'B' whatever the ECG. Because letters are balanced across learners, this bias does not favour any group; it pushes every group's accuracy towards 50% and hides differences.
- Underneath the bias the probabilities carry information after training. That is why scores (b) and (c) separate the groups while score (a) does not.
- Practice with feedback gave the largest gain. The AI's cues gave a gain of similar size or a little smaller, and clearly more than the placebo lesson in the main run.
- In the blinded run the pre-test is *below* chance by score (b): with meaningless names and no lesson, the model's prior ordering of the numbers runs slightly against the truth. Training reverses it.
- With no training the model does not improve, as it should not.
- The placebo group also improves, because every taught group sees the same 24 practice cases with correct answers. In the main run the placebo group ends well below feedback only (58.7% against 69.2%): a wrong lesson seems to get in the way of learning from the cases. The AI's lesson, which is right, also ends a little below feedback only. So in this pilot a written lesson did not add to the practice cases, whatever its content; the open question is whether that is the design (24 cases already teach the pattern) or this model.
- The prior-knowledge probe says 'higher in older adults' for 17 of 18 measurements with near-certainty, which looks like the same yes-bias, so the probe is not informative as written (P20).

**Prompt length.** Feedback and cue prompts are about 9,000 to 12,000 tokens (24 practice ECGs of 18 numbers each, plus the interference block). This is long for a 3B model and may be part of why answers collapse; the pilot cannot separate the two.

## 8. Self-checks

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

## 9. Problems found, and what was done

| # | Problem | Severity | Status |
| --- | --- | --- | --- |
| P11 | Same seed, different teacher and cues (32-bit summation depended on memory layout) | High | Fixed in 0.3 (64-bit); test added |
| P12 | `check-data` failed on a fresh folder | Low | Fixed in 0.3 |
| P13 | Results saved only every 25 learners; 20 finished learners lost in a restart | Medium | Fixed in 0.3 (saved after every learner) |
| P14 | Cue list fragile (5 vs 8 cues) | High (design) | Decision D9: consensus of 5 teachers; self-check C16 |
| P15 | Placebo lesson could point the wrong way (36%) | Medium (design) | Decision D10: seeded re-draw to 45-55%; self-check C17 |
| P16 | Experiment B: practice alone reaches the ceiling | Design | **Open: author's decision** |
| P17 | Out of memory on long prompts (13.9 GB) | High | Fixed in 0.3 (prompt fed in chunks; last-position logits only) |
| P18 | Out of memory again in the cue group (cache copied 8 times) | High | Fixed in 0.3 (memory budget for cache copies) |
| P19 | Experiment A: letter bias swamps the accuracy score; the collapse rule would exclude the model | High (design) | **Open: author's decision** on the outcome measure and the collapse rule |
| P20 | Prior-knowledge probe answers 'higher' for almost everything | Low | **Open**: ask each question both ways, or drop the probe |
| P21 | Run-time estimates in the README were far too low for CPU; long prompts cost about 11 minutes per learner on 4 cores | Low | README updated with measured times |
| P22 | Pilot seal is applied after the feature table is loaded: a unit check and the missing-value count run over all rows, sealed ones included | Low | Disclosed exactly in `prereg/PILOT_DISCLOSURE.md` section 2; no age-feature relation is computed there |

Details are in `ISSUES.md`.

## 10. What the pilot means for the preregistration

Decisions already taken by the author during the pilot: D7 (protected split), D8 (Qwen only for the pilot), D9 (cue consensus), D10 (neutral placebo), D11 (rerun on v0.3 at 30 per group).

Open decisions, for the author, before freezing:

1. **Outcome measure for Experiment A (P19).** Keep raw accuracy as primary, or switch to a bias-free score (bias-corrected accuracy or AUROC), and what the collapse rule should then exclude.
2. **Experiment B (P16).** Keep as is (expecting a null), make practice scarcer so there is room for cues to help, or drop it to a secondary role.
3. **Is H1 the right primary?** In this pilot the AI's cues did not beat feedback-only anywhere. H7 (real cues beat placebo) and 'cues beat no training' looked more promising. One small model, so this is weak evidence, but the choice of primary must be made before the real data are seen.
4. **Prompt length and practice size** for language models (24 practice ECGs make 9-12k-token prompts).
5. **Models:** add the Hugging Face token so Llama-3.2-3B and Gemma-2-2B run; pin all three commits.

**What must be disclosed in the preregistration** is drafted in `prereg/PILOT_DISCLOSURE.md`: this pilot happened, which data it touched and did not touch, what was seen, and which design choices were changed because of it.

## 11. Limits of this pilot

- One language model of three; 30 learners per group; one seed.
- Folds 1-9 only, with a different split from the confirmatory study (teacher on 7 folds instead of 8; a different transfer device).
- Scores (b) and (c) in Section 7 were chosen after looking at the data.
- CPU run in bfloat16; probabilities for very long prompts are precise to a few hundredths (P17).
- The v0.3 run was resumed several times and two memory fixes were applied between resumes.
- The v0.2 Experiment A results (20 learners) were lost and are not reported.

## 12. Files

`pilot/v0.3_run/` holds the full outputs of the v0.3 pilot run (learner-level and item-level CSVs, lessons, example prompts, cues, teacher metrics, logs, manifest, SHA256SUMS). `pilot/v0.2_run1/` and `pilot/v0.2_run2/` hold the two v0.2 runs. Teacher weights and cohort tables derived from PTB-XL are included; PTB-XL itself is not (download it from PhysioNet).
