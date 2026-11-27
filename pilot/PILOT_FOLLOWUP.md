# Follow-up to the exploratory pilot (2026-11-14)

*AI-to-learner knowledge transfer study (ECG age). Author: KEYUR SHAILESHKUMAR PATEL (ORCID https://orcid.org/0009-0008-5261-043X). MSc Artificial Intelligence student, Department of Computer Science, Brunel University London, UK; Independent researcher. Conducted independently; not an official Brunel University London project.*

> **Status: EXPLORATORY.** Everything here used PTB-XL folds 1-9 only. Fold 10 and the held-out device (AT-6 C 5.5) are still sealed. Drafted with help from an AI assistant (Claude, by Anthropic) under the author's direction.

## 1. Decision D12 is built in (version 0.4.0)

On 2026-11-14 the author chose **bias-corrected accuracy** as the primary score of Experiment A. It is now in the code (`experiment_llm.primary_score: bias_corrected`). The raw score and each learner's AUROC are reported beside it. The pilot data were re-scored with the package's own analysis, and it gives the same numbers as the pilot report:

| Run | No training | Feedback only | AI cues | Scrambled | H1: cues minus feedback | H7: cues minus scrambled |
| --- | --- | --- | --- | --- | --- | --- |
| Main | 49.9% | 69.2% | 65.3% | 58.7% | -3.9 points (not supported) | +6.6 points, Holm p = 0.02 |
| Blinded | 44.7% | 66.0% | 66.6% | 63.0% | +0.7 points (not supported) | +3.6 points, Holm p = 0.13 |

**Proposed with it, for the author to confirm:** the old collapse rule removed a model when its learners answered one letter. With the new score that no longer hides learning, so the collapse flag is still reported but no longer removes a model. In its place a *flat rule*: a learner is flat on a test if more than half of its items tie at its own median (they cannot be ranked); a model does not count if more than half of its learners are flat on the post-test. In the pilot no Qwen learner is flat, so Qwen counts.

## 2. New problem found: answer probabilities were too coarse (P23)

The models run in a 16-bit number format (bfloat16). The model's own output had only about 8 significant bits, so the 40 answers of one test fell on about 13 distinct probability values. On average 12% of a learner's items were tied at its median, and the tie rate differed by group (post-test, main run: feedback only 6.5%, AI cues 11.6%, scrambled 16.5%). Tied items get half credit, which pulls a score towards 50%. So part of the gap between feedback and cues in the pilot may come from this, not from learning. The pilot cannot tell how much.

**Fix in 0.4.0:** the two answer letters are now re-read in 32-bit precision from the model's last internal state (`experiment_llm.answer_precision: float32`). The model's own 16-bit values are kept as a check. Tested on Qwen2.5-3B: the two read-outs agree within the 16-bit rounding step (largest difference 0.124 in logit, 0.03 in probability), and all 16 test answers stayed on the same side.

**Still open:** the 16-bit format also adds noise inside the model. Reading the same prompt in differently sized pieces moved a probability by up to 0.06. A 32-bit run on a GPU would remove this; it needs about twice the memory. This is a choice for the confirmatory run.

## 3. Faster: the lesson is read once per learner

The post-test and the new-device test use the same prompt, and the delayed test only adds the interference block. The code now keeps the prompt in memory between a learner's tests instead of reading it three times. On the test above the same three tests took 269 seconds instead of 354 (24% less), with identical answers for the shared prompt.

## 4. Experiment B: fewer practice ECGs do not help the cues (P16)

Thirty small nets per group at each practice size, same split, cues and seeds as the pilot. Accuracy right after training:

| Practice ECGs | No training | Feedback only | AI cues | Scrambled | Cues minus feedback | Two-sided p |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | 51.6% | 64.6% | 60.0% | 63.8% | -4.6 | 0.08 |
| 8 | 48.2% | 68.6% | 68.4% | 69.5% | -0.2 | 0.91 |
| 16 | 52.1% | 73.8% | 73.1% | 72.0% | -0.7 | 0.62 |
| 32 | 49.3% | 76.6% | 77.2% | 75.4% | +0.7 | 0.53 |
| 64 (as planned) | 48.2% | 80.8% | 80.8% | 80.5% | 0.0 | 1.00 |
| 128 | 47.6% | 82.8% | 84.0% | 83.9% | +1.2 | 0.09 |

**Reading.** The idea that practice alone "hits a ceiling" and leaves no room for the cues is wrong: with only 8 or 16 practice ECGs, practice reaches 69% to 74%, far below the teacher, and the cues still add nothing. The scrambled cues do as well as the real ones at every size. So for these small nets, the extra task of predicting the cue values does not carry the AI's knowledge into the age decision, whatever the amount of practice. Changing the practice size will not rescue Experiment B. After the interference task the cue and scrambled groups keep a little more than feedback only at most sizes, again by similar amounts.

## 5. What still needs the author's judgment

*Update, later on 2026-11-14: points 1 to 3 are decided. D15: the flat rule is confirmed. D13: Experiment B stays as designed but is secondary. D14: H1 stays the primary hypothesis.*

1. Confirm or change the flat rule (section 1).
2. Experiment B (section 4): keep it and expect no difference, change how the cues are given to the nets, or drop it to a secondary role.
3. Which hypothesis is primary, given that the cues did not beat feedback-only practice in any pilot run.
4. Practice size and prompt length in Experiment A. A short Qwen run with 6 practice cases would show whether the cues add more when practice is scarce (about 5 hours here).
5. 32-bit runs on a GPU for the confirmatory study (section 2).
6. The prior-knowledge probe (P20), and the Hugging Face token for the other two models.

## 6. Files

`pilot/followup/`: `nets_practice/` (learner files and log of the sweep), `backend_microtest.out` and `.json`, and the scripts `nets_practice_sweep.py`, `backend_microtest.py`, `llm_practice_sweep.py` (written, not run).
