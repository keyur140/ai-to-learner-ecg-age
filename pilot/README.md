# pilot/: the exploratory real-data pilot (before preregistration)

*AI-to-learner knowledge transfer study (ECG age). Author: KEYUR SHAILESHKUMAR PATEL (ORCID https://orcid.org/0009-0008-5261-043X). MSc Artificial Intelligence student, Department of Computer Science, Brunel University London, UK; Independent researcher. Conducted independently; not an official Brunel University London project.*

**Everything in this folder is EXPLORATORY.** It was produced before preregistration, on PTB-XL folds 1-9 only. Fold 10 and the held-out recording device (AT-6 C 5.5) were sealed and never opened (decision D7). Nothing here tests a hypothesis, and nothing here is combined with the confirmatory run.

| File or folder | What it is |
| --- | --- |
| `EXPLORATORY_PILOT_REPORT.md`, `.pdf` | The report: what was run, what was seen, what was fixed, what is still open. Start here. |
| `PILOT_FOLLOWUP.md`, `followup/` | Follow-up of 2026-11-14 (version 0.4.0): the pilot re-scored with the bias-corrected score, the finer answer read-out (P23), and a practice-size sweep for Experiment B. |
| `PREREG_CHANGES_PROPOSED.md` | The wording changes the preregistration needed after the pilot, old text beside new. Approved by the author on 2026-11-15 and applied; frozen in version 1.0.0 on 2026-11-27. |
| `v0.3_run/` | Full output of the version 0.3.0 pilot run (`python run_study.py run --pilot 30 --llm-models Qwen/Qwen2.5-3B-Instruct`): learner and item files for Experiment B, Experiment A main and blinded; cues; lessons; example prompts; five teachers; `checks.json` (17 of 17 pass); `REPORT.pdf`; `run.log`; `manifest.json`; `SHA256SUMS.txt`. |
| `v0.2_run1/` | First version 0.2.0 pilot run: teacher, cues, Experiment B. |
| `v0.2_run2/` | Second version 0.2.0 pilot run: teacher and cues only. Same seed as run 1, different result: this is the reproducibility bug P11. |
| `make_pilot_report.py` | Builds the report and the disclosure from the run folders (paths inside are those of the machine it ran on). |
| `resume_llm_only.py` | Helper used for the last resumes of the version 0.3.0 run: continues the language-model learners from the saved cues and settings without retraining the teachers. `v0.3_run/resume_llm_only.log` is its log. |
| `consensus_check.py` | Teacher-and-cues-only check of the five-teacher consensus (decision D9), run before the full version 0.3.0 pilot. |
| `md2pdf.py`, `build_package.py` | Helpers that printed the report to PDF and assembled this package. |

The disclosure that becomes part of the preregistration is `../prereg/PILOT_DISCLOSURE.md`.

To re-check the run files: `python run_study.py verify pilot/v0.3_run` from the package root.

The run folders hold tables and teacher weights derived from PTB-XL and PTB-XL+ (Wagner et al., 2020; Strodthoff et al., 2023; both CC BY 4.0). The datasets themselves are not included; `python run_study.py check-data` downloads them from PhysioNet and checks them against PhysioNet's checksums.
