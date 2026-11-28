# Data card

**Prepared by:** KEYUR SHAILESHKUMAR PATEL (ORCID: https://orcid.org/0009-0008-5261-043X),
MSc Artificial Intelligence student, Department of Computer Science, Brunel University London, UK; Independent researcher.
Conducted independently; not an official Brunel University London project. Version 1.0.1. Licence: CC BY 4.0.

| Item | Detail |
| --- | --- |
| Sources | PTB-XL v1.0.3 (Wagner et al., 2020) and PTB-XL+ v1.0.1 (Strodthoff et al., 2023), PhysioNet, CC BY 4.0 |
| Content | 21,799 10-second 12-lead ECGs (18,869 patients), 100 Hz; 12SL median-beat features |
| Labels used | younger = age ≤ 39, older = age ≥ 65; ages 40-64 excluded; ages >89 (coded 300) recoded to 90 |
| Features used | 18-measurement panel fixed in advance (configs/study.yaml) |
| Integrity | Every file checked against PhysioNet's SHA256SUMS.txt; record and patient counts compared with 21,799 and 18,869 |
| Splits | By patient: teacher train/val = folds 1-8 (validation patients by a fixed SHA-256 rule); cue extraction = fold 9; learner tests = fold 10; transfer = held-out device (other-device ECGs of those patients set aside) |
| Within a learner | One ECG per patient per set; pre, post and delayed tests use different patients |
| Personal data | De-identified by the dataset creators; no re-identification attempted; nothing is redistributed in this package |
| Human participants | None. All learners are software |
| Known limits | One country and one hospital network; cues describe what this AI uses and may include dataset quirks |
