# OSF registration

*AI-to-learner knowledge transfer study (ECG age). Author: KEYUR SHAILESHKUMAR PATEL (ORCID https://orcid.org/0009-0008-5261-043X). Conducted independently; not an official Brunel University London project.*

| | |
| --- | --- |
| Registration | https://osf.io/s8rdx/ |
| DOI | [10.17605/OSF.IO/S8RDX](https://doi.org/10.17605/OSF.IO/S8RDX) |
| Template | OSF Preregistration |
| Registered | 2026-11-27, 23:56 UTC; public |
| Plan version | 1.0.1 (`prereg/preregistration.md`, `prereg/PILOT_DISCLOSURE.md`) |
| Confirmatory data seen before registering | None (PTB-XL fold 10 and the held-out device were never opened) |

## Lock fingerprints (SHA-256, `prereg/LOCK.json`, locked 2026-11-27T19:17:38Z)

| What | SHA-256 |
| --- | --- |
| Plan + pilot disclosure | `26081e90a8882ad5a29825a0902e95df927c9f2757568d711e135f2fb4668d3c` |
| Settings (`configs/study.yaml`) | `4c548d8a459ab1772a6b916b6c6b49b0b89b3fa60711646c8d4558de94747e48` |
| Code (`src/**/*.py`) | `7c4ea7200b3661dbb6751772372fad21e3307f8770487634bf90f3ed339dbfbe` |
| Run script (`run_study.py`) | `e44176b4deb881402c5ee6d642a6b89385b0a24facd7219e6ab187324a2b3c72` |

## Files archived with the registration

All five were checked after archiving: the SHA-256 on OSF matches the local file.

| File | SHA-256 |
| --- | --- |
| `LOCK.json` | `8354417e221b2f8703f3fc62ef9a0cbc5c67e9c3baf709a7c1d6d1d5acfcbe07` |
| `PILOT_DISCLOSURE_v1.0.1.pdf` | `e672f91c8e6face83b59ccde6d48a6c749b981d9c9e5447bd198715da6f5bc92` |
| `VALIDATION_REPORT.pdf` | `eb585c86296d209235cf65d31f17e50d0e621e5e8fb89804393bd9efe3fff65a` |
| `ai2human_v1.0.1.zip` | `9496c42a139339d01c22d86602c023fea53eb434ab2e3c6ed3efe18c610356be` |
| `preregistration_v1.0.1.pdf` | `4fbf094c550d0ba977bd420afe708572ebde33d78354f5df6de02eb7080ed23b` |

## Checks before and after submitting

- The answer boxes were built word for word from `prereg/preregistration.md`. The only changes were removing Markdown marks and writing out the characters that OSF does not accept.
- Before submitting, the draft was read back from OSF and checked: 91 of 91 checks passed. These covered every answer, the tick boxes, the required fields, the file hashes, the lock fingerprints, the absence of any email, and draft status.
- After submitting, the registration was read back: every answer is identical to the plan.

## Note on this repository after registration

This file and `README.md` were edited after registration to add the OSF link. Every other file matches `PACKAGE_SHA256SUMS.txt`, which describes the registered package `ai2human_v1.0.1.zip`. To check them: `grep -v -e ./README.md PACKAGE_SHA256SUMS.txt | sha256sum -c`.
