# AI-to-learner knowledge transfer study (ECG age)
# Author: KEYUR SHAILESHKUMAR PATEL (ORCID: https://orcid.org/0009-0008-5261-043X)
# MSc Artificial Intelligence student, Department of Computer Science, Brunel University London, UK; Independent researcher
# Conducted independently; not an official Brunel University London project. Licence: MIT
"""Per-learner randomisation: every learner gets its own seed, practice set and test items."""
from __future__ import annotations

import numpy as np
import pandas as pd

EXP_CODES = {"llm": 1, "nets": 2, "llm_blind": 3}


def learner_rng(master_seed: int, exp: str, cond_idx: int, learner: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([master_seed, EXP_CODES[exp], cond_idx, learner]))


def balanced(rng, pool_idx: np.ndarray, labels: np.ndarray, n: int, patients: np.ndarray,
             exclude_patients: set | None = None) -> np.ndarray:
    """n items, half younger and half older, at most one ECG per patient, none from
    `exclude_patients`. Raises ValueError if the pool is too small."""
    exclude_patients = exclude_patients or set()
    out: list[int] = []
    for lab, k in ((0, n // 2), (1, n - n // 2)):
        cand = pool_idx[labels[pool_idx] == lab]
        cand = cand[rng.permutation(len(cand))]
        seen: set = set()
        for i in cand:
            p = patients[i]
            if p in exclude_patients or p in seen:
                continue
            seen.add(p)
            out.append(int(i))
            if len(seen) == k:
                break
        if len(seen) < k:
            raise ValueError(f"pool too small: need {k} patients of class {lab}, have {len(seen)}")
    arr = np.array(out)
    rng.shuffle(arr)
    return arr


def draw_learner_sets(rng, meta: pd.DataFrame, n_practice: int, n_items: int, extra_practice: int = 0) -> dict:
    """Row positions (into meta) for one learner's practice and test sets.

    Every set is balanced 50/50 and has at most one ECG per patient. pre, post and
    delayed share no patient when the test pool allows it; otherwise each is drawn
    independently and `tests_disjoint` is set to False in the learner record."""
    role = meta["role"].values
    lab = meta["label"].values.astype(int)
    pat = meta["patient_id"].values
    train_pool = np.where(role == "teacher_train")[0]
    test_pool = np.where(role == "test")[0]
    tr_pool = np.where(role == "transfer")[0]
    sets = {"practice": balanced(rng, train_pool, lab, n_practice + extra_practice, pat)}
    used: set = set()
    disjoint = True
    for name in ("pre", "post", "delayed"):
        try:
            s = balanced(rng, test_pool, lab, n_items, pat, used)
        except ValueError:
            s = balanced(rng, test_pool, lab, n_items, pat)
            disjoint = False
        used |= set(pat[s].tolist())
        sets[name] = s
    if len(tr_pool):
        n_pat = min(len(set(pat[tr_pool[lab[tr_pool] == c]])) for c in (0, 1))
        sets["transfer"] = balanced(rng, tr_pool, lab, min(n_items, 2 * n_pat), pat)
    sets["tests_disjoint"] = disjoint
    return sets
