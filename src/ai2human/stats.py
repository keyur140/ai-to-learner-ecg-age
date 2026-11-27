# AI-to-learner knowledge transfer study (ECG age)
# Author: KEYUR SHAILESHKUMAR PATEL (ORCID: https://orcid.org/0009-0008-5261-043X)
# MSc Artificial Intelligence student, Department of Computer Science, Brunel University London, UK; Independent researcher
# Conducted independently; not an official Brunel University London project. Licence: MIT
"""Preregistered statistics: effect sizes, tests, bootstrap CIs, multiplicity, power."""
from __future__ import annotations

import numpy as np
from scipy import stats


def bias_corrected_credit(p_older, labels) -> np.ndarray:
    """Per-item credit under the bias-corrected score (decision D12).

    One learner's items on one test are split at that learner's own median P(older): above the
    median counts as the answer 'older', below as 'younger', exactly at the median gets half credit.
    Tests are half older and half younger, so chance is 0.5 whatever letter or class a learner favours."""
    p, y = np.asarray(p_older, float), np.asarray(labels).astype(int)
    m = np.median(p)
    return np.where(p > m, y == 1, np.where(p < m, y == 0, 0.5)).astype(float)


def median_tie_share(p_older) -> float:
    """Share of a learner's items whose P(older) equals the learner's median (they cannot be ranked)."""
    p = np.asarray(p_older, float)
    return float(np.mean(p == np.median(p)))


def auroc(p_older, labels) -> float:
    """Area under the ROC curve from ranks (ties get mid-ranks). 0.5 when undefined."""
    p, y = np.asarray(p_older, float), np.asarray(labels).astype(int)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return 0.5
    r = stats.rankdata(p)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def cohens_d(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else float("nan")


def welch_test(a, b, alternative: str = "greater") -> dict:
    t, p = stats.ttest_ind(a, b, equal_var=False, alternative=alternative)
    return {"t": float(t), "p": float(p), "mean_a": float(np.mean(a)), "mean_b": float(np.mean(b)),
            "diff": float(np.mean(a) - np.mean(b)), "d": cohens_d(a, b), "n_a": len(a), "n_b": len(b)}


def bootstrap_diff_ci(a, b, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    diffs = rng.choice(a, (n_boot, len(a))).mean(1) - rng.choice(b, (n_boot, len(b))).mean(1)
    return float(np.quantile(diffs, alpha / 2)), float(np.quantile(diffs, 1 - alpha / 2))


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def power_two_sample(n_per_group: int, d: float, alpha: float = 0.05, one_sided: bool = True) -> float:
    """Power of a two-sample t-test (noncentral t)."""
    df = 2 * n_per_group - 2
    nc = d * np.sqrt(n_per_group / 2)
    crit = stats.t.ppf(1 - alpha if one_sided else 1 - alpha / 2, df)
    pw = 1 - stats.nct.cdf(crit, df, nc)
    if not one_sided:
        pw += stats.nct.cdf(-crit, df, nc)
    return float(pw)


def min_n_for_power(d: float, power: float = 0.9, alpha: float = 0.05, one_sided: bool = True) -> int:
    n = 2
    while power_two_sample(n, d, alpha, one_sided) < power:
        n += 1
    return n


def tost_equivalence(a, b, margin: float) -> dict:
    """Two one-sided tests: is |mean(a)-mean(b)| < margin? Used for 'no difference' claims."""
    p_low = stats.ttest_ind(np.asarray(a) + margin, b, equal_var=False, alternative="greater").pvalue
    p_high = stats.ttest_ind(np.asarray(a) - margin, b, equal_var=False, alternative="less").pvalue
    return {"p": float(max(p_low, p_high)), "margin": margin}
