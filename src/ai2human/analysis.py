# AI-to-learner knowledge transfer study (ECG age)
# Author: KEYUR SHAILESHKUMAR PATEL (ORCID: https://orcid.org/0009-0008-5261-043X)
# MSc Artificial Intelligence student, Department of Computer Science, Brunel University London, UK; Independent researcher
# Conducted independently; not an official Brunel University London project. Licence: MIT
"""Preregistered analysis. Run separately for each experiment (llm, nets).

Primary (H1): post-test accuracy, cue_training > feedback_only, one-sided Welch t-test, alpha 0.05,
and the mean difference must reach the preregistered minimum meaningful difference.
Secondary (Holm-corrected within experiment):
  H2 post: cue_training > no_training
  H3 post: feedback_only > no_training
  H4 delayed: cue_training > feedback_only        (retention)
  H5 transfer: cue_training > feedback_only       (held-out device)
  H6 gain (post - pre): cue_training > feedback_only
  H7 post: cue_training > scrambled_cues          (does the AI's content matter?)
Language models (Experiment A): the accuracy in every hypothesis is the primary score named in
experiment_llm.primary_score (v0.4, decision D12: bias-corrected accuracy); the raw score and
AUROC are reported as robustness. Only models that pass the known-answer gate and are not
uninformative (flat rule; see exp_llm) count; all models are reported (sensitivity: all models).
The blinded-labels run is analysed the same way and reported as a secondary check.
Robustness (reported, not used for claims): ANCOVA post ~ condition + pre (HC3);
same tests on mean p(correct); per-model results; item-level regression with
two-way cluster-robust errors (learner and ECG).
Every null is shown with its minimum detectable effect (80% power).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats as sps

from . import stats as S

SECONDARY = {
    "H2": ("post_acc", "cue_training", "no_training"),
    "H3": ("post_acc", "feedback_only", "no_training"),
    "H4": ("delayed_acc", "cue_training", "feedback_only"),
    "H5": ("transfer_acc", "cue_training", "feedback_only"),
    "H6": ("gain_acc", "cue_training", "feedback_only"),
    "H7": ("post_acc", "cue_training", "scrambled_cues"),
}


def mde_diff(x, y, alpha=0.05, power=0.80) -> float:
    """Smallest true accuracy difference the test would detect 80% of the time (one-sided)."""
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float("nan")
    sp = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2))
    d = 0.01
    while S.power_two_sample(min(nx, ny), d, alpha) < power and d < 5:
        d += 0.005
    return float(d * sp)


def twoway_cluster_lpm(items: pd.DataFrame, a: str, b: str, test: str = "post", score: str = "raw") -> dict:
    """Item-level linear probability model: credit ~ 1 + [group a], with standard errors
    clustered two ways (learner and ECG; Cameron, Gelbach and Miller 2011).
    credit = 1/0 for right/wrong under the raw score, or the bias-corrected credit (1, 0.5, 0)."""
    d = items[(items.test == test) & items.condition.isin([a, b])].copy()
    if d.empty:
        return {"skipped": "no items"}
    if score == "bias_corrected":
        y = np.empty(len(d))
        for _, ix in pd.Series(np.arange(len(d))).groupby([d.condition.values, d.learner.values]):
            y[ix.values] = S.bias_corrected_credit(d.p_older.values[ix.values], d.label.values[ix.values])
    else:
        y = ((d.p_older > 0.5).astype(int) == d.label).astype(float).values
    X = np.column_stack([np.ones(len(d)), (d.condition == a).astype(float).values])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    e = y - X @ beta

    def meat(groups):
        M = np.zeros((2, 2))
        for _, ix in pd.Series(np.arange(len(d))).groupby(groups.values):
            s = X[ix.values].T @ e[ix.values]
            M += np.outer(s, s)
        return M

    lk = d["condition"].astype(str) + ":" + d["learner"].astype(str)
    ek = d["row"].astype(str)
    both = lk + "|" + ek
    V = XtX_inv @ (meat(lk) + meat(ek) - meat(both)) @ XtX_inv
    se = float(np.sqrt(max(V[1, 1], 1e-30)))
    z = beta[1] / se
    return {"coef": float(beta[1]), "se_twoway": se, "z": float(z), "p_one_sided": float(1 - sps.norm.cdf(z)),
            "n_items": int(len(d)), "n_learners": int(lk.nunique()), "n_ecgs": int(ek.nunique())}


def describe(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cond, g in df.groupby("condition"):
        for t in ["pre", "post", "delayed", "transfer", "gain"]:
            col = f"{t}_acc"
            if col not in g:
                continue
            v = g[col].dropna()
            if len(v) == 0:
                continue
            se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float("nan")
            rows.append({"condition": cond, "test": t, "n": len(v), "mean_acc": v.mean(), "sd": v.std(ddof=1),
                         "ci_low": v.mean() - 1.96 * se, "ci_high": v.mean() + 1.96 * se})
    return pd.DataFrame(rows)


def contrast(df, col, a, b, alpha, seed, n_boot):
    if col not in df:
        return {"skipped": f"{col} not measured"}
    x = df.loc[df.condition == a, col].dropna().values
    y = df.loc[df.condition == b, col].dropna().values
    if len(x) < 2 or len(y) < 2:
        return {"skipped": "not enough learners"}
    if np.var(x) == 0 and np.var(y) == 0:
        return {"skipped": "no variation in either group (test undefined)", "outcome": col, "group_a": a, "group_b": b}
    r = S.welch_test(x, y, "greater")
    r["ci95_diff"] = S.bootstrap_diff_ci(x, y, n_boot, 0.05, seed)
    r["mde_80"] = mde_diff(x, y, alpha)
    r.update({"outcome": col, "group_a": a, "group_b": b})
    return r


def analyse(df: pd.DataFrame, cfg: dict, teacher_metrics: dict, cue_report: dict, exp: str,
            items: pd.DataFrame | None = None, gates: dict | None = None) -> dict:
    ac = cfg["analysis"]
    seed = cfg["study"]["master_seed"]
    df_all = df.copy()
    df_all["gain_acc"] = df_all["post_acc"] - df_all["pre_acc"]
    out = {"experiment": exp, "n_learners_total": int(len(df_all))}
    score = cfg["experiment_llm"].get("primary_score", "raw") if gates is not None else "raw"
    out["primary_score"] = score
    out["role"] = cfg.get("experiment_nets", {}).get("role", "primary") if exp == "nets" else "primary"
    if gates is not None:  # language models: only informative models count (gate + collapse rule)
        out["models"] = gates
        keep = [m for m, g in gates.items() if g.get("informative")]
        out["models_counted"] = keep
        out["models_uninformative"] = [m for m in gates if m not in keep]
        df = df_all[df_all.model.isin(keep)].copy()
        if items is not None:
            items = items[items.model.isin(keep)]
        if df.empty:
            out["H1_primary"] = {"skipped": "no language model passed the known-answer gate and the flat/collapse rule"}
            out["secondary"] = {}
            out["descriptives"] = describe(df_all).to_dict("records")
            out["robustness"] = {}
            return out
    else:
        df = df_all
    out["n_learners"] = int(len(df))
    out["n_per_condition"] = df.condition.value_counts().to_dict()

    out["H1_primary"] = contrast(df, "post_acc", "cue_training", "feedback_only", ac["alpha"], seed, ac["bootstrap"])
    h1p = out["H1_primary"].get("p", float("nan"))
    sesoi = ac.get("min_meaningful_diff", 0.0)
    out["H1_primary"]["min_meaningful_diff"] = sesoi
    out["H1_primary"]["significant"] = bool(h1p < ac["alpha"])
    out["H1_primary"]["meaningful"] = bool(out["H1_primary"].get("diff", -1) >= sesoi)
    out["H1_primary"]["supported"] = out["H1_primary"]["significant"] and out["H1_primary"]["meaningful"]
    if "skipped" in out["H1_primary"]:
        out["H1_primary"]["supported"] = False
    elif not out["H1_primary"]["supported"]:
        a = df.loc[df.condition == "cue_training", "post_acc"].values
        b = df.loc[df.condition == "feedback_only", "post_acc"].values
        out["H1_equivalence_TOST"] = S.tost_equivalence(a, b, ac["equivalence_margin"])

    sec = {k: contrast(df, c, a, b, ac["alpha"], seed, ac["bootstrap"]) for k, (c, a, b) in SECONDARY.items()}
    praw = {k: v["p"] for k, v in sec.items() if "p" in v}
    padj = S.holm(praw)
    for k in sec:
        if k in padj:
            sec[k]["p_holm"] = padj[k]
            sec[k]["supported"] = bool(padj[k] < ac["alpha"])
    out["secondary"] = sec

    # Pre-test near chance? (precondition of the "skill they never had" claim)
    pre = df["pre_acc"].dropna().values
    t, p = sps.ttest_1samp(pre, 0.5)
    out["pretest_vs_chance"] = {"mean": float(pre.mean()), "t": float(t), "p_two_sided": float(p)}

    # Transfer fraction: share of the teacher's above-chance advantage the learners reach
    t_acc = teacher_metrics.get("test", {}).get("teacher_accuracy")
    if t_acc and t_acc > 0.5:
        out["transfer_fraction"] = {
            c: float((df.loc[df.condition == c, "post_acc"].mean() - 0.5) / (t_acc - 0.5))
            for c in cfg["conditions"]}
        out["teacher_test_accuracy"] = t_acc
    t_auc = teacher_metrics.get("test", {}).get("teacher_auroc")
    out["preconditions"] = {
        "teacher_test_auroc": t_auc,
        "teacher_is_expert (AUROC >= 0.80)": bool(t_auc is not None and t_auc >= 0.80),
        "pretest_mean": float(pre.mean()),
        "new_skill (pretest mean <= 0.65)": bool(pre.mean() <= 0.65),
    }
    out["rule_tally_accuracy_test"] = cue_report.get("test", {}).get("rule_tally_accuracy_true")

    # Robustness
    rob = {}
    d2 = df[df.condition.isin(["cue_training", "feedback_only"])].copy()
    d2["cue"] = (d2.condition == "cue_training").astype(int)
    try:
        f = "post_acc ~ cue + pre_acc" if d2["pre_acc"].std() > 0 else "post_acc ~ cue"
        m = smf.ols(f, data=d2).fit(cov_type="HC3")
        rob["ancova_cue_coef"] = {"coef": float(m.params["cue"]), "p_two_sided": float(m.pvalues["cue"]),
                                  "ci95": [float(x) for x in m.conf_int().loc["cue"]]}
    except Exception as e:
        rob["ancova_cue_coef"] = f"failed: {e}"
    if "post_p_correct" in df:
        rob["H1_on_p_correct"] = contrast(df, "post_p_correct", "cue_training", "feedback_only", ac["alpha"], seed, 2000)
    if "model" in df and df["model"].nunique() > 1:
        try:
            m = smf.ols("post_acc ~ cue + C(model)", data=d2).fit(cov_type="HC3")
            rob["cue_adjusted_for_model"] = {"coef": float(m.params["cue"]), "p_two_sided": float(m.pvalues["cue"])}
        except Exception as e:
            rob["cue_adjusted_for_model"] = f"failed: {e}"
        rob["per_model_H1"] = {mname: contrast(g, "post_acc", "cue_training", "feedback_only", ac["alpha"], seed, 2000)
                               for mname, g in df.groupby("model")}
    if "post_p_older_mean" in df:
        rob["response_bias_p_older_mean"] = df.groupby("condition")["post_p_older_mean"].mean().to_dict()
    if items is not None and ac.get("item_level_robustness", True):
        rob["item_level_twoway_cluster_H1"] = twoway_cluster_lpm(items, "cue_training", "feedback_only", score=score)
    for col, key in (("post_acc_raw", "H1_on_raw_accuracy"), ("post_acc_bc", "H1_on_bias_corrected_accuracy"),
                     ("post_auroc", "H1_on_auroc")):
        if col in df and not (col == "post_acc_raw" and score == "raw") and not (col == "post_acc_bc" and score == "bias_corrected"):
            rob[key] = contrast(df, col, "cue_training", "feedback_only", ac["alpha"], seed, 2000)
    for col, key in (("post_acc_raw", "raw_accuracy_post_by_group"), ("post_auroc", "auroc_post_by_group"),
                     ("post_tie_share", "median_tie_share_post_by_group")):
        if col in df:
            rob[key] = df.groupby("condition")[col].mean().to_dict()
    if gates is not None and out.get("models_uninformative"):
        rob["H1_all_models_including_uninformative"] = contrast(
            df_all, "post_acc", "cue_training", "feedback_only", ac["alpha"], seed, 2000)
    coll = {t: float(df_all[f"{t}_collapsed"].mean()) for t in ("pre", "post", "delayed", "transfer")
            if f"{t}_collapsed" in df_all}
    if not coll and "post_majority_share" in df_all:
        coll = {t: float((df_all[f"{t}_majority_share"] >= cfg["experiment_llm"]["collapse_share"]).mean())
                for t in ("pre", "post", "delayed", "transfer") if f"{t}_majority_share" in df_all}
    out["collapse_share_of_learners"] = coll
    out["flat_share_of_learners"] = {t: float(df_all[f"{t}_flat"].mean()) for t in ("pre", "post", "delayed", "transfer")
                                     if f"{t}_flat" in df_all}
    out["robustness"] = rob
    out["descriptives"] = describe(df).to_dict("records")
    return out


def save_analysis(res: dict, path):
    path.write_text(json.dumps(res, indent=2, default=float))
