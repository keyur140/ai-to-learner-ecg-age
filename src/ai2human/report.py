# AI-to-learner knowledge transfer study (ECG age)
# Author: KEYUR SHAILESHKUMAR PATEL (ORCID: https://orcid.org/0009-0008-5261-043X)
# MSc Artificial Intelligence student, Department of Computer Science, Brunel University London, UK; Independent researcher
# Conducted independently; not an official Brunel University London project. Licence: MIT
"""The run report: figures plus one list of content blocks, written out as
REPORT.md / REPORT.html (here) and REPORT.pdf (pdf_report.py) from the same blocks,
so the three versions always say the same thing.

Block types: ("h1", text) ("h2", text) ("p", text) ("bullets", [text]) ("box", title, text)
("table", header, rows) ("img", path, caption) ("code", text) ("pagebreak",).
Inline **bold** is allowed in text.
"""
from __future__ import annotations

import html as _html
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

COND = ["no_training", "feedback_only", "cue_training", "scrambled_cues"]
COLORS = {"no_training": "#9aa0a6", "feedback_only": "#4c78a8", "cue_training": "#e45756", "scrambled_cues": "#f2a541"}
NAMES = {"no_training": "No training", "feedback_only": "Feedback only", "cue_training": "AI-cue training",
         "scrambled_cues": "Scrambled cues (placebo)"}
TESTS = ["pre", "post", "delayed", "transfer"]
TEST_NAMES = {"pre": "Pre-test", "post": "Post-test (primary)", "delayed": "Delayed", "transfer": "New device", "gain": "Gain"}
HYP_TEXT = {
    "H1 (primary)": "AI-cue training beats feedback only, right after training",
    "H2": "AI-cue training beats no training, right after training",
    "H3": "Feedback only beats no training, right after training",
    "H4": "AI-cue training beats feedback only after the interference task (delayed)",
    "H5": "AI-cue training beats feedback only on the held-out device (transfer)",
    "H6": "AI-cue learners improve more from pre- to post-test than feedback-only learners",
    "H7": "The AI's real cues beat scrambled cues in the same format (placebo control)",
}
TITLE = "Can learners acquire a perceptual skill they never had from AI-derived cues? (ECG age)"
SHORT_TITLE = "ECG-age cue-transfer run report"
AI_USE = ("Use of AI tools. The research questions, hypotheses and design choices are the author's. The code and draft "
          "documents were written with help from an AI assistant (Claude, by Anthropic) under the author's direction and "
          "checked by the author. Separately, the study uses AI models as its objects of study: the AI teacher (a neural "
          "network trained here) and the learners (open language models and small neural networks).")
LICENCE = "Licence: text and figures CC BY 4.0; code MIT."


# ------------------------------------------------------------------ figures
def _clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig_accuracy(desc: pd.DataFrame, title: str, path: Path):
    conds = [c for c in COND if c in set(desc.condition)]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    w = 0.8 / max(len(conds), 1)
    for k, cond in enumerate(conds):
        d = desc[desc.condition == cond].set_index("test").reindex(TESTS)
        x = [i + (k - (len(conds) - 1) / 2) * w for i in range(len(TESTS))]
        ax.bar(x, d.mean_acc, w, color=COLORS[cond], label=NAMES[cond])
        ax.errorbar(x, d.mean_acc, yerr=[d.mean_acc - d.ci_low, d.ci_high - d.mean_acc], fmt="none", ecolor="black",
                    capsize=2, lw=0.8)
    ax.axhline(0.5, color="black", ls="--", lw=0.8)
    ax.text(-0.45, 0.505, "chance", fontsize=8, ha="left")
    ax.set_xticks(range(len(TESTS)), [TEST_NAMES[t] for t in TESTS])
    ax.set_ylabel("Accuracy (mean, 95% CI)")
    ax.set_ylim(0.3, 1.0)
    ax.set_title(title, fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    _clean(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_cues(cues: dict, path: Path):
    cs = cues["cues"]
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(cs) + 1))
    labels = [f"{c['label']} ({'higher' if c['older_is_higher'] else 'lower'} = older)" for c in cs][::-1]
    ax.barh(labels, [c["weight"] for c in cs][::-1], color=COLORS["cue_training"])
    ax.set_xlabel("Share of cue weight")
    ax.set_title("AI-derived cues", fontsize=10)
    _clean(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_teacher(tm: dict, path: Path):
    h = pd.DataFrame(tm["history"])
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(h.epoch, h.val_auroc, color=COLORS["feedback_only"], label="validation AUROC")
    ax.axvline(tm["best_epoch"], color="black", ls=":", lw=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUROC")
    ax.set_title("Teacher training (dotted line: epoch kept)", fontsize=10)
    _clean(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ helpers
def fp(p):
    if p is None or p != p:
        return "n/a"
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def pct(x, d=1):
    return "n/a" if x is None or x != x else f"{100 * x:.{d}f}%"


def pts(x):
    return "n/a" if x is None or x != x else f"{100 * x:+.1f} points"


def _label(ctx):
    lab = ctx.get("label", "")
    if ctx.get("synthetic"):
        return "SYNTHETIC DATA - NOT SCIENTIFIC RESULTS"
    return lab


EXP_TITLES = {
    "llm": "Experiment A: language-model learners, taught in words",
    "llm_blinded": "Experiment A, blinded-labels run (measurement names hidden)",
    "nets": "Experiment B: small neural-net learners, taught with data",
}


def short_answer(exp: str, res: dict) -> tuple[list[str], str]:
    """Bullets with numbers, and one plain-words sentence, for one experiment."""
    name = EXP_TITLES[exp].split(":")[0] if exp != "llm_blinded" else "Experiment A (blinded run)"
    h1 = res.get("H1_primary", {})
    if "skipped" in h1:
        return [f"{name}: H1 could not be tested ({h1['skipped']})."], f"{name} gives no answer, because {h1['skipped']}."
    b = [f"{name}: right after training, AI-cue learners got {pct(h1['mean_a'])} right; feedback-only learners got "
         f"{pct(h1['mean_b'])} (difference {pts(h1['diff'])}, 95% range {pts(h1['ci95_diff'][0])} to {pts(h1['ci95_diff'][1])}; "
         f"one-sided p = {fp(h1['p'])}). H1: **{'SUPPORTED' if h1.get('supported') else 'NOT SUPPORTED'}**."]
    h7 = res.get("secondary", {}).get("H7", {})
    if "p" in h7:
        b.append(f"{name}: real cues vs scrambled cues: {pct(h7['mean_a'])} vs {pct(h7['mean_b'])} "
                 f"(difference {pts(h7['diff'])}; Holm p = {fp(h7.get('p_holm'))}). H7: **{'supported' if h7.get('supported') else 'not supported'}**.")
    pre = res.get("preconditions", {})
    warn = [k.split(" (")[0] for k, v in pre.items() if isinstance(v, bool) and not v]
    if warn:
        b.append(f"{name}: preregistered precondition not met: {', '.join(warn)}. Read H1 with this in mind.")
    if h1.get("supported"):
        plain = (f"In {name}, learners taught the AI's cues did better than learners who only practised with feedback, "
                 f"by about {abs(100 * h1['diff']):.0f} in every 100 cases, and the difference is larger than the smallest one we said would matter.")
    elif h1.get("significant") and not h1.get("meaningful"):
        plain = (f"In {name}, the AI's cues helped a little, but by less than the smallest difference we said in advance would matter, "
                 f"so we count it as no meaningful help.")
    else:
        mde = h1.get("mde_80")
        plain = (f"In {name}, learners taught the AI's cues did not do clearly better than learners who only practised with feedback. "
                 f"The test could have found a true difference of about {abs(100 * mde):.1f} points or more most of the time."
                 if mde == mde and mde is not None else f"In {name}, the AI's cues did not clearly help.")
        tost = res.get("H1_equivalence_TOST")
        if tost and tost["p"] < 0.05:
            plain += f" The equivalence test also says any difference is smaller than {100 * tost['margin']:.0f} points."
    return b, plain


def _hyp_rows(res: dict):
    rows = []
    allh = {"H1 (primary)": res.get("H1_primary", {}), **res.get("secondary", {})}
    for k, r in allh.items():
        if "skipped" in r or "p" not in r:
            rows.append([k, HYP_TEXT.get(k, ""), "", "", "", "", "", "", f"skipped: {r.get('skipped', 'n/a')}"])
            continue
        ci = r.get("ci95_diff", (float("nan"),) * 2)
        rows.append([k, HYP_TEXT.get(k, ""), f"{100 * r['diff']:+.1f}", f"[{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}]",
                     f"{r['d']:.2f}", fp(r["p"]), fp(r.get("p_holm")) if k != "H1 (primary)" else "-",
                     f"{100 * r.get('mde_80', float('nan')):.1f}", "yes" if r.get("supported") else "no"])
    return rows


def _desc_rows(res: dict):
    d = pd.DataFrame(res.get("descriptives", []))
    rows = []
    if d.empty:
        return rows
    for c in COND:
        g = d[d.condition == c].set_index("test")
        if g.empty:
            continue
        row = [NAMES[c], str(int(g["n"].iloc[0]))]
        for t in TESTS:
            row.append(f"{pct(g.loc[t, 'mean_acc'])} [{pct(g.loc[t, 'ci_low'])}, {pct(g.loc[t, 'ci_high'])}]" if t in g.index else "-")
        rows.append(row)
    return rows


def _prior_knowledge(gates: dict, cues: dict):
    rows = []
    for m, g in gates.items():
        kn = g.get("probe_known_direction")
        if not kn:
            continue
        agree = sum(kn.get(c["id"], 0) == (1 if c["older_is_higher"] else -1) for c in cues["cues"])
        incons = sum(v == 0 for v in kn.values())
        rows.append([m, f"{agree} of {len(cues['cues'])}", f"{incons} of {len(kn)}"])
    return rows


def experiment_blocks(exp: str, res: dict, ctx: dict, run_dir: Path, fig_dir: Path, num: str):
    B = [("h1", f"{num}. {EXP_TITLES[exp]}")]
    desc = pd.DataFrame(res.get("descriptives", []))
    if not desc.empty:
        fig_accuracy(desc, EXP_TITLES[exp], fig_dir / f"accuracy_{exp}.png")
    gates = res.get("models")
    if gates:
        bc = res.get("primary_score") == "bias_corrected"
        coll_excl = any(g.get("collapse_excludes_model", not bc) for g in gates.values())
        rule = ("Before a model's learners count, it must follow one explicit rule on 40 cases (at least 85% right). "
                + ("It must also not be flat: fewer than half of its learners may give the same probability to more than half of the "
                   "post-test cases (such answers cannot be ranked). " if bc else "")
                + ("Fewer than half of its learners may give the same answer to 95% or more of the post-test cases. " if coll_excl else
                   "The share of learners that give one answer to 95% or more of the cases ('collapsed') is shown but does not remove a model, "
                   "because the primary score does not depend on which letter a learner favours. ")
                + "Models that fail are reported but do not count.")
        B += [("h2", "Which learner models count"),
              ("p", rule),
              ("table", ["Model", "Gate accuracy", "Gate", "Learners flat (post)", "Learners collapsed (post)", "Counts for H1"],
               [[m, pct(g["gate_accuracy"]), "pass" if g["gate_pass"] else "FAIL", pct(g.get("post_flat_fraction")),
                 pct(g.get("post_collapse_fraction")), "yes" if g.get("informative") else "no"] for m, g in gates.items()])]
        pk = _prior_knowledge(gates, ctx["cues"])
        if pk:
            B += [("p", "Prior-knowledge probe: before any lesson, each model was asked, for each measurement, whether it is usually higher "
                        "in older adults, twice, with the answer letters swapped. A direction counts as known only when both answers agree; a model "
                        "that always picks the same letter knows nothing by this rule. High numbers mean the lesson may teach little that is "
                        "new; the blinded run is the check."),
                  ("table", ["Model", "Cue directions already known", "Measurements answered inconsistently"], pk)]
    score_note = (" Score: bias-corrected accuracy. Each learner's cases on a test are split at that learner's own median probability of "
                  "'older' (above = older, below = younger, exactly at the median = half credit). Tests are half older and half younger, "
                  "so chance is 50% whichever letter a learner favours. The raw score (older when the probability is above 0.5) is "
                  "under Robustness." if res.get("primary_score") == "bias_corrected" else "")
    B += [("h2", "Results by group"), ("p", f"Learners counted: {res.get('n_learners', 0)} "
                                          f"({', '.join(f'{NAMES.get(k, k)} {v}' for k, v in res.get('n_per_condition', {}).items())})."
                                          + score_note),
          ("table", ["Group", "n", "Pre", "Post", "Delayed", "New device"], _desc_rows(res))]
    if (fig_dir / f"accuracy_{exp}.png").exists():
        B.append(("img", fig_dir / f"accuracy_{exp}.png", "Mean accuracy per group and test, with 95% confidence intervals."))
    if res.get("role") == "secondary":
        B += [("p", "This experiment is secondary (decision D13). It is run and analysed exactly as designed, but its H1 is not a primary "
                    "claim of the study: the preregistration states in advance that no difference between AI cues and feedback only is "
                    "expected here.")]
    B += [("h2", "Hypotheses"),
          ("p", "Differences are in percentage points of accuracy. MDE = the smallest true difference this test would find 80% of the time. "
                "H1 needs p < 0.05 AND a difference of at least "
                f"{100 * res.get('H1_primary', {}).get('min_meaningful_diff', 0.03):.0f} points. H2-H7 are Holm-corrected."),
          ("table", ["", "Claim", "Diff", "95% CI", "d", "p", "p (Holm)", "MDE", "Supported"], _hyp_rows(res))]
    bul, plain = short_answer(exp, res)
    B += [("box", "What this means in plain words", plain)]
    pre = res.get("preconditions", {})
    if pre:
        B += [("h2", "Preregistered preconditions"),
              ("table", ["Check", "Value"], [[k, (f"{v:.3f}" if isinstance(v, float) else str(v))] for k, v in pre.items()])]
    rob = res.get("robustness", {})
    rb = []
    a = rob.get("ancova_cue_coef")
    if isinstance(a, dict):
        rb.append(f"Adjusting for each learner's pre-test score (ANCOVA): cue effect {pts(a['coef'])}, two-sided p = {fp(a['p_two_sided'])}.")
    a = rob.get("H1_on_p_correct")
    if isinstance(a, dict) and "p" in a:
        rb.append(f"Same test on mean probability of the right answer: difference {pts(a['diff'])}, p = {fp(a['p'])}.")
    a = rob.get("item_level_twoway_cluster_H1")
    if isinstance(a, dict) and "coef" in a:
        rb.append(f"Item-level model with errors clustered by learner and by ECG (allows for learners sharing test ECGs): "
                  f"{pts(a['coef'])}, SE {100 * a['se_twoway']:.2f} points, one-sided p = {fp(a['p_one_sided'])} "
                  f"({a['n_items']} answers, {a['n_ecgs']} distinct ECGs).")
    for key, name in (("H1_on_raw_accuracy", "raw accuracy (older when the probability is above 0.5)"),
                      ("H1_on_bias_corrected_accuracy", "bias-corrected accuracy")):
        a = rob.get(key)
        if isinstance(a, dict) and "p" in a:
            rb.append(f"Same test on {name}: difference {pts(a['diff'])}, p = {fp(a['p'])}.")
    a = rob.get("H1_on_auroc")
    if isinstance(a, dict) and "p" in a:
        rb.append(f"Same test on each learner's AUROC (how well its probabilities rank the cases; 0.50 = no information): "
                  f"difference {a['diff']:+.3f}, p = {fp(a['p'])}.")
    for key, name, fmt in (("raw_accuracy_post_by_group", "Raw accuracy at post-test", pct),
                           ("auroc_post_by_group", "AUROC at post-test", lambda v: f"{v:.3f}"),
                           ("median_tie_share_post_by_group", "Share of post-test cases tied at the learner's median (cannot be ranked)", pct)):
        if rob.get(key):
            rb.append(f"{name}: " + ", ".join(f"{NAMES.get(k, k)} {fmt(v)}" for k, v in rob[key].items()) + ".")
    a = rob.get("H1_all_models_including_uninformative")
    if isinstance(a, dict) and "p" in a:
        rb.append(f"H1 counting every model, including those that did not count: {pts(a['diff'])}, p = {fp(a['p'])}.")
    for m, r in (rob.get("per_model_H1") or {}).items():
        if "p" in r:
            rb.append(f"Model {m} alone: {pts(r['diff'])}, p = {fp(r['p'])}.")
    if rob.get("response_bias_p_older_mean"):
        rb.append("Average probability given to 'older' (0.5 = no bias): " +
                  ", ".join(f"{NAMES.get(k, k)} {v:.2f}" for k, v in rob["response_bias_p_older_mean"].items()) + ".")
    if res.get("collapse_share_of_learners"):
        rb.append("Share of learners whose answers collapsed to one class: " +
                  ", ".join(f"{TEST_NAMES[k]} {pct(v)}" for k, v in res["collapse_share_of_learners"].items()) + ".")
    if "transfer_fraction" in res:
        rb.append("Transfer fraction (share of the AI teacher's advantage over chance reached at post-test): " +
                  ", ".join(f"{NAMES[c]} {v:.2f}" for c, v in res["transfer_fraction"].items()) + ".")
    if rb:
        B += [("h2", "Robustness checks (reported, not used for the verdict)"), ("bullets", rb)]
    return B


def build_blocks(run_dir: Path, ctx: dict) -> list:
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    au = ctx.get("author") or {}
    exps = [e for e in ("llm", "llm_blinded", "nets") if ctx.get(f"analysis_{e}")]
    checks = ctx.get("checks", [])
    n_fail = sum(c["result"] == "FAIL" for c in checks)
    B = [("cover", {"title": TITLE, "subtitle": f"Run report - {_label(ctx)}", "author": au, "lines": [
        f"Run: {ctx['run_id']}  |  Software version {ctx.get('version', '?')}",
        f"Preregistration: {ctx['prereg_lock_status']}",
        AI_USE, LICENCE]})]

    # 1. short answer
    bullets, plains = [], []
    for e in exps:
        b, p = short_answer(e, ctx[f"analysis_{e}"])
        bullets += b
        plains.append(p)
    if ctx.get("synthetic"):
        bullets.insert(0, "**These are fake data and fake learners.** The numbers show only that the software works and what it reports.")
    bullets.append(f"Self-checks: {sum(c['result'] == 'pass' for c in checks)} passed, {n_fail} failed, "
                   f"{sum(c['result'] == 'n/a' for c in checks)} not applicable (Section 3).")
    B += [("h1", "1. The short answer"), ("bullets", bullets), ("box", "What this means in plain words", " ".join(plains))]

    # 2. what this is
    what = [("h1", "2. What this report is, and what it is not")]
    if ctx.get("synthetic"):
        what.append(("p", "This run used made-up ECG-like data and mock learners that follow simple scripted rules. It is a test of the "
                          "machinery, not a finding about real learners. Nothing here may be quoted as a result about ECGs, people or AI models."))
    elif ctx.get("label") == "EXPLORATORY":
        what.append(("p", "This run used the real PTB-XL data but was NOT made under a locked preregistration. It is exploratory: useful for "
                          "ideas, not for testing the hypotheses."))
    else:
        what.append(("p", "This run tests the preregistered hypotheses on PTB-XL. The plan, code and settings were frozen before it ran; "
                          "their fingerprints are in Section 3 and must match the ones on OSF."))
    what.append(("p", "It is a fully computational study: the 'learners' are machine learners (language models and small neural networks), "
                      "not people. It asks whether a skill an AI found can be handed on in a teachable form, which is a first step before any "
                      "study with people. It does not show that people would learn the same way."))
    B += what

    # 3. trust
    B += [("h1", "3. Can we trust the run? (checks)"),
          ("p", "Each check recomputes something a second, independent way. A FAIL is shown here and must be explained in ISSUES.md before "
                "any number from this run is quoted."),
          ("table", ["#", "Check", "Result", "Detail"], [[c["id"], c["check"], c["result"], c["detail"]] for c in checks]),
          ("h2", "Fingerprints (SHA-256)"),
          ("table", ["Item", "SHA-256"], [["Preregistration (prereg/*.md)", ctx["prereg_digest"]], ["Config (configs/study.yaml)", ctx["config_sha"]],
                                         ["Code (src/*.py)", ctx["code_digest"]]])]
    if ctx.get("deviations"):
        B += [("h2", "Deviations from the preregistration"), ("bullets", ctx["deviations"])]

    # 4. data
    cs = ctx["cohort"]
    integ = cs.get("source", {}).get("integrity")
    B += [("h1", "4. Data and splits"),
          ("p", f"Source: {cs['source']['dataset']}. {cs['n_total_records']} ECGs, {cs['n_in_age_groups']} in the two age groups "
                f"(under 40 or 65 and over), {cs['n_cohort']} with the measurements, from {cs['n_patients']} patients. Roles are given by "
                f"patient, so no patient is in two roles; {cs['ecgs_set_aside_for_patient_overlap']} ECGs were set aside for this. "
                f"Held-out device: {cs['transfer_device']}."),
          ("table", ["Role", "ECGs", "Older (65+)", "Used for"],
           [[r, str(v["n"]), str(v["n_older"]), {"teacher_train": "teaching the AI; learners' practice cases", "teacher_val": "stopping the AI's training",
                                                  "cue": "finding the cues", "test": "learners' pre, post and delayed tests",
                                                  "transfer": "learners' new-device test", "unused": "set aside (patient overlap)"}[r]]
            for r, v in cs["roles"].items() if v["n"]])]
    if integ:
        B.append(("p", f"Data integrity: PhysioNet checksum file present: {integ.get('sha256sums_present')}; status "
                       f"{integ.get('status')}; {integ.get('files_checked', 0)} files checked. Records {integ.get('n_records')} and patients "
                       f"{integ.get('n_patients')} match the published counts: {integ.get('counts_match_published')}."))

    # 5. teacher
    tm = ctx["teacher"]
    fig_teacher(tm, fig_dir / "teacher_training.png")
    rows = [[r, f"{tm[r]['teacher_auroc']:.3f}", pct(tm[r]["teacher_accuracy"]), f"{tm[r]['features_baseline_auroc']:.3f}",
             pct(tm[r]["features_baseline_accuracy"])] for r in ("teacher_val", "cue", "test", "transfer") if r in tm]
    B += [("h1", "5. The AI teacher"),
          ("p", "A 1D ResNet read the raw 12-lead ECG and learned to tell younger from older adults. 'Measurements only' is a simple "
                "model that sees only the 18 measurements the learners see: it shows how much is possible from those numbers alone."),
          ("table", ["Split", "AI AUROC", "AI accuracy", "Measurements-only AUROC", "Measurements-only accuracy"], rows),
          ("img", fig_dir / "teacher_training.png", "Validation AUROC by epoch.")]

    # 6. cues
    cues = ctx["cues"]
    fig_cues(cues, fig_dir / "cues.png")
    rep = cues["report"]
    B += [("h1", "6. The cues taken from the AI"),
          ("p", "On the cue fold only, a sparse model was fitted to the AI's own guesses (not the true ages); the measurements it kept "
                "became plain rules. This is the lesson the AI-cue group reads:"),
          ("code", (run_dir / "cue_lesson.txt").read_text()),
          ("img", fig_dir / "cues.png", "Weight of each cue in the lesson."),
          ("p", "The placebo group reads this lesson instead (same format, measurements the AI did not pick, random directions):"),
          ("code", (run_dir / "scrambled_lesson.txt").read_text()),
          ("table", ["Split", "Surrogate agrees with AI", "Rules agree with AI", "Rules right (true age)", "Scrambled rules right"],
           [[r, pct(rep[r]["surrogate_fidelity_to_teacher"]), pct(rep[r]["rule_tally_fidelity_to_teacher"]),
             pct(rep[r]["rule_tally_accuracy_true"]), pct(rep[r].get("scrambled_tally_accuracy_true"))]
            for r in ("cue", "test", "transfer") if r in rep])]
    s = ctx.get("saliency")
    if s:
        B += [("h2", "Saliency check (does the AI look where the cues say?)"),
              ("table", ["ECG segment", "Share of AI attention", "Share of time", "Density ratio"],
               [[k, pct(v["saliency_share"]), pct(v["time_share"]), f"{v['density_ratio']:.2f}"] for k, v in s["segments"].items()]),
              ("p", f"Densest attention: {s['densest_saliency_segment']}. Segment holding most cue weight: {s['segment_with_most_cue_weight']}. "
                    f"Agree: {s['agree']}.")]

    num = 7
    for e in exps:
        B += experiment_blocks(e, ctx[f"analysis_{e}"], ctx, run_dir, fig_dir, str(num))
        num += 1

    # problems, can/cannot, glossary, how to check, FAQ
    fails = [f"{c['id']} {c['check']}: {c['detail']}" for c in checks if c["result"] == "FAIL"]
    B += [("h1", f"{num}. Problems found"),
          ("bullets", fails or ["No self-check failed in this run. Known design limits are listed in ISSUES.md and in Section "
                                f"{num + 1}."])]
    num += 1
    can = ["Whether, in this setting, machine learners taught the AI's cues beat machine learners given only feedback (H1), and whether the "
           "AI's real cues beat a placebo lesson in the same format (H7)."]
    cannot = ["That people would learn the same way. All learners here are machines.",
              "That the cues are how the AI 'really' works. They are a simple summary that agrees with the AI most of the time (Section 6).",
              "Anything clinical. Telling younger from older ECGs is a test task, not a diagnosis."]
    if ctx.get("synthetic"):
        cannot.insert(0, "Anything at all about real ECGs or real models: this run used fake data and scripted learners.")
    B += [("h1", f"{num}. What we can and cannot say"), ("h2", "We can say"), ("bullets", can), ("h2", "We cannot say"), ("bullets", cannot)]
    num += 1
    B += [("h1", f"{num}. Words used"), ("table", ["Word", "Meaning"], GLOSSARY)]
    num += 1
    B += [("h1", f"{num}. How to check this report"),
          ("bullets", [f"Master seed: {ctx.get('seed', '20261026')}. Every learner's cases and letters come from its own seed, so any learner can be re-made.",
                       "Re-hash every output: python run_study.py verify runs/" + ctx["run_id"],
                       "Compare the three fingerprints in Section 3 with prereg/LOCK.json and with the OSF registration.",
                       "All learner-level results are in learners_*.csv; every single answer is in items_*.csv; checks.json holds the self-checks."])]
    num += 1
    B += [("h1", f"{num}. Common questions"), ("table", ["Question", "Answer"], FAQ)]
    return B


GLOSSARY = [
    ["AI teacher", "The neural network trained here to tell younger from older adults from the raw ECG."],
    ["Cue", "A plain rule taken from the AI, e.g. 'heart rate tends to be lower in older adults'."],
    ["Learner", "One machine 'participant': one language-model session or one small neural network, with its own cases."],
    ["Feedback only", "Learners practise on cases and are told the right answer, but get no rules."],
    ["Scrambled cues (placebo)", "A lesson that looks the same but uses measurements the AI did not pick, with random directions."],
    ["Known-answer gate", "A test that a model can follow one simple written rule. Models that cannot are not counted."],
    ["Collapse", "When a learner gives the same answer to nearly every case. Its accuracy then says nothing."],
    ["AUROC", "How well a model ranks older above younger; 0.5 is guessing, 1.0 is perfect."],
    ["p-value", "How surprising a result would be if there were no real difference. Small = unlikely to be luck."],
    ["Holm correction", "A rule that makes p-values stricter when several hypotheses are tested together."],
    ["MDE", "Minimum detectable effect: the smallest true difference the test would find 80% of the time."],
    ["Equivalence test (TOST)", "A test of whether a difference is smaller than a set margin, so 'no difference' can be claimed."],
    ["Preregistration", "Publishing the plan before running the test, so it cannot be changed after."],
    ["SHA-256", "A file fingerprint. Change one letter and the fingerprint changes."],
]
FAQ = [
    ["Did an AI do the research?", "No. The idea, hypotheses and every design choice are the author's. An AI assistant helped write code and "
                                   "drafts under the author's direction. Separately, AI models are what the study tests."],
    ["Why machines and not people?", "It is cheaper, fully repeatable and can use 1,002 learners per group. If cues do not transfer even to "
                                     "machines, a study with people is unlikely to work; if they do, it is the next step."],
    ["Why a placebo lesson?", "A lesson might help just because it is a lesson. The placebo has the same format and length, so only the AI's "
                              "content differs."],
    ["Why hide the measurement names in one run?", "Language models may already know how ECGs change with age. With names hidden, they can "
                                                   "only use what the lesson tells them."],
    ["What if no model passes the gate?", "Then Experiment A gives no answer for H1, and the report says so. The rule was fixed in advance."],
]


# ------------------------------------------------------------------ markdown / html
def _md_inline(t: str) -> str:
    return t


def blocks_to_md(B: list) -> str:
    out = []
    for b in B:
        k = b[0]
        if k == "cover":
            c = b[1]
            out += [f"# {c['title']}", "", f"**{c['subtitle']}**", ""]
            au = c.get("author") or {}
            if au:
                out += [f"**{au['name']}** · ORCID {au['orcid']}  ", "; ".join(au["affiliations"]) + ".  ", f"*{au['note']}*", ""]
            out += [f"{x}  " for x in c["lines"]] + [""]
        elif k == "h1":
            out += ["", f"## {b[1]}", ""]
        elif k == "h2":
            out += ["", f"### {b[1]}", ""]
        elif k == "p":
            out += [b[1], ""]
        elif k == "bullets":
            out += [f"- {x}" for x in b[1]] + [""]
        elif k == "box":
            out += [f"> **{b[1]}.** {b[2]}", ""]
        elif k == "table":
            out += ["| " + " | ".join(b[1]) + " |", "|" + " --- |" * len(b[1])]
            out += ["| " + " | ".join(str(x).replace("|", "/") for x in r) + " |" for r in b[2]] + [""]
        elif k == "img":
            p = Path(b[1])
            out += [f"![{b[2]}](figures/{p.name})" if p.parent.name == "figures" else f"![{b[2]}]({p.name})", f"*{b[2]}*", ""]
        elif k == "code":
            out += ["```", b[1].rstrip(), "```", ""]
    return "\n".join(out)


def md_to_html(text: str, title: str) -> str:
    try:
        import markdown

        body = markdown.markdown(text, extensions=["tables", "fenced_code"])
    except ImportError:
        body = f"<pre>{_html.escape(text)}</pre>"
    return ("<!doctype html><html><head><meta charset='utf-8'><title>" + _html.escape(title) + "</title><style>"
            "body{font-family:system-ui,sans-serif;max-width:980px;margin:2rem auto;padding:0 1rem;line-height:1.5;color:#1b1b1b}"
            "table{border-collapse:collapse;margin:0.5rem 0}td,th{border:1px solid #ccc;padding:4px 8px;vertical-align:top}"
            "img{max-width:100%}pre{background:#f5f5f5;padding:1rem;overflow-x:auto;white-space:pre-wrap}"
            "blockquote{border-left:4px solid #e45756;margin:0.8rem 0;padding:0.4rem 1rem;background:#fdf3f2}"
            "</style></head><body>" + body + "</body></html>")


def write_report(run_dir: Path, ctx: dict) -> Path:
    B = build_blocks(run_dir, ctx)
    ctx["_blocks"] = B
    text = blocks_to_md(B)
    (run_dir / "REPORT.md").write_text(text)
    (run_dir / "REPORT.html").write_text(md_to_html(text, f"Run report {ctx['run_id']}"))
    return run_dir / "REPORT.md"


INFO_KEYS = ("author", "run_id", "version", "mode", "label", "smoke", "synthetic", "prereg_digest", "prereg_lock_status",
             "config_sha", "code_digest", "deviations", "seed")


def load_ctx(run_dir: Path) -> dict:
    """Rebuild a report context from a finished run's files (python run_study.py report runs/<id>)."""
    run_dir = Path(run_dir)
    rd = lambda n: json.loads((run_dir / n).read_text()) if (run_dir / n).exists() else None  # noqa: E731
    ctx = dict(rd("run_info.json") or {})
    ctx.update(cohort=rd("cohort_summary.json"), teacher=rd("teacher_metrics.json"), cues=rd("cues.json"),
               saliency=rd("saliency.json"), checks=rd("checks.json") or [])
    for e in ("llm", "llm_blinded", "nets"):
        a = rd(f"analysis_{e}.json")
        if a:
            ctx[f"analysis_{e}"] = a
    return ctx
