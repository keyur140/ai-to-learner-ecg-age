# AI-to-learner knowledge transfer study (ECG age)
# Author: KEYUR SHAILESHKUMAR PATEL (ORCID: https://orcid.org/0009-0008-5261-043X)
# MSc Artificial Intelligence student, Department of Computer Science, Brunel University London, UK; Independent researcher
# Conducted independently; not an official Brunel University London project. Licence: MIT
"""The synthetic validation report (Markdown, HTML and PDF from the same blocks)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from . import __version__
from .report import AI_USE, COND, GLOSSARY, LICENCE, NAMES, TITLE, _desc_rows, _hyp_rows, blocks_to_md, md_to_html, pct

VTITLE = "ECG-age cue-transfer synthetic validation"


def _fig_power(sim: dict, path: Path, sesoi: float):
    c = pd.DataFrame(sim["power_curve"])
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.plot(100 * c.true_diff, c.p_below_alpha, "o-", color="#4c78a8", label="p < 0.05 only")
    ax.plot(100 * c.true_diff, c.H1_rule, "o-", color="#e45756", label="H1 rule (also needs diff >= 3)")
    ax.axvline(100 * sesoi, color="black", ls=":", lw=0.8)
    ax.axhline(0.8, color="grey", ls="--", lw=0.6)
    ax.axhline(0.05, color="grey", ls="--", lw=0.6)
    ax.set_xlabel("True difference, AI-cue minus feedback-only (accuracy points)")
    ax.set_ylabel("Chance of a 'yes'")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fig_worlds(summary: dict, path: Path):
    rows = []
    for wid, w in summary["worlds"].items():
        a = w["analyses"].get("llm")
        if not a:
            continue
        d = pd.DataFrame(a["descriptives"])
        for c in COND:
            r = d[(d.condition == c) & (d.test == "post")]
            if len(r):
                rows.append((wid, c, float(r.mean_acc.iloc[0]), float(r.ci_low.iloc[0]), float(r.ci_high.iloc[0])))
    if not rows:
        return False
    df = pd.DataFrame(rows, columns=["w", "c", "m", "lo", "hi"])
    ws = list(dict.fromkeys(df.w))
    from .report import COLORS

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    wdt = 0.2
    for k, c in enumerate(COND):
        d = df[df.c == c].set_index("w").reindex(ws)
        x = [i + (k - 1.5) * wdt for i in range(len(ws))]
        ax.bar(x, d.m, wdt, color=COLORS[c], label=NAMES[c])
        ax.errorbar(x, d.m, yerr=[d.m - d.lo, d.hi - d.m], fmt="none", ecolor="black", capsize=2, lw=0.7)
    ax.axhline(0.5, color="black", ls="--", lw=0.8)
    ax.set_xticks(range(len(ws)), [f"{w}\n{summary['worlds'][w]['name']}" for w in ws], fontsize=8)
    ax.set_ylabel("Post-test accuracy (all learners)")
    ax.set_ylim(0.3, 1.0)
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def build_blocks(out: Path, summary: dict, cfg: dict) -> list:
    fig = out / "figures"
    fig.mkdir(exist_ok=True)
    au = cfg.get("author", {})
    W = summary["worlds"]
    sim = summary.get("simulation", {})
    rep = summary.get("reproducibility", {})
    n = cfg["experiment_llm"]["n_per_condition"]
    sesoi = cfg["analysis"]["min_meaningful_diff"]
    n_exp = sum(len(w["expectations"]) for w in W.values())
    n_met = sum(e["met"] for w in W.values() for e in w["expectations"])
    curve = {round(r["true_diff"], 3): r for r in sim.get("power_curve", [])}

    def pw(d):
        return pct(curve.get(d, {}).get("H1_rule"), 0)

    B = [("cover", {"title": TITLE, "subtitle": "Synthetic validation report - SYNTHETIC DATA - NOT SCIENTIFIC RESULTS",
                    "author": au, "lines": [
                        f"Validation run: {out.name}  |  Software version {__version__}  |  {'QUICK (reduced sizes)' if summary['quick'] else 'full'}",
                        "Status: exploratory validation made BEFORE preregistration. No real ECGs were used.",
                        AI_USE, LICENCE]})]
    short = [f"The pipeline behaved as planned in **{n_met} of {n_exp}** pre-stated checks across {len(W)} fake worlds where the right answer was known in advance."]
    for wid, w in W.items():
        miss = [e["what"] for e in w["expectations"] if not e["met"]]
        short.append(f"{wid} {w['name']}: {len(w['expectations']) - len(miss)} of {len(w['expectations'])} as expected"
                     + (f"; NOT as expected: {'; '.join(miss)}." if miss else "."))
    h7s = [(wid, w["analyses"]["llm"]["secondary"]["H7"]) for wid, w in W.items()
           if w["signal"] and any("student" in m or "strong" in m for m in w["models"]) and w["analyses"].get("llm") and "p" in w["analyses"]["llm"].get("secondary", {}).get("H7", {})]
    if h7s:
        short.append("Large differences are found: real cues beat scrambled cues by "
                     + ", ".join(f"{100 * h['diff']:+.0f} points ({wid}, {'supported' if h.get('supported') else 'not supported'})" for wid, h in h7s)
                     + " in the worlds where the learners read lessons.")
    if sim:
        short.append(f"At the planned size ({n:,} learners per group), the H1 rule gave a false 'yes' in {pct(sim['false_positive_rate_H1'], 2)} "
                     f"of simulated no-effect studies (limit 5%). It found a true {100 * 0.05:.0f}-point effect {pw(0.05)} of the time, "
                     f"a 4-point effect {pw(0.04)} and a 3-point effect only {pw(0.03)}.")
        short.append(f"Holm correction kept the chance of any false 'yes' among H2-H7 at {pct(sim['holm_familywise_error'], 1)} (limit 5%).")
    if rep:
        short.append(f"Re-running the same settings gave byte-identical results ({rep['repeat_files_compared']} files): "
                     f"{'yes' if rep['repeat_identical'] else 'NO'}. Changing one byte of a results file was caught: "
                     f"{'yes' if not rep['tamper']['after_one_byte_change'] else 'NO'}.")
    below = [f"{wid}" for wid, w in W.items() for e in w["expectations"] if not e["met"] and "below the 3-point bar" in e["observed"]]
    plain = (f"We built {len(W)} pretend worlds where we knew in advance what should happen, and ran the whole study in each. The software "
             "said 'the cues help' when they clearly did, 'no' when they did not, caught learners that already knew the answer, threw out "
             "broken learners, and noticed when there was nothing to learn. These are machine tests only: they say nothing yet about "
             "real ECGs or real AI models.")
    if below:
        plain += (f" In {', '.join(sorted(set(below)))} the help we planted turned out smaller than the 3-point bar, so the study said "
                  "'real but too small to count', which is what the rule is for; our advance guess of the size was too high.")
    if curve.get(0.03, {}).get("H1_rule", 1) < 0.7:
        plain += (f" One design fact to know: because H1 also needs the observed difference to reach {100 * sesoi:.0f} points, a true "
                  f"effect of exactly {100 * sesoi:.0f} points is found only about half the time. Effects of 4 points or more are found "
                  f"almost always.")
    B += [("h1", "1. The short answer"), ("bullets", short), ("box", "What this means in plain words", plain)]

    B += [("h1", "2. What this report is, and what it is not"),
          ("p", "This is the exploratory validation run made before the study is preregistered. It uses made-up ECG-like data "
                "and scripted mock learners. Its only job is to show that the software, the checks and the statistics do what the "
                "plan says, so that the preregistration can point to it. It may be cited in the preregistration as 'synthetic "
                "validation'. It is not a pilot on real data: no PTB-XL record was opened (decision D3)."),
          ("p", "Each fake world states in advance what the pipeline must report. A world 'passes' only if every one of its "
                "statements comes true. The worlds use the preregistered code unchanged; only data size, training length and "
                "the learner type are set for the world. Language-model worlds use the full planned number of learners "
                f"({n:,} per group, main and blinded runs). Neural-net worlds use 30 learners per group to save time; their power "
                "at full size is covered by the simulations in Section 5.")]

    B += [("h1", "3. How the fake worlds are built"),
          ("bullets", ["Fake ECGs: 12-lead, 10-second signals with patients, folds and recording devices like PTB-XL. In worlds with "
                       "signal, age shifts some of the 18 measurements and the waveform; in W5 it shifts nothing.",
                       "Mock learners read the same prompts real models would read and answer with a probability for A or B. A "
                       "'student' applies any numbered rules it finds and learns a little from practice cases; a 'null' learner "
                       "ignores lessons; a 'memoriser' already knows the AI's cues when real names are shown; 'collapse' always says "
                       "'older' outside the gate; 'random' guesses.",
                       "Neural-net learners are real small networks trained on the fake signals, exactly as in the study."]),
          ("table", ["World", "Story", "Learner models", "Age signal"],
           [[wid + (" (added)" if w.get("added") else ""), w["story"], ", ".join(w["models"]), "yes" if w["signal"] else "none"]
            for wid, w in W.items()])]

    B += [("h1", "4. Results in each world")]
    if _fig_worlds(summary, fig / "worlds_post.png"):
        B.append(("img", fig / "worlds_post.png", "Post-test accuracy by group in each world (language-model learners, main run, all models)."))
    for wid, w in W.items():
        B += [("h2", f"{wid}. {w['name']}"), ("p", w["story"]),
              ("table", ["Must happen", "Happened", "What we saw"],
               [[e["what"], "yes" if e["met"] else "NO", e["observed"]] for e in w["expectations"]])]
        a = w["analyses"].get("llm")
        if a and a.get("descriptives"):
            B += [("p", f"Language-model learners (main run). Teacher AUROC {w['teacher_test_auroc']:.3f}; cues: "
                        f"{', '.join(w['cues'])}; the cue rules alone get {pct(w['rule_tally_accuracy_test'])} right on the test fold."),
                  ("table", ["Group", "n", "Pre", "Post", "Delayed", "New device"], _desc_rows(a))]

    if sim:
        _fig_power(sim, fig / "power_curve.png", sesoi)
        inp = sim["inputs"]
        cc = sim["cluster_calibration"]
        B += [("h1", f"5. How the statistics behave at the planned size ({n:,} per group)"),
              ("p", f"Simulated learners were given the spread seen in W1 (feedback-only mean {pct(inp['feedback_only_mean'])}, SD "
                    f"{100 * inp['learner_sd']:.1f} points, {inp['items_per_test']} test items each). Each figure below is from "
                    f"{inp['replications']:,} simulated studies."),
              ("img", fig / "power_curve.png", "Chance that the study says 'yes' for H1, by the true size of the effect. Dotted line: the "
                                               "preregistered minimum meaningful difference."),
              ("table", ["True difference (points)", "p < 0.05", "H1 rule (p < 0.05 and >= 3 points)"],
               [[f"{100 * r['true_diff']:.1f}", pct(r["p_below_alpha"], 1), pct(r["H1_rule"], 1)] for r in sim["power_curve"]]),
              ("bullets", [f"False 'yes' for H1 when there is no effect: {pct(sim['false_positive_rate_H1'], 2)} (Welch test alone: "
                           f"{pct(sim['false_positive_rate_welch_only'], 2)}).",
                           f"Chance of at least one false 'yes' among H2-H7 after Holm correction, when nothing works: "
                           f"{pct(sim['holm_familywise_error'], 1)}.",
                           f"Learners sharing test ECGs: with {cc['learners_per_group']} learners per group drawing from {cc['ecg_pool']} "
                           f"ECGs of mixed difficulty, the learner-level Welch test said 'yes' falsely {pct(cc['welch_false_positive'], 1)} "
                           f"of the time and the item-level two-way cluster test {pct(cc['twoway_cluster_false_positive'], 1)} "
                           f"({cc['replications']} simulated studies; 5% is right)."]),
              ("h2", "Known-answer gate and collapse rule"),
              ("table", ["Model's true rule-following accuracy", "Chance it passes the gate (>= 34 of 40)"],
               [[pct(float(q), 0), pct(v, 1)] for q, v in sim["gate_pass_probability"].items()]),
              ("table", ["Learner's lean toward one answer", "Chance it is flagged as collapsed (>= 38 of 40 the same)"],
               [[pct(float(q), 0), pct(v, 2)] for q, v in sim["collapse_flag_probability"].items()])]

    if rep:
        t = rep["tamper"]
        B += [("h1", "6. Reproducibility and tamper test"),
              ("bullets", [f"The W1 settings (reduced size) were run twice from scratch. {rep['repeat_files_compared']} result files compared "
                           f"(logs and run IDs left out): identical = {rep['repeat_identical']}."
                           + (f" Different: {', '.join(rep['repeat_files_different'])}." if rep["repeat_files_different"] else ""),
                           f"`verify` on the second run before tampering: {'all match' if t['before'] else 'MISMATCH'}. After changing one "
                           f"byte in learners_llm.csv: {'caught' if not t['after_one_byte_change'] else 'NOT caught'} "
                           f"({', '.join(t['changed_files_found'])}). After putting it back: {'all match' if t['after_restore'] else 'MISMATCH'}."])]

    issues = []
    for wid, w in W.items():
        for e in w["expectations"]:
            if e["met"]:
                continue
            msg = f"{wid}: expected '{e['what']}' but saw {e['observed']}."
            if "below the 3-point bar" in e["observed"]:
                msg += (" Why: the mock learners already score about 83% from 30 practice cases, so the planted lesson adds only "
                        "1 to 2 points. The H1 rule correctly calls a gain this size detectable but not meaningful. The advance "
                        "expectation was set too high; it is kept as written (ISSUES.md P9, P10).")
            issues.append(msg)
    if curve.get(0.03, {}).get("H1_rule", 1) < 0.7:
        issues.append(f"Design property (not a bug): requiring the observed difference to reach {100 * sesoi:.0f} points means a true "
                      f"{100 * sesoi:.0f}-point effect is found only {pw(0.03)} of the time; 80% power is reached at about 4 points. "
                      "The preregistration states this.")
    B += [("h1", "7. Problems found and what changed"),
          ("bullets", issues or ["No world failed any of its pre-stated checks."])]

    B += [("h1", "8. What we can and cannot say"), ("h2", "We can say"),
          ("bullets", ["The software, self-checks and statistics behave as the plan says in the situations tested here.",
                       "At the planned size the H1 rule keeps false positives well under 5% and finds effects of 4 points or more almost always.",
                       "The gate, collapse rule, prior-knowledge probe and blinded run each catch the problem they were built for."]),
          ("h2", "We cannot say"),
          ("bullets", ["Anything about real ECGs, real AI models, or people. Mock learners are scripts.",
                       "That real language models will pass the gate or not collapse. That is only known when the study runs.",
                       "That the real effect, if any, is as large as in W1. W1 was built to have an effect."])]
    B += [("h1", "9. Words used"), ("table", ["Word", "Meaning"], GLOSSARY + [
        ["Mock learner", "A small script that pretends to be a learner, used only to test the software."],
        ["Synthetic world", "Made-up data and learners where the right answer is known in advance."],
        ["Power", "The chance that the study finds an effect that is really there."]])]
    B += [("h1", "10. How to check this report"),
          ("bullets", ["Re-make it: python run_study.py validate (about an hour on a laptop CPU; add --quick for a 10-minute version).",
                       "Every file in this folder is listed with its SHA-256 in SHA256SUMS.txt; each world folder has its own manifest and SHA256SUMS.txt.",
                       "validation_summary.json holds every number in this report.",
                       f"Master seed {cfg['study']['master_seed']}; simulations use seed {cfg['study']['master_seed'] + 99}."])]
    B += [("h1", "11. Common questions"), ("table", ["Question", "Answer"], [
        ["Did an AI do the research?", "No. The idea, hypotheses and every design choice are the author's. An AI assistant helped write code "
                                       "and drafts under the author's direction."],
        ["Why test on fake data first?", "So that any bug is found before the plan is locked, without looking at the real data and without "
                                         "being tempted to change the plan after seeing real results."],
        ["Why didn't you run a small real pilot?", "Decided in advance (D3): looking at real data before locking the plan would weaken the "
                                                    "preregistration. The fake worlds test the machinery instead."],
        ["Can these numbers go in a paper as results?", "No. They can be cited as validation of the method, labelled synthetic."]])]

    B += [("pagebreak",), ("h1", "Appendix A. Hypothesis tables for every world")]
    for wid, w in W.items():
        for e, a in w["analyses"].items():
            B += [("h2", f"{wid} {w['name']}: {e}"),
                  ("table", ["", "Claim", "Diff", "95% CI", "d", "p", "p (Holm)", "MDE", "Supported"], _hyp_rows(a))]
    B += [("h1", "Appendix B. Self-checks in every world"),
          ("table", ["World"] + [c["id"] for c in next(iter(W.values()))["checks"]],
           [[wid] + [c["result"] for c in w["checks"]] for wid, w in W.items()])]
    return B


def write_validation_report(out: Path, summary: dict, cfg: dict):
    from .pdf_report import render

    B = build_blocks(out, summary, cfg)
    text = blocks_to_md(B)
    (out / "VALIDATION_REPORT.md").write_text(text)
    (out / "VALIDATION_REPORT.html").write_text(md_to_html(text, "Synthetic validation report"))
    render(B, out / "VALIDATION_REPORT.pdf", VTITLE + " (SYNTHETIC)")
