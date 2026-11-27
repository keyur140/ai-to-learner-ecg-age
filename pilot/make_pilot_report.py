# Builds the exploratory pilot report (Markdown) from the pilot run folders. Pilot-side script, not part of the study pipeline.
import json, sys, hashlib, os
FREEZE_NOTE = os.environ.get("FREEZE_NOTE", "pending")
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

V3 = Path("/home/user/dev/ai2human_v03/runs/20261107T164810Z_study_pilot")
V2A = Path("/home/user/pilot/ai2human_v02/runs/20261107T114352Z_study_pilot")
V2B = Path("/home/user/pilot/ai2human_v02/runs/20261107T120108Z_study_pilot")
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
ORDER = ["no_training", "feedback_only", "cue_training", "scrambled_cues"]
NAME = {"no_training": "No training", "feedback_only": "Feedback only", "cue_training": "AI cues", "scrambled_cues": "Scrambled cues (placebo)"}
TESTS = ["pre", "post", "delayed", "transfer"]
J = lambda p: json.loads(Path(p).read_text())
pct = lambda x: "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100*x:.1f}%"

def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a.mean() - b.mean()
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    t = d / se if se > 0 else float("nan")
    p2 = 2 * stats.t.sf(abs(t), df)
    h = stats.t.ppf(0.975, df) * se
    return d, d - h, d + h, p2

def table(rows, head):
    out = ["| " + " | ".join(head) + " |", "| " + " | ".join(["---"] * len(head)) + " |"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)

def per_learner(items):
    it = items.copy()
    it["med"] = it.groupby(["condition", "learner", "test"]).p_older.transform("median")
    it["bc"] = ((it.p_older > it.med).astype(int) == it.label).astype(float)
    it.loc[it.p_older == it.med, "bc"] = 0.5
    it["raw"] = ((it.p_older > 0.5).astype(int) == it.label).astype(float)
    it["say_older"] = (it.p_older > 0.5).astype(float)
    g = it.groupby(["condition", "learner", "test"])
    auc = g.apply(lambda x: roc_auc_score(x.label, x.p_older) if x.label.nunique() > 1 and x.p_older.nunique() > 1 else 0.5)
    df = pd.DataFrame({"raw": g.raw.mean(), "bc": g.bc.mean(), "auc": auc,
                       "one_answer": g.say_older.mean().apply(lambda s: max(s, 1 - s))}).reset_index()
    return df

def group_table(pl, metric, fmt=pct):
    rows = []
    for c in ORDER:
        s = pl[pl.condition == c]
        if s.empty: continue
        n = s.learner.nunique()
        rows.append([NAME[c], n] + [fmt(s[s.test == t][metric].mean()) for t in TESTS])
    return table(rows, ["Group", "Learners", "Before", "Right after", "After interference", "New device"])

def contrasts(pl, metric, unit="points"):
    rows = []
    p = pl[pl.test == "post"]
    for a, b, lab in [("cue_training", "feedback_only", "AI cues minus feedback only (the H1 comparison)"),
                      ("cue_training", "scrambled_cues", "AI cues minus scrambled cues (the H7 comparison)"),
                      ("cue_training", "no_training", "AI cues minus no training"),
                      ("feedback_only", "no_training", "Feedback only minus no training")]:
        u, v = p[p.condition == a][metric], p[p.condition == b][metric]
        if len(u) < 3 or len(v) < 3: continue
        d, lo, hi, p2 = welch(u, v)
        rows.append([lab, f"{len(u)} vs {len(v)}", f"{100*d:+.1f}", f"{100*lo:+.1f} to {100*hi:+.1f}", f"{p2:.3f}" if p2 >= 0.001 else "<0.001"])
    return table(rows, ["Comparison (right after training)", "Learners", f"Difference ({unit})", "95% range", "Two-sided p"])

M = []
A = M.append
c3, t3, coh = J(V3 / "cues.json"), J(V3 / "teacher_metrics.json"), J(V3 / "cohort_summary.json")
c2a, c2b, t2a, t2b = J(V2A / "cues.json"), J(V2B / "cues.json"), J(V2A / "teacher_metrics.json"), J(V2B / "teacher_metrics.json")
nets3, nets2 = pd.read_csv(V3 / "learners_nets.csv"), pd.read_csv(V2A / "learners_nets.csv")
llm = {k: (pd.read_csv(V3 / f"learners_{k}.csv"), pd.read_csv(V3 / f"items_{k}.csv")) for k in ["llm", "llm_blinded"]}
gate = {k: list(J(V3 / f"gate_{k}.json").values())[0] for k in ["llm", "llm_blinded"]}
pl = {k: per_learner(v[1]) for k, v in llm.items()}
n_main, n_bl = len(llm["llm"][0]), len(llm["llm_blinded"][0])
checks = J(V3 / "checks.json") if (V3 / "checks.json").exists() else None

A("# Exploratory pilot on real data: report\n")
A("*AI-to-learner knowledge transfer study (ECG age).*\n")
A("**Author:** KEYUR SHAILESHKUMAR PATEL · ORCID [0009-0008-5261-043X](https://orcid.org/0009-0008-5261-043X)  \nMSc Artificial Intelligence student, Department of Computer Science, Brunel University London, UK · Independent researcher\n")
A("*Conducted independently; not an official Brunel University London project.*\n")
A("**Use of AI tools.** The research questions, hypotheses and design choices are the author's. The code and this draft were written with help from an AI assistant (Claude, by Anthropic) under the author's direction and checked by the author. Separately, the study uses AI models as its objects of study.  \n**Licence.** Text and figures CC BY 4.0; code MIT.\n")
A("> **Status: EXPLORATORY. Nothing here tests a hypothesis.** This pilot was run before preregistration, on purpose, to find real-data problems. It used PTB-XL folds 1-9 only. Fold 10 and the held-out recording device (AT-6 C 5.5) were sealed and were never opened (decision D7). Every number below can change in the confirmatory study.\n")

A("## 1. The short answer\n")
bcm = pl["llm"][pl["llm"].test == "post"].groupby("condition").bc.mean()
rawm = pl["llm"][pl["llm"].test == "post"].groupby("condition").raw.mean()
A(f"1. **The data and the AI teacher are fine.** PTB-XL matched PhysioNet's checksums ({coh['source']['integrity']['files_checked']:,} files). The teacher reached AUROC {t3['test']['teacher_auroc']:.3f} on the pilot's test fold (precondition: at least 0.80).")
A("2. **The pilot found a reproducibility bug** (P11): the same seed gave different teachers and different cue lists (5 cues, then 8). It is fixed in version 0.3.")
A(f"3. **The cue list was fragile, and is now stable** (P14, decision D9). With five teachers voting, four cues are unanimous (PR interval, T-wave height in lead II, R-wave height in aVL, QRS axis) and two more pass 3 of 5 (QTc, S-wave depth in V1). As a simple tally these six rules are right {pct(c3['report']['test']['rule_tally_accuracy_true'])} of the time.")
A("4. **The placebo lesson was misleading, and is now neutral** (P15, decision D10): it scored 36% on its own; the fixed re-draw rule brings it to 46-47%.")
A(f"5. **Experiment B (small nets) hits a ceiling.** Practice alone reaches {pct(nets3[nets3.condition=='feedback_only'].post_acc.mean())}; adding the AI's cues gives {pct(nets3[nets3.condition=='cue_training'].post_acc.mean())}. As designed it is unlikely to show a cue benefit (P16, open decision).")
A(f"6. **Experiment A (Qwen2.5-3B) is dominated by letter bias.** By the planned accuracy score every group is near chance ({pct(rawm.get('no_training'))} to {pct(rawm.get('feedback_only'))}) and the preregistered collapse rule marks the model uninformative. But the model's probabilities do rank the ECGs: with a bias-corrected score (defined after seeing the data, so exploratory) the groups separate: no training {pct(bcm.get('no_training'))}, scrambled {pct(bcm.get('scrambled_cues'))}, AI cues {pct(bcm.get('cue_training'))}, feedback only {pct(bcm.get('feedback_only'))} (P19, open decision).")
A("7. **The language-model code could not run long prompts** without running out of memory (P17, P18). Fixed in 0.3.")
A("8. **In this pilot the AI's cues did not beat practice with feedback**, in either experiment, by any score. By the bias-corrected score they beat the placebo in the main run of Experiment A (+6.6 points); in the blinded run the gap was smaller (+3.6 points) and not clear. This is one small model and 30 learners per group, so it is a warning for the design, not a result.\n")

A("## 2. What was run\n")
A(table([
    ["v0.2 pilot, run 1", "0.2.0", "Teacher, cues, Experiment B (30 nets per group)", "Complete. First run on this machine (signals read fresh from the WFDB files)."],
    ["v0.2 pilot, run 2", "0.2.0", "Teacher, cues, then Experiment A", "Teacher and cues complete. Showed the reproducibility bug. Experiment A stopped at 20 learners (machine restart, results not saved: P13)."],
    ["Consensus check", "0.3.0 (in development)", "Five teachers and cue consensus only", "Complete. Same cues as the later full run."],
    ["v0.3 pilot", "0.3.0", "Five teachers, cues, Experiment B (30 per group), Experiment A main and blinded (Qwen2.5-3B, 30 per group)", f"Complete: Experiment B, Experiment A main ({n_main} learners) and blinded ({n_bl} learners)." if n_bl == 120 else f"Experiment B and Experiment A main complete ({n_main} learners). Blinded: {n_bl} of 120 learners."],
], ["Run", "Code version", "What", "Status"]))
A("\nAll runs used master seed 20261026, the protected pilot split (teacher folds 1-7, cue fold 8, test fold 9; transfer device CS-12, the largest remaining device), CPU only (4 cores, no GPU), and `python run_study.py run --pilot 30`.")
A(f"\nThe v0.3 run was interrupted several times (machine restarts and two out-of-memory kills) and resumed each time. Learners are seeded one by one, so a resumed learner gets exactly the prompts it would have got. Two memory fixes (P17, P18) were made between resumes; they change how the same computation is batched, not what is computed. Learners finished before and after each fix are in the same files. For the last resumes a small helper script (`pilot/resume_llm_only.py`) continued the language-model learners from the saved cues and settings without retraining the five teachers; a final normal `--resume` then retrained the teachers, checked that they match the saved ones (self-check C14), and wrote the self-checks and the run report. See Section 9.\n")
A("**Models.** Llama-3.2-3B and Gemma-2-2B are licence-gated and no Hugging Face token was set, so the pilot used Qwen2.5-3B-Instruct only (decision D8; commit `aa8e72537993ba99e69dfaafa59ed015b17504d1`, the head of the repository on 2026-11-07; not pinned in the config).\n")

A("## 3. Data and cohort (pilot split)\n")
r = coh["roles"]
A(table([[k.replace("_", " "), v["n"], v["n_older"], v["n"] - v["n_older"]] for k, v in r.items()], ["Role", "ECGs", "Older (65+)", "Younger (under 40)"]))
pr = coh["protected"]
A(f"\n{coh['n_total_records']:,} records, {coh['n_in_age_groups']:,} in the two age groups. **Sealed and never opened: {pr['sealed_ecgs']:,} ECGs of {pr['sealed_patients']:,} patients** (fold {pr['sealed_fold']} and device {pr['sealed_device']}). {coh['n_cohort']:,} ECGs of {coh['n_patients']:,} patients remained. No patient is in two roles (self-check C1).\n")
A("Note for the design: the pilot's transfer device (CS-12) is 91% older adults, so learners' transfer tests draw their younger half from a small pool (198 ECGs).\n")

A("## 4. The AI teacher\n")
rows = [["v0.2 run 1 (fresh read)", t2a["best_epoch"], f"{t2a['test']['teacher_auroc']:.3f}", pct(t2a['test']['teacher_accuracy'])],
        ["v0.2 run 2 (cached read)", t2b["best_epoch"], f"{t2b['test']['teacher_auroc']:.3f}", pct(t2b['test']['teacher_accuracy'])],
        ["v0.3 primary teacher", t3["best_epoch"], f"{t3['test']['teacher_auroc']:.3f}", pct(t3['test']['teacher_accuracy'])]]
rows += [[f"v0.3 consensus teacher {e['k']}", e["best_epoch"], f"{e['test_auroc']:.3f}", ""] for e in t3["ensemble_for_cues"]]
A(table(rows, ["Teacher", "Best epoch", "AUROC, test fold", "Accuracy"]))
A(f"\nA plain logistic regression on the 18 panel measurements reaches AUROC {t3['test']['features_baseline_auroc']:.3f} ({pct(t3['test']['features_baseline_accuracy'])}): most of the age signal the teacher uses is already in the numbers learners see. On the transfer device the primary teacher scores AUROC {t3['transfer']['teacher_auroc']:.3f}.\n")
A("**The reproducibility bug (P11).** Runs 1 and 2 used the same code, seed and data, and should have been identical. They were not: the per-lead scaling was summed in 32-bit floats over about 5 million values, and the result depended on whether the signals had just been read from disk or loaded from the cache (standard deviations 0.5% apart). Version 0.3 sums in 64-bit; a fresh read and a cached read now agree bit for bit, and the v0.3 run reproduced the separate consensus check exactly (same cues, same teacher AUROC).\n")

A("## 5. The cues\n")
ids = sorted({x["id"] for c in (c2a, c2b, c3) for x in c["cues"]}, key=lambda f: -c3["report"]["consensus"]["votes"].get(f, 0))
lab = {p["id"]: p["label"] for p in __import__("yaml").safe_load(open(V3 / "config_used.yaml"))["panel"]}
inn = lambda c, f: "yes" if f in [x["id"] for x in c["cues"]] else "-"
A(table([[lab[f], inn(c2a, f), inn(c2b, f), f"{c3['report']['consensus']['votes'].get(f,0)} of 5", inn(c3, f)] for f in ids],
        ["Measurement", "v0.2 run 1 (5 cues)", "v0.2 run 2 (8 cues)", "v0.3: teachers that chose it", "v0.3 cue"]))
A("\n**Why the list swung (P14).** The rule takes the sparsest setting with at least 5 stable features. In run 1 that setting gave exactly 5; with a barely different teacher it gave 4, so the rule moved to a looser setting and took 8. **Decision D9:** train five teachers (seeds 0-4), run the same selection on each, keep features chosen by at least 3. All five teachers agree on the direction of every kept cue.\n")
A("**The v0.3 lesson, as learners saw it:**\n")
A("```\n" + (V3 / "cue_lesson.txt").read_text() + "\n```\n")
rep = c3["report"]["test"]
A(table([["AI cue rules, applied as a tally", pct(rep["rule_tally_accuracy_true"]), pct(rep["rule_tally_fidelity_to_teacher"])],
         ["Surrogate model on the 6 cues", pct(rep["surrogate_accuracy_true"]), pct(rep["surrogate_fidelity_to_teacher"])],
         ["Scrambled (placebo) rules, as a tally", pct(rep["scrambled_tally_accuracy_true"]), "-"],
         ["Teacher itself", pct(t3["test"]["teacher_accuracy"]), "100%"]],
        ["On the pilot test fold", "Agrees with true age group", "Agrees with the teacher"]))
A(f"\n**The placebo (P15, decision D10).** With the v0.3 cues the first placebo draw scored 36% as a tally: its random directions mostly pointed the wrong way, so it could have taught the opposite and inflated H7. The fixed rule now re-draws (seeded) until the tally is within 45-55% on the cue fold. Here: draw {c3['scrambled'][0]['placebo_draw']}, {pct(c3['scrambled'][0]['placebo_cue_fold_accuracy'])} on the cue fold, {pct(rep['scrambled_tally_accuracy_true'])} on the test fold.\n")

A("## 6. Experiment B: small neural nets (30 per group)\n")
def nets_table(d):
    return table([[NAME[c], len(d[d.condition == c])] + [pct(d[d.condition == c][f"{t}_acc"].mean()) for t in TESTS] for c in ORDER],
                 ["Group", "Learners", "Before", "Right after", "After interference", "New device"])
A("**v0.3 (consensus cues, neutral placebo):**\n"); A(nets_table(nets3))
A("\n**v0.2 run 1 (5 cues, old placebo):**\n"); A(nets_table(nets2))
pn = nets3.rename(columns={"post_acc": "raw"}).assign(test="post")
A("\n" + contrasts(pn, "raw"))
A("\n**Reading.** Sixty-four labelled ECGs are enough for a small net to reach about 81% by itself; the extra cue-prediction loss adds nothing right after training. After the interference task the cue and scrambled groups keep more than feedback only, by a similar amount, so that difference comes from the extra training signal, not from the AI's content. This is the ceiling already seen in synthetic world W1 (issues P9, P10), now on real data (P16).\n")

A("## 7. Experiment A: language-model learners (Qwen2.5-3B, 30 per group)\n")
for k, title in [("llm", "Main run (real measurement names)"), ("llm_blinded", "Blinded run (names replaced by 'Measure 1..18')")]:
    d, g = llm[k][0], gate[k]
    A(f"### {title}: {len(d)} learners\n")
    if k == "llm_blinded" and len(d) < 120:
        A(f"*Incomplete: {len(d)} of 120 learners ({', '.join(f'{NAME[c]} {int((d.condition==c).sum())}' for c in ORDER)}).*\n")
    A(f"Known-answer gate: {pct(g['gate_accuracy'])} (bar 85%): {'passed' if g['gate_pass'] else 'FAILED'}. " + (f"Share of learners giving nearly one answer on the post-test: {pct(g['post_collapse_fraction'])}; by the preregistered rule (over half) the model is **{'uninformative' if g.get('collapsed_model') else 'informative'}**.\n" if "post_collapse_fraction" in g else "\n"))
    A("**(a) Planned score: accuracy (answer = the letter with the higher probability).**\n"); A(group_table(pl[k], "raw"))
    A("\nShare of items on which a learner gave its more frequent answer (50% = balanced, 100% = one answer only):\n"); A(group_table(pl[k], "one_answer"))
    A("\n**(b) Exploratory, defined after seeing the data: bias-corrected accuracy.** Each learner's items are split at that learner's own median probability of 'older' on that test (tests are half older, half younger, so chance is 50%).\n"); A(group_table(pl[k], "bc"))
    A("\n" + contrasts(pl[k], "bc"))
    A("\n**(c) Exploratory: AUROC of the learner's probability of 'older'** (0.50 = no information, 1.00 = perfect ranking).\n"); A(group_table(pl[k], "auc", lambda x: f"{x:.3f}")); A("")
A("**Reading.**\n")
A("- Qwen passes the short known-answer gate, so it can apply one written rule. On the real task it answers almost only 'A' or only 'B' whatever the ECG. Because letters are balanced across learners, this bias does not favour any group; it pushes every group's accuracy towards 50% and hides differences.\n- Underneath the bias the probabilities carry information after training. That is why scores (b) and (c) separate the groups while score (a) does not.\n- Practice with feedback gave the largest gain. The AI's cues gave a gain of similar size or a little smaller, and clearly more than the placebo lesson in the main run.\n- In the blinded run the pre-test is *below* chance by score (b): with meaningless names and no lesson, the model's prior ordering of the numbers runs slightly against the truth. Training reverses it.\n- With no training the model does not improve, as it should not.\n- The placebo group also improves, because every taught group sees the same 24 practice cases with correct answers. In the main run the placebo group ends well below feedback only (58.7% against 69.2%): a wrong lesson seems to get in the way of learning from the cases. The AI's lesson, which is right, also ends a little below feedback only. So in this pilot a written lesson did not add to the practice cases, whatever its content; the open question is whether that is the design (24 cases already teach the pattern) or this model.\n- The prior-knowledge probe says 'higher in older adults' for 17 of 18 measurements with near-certainty, which looks like the same yes-bias, so the probe is not informative as written (P20).\n")
A("**Prompt length.** Feedback and cue prompts are about 9,000 to 12,000 tokens (24 practice ECGs of 18 numbers each, plus the interference block). This is long for a 3B model and may be part of why answers collapse; the pilot cannot separate the two.\n")

A("## 8. Self-checks\n")
if checks:
    A(table([[c["id"], c["check"], c["result"], c["detail"]] for c in checks], ["#", "Check", "Result", "Detail"]))
else:
    A("*The self-checks C1-C17 of the v0.3 run are written when the run completes. In the v0.2 run 1 (Experiment B only), 10 passed, 0 failed, 5 were not applicable, including C15 (no sealed ECG was used: 0 found).*")
A("")

A("## 9. Problems found, and what was done\n")
A(table([
    ["P11", "Same seed, different teacher and cues (32-bit summation depended on memory layout)", "High", "Fixed in 0.3 (64-bit); test added"],
    ["P12", "`check-data` failed on a fresh folder", "Low", "Fixed in 0.3"],
    ["P13", "Results saved only every 25 learners; 20 finished learners lost in a restart", "Medium", "Fixed in 0.3 (saved after every learner)"],
    ["P14", "Cue list fragile (5 vs 8 cues)", "High (design)", "Decision D9: consensus of 5 teachers; self-check C16"],
    ["P15", "Placebo lesson could point the wrong way (36%)", "Medium (design)", "Decision D10: seeded re-draw to 45-55%; self-check C17"],
    ["P16", "Experiment B: practice alone reaches the ceiling", "Design", "**Open: author's decision**"],
    ["P17", "Out of memory on long prompts (13.9 GB)", "High", "Fixed in 0.3 (prompt fed in chunks; last-position logits only)"],
    ["P18", "Out of memory again in the cue group (cache copied 8 times)", "High", "Fixed in 0.3 (memory budget for cache copies)"],
    ["P19", "Experiment A: letter bias swamps the accuracy score; the collapse rule would exclude the model", "High (design)", "**Open: author's decision** on the outcome measure and the collapse rule"],
    ["P20", "Prior-knowledge probe answers 'higher' for almost everything", "Low", "**Open**: ask each question both ways, or drop the probe"],
    ["P21", "Run-time estimates in the README were far too low for CPU; long prompts cost about 11 minutes per learner on 4 cores", "Low", "README updated with measured times"],
    ["P22", "Pilot seal is applied after the feature table is loaded: a unit check and the missing-value count run over all rows, sealed ones included", "Low", "Disclosed exactly in `prereg/PILOT_DISCLOSURE.md` section 2; no age-feature relation is computed there"],
], ["#", "Problem", "Severity", "Status"]))
A("\nDetails are in `ISSUES.md`.\n")

A("## 10. What the pilot means for the preregistration\n")
A("Decisions already taken by the author during the pilot: D7 (protected split), D8 (Qwen only for the pilot), D9 (cue consensus), D10 (neutral placebo), D11 (rerun on v0.3 at 30 per group).\n")
A("Open decisions, for the author, before freezing:\n")
A("1. **Outcome measure for Experiment A (P19).** Keep raw accuracy as primary, or switch to a bias-free score (bias-corrected accuracy or AUROC), and what the collapse rule should then exclude.\n2. **Experiment B (P16).** Keep as is (expecting a null), make practice scarcer so there is room for cues to help, or drop it to a secondary role.\n3. **Is H1 the right primary?** In this pilot the AI's cues did not beat feedback-only anywhere. H7 (real cues beat placebo) and 'cues beat no training' looked more promising. One small model, so this is weak evidence, but the choice of primary must be made before the real data are seen.\n4. **Prompt length and practice size** for language models (24 practice ECGs make 9-12k-token prompts).\n5. **Models:** add the Hugging Face token so Llama-3.2-3B and Gemma-2-2B run; pin all three commits.\n")
A("**What must be disclosed in the preregistration** is drafted in `prereg/PILOT_DISCLOSURE.md`: this pilot happened, which data it touched and did not touch, what was seen, and which design choices were changed because of it.\n")

A("## 11. Limits of this pilot\n")
A("- One language model of three; 30 learners per group; one seed.\n- Folds 1-9 only, with a different split from the confirmatory study (teacher on 7 folds instead of 8; a different transfer device).\n- Scores (b) and (c) in Section 7 were chosen after looking at the data.\n- CPU run in bfloat16; probabilities for very long prompts are precise to a few hundredths (P17).\n- The v0.3 run was resumed several times and two memory fixes were applied between resumes.\n- The v0.2 Experiment A results (20 learners) were lost and are not reported.\n")

A("## 12. Files\n")
A("`pilot/v0.3_run/` holds the full outputs of the v0.3 pilot run (learner-level and item-level CSVs, lessons, example prompts, cues, teacher metrics, logs, manifest, SHA256SUMS). `pilot/v0.2_run1/` and `pilot/v0.2_run2/` hold the two v0.2 runs. Teacher weights and cohort tables derived from PTB-XL are included; PTB-XL itself is not (download it from PhysioNet).\n")
(OUT / "EXPLORATORY_PILOT_REPORT.md").write_text("\n".join(M))
print("written", OUT / "EXPLORATORY_PILOT_REPORT.md", len("\n".join(M)))


# ---------------------------------------------------------------- prereg disclosure
D = []
B = D.append
b_main, b_bl = pl["llm"][pl["llm"].test == "post"], pl["llm_blinded"][pl["llm_blinded"].test == "post"]
def g(df, c, m): 
    v = df[df.condition == c][m]
    return pct(v.mean()) if len(v) else "n/a"
B("# Disclosure of the exploratory real-data pilot\n")
B("*AI-to-learner knowledge transfer study (ECG age). Author: KEYUR SHAILESHKUMAR PATEL (ORCID https://orcid.org/0009-0008-5261-043X). Conducted independently; not an official Brunel University London project.*\n")
B(f"*Part of the preregistration: this file is hashed into `prereg/LOCK.json` together with `preregistration.md`. Drafted with help from an AI assistant (Claude, by Anthropic) under the author's direction; checked and confirmed by the author before freezing ({FREEZE_NOTE}).*\n")
B("## 1. What happened\n")
B("Version 0.2.0 planned no real-data run before preregistration (decision D3). On 2026-11-07 the author decided to run an exploratory pilot on real data first, in a protected form (decision D7). It ran from 2026-11-07 to 2026-11-12. This file states what was looked at, what was not, what was seen, and what was changed because of it, so that readers can judge the confirmatory study with that in mind.\n")
B("## 2. Data that were used, and data that were not\n")
B(table([
    ["PTB-XL folds 1-7", "Teacher training and validation; learners' practice ECGs", "Teacher training (with fold 8)"],
    ["PTB-XL fold 8", "Cue extraction", "Teacher training"],
    ["PTB-XL fold 9", "Learners' tests (pre, post, delayed)", "Cue extraction"],
    ["Device CS-12 (largest device after sealing)", "Learners' transfer test", "Part of folds 1-9 as above"],
    ["**PTB-XL fold 10**", "**Sealed. Never opened.**", "**Learners' tests: the data H1 is tested on**"],
    ["**Device AT-6 C 5.5** (2nd-largest overall)", "**Sealed. Never opened.**", "**Transfer test**"],
], ["Data", "Role in the pilot", "Role in the confirmatory study"]))
B(f"\nThe seal is enforced in code (`run --pilot`): {pr['sealed_ecgs']:,} ECGs of {pr['sealed_patients']:,} patients were removed before anything else ran, and self-check C15 counts sealed ECGs in the run (0 found). Patients on the sealed device were removed with all their ECGs.\n")
B("**What the code touched in the sealed data, stated exactly (P22).** The seal is applied in the cohort step, after three things that run over the whole dataset, sealed rows included: (1) every downloaded file is hashed and compared with PhysioNet's checksums (bytes only, nothing is parsed); (2) the PTB-XL+ feature table is loaded whole, and for each amplitude measurement the code takes the median absolute value over all rows to decide whether the file stores millivolts or microvolts; (3) to pick the sealed device by the preregistered rule ('2nd-largest'), the code counts eligible ECGs per recording device, and 'eligible' uses each ECG's age group and how many of its 18 panel values are missing. None of these relates age to any ECG feature, none is printed per ECG, and no ECG signal of a sealed ECG was read (signals are loaded only for the kept rows). After the seal, sealed rows are dropped from every table and are not written to any output file.\n")
B("## 3. What the author has now seen\n")
B("Before this pilot the author had not analysed how age relates to any ECG feature in PTB-XL. That is no longer true for folds 1-9. The author has seen:\n")
B(f"- that the teacher separates the two age groups very well on fold 9 (AUROC {t3['test']['teacher_auroc']:.3f}), and that a logistic regression on the 18 panel measurements also does (AUROC {t3['test']['features_baseline_auroc']:.3f});")
B("- which measurements the teacher's guesses load on in fold 8, and in which direction: PR interval (longer in older), T-wave height in lead II (lower), R-wave height in aVL (higher), QRS axis (lower), QTc (longer), S-wave depth in V1 (higher); and that a tally of these six rules is right about " + pct(c3['report']['test']['rule_tally_accuracy_true']) + " of the time on fold 9;")
B(f"- Experiment B on fold 9, 30 nets per group: after training, feedback only {pct(nets3[nets3.condition=='feedback_only'].post_acc.mean())}, AI cues {pct(nets3[nets3.condition=='cue_training'].post_acc.mean())}, scrambled {pct(nets3[nets3.condition=='scrambled_cues'].post_acc.mean())}, no training {pct(nets3[nets3.condition=='no_training'].post_acc.mean())};")
B(f"- Experiment A with one of the three planned models (Qwen2.5-3B), 30 learners per group, main run ({n_main} learners) and blinded run ({n_bl} learners). By the planned accuracy score: feedback only {g(b_main,'feedback_only','raw')}, AI cues {g(b_main,'cue_training','raw')}, scrambled {g(b_main,'scrambled_cues','raw')}, no training {g(b_main,'no_training','raw')}; the model answered almost only one letter and was marked uninformative by the collapse rule. By a bias-corrected score chosen after seeing this (each learner's items split at its own median probability): feedback only {g(b_main,'bc','bc') if False else g(b_main,'feedback_only','bc')}, AI cues {g(b_main,'cue_training','bc')}, scrambled {g(b_main,'scrambled_cues','bc')}, no training {g(b_main,'no_training','bc')} (main); feedback only {g(b_bl,'feedback_only','bc')}, AI cues {g(b_bl,'cue_training','bc')}, scrambled {g(b_bl,'scrambled_cues','bc')}, no training {g(b_bl,'no_training','bc')} (blinded).\n")
B("- A follow-up on 2026-11-14 (`pilot/PILOT_FOLLOWUP.md`): Experiment B on fold 9 with 4, 8, 16, 32, 64 and 128 practice ECGs, 30 nets per group each. At no size did the AI's cues beat feedback only (differences -4.6 to +1.2 points), and the scrambled cues did as well as the real ones.\n")
B("In short: **in the pilot the AI's cues did not beat feedback-only practice** (the H1 comparison) in either experiment, and by the bias-corrected score they beat the placebo in the main run of Experiment A (+6.6 points) but not clearly in the blinded run (+3.6 points). The author knows this when writing the final preregistration.\n")
B("The author has not seen: any ECG, feature or result from fold 10 or from device AT-6 C 5.5; any result from Llama-3.2-3B or Gemma-2-2B; any result at the planned sample size.\n")
B("## 4. Replacement for the 'existing data' statement\n")
B("The bracketed statement in section 3 of the preregistration is replaced by:\n")
B("> As of the date of this preregistration, the author has run one exploratory pilot on PTB-XL folds 1-9 (described in `prereg/PILOT_DISCLOSURE.md` and reported in full in `pilot/EXPLORATORY_PILOT_REPORT.md`). In it the author saw how the teacher's guesses relate to the 18 panel measurements and saw exploratory group results for 30 learners per group. The author has not opened, analysed or seen any summary of the confirmatory test data: PTB-XL fold 10 and all ECGs from the held-out recording device. The synthetic validation (section 7) opened no PTB-XL file.\n")
B("## 5. Changes made because of the pilot\n")
B("Each was decided by the author on the date shown, after seeing the pilot result named, and before any confirmatory data were seen.\n")
B(table([
    ["D9", "2026-11-07", "Cue consensus across five teacher seeds (keep a cue if at least 3 of 5 teachers choose it)", "Two runs gave 5 and then 8 cues (P11, P14)", "Changes which cues are taught. Does not use outcomes."],
    ["D10", "2026-11-08", "Placebo lesson re-drawn (fixed seed) until its rules score 45-55% on the cue fold. (Built into the v0.3 pilot run on 2026-11-07 as the assistant's recommended option; the rule was therefore fixed in code before any v0.3 learner ran. The author confirmed it on 2026-11-08; by then the v0.3 main run of Experiment A had finished, so the confirmation was not blind to those results.)", "The placebo scored 36% as a tally (P15)", "Makes H7 harder to pass, not easier: a misleading placebo would have inflated H7."],
    ["D12", "2026-11-14", "Primary score of Experiment A changed from raw accuracy to bias-corrected accuracy (each learner's items split at its own median probability).", "Qwen2.5-3B answered almost only one letter; raw accuracy was near 50% in every group while the probabilities still ranked the ECGs (P19)", "**Yes, it could.** This is a change of the primary outcome made after seeing pilot outcomes, including the group means under both scores. It does not favour one group over another by construction (it is applied identically to every learner and chance stays 50%), but it was chosen knowing that it shows group differences in folds 1-9."],
    ["D13", "2026-11-14", "Experiment B becomes secondary; it runs exactly as designed", "Cues did not beat feedback in Experiment B at any practice size from 4 to 128 ECGs (P16)", "Removes a primary test that the pilot says would fail. It does not change any test or make any claim easier to support; the expected null is stated in advance."],
    ["D14", "2026-11-14", "No change: H1 (cues beat feedback-only practice by at least 3 points) stays primary in Experiment A", "Cues did not beat feedback in the pilot; they beat the placebo in the main run", "No. The author kept the original question although the pilot suggests it may fail."],
    ["D15", "2026-11-14", "Flat rule replaces the collapse rule for deciding which models count", "The collapse rule would have removed Qwen2.5-3B (P19)", "It keeps a model the old rule would remove. Chosen knowing that this model shows group differences under the new score in folds 1-9."],
    ["D16", "2026-11-14", "No change: 24 practice cases are kept in Experiment A", "24 cases alone taught Qwen2.5-3B as much as the cues did; prompts of 9,000 to 12,000 tokens", "No. The author declined to change the practice size or to look at more pilot data."],
    ["D17", "2026-11-20", "No change: the confirmatory run stays in bfloat16 with the float32 answer read-out", "Rounding noise up to 0.06 in probability inside the 16-bit model (P23)", "No. The noise is random and the same for every group."],
    ["D18", "2026-11-20", "The prior-knowledge probe asks each question twice with the letters swapped", "The probe said 'higher' for 17 of 18 measurements (P20)", "No. The probe is descriptive only."],
    ["Fix P23", "2026-11-14", "Answer letters read in float32", "About 13 distinct probabilities per 40-item test; tie rates differed by group", "Measurement precision. Reduces ties, which pulled scores towards 50% unevenly across groups."],
    ["Fix P11", "2026-11-07", "Input scaling summed in 64-bit", "Same seed gave different teachers", "Reproducibility only."],
    ["Fixes P13, P17, P18", "2026-11-07/08", "Save after every learner; bounded memory for long prompts", "Lost results; out-of-memory kills", "Same computation, different batching. Probabilities can differ in the second or third decimal (bfloat16)."],
], ["Ref", "Date", "Change", "Seen in the pilot", "Could it favour the hypotheses?"]))
B("\nNo other design choice was changed after the pilot. On 2026-11-27, before freezing, the three learner models were pinned to the exact Hugging Face commits current on that day (Qwen2.5-3B-Instruct `aa8e7253`, Llama-3.2-3B-Instruct `0cb88a4f`, Gemma-2-2b-it `299a8560`; full hashes in `configs/study.yaml`), as the plan required; this needs no judgment and uses no data. The Hugging Face token needed to download Llama and Gemma will be added before the confirmatory run; it changes nothing in the plan.\n")
B("## 6. Why the confirmatory test is still a fair test, and what risk remains\n")
B("- The data H1 is tested on (fold 10, and the held-out device for H5) were never opened, so no choice could have been tuned to them.\n- The confirmatory study retrains everything from scratch with its own split (teacher on folds 1-8, cues from fold 9); no teacher, cue list or lesson from the pilot is reused.\n- The confirmatory learners draw their practice ECGs from folds 1-8, which the pilot also used. Practice ECGs are shown with their true labels in every taught group, so this cannot favour one group.\n")
B("Risks that remain, stated plainly:\n")
B("- The author chose design details knowing the pilot's results on folds 1-9. Folds 1-9 and fold 10 come from the same hospital network and the same population, so a design tuned on one tends to behave similarly on the other. This is the usual cost of piloting; the protection is that every such choice is listed in section 5 with its date, and that the analysis is frozen before fold 10 is opened.\n- The pilot used one language model. Design choices that suit Qwen2.5-3B may not suit the other two.\n- The bias-corrected score in section 3 was invented after seeing the planned score fail. If it becomes part of the confirmatory analysis, that is a choice made with knowledge of pilot outcomes, and it is listed as such.\n")
B("## 7. Where everything is\n")
B("`pilot/EXPLORATORY_PILOT_REPORT.md` (and `.pdf`): full report. `pilot/v0.3_run/`, `pilot/v0.2_run1/`, `pilot/v0.2_run2/`: run outputs with logs, manifests and SHA-256 sums. `pilot/PILOT_FOLLOWUP.md`: follow-up of 2026-11-14. `ISSUES.md` P11-P23. `DECISIONS.md` D7-D18. `CHANGELOG.md` 0.3.0 to 0.5.1.\n")
(OUT / "PILOT_DISCLOSURE.md").write_text("\n".join(D))
print("written", OUT / "PILOT_DISCLOSURE.md")
