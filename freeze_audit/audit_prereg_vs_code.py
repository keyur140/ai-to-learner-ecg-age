# Audit: does prereg/preregistration.md state exactly what configs/study.yaml and the code do?
# Each check reads the value from the config or code and looks for the exact statement in the prereg text.
# Usage: python audit_prereg_vs_code.py <package_root>   -> prints a table, writes audit_prereg_vs_code.json, exit 1 on any mismatch
import json, re, sys
from pathlib import Path
import yaml

ROOT = Path(sys.argv[1])
P = (ROOT / "prereg/preregistration.md").read_text()
C = yaml.safe_load((ROOT / "configs/study.yaml").read_text())
src = {f.name: f.read_text() for f in (ROOT / "src/ai2human").glob("*.py")}
d, t, cu, el, en, an = C["data"], C["teacher"], C["cues"], C["experiment_llm"], C["experiment_nets"], C["analysis"]
pct = lambda x: f"{round(100 * x):d}%"
rows = []

def chk(item, value, *phrases, where="prereg"):
    text = P if where == "prereg" else where
    miss = [ph for ph in phrases if (ph.search(text) is None if isinstance(ph, re.Pattern) else ph not in text)]
    rows.append({"item": item, "code_value": str(value), "prereg_says": [getattr(p, "pattern", p) for p in phrases], "match": not miss,
                 "missing": [getattr(p, "pattern", p) for p in miss]})

# study and data
chk("Master seed", C["study"]["master_seed"], f"[{C['study']['master_seed']},", f"seed {C['study']['master_seed']}")
chk("Younger group", d["young_max_age"], f"younger = age {d['young_max_age']} or under")
chk("Older group", d["older_min_age"], f"older = age {d['older_min_age']} or over")
chk("Missing panel features allowed", d["max_missing_panel"], f"missing more than {d['max_missing_panel']} of the 18 panel features")
chk("Teacher folds", d["teacher_train_folds"], "teacher training: folds 1-8")
chk("Cue fold", d["cue_fold"], f"cue extraction: fold {d['cue_fold']}")
chk("Test fold", d["test_fold"], f"learner tests: fold {d['test_fold']}")
chk("Validation fraction", d["teacher_val_fraction"], f"below {d['teacher_val_fraction']:.2f}")
chk("Transfer device rank", d["transfer_device_rank"], "2nd-most eligible ECGs")
chk("Panel size", len(C["panel"]), "these 18 numbers")
for p in C["panel"]:
    lab = p["label"].split(",")[0].replace("-wave", "").replace(" (Bazett)", "")
chk("Panel labels (sample)", "QTc (Bazett), PR interval, Q-wave depth (III)", "QTc (Bazett)", "PR interval", "Q-wave depth (III)")
# teacher
chk("Teacher architecture", f"{t['arch']} {t['channels']}ch {t['blocks']} blocks", f"1D ResNet ({t['channels']} channels, {t['blocks']} blocks)")
chk("Early stopping", f"patience {t['patience']}, {t['epochs']} epochs", f"patience {t['patience']}, max {t['epochs']} epochs")
chk("Teacher seeds", cu["n_teacher_seeds"], f"k = 0 to {cu['n_teacher_seeds'] - 1}", "seeds 20261026 + 1000k")
chk("Teacher seed rule in code", "base + 1000*k", "1000 * k", where=src["teacher.py"])
# cues
chk("Bootstraps", cu["stability_bootstraps"], f"{cu['stability_bootstraps']} bootstraps")
chk("C grid", f"{min(cu['l1_C_grid'])}-{max(cu['l1_C_grid'])}", f"C grid {min(cu['l1_C_grid'])}-{max(cu['l1_C_grid'])}")
chk("Stability threshold", cu["stability_threshold"], f"selection frequency {cu['stability_threshold']} or more")
chk("Cue count min/max", f"{cu['k_min']}-{cu['k_max']}", f"at least {cu['k_min']} features", f"at most {cu['k_max']} are kept")
chk("Consensus", f"{cu['consensus_min']} of {cu['n_teacher_seeds']}", f"at least {cu['consensus_min']} of the {cu['n_teacher_seeds']} teachers")
chk("Placebo seed", C["study"]["master_seed"] + 7, f"seed {C['study']['master_seed'] + 7} (master seed + 7)")
chk("Placebo seed in code", "master_seed + 7", '["master_seed"] + 7', where=src["cues.py"])
chk("Placebo band", cu["scrambled_band"], f"between {pct(cu['scrambled_band'][0])} and {pct(cu['scrambled_band'][1])}")
chk("Placebo max draws", cu["scrambled_max_draws"], f"{cu['scrambled_max_draws']:,} draws")
chk("Clearest cases", 2 * cu["n_prototypes_per_class"], f"{2 * cu['n_prototypes_per_class']} most confident")
chk("Saliency ECGs", C["saliency"]["n_ecgs"], f"{C['saliency']['n_ecgs']} cue-fold ECGs")
# groups and tests
chk("Groups", C["conditions"], *[c for c in C["conditions"]])
chk("LLM practice cases", el["n_practice"], f"{el['n_practice']} practice cases", f"{el['n_practice'] + 2 * cu['n_prototypes_per_class']} practice cases with correct answers ({el['n_practice']} + {2 * cu['n_prototypes_per_class']} extra)")
chk("Nets practice ECGs", en["n_practice"], f"{en['n_practice']} labelled practice ECGs")
chk("Nets steps", en["steps"], f"{en['steps']} steps")
chk("Nets cue loss weight", en["cue_loss_weight"], f"(weight {en['cue_loss_weight']})")
chk("Interference A", el["distractor_tokens"], f"about {el['distractor_tokens']:,} tokens")
chk("Interference B", f"{en['distractor_n']} ECGs, {en['distractor_steps']} steps", f"from {en['distractor_n']} other ECGs for {en['distractor_steps']} steps")
chk("Items per test", f"A {el['n_items_per_test']}, B {en['n_items_per_test']}", f"A: {el['n_items_per_test']} items per test", f"B: {en['n_items_per_test']} items per test")
chk("Experiment codes", "llm=1 nets=2 llm_blind=3", "llm = 1, nets = 2, llm_blinded = 3")
chk("Experiment codes in code", '{"llm": 1, "nets": 2, "llm_blind": 3}', '{"llm": 1, "nets": 2, "llm_blind": 3}', where=src["sampling.py"])
# sample size
n = el["n_per_condition"]
chk("Learners per group (A main)", n, f"{n:,} learners per group")
chk("Learners per group (A blinded)", el["blinded_n_per_condition"], f"Experiment A blinded run {4 * el['blinded_n_per_condition']:,}")
chk("Learners per group (B)", en["n_per_condition"], f"Experiment B {4 * en['n_per_condition']:,}")
chk("Models", el["models"], "3 language models", "Qwen2.5-3B", "Llama 3.2", "Gemma 2")
chk("Per model per group", n // len(el["models"]), f"{n // len(el['models'])} learners per group")
chk("Model revisions pinned", el.get("revisions") or "NOT PINNED", "pinned to exact Hugging Face commits")
rows[-1]["match"] = rows[-1]["match"] and bool(el.get("revisions")) and len(el["revisions"]) == len(el["models"])
# gate and model rules
g = el["gate"]
chk("Gate items and bar", f"{g['n_items']}, {g['min_accuracy']}", f"{g['n_items']} teacher-training ECGs", f"at least {pct(g['min_accuracy'])} ({round(g['min_accuracy'] * g['n_items'])} of {g['n_items']})")
chk("Gate rule feature", g["rule_feature"], "heart rate is above")
chk("Flat rule", f"tie share > {el['flat_tie_share']}, model > {el['flat_model_fraction']}", "more than half of its items tie", "more than half of its learners are flat")
chk("Collapse flag", el["collapse_share"], f"{pct(el['collapse_share'])} or more of its answers")
chk("Collapse does not remove a model", el["collapse_excludes_model"], "does not remove a model")
rows[-1]["match"] = rows[-1]["match"] and el["collapse_excludes_model"] is False
chk("Primary score A", el["primary_score"], "primary score: bias-corrected accuracy", "half credit")
rows[-1]["match"] = rows[-1]["match"] and el["primary_score"] == "bias_corrected"
chk("Answer read-out", el["answer_precision"], "read in float32")
chk("Model precision", el["dtype"], "16-bit", where=P + (ROOT / "DECISIONS.md").read_text())
chk("Blinded run", el["blinded_run"], '"Measure 1" to "Measure 18"')
chk("Probe both ways", "both orderings", "asked twice")
chk("Experiment B role", en["role"], "Experiment B is secondary")
# analysis
chk("Alpha", an["alpha"], f"alpha = {an['alpha']}")
chk("One-sided", an["primary_alternative"], "One-sided Welch")
chk("Meaningful difference", an["min_meaningful_diff"], f"at least **{round(100 * an['min_meaningful_diff'])} accuracy points**")
chk("Equivalence margin", an["equivalence_margin"], f"margin {round(100 * an['equivalence_margin'])} points")
chk("Bootstrap", an["bootstrap"], f"{an['bootstrap']:,}-sample bootstrap")
chk("Holm", "Holm across H2-H7", "Holm correction across H2-H7")
chk("Teacher precondition", 0.80, "AUROC of 0.80 or more")
chk("Pre-test precondition", 0.65, "0.65 or less")
chk("Self-checks", "C1-C17", "C1-C17")
nchecks = len(set(re.findall(r'add\("(C\d+)"', src["checks.py"])))
rows[-1]["code_value"] = f"{nchecks} checks defined in checks.py"
rows[-1]["match"] = rows[-1]["match"] and nchecks == 17
# hypotheses
for h in ["H1", "H2", "H3", "H4", "H5", "H6", "H7"]:
    chk(f"Hypothesis {h} in analysis code", h, f"**{h} " if h in ("H1", "H7") else f"- {h} ", f'"{h}' if h != "H1" else "H1_primary", where=P + src["analysis.py"])
# blanks left
NOT_BLANKS = {"[brackets]", "[20261026, experiment, group, learner_index]", "[its training median]"}  # wording, not blanks
blanks = [b for b in re.findall(r"\[[^\]\(]{3,80}\]", P) if b not in NOT_BLANKS]
rows.append({"item": "Blanks still to fill", "code_value": blanks, "prereg_says": [], "match": not blanks, "missing": blanks})

bad = [r for r in rows if not r["match"]]
for r in rows:
    print(("OK   " if r["match"] else "MISS ") + r["item"] + ": " + str(r["code_value"]) + ("" if r["match"] else "  <- not found: " + "; ".join(map(str, r["missing"]))))
print(f"\n{len(rows) - len(bad)} of {len(rows)} match")
Path("audit_prereg_vs_code.json").write_text(json.dumps(rows, indent=1))
sys.exit(1 if bad else 0)
