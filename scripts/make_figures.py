# Figures for the README and the paper. Every number is read from the run files; nothing is typed in.
# Colour-blind-safe Okabe-Ito palette. Usage, from the package root: python scripts/make_figures.py . docs/figures
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(ROOT / "src"))
from ai2human.exp_llm import rescore_from_items

OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
V3 = ROOT / "pilot" / "v0.3_run"
VAL = ROOT / "validation"
SW = ROOT / "pilot" / "followup" / "nets_practice"
COL = {"no_training": "#999999", "feedback_only": "#0072B2", "cue_training": "#D55E00", "scrambled_cues": "#009E73"}
NAME = {"no_training": "No training", "feedback_only": "Feedback only", "cue_training": "AI cues", "scrambled_cues": "Scrambled cues (placebo)"}
ORDER = list(COL)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
ec = {"collapse_share": 0.95, "primary_score": "bias_corrected", "flat_tie_share": 0.5}
numbers = {}

# 1. Experiment A pilot: bias-corrected accuracy by group and test, main and blinded
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for ax, (k, title) in zip(axes, [("llm", "Main run (real names)"), ("llm_blinded", "Blinded run (Measure 1-18)")]):
    df = rescore_from_items(pd.read_csv(V3 / f"learners_{k}.csv"), pd.read_csv(V3 / f"items_{k}.csv"), ec)
    tests = ["pre", "post", "delayed", "transfer"]
    x = np.arange(len(tests)); w = 0.2
    numbers[k] = {}
    for gi, c in enumerate(ORDER):
        d = df[df.condition == c]
        m = [100 * d[f"{t}_acc_bc"].mean() for t in tests]
        se = [100 * d[f"{t}_acc_bc"].std(ddof=1) / np.sqrt(len(d)) for t in tests]
        numbers[k][c] = dict(zip(tests, [round(v, 1) for v in m]))
        ax.bar(x + (gi - 1.5) * w, m, w, yerr=[1.96 * s for s in se], color=COL[c], label=NAME[c], capsize=2, error_kw={"lw": 0.8})
    ax.axhline(50, color="black", lw=0.8, ls="--"); ax.set_xticks(x, ["Before", "After", "After interference", "New device"])
    ax.set_title(title); ax.set_ylim(30, 80)
axes[0].set_ylabel("Bias-corrected accuracy (%)\n50% = chance")
h, l = axes[1].get_legend_handles_labels(); fig.legend(h, l, frameon=False, fontsize=9, loc="lower center", ncol=4)
fig.suptitle("Pilot, Experiment A: Qwen2.5-3B learners, 30 per group, folds 1-9 (exploratory)", fontsize=11)
fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(OUT / "pilot_expA.png"); plt.close(fig)

# 2. Experiment B practice-size sweep
fig, ax = plt.subplots(figsize=(6, 3.8))
ns = [4, 8, 16, 32, 64, 128]; numbers["nets_sweep"] = {}
for c in ORDER:
    m, lo, hi = [], [], []
    for n in ns:
        v = pd.read_csv(SW / f"n{n:03d}/learners_nets.csv"); v = 100 * v[v.condition == c].post_acc
        m.append(v.mean()); h = 1.96 * v.std(ddof=1) / np.sqrt(len(v)); lo.append(v.mean() - h); hi.append(v.mean() + h)
    numbers["nets_sweep"][c] = dict(zip(ns, [round(v, 1) for v in m]))
    ax.plot(ns, m, "-o", color=COL[c], label=NAME[c], ms=4); ax.fill_between(ns, lo, hi, color=COL[c], alpha=0.12)
ax.set_xscale("log", base=2); ax.set_xticks(ns, [str(n) for n in ns]); ax.axhline(50, color="black", lw=0.8, ls="--")
ax.set_xlabel("Labelled practice ECGs per net"); ax.set_ylabel("Accuracy after training (%)")
ax.set_title("Pilot, Experiment B: small nets, 30 per group (exploratory)", fontsize=11); ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "pilot_expB_sweep.png"); plt.close(fig)

# 3. Benchmarks on the pilot test fold: teacher, baseline, cue rules, placebo
tm = json.loads((V3 / "teacher_metrics.json").read_text()); cues = json.loads((V3 / "cues.json").read_text())
rep = cues["report"]["test"]
bench = [("AI teacher (1D ResNet, raw ECG)", 100 * tm["test"]["teacher_accuracy"]),
         ("Logistic regression on the 18 measurements", 100 * tm["test"]["features_baseline_accuracy"]),
         ("Surrogate model on the 6 cues", 100 * rep["surrogate_accuracy_true"]),
         ("6 AI cue rules as a simple tally", 100 * rep["rule_tally_accuracy_true"]),
         ("Placebo rules as a tally", 100 * rep["scrambled_tally_accuracy_true"])]
numbers["benchmarks_accuracy"] = {k: round(v, 1) for k, v in bench}
numbers["teacher_auroc"] = round(tm["test"]["teacher_auroc"], 3); numbers["baseline_auroc"] = round(tm["test"]["features_baseline_auroc"], 3)
fig, ax = plt.subplots(figsize=(8, 3.4))
cols = ["#0072B2", "#56B4E9", "#E69F00", "#D55E00", "#009E73"]
ax.barh(range(len(bench))[::-1], [v for _, v in bench], color=cols)
for i, (lab, v) in enumerate(bench):
    ax.text(v + 1.5, len(bench) - 1 - i, f"{v:.1f}%", va="center", fontsize=9, bbox=dict(facecolor="white", edgecolor="none", pad=1))
ax.set_yticks(range(len(bench))[::-1], [b for b, _ in bench]); ax.set_xlim(0, 100); ax.axvline(50, color="black", lw=0.8, ls="--")
ax.set_xlabel("Accuracy on the pilot test fold (fold 9), %"); ax.set_title("Benchmarks: telling younger from older adults\n(dashed line = 50%, chance)", fontsize=11)
fig.tight_layout(); fig.savefig(OUT / "benchmarks.png"); plt.close(fig)

# 4. Validation: power of the H1 rule
sim = json.loads((VAL / "validation_summary.json").read_text())["simulation"]["power_curve"]
fig, ax = plt.subplots(figsize=(6, 3.4))
x = [100 * r["true_diff"] for r in sim]
ax.plot(x, [100 * r["p_below_alpha"] for r in sim], "-o", color="#56B4E9", ms=4, label="Welch test alone (p < 0.05)")
ax.plot(x, [100 * r["H1_rule"] for r in sim], "-o", color="#D55E00", ms=4, label="H1 rule (p < 0.05 and at least 3 points)")
ax.axvline(3, color="black", lw=0.8, ls=":"); ax.set_xlabel("True difference, AI cues minus feedback only (points)")
ax.set_ylabel("Chance of 'yes' (%)"); ax.set_title("Synthetic validation: 4,000 simulated studies per point", fontsize=11); ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "validation_power.png"); plt.close(fig)
numbers["power_curve"] = sim
(OUT / "figure_numbers.json").write_text(json.dumps(numbers, indent=1))
print("figures:", sorted(p.name for p in OUT.glob("*.png")))
