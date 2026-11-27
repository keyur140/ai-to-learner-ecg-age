# AI-to-learner knowledge transfer study (ECG age)
# Author: KEYUR SHAILESHKUMAR PATEL (ORCID: https://orcid.org/0009-0008-5261-043X)
# MSc Artificial Intelligence student, Department of Computer Science, Brunel University London, UK; Independent researcher
# Conducted independently; not an official Brunel University London project. Licence: MIT
"""Synthetic validation, run BEFORE preregistration (no real ECGs are touched).

Six fake worlds (five planned, one added after the dry run), each built so that the right answer is known in advance, are run
through the full pipeline. Every world states, before it runs, what the pipeline
must report; the validation report shows expected vs observed for each.

  W1 informative   ECG-like data where age leaves a trace; mock learners that apply
                   the rules they read. Expected: H1 and H7 supported, all checks pass.
  W1b strong       added after the dry run: learners lean on written rules, so the
                   planted gain is large. Expected: H1 (both runs) and H7 supported.
  W2 null learners learners that follow an explicit rule but learn nothing from
                   lessons. Expected: gate passed, H1 and H7 NOT supported.
  W3 prior knowl.  learners that already 'know' the cues when names are shown.
                   Expected: probe flags it, pre-test far above chance in the main run
                   (precondition fails); blinded run still shows the lesson effect.
  W4 broken models one good model, one that collapses to 'older', one random.
                   Expected: random fails the gate, collapse is flagged, only the good
                   model counts.
  W5 no signal     age leaves no trace in the data. Expected: teacher near chance
                   (precondition fails), H1 not supported in either experiment.

Plus simulations of the preregistered statistics at the confirmatory size
(1,002 learners per group): false-positive rate, power curve, Holm family-wise error,
gate pass probability, false collapse flags, and calibration of the item-level
two-way cluster test. Plus a reproducibility re-run and a tamper test.
"""
from __future__ import annotations

import copy
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from . import stats as S
from .analysis import twoway_cluster_lpm
from .repro import get_logger, sha256_file, sha256_tree

STUDENTS = ["mock-student-1", "mock-student-2", "mock-student-3"]
WORLDS = {
    "W1": {"name": "Informative world", "models": STUDENTS, "signal": 1.0, "nets": True,
           "story": "Age leaves a trace in the ECG and the measurements. Learners read and apply the rules they are given, "
                    "and learn a little from practice cases."},
    "W1b": {"name": "Strong teaching world", "models": ["mock-strong-1", "mock-strong-2", "mock-strong-3"], "signal": 1.0,
            "nets": False, "added": "after the dry run",
            "story": "Same data as W1, but the learners lean much more on written rules than on practice cases, so a "
                     "good lesson should give a large gain. Added after the dry run showed W1's planted gain was below 3 points."},
    "W2": {"name": "Null learners", "models": ["mock-null-1", "mock-null-2", "mock-null-3"], "signal": 1.0, "nets": False,
           "story": "Same data. Learners can follow one explicit rule (so they pass the gate) but learn nothing from lessons "
                    "or practice: their answers are noise."},
    "W3": {"name": "Prior knowledge", "models": ["mock-memorizer-1", "mock-memorizer-2", "mock-memorizer-3"], "signal": 1.0,
           "nets": False,
           "story": "Same data. Learners already 'know' the AI's cues whenever the real measurement names are shown, with or "
                    "without a lesson, like a model that memorised ECG textbooks."},
    "W4": {"name": "Broken models", "models": ["mock-student-1", "mock-collapse-1", "mock-random-1"], "signal": 1.0, "nets": False,
           "story": "One good learner model, one that follows an explicit rule but otherwise always says 'older', and one that "
                    "answers at random."},
    "W5": {"name": "No signal", "models": STUDENTS, "signal": 0.0, "nets": True,
           "story": "Age leaves no trace at all in the ECG or the measurements. There is nothing for the AI to find."},
}


def world_cfg(cfg_raw: dict, w: dict, quick: bool) -> dict:
    c = copy.deepcopy(cfg_raw)
    c["_smoke_n"] = 3000 if quick else 6000
    c["_synthetic_signal"] = w["signal"]
    c["teacher"].update(epochs=3 if quick else 8, channels=16 if quick else 32, blocks=2 if quick else 3)
    c["cues"]["stability_bootstraps"] = 20 if quick else 100
    c["saliency"].update(n_ecgs=32 if quick else 64, ig_steps=8 if quick else 16)
    n_llm = 6 if quick else c["experiment_llm"]["n_per_condition"]  # full confirmatory size for language-model worlds
    c["experiment_llm"].update(backend="mock", models=w["models"], n_per_condition=n_llm, blinded_n_per_condition=n_llm)
    if quick:
        c["experiment_llm"].update(n_items_per_test=20, n_practice=8)
        c["experiment_llm"]["gate"] = {**c["experiment_llm"]["gate"], "n_items": 20}
    c["experiment_nets"].update(enabled=w["nets"], n_per_condition=6 if quick else 30, steps=30 if quick else 150,
                                distractor_steps=30 if quick else 100, n_items_per_test=60, distractor_n=64 if quick else 128)
    c["analysis"]["bootstrap"] = 500 if quick else 2000
    return c


# ----------------------------------------------------------------- expectations
def _h(ctx, exp, key="H1_primary"):
    a = ctx.get(f"analysis_{exp}") or {}
    return a.get(key) if key == "H1_primary" else a.get("secondary", {}).get(key, {})


def _all_checks_pass(ctx):
    return all(c["result"] != "FAIL" for c in ctx.get("checks", []))


def expectations(wid: str, ctx: dict) -> list[tuple[str, bool, str]]:
    """(what must happen, did it happen, what we saw)."""
    E = []
    chk = ctx.get("checks", [])
    E.append(("No self-check fails", _all_checks_pass(ctx),
              f"{sum(c['result'] == 'pass' for c in chk)} pass, {sum(c['result'] == 'FAIL' for c in chk)} fail, "
              f"{sum(c['result'] == 'n/a' for c in chk)} n/a"))
    a = ctx.get("analysis_llm") or {}
    ab = ctx.get("analysis_llm_blinded") or {}
    h1, h1b = a.get("H1_primary", {}), ab.get("H1_primary", {})
    h7 = a.get("secondary", {}).get("H7", {})
    pre = a.get("preconditions", {})
    t_auc = ctx["teacher"].get("test", {}).get("teacher_auroc", float("nan"))

    def d(h):
        if "p" not in h:
            return h.get("skipped", "n/a")
        tag = ""
        if "meaningful" in h and h["p"] < 0.05 and not h["meaningful"]:
            tag = " (significant, but below the 3-point bar)"
        return f"diff {100 * h['diff']:+.1f} pts, p = {h['p']:.3g}{tag}"

    if wid == "W1b":
        E += [("H1 supported (main run)", bool(h1.get("supported")), d(h1)),
              ("H1 supported (blinded run)", bool(h1b.get("supported")), d(h1b)),
              ("H7 supported: real cues beat scrambled cues", bool(h7.get("supported")), d(h7))]
    elif wid == "W1":
        E += [("AI teacher is an expert (AUROC >= 0.80)", t_auc >= 0.80, f"AUROC {t_auc:.3f}"),
              ("H1 supported (main run)", bool(h1.get("supported")), d(h1)),
              ("H1 supported (blinded run)", bool(h1b.get("supported")), d(h1b)),
              ("H7 supported: real cues beat scrambled cues", bool(h7.get("supported")), d(h7)),
              ("Pre-test near chance (new skill)", bool(pre.get("new_skill (pretest mean <= 0.65)")), f"pre-test mean {pre.get('pretest_mean', float('nan')):.3f}"),
              ("All three models count", len(a.get("models_counted", [])) == 3, f"{a.get('models_counted')}")]
    elif wid == "W2":
        E += [("All models pass the known-answer gate", all(g["gate_pass"] for g in a.get("models", {}).values()),
               ", ".join(f"{g['gate_accuracy']:.2f}" for g in a.get("models", {}).values())),
              ("H1 NOT supported (main run)", not h1.get("supported"), d(h1)),
              ("H1 NOT supported (blinded run)", not h1b.get("supported"), d(h1b)),
              ("H7 NOT supported", not h7.get("supported"), d(h7))]
    elif wid == "W3":
        cues = ctx["cues"]["cues"]
        known = []
        for g in a.get("models", {}).values():
            kn = g.get("probe_known_direction", {})
            known.append(sum(kn.get(c["id"], 0) == (1 if c["older_is_higher"] else -1) for c in cues))
        E += [("Probe shows the models already know most cue directions", bool(known) and min(known) >= len(cues) / 2 + 0.5,
               f"{known} of {len(cues)}"),
              ("Main run: pre-test far above chance, so the 'new skill' precondition fails",
               pre.get("new_skill (pretest mean <= 0.65)") is False, f"pre-test mean {pre.get('pretest_mean', float('nan')):.3f}"),
              ("Blinded run: pre-test back near chance", bool(ab.get("preconditions", {}).get("new_skill (pretest mean <= 0.65)")),
               f"pre-test mean {ab.get('preconditions', {}).get('pretest_mean', float('nan')):.3f}"),
              ("Blinded run still shows the lesson effect (H1 supported)", bool(h1b.get("supported")), d(h1b))]
    elif wid == "W4":
        g = a.get("models", {})
        E += [("Random model fails the gate", not g.get("mock-random-1", {}).get("gate_pass", True),
               f"gate accuracy {g.get('mock-random-1', {}).get('gate_accuracy', float('nan')):.2f}"),
              ("Collapsing model passes the gate but is flagged (collapsed, and flat: its answers cannot be ranked)",
               g.get("mock-collapse-1", {}).get("gate_pass") is True and g.get("mock-collapse-1", {}).get("collapsed_model") is True
               and g.get("mock-collapse-1", {}).get("informative") is False,
               f"gate {g.get('mock-collapse-1', {}).get('gate_accuracy', float('nan')):.2f}, "
               f"collapsed learners {100 * g.get('mock-collapse-1', {}).get('post_collapse_fraction', float('nan')):.0f}%, "
               f"flat learners {100 * g.get('mock-collapse-1', {}).get('post_flat_fraction', float('nan')):.0f}%"),
              ("Only the good model counts for H1", a.get("models_counted") == ["mock-student-1"], f"{a.get('models_counted')}"),
              ("H1 still testable on the good model", "p" in h1, d(h1)),
              ("Sensitivity analysis with all models is reported",
               "p" in a.get("robustness", {}).get("H1_all_models_including_uninformative", {}),
               d(a.get("robustness", {}).get("H1_all_models_including_uninformative", {})))]
    elif wid == "W5":
        hn = (ctx.get("analysis_nets") or {}).get("H1_primary", {})
        E += [("Teacher near chance, so the 'expert' precondition fails", t_auc < 0.80, f"AUROC {t_auc:.3f}"),
              ("H1 NOT supported (language-model learners)", not h1.get("supported"), d(h1)),
              ("H1 NOT supported (neural-net learners)", not hn.get("supported"), d(hn))]
    return E


# ----------------------------------------------------------------- statistics simulations
def _learner_sd(ctx_w1: dict | None) -> tuple[float, float]:
    """Mean and SD of post-test accuracy of feedback-only learners in W1 (fallback 0.70 / 0.08)."""
    try:
        d = pd.DataFrame(ctx_w1["analysis_llm"]["descriptives"])
        r = d[(d.condition == "feedback_only") & (d.test == "post")].iloc[0]
        return float(r.mean_acc), float(r.sd)
    except Exception:
        return 0.70, 0.08


def simulate_stats(cfg: dict, base_mean: float, sd: float, n: int, n_items: int, quick: bool, log) -> dict:
    rng = np.random.default_rng(cfg["study"]["master_seed"] + 99)
    alpha, sesoi = cfg["analysis"]["alpha"], cfg["analysis"]["min_meaningful_diff"]
    reps = 400 if quick else 4000
    # learner accuracy = item-sampling noise + between-learner spread; total SD matched to `sd`
    p_item = base_mean
    binom_var = p_item * (1 - p_item) / n_items
    tau = float(np.sqrt(max(sd ** 2 - binom_var, 1e-6)))

    def draw(mu, size):
        p = np.clip(rng.normal(mu, tau, size), 0.01, 0.99)
        return rng.binomial(n_items, p) / n_items

    out = {"inputs": {"n_per_group": n, "items_per_test": n_items, "feedback_only_mean": base_mean, "learner_sd": sd,
                      "between_learner_sd": tau, "replications": reps, "alpha": alpha, "min_meaningful_diff": sesoi}}
    diffs = [0.0, 0.01, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05, 0.075, 0.10]
    curve = []
    for dlt in diffs:
        sig = rule = 0
        for _ in range(reps):
            a, b = draw(base_mean + dlt, n), draw(base_mean, n)
            p = sps.ttest_ind(a, b, equal_var=False, alternative="greater").pvalue
            sig += p < alpha
            rule += (p < alpha) and (a.mean() - b.mean() >= sesoi)
        curve.append({"true_diff": dlt, "p_below_alpha": sig / reps, "H1_rule": rule / reps})
    out["power_curve"] = curve
    out["false_positive_rate_H1"] = curve[0]["H1_rule"]
    out["false_positive_rate_welch_only"] = curve[0]["p_below_alpha"]
    log(f"sim: H1 false-positive rate {curve[0]['H1_rule']:.4f}; power at +3 pts {curve[4]['H1_rule']:.3f}, +5 pts {curve[7]['H1_rule']:.3f}")

    # Holm family (H2-H7) under the global null: family-wise error rate
    fam = 0
    reps_h = reps // 2
    for _ in range(reps_h):
        g = {c: draw(base_mean, n) for c in ("no", "fb", "cue", "scr")}
        pre = {c: draw(0.5, n) for c in ("fb", "cue")}
        dl = {c: draw(base_mean, n) for c in ("fb", "cue")}
        tr = {c: draw(base_mean, n) for c in ("fb", "cue")}
        ps = {"H2": sps.ttest_ind(g["cue"], g["no"], equal_var=False, alternative="greater").pvalue,
              "H3": sps.ttest_ind(g["fb"], g["no"], equal_var=False, alternative="greater").pvalue,
              "H4": sps.ttest_ind(dl["cue"], dl["fb"], equal_var=False, alternative="greater").pvalue,
              "H5": sps.ttest_ind(tr["cue"], tr["fb"], equal_var=False, alternative="greater").pvalue,
              "H6": sps.ttest_ind(g["cue"] - pre["cue"], g["fb"] - pre["fb"], equal_var=False, alternative="greater").pvalue,
              "H7": sps.ttest_ind(g["cue"], g["scr"], equal_var=False, alternative="greater").pvalue}
        fam += min(S.holm(ps).values()) < alpha
    out["holm_familywise_error"] = fam / reps_h

    # Known-answer gate: chance a model with true rule-following accuracy q passes (>= 85% of 40)
    g = cfg["experiment_llm"]["gate"]
    k = int(np.ceil(g["min_accuracy"] * g["n_items"]))
    out["gate_pass_probability"] = {f"{q:.2f}": float(sps.binom.sf(k - 1, g["n_items"], q)) for q in
                                    (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)}
    # Collapse flag: chance a learner with a given lean toward 'older' is flagged on a 40-item test
    cs = cfg["experiment_llm"]["collapse_share"]
    kk = int(np.ceil(cs * n_items))
    out["collapse_flag_probability"] = {f"{q:.2f}": float(sps.binom.sf(kk - 1, n_items, q) + sps.binom.cdf(n_items - kk, n_items, q))
                                        for q in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)}

    # Item-level two-way cluster test: calibration under the null when learners share test ECGs
    reps_c = 60 if quick else 300
    n_c = min(n, 300)
    pool = 400
    rej_w = rej_c = 0
    for _ in range(reps_c):
        ease = rng.normal(0, 1.0, pool)  # some ECGs are easier than others
        rows = []
        accs = {"a": [], "b": []}
        for grp in ("a", "b"):
            for L in range(n_c):
                skill = rng.normal(0.85, 0.35)
                its = rng.choice(pool, n_items, replace=False)
                pc = 1 / (1 + np.exp(-(skill + ease[its])))
                cor = rng.random(n_items) < pc
                accs[grp].append(cor.mean())
                rows.append(pd.DataFrame({"test": "post", "condition": grp, "learner": L, "row": its,
                                          "label": 1, "p_older": np.where(cor, 0.9, 0.1)}))
        items = pd.concat(rows, ignore_index=True)
        rej_w += sps.ttest_ind(accs["a"], accs["b"], equal_var=False, alternative="greater").pvalue < alpha
        rej_c += twoway_cluster_lpm(items, "a", "b")["p_one_sided"] < alpha
    out["cluster_calibration"] = {"replications": reps_c, "learners_per_group": n_c, "ecg_pool": pool,
                                  "welch_false_positive": rej_w / reps_c, "twoway_cluster_false_positive": rej_c / reps_c}
    out["analytic_power"] = {f"d={d}": S.power_two_sample(n, d, alpha) for d in (0.05, 0.08, 0.11, 0.13, 0.2)}
    return out


# ----------------------------------------------------------------- reproducibility + tamper
def repro_and_tamper(cfg_raw: dict, out: Path, execute, log) -> dict:
    from .report import INFO_KEYS  # noqa: F401  (ensures the same code path as a run)

    res = {}
    runs = []
    for k in (1, 2):
        rd = out / f"R{k}_repeat"
        rd.mkdir(exist_ok=True)
        lg = get_logger(rd)
        c = world_cfg(cfg_raw, WORLDS["W1"], quick=True)
        execute(c, cfg_raw, "validate", rd, "n/a (synthetic)", "SYNTHETIC", [], lg, write_reports=False)
        runs.append(rd)
    skip = {"run.log", "manifest.json", "SHA256SUMS.txt", "run_info.json"}
    h1 = {k: v for k, v in sha256_tree(runs[0]).items() if k not in skip}
    h2 = {k: v for k, v in sha256_tree(runs[1]).items() if k not in skip}
    diff = sorted(k for k in set(h1) | set(h2) if h1.get(k) != h2.get(k))
    res["repeat_identical"] = not diff
    res["repeat_files_compared"] = len(h1)
    res["repeat_files_different"] = diff
    # tamper: change one byte of a results file and re-verify
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("run_study", Path(__file__).resolve().parents[2] / "run_study.py")
    rs = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("run_study_v", rs)
    spec.loader.exec_module(rs)
    before = rs.verify_run(runs[1])
    f = runs[1] / "learners_llm.csv"
    data = f.read_bytes()
    f.write_bytes(data.replace(b"0.", b"1.", 1))
    after = rs.verify_run(runs[1])
    f.write_bytes(data)
    restored = rs.verify_run(runs[1])
    res["tamper"] = {"before": before["ok"], "after_one_byte_change": after["ok"], "changed_files_found": after["changed"],
                     "after_restore": restored["ok"]}
    log(f"repro: identical={res['repeat_identical']} ({len(h1)} files); tamper detected={not after['ok']}")
    return res


# ----------------------------------------------------------------- driver
def run_validation(cfg_raw: dict, out: Path, log, execute, quick: bool = False, worlds: list[str] | None = None) -> dict:
    t0 = time.time()
    summary = {"quick": quick, "worlds": {}, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    ctxs = {}
    for wid in (worlds or list(WORLDS)):
        w = WORLDS[wid]
        rd = out / f"{wid}_{w['name'].lower().replace(' ', '_')}"
        rd.mkdir(exist_ok=True)
        lg = get_logger(rd)
        lg.info(f"=== {wid} {w['name']}: {w['story']}")
        c = world_cfg(cfg_raw, w, quick)
        t = time.time()
        ctx = execute(c, cfg_raw, "validate", rd, "n/a (synthetic validation)", "SYNTHETIC", [], lg, write_reports=True)
        E = expectations(wid, ctx)
        ctxs[wid] = ctx
        summary["worlds"][wid] = {"name": w["name"], "story": w["story"], "added": w.get("added"), "models": w["models"], "signal": w["signal"],
                                  "nets": w["nets"], "seconds": round(time.time() - t, 1), "dir": rd.name,
                                  "expectations": [{"what": e[0], "met": bool(e[1]), "observed": e[2]} for e in E],
                                  "teacher_test_auroc": ctx["teacher"].get("test", {}).get("teacher_auroc"),
                                  "cues": [c_["label"] for c_ in ctx["cues"]["cues"]],
                                  "rule_tally_accuracy_test": ctx["cues"]["report"].get("test", {}).get("rule_tally_accuracy_true"),
                                  "analyses": {e: ctx[f"analysis_{e}"] for e in ("llm", "llm_blinded", "nets") if ctx.get(f"analysis_{e}")},
                                  "checks": ctx.get("checks")}
        log = get_logger(out)
        log.info(f"{wid}: {sum(e[1] for e in E)}/{len(E)} expectations met")
        (out / "validation_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    log = get_logger(out)
    base, sd = _learner_sd(ctxs.get("W1"))
    ec = cfg_raw["experiment_llm"]
    summary["simulation"] = simulate_stats(cfg_raw, base, sd, ec["n_per_condition"], ec["n_items_per_test"], quick, log.info)
    summary["reproducibility"] = repro_and_tamper(cfg_raw, out, execute, get_logger(out).info)
    log = get_logger(out)
    summary["seconds_total"] = round(time.time() - t0, 1)
    (out / "validation_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    from .validation_report import write_validation_report

    write_validation_report(out, summary, cfg_raw)
    # fingerprints of everything
    sums = sha256_tree(out)
    with open(out / "SHA256SUMS.txt", "w") as f:
        for k, v in sums.items():
            if k != "SHA256SUMS.txt":
                f.write(f"{v}  {k}\n")
    log.info(f"validation done in {summary['seconds_total']} s -> {out}")
    return summary
