# AI-to-learner knowledge transfer study (ECG age)
# Author: KEYUR SHAILESHKUMAR PATEL (ORCID: https://orcid.org/0009-0008-5261-043X)
# MSc Artificial Intelligence student, Department of Computer Science, Brunel University London, UK; Independent researcher
# Conducted independently; not an official Brunel University London project. Licence: MIT
"""The study pipeline, shared by the confirmatory run, the smoke test and the synthetic validation."""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from .analysis import analyse, save_analysis
from .checks import run_checks
from .cohort import build_cohort
from .cues import extract_cues
from .saliency import run_saliency
from .teacher import pick_device, train_teacher


def run_pipeline(cfg: dict, run_dir: Path, synthetic: bool, ctx: dict, man, log=print) -> dict:
    t = time.time()
    meta, feats, sigs, cohort = build_cohort(cfg, synthetic, run_dir, log)
    man.record_stage("cohort", t, n=cohort["n_cohort"])

    t = time.time()
    model, X, tpreds, tmetrics = train_teacher(cfg, meta, feats, sigs, run_dir, log)
    man.record_stage("teacher", t, test_auroc=tmetrics.get("test", {}).get("teacher_auroc"))

    t = time.time()
    cues = extract_cues(cfg, meta, feats, tpreds, run_dir, log)
    man.record_stage("cues", t, selected=[c["id"] for c in cues["cues"]])

    t = time.time()
    device = pick_device(cfg["experiment_nets"].get("device", "auto"))
    sal = run_saliency(cfg, model, X, meta, cues, run_dir, device, log)
    man.record_stage("saliency", t)
    del model

    ctx.update(cohort=cohort, teacher=tmetrics, cues=cues, saliency=sal)
    results = {}
    if cfg["experiment_nets"]["enabled"]:
        from .exp_nets import run_experiment_nets

        t = time.time()
        dfn = run_experiment_nets(cfg, X, meta, feats, cues, run_dir, device, log)
        man.record_stage("experiment_nets", t, n=len(dfn))
        a = analyse(dfn, cfg, tmetrics, cues["report"], "nets")
        save_analysis(a, run_dir / "analysis_nets.json")
        ctx["analysis_nets"] = a
        en = cfg["experiment_nets"]
        results["nets"] = {"df": dfn, "items": None, "analysis": a, "gates": None, "leaks": None,
                           "exp_code": "nets", "n_practice": en["n_practice"], "n_items": en["n_items_per_test"], "extra": 0}
    del X, sigs

    if cfg["experiment_llm"]["enabled"]:
        from .exp_llm import run_experiment_llm

        el = cfg["experiment_llm"]
        n_proto = 2 * cfg["cues"]["n_prototypes_per_class"]
        runs = [("llm", False)] + ([("llm_blinded", True)] if el.get("blinded_run") else [])
        for name, blinded in runs:
            t = time.time()
            df, gates, leaks = run_experiment_llm(cfg, meta, feats, cues, run_dir, log, blinded=blinded)
            man.record_stage(f"experiment_{name}", t, n=len(df))
            items = pd.read_csv(run_dir / f"items_{name}.csv")
            a = analyse(df, cfg, tmetrics, cues["report"], name, items=items, gates=gates)
            save_analysis(a, run_dir / f"analysis_{name}.json")
            ctx[f"analysis_{name}"] = a
            results[name] = {"df": df, "items": items, "analysis": a, "gates": gates, "leaks": leaks,
                             "exp_code": "llm_blind" if blinded else "llm", "n_practice": el["n_practice"],
                             "n_items": el["n_items_per_test"], "extra": n_proto}

    t = time.time()
    ctx["checks"] = run_checks(run_dir, cfg, meta, feats, cues, cohort, tmetrics, results)
    man.record_stage("self_checks", t, passed=sum(c["result"] == "pass" for c in ctx["checks"]))
    return ctx
