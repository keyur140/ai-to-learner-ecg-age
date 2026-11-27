# AI-to-learner knowledge transfer study (ECG age)
# Author: KEYUR SHAILESHKUMAR PATEL (ORCID: https://orcid.org/0009-0008-5261-043X)
# MSc Artificial Intelligence student, Department of Computer Science, Brunel University London, UK; Independent researcher
# Conducted independently; not an official Brunel University London project. Licence: MIT
"""Self-checks C1-C17. Each recomputes something a second, independent way.

A run's report shows every check with pass / FAIL and the detail. A FAIL never
stops the report: it is shown at the top, and the run's findings must not be
quoted until it is explained in ISSUES.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from . import stats as S
from .cues import tally
from .repro import sha256_file
from .sampling import draw_learner_sets, learner_rng


def _welch_manual(x, y):
    """Welch t-test written out by hand (independent of scipy.stats.ttest_ind)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    vx, vy, nx, ny = x.var(ddof=1), y.var(ddof=1), len(x), len(y)
    se = np.sqrt(vx / nx + vy / ny)
    t = (x.mean() - y.mean()) / se
    df = (vx / nx + vy / ny) ** 2 / ((vx / nx) ** 2 / (nx - 1) + (vy / ny) ** 2 / (ny - 1))
    return float(t), float(sps.t.sf(t, df))


def _holm_manual(p: dict) -> dict:
    from statsmodels.stats.multitest import multipletests

    keys = list(p)
    adj = multipletests([p[k] for k in keys], method="holm")[1]
    return dict(zip(keys, map(float, adj)))


def run_checks(run_dir: Path, cfg: dict, meta: pd.DataFrame, feats: pd.DataFrame, cues: dict, cohort: dict,
               teacher: dict, results: dict) -> list[dict]:
    """results: {exp_name: {"df": learners, "items": items or None, "analysis": dict, "gates": dict|None,
    "leaks": dict|None, "exp_code": str, "n_practice": int, "n_items": int, "extra": int}}"""
    C = []

    def add(cid, name, ok, detail):
        C.append({"id": cid, "check": name, "result": "pass" if ok else ("n/a" if ok is None else "FAIL"),
                  "detail": detail})

    add("C1", "No patient appears in two roles", cohort["patients_in_more_than_one_role"] == 0,
        f"{cohort['patients_in_more_than_one_role']} shared; {cohort['ecgs_set_aside_for_patient_overlap']} ECGs set aside")
    tr_ok = set(teacher["roles_used_for_training"]) <= {"teacher_train"} and \
        set(teacher["roles_used_for_early_stopping"]) <= {"teacher_val"}
    add("C2", "Teacher saw only its training and validation ECGs", tr_ok,
        f"train roles {teacher['roles_used_for_training']}, early-stopping roles {teacher['roles_used_for_early_stopping']}")
    rep = cues["report"]
    add("C3", "Cues built only from the cue fold, from the teacher's guesses", rep["roles_used"] == ["cue"],
        f"{rep['n_rows_used']} ECGs, roles {rep['roles_used']}")

    # C4/C5: re-draw learners from their seeds; check roles, balance, one ECG per patient, and match the item files
    role = meta["role"].values
    lab = meta["label"].values.astype(int)
    pat = meta["patient_id"].values
    bad, mism, checked = [], 0, 0
    for name, r in results.items():
        df = r["df"]
        if df is None or df.empty:
            continue
        sample = df.sample(min(20, len(df)), random_state=0)
        for _, row in sample.iterrows():
            ci = cfg["conditions"].index(row["condition"])
            rng = learner_rng(cfg["study"]["master_seed"], r["exp_code"], ci, int(row["learner"]))
            if r["exp_code"] == "nets":
                rng.integers(0, 2**31 - 1)  # the nets runner draws the torch seed first
            sets = draw_learner_sets(rng, meta, r["n_practice"], r["n_items"], extra_practice=r["extra"])
            checked += 1
            if not set(role[sets["practice"]]) <= {"teacher_train"}:
                bad.append("practice role")
            for t in ("pre", "post", "delayed"):
                if not set(role[sets[t]]) <= {"test"}:
                    bad.append(f"{t} role")
            if "transfer" in sets and not set(role[sets["transfer"]]) <= {"transfer"}:
                bad.append("transfer role")
            for t in ("practice", "pre", "post", "delayed", "transfer"):
                if t in sets:
                    s = sets[t]
                    if len(set(pat[s])) != len(s):
                        bad.append(f"{t} repeats a patient")
                    if t != "practice" and abs(lab[s].mean() - 0.5) > 0.5 / len(s) + 1e-9:
                        bad.append(f"{t} unbalanced")
            items = r.get("items")
            if items is not None:
                mine = items[(items.condition == row["condition"]) & (items.learner == row["learner"])]
                for t in ("pre", "post", "delayed", "transfer"):
                    if t in sets:
                        got = sorted(mine.loc[mine.test == t, "row"].tolist())
                        if got != sorted(sets[t].tolist()):
                            mism += 1
    add("C4", "Learner sets come from the right roles, balanced, one ECG per patient", not bad,
        f"{checked} learners re-drawn from their seeds; problems: {sorted(set(bad)) or 'none'}")
    any_items = any(r.get("items") is not None for r in results.values())
    add("C5", "Seeds reproduce exactly the test items each learner saw", (mism == 0) if any_items else None,
        f"{mism} mismatched test sets" if any_items else "no item-level files in this run")

    # C6: learner accuracies recomputed from item files
    maxdiff, n6 = 0.0, 0
    for r in results.values():
        items, df = r.get("items"), r["df"]
        if items is None or df is None or df.empty:
            continue
        primary = cfg["experiment_llm"].get("primary_score", "raw")

        def _recount(g):
            # written out by hand (sort and cut), not through stats.bias_corrected_credit
            p, y = g.p_older.values, g.label.values
            raw = float(np.mean((p > 0.5) == (y == 1)))
            srt = np.sort(p)
            n = len(srt)
            cut = srt[n // 2] if n % 2 else (srt[n // 2 - 1] + srt[n // 2]) / 2
            bc = (np.sum((p > cut) & (y == 1)) + np.sum((p < cut) & (y == 0)) + 0.5 * np.sum(p == cut)) / n
            return pd.Series({"raw": raw, "bc": float(bc)})

        acc = items.groupby(["condition", "learner", "test"])[["p_older", "label"]].apply(_recount)
        for _, row in df.iterrows():
            for t in ("pre", "post", "delayed", "transfer"):
                k = (row["condition"], row["learner"], t)
                if k not in acc.index:
                    continue
                want = {f"{t}_acc": acc.loc[k, "bc" if primary == "bias_corrected" else "raw"],
                        f"{t}_acc_raw": acc.loc[k, "raw"], f"{t}_acc_bc": acc.loc[k, "bc"]}
                for col, v in want.items():
                    if col in row and pd.notna(row[col]):
                        maxdiff = max(maxdiff, abs(v - row[col]))
                        n6 += 1
    add("C6", "Learner accuracies (primary score and raw score) recomputed from the item files", (maxdiff < 1e-9) if n6 else None,
        f"{n6} values, largest difference {maxdiff:.2e}" if n6 else "no item-level files")

    # C7/C8: H1 p-values and Holm adjustment recomputed independently
    d7, d8 = [], []
    for name, r in results.items():
        a, df = r["analysis"], r["df"]
        h1 = a.get("H1_primary", {})
        if "p" in h1:
            sub = df if not a.get("models_counted") else df[df.model.isin(a["models_counted"])]
            x = sub.loc[sub.condition == "cue_training", "post_acc"].values
            y = sub.loc[sub.condition == "feedback_only", "post_acc"].values
            _, p2 = _welch_manual(x, y)
            d7.append(abs(p2 - h1["p"]))
        sec = a.get("secondary", {})
        raw = {k: v["p"] for k, v in sec.items() if "p" in v}
        if raw:
            ref = _holm_manual(raw)
            d8.append(max(abs(ref[k] - sec[k]["p_holm"]) for k in raw))
    add("C7", "H1 p-values recomputed with a hand-written Welch test", (max(d7) < 1e-8) if d7 else None,
        f"{len(d7)} experiments, largest difference {max(d7) if d7 else 0:.1e}")
    add("C8", "Holm adjustment recomputed with statsmodels", (max(d8) < 1e-12) if d8 else None,
        f"{len(d8)} families, largest difference {max(d8) if d8 else 0:.1e}")

    # C9: rule-tally accuracy recomputed from the saved files
    saved = pd.read_csv(run_dir / "panel_features.csv", index_col=0)
    m = (meta["role"] == "test").values
    if m.sum():
        acc = float((tally(saved.values[m], cues["cues"], list(saved.columns)) == lab[m]).mean())
        add("C9", "Cue-rule accuracy recomputed from the saved files",
            abs(acc - rep["test"]["rule_tally_accuracy_true"]) < 1e-12, f"{acc:.4f} vs {rep['test']['rule_tally_accuracy_true']:.4f}")
    leaks = [r.get("leaks") for r in results.values() if r.get("leaks")]
    if leaks:
        n_case = sum(l["case_lines_flagged"] for l in leaks)
        n_ctx = sum(len(l["context_words_flagged"]) for l in leaks)
        add("C10", "Prompt leak scan: no IDs, sex, device, site or dates in learner text", n_case == 0 and n_ctx == 0,
            f"{n_case} flagged case lines, {n_ctx} flagged lesson texts")
    else:
        add("C10", "Prompt leak scan", None, "no language-model run")
    shares, worst = [], 0.0
    for r in results.values():
        df = r["df"]
        if df is not None and "older_is_A" in df:
            for _, g in df.groupby(["condition", "model"]):
                sh = g["older_is_A"].mean()
                shares.append(sh)
                worst = max(worst, abs(sh - 0.5) * len(g))  # in learners
    add("C11", "Answer letters balanced: in every group and model, 'older = A' for half the learners (within one)",
        (worst <= 0.5 + 1e-9) if shares else None,
        f"{len(shares)} group-model cells, largest imbalance {worst:.1f} learners" if shares else "no language-model run")
    g_ok, g_n = True, 0
    for r in results.values():
        for gm in (r.get("gates") or {}).values():
            g_n += 1
            ecl = cfg["experiment_llm"]
            g_ok &= gm["gate_pass"] == (gm["gate_accuracy"] >= ecl["gate"]["min_accuracy"])
            if "informative" in gm:  # which models count: gate, flat rule, and the collapse rule only if switched on
                flat = gm.get("post_flat_fraction", 0.0) > ecl.get("flat_model_fraction", 0.5)
                coll = gm.get("post_collapse_fraction", 0.0) > ecl["collapse_model_fraction"]
                g_ok &= gm["informative"] == (gm["gate_pass"] and not flat
                                              and not (ecl.get("collapse_excludes_model", True) and coll))
    add("C12", "Known-answer gate and which-models-count decisions follow the preregistered rules", g_ok if g_n else None,
        f"{g_n} model decisions")
    overlap = len({c["id"] for c in cues["cues"]} & {c["id"] for c in cues["scrambled"]})
    room = len(cues["feature_order"]) - len(cues["cues"]) >= len(cues["cues"])
    add("C13", "Scrambled lesson uses features the AI did not pick", overlap == 0 or not room,
        f"{overlap} shared features; scrambled-rule accuracy on test {rep['test'].get('scrambled_tally_accuracy_true', float('nan')):.3f}")
    w = run_dir / "teacher.pt"
    ens = teacher.get("ensemble_for_cues") or []
    ens_ok = all((run_dir / f"teacher_seed{e['k']}.pt").exists()
                 and sha256_file(run_dir / f"teacher_seed{e['k']}.pt") == e["weights_sha256"] for e in ens)
    add("C14", "Teacher weights match their recorded SHA-256", w.exists() and sha256_file(w) == teacher["weights_sha256"] and ens_ok,
        teacher["weights_sha256"][:16] + "..." + (f" (+{len(ens)} consensus teachers)" if ens else ""))
    con = rep.get("consensus")
    if con:
        sel = [c["id"] for c in cues["cues"]]
        recount = {f: sum(f in t["selected"] for t in con["per_teacher"]) for f in sel}
        ok = all(recount[f] == con["votes"][f] for f in sel) and (con["fallback"] or all(recount[f] >= con["min_votes"] for f in sel))
        add("C16", f"Cue consensus: every cue was chosen by at least {con['min_votes']} of {con['n_teachers']} teachers (recounted)",
            ok, "votes " + ", ".join(f"{f} {recount[f]}" for f in sel) + ("; FALLBACK used" if con["fallback"] else ""))
    pr = cohort.get("protected")
    if pr:
        touched = int(meta["strat_fold"].eq(pr["sealed_fold"]).sum() + meta["device"].eq(pr["sealed_device"]).sum())
        add("C15", "Protected pilot: no ECG from the sealed test fold or the sealed device was used", touched == 0,
            f"{pr['sealed_ecgs']} ECGs of {pr['sealed_patients']} patients sealed; {touched} found in the run")
    sc0 = cues["scrambled"][0] if cues["scrambled"] else {}
    if "placebo_in_band" in sc0:
        lo, hi = cfg["cues"].get("scrambled_band", [0.45, 0.55])
        cm = (meta["role"] == "cue").values
        from .cues import tally as _tally
        re_acc = float((_tally(feats.values[cm], cues["scrambled"], cues["feature_order"]) == meta["label"].values[cm]).mean())
        add("C17", f"Placebo is neutral: scrambled rules score {lo:.0%}-{hi:.0%} on the cue fold (recomputed)",
            lo <= re_acc <= hi, f"{re_acc:.3f} (draw {sc0['placebo_draw']}); on the learners' test pool "
            f"{rep['test'].get('scrambled_tally_accuracy_true', float('nan')):.3f}")
    (run_dir / "checks.json").write_text(json.dumps(C, indent=2))
    return C
