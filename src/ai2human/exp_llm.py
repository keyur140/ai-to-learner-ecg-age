# AI-to-learner knowledge transfer study (ECG age)
# Author: KEYUR SHAILESHKUMAR PATEL (ORCID: https://orcid.org/0009-0008-5261-043X)
# MSc Artificial Intelligence student, Department of Computer Science, Brunel University London, UK; Independent researcher
# Conducted independently; not an official Brunel University London project. Licence: MIT
"""Experiment A: open-source language-model learners taught in words.

Every learner (one 'participant') gets its own practice ECGs, test ECGs and case order,
drawn from its own seed. The answer letters (older = A or B) alternate, so they are
exactly balanced within each model and group. Groups:
  no_training    : neutral, length-matched lesson about what the measurements are
  feedback_only  : neutral lesson + practice cases with correct answers
  cue_training   : AI-derived cue lesson + the AI's most confident ("clearest") cases
                   + the same practice cases
  scrambled_cues : placebo lesson in the same format (features the AI did not pick,
                   random directions) + the same number of "clearest" cases, drawn at
                   random, + the same practice cases
feedback_only gets as many extra ordinary practice cases as the others get "clearest"
cases, so every taught group sees the same number of examples.
Tests: pre (intro only), post, transfer (held-out device), delayed (after an
unrelated interference task in the context).

Two runs: 'main' (real measurement names) and 'blinded' (names hidden as
"Measure 1".."Measure 18", no units), each with its own learner seeds.
Before a model's learners count, it must pass the known-answer gate; every model
also answers a prior-knowledge probe. Every prompt is leak-scanned.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import prompts as PR
from . import stats as S
from .llm_backends import make_backend
from .sampling import draw_learner_sets, learner_rng

TESTS = ("pre", "post", "delayed", "transfer")


def test_scores(p_older, y, ec: dict) -> dict:
    """All per-learner scores for one test, from the item probabilities alone.

    acc_raw: share correct with the answer taken as 'older' when P(older) > 0.5 (the v0.2 score).
    acc_bc : bias-corrected accuracy (decision D12): items split at the learner's own median P(older).
    acc    : the primary score named in experiment_llm.primary_score; every hypothesis uses this one."""
    p_older, y = np.asarray(p_older, float), np.asarray(y).astype(int)
    older_share = float((p_older > 0.5).mean())
    out = {"acc_raw": float(((p_older > 0.5) == (y == 1)).mean()),
           "acc_bc": float(S.bias_corrected_credit(p_older, y).mean()),
           "auroc": S.auroc(p_older, y),
           "p_correct": float(np.where(y == 1, p_older, 1 - p_older).mean()),
           "p_older_mean": float(p_older.mean()),  # response bias check
           "majority_share": max(older_share, 1 - older_share),
           "tie_share": S.median_tie_share(p_older)}
    out["collapsed"] = bool(out["majority_share"] >= ec["collapse_share"])
    out["flat"] = bool(out["tie_share"] > ec.get("flat_tie_share", 0.5))
    out["acc"] = out["acc_bc"] if ec.get("primary_score", "raw") == "bias_corrected" else out["acc_raw"]
    return out


def rescore_from_items(df: pd.DataFrame, items: pd.DataFrame, ec: dict) -> pd.DataFrame:
    """Recompute every score column of the learner table from the item file (one source of truth).
    Also brings learner rows saved by an older version up to date after --resume."""
    df = df.copy()
    pos = {(r.condition, r.learner): k for k, r in zip(df.index, df.itertuples())}
    for (cond, learner, test), g in items.groupby(["condition", "learner", "test"], sort=False):
        k = pos.get((cond, learner))
        if k is None:
            continue
        for name, v in test_scores(g.p_older.values, g.label.values, ec).items():
            df.loc[k, f"{test}_{name}"] = v
    for t in TESTS:
        for name in ("collapsed", "flat"):
            if f"{t}_{name}" in df:
                df[f"{t}_{name}"] = df[f"{t}_{name}"].astype(bool)
    return df


def _score(backend, context, idxs, case_txt, labels, letters, test, rec, items, learner_key, ec):
    qs = [PR.test_question(case_txt[i], letters) for i in idxs]
    pa = np.array(backend.p_a(context, qs))
    a_is_older = letters.letter[1] == "A"
    p_older = pa if a_is_older else 1 - pa
    y = labels[idxs]
    pc = np.where(y == 1, p_older, 1 - p_older)
    for name, v in test_scores(p_older, y, ec).items():
        rec[f"{test}_{name}"] = v
    for i, p, c in zip(idxs, p_older, pc):
        items.append({**learner_key, "test": test, "row": int(i), "label": int(labels[i]),
                      "p_older": float(p), "p_correct": float(c)})


class Materials:
    """Everything a run shows to learners, built once (and leak-scanned once)."""

    def __init__(self, cfg, meta, feats, cues, out_dir: Path, blinded: bool):
        panel = cfg["panel"]
        self.blind = cues["blinded_labels"] if blinded else None
        self.case_txt = [PR.format_case(r, panel, self.blind) for r in feats.values]
        tag = "_blinded" if blinded else ""
        self.cue_lesson = (out_dir / f"cue_lesson{tag}.txt").read_text()
        self.scr_lesson = (out_dir / f"scrambled_lesson{tag}.txt").read_text()
        words = len(self.cue_lesson.split())
        self.neutral = PR.neutral_lesson_blinded(len(panel), words) if blinded else PR.neutral_lesson(panel, words)
        (out_dir / f"neutral_lesson{tag}.txt").write_text(self.neutral)
        pos = {eid: k for k, eid in enumerate(meta.index)}
        self.protos = [(pos[e], 1) for e in cues["prototypes"]["older"]] + [(pos[e], 0) for e in cues["prototypes"]["younger"]]
        allowed = set(self.blind.values()) if blinded else {p["label"] for p in panel}
        self.leaks = {"case_lines_flagged": 0, "context_words_flagged": {}}
        for t in self.case_txt:
            self.leaks["case_lines_flagged"] += len(PR.leak_scan_case(t, allowed))
        for name, txt in (("intro", PR.INTRO), ("neutral", self.neutral), ("cue", self.cue_lesson), ("scrambled", self.scr_lesson)):
            hits = PR.leak_scan_context(txt)
            if hits:
                self.leaks["context_words_flagged"][name] = hits


def run_gate_and_probe(cfg, backend, model_name, meta, feats, mats: Materials, blinded: bool, log=print) -> dict:
    """Known-answer gate (one explicit rule) and prior-knowledge probe for one model."""
    ec = cfg["experiment_llm"]
    g = ec["gate"]
    panel = cfg["panel"]
    fid = g["rule_feature"]
    col = list(feats.columns).index(fid)
    pool = np.where(meta["role"].values == "teacher_train")[0]
    thr = float(np.median(feats.values[pool, col]))
    rule_labels = (feats.values[:, col] > thr).astype(int)
    rng = np.random.default_rng(cfg["study"]["master_seed"] + 11)
    items = np.concatenate([rng.choice(pool[rule_labels[pool] == c], g["n_items"] // 2, replace=False) for c in (0, 1)])
    unit = next(p["unit"] for p in panel if p["id"] == fid)
    label = mats.blind[fid] if blinded else next(p["label"] for p in panel if p["id"] == fid)
    thr_txt = PR.fmt_value(thr, unit).split(" ")[0] if blinded else PR.fmt_value(thr, unit)
    accs = []
    for older_is_a in (True, False):  # both letter mappings, half the items each
        letters = PR.Letters(older_is_a)
        half = items[::2] if older_is_a else items[1::2]
        qs = [PR.test_question(mats.case_txt[i], letters) for i in half]
        pa = np.array(backend.p_a(PR.gate_intro(label, thr_txt), qs))
        p_older = pa if older_is_a else 1 - pa
        accs.extend(((p_older > 0.5) == (rule_labels[half] == 1)).tolist())
    acc = float(np.mean(accs))
    out = {"model": model_name, "run": "blinded" if blinded else "main", "gate_accuracy": acc,
           "gate_items": len(accs), "gate_pass": bool(acc >= g["min_accuracy"]), "gate_rule": f"{label} > {thr_txt}"}
    if not blinded:  # probe: does the model already 'know' how each measurement shifts with age?
        # Asked both ways (decision D18): P(higher) with A = higher, and with A = lower (1 - P(A)).
        # A direction counts as known only if both orderings agree.
        probe, probe_a, probe_b, known = {}, {}, {}, {}
        for p in panel:
            pa_first = float(backend.p_a("", [PR.probe_question(p["label"], True)])[0])
            pb_first = 1.0 - float(backend.p_a("", [PR.probe_question(p["label"], False)])[0])
            probe_a[p["id"]], probe_b[p["id"]] = pa_first, pb_first
            probe[p["id"]] = (pa_first + pb_first) / 2
            known[p["id"]] = PR.probe_known(pa_first, pb_first)
        out["probe_p_higher_in_older"] = probe  # mean of the two orderings
        out["probe_p_higher_A_first"], out["probe_p_higher_B_first"] = probe_a, probe_b
        out["probe_known_direction"] = known  # +1 higher, -1 lower, 0 inconsistent
    log(f"gate {model_name} ({out['run']}): accuracy {acc:.2f} -> {'PASS' if out['gate_pass'] else 'FAIL'}")
    return out


def run_experiment_llm(cfg, meta, feats, cues, out_dir: Path, log=print, blinded: bool = False):
    ec = cfg["experiment_llm"]
    tag = "_blinded" if blinded else ""
    exp_code = "llm_blind" if blinded else "llm"
    n_per = ec["blinded_n_per_condition"] if blinded else ec["n_per_condition"]
    labels = meta["label"].values.astype(int)
    mats = Materials(cfg, meta, feats, cues, out_dir, blinded)
    case_txt = mats.case_txt
    n_proto = len(mats.protos)
    cue_rows = np.where(meta["role"].values == "cue")[0]

    models = ec["models"]
    csv, icsv, gcsv = out_dir / f"learners_llm{tag}.csv", out_dir / f"items_llm{tag}.csv", out_dir / f"gate_llm{tag}.json"
    rows, items, done = [], [], set()
    gates = json.loads(gcsv.read_text()) if gcsv.exists() else {}
    if csv.exists():
        rows = pd.read_csv(csv).to_dict("records")
        items = pd.read_csv(icsv).to_dict("records") if icsv.exists() else []
        done = {(r["condition"], r["learner"]) for r in rows}
        log(f"resuming LLM{tag} experiment: {len(done)} learners already done")

    # learner i -> models[i % M]: balanced across models within each condition
    for mi, model_name in enumerate(models):
        todo = [(ci, c, i) for ci, c in enumerate(cfg["conditions"]) for i in range(n_per)
                if i % len(models) == mi and (c, i) not in done]
        if not todo and model_name in gates:
            continue
        log(f"loading learner model {model_name} for {len(todo)} learners ({'blinded' if blinded else 'main'} run)")
        backend = make_backend(cfg, model_name, cues, cfg["panel"], log)
        if model_name not in gates:
            gates[model_name] = run_gate_and_probe(cfg, backend, model_name, meta, feats, mats, blinded, log)
            gcsv.write_text(json.dumps(gates, indent=2))
        last_save = time.time()
        for n_done, (ci, cond, i) in enumerate(todo, 1):
            rng = learner_rng(cfg["study"]["master_seed"], exp_code, ci, i)
            sets = draw_learner_sets(rng, meta, ec["n_practice"], ec["n_items_per_test"], extra_practice=n_proto)
            letters = PR.Letters(older_is_a=(i // len(models)) % 2 == 0)  # exactly balanced within model and group
            practice = [(case_txt[j], labels[j]) for j in sets["practice"]]
            base_practice, extra = practice[: ec["n_practice"]], practice[ec["n_practice"]:]
            key = {"experiment": f"llm{tag}", "condition": cond, "learner": i, "model": model_name}
            rec = {**key, "older_is_A": letters.letter[1] == "A", "tests_disjoint": sets["tests_disjoint"],
                   "gate_pass": gates[model_name]["gate_pass"]}

            head = PR.INTRO + "\n\n"
            _score(backend, head.strip(), sets["pre"], case_txt, labels, letters, "pre", rec, items, key, ec)
            practice_txt = PR.practice_block(base_practice, letters, "PRACTICE CASES WITH FEEDBACK:")
            if cond == "no_training":
                lesson = mats.neutral
            elif cond == "feedback_only":
                lesson = mats.neutral + "\n\n" + PR.practice_block(base_practice + extra, letters, "PRACTICE CASES WITH FEEDBACK:")
            elif cond == "cue_training":
                proto_cases = [(case_txt[j], lab) for j, lab in mats.protos]
                lesson = (mats.cue_lesson + "\n\n" + PR.practice_block(proto_cases, letters, "CLEAREST EXAMPLES (the AI is most sure about these):")
                          + "\n\n" + practice_txt)
            else:  # scrambled_cues: same format, placebo content, random "clearest" cases
                rand = rng.choice(cue_rows, n_proto, replace=False)
                proto_cases = [(case_txt[j], labels[j]) for j in rand]
                lesson = (mats.scr_lesson + "\n\n" + PR.practice_block(proto_cases, letters, "CLEAREST EXAMPLES (the AI is most sure about these):")
                          + "\n\n" + practice_txt)
            ctx = head + lesson
            _score(backend, ctx, sets["post"], case_txt, labels, letters, "post", rec, items, key, ec)
            if "transfer" in sets:
                _score(backend, ctx, sets["transfer"], case_txt, labels, letters, "transfer", rec, items, key, ec)
            ctx_delayed = ctx + "\n\n" + PR.distractor_block(rng, ec["distractor_tokens"])
            _score(backend, ctx_delayed, sets["delayed"], case_txt, labels, letters, "delayed", rec, items, key, ec)
            rec["context_words_post"] = len(ctx.split())
            ex = out_dir / f"example_prompts{tag}" / f"{cond}.txt"
            if not ex.exists():
                ex.parent.mkdir(exist_ok=True)
                ex.write_text(ctx_delayed + "\n\n" + PR.test_question(case_txt[sets["delayed"][0]], letters))
            rows.append(rec)
            # Checkpoint (issue P13: a machine restart lost 20 unsaved learners): at most every
            # checkpoint_seconds, so a real model (minutes per learner) saves after every learner, while
            # fast mock learners in the synthetic validation do not rewrite the whole file thousands of
            # times (issue P24). Items first, then learners, because the learner file decides what --resume skips.
            if time.time() - last_save >= ec.get("checkpoint_seconds", 30) or n_done == len(todo):
                pd.DataFrame(items).to_csv(icsv, index=False)
                pd.DataFrame(rows).to_csv(csv, index=False)
                last_save = time.time()
            if n_done % 5 == 0 or n_done == len(todo):
                log(f"LLM{tag} {model_name}: {n_done}/{len(todo)} learners")
        backend.close()
    items_df = pd.DataFrame(items)
    items_df.to_csv(icsv, index=False)
    df = rescore_from_items(pd.DataFrame(rows), items_df, ec)
    df.to_csv(csv, index=False)
    (out_dir / f"leak_scan{tag}.json").write_text(json.dumps(mats.leaks, indent=2))
    # Which models count (decided per run). Always: the known-answer gate.
    # Flat rule (v0.4, with the bias-corrected score): a learner is flat on a test if more than
    # flat_tie_share of its items tie at its median, so they cannot be ranked; a model is uninformative
    # if more than flat_model_fraction of its learners are flat on the post-test.
    # Collapse rule (one answer for >= collapse_share of the items): always reported; it excludes a
    # model only when collapse_excludes_model is true (the v0.2 rule, used with the raw score).
    for m, g in gates.items():
        sub = df[df.model == m]
        g["post_collapse_fraction"] = float(sub["post_collapsed"].mean()) if len(sub) else float("nan")
        g["collapsed_model"] = bool(g["post_collapse_fraction"] > ec["collapse_model_fraction"])
        g["post_flat_fraction"] = float(sub["post_flat"].mean()) if len(sub) else float("nan")
        g["flat_model"] = bool(g["post_flat_fraction"] > ec.get("flat_model_fraction", 0.5))
        g["primary_score"] = ec.get("primary_score", "raw")
        g["collapse_excludes_model"] = bool(ec.get("collapse_excludes_model", True))
        g["informative"] = bool(g["gate_pass"] and not g["flat_model"]
                                and not (ec.get("collapse_excludes_model", True) and g["collapsed_model"]))
    gcsv.write_text(json.dumps(gates, indent=2))
    df["model_informative"] = df["model"].map({m: g["informative"] for m, g in gates.items()})
    df.to_csv(csv, index=False)
    return df, gates, mats.leaks
