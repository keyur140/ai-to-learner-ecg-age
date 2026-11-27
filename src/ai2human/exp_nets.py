# AI-to-learner knowledge transfer study (ECG age)
# Author: KEYUR SHAILESHKUMAR PATEL (ORCID: https://orcid.org/0009-0008-5261-043X)
# MSc Artificial Intelligence student, Department of Computer Science, Brunel University London, UK; Independent researcher
# Conducted independently; not an official Brunel University London project. Licence: MIT
"""Experiment B: small neural-net learners taught with data.

Groups (all four use the same number of practice ECGs and training steps):
  no_training    : practice ECGs with SHUFFLED labels (practice without real feedback)
  feedback_only  : practice ECGs with their true labels
  cue_training   : same, plus an extra loss for predicting the AI-derived cue values
  scrambled_cues : same, plus an extra loss for predicting the scrambled (placebo) cue values
Tests: pre (before practice), post, transfer (held-out device), delayed (after an
interfering task: learning sex from other ECGs with the same network).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .models import Student
from .repro import set_seed
from .sampling import draw_learner_sets, learner_rng


@torch.no_grad()
def _eval(model, X, y, device) -> dict:
    model.eval()
    logit, _, _ = model(torch.from_numpy(X).to(device))
    p = torch.sigmoid(logit).cpu().numpy()
    pc = np.where(y == 1, p, 1 - p)
    older = (p > 0.5).mean()
    return {"acc": float(((p > 0.5) == (y == 1)).mean()), "p_correct": float(pc.mean()),
            "majority_share": float(max(older, 1 - older))}


def run_learner(cfg, cond, cond_idx, i, X, meta, cue_sets, device, distractor_labels):
    ec = cfg["experiment_nets"]
    rng = learner_rng(cfg["study"]["master_seed"], "nets", cond_idx, i)
    seed = int(rng.integers(0, 2**31 - 1))
    set_seed(seed)
    sets = draw_learner_sets(rng, meta, ec["n_practice"], ec["n_items_per_test"])
    y = meta["label"].values.astype(int)
    cue_targets = cue_sets["scrambled_cues" if cond == "scrambled_cues" else "cue_training"]
    model = Student(cue_targets.shape[1], ec["channels"]).to(device)
    rec = {"experiment": "nets", "condition": cond, "learner": i, "seed": seed, "tests_disjoint": sets["tests_disjoint"]}
    for k, v in _eval(model, X[sets["pre"]], y[sets["pre"]], device).items():
        rec[f"pre_{k}"] = v

    bce, mse = nn.BCEWithLogitsLoss(), nn.MSELoss()
    P = sets["practice"]
    y_train = y.copy()
    if cond == "no_training":  # shuffled labels: same ECGs, same steps, no real feedback
        y_train[P] = rng.permutation(y[P])
    opt = torch.optim.AdamW(model.parameters(), lr=ec["lr"], weight_decay=ec["weight_decay"])
    for step in range(ec["steps"]):
        model.train()
        b = P[rng.integers(0, len(P), ec["batch_size"])]
        xb = torch.from_numpy(X[b]).to(device)
        logit, cue_pred, _ = model(xb)
        loss = bce(logit, torch.from_numpy(y_train[b].astype(np.float32)).to(device))
        if cond in ("cue_training", "scrambled_cues"):
            loss = loss + ec["cue_loss_weight"] * mse(cue_pred, torch.from_numpy(cue_targets[b]).to(device))
        opt.zero_grad()
        loss.backward()
        opt.step()
    for name in ("post", "transfer"):
        if name in sets:
            for k, v in _eval(model, X[sets[name]], y[sets[name]], device).items():
                rec[f"{name}_{k}"] = v

    # interference: learn a different task (sex) on other ECGs, then re-test age
    role = meta["role"].values
    excl = set(sets["practice"].tolist())
    pool = np.array([j for j in np.where(role == "teacher_train")[0] if j not in excl])
    D = rng.choice(pool, min(ec["distractor_n"], len(pool)), replace=False)
    params = list(model.backbone.parameters()) + list(model.aux_head.parameters())
    opt2 = torch.optim.AdamW(params, lr=ec["lr"], weight_decay=ec["weight_decay"])
    for step in range(ec["distractor_steps"]):
        model.train()
        b = D[rng.integers(0, len(D), ec["batch_size"])]
        _, _, aux = model(torch.from_numpy(X[b]).to(device))
        loss = bce(aux, torch.from_numpy(distractor_labels[b].astype(np.float32)).to(device))
        opt2.zero_grad()
        loss.backward()
        opt2.step()
    for k, v in _eval(model, X[sets["delayed"]], y[sets["delayed"]], device).items():
        rec[f"delayed_{k}"] = v
    return rec


def run_experiment_nets(cfg, X, meta, feats, cues, out_dir: Path, device, log=print) -> pd.DataFrame:
    ec = cfg["experiment_nets"]
    ids = cues["feature_order"]
    mu_all, sd_all = np.array(cues["scaler_mean"]), np.array(cues["scaler_scale"])

    def targets(cue_list):
        cols = [ids.index(c["id"]) for c in cue_list]
        return ((feats.values[:, cols] - mu_all[cols]) / sd_all[cols]).astype(np.float32)

    cue_sets = {"cue_training": targets(cues["cues"]), "scrambled_cues": targets(cues["scrambled"])}
    sex = meta["sex"].values.astype(int)
    rows = []
    csv = out_dir / "learners_nets.csv"
    done = set()
    if csv.exists():  # resume support
        old = pd.read_csv(csv)
        rows = old.to_dict("records")
        done = {(r["condition"], r["learner"]) for r in rows}
        log(f"resuming nets experiment: {len(done)} learners already done")
    for ci, cond in enumerate(cfg["conditions"]):
        for i in range(ec["n_per_condition"]):
            if (cond, i) in done:
                continue
            rows.append(run_learner(cfg, cond, ci, i, X, meta, cue_sets, device, sex))
            if (i + 1) % 25 == 0 or i + 1 == ec["n_per_condition"]:
                pd.DataFrame(rows).to_csv(csv, index=False)
                log(f"nets {cond}: {i+1}/{ec['n_per_condition']} learners")
    df = pd.DataFrame(rows)
    df.to_csv(csv, index=False)
    return df
