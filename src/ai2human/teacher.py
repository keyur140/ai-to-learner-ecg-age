# AI-to-learner knowledge transfer study (ECG age)
# Author: KEYUR SHAILESHKUMAR PATEL (ORCID: https://orcid.org/0009-0008-5261-043X)
# MSc Artificial Intelligence student, Department of Computer Science, Brunel University London, UK; Independent researcher
# Conducted independently; not an official Brunel University London project. Licence: MIT
"""Train the AI teacher (young vs older from raw ECG) and a features-only baseline."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .models import Teacher
from .repro import set_seed, sha256_file


def pick_device(pref: str = "auto") -> torch.device:
    if pref == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() and pref == "auto":
        return torch.device("mps")
    return torch.device("cpu")


def lead_norm_stats(sigs: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # float64 accumulation (issue P11): float32 sums over ~5 million values per lead depended on how
    # the array sat in memory, so a fresh read and a cached read gave sd values 0.5% apart.
    sub = np.asarray(sigs[idx], dtype=np.float64)
    mu = sub.mean(axis=(0, 2), keepdims=True)[0]
    sd = sub.std(axis=(0, 2), keepdims=True)[0] + 1e-6
    return mu.astype(np.float32), sd.astype(np.float32)


def teacher_seeds(cfg: dict) -> list[int]:
    """Seeds of the teacher ensemble used for cue consensus (decision D9). Seed 0 is the primary teacher."""
    base = cfg["study"]["master_seed"]
    return [base + 1000 * k for k in range(int(cfg["cues"].get("n_teacher_seeds", 1)))]


@torch.no_grad()
def predict(model, X: np.ndarray, device, bs: int = 256) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        out.append(torch.sigmoid(model(torch.from_numpy(X[i : i + bs]).to(device))).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def _train_one(tc: dict, seed: int, X, y, tr, va, device, wpath: Path, log, tag: str = ""):
    set_seed(seed)
    model = Teacher(tc["channels"], tc["blocks"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, tc["epochs"])
    lossf = nn.BCEWithLogitsLoss()
    g = torch.Generator().manual_seed(seed)
    best, best_ep, wait, hist = -1.0, -1, 0, []
    for ep in range(tc["epochs"]):
        model.train()
        perm = torch.randperm(len(tr), generator=g).numpy()
        tot = 0.0
        for i in range(0, len(perm), tc["batch_size"]):
            b = tr[perm[i : i + tc["batch_size"]]]
            xb = torch.from_numpy(X[b]).to(device)
            # light augmentation: random amplitude scaling and time shift
            xb = xb * (1 + 0.1 * torch.randn(len(b), 1, 1, device=device))
            xb = torch.roll(xb, int(torch.randint(-50, 51, (1,), generator=g)), dims=2)
            loss = lossf(model(xb), torch.from_numpy(y[b]).to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        sched.step()
        pv = predict(model, X[va], device)
        auc = roc_auc_score(y[va], pv) if len(np.unique(y[va])) > 1 else float("nan")
        hist.append({"epoch": ep + 1, "train_loss": tot / len(tr), "val_auroc": auc})
        log(f"teacher{tag} epoch {ep+1:>3}: loss {tot/len(tr):.4f}  val AUROC {auc:.4f}")
        if auc > best:
            best, best_ep, wait = auc, ep + 1, 0
            torch.save(model.state_dict(), wpath)
        else:
            wait += 1
            if wait >= tc["patience"]:
                log(f"early stop{tag} at epoch {ep+1} (best {best_ep})")
                break
    model.load_state_dict(torch.load(wpath, map_location=device))
    return model, best_ep, hist


def train_teacher(cfg: dict, meta: pd.DataFrame, feats: pd.DataFrame, sigs: np.ndarray, out_dir: Path, log=print):
    """Train the primary teacher (seed 0, used for everything downstream) and, if cues.n_teacher_seeds > 1,
    further teachers with other seeds whose guesses are used only for the cue consensus (decision D9)."""
    tc = cfg["teacher"]
    seeds = teacher_seeds(cfg)
    device = pick_device()
    log(f"teacher device: {device}; {len(seeds)} teacher seed(s)")
    role = meta["role"].values
    y = meta["label"].values.astype(np.float32)
    tr, va = np.where(role == "teacher_train")[0], np.where(role == "teacher_val")[0]
    mu, sd = lead_norm_stats(sigs, tr)
    X = ((sigs - mu) / sd).astype(np.float32)
    np.savez(out_dir / "lead_norm.npz", mu=mu, sd=sd)

    wpath = out_dir / "teacher.pt"
    model, best_ep, hist = _train_one(tc, seeds[0], X, y, tr, va, device, wpath, log)
    p_all = predict(model, X, device)
    preds = pd.DataFrame({"p_older": p_all, "label": y, "role": role}, index=meta.index)
    ens = []
    for k, sd_k in enumerate(seeds[1:], 1):
        wk = out_dir / f"teacher_seed{k}.pt"
        mk, be_k, _ = _train_one(tc, sd_k, X, y, tr, va, device, wk, log, tag=f" seed{k}")
        pk = predict(mk, X, device)
        preds[f"p_older_seed{k}"] = pk
        te = role == "test"
        ens.append({"k": k, "seed": sd_k, "best_epoch": be_k, "weights_sha256": sha256_file(wk),
                    "test_auroc": float(roc_auc_score(y[te], pk[te])) if len(np.unique(y[te])) > 1 else None})
        log(f"teacher seed{k}: test AUROC {ens[-1]['test_auroc']}")
        del mk
    preds.to_csv(out_dir / "teacher_predictions.csv")

    # features-only baseline trained on TRUE labels (context: how much do the panel numbers carry?)
    base = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    trm = role == "teacher_train"
    base.fit(feats.values[trm], y[trm])
    p_base = base.predict_proba(feats.values)[:, 1]

    metrics = {"best_epoch": best_ep, "history": hist, "device": str(device), "weights_sha256": sha256_file(wpath),
               "seed": seeds[0], "ensemble_for_cues": ens,
               "lead_norm_sha256": sha256_file(out_dir / "lead_norm.npz"),
               "roles_used_for_training": sorted(set(role[tr])), "roles_used_for_early_stopping": sorted(set(role[va])),
               "n_train": int(len(tr)), "n_val": int(len(va))}
    for r in ["teacher_val", "cue", "test", "transfer"]:
        m = role == r
        if m.sum() == 0 or len(np.unique(y[m])) < 2:
            continue
        metrics[r] = {
            "n": int(m.sum()),
            "teacher_auroc": float(roc_auc_score(y[m], p_all[m])),
            "teacher_accuracy": float(accuracy_score(y[m], p_all[m] > 0.5)),
            "features_baseline_auroc": float(roc_auc_score(y[m], p_base[m])),
            "features_baseline_accuracy": float(accuracy_score(y[m], p_base[m] > 0.5)),
        }
    (out_dir / "teacher_metrics.json").write_text(json.dumps(metrics, indent=2))
    log(f"teacher test: {metrics.get('test')}")
    return model, X, preds, metrics
