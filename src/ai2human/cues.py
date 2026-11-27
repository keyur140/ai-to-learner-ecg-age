# AI-to-learner knowledge transfer study (ECG age)
# Author: KEYUR SHAILESHKUMAR PATEL (ORCID: https://orcid.org/0009-0008-5261-043X)
# MSc Artificial Intelligence student, Department of Computer Science, Brunel University London, UK; Independent researcher
# Conducted independently; not an official Brunel University London project. Licence: MIT
"""Turn what the AI teacher uses into 5-10 teachable cues (feature surrogate).

Method (preregistered):
1. On the cue fold only, take the teacher's own guesses (p_older > 0.5), not the true ages.
2. Stability selection: L1 logistic regression of the teacher's guesses on the
   standardised panel, refit on bootstrap resamples, at each C in a fixed grid.
3. Pick the sparsest C whose stable set (selection frequency >= threshold) has at
   least k_min features; keep at most k_max (highest frequency first).
3b. (v0.3, decision D9) Repeat steps 2-3 on the guesses of each extra teacher seed. Keep the
   features chosen by at least consensus_min of the n_teacher_seeds teachers, ordered by votes,
   then mean selection frequency; top up to k_min the same way if too few agree (logged).
4. Each cue becomes a plain rule: direction from an L2 refit on the cues, cut-point =
   midpoint of the medians of the two teacher-guessed groups.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


def _fmt(v: float, unit: str) -> str:
    if unit == "mV":
        return f"{v:.2f} {unit}"
    return f"{v:.0f} {unit}"


def _l1_logreg(C: float) -> LogisticRegression:
    """L1 logistic regression across scikit-learn versions (penalty= was deprecated in 1.8)."""
    import sklearn

    major, minor = (int(x) for x in sklearn.__version__.split(".")[:2])
    if (major, minor) >= (1, 8):
        return LogisticRegression(l1_ratio=1.0, C=C, solver="liblinear", max_iter=2000)
    return LogisticRegression(penalty="l1", C=C, solver="liblinear", max_iter=2000)


def stability_selection(X: np.ndarray, y: np.ndarray, C: float, n_boot: int, rng) -> np.ndarray:
    n = len(y)
    counts = np.zeros(X.shape[1])
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        if len(np.unique(y[idx])) < 2:
            continue
        m = _l1_logreg(C)
        m.fit(X[idx], y[idx])
        counts += np.abs(m.coef_[0]) > 1e-8
    return counts / n_boot


def _select(X, yhat, cc, ids, rng, log):
    """Steps 2-3 of the method for one teacher's guesses. Returns (C, freq, selected indices, path)."""
    path, chosen = [], None
    for C in sorted(cc["l1_C_grid"]):
        freq = stability_selection(X, yhat, C, cc["stability_bootstraps"], rng)
        stable = [i for i in np.argsort(-freq) if freq[i] >= cc["stability_threshold"]]
        path.append({"C": C, "n_stable": len(stable), "freq": {ids[i]: float(freq[i]) for i in range(len(ids))}})
        log(f"cue stability C={C}: {len(stable)} stable features")
        if len(stable) >= cc["k_min"] and chosen is None:
            chosen = (C, freq, stable[: cc["k_max"]])
            break
    if chosen is None:
        C, freq = path[-1]["C"], np.array(list(path[-1]["freq"].values()))
        chosen = (C, freq, list(np.argsort(-freq)[: cc["k_min"]]))
        log("no C reached k_min stable features; took top-k_min by frequency at largest C (logged as rule fallback)")
    return (*chosen, path)


def extract_cues(cfg: dict, meta: pd.DataFrame, feats: pd.DataFrame, teacher_preds: pd.DataFrame,
                 out_dir: Path, log=print) -> dict:
    cc = cfg["cues"]
    panel = {p["id"]: p for p in cfg["panel"]}
    ids = list(feats.columns)
    rng = np.random.default_rng(cfg["study"]["master_seed"] + 2)
    cue_mask = (meta["role"] == "cue").values
    Xraw = feats.values[cue_mask]
    yhat = (teacher_preds["p_older"].values[cue_mask] > 0.5).astype(int)
    scaler = StandardScaler().fit(Xraw)
    X = scaler.transform(Xraw)

    C, freq, sel, path = _select(X, yhat, cc, ids, rng, log)
    consensus = None
    seed_cols = sorted(c for c in teacher_preds.columns if c.startswith("p_older_seed"))
    if seed_cols:
        # Cue consensus across teacher seeds (decision D9): run the same selection on each extra
        # teacher's guesses, keep features chosen by at least consensus_min of all teachers.
        per = [{"teacher": 0, "C": C, "selected": [ids[i] for i in sel], "freq": {ids[i]: float(freq[i]) for i in sel}}]
        for col in seed_cols:
            k = int(col.replace("p_older_seed", ""))
            yk = (teacher_preds[col].values[cue_mask] > 0.5).astype(int)
            rk = np.random.default_rng(cfg["study"]["master_seed"] + 2 + 1000 * k)
            Ck, fk, sk, _ = _select(X, yk, cc, ids, rk, lambda m, k=k: log(f"[seed{k}] {m}"))
            per.append({"teacher": k, "C": Ck, "selected": [ids[i] for i in sk], "freq": {ids[i]: float(fk[i]) for i in sk}})
        n_t = len(per)
        need = int(cc.get("consensus_min", n_t // 2 + 1))
        votes = {f: sum(f in t["selected"] for t in per) for f in ids}
        mfreq = {f: float(np.mean([t["freq"].get(f, 0.0) for t in per])) for f in ids}
        order = sorted(ids, key=lambda f: (-votes[f], -mfreq[f], ids.index(f)))
        agreed = [f for f in order if votes[f] >= need]
        fallback = len(agreed) < cc["k_min"]
        chosen_ids = (agreed if not fallback else order[: cc["k_min"]])[: cc["k_max"]]
        if fallback:
            log(f"cue consensus: only {len(agreed)} features chosen by >= {need}/{n_t} teachers; "
                f"topped up to k_min by votes, then mean frequency (logged as rule fallback)")
        sel = [ids.index(f) for f in chosen_ids]
        freq = np.array([mfreq[f] for f in ids])
        consensus = {"n_teachers": n_t, "min_votes": need, "votes": votes, "mean_frequency": mfreq,
                     "per_teacher": per, "agreed": agreed, "fallback": fallback, "selected": chosen_ids}
        log(f"cue consensus ({n_t} teachers, need {need}): votes "
            f"{ {f: votes[f] for f in order if votes[f]} } -> {chosen_ids}")

    refit = LogisticRegression(max_iter=5000).fit(X[:, sel], yhat)
    coefs = refit.coef_[0]
    cues = []
    for j, i in enumerate(sel):
        fid = ids[i]
        unit = panel[fid]["unit"]
        older_vals, young_vals = Xraw[yhat == 1, i], Xraw[yhat == 0, i]
        thr = float((np.median(older_vals) + np.median(young_vals)) / 2)
        older_higher = bool(coefs[j] > 0)
        cues.append({
            "id": fid, "label": panel[fid]["label"], "unit": unit,
            "selection_frequency": float(freq[i]), "std_coef": float(coefs[j]),
            "older_is_higher": older_higher, "threshold": thr,
            "median_teacher_older": float(np.median(older_vals)), "median_teacher_younger": float(np.median(young_vals)),
        })
    cues.sort(key=lambda c: -abs(c["std_coef"]))
    total = sum(abs(c["std_coef"]) for c in cues)
    for r, c in enumerate(cues, 1):
        c["rank"] = r
        c["weight"] = abs(c["std_coef"]) / total
        c["rule"] = rule_text(c, c["label"], c["unit"])

    y_cue = meta["label"].values[cue_mask].astype(int)
    scrambled = scrambled_cues(cfg, cues, ids, Xraw, panel, y_cue)
    log(f"placebo: draw {scrambled[0].get('placebo_draw')}, cue-fold tally accuracy "
        f"{scrambled[0].get('placebo_cue_fold_accuracy', float('nan')):.3f}")

    if consensus:  # do the other teachers agree on each cue's direction?
        for c in cues:
            j = sel.index(ids.index(c["id"]))
            agree = []
            for col in seed_cols:
                yk = (teacher_preds[col].values[cue_mask] > 0.5).astype(int)
                ck = LogisticRegression(max_iter=5000).fit(X[:, sel], yk).coef_[0][j]
                agree.append(bool((ck > 0) == c["older_is_higher"]))
            c["teacher_votes"] = consensus["votes"][c["id"]]
            c["direction_agreement"] = (1 + sum(agree)) / (1 + len(agree))
    report = {"C": C, "selected": [c["id"] for c in cues], "stability_path": path, "consensus": consensus,
              "n_rows_used": int(cue_mask.sum()), "roles_used": sorted(set(meta["role"].values[cue_mask]))}
    for r in ["cue", "test", "transfer"]:
        m = (meta["role"] == r).values
        if m.sum() == 0:
            continue
        Fm = feats.values[m]
        t_guess = (teacher_preds["p_older"].values[m] > 0.5).astype(int)
        y = meta["label"].values[m].astype(int)
        sur = refit.predict(scaler.transform(Fm)[:, sel])
        rt = tally(Fm, cues, ids)
        rs = tally(Fm, scrambled, ids)
        report[r] = {
            "scrambled_tally_accuracy_true": float(accuracy_score(y, rs)),
            "surrogate_fidelity_to_teacher": float(accuracy_score(t_guess, sur)),
            "surrogate_accuracy_true": float(accuracy_score(y, sur)),
            "rule_tally_fidelity_to_teacher": float(accuracy_score(t_guess, rt)),
            "rule_tally_accuracy_true": float(accuracy_score(y, rt)),
        }
    log(f"cues: {[c['id'] for c in cues]}; test fidelity {report.get('test', {})}")

    # Prototypes ("exaggerated" examples): the teacher's most confident cue-fold cases
    p = teacher_preds["p_older"].values[cue_mask]
    idx_cue = meta.index.values[cue_mask]
    k = cc["n_prototypes_per_class"]
    protos = {"older": [int(idx_cue[i]) for i in np.argsort(-p)[:k]],
              "younger": [int(idx_cue[i]) for i in np.argsort(p)[:k]]}

    out = {"cues": cues, "prototypes": protos, "report": report,
           "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(), "feature_order": ids}
    out["scrambled"] = scrambled
    out["blinded_labels"] = blinded_labels(cfg)
    (out_dir / "cues.json").write_text(json.dumps(out, indent=2))
    write_lessons(out, cfg, out_dir)
    return out


def tally(F: np.ndarray, cue_list: list[dict], ids: list[str]) -> np.ndarray:
    """Apply a lesson's rules as written: weighted vote, 'older' if more than half the weight agrees."""
    score = np.zeros(len(F))
    for c in cue_list:
        col = ids.index(c["id"])
        older_side = F[:, col] > c["threshold"] if c["older_is_higher"] else F[:, col] < c["threshold"]
        score += c["weight"] * older_side
    return (score > 0.5).astype(int)


def rule_text(c: dict, label: str, unit: str | None) -> str:
    side = "higher" if c["older_is_higher"] else "lower"
    other = "lower" if c["older_is_higher"] else "higher"
    cut = _fmt(c["threshold"], unit) if unit else f"{c['threshold']:.3g}"
    return f"{label}: in older adults it tends to be {side} (cut-point about {cut}; {other} points to younger)."


def scrambled_cues(cfg: dict, cues: list[dict], ids: list[str], Xraw: np.ndarray, panel: dict,
                   y_ref: np.ndarray | None = None) -> list[dict]:
    """Placebo lesson (preregistered rule): same number of cues, same weights and format,
    but the features are ones the AI did NOT select (topped up with selected ones only if
    the panel runs out), each with a random direction and its cue-fold median as cut-point.

    v0.3 (decision D10): the placebo must be neutral, not misleading. With a fixed seed, draws are
    repeated until the placebo rules, applied as a tally, score within the band
    cues.scrambled_band (default 0.45-0.55) against the TRUE labels of the cue fold. The first draw
    that passes is used; the number of draws is recorded. Test data are never used."""
    rng = np.random.default_rng(cfg["study"]["master_seed"] + 7)
    lo, hi = cfg["cues"].get("scrambled_band", [0.45, 0.55])
    max_draws = int(cfg["cues"].get("scrambled_max_draws", 1000))
    chosen = {c["id"] for c in cues}
    best = None
    for draw in range(1, max_draws + 1):
        pool = [f for f in ids if f not in chosen]
        rng.shuffle(pool)
        extra = [f for f in ids if f in chosen]
        rng.shuffle(extra)
        feats_for = (pool + extra)[: len(cues)]
        out = []
        for c, fid in zip(cues, feats_for):
            col = ids.index(fid)
            s = {"id": fid, "label": panel[fid]["label"], "unit": panel[fid]["unit"],
                 "older_is_higher": bool(rng.integers(0, 2)), "threshold": float(np.median(Xraw[:, col])),
                 "rank": c["rank"], "weight": c["weight"]}
            s["rule"] = rule_text(s, s["label"], s["unit"])
            out.append(s)
        if y_ref is None:
            return out
        acc = float(accuracy_score(y_ref, tally(Xraw, out, ids)))
        if best is None or abs(acc - 0.5) < abs(best[1] - 0.5):
            best = (out, acc, draw)
        if lo <= acc <= hi:
            break
    out, acc, draw = best
    for s in out:
        s["placebo_draw"] = draw
        s["placebo_cue_fold_accuracy"] = acc
        s["placebo_in_band"] = bool(lo <= acc <= hi)
    return out


def blinded_labels(cfg: dict) -> dict:
    """Neutral names for the blinded-labels run: panel order, 'Measure 1' ... 'Measure 18', no units."""
    return {p["id"]: f"Measure {k}" for k, p in enumerate(cfg["panel"], 1)}


def lesson_text(cue_list: list[dict], blind: dict | None = None) -> str:
    lines = ["LESSON: How an expert AI tells younger (under 40) from older (65+) adults on an ECG.",
             "The AI relies on these measurements, most important first:"]
    for c in cue_list:
        rule = rule_text(c, blind[c["id"]], None) if blind else c["rule"]
        lines.append(f"{c['rank']}. {rule}")
    lines.append("No single cue is decisive. Count how many cues point to 'older', giving more weight "
                 "to the cues near the top of the list, and decide by the balance.")
    return "\n".join(lines)


def cue_lesson_text(cues: list[dict]) -> str:
    return lesson_text(cues)


def write_lessons(out: dict, cfg: dict, out_dir: Path) -> None:
    b = out["blinded_labels"]
    (out_dir / "cue_lesson.txt").write_text(lesson_text(out["cues"]))
    (out_dir / "scrambled_lesson.txt").write_text(lesson_text(out["scrambled"]))
    (out_dir / "cue_lesson_blinded.txt").write_text(lesson_text(out["cues"], b))
    (out_dir / "scrambled_lesson_blinded.txt").write_text(lesson_text(out["scrambled"], b))
