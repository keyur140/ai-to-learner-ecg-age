# AI-to-learner knowledge transfer study (ECG age)
# Author: KEYUR SHAILESHKUMAR PATEL (ORCID: https://orcid.org/0009-0008-5261-043X)
# MSc Artificial Intelligence student, Department of Computer Science, Brunel University London, UK; Independent researcher
# Conducted independently; not an official Brunel University London project. Licence: MIT
"""Saliency check: where in the ECG does the teacher look? (integrated gradients)

Descriptive check of the cue list (not a hypothesis test): if the cues are mostly
about the T wave, the teacher's saliency should also be dense in the ST-T segment.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import find_peaks

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
# windows relative to the R peak, in seconds
WINDOWS = {"P": (-0.25, -0.08), "QRS": (-0.06, 0.06), "ST-T": (0.08, 0.40)}


def cue_segment(fid: str) -> str:
    if fid.startswith(("P_", "PR_")):
        return "P"
    if fid.startswith(("QRS", "R_", "S_", "Q_Amp")):
        return "QRS"
    if fid.startswith(("T_", "ST_", "QT")):
        return "ST-T"
    return "rhythm"


def integrated_gradients(model, x: torch.Tensor, steps: int) -> torch.Tensor:
    model.eval()
    base = torch.zeros_like(x)
    total = torch.zeros_like(x)
    for a in torch.linspace(1 / steps, 1, steps):
        xi = (base + a * (x - base)).requires_grad_(True)
        out = model(xi).sum()
        (g,) = torch.autograd.grad(out, xi)
        total += g
    return (x - base) * total / steps


def run_saliency(cfg: dict, model, X: np.ndarray, meta: pd.DataFrame, cues: dict, out_dir: Path, device, log=print):
    sc = cfg["saliency"]
    fs = cfg["data"]["sampling_rate"]
    rng = np.random.default_rng(cfg["study"]["master_seed"] + 3)
    idx = np.where(meta["role"].values == "cue")[0]
    idx = rng.choice(idx, min(sc["n_ecgs"], len(idx)), replace=False)
    lead_share = np.zeros(12)
    seg_mass = {k: 0.0 for k in WINDOWS}
    seg_time = {k: 0.0 for k in WINDOWS}
    total_mass, total_time = 0.0, 0.0
    for start in range(0, len(idx), 32):
        b = idx[start : start + 32]
        xb = torch.from_numpy(X[b]).to(device)
        ig = integrated_gradients(model, xb, sc["ig_steps"]).abs().detach().cpu().numpy()
        lead_share += ig.sum(axis=2).sum(axis=0)
        for k, row in enumerate(b):
            lead2 = X[row, 1]
            peaks, _ = find_peaks(lead2, distance=int(0.3 * fs), prominence=np.std(lead2))
            mass_t = ig[k].sum(axis=0)
            total_mass += mass_t.sum()
            total_time += len(mass_t)
            taken = np.zeros(len(mass_t), bool)
            for name in ("QRS", "ST-T", "P"):  # precedence: a sample belongs to one segment only
                lo, hi = WINDOWS[name]
                mask = np.zeros(len(mass_t), bool)
                for pk in peaks:
                    a, z = pk + int(lo * fs), pk + int(hi * fs)
                    mask[max(a, 0) : min(z, len(mass_t))] = True
                mask &= ~taken
                taken |= mask
                seg_mass[name] += mass_t[mask].sum()
                seg_time[name] += mask.sum()
    lead_share = lead_share / lead_share.sum()
    seg = {}
    for name in WINDOWS:
        share = seg_mass[name] / total_mass if total_mass else float("nan")
        tshare = seg_time[name] / total_time if total_time else float("nan")
        seg[name] = {"saliency_share": float(share), "time_share": float(tshare),
                     "density_ratio": float(share / tshare) if tshare else float("nan")}
    cue_w = {}
    for c in cues["cues"]:
        s = cue_segment(c["id"])
        cue_w[s] = cue_w.get(s, 0.0) + c["weight"]
    densest = max(seg, key=lambda k: seg[k]["density_ratio"])
    top_cue_seg = max(cue_w, key=cue_w.get)
    result = {
        "n_ecgs": int(len(idx)),
        "lead_share": dict(zip(LEADS, map(float, lead_share))),
        "segments": seg,
        "cue_weight_by_segment": cue_w,
        "densest_saliency_segment": densest,
        "segment_with_most_cue_weight": top_cue_seg,
        "agree": densest == top_cue_seg,
    }
    (out_dir / "saliency.json").write_text(json.dumps(result, indent=2))
    log(f"saliency: densest segment {densest}, most cue weight {top_cue_seg}")
    return result
