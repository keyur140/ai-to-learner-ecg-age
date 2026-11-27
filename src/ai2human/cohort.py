# AI-to-learner knowledge transfer study (ECG age)
# Author: KEYUR SHAILESHKUMAR PATEL (ORCID: https://orcid.org/0009-0008-5261-043X)
# MSc Artificial Intelligence student, Department of Computer Science, Brunel University London, UK; Independent researcher
# Conducted independently; not an official Brunel University London project. Licence: MIT
"""Build the study cohort: labels, panel features, and the preregistered splits.

Split roles (each ECG has exactly one):
  teacher_train / teacher_val : folds 1-8, used to train the AI teacher
  cue                         : fold 9, the teacher's guesses here define the cues
  test                        : fold 10, learners' pre/post/delayed tests
  transfer                    : every ECG from the held-out device (any fold)
Learner practice examples are drawn from teacher_train (true labels shown).

Roles are assigned by PATIENT, not by ECG: no patient ever appears in two roles.
PTB-XL folds are already patient-wise. Patients with any ECG on the transfer device
go to the transfer role; their ECGs from other devices are set aside ("unused").
The validation part of folds 1-8 is chosen per patient by a fixed SHA-256 rule.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D


def label_age(age: pd.Series, young_max: int, older_min: int) -> pd.Series:
    lab = pd.Series(np.nan, index=age.index)
    lab[age <= young_max] = 0
    lab[age >= older_min] = 1
    return lab


def build_cohort(cfg: dict, smoke: bool, out_dir: Path, log=print):
    dcfg = cfg["data"]
    panel = cfg["panel"]
    panel_ids = [p["id"] for p in panel]
    if smoke:
        meta, sigs, feats = D.synthetic_ptbxl(panel_ids, n=cfg.get("_smoke_n", 3000), seed=cfg["study"]["master_seed"],
                                              signal=cfg.get("_synthetic_signal", 1.0))
        source = {"dataset": "SYNTHETIC (pipeline test only; not real ECGs)"}
    else:
        root = D.fetch_ptbxl(dcfg["dir"], with_plus=True, log=log)
        meta = D.load_metadata(root)
        integrity = D.verify_against_physionet(root, log)
        if integrity["status"] == "mismatch":
            raise RuntimeError(f"PTB-XL files differ from PhysioNet's SHA256SUMS: {integrity['mismatched'][:5]}")
        integrity["n_records"] = int(len(meta))
        integrity["n_patients"] = int(meta["patient_id"].nunique())
        integrity["counts_match_published"] = (integrity["n_records"] == D.EXPECTED["records"]
                                               and integrity["n_patients"] == D.EXPECTED["patients"])
        log(f"PTB-XL: {integrity['n_records']} records, {integrity['n_patients']} patients "
            f"(published: {D.EXPECTED['records']}, {D.EXPECTED['patients']})")
        raw = D.load_plus_features(Path(dcfg["dir"]), dcfg["feature_source"])
        if raw is None:
            raise FileNotFoundError("PTB-XL+ features not found; the fetch step should have downloaded them")
        feats = D.extract_panel(raw, panel, D.load_feature_description(Path(dcfg["dir"])), dcfg["feature_source"], log)
        feats = feats.reindex(meta.index)
        sigs = None  # loaded lazily below for the kept rows only
        source = {"dataset": "PTB-XL 1.0.3 + PTB-XL+ 1.0.1", "root": str(root), "integrity": integrity}

    meta = meta.copy()
    meta["label"] = label_age(meta["age"], dcfg["young_max_age"], dcfg["older_min_age"])
    n_all = len(meta)
    keep = meta["label"].notna()
    n_age = int(keep.sum())
    n_missing = feats.isna().sum(axis=1)
    keep &= n_missing <= dcfg["max_missing_panel"]
    meta, feats = meta[keep], feats[keep]
    log(f"cohort: {n_all} ECGs -> {n_age} in age groups -> {len(meta)} with panel features")

    # transfer device (rule fixed in the prereg)
    dev_counts = meta["device"].value_counts()
    rank = dcfg["transfer_device_rank"]
    transfer_device = dev_counts.index[rank - 1] if len(dev_counts) >= rank else None
    protected = None
    if cfg.get("_protected"):
        # Protected exploratory run: seal the confirmatory test data BEFORE anything else is computed.
        # Every patient with an ECG in the confirmatory test fold or on the confirmatory transfer device
        # is removed; their signals are never loaded. The remaining ECGs get their own splits.
        p = cfg["_protected"]
        sealed_ecg = meta["strat_fold"].eq(p["sealed_fold"]) | meta["device"].eq(transfer_device)
        sealed_pat = set(meta.loc[sealed_ecg, "patient_id"])
        drop = meta["patient_id"].isin(sealed_pat)
        protected = {"sealed_fold": p["sealed_fold"], "sealed_device": transfer_device,
                     "sealed_ecgs": int(drop.sum()), "sealed_patients": len(sealed_pat)}
        meta, feats = meta[~drop], feats[~drop]
        if smoke:
            sigs = sigs[np.flatnonzero(keep.values)[~drop.values]]
            keep = pd.Series(True, index=meta.index)
        dev_counts = meta["device"].value_counts()
        transfer_device = dev_counts.index[rank - 1] if len(dev_counts) >= rank else None
        protected["exploratory_transfer_device"] = transfer_device
        log(f"PROTECTED: sealed {protected['sealed_ecgs']} ECGs of {protected['sealed_patients']} patients "
            f"(fold {p['sealed_fold']} + device {protected['sealed_device']!r}); exploratory transfer device {transfer_device!r}")
    role = pd.Series("unused", index=meta.index)
    in_train = meta["strat_fold"].isin(dcfg["teacher_train_folds"])
    seed = cfg["study"]["master_seed"]

    def patient_u(pid) -> float:  # fixed SHA-256 rule -> uniform number in [0, 1)
        h = hashlib.sha256(f"{seed}:val:{int(pid)}".encode()).hexdigest()
        return int(h[:12], 16) / 16**12

    val_mask = meta["patient_id"].map(patient_u) < dcfg["teacher_val_fraction"]
    role[in_train & ~val_mask] = "teacher_train"
    role[in_train & val_mask] = "teacher_val"
    role[meta["strat_fold"] == dcfg["cue_fold"]] = "cue"
    role[meta["strat_fold"] == dcfg["test_fold"]] = "test"
    n_set_aside = 0
    if transfer_device is not None:
        on_dev = meta["device"] == transfer_device
        tr_patients = set(meta.loc[on_dev, "patient_id"])
        of_tr_patient = meta["patient_id"].isin(tr_patients)
        role[of_tr_patient & ~on_dev] = "unused"
        n_set_aside = int((of_tr_patient & ~on_dev).sum())
        role[on_dev] = "transfer"
    meta["role"] = role
    # Self-check C1: a patient has exactly one role (ignoring set-aside ECGs)
    used = meta[meta.role != "unused"]
    roles_per_patient = used.groupby("patient_id")["role"].nunique()
    shared = int((roles_per_patient > 1).sum())
    if shared:
        raise RuntimeError(f"C1 failed: {shared} patients appear in more than one role")

    # impute remaining missing panel values with teacher_train medians (no test leakage)
    med = feats[meta["role"] == "teacher_train"].median()
    feats = feats.fillna(med)

    if not smoke:
        sigs = D.load_signals(Path(source["root"]), meta, dcfg["sampling_rate"])
    else:
        sigs = sigs[keep.values]
    sigs = np.asarray(sigs, dtype=np.float32)

    summary = {
        "source": source,
        "n_total_records": n_all,
        "n_in_age_groups": n_age,
        "n_cohort": int(len(meta)),
        "transfer_device": transfer_device,
        "n_patients": int(meta["patient_id"].nunique()),
        "ecgs_set_aside_for_patient_overlap": n_set_aside,
        "patients_in_more_than_one_role": shared,
        "protected": protected,
        "device_counts": {str(k): int(v) for k, v in dev_counts.items()},
        "roles": {
            r: {"n": int((meta.role == r).sum()), "n_older": int(((meta.role == r) & (meta.label == 1)).sum())}
            for r in ["teacher_train", "teacher_val", "cue", "test", "transfer", "unused"]
        },
        "imputation_medians": {k: float(v) for k, v in med.items()},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    meta[["patient_id", "age", "sex", "strat_fold", "device", "label", "role"]].to_csv(out_dir / "cohort.csv")
    feats.to_csv(out_dir / "panel_features.csv")
    (out_dir / "cohort_summary.json").write_text(json.dumps(summary, indent=2))
    log(f"roles: {json.dumps(summary['roles'])}; transfer device = {transfer_device!r}")
    return meta, feats, sigs, summary
