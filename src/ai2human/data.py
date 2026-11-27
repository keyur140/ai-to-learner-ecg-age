# AI-to-learner knowledge transfer study (ECG age)
# Author: KEYUR SHAILESHKUMAR PATEL (ORCID: https://orcid.org/0009-0008-5261-043X)
# MSc Artificial Intelligence student, Department of Computer Science, Brunel University London, UK; Independent researcher
# Conducted independently; not an official Brunel University London project. Licence: MIT
"""PTB-XL data access (download, load, split) plus a synthetic stand-in for smoke tests.

PTB-XL v1.0.3 (Wagner et al., 2020, Sci Data) is open (CC BY 4.0) on PhysioNet.
We use the 100 Hz version and the official stratified folds:
folds 1-8 = train, 9 = validation, 10 = test (human-validated labels).
"""
from __future__ import annotations

import ast
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

PTBXL_ZIP = (
    "https://physionet.org/static/published-projects/ptb-xl/"
    "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"
)
PTBXL_PLUS_ZIP = (
    "https://physionet.org/static/published-projects/ptb-xl-plus/"
    "ptb-xl-a-comprehensive-electrocardiographic-feature-dataset-1.0.1.zip"
)
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def _download(url: str, dest: Path, log=print) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        log(f"already downloaded: {dest}")
        return dest
    log(f"downloading {url} -> {dest} (this can take a while)")
    tmp = dest.with_suffix(".part")
    try:
        with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, length=1 << 20)
    except Exception as e:
        raise SystemExit(
            f"Download failed ({e}).\nDownload this file yourself in a browser:\n  {url}\n"
            f"and save it as:\n  {dest.resolve()}\nthen run the same command again.")
    tmp.rename(dest)
    return dest


def _extract(zip_path: Path, out_dir: Path, log=print) -> Path:
    marker = out_dir / ".extracted"
    if marker.exists():
        return out_dir
    log(f"extracting {zip_path.name}")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out_dir)
    marker.touch()
    return out_dir


def find_ptbxl_root(base: Path) -> Path:
    hits = list(base.rglob("ptbxl_database.csv"))
    if not hits:
        raise FileNotFoundError(f"ptbxl_database.csv not found under {base}")
    return hits[0].parent


def fetch_ptbxl(data_dir: str | Path, with_plus: bool = True, log=print) -> Path:
    data_dir = Path(data_dir)
    z = _download(PTBXL_ZIP, data_dir / "ptbxl.zip", log)
    _extract(z, data_dir / "ptbxl", log)
    if with_plus:
        z2 = _download(PTBXL_PLUS_ZIP, data_dir / "ptbxl_plus.zip", log)
        _extract(z2, data_dir / "ptbxl_plus", log)
    return find_ptbxl_root(data_dir / "ptbxl")


EXPECTED = {"records": 21799, "patients": 18869}


def verify_against_physionet(root: Path, log=print, force: bool = False) -> dict:
    """Check the download against PhysioNet's own SHA256SUMS.txt.

    status: ok | mismatch | no_sums_file. Results are cached in .physionet_verified.json
    (keyed by the SHA256SUMS digest); force=True re-hashes everything."""
    import hashlib

    sums = root / "SHA256SUMS.txt"
    out = {"sha256sums_present": sums.exists(), "status": "no_sums_file"}
    if not sums.exists():
        log("PhysioNet SHA256SUMS.txt not found next to the data: integrity not verified (shown in the report)")
    if sums.exists():
        digest = hashlib.sha256(sums.read_bytes()).hexdigest()
        cache = root / ".physionet_verified.json"
        if cache.exists() and not force:
            c = json.loads(cache.read_text())
            if c.get("sums_digest") == digest:
                log("PhysioNet checksums: already verified")
                return c
        bad, missing, n = [], [], 0
        for line in sums.read_text().splitlines():
            if not line.strip():
                continue
            h, name = line.split(maxsplit=1)
            f = root / name.strip().lstrip("*")
            if not f.exists():
                missing.append(name)
                continue
            n += 1
            hh = hashlib.sha256()
            with open(f, "rb") as fh:
                for b in iter(lambda: fh.read(1 << 20), b""):
                    hh.update(b)
            if hh.hexdigest() != h:
                bad.append(name)
        out.update(status="mismatch" if bad else ("incomplete" if missing else "ok"), sums_digest=digest, files_checked=n, mismatched=bad[:50], n_mismatched=len(bad),
                   missing=missing[:50], n_missing=len(missing))
        log(f"PhysioNet checksums: {n} files checked, {len(bad)} mismatched, {len(missing)} missing")
        (root / ".physionet_verified.json").write_text(json.dumps(out, indent=2))
    return out


def load_metadata(root: Path) -> pd.DataFrame:
    df = pd.read_csv(root / "ptbxl_database.csv", index_col="ecg_id")
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)
    # PTB-XL encodes age > 89 as 300 for anonymisation; recode to 90.
    df.loc[df["age"] >= 300, "age"] = 90
    return df


def load_signals(root: Path, df: pd.DataFrame, sampling_rate: int = 100) -> np.ndarray:
    """Return array (n, 12, T). Uses wfdb. Caches to .npy next to the data."""
    import wfdb

    cache = root / f"signals_{sampling_rate}hz_{len(df)}.npy"
    ids_cache = cache.with_suffix(".ids.npy")
    if cache.exists() and ids_cache.exists() and np.array_equal(np.load(ids_cache), df.index.values):
        return np.load(cache, mmap_mode="r")
    col = "filename_lr" if sampling_rate == 100 else "filename_hr"
    sigs = np.stack([wfdb.rdsamp(str(root / f))[0].T.astype(np.float32) for f in df[col]])
    np.save(cache, sigs)
    np.save(ids_cache, df.index.values)
    return sigs


def load_plus_features(data_dir: Path, source: str = "12sl") -> pd.DataFrame | None:
    """PTB-XL+ median-beat features (intervals, amplitudes, axes). Returns None if missing."""
    base = Path(data_dir) / "ptbxl_plus"
    hits = sorted(p for p in base.rglob(f"{source.lower()}_features.csv"))
    if not hits:
        return None
    feats = pd.read_csv(hits[0])
    idcol = "ecg_id" if "ecg_id" in feats.columns else feats.columns[0]
    return feats.set_index(idcol)


def load_feature_description(data_dir: Path) -> pd.DataFrame | None:
    hits = sorted(Path(data_dir).joinpath("ptbxl_plus").rglob("feature_description.csv"))
    return pd.read_csv(hits[0]) if hits else None


def extract_panel(feats: pd.DataFrame, panel: list[dict], desc: pd.DataFrame | None,
                  source: str = "12sl", log=print) -> pd.DataFrame:
    """Pull the preregistered panel out of a PTB-XL+ feature table.

    Feature names are matched in this order: the harmonised id itself
    (e.g. R_Amp_V5), then the source-specific name from feature_description.csv
    (e.g. R_PeakAmpl_V5 for 12SL). Amplitudes are converted to mV when the
    source stores microvolts (median |value| > 20 is taken as microvolts).
    """
    out = {}
    lookup: dict[str, str] = {}
    if desc is not None and f"{source}_feature" in desc.columns:
        for _, r in desc.iterrows():
            src = r.get(f"{source}_feature")
            if isinstance(src, str) and not src.startswith("$"):
                lookup[str(r["id"])] = src
    for item in panel:
        fid = item["id"]
        col = None
        if fid in feats.columns:
            col = fid
        else:
            # lead-specific: template id ends in _X in the description file
            stem, _, lead = fid.rpartition("_")
            for key in (fid, f"{stem}_X"):
                src = lookup.get(key)
                if src is None:
                    continue
                cand = src.replace("_X", f"_{lead}") if key.endswith("_X") else src
                if cand in feats.columns:
                    col = cand
                    break
        if col is None:
            raise KeyError(f"panel feature {fid} not found in {source} features; edit configs/study.yaml panel")
        v = pd.to_numeric(feats[col], errors="coerce")
        if item.get("unit") == "mV" and v.abs().median() > 20:
            v = v / 1000.0
            log(f"  {fid}: converted microvolts -> mV")
        out[fid] = v
        log(f"  panel {fid:<22} <- column {col}")
    return pd.DataFrame(out, index=feats.index)


# ---------------------------------------------------------------- synthetic
def synthetic_ptbxl(panel_ids: list[str], n: int = 3000, T: int = 1000, seed: int = 0, signal: float = 1.0):
    """Fake ECG-like data where age shifts a few panel features and the waveform.

    ONLY for smoke tests and the synthetic validation worlds. Results on this data mean nothing.
    `signal` scales how strongly age changes the ECG (0 = age leaves no trace).
    About 15% of patients have two ECGs, sometimes on different devices, so the
    patient-level split rules are exercised. Returns (metadata, signals (n,12,T), features).
    """
    rng = np.random.default_rng(seed)
    n_pat = int(n / 1.15)
    pat_age = np.where(rng.random(n_pat) < 0.5, rng.integers(18, 40, n_pat), rng.integers(65, 90, n_pat)).astype(float)
    mid = rng.random(n_pat) < 0.15
    pat_age[mid] = rng.integers(40, 65, mid.sum())
    pat_sex = rng.integers(0, 2, n_pat)
    pat_fold = rng.integers(1, 11, n_pat)  # PTB-XL folds are patient-wise
    patient = np.concatenate([np.arange(n_pat), rng.choice(n_pat, n - n_pat, replace=False)])
    rng.shuffle(patient)
    age, sex = pat_age[patient], pat_sex[patient]
    z = signal * (age - 55) / 20
    base = {"HR_Ventr_Global": (70, -3, 10), "PR_Int_Global": (155, 10, 20), "QRS_Dur_Global": (95, 3, 10),
            "QT_Int_Global": (390, 5, 25), "QT_IntBazett_Global": (415, 8, 20), "P_AxisFront_Global": (55, -5, 20),
            "R_AxisFrontal_Global": (45, -20, 25), "T_AxisFront_Global": (45, 5, 20), "R_Amp_V5": (1.6, -0.25, 0.4),
            "T_Amp_V1": (0.05, 0.08, 0.1), "S_Amp_V1": (-0.9, 0.1, 0.3)}
    unit_default = {"ms": (100, 0.0, 15), "deg": (45, 0.0, 20), "bpm": (70, 0.0, 10), "mV": (0.3, 0.0, 0.1)}
    feats = {}
    for fid in panel_ids:
        guess_unit = "ms" if ("Dur" in fid or "Int" in fid) else "deg" if "Axis" in fid else "mV"
        mu, slope, sd = base.get(fid, unit_default[guess_unit])
        feats[fid] = mu + slope * z + sd * rng.normal(size=n)
    feats = pd.DataFrame(feats)
    t = np.arange(T) / 100.0
    hr = feats.get("HR_Ventr_Global", pd.Series(np.full(n, 70.0))).clip(35, 180).values
    qrs = feats.get("QRS_Dur_Global", pd.Series(np.full(n, 95.0))).clip(50, 200).values
    ramp = feats.get("R_Amp_V5", pd.Series(np.full(n, 1.5))).values
    tamp = feats.get("T_Amp_V1", pd.Series(np.full(n, 0.1))).values
    axis = feats.get("R_AxisFrontal_Global", pd.Series(np.full(n, 45.0))).values
    sigs = np.zeros((n, 12, T), dtype=np.float32)
    lead_angles = np.linspace(-30, 120, 12)
    for i in range(n):
        period = 60.0 / hr[i]
        phase = (t % period) / period
        w = qrs[i] / 1000 / period / 3
        beat = ramp[i] * np.exp(-((phase - 0.3) ** 2) / (2 * w**2))
        beat += 3 * tamp[i] * np.exp(-((phase - 0.6) ** 2) / 0.004)
        beat += 0.12 * np.exp(-((phase - 0.15) ** 2) / 0.001)
        gains = np.cos(np.deg2rad(axis[i] - lead_angles))
        sigs[i] = gains[:, None] * beat[None, :] + rng.normal(0, 0.05, (12, T))
    devices = rng.choice(["CS-12   E", "AT-6 C 5.5", "CS100    3", "AT-60    3"], n, p=[0.55, 0.25, 0.12, 0.08])
    meta = pd.DataFrame({"patient_id": patient.astype(float), "age": age, "sex": sex,
                         "strat_fold": pat_fold[patient], "device": devices},
                        index=pd.Index(np.arange(1, n + 1), name="ecg_id"))
    feats.index = meta.index
    return meta, sigs, feats
