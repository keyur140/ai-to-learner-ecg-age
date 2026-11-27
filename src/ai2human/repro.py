# AI-to-learner knowledge transfer study (ECG age)
# Author: KEYUR SHAILESHKUMAR PATEL (ORCID: https://orcid.org/0009-0008-5261-043X)
# MSc Artificial Intelligence student, Department of Computer Science, Brunel University London, UK; Independent researcher
# Conducted independently; not an official Brunel University London project. Licence: MIT
"""Reproducibility utilities: seeds, SHA-256 hashing, environment capture, run manifests.

Every run writes runs/<run_id>/manifest.json with hashes of the config, the
preregistration, all source files, the input data, and every output file.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHUNK = 1 << 20


def set_seed(seed: int) -> None:
    """Seed every RNG we use. Torch is seeded only if installed."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(obj: Any) -> str:
    return sha256_bytes(json.dumps(obj, sort_keys=True, default=str).encode())


def sha256_tree(root: str | Path, patterns: tuple[str, ...] = ("*",)) -> dict[str, str]:
    """Hash every file under root matching patterns. Returns {relative_path: sha256}."""
    root = Path(root)
    out: dict[str, str] = {}
    if not root.exists():
        return out
    for pat in patterns:
        for p in sorted(root.rglob(pat)):
            if p.is_file() and "__pycache__" not in p.parts:
                out[str(p.relative_to(root))] = sha256_file(p)
    return dict(sorted(out.items()))


def tree_digest(hashes: dict[str, str]) -> str:
    """One hash summarising a whole file tree (order-independent)."""
    return sha256_json(hashes)


def environment_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    pkgs = {}
    for name in ["numpy", "pandas", "scipy", "sklearn", "statsmodels", "torch", "wfdb", "yaml", "matplotlib"]:
        try:
            mod = __import__(name)
            pkgs[name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            pkgs[name] = None
    info["packages"] = pkgs
    try:
        import torch

        info["cuda"] = torch.cuda.is_available()
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        info["cuda"] = None
    try:
        info["pip_freeze"] = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, timeout=60
        ).stdout.splitlines()
    except Exception as e:  # pragma: no cover
        info["pip_freeze"] = f"unavailable: {e}"
    return info


def get_logger(run_dir: Path, name: str = "ai2human") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%dT%H:%M:%S")
    fh = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


@dataclass
class RunManifest:
    run_id: str
    run_dir: Path
    config: dict[str, Any]
    stages: list[dict[str, Any]] = field(default_factory=list)

    def record_stage(self, name: str, started: float, **extra: Any) -> None:
        self.stages.append({"stage": name, "seconds": round(time.time() - started, 2), **extra})
        self.save(partial=True)

    def save(self, partial: bool = False) -> Path:
        prereg_dir = PROJECT_ROOT / "prereg"
        manifest = {
            "run_id": self.run_id,
            "author": self.config.get("author"),
            "status": "partial" if partial else "complete",
            "config_sha256": sha256_json(self.config),
            "config": self.config,
            "prereg_sha256": sha256_tree(prereg_dir),
            "code_sha256": sha256_tree(PROJECT_ROOT / "src", ("*.py",)),
            "code_digest": tree_digest(sha256_tree(PROJECT_ROOT / "src", ("*.py",))),
            "stages": self.stages,
            "environment": environment_info() if not partial else None,
        }
        if not partial:
            outputs = {
                k: v
                for k, v in sha256_tree(self.run_dir).items()
                if k not in ("manifest.json", "SHA256SUMS.txt")
            }
            manifest["outputs_sha256"] = outputs
            with open(self.run_dir / "SHA256SUMS.txt", "w") as f:
                for k, v in outputs.items():
                    f.write(f"{v}  {k}\n")
        path = self.run_dir / "manifest.json"
        path.write_text(json.dumps(manifest, indent=2, default=str))
        return path


def new_run_dir(base: str | Path, tag: str) -> tuple[str, Path]:
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"_{tag}"
    d = Path(base) / run_id
    d.mkdir(parents=True, exist_ok=False)
    return run_id, d
