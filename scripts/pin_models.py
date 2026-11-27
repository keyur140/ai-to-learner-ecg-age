# AI-to-learner knowledge transfer study (ECG age)
# Author: KEYUR SHAILESHKUMAR PATEL (ORCID: https://orcid.org/0009-0008-5261-043X)
# MSc Artificial Intelligence student, Department of Computer Science, Brunel University London, UK; Independent researcher
# Conducted independently; not an official Brunel University London project. Licence: MIT
"""Pin each learner model to its current exact Hugging Face commit, BEFORE freezing.

Writes experiment_llm.revisions in configs/study.yaml, e.g.
  revisions: {Qwen/Qwen2.5-3B-Instruct: 'aa8e72537993ba99e69dfaafa59ed015b17504d1', ...}
Needs internet access to huggingface.co, and a token (HF_TOKEN in .env) for the gated
Llama and Gemma models. Run once, then: python run_study.py freeze
"""
import os
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
cfg_path = ROOT / "configs" / "study.yaml"
env = ROOT / ".env"
if env.exists():
    for line in env.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
if (ROOT / "prereg" / "LOCK.json").exists():
    sys.exit("The preregistration is already frozen. Pinning now would be a deviation.")

from huggingface_hub import HfApi  # noqa: E402

cfg = yaml.safe_load(cfg_path.read_text())
api = HfApi(token=os.environ.get("HF_TOKEN"))
pins = {}
for m in cfg["experiment_llm"]["models"]:
    info = api.model_info(m)
    pins[m] = info.sha
    print(f"{m}: {info.sha}")
# Edit only the revisions line, so comments in the YAML file are kept
text = cfg_path.read_text()
new = "  revisions: {" + ", ".join(f"{k}: '{v}'" for k, v in pins.items()) + "}"
text, n = re.subn(r"^  revisions: .*$", new, text, count=1, flags=re.M)
if n != 1:
    sys.exit("could not find the 'revisions:' line in configs/study.yaml")
cfg_path.write_text(text)
print("configs/study.yaml updated. Next: python run_study.py freeze")
