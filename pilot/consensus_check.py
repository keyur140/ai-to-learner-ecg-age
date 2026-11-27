# Teacher + cue stage only, v0.3 code, protected pilot splits (folds 1-9). Exploratory check of decision D9.
import sys, json, copy, logging
from pathlib import Path
ROOT = Path("/home/user/dev/ai2human_v03"); sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))
import yaml
import run_study as RS
from ai2human.cohort import build_cohort
from ai2human.teacher import train_teacher
from ai2human.cues import extract_cues
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
logging.basicConfig(filename=out / "run.log", level=logging.INFO, format="%(asctime)s | %(message)s")
log = lambda m: (logging.info(m), print(m, flush=True))
cfg = RS.pilot_overrides(yaml.safe_load(open(ROOT / "configs/study.yaml")), 30)
import os; os.chdir(ROOT)
meta, feats, sigs, cohort = build_cohort(cfg, False, out, log)
model, X, tp, tm = train_teacher(cfg, meta, feats, sigs, out, log)
cues = extract_cues(cfg, meta, feats, tp, out, log)
log("FINAL " + json.dumps({"selected": [c["id"] for c in cues["cues"]], "votes": {c["id"]: c.get("teacher_votes") for c in cues["cues"]},
     "dir_agree": {c["id"]: c.get("direction_agreement") for c in cues["cues"]}, "test": cues["report"].get("test"),
     "teacher_test": tm.get("test"), "seeds": tm.get("ensemble_for_cues")}))
