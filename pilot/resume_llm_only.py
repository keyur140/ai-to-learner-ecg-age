# Resume helper for the exploratory pilot only (not part of the study package).
# The sandbox restarts often; a normal --resume retrains the 5 teachers (about 22 min) before any learner.
# This continues the language-model learners directly from the saved cohort rules and cues.json of the
# same run (cues were shown to be identical across reruns). Learner seeds, prompts and scoring are the
# package's own run_experiment_llm. The final reports/checks are still produced by a normal --resume.
import json, sys, logging
from pathlib import Path
ROOT = Path("/home/user/dev/ai2human_v03"); sys.path.insert(0, str(ROOT / "src"))
import os; os.chdir(ROOT)
import yaml
from ai2human.cohort import build_cohort
from ai2human.exp_llm import run_experiment_llm
run_dir = ROOT / "runs/20261107T164810Z_study_pilot"
cfg = yaml.safe_load(open(run_dir / "config_used.yaml"))
logging.basicConfig(level=logging.INFO, format="%(asctime)s | INFO | %(message)s", datefmt="%Y-%m-%dT%H:%M:%S",
                    handlers=[logging.FileHandler(run_dir / "resume_llm_only.log"), logging.StreamHandler()])
log = logging.info
log("resume_llm_only: continuing language-model learners without retraining teachers")
cues_before = (run_dir / "cues.json").read_bytes()
meta, feats, sigs, cohort = build_cohort(cfg, False, run_dir, log)
del sigs
cues = json.loads(cues_before)
for blinded in (False, True):
    df, gates, leaks = run_experiment_llm(cfg, meta, feats, cues, run_dir, log, blinded=blinded)
    log(f"{'blinded' if blinded else 'main'}: {len(df)} learners in file")
log("resume_llm_only: all language-model learners done")
