# Pilot-side exploratory sweep (NOT part of the study pipeline). Experiment A, Qwen2.5-3B, main run only.
# Question: with fewer practice cases (shorter prompts), do the AI's cues add more to practice?
# Same protected pilot split, same cues and lessons as the v0.3 pilot run; only experiment_llm.n_practice
# and the number of learners change. Uses the v0.4 backend (float32 answer read-out, cache reuse) and score.
# Usage: python llm_practice_sweep.py <n_practice> <learners_per_group>
import json, sys, logging, os, shutil
from pathlib import Path
import yaml
ROOT = Path("/home/user/dev/ai2human_v03"); sys.path.insert(0, str(ROOT / "src")); os.chdir(ROOT)
from ai2human.cohort import build_cohort
from ai2human.exp_llm import run_experiment_llm

n_practice, n_per = int(sys.argv[1]), int(sys.argv[2])
PILOT = ROOT / "runs/20261107T164810Z_study_pilot"
OUT = Path(f"/home/user/dev/sweeps/llm_practice/n{n_practice:02d}"); OUT.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | INFO | %(message)s", datefmt="%Y-%m-%dT%H:%M:%S",
                    handlers=[logging.FileHandler(OUT / "sweep.log"), logging.StreamHandler()])
log = logging.info
cfg = yaml.safe_load(open(PILOT / "config_used.yaml"))
new = yaml.safe_load(open(ROOT / "configs/study.yaml"))["experiment_llm"]
ec = cfg["experiment_llm"]
for k in ("primary_score", "flat_tie_share", "flat_model_fraction", "collapse_excludes_model", "answer_precision"):
    ec[k] = new[k]
ec.update(n_practice=n_practice, n_per_condition=n_per, blinded_run=False, models=["Qwen/Qwen2.5-3B-Instruct"])
(OUT / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
for f in ("cue_lesson.txt", "scrambled_lesson.txt", "cues.json"):
    shutil.copy2(PILOT / f, OUT / f)
meta, feats, sigs, cohort = build_cohort(cfg, False, OUT, log)
del sigs
assert cohort["protected"]["sealed_ecgs"] == 3646 and not (meta["strat_fold"] == 10).any()
cues = json.loads((PILOT / "cues.json").read_text())
log(f"=== Experiment A sweep: n_practice = {n_practice}, {n_per} learners per group, main run")
df, gates, leaks = run_experiment_llm(cfg, meta, feats, cues, OUT, log, blinded=False)
log("post (bias-corrected) by group: " + json.dumps(df.groupby("condition").post_acc.mean().round(4).to_dict()))
log("sweep done")
