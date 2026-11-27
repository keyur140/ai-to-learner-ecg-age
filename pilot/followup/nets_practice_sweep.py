# Pilot-side exploratory sweep for issue P16 (Experiment B ceiling). NOT part of the study pipeline.
# Question: how many labelled practice ECGs does a small net need before practice alone reaches the ceiling?
# Same protected pilot split as the pilot (folds 1-9; fold 10 and the held-out device sealed), same teacher
# scaling and the same cues as the v0.3 pilot run; only experiment_nets.n_practice changes. 30 nets per group.
import json, sys, logging, os
from pathlib import Path
import numpy as np, pandas as pd, yaml
ROOT = Path("/home/user/dev/ai2human_v03"); sys.path.insert(0, str(ROOT / "src")); os.chdir(ROOT)
from ai2human.cohort import build_cohort
from ai2human.exp_nets import run_experiment_nets
from ai2human.teacher import pick_device

PILOT = ROOT / "runs/20261107T164810Z_study_pilot"
OUT = Path("/home/user/dev/sweeps/nets_practice"); OUT.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | INFO | %(message)s", datefmt="%Y-%m-%dT%H:%M:%S",
                    handlers=[logging.FileHandler(OUT / "sweep.log"), logging.StreamHandler()])
log = logging.info
cfg = yaml.safe_load(open(PILOT / "config_used.yaml"))
meta, feats, sigs, cohort = build_cohort(cfg, False, OUT, log)
assert cohort["protected"]["sealed_ecgs"] == 3646 and not (meta["strat_fold"] == 10).any()
nz = np.load(PILOT / "lead_norm.npz")
X = ((sigs - nz["mu"]) / nz["sd"]).astype(np.float32)
del sigs
cues = json.loads((PILOT / "cues.json").read_text())
device = pick_device("cpu")
for n in [int(a) for a in sys.argv[1:]] or [4, 8, 16, 32, 64, 128]:
    c = json.loads(json.dumps(cfg))
    c["experiment_nets"]["n_practice"] = n
    c["experiment_nets"]["n_per_condition"] = 30
    d = OUT / f"n{n:03d}"; d.mkdir(exist_ok=True)
    log(f"=== n_practice = {n}")
    df = run_experiment_nets(c, X, meta, feats, cues, d, device, log)
    log("post accuracy by group: " + json.dumps(df.groupby("condition").post_acc.mean().round(4).to_dict()))
log("sweep done")
