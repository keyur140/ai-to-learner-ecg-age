#!/usr/bin/env python3
# AI-to-learner knowledge transfer study (ECG age)
# Author: KEYUR SHAILESHKUMAR PATEL (ORCID: https://orcid.org/0009-0008-5261-043X)
# MSc Artificial Intelligence student, Department of Computer Science, Brunel University London, UK; Independent researcher
# Conducted independently; not an official Brunel University London project. Licence: MIT
"""Run the preregistered study, or one of its helper steps.

  python run_study.py smoke                 # 5-minute pipeline check on fake data (no downloads)
  python run_study.py validate              # synthetic validation: 5 fake worlds + power simulation
  python run_study.py check-data            # verify PTB-XL against PhysioNet's SHA256SUMS
  python run_study.py freeze                # lock prereg + config + code (writes prereg/LOCK.json)
  python run_study.py run                   # the confirmatory study (needs the lock)
  python run_study.py run --exploratory     # same pipeline without a lock, labelled EXPLORATORY
  python run_study.py run --pilot 30        # protected exploratory pilot: test fold + held-out device sealed
  python run_study.py verify runs/<run_id>  # re-hash every output of a run against its SHA256SUMS.txt
  python run_study.py report runs/<run_id>  # rebuild REPORT.md/.html/.pdf from a run's files

Options for `run`: --only llm|nets, --resume runs/<run_id>, --llm-backend hf|openai|mock,
--llm-models ..., --allow-deviation "reason" (if anything changed after freezing).

Everything a run produces goes to runs/<run_id>/: run.log, cohort, teacher weights,
cues and lessons, learner-level and item-level results, gates, leak scan,
self-checks, analysis JSON, REPORT.md/.html/.pdf, figures, manifest.json and SHA256SUMS.txt.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from ai2human import __version__, repro  # noqa: E402

LOCK = ROOT / "prereg" / "LOCK.json"


def load_env():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def smoke_overrides(cfg: dict) -> dict:
    c = copy.deepcopy(cfg)
    c["_smoke_n"] = 3000
    c["teacher"].update(epochs=3, channels=16, blocks=2)
    c["cues"]["stability_bootstraps"] = 20
    c["cues"].update(n_teacher_seeds=3, consensus_min=2)
    c["saliency"].update(n_ecgs=32, ig_steps=8)
    c["experiment_llm"].update(backend="mock", models=["mock-student-1", "mock-student-2", "mock-student-3"],
                               n_per_condition=6, blinded_n_per_condition=6, n_items_per_test=20, n_practice=8)
    c["experiment_llm"]["gate"] = {**c["experiment_llm"]["gate"], "n_items": 20}
    c["experiment_nets"].update(n_per_condition=6, steps=30, distractor_steps=30, n_items_per_test=60, distractor_n=64)
    c["analysis"]["bootstrap"] = 500
    return c


def digests():
    return (repro.tree_digest(repro.sha256_tree(ROOT / "prereg", ("*.md",))),
            repro.sha256_file(ROOT / "configs" / "study.yaml"),
            repro.tree_digest(repro.sha256_tree(ROOT / "src", ("*.py",))))


def prereg_status(mode: str, allow_deviation: str | None, exploratory: bool):
    """mode: run | smoke | validate. Returns (status text, deviations, label)."""
    digest, cfg_sha, code = digests()
    if mode != "run":
        return f"not applicable ({mode} on synthetic data)", [], mode.upper()
    if not LOCK.exists():
        if exploratory:
            return "NOT LOCKED: exploratory run, not confirmatory", [], "EXPLORATORY"
        sys.exit("No prereg/LOCK.json. Freeze first (python run_study.py freeze), or run with --exploratory.")
    L = json.loads(LOCK.read_text())
    mism = [k for k, v in (("prereg_digest", digest), ("config_sha256", cfg_sha), ("code_digest", code)) if L.get(k) != v]
    if not mism:
        return f"LOCKED and unchanged (locked {L.get('locked_utc')})", [], "CONFIRMATORY"
    if exploratory:
        return f"changed after lock in {mism}: exploratory run", [], "EXPLORATORY"
    if allow_deviation:
        return f"DEVIATION: {mism} changed after lock", [f"{', '.join(mism)} changed after lock. Reason given: {allow_deviation}"], "CONFIRMATORY WITH DEVIATION"
    sys.exit(f"Prereg lock mismatch in {mism}. Revert the change, or rerun with "
             f"--allow-deviation \"reason\" (it will be written into the report).")


def execute(cfg: dict, cfg_raw: dict, mode: str, run_dir: Path, status: str, label: str, deviations: list, log,
            write_reports: bool = True, extra_ctx: dict | None = None) -> dict:
    from ai2human.pipeline import run_pipeline

    digest, cfg_sha, code = digests()
    (run_dir / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    man = repro.RunManifest(run_dir.name, run_dir, cfg)
    repro.set_seed(cfg["study"]["master_seed"])
    ctx = {"author": cfg.get("author"), "run_id": run_dir.name, "version": __version__, "mode": mode, "label": label,
           "smoke": mode != "run", "synthetic": mode != "run", "prereg_digest": digest, "prereg_lock_status": status,
           "config_sha": cfg_sha, "code_digest": code, "deviations": deviations, "seed": cfg["study"]["master_seed"],
           **(extra_ctx or {})}
    from ai2human.report import INFO_KEYS

    (run_dir / "run_info.json").write_text(json.dumps({k: ctx.get(k) for k in INFO_KEYS}, indent=2))
    run_pipeline(cfg, run_dir, mode != "run", ctx, man, log.info)
    if write_reports:
        from ai2human.report import write_report

        write_report(run_dir, ctx)
        try:
            from ai2human.pdf_report import run_report_pdf

            run_report_pdf(run_dir, ctx)
        except ImportError as e:  # reportlab missing: Markdown/HTML still written
            log.info(f"PDF report skipped ({e})")
        log.info("reports written: REPORT.md / REPORT.html / REPORT.pdf")
    log.info(f"done. writing manifest.json + SHA256SUMS.txt in {run_dir}")
    for h in log.handlers:  # run.log is complete before it is hashed
        h.flush()
    man.save(partial=False)
    return ctx


def pilot_overrides(cfg: dict, n: int) -> dict:
    """Protected exploratory pilot (decision D7): seal the confirmatory test fold and held-out device,
    split the remaining folds 1-9 the same way, and use n learners per group."""
    c = copy.deepcopy(cfg)
    d = c["data"]
    sealed = d["test_fold"]
    rest = sorted(set(d["teacher_train_folds"]) | {d["cue_fold"]})  # folds 1-9
    d.update(teacher_train_folds=rest[:-2], cue_fold=rest[-2], test_fold=rest[-1])
    c["_protected"] = {"sealed_fold": sealed}
    c["experiment_llm"].update(n_per_condition=n, blinded_n_per_condition=n)
    c["experiment_nets"]["n_per_condition"] = n
    return c


def cmd_run(args, mode="run"):
    load_env()
    cfg_raw = yaml.safe_load(open(args.config))
    cfg = smoke_overrides(cfg_raw) if mode == "smoke" else copy.deepcopy(cfg_raw)
    pilot = getattr(args, "pilot", None)
    if pilot:
        cfg = pilot_overrides(cfg, pilot)
        if mode == "run":
            args.exploratory = True
    if getattr(args, "only", None):
        cfg["experiment_llm"]["enabled"] = args.only == "llm"
        cfg["experiment_nets"]["enabled"] = args.only == "nets"
    if getattr(args, "llm_backend", None):
        cfg["experiment_llm"]["backend"] = args.llm_backend
    if getattr(args, "llm_models", None):
        cfg["experiment_llm"]["models"] = args.llm_models
    status, deviations, label = prereg_status(mode, getattr(args, "allow_deviation", None), getattr(args, "exploratory", False))
    if pilot:
        label = ("EXPLORATORY PILOT (protected: confirmatory test fold and held-out device sealed)" if mode == "run"
                 else label + " + PILOT (protected)")
    if mode == "run" and (args.llm_backend or args.llm_models or args.only):
        deviations.append(f"Command-line overrides: only={args.only}, backend={args.llm_backend}, models={args.llm_models}")
    if mode == "run" and cfg["experiment_llm"]["backend"] == "hf" and not cfg["experiment_llm"].get("revisions"):
        deviations.append("Learner models were not pinned to exact Hugging Face commits (run scripts/pin_models.py before freezing).")
    if getattr(args, "resume", None):
        run_dir = Path(args.resume)
    else:
        tag = {"run": "study", "smoke": "smoke"}[mode] + ("_pilot" if pilot else "")
        _, run_dir = repro.new_run_dir(ROOT / cfg["study"]["output_dir"], tag)
    log = repro.get_logger(run_dir)
    log.info(f"ai2human {__version__} | run {run_dir.name} | {label} | prereg: {status}")
    execute(cfg, cfg_raw, mode, run_dir, status, label, deviations, log)
    return run_dir


def cmd_validate(args):
    from ai2human.validate import run_validation

    if args.report_only:
        from ai2human.validation_report import write_validation_report

        out = Path(args.report_only)
        summary = json.loads((out / "validation_summary.json").read_text())
        write_validation_report(out, summary, yaml.safe_load(open(args.config)))
        print(f"rebuilt VALIDATION_REPORT.* in {out} from validation_summary.json")
        return out

    cfg_raw = yaml.safe_load(open(args.config))
    _, out = repro.new_run_dir(ROOT / cfg_raw["study"]["output_dir"], "validation")
    log = repro.get_logger(out)
    log.info(f"ai2human {__version__} | synthetic validation {out.name} | quick={args.quick}")
    run_validation(cfg_raw, out, log, execute, quick=args.quick, worlds=args.worlds)
    return out


def cmd_freeze(args):
    if LOCK.exists() and not args.force:
        sys.exit("prereg/LOCK.json already exists. Use --force only if you have NOT run the confirmatory study yet.")
    cfg = yaml.safe_load(open(ROOT / "configs" / "study.yaml"))
    if cfg["experiment_llm"]["backend"] == "hf" and not cfg["experiment_llm"].get("revisions"):
        print("WARNING: learner models are not pinned to exact commits. Run scripts/pin_models.py first "
              "(needs internet access to huggingface.co), or the confirmatory report will list this as a deviation.")
    prereg = repro.sha256_tree(ROOT / "prereg", ("*.md",))
    code = repro.sha256_tree(ROOT / "src", ("*.py",))
    data = {
        "version": __version__,
        "locked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "prereg_files": prereg, "prereg_digest": repro.tree_digest(prereg),
        "config_sha256": repro.sha256_file(ROOT / "configs" / "study.yaml"),
        "code_files": code, "code_digest": repro.tree_digest(code),
        "run_script_sha256": repro.sha256_file(ROOT / "run_study.py"),
    }
    LOCK.write_text(json.dumps(data, indent=2))
    print(json.dumps({k: data[k] for k in ("locked_utc", "prereg_digest", "config_sha256", "code_digest")}, indent=2))
    print("\nPaste these four lines into your OSF preregistration, upload the zip, THEN run: python run_study.py run")


def verify_run(run_dir: Path) -> dict:
    """Re-hash every file listed in SHA256SUMS.txt. Returns {"ok", "checked", "changed", "missing", "extra"}."""
    sums = run_dir / "SHA256SUMS.txt"
    if not sums.exists():
        return {"ok": False, "error": "no SHA256SUMS.txt"}
    listed = {}
    for line in sums.read_text().splitlines():
        h, _, name = line.partition("  ")
        listed[name] = h
    changed = [n for n, h in listed.items() if (run_dir / n).exists() and repro.sha256_file(run_dir / n) != h]
    missing = [n for n in listed if not (run_dir / n).exists()]
    now = set(repro.sha256_tree(run_dir)) - {"manifest.json", "SHA256SUMS.txt"}
    extra = sorted(now - set(listed))
    return {"ok": not (changed or missing), "checked": len(listed), "changed": changed, "missing": missing, "extra": extra}


def cmd_verify(args):
    r = verify_run(Path(args.run_dir))
    print(json.dumps(r, indent=2))
    sys.exit(0 if r.get("ok") else 1)


def cmd_report(args):
    from ai2human.pdf_report import run_report_pdf
    from ai2human.report import load_ctx, write_report

    rd = Path(args.run_dir)
    ctx = load_ctx(rd)
    write_report(rd, ctx)
    run_report_pdf(rd, ctx)
    print(f"rewrote REPORT.md/.html/.pdf in {rd} (SHA256SUMS.txt is not changed; `verify` will list them as changed)")


def cmd_check_data(args):
    from ai2human.data import fetch_ptbxl, verify_against_physionet

    cfg = yaml.safe_load(open(args.config))
    # downloads and unpacks PTB-XL first if needed (issue P12: check-data used to fail on a fresh folder)
    r = verify_against_physionet(fetch_ptbxl(ROOT / cfg["data"]["dir"]), print, force=True)
    print(json.dumps(r, indent=2))
    sys.exit(0 if r.get("status") == "ok" else 1)


def main():
    argv = sys.argv[1:]
    if argv and argv[0] == "--smoke":  # v0.1 spelling still works
        argv = ["smoke"] + argv[1:]
    if not argv or argv[0].startswith("-") and argv[0] not in ("-h", "--help"):
        argv = ["run"] + argv
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "study.yaml"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="the study on PTB-XL")
    r.add_argument("--only", choices=["llm", "nets"])
    r.add_argument("--resume")
    r.add_argument("--llm-backend", choices=["hf", "openai", "mock"])
    r.add_argument("--llm-models", nargs="+")
    r.add_argument("--allow-deviation")
    r.add_argument("--exploratory", action="store_true", help="allow running without (or against a changed) lock")
    r.add_argument("--pilot", type=int, metavar="N", help="protected exploratory pilot with N learners per group "
                   "(seals the test fold and held-out device; implies --exploratory)")
    s = sub.add_parser("smoke", help="tiny run on fake data with mock learners")
    s.add_argument("--only", choices=["llm", "nets"])
    s.add_argument("--pilot", type=int, metavar="N", help="test the protected pilot mode on fake data")
    v = sub.add_parser("validate", help="synthetic validation worlds + power simulation")
    v.add_argument("--quick", action="store_true", help="small sizes (for tests)")
    v.add_argument("--worlds", nargs="+", help="subset, e.g. W1 W5")
    v.add_argument("--report-only", help="rebuild the report of a finished validation folder")
    f = sub.add_parser("freeze", help="write prereg/LOCK.json")
    f.add_argument("--force", action="store_true")
    ve = sub.add_parser("verify", help="re-hash a run's outputs")
    ve.add_argument("run_dir")
    sub.add_parser("check-data", help="verify PTB-XL files against PhysioNet SHA256SUMS")
    rp = sub.add_parser("report", help="rebuild the reports of a finished run from its files")
    rp.add_argument("run_dir")
    for p in (r, s, v, f, ve, rp):
        p.add_argument("--config", default=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    {"run": cmd_run, "smoke": lambda a: cmd_run(a, "smoke"), "validate": cmd_validate, "freeze": cmd_freeze,
     "verify": cmd_verify, "check-data": cmd_check_data, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
