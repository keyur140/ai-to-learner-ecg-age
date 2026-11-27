# Pilot-side helper: assemble the v0.3.0 package folder and zip it. Usage: python build_package.py [out_zip]
import hashlib, shutil, sys, zipfile
from pathlib import Path

DEV = Path("/home/user/dev")
SRC = DEV / "ai2human_v03"
V3 = SRC / "runs/20261107T164810Z_study_pilot"
V2 = Path("/home/user/pilot/ai2human_v02/runs")
STAGE = DEV / "stage"
PKG = STAGE / "ai2human_v03"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else DEV / "ai2human_v1.0.0.zip"
IGN = shutil.ignore_patterns("__pycache__", ".pytest_cache", ".venv", "data", "runs", "PACKAGE_SHA256SUMS.txt", "*.pyc")


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


shutil.rmtree(STAGE, ignore_errors=True)
shutil.copytree(SRC, PKG, ignore=IGN, symlinks=False)
pil = PKG / "pilot"
pil.mkdir()
shutil.copytree(V3, pil / "v0.3_run", ignore=shutil.ignore_patterns("__pycache__"))
shutil.copytree(V2 / "20261107T114352Z_study_pilot", pil / "v0.2_run1")
shutil.copytree(V2 / "20261107T120108Z_study_pilot", pil / "v0.2_run2")
for f in ["pilot_report/EXPLORATORY_PILOT_REPORT.md", "pilot_report/EXPLORATORY_PILOT_REPORT.pdf",
          "pilot_report/PREREG_CHANGES_PROPOSED.md", "pilot_report/PILOT_FOLLOWUP.md", "make_pilot_report.py", "resume_llm_only.py",
          "consensus_check.py", "tools/md2pdf.py", "tools/build_package.py"]:
    shutil.copy2(DEV / f, pil / Path(f).name)
fu = pil / "followup"
(fu / "nets_practice").mkdir(parents=True)
for n in (4, 8, 16, 32, 64, 128):
    (fu / "nets_practice" / f"n{n:03d}").mkdir()
    shutil.copy2(DEV / f"sweeps/nets_practice/n{n:03d}/learners_nets.csv", fu / "nets_practice" / f"n{n:03d}")
shutil.copy2(DEV / "sweeps/nets_practice/sweep.log", fu / "nets_practice")
for f in ["backend_microtest.out", "backend_microtest.json", "tools/nets_practice_sweep.py", "tools/backend_microtest.py", "tools/llm_practice_sweep.py"]:
    shutil.copy2(DEV / f, fu / Path(f).name)
VAL = SRC / "runs/20261120T132404Z_validation"
vd = PKG / "validation"
vd.mkdir()
for f in ["VALIDATION_REPORT.pdf", "VALIDATION_REPORT.md", "VALIDATION_REPORT.html", "validation_summary.json", "SHA256SUMS.txt", "run.log"]:
    shutil.copy2(VAL / f, vd / f)
shutil.copytree(VAL / "figures", vd / "figures")
(pil / "README.md").write_text((DEV / "tools/pilot_README.md").read_text())
shutil.copy2(DEV / "pilot_report/PILOT_DISCLOSURE.md", PKG / "prereg/PILOT_DISCLOSURE.md")
fa = PKG / "freeze_audit"
fa.mkdir()
for f in ["audit_prereg_vs_code.py", "audit_prereg_vs_code.json", "smoke_v1.0.0.log"]:
    shutil.copy2(DEV / "publish" / f, fa / f)

files = sorted(p for p in PKG.rglob("*") if p.is_file())
lines = [f"{sha(p)}  ./{p.relative_to(PKG).as_posix()}" for p in files]
(PKG / "PACKAGE_SHA256SUMS.txt").write_text("\n".join(lines) + "\n")

OUT.unlink(missing_ok=True)
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for p in sorted(PKG.rglob("*")):
        if p.is_file():
            z.write(p, p.relative_to(STAGE).as_posix())
print(f"{len(files) + 1} files, {OUT.stat().st_size / 1e6:.1f} MB")
print(f"{sha(OUT)}  {OUT.name}")
