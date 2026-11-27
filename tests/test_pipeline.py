# AI-to-learner knowledge transfer study (ECG age)
# Author: KEYUR SHAILESHKUMAR PATEL (ORCID: https://orcid.org/0009-0008-5261-043X)
# MSc Artificial Intelligence student, Department of Computer Science, Brunel University London, UK; Independent researcher
# Conducted independently; not an official Brunel University London project. Licence: MIT
"""Run with: pytest -q   (takes a few minutes on CPU)"""
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def test_hashing_is_stable(tmp_path):
    from ai2human.repro import sha256_file, sha256_json

    f = tmp_path / "x.txt"
    f.write_text("abc")
    assert sha256_file(f) == hashlib.sha256(b"abc").hexdigest()
    assert sha256_json({"b": 1, "a": 2}) == sha256_json({"a": 2, "b": 1})


def test_power():
    from ai2human.stats import power_two_sample

    assert 0.89 < power_two_sample(300, 0.24) < 0.92


def test_learner_seeds_are_independent_and_repeatable():
    from ai2human.sampling import learner_rng

    a = learner_rng(1, "llm", 0, 5).integers(0, 10**9, 3)
    b = learner_rng(1, "llm", 0, 5).integers(0, 10**9, 3)
    c = learner_rng(1, "llm", 1, 5).integers(0, 10**9, 3)
    assert (a == b).all() and not (a == c).all()


def _run_smoke():
    out = subprocess.run([sys.executable, "-W", "ignore", str(ROOT / "run_study.py"), "smoke"],
                         capture_output=True, text=True, timeout=1800)
    assert out.returncode == 0, out.stderr[-3000:]
    runs = sorted((ROOT / "runs").glob("*_smoke"))
    return runs[-1]


def test_smoke_run_is_complete_and_reproducible():
    r1 = _run_smoke()
    r2 = _run_smoke()
    for name in ["REPORT.md", "REPORT.html", "manifest.json", "SHA256SUMS.txt", "analysis_llm.json", "analysis_nets.json"]:
        assert (r1 / name).exists(), name
    for name in ["learners_nets.csv", "learners_llm.csv", "cues.json"]:
        assert (r1 / name).read_bytes() == (r2 / name).read_bytes(), f"{name} differs between identical runs"


def test_smoke_v02_outputs_checks_and_verify():
    r = sorted((ROOT / "runs").glob("*_smoke"))[-1] if list((ROOT / "runs").glob("*_smoke")) else _run_smoke()
    import json

    for name in ["REPORT.pdf", "checks.json", "analysis_llm_blinded.json", "gate_llm.json", "leak_scan.json",
                 "scrambled_lesson.txt", "cue_lesson_blinded.txt", "run_info.json"]:
        assert (r / name).exists(), name
    checks = json.loads((r / "checks.json").read_text())
    # C15 only appears in protected pilot runs; C16 (cue consensus) and C17 (neutral placebo) are new in v0.3
    assert [c["id"] for c in checks] == [f"C{i}" for i in range(1, 15)] + ["C16", "C17"]
    assert not [c for c in checks if c["result"] == "FAIL"], checks
    sys.path.insert(0, str(ROOT))
    import run_study

    v = run_study.verify_run(r)
    assert v["ok"] and not v["changed"], v


def test_verify_catches_tampering(tmp_path):
    sys.path.insert(0, str(ROOT))
    import run_study
    from ai2human.repro import sha256_file

    (tmp_path / "a.csv").write_text("x,1\n")
    (tmp_path / "SHA256SUMS.txt").write_text(f"{sha256_file(tmp_path / 'a.csv')}  a.csv\n")
    assert run_study.verify_run(tmp_path)["ok"]
    (tmp_path / "a.csv").write_text("x,2\n")
    r = run_study.verify_run(tmp_path)
    assert not r["ok"] and r["changed"] == ["a.csv"]


def test_holm_and_welch_match_references():
    import numpy as np
    from ai2human.checks import _holm_manual, _welch_manual
    from ai2human.stats import holm, welch_test

    p = {"a": 0.01, "b": 0.04, "c": 0.03}
    assert all(abs(holm(p)[k] - _holm_manual(p)[k]) < 1e-12 for k in p)
    rng = np.random.default_rng(0)
    x, y = rng.normal(0.6, 0.1, 50), rng.normal(0.55, 0.12, 60)
    assert abs(welch_test(x, y)["p"] - _welch_manual(x, y)[1]) < 1e-10


def test_mock_learner_reads_rules_and_gate():
    from ai2human.llm_backends import MockBackend

    cues = {"cues": [{"id": "HR", "label": "Heart rate", "older_is_higher": False, "threshold": 70, "weight": 1.0}]}
    m = MockBackend("mock-student-1", cues, [{"id": "HR", "label": "Heart rate", "unit": "bpm"}])
    ctx = "1. Heart rate: in older adults it tends to be lower (cut-point about 70 bpm; higher points to younger)."
    from ai2human.prompts import Letters, test_question

    L = Letters(older_is_a=True)
    pa = m.p_a(ctx, [test_question("Heart rate: 55 bpm", L), test_question("Heart rate: 95 bpm", L)])
    assert pa[0] > 0.5 > pa[1]
    gate = "a case is an older adult (65 or over) exactly when Heart rate is above 70; otherwise"
    pg = m.p_a(gate, [test_question("Heart rate: 75 bpm", L), test_question("Heart rate: 65 bpm", L)])
    assert pg[0] > 0.5 > pg[1]


def test_lead_norm_stats_do_not_depend_on_memory_layout(tmp_path):
    """Issue P11: a fresh array and the same values loaded from .npy must give identical scaling."""
    import numpy as np
    from ai2human.teacher import lead_norm_stats

    rng = np.random.default_rng(0)
    a = (rng.standard_normal((900, 12, 1000)) * 0.2).astype(np.float32)
    np.save(tmp_path / "s.npy", a)
    b = np.load(tmp_path / "s.npy", mmap_mode="r")
    idx = np.arange(0, 900, 2)
    (m1, s1), (m2, s2) = lead_norm_stats(a, idx), lead_norm_stats(b, idx)
    assert np.array_equal(m1, m2) and np.array_equal(s1, s2)
    ref = a[idx].astype(np.float64).std(axis=(0, 2))
    assert np.allclose(s1[:, 0], ref + 1e-6, rtol=1e-6)


def test_placebo_is_neutral_on_cue_fold():
    """Decision D10: the scrambled lesson's tally must score inside the band on the cue fold."""
    import numpy as np
    from ai2human.cues import scrambled_cues, tally

    rng = np.random.default_rng(1)
    ids = [f"f{i}" for i in range(12)]
    X = rng.standard_normal((400, 12))
    y = (X[:, :6].sum(1) > 0).astype(int)  # every feature carries some signal
    panel = {f: {"label": f, "unit": "ms"} for f in ids}
    cues = [{"id": f"f{i}", "rank": i + 1, "weight": 1 / 3} for i in range(3)]
    cfg = {"study": {"master_seed": 5}, "cues": {"scrambled_band": [0.45, 0.55]}}
    sc = scrambled_cues(cfg, cues, ids, X, panel, y)
    acc = (tally(X, sc, ids) == y).mean()
    assert 0.45 <= acc <= 0.55 and sc[0]["placebo_in_band"]
    assert not {c["id"] for c in sc} & {c["id"] for c in cues}


def test_bias_corrected_score_ignores_letter_bias():
    import numpy as np

    from ai2human.exp_llm import test_scores
    from ai2human.stats import auroc, bias_corrected_credit

    ec = {"collapse_share": 0.95, "primary_score": "bias_corrected", "flat_tie_share": 0.5}
    rng = np.random.default_rng(0)
    y = np.array([0, 1] * 20)
    signal = 0.02 * (y - 0.5) + rng.normal(0, 0.01, 40)   # informative, but tiny next to the bias
    biased = np.clip(0.97 + signal, 0, 1)                 # says "older" for every item
    s = test_scores(biased, y, ec)
    assert s["acc_raw"] == 0.5 and s["collapsed"] and not s["flat"]
    assert s["acc_bc"] > 0.75 and s["acc"] == s["acc_bc"]
    # the same learner with the opposite bias gets exactly the same bias-corrected score
    assert test_scores(np.clip(0.03 + signal, 0, 1), y, ec)["acc_bc"] == s["acc_bc"]
    # hand count: 4 items, median 0.45; above -> older, below -> younger
    assert bias_corrected_credit([0.1, 0.4, 0.5, 0.9], [0, 1, 1, 1]).tolist() == [1.0, 0.0, 1.0, 1.0]
    # ties at the median get half credit; a constant learner scores exactly 0.5 and is flat
    c = test_scores(np.full(40, 0.97), y, ec)
    assert c["acc_bc"] == 0.5 and c["flat"] and c["tie_share"] == 1.0
    assert abs(auroc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) - 0.75) < 1e-12
    assert test_scores(biased, y, {**ec, "primary_score": "raw"})["acc"] == 0.5


def test_probe_counts_a_direction_only_when_both_orderings_agree():
    from ai2human.prompts import probe_known, probe_question

    assert "A) higher" in probe_question("QRS axis", True) and "A) lower" in probe_question("QRS axis", False)
    # a model that always answers "A" says P(higher) = 0.99 with A = higher and 0.01 with A = lower
    assert probe_known(0.99, 1 - 0.99) == 0
    assert probe_known(0.9, 0.8) == 1 and probe_known(0.2, 0.1) == -1
