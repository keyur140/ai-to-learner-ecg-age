# Pilot-side check of the v0.4 backend changes on the real model (Qwen2.5-3B, CPU).
# Uses the saved example prompt of the pilot's cue group, cut to a shorter context so it runs in minutes.
# Compares: (a) v0.3 behaviour (model logits, no cache reuse), (b) float32 read-out, (c) float32 + cache reuse.
import json, os, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, "/home/user/dev/ai2human_v03/src")
from ai2human.llm_backends import HFBackend

R = Path("/home/user/dev/ai2human_v03/runs/20261107T164810Z_study_pilot")
txt = (R / "example_prompts/cue_training.txt").read_text()
ctx_full, _, q_tail = txt.rpartition("\n\nTEST CASE:")
cases = [c for c in ctx_full.split("\nCase ")[1:]]
n_ctx = int(os.environ.get("N_CASES", 8))
head = ctx_full.split("\nCase ")[0]
ctx = head + "".join("\nCase " + c for c in cases[:n_ctx])
ctx_delayed = ctx + "\n\nUNRELATED TASK: the weather in three towns was recorded for a week. " * 12
# questions: reuse the bodies of later practice cases as test cases (the answer line removed)
tmpl = "TEST CASE:" + q_tail
qbody = q_tail.split("\n\n")[0]
qs = []
for c in cases[n_ctx:n_ctx + 16]:
    body = c.split("Correct answer:")[0].split(":\n", 1)[-1].strip()
    qs.append(tmpl.replace(qbody.strip(), body) if qbody.strip() in tmpl else tmpl)
log = lambda m: print("   ", m, flush=True)
res = {}
for tag, prec, reuse in (("a_v03", "model", "0"), ("b_fp32", "float32", "0"), ("c_fp32_reuse", "float32", "1")):
    os.environ["AI2HUMAN_REUSE_CACHE"] = reuse
    be = HFBackend("Qwen/Qwen2.5-3B-Instruct", "bfloat16", 8, log, answer_precision=prec)
    t = time.time()
    r = {"post": be.p_a(ctx, qs), "transfer": be.p_a(ctx, qs[::-1])[::-1], "delayed": be.p_a(ctx_delayed, qs)}
    r["seconds"] = time.time() - t
    r["fp32_maxdiff"] = be.fp32_maxdiff
    res[tag] = r
    print(tag, "seconds %.0f" % r["seconds"], "distinct post:", len(set(r["post"])), "of", len(qs), "| fp32 vs model logits max diff %.3f" % be.fp32_maxdiff, flush=True)
    be.close(); del be
A, B, C = (res[k] for k in ("a_v03", "b_fp32", "c_fp32_reuse"))
for t in ("post", "transfer", "delayed"):
    a, b, c = (np.array(x[t]) for x in (A, B, C))
    print(t, "| max |p fp32 - p model| %.4f" % np.abs(b - a).max(), "| max |p reuse - p no reuse| %.5f" % np.abs(c - b).max(),
          "| same side of 0.5 (a vs b): %d/%d" % (((a > .5) == (b > .5)).sum(), len(a)))
print("post vs transfer (same prompt, reversed order), fp32+reuse: max diff %.5f" % np.abs(np.array(C["post"]) - np.array(C["transfer"])).max())
json.dump(res, open("/home/user/dev/backend_microtest.json", "w"), indent=1)
