# AI-to-learner knowledge transfer study (ECG age)
# Author: KEYUR SHAILESHKUMAR PATEL (ORCID: https://orcid.org/0009-0008-5261-043X)
# MSc Artificial Intelligence student, Department of Computer Science, Brunel University London, UK; Independent researcher
# Conducted independently; not an official Brunel University London project. Licence: MIT
"""Language-model backends. Each returns P(letter A) for a (context, question) pair.

hf     : local open-source model via Hugging Face transformers (default, free)
openai : any OpenAI-compatible server (vLLM, Ollama, LM Studio, or a paid API)
mock   : a fake learner for smoke tests only (results are meaningless)
"""
from __future__ import annotations

import copy
import gc
import math
import os
import re

import numpy as np


def _trim_heap():
    """Give freed memory back to the system after each long prompt (issue P18: RSS crept up over hours)."""
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

MARK = "⁣ITEM⁣"


class Backend:
    name = "base"

    def p_a(self, context: str, questions: list[str]) -> list[float]:
        raise NotImplementedError

    def close(self):
        pass


class HFBackend(Backend):
    def __init__(self, model_name: str, dtype: str = "bfloat16", batch_size: int = 8, log=print,
                 revision: str | None = None, answer_precision: str = "float32"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.name = model_name
        self.bs = batch_size
        self.log = log
        tok_kwargs = {"token": os.environ.get("HF_TOKEN")} if os.environ.get("HF_TOKEN") else {}
        if revision:  # exact commit pinned by scripts/pin_models.py
            tok_kwargs["revision"] = revision
        log(f"loading {model_name} at revision {revision or 'latest (NOT pinned)'}")
        self.tok = AutoTokenizer.from_pretrained(model_name, **tok_kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=getattr(torch, dtype), device_map="auto", **tok_kwargs
        ).eval()
        self.dev = next(self.model.parameters()).device
        self.ids_a = self._letter_ids("A")
        self.ids_b = self._letter_ids("B")
        self.use_cache = True
        self.cache_checked = False
        self.prefill_chunk = int(os.environ.get("AI2HUMAN_PREFILL_CHUNK", 1024))
        mc = self.model.config
        mc = getattr(mc, "text_config", mc)
        n_kv = getattr(mc, "num_key_value_heads", None) or mc.num_attention_heads
        head_dim = getattr(mc, "head_dim", None) or mc.hidden_size // mc.num_attention_heads
        elt = torch.finfo(getattr(torch, dtype)).bits // 8
        self.kv_bytes_per_token = 2 * mc.num_hidden_layers * n_kv * head_dim * elt
        self.kv_budget = float(os.environ.get("AI2HUMAN_KV_BUDGET_GB", 1.5)) * 1e9
        # Reuse the prompt cache between a learner's tests (v0.4): the post and transfer tests share one
        # prompt and the delayed test only adds the interference block, so the long lesson is read once.
        self.reuse_cache = os.environ.get("AI2HUMAN_REUSE_CACHE", "1") != "0"
        self._kept = None  # (token ids of the cached prompt, cache)
        # Answer read-out at full precision (v0.4, issue P23). In bfloat16 the model's own logits have
        # 8 significant bits, so the 40 answers of a test fell on about 13 distinct probabilities and
        # many test cases tied. The two answer letters are therefore re-read in float32 from the last
        # hidden state: logit = hidden . W[letter] (+ bias), with the model's own soft-capping if any.
        # The model's bfloat16 logits for the same letters are kept as a check (max difference logged;
        # if they ever disagree by more than 0.75 the float32 read-out is switched off for the run).
        self.fp32 = answer_precision == "float32"
        self.fp32_maxdiff = 0.0
        self._h = None
        if self.fp32:
            head = self.model.get_output_embeddings()
            cfg0 = self.model.config
            self.softcap = getattr(cfg0, "final_logit_softcapping", None)
            if head is None or not hasattr(head, "weight"):
                self.fp32 = False
                log("float32 answer read-out not available for this model (no output layer found); using model logits")
            else:
                ids = torch.tensor(self.ids_a + self.ids_b, device=head.weight.device)
                self.W = head.weight.detach()[ids].float()
                self.b = head.bias.detach()[ids].float() if getattr(head, "bias", None) is not None else None
                head.register_forward_pre_hook(lambda mod, args: setattr(self, "_h", args[0]))

    def _letter_ids(self, L):
        ids = set()
        for s in (L, " " + L):
            t = self.tok.encode(s, add_special_tokens=False)
            if len(t) == 1:
                ids.add(t[0])
        return sorted(ids)

    def _split(self, context: str):
        msgs = [{"role": "user", "content": context + "\n\n" + MARK}]
        try:
            text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            text = context + "\n\n" + MARK + "\nAnswer:"
        pre, post = text.split(MARK)
        return self.tok.encode(pre, add_special_tokens=False), post

    def _score_logits(self, last_logits, last_hidden=None):
        """P(A) renormalised over the two letters. The softmax normaliser cancels in log P(A) - log P(B),
        so only the logits of the letter tokens matter; they are read in float64 arithmetic from float32."""
        torch = self.torch
        na = len(self.ids_a)
        z = last_logits[:, self.ids_a + self.ids_b].float()
        if self.fp32 and last_hidden is not None:
            zf = last_hidden.to(self.W.device).float() @ self.W.T
            if self.b is not None:
                zf = zf + self.b
            if self.softcap:
                zf = torch.tanh(zf / self.softcap) * self.softcap
            diff = float((zf - z.to(zf.device)).abs().max())
            self.fp32_maxdiff = max(self.fp32_maxdiff, diff)
            if diff > 0.75:
                self.log(f"float32 answer read-out disagrees with the model's logits by {diff:.2f}; switched off")
                self.fp32 = False
            else:
                z = zf
        z = z.double()
        a = torch.logsumexp(z[:, :na], -1)
        b = torch.logsumexp(z[:, na:], -1)
        return torch.sigmoid(a - b).cpu().numpy().tolist()

    def _full(self, pre_ids, suffixes, bs=None):
        torch = self.torch
        out = []
        bs = bs or self.bs
        for i in range(0, len(suffixes), bs):
            seqs = [pre_ids + s for s in suffixes[i : i + bs]]
            L = max(map(len, seqs))
            pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
            ids = torch.full((len(seqs), L), pad, dtype=torch.long)
            att = torch.zeros((len(seqs), L), dtype=torch.long)
            for j, s in enumerate(seqs):  # left padding
                ids[j, L - len(s):] = torch.tensor(s)
                att[j, L - len(s):] = 1
            with torch.no_grad():
                # logits_to_keep=1: only the last position is needed (issue P17: full logits over a
                # 10k-token prompt x 151k vocabulary need several GB and were OOM-killed)
                self._h = None
                logits = self.model(input_ids=ids.to(self.dev), attention_mask=att.to(self.dev),
                                    logits_to_keep=1).logits[:, -1]
                h = self._h[:, -1] if self._h is not None and self._h.shape[0] == logits.shape[0] else None
            out += self._score_logits(logits, h)
        return out

    def _prefill(self, pre_ids):
        """Cache of the shared prompt. Reuses the cache kept from the previous call for the longest
        common token prefix (cropping it if needed) and reads only the new tokens."""
        torch = self.torch
        base, start = None, 0
        if self.reuse_cache and self._kept is not None:
            old_ids, old = self._kept
            n = 0
            for x, y in zip(old_ids, pre_ids):
                if x != y:
                    break
                n += 1
            if n >= 64:
                if n < len(old_ids):
                    old.crop(n)
                base, start = old, n
            else:
                del old
        self._kept = None
        with torch.no_grad():
            # Prefill in chunks (issue P17): one pass over a ~11k-token prompt builds a
            # 16 x 11k x 11k attention matrix (about 8 GB) and was OOM-killed. Chunks give the same cache.
            step = self.prefill_chunk
            for c0 in range(start, len(pre_ids), step):
                part = torch.tensor([pre_ids[c0 : c0 + step]], device=self.dev)
                pos = torch.arange(c0, c0 + part.shape[1], device=self.dev).unsqueeze(0)
                base = self.model(input_ids=part, past_key_values=base, position_ids=pos, use_cache=True,
                                  logits_to_keep=1).past_key_values
        if self.reuse_cache:
            self._kept = (list(pre_ids), base)
        return base

    def _cached(self, pre_ids, suffixes):
        torch = self.torch
        base = self._prefill(pre_ids)
        out = []
        P = len(pre_ids)
        # Rows per suffix batch are capped by a memory budget for the copied cache (issue P17, second
        # crash at 13.9 GB in the cue group): each row holds a full copy of the prompt's cache.
        rows = max(1, min(self.bs, int(self.kv_budget // max(1, P * self.kv_bytes_per_token))))
        for i in range(0, len(suffixes), rows):
            chunk = suffixes[i : i + rows]
            L = max(map(len, chunk))
            pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
            ids = torch.full((len(chunk), L), pad, dtype=torch.long)
            att = torch.zeros((len(chunk), P + L), dtype=torch.long)
            att[:, :P] = 1
            for j, s in enumerate(chunk):  # right padding; read logits at each row's last real token
                ids[j, : len(s)] = torch.tensor(s)
                att[j, P : P + len(s)] = 1
            cache = copy.deepcopy(base)
            if len(chunk) > 1:
                cache.batch_repeat_interleave(len(chunk))
            pos = torch.arange(P, P + L, device=self.dev).unsqueeze(0).expand(len(chunk), -1)
            with torch.no_grad():
                self._h = None
                logits = self.model(input_ids=ids.to(self.dev), attention_mask=att.to(self.dev),
                                    past_key_values=cache, position_ids=pos, use_cache=True).logits
            last = torch.stack([logits[j, len(s) - 1] for j, s in enumerate(chunk)])
            h = None
            if self._h is not None and self._h.shape[:2] == logits.shape[:2]:
                h = torch.stack([self._h[j, len(s) - 1] for j, s in enumerate(chunk)])
            out += self._score_logits(last, h)
            del cache, logits
            self._h = None
        del base
        gc.collect()
        _trim_heap()
        return out

    def p_a(self, context, questions):
        pre_ids, post = self._split(context)
        suffixes = [self.tok.encode(q + post, add_special_tokens=False) for q in questions]
        if self.use_cache:
            try:
                res = self._cached(pre_ids, suffixes)
                if not self.cache_checked:  # verify once against the plain path
                    ref = self._full(pre_ids, suffixes[:1], bs=1)
                    diff = max(abs(x - y) for x, y in zip(res[:1], ref))
                    self.cache_checked = True
                    if diff > 0.02:
                        self.log(f"prefix cache disagrees with full pass by {diff:.3f}; disabling cache")
                        self.use_cache = False
                        return self._full(pre_ids, suffixes)
                    self.log(f"prefix cache verified (max diff {diff:.4f})")
                return res
            except Exception as e:
                self.log(f"prefix cache unavailable ({type(e).__name__}: {e}); using full passes")
                self.use_cache = False
        return self._full(pre_ids, suffixes)

    def close(self):
        if self.fp32_maxdiff:
            self.log(f"float32 answer read-out: largest difference from the model's own logits {self.fp32_maxdiff:.3f}")
        self._kept = None
        del self.model
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


class OpenAIBackend(Backend):
    """OpenAI-compatible chat endpoint. Needs logprobs for exact probabilities;
    falls back to the text answer (p = 1 or 0) when logprobs are not returned."""

    def __init__(self, model_name: str, base_url: str | None, log=print):
        from openai import OpenAI

        self.name = model_name
        self.client = OpenAI(base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
                             api_key=os.environ.get("OPENAI_API_KEY", "not-needed"))
        self.log = log
        self.fallbacks = 0

    def _one(self, context, q):
        import time

        for attempt in range(6):
            try:
                r = self.client.chat.completions.create(
                    model=self.name, messages=[{"role": "user", "content": context + "\n\n" + q}],
                    max_tokens=1, temperature=0, logprobs=True, top_logprobs=10, seed=0)
                ch = r.choices[0]
                if ch.logprobs and ch.logprobs.content:
                    pa = pb = 0.0
                    for t in ch.logprobs.content[0].top_logprobs:
                        tok = t.token.strip().upper()
                        if tok == "A":
                            pa += math.exp(t.logprob)
                        elif tok == "B":
                            pb += math.exp(t.logprob)
                    if pa + pb > 0:
                        return pa / (pa + pb)
                self.fallbacks += 1
                txt = (ch.message.content or "").strip().upper()
                return 1.0 if txt.startswith("A") else 0.0 if txt.startswith("B") else 0.5
            except Exception as e:
                wait = 2 ** attempt
                self.log(f"API error ({e}); retry in {wait}s")
                time.sleep(wait)
        raise RuntimeError("API failed 6 times")

    def p_a(self, context, questions):
        return [self._one(context, q) for q in questions]


class MockBackend(Backend):
    """Fake learners for smoke tests and the synthetic validation worlds.
    Their results say NOTHING about real learners. Behaviour by model name:

      mock            reads the context like a diligent student: applies any numbered
                      rules it finds (real or scrambled), and learns a little from the
                      practice cases (5-nearest-neighbour vote)
      mock-strong-*   a student that leans more on written rules (95% vs 75%)
      mock-null       follows an explicit gate rule, but learns nothing from lessons or
                      practice; answers are noise around 0.5
      mock-memorizer  already "knows" the AI's cues whenever real measurement names
                      are shown, with or without a lesson (prior knowledge)
      mock-collapse   follows an explicit gate rule, but otherwise always answers "older"
      mock-random     random answers (fails the known-answer gate)
    """

    RULE = re.compile(r"^\d+\. (.+?): in older adults it tends to be (higher|lower) \(cut-point about (-?[\d.]+)", re.M)
    GATE = re.compile(r"exactly when (.+?) is above (-?[\d.]+)")
    VAL = re.compile(r"^(.+?): (-?[\d.]+)", re.M)

    def __init__(self, model_name: str, cues: dict, panel: list[dict], seed: int = 0):
        self.name = model_name
        kinds = [k for k in ("collapse", "random", "null", "memorizer") if k in model_name]
        self.kind = kinds[0] if kinds else "student"
        self.rule_weight = 0.95 if "strong" in model_name else 0.75  # 'strong' students lean on written rules more
        self.cues = cues["cues"] if isinstance(cues, dict) else cues
        self.label_of = {p["id"]: p["label"] for p in panel}
        self.rng = np.random.default_rng(seed + sum(map(ord, model_name)))
        self._ctx_cache: dict = {}

    def _parse_context(self, context: str):
        key = hash(context)
        if key in self._ctx_cache:
            return self._ctx_cache[key]
        rules = [(m.group(1), m.group(2) == "higher", float(m.group(3))) for m in self.RULE.finditer(context)]
        n = len(rules)
        rules = [(lab, hi, thr, (n - k) / sum(range(1, n + 1))) for k, (lab, hi, thr) in enumerate(rules)]
        cases = []
        for block in context.split("\nCase ")[1:]:
            body, _, ans = block.partition("Correct answer: ")
            vals = {m.group(1).strip(): float(m.group(2)) for m in self.VAL.finditer(body.split(":\n", 1)[-1])}
            older = "older" in ans.split("\n")[0]
            cases.append((vals, older))
        g = self.GATE.search(context)
        out = (rules, cases, (g.group(1), float(g.group(2))) if g else None)
        self._ctx_cache[key] = out
        return out

    @staticmethod
    def _rule_score(rules, vals):
        s = tot = 0.0
        for lab, hi, thr, w in rules:
            v = vals.get(lab)
            if v is None:
                continue
            s += w * ((v > thr) == hi)
            tot += w
        return s / tot if tot else 0.5

    @staticmethod
    def _knn(cases, vals, k=5):
        if len(cases) < k:
            return 0.5
        keys = [kk for kk in vals if all(kk in c[0] for c in cases)]
        if not keys:
            return 0.5
        M = np.array([[c[0][kk] for kk in keys] for c in cases])
        mu, sd = M.mean(0), M.std(0) + 1e-9
        x = (np.array([vals[kk] for kk in keys]) - mu) / sd
        d = (((M - mu) / sd - x) ** 2).sum(1)
        nn = np.argsort(d)[:k]
        return float(np.mean([cases[j][1] for j in nn]))

    def p_a(self, context, questions):
        if self.kind == "collapse" and not self.GATE.search(context):
            return [0.97 if "A) older" in q else 0.03 for q in questions]
        if self.kind == "random":
            p = np.clip(0.5 + self.rng.normal(0, 0.3, len(questions)), 0.01, 0.99)
            return [float(x) if "A) older" in q else float(1 - x) for x, q in zip(p, questions)]
        rules, cases, gate = self._parse_context(context)
        if self.kind in ("null", "collapse") and not gate:
            rules, cases = [], []
        out = []
        for q in questions:
            older_is_a = "A) older" in q
            if q.startswith("Question about adults"):  # prior-knowledge probe
                score = 0.5
                if self.kind == "memorizer":
                    for c in self.cues:
                        if f"the {c['label']} usually" in q:
                            score = 0.9 if c["older_is_higher"] else 0.1
                out.append(score if "A) higher" in q else 1 - score)  # score = P(higher)
                continue
            vals = {m.group(1).strip(): float(m.group(2)) for m in self.VAL.finditer(q.split("TEST CASE:\n", 1)[-1])}
            if gate:
                score = 0.95 if vals.get(gate[0], -1e9) > gate[1] else 0.05
            else:
                parts = []
                if rules:
                    parts.append((self.rule_weight, 0.15 + 0.7 * self._rule_score(rules, vals)))
                if cases:
                    parts.append((1 - self.rule_weight if rules else 1.0, 0.5 + 0.6 * (self._knn(cases, vals) - 0.5)))
                if self.kind == "memorizer" and any(c["label"] in vals for c in self.cues):
                    prior = [(c["label"], c["older_is_higher"], c["threshold"], c["weight"]) for c in self.cues]
                    parts = [(1.0, 0.15 + 0.7 * self._rule_score(prior, vals))]
                score = sum(w * v for w, v in parts) / sum(w for w, _ in parts) if parts else 0.5
            score = float(np.clip(score + self.rng.normal(0, 0.12), 0.01, 0.99))
            out.append(score if older_is_a else 1 - score)
        return out


def make_backend(cfg: dict, model_name: str, cues, panel, log=print) -> Backend:
    ec = cfg["experiment_llm"]
    if ec["backend"] == "mock":
        return MockBackend(model_name, cues, panel, cfg["study"]["master_seed"])
    if ec["backend"] == "openai":
        return OpenAIBackend(model_name, ec.get("openai_base_url"), log)
    return HFBackend(model_name, ec.get("dtype", "bfloat16"), ec.get("batch_size", 8), log,
                     revision=(ec.get("revisions") or {}).get(model_name),
                     answer_precision=ec.get("answer_precision", "float32"))
