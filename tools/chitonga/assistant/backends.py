"""The real model and toolbox calls behind the pipeline, in one place, all imported lazily (importing this file loads
nothing; the unit tests replace the whole object with a stub of the same methods).

  translate(text, direction, model)  translator_service.Service.translate, term guard on (as served), keep=True so the
                                      service's own n-best come back as 'alts' (re-render material; single-piece input)
  translate_batch(texts, dir, model) the batched path below (render.py uses it for every answer)
  readback(toi) / readback_batch     R's plain decode toi>en (the checker stays R whatever renders; negation sweep idem)
  similarity(a, b) / similarity_batch  bge-small cosine via h3_select.embed after h3_select.neutral (gender-neutral)
  words(toi)                         attest.line (the allowlist word check; not-Tonga list; contact evidence)
  word_verdict(w)                    attest.word(w)[0]
  intake_note(text)                  attest.note (the unknown words of a Tonga request)
  negation(en, toi)                  tools/chitonga/train/negation_sweep.auto_flags (neg_dropped / obl_dropped / neg_added)
  en_has_neg(text)                   negation_test.EN_NEG
  en_vocab()                         h2_text._en_vocab (as translator_service.detect_direction)
  draft(messages)                    the local UNTOUCHED Gemma 4 E4B instruct through llama-server (OpenAI endpoint)
  claude_draft(messages)             Claude through the Claude Code CLI, headless, from an empty folder. 22 Oct 2026:
                                      the API key in .env.local answers 401; Chris: "use the subscription for now"
  timing                             seconds per step of the current request (the pipeline resets it per request)

BATCHED PATH (22 Oct 2026, the Jesuit; measured on the Beelink the same day, renderer a050, six answer sentences):
one Service.translate call per sentence took 12-13 s each; the six as one text took 27.9 s. Of those, the candidates took
13.5 s (soup 5-best 3.6, soup beam 4 3.2, each seed reader 3.4) and the service's three-reader pool check the rest.
translate_batch keeps the served soup's own beams (beam 4 first = what the released page gives, then the 5-best) and
drops (a) the two seed readers: with max_attempts 4, render.py never reached them (pool positions 6-7), and (b) the pool
check: render.py's own checks (words, kept terms, numbers, negation, read-back by R) are the gate; the service's cheap
alarms (copy-through, loop, notation lost) are kept per candidate. The term guard and the translation memory stay on.

Model swap: translator_service.SOUP is a module constant read when the 'soup' reader first loads, so each renderer gets
its own Service built (and first used) with SOUP set to that model, under one lock; SOUP is put back after each call.
"""
import collections
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
LLM = os.path.normpath(os.path.join(HERE, "..", "llm"))
TRAIN = os.path.normpath(os.path.join(HERE, "..", "train"))
for p in (LLM, TRAIN):
    if p not in sys.path:
        sys.path.insert(0, p)

_LOCK = threading.RLock()


class Real:
    def __init__(self, cfg, models_dir=None, renderers=None):
        import config as C
        self.cfg = cfg
        self.models_dir = models_dir or C.MODELS_DIR
        self.renderers = renderers or C.RENDERERS
        self._svc = {}
        self._vocab = None
        self.timing = collections.defaultdict(float)
        self.last_gemma = None
        self.last_claude = None

    def _tick(self, key, t0):
        self.timing[key] = round(self.timing[key] + time.time() - t0, 2)

    # translator ---------------------------------------------------------------
    def _dir(self, model):
        return os.path.join(self.models_dir, self.renderers[model])

    def _service(self, model):
        import translator_service as TS
        if model not in self._svc:
            self._svc[model] = TS.Service()
        return TS, self._svc[model]

    def translate(self, text, direction, model="R"):
        t0 = time.time()
        TS, svc = self._service(model)
        with _LOCK:
            old = TS.SOUP
            TS.SOUP = self._dir(model)
            try:
                r = svc.translate(text, direction, keep=True)
            finally:
                TS.SOUP = old
        alts = []
        pcs = r.get("pieces") or []
        if len(pcs) == 1 and pcs[0].get("candidates") and r.get("plan") is not None:
            q = pcs[0]
            for c in q["candidates"]:
                t = TS.assemble(r["plan"], [TS.render(q, c["text"])])
                if t and t != r["text"]:
                    alts.append(t)
        self._tick("translate_" + direction, t0)
        return {"text": r.get("text") or "", "alarms": list(r.get("alarms") or []), "alts": alts,
                "stage": r.get("stage"), "model": model, "t": r.get("t")}

    def translate_batch(self, texts, direction, model="R"):
        """[{'text', 'alts', 'alarms': [per candidate, aligned with [text] + alts]}] per text (BATCHED PATH above)"""
        t0 = time.time()
        TS, svc = self._service(model)
        with _LOCK:
            old, old_cfg = TS.SOUP, svc.cfg
            TS.SOUP = self._dir(model)
            svc.cfg = dict(svc.cfg, seeds=())
            svc._tl.ctx = ""                   # the term guard reads each piece itself (no trigger across sentences)
            try:
                plans, prep, owner = [], [], []
                for i, text in enumerate(texts):
                    plan, pieces = TS.layout(text, direction, svc.cfg["split_words"])
                    plans.append(plan)
                    for p in pieces:
                        pt, cap, stop = TS.phone_form(p["text"])
                        prep.append({"src": p["text"], "text": pt, "map": p["map"], "counts": p["counts"],
                                     "caps": p["caps"], "stop_added": stop, "cap_added": cap})
                        owner.append(i)
                mem = [svc.memory(q["src"], direction) if svc.cfg.get("memory", True) else None for q in prep]
                cands = self._candidates(TS, svc, prep, direction) if prep else []
            finally:
                TS.SOUP, svc.cfg = old, old_cfg
        out = []
        for i, text in enumerate(texts):
            per_piece = []
            for j in [j for j, o in enumerate(owner) if o == i]:
                q = prep[j]
                cs = [c["text"] for c in cands[j]]
                if mem[j]:
                    cs = [mem[j]["text"]] + [c for c in cs if c != mem[j]["text"]]
                per_piece.append([(TS.render(q, c), c, q) for c in cs])
            if not per_piece:                  # nothing to translate (numbers / notation only): copied by the layout
                out.append({"text": TS.assemble(plans[i], []), "alts": [], "alarms": [[]], "model": model})
                continue
            texts_i, alarms_i = [], []
            for k in range(max(len(x) for x in per_piece)):   # rank-aligned: candidate k of every piece
                chosen = [x[min(k, len(x) - 1)] for x in per_piece]
                t = TS.assemble(plans[i], [c[0] for c in chosen])
                if t in texts_i:
                    continue
                al = []
                for _, raw, q in chosen:
                    if TS.has_loop(raw, q["text"]):
                        al.append("loop")
                    elif TS.is_copy(q["text"], raw):
                        al.append("copy_through")
                    al += [f"notation_lost: {x}" for x in TS.lost(raw, q)]
                texts_i.append(t)
                alarms_i.append(al)
            out.append({"text": texts_i[0], "alts": texts_i[1:], "alarms": alarms_i, "model": model})
        self._tick("translate_batch_" + direction, t0)
        return out

    def _candidates(self, TS, svc, prep, direction):
        """[[{'text'}], ...] per piece: ONE soup decode, beam 4 with 4 hypotheses. Its first hypothesis is the beam-4
        1-best that Service.candidates puts first ('soup:b4', what the released page gives): the same search, so the
        served first candidate is unchanged. Service.candidates decodes the soup twice (5-best at beam 5, then beam 4);
        render.py tries at most 4 candidates, all of which this one decode gives (22 Oct evening: the merge roughly
        halves the decode; measured in the LOGBOOK entry). Kept from Service.candidates: the no-repeat decode for a
        looping beam, and the stand-in route for a name the soup loses (h2_text.safe_en)."""
        import h2_text as T
        texts = [q["text"] for q in prep]
        k = self.cfg.get("max_attempts", 4)
        soup = svc.nbest("soup", texts, direction, k, beam=4)
        gi = [i for i in range(len(prep)) if TS.has_loop(soup[i][0][0], texts[i])]
        guard = svc.nbest("soup", [texts[i] for i in gi], direction, 1, guard=True) if gi else []
        out = []
        for i, q in enumerate(prep):
            items = [t for t, _ in soup[i]]
            if i in gi:
                items.append(guard[gi.index(i)][0][0])
            if direction == "en>toi" and T.en_names(q["text"]):
                nm = T.en_names(q["text"])
                if any(n.lower() not in soup[i][0][0].lower() for n in nm):
                    t2, _ = T.safe_en(q["text"], lambda x: svc.nbest("soup", [x], direction, 1, beam=4)[0][0][0])
                    items.append(t2)
            out.append([{"text": t} for t in dict.fromkeys(x.strip() for x in items if x and x.strip())])
        return out

    def readback(self, toi):
        return self.readback_batch([toi])[0]

    def readback_batch(self, tois):
        if not tois:
            return []
        t0 = time.time()
        TS, svc = self._service(self.cfg["readback"])
        with _LOCK:
            old = TS.SOUP
            TS.SOUP = self._dir(self.cfg["readback"])
            try:
                r = svc.reader("soup")
                with svc._lock_for("soup"):
                    res = r._run(list(tois), TS.LANG["toi"], TS.LANG["en"], 256)
            finally:
                TS.SOUP = old
        self._tick("readback", t0)
        return res

    # checks -------------------------------------------------------------------
    def similarity(self, a, b):
        return self.similarity_batch([(a, b)])[0]

    def similarity_batch(self, pairs):
        if not pairs:
            return []
        t0 = time.time()
        import h3_select as S
        v = S.embed([S.neutral(x) for a, b in pairs for x in (a, b)])
        self._tick("similarity", t0)
        return [float((v[2 * i] * v[2 * i + 1]).sum()) for i in range(len(pairs))]

    def words(self, toi):
        t0 = time.time()
        import attest
        try:
            return attest.line(toi)
        finally:
            self._tick("attest", t0)

    def word_verdict(self, w):
        import attest
        return attest.word(w)[0]

    def word_note(self, w):
        import attest
        return attest.word(w)

    def intake_note(self, text):
        import attest
        return attest.note(text)

    def negation(self, en, toi):
        import negation_sweep as NS
        return NS.auto_flags({"dir": "en>toi", "src": en}, toi)

    def en_has_neg(self, text):
        import negation_test as N
        return bool(N.EN_NEG.search(text or ""))

    def en_vocab(self):
        if self._vocab is None:
            import translator_service as TS
            self._vocab = TS._en_vocab()
        return self._vocab

    # the composers --------------------------------------------------------------
    def draft(self, messages):
        """the untouched Gemma (llama-server). Returns the text; self.last_gemma holds time and token counts"""
        t0 = time.time()
        # thinking OFF (22 Oct showcase: with it on, 3 of 4 drafts spent the 900-token budget on hidden reasoning and the
        # JSON was cut off; 'enable_thinking': false gives the same answer without it). The weights stay untouched.
        body = {"messages": messages, "temperature": 0, "max_tokens": 1500,
                "chat_template_kwargs": {"enable_thinking": False}}
        req = urllib.request.Request(self.cfg["gemma_url"].rstrip("/") + "/v1/chat/completions",
                                     data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        r = json.load(urllib.request.urlopen(req, timeout=self.cfg.get("gemma_timeout_s", 300)))
        self.last_gemma = {"t": round(time.time() - t0, 2), "usage": r.get("usage"), "model": r.get("model"),
                           "cost_usd": 0.0, "billing": "local (Beelink)"}
        return r["choices"][0]["message"]["content"]

    def claude_draft(self, messages):
        """Claude through the Claude Code CLI (subscription), headless. Our system prompt REPLACES the CLI's own; no
        tools, no settings, no MCP, no saved session; run from an empty folder, so no CLAUDE.md or repo file is seen.
        The request has already been scrubbed of names and numbers (dual.scrub). Returns the reply text;
        self.last_claude holds time, model, tokens and the CLI's notional list-price cost (the subscription pays; the
        figure is logged so a later API run can be costed honestly)."""
        t0 = time.time()
        sysmsg = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        user = "\n\n".join(m["content"] for m in messages if m["role"] != "system")
        cwd = self.cfg.get("claude_cwd") or os.path.join(tempfile.gettempdir(), "zamai_claude_empty")
        os.makedirs(cwd, exist_ok=True)
        if os.listdir(cwd):
            raise RuntimeError(f"claude_cwd is not empty: {cwd}")
        cmd = [self.cfg.get("claude_cli", "claude"), "-p", "--output-format", "json", "--system-prompt", sysmsg,
               "--tools", "", "--setting-sources", "", "--strict-mcp-config", "--no-session-persistence",
               "--model", self.cfg.get("claude_model", "opus")]
        p = subprocess.run(cmd, input=user, capture_output=True, text=True, encoding="utf-8", cwd=cwd,
                           timeout=self.cfg.get("claude_timeout_s", 120))
        try:
            d = json.loads(p.stdout)
        except json.JSONDecodeError:
            raise RuntimeError(f"claude CLI rc {p.returncode}: {(p.stderr or p.stdout)[:200]}")
        mu = d.get("modelUsage") or {}
        main = max(mu, key=lambda k: mu[k].get("outputTokens", 0)) if mu else None
        self.last_claude = {"t": round(time.time() - t0, 2), "model": main,
                            "usage": {"input": (d.get("usage") or {}).get("input_tokens"),
                                      "output": (d.get("usage") or {}).get("output_tokens")},
                            "cost_usd_notional": d.get("total_cost_usd"), "billing": "subscription (Claude Code CLI)"}
        if d.get("is_error"):
            raise RuntimeError(f"claude CLI error: {str(d.get('result'))[:200]}")
        return d.get("result") or ""
