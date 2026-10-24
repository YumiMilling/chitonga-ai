"""ON THE POD (or the Beelink, for the smoke): every pinned GGUF answers every frozen request body (24 Oct 2026, the Jesuit).

Step A of the reader bake-off (internal/chitonga_train/runs/2026-10-24_reader_bakeoff/RULE.md). Reads pack/bodies.jsonl
(the exact chat requests the harness would send, made on the Beelink by reader_bakeoff.py capture) and
pack/models.json (repo, file, sha256 per arm), and for each arm in order:
  1. downloads the GGUF from Hugging Face at datacentre speed (never through the Beelink), asserts its sha256;
  2. starts llama-server with the SAME flags the harness uses on the Beelink (reader: -c 4096 -np 1 --jinja
     --reasoning off --reasoning-budget 0, as v14_eval_harness.serve; digest: -c 8192 -np 1, as notes_digest.Gemma);
  3. one SMOKE request first (a reply must come back before any result counts), then every body in turn, greedy as
     the body says; writes out/<tag>.jsonl {key, raw, ms, http, err, timings};
  4. deletes the GGUF (disk) and goes on.
Progress lines use the Training panel's grammar (tools/beelink/training.py): 'TRAIN <tag>', 'THROUGHPUT CHECK step N'
(N = bodies answered), '<tag> TRAINING DONE', 'RUN FAILED <tag> <why>', and at the end F20_DONE / F20_FAILED.
Robustness arms (role 'robustness', the Q8_0 files) are skipped once --soft-min minutes have passed, so a slow card
cuts robustness, never the decision arms.

    python3 pod_reader_bakeoff.py --pack /root/pack --out /root/out --mdir /root/m --llama <llama-server> [--only TAG,..]
"""
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import urllib.request


def log(msg):
    print(msg, flush=True)


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def download(m, mdir, token):
    from huggingface_hub import hf_hub_download  # noqa: PLC0415
    t0 = time.time()
    p = hf_hub_download(m["repo"], m["file"], local_dir=mdir, token=token or None)
    got = sha_file(p)
    if got != m["sha256"]:
        os.remove(p)
        raise RuntimeError(f"sha256 {got[:12]} != pinned {m['sha256'][:12]}")
    log(f"DOWNLOADED {m['tag']} {m['repo']}/{m['file']} {os.path.getsize(p) / 1e9:.2f} GB in {time.time() - t0:.0f}s sha256 OK")
    return p


def up(url):
    try:
        with urllib.request.urlopen(url + "/health", timeout=5) as f:
            return json.load(f).get("status") == "ok"
    except Exception:  # noqa: BLE001
        return False


def serve(llama, gguf, port, kind, logf, slots=1, flags=None):
    """kind reader: the harness's flags (-c 4096 -np 1 --jinja --reasoning off); digest: notes_digest.Gemma's (-c 8192);
    slots > 1 (sizing only): continuous batching over `slots` parallel slots, 3072 context each"""
    args = [llama, "-m", gguf, "-ngl", "99", "--host", "127.0.0.1", "--port", str(port), "-np", str(slots)]
    ctx = "4096" if slots == 1 else str(3072 * slots)
    if kind == "reader":
        # flags: a post-hoc retry may replace the harness's reader flags (24 Oct: '--reasoning off --reasoning-budget 0'
        # made Granite 4.2 and Nemotron 3 Nano fail the JSON grammar on token 38; '--jinja' alone works); never the default
        args += ["-c", ctx] + (list(flags) if flags is not None else ["--jinja", "--reasoning", "off", "--reasoning-budget", "0"])
    else:
        args += ["-c", "8192"]
    p = subprocess.Popen(args, stdout=logf, stderr=subprocess.STDOUT, cwd=os.path.dirname(llama) or None)
    url = f"http://127.0.0.1:{port}"
    t0 = time.time()
    while time.time() - t0 < 600:
        if up(url):
            return p, url
        if p.poll() is not None:
            raise RuntimeError(f"llama-server exited rc={p.returncode} while loading")
        time.sleep(2)
    p.kill()
    raise RuntimeError("llama-server not up in 600 s")


def stop(p):
    if p and p.poll() is None:
        p.terminate()
        try:
            p.wait(30)
        except Exception:  # noqa: BLE001
            p.kill()


class MemWatch:
    """peak GPU memory (nvidia-smi, every second) and peak resident memory of the server process (psutil) while an
    arm runs; both minus nothing: the idle GPU baseline is read before the server starts (RULE v2 section 4)"""

    def __init__(self, pid=None):
        import threading  # noqa: PLC0415
        self.pid, self.gpu_peak, self.rss_peak, self.stop_ = pid, None, None, False
        self.t = threading.Thread(target=self.run, daemon=True)

    @staticmethod
    def gpu_used():
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=10)
            return int(r.stdout.split()[0])
        except Exception:  # noqa: BLE001
            return None

    def run(self):
        try:
            import psutil  # noqa: PLC0415
            proc = psutil.Process(self.pid) if self.pid else None
        except Exception:  # noqa: BLE001
            proc = None
        while not self.stop_:
            g = self.gpu_used()
            if g is not None:
                self.gpu_peak = max(self.gpu_peak or 0, g)
            if proc is not None:
                try:
                    self.rss_peak = max(self.rss_peak or 0, proc.memory_info().rss // 2 ** 20)
                except Exception:  # noqa: BLE001
                    proc = None
            time.sleep(1)

    def start(self):
        self.t.start()
        return self

    def end(self):
        self.stop_ = True
        self.t.join(5)


def speed_of(rows):
    """latency p50 / p90 (ms) and summed prompt / generation tokens per second from llama-server's timings"""
    ms = sorted(r["ms"] for r in rows if r.get("raw") is not None)
    tm = [r["timings"] for r in rows if r.get("timings")]
    pn, pms = sum(t.get("prompt_n") or 0 for t in tm), sum(t.get("prompt_ms") or 0 for t in tm)
    gn, gms = sum(t.get("predicted_n") or 0 for t in tm), sum(t.get("predicted_ms") or 0 for t in tm)
    q = lambda f: ms[max(0, math.ceil(len(ms) * f) - 1)] if ms else None   # noqa: E731  nearest rank
    return {"n": len(ms), "p50_ms": q(0.5), "p90_ms": q(0.9), "prompt_tok_s": round(pn / pms * 1000, 1) if pms else None,
            "gen_tok_s": round(gn / gms * 1000, 1) if gms else None, "prompt_tokens_mean": round(pn / len(tm)) if tm else None,
            "gen_tokens_mean": round(gn / len(tm)) if tm else None}


def ask(url, body, timeout):
    t0 = time.time()
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as f:
            r = json.loads(f.read())
        ch = r["choices"][0]
        return {"raw": ch["message"].get("content"), "finish": ch.get("finish_reason"), "http": 200,
                "prompt_tokens": (r.get("usage") or {}).get("prompt_tokens"),
                "ms": round((time.time() - t0) * 1000), "err": None,
                "timings": {k: (r.get("timings") or {}).get(k) for k in ("prompt_n", "prompt_ms", "predicted_n", "predicted_ms")}}
    except Exception as e:  # noqa: BLE001
        return {"raw": None, "http": getattr(e, "code", None), "ms": round((time.time() - t0) * 1000),
                "err": repr(e)[:200], "timings": None}


def sizing(a, tag, g, reader, logf):
    """information only (coordinator, for Chris: "how many teachers one server could serve"): the real captured requests
    fired concurrently (closed loop, c clients) at -np 8 and -np 16; capacity = the best req/s whose p95 stayed under
    3 s; prefix cache = prompt tokens computed vs prompt tokens sent (the system prompt is shared)"""
    import random  # noqa: PLC0415
    import threading  # noqa: PLC0415
    order = [b["body"] for b in reader if b["kind"] == "reader"]
    random.Random(7).shuffle(order)
    out = {}
    for slots, levels in ((8, (1, 2, 4, 8)), (16, (8, 12, 16))):
        srv, url = serve(a.llama, g, a.port, "reader", logf, slots=slots, flags=getattr(a, "_flags", None))
        try:
            ask(url, order[0], a.timeout)
            lv, sent, computed = [], 0, 0
            for c in levels:
                n = min(a.sizing_req, max(len(order), c))
                todo = list(range(n))
                lat, res, lk = [], [], threading.Lock()

                def worker():
                    while True:
                        with lk:
                            if not todo:
                                return
                            i = todo.pop(0)
                        r = ask(url, order[i % len(order)], a.timeout)
                        with lk:
                            res.append(r)
                t0 = time.time()
                th = [threading.Thread(target=worker) for _ in range(c)]
                for t in th:
                    t.start()
                for t in th:
                    t.join()
                el = time.time() - t0
                ok = [r for r in res if r["raw"] is not None]
                ms = sorted(r["ms"] for r in ok)
                p95 = ms[max(0, math.ceil(len(ms) * 0.95) - 1)] / 1000 if ms else None
                sent += sum(r.get("prompt_tokens") or 0 for r in ok)
                computed += sum((r.get("timings") or {}).get("prompt_n") or 0 for r in ok)
                lv.append({"clients": c, "n": len(res), "errors": len(res) - len(ok), "rps": round(len(ok) / el, 2),
                           "p50_s": round(ms[len(ms) // 2] / 1000, 2) if ms else None, "p95_s": round(p95, 2) if p95 else None})
                log(f"SIZING {tag} np={slots} clients={c} {lv[-1]['rps']} req/s p95 {lv[-1]['p95_s']} s ({lv[-1]['errors']} errors)")
            good = [x for x in lv if x["p95_s"] is not None and x["p95_s"] < 3.0 and not x["errors"]]
            best = max(good, key=lambda x: x["rps"]) if good else None
            nreq = sum(x["n"] - x["errors"] for x in lv)
            out[str(slots)] = {"levels": lv, "cap_rps": best["rps"] if best else 0.0, "cap_p95_s": best["p95_s"] if best else None,
                               "prompt_tokens_mean": round(sent / nreq) if nreq else None,
                               "prompt_computed_mean": round(computed / nreq) if nreq else None,
                               "cache_saving": round(1 - computed / sent, 3) if sent else None}
        finally:
            stop(srv)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default="/root/pack")
    ap.add_argument("--out", default="/root/out")
    ap.add_argument("--mdir", default="/root/m")
    ap.add_argument("--llama", required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--soft-min", type=float, default=240)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--keep-gguf", action="store_true")
    ap.add_argument("--sizing-req", type=int, default=64, help="requests per concurrency level in the sizing step")
    ap.add_argument("--no-sizing", action="store_true")
    a = ap.parse_args()
    t_all = time.time()
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.mdir, exist_ok=True)
    token = os.environ.get("HF_TOKEN", "")
    bodies = [json.loads(l) for l in open(os.path.join(a.pack, "bodies.jsonl"), encoding="utf-8") if l.strip()]
    models = json.load(open(os.path.join(a.pack, "models.json"), encoding="utf-8"))["models"]
    if a.only:
        want = a.only.split(",")
        models = [m for m in models if m["tag"] in want]
    reader = [b for b in bodies if b["kind"] == "reader"] + [b for b in bodies if b["kind"] == "reader_aux"]
    digest = [b for b in bodies if b["kind"] == "digest"]
    log(f"PLAN {len(models)} arms x ({len(reader)} reader + {len(digest)} digest bodies); soft limit {a.soft_min} min for robustness arms")
    man = {"llama": a.llama, "arms": {}, "bodies": len(bodies)}
    try:
        man["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                                    capture_output=True, text=True, timeout=20).stdout.strip() or None
    except Exception:  # noqa: BLE001
        man["gpu"] = "no NVIDIA GPU (CPU run)"
    try:
        man["llama_version"] = subprocess.run([a.llama, "--version"], capture_output=True, text=True, timeout=60).stderr.strip()[-200:]
    except Exception as e:  # noqa: BLE001
        man["llama_version"] = repr(e)[:100]
    failed = 0
    for m in models:
        tag = m["tag"]
        el = (time.time() - t_all) / 60
        if m.get("role") == "robustness" and el > a.soft_min:
            log(f"SKIPPED {tag} (robustness arm, {el:.0f} min > soft limit {a.soft_min})")
            man["arms"][tag] = {"skipped": "time"}
            continue
        log(f"TRAIN {tag}")
        rec = {"repo": m["repo"], "file": m["file"], "sha256": m["sha256"], "t0": time.time()}
        srv = None
        logf = open(os.path.join(a.out, f"llama_{tag}.log"), "w")
        try:
            g = download(m, a.mdir, token)
            rec["sha256_verified"] = True
            out = open(os.path.join(a.out, f"{tag}.jsonl"), "w", encoding="utf-8", newline="\n")
            rec["gb"] = round(os.path.getsize(g) / 1e9, 2)
            for kind, group in (("reader", reader), ("digest", digest)):
                if not group:
                    continue
                t1 = time.time()
                base_gpu = MemWatch.gpu_used()
                srv, url = serve(a.llama, g, a.port, kind, logf, flags=m.get("server_flags"))
                mw = MemWatch(srv.pid).start()
                rows = []
                rec[f"load_s_{kind}"] = round(time.time() - t1, 1)
                if kind == "reader":
                    sm = ask(url, group[0]["body"], a.timeout)
                    if sm["raw"] is None:
                        raise RuntimeError(f"smoke request failed: {sm['err']}")
                    log(f"SMOKE OK {tag} ({sm['ms']} ms): {str(sm['raw'])[:80]!r}")
                for i, b in enumerate(group, 1):
                    r = ask(url, b["body"], a.timeout if kind == "reader" else 600)
                    if b["kind"] == "reader":
                        rows.append(r)
                    out.write(json.dumps(dict(key=b["key"], kind=b["kind"], **r), ensure_ascii=False) + "\n")
                    out.flush()
                    if kind == "reader" and (i % 10 == 0 or i == len(group)):
                        log(f"THROUGHPUT CHECK step {i} ({tag}, {r['ms']} ms last)")
                mw.end()
                if kind == "reader":
                    sp = speed_of(rows)
                    sp.update(gpu_mib_peak=mw.gpu_peak, gpu_mib_idle=base_gpu,
                              gpu_mib_arm=(mw.gpu_peak - base_gpu) if mw.gpu_peak is not None and base_gpu is not None else None,
                              server_rss_mib_peak=mw.rss_peak)
                    rec["speed"] = sp
                    log(f"SPEED {tag} p50 {sp['p50_ms']} ms p90 {sp['p90_ms']} ms prompt {sp['prompt_tok_s']} tok/s gen "
                        f"{sp['gen_tok_s']} tok/s GPU {sp['gpu_mib_arm']} MiB RSS {sp['server_rss_mib_peak']} MiB")
                stop(srv)
                srv = None
            out.close()
            if m.get("role") not in ("robustness", "retry") and not a.no_sizing:
                try:
                    rec["sizing"] = sizing(a, tag, g, reader, logf)
                except Exception as e:  # noqa: BLE001  (information only: never costs the arm its replies)
                    rec["sizing"] = {"failed": repr(e)[:200]}
                    log(f"SIZING FAILED {tag} {repr(e)[:150]}")
            n_err = sum(1 for l in open(os.path.join(a.out, f"{tag}.jsonl"), encoding="utf-8") if json.loads(l)["err"])
            rec.update(seconds=round(time.time() - rec["t0"]), errors=n_err)
            log(f"{tag} TRAINING DONE ({len(reader)} + {len(digest)} bodies, {n_err} errors, {rec['seconds']} s)")
        except Exception as e:  # noqa: BLE001
            failed += 1
            rec["failed"] = repr(e)[:300]
            log(f"RUN FAILED {tag} {repr(e)[:200]}")
        finally:
            stop(srv)
            logf.close()
            if not a.keep_gguf:
                for f in os.listdir(a.mdir):
                    if f.endswith(".gguf"):
                        try:
                            os.remove(os.path.join(a.mdir, f))
                        except OSError:
                            pass
        rec.pop("t0", None)
        man["arms"][tag] = rec
        json.dump(man, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    man["minutes"] = round((time.time() - t_all) / 60, 1)
    json.dump(man, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    done = [t for t, r in man["arms"].items() if "seconds" in r]
    if not done:
        log("F20_FAILED no arm finished")
        sys.exit(1)
    log(f"F20_DONE {len(done)} arms answered, {failed} failed, {man['minutes']} min")


if __name__ == "__main__":
    main()
