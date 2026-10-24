"""WiSE-FT weight interpolation of two checkpoints of ONE model (22 Oct 2026, the Jesuit; Chris: "Go" on step 1 of
council/NEXT_ROUND_2026-10-22_jesuit.md). Rules in internal/chitonga_train/runs/2026-10-22_merge/PREREG.md.

    python -X utf8 tools/chitonga/train/merge_wise.py <R folder> <K folder> <alpha> <out folder>

theta = (1 - alpha) * R + alpha * K, for every stored floating tensor (Wortsman et al., "Robust fine-tuning of zero-shot
models", CVPR 2022). Valid here because K was fine-tuned FROM R, so both sit in one basin.

WHY streamed and hand-written safetensors: R and K are 5.5 GB each in fp32 and the Beelink has 28 GB shared with
everything else. The output shard header (names, shapes, offsets) is known from the parents' metadata, so it is written
first and each tensor is computed, written and dropped one at a time: peak memory is about three copies of the largest
tensor (the 256,205 x 1,024 embedding, 1 GB), not three models.

WHY this is free of the soup bug (stage15b_soup.py): NLLB / M2M100 ties four state-dict keys (model.shared,
encoder.embed_tokens, decoder.embed_tokens, lm_head) to ONE tensor, and save_pretrained stores that tensor ONCE. Working
on the stored tensors, not on model.state_dict(), the tie is interpolated exactly once. Asserted: R and K store the same
names, shapes and dtypes; non-float tensors are equal and copied; the parents differ; every written tensor, read back
from disk, equals (1 - alpha) * R + alpha * K of fresh reads (max abs error < 1e-6); the reloaded model keeps the
embedding / lm_head tie and its embedding equals the merged stored tensor. Prints MERGE_OK last; anything else fails.
"""
import json
import os
import shutil
import struct
import sys

import torch
from safetensors import safe_open

DT = {torch.float32: "F32", torch.float16: "F16", torch.bfloat16: "BF16", torch.int64: "I64", torch.int32: "I32",
      torch.bool: "BOOL"}
NBYTES = {"F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "BOOL": 1}


def shards(folder):
    idx = os.path.join(folder, "model.safetensors.index.json")
    if os.path.exists(idx):
        wm = json.load(open(idx, encoding="utf-8"))["weight_map"]
        files = sorted(set(wm.values()))
        return files, {f: sorted(k for k, v in wm.items() if v == f) for f in files}
    with safe_open(os.path.join(folder, "model.safetensors"), "pt") as f:
        return ["model.safetensors"], {"model.safetensors": sorted(f.keys())}


def merged(tr, tk, alpha):
    if not tr.is_floating_point():
        assert torch.equal(tr, tk), "a non-float tensor differs between the parents"
        return tr
    return ((1.0 - alpha) * tr.float() + alpha * tk.float()).to(tr.dtype)


def main(rdir, kdir, alpha, out):
    assert 0.0 < alpha < 1.0, alpha
    os.makedirs(out, exist_ok=True)
    files, keys = shards(rdir)
    kfiles, kkeys = shards(kdir)
    assert files == kfiles and keys == kkeys, "R and K are not stored the same way"
    differ = total = 0
    for fn in files:
        with safe_open(os.path.join(rdir, fn), "pt") as fr, safe_open(os.path.join(kdir, fn), "pt") as fk:
            meta, off = {}, 0
            for k in keys[fn]:
                sr, sk = fr.get_slice(k), fk.get_slice(k)
                assert sr.get_shape() == sk.get_shape() and sr.get_dtype() == sk.get_dtype(), k
                n = NBYTES[sr.get_dtype()]
                for d in sr.get_shape():
                    n *= d
                meta[k] = {"dtype": sr.get_dtype(), "shape": sr.get_shape(), "data_offsets": [off, off + n]}
                off += n
            meta["__metadata__"] = {"format": "pt", "wise_ft": f"(1-{alpha})*R + {alpha}*K"}
            head = json.dumps(meta, separators=(",", ":")).encode("utf-8")
            head += b" " * ((8 - len(head) % 8) % 8)
            with open(os.path.join(out, fn), "wb") as w:
                w.write(struct.pack("<Q", len(head)))
                w.write(head)
                for k in keys[fn]:
                    tr, tk = fr.get_tensor(k), fk.get_tensor(k)
                    differ += int(tr.is_floating_point() and not torch.equal(tr, tk))
                    total += 1
                    m = merged(tr, tk, alpha).contiguous()
                    assert DT[m.dtype] == meta[k]["dtype"], k
                    b = m.view(torch.uint8).numpy().tobytes() if m.dim() else m.reshape(1).view(torch.uint8).numpy().tobytes()
                    assert len(b) == meta[k]["data_offsets"][1] - meta[k]["data_offsets"][0], k
                    w.write(b)
                    del tr, tk, m, b
        print(f"written {fn}: {len(keys[fn])} tensors", flush=True)
    assert differ > 0, "R and K are identical: nothing to merge"
    for f in os.listdir(rdir):
        p = os.path.join(rdir, f)
        if os.path.isfile(p) and not f.endswith(".safetensors") and f not in ("README.md", "LICENSE.md"):
            shutil.copy(p, os.path.join(out, f))
    open(os.path.join(out, "README.md"), "w", encoding="utf-8").write(
        f"# chitonga-nllb WiSE-FT merge, alpha {alpha}\n\n(1 - {alpha}) * R + {alpha} * K, every stored tensor; "
        "R = zamai-zambia/chitonga-nllb-s13, K = PRIVATE_HF_ORG/chitonga-nllb-s16-fullA-soup-step8000. "
        "Built on the Beelink by tools/chitonga/train/merge_wise.py, 22 Oct 2026. Judged in "
        "internal/chitonga_train/runs/2026-10-22_merge/.\n")
    # check from disk against fresh reads of the parents
    worst = 0.0
    n = 0
    for fn in files:
        with safe_open(os.path.join(rdir, fn), "pt") as fr, safe_open(os.path.join(kdir, fn), "pt") as fk, \
                safe_open(os.path.join(out, fn), "pt") as fo:
            assert sorted(fo.keys()) == keys[fn], fn
            for k in keys[fn]:
                o, tr, tk = fo.get_tensor(k), fr.get_tensor(k), fk.get_tensor(k)
                if o.is_floating_point():
                    worst = max(worst, (o.float() - ((1.0 - alpha) * tr.float() + alpha * tk.float())).abs().max().item())
                else:
                    assert torch.equal(o, tr), k
                n += 1
                del o, tr, tk
    assert worst < 1e-6, f"merge differs from the interpolation of its parents by {worst}"
    # the tie, through transformers
    from transformers import AutoModelForSeq2SeqLM
    mo = AutoModelForSeq2SeqLM.from_pretrained(out, torch_dtype=torch.float32)
    emb = mo.get_input_embeddings().weight
    assert emb.data_ptr() == mo.lm_head.weight.data_ptr(), "the embedding / output tie was broken"
    assert mo.model.encoder.embed_tokens.weight.data_ptr() == emb.data_ptr() == mo.model.decoder.embed_tokens.weight.data_ptr()
    ekey = [k for ks in keys.values() for k in ks if k.endswith(("shared.weight", "embed_tokens.weight", "lm_head.weight"))]
    fn = [f for f in files if ekey[0] in keys[f]][0]
    with safe_open(os.path.join(out, fn), "pt") as fo:
        assert torch.equal(fo.get_tensor(ekey[0]), emb.detach()), "the loaded embedding is not the merged stored tensor"
    print(f"merge a={alpha}: {n} stored tensors checked from disk against the parents (max error {worst:.1e}); "
          f"{differ} of {total} differ between R and K; tied embedding stored as {ekey} and interpolated once", flush=True)
    print("MERGE_OK", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4])
