"""Two TINY NLLB-shaped models for the CPU smoke test of run_stage19.sh (the Jesuit, 24 Oct 2026; pod rules: every code
path is run end to end on the Beelink before any machine is created).

  <out>/init_tiny   stands in for a050: the real a050 tokenizer (toi_Latn = 256204), a 1-layer M2M100 with the real
                    vocabulary size, saved as safetensors (merge_wise.py reads the stored tensors)
  <out>/base_tiny   stands in for facebook/nllb-200-3.3B: the SAME sentencepiece model but WITHOUT toi_Latn (256,204
                    rows), so train_nllb.py's --new-code toi_Latn --init-from bem_Latn path (the 3.3B arm's phase 1) runs
                    exactly as on the droplet, including the resize and the tied-embedding assert

    python -X utf8 tools/chitonga/train/smoke_tiny_nllb.py <out dir> [a050 folder]
Nothing here is a model anyone should use: random weights, d_model 32.
"""
import json
import os
import shutil
import sys

import torch
from transformers import AutoTokenizer, M2M100Config, M2M100ForConditionalGeneration

out = sys.argv[1]
A050 = sys.argv[2] if len(sys.argv) > 2 else r"models\chitonga-nllb-wise-a050"
torch.manual_seed(0)
os.makedirs(out, exist_ok=True)


def tiny(vocab, src_cfg):
    c = json.load(open(os.path.join(src_cfg, "config.json"), encoding="utf-8"))
    cfg = M2M100Config(vocab_size=vocab, d_model=32, encoder_layers=1, decoder_layers=1, encoder_attention_heads=2,
                       decoder_attention_heads=2, encoder_ffn_dim=64, decoder_ffn_dim=64, max_position_embeddings=1024,
                       pad_token_id=c["pad_token_id"], bos_token_id=c["bos_token_id"], eos_token_id=c["eos_token_id"],
                       decoder_start_token_id=c["decoder_start_token_id"], scale_embedding=True,
                       tie_word_embeddings=True)
    return M2M100ForConditionalGeneration(cfg)


# the a050 stand-in
tok = AutoTokenizer.from_pretrained(A050)
d1 = os.path.join(out, "init_tiny")
m = tiny(len(tok), A050)
m.save_pretrained(d1, safe_serialization=True)
tok.save_pretrained(d1)
# the 3.3B base stand-in: the tokenizer without toi_Latn (a050's added token removed from every tokenizer file)
d2 = os.path.join(out, "base_tiny")
os.makedirs(d2, exist_ok=True)
shutil.copy(os.path.join(A050, "sentencepiece.bpe.model"), d2)
tc = json.load(open(os.path.join(A050, "tokenizer_config.json"), encoding="utf-8"))
tc["additional_special_tokens"] = [t for t in tc["additional_special_tokens"] if t != "toi_Latn"]
tc["added_tokens_decoder"] = {k: v for k, v in tc["added_tokens_decoder"].items() if v["content"] != "toi_Latn"}
json.dump(tc, open(os.path.join(d2, "tokenizer_config.json"), "w", encoding="utf-8"), indent=1)
sm = json.load(open(os.path.join(A050, "special_tokens_map.json"), encoding="utf-8"))
sm["additional_special_tokens"] = [t for t in sm.get("additional_special_tokens", []) if t != "toi_Latn"]
json.dump(sm, open(os.path.join(d2, "special_tokens_map.json"), "w", encoding="utf-8"), indent=1)
tb = AutoTokenizer.from_pretrained(d2)
assert "toi_Latn" not in tb.get_vocab() and len(tb) == len(tok) - 1, (len(tb), len(tok))
m2 = tiny(len(tb), A050)
m2.save_pretrained(d2, safe_serialization=True)
tb.save_pretrained(d2)
print("TINY OK", d1, len(tok), "|", d2, len(tb))
