"""Phase 4 smoke check: load a Qwen3.5 checkpoint and run a forward pass.

Read-only reproduction aid: it does not touch any repository source file.
Usage: python check_model_load.py <model_dir> <tag>
"""
import json
import os
import sys
import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

path, tag = sys.argv[1], sys.argv[2]
t0 = time.time()
print(f"[{tag}] path={path}", flush=True)

cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
print(f"[{tag}] config ok: {cfg.model_type} layers={getattr(cfg,'num_hidden_layers',None)} "
      f"hidden={getattr(cfg,'hidden_size',None)} vocab={getattr(cfg,'vocab_size',None)}", flush=True)
print(f"[{tag}] architectures={getattr(cfg,'architectures',None)}", flush=True)

tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
print(f"[{tag}] tokenizer ok: vocab={len(tok)} eos={tok.eos_token_id} "
      f"chat_template={'yes' if tok.chat_template else 'no'}", flush=True)

model = AutoModelForCausalLM.from_pretrained(
    path, dtype=torch.bfloat16, trust_remote_code=True,
).to("cuda:0").eval()
mem = torch.cuda.max_memory_allocated() / 2**30
n = sum(p.numel() for p in model.parameters())
print(f"[{tag}] loaded in {time.time()-t0:.1f}s params={n/1e9:.2f}B peak_gpu={mem:.1f}GiB", flush=True)

# forward + backward: checks that the GDN/hybrid path computes finite values
ids = tok("Customer: I need to change my flight. Agent:", return_tensors="pt").to("cuda:0")
with torch.autocast("cuda", dtype=torch.bfloat16):
    out = model(**ids, labels=ids["input_ids"])
loss = out.loss
loss.backward()
gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9).item()
finite = bool(torch.isfinite(loss).all())
print(f"[{tag}] fwd/bwd ok loss={loss.item():.4f} finite={finite} grad_norm={gnorm:.4f}", flush=True)

# one greedy generation step: checks the KV/GDN state machinery on a real delay
with torch.no_grad():
    g = model.generate(**ids, max_new_tokens=16, do_sample=False)
gen = tok.decode(g[0][ids["input_ids"].shape[1]:], skip_special_tokens=False)
print(f"[{tag}] generate ok: {gen[:120]!r}", flush=True)

print(json.dumps({"tag": tag, "params_B": round(n / 1e9, 3), "loss": round(float(loss), 4),
                  "peak_gpu_GiB": round(mem, 1), "finite": finite,
                  "grad_norm": round(gnorm, 4), "load_s": round(time.time() - t0, 1)}))
