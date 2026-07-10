"""Shared frozen text-embedding featurizer for the prompt router.

A single module so the embedding model is loaded once per process and reused
across training (train_router.py) and inference (collect_traces.py,
run_ablation.py) call sites, and so all three agree on the exact pooling.
"""

from __future__ import annotations

import os
import threading
from typing import Any

import numpy as np

DEFAULT_EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"

_cache: dict[str, tuple[Any, Any, str]] = {}
_cache_lock = threading.Lock()


def _load(model_id: str) -> tuple[Any, Any, str]:
    # collect_traces.py calls embed() from many ThreadPoolExecutor workers at
    # once; without this lock, concurrent first-access threads race on
    # transformers' lazy-import machinery (and would otherwise each load a
    # separate full copy of the model).
    if model_id in _cache:
        return _cache[model_id]
    with _cache_lock:
        if model_id not in _cache:
            import torch
            from transformers import AutoModel, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
            model = AutoModel.from_pretrained(model_id, dtype=torch.float32)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device).eval()
            _cache[model_id] = (tokenizer, model, device)
    return _cache[model_id]


def embed(texts: list[str], model_id: str = "", batch_size: int = 32) -> np.ndarray:
    """Embed prompts with a frozen decoder embedding model (last-token pooling).

    Last-token pooling with left-padding is the standard convention for
    decoder-based embedding models like Qwen3-Embedding: the final position
    always holds the true last real token regardless of sequence length.
    """
    import torch

    model_id = model_id or os.environ.get("ROUTER_EMBED_MODEL", DEFAULT_EMBED_MODEL)
    tokenizer, model, device = _load(model_id)
    if not texts:
        return np.zeros((0, model.config.hidden_size), dtype=np.float32)

    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = [str(t) for t in texts[start:start + batch_size]]
        encoded = tokenizer(
            batch, padding=True, truncation=True, max_length=2048, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            last_hidden = model(**encoded).last_hidden_state
        pooled = last_hidden[:, -1, :]
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
        vectors.append(pooled.float().cpu().numpy())
    return np.concatenate(vectors, axis=0)
