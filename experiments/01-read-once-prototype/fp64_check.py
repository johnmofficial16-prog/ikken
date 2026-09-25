"""fp32 vs fp64 parity through the `ikken` library's check_parity (Ettin-68m, ModernBERT-base).

In fp32, packed-vs-alone hidden-state differences on real checkpoints are rounding noise amplified by
large activations. Running the same check in fp64 shows how far they shrink. Note: transformers'
ModernBERT applies rotary embeddings in float32 even in an fp64 run (`apply_rotary_pos_emb` casts q and
k with `.float()`), so an fp64 run is not fully fp64.

Usage: python fp64_check.py   (weights must already be in the HF cache: run fetch_models.py first)
Output: results/fp64_parity_check.json
"""
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import RESULTS, load_backbone  # noqa: E402
from synth import long_state_ids, question_pool  # noqa: E402

from ikken import check_parity, choice_block  # noqa: E402

out = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "torch": torch.__version__,
       "state_tokens": 510, "questions": 20, "question_pool_seed": 1234, "runs": []}
for name in ("ettin-68m", "modernbert-base"):
    tok, enc, _ = load_backbone(name)
    blocks = [choice_block(tok, q["text"], q["options"]) for q in question_pool(20, seed=1234)]
    state = long_state_ids(tok, 510, seed=0)
    for dtype in (torch.float32, torch.float64):
        r = check_parity(enc.to(dtype), state, blocks, cls_id=tok.cls_token_id, sep_id=tok.sep_token_id,
                         pad_id=tok.pad_token_id)
        out["runs"].append({"backbone": name, "dtype": str(dtype).replace("torch.", ""), **r})
        print(out["runs"][-1], flush=True)
with open(os.path.join(RESULTS, "fp64_parity_check.json"), "w") as fh:
    json.dump(out, fh, indent=1)
