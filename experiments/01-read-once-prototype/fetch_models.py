"""Download the backbones into the Hugging Face cache (HF_HOME) and verify the safetensors weights.

The Ettin encoders ship only `pytorch_model.bin` on `main`. Hugging Face's SFconvertbot has
opened PRs that add `model.safetensors`; we load from the newest bot PR with
`use_safetensors=True` and check every tensor against `main`'s .bin, which is opened with
`torch.load(weights_only=True)` (restricted unpickler: tensors and primitive types only).

Set HF_HOME to choose where the weights are cached.
"""
import json
import os
import sys
import time

import torch
from huggingface_hub import HfApi, hf_hub_download
from safetensors.torch import load_file

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import MODELS, RESULTS  # noqa: E402

api = HfApi()
out = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "models": {}}
only = sys.argv[1:]
for name, spec in MODELS.items():
    if only and name not in only:
        continue
    rid, rev = spec["repo"], spec["revision"]
    t0 = time.time()
    rec = {"repo": rid, "revision": rev}
    minfo = api.model_info(rid, revision=rev)
    rec["resolved_sha"] = minfo.sha
    files = [s.rfilename for s in minfo.siblings]
    for f in ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
              "model.safetensors"):
        if f in files:
            hf_hub_download(rid, f, revision=rev)
    st_path = hf_hub_download(rid, "model.safetensors", revision=rev)
    rec["safetensors_bytes"] = os.path.getsize(st_path)
    rec["download_s"] = round(time.time() - t0, 1)
    if spec.get("verify_against_main_bin"):
        bin_path = hf_hub_download(rid, "pytorch_model.bin", revision=spec["main_sha"])
        a = load_file(st_path)
        b = torch.load(bin_path, map_location="cpu", weights_only=True)
        b = {k: v for k, v in b.items()}
        shared = sorted(set(a) & set(b))
        mism = [k for k in shared if a[k].shape != b[k].shape or not torch.equal(a[k], b[k].to(a[k].dtype))]
        only_bin = sorted(set(b) - set(a))
        # safetensors cannot store shared tensors twice: the converter keeps the MLM `decoder.weight`
        # and drops the tied input embedding. Check the tie holds (both in the .bin and across files).
        tied = {}
        for k in only_bin:
            if k.endswith("tok_embeddings.weight") and "decoder.weight" in a:
                tied[k] = {"bin_embedding_equals_bin_decoder": bool(torch.equal(b[k], b.get("decoder.weight", b[k] * 0 + 1))),
                           "bin_embedding_equals_safetensors_decoder": bool(torch.equal(b[k], a["decoder.weight"]))}
        rec["verify"] = {
            "tensors_safetensors": len(a), "tensors_bin": len(b), "shared": len(shared),
            "only_in_safetensors": sorted(set(a) - set(b))[:10], "only_in_bin": only_bin[:10],
            "mismatched": mism[:10], "tied_checks": tied,
            "all_bin_tensors_recovered": (not mism) and all(
                v["bin_embedding_equals_safetensors_decoder"] for v in tied.values()) and len(only_bin) == len(tied),
            "bin_sha": spec["main_sha"],
        }
        del a, b
        # the .bin is only needed for this check; drop it from the cache to save space
        try:
            os.remove(bin_path)
        except OSError:
            pass
    out["models"][name] = rec
    print(name, json.dumps(rec))

path = os.path.join(RESULTS, "fetch_models.json")
prev = {}
if os.path.exists(path):
    with open(path) as fh:
        prev = json.load(fh)
prev.setdefault("models", {}).update(out["models"])
prev["timestamp"] = out["timestamp"]
with open(path, "w") as fh:
    json.dump(prev, fh, indent=1)
