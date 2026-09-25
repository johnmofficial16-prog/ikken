"""Shared helpers for experiment 01: environment capture, timing stats, backbone loading.

Every script records the machine state (power source, plan, background CPU, versions) next to
its numbers, because a 15 W laptop shared with other agents is a moving target.
"""
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from importlib import metadata

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)

# Weights come from Hugging Face's SFconvertbot safetensors PRs (main ships only a pickle .bin);
# fetch_models.py checks them tensor-by-tensor against main's .bin.
MODELS = {
    "ettin-17m": {"repo": "jhu-clsp/ettin-encoder-17m", "revision": "59c53d9a19ee484b200676d55e4ae240528b2bb9",  # SFconvertbot PR #5
                  "main_sha": "987607455c61e7a5bbc85f7758e0512ea6d0ae4c", "verify_against_main_bin": True},
    "ettin-32m": {"repo": "jhu-clsp/ettin-encoder-32m", "revision": "27b98f697261919d35f3f995fee5c690921e5404",  # bot PR #4
                  "main_sha": "1b8ba06455dd44f80fc9c1ca9e22806157a57379", "verify_against_main_bin": True},
    "ettin-68m": {"repo": "jhu-clsp/ettin-encoder-68m", "revision": "d446589aef55657267b913c9bcce6c98b69061c7",  # bot PR #4
                  "main_sha": "ac19ae4bc51093b31c475665ac872a936d056cc2", "verify_against_main_bin": True},
    "modernbert-base": {"repo": "answerdotai/ModernBERT-base",
                        "revision": "8949b909ec900327062f0ebf497f51aef5e6f0c8"},
}

PKGS = ("torch", "transformers", "tokenizers", "huggingface_hub", "safetensors", "numpy", "psutil")


def pkg_versions():
    out = {}
    for p in PKGS:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    return out


def power_state():
    try:
        scheme = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True,
                                text=True, timeout=15).stdout.strip()
    except Exception as e:
        scheme = "unavailable (%s)" % e
    bat = None
    try:
        import psutil
        b = psutil.sensors_battery()
        bat = None if b is None else {"percent": b.percent, "plugged_in": b.power_plugged}
    except Exception:
        pass
    return {"power_scheme": scheme, "battery": bat}


def env_info(sample_cpu=True, **extra):
    import psutil
    import torch
    vm = psutil.virtual_memory()
    info = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cpus": psutil.cpu_count(logical=True),
        "ram_total_gb": round(vm.total / 1e9, 2),
        "ram_available_gb": round(vm.available / 1e9, 2),
        "versions": pkg_versions(),
        "torch_threads": torch.get_num_threads(),
        "torch_cpu_capability": torch.backends.cpu.get_cpu_capability(),
    }
    info.update(power_state())
    if sample_cpu:
        # busy % across all logical CPUs from other processes, sampled before we start
        info["background_cpu_percent_1s"] = psutil.cpu_percent(interval=1.0)
    info.update(extra)
    return info


def pct(xs, q):
    s = sorted(xs)
    if not s:
        return float("nan")
    k = (len(s) - 1) * q / 100.0
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def stats_ms(samples_s):
    ms = [x * 1000.0 for x in samples_s]
    return {
        "n": len(ms),
        "median_ms": round(statistics.median(ms), 2),
        "p90_ms": round(pct(ms, 90), 2),
        "mean_ms": round(statistics.fmean(ms), 2),
        "stdev_ms": round(statistics.stdev(ms), 2) if len(ms) > 1 else 0.0,
        "min_ms": round(min(ms), 2),
        "max_ms": round(max(ms), 2),
        "cv_percent": round(100 * statistics.stdev(ms) / statistics.fmean(ms), 1) if len(ms) > 1 else 0.0,
    }


def load_backbone(name, attn="sdpa", offline=True):
    """Return (tokenizer, encoder, spec). Encoder is the bare ModernBertModel in fp32, eval mode."""
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    if offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    spec = MODELS[name]
    tok = AutoTokenizer.from_pretrained(spec["repo"], revision=spec["revision"])
    # Load through the MLM class: the safetensors files store the tied input embedding only as
    # `decoder.weight`, so a bare AutoModel load would leave the embeddings randomly initialised.
    mlm, info = AutoModelForMaskedLM.from_pretrained(
        spec["repo"], revision=spec["revision"], use_safetensors=True, attn_implementation=attn,
        dtype=torch.float32, output_loading_info=True)
    missing = [k for k in info.get("missing_keys", []) if k]
    if missing:
        raise RuntimeError("missing weights for %s: %s" % (name, missing[:10]))
    if not torch.equal(mlm.model.embeddings.tok_embeddings.weight, mlm.decoder.weight):
        raise RuntimeError("input embedding is not tied to decoder.weight for %s" % name)
    enc = mlm.model
    enc.eval()
    return tok, enc, spec


def backbone_summary(enc):
    c = enc.config
    n_params = sum(p.numel() for p in enc.parameters())
    n_emb = enc.embeddings.tok_embeddings.weight.numel()
    return {
        "class": type(enc).__name__, "model_type": c.model_type, "hidden_size": c.hidden_size,
        "num_layers": c.num_hidden_layers, "num_heads": c.num_attention_heads,
        "intermediate_size": c.intermediate_size, "layer_types": list(c.layer_types),
        "local_attention": getattr(c, "local_attention", None),
        "sliding_window_half": getattr(c, "sliding_window", None),
        "max_position_embeddings": c.max_position_embeddings,
        "rope_parameters": c.rope_parameters, "vocab_size": c.vocab_size,
        "params_total": n_params, "params_non_embedding": n_params - n_emb,
        "attn_implementation": c._attn_implementation,
    }


def dump(name, obj):
    path = os.path.join(RESULTS, name)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, default=str)
    return path
