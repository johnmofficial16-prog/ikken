"""Kaggle T4 smoke test for the read-once encoder.

Run in a Kaggle notebook with Accelerator = "GPU T4 x2" and Internet on. Each test is
independent: a failure is recorded, not fatal. Results are rewritten to smoke_results.json after
every test, so a session that dies part-way still leaves everything measured so far.

Tests, in priority order:
  env               versions and GPUs
  exactness         read-once packed vs single-question outputs on CUDA, in fp32 and fp16, plus the
                    sequence-index sliding-window negative control
  activations       per-module max |activation| in fp32 (headroom against the fp16 max of 65,504)
                    and a non-finite check in an fp16 forward pass
  train_*           training throughput, peak memory and stability (fp16 AMP; one fp32 and one
                    Laya-style cross-encoder run for comparison)
  sdpa_backends     which SDPA kernels accept the custom mask on this GPU, and their speed
  flex              whether FlexAttention compiles and matches dense SDPA on this GPU
  resume            checkpoint save, reload and continue gives the same loss

Usage on Kaggle:        python smoke.py
Usage for a dry run:    python smoke.py --dry-run    (CPU, tiny shapes; checks the code paths only)
"""
import argparse
import gc
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from readonce import (ChoiceModel, choice_loss, collate, encode_block,  # noqa: E402
                      pack_cross, pack_readonce, build_readonce_batch)
from synth import long_state_ids, make_dataset, question_pool  # noqa: E402

# Pinned safetensors revisions (the same ones experiment 01 verified against each repo's main .bin).
MODELS = {
    "ettin-32m": ("jhu-clsp/ettin-encoder-32m", "27b98f697261919d35f3f995fee5c690921e5404"),
    "ettin-68m": ("jhu-clsp/ettin-encoder-68m", "d446589aef55657267b913c9bcce6c98b69061c7"),
    "modernbert-base": ("answerdotai/ModernBERT-base", "8949b909ec900327062f0ebf497f51aef5e6f0c8"),
}
FP16_MAX = 65504.0
OUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(OUT_DIR, "smoke_results.json")
RESULTS = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "tests": {}}


# ----------------------------------------------------------------------------- bookkeeping
def save():
    with open(OUT, "w") as fh:
        json.dump(RESULTS, fh, indent=1, default=str)


def free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def record(name, fn, deadline):
    if time.time() > deadline:
        RESULTS["tests"][name] = {"ok": False, "skipped": "time budget exhausted"}
        save()
        print(f"[SKIP] {name}", flush=True)
        return
    t0 = time.time()
    try:
        RESULTS["tests"][name] = {"ok": True, "result": fn()}
    except Exception as e:  # noqa: BLE001 - every failure is data here
        RESULTS["tests"][name] = {"ok": False, "error": repr(e)[:500], "trace": traceback.format_exc()[-3000:]}
    RESULTS["tests"][name]["seconds"] = round(time.time() - t0, 1)
    save()
    free()
    t = RESULTS["tests"][name]
    print(f"[{'OK ' if t['ok'] else 'ERR'}] {name} ({t['seconds']} s)", flush=True)
    if not t["ok"]:
        print("      " + t["error"], flush=True)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def dev(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def median(xs):
    s = sorted(xs)
    return s[len(s) // 2] if s else float("nan")


# ----------------------------------------------------------------------------- models and data
def load(name, device):
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    repo, rev = MODELS[name]
    tok = AutoTokenizer.from_pretrained(repo, revision=rev)
    # Load through the MLM class: these safetensors store the tied input embedding only as
    # decoder.weight, so a bare AutoModel load would leave the embeddings randomly initialised.
    mlm = AutoModelForMaskedLM.from_pretrained(repo, revision=rev, use_safetensors=True,
                                               attn_implementation="sdpa", dtype=torch.float32)
    enc = mlm.model
    if not torch.equal(enc.embeddings.tok_embeddings.weight, mlm.decoder.weight):
        raise RuntimeError("input embedding not tied to decoder.weight for " + name)
    return tok, enc.to(device)


def new_model(enc, device, train):
    torch.manual_seed(0)  # identical scorer initialisation in every test
    m = ChoiceModel(enc).to(device)
    return m.train() if train else m.eval()


def packed_rows(tok, S, K, n_rows, seed):
    """n_rows read-once rows, each with an S-token state block ([CLS] + S-2 + [SEP]) and K questions."""
    items, labels = [], []
    for r in range(n_rows):
        sids = long_state_ids(tok, S - 2, seed=seed + r)
        qs = question_pool(K, seed=seed * 1000 + r)
        items.append(pack_readonce(tok, sids, [encode_block(tok, q) for q in qs]))
        labels.append([q["answer"] for q in qs])
    return collate(items, tok.pad_token_id, labels)


def cross_rows(tok, S, K, n_states, seed):
    """Laya-style rows: one row per (state, question), so each state is re-read K times."""
    items, labels = [], []
    for r in range(n_states):
        sids = long_state_ids(tok, S - 2, seed=seed + r)
        for q in question_pool(K, seed=seed * 1000 + r):
            items.append(pack_cross(tok, sids, encode_block(tok, q)))
            labels.append([q["answer"]])
    return collate(items, tok.pad_token_id, labels)


# ----------------------------------------------------------------------------- tests
def t_env(device):
    import huggingface_hub
    import tokenizers
    import transformers
    info = {"python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "transformers": transformers.__version__,
            "tokenizers": tokenizers.__version__, "huggingface_hub": huggingface_hub.__version__,
            "device": str(device)}
    if torch.cuda.is_available():
        info["gpus"] = [{"name": torch.cuda.get_device_name(i),
                         "capability": list(torch.cuda.get_device_capability(i)),
                         "mem_gb": round(torch.cuda.get_device_properties(i).total_memory / 1e9, 2)}
                        for i in range(torch.cuda.device_count())]
        try:
            info["nvidia_smi"] = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"],
                capture_output=True, text=True, timeout=30).stdout
        except Exception as e:  # noqa: BLE001
            info["nvidia_smi"] = repr(e)
    return info


def gpu_state():
    """GPU 0 clocks, temperature, power and active throttle reasons (nvidia-smi), or None off-GPU."""
    if not torch.cuda.is_available():
        return None
    q = "clocks.sm,clocks.max.sm,temperature.gpu,power.draw,clocks_throttle_reasons.active"
    try:
        out = subprocess.run(["nvidia-smi", "-i", "0", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout.strip()
        return dict(zip(q.split(","), [x.strip() for x in out.split(",")]))
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}


def t_exactness(name, device, n_records, long_S, long_K):
    """Each question's probabilities in a packed pass vs the same question packed alone."""
    tok, enc = load(name, device)
    model = new_model(enc, device, train=False)
    cases = [(ex["state"], ex["questions"]) for ex in make_dataset(n_records, seed=123)]
    long_state = tok.decode(long_state_ids(tok, long_S - 2, seed=7))
    cases.append((long_state, question_pool(long_K, seed=7)))
    modes = [("fp32", False, "position"), ("fp16", True, "position"),
             ("fp32_negative_control_index_window", False, "index")]
    out = {}
    for label, half, window_on in modes:
        if half and device.type != "cuda":
            continue
        maxdiff, agree, n = 0.0, 0, 0
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.float16, enabled=half):
            for state, qs in cases:
                lp, _ = model(dev(build_readonce_batch(tok, [state], [qs]), device),
                              mode="readonce", window_on=window_on)
                pp = torch.softmax(lp.float(), -1)
                for j, q in enumerate(qs):
                    ls, _ = model(dev(build_readonce_batch(tok, [state], [[q]]), device), mode="readonce")
                    ps = torch.softmax(ls.float(), -1)
                    k = len(q["options"])
                    maxdiff = max(maxdiff, (pp[j, :k] - ps[0, :k]).abs().max().item())
                    agree += int(pp[j, :k].argmax().item() == ps[0, :k].argmax().item())
                    n += 1
        out[label] = {"max_abs_prob_diff": maxdiff, "argmax_agree": agree, "n_questions": n}
    out["cases"] = {"short_records": n_records, "long_state_tokens": long_S, "long_state_questions": long_K}
    return out


def t_activations(name, device, S, K, n_rows):
    """Largest |activation| per module (fp32) and non-finite values in an fp16 forward pass."""
    tok, enc = load(name, device)
    model = new_model(enc, device, train=False)
    batch = dev(packed_rows(tok, S, K, n_rows, seed=11), device)
    stats = {}

    def mk(nm):
        def hook(_mod, _inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            if torch.is_tensor(t) and t.is_floating_point():
                a = t.detach().abs()
                fin = torch.isfinite(a)
                mx = a[fin].max().item() if fin.any() else float("nan")
                prev = stats.get(nm, {"max": 0.0, "nonfinite": 0})
                stats[nm] = {"max": max(prev["max"], mx), "nonfinite": prev["nonfinite"] + int((~fin).sum().item())}
        return hook

    hooks = []
    for nm, mod in model.encoder.named_modules():
        is_leaf = len(list(mod.children())) == 0
        is_layer = nm.startswith("layers.") and nm.count(".") == 1  # residual stream after each layer
        if nm and (is_leaf or is_layer):
            hooks.append(mod.register_forward_hook(mk(nm)))
    out = {"shape": {"S": S, "K": K, "rows": n_rows, "L": int(batch["input_ids"].shape[1])}}
    try:
        for label, half in (("fp32", False), ("fp16", True)):
            if half and device.type != "cuda":
                continue
            stats.clear()
            with torch.no_grad(), torch.autocast(device.type, dtype=torch.float16, enabled=half):
                logits, _ = model(batch, mode="readonce")
            gmax = max(v["max"] for v in stats.values())
            top = sorted(stats.items(), key=lambda kv: -(kv[1]["max"] if kv[1]["max"] == kv[1]["max"] else 0))[:8]
            out[label] = {"global_max_abs": round(gmax, 1), "headroom_vs_fp16_max": round(FP16_MAX / gmax, 2),
                          "nonfinite_values": sum(v["nonfinite"] for v in stats.values()),
                          "logits_finite": bool(torch.isfinite(logits[batch["q_mask"]]).all().item()),
                          "top_modules": [(k, round(v["max"], 1)) for k, v in top]}
    finally:
        for h in hooks:
            h.remove()
    return out


def t_train(name, device, S, K, bs, steps, amp, arch="readonce", n_batches=4):
    """Median step time, tokens/s, questions/s, peak memory and loss stability over `steps` steps."""
    tok, enc = load(name, device)
    model = new_model(enc, device, train=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5)
    use_amp = amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    if arch == "readonce":
        batches = [dev(packed_rows(tok, S, K, bs, seed=1000 + i), device) for i in range(n_batches)]
    else:  # bs = number of states; rows = bs * K
        batches = [dev(cross_rows(tok, S, K, bs, seed=1000 + i), device) for i in range(n_batches)]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    warm, times, losses, nonfinite = 3, [], [], 0
    try:
        for step in range(steps + warm):
            b = batches[step % n_batches]
            sync(device)
            t0 = time.perf_counter()
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                logits, _ = model(b, mode="readonce" if arch == "readonce" else "stock")
                loss = choice_loss(logits, b["labels"])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sync(device)
            if step >= warm:
                times.append(time.perf_counter() - t0)
            lv = loss.item()
            losses.append(lv)
            nonfinite += int(not math.isfinite(lv))
    except torch.cuda.OutOfMemoryError as e:
        return {"oom": True, "error": repr(e)[:300], "arch": arch, "S": S, "K": K, "bs": bs}
    gpu_after = gpu_state()
    tokens = [int(b["attention_mask"].sum().item()) for b in batches]
    questions = [int(b["labels"].numel()) for b in batches]
    step_s = median(times)
    return {"arch": arch, "amp_fp16": use_amp, "S": S, "K": K, "bs": bs,
            "rows_per_step": int(batches[0]["input_ids"].shape[0]), "L": int(batches[0]["input_ids"].shape[1]),
            "median_step_ms": round(step_s * 1000, 1),
            "tokens_per_s": round(sum(tokens) / len(tokens) / step_s, 1),
            "questions_per_s": round(sum(questions) / len(questions) / step_s, 1),
            "peak_mem_gb": round(torch.cuda.max_memory_allocated(device) / 1e9, 2) if device.type == "cuda" else None,
            "loss_first": round(losses[0], 4), "loss_last": round(losses[-1], 4),
            "nonfinite_losses": nonfinite,
            "grad_scaler_scale_end": scaler.get_scale() if use_amp else None,
            "gpu_after": gpu_after}


def t_sdpa_backends(name, device, S, K, bs):
    """Forward+backward with the custom mask under each SDPA backend (fp16 autocast)."""
    from torch.nn.attention import SDPBackend, sdpa_kernel
    tok, enc = load(name, device)
    model = new_model(enc, device, train=True)
    b = dev(packed_rows(tok, S, K, bs, seed=5), device)
    out = {"S": S, "K": K, "bs": bs}
    for bk in (SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH):
        try:
            times = []
            with sdpa_kernel(bk):
                for i in range(6):
                    sync(device)
                    t0 = time.perf_counter()
                    with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                        logits, _ = model(b, mode="readonce")
                        loss = choice_loss(logits, b["labels"])
                    loss.backward()
                    model.zero_grad(set_to_none=True)
                    sync(device)
                    if i >= 2:
                        times.append(time.perf_counter() - t0)
            out[bk.name] = {"ok": True, "median_fwd_bwd_ms": round(median(times) * 1000, 1)}
        except Exception as e:  # noqa: BLE001
            out[bk.name] = {"ok": False, "error": repr(e)[:400]}
        free()
    return out


def t_flex(device, S, K, q_len=40, heads=8, head_dim=64):
    """FlexAttention with the read-once block mask: does it compile here, and does it match SDPA?"""
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
    seg_list = [0] * S
    for k in range(1, K + 1):
        seg_list += [k] * q_len
    seg = torch.tensor(seg_list, device=device)
    L = seg.numel()
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    q, k_, v = (torch.randn(1, heads, L, head_dim, device=device, dtype=dtype) for _ in range(3))
    dense = ((seg[None, :] == 0) | (seg[None, :] == seg[:, None]))[None, None]
    ref = F.scaled_dot_product_attention(q, k_, v, attn_mask=dense)
    res = {"L": L}

    def mask_mod(b, h, qi, ki):
        return (seg[ki] == 0) | (seg[ki] == seg[qi])

    try:
        bm = create_block_mask(mask_mod, B=None, H=None, Q_LEN=L, KV_LEN=L, device=device)
        fn = torch.compile(flex_attention)
        o = fn(q, k_, v, block_mask=bm)
        sync(device)
        res["compiled_ok"] = True
        res["max_abs_diff_vs_sdpa"] = (o.float() - ref.float()).abs().max().item()
        ft, st = [], []
        for _ in range(10):
            sync(device)
            t0 = time.perf_counter()
            fn(q, k_, v, block_mask=bm)
            sync(device)
            ft.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            F.scaled_dot_product_attention(q, k_, v, attn_mask=dense)
            sync(device)
            st.append(time.perf_counter() - t0)
        res["flex_median_ms"] = round(median(ft) * 1000, 3)
        res["sdpa_dense_median_ms"] = round(median(st) * 1000, 3)
    except Exception as e:  # noqa: BLE001
        res["compiled_ok"] = False
        res["error"] = repr(e)[:600]
    return res


def t_resume(name, device):
    """One step, save, one more step; reload the checkpoint and take the same step again."""
    tok, enc = load(name, device)
    model = new_model(enc, device, train=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5)
    b = dev(packed_rows(tok, 512, 4, 4, seed=21), device)

    def step(m, o):
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits, _ = m(b, mode="readonce")
            loss = choice_loss(logits, b["labels"])
        o.zero_grad(set_to_none=True)
        loss.backward()
        o.step()
        return loss.item()

    step(model, opt)
    path = os.path.join(OUT_DIR, "resume_ckpt.pt")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict()}, path)
    l_continued = step(model, opt)
    _, enc2 = load(name, device)
    model2 = new_model(enc2, device, train=True)
    ck = torch.load(path, map_location=device)  # our own file, written above
    model2.load_state_dict(ck["model"])
    opt2 = torch.optim.AdamW(model2.parameters(), lr=3e-5)
    opt2.load_state_dict(ck["opt"])
    l_resumed = step(model2, opt2)
    size_mb = round(os.path.getsize(path) / 1e6, 1)
    os.remove(path)
    return {"loss_continued": l_continued, "loss_resumed": l_resumed,
            "abs_diff": abs(l_continued - l_resumed), "checkpoint_mb": size_mb}


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="CPU, tiny shapes: checks code paths only")
    ap.add_argument("--budget-min", type=float, default=50.0, help="skip remaining tests after this")
    ap.add_argument("--throughput-reps", type=int, default=0,
                    help="if > 0: only run the read-once vs per-question training-throughput pair this many "
                         "times, alternating which goes first, then stop")
    args = ap.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.dry_run else "cpu")
    deadline = time.time() + args.budget_min * 60
    RESULTS["mode"] = "dry-run" if args.dry_run else "full"
    d = args.dry_run

    record("env", lambda: t_env(device), deadline)
    if args.throughput_reps > 0:
        pair = [("readonce", 2 if d else 16), ("cross", 2 if d else 4)]  # Ettin-68m, K=8, fp16 on GPU
        S_tp, steps_tp = (128, 2) if d else (512, 30)
        for r in range(args.throughput_reps):
            for arch, bs in (pair if r % 2 == 0 else pair[::-1]):
                record(f"tp_r{r}_{arch}", lambda arch=arch, bs=bs:
                       t_train("ettin-68m", device, S_tp, 8, bs, steps_tp, True, arch), deadline)
        RESULTS["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        save()
        return
    record("exactness_ettin-68m", lambda: t_exactness("ettin-68m", device, n_records=2 if d else 6,
                                                      long_S=256 if d else 2048, long_K=3 if d else 10), deadline)
    act_models = ["ettin-68m"] if d else ["ettin-68m", "modernbert-base", "ettin-32m"]
    for nm in act_models:
        record(f"activations_{nm}", lambda nm=nm: t_activations(nm, device, S=256 if d else 2048,
                                                                 K=2 if d else 8, n_rows=1 if d else 2), deadline)

    steps = 2 if d else 20
    if d:
        train_cfgs = [("ettin-68m", 128, 2, 2, True, "readonce")]
    else:
        train_cfgs = [("ettin-68m", 512, 8, 16, True, "readonce"),
                      ("ettin-68m", 2048, 8, 4, True, "readonce"),
                      ("ettin-32m", 512, 8, 16, True, "readonce"),
                      ("ettin-32m", 2048, 8, 4, True, "readonce"),
                      ("modernbert-base", 512, 8, 8, True, "readonce"),
                      ("modernbert-base", 2048, 8, 2, True, "readonce"),
                      ("ettin-68m", 512, 8, 16, False, "readonce"),   # fp32 comparison
                      ("ettin-68m", 512, 8, 4, True, "cross")]        # Laya-style: 4 states x 8 questions
    for nm, S, K, bs, amp, arch in train_cfgs:
        label = f"train_{arch}_{nm}_S{S}_K{K}_bs{bs}_{'fp16' if amp else 'fp32'}"
        record(label, lambda nm=nm, S=S, K=K, bs=bs, amp=amp, arch=arch:
               t_train(nm, device, S, K, bs, steps, amp, arch), deadline)

    record("sdpa_backends_ettin-68m", lambda: t_sdpa_backends("ettin-68m", device, S=256 if d else 2048,
                                                              K=2 if d else 8, bs=1 if d else 2), deadline)
    record("flex", lambda: t_flex(device, S=256 if d else 2048, K=2 if d else 8), deadline)
    record("resume_ettin-32m", lambda: t_resume("ettin-68m" if d else "ettin-32m", device), deadline)

    RESULTS["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save()
    ok = sum(1 for t in RESULTS["tests"].values() if t.get("ok"))
    print(f"\nDone: {ok}/{len(RESULTS['tests'])} tests OK. Results: {OUT}", flush=True)


if __name__ == "__main__":
    main()
