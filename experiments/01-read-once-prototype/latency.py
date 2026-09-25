"""Experiment 2: CPU latency vs number of questions K, read-once packing vs Laya-style re-reading.

read-once : 1 row  = [CLS] state [SEP] Q1 ... QK, custom masks + parallel positions (stock HF
            ModernBertModel, masks passed as a per-layer-type dict)
baseline  : K rows = [CLS] Qi state [SEP], stock HF path (2D padding mask), batched in chunks of up
            to BS_MAX rows (fewer when the [B,1,L,L] SDPA mask would exceed ~1 GB of float32)

Both include the scorer head. Input building (tokenise questions + pack + collate) is timed separately.
Reps are interleaved (read-once, baseline, read-once, ...) so background noise hits both. Each
config gets a warm-up, then reps until max(min_reps, time budget) or max_reps.
A baseline config whose estimated time (K x the measured K=1 time) exceeds SKIP_S is not run; it
is recorded as skipped with that estimate (an estimate, not a measurement).

Usage: python latency.py MODEL [S,S,...] [K,K,...] [tag]
"""
import gc
import sys
import time

import torch

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from common import backbone_summary, dump, env_info, load_backbone, power_state, stats_ms  # noqa: E402
from readonce import ChoiceModel, collate, encode_block, pack_cross, pack_readonce  # noqa: E402
from synth import long_state_ids, question_pool  # noqa: E402
import prefixkv  # noqa: E402

BS_MAX = int(__import__("os").environ.get("RO_BS_MAX", "16"))  # RO_BS_MAX=1: unbatched, no padding
SKIP_S = 240.0        # skip a baseline config if its estimated time per rep exceeds this
ONE_REP_S = float(__import__("os").environ.get("RO_ONE_REP_S", "20"))  # above this est. time per rep: one timed rep, no warm-up
MASK_BYTES = 1.0e9    # cap for B * L^2 * 4 bytes in the baseline's float SDPA mask


def build_readonce(tok, state_ids, qs):
    blocks = [encode_block(tok, q) for q in qs]
    return collate([pack_readonce(tok, state_ids, blocks)], tok.pad_token_id)


def build_baseline(tok, state_ids, qs):
    items = [pack_cross(tok, state_ids, encode_block(tok, q)) for q in qs]
    L = max(len(it["ids"]) for it in items)
    bs = max(1, min(BS_MAX, int(MASK_BYTES // (4 * L * L))))
    return [collate(items[i:i + bs], tok.pad_token_id) for i in range(0, len(items), bs)], bs


@torch.inference_mode()
def run_readonce(model, batch):
    logits, _ = model(batch, mode="readonce")
    return logits


@torch.inference_mode()
def run_baseline(model, chunks):
    return [model(c, mode="stock")[0] for c in chunks]


@torch.inference_mode()
def run_prefixkv(model, tok, state_ids, blocks):
    return prefixkv.answer(model, tok, state_ids, blocks)


def timed(fn):
    t = time.perf_counter()
    fn()
    return time.perf_counter() - t


def bench(model_name, S_list, K_list, tag="", min_reps=3, max_reps=10, budget_s=12.0, skip_s=SKIP_S):
    tok, enc, spec = load_backbone(model_name, attn="sdpa")
    torch.manual_seed(0)
    model = ChoiceModel(enc).eval()
    pool = question_pool(max(K_list), seed=1234)
    out = {"env": env_info(), "model": model_name, "backbone": backbone_summary(enc), "tag": tag,
           "settings": {"BS_MAX": BS_MAX, "SKIP_S": skip_s, "ONE_REP_S": ONE_REP_S, "MASK_BYTES": MASK_BYTES,
                        "min_reps": min_reps, "max_reps": max_reps, "budget_s": budget_s},
           "question_pool_n_options": [len(q["options"]) for q in pool], "runs": []}
    for S in S_list:
        state_ids = long_state_ids(tok, S - 2, seed=S)
        block_env = {"power": power_state(), "background_cpu_percent_1s": env_info()["background_cpu_percent_1s"]}
        per_row_1 = None
        for K in K_list:
            qs = pool[:K]
            gc.collect()
            # --- input building (tokenise questions + pack + collate), both methods
            tb_ro = [timed(lambda: build_readonce(tok, state_ids, qs)) for _ in range(3)]
            tb_bl = [timed(lambda: build_baseline(tok, state_ids, qs)) for _ in range(3)]
            ro_batch = build_readonce(tok, state_ids, qs)
            bl_chunks, bs = build_baseline(tok, state_ids, qs)
            L_ro = ro_batch["input_ids"].shape[1]
            q_tokens = L_ro - S
            rec = {"S": S, "K": K, "readonce_len": L_ro, "question_tokens": q_tokens,
                   "baseline_rows": K, "baseline_row_len_max": max(c["input_ids"].shape[1] for c in bl_chunks),
                   "baseline_tokens": int(sum(c["attention_mask"].sum().item() for c in bl_chunks)),
                   "baseline_batch_size": bs,
                   "build_ms": {"readonce": stats_ms(tb_ro), "baseline": stats_ms(tb_bl)}}
            # --- warm-up (skipped for very long baseline configs: the model is already warm and a
            #     single timed rep of >= ONE_REP_S is dominated by compute, not first-call overhead)
            w_ro = timed(lambda: run_readonce(model, ro_batch))
            blocks = [encode_block(tok, q) for q in qs]
            w_kv = timed(lambda: run_prefixkv(model, tok, state_ids, blocks))
            l_ro, l_kv = run_readonce(model, ro_batch), run_prefixkv(model, tok, state_ids, blocks)
            fin = torch.isfinite(l_ro)
            rec["prefixkv_vs_packed_max_abs_diff_logits"] = (l_ro[fin] - l_kv[fin]).abs().max().item()
            est_bl = K * per_row_1 if per_row_1 else None
            do_bl = est_bl is None or est_bl <= skip_s
            single = est_bl is not None and est_bl > ONE_REP_S
            w_bl = timed(lambda: run_baseline(model, bl_chunks)) if (do_bl and not single) else None
            if K == 1 and w_bl is not None:
                per_row_1 = w_bl
            if w_bl is not None and w_bl > ONE_REP_S:
                single = True
            ro_t, bl_t, kv_t = [], [], []

            def need(ts, target):
                return len(ts) < max_reps and (len(ts) < target or sum(ts) < budget_s)

            def need_bl():
                return do_bl and (len(bl_t) < 1 if single else need(bl_t, min_reps))

            while need(ro_t, min_reps) or need(kv_t, min_reps) or need_bl():
                if need(ro_t, min_reps):
                    ro_t.append(timed(lambda: run_readonce(model, ro_batch)))
                if need(kv_t, min_reps):
                    kv_t.append(timed(lambda: run_prefixkv(model, tok, state_ids, blocks)))
                if need_bl():
                    bl_t.append(timed(lambda: run_baseline(model, bl_chunks)))
            rec["readonce"] = stats_ms(ro_t)
            rec["readonce"]["warmup_ms"] = round(w_ro * 1000, 1)
            rec["prefixkv"] = stats_ms(kv_t)
            rec["prefixkv"]["warmup_ms"] = round(w_kv * 1000, 1)
            if do_bl:
                rec["baseline"] = stats_ms(bl_t)
                rec["baseline"]["warmup_ms"] = round(w_bl * 1000, 1) if w_bl is not None else None
                rec["baseline"]["single_rep_no_warmup"] = bool(single and w_bl is None)
                rec["speedup_median"] = round(rec["baseline"]["median_ms"] / rec["readonce"]["median_ms"], 2)
                rec["speedup_prefixkv_median"] = round(rec["baseline"]["median_ms"] / rec["prefixkv"]["median_ms"], 2)
            else:
                rec["baseline"] = {"skipped": True, "estimated_ms_per_rep_ESTIMATE": round(est_bl * 1000),
                                   "basis": "K x measured K=1 warm-up time at this S"}
                rec["speedup_ESTIMATE"] = round(est_bl * 1000 / rec["readonce"]["median_ms"], 1)
                rec["speedup_prefixkv_ESTIMATE"] = round(est_bl * 1000 / rec["prefixkv"]["median_ms"], 1)
            print(model_name, "S=%d K=%d L_ro=%d" % (S, K, L_ro), "RO", rec["readonce"]["median_ms"], "/",
                  rec["readonce"]["p90_ms"], "(n=%d)" % rec["readonce"]["n"],
                  "KV", rec["prefixkv"]["median_ms"], "/", rec["prefixkv"]["p90_ms"], "BL",
                  rec["baseline"].get("median_ms", rec["baseline"].get("estimated_ms_per_rep_ESTIMATE")), "/",
                  rec["baseline"].get("p90_ms"), "(n=%s, bs=%d)" % (rec["baseline"].get("n", "skip"), bs),
                  "x", rec.get("speedup_median", rec.get("speedup_ESTIMATE")), flush=True)
            out["runs"].append(rec)
            dump("latency_%s%s.json" % (model_name, tag), out)  # save as we go
        block_env["power_end"] = power_state()
        out.setdefault("per_S_env", {})[str(S)] = block_env
    out["env_end"] = env_info(sample_cpu=False)
    print(dump("latency_%s%s.json" % (model_name, tag), out))


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "ettin-68m"
    S_list = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [128, 512, 2048]
    K_list = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [1, 2, 5, 10, 20, 50]
    tag = sys.argv[4] if len(sys.argv) > 4 else ""
    skip = float(sys.argv[5]) if len(sys.argv) > 5 else SKIP_S
    bench(name, S_list, K_list, tag, skip_s=skip)
