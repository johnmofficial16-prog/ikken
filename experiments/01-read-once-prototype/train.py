"""Experiment 3 (and 4): learnability of read-once vs Laya-style per-question encoding.

Both architectures start from the same pretrained encoder + the same freshly initialised scorer
(same seed), see the same records and the same (record, question) pairs in the same order, and take
the same number of optimizer steps:
  readonce : one packed row per record ([CLS] state [SEP] Q1 ... Qk), custom masks, parallel positions
  cross    : one row per (record, question) ([CLS] Q state [SEP]), full attention (Laya-style)

Eval on held-out records (different seed): accuracy, ECE (15 bins, top-1 confidence), NLL and
Brier, overall / by family (lookup vs numeric) / by template, with chance level.

Usage: python train.py --arch readonce|cross [--model ettin-32m] [--steps 300] [--batch 8] [--k 4]
                       [--style raw|digits|derived] [--threads 8] [--tag x]
"""
import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import dump, env_info, load_backbone  # noqa: E402
from readonce import ChoiceModel, build_cross_batch, build_readonce_batch, choice_loss  # noqa: E402
from synth import make_dataset  # noqa: E402

SCRATCH_PRED = os.environ.get("R")  # optional: directory for per-question prediction files


def ece(conf, correct, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            e += sel.mean() * abs(conf[sel].mean() - correct[sel].mean())
    return float(e)


def metrics(rows):
    if not rows:
        return {}
    p_true = np.array([r["p_true"] for r in rows])
    conf = np.array([r["conf"] for r in rows])
    corr = np.array([r["correct"] for r in rows], dtype=float)
    brier = np.array([r["brier"] for r in rows])
    return {"n": len(rows), "accuracy": round(float(corr.mean()), 4),
            "chance": round(float(np.mean([1.0 / r["n_opt"] for r in rows])), 4),
            "ece": round(ece(conf, corr), 4), "nll": round(float(-np.log(np.clip(p_true, 1e-12, 1)).mean()), 4),
            "brier": round(float(brier.mean()), 4), "mean_conf": round(float(conf.mean()), 4)}


@torch.no_grad()
def evaluate(model, tok, data, arch, bs=16):
    model.eval()
    rows = []
    for i in range(0, len(data), bs):
        chunk = data[i:i + bs]
        states = [ex["state"] for ex in chunk]
        qs = [ex["questions"] for ex in chunk]
        batch = (build_readonce_batch if arch == "readonce" else build_cross_batch)(tok, states, qs, with_labels=True)
        logits, _ = model(batch, mode="readonce" if arch == "readonce" else "stock")
        probs = torch.softmax(logits.double(), -1)
        flat_q = [q for ex in chunk for q in ex["questions"]]
        for j, q in enumerate(flat_q):
            n = len(q["options"])
            p = probs[j, :n].numpy()
            y = q["answer"]
            onehot = np.zeros(n)
            onehot[y] = 1
            rows.append({"template": q["template"], "family": q["family"], "n_opt": n,
                         "p_true": float(p[y]), "conf": float(p.max()), "correct": int(p.argmax() == y),
                         "brier": float(((p - onehot) ** 2).sum())})
    model.train()
    out = {"overall": metrics(rows)}
    for fam in sorted({r["family"] for r in rows}):
        out["family:" + fam] = metrics([r for r in rows if r["family"] == fam])
    for t in sorted({r["template"] for r in rows}):
        out["template:" + t] = metrics([r for r in rows if r["template"] == t])
    for lo, hi in [(2, 2), (3, 9), (10, 30)]:
        out["options:%d-%d" % (lo, hi)] = metrics([r for r in rows if lo <= r["n_opt"] <= hi])
    return out, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=["readonce", "cross"])
    ap.add_argument("--model", default="ettin-32m")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=8, help="records per step")
    ap.add_argument("--k", type=int, default=4, help="questions per record per step (sampled)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--style", default="raw", choices=["raw", "digits", "derived"])
    ap.add_argument("--families", default="all", help="all | numeric (train+eval only numeric questions)")
    ap.add_argument("--eval_n", type=int, default=300)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--time_steps_only", type=int, default=0, help="time N steps and exit (no eval)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    tok, enc, spec = load_backbone(a.model, attn="sdpa")
    torch.manual_seed(a.seed)  # identical scorer init for both archs
    model = ChoiceModel(enc)
    model.train()
    enc_params = list(model.encoder.parameters())
    head_params = list(model.scorer.parameters())
    opt = torch.optim.AdamW([{"params": enc_params, "lr": a.lr}, {"params": head_params, "lr": a.head_lr}],
                            weight_decay=0.01)
    warm = max(1, int(a.warmup * a.steps))

    def lr_lambda(s):  # linear warm-up, then linear decay to 0
        if s < warm:
            return (s + 1) / warm
        return max(0.0, (a.steps - s) / max(1, a.steps - warm))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    def keep(q):
        return a.families == "all" or q["family"] == a.families

    # identical training stream for both archs: records + sampled question subsets from one seeded RNG
    train = make_dataset(a.steps * a.batch, seed=1000 + a.seed, style=a.style)
    rng = random.Random(a.seed)
    stream = []
    for ex in train:
        qs = [q for q in ex["questions"] if keep(q)]
        stream.append({"state": ex["state"], "questions": rng.sample(qs, min(a.k, len(qs)))})
    test = make_dataset(a.eval_n, seed=999_999, style=a.style)
    test = [{"state": ex["state"], "questions": [q for q in ex["questions"] if keep(q)]} for ex in test]
    test = [ex for ex in test if ex["questions"]]

    log, t0, tok_count = [], time.time(), 0
    step_times = []
    for step in range(a.steps):
        chunk = stream[step * a.batch:(step + 1) * a.batch]
        states = [ex["state"] for ex in chunk]
        qs = [ex["questions"] for ex in chunk]
        ts = time.time()
        if a.arch == "readonce":
            batch = build_readonce_batch(tok, states, qs, with_labels=True)
            logits, _ = model(batch, mode="readonce")
        else:
            batch = build_cross_batch(tok, states, qs, with_labels=True)
            logits, _ = model(batch, mode="stock")
        loss = choice_loss(logits, batch["labels"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        step_times.append(time.time() - ts)
        tok_count += int(batch["attention_mask"].sum())
        if step % 10 == 0 or step == a.steps - 1:
            acc = (logits.argmax(-1) == batch["labels"]).float().mean().item()
            log.append({"step": step, "loss": round(loss.item(), 4), "batch_acc": round(acc, 3),
                        "grad_norm": round(float(gn), 3), "elapsed_s": round(time.time() - t0, 1),
                        "rows": int(batch["input_ids"].shape[0]), "row_len": int(batch["input_ids"].shape[1])})
            print(a.arch, log[-1], flush=True)
        if a.time_steps_only and step + 1 >= a.time_steps_only:
            st = sorted(step_times[1:]) or step_times
            print(json.dumps({"arch": a.arch, "model": a.model, "median_step_s": st[len(st) // 2],
                              "tokens_per_step": tok_count / (step + 1)}))
            return
    train_s = time.time() - t0
    te = time.time()
    ev, rows = evaluate(model, tok, test, a.arch)
    eval_s = time.time() - te

    # after training: packed-K answers still equal packed-alone answers? (read-once only, 20 records)
    post = None
    if a.arch == "readonce":
        model.eval()
        with torch.no_grad():
            d = 0.0
            for ex in test[:20]:
                bk = build_readonce_batch(tok, [ex["state"]], [ex["questions"]])
                lk, _ = model(bk)
                for j, q in enumerate(ex["questions"]):
                    b1 = build_readonce_batch(tok, [ex["state"]], [[q]])
                    l1, _ = model(b1)
                    n = len(q["options"])
                    d = max(d, (lk[j, :n].double().softmax(-1) - l1[0, :n].double().softmax(-1)).abs().max().item())
        post = {"records": 20, "max_abs_diff_probs_packed_vs_single": d}

    out = {"env": env_info(sample_cpu=False), "args": vars(a), "model_spec": spec,
           "train_seconds": round(train_s, 1), "eval_seconds": round(eval_s, 1),
           "median_step_s": round(sorted(step_times)[len(step_times) // 2], 3),
           "train_tokens": tok_count, "train_questions": a.steps * a.batch * a.k,
           "log": log, "eval": ev, "post_training_exactness": post}
    name = "train_%s_%s_%s%s.json" % (a.arch, a.model, a.style, a.tag)
    print(dump(name, out))
    if SCRATCH_PRED:
        with open(os.path.join(SCRATCH_PRED, "pred_" + name), "w") as fh:
            json.dump(rows, fh)
    print(json.dumps(ev["overall"]), json.dumps({k: v["accuracy"] for k, v in ev.items()}))


if __name__ == "__main__":
    main()
