"""CPU latency: read the state once vs re-read it for every question (same encoder, same questions).

The re-read baseline is one row per question, [CLS] question state [SEP], with full attention and
ordinary positions, the way per-question cross-encoders work. Both sides run as one batch.

Usage:
  python examples/bench_cpu.py --states 512 2048 --questions 1 5 20 --threads 8
"""
import argparse
import statistics
import time

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from ikken import choice_block, collate, encode, pack

MODELS = {
    "ettin-17m": ("jhu-clsp/ettin-encoder-17m", "59c53d9a19ee484b200676d55e4ae240528b2bb9"),
    "ettin-32m": ("jhu-clsp/ettin-encoder-32m", "27b98f697261919d35f3f995fee5c690921e5404"),
    "ettin-68m": ("jhu-clsp/ettin-encoder-68m", "d446589aef55657267b913c9bcce6c98b69061c7"),
}
FILLER = ("Service Harbor-12 in eu-west, tier api, status degraded. Cloud spend this week $1,240.50 "
          "against a budget of $1,500.00. Error rate 1.84% this week, 2.31% last week. 3 alerts open. "
          "Support ticket from Northwind, plan Enterprise, product Reporting, priority P2, SLA 14 hours left. ")
QUESTIONS = [("Is spend above budget?", ["yes", "no"]), ("Did the error rate fall week over week?", ["yes", "no"]),
             ("Priority?", ["P1", "P2", "P3", "P4"]), ("Which product?", ["Billing", "Login", "Reporting", "API"]),
             ("Are more than 2 alerts open?", ["yes", "no"])]


def median_time(fn, reps):
    fn()  # warm-up
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ettin-68m", choices=sorted(MODELS))
    ap.add_argument("--states", type=int, nargs="+", default=[512, 2048])
    ap.add_argument("--questions", type=int, nargs="+", default=[1, 5, 20])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    repo, rev = MODELS[args.model]
    tok = AutoTokenizer.from_pretrained(repo, revision=rev)
    model = AutoModelForMaskedLM.from_pretrained(repo, revision=rev, use_safetensors=True,
                                                 attn_implementation="sdpa").model.eval()
    filler_ids = tok(FILLER * 64, add_special_tokens=False)["input_ids"]
    print(f"{args.model}, {args.threads} threads, median of {args.reps} runs\n")
    print(f"{'state':>6} {'K':>4} {'read-once ms':>13} {'re-read ms':>11} {'speed-up':>9}")
    for S in args.states:
        state = filler_ids[:S - 2]
        for K in args.questions:
            blocks = [choice_block(tok, *QUESTIONS[i % len(QUESTIONS)]) for i in range(K)]
            ro_batch = collate([pack(state, blocks, cls_id=tok.cls_token_id, sep_id=tok.sep_token_id)],
                               pad_id=tok.pad_token_id)
            rows = [[tok.cls_token_id] + b.ids + state + [tok.sep_token_id] for b in blocks]
            L = max(map(len, rows))
            rr_ids = torch.tensor([r + [tok.pad_token_id] * (L - len(r)) for r in rows])
            rr_mask = torch.tensor([[1] * len(r) + [0] * (L - len(r)) for r in rows])
            with torch.no_grad():
                t_ro = median_time(lambda: encode(model, ro_batch), args.reps)
                t_rr = median_time(lambda: model(input_ids=rr_ids, attention_mask=rr_mask), args.reps)
            print(f"{S:>6} {K:>4} {t_ro * 1000:>13.0f} {t_rr * 1000:>11.0f} {t_rr / t_ro:>8.1f}x", flush=True)


if __name__ == "__main__":
    main()
