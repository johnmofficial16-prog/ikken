"""Paired comparison of two train.py runs evaluated on the same held-out questions (same order).

Reads the per-question prediction files train.py writes to the directory in $R ($R/pred_*.json) and
reports accuracy / ECE / NLL differences with a paired bootstrap (2,000 resamples of questions) and an
exact McNemar test on the discordant pairs. Output: results/compare_<a>_vs_<b>.json

Usage: python compare_train.py pred_train_readonce_...json pred_train_cross_...json
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import dump  # noqa: E402


def ece(conf, corr, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (conf > lo) & (conf <= hi)
        if s.any():
            e += s.mean() * abs(conf[s].mean() - corr[s].mean())
    return e


def mcnemar_exact(b, c):
    n, k = b + c, min(b, c)
    if n == 0:
        return 1.0
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def arrays(rows):
    return (np.array([r["correct"] for r in rows], float), np.array([r["conf"] for r in rows]),
            np.array([-math.log(max(r["p_true"], 1e-12)) for r in rows]))


def compare(ra, rb, seed=0, n_boot=2000):
    ca, fa, la = arrays(ra)
    cb, fb, lb = arrays(rb)
    rng = np.random.default_rng(seed)
    n = len(ra)
    idx = rng.integers(0, n, size=(n_boot, n))
    d_acc = ca[idx].mean(1) - cb[idx].mean(1)
    d_nll = la[idx].mean(1) - lb[idx].mean(1)
    d_ece = np.array([ece(fa[i], ca[i]) - ece(fb[i], cb[i]) for i in idx[:500]])
    b = int(((ca == 1) & (cb == 0)).sum())
    c = int(((ca == 0) & (cb == 1)).sum())
    return {"n": n, "acc_a": round(ca.mean(), 4), "acc_b": round(cb.mean(), 4),
            "acc_diff_a_minus_b": round(ca.mean() - cb.mean(), 4),
            "acc_diff_95ci": [round(float(np.percentile(d_acc, 2.5)), 4), round(float(np.percentile(d_acc, 97.5)), 4)],
            "ece_a": round(ece(fa, ca), 4), "ece_b": round(ece(fb, cb), 4),
            "ece_diff_95ci": [round(float(np.percentile(d_ece, 2.5)), 4), round(float(np.percentile(d_ece, 97.5)), 4)],
            "nll_a": round(la.mean(), 4), "nll_b": round(lb.mean(), 4),
            "nll_diff_95ci": [round(float(np.percentile(d_nll, 2.5)), 4), round(float(np.percentile(d_nll, 97.5)), 4)],
            "mcnemar_a_right_b_wrong": b, "mcnemar_a_wrong_b_right": c, "mcnemar_exact_p": round(mcnemar_exact(b, c), 4)}


if __name__ == "__main__":
    base = os.environ.get("R", ".")
    pa, pb = [p if os.path.isabs(p) else os.path.join(base, p) for p in sys.argv[1:3]]
    with open(pa) as fh:
        ra = json.load(fh)
    with open(pb) as fh:
        rb = json.load(fh)
    assert len(ra) == len(rb) and all(x["template"] == y["template"] for x, y in zip(ra, rb)), "not the same eval set"
    out = {"a": os.path.basename(pa), "b": os.path.basename(pb), "overall": compare(ra, rb)}
    for fam in sorted({r["family"] for r in ra}):
        sel = [i for i, r in enumerate(ra) if r["family"] == fam]
        out["family:" + fam] = compare([ra[i] for i in sel], [rb[i] for i in sel])
    name = "compare_%s_vs_%s.json" % tuple(os.path.basename(p)[len("pred_train_"):-5] for p in (pa, pb))
    print(dump(name, out))
    print(json.dumps(out, indent=1))
