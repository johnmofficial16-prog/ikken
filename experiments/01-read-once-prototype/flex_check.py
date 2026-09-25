"""GPU-portability check: express the read-once mask as a FlexAttention mask_mod / BlockMask.

On a GPU the dense [B,1,L,L] mask used on CPU is the wrong tool: FlashAttention-2 (HF's
flash_attention_2 path) takes only a 2D padding mask, and SDPA with an arbitrary mask falls back to
kernels that compute every masked block. FlexAttention takes a mask_mod and skips empty 128x128
blocks. Here we (1) check that the mask_mod gives the same attention output as our dense SDPA mask
(eager flex_attention on CPU; no torch.compile, which needs a C++ toolchain on Windows), and
(2) report the BlockMask sparsity, i.e. the fraction of attention blocks a block-sparse kernel skips,
for the latency grid's shapes. (1) is measured; the FLOP saving in (2) is arithmetic, not a timing.

Usage: python flex_check.py
"""
import os
import sys
import warnings

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import dump, env_info  # noqa: E402
from readonce import build_masks  # noqa: E402

warnings.filterwarnings("ignore")


def layout(S, q_lens):
    """seg/pos for one packed row: [CLS] state [SEP] = S tokens, then blocks of the given lengths."""
    seg, pos = [0] * S, list(range(S))
    for k, n in enumerate(q_lens, start=1):
        seg += [k] * n
        pos += list(range(S, S + n))
    return torch.tensor([seg]), torch.tensor([pos])


def main():
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
    out = {"env": env_info(sample_cpu=False), "torch": torch.__version__, "checks": [], "sparsity": []}
    window = 64
    # (1) numerical equivalence on a mid-size shape (eager flex materialises scores, so keep it small)
    torch.manual_seed(0)
    for S, K in [(512, 10), (1024, 20)]:
        q_lens = [int(x) for x in torch.randint(12, 60, (K,))]
        seg, pos = layout(S, q_lens)
        L = seg.shape[1]

        def full_mod(b, h, qi, ki):
            qs, ks = seg[b, qi], seg[b, ki]
            return (ks == 0) | (ks == qs)

        def local_mod(b, h, qi, ki):
            return full_mod(b, h, qi, ki) & ((pos[b, qi] - pos[b, ki]).abs() <= window)

        H, D = 8, 64
        q, k, v = (torch.randn(1, H, L, D) for _ in range(3))
        dense = build_masks(seg, pos, window)
        rec = {"S": S, "K": K, "L": L}
        for name, mod in [("full_attention", full_mod), ("sliding_attention", local_mod)]:
            bm = create_block_mask(mod, 1, None, L, L, device="cpu", _compile=False)
            o_flex = flex_attention(q, k, v, block_mask=bm)
            o_sdpa = F.scaled_dot_product_attention(q, k, v, attn_mask=dense[name])
            rec[name] = {"max_abs_diff_vs_dense_sdpa": (o_flex - o_sdpa).abs().max().item(),
                         "block_sparsity_percent": round(float(bm.sparsity()), 1)}
        out["checks"].append(rec)
        print(rec, flush=True)
    # (2) BlockMask sparsity for the latency grid (question blocks of ~35 tokens, the pool's mean)
    for S in (128, 512, 2048):  # 8K omitted: create_block_mask without compile evaluates all L^2 pairs
        for K in (1, 5, 20, 50):
            seg, pos = layout(S, [35] * K)
            L = seg.shape[1]

            def full_mod(b, h, qi, ki):
                qs, ks = seg[b, qi], seg[b, ki]
                return (ks == 0) | (ks == qs)

            bm = create_block_mask(full_mod, 1, None, L, L, device="cpu", _compile=False)
            # exact fraction of allowed (q,k) pairs vs dense L^2
            allowed = S * S + K * 35 * (S + 35)
            rec = {"S": S, "K": K, "L": L, "block_sparsity_percent_full_layers": round(float(bm.sparsity()), 1),
                   "allowed_pairs_fraction_of_dense": round(allowed / (L * L), 3)}
            out["sparsity"].append(rec)
            print(rec, flush=True)
    print(dump("flex_check.json", out))


if __name__ == "__main__":
    main()
