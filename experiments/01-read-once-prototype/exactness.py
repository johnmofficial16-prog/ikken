"""Experiment 1: does a question's output in a K-question packed pass equal its output when packed
alone with the same state (K=1)?

For each backbone / attention implementation / dtype / state length:
  * packed pass with K questions vs K separate K=1 packed passes: max |diff| of every question-block
    hidden state, of the option logits and probabilities (scorer = randomly initialised MLP, seed 0),
    and argmax agreement;
  * state hidden states in the packed pass vs a state-only pass;
  * order invariance (questions packed in reverse order);
  * padded batch (two different states in one batch) vs each alone;
  * sanity: our mask builder with no packing (seg all 0) vs the stock HF path (2D padding mask);
  * negative controls: sequential positions, blocks allowed to see each other, sliding window on
    sequence index (the stock behaviour) -> these SHOULD change the answers.

Usage: python exactness.py [model ...]
"""
import sys
import time

import torch

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from common import backbone_summary, dump, env_info, load_backbone  # noqa: E402
from readonce import ChoiceModel, collate, encode_block, pack_cross, pack_readonce  # noqa: E402
from synth import long_state_ids, question_pool  # noqa: E402


def block_slices(item):
    """[(start, end)] of each question block in a packed item."""
    seg = item["seg"]
    out, k = [], 1
    while True:
        idx = [i for i, s in enumerate(seg) if s == k]
        if not idx:
            return out
        out.append((idx[0], idx[-1] + 1))
        k += 1


@torch.inference_mode()
def forward(model, items, pad_id, **kw):
    batch = collate(items, pad_id)
    logits, h = model(batch, **kw)
    return logits, h, batch


def compare_packed(model, tok, state_ids, blocks, **kw):
    """Packed-K vs each question alone (K=1). Returns diff stats (and per-question details)."""
    pad = tok.pad_token_id
    item = pack_readonce(tok, state_ids, blocks, positions=kw.pop("positions", "parallel"))
    logits_k, h_k, _ = forward(model, [item], pad, **kw)
    sl = block_slices(item)
    S = item["state_len"]
    d_hidden = d_marker = d_logit = d_prob = 0.0
    agree, scale = 0, 0.0
    for i, b in enumerate(blocks):
        one = pack_readonce(tok, state_ids, [b])
        logits_1, h_1, _ = forward(model, [one], pad)  # reference: always the correct K=1 construction
        a, e = sl[i]
        hk, h1 = h_k[0, a:e], h_1[0, S:S + len(b[0])]
        d_hidden = max(d_hidden, (hk - h1).abs().max().item())
        mk = torch.tensor(b[1])
        d_marker = max(d_marker, (hk[mk] - h1[mk]).abs().max().item())
        n = len(b[1])
        lk, l1 = logits_k[i, :n].double(), logits_1[0, :n].double()
        d_logit = max(d_logit, (lk - l1).abs().max().item())
        d_prob = max(d_prob, (lk.softmax(-1) - l1.softmax(-1)).abs().max().item())
        agree += int(lk.argmax() == l1.argmax())
        scale = max(scale, h1.abs().max().item())
    return {"max_abs_diff_block_hidden": d_hidden, "max_abs_diff_marker_hidden": d_marker,
            "max_abs_diff_logits": d_logit, "max_abs_diff_probs": d_prob,
            "argmax_agree": "%d/%d" % (agree, len(blocks)), "max_abs_hidden_value": scale,
            "packed_len": len(item["ids"]), "state_len": S,
            "question_tokens": sum(len(b[0]) for b in blocks)}, logits_k, h_k


def run(name, attn="sdpa", dtype=torch.float32, S_list=(128, 512, 2048), K=20, controls_at=512):
    tok, enc, spec = load_backbone(name, attn=attn)
    torch.manual_seed(0)
    model = ChoiceModel(enc).to(dtype).eval()
    pad = tok.pad_token_id
    res = {"model": name, "attn": attn, "dtype": str(dtype), "backbone": backbone_summary(enc), "runs": []}
    for S in S_list:
        t0 = time.time()
        state_ids = long_state_ids(tok, S - 2, seed=S)  # + [CLS] + [SEP] = S tokens
        qs = question_pool(K, seed=S)
        blocks = [encode_block(tok, q) for q in qs]
        r = {"S": S, "K": K, "n_options": [len(q["options"]) for q in qs]}
        r["packed_vs_single"], logits_k, h_k = compare_packed(model, tok, state_ids, blocks)

        # state hidden states: packed pass vs state-only pass
        _, h0, _ = forward(model, [pack_readonce(tok, state_ids, [])], pad)
        r["state_hidden_max_abs_diff_vs_state_only"] = (h_k[0, :S] - h0[0, :S]).abs().max().item()

        # order invariance: reverse the packing order
        rev = pack_readonce(tok, state_ids, blocks[::-1])
        logits_r, _, _ = forward(model, [rev], pad)
        d = 0.0
        for i, b in enumerate(blocks):
            n = len(b[1])
            j = K - 1 - i
            d = max(d, (logits_r[j, :n].double().softmax(-1) - logits_k[i, :n].double().softmax(-1)).abs().max().item())
        r["reverse_order_max_abs_diff_probs"] = d

        # padded batch: this state with K questions + a shorter state with 3 questions, in one batch
        other_ids = long_state_ids(tok, max(8, S // 3), seed=S + 1)
        other = pack_readonce(tok, other_ids, blocks[:3])
        logits_b, _, _ = forward(model, [pack_readonce(tok, state_ids, blocks), other], pad)
        logits_o, _, _ = forward(model, [other], pad)
        d_b = 0.0
        for i, b in enumerate(blocks):
            n = len(b[1])
            d_b = max(d_b, (logits_b[i, :n] - logits_k[i, :n]).abs().max().item())
        for i, b in enumerate(blocks[:3]):
            n = len(b[1])
            d_b = max(d_b, (logits_b[K + i, :n] - logits_o[i, :n]).abs().max().item())
        r["padded_batch_max_abs_diff_logits"] = d_b

        # sanity: our mask builder on an unpacked Laya-style row == stock HF masks
        cross = [pack_cross(tok, state_ids, b) for b in blocks[:4]]
        l_ours, _, _ = forward(model, cross, pad)
        l_stock, _, _ = forward(model, cross, pad, mode="stock")
        fin = torch.isfinite(l_ours)
        r["custom_mask_vs_stock_on_unpacked_rows_max_abs_diff_logits"] = (l_ours[fin] - l_stock[fin]).abs().max().item()

        if S == controls_at:
            ctrl = {}
            for label, kw in [("sequential_positions", {"positions": "sequential"}),
                              ("blocks_see_each_other", {"isolate_blocks": False}),
                              ("sliding_window_on_sequence_index", {"window_on": "index"})]:
                c, _, _ = compare_packed(model, tok, state_ids, blocks, **kw)
                ctrl[label] = {k: c[k] for k in ("max_abs_diff_block_hidden", "max_abs_diff_logits",
                                                 "max_abs_diff_probs", "argmax_agree")}
            r["negative_controls"] = ctrl
        r["seconds"] = round(time.time() - t0, 1)
        res["runs"].append(r)
        print(name, attn, dtype, S, {k: v for k, v in r["packed_vs_single"].items() if k.startswith("max") or k == "argmax_agree"},
              "state", "%.2e" % r["state_hidden_max_abs_diff_vs_state_only"],
              "rev", "%.2e" % r["reverse_order_max_abs_diff_probs"], "pad", "%.2e" % r["padded_batch_max_abs_diff_logits"],
              "stock", "%.2e" % r["custom_mask_vs_stock_on_unpacked_rows_max_abs_diff_logits"], r["seconds"], "s", flush=True)
        if "negative_controls" in r:
            print("   controls:", r["negative_controls"], flush=True)
    return res


if __name__ == "__main__":
    names = sys.argv[1:] or ["ettin-68m"]
    out = {"env": env_info(), "results": []}
    fname = "exactness_%s.json" % "_".join(names)
    for name in names:
        if name == "modernbert-base":  # largest model: skip S=2048 to fit the time box
            out["results"].append(run(name, S_list=(128, 512)))
        else:
            out["results"].append(run(name))
        dump(fname, out)  # save after every backbone so a cut-off loses nothing
        if name == "ettin-17m":  # cheap extra configurations on the smallest model
            out["results"].append(run(name, attn="eager", S_list=(512,)))
            out["results"].append(run(name, dtype=torch.float64, S_list=(512,)))
            out["results"].append(run(name, S_list=(7936,), K=10, controls_at=None))
        if name == "ettin-68m":
            out["results"].append(run(name, S_list=(512,), K=50, controls_at=None))
        dump(fname, out)
    out["env_end"] = env_info(sample_cpu=False)
    print(dump(fname, out))
