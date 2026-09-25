# Ikken: read once, answer many

*Ikken (一見) is Japanese for "at a glance".*

Ikken gives ModernBERT-family encoders exact multi-question encoding. You ask K typed questions about one input and pay for
roughly one encoding of that input, not K. Every question's hidden states are identical, to floating-point
tolerance, to asking that question alone.

It works with any ModernBERT-architecture encoder (ModernBERT, Ettin, mmBERT) in transformers ≥ 5.17. Licence: Apache-2.0.

## Why

Decision models take a state (an email thread, a ticket, a log) plus several typed questions, and return calibrated answers.
Examples are TypeSafe's Jev and the open Laya and Kev. If such a model re-reads the state for every question, its cost grows
with the number of questions. Laya's maintainer describes each question as formatting the state after the question
markers, which re-encodes the state per question ([laya#49](https://github.com/NandhaKishorM/laya/issues/49)), and
Laya's benchmarks put each extra question at about 7.0 ms on a GB10 and ~14.9 ms on a T4
([BENCHMARKS.md](https://github.com/NandhaKishorM/laya/blob/010bacef009c855ccba814b51f7c8e1d38ab5e3f/BENCHMARKS.md#L212)).

## How it works

```
[CLS] state [SEP] | question 1 | question 2 | ... | question K
 positions 0..S-1  | S, S+1, …  | S, S+1, …  |     | S, S+1, …    ← every question restarts at S
```

- State tokens attend only to state tokens, so the state encoding is the same whatever you ask.
- Question *k* attends to the state and to itself, never to another question.
- Every question block starts at the same position ("parallel positions"), so it sees the state exactly as if it were
  the only question.

As a result, each question's hidden states equal those of the same question packed alone. `check_parity` measures this
on your own model and inputs.

## The trap: ModernBERT's sliding window

ModernBERT interleaves global attention layers with local layers that see a 128-token window. In transformers, that
window is built from **sequence index**: `abs(q_idx - kv_idx) <= sliding_window` in `masking_utils`. Rotary
embeddings, however, use **position ids**.

With parallel positions the two disagree. Questions 2..K sit further from the state by sequence index than by
position, so the local layers silently show them a different slice of the state. Nothing raises an error; answers just
change. With an untrained scorer, the stock window flipped 8 of 20 answers on Ettin-68m on a laptop CPU, and 7 of 64
on a T4 GPU.

Ikken builds the window from position ids, and the test suite includes the negative control.

FlashAttention cannot express this mask, because it applies its window by sequence index inside the kernel. Use
`attn_implementation="sdpa"` (its memory-efficient kernel accepts the mask, including on T4) or `"eager"`.

## Results (measured, 22–23 Sep 2026)

| What | Result |
|---|---|
| Parity on CPU, fp32 (Ettin-17m, Ettin-68m, ModernBERT-base; 128–2,048-token states, and 7,936 on Ettin-17m; 10–50 questions) | Option probabilities within **1.0e-7** of each question asked alone; 20/20 answers agree. In fp64 (Ettin-17m) the probabilities are identical |
| Parity on a T4 GPU (Ettin-68m, 64 questions incl. a 2,048-token state) | fp32 **1.8e-7**, fp16 **2.1e-4**; 64/64 answers agree |
| Latency, 15 W laptop CPU (Ryzen 7 5825U, Ettin-68m, fp32, 20 questions) | 2,048-token state: **3.7 s** vs 51.2 s re-reading (**13.7×**). 512 tokens: 1.3 s vs 11.5 s (8.9×). Medians of 3–10 interleaved runs |
| Training on one free Kaggle T4 (Ettin-68m, 512-token states, 8 questions each) | **135 vs 25.9** questions/s against per-question re-reading (**5.3×**; median of 5 alternating pairs, range 4.8–5.4×), with less memory (8.3 vs 12.1 GB) |
| Accuracy after identical training (small synthetic test, Ettin-17m, 150 steps) | 0.790 vs 0.752 overall, but read-once was worse on numeric questions (0.467 vs 0.492; 95% CI −4.5 to −0.6 points) and better on lookups. One seed; an earlier run with different wording gave 0.742 vs 0.728. Not evidence either way yet |

**Rule of thumb:** N questions cost at least 1 + N·q/S single-question passes, where q is the question length and S
the state length. The saving is large for long documents, threads and logs, and small for short records, where
the questions dominate the packed sequence and attention cost grows with its length: at 128 tokens, 20 questions
cost 12.6× one (the formula alone gives about 9×).

Details, raw logs and scripts:
- [experiments/01-read-once-prototype](experiments/01-read-once-prototype/RESULTS.md)
- [experiments/02-kaggle-t4-smoke](experiments/02-kaggle-t4-smoke/RESULTS.md)

## Usage

```python
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from ikken import ReadOnceEncoder, choice_block

repo, rev = "jhu-clsp/ettin-encoder-17m", "59c53d9a19ee484b200676d55e4ae240528b2bb9"  # safetensors revision
tok = AutoTokenizer.from_pretrained(repo, revision=rev)
encoder = AutoModelForMaskedLM.from_pretrained(repo, revision=rev, use_safetensors=True,
                                               attn_implementation="sdpa").model.eval()

ro = ReadOnceEncoder(encoder, tok)
questions = [choice_block(tok, "Priority?", ["P1", "P2", "P3", "P4"]),
             choice_block(tok, "Is the customer blocked?", ["yes", "no"])]
batch, rows = ro.pack(["Ticket 4812: dashboard empty since Monday's upgrade, 40 users blocked."], [questions])
with torch.no_grad():
    hidden = ro.encode(batch)                  # the state is encoded once
option_states = ro.markers(hidden, rows)[0]    # one [n_options, hidden] tensor per question
print(ro.check_parity("Ticket 4812: ...", questions))
```

The lower-level `pack`, `collate`, `read_once_masks` and `encode` functions work with raw token ids and custom blocks
(`Block(ids, markers)`), if you are training your own head.

## Limits

- **You must train with the mask.** The state no longer sees the question, so this is a different computation.
  Applying the masks to a model trained as a per-question cross-encoder changes its answers. In the literature, turning
  a block mask on without training cost a decoder an average of 16.2 points on four RAG benchmarks, and fine-tuning
  with it recovered to within about 1.3 points ([Block-Attention](https://arxiv.org/abs/2409.15355)).
- **Some questions may lose accuracy.** Questions that need the state re-read in light of the question, such as
  multi-hop comparisons, may lose more. In our small test read-once was already slightly worse on numeric questions; it could not yet test reasoning questions.
- **The speed-up depends on state length:** see the rule of thumb above.
- **Supported setups:** SDPA or eager attention only, on ModernBERT-architecture encoders, with transformers ≥ 5.17.

## Prior work

The idea is not new. Kev (a decoder-based decision model) isolates questions over a shared state. TypeSafe's Jev
documentation describes evaluating every question "in parallel and in isolation against the same state". Research on
the pattern includes [Block-Attention](https://arxiv.org/abs/2409.15355) and
[Parallel Context Windows](https://arxiv.org/abs/2212.10947), as well as prompt caching.

What this library adds is an exact, tested implementation for ModernBERT-family encoders, the sliding-window fix, and
a parity check you can run on your own model.

## Reproduce

```bash
pip install -e ".[test]"
pytest                          # 17 tests, tiny random ModernBERT, about 15 s on a laptop, no downloads
python examples/quickstart.py   # Ettin-17m from the Hub
python examples/bench_cpu.py --states 512 2048 --questions 1 5 20
```

## License

Apache-2.0. The model weights used in the examples keep their own licences (Ettin: MIT).
