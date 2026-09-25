# 01 — Read once, answer many: prototype and CPU measurements

Run: 23 Sep 2026, 17:22–20:12 (local machine time) on the project laptop, a single fresh rerun of every script
below. Raw numbers are in `results/*.json`; `python summarize.py` prints the tables. **All numbers are measured
unless marked ESTIMATE.**

## The design being tested

```
one row per state:   [CLS] state [SEP] | Q1 block | Q2 block | ... | QK block
Q block:             "choice question: <text>" [SEP] [MASK] opt_0 [MASK] opt_1 ... [SEP]
attention:           state -> state only;  block i -> state + block i;  no block -> other block
positions:           state 0..S-1; every block restarts at S ("parallel positions")
local layers:        sliding window (|dpos| <= 64) computed on POSITION ids, not sequence index
head:                MLP scorer on each [MASK] marker, softmax over that question's options
baseline (Laya-style): K rows [CLS] Q block state [SEP], full attention, same scorer
```

It runs on the **stock** Hugging Face `ModernBertModel` (transformers 5.17.0). The forward takes `position_ids`
and accepts `attention_mask` as a per-layer-type dict of 4D masks, so the model needs no code changes.

## Bottom line

| Question | Answer from this machine |
|---|---|
| Does the packed pass reproduce single-question outputs? | **Yes.** On Ettin-17m, Ettin-68m and ModernBERT-base, at S = 128–7,936 tokens and K = 10–50, option probabilities match the K = 1 pass to ≤ 1.0e-7 (fp32, logits ≤ 1.0e-6); argmax agreed for every question in every configuration, including K = 50. fp64 is bit-identical on Ettin-17m (0.0 everywhere). This holds only with parallel positions, block isolation and a **position-space** sliding window: each of three wrong variants flips at least 1 and up to 12 of 20 answers, depending on backbone and variant (see the negative-controls table). |
| How does latency scale with K? (Ettin-68m, fp32, 8 threads, AC) | At S = 2,048: 20 questions cost 3.73 s vs 51.2 s for the batched Laya-style re-read — **13.7× faster**, 2.03× the cost of one question. At S = 512, 20 questions: 1.29 s vs 11.5 s (**8.9×**), 4.09× one question. Against an **unpadded**, batch-size-1 baseline the gains hold: 20 questions at S = 2,048 are 2.66 s vs 36.7 s (**13.8×**, 1.43× one question), and at S = 512, 777 ms vs 6.88 s (**8.8×**, 2.29× one question). Roughly, K questions cost about `1 + K·q/S` single passes (q = one question's tokens): the packed pass is cheap only once the state dwarfs the questions. |
| Does question-agnostic encoding cost accuracy or calibration? (Ettin-17m, 150 identical steps) | **Not clearly, and not stably.** This run: accuracy 0.790 vs 0.752 for the Laya-style cross-encoder (paired diff +0.038, 95% CI +0.027 to +0.049, exact McNemar p < 0.0001). Read-once is worse on the numeric slice (diff −0.026, 95% CI −0.044 to −0.006). **A separately run, earlier pass of the same experiment with different synthetic wording gave 0.742 vs 0.728 (not significant)** — so this accuracy comparison moves with the synthetic text and is not a stable result in either direction; it is not a claim that either architecture learns better. |
| What carries over to a GPU run on ModernBERT? | HF's `flash_attention_2` path accepts only 2D padding masks, so the packed mask cannot use it. Use FlexAttention (HF ModernBERT supports it; the `mask_mod` matched the dense mask to ≤ 6.6e-7 here; no attention dropout), SDPA with a dense mask (no FA backend, O(L²) mask), or the two-stage **prefix-KV** form, which also matched to within a few ×1e-7. FA2 does not support T4s. |

## Environment

AMD Ryzen 7 5825U (8C/16T, 15 W), 15.4 GB RAM, Windows 11 Pro 10.0.26200, Balanced power plan.
Python 3.13.1, torch 2.14.0+cpu (AVX2), transformers 5.17.0, tokenizers 0.23.2, safetensors 0.8.0.
Power: exactness and training ran on **battery** (exactness 81% → 73%; training 67% → 50%, one architecture after
the other, not concurrently). Latency ran on **AC** (45% → 77% across both files). Background CPU sampled 6–26%
across runs. Each JSON records `env.battery`, `power_scheme` and `background_cpu_percent_1s`.

Backbones (safetensors, pinned by commit; see `results/fetch_models.json`):

| name | repo @ commit | layers | d | params (non-emb.) | window / global |
|---|---|---|---|---|---|
| ettin-17m | jhu-clsp/ettin-encoder-17m @ 59c53d9 | 7 | 256 | 16.8M (3.9M) | 128-token local, every 3rd layer global |
| ettin-68m | jhu-clsp/ettin-encoder-68m @ d446589 | 19 | 512 | 68.1M (42.4M) | same |
| modernbert-base | answerdotai/ModernBERT-base @ 8949b90 | 22 | 768 | 149.0M (110.3M) | same |

The Ettin repos have only a pickle `.bin` on `main`. The safetensors revision above was checked tensor-by-tensor
against it: all identical, download 13–35 s. The input embedding exists only as the tied `decoder.weight`, so the
code loads through `AutoModelForMaskedLM` and asserts that no keys are missing.

## 1. Exactness (`exactness.py` → `results/exactness_ettin-17m_ettin-68m_modernbert-base.json`)

This compares each question's outputs in a K-question packed pass with the same question packed alone (K = 1).
The scorer is a random-init MLP (seed 0).

| backbone | attn / dtype | S | K | max abs diff, option probs | max abs diff, logits | argmax agree |
|---|---|---|---|---|---|---|
| ettin-17m | sdpa fp32 | 128 / 512 / 2048 | 20 | 6.3e-8 / 8.2e-8 / 7.5e-8 | ≤ 3.0e-7 | 20/20 each |
| ettin-17m | eager fp32 | 512 | 20 | 5.8e-8 | 2.8e-7 | 20/20 |
| ettin-17m | sdpa **fp64** | 512 | 20 | **0.0** | **0.0** | 20/20 |
| ettin-17m | sdpa fp32 | **7,936** | 10 | 5.6e-8 | 2.2e-7 | 10/10 |
| ettin-68m | sdpa fp32 | 128 / 512 / 2048 | 20 | 4.5e-8 / 1.0e-7 / 9.7e-8 | ≤ 4.5e-7 | 20/20 each |
| ettin-68m | sdpa fp32 | 512 | **50** | 8.9e-8 | 1.0e-6 | 50/50 |
| modernbert-base | sdpa fp32 | 128 / 512 | 20 | 9.8e-8 / 7.8e-8 | ≤ 1.0e-6 | 20/20 each |

Also measured, in every configuration above:
- Reversing the packing order changed the probabilities by ≤ 2.0e-7.
- Packing two states of different lengths into one padded batch changed the logits by ≤ 4.2e-7.
- The state's hidden states in the K-question pass equal a state-only pass (0.0 difference; 7.2e-5 at 7,936 tokens).
- The mask builder reproduces the stock HF path exactly on unpacked rows (0.0).

**Hidden-state gaps scale with the backbone, not cleanly with peak activation.** At S = 512, K = 20 (fp32, sdpa),
the largest per-token hidden-state gap between the packed and single-question pass, alongside that run's largest
hidden-state magnitude for scale: ettin-17m 3.4e-5 (max activation 71), ettin-68m 1.5e-3 (max activation 147),
modernbert-base 1.3e-2 (max activation 43). ModernBERT-base has the smallest peak activation of the three here but
the largest gap — so "larger activations, larger fp32 gap" holds between Ettin-17m and Ettin-68m but not against
ModernBERT-base at this S. Despite that, option probabilities and argmax are exact to ≤ 1e-7 / 100% for all three.

Two additional fp64 checks with the library's `check_parity` (510-token state, 20 questions; `fp64_check.py` →
`results/fp64_parity_check.json`): Ettin-68m's hidden-state gap shrinks to **2.9e-12** in fp64, and ModernBERT-base's to
**3.0e-7**. ModernBERT-base's fp64 residual does not vanish like Ettin-68m's because transformers'
`apply_rotary_pos_emb` casts query and key to float32 even inside an fp64 run (`modeling_modernbert.py`, lines
217–218), so a few ULPs of fp32 rounding survive every layer.

**Negative controls** (S = 512, K = 20; each one *should* change answers):

| variant | ettin-17m logit diff / argmax agree | ettin-68m | modernbert-base |
|---|---|---|---|
| ordinary sequential positions | 0.038 / 17 of 20 | 0.053 / 15 of 20 | 0.053 / 14 of 20 |
| blocks may attend to each other | 0.18 / 9 of 20 | 0.11 / 10 of 20 | 0.20 / 8 of 20 |
| sliding window on sequence index (the stock HF helper) | 0.010 / 19 of 20 | 0.035 / 12 of 20 | 0.056 / 15 of 20 |

## 2. Latency vs K (`latency.py` → `results/latency_ettin-68m.json`)

Backbone: Ettin-68m, fp32, SDPA, 8 threads, on AC, ≥3 timed reps in every cell (`RO_ONE_REP_S=1000` forces this
even where the default single-rep-above-20s cutoff would otherwise apply).
- Question pool seed 1234: 2–30 options, about half yes/no.
- Baseline = K rows `[CLS] Q state [SEP]` on the stock HF path, batched ≤ 16 rows with padding.
- Read-once packed = one row with the custom masks (mask construction is inside the timing).
- Prefix-KV = the same function in two stages (`prefixkv.py`).
- Every number includes the scorer. Reps are interleaved; median / p90 ms (n).

| S | K | Q tokens | baseline median / p90 ms (n) | read-once packed median / p90 ms (n) | speedup | prefix-KV median / p90 ms (n) | speedup | best read-once ÷ 1 question |
|---|---|---|---|---|---|---|---|---|
| 128 | 1 | 23 | 76 / 80 (10) | 78 / 79 (10) | 1.0x | 103 / 107 (10) | 0.7x | 1.02x |
| 128 | 2 | 42 | 138 / 147 (10) | 85 / 91 (10) | 1.6x | 114 / 119 (10) | 1.2x | 1.12x |
| 128 | 5 | 171 | 471 / 487 (10) | 137 / 144 (10) | 3.4x | 266 / 281 (10) | 1.8x | 1.80x |
| 128 | 10 | 322 | 1,016 / 1,304 (10) | 218 / 263 (10) | 4.7x | 471 / 590 (10) | 2.2x | 2.86x |
| 128 | 20 | 1,048 | 5,197 / 5,252 (3) | 958 / 1,099 (10) | 5.4x | 3,899 / 3,933 (4) | 1.3x | 12.56x |
| 128 | 50 | 2,068 | 11,344 / 11,350 (3) | 2,043 / 2,133 (6) | 5.5x | 8,561 / 8,622 (3) | 1.3x | 26.77x |
| 512 | 1 | 23 | 316 / 337 (10) | 322 / 338 (10) | 1.0x | 376 / 404 (10) | 0.8x | 1.02x |
| 512 | 2 | 42 | 640 / 712 (10) | 337 / 375 (10) | 1.9x | 384 / 440 (10) | 1.7x | 1.07x |
| 512 | 5 | 171 | 1,914 / 2,008 (7) | 397 / 412 (10) | 4.8x | 594 / 658 (10) | 3.2x | 1.26x |
| 512 | 10 | 322 | 3,973 / 4,132 (3) | 539 / 565 (10) | 7.4x | 888 / 959 (10) | 4.5x | 1.71x |
| 512 | 20 | 1,048 | **11,479 / 11,969 (3)** | **1,292 / 1,464 (9)** | **8.9x** | 4,327 / 4,472 (3) | 2.6x | 4.09x |
| 512 | 50 | 2,068 | 27,913 / 28,573 (3) | 2,753 / 2,833 (5) | 10.1x | 11,306 / 11,558 (3) | 2.5x | 8.72x |
| 2048 | 1 | 23 | 1,836 / 1,958 (7) | 1,883 / 2,053 (7) | 1.0x | 1,886 / 2,003 (7) | 1.0x | 1.03x |
| 2048 | 2 | 42 | 4,060 / 4,075 (3) | 1,984 / 2,049 (7) | 2.0x | 2,048 / 2,070 (6) | 2.0x | 1.08x |
| 2048 | 5 | 171 | 10,799 / 11,056 (3) | 2,275 / 2,408 (6) | 4.8x | 2,565 / 2,596 (5) | 4.2x | 1.24x |
| 2048 | 10 | 322 | 22,614 / 23,778 (3) | 2,620 / 2,766 (5) | 8.6x | 3,214 / 3,376 (4) | 7.0x | 1.43x |
| 2048 | 20 | 1,048 | **51,213 / 51,732 (3)** | **3,733 / 3,880 (4)** | **13.7x** | 8,138 / 8,178 (3) | 6.3x | 2.03x |
| 2048 | 50 | 2,068 | 120,104 / 122,845 (3) | 5,224 / 5,634 (3) | 23.0x | 18,036 / 19,274 (3) | 6.7x | 2.84x |

Notes on reading this table:
- **Variance.** CV is mostly 1–11% (worst 14%, n = 3–10 reps per cell). Background CPU sampled 6–26% across the
  run.
- **Padding inflates the baseline at short states.** Real / padded baseline tokens are 0.42–0.45 at S = 128 with
  K ≥ 20, 0.71–0.72 at S = 512, and 0.90–0.91 at S = 2,048. The unpadded check below isolates this.
- **Prefix-KV** matched the packed logits closely in every config but, as written, pads every question block to the
  longest in the batch, which makes it slower than the packed pass on CPU.
- **Dense-mask waste.** At K = 50, only 54% / 24% / 9% of the allowed attention pairs are actually needed for
  S = 2,048 / 512 / 128 (`results/flex_check.json`, mask-structure only — see note below). CPU SDPA with a dense
  mask computes all of them regardless.

**Unpadded-baseline check** (`RO_BS_MAX=1 python latency.py ettin-68m 128,512,2048 1,5,20 _bs1 1000` →
`results/latency_ettin-68m_bs1.json`, also on AC).
- The baseline here runs one row per forward, so there is no padding at all.
- Requesting only K ∈ {1, 5, 20} made `question_pool` draw a different, shorter question set (26 tokens/question
  here vs 23–42 in the table above); compare within this table only.

| S | K | Q tokens | unpadded baseline median / p90 ms (n) | read-once packed median / p90 ms (n) | speedup | prefix-KV median / p90 ms (n) | read-once ÷ 1 question |
|---|---|---|---|---|---|---|---|
| 128 | 1 | 26 | 80 / 84 (10) | 79 / 82 (10) | 1.0x | 104 / 108 (10) | 0.99x |
| 128 | 5 | 224 | 446 / 474 (10) | 179 / 198 (10) | 2.5x | 286 / 300 (10) | 2.24x |
| 128 | 20 | 549 | 1,730 / 1,820 (8) | 387 / 425 (10) | **4.5x** | 1,026 / 1,084 (10) | 4.84x |
| 512 | 1 | 26 | 340 / 373 (10) | 355 / 391 (10) | 1.0x | 412 / 437 (10) | 1.05x |
| 512 | 5 | 224 | 1,704 / 1,836 (8) | 491 / 542 (10) | 3.5x | 628 / 726 (10) | 1.44x |
| 512 | 20 | 549 | **6,884 / 6,907 (3)** | **777 / 802 (10)** | **8.8x** | 1,607 / 1,744 (8) | 2.29x |
| 2048 | 1 | 26 | 1,862 / 1,914 (7) | 1,885 / 1,931 (7) | 1.0x | 1,932 / 1,965 (7) | 1.01x |
| 2048 | 5 | 224 | 9,621 / 9,922 (3) | 2,246 / 2,272 (6) | 4.3x | 2,508 / 2,622 (5) | 1.21x |
| 2048 | 20 | 549 | **36,660 / 37,480 (3)** | **2,663 / 2,765 (5)** | **13.8x** | 4,050 / 4,465 (3) | 1.43x |

- **Padding was not the main source of the speedup.** Without it, read-once is still 4.5x / 8.8x / 13.8x faster at
  20 questions for S = 128 / 512 / 2,048.
- The two headline cells above (S = 2,048, K = 20 and S = 512, K = 20) are consistent across the padded and unpadded
  baselines: roughly 13.7–13.8× at S = 2,048 and 8.8–8.9× at S = 512.

## 3. Learnability (`train.py`, `compare_train.py` → `results/train_*_ettin-17m_raw.json`, `results/compare_*.json`)

**Set-up.**
- Ettin-17m, identical for both models: pretrained encoder, seeded scorer init (seed 0), data stream and order.
- 150 steps × (8 synthetic records × 4 sampled questions) = 4,800 training questions, one seed.
- AdamW: encoder 2e-4, head 1e-3, wd 0.01; 10% warm-up, then linear decay; clip 1.0; fp32; 4 threads.
- The two architectures were trained one after the other on battery power, not concurrently.
- Held-out set: 300 records from a different seed, 2,704 questions.
- The Laya-style model sees `[CLS] Q state [SEP]` with full attention, so its state encoding is question-aware.

| slice (n) | chance | read-once acc / ECE | cross-encoder acc / ECE |
|---|---|---|---|
| overall (2,704) | 0.339 | **0.790 / 0.020** | **0.752 / 0.033** |
| lookup (1,648) | 0.307 | 0.997 / 0.032 | 0.918 / 0.020 |
| numeric (1,056) | 0.389 | 0.467 / 0.082 | 0.492 / 0.055 |
| 2 options (1,328) | 0.500 | 0.734 / 0.018 | 0.637 / 0.036 |
| 3–9 options (1,040) | 0.226 | 0.850 / 0.013 | 0.877 / 0.019 |
| 10–30 options (336) | 0.054 | 0.824 / 0.093 | 0.819 / 0.107 |
| 9 categorical lookup templates (2–30 options) | 0.10–0.36 | 1.000 (all nine) | 1.000 (all nine) |
| spend_bucket (3–30 options, numeric) | 0.086 | 0.447 / 0.279 | 0.454 / 0.268 |
| peak_day (7 options, numeric) | 0.143 | 0.138 / 0.010 | 0.303 / 0.066 |
| 7 yes/no templates (lookup and numeric, acc range) | 0.5 | 0.48–1.00 | 0.35–0.74 |

**Paired comparison** (same 2,704 questions; 2,000-sample bootstrap; exact McNemar; from `compare_readonce_ettin-17m_raw_vs_cross_ettin-17m_raw.json`):

| slice | accuracy diff, read-once minus cross (95% CI) | McNemar p | ECE diff (CI) | NLL, read-once vs cross |
|---|---|---|---|---|
| overall | +0.038 (+0.027, +0.049) | < 0.0001 | −0.013 (−0.022, +0.002) | 0.437 vs 0.464 |
| lookup | +0.079 (+0.066, +0.093) | < 0.0001 | +0.012 (+0.000, +0.023) | 0.046 vs 0.119 |
| numeric | **−0.026 (−0.044, −0.006)** | 0.0093 | +0.027 (−0.011, +0.053) | 1.047 vs 1.001 |

Read-once is ahead overall and on lookup questions, and measurably worse on the numeric slice in this run
(McNemar 37 read-once-wrong/right vs 64 the other way on numeric, p = 0.0093).

**This is not a stable result.** A separately run, earlier pass of the same experiment (same steps, seed and
architecture) with different synthetic wording gave **0.742 vs 0.728** for read-once vs cross-encoder — a much
smaller, non-significant gap. Between that run and this one, only the synthetic text generator changed. So the
accuracy comparison between these two architectures moves with the wording of the synthetic data at this training
budget, and neither number should be read as a claim that one architecture learns better than the other.

**Cost of the same 150 steps.** Read-once used 287k training tokens and a 1.46 s median step (211 s total). The
cross-encoder used 638k tokens (2.2× as many) and a 5.13 s median step (709 s total). Evaluating 2,704 questions
took 42 s vs 160 s.

**After training the packed pass is still exact:** ≤ 1.3e-6 difference in probabilities, packed vs alone, on 20
held-out records (read-once run's own post-training check).

**Not done (time box).** More seeds, a larger backbone, longer training so the cross-encoder masters numeric
questions, and experiment 4 (numeric preprocessing). `synth.py` already has the `digits` and `derived` renderings
(`train.py --style digits|derived`).

## 4. GPU portability (code reading + `flex_check.py` → `results/flex_check.json`)

`flex_check.json` measures only mask structure and sparsity, which depends on token counts (S, K) and model
config, not on the synthetic text — it was captured earlier the same day (23 Sep 2026, 11:07) than the rest of
this experiment's files (17:22–20:12) and is still current.

- **What runs today.** HF `ModernBertModel` with `sdpa` or `eager` attention takes the per-layer-type 4D mask dict
  and `position_ids` unchanged. transformers 5.17 no longer unpads inside the ModernBERT forward.
- **`flash_attention_2` cannot express the design.** `_flash_attention_forward` treats `attention_mask` as a 2D
  padding mask (`_upad_input` → `attention_mask.sum(dim=-1)`). Separately, `_is_packed_sequence` would read parallel
  `position_ids` as packed documents on models that forward them. The FlashAttention README lists causal /
  sliding-window / ALiBi / softcap only, and FA2 needs Ampere or newer; T4 has a separate repo.
- **FlexAttention.** The read-once `mask_mod`s gave the same attention output as the dense SDPA mask (≤ 6.6e-7 at
  S = 512/1,024, K = 10/20; eager, CPU, not compiled). The BlockMask skips 25–35% of 128×128 blocks in global
  layers and 59–78% in local layers at those sizes. HF's flex path accepts a `BlockMask` but raises on attention
  dropout > 0. Performance on a GPU is measured separately in experiment 02.
- **Prefix-KV** (`prefixkv.py`) needs no custom mask. Stage 1 is the state alone; stage 2 is each block attending to
  [state K/V of that layer; its own K/V]. Laid out like that, key index = key position, so FlashAttention's
  bottom-right-aligned `window_size=(64, 64)` equals ModernBERT's position-space window. It matched the packed pass
  to within a few ×1e-7 (section 2) and also gives a reusable state cache: new questions later cost only their own
  tokens. It needs length bucketing (CPU) or varlen kernels (GPU) to avoid padding waste.
- **The trap to document for users.** HF's stock sliding-window mask uses sequence-index distance. With parallel
  positions it silently changes answers (8 of 20 flipped on Ettin-68m at S = 512, K = 20 with an untrained scorer;
  see the negative-controls table above).

## Files

| file | what |
|---|---|
| `readonce.py` | the prototype: block encoding, packing, 4D masks (position-space window), `ChoiceModel` (encoder + marker scorer), Laya-style rows |
| `prefixkv.py` | same function in two stages (state once, keep per-layer K/V; question blocks attend to [state K/V; own K/V]); no L×L mask |
| `synth.py` | synthetic service-metrics records (region, tier, status, daily cloud spend, requests, errors, alerts, this/last-week error rate) and support tickets, Choice questions (2–30 options) and yes/no questions with deterministic answers. Its templates are synthetic text, written for measurement only, not for training a released model |
| `exactness.py` | experiment 1 |
| `latency.py` | experiment 2 (read-once packed, read-once prefix-KV, batched Laya-style baseline) |
| `train.py` | experiment 3 (same data, steps, seed and scorer init for both architectures; accuracy, ECE, NLL, Brier) |
| `compare_train.py` | paired bootstrap CIs and exact McNemar between two `train.py` prediction files |
| `flex_check.py` | FlexAttention mask_mod vs dense SDPA mask on CPU; BlockMask sparsity |
| `fp64_check.py` | fp32 vs fp64 parity through the `ikken` library's `check_parity` (Ettin-68m, ModernBERT-base) |
| `fetch_models.py`, `common.py`, `summarize.py` | download + verify weights; env capture and loading; tables |

## How to run

```
HF_HOME=<cache dir> python fetch_models.py
python exactness.py ettin-17m ettin-68m modernbert-base
RO_ONE_REP_S=1000 python latency.py ettin-68m 128,512,2048 1,2,5,10,20,50 "" 1000
RO_BS_MAX=1 RO_ONE_REP_S=1000 python latency.py ettin-68m 128,512,2048 1,5,20 _bs1 1000
R=<pred dir> python train.py --arch readonce --model ettin-17m --steps 150 --batch 8 --k 4 --lr 2e-4 --threads 4
R=<pred dir> python train.py --arch cross --model ettin-17m --steps 150 --batch 8 --k 4 --lr 2e-4 --threads 4
R=<pred dir> python compare_train.py pred_train_readonce_ettin-17m_raw.json pred_train_cross_ettin-17m_raw.json
python summarize.py
```

`RO_ONE_REP_S=1000` forces at least 3 timed reps even for slow configs — the default takes a single, un-warmed-up
rep once one rep alone would exceed 20 s.

---
Raw logs: absolute local paths were replaced with `<local>` or a repo-relative path by the publishing script;
nothing else was changed.
