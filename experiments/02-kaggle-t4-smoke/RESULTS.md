# 02: Kaggle T4 smoke test (measured, 23 Sep 2026)

This experiment tests the read-once encoder on Kaggle's free GPUs, in two runs on the same day:

- **Smoke run**, 11:53:54–12:01:54 (about 8 min), a **GPU T4 x2** Kaggle notebook, using about 8 minutes of the
  30 h weekly quota. Raw results: `results/smoke_results_kaggle_t4.json` (9,926 bytes, SHA-256
  `def04359…b5b1b5a32bb`). Code: `smoke.py`.
- **Throughput-repeats run**, 12:11:31–12:18:02 (about 6.5 min), same accelerator, about 6.5 more minutes of quota:
  5 alternating read-once / per-question ("cross") training passes on Ettin-68m, S = 512, K = 8, fp16, 30 steps
  each, with `nvidia-smi` clocks, temperature and power sampled after every run. Raw results:
  `results/throughput_repeats_kaggle_t4.json` (7,119 bytes, SHA-256 `3af9aa06…1ae27c`).

Only `cuda:0` was used in both runs. `build_notebook.py` packs the prototype's `readonce.py` and `synth.py` with
`smoke.py` into `smoke.ipynb` (full run), and `python build_notebook.py --throughput-reps 5` builds
`throughput.ipynb` (the repeated pair only). `dryrun_cpu_results.json` is the laptop dry run that checked the code
paths before either GPU run.

## Environment

Python 3.12.13, torch 2.10.0+cu128 (CUDA 12.8), transformers 5.17.0 (pinned via pip), tokenizers 0.23.2,
huggingface_hub 1.32.0, 2× Tesla T4 (sm_75, 15.64 GB each, driver 580.159.04).

## Bottom line

| Question | Answer (measured) |
|---|---|
| Is read-once still exact on a GPU? | **Yes.** Ettin-68m, 64 questions (6 records, plus a 2,048-token state with 10 questions): fp32 max \|Δp\| **1.8e-7**, fp16 **2.1e-4**, 64/64 answers agree both ways. The sequence-index window (negative control, fp32) moves the probabilities by 1.2e-2 and flips 7 of 64 answers (57/64 agree), so the trap reproduces on GPU. |
| Does fp16 overflow? (fp16 max 65,504) | **Not at 32m or 68m, at S = 2,048, K = 8.** Peak activation 4,817 (Ettin-32m, 13.6× headroom) and 8,719–8,720 (Ettin-68m, 7.5× headroom), zero non-finite values either dtype. ModernBERT-base peaks at **42,464–42,470, only 1.5× headroom**. Risky, though nothing overflowed here. |
| How fast does it train on one free T4? (fp16 AMP, single pass) | Ettin-68m: **117.0 questions/s** at 512-token states (8 q/state), 36.5 q/s at 2,048 tokens (2,048-token states cost about 3.2× more per question). Ettin-32m: **338.4 q/s** at 512 tokens, 94.6 q/s at 2,048. ModernBERT-base: 70.3 q/s and 17.9 q/s. Ettin-68m in **fp32 at the same batch size ran out of memory** (see below) — no fp32 speed number at this config. |
| Read-once vs Laya-style training cost? | In this run's single pass, **4.7× more questions per second** (Ettin-68m, S = 512, K = 8: 117.0 vs 24.8 q/s, 8.29 vs 12.08 GB peak memory). **A repeated, alternating measurement (5 pairs, below) gives a more reliable median of 5.3×** (range 4.8×–5.4×), because a single pass' throughput depends heavily on how hot the GPU already is. |
| Does the custom mask work with fast attention kernels on T4? | **Yes.** SDPA's memory-efficient kernel accepts it: 450.5 ms fwd+bwd vs 1,338.7 ms for the math kernel (**3.0× faster**). |
| FlexAttention on T4? | **It compiles and is correct** (max diff 1.2e-4 vs SDPA in fp16). But it is **9.4× slower** than dense-mask SDPA for this shape (29.0 vs 3.1 ms median). Use SDPA on T4. |
| Checkpoint and resume? | **Exact.** The resumed loss equals the continued loss (diff 0.0, both 1.4306). Ettin-32m plus AdamW state is 384.5 MB. |

## 1. Exactness and fp16 headroom (`smoke.py` tests `exactness_ettin-68m`, `activations_*`)

Ettin-68m, 64 questions across 6 short records plus one 2,048-token state with 10 questions:

| dtype | max abs prob diff | argmax agree |
|---|---|---|
| fp32 | 1.8e-7 | 64/64 |
| fp16 | 2.1e-4 | 64/64 |
| fp32, sequence-index window (negative control) | 1.2e-2 | 57/64 |

fp16 activation headroom (S = 2,048, K = 8, 2 rows, L = 2,621 packed tokens; fp16 max representable value is
65,504):

| backbone | peak activation (fp32 / fp16) | headroom vs fp16 max | non-finite values |
|---|---|---|---|
| ettin-32m | 4,816.8 / 4,815.6 | 13.6× | 0 |
| ettin-68m | 8,718.8 / 8,720.0 | 7.5× | 0 |
| modernbert-base | 42,470.4 / 42,464.0 | 1.5× | 0 |

Nothing overflowed in this run, but ModernBERT-base's headroom is thin enough that a longer state or a different
seed could push it over; the two smaller backbones have a comfortable margin.

## 2. Training throughput, single pass (`smoke.py` tests `train_*`)

All fp16 AMP, K = 8 questions/state, batch sizes chosen to roughly fill a T4:

| backbone | S | rows/step | median step | tokens/s | questions/s | peak mem |
|---|---|---|---|---|---|---|
| ettin-68m | 512 | 16 | 1,094.4 ms | 11,471.6 | 117.0 | 8.29 GB |
| ettin-68m | 2,048 | 4 | 876.0 ms | 10,490.3 | 36.5 | 5.99 GB |
| ettin-32m | 512 | 16 | 378.3 ms | 33,190.9 | 338.4 | 3.53 GB |
| ettin-32m | 2,048 | 4 | 338.2 ms | 27,167.3 | 94.6 | 2.66 GB |
| modernbert-base | 512 | 8 | 910.8 ms | 6,823.1 | 70.3 | 8.33 GB |
| modernbert-base | 2,048 | 2 | 894.4 ms | 5,215.1 | 17.9 | 6.58 GB |
| ettin-68m, **fp32** | 512 | 16 | — | — | — | **OOM** |
| cross-encoder, ettin-68m | 512 | 32 (4 states × 8 q) | 1,289.8 ms | 13,475.5 | 24.8 | 12.08 GB |

No non-finite losses anywhere fp16 ran; GradScaler stayed at 32,768–65,536.

**The fp32 run at the same batch size (16 rows, S = 512, K = 8) ran out of memory**, reported as it happened:
`CUDA out of memory. Tried to allocate 34.00 MiB. GPU 0 has a total capacity of 14.56 GiB of which 26.81 MiB is
free. Including non-PyTorch memory, this process has 14.53 GiB memory in use. Of the allocated memory 12.48 GiB is
allocated by PyTorch...`. So there is no fp32-vs-fp16 speed comparison at this batch size from this run; fp32 would
need a smaller batch or gradient checkpointing to fit in 14.56 GiB of usable T4 memory.

## 3. Training throughput, repeated and alternated (`results/throughput_repeats_kaggle_t4.json`)

Five pairs of runs, alternating architecture so neither one always goes first or last, all Ettin-68m, S = 512,
K = 8, fp16, 30 timed steps (3 warm-up). `nvidia-smi` was sampled right after each run.

| round | arch | questions/s | SM clock (MHz) | temp (°C) | power draw (W) |
|---|---|---|---|---|---|
| 0 | read-once | 153.0 | 1,425 | 59 | 39.5 |
| 0 | cross | 28.6 | 1,365 | 69 | 59.6 |
| 1 | cross | 27.1 | 1,290 | 78 | 65.9 |
| 1 | read-once | 130.4 | 1,020 | 77 | 68.0 |
| 2 | read-once | 137.7 | 1,065 | 76 | 59.5 |
| 2 | cross | 25.9 | 1,080 | 77 | 38.5 |
| 3 | cross | 25.7 | 1,305 | 77 | 39.1 |
| 3 | read-once | 135.2 | 1,215 | 77 | 41.9 |
| 4 | read-once | 134.9 | 1,245 | 77 | 68.3 |
| 4 | cross | 25.9 | 1,170 | 77 | 38.8 |

| | median q/s | range |
|---|---|---|
| read-once | **135.2** | 130.4 – 153.0 |
| cross | **25.9** | 25.7 – 28.6 |
| paired ratio (read-once ÷ cross, same round) | **5.26×** | 4.81× – 5.35× |

Peak memory was constant across every round: 8.29 GB (read-once) vs 12.08 GB (cross).

**Why the single-pass numbers disagree.** Over these ten runs the GPU heats from about 59 °C to about 77 °C, and
the SM clock drops from about 1,425 MHz at the start to roughly 1,020–1,305 MHz once it's warm — `nvidia-smi`
reports an active throttle reason (`0x4`, a power cap) throughout. So a single run's throughput depends on when it
ran and how hot the card already was. An earlier single-pass comparison, run in a different Kaggle session, reported
153 vs 24.5 q/s (6.2×), with the read-once run happening on a cold GPU. This run's own single-pass smoke test
(section 2) got 117.0 vs 24.8 q/s (4.7×). The repeated, alternating measurement above — median 5.3×, range
4.8×–5.4× — averages out the clock/thermal state and **supersedes both single-pass numbers**.

## 4. Attention kernels on T4 (`smoke.py` tests `sdpa_backends_ettin-68m`, `flex`)

Ettin-68m, S = 2,048, K = 8, fwd+bwd timing:

| kernel | median fwd+bwd | vs math |
|---|---|---|
| SDPA, memory-efficient (`EFFICIENT_ATTENTION`) | 450.5 ms | 3.0× faster |
| SDPA, math (`MATH`) | 1,338.7 ms | 1.0× |

**FlexAttention** compiled and matched SDPA to 1.2e-4 max abs diff (fp16, eager not needed — it compiles on T4),
but is 9.4× slower for this shape: 29.0 ms median forward vs 3.1 ms for dense-mask SDPA. Use SDPA on T4, not Flex.

## 5. Checkpoint and resume (`smoke.py` test `resume_ettin-32m`)

Continuing training without interruption and resuming from a saved checkpoint gave identical loss: 1.4305552244
either way (abs diff 0.0). Ettin-32m's checkpoint plus AdamW optimizer state is 384.5 MB.

## What this means for the plan

- **Training fits the free tier easily, with some clock-dependent slack.** Using the repeated-measurement median
  (135.2 q/s, more reliable than either single pass), one epoch over 1M decisions (512-token states, 8 questions
  each) takes roughly **2.1 GPU-hours for Ettin-68m** (10⁶ / 135.2 q/s); this run's own single-pass smoke number
  (117.0 q/s) gives about 2.4 h. Ettin-32m's single-pass number (338.4 q/s) gives about **0.8 h** per epoch. Three
  epochs of Ettin-68m is roughly 6.2–7.1 h, about a fifth to a quarter of one week's 30-hour quota. States of
  2,048 tokens cost about 3.2× more per question (Ettin-68m: 117.0 vs 36.5 q/s). GPU quota is not the constraint.
- **Stick to 32m or 68m for fp16.** The 150m-class ModernBERT-base has thin fp16 headroom (1.5×); training it in
  fp32 at this batch size overflowed T4 memory, so it needs a smaller batch, gradient checkpointing, or fp32 only
  for its outlier layers.
- **On T4, use dense-mask SDPA (the memory-efficient kernel)**, not FlexAttention — FlexAttention is correct but
  9× slower here.
- **Read-once lowers training cost too**, not just inference: a repeated, alternating measurement puts it at a
  median 5.3× more questions per GPU-second than per-question re-encoding, not the 6.2× a single cold-GPU pass
  suggested.

## Caveats

- Single-pass numbers (sections 1–2) are one seed, one GPU, 20 timed steps after 3 warm-up steps unless noted;
  section 3 repeats and alternates 5 pairs of 30-step runs specifically because single-pass throughput turned out
  to depend on GPU temperature and clock state.
- The data is the prototype's synthetic records (`synth.py`): service-metrics records (region, tier, status, daily
  cloud spend, requests, errors, alerts, this/last-week error rate) and support tickets. Its templates are
  synthetic text written for measurement only, not for training a released model.
- Losses falling on synthetic data say nothing about real-world quality.
- Tokens/s counts non-padding tokens. Cross-encoder rows repeat the state for each question, hence its larger
  `rows_per_step` and peak memory at the same K.

## Files

| file | what |
|---|---|
| `smoke.py` | the full smoke test: env, exactness, fp16 activation headroom, training throughput per backbone, fp32 OOM check, SDPA/Flex kernel timing, resume check |
| `build_notebook.py` | packs `readonce.py`, `synth.py` and `smoke.py` into `smoke.ipynb`; `--throughput-reps N` builds `throughput.ipynb` instead |
| `smoke.ipynb` | the full run, as executed on Kaggle |
| `throughput.ipynb` | built with `python build_notebook.py --throughput-reps 5`; only the repeated read-once/cross throughput pair |
| `dryrun_cpu_results.json` | the laptop dry run that checked the code paths before spending GPU quota |
| `results/smoke_results_kaggle_t4.json` | raw output of the smoke run |
| `results/throughput_repeats_kaggle_t4.json` | raw output of the 5 alternating throughput pairs |

## How to run

On Kaggle, with **Accelerator = GPU T4 x2** and **Internet = on**:

```
python build_notebook.py                      # writes smoke.ipynb
python build_notebook.py --throughput-reps 5   # writes throughput.ipynb
```

Upload the resulting notebook and run all cells; each test is independent and results are saved to
`/kaggle/working/smoke_results.json` after every test.

---
Raw logs: absolute local paths were replaced with `<local>` or a repo-relative path by the publishing script;
nothing else was changed.
