"""Read-once, answer-many packing for a ModernBERT-style encoder (prototype).

Packed sequence (one row per state):

    [CLS] state [SEP] | Q1 block | Q2 block | ... | QK block

    Q block = "choice question: <text>" [SEP] [MASK] opt_0 [MASK] opt_1 ... [SEP]

Attention (built here, passed to the stock Hugging Face ModernBertModel as a per-layer-type mask dict):
    * state tokens attend only to state tokens (the state encoding is question-agnostic);
    * tokens of block i attend to all state tokens and to block i;
    * no attention between different blocks.
Position ids: the state uses 0..S-1 and every block restarts at S ("parallel positions"), so each
question sees the state exactly as if it were the only question.

ModernBERT's local layers use a sliding window. The stock mask computes it on *sequence index*
(|q_idx - kv_idx| <= 64); with parallel positions the window must be computed on *position id*
instead, otherwise block 2..K would be "far" from the end of the state and see less of it than
block 1 does. `build_masks(window_on="position")` does that.

Baseline ("Laya-style" cross-encoder): one row per question, [CLS] Q block state [SEP], full
bidirectional attention, so the state is re-encoded for every question (and is question-aware).
Both models share the same scorer: an MLP over the hidden state at each [MASK] marker, softmax
over that question's options. Yes/no questions are 2-option choice questions.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------- tokenisation
def encode_state(tok, text: str, max_tokens: int | None = None) -> list[int]:
    ids = tok(text.replace(tok.mask_token, " "), add_special_tokens=False)["input_ids"]
    return ids[:max_tokens] if max_tokens else ids


def encode_block(tok, q: dict, max_opt_tokens: int = 16, max_text_tokens: int = 96):
    """One question block. Returns (ids, marker offsets relative to the block start)."""
    texts = ["choice question: " + q["text"]] + [" " + o for o in q["options"]]
    texts = [t.replace(tok.mask_token, " ") for t in texts]
    enc = tok(texts, add_special_tokens=False)["input_ids"]
    ids = enc[0][:max_text_tokens] + [tok.sep_token_id]
    markers = []
    for opt_ids in enc[1:]:
        markers.append(len(ids))
        ids.append(tok.mask_token_id)
        ids.extend(opt_ids[:max_opt_tokens])
    ids.append(tok.sep_token_id)
    return ids, markers


# ----------------------------------------------------------------------------- packing
def pack_readonce(tok, state_ids: list[int], blocks: list[tuple[list[int], list[int]]],
                  positions: str = "parallel") -> dict:
    """[CLS] state [SEP] + blocks. seg: 0 = state, k = k-th block (1-based)."""
    ids = [tok.cls_token_id] + list(state_ids) + [tok.sep_token_id]
    S = len(ids)
    pos, seg, q_markers = list(range(S)), [0] * S, []
    for k, (bids, bmark) in enumerate(blocks, start=1):
        off = len(ids)
        ids.extend(bids)
        start = S if positions == "parallel" else off  # "sequential" = ordinary 0..L-1 positions
        pos.extend(range(start, start + len(bids)))
        seg.extend([k] * len(bids))
        q_markers.append([off + m for m in bmark])
    return {"ids": ids, "pos": pos, "seg": seg, "q_markers": q_markers, "state_len": S}


def pack_cross(tok, state_ids: list[int], block: tuple[list[int], list[int]]) -> dict:
    """Laya-style row: [CLS] Q block state [SEP]; ordinary positions, full attention (seg all 0)."""
    bids, bmark = block
    ids = [tok.cls_token_id] + list(bids) + list(state_ids) + [tok.sep_token_id]
    return {"ids": ids, "pos": list(range(len(ids))), "seg": [0] * len(ids),
            "q_markers": [[1 + m for m in bmark]], "state_len": None}


def collate(items: list[dict], pad_id: int, labels: list[list[int]] | None = None) -> dict:
    """Pad rows; flatten every question of every row into a [Q, M] table of marker indices
    into the flattened [B*L] hidden states."""
    B, L = len(items), max(len(it["ids"]) for it in items)
    ids = torch.full((B, L), pad_id, dtype=torch.long)
    pos = torch.zeros((B, L), dtype=torch.long)
    seg = torch.full((B, L), -1, dtype=torch.long)
    att = torch.zeros((B, L), dtype=torch.long)
    qm_rows, q_row = [], []
    for b, it in enumerate(items):
        n = len(it["ids"])
        ids[b, :n] = torch.tensor(it["ids"])
        pos[b, :n] = torch.tensor(it["pos"])
        seg[b, :n] = torch.tensor(it["seg"])
        att[b, :n] = 1
        for m in it["q_markers"]:
            qm_rows.append([b * L + x for x in m])
            q_row.append(b)
    Q, M = len(qm_rows), max((len(m) for m in qm_rows), default=1)
    qm = torch.zeros((Q, M), dtype=torch.long)
    qmask = torch.zeros((Q, M), dtype=torch.bool)
    for i, m in enumerate(qm_rows):
        qm[i, :len(m)] = torch.tensor(m)
        qmask[i, :len(m)] = True
    out = {"input_ids": ids, "position_ids": pos, "seg": seg, "attention_mask": att,
           "q_markers": qm, "q_mask": qmask, "q_row": torch.tensor(q_row)}
    if labels is not None:
        out["labels"] = torch.tensor([y for row in labels for y in row], dtype=torch.long)
    return out


# ----------------------------------------------------------------------------- masks
def build_masks(seg: torch.Tensor, pos: torch.Tensor, window: int, impl: str = "sdpa",
                dtype: torch.dtype = torch.float32, window_on: str = "position",
                isolate_blocks: bool = True) -> dict:
    """4D masks [B, 1, L, L] for ModernBertModel's {"full_attention", "sliding_attention"} layers.

    seg: [B, L] (0 = state, k>0 = block k, -1 = padding); pos: [B, L] position ids.
    True = may attend (sdpa); eager gets the additive 0 / finfo.min form.
    """
    B, L = seg.shape
    qs, ks = seg[:, :, None], seg[:, None, :]
    if isolate_blocks:
        allowed = (ks == 0) | (ks == qs)
    else:  # negative control: blocks may see each other (state still sees only state)
        allowed = (ks == 0) | ((qs > 0) & (ks > 0))
    allowed = allowed & (qs >= 0) & (ks >= 0)
    eye = torch.eye(L, dtype=torch.bool, device=seg.device)[None]
    pad_q = (qs < 0) & eye  # padding rows attend to themselves only (avoids all-masked NaN rows)
    allowed = allowed | pad_q
    if window_on == "position":
        dist = (pos[:, :, None] - pos[:, None, :]).abs()
    else:  # what the stock mask does: distance in sequence index
        idx = torch.arange(L, device=seg.device)
        dist = (idx[:, None] - idx[None, :]).abs()[None]
    local = (allowed & (dist <= window)) | pad_q
    masks = {"full_attention": allowed[:, None], "sliding_attention": local[:, None]}
    if impl == "eager":
        neg = torch.finfo(dtype).min
        masks = {k: torch.zeros(v.shape, dtype=dtype).masked_fill(~v, neg) for k, v in masks.items()}
    return masks


# ----------------------------------------------------------------------------- model
class ChoiceModel(nn.Module):
    """Encoder + per-option scorer at each [MASK] marker; softmax over a question's options."""

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        d = encoder.config.hidden_size
        self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.window = encoder.config.sliding_window  # half-window, e.g. 64

    def encode(self, batch: dict, mode: str = "readonce", **mask_kw) -> torch.Tensor:
        if mode == "stock":  # plain HF path: 2D padding mask, default positions and masks
            return self.encoder(input_ids=batch["input_ids"],
                                attention_mask=batch["attention_mask"]).last_hidden_state
        impl = self.encoder.config._attn_implementation
        masks = build_masks(batch["seg"], batch["position_ids"], self.window, impl=impl,
                            dtype=next(self.encoder.parameters()).dtype, **mask_kw)
        return self.encoder(input_ids=batch["input_ids"], attention_mask=masks,
                            position_ids=batch["position_ids"]).last_hidden_state

    def score(self, h: torch.Tensor, batch: dict) -> torch.Tensor:
        flat = h.reshape(-1, h.size(-1))
        m = flat[batch["q_markers"]]  # [Q, M, d]
        logits = self.scorer(m).squeeze(-1).float()
        return logits.masked_fill(~batch["q_mask"], float("-inf"))

    def forward(self, batch: dict, mode: str = "readonce", **mask_kw):
        h = self.encode(batch, mode, **mask_kw)
        return self.score(h, batch), h


def choice_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, labels)


# ----------------------------------------------------------------------------- convenience
def build_readonce_batch(tok, states: list[str], questions: list[list[dict]], max_state_tokens=None,
                         positions: str = "parallel", with_labels: bool = False) -> dict:
    items = []
    for text, qs in zip(states, questions):
        sids = encode_state(tok, text, max_state_tokens)
        items.append(pack_readonce(tok, sids, [encode_block(tok, q) for q in qs], positions=positions))
    labels = [[q["answer"] for q in qs] for qs in questions] if with_labels else None
    return collate(items, tok.pad_token_id, labels)


def build_cross_batch(tok, states: list[str], questions: list[list[dict]], max_state_tokens=None,
                      with_labels: bool = False) -> dict:
    """One row per (state, question): the state is re-read for every question."""
    items, labels = [], []
    for text, qs in zip(states, questions):
        sids = encode_state(tok, text, max_state_tokens)
        for q in qs:
            items.append(pack_cross(tok, sids, encode_block(tok, q)))
            labels.append([q.get("answer", -1)])
    return collate(items, tok.pad_token_id, labels if with_labels else None)
