"""Read once, answer many: exact multi-question encoding for ModernBERT-family encoders.

A decision model often asks several questions about the same input (the "state"). Re-encoding the
state for every question costs one full pass per question. Read-once packs the state and K question
blocks into a single sequence:

    [CLS] state [SEP] | block 1 | block 2 | ... | block K

and encodes it with a block attention mask and parallel positions:

* state tokens attend only to state tokens, so the state encoding does not depend on the questions;
* the tokens of block k attend to the state and to block k, never to other blocks;
* every block's position ids restart right after the state, so each block sees the state exactly as
  it would if it were the only question.

Each block's hidden states then equal those of the same block packed alone, up to floating-point
tolerance, however many other blocks share the sequence. `check_parity` measures this for any model.

The ModernBERT trap: ModernBERT's local-attention layers see only tokens within a sliding window.
transformers builds that window from *sequence index* while rotary embeddings use *position ids*.
With parallel positions the two disagree, and every block after the first silently sees a different
slice of the state. `read_once_masks` computes the window from position ids instead.

Requirements: a ModernBERT-architecture encoder (ModernBERT, Ettin, mmBERT) in transformers >= 5.17,
which accepts a per-layer-type dict of attention masks, loaded with ``attn_implementation="sdpa"`` or
``"eager"``. FlashAttention applies its window by sequence index inside the kernel and cannot express
this mask.

Read-once changes what a model computes (the state no longer sees the question), so a model must be
trained or fine-tuned with these masks. Applying them to a model trained as a cross-encoder changes
its answers.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal, Sequence

import torch

__all__ = [
    "Block",
    "PackedRow",
    "choice_block",
    "pack",
    "collate",
    "read_once_masks",
    "encode",
    "check_parity",
    "ReadOnceEncoder",
]


# ----------------------------------------------------------------------------- data structures
@dataclass
class Block:
    """One question block: token ids plus the offsets (within the block) of any marker tokens whose
    hidden states a head will read, e.g. one marker before each answer option."""

    ids: list[int]
    markers: list[int] = field(default_factory=list)

    def __post_init__(self):
        if not self.ids:
            raise ValueError("a block needs at least one token")
        if any(not 0 <= m < len(self.ids) for m in self.markers):
            raise ValueError("marker offsets must index into the block")


@dataclass
class PackedRow:
    """One packed sequence: [CLS] state [SEP] followed by the blocks."""

    input_ids: list[int]
    position_ids: list[int]
    segment_ids: list[int]  # 0 = state, k = block k (1-based)
    block_starts: list[int]  # index of each block's first token
    block_lengths: list[int]
    markers: list[list[int]]  # absolute marker indices, per block
    state_len: int  # number of state tokens, including [CLS] and [SEP]


# ----------------------------------------------------------------------------- packing
def choice_block(tokenizer, question: str, options: Sequence[str], *, marker_token: str | None = None,
                 max_question_tokens: int = 96, max_option_tokens: int = 16) -> Block:
    """A multiple-choice block: question tokens, [SEP], then a marker before each option.

    The marker defaults to the tokenizer's mask token. Marker hidden states are what a choice head
    scores, one per option.
    """
    if isinstance(options, str) or not options:
        raise ValueError("options must be a non-empty list of strings")
    if any(not isinstance(o, str) or not o.strip() for o in options):
        raise ValueError("every option must be a non-empty string")
    marker = marker_token or tokenizer.mask_token
    if marker is None:
        raise ValueError("the tokenizer has no mask token; pass marker_token=")
    marker_id = tokenizer.convert_tokens_to_ids(marker)
    if marker_id is None or (marker_id == tokenizer.unk_token_id and marker != tokenizer.unk_token):
        raise ValueError(f"marker token {marker!r} is not in the tokenizer's vocabulary")
    if tokenizer.sep_token_id is None:
        raise ValueError("the tokenizer has no sep token")
    enc = tokenizer([question] + [" " + o for o in options], add_special_tokens=False)["input_ids"]
    if len(enc[0]) > max_question_tokens:
        warnings.warn(f"question truncated from {len(enc[0])} to {max_question_tokens} tokens "
                      "(raise max_question_tokens to keep it whole)", stacklevel=2)
    long_opts = sum(len(o) > max_option_tokens for o in enc[1:])
    if long_opts:
        warnings.warn(f"{long_opts} option(s) truncated to {max_option_tokens} tokens "
                      "(raise max_option_tokens to keep them whole)", stacklevel=2)
    ids = list(enc[0][:max_question_tokens]) + [tokenizer.sep_token_id]
    markers = []
    for opt in enc[1:]:
        markers.append(len(ids))
        ids.append(marker_id)
        ids.extend(opt[:max_option_tokens])
    ids.append(tokenizer.sep_token_id)
    return Block(ids, markers)


def pack(state_ids: Sequence[int], blocks: Sequence[Block], *, cls_id: int, sep_id: int) -> PackedRow:
    """Pack one state and its blocks with parallel positions (every block starts at the same position)."""
    if not blocks:
        raise ValueError("pack needs at least one block")
    ids = [cls_id, *state_ids, sep_id]
    state_len = len(ids)
    pos, seg = list(range(state_len)), [0] * state_len
    starts, lengths, markers = [], [], []
    for k, b in enumerate(blocks, start=1):
        start = len(ids)
        ids.extend(b.ids)
        pos.extend(range(state_len, state_len + len(b.ids)))
        seg.extend([k] * len(b.ids))
        starts.append(start)
        lengths.append(len(b.ids))
        markers.append([start + m for m in b.markers])
    return PackedRow(ids, pos, seg, starts, lengths, markers, state_len)


def collate(rows: Sequence[PackedRow], *, pad_id: int) -> dict[str, torch.Tensor]:
    """Right-pad rows into tensors. Padding gets segment id -1."""
    B, L = len(rows), max(len(r.input_ids) for r in rows)
    input_ids = torch.full((B, L), pad_id, dtype=torch.long)
    position_ids = torch.zeros((B, L), dtype=torch.long)
    segment_ids = torch.full((B, L), -1, dtype=torch.long)
    attention_mask = torch.zeros((B, L), dtype=torch.long)
    for b, r in enumerate(rows):
        n = len(r.input_ids)
        input_ids[b, :n] = torch.tensor(r.input_ids)
        position_ids[b, :n] = torch.tensor(r.position_ids)
        segment_ids[b, :n] = torch.tensor(r.segment_ids)
        attention_mask[b, :n] = 1
    return {"input_ids": input_ids, "position_ids": position_ids, "segment_ids": segment_ids,
            "attention_mask": attention_mask}


# ----------------------------------------------------------------------------- masks
_ROW_CHUNK = 512


def read_once_masks(segment_ids: torch.Tensor, position_ids: torch.Tensor, sliding_window: int, *,
                    window_on: Literal["position", "index"] = "position") -> dict[str, torch.Tensor]:
    """Boolean masks [B, 1, L, L] (True = may attend) for ModernBERT's two layer types.

    ``sliding_window`` is the model's half-window (``config.sliding_window``, 64 for a 128-token local
    window). ``window_on="index"`` reproduces what transformers does by default; it exists only to
    demonstrate the trap and should not be used otherwise.
    """
    if window_on not in ("position", "index"):
        raise ValueError(f"window_on must be 'position' or 'index', got {window_on!r}")
    B, L = segment_ids.shape
    q, k = segment_ids[:, :, None], segment_ids[:, None, :]
    allowed = (k == 0) | (k == q)
    allowed &= (q >= 0) & (k >= 0)
    idx = torch.arange(L, device=segment_ids.device)
    pad_self = (q < 0) & (idx[:, None] == idx[None, :])  # padding rows attend to themselves: no empty row
    allowed |= pad_self
    pos = position_ids.to(torch.int32) if window_on == "position" else idx.to(torch.int32)[None].expand(B, L)
    local = allowed.clone()
    for i in range(0, L, _ROW_CHUNK):  # [B, chunk, L] distances at a time, not [B, L, L]
        j = min(i + _ROW_CHUNK, L)
        local[:, i:j] &= (pos[:, i:j, None] - pos[:, None, :]).abs() <= sliding_window
    local |= pad_self
    return {"full_attention": allowed[:, None], "sliding_attention": local[:, None]}


def _check_model(model):
    from transformers import ModernBertModel

    if not isinstance(model, ModernBertModel):
        if isinstance(getattr(model, "_orig_mod", None), ModernBertModel):
            raise TypeError("pass the uncompiled encoder: use the compiled module's `._orig_mod` "
                            "(torch.compile support is untested)")
        if isinstance(getattr(model, "model", None), ModernBertModel):
            raise TypeError(f"pass the bare encoder, not {type(model).__name__}: use its `.model` "
                            "(e.g. AutoModelForMaskedLM.from_pretrained(...).model)")
        raise TypeError("read-once supports ModernBERT-architecture encoders only (ModernBERT, Ettin, mmBERT, "
                        f"i.e. transformers' ModernBertModel); got {type(model).__name__}")
    config = model.config
    impl = getattr(config, "_attn_implementation", None) or "eager"  # transformers runs eager when unset
    if impl not in ("sdpa", "eager"):
        raise ValueError(f"read-once needs attn_implementation 'sdpa' or 'eager', got {impl!r}. "
                         "FlashAttention applies its sliding window by sequence index inside the kernel.")
    return impl


def encode(model, batch: dict[str, torch.Tensor], *,
           window_on: Literal["position", "index"] = "position") -> torch.Tensor:
    """Run a ModernBERT-architecture encoder over a collated read-once batch; returns last_hidden_state."""
    impl = _check_model(model)
    masks = read_once_masks(batch["segment_ids"], batch["position_ids"], model.config.sliding_window,
                            window_on=window_on)
    if impl == "eager":  # eager attention adds the mask to the scores
        dtype = next(model.parameters()).dtype
        neg = torch.finfo(dtype).min
        masks = {name: torch.zeros(m.shape, dtype=dtype, device=m.device).masked_fill(~m, neg)
                 for name, m in masks.items()}
    out = model(input_ids=batch["input_ids"], attention_mask=masks, position_ids=batch["position_ids"])
    return out.last_hidden_state


def _to(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def check_parity(model, state_ids: Sequence[int], blocks: Sequence[Block], *, cls_id: int, sep_id: int,
                 pad_id: int, window_on: Literal["position", "index"] = "position") -> dict[str, float]:
    """Encode all blocks together, then each block alone, and report the largest difference.

    Returns the max |difference| over every block token's hidden state and over the state tokens, plus
    the largest hidden-state magnitude for scale. With a correct setup the differences are rounding
    noise from different summation orders. In fp32 on real checkpoints they reach ~1e-4 of
    ``max_abs_hidden`` on models with large activations (ModernBERT-base), while option logits and
    probabilities agree to ~1e-6 and ~1e-7. In fp64 they shrink by orders of magnitude.

    What this proves: packing K blocks together gives the same result as packing each block alone, on
    your model and inputs. Both sides use this library's masks, so it does not by itself prove that
    the masks match stock transformers; the test suite checks that against transformers' own mask
    builders. The model is put in eval mode for the check (dropout would make it meaningless).
    """
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    try:
        row = pack(state_ids, blocks, cls_id=cls_id, sep_id=sep_id)
        h_all = encode(model, _to(collate([row], pad_id=pad_id), device), window_on=window_on)[0]
        worst_block, worst_state = 0.0, 0.0
        for k, b in enumerate(blocks):
            single = pack(state_ids, [b], cls_id=cls_id, sep_id=sep_id)
            h_one = encode(model, _to(collate([single], pad_id=pad_id), device), window_on=window_on)[0]
            a, s = row.block_starts[k], single.block_starts[0]
            worst_block = max(worst_block,
                              (h_all[a:a + len(b.ids)] - h_one[s:s + len(b.ids)]).abs().max().item())
            worst_state = max(worst_state, (h_all[:row.state_len] - h_one[:single.state_len]).abs().max().item())
    finally:
        model.train(was_training)
    return {"max_abs_diff_blocks": worst_block, "max_abs_diff_state": worst_state, "n_blocks": len(blocks),
            "max_abs_hidden": h_all.abs().max().item()}


# ----------------------------------------------------------------------------- convenience wrapper
class ReadOnceEncoder:
    """Tokenise, pack, encode and gather marker states for a ModernBERT-architecture encoder.

    >>> ro = ReadOnceEncoder(model, tokenizer)
    >>> blocks = [choice_block(tokenizer, "Is this urgent?", ["yes", "no"])]
    >>> batch, rows = ro.pack(["Server down since 9am, customers can't log in."], [blocks])
    >>> markers = ro.markers(ro.encode(batch), rows)   # markers[row][block] -> [n_options, hidden]
    """

    def __init__(self, model, tokenizer):
        _check_model(model)
        self.model, self.tokenizer = model, tokenizer
        self.cls_id, self.sep_id, self.pad_id = tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id
        missing = [n for n, v in (("cls", self.cls_id), ("sep", self.sep_id), ("pad", self.pad_id)) if v is None]
        if missing:
            raise ValueError(f"the tokenizer has no {', '.join(missing)} token id")

    def state_ids(self, text: str, max_tokens: int | None = None) -> list[int]:
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        return ids[:max_tokens] if max_tokens is not None else ids

    def pack(self, states: Sequence[str], blocks: Sequence[Sequence[Block]], *, max_state_tokens: int | None = None):
        """Pack a batch: ``states[i]`` is one text and ``blocks[i]`` the list of blocks asked about it."""
        if isinstance(states, str):
            raise TypeError("states must be a list of strings, one per row (got a single string)")
        if len(states) != len(blocks):
            raise ValueError(f"got {len(states)} states but {len(blocks)} block lists")
        for bl in blocks:
            if isinstance(bl, Block) or not all(isinstance(b, Block) for b in bl):
                raise TypeError("blocks must be a list holding one list of Block per state")
        rows = [pack(self.state_ids(s, max_state_tokens), bl, cls_id=self.cls_id, sep_id=self.sep_id)
                for s, bl in zip(states, blocks)]
        return collate(rows, pad_id=self.pad_id), rows

    def encode(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        device = next(self.model.parameters()).device
        return encode(self.model, _to(batch, device))

    @staticmethod
    def markers(hidden: torch.Tensor, rows: Sequence[PackedRow]) -> list[list[torch.Tensor]]:
        return [[hidden[b, idx] for idx in row.markers] for b, row in enumerate(rows)]

    def check_parity(self, state: str, blocks: Sequence[Block], *,
                     max_state_tokens: int | None = None) -> dict[str, float]:
        return check_parity(self.model, self.state_ids(state, max_state_tokens), blocks, cls_id=self.cls_id,
                            sep_id=self.sep_id, pad_id=self.pad_id)
