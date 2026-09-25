"""Read-once, two-stage form ("prefix KV"): the same function as the packed/masked pass, no L x L mask.

Because state tokens never attend to question tokens, the state's hidden states at every layer do
not depend on the questions. So:
  stage 1: run the encoder on [CLS] state [SEP] alone and keep each layer's rotary-applied K and V;
  stage 2: run all K question blocks as a batch of K short rows (positions S..S+len-1); at each
           layer a block attends to concat(state K/V of that layer, its own K/V).
Attention work drops from (S + sum Lq)^2 (dense mask) to S^2 + sum Lq * (S + Lq), and each stage
is an ordinary attention call. On a GPU this is FlashAttention-varlen friendly: the only mask is
padding plus ModernBERT's local window, and with keys laid out as [state; block] the key index equals
the key position, so a (64, 64) window with bottom-right alignment is the position-space window.

This re-implements the ModernBertEncoderLayer forward with the module's own weights (transformers
5.17 layout: attn_norm, attn.Wqkv, attn.Wo, mlp_norm, mlp; final_norm; rotary_emb per layer type).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers.models.modernbert.modeling_modernbert import apply_rotary_pos_emb


def _qkv(layer, x, cos, sin):
    B, L, _ = x.shape
    attn = layer.attn
    qkv = attn.Wqkv(x).view(B, L, 3, -1, attn.head_dim)
    q, k, v = (t.transpose(1, 2) for t in qkv.unbind(dim=2))  # [B, H, L, Dh]
    q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
    return q, k, v


def _finish(layer, h, o):
    B, H, L, Dh = o.shape
    o = layer.attn.Wo(o.transpose(1, 2).reshape(B, L, H * Dh))
    h = h + o
    return h + layer.mlp(layer.mlp_norm(h))


@torch.inference_mode()
def encode_state(enc, state_ids: torch.Tensor):
    """state_ids: [1, S]. Returns (final hidden [1,S,d], per-layer (k, v) caches)."""
    S = state_ids.shape[1]
    pos = torch.arange(S)[None]
    h = enc.embeddings(input_ids=state_ids)
    rope = {t: enc.rotary_emb(h, pos, t) for t in set(enc.config.layer_types)}
    win = enc.config.sliding_window
    idx = torch.arange(S)
    local = ((idx[:, None] - idx[None, :]).abs() <= win)[None, None]
    cache = []
    for layer in enc.layers:
        cos, sin = rope[layer.attention_type]
        q, k, v = _qkv(layer, layer.attn_norm(h), cos, sin)
        mask = local if layer.attention_type == "sliding_attention" else None
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=layer.attn.head_dim ** -0.5)
        cache.append((k, v))
        h = _finish(layer, h, o)
    return enc.final_norm(h), cache


@torch.inference_mode()
def encode_blocks(enc, cache, S: int, blocks: list[list[int]], pad_id: int):
    """blocks: token ids of each question block. Returns final hidden [K, Lq_max, d] (padded rows)."""
    K, Lq = len(blocks), max(len(b) for b in blocks)
    ids = torch.full((K, Lq), pad_id, dtype=torch.long)
    valid = torch.zeros((K, Lq), dtype=torch.bool)
    for i, b in enumerate(blocks):
        ids[i, :len(b)] = torch.tensor(b)
        valid[i, :len(b)] = True
    pos = (S + torch.arange(Lq))[None].expand(K, -1)
    h = enc.embeddings(input_ids=ids)
    rope = {t: enc.rotary_emb(h, pos, t) for t in set(enc.config.layer_types)}
    win = enc.config.sliding_window
    # keys = [state (positions 0..S-1) ; block (positions S..S+Lq-1)], so key index == key position
    qpos = S + torch.arange(Lq)
    kpos = torch.arange(S + Lq)
    kvalid = torch.cat([torch.ones((K, S), dtype=torch.bool), valid], dim=1)       # [K, S+Lq]
    # a padded query always sees its own key: no all-masked rows (NaN would leak via 0 * NaN)
    self_pad = (~valid)[:, :, None] & (kpos[None, None, :] == qpos[None, :, None])  # [K, Lq, S+Lq]
    near = (qpos[:, None] - kpos[None, :]).abs() <= win                           # [Lq, S+Lq]
    full = (kvalid[:, None, :] | self_pad)[:, None]
    local = ((kvalid[:, None, :] & near[None]) | self_pad)[:, None]
    for li, layer in enumerate(enc.layers):
        cos, sin = rope[layer.attention_type]
        q, k, v = _qkv(layer, layer.attn_norm(h), cos, sin)
        ks, vs = cache[li]
        k = torch.cat([ks.expand(K, -1, -1, -1), k], dim=2)
        v = torch.cat([vs.expand(K, -1, -1, -1), v], dim=2)
        mask = local if layer.attention_type == "sliding_attention" else full
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=layer.attn.head_dim ** -0.5)
        h = _finish(layer, h, o)
    return enc.final_norm(h), valid


@torch.inference_mode()
def answer(model, tok, state_ids: list[int], blocks: list[tuple[list[int], list[int]]]):
    """ChoiceModel-compatible scoring. Returns [K, M] logits (-inf for missing options)."""
    enc = model.encoder
    sid = torch.tensor([[tok.cls_token_id] + list(state_ids) + [tok.sep_token_id]])
    _, cache = encode_state(enc, sid)
    hb, _ = encode_blocks(enc, cache, sid.shape[1], [b[0] for b in blocks], tok.pad_token_id)
    M = max(len(b[1]) for b in blocks)
    idx = torch.zeros((len(blocks), M), dtype=torch.long)
    m = torch.zeros((len(blocks), M), dtype=torch.bool)
    for i, (_, mk) in enumerate(blocks):
        idx[i, :len(mk)] = torch.tensor(mk)
        m[i, :len(mk)] = True
    g = torch.gather(hb, 1, idx[:, :, None].expand(-1, -1, hb.size(-1)))
    return model.scorer(g).squeeze(-1).float().masked_fill(~m, float("-inf"))
