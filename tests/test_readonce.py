"""Exactness tests on a tiny, randomly initialised ModernBERT (no downloads, runs in seconds).

Parity is a property of the masks and positions, not of the weights, so random weights test it fully.
"""
import pytest
import torch
from transformers import ModernBertConfig, ModernBertModel

from ikken import Block, check_parity, choice_block, collate, encode, pack, read_once_masks

CLS, SEP, PAD, MASK = 1, 2, 0, 3


def tiny_model(attn="sdpa", seed=0):
    torch.manual_seed(seed)
    cfg = ModernBertConfig(vocab_size=97, hidden_size=64, intermediate_size=96, num_hidden_layers=6,
                           num_attention_heads=4, local_attention=16, global_attn_every_n_layers=3,
                           max_position_embeddings=512, pad_token_id=PAD, cls_token_id=CLS,
                           sep_token_id=SEP, attn_implementation=attn)
    model = ModernBertModel(cfg).eval()
    assert model.config.sliding_window == 8  # half of local_attention: the window is small, so the trap shows
    return model


def rand_ids(n, gen):
    return torch.randint(4, 97, (n,), generator=gen).tolist()


def blocks_for(gen, k, lo=5, hi=13):
    out = []
    for _ in range(k):
        n = int(torch.randint(lo, hi, (1,), generator=gen))
        out.append(Block(rand_ids(n, gen), markers=[0]))
    return out


@pytest.mark.parametrize("attn", ["sdpa", "eager"])
def test_packed_equals_alone(attn):
    model, gen = tiny_model(attn), torch.Generator().manual_seed(1)
    rep = check_parity(model, rand_ids(40, gen), blocks_for(gen, 6), cls_id=CLS, sep_id=SEP, pad_id=PAD)
    assert rep["max_abs_diff_blocks"] < 1e-5, rep
    assert rep["max_abs_diff_state"] < 1e-5, rep


def test_stock_index_window_breaks_parity():
    """The trap: a sliding window on sequence index lets later blocks see a different slice of the state."""
    model, gen = tiny_model(), torch.Generator().manual_seed(2)
    rep = check_parity(model, rand_ids(40, gen), blocks_for(gen, 6), cls_id=CLS, sep_id=SEP, pad_id=PAD,
                       window_on="index")
    assert rep["max_abs_diff_blocks"] > 1e-3, rep


def test_block_order_does_not_matter():
    model, gen = tiny_model(), torch.Generator().manual_seed(3)
    state, blocks = rand_ids(30, gen), blocks_for(gen, 4)
    fwd, rev = pack(state, blocks, cls_id=CLS, sep_id=SEP), pack(state, blocks[::-1], cls_id=CLS, sep_id=SEP)
    with torch.no_grad():
        h = encode(model, collate([fwd, rev], pad_id=PAD))
    for k, b in enumerate(blocks):
        a, r = fwd.block_starts[k], rev.block_starts[len(blocks) - 1 - k]
        assert (h[0, a:a + len(b.ids)] - h[1, r:r + len(b.ids)]).abs().max().item() < 1e-5


def test_padding_in_a_batch_does_not_change_results():
    model, gen = tiny_model(), torch.Generator().manual_seed(4)
    rows = [pack(rand_ids(n, gen), blocks_for(gen, k), cls_id=CLS, sep_id=SEP) for n, k in ((50, 3), (12, 5))]
    with torch.no_grad():
        together = encode(model, collate(rows, pad_id=PAD))
        for b, row in enumerate(rows):
            alone = encode(model, collate([row], pad_id=PAD))[0]
            n = len(row.input_ids)
            assert (together[b, :n] - alone[:n]).abs().max().item() < 1e-5


def test_masks_isolate_blocks_and_restart_positions():
    row = pack([10, 11, 12], [Block([20, 21]), Block([30, 31, 32])], cls_id=CLS, sep_id=SEP)
    assert row.position_ids == [0, 1, 2, 3, 4, 5, 6, 5, 6, 7]  # both blocks start right after the state
    batch = collate([row], pad_id=PAD)
    full = read_once_masks(batch["segment_ids"], batch["position_ids"], sliding_window=64)["full_attention"][0, 0]
    s, b1, b2 = slice(0, 5), slice(5, 7), slice(7, 10)
    assert full[s, s].all() and not full[s, b1].any() and not full[s, b2].any()  # state sees only the state
    assert full[b1, s].all() and full[b1, b1].all() and not full[b1, b2].any()  # block 1 never sees block 2
    assert full[b2, s].all() and full[b2, b2].all() and not full[b2, b1].any()


def test_flash_attention_is_rejected():
    model = tiny_model()
    model.config._attn_implementation = "flash_attention_2"
    batch = collate([pack([10, 11], [Block([20])], cls_id=CLS, sep_id=SEP)], pad_id=PAD)
    with pytest.raises(ValueError, match="sdpa"):
        encode(model, batch)


def test_block_validation():
    with pytest.raises(ValueError):
        Block([])
    with pytest.raises(ValueError):
        Block([5, 6], markers=[2])


# ----------------------------------------------------------------------------- against stock transformers
def reference_masks(model, batch):
    """The read-once masks built independently with transformers' own mask builders."""
    from transformers.masking_utils import and_masks, create_bidirectional_mask

    seg, pos, w = batch["segment_ids"], batch["position_ids"], model.config.sliding_window

    def blocks(b, h, q, kv):
        return (seg[b, kv] == 0) | (seg[b, kv] == seg[b, q])

    def window(b, h, q, kv):
        return (pos[b, q] - pos[b, kv]).abs() <= w

    kw = dict(config=model.config, inputs_embeds=model.embeddings(input_ids=batch["input_ids"]),
              attention_mask=batch["attention_mask"])
    return {"full_attention": create_bidirectional_mask(**kw, and_mask_function=blocks),
            "sliding_attention": create_bidirectional_mask(**kw, and_mask_function=and_masks(blocks, window))}


def mixed_batch(gen):
    rows = [pack(rand_ids(n, gen), blocks_for(gen, k, 3, 20), cls_id=CLS, sep_id=SEP) for n, k in ((45, 4), (7, 6))]
    return rows, collate(rows, pad_id=PAD)


@pytest.mark.parametrize("chunk", [512, 7])  # 7: the window is built over many row chunks
def test_masks_equal_transformers_reference(chunk, monkeypatch):
    import ikken.readonce

    monkeypatch.setattr(ikken.readonce, "_ROW_CHUNK", chunk)
    model, gen = tiny_model(), torch.Generator().manual_seed(5)
    rows, batch = mixed_batch(gen)
    ours = read_once_masks(batch["segment_ids"], batch["position_ids"], model.config.sliding_window)
    ref = reference_masks(model, batch)
    for name in ours:
        for b, row in enumerate(rows):
            n = len(row.input_ids)  # padding query rows differ by design (ours attend to themselves)
            assert torch.equal(ours[name][b, 0, :n], ref[name][b, 0, :n]), name


@pytest.mark.parametrize("attn", ["sdpa", "eager"])
def test_hidden_states_equal_transformers_reference(attn):
    model, gen = tiny_model(attn), torch.Generator().manual_seed(6)
    rows, batch = mixed_batch(gen)
    with torch.no_grad():
        ours = encode(model, batch)
        ref_masks = reference_masks(model, batch)  # bool for sdpa, additive float for eager
        ref = model(input_ids=batch["input_ids"], attention_mask=ref_masks,
                    position_ids=batch["position_ids"]).last_hidden_state
    for b, row in enumerate(rows):
        n = len(row.input_ids)
        assert (ours[b, :n] - ref[b, :n]).abs().max().item() < 1e-5


def test_state_equals_stock_encoding_of_state_alone():
    model, gen = tiny_model(), torch.Generator().manual_seed(7)
    state = rand_ids(60, gen)
    row = pack(state, blocks_for(gen, 5), cls_id=CLS, sep_id=SEP)
    with torch.no_grad():
        packed = encode(model, collate([row], pad_id=PAD))[0, :row.state_len]
        stock = model(input_ids=torch.tensor([[CLS, *state, SEP]])).last_hidden_state[0]
    assert (packed - stock).abs().max().item() < 1e-5


def test_unset_attn_implementation_runs_as_eager():
    model, gen = tiny_model(), torch.Generator().manual_seed(8)
    batch = collate([pack(rand_ids(30, gen), blocks_for(gen, 3), cls_id=CLS, sep_id=SEP)], pad_id=PAD)
    with torch.no_grad():
        sdpa = encode(model, batch)
        model.config._attn_implementation = None
        unset = encode(model, batch)
    assert (sdpa - unset).abs().max().item() < 1e-5


# ----------------------------------------------------------------------------- input checks
def test_rejects_decoders_and_head_wrappers():
    from transformers import ModernBertDecoderConfig, ModernBertDecoderModel, ModernBertForMaskedLM

    batch = collate([pack([10, 11], [Block([20])], cls_id=CLS, sep_id=SEP)], pad_id=PAD)
    dec = ModernBertDecoderModel(ModernBertDecoderConfig(vocab_size=97, hidden_size=64, intermediate_size=96,
                                                         num_hidden_layers=2, num_attention_heads=4,
                                                         pad_token_id=PAD, bos_token_id=CLS, eos_token_id=SEP,
                                                         cls_token_id=CLS, sep_token_id=SEP))
    with pytest.raises(TypeError, match="encoders only"):
        encode(dec, batch)
    mlm = ModernBertForMaskedLM(tiny_model().config)
    with pytest.raises(TypeError, match=r"\.model"):
        encode(mlm, batch)


class StubTokenizer:
    """Whitespace tokenizer with BERT-style special tokens, enough for the packing helpers."""
    cls_token_id, sep_token_id, pad_token_id, unk_token_id = CLS, SEP, PAD, 4
    mask_token, unk_token = "[MASK]", "[UNK]"

    def convert_tokens_to_ids(self, t):
        return {"[MASK]": MASK, "[UNK]": 4}.get(t, 4)

    def __call__(self, text, add_special_tokens=False):
        enc = lambda s: [5 + sum(map(ord, w)) % 90 for w in s.split()]  # noqa: E731
        return {"input_ids": enc(text) if isinstance(text, str) else [enc(t) for t in text]}


def test_choice_block_checks_and_warns():
    tok = StubTokenizer()
    b = choice_block(tok, "Is it urgent?", ["yes", "no"])
    assert [b.ids[m] for m in b.markers] == [MASK, MASK]
    for bad in ([], "yes", ["yes", ""]):
        with pytest.raises(ValueError):
            choice_block(tok, "q?", bad)
    with pytest.raises(ValueError, match="vocabulary"):
        choice_block(tok, "q?", ["a", "b"], marker_token="[NOPE]")
    with pytest.warns(UserWarning, match="question truncated"):
        choice_block(tok, "word " * 200, ["a", "b"])
    with pytest.warns(UserWarning, match="option"):
        choice_block(tok, "q?", ["x " * 40, "b"])


def test_encoder_pack_validates_inputs():
    from ikken import ReadOnceEncoder

    tok = StubTokenizer()
    ro = ReadOnceEncoder(tiny_model(), tok)
    q = [choice_block(tok, "Is it urgent?", ["yes", "no"])]
    with pytest.raises(TypeError, match="single string"):
        ro.pack("server down since 9am", [q])
    with pytest.raises(ValueError, match="3 states but 2"):
        ro.pack(["a b", "c d", "e f"], [q, q])
    with pytest.raises(TypeError, match="one list of Block"):
        ro.pack(["a b"], q)
    assert ro.state_ids("a b c", max_tokens=0) == []
