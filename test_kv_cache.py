"""KV cache correctness: feeding tokens incrementally through the cache must give the same
logits as one full forward pass. Tiny random model on CPU, so it runs in about a second."""
import pytest, torch
from gpt2 import GPT, GPTConfig

@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    cfg = GPTConfig(block_size=64, vocab_size=100, n_layer=2, n_head=4, n_embd=32)
    return GPT(cfg).eval()

def run_in_chunks(model, ids, chunks):
    # feed ids through the cache chunk by chunk, return the logits of every position
    assert sum(chunks) == ids.size(1)
    cache, outs, start = model.new_kv_cache(), [], 0
    with torch.no_grad():
        for n in chunks:
            outs.append(model(ids[:, start:start + n], kv_cache=cache))
            start += n
    return torch.cat(outs, dim=1), cache

@pytest.mark.parametrize("chunks", [
    [20],                    # whole prompt at once (prefill only)
    [1] * 20,                # one token at a time (pure decode)
    [5, 1, 3, 1, 6, 4],      # prefill, then mixed chunk sizes: catches wrong masks / positions
])
def test_cached_logits_match_full_forward(model, chunks):
    ids = torch.randint(0, 100, (2, 20))               # batch of 2
    with torch.no_grad():
        full = model(ids)
    cached, cache = run_in_chunks(model, ids, chunks)
    assert cache.seq_len() == 20
    assert torch.allclose(cached, full, atol=1e-5), f"max diff {(cached - full).abs().max():.3e}"

def test_cached_greedy_matches_uncached(model):
    # the cache is an optimisation: it must not change a single generated token
    ids = torch.randint(0, 100, (1, 10))
    a = model.greedy_search(ids, max_new_tokens=40, use_cache=True)
    b = model.greedy_search(ids, max_new_tokens=40, use_cache=False)
    assert a.tolist() == b.tolist()

def test_cached_greedy_stops_at_block_size(model):
    ids = torch.randint(0, 100, (1, 10))
    out = model.greedy_search(ids, max_new_tokens=500, use_cache=True)
    assert out.size(1) == model.config.block_size      # 64 for this tiny model

# ---- batched decoding: B prompts of the same length generated together ----

def test_batched_greedy_cached_matches_uncached(model):
    ids = torch.randint(0, 100, (3, 10))
    a = model.greedy_search(ids, max_new_tokens=30, use_cache=True)
    b = model.greedy_search(ids, max_new_tokens=30, use_cache=False)
    assert a.shape == (3, 40)
    assert a.tolist() == b.tolist()

def test_batched_greedy_rows_match_single_runs(model):
    # each row of a batch must come out exactly as if it had been generated alone
    ids = torch.randint(0, 100, (4, 10))
    batched = model.greedy_search(ids, max_new_tokens=25, use_cache=True)
    for i in range(4):
        single = model.greedy_search(ids[i:i + 1], max_new_tokens=25, use_cache=True)
        assert batched[i].tolist() == single[0].tolist(), f"row {i} differs"

def test_batched_eos_pads_finished_rows(model):
    ids = torch.randint(0, 100, (3, 10))
    free = model.greedy_search(ids, max_new_tokens=20)            # no EOS: what each row would say
    eos = free[0, 13].item()                                       # row 0's 4th new token plays the role of EOS
    out = model.greedy_search(ids, max_new_tokens=20, eos_token_id=eos)
    for row in out[:, 10:].tolist():
        if eos in row:
            assert all(t == eos for t in row[row.index(eos):]), "tokens after EOS must be EOS padding"

# ---- cached top-k / top-p ----

def test_cached_top_k_one_equals_greedy(model):
    ids = torch.randint(0, 100, (2, 10))
    a = model.top_k_sampling(ids, max_new_tokens=25, k=1, use_cache=True)
    assert a.tolist() == model.greedy_search(ids, max_new_tokens=25).tolist()

@pytest.mark.parametrize("method,kwargs", [("top_k", dict(k=20)), ("top_p", dict(p=0.9))])
def test_cached_sampling_matches_uncached_with_same_seed(model, method, kwargs):
    # same seed => same random draws; the cache only changes how the logits are computed, so the tokens must agree
    ids = torch.randint(0, 100, (2, 10))
    outs = []
    for use_cache in (True, False):
        torch.manual_seed(123)
        outs.append(model.generate(ids, max_new_tokens=30, method=method, use_cache=use_cache, **kwargs))
    assert outs[0].tolist() == outs[1].tolist()
