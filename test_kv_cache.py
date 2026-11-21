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
