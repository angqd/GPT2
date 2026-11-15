"""Parity tests: hand-written GPT (train_gpt2.py) vs Hugging Face GPT2LMHeadModel('gpt2').
Run with `pytest test_gpt2_parity.py` or `python test_gpt2_parity.py [--debug]`.
Only reports divergences; never modifies the model code."""
import sys, pathlib
from functools import lru_cache
import pytest, torch

torch.manual_seed(0)
TOL = 1e-4
RESULTS = []                 # (test name, max abs diff or None, status) for the summary table
LOGITS_FAILED = False        # lets the layer-by-layer test run only when logits mismatch
TEXT = ("The quick brown fox jumps over the lazy dog. In a hole in the ground there lived a hobbit. "
        "It was the best of times, it was the worst of times, it was the age of wisdom. ") * 40

@lru_cache(None)
def models():
    # train_gpt2.py runs sampling code (input(), mps) at import time, so exec only the
    # class definitions that come before the SAMPLING marker instead of importing it.
    src = pathlib.Path(__file__).with_name("train_gpt2.py").read_text()
    ns = {"__name__": "train_gpt2_defs"}
    exec(compile(src.split("################### SAMPLING")[0], "train_gpt2.py", "exec"), ns)
    from transformers import GPT2LMHeadModel
    mine = ns["GPT"].from_pretrained("gpt2").eval().float()
    hf = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager").eval().float()
    return mine, hf, ns["GPT"]

@lru_cache(None)
def tokenizers():
    import tiktoken
    from transformers import GPT2TokenizerFast
    return tiktoken.get_encoding("gpt2"), GPT2TokenizerFast.from_pretrained("gpt2")

def encode(text):
    tik, hft = tokenizers()
    a, b = tik.encode(text), hft.encode(text)
    assert a == b, "tiktoken and HF tokenizer disagree on token IDs"   # same IDs for both models
    return a

CASES = {
    "1 token":       lambda: torch.tensor([encode("Hello")[:1]]),
    "short sentence":lambda: torch.tensor([encode("Hello, I'm a language model,")]),
    "~200 tokens":   lambda: torch.tensor([encode(TEXT)[:200]]),
    "batch of 2":    lambda: torch.tensor([encode(TEXT)[:16], encode(TEXT)[16:32]]),
    "1024 tokens":   lambda: torch.tensor([encode(TEXT)[:1024]]),
}

def record(name, diff, ok, note=""):
    RESULTS.append((name, diff, ("SKIP" if ok is None else "PASS" if ok else "FAIL") + note))

@pytest.mark.parametrize("name", list(CASES))
def test_logits_match(name):
    # Same token tensor into both models; compare full (B, T, 50257) logits.
    global LOGITS_FAILED
    mine, hf, _ = models()
    ids = CASES[name]()
    with torch.no_grad():
        a, b = mine(ids), hf(ids).logits
    assert a.shape == b.shape == (*ids.shape, 50257), f"shape mismatch: mine {tuple(a.shape)} vs HF {tuple(b.shape)}"
    d = (a - b).abs()
    close = torch.allclose(a, b, atol=TOL, rtol=TOL)
    argmax_ok = bool((a.argmax(-1) == b.argmax(-1)).all())     # same top-1 token at every position
    record(f"logits: {name}", d.max().item(), close and argmax_ok,
           "" if argmax_ok else " (argmax differs)")
    if not (close and argmax_ok): LOGITS_FAILED = True
    assert close and argmax_ok, f"{name}: max abs diff {d.max():.3e}, mean {d.mean():.3e}, argmax agree={argmax_ok}"

def test_layerwise_localisation():
    # Runs only on a logits failure or --debug. Compares embeddings + each block's output with
    # HF hidden_states (HF's last entry is AFTER ln_f, so the last block is compared via ln_f).
    if not (LOGITS_FAILED or "--debug" in sys.argv):
        pytest.skip("logits matched; pass --debug to force layer-by-layer comparison")
    mine, hf, _ = models()
    ids = CASES["short sentence"]()
    outs = {}
    hooks = [m.register_forward_hook(lambda m, i, o, k=k: outs.__setitem__(k, o))
             for k, m in [("wte", mine.transformer.wte), ("wpe", mine.transformer.wpe), ("ln_f", mine.transformer.ln_f)]]
    hooks += [blk.register_forward_hook(lambda m, i, o, k=k: outs.__setitem__(f"h{k}", o))
              for k, blk in enumerate(mine.transformer.h)]
    try:
        with torch.no_grad():
            mine(ids); hs = hf(ids, output_hidden_states=True).hidden_states
    finally:
        for h in hooks: h.remove()
    n = len(mine.transformer.h)
    stages = [("embeddings (wte+wpe)", outs["wte"] + outs["wpe"], hs[0])]
    stages += [(f"block {i}", outs[f"h{i}"], hs[i + 1]) for i in range(n - 1)]
    stages += [(f"block {n-1} + ln_f", outs["ln_f"], hs[n])]
    first = None
    for name, x, y in stages:
        d = (x - y).abs().max().item()
        print(f"  {name:24s} max abs diff {d:.3e}")
        if d > TOL and first is None: first = name
    record("layerwise first bad stage", None, first is None, f" ({first})" if first else "")
    assert first is None, f"first divergence at: {first}"

def test_greedy_generation_matches():
    mine, hf, _ = models()
    if not hasattr(mine, "generate"):
        record("greedy generation", None, None, " (GPT has no generate())")
        pytest.skip("GPT has no generate() method")
    ids = torch.tensor([encode("Hello, I'm a language model,")])
    with torch.no_grad():
        a = mine.generate(ids, max_new_tokens=50, do_sample=False)   # assumes this signature
        b = hf.generate(ids, do_sample=False, max_new_tokens=50, pad_token_id=50256)
    tik, _ = tokenizers()
    print("mine:", tik.decode(a[0].tolist()), "\nHF:  ", tik.decode(b[0].tolist()))
    record("greedy generation", None, a.tolist() == b.tolist())
    assert a.tolist() == b.tolist()

def test_param_count_and_weight_tying():
    mine, hf, _ = models()
    n_mine = sum(p.numel() for p in mine.parameters())    # .parameters() dedups tied tensors
    n_hf = sum(p.numel() for p in hf.parameters())
    tied = mine.lm_head.weight.data_ptr() == mine.transformer.wte.weight.data_ptr()
    record("param count equal", abs(n_mine - n_hf), n_mine == n_hf, f" (mine {n_mine:,} vs HF {n_hf:,})")
    record("lm_head tied to wte", None, tied)
    assert n_mine == n_hf, f"param count: mine {n_mine:,} vs HF {n_hf:,}"
    assert tied, "lm_head.weight and wte.weight are not the same tensor"

if __name__ == "__main__":
    tests = [(f"logits: {n}", lambda n=n: test_logits_match(n)) for n in CASES]
    tests += [("layerwise", test_layerwise_localisation), ("greedy", test_greedy_generation_matches),
              ("params/tying", test_param_count_and_weight_tying)]
    for label, fn in tests:
        try: fn()
        except (AssertionError, pytest.skip.Exception) as e: print(f"[{label}] {type(e).__name__}: {e}")
        except Exception as e:
            record(label, None, False, f" (crashed: {type(e).__name__}: {e})")
    print(f"\n{'test':32s} | {'max abs diff':>12s} | result")
    for name, d, status in RESULTS:
        print(f"{name:32s} | {('%.3e' % d) if d is not None else '-':>12s} | {status}")
