# GPT-2 from scratch

A hand-written GPT-2 (124M) in PyTorch, following Karpathy's
["Let's reproduce GPT-2 (124M)"](https://www.youtube.com/watch?v=l8pRSuU81PU) tutorial.
It can load the OpenAI/Hugging Face weights, generate text with four decoding strategies,
report performance stats on Apple Silicon (MPS), and be checked against Hugging Face's reference model.

## Files

| file | what it is |
|---|---|
| `gpt2.py` | the model (`CausalSelfAttention`, `MLP`, `Block`, `GPT`), `GPT.from_pretrained`, the generation methods, and the command-line generator |
| `test_gpt2_parity.py` | parity tests of this model against Hugging Face `GPT2LMHeadModel` |
| `test_kv_cache.py` | KV cache correctness tests (tiny random model, about a second) |
| `bench.py` | repeatable MPS benchmark; saves the median stats of several runs per strategy to `results/<label>.json` |
| `results/` | saved benchmark results (`before_tying.json`, `tied.json`, `kv_greedy.json`) |

## Setup

Python 3.11, with a virtual environment in `.venv/`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch numpy tiktoken transformers datasets tqdm requests pytest
```

The first run downloads the GPT-2 weights from Hugging Face (about 500 MB).

## Generate text

```bash
python gpt2.py -s greedy      # or: beam | top_k | top_p     (long form: --strategy)
```

The script loads GPT-2 124M onto MPS, then asks for the parameters of the chosen strategy.
Press Enter to keep the default shown in brackets.

| strategy | what it does | questions asked |
|---|---|---|
| `greedy` | always picks the most likely next token (deterministic) | prompt, `max_new_tokens` (50), stop at `<\|endoftext\|>` (y) |
| `beam` | keeps the `num_beams` best sequences by total log-probability and returns the best | the above, plus `num_beams` (4), `length_penalty` (1.0) |
| `top_k` | samples among the `k` most likely tokens | the above, plus `k` (50), `temperature` (1.0), `seed` (42) |
| `top_p` | nucleus sampling: samples from the smallest set of tokens whose cumulative probability reaches `p` | the above, plus `p` (0.9), `temperature` (1.0), `seed` (42) |

Options:

- `--device cpu` runs on CPU instead of MPS. The default is `mps`, and the script exits with an error if MPS is not available.
- `--stats` prints performance stats after generating (see below).

At startup the script checks that every parameter and buffer, including the causal-mask buffer, is on the chosen device.

### Performance stats (`--stats`)

```bash
python gpt2.py -s beam --stats
```

It does one untimed warm-up pass first (the first MPS run compiles Metal shaders), then reports:

- total time, tokens/sec and ms/token
- time-to-first-token (the prompt pass) and the mean and last time of the later passes
- MPS memory: model weights, approximate peak during generation, and the driver total against the recommended max

Timing uses `torch.mps.synchronize()`, because MPS is asynchronous. Peak memory is sampled after each forward pass, so it is approximate. Memory is only reported on MPS.

### Benchmark (`bench.py`)

```bash
python bench.py my_label           # four strategies; writes results/my_label.json
python bench.py my_label kv        # greedy with vs without the KV cache at 60/200/800 tokens
```

It runs greedy, beam (4 beams), top-k and top-p on a fixed prompt on MPS, 5 runs each, and saves the median tokens/sec, ms/token, time-to-first-token, mean later-pass time, and weights and peak memory, plus the parameter count, whether the weights are tied, the torch version and the git commit. Use it to compare before and after a change.

### Using the generation methods from Python

Importing `gpt2` no longer runs anything, so you can use the model directly. Each method takes one prompt of shape `(1, T)` and returns the full sequence of shape `(1, T + new_tokens)`:

```python
import torch, tiktoken
from gpt2 import GPT

model = GPT.from_pretrained('gpt2').eval()
enc = tiktoken.get_encoding('gpt2')
ids = torch.tensor([enc.encode("Hello, I'm a language model,")])

out = model.greedy_search(ids, max_new_tokens=40)
out = model.beam_search(ids, max_new_tokens=40, num_beams=4, length_penalty=1.0)
out = model.top_k_sampling(ids, max_new_tokens=40, k=50, temperature=1.0)
out = model.top_p_sampling(ids, max_new_tokens=40, p=0.9, temperature=1.0)
out = model.generate(ids, 40, method='beam', num_beams=4)    # dispatcher: greedy | beam | top_k | top_p
print(enc.decode(out[0].tolist()))
```

All methods accept `eos_token_id` (for example `50256`) to stop at end-of-text. Call `torch.manual_seed(...)` first if you want repeatable sampling.

### KV cache

`greedy_search` uses a KV cache by default. Each layer stores the keys and values of every token processed so far, so each decode step pushes only the newest token through the model instead of the whole sequence.

```python
model.greedy_search(ids, max_new_tokens=100)                    # cached (default)
model.greedy_search(ids, max_new_tokens=100, use_cache=False)   # old path, recomputes everything
logits = model(new_tokens, kv_cache=cache)                      # low level: cache = model.new_kv_cache()
```

How it works:
- **Prefill:** the first call feeds the whole prompt with an empty cache. **Decode:** every later call feeds only the one token just chosen. The same `forward` handles both.
- With a cache, `forward` takes only the new tokens. Position embeddings start at the number of cached tokens, and the causal mask uses rows `T_past : T_past + T` and columns `: T_past + T`.
- The cache stores absolute positions, so cached generation stops at `block_size` (1024 total tokens) instead of sliding a window.
- The cache is used for greedy search only. Top-k, top-p and beam search still recompute the whole sequence each step.

## Tests

The parity tests compare this model with Hugging Face's GPT-2 on CPU, in float32:

```bash
python -m pytest test_kv_cache.py -q  # KV cache tests only (about a second)
python test_gpt2_parity.py            # prints a summary table
python test_gpt2_parity.py --debug    # also runs the layer-by-layer comparison
pytest test_gpt2_parity.py -q
```

| test | what it checks |
|---|---|
| logits | logits match Hugging Face (atol and rtol 1e-4, same argmax at every position) for a 1-token input, a short sentence, about 200 tokens, a batch of 2, and exactly 1024 tokens |
| layer-by-layer | embeddings and each block's output against Hugging Face's hidden states; reports the first stage that diverges. It runs only if the logits test fails or you pass `--debug`. |
| greedy generation | `greedy_search` gives the same 50 tokens as Hugging Face `generate(do_sample=False)` |
| params and tying | the parameter count equals Hugging Face's, and `lm_head` is tied to `wte` |

`test_kv_cache.py` checks that logits computed through the cache (whole prompt, one token at a time, and mixed chunk sizes) equal a full forward pass, that cached greedy output equals uncached output token for token, and that cached generation stops at `block_size`.

The test file runs only the part of `gpt2.py` above the `################### SAMPLING` comment, so keep that comment line in the file.

## Results

Measured on an Apple Silicon Mac with MPS. The benchmark numbers are saved in `results/`. Timings vary by a few tokens/sec between runs, so differences of a few percent are noise.

**Parity tests** (all pass; the tie check passes since the weight-tying fix)

| test | max abs diff | result |
|---|---|---|
| logits: 1 token | 2.6e-4 | pass |
| logits: short sentence, ~200 tokens, batch of 2, 1024 tokens | 0 | pass |
| greedy generation | – | pass |
| param count equal | – | pass: 124,439,808 vs Hugging Face's 124,439,808 |
| `lm_head` tied to `wte` | – | pass |

**Generation checks** (prompt `"Hello, I'm a language model,"`)

- `greedy_search` matches Hugging Face's greedy output exactly over 50 tokens.
- `beam_search` with 4 beams matches Hugging Face's 4-beam output exactly, and with 1 beam it equals greedy.
- `top_k_sampling` with `k=1` and `top_p_sampling` with `p` near 0 both reduce to greedy.

**Weight tying fix.** `GPT.__init__` now sets `self.transformer.wte.weight = self.lm_head.weight`. Before the fix the model had 163,037,184 parameters, because the 50257×768 matrix was stored twice. Now it has 124,439,808, the same as Hugging Face. The tie survives `.to('mps')`, and a backward pass gives one shared gradient tensor.

**Benchmark baseline** (`results/before_tying.json` vs `results/tied.json`; median of 5 runs; tokens/sec, higher is better)

| strategy | new tokens | tokens/s before tying | tokens/s tied | weights before | weights tied | peak before | peak tied |
|---|---|---|---|---|---|---|---|
| greedy | 60 | 42.9 | 47.9 | 671 MB | 523 MB | 684 MB | 536 MB |
| beam (4 beams) | 40 | 29.9 | 30.2 | 671 MB | 523 MB | 710 MB | 562 MB |
| top-k (k=50) | 60 | 48.1 | 47.3 | 671 MB | 523 MB | 685 MB | 537 MB |
| top-p (p=0.9) | 60 | 45.5 | 45.4 | 671 MB | 523 MB | 685 MB | 537 MB |

- Tying saves 148 MB of weight memory (38.6M parameters at 4 bytes each). Speed is unchanged within noise.
- These runs are from before the KV cache existed. Time-to-first-token is about 13 ms (20 ms for beam), and later passes are about 21 ms (31 ms for beam).
- Beam search is about 1.5× slower per token than greedy, not 4×, even though it processes 4× the data.
- Memory use is about 0.5 GB, against the 12 GB macOS recommends.
- CPU was not part of the saved baseline. In one earlier run, greedy on CPU was about 26 tokens/s, so MPS was roughly 1.7× faster at this size.

**KV cache, greedy search** (`results/kv_greedy.json`; median of 3 runs; MPS)

| new tokens | uncached | cached | speedup | peak memory (uncached → cached) |
|---|---|---|---|---|
| 60 | 45.9 tok/s (21.8 ms/token) | 89.8 tok/s (11.1 ms/token) | 1.96× | 549 → 529 MB |
| 200 | 30.8 tok/s (32.5 ms/token) | 81.3 tok/s (12.3 ms/token) | 2.65× | 603 → 538 MB |
| 800 | 3.2 tok/s (314.9 ms/token) | 63.3 tok/s (15.8 ms/token) | 19.9× | 835 → 586 MB |

- Uncached cost per token climbs steeply with length (22 → 33 → 315 ms), because every step re-processes the whole sequence. Cached cost per token is nearly flat (11 → 12 → 16 ms).
- The cache itself costs about 63 MB at 808 tokens, in line with the estimate of about 72 KB per token for GPT-2 124M in float32.
- Peak memory is sampled after each forward pass, so it is approximate.

## Known gaps

- No training loop, data loading or loss yet. `GPT.forward` returns logits only.
- The KV cache is used for greedy search only. Top-k, top-p and beam search don't use it yet. Beam search will also need the cache rows reordered whenever beams are reshuffled.
- Beam search calls `.item()` for each candidate at every step, which forces a GPU sync. It is a possible speed-up, and `bench.py` can measure it.
