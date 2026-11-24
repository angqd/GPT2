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
| `plot_results.py` | turns the saved `results/*.json` into charts in `results/plots/` (needs `pip install matplotlib`) |
| `results/` | saved benchmark results, one JSON file per `bench.py` run |

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
python bench.py my_label strategies_uncached   # the same four with the KV cache off
python bench.py batch batch        # cached greedy at batch size 1, 2, 4, 8, 16, 32 (100 new tokens)
python bench.py sample_kv sample_kv  # top-k and top-p with vs without the KV cache (200 tokens)
python plot_results.py             # charts from the saved results, written to results/plots/
```

For the batch case set, `tokens_per_s` counts every sequence in the batch (aggregate throughput) and `ms_per_token` is the time per decode step. Close other apps first (browser, Docker): background load changes the timings.

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
- Greedy, top-k and top-p use the cache by default (`use_cache=False` turns it off). Beam search still recomputes the whole sequence each step.
- Greedy, top-k and top-p also take a **batch** of prompts, `ids` of shape `(B, T)`, all the same length (no padding support). Each row is generated independently. With `eos_token_id`, a row that has finished keeps emitting EOS as padding, and generation stops when every row is done.

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

**Weight tying.** `GPT.__init__` sets `self.transformer.wte.weight = self.lm_head.weight`, so the 50257×768 matrix is stored once. The model has 124,439,808 parameters, the same as Hugging Face. The tie survives `.to('mps')`, and a backward pass gives one shared gradient tensor.

All results below were measured in one sitting on 2026-11-24 with nothing else running (load average about 2 at the start of each run, no Docker). Five back-to-back runs of the same benchmark differed by under 2% (greedy 91.4 to 92.9 tokens/s), so differences of a few percent are noise.

**Strategies, without and with the KV cache** (`results/baseline_uncached.json`, `results/strategies_cached.json`; median of 5 runs; 60 new tokens, 40 for beam)

| strategy | no cache | KV cache | speedup | peak memory (no cache → cache) |
|---|---|---|---|---|
| greedy | 49.6 tok/s (20.1 ms/token) | 92.0 tok/s (10.9 ms/token) | 1.85× | 549 → 529 MB |
| top-k (k=50) | 48.3 tok/s (20.7 ms/token) | 89.2 tok/s (11.2 ms/token) | 1.85× | 537 → 529 MB |
| top-p (p=0.9) | 47.8 tok/s (20.9 ms/token) | 88.4 tok/s (11.3 ms/token) | 1.85× | 537 → 529 MB |
| beam (4 beams) | 31.4 tok/s (31.9 ms/token) | no cache yet | – | 562 MB |

- Weights take 523 MB. Beam search is about 1.5× slower per token than greedy, not 4×, even though it processes 4× the data.
- Memory use is about 0.5 GB, against the 12 GB macOS recommends.
- CPU was not part of the saved results. In one earlier run, greedy on CPU was about 26 tokens/s, so MPS was roughly 1.7× faster at this size.

**Cached top-k and top-p** (`results/sample_kv.json`; 200 new tokens; median of 3)

| strategy | no cache | KV cache | speedup |
|---|---|---|---|
| top-k | 33.0 tok/s (30.3 ms/token) | 87.6 tok/s (11.4 ms/token) | 2.66× |
| top-p | 32.6 tok/s (30.7 ms/token) | 86.2 tok/s (11.6 ms/token) | 2.64× |

**Batched decoding** (`results/batch.json`; cached greedy, B copies of the prompt, 100 new tokens; median of 3)

| batch | ms per decode step | aggregate tokens/s | ms per sequence-token | peak memory |
|---|---|---|---|---|
| 1 | 10.8 | 90 | 11.1 | 531 MB |
| 2 | 13.5 | 145 | 6.9 | 539 MB |
| 4 | 14.8 | 265 | 3.8 | 556 MB |
| 8 | 17.8 | 442 | 2.3 | 589 MB |
| 16 | 23.3 | 675 | 1.5 | 653 MB |
| 32 | 30.3 | 1,029 | 1.0 | 782 MB |

- Throughput grows 11.4× for a 32× larger batch, so each sequence's tokens get 11× cheaper. The step time is not flat, though: it triples from batch 1 to 32, at roughly 0.4 to 0.75 ms per extra sequence. So a step is not purely memory-bound. The extra compute, and the extra cache reads and copies, start to show well below the batch size where a 3.6 TFLOP/s peak would predict the crossover.

**KV cache, greedy search** (`results/kv_greedy.json`; median of 3 runs; MPS)

| new tokens | uncached | cached | speedup | peak memory (uncached → cached) |
|---|---|---|---|---|
| 60 | 48.5 tok/s (20.6 ms/token) | 91.5 tok/s (10.9 ms/token) | 1.89× | 549 → 529 MB |
| 200 | 30.1 tok/s (33.2 ms/token) | 80.8 tok/s (12.4 ms/token) | 2.68× (from the re-check, see note) | 603 → 538 MB |
| 800 | 4.6 tok/s (218.0 ms/token) | 73.8 tok/s (13.6 ms/token) | 16.1× | 835 → 586 MB |

- Uncached cost per token climbs steeply with length (21 → 33 → 218 ms), because every step re-processes the whole sequence. Cached cost per token is nearly flat (11 → 12 → 14 ms).
- **Note on the 200-token row:** the first sweep (`kv_greedy.json`) measured 41.0 ms/token uncached (3.5× speedup), which looked too high. A re-check (`results/kv200_check.json`, 5 runs, every run saved) gave 33.2 ms/token with all runs between 33.0 and 33.4 ms, and the untimed warm-up pass took 7.0 s against 6.6 s for a timed run. So it was not a cold start. The first value was a transient slowdown, and the row above uses the re-check. Re-run it with `python bench.py kv200_check kv200`.
- The cache itself costs about 57 MB at 808 tokens (the peak-memory difference is larger because the uncached path also builds full-sequence logits), in line with the estimate of about 72 KB per token for GPT-2 124M in float32.
- Peak memory is sampled after each forward pass, so it is approximate.

## Known gaps

- No training loop, data loading or loss yet. `GPT.forward` returns logits only.
- Beam search doesn't use the KV cache yet. It needs the cache rows reordered whenever beams are reshuffled.
- Batched generation needs equal-length prompts. Mixed lengths would need padding and an attention mask.
- Beam search calls `.item()` for each candidate at every step, which forces a GPU sync. It is a possible speed-up, and `bench.py` can measure it.
