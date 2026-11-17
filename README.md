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

There is no KV cache, so each step re-runs the whole sequence.

## Tests

The parity tests compare this model with Hugging Face's GPT-2 on CPU, in float32:

```bash
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

The test file runs only the part of `gpt2.py` above the `################### SAMPLING` comment, so keep that comment line in the file.

## Results

These come from one-off runs on an Apple Silicon Mac. They were not saved to a file, so treat them as rough.

**Parity tests** (as of the last run)

| test | max abs diff | result |
|---|---|---|
| logits: 1 token | 2.6e-4 | pass |
| logits: short sentence, ~200 tokens, batch of 2, 1024 tokens | 0 | pass |
| greedy generation | – | pass |
| param count equal | – | fail: 163,037,184 vs Hugging Face's 124,439,808 |
| `lm_head` tied to `wte` | – | fail |

The two failures have one cause: `GPT.__init__` does not tie `lm_head.weight` to `transformer.wte.weight`, so the output matrix is counted twice. The logits match anyway, because `from_pretrained` copies the same values into both.

**Generation checks** (prompt `"Hello, I'm a language model,"`)

- `greedy_search` matches Hugging Face's greedy output exactly over 50 tokens.
- `beam_search` with 4 beams matches Hugging Face's 4-beam output exactly, and with 1 beam it equals greedy.
- `top_k_sampling` with `k=1` and `top_p_sampling` with `p` near 0 both reduce to greedy.

**`--stats` on MPS and CPU**

| run | tokens/s | ms/token | time-to-first-token | memory |
|---|---|---|---|---|
| greedy, MPS, 60 tokens | 44.8 | 22.3 | 26 ms | weights 671 MB, peak about 684 MB |
| beam (4 beams), MPS, 40 tokens | 21.6 | 46.2 | 22 ms | peak about 710 MB |
| greedy, CPU, 20 tokens | 25.9 | 38.5 | 11.6 ms | not reported |

At these short lengths, later passes are about as fast as the first (about 22 ms), so per-step time is dominated by fixed GPU overhead, not sequence length. The cost of having no KV cache would only show up at several hundred tokens.

## Known gaps

- `lm_head` and `wte` are not weight-tied yet (see Results).
- No training loop, data loading or loss yet. `GPT.forward` returns logits only.
- No KV cache.
