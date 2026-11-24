# Making sense of our decoding results (before and after the KV cache)

Teaching notes, 2026-11-24. Based on the ideas in "Making Deep Learning Go Brrr From First Principles", explained from
memory (not re-read). Benchmark numbers come from the files in `results/` (see section 2 for which file feeds which
table). Hardware numbers (100 GB/s, 3.6 TFLOP/s) are rough assumptions for a base M2, not measurements.

## 1. The three ways a GPU can be slow

Picture a factory (the compute units), a warehouse (memory), a conveyor belt between them (bandwidth), and a manager
who sends the orders (the Python code launching GPU kernels).

| regime | what's limiting | how to spot it |
|---|---|---|
| Compute-bound | The factory is at full speed | Time grows in step with the FLOPs |
| Memory-bandwidth-bound | The factory waits for the belt | Time tracks bytes moved, not FLOPs |
| Overhead-bound | The factory waits for the manager | Making the work bigger costs almost nothing |

The test that tells them apart is **arithmetic intensity**: the FLOPs you do per byte you move. A chip has a break-even
value, the "ridge point". Below it you're waiting on memory; above it you're waiting on compute. For the M2 that's about
3.6 TFLOP/s divided by 100 GB/s, roughly **36 FLOPs per byte** (assumed numbers).

**A rule that works on any result:** work out which resource a step is waiting on. Only speeding up that resource helps.

## 2. Before and after: the actual results

All measured on one Apple M2 (16 GB) with MPS, float32, nothing else running, prompt `"Hello, I'm a language model,"`.

### 2a. The KV cache (greedy decoding; `results/kv_greedy.json`, 200-token row from `results/kv200_check.json`)

| new tokens | BEFORE: no cache | AFTER: KV cache | speedup |
|---|---|---|---|
| 60 | 20.6 ms/token | 10.9 ms/token | **1.9x** |
| 200 | 33.2 ms/token | 12.4 ms/token | **2.7x** |
| 800 | 218.0 ms/token | 13.6 ms/token | **16.1x** |

Read it like this: without the cache, every token costs more than the last one, so the cost per token climbs
(21 -> 33 -> 218 ms). With the cache it barely moves (11 -> 12 -> 14 ms). The gap between the two lines is the speedup,
and it keeps widening as the text gets longer.

### 2b. The same change for every strategy (60 new tokens; `baseline_uncached.json` vs `strategies_cached.json`)

| strategy | BEFORE | AFTER | speedup |
|---|---|---|---|
| greedy | 49.6 tok/s | 92.0 tok/s | 1.85x |
| top-k | 48.3 tok/s | 89.2 tok/s | 1.85x |
| top-p | 47.8 tok/s | 88.4 tok/s | 1.85x |
| beam (4 beams) | 31.4 tok/s | not cached yet | - |

At 200 tokens (`sample_kv.json`) top-k and top-p speed up 2.66x and 2.64x, the same as greedy. The cache is a property
of the model, not of the decoding rule.

### 2c. Batching (cached greedy; `results/batch.json`)

| batch | ms per step | tokens/s (all rows) | ms per sequence-token |
|---|---|---|---|
| 1 | 10.8 | 90 | 11.1 |
| 2 | 13.5 | 145 | 6.9 |
| 4 | 14.8 | 265 | 3.8 |
| 8 | 17.8 | 442 | 2.3 |
| 16 | 23.3 | 675 | 1.5 |
| 32 | 30.3 | 1,029 | 1.0 |

Read it like this: each row of the batch is a separate sequence. Going from 1 to 32 sequences makes a step 2.8x slower
but produces 32 tokens instead of 1, so throughput rises 11.4x and each sequence-token gets 11x cheaper.

### 2d. Memory

The weights take 523 MB. The cache costs about 72 KB per token (2 x 12 layers x 768 x 4 bytes): 57 MB at 808 tokens.
Peak memory at 800 tokens: 835 MB without the cache, 586 MB with it. The uncached run is bigger because it computes
logits for every position (about 155 MB at 808 tokens) and builds full attention score matrices.

## 3. How to understand it

**Why does the cache help so much more at 800 tokens than at 60?**
It removes compute, and the uncached run only had a lot of compute to remove at long lengths.
- Cached decode: each step reads all 498 MB of weights and does about 2 FLOPs per parameter, which is 0.5 FLOPs per
  byte, far below the ridge (~36). So it is memory-bound plus some launch overhead. The floor is about 5 ms for the
  weight reads (498 MB / 100 GB/s) plus overhead; we measure 11 ms.
- Uncached decode at 60 tokens: the sequence is short, so the FLOPs per byte sit near the ridge. Reprocessing 68 tokens
  costs only about 2x one token, because the extra compute fits into time the chip spent waiting on memory anyway.
- Uncached decode at 800 tokens: far above the ridge, so it is compute-bound and time tracks FLOPs. The cache removes
  almost all of that compute.

So the cache converts a compute problem into a bandwidth problem. It does not make the weight reads go away, which is
why cached time per token never drops below roughly 11 ms.

**Why is the speedup (16x) so much smaller than the FLOP formula says (about 400x)?**
The formula counts only arithmetic. Real time includes reading weights and launching hundreds of small kernels, and
neither shrinks with the cache. Those costs set the 11 ms floor.

**What did the cache cost us?**
It adds traffic. At 800 tokens it holds 57 MB, read at every step, and `torch.cat` copies the whole cache each step.
My back-of-envelope estimate predicted about +34% per step from this; we measured +24% (10.9 -> 13.6 ms). Same ballpark.
A preallocated cache buffer would avoid the copy.

**Batching: my prediction was wrong, and that's the interesting part.**
I expected the step time to stay flat (about 11 ms) as the batch grew, because the weights are read once however many
rows you have. It rose to 30 ms at batch 32, about 0.4 to 0.75 ms per extra sequence. So a step is not purely
memory-bound. The likely reasons (a hypothesis, not separately measured):
- The chip only reaches a fraction of its peak FLOP/s on real code, so the effective ridge point is much lower than the
  ~36 FLOPs per byte computed from the peak numbers. Compute starts to matter at small batches.
- Each extra row adds cache reads and copies, and a bigger logits tensor.
Throughput still improves a lot (11x for 32x the batch). The lesson: the ridge-point model tells you which side you
are on, but the exact crossover depends on how much of the peak your code really reaches.

**What the benchmark can and cannot tell you**
- Runs agree within about 2% when the machine is quiet. Differences of a few percent are noise.
- One case in the first sweep (200 tokens, uncached) came out at 41 ms instead of 33 ms. A re-check with every run
  saved gave 33.0 to 33.4 ms on all five runs, and the warm-up pass was not slower than a timed run, so it was a
  one-off slowdown, not a cold start.
- Timings use forward hooks that force a GPU sync after each pass, which slightly slows the cached path.
- Peak memory is sampled after each forward pass, so it can miss brief spikes.

## 4. MPS vs an NVIDIA GPU

The hierarchy is the same on both. What differs is where the memory sits and how big each level is.

| | Apple M2 (MPS) | Typical NVIDIA datacenter GPU (A100-class) |
|---|---|---|
| Main memory | One pool shared by CPU and GPU (unified memory) | Separate "VRAM" on the card (HBM) |
| Bandwidth | about 100 GB/s | about 1.5-2 TB/s, 15-20x more |
| Size | 16 GB total; macOS recommends the GPU stay under about 12 GB | 40-80 GB, all dedicated |
| CPU-to-GPU copy | None, it's the same memory | Over PCIe (tens of GB/s), a real cost |
| On-chip fast memory ("SRAM") | Small per-core scratch plus a system cache (M2 sizes not verified) | Registers and shared memory, about 20 MB total, plus a larger L2 cache |
| Matrix-multiply hardware | None dedicated on the M2 GPU (my understanding) | Tensor cores |

- "VRAM" on a Mac is just your RAM.
- SRAM is the small, very fast on-chip memory. Operator fusion (the article's big optimization) keeps intermediate
  results there instead of writing them to main memory and reading them back. FlashAttention is the famous example.
- The same 498 MB weight read would take about 0.3 ms on an A100. Overhead dominates even more there, which is why CUDA
  tooling leans on CUDA graphs and `torch.compile`. MPS is generally less tuned for this (not measured here).

## 5. Check your understanding (with answers so far)

1. *Cached decode at batch 8: what step time, and which regime?* Prediction: about 11 ms, memory-bound. Measured: 17.8 ms.
   The step is mostly memory-bound at small batch but not purely, as section 3 explains.
2. *The uncached run at T=800 is compute-bound. Does doubling memory bandwidth change it? What about the cached run?*
   Uncached: no. Cached: yes, but not by half. The weight reads (about 5 ms of 11) would halve, and the overhead part
   would stay, so roughly 11 -> 8.5 ms.
3. *Why doesn't doubling the compute speed help the cached step?* It is already waiting on memory and overhead, not on
   arithmetic.
4. *New:* at what batch size would you expect the cached step to double in time (about 22 ms)? Our data says around
   batch 16. Why might it be earlier than the theoretical ridge at batch 72?
