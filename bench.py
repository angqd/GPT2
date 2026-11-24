"""Repeatable MPS benchmark: python bench.py <label> [case_set]  ->  results/<label>.json
Runs generation cases a few times on a fixed prompt and saves the median stats.
case_set: strategies (default, compares the four strategies), strategies_uncached (the same four with the KV cache off:
the original baseline), kv (greedy with vs without the KV cache), batch (cached greedy at growing batch sizes),
sample_kv (top-k / top-p with vs without the KV cache)."""
import json, os, platform, statistics, subprocess, sys, time
from pathlib import Path
import tiktoken, torch
from gpt2 import GPT, GenStats

PROMPT = "Hello, I'm a language model,"
RUNS = 5   # per strategy; we keep the median because MPS timings are noisy
CASES = {  # name -> (generate kwargs, new tokens [, batch size])
    "greedy": (dict(method="greedy"), 60),
    "beam4":  (dict(method="beam", num_beams=4), 40),
    "top_k":  (dict(method="top_k", k=50), 60),
    "top_p":  (dict(method="top_p", p=0.9), 60),
}
CASES_UNCACHED = {  # the original pre-KV-cache baseline: same four strategies, cache off (beam never had one)
    name: ((dict(kw, use_cache=False) if kw["method"] != "beam" else kw), n) for name, (kw, n) in CASES.items()
}
KV_CASES = {  # greedy with and without the KV cache, at growing lengths
    f"greedy_{'cached' if c else 'uncached'}_{n}": (dict(method="greedy", use_cache=c), n)
    for n in (60, 200, 800) for c in (False, True)
}
BATCH_CASES = {  # cached greedy decoding of B copies of the prompt at once: does a step get slower with B?
    f"greedy_b{b}": (dict(method="greedy", use_cache=True), 100, b) for b in (1, 2, 4, 8, 16, 32)
}
SAMPLE_KV_CASES = {  # sampling with and without the KV cache
    f"{m}_{'cached' if c else 'uncached'}_200": (dict(method=m, use_cache=c, **extra), 200)
    for m, extra in (("top_k", dict(k=50)), ("top_p", dict(p=0.9))) for c in (False, True)
}
KV200_CASES = {k: v for k, v in KV_CASES.items() if k.endswith("_200")}   # just the 200-token pair, for re-checks
CASE_SETS = {  # name -> (cases, runs per case)
    "strategies": (CASES, RUNS), "strategies_uncached": (CASES_UNCACHED, RUNS), "kv": (KV_CASES, 3),
    "kv200": (KV200_CASES, 5), "batch": (BATCH_CASES, 3), "sample_kv": (SAMPLE_KV_CASES, 3),
}

def run_once(model, device, ids, kwargs, n_new):
    torch.manual_seed(42); torch.mps.manual_seed(42)
    st = GenStats(model, device)
    st.warmup(ids)                                   # untimed: compiles Metal shaders
    st.sync(); t0 = time.perf_counter()
    out = model.generate(ids, max_new_tokens=n_new, **kwargs)
    st.sync(); total = time.perf_counter() - t0
    for h in st.hooks: h.remove()
    t = st.step_times
    B = ids.size(0)
    # tokens_per_s counts every sequence in the batch (aggregate throughput); ms_per_token is the time per decode
    # step, i.e. per generated position, which every sequence in the batch shares
    return dict(tokens_per_s=B * n_new / total, ms_per_token=total / n_new * 1000, ttft_ms=t[0] * 1000,
                later_mean_ms=sum(t[1:]) / len(t[1:]) * 1000, weights_mb=st.weights_mem / 2**20,
                peak_mb=st.peak_mem / 2**20), out

def main(label, case_set="strategies"):
    cases, runs_per_case = CASE_SETS[case_set]
    load_at_start = os.getloadavg()          # 1/5/15 min load averages: a record of how busy the machine was
    device = torch.device("mps")
    model = GPT.from_pretrained("gpt2").eval().to(device)
    enc = tiktoken.get_encoding("gpt2")
    ids1 = torch.tensor([enc.encode(PROMPT)], device=device)
    info = dict(label=label, date=time.strftime("%Y-%m-%d %H:%M"), torch=torch.__version__,
                machine=platform.machine(), loadavg_at_start=[round(x, 2) for x in load_at_start], prompt=PROMPT, runs_per_case=runs_per_case, case_set=case_set,
                commit=subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(),
                n_params=sum(p.numel() for p in model.parameters()),
                tied=model.lm_head.weight.data_ptr() == model.transformer.wte.weight.data_ptr())
    results = {}
    for name, (kwargs, n_new, *rest) in cases.items():
        B = rest[0] if rest else 1
        ids = ids1.repeat(B, 1)                                    # B identical prompts (same length, so no padding)
        tw = time.perf_counter()
        model.generate(ids, max_new_tokens=n_new, **kwargs)        # untimed full pass: warms every kernel/shape this case uses
        torch.mps.synchronize(); warm_s = time.perf_counter() - tw  # kept: a much slower first pass means a cold start
        runs = [run_once(model, device, ids, kwargs, n_new)[0] for _ in range(runs_per_case)]
        results[name] = {k: round(statistics.median(r[k] for r in runs), 2) for k in runs[0]}
        results[name]["new_tokens"] = n_new
        results[name]["batch"] = B
        results[name]["warm_pass_s"] = round(warm_s, 2)
        results[name]["ms_per_token_runs"] = [round(r["ms_per_token"], 2) for r in runs]   # every run, in order
        results[name]["ms_per_token_min_max"] = [round(min(r["ms_per_token"] for r in runs), 2),
                                                 round(max(r["ms_per_token"] for r in runs), 2)]
        print(name, results[name])
    Path("results").mkdir(exist_ok=True)
    Path(f"results/{label}.json").write_text(json.dumps(dict(info=info, results=results), indent=2))
    print(f"saved results/{label}.json | params {info['n_params']:,} | tied {info['tied']}")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "bench", sys.argv[2] if len(sys.argv) > 2 else "strategies")
