"""Repeatable MPS benchmark: python bench.py <label> [kv]  ->  results/<label>.json
Runs generation cases a few times on a fixed prompt and saves the median stats.
Default case set compares the four strategies; `kv` compares greedy with and without the KV cache."""
import json, platform, statistics, subprocess, sys, time
from pathlib import Path
import tiktoken, torch
from gpt2 import GPT, GenStats

PROMPT = "Hello, I'm a language model,"
RUNS = 5   # per strategy; we keep the median because MPS timings are noisy
CASES = {  # name -> (generate kwargs, new tokens)
    "greedy": (dict(method="greedy"), 60),
    "beam4":  (dict(method="beam", num_beams=4), 40),
    "top_k":  (dict(method="top_k", k=50), 60),
    "top_p":  (dict(method="top_p", p=0.9), 60),
}
KV_CASES = {  # greedy with and without the KV cache, at growing lengths
    f"greedy_{'cached' if c else 'uncached'}_{n}": (dict(method="greedy", use_cache=c), n)
    for n in (60, 200, 800) for c in (False, True)
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
    return dict(tokens_per_s=n_new / total, ms_per_token=total / n_new * 1000, ttft_ms=t[0] * 1000,
                later_mean_ms=sum(t[1:]) / len(t[1:]) * 1000, weights_mb=st.weights_mem / 2**20,
                peak_mb=st.peak_mem / 2**20), out

def main(label, case_set="strategies"):
    cases, runs_per_case = (KV_CASES, 3) if case_set == "kv" else (CASES, RUNS)
    device = torch.device("mps")
    model = GPT.from_pretrained("gpt2").eval().to(device)
    enc = tiktoken.get_encoding("gpt2")
    ids = torch.tensor([enc.encode(PROMPT)], device=device)
    info = dict(label=label, date=time.strftime("%Y-%m-%d %H:%M"), torch=torch.__version__,
                machine=platform.machine(), prompt=PROMPT, runs_per_case=runs_per_case, case_set=case_set,
                commit=subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(),
                n_params=sum(p.numel() for p in model.parameters()),
                tied=model.lm_head.weight.data_ptr() == model.transformer.wte.weight.data_ptr())
    results = {}
    for name, (kwargs, n_new) in cases.items():
        model.generate(ids, max_new_tokens=n_new, **kwargs)   # untimed full pass: warms every kernel/shape this case uses
        runs = [run_once(model, device, ids, kwargs, n_new)[0] for _ in range(runs_per_case)]
        results[name] = {k: round(statistics.median(r[k] for r in runs), 2) for k in runs[0]}
        results[name]["new_tokens"] = n_new
        print(name, results[name])
    Path("results").mkdir(exist_ok=True)
    Path(f"results/{label}.json").write_text(json.dumps(dict(info=info, results=results), indent=2))
    print(f"saved results/{label}.json | params {info['n_params']:,} | tied {info['tied']}")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "bench", sys.argv[2] if len(sys.argv) > 2 else "strategies")
