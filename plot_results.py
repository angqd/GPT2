"""Turn the saved benchmark JSON files in results/ into PNG charts in results/plots/.
    python plot_results.py            (needs matplotlib: pip install matplotlib)
Each chart is skipped, with a message, if its results file does not exist yet:
    results/kv_greedy.json    from  python bench.py kv_greedy kv
    results/batch.json        from  python bench.py batch batch
    results/sample_kv.json    from  python bench.py sample_kv sample_kv
"""
import json, sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")                      # write files, no window
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("matplotlib is not installed: pip install matplotlib")

RESULTS, OUT = Path("results"), Path("results/plots")

# Rough M2 numbers used for the "theory" lines. Assumptions, not measurements: adjust for your chip.
BANDWIDTH_GBPS = 100.0       # unified memory bandwidth (base M2)
PEAK_TFLOPS = 3.6            # fp32 GPU peak (base M2, 10 core)
WEIGHT_BYTES = 124_439_808 * 4
FLOPS_PER_TOKEN = 2 * 124_439_808          # about 2 FLOPs per parameter per token (ignores attention)
RIDGE = PEAK_TFLOPS * 1e12 / (BANDWIDTH_GBPS * 1e9)   # FLOPs per byte where compute and memory balance

def load(name):
    p = RESULTS / f"{name}.json"
    if not p.exists():
        print(f"skip: {p} not found")
        return None
    return json.loads(p.read_text())["results"]

def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)
    print(f"wrote {OUT / name}")

def plot_kv():
    r = load("kv_greedy")
    if not r: return
    recheck = load("kv200_check")      # the 200-token pair was re-measured with 5 runs after a slow outlier in the sweep
    if recheck:
        r = {**r, **recheck}
    lengths = sorted({v["new_tokens"] for v in r.values()})
    unc = [r[f"greedy_uncached_{n}"] for n in lengths]
    cac = [r[f"greedy_cached_{n}"] for n in lengths]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))
    a.plot(lengths, [x["ms_per_token"] for x in unc], "o-", label="no cache")
    a.plot(lengths, [x["ms_per_token"] for x in cac], "o-", label="KV cache")
    a.axhline(WEIGHT_BYTES / (BANDWIDTH_GBPS * 1e9) * 1000, ls="--", c="gray", label="memory floor (weights / bandwidth)")
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xticks(lengths, labels=[str(n) for n in lengths]); a.minorticks_off()
    a.set_xlabel("new tokens generated"); a.set_ylabel("ms per token (log)"); a.set_title("Cost per token"); a.legend()
    sp = [u["ms_per_token"] / c["ms_per_token"] for u, c in zip(unc, cac)]
    bars = b.bar([str(n) for n in lengths], sp)
    b.bar_label(bars, labels=[f"{s:.1f}x" for s in sp])
    b.set_xlabel("new tokens generated"); b.set_ylabel("speedup (cached vs uncached)"); b.set_title("KV cache speedup")
    save(fig, "kv_speedup.png")

def plot_batch():
    r = load("batch")
    if not r: return
    rows = sorted(r.values(), key=lambda v: v["batch"])
    B = [v["batch"] for v in rows]
    step = [v["later_mean_ms"] for v in rows]
    thr = [v["tokens_per_s"] for v in rows]
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(15, 4))
    a.plot(B, step, "o-", label="measured decode step")
    a.axhline(step[0], ls="--", c="gray", label="batch-1 step time (flat = memory bound)")
    a.set_xscale("log", base=2); a.set_xlabel("batch size"); a.set_ylabel("ms per decode step"); a.set_title("Step time vs batch"); a.legend()
    b.plot(B, thr, "o-", label="measured")
    b.plot(B, [thr[0] * x / B[0] for x in B], "--", c="gray", label="perfect linear scaling")
    b.set_xscale("log", base=2); b.set_yscale("log"); b.set_xlabel("batch size"); b.set_ylabel("aggregate tokens/s (log)")
    b.set_title("Throughput vs batch"); b.legend()
    # roofline view: arithmetic intensity of a decode step grows with the batch (the weights are read once for all rows)
    intensity = [FLOPS_PER_TOKEN * x / WEIGHT_BYTES for x in B]
    c.plot(B, intensity, "o-", label="decode step (theory, no attention)")
    c.axhline(RIDGE, ls="--", c="red", label=f"M2 ridge point ~{RIDGE:.0f} FLOP/byte (assumed)")
    c.set_xscale("log", base=2); c.set_yscale("log"); c.set_xlabel("batch size"); c.set_ylabel("FLOPs per byte moved")
    c.set_title(f"Theory: ridge reached at B~{RIDGE / (FLOPS_PER_TOKEN / WEIGHT_BYTES):.0f} (real chips cross earlier)")
    c.legend(loc="lower right")
    save(fig, "batch_scaling.png")

def plot_sample_kv():
    r = load("sample_kv")
    if not r: return
    methods = ["top_k", "top_p"]
    fig, ax = plt.subplots(figsize=(6, 4))
    w = 0.35
    for i, (mode, label) in enumerate((("uncached", "no cache"), ("cached", "KV cache"))):
        vals = [r[f"{m}_{mode}_200"]["tokens_per_s"] for m in methods]
        bars = ax.bar([x + (i - 0.5) * w for x in range(len(methods))], vals, w, label=label)
        ax.bar_label(bars, labels=[f"{v:.0f}" for v in vals])
    ax.set_xticks(range(len(methods))); ax.set_xticklabels(methods)
    ax.set_ylabel("tokens/s (200 new tokens)"); ax.set_title("Sampling with and without the KV cache"); ax.legend()
    save(fig, "sampling_kv.png")

if __name__ == "__main__":
    plot_kv(); plot_batch(); plot_sample_kv()
