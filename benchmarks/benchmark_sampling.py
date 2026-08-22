import argparse
import csv
import statistics
import sys
from pathlib import Path

import torch

from . import harness
from .implementations import (
    EAGER_FNS,
    PROBS_INPUT_IMPLS,
    build_compile,
    build_eager,
    build_graph,
    sample_tight,
)

FIELDS = [
    "impl", "batch", "vocab", "top_k", "top_p", "dtype", "residency", "cold_method",
    "median_us", "p05_us", "p95_us", "mean_us", "stdev_us", "reps", "iters", "warmup",
    "logits_bytes", "dram_floor_us", "floor_frac_of_latency",
    "validation_in_timed_region", "round", "git_commit",
]

L2_MULTIPLIER = 4


def make_buffers(batch, vocab, dtype, device, hot, l2_bytes, seed=0, as_probs=False):
    torch.manual_seed(seed)
    itemsize = 4 if as_probs else dtype.itemsize
    per = batch * vocab * itemsize
    n = 1 if hot else max(2, int(L2_MULTIPLIER * l2_bytes / per) + 1)
    out = []
    for _ in range(n):
        x = torch.randn(batch, vocab, device=device) * 4
        out.append(torch.softmax(x, dim=-1) if as_probs else x.to(dtype))
    return out


def build(impl_name, buffers, top_k, top_p, hot):
    if impl_name in EAGER_FNS:
        return build_eager(impl_name, EAGER_FNS[impl_name], buffers, top_k, top_p, hot)
    if impl_name == "compile":
        return build_compile(sample_tight, buffers, top_k, top_p, hot)
    if impl_name == "graph_eager":
        return build_graph(sample_tight, buffers, top_k, top_p, hot)
    if impl_name == "graph_compile":
        return build_graph(
            sample_tight, buffers, top_k, top_p, hot, "graph_compile", compile_first=True
        )
    raise ValueError(impl_name)


def run_cell(impl_name, batch, vocab, top_k, top_p, dtype, hot, l2_bytes, bw, reps, rnd, commit):
    device = "cuda"
    as_probs = impl_name in PROBS_INPUT_IMPLS
    buffers = make_buffers(batch, vocab, dtype, device, hot, l2_bytes, seed=rnd, as_probs=as_probs)
    try:
        impl = build(impl_name, buffers, top_k, top_p, hot)
    except Exception as e:
        print(f"    ! {impl_name} build failed: {type(e).__name__}: {str(e)[:120]}")
        return None

    iters = harness.calibrate_iters(impl.call)
    warmup = max(10, iters // 10)
    samples = harness.repeat_amortized(impl.call, reps, iters, warmup)

    nbytes = batch * vocab * (4 if as_probs else dtype.itemsize)
    floor_us = nbytes / bw * 1e6
    med = statistics.median(samples)
    del buffers, impl
    torch.cuda.empty_cache()

    return {
        "impl": impl_name,
        "batch": batch,
        "vocab": vocab,
        "top_k": top_k,
        "top_p": top_p,
        "dtype": str(dtype).replace("torch.", ""),
        "residency": "hot" if hot else "cold",
        "cold_method": "n/a" if hot else ("rotate_into_static" if "graph" in impl_name else "rotate"),
        "median_us": f"{med:.3f}",
        "p05_us": f"{harness.percentile(samples, 0.05):.3f}",
        "p95_us": f"{harness.percentile(samples, 0.95):.3f}",
        "mean_us": f"{statistics.mean(samples):.3f}",
        "stdev_us": f"{statistics.stdev(samples) if len(samples) > 1 else 0.0:.3f}",
        "reps": reps,
        "iters": iters,
        "warmup": warmup,
        "logits_bytes": nbytes,
        "dram_floor_us": f"{floor_us:.3f}",
        "floor_frac_of_latency": f"{floor_us / med:.6f}",
        "validation_in_timed_region": "false",
        "round": rnd,
        "git_commit": commit,
    }


def cells(args):
    """Main sweep at the anchor point, plus a parameter-sensitivity slice."""
    out = []
    for dtype in (torch.float16, torch.bfloat16):
        for hot in (True, False):
            for batch in args.batches:
                out.append((batch, args.anchor_vocab, 50, 0.90, dtype, hot))
    for vocab in args.vocabs:
        for top_k in (20, 50, 100):
            for top_p in (0.90, 0.95):
                for batch in (1, 32):
                    out.append((batch, vocab, top_k, top_p, torch.bfloat16, True))
    seen, uniq = set(), []
    for c in out:
        key = tuple(str(x) for x in c)
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    return uniq


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/raw/sampling_ladder.csv")
    ap.add_argument("--env-out", default="results/raw/environment.json")
    ap.add_argument("--impls", nargs="+", default=None)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    ap.add_argument("--vocabs", type=int, nargs="+", default=[128256, 151936])
    ap.add_argument("--anchor-vocab", type=int, default=151936)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args(argv)

    impls = args.impls or (
        list(EAGER_FNS) + ["compile", "graph_eager", "graph_compile"]
    )

    busy, apps = harness.gpu_is_busy()
    if busy:
        print(f"REFUSING: GPU is busy: {apps}")
        return 2

    locked = harness.try_lock_clocks()
    Path(args.env_out).parent.mkdir(parents=True, exist_ok=True)
    env = harness.dump_env(args.env_out, locked)
    bw = harness.measured_dram_bandwidth()
    env["measured_hbm_bandwidth_bytes_per_s"] = bw
    import json

    json.dump(env, open(args.env_out, "w"), indent=2, sort_keys=True)
    print(f"env -> {args.env_out}   clocks_locked={locked}   HBM={bw/1e9:.1f} GB/s")

    grid = cells(args)
    l2 = env["gpu_l2_bytes"]
    rows, total = [], len(grid) * len(impls) * args.rounds
    done = 0
    for rnd in range(args.rounds):
        # rotate impl order every round so no rung is always measured first
        order = impls[rnd % len(impls):] + impls[: rnd % len(impls)]
        for cell in grid:
            batch, vocab, top_k, top_p, dtype, hot = cell
            for impl_name in order:
                r = run_cell(
                    impl_name, batch, vocab, top_k, top_p, dtype, hot, l2, bw,
                    args.reps, rnd, env["git_commit"],
                )
                done += 1
                if r:
                    rows.append(r)
                    print(
                        f"[{done}/{total}] r{rnd} {impl_name:20s} B={batch:3d} V={vocab} "
                        f"k={top_k:3d} p={top_p} {r['dtype']:8s} {r['residency']:4s} "
                        f"-> {float(r['median_us']):9.2f} us"
                    )
                sys.stdout.flush()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
