"""Stage 1 of the kernel spike: what does the rig charge for a launch, and for one full pass?

Bounds any fused kernel from below on the same instrument as the ladder, before the real
kernel exists. See results/DECISION.md and docs/benchmark_methodology.md.
"""

import argparse
import csv
import statistics

import torch

from . import harness
from .regime import ANCHOR_VOCAB, BENCH_BATCH_SIZES

try:
    import fused_sampling

    from .fused import sample_fused

    HAVE_FUSED = True
except Exception:
    HAVE_FUSED = False

FIELDS = [
    "op",
    "batch",
    "vocab",
    "dtype",
    "splits",
    "blocks",
    "median_us",
    "p05_us",
    "p95_us",
    "mean_us",
    "stdev_us",
    "reps",
    "iters",
    "warmup",
    "logits_bytes",
    "dram_floor_us",
    "floor_frac_of_latency",
    "git_commit",
]

SPLIT_GRID = (1, 8, 35, 70, 140, 280)
KERNEL_SPLIT_GRID = (1, 2, 4, 8, 16, 20, 32)
from .regime import BENCH_TOP_K_VALUES  # noqa: E402


def ops(x, batch):
    out = [("noop", 0, batch, lambda: fused_sampling.probe_noop(x)),
           ("scan_rowblock", 0, batch, lambda: fused_sampling.probe_scan_rowblock(x)),
           ("torch_argmax", 0, 0, lambda: torch.argmax(x, dim=-1))]
    for s in SPLIT_GRID:
        out.append((f"scan_split", s, s * batch, (lambda s=s: lambda: fused_sampling.probe_scan_split(x, s))()))
    return out


def run(op, splits, blocks, fn, batch, vocab, dtype, reps, bw, commit):
    iters = harness.calibrate_iters(fn)
    warmup = max(10, iters // 10)
    xs = harness.repeat_amortized(fn, reps, iters, warmup)
    med = statistics.median(xs)
    nbytes = batch * vocab * dtype.itemsize
    floor_us = nbytes / bw * 1e6
    return {
        "op": op,
        "batch": batch,
        "vocab": vocab,
        "dtype": str(dtype).replace("torch.", ""),
        "splits": splits,
        "blocks": blocks,
        "median_us": f"{med:.3f}",
        "p05_us": f"{harness.percentile(xs, 0.05):.3f}",
        "p95_us": f"{harness.percentile(xs, 0.95):.3f}",
        "mean_us": f"{statistics.mean(xs):.3f}",
        "stdev_us": f"{statistics.stdev(xs) if len(xs) > 1 else 0.0:.3f}",
        "reps": reps,
        "iters": iters,
        "warmup": warmup,
        "logits_bytes": nbytes,
        "dram_floor_us": f"{floor_us:.3f}",
        "floor_frac_of_latency": f"{floor_us / med:.6f}",
        "git_commit": commit,
    }


def splits_sweep(args, bw, commit):
    """How many row-slices the fused kernel should cut the vocabulary into.

    It is latency-bound on its phase chain rather than throughput-bound, so this does not
    simply track SM count -- the auto rule in csrc/fused_sampling.cu comes from here.
    """
    rows = []
    for rnd in range(args.rounds):
        for batch in args.batches:
            x = (torch.randn(batch, args.vocab, device="cuda") * 4).to(torch.bfloat16)
            for top_k in BENCH_TOP_K_VALUES:
                for sp in KERNEL_SPLIT_GRID:
                    if sp * top_k > 1024:
                        continue
                    fn = (lambda sp=sp, k=top_k: lambda: sample_fused(x, k, 0.9, splits=sp))()
                    r = run(f"fused_k{top_k}", sp, sp * batch, fn, batch, args.vocab,
                            torch.bfloat16, args.reps, bw, commit)
                    r["round"] = rnd
                    rows.append(r)
                    print(f"  r{rnd} B={batch:<3} k={top_k:<4} splits={sp:<3} {r['median_us']:>10} us")
            del x
            torch.cuda.empty_cache()
    return rows


def write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS + ["round"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/raw/kernel_floor.csv")
    ap.add_argument("--splits-out", default="results/raw/kernel_splits.csv")
    ap.add_argument("--vocab", type=int, default=ANCHOR_VOCAB)
    ap.add_argument("--batches", type=int, nargs="+", default=list(BENCH_BATCH_SIZES))
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()

    if not HAVE_FUSED:
        print("! fused_sampling extension not importable; run python setup.py build_ext --inplace")
        return 2
    busy, apps = harness.gpu_is_busy()
    if busy:
        print(f"! GPU is busy, refusing to measure: {apps}")
        return 2

    locked = harness.try_lock_clocks()
    env = harness.environment(locked)
    bw = harness.measured_dram_bandwidth()
    print(f"measured HBM bandwidth: {bw / 1e9:.1f} GB/s")

    rows = []
    for rnd in range(args.rounds):
        for batch in args.batches:
            x = (torch.randn(batch, args.vocab, device="cuda") * 4).to(torch.bfloat16)
            spec = ops(x, batch)
            spec = spec[rnd % len(spec):] + spec[: rnd % len(spec)]
            for op, splits, blocks, fn in spec:
                r = run(op, splits, blocks, fn, batch, args.vocab, torch.bfloat16, args.reps, bw,
                        env["git_commit"])
                r["round"] = rnd
                rows.append(r)
                print(f"  r{rnd} B={batch:<3} {op:<14} splits={splits:<4} {r['median_us']:>10} us")
            del x
            torch.cuda.empty_cache()

    write(args.out, rows)
    write(args.splits_out, splits_sweep(args, bw, env["git_commit"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
