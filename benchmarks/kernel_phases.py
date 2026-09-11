"""B0 of the post-spike roadmap: where does the fused kernel's 20.6 us actually go?

results/SPIKE.md attributes the 4.5x gap to its own floor to "three row passes, ~50 block syncs,
and a 55-stage bitonic merge" -- an inference, not a measurement, because ncu is blocked on this
box (RmProfilingAdminOnly: 1). This runs phases 1..n of the real kernel and stops, so each phase
costs a difference of two timings on the same instrument as the ladder.

Phases, as instantiated by csrc/fused_sampling.cu:
  1  high-byte histogram          -- traversal
  2  + fold, suffix scan, bucket search
  3  + low-byte histogram in that bucket -- traversal
  4  + fold, suffix scan, exact threshold
  5  + gather, tie resolve, store  -- traversal; the whole partial kernel
  6  + merge bitonic sort          -- second launch
  7  + softmax, cumsum, top-p cut, draw; identical to sample_fused
"""

import argparse
import csv
import statistics

import torch

from . import harness
from .regime import ANCHOR_VOCAB, BENCH_BATCH_SIZES, BENCH_TOP_K_VALUES

try:
    import fused_sampling

    from .fused import sample_fused

    HAVE_FUSED = True
except Exception:
    HAVE_FUSED = False

N_PHASES = 7
PHASE_LABEL = {
    1: "high-byte histogram",
    2: "+ bucket search",
    3: "+ low-byte histogram",
    4: "+ exact threshold",
    5: "+ gather, ties, store",
    6: "+ merge sort",
    7: "+ softmax, cut, draw",
}

FIELDS = [
    "op", "phase", "batch", "vocab", "top_k", "dtype", "splits", "blocks",
    "median_us", "p05_us", "p95_us", "mean_us", "stdev_us", "reps", "iters", "warmup",
    "round", "git_commit",
]


def auto_splits(batch, top_k):
    """Mirrors make_plan in csrc/fused_sampling.cu; recorded so a row is self-describing."""
    return max(1, min(min(8, max(4, 128 // batch)), 1024 // top_k))


def run(op, phase, fn, batch, vocab, top_k, reps, commit, rnd):
    iters = harness.calibrate_iters(fn)
    warmup = max(10, iters // 10)
    xs = harness.repeat_amortized(fn, reps, iters, warmup)
    med = statistics.median(xs)
    sp = auto_splits(batch, top_k)
    return {
        "op": op,
        "phase": phase,
        "batch": batch,
        "vocab": vocab,
        "top_k": top_k,
        "dtype": "bfloat16",
        "splits": sp,
        "blocks": sp * batch,
        "median_us": f"{med:.3f}",
        "p05_us": f"{harness.percentile(xs, 0.05):.3f}",
        "p95_us": f"{harness.percentile(xs, 0.95):.3f}",
        "mean_us": f"{statistics.mean(xs):.3f}",
        "stdev_us": f"{statistics.stdev(xs) if len(xs) > 1 else 0.0:.3f}",
        "reps": reps,
        "iters": iters,
        "warmup": warmup,
        "round": rnd,
        "git_commit": commit,
    }


def spec(x, top_k, top_p):
    out = [("noop", 0, lambda: fused_sampling.probe_noop(x))]
    for ph in range(1, N_PHASES + 1):
        out.append((f"phase{ph}", ph,
                    (lambda ph=ph: lambda: fused_sampling.probe_phase(x, top_k, top_p, ph, 0))()))
    # control: phase 7 is the production path, so these two must agree
    out.append(("full", N_PHASES, lambda: sample_fused(x, top_k, top_p)))
    return out


def breakdown(rows):
    """Median across rounds per (batch, top_k, phase), then adjacent differences."""
    med = {}
    for r in rows:
        med.setdefault((r["batch"], r["top_k"], r["op"]), []).append(float(r["median_us"]))
    med = {k: statistics.median(v) for k, v in med.items()}
    out = []
    for (b, k, op) in sorted({(b, k, op) for (b, k, op) in med}):
        if not op.startswith("phase"):
            continue
        ph = int(op[5:])
        prev = med[(b, k, "noop")] if ph == 1 else med[(b, k, f"phase{ph - 1}")]
        cur = med[(b, k, op)]
        out.append({
            "batch": b, "top_k": k, "phase": ph, "label": PHASE_LABEL[ph],
            "cumulative_us": round(cur, 3), "delta_us": round(cur - prev, 3),
            "pct_of_total": round(100.0 * (cur - prev) / med[(b, k, f"phase{N_PHASES}")], 1),
        })
    return out


def write(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/raw/kernel_phases.csv")
    ap.add_argument("--breakdown-out", default="results/raw/kernel_phase_breakdown.csv")
    ap.add_argument("--vocab", type=int, default=ANCHOR_VOCAB)
    ap.add_argument("--batches", type=int, nargs="+", default=list(BENCH_BATCH_SIZES))
    ap.add_argument("--top-k", type=int, nargs="+", default=list(BENCH_TOP_K_VALUES))
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()

    if not HAVE_FUSED:
        print("! fused_sampling extension not importable; run python setup.py build_ext --inplace")
        return 2
    if not hasattr(fused_sampling, "probe_phase"):
        print("! the built .so predates probe_phase; rebuild with python setup.py build_ext --inplace")
        return 2
    busy, apps = harness.gpu_is_busy()
    if busy:
        print(f"! GPU is busy, refusing to measure: {apps}")
        return 2

    env = harness.environment(harness.try_lock_clocks())
    rows = []
    for rnd in range(args.rounds):
        for batch in args.batches:
            x = (torch.randn(batch, args.vocab, device="cuda") * 4).to(torch.bfloat16)
            for top_k in args.top_k:
                s = spec(x, top_k, args.top_p)
                s = s[rnd % len(s):] + s[: rnd % len(s)]
                for op, ph, fn in s:
                    r = run(op, ph, fn, batch, args.vocab, top_k, args.reps, env["git_commit"], rnd)
                    rows.append(r)
                    print(f"  r{rnd} B={batch:<3} k={top_k:<4} {op:<8} {r['median_us']:>9} us")
            del x
            torch.cuda.empty_cache()

    write(args.out, rows, FIELDS)
    bd = breakdown(rows)
    write(args.breakdown_out, bd,
          ["batch", "top_k", "phase", "label", "cumulative_us", "delta_us", "pct_of_total"])

    print("\nanchor breakdown, V=%d bf16, k=50:" % args.vocab)
    for b in args.batches:
        print(f"  B={b}")
        for r in bd:
            if r["batch"] == b and r["top_k"] == 50:
                print(f"    {r['phase']}  {r['label']:<24} +{r['delta_us']:>7.2f} us"
                      f"  ({r['pct_of_total']:>5.1f}%)  cum {r['cumulative_us']:>7.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
