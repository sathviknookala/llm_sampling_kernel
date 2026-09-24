"""Per-kernel CUPTI trace of one sampling step: launch count and device time, per path.

torch.profiler's CUDA activity trace needs no counter permission, unlike ncu. device_us is the sum
of kernel durations -- device-busy time. It excludes host launch overhead and the idle gaps between
dependent launches, so it is a floor on the path's latency, not the ladder's measured latency.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from . import harness
from .implementations import EAGER_FNS, PROBS_INPUT_IMPLS
from .regime import ANCHOR_TOP_K, ANCHOR_TOP_P, ANCHOR_VOCAB

FIELDS = [
    "impl", "batch", "vocab", "top_k", "top_p", "dtype", "kind", "name",
    "launches_per_step", "device_us_per_step", "steps", "git_commit",
]
IMPLS = ("fused_kernel", "hf_eager", "flashinfer_from_probs")


def kind_of(name):
    n = name.lower()
    if n.startswith("memcpy"):
        return "memcpy"
    if n.startswith("memset"):
        return "memset"
    return "kernel"


def trace(fn, steps):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(steps):
            fn()
        torch.cuda.synchronize()
    agg = defaultdict(lambda: [0, 0.0])
    for e in p.events():
        if e.device_type.name == "CUDA":
            agg[e.name][0] += 1
            agg[e.name][1] += e.device_time_total
    return {n: (c / steps, us / steps) for n, (c, us) in agg.items()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/raw/kernel_trace.csv")
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 32])
    ap.add_argument("--steps", type=int, default=50)
    args = ap.parse_args(argv)

    busy, apps = harness.gpu_is_busy()
    if busy:
        print(f"! GPU is busy, refusing to measure: {apps}")
        return 2

    commit = harness.git_commit()
    k, p, v = ANCHOR_TOP_K, ANCHOR_TOP_P, ANCHOR_VOCAB
    rows = []
    for batch in args.batches:
        torch.manual_seed(0)
        logits = (torch.randn(batch, v, device="cuda") * 4).to(torch.bfloat16)
        probs = torch.softmax(logits.float(), dim=-1)
        for impl in IMPLS:
            if impl not in EAGER_FNS:
                print(f"  ! {impl} not importable in this env, skipped")
                continue
            fn = EAGER_FNS[impl]
            x = probs if impl in PROBS_INPUT_IMPLS else logits
            per = trace(lambda: fn(x, k, p), args.steps)
            base = {"impl": impl, "batch": batch, "vocab": v, "top_k": k, "top_p": p,
                    "dtype": "bfloat16", "steps": args.steps, "git_commit": commit}
            for name, (c, us) in sorted(per.items(), key=lambda kv: -kv[1][1]):
                rows.append({**base, "kind": kind_of(name), "name": name,
                             "launches_per_step": round(c, 3), "device_us_per_step": round(us, 3)})
            for kind in ("kernel", "memcpy", "memset"):
                sel = [cu for n, cu in per.items() if kind_of(n) == kind]
                rows.append({**base, "kind": f"total_{kind}", "name": "",
                             "launches_per_step": round(sum(c for c, _ in sel), 3),
                             "device_us_per_step": round(sum(u for _, u in sel), 3)})
            tk = rows[-3]
            print(f"  B={batch:3d} {impl:22s} {tk['launches_per_step']:6.1f} kernels "
                  f"{tk['device_us_per_step']:8.2f} us device  "
                  f"(+{rows[-2]['launches_per_step']:.0f} memcpy, +{rows[-1]['launches_per_step']:.0f} memset)")
            if impl == "fused_kernel":
                for name, (c, us) in per.items():
                    print(f"        {us:7.2f} us  x{c:.0f}  {name[:90]}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
