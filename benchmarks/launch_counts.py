import argparse
import csv
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from . import harness
from .implementations import (
    EAGER_FNS,
    PROBS_INPUT_IMPLS,
    build_compile,
    build_graph,
    sample_tight,
)

FIELDS = ["impl", "batch", "vocab", "top_k", "top_p", "dtype", "kernels", "note", "git_commit"]


def count(fn):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        fn()
        torch.cuda.synchronize()
    return sum(
        e.count
        for e in p.key_averages()
        if e.device_type.name == "CUDA" and e.self_device_time_total > 0
    )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/raw/launch_counts.csv")
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32])
    ap.add_argument("--vocab", type=int, default=151936)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.90)
    args = ap.parse_args(argv)

    commit = harness.git_commit()
    rows = []
    for batch in args.batches:
        torch.manual_seed(0)
        logits = (torch.randn(batch, args.vocab, device="cuda") * 4).to(torch.bfloat16)
        probs = torch.softmax(logits.float(), dim=-1)
        measured = []

        for name, fn in EAGER_FNS.items():
            x = probs if name in PROBS_INPUT_IMPLS else logits
            measured.append((name, count(lambda: fn(x, args.top_k, args.top_p)), ""))

        c = build_compile(sample_tight, [logits], args.top_k, args.top_p, hot=True)
        measured.append(("compile", count(c.call), "inductor-generated"))
        for nm, cf in (("graph_eager", False), ("graph_compile", True)):
            g = build_graph(sample_tight, [logits], args.top_k, args.top_p, True, nm, cf)
            measured.append((nm, count(g.call), "device kernels unchanged; replay removes the host-side launches"))

        for name, k, note in measured:
            print(f"  B={batch:3d} {name:24s} {k:4d} kernels  {note}")
            rows.append({
                "impl": name, "batch": batch, "vocab": args.vocab, "top_k": args.top_k,
                "top_p": args.top_p, "dtype": "bfloat16", "kernels": k, "note": note,
                "git_commit": commit,
            })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
