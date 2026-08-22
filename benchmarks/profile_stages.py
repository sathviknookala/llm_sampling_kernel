import argparse
import csv
import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile
from transformers.generation.logits_process import TopKLogitsWarper, TopPLogitsWarper

from . import harness
from .implementations import sample_tight
from .hf_baseline import sample_hf

FIELDS = [
    "path", "stage", "batch", "vocab", "top_k", "top_p", "dtype",
    "isolated_us", "kernels", "pct_of_path_total", "path_total_us", "git_commit",
]


def stages_hf(x, top_k, top_p):
    kw = TopKLogitsWarper(top_k=top_k, min_tokens_to_keep=1)
    pw = TopPLogitsWarper(top_p=top_p, min_tokens_to_keep=1)
    f32 = x.float()
    after_k = kw(None, f32)
    after_p = pw(None, after_k)
    probs = torch.softmax(after_p, dim=-1)
    srt = torch.sort(after_k, descending=False)

    return [
        ("cast_fp32", lambda: x.float()),
        ("topk_warper", lambda: kw(None, f32)),
        ("topp_warper", lambda: pw(None, after_k)),
        ("  topp:full_sort", lambda: torch.sort(after_k, descending=False)),
        ("  topp:softmax", lambda: srt.values.softmax(dim=-1)),
        ("  topp:cumsum", lambda: srt.values.softmax(dim=-1).cumsum(dim=-1)),
        ("  topp:scatter+mask", lambda: after_k.masked_fill(
            torch.zeros_like(after_k, dtype=torch.bool).scatter(
                1, srt.indices, torch.zeros_like(after_k, dtype=torch.bool)
            ), -float("inf"))),
        ("final_softmax", lambda: torch.softmax(after_p, dim=-1)),
        ("multinomial", lambda: torch.multinomial(probs, 1)),
        ("TOTAL", lambda: sample_hf(x, top_k, top_p)),
    ]


def stages_tight(x, top_k, top_p):
    f32 = x.float()
    vals, ids = torch.topk(f32, top_k, dim=-1)
    probs = torch.softmax(vals, dim=-1)
    cumsum = probs.cumsum(dim=-1)
    keep = (cumsum - probs) < top_p
    filt = torch.where(keep, probs, torch.zeros_like(probs))
    renorm = filt / filt.sum(dim=-1, keepdim=True)
    choice = torch.multinomial(renorm, 1)

    return [
        ("cast_fp32", lambda: x.float()),
        ("topk", lambda: torch.topk(f32, top_k, dim=-1)),
        ("softmax_k", lambda: torch.softmax(vals, dim=-1)),
        ("cumsum_k", lambda: probs.cumsum(dim=-1)),
        ("topp_mask_k", lambda: (cumsum - probs) < top_p),
        ("renorm_k", lambda: filt / filt.sum(dim=-1, keepdim=True)),
        ("multinomial_k", lambda: torch.multinomial(renorm, 1)),
        ("gather", lambda: ids.gather(-1, choice).squeeze(-1)),
        ("TOTAL", lambda: sample_tight(x, top_k, top_p)),
    ]


def count_kernels(fn):
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
    ap.add_argument("--out", default="results/raw/stage_profile.csv")
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32])
    ap.add_argument("--vocab", type=int, default=151936)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.90)
    args = ap.parse_args(argv)

    busy, apps = harness.gpu_is_busy()
    if busy:
        print(f"REFUSING: GPU busy: {apps}")
        return 2

    commit = harness.git_commit()
    rows = []
    for batch in args.batches:
        torch.manual_seed(0)
        x = (torch.randn(batch, args.vocab, device="cuda") * 4).to(torch.bfloat16)
        for path, builder in (("hf_eager", stages_hf), ("tight_eager", stages_tight)):
            built = builder(x, args.top_k, args.top_p)
            total_us = None
            measured = []
            for name, fn in built:
                iters = harness.calibrate_iters(fn, target_ms=80.0)
                us = harness.amortized_us(fn, iters, max(10, iters // 10))
                k = count_kernels(fn)
                measured.append((name, us, k))
                if name == "TOTAL":
                    total_us = us
            for name, us, k in measured:
                rows.append({
                    "path": path,
                    "stage": name.strip(),
                    "batch": batch,
                    "vocab": args.vocab,
                    "top_k": args.top_k,
                    "top_p": args.top_p,
                    "dtype": "bfloat16",
                    "isolated_us": f"{us:.3f}",
                    "kernels": k,
                    "pct_of_path_total": f"{100 * us / total_us:.2f}",
                    "path_total_us": f"{total_us:.3f}",
                    "git_commit": commit,
                })
                print(f"  B={batch:3d} {path:12s} {name:22s} {us:9.2f}us {k:3d}k "
                      f"{100*us/total_us:6.1f}%")
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
