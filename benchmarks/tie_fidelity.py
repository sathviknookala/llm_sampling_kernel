import argparse
import csv
import subprocess
import sys
from pathlib import Path

import torch

from .hf_baseline import hf_probs
from .reference import sample_eager

FIELDS = [
    "dtype", "vocab", "batch", "top_k", "top_p",
    "hf_tv_vs_fp32_mean", "hf_tv_vs_fp32_max",
    "ours_tv_vs_fp32_mean", "ours_tv_vs_fp32_max",
    "tv_vs_hf_mean", "tv_vs_hf_max",
    "rows_with_boundary_tie", "tie_multiplicity_median", "tie_multiplicity_max",
    "ours_closer_to_fp32", "git_commit",
]


def probs_vector(logits, top_k, top_p, generator):
    s = sample_eager(logits, top_k, top_p, generator=generator, return_stages=True)
    full = torch.zeros(logits.shape[0], logits.shape[1], device=logits.device)
    return full.scatter_(-1, s.topk_ids, s.renormed)


def tv(a, b):
    return 0.5 * (a - b).abs().sum(-1)


def measure(logits32, dtype, top_k, top_p, generator):
    xq = logits32.to(dtype)
    truth = probs_vector(logits32, top_k, top_p, generator)
    ours = probs_vector(xq, top_k, top_p, generator)
    hf = hf_probs(xq, top_k, top_p)

    tv_hf32, tv_ours32, tv_hf = tv(hf, truth), tv(ours, truth), tv(ours, hf)
    xf = xq.float()
    kth = xf.topk(min(top_k, xf.shape[-1]), dim=-1).values[:, -1]
    mult = (xf == kth.unsqueeze(-1)).sum(-1)

    return {
        "dtype": str(dtype).replace("torch.", ""),
        "vocab": logits32.shape[1],
        "batch": logits32.shape[0],
        "top_k": top_k,
        "top_p": top_p,
        "hf_tv_vs_fp32_mean": f"{tv_hf32.mean():.6f}",
        "hf_tv_vs_fp32_max": f"{tv_hf32.max():.6f}",
        "ours_tv_vs_fp32_mean": f"{tv_ours32.mean():.6f}",
        "ours_tv_vs_fp32_max": f"{tv_ours32.max():.6f}",
        "tv_vs_hf_mean": f"{tv_hf.mean():.6f}",
        "tv_vs_hf_max": f"{tv_hf.max():.6f}",
        "rows_with_boundary_tie": f"{(mult > 1).float().mean():.4f}",
        "tie_multiplicity_median": int(mult.median()),
        "tie_multiplicity_max": int(mult.max()),
        "ours_closer_to_fp32": int(tv_ours32.mean() <= tv_hf32.mean()),
    }


def git_commit():
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/raw/tie_fidelity.csv")
    ap.add_argument("--vocab", type=int, default=128256)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    logits32 = torch.randn(args.batch, args.vocab, device=args.device) * 4
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    commit = git_commit()

    rows = []
    for dtype in (torch.float16, torch.bfloat16):
        for top_k in (20, 50, 100):
            for top_p in (0.90, 0.95, 1.0):
                row = measure(logits32, dtype, top_k, top_p, generator)
                row["git_commit"] = commit
                rows.append(row)
                print(
                    f"{row['dtype']:9s} k={top_k:3d} p={top_p:.2f}  "
                    f"hf->fp32 {row['hf_tv_vs_fp32_mean']}  "
                    f"ours->fp32 {row['ours_tv_vs_fp32_mean']}  "
                    f"ties {row['rows_with_boundary_tie']}"
                )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    losses = [r for r in rows if not r["ours_closer_to_fp32"]]
    print(f"\n{len(rows)} rows -> {args.out}")
    print(f"configs where fixed-k is NOT closer to fp32 than HF: {len(losses)}")
    return 1 if losses else 0


if __name__ == "__main__":
    sys.exit(main())
