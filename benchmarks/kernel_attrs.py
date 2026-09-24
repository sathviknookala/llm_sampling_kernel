"""Static resources and theoretical occupancy of every production kernel instantiation.

cudaFuncGetAttributes + cudaOccupancyMaxActiveBlocksPerMultiprocessor, at the shipped block size.
This is the figure Nsight Compute reports as theoretical occupancy; it needs no counter
permission, so it survives RmProfilingAdminOnly: 1. Achieved occupancy does not.
"""

import argparse
import csv
from pathlib import Path

import torch

from . import harness
from .kernel_phases import auto_splits
from .regime import BENCH_BATCH_SIZES, BENCH_TOP_K_VALUES

import fused_sampling

FIELDS = [
    "kernel", "P", "block", "regs_per_thread", "local_bytes", "static_smem_bytes",
    "max_blocks_per_sm", "active_warps_per_sm", "max_warps_per_sm", "theoretical_occupancy_pct",
    "limiter", "shipped_at", "gpu", "git_commit",
]


def merge_width(batch, top_k):
    """Mirrors make_plan: the merge instantiation a (B, k) point dispatches to."""
    n = auto_splits(batch, top_k) * top_k
    return max(32, 1 << (n - 1).bit_length())


def limiters(props, block, regs, smem, blocks):
    warps = -(-block // 32)
    reg_warp = -(-regs * 32 // 256) * 256
    limits = {
        "threads": props.max_threads_per_multi_processor // block,
        "registers": (props.regs_per_multiprocessor // reg_warp) // warps,
        # 1 KB of shared memory is reserved per resident block on sm_80+
        "shared_memory": props.shared_memory_per_multiprocessor // (smem + 1024),
    }
    return "+".join(k for k, v in limits.items() if v == blocks) or "other"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/raw/kernel_attrs.csv")
    args = ap.parse_args(argv)

    props = torch.cuda.get_device_properties(0)
    max_warps = props.max_threads_per_multi_processor // 32
    shipped = {}
    for b in BENCH_BATCH_SIZES:
        for k in BENCH_TOP_K_VALUES:
            shipped.setdefault(merge_width(b, k), []).append(f"B{b}k{k}")

    commit = harness.git_commit()
    rows = []
    for name, P, block, regs, local, smem, blocks in fused_sampling.kernel_attrs():
        warps = blocks * (-(-block // 32))
        where = "all" if name == "topk_partial_kernel" else " ".join(shipped.get(P, [])) or "none"
        rows.append({
            "kernel": name, "P": P or "", "block": block, "regs_per_thread": regs,
            "local_bytes": local, "static_smem_bytes": smem, "max_blocks_per_sm": blocks,
            "active_warps_per_sm": warps, "max_warps_per_sm": max_warps,
            "theoretical_occupancy_pct": round(100.0 * warps / max_warps, 1),
            "limiter": limiters(props, block, regs, smem, blocks), "shipped_at": where,
            "gpu": props.name, "git_commit": commit,
        })
        r = rows[-1]
        print(f"  {name:26s} P={P:5d} regs={regs:4d} local={local:3d}B smem={smem:6d}B "
              f"blocks/SM={blocks:2d} occ={r['theoretical_occupancy_pct']:5.1f}% "
              f"[{r['limiter']}] {where}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
