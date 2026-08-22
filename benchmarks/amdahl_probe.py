import argparse
import csv
import json
import sys
from pathlib import Path

import torch

from . import harness
from .implementations import EAGER_FNS, build_graph, sample_tight

FIELDS = [
    "model", "params", "vocab", "batch", "prompt_len", "dtype",
    "decode_step_us", "weight_read_floor_us", "decode_vs_floor_ratio",
    "sampler", "sampling_us", "sampling_pct_of_step",
    "speedup_2x_gain_pct", "speedup_5x_gain_pct", "speedup_10x_gain_pct",
    "speedup_inf_gain_pct", "measured", "git_commit",
]


def amdahl_gain_pct(frac, speedup):
    """% reduction in total step time if the sampling fraction is accelerated by `speedup`."""
    if speedup == float("inf"):
        return 100.0 * frac
    return 100.0 * frac * (1 - 1 / speedup)


def decode_step_us(model, vocab, batch, prompt_len, device):
    """Time one dependent decode forward: [B,1] token in, cache of length prompt_len."""
    from transformers import DynamicCache

    ids = torch.randint(0, vocab, (batch, prompt_len), device=device)
    cache = DynamicCache()
    with torch.no_grad():
        model(ids, past_key_values=cache, use_cache=True)
    base_len = cache.get_seq_length()
    nxt = torch.randint(0, vocab, (batch, 1), device=device)

    def fwd():
        # crop back so the cache length -- and therefore the attention work -- is identical
        # on every timed iteration
        cache.crop(base_len)
        with torch.no_grad():
            model(nxt, past_key_values=cache, use_cache=True)

    iters = harness.calibrate_iters(fwd, target_ms=400.0, min_iters=10, max_iters=300)
    return harness.amortized_us(fwd, iters, max(5, iters // 5)), iters


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/raw/amdahl_probe.csv")
    ap.add_argument("--models", nargs="+", default=["Qwen/Qwen2-0.5B", "mistralai/Mistral-7B-v0.1"])
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--prompt-len", type=int, default=512)
    args = ap.parse_args(argv)

    busy, apps = harness.gpu_is_busy()
    if busy:
        print(f"REFUSING: GPU busy: {apps}")
        return 2

    from transformers import AutoConfig, AutoModelForCausalLM

    bw = harness.measured_dram_bandwidth()
    commit = harness.git_commit()
    rows = []

    for name in args.models:
        try:
            cfg = AutoConfig.from_pretrained(name)
            model = AutoModelForCausalLM.from_pretrained(
                name, dtype=torch.bfloat16
            ).eval().to("cuda")
        except Exception as e:
            print(f"! skipping {name}: {type(e).__name__}: {str(e)[:160]}")
            continue

        params = sum(p.numel() for p in model.parameters())
        vocab = cfg.vocab_size
        floor_us = params * 2 / bw * 1e6
        print(f"\n=== {name}: {params/1e9:.2f}B params, V={vocab}, "
              f"weight-read floor {floor_us/1000:.2f} ms ===")

        for batch in args.batches:
            try:
                step_us, iters = decode_step_us(model, vocab, batch, args.prompt_len, "cuda")
            except Exception as e:
                print(f"  ! decode step failed B={batch}: {type(e).__name__}: {str(e)[:160]}")
                continue
            print(f"  B={batch}: decode step {step_us/1000:.3f} ms "
                  f"({step_us/floor_us:.2f}x the weight-read floor, {iters} iters)")

            torch.manual_seed(0)
            logits = (torch.randn(batch, vocab, device="cuda") * 4).to(torch.bfloat16)
            samplers = {}
            for sname in ("hf_eager", "flashinfer"):
                if sname in EAGER_FNS:
                    fn = EAGER_FNS[sname]
                    it = harness.calibrate_iters(lambda: fn(logits, 50, 0.90))
                    samplers[sname] = harness.amortized_us(
                        lambda: fn(logits, 50, 0.90), it, max(10, it // 10)
                    )
            try:
                g = build_graph(sample_tight, [logits], 50, 0.90, hot=True)
                it = harness.calibrate_iters(g.call)
                samplers["graph_eager"] = harness.amortized_us(g.call, it, max(10, it // 10))
            except Exception as e:
                print(f"    ! graph sampler failed: {type(e).__name__}")

            for sname, s_us in samplers.items():
                frac = s_us / step_us
                rows.append({
                    "model": name,
                    "params": params,
                    "vocab": vocab,
                    "batch": batch,
                    "prompt_len": args.prompt_len,
                    "dtype": "bfloat16",
                    "decode_step_us": f"{step_us:.2f}",
                    "weight_read_floor_us": f"{floor_us:.2f}",
                    "decode_vs_floor_ratio": f"{step_us/floor_us:.3f}",
                    "sampler": sname,
                    "sampling_us": f"{s_us:.2f}",
                    "sampling_pct_of_step": f"{100*frac:.3f}",
                    "speedup_2x_gain_pct": f"{amdahl_gain_pct(frac, 2):.4f}",
                    "speedup_5x_gain_pct": f"{amdahl_gain_pct(frac, 5):.4f}",
                    "speedup_10x_gain_pct": f"{amdahl_gain_pct(frac, 10):.4f}",
                    "speedup_inf_gain_pct": f"{amdahl_gain_pct(frac, float('inf')):.4f}",
                    "measured": "true",
                    "git_commit": commit,
                })
                print(f"    {sname:14s} {s_us:8.1f} us = {100*frac:6.3f}% of step "
                      f"-> 10x sampler saves {amdahl_gain_pct(frac,10):.3f}% of decode")
            sys.stdout.flush()

        del model
        torch.cuda.empty_cache()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
