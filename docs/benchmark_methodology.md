# Benchmark Methodology

How sampling latency is measured in this repo, and what each number does and does not include.
Read this before quoting or adding a benchmark number.

## What is being measured

One decode step's sampling: `[B, V] logits -> [B] int64 token ids`, top-k → softmax → cumsum →
top-p mask → renormalize → sample, per `benchmarks/SEMANTICS.md`.

**Temperature is outside the timed operation.** HF applies it as a separate warper, i.e. an extra
full-vocabulary pass. Including it would inflate the baseline with work the fused kernel absorbs
for free, so every rung here measures top-k/top-p sampling alone.

## Latency definition

`median_us` is **amortized device time per sampling call**:

```
warm up W times
record CUDA event
call the sampler N times back to back
record CUDA event, synchronize
per_call = elapsed_ms * 1000 / N
```

Repeated `reps` times per cell; the CSV carries median, p05, p95, mean, stdev, `iters`, `warmup`.

**Why amortized and not per-call.** CUDA-event bracketing costs several microseconds on a call that
launches nothing. The fastest rungs here are ~70-90 µs, but at small `B` a future fused kernel is
expected well below that, and per-call bracketing would put the instrument inside the measurement.
`N` is calibrated per cell to target ~120 ms of work, so timer overhead is a rounding error.

This is throughput-style timing: launches from consecutive iterations overlap, so it **understates
per-call launch latency** in a real decode loop where the sampler is called once between dependent
GEMMs. It is the right instrument for comparing rungs and the wrong one for claiming an absolute
end-to-end decode cost. The Amdahl probe measures the dependent-call case separately.

## Validation is never inside the timed region

Every timed call runs with `check_inputs=False`. `validation_in_timed_region=false` is recorded on
every row. This matters twice: the NaN/Inf check costs a device-to-host sync, and it is exactly the
work a fused kernel gets to skip — leaving it on would flatter the kernel against its own baseline.

**Never feed a known-bad tensor with checks off.** `torch.multinomial`'s device-side assert poisons
the CUDA context for the whole process, so one bad row loses the entire run rather than one cell.

## Ordering control

Earlier work on the sibling repo measured first-rung ordering artifacts up to 25%. The ladder is
therefore run for `--rounds` complete passes, and **the implementation order is rotated by one
position each round**, so no rung is always measured first on a freshly-idled GPU. Every round's
rows are kept in the raw CSV with a `round` column; the summary reports round-to-round spread so
the artifact is visible rather than averaged away.

## Hot vs cold L2 residency

The logits tensor fits in this GPU's 48 MB L2 for **every** configuration in the target regime
(largest: B=32, V=151936, FP16 = 9.3 MB). Repeatedly sampling one buffer therefore measures an
L2-resident workload.

Both conditions are measured and neither is called correct:

- **hot** — one buffer, reused every iteration. Approximates logits still resident after the LM
  head wrote them.
- **cold** — rotate through enough distinct buffers to total 4× L2, so a given buffer is evicted
  before it is read again. `cold_method=rotate`.

**CUDA-graph rungs cannot express `rotate` directly**: replay requires a fixed input address, so
the cold variant copies from the rotating pool into the graph's static input inside the timed
region. That copy is real work a graph-based serving path also pays, but it is not the same
measurement as `rotate`, and it is recorded as `cold_method=rotate_into_static`.

## Memory-read floor

`dram_floor_us = B * V * itemsize / measured_bandwidth`, where the bandwidth is measured on this
GPU by a large device-to-device copy at run time (not a datasheet figure) and stored in
`results/raw/environment.json`. `floor_frac_of_latency` is the ratio to measured latency. This
exists to test the original memory-bound hypothesis, not to assume it.

## Per-stage attribution

`benchmarks/profile_stages.py` times each stage **in isolation** with the same amortized timer, and
separately counts CUDA kernels per stage with `torch.profiler`. Isolated timing is used for
attribution rather than profiler device time, because the profiler perturbs execution; kernel
counts come from the profiler because they are structural and unaffected by its overhead.

Isolated stage timings do not sum exactly to the measured total — each stage pays its own launch
and the allocator behaves differently in isolation. They are an attribution tool, not a
decomposition identity.

## GPU state

Recorded in `results/raw/environment.json` on every run: GPU name, SM version, L2 size, SM count,
driver, torch/CUDA/transformers/flashinfer versions, max SM clock, persistence mode, whether any
other process held GPU memory at start, and the git commit.

**Clocks cannot be locked on this machine** — `nvidia-smi -lgc` and `-pm` both return insufficient
permissions for this user, so `clocks_locked=false` on every row. Numbers therefore carry whatever
boost-clock variation the GPU chose. The run refuses to start if another process holds more than
512 MiB.

## Baseline ladder

| Rung | What it is |
|---|---|
| `hf_eager` | HF `transformers` warpers + full-vocab softmax + multinomial over `V`. The Phase 1 contract. |
| `ref_eager_fullsort` | This repo's semantic reference — full `[B,V]` stable sort for the specified tie-break. Correctness oracle, not a performance target. |
| `tight_eager` | `torch.topk` collapse to `[B,K]`, then all later stages at `K`. Isolates "stop working at full-vocab width" from "fuse". **Not tie-exact** — uses `torch.topk`, whose tie order is unspecified. |
| `compile` | `torch.compile(tight_eager, dynamic=False)`, warmed before timing. |
| `graph_eager` | CUDA-graph capture/replay of `tight_eager`. |
| `graph_compile` | CUDA-graph capture/replay of the compiled version. |
| `flashinfer` | FlashInfer `top_k_top_p_sampling_from_probs`, logits→token (full-vocab softmax inside the timed region, matching every other rung's contract). |
| `flashinfer_from_probs` | The same kernel, probs→token — its cost alone, the way vLLM calls it. |

### FlashInfer environment caveat

FlashInfer's published wheel requires torch ≥2.13 built against CUDA 13, and this machine's driver
(575.64.03) supports only CUDA 12.9 — installing it into the main environment would have replaced
the pinned torch 2.11.0+cu128 the semantics were verified against, and the CUDA-13 build cannot
reach the GPU at all on this driver.

It is therefore installed in an **isolated virtualenv** (`~/.venv_flashinfer`) with
`torch 2.9.1+cu128` and `transformers 5.12.1`, where its kernels JIT-compile for `sm_120` against
the local CUDA 12.9 `nvcc`. **The whole ladder is run inside that venv**, so FlashInfer is compared
against rungs measured under identical torch and CUDA — no cross-environment comparison is made.
The torch version differs from the repo's pinned 2.11.0; the environment JSON records which was
used, and any conclusion resting on a small margin should be re-checked under both.

### FlashInfer semantic difference

FlashInfer applies top-k and top-p jointly to the **full-vocabulary** probability distribution.
This repo's semantics renormalize within the top-k survivors *before* applying top-p, so the same
`top_p` retains fewer tokens here. FlashInfer is included as a **performance** rung, not as a
correctness oracle, and it is not gated against `reference.py`.

## Reproduction

```
python -m benchmarks.benchmark_sampling --rounds 3 --reps 5     # results/raw/sampling_ladder.csv
python -m benchmarks.profile_stages                             # results/raw/stage_profile.csv
python -m benchmarks.amdahl_probe                               # results/raw/amdahl_probe.csv
python -m benchmarks.summarize                                  # results/summary_ladder.md
python -m benchmarks.tie_fidelity                               # results/raw/tie_fidelity.csv
```

For the FlashInfer rungs, run with the venv interpreter and its `bin` on `PATH` (JIT needs `ninja`
and `nvcc`):

```
PATH=~/.venv_flashinfer/bin:/home/sathvik/cuda-12.9/bin:$PATH \
  ~/.venv_flashinfer/bin/python -m benchmarks.benchmark_sampling
```

## Limitations

- **Clocks are not locked** — no permission on this machine.
- **Amortized timing understates dependent-call launch latency** by design; see above.
- **Synthetic Gaussian logits.** Real decode logits are heavy-tailed. This changes tie density and
  can change top-p cutoff positions, so per-stage costs may shift on real data.
- **No `ncu` counters.** DRAM traffic is a computed floor from a measured copy bandwidth, not
  measured traffic; `ncu` permissions are unverified on this machine.
- **`torch.compile` runs at default mode.** `max-autotune` was not swept.
