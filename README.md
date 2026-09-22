# Fused Top-K / Top-P CUDA Sampling Kernel

A custom CUDA implementation of decode-time **top-k + top-p sampling** for large-vocabulary LLMs. The kernel specializes the `[B, V] logits → [B] token ids` path for low decode batches and 128K–152K vocabularies, targeting the fixed-cost selection, sorting, and launch overhead left by general-purpose sampling implementations.

On an **NVIDIA RTX PRO 4000 Blackwell (SM120)**, the current implementation runs **2.28×–4.09× faster than FlashInfer's `top_k_top_p_sampling_from_probs`** across the measured parameter grid. At the anchor configuration (`V=151936`, `k=50`, `p=0.90`, BF16, hot L2), latency falls from **75.4 µs to 22.5 µs at B=1**.

Every number in this README comes from a committed artifact under `results/`, named at the point it is used. All performance measurements were taken at commit `df3e3c0`.

## Project Objective

The project asks a narrow systems question:

> How much decode-time sampling latency remains after replacing generic full-vocabulary PyTorch operations with a workload-specialized CUDA implementation?

The operation is:

```text
[B, V] logits
    → top-k
    → softmax over survivors
    → cumulative probability
    → top-p cutoff
    → renormalization
    → random draw
    → [B] token ids
```

The original Hugging Face eager path performs much of this work over the full vocabulary and launches **64–70 CUDA kernels per sampling call** (`results/raw/launch_counts.csv`). A tighter PyTorch implementation and CUDA Graphs remove much of that overhead before any custom CUDA is written.

For this reason, the primary performance baseline is not Hugging Face eager. It is:

```text
FlashInfer 0.6.17
top_k_top_p_sampling_from_probs
```

The primary metric is **amortized device latency per sampling call in microseconds**.

The current result is an **operator-level optimization**. Sampling accounts for only approximately 0.16–1.2% of the measured end-to-end decode step in the model probes in this repository (`results/raw/amdahl_probe.csv`), so the project does not claim a comparable end-to-end decode speedup.

## Kernel Design

The implementation is in `csrc/fused_sampling.cu` and maps one logical sampling operation onto two CUDA kernels.

```text
logits [B, V]
      │
      ▼
┌───────────────────────────┐
│ topk_partial_kernel       │
│ grid = (splits, B)        │
│                           │
│ 16-bit radix selection    │
│ exact k-boundary handling │
│ per-slice top-k           │
└─────────────┬─────────────┘
              │ splits × K packed candidates
              ▼
┌───────────────────────────┐
│ merge_sample_kernel       │
│ grid = B                  │
│                           │
│ merge candidate lists     │
│ FP32 softmax              │
│ top-p cutoff              │
│ random draw               │
└─────────────┬─────────────┘
              │
              ▼
        token ids [B]
```

### 16-bit selection

FP16 and BF16 logits are both 16-bit sign-magnitude values. The kernel maps their bit pattern to an order-preserving unsigned key — `key = b ^ (b & 0x8000 ? 0xFFFF : 0x8000)` — so that descending-by-value and descending-by-key are the same ordering for both dtypes. Selection then needs **two 8-bit radix passes** rather than the four an FP32 key would need.

Each candidate is packed with the complement of its vocabulary index into a single 64-bit word:

```text
(value key, ~token id)
```

This makes every packed value unique and reduces the tie rule to a single 64-bit comparison. Equal logits resolve to the **lowest token id**, matching the repository's reference semantics.

### Split top-k

For low batch sizes, assigning one block to an entire 128K–152K-element row leaves most of the GPU idle. `topk_partial_kernel` therefore splits each row across several blocks.

Each split:

1. scans its section of the vocabulary,
2. builds a high-byte histogram,
3. locates the top-k boundary bucket,
4. builds a low-byte histogram inside that bucket,
5. resolves the exact threshold and boundary ties,
6. emits its local top-k candidates.

The split count is chosen from the batch size — `min(8, max(4, 128/B))`, then clamped by the merge buffer and by the vocabulary slice size. The rule is measured (`results/raw/kernel_splits.csv`) but not yet optimal; see *Experimental / future work*.

### Candidate merge and sampling

The second kernel merges the `splits × K` candidates, decodes the values back out of the keys — the logits are never read a second time — and performs the remaining operation without returning to a full `[B, V]` representation.

It performs:

- candidate merge/sort,
- FP32 exponentiation and normalization,
- cumulative probability calculation,
- top-p cutoff on the exclusive prefix,
- an inverse-CDF draw against `u · Z_p`.

**No normalized probability tensor is materialized.** Scaling the uniform draw by the retained mass `Z_p` replaces renormalization and yields the identical categorical distribution.

The merge that every measurement in this README was taken on is a shared-memory bitonic sort. A register-resident warp merge (`fs::warp_merge_sort`, one warp, `__shfl_xor_sync` across lanes) has since landed on the `register-merge` branch; it compiles without spill but **has not been tested or benchmarked**, and no number here reflects it.

### Numerical behavior

Input logits are FP16 or BF16, while probability computations and reductions use **FP32**. `--use_fast_math` is deliberately off, so the kernel stays within reach of the FP32 reference it is gated against.

The kernel also contains an exact fallback for unusually large k-boundary tie sets — index bucketing plus a bitmap over the boundary bucket, which cannot overflow because the bucket is `2^shift` wide — rather than relying on a fixed-capacity approximation.

## Optimization Strategy

The measured development path was:

```text
Hugging Face full-vocabulary eager sampling
        ↓
collapse work to K candidates after top-k
        ↓
CUDA Graph replay to reduce host launch overhead
        ↓
compare against FlashInfer production sampler
        ↓
specialized split-row CUDA selection
        ↓
two-pass 16-bit radix threshold search
        ↓
fuse merge + softmax + top-p + sampling
        ↓
optimize warp/block reductions and boundary search
```

Several measurements shaped the design:

- Full-vocabulary DRAM traffic was **not** the dominant cost of the eager implementation — reading the logits is 0.17–0.80% of `hf_eager` latency (`results/DECISION.md`).
- Hugging Face's full-vocabulary sort dominated at larger batches — 65% of `hf_eager` at B=32 (`results/raw/stage_profile.csv`).
- Low-batch execution was primarily limited by launch latency and sequential phase cost; at B=1 every eager stage costs 55–68 µs regardless of size.
- Replacing full-vocabulary processing with `torch.topk` captured much of the easy algorithmic gain before custom CUDA.
- CUDA Graphs reduced host-launch overhead but left the underlying device work intact.
- After implementing the fused CUDA path, phase ablation identified the **candidate merge** as the largest individual B=1 cost.

More aggressive register-resident selection and a barrier-free merge remain optimization directions; they are not included in the measured headline result.

## Benchmark Methodology

The primary measurements were collected on:

| Component | Configuration |
|---|---|
| GPU | NVIDIA RTX PRO 4000 Blackwell |
| Architecture | SM120 |
| SM count | 70 |
| GPU memory | 25.2 GB |
| L2 cache | 48 MB |
| Driver | 575.64.03 |
| Local CUDA toolkit | CUDA 12.9 (`nvcc`) |
| PyTorch | 2.9.1+cu128 |
| FlashInfer | 0.6.17 |
| Transformers | 5.12.1 |
| Measured HBM copy bandwidth | ~553 GB/s |

Captured per run in `results/raw/environment_spike.json`. The FlashInfer comparison is run in the same Python process and software environment as the custom kernel.

### Target regime

```text
Vocabulary:  128256, 151936
Batch:       low decode batches up to 32
Top-k:       20, 50, 100
Top-p:       0.90, 0.95
Input dtype: FP16, BF16
Compute:     FP32
```

### Timing

Reported `median_us` is amortized CUDA device time per sampling operation:

```text
warm up
record CUDA event
run N sampling calls
record CUDA event
synchronize
latency = elapsed / N
```

`N` is calibrated per benchmark cell to target approximately 120 ms of total measured work.

Each cell uses:

- **5 repetitions**
- approximately 10% as many warmup calls as timed iterations, with a minimum of 10 (later repetitions reuse a quarter of that)
- **3 full benchmark rounds**
- rotated implementation ordering between rounds

Validation is performed outside the timed region, and **temperature is excluded from the timed operation** for every rung — including it would charge the baselines an extra full-vocabulary pass the fused kernel absorbs for free.

The benchmark records median, p05, p95, mean, standard deviation, iteration count, warmup count, configuration, cache condition, git revision, and environment metadata.

### L2 residency

The logits tensor fits in this GPU's 48 MB L2 at every point in the target regime (largest: B=32, V=151936, FP16 = 9.3 MB), so both cache conditions are explicitly tested:

- **hot** — reuse the same tensor; approximates sampling immediately after the LM head writes logits.
- **cold** — rotate through distinct buffers totalling 4× L2, so a buffer is evicted before it is read again.

The primary headline uses the hot condition.

### Baselines

The benchmark ladder includes:

- Hugging Face eager sampling
- full-sort semantic reference
- tighter eager PyTorch using `torch.topk`
- `torch.compile`
- CUDA Graph eager
- CUDA Graph + compiled
- FlashInfer logits → token
- FlashInfer probabilities → token
- custom fused CUDA kernel

`flashinfer_from_probs` is the primary performance bar.

FlashInfer receives an already-computed `[B,V]` probability tensor, while the custom kernel receives raw logits and performs its own probability computation. FlashInfer also applies top-k/top-p to the full-vocabulary distribution where this kernel renormalizes within the top-k survivors first; it is therefore a **performance baseline, not a correctness oracle**.

Full methodology is documented in `docs/benchmark_methodology.md`.

## Results

### Headline

Across the complete measured parameter grid:

**custom fused kernel: 2.28×–4.09× faster than `flashinfer_from_probs`**

At the anchor configuration:

```text
V = 151936
top_k = 50
top_p = 0.90
dtype = BF16
residency = hot
```

| Batch | FlashInfer from probs | Fused kernel | Speedup |
|---:|---:|---:|---:|
| 1 | 75.4 µs | **22.5 µs** | **3.35×** |
| 4 | 75.4 µs | **22.5 µs** | **3.35×** |
| 8 | 75.5 µs | **22.6 µs** | **3.35×** |
| 16 | 75.4 µs | **26.6 µs** | **2.83×** |
| 32 | 101.2 µs | **34.8 µs** | **2.91×** |

The low-batch latency remains almost flat through B=8 despite an 8× increase in total input work, consistent with the measured latency-bound behavior of this regime.

### Top-k sensitivity

For `V=151936`, BF16, hot L2:

| Configuration | Fused kernel | Speedup vs FlashInfer |
|---|---:|---:|
| B=1, k=20 | **20.3 µs** | **3.69×** |
| B=1, k=50 | **22.5 µs** | **3.35×** |
| B=1, k=100 | **30.3 µs** | **2.48×** |
| B=32, k=20 | **30.7 µs** | **3.35×** |
| B=32, k=50 | **34.8 µs** | **2.91×** |
| B=32, k=100 | **38.3 µs** | **2.69×** |

FP16 and BF16 were measured against each other at `V=151936`, `k=50`, `p=0.90`: they agree to within 0.1% at B ≤ 8 hot, and the widest divergence across those ten cells is 7.8% (B=1, cold L2), inside this rung's round-to-round spread.

Cold-L2 measurements reduce the advantage, with measured speedups of approximately **2.53×–3.37×**.

Full measurements are stored in `results/raw/spike_ladder.csv` (1134 rows).

## Why the Kernel Is Faster

The speedup does not come primarily from reducing HBM traffic.

The repository's initial profiling rejected that hypothesis: reading the logits accounted for 0.17–0.80% of Hugging Face eager runtime across the initial benchmark regime, and the `[B, V]` tensor is L2-resident throughout.

The measured gains instead come from changing the structure of the sampling operation.

### Avoiding the full-vocabulary sort

Generic sampling implementations may sort or otherwise process the entire 128K–152K vocabulary despite eventually retaining only tens of candidates.

The custom kernel finds the top-k boundary with two 8-bit radix histogram passes and gathers only the required candidates.

### Specializing selection for FP16/BF16

The 16-bit input representation allows the selection problem to operate directly on transformed integer keys. This avoids a generic floating-point sorting pipeline.

### Parallelizing a single vocabulary row

At B=1, one block per sequence would expose very little GPU parallelism.

Splitting each vocabulary row across several blocks allows a single sampling request to occupy more of the 70-SM GPU.

### Keeping later work at K-scale

Once local top-k candidates have been produced, subsequent work operates over `splits × K` or `K`, rather than returning to `[B,V]`.

### Collapsing the sampling tail

Merge, softmax, cumulative probability, top-p cutoff, and the random draw are handled by the second CUDA kernel rather than a chain of independent framework operations.

### Measured remaining bottleneck

Phase ablation at B=1, `V=151936`, `k=50` (`results/raw/kernel_phase_breakdown.csv`, derived from 9 rounds in `results/raw/kernel_phases.csv`) attributes the 22.5 µs total approximately as:

| Component | Latency | Share |
|---|---:|---:|
| dispatch floor | 2.51 µs | 11% |
| high-byte histogram | 1.62 µs | 7% |
| high-byte search | 0.06 µs | <1% |
| low-byte histogram | 2.27 µs | 10% |
| threshold search | 0.15 µs | 1% |
| gather + tie resolution | 3.68 µs | 16% |
| candidate bitonic merge | **8.12 µs** | **36%** |
| softmax + top-p + draw | 4.09 µs | 18% |
| **Total** | **22.51 µs** | |

The candidate merge is therefore the largest measured low-batch optimization target. For scale, `results/raw/kernel_floor.csv` puts one full pass over the vocabulary at **4.49 µs** at B=1, so the kernel is 5.0× above its own floor.

Because `ncu` is unavailable on this machine, this attribution is a wall-clock phase ablation — phases 1..n of the real kernel are run and the kernel then stops — not a hardware-counter profile. Individual sub-2 µs phase deltas at B ≤ 8 are not resolvable; a ~2.05 µs timing quantum migrates between adjacent phases across rounds.

## Repository Structure

```text
.
├── csrc/
│   ├── fused_sampling.cu       # CUDA selection, merge, and sampling kernels
│   ├── fused_sampling.cuh      # CUDA helpers (sorts, reductions, packing)
│   ├── probes.cu               # floor probes: noop, row-block scan, split scan
│   └── bindings.cpp            # PyTorch extension bindings
│
├── benchmarks/
│   ├── SEMANTICS.md            # locked operation contract
│   ├── regime.py               # single source of shapes, dtypes, anchors
│   ├── reference.py            # semantic reference implementation
│   ├── implementations.py      # ladder rungs (HF, eager, compile, graph, FlashInfer)
│   ├── hf_baseline.py          # Hugging Face sampling path
│   ├── fused.py                # Python front end for the CUDA extension
│   ├── harness.py              # timing, calibration, environment capture
│   ├── benchmark_sampling.py   # primary benchmark ladder
│   ├── kernel_floor.py         # launch/scan lower-bound and split-count probes
│   ├── kernel_phases.py        # cumulative kernel phase ablation
│   ├── profile_stages.py       # framework-stage attribution
│   ├── launch_counts.py        # kernels launched per sampling call
│   ├── amdahl_probe.py         # sampling vs complete decode step
│   ├── tie_fidelity.py         # low-precision semantic fidelity
│   └── summarize.py            # benchmark summaries
│
├── tests/
│   ├── test_fused_kernel.py    # kernel correctness, ties, RNG/graph tests
│   ├── test_reference.py       # semantic reference tests
│   ├── test_hf_conformance.py  # reference vs Hugging Face conformance
│   └── test_harness.py         # timing-harness tests
│
├── results/
│   ├── raw/                    # complete benchmark CSV/JSON artifacts
│   ├── DECISION.md             # baseline investigation / go-no-go
│   ├── SPIKE.md                # fused-kernel result and ablations
│   ├── summary_spike.md        # summarized fused-kernel ladder
│   └── summary_ladder.md       # summarized pre-kernel ladder
│
├── docs/
│   └── benchmark_methodology.md
│
└── setup.py                    # CUDA extension build
```

## Reproducing the Benchmarks

The kernel and FlashInfer are benchmarked in the same isolated environment.

```bash
export PATH=~/.venv_flashinfer/bin:/home/sathvik/cuda-12.9/bin:$PATH
V=~/.venv_flashinfer/bin/python
```

Build the extension:

```bash
$V setup.py build_ext --inplace
```

Run correctness tests:

```bash
$V -m pytest tests/ -q
```

Run hardware-floor and split-count probes:

```bash
$V -m benchmarks.kernel_floor
```

Run the phase ablation:

```bash
$V -m benchmarks.kernel_phases --rounds 9
```

Run the primary benchmark:

```bash
$V -m benchmarks.benchmark_sampling \
    --rounds 3 \
    --reps 5 \
    --out results/raw/spike_ladder.csv \
    --env-out results/raw/environment_spike.json
```

Generate the summary:

```bash
$V -m benchmarks.summarize \
    --raw results/raw/spike_ladder.csv \
    --out results/summary_spike.md
```

Primary outputs:

```text
results/raw/spike_ladder.csv
results/raw/environment_spike.json
results/raw/kernel_floor.csv
results/raw/kernel_splits.csv
results/raw/kernel_phases.csv
results/raw/kernel_phase_breakdown.csv
results/summary_spike.md
```

A `.so` built here is not portable across PyTorch minor versions — rebuild in the environment you intend to benchmark in.

## Correctness

Performance measurements are gated by a separate correctness suite.

The custom kernel is compared against the repository's explicit semantic reference (`benchmarks/reference.py`, contract in `benchmarks/SEMANTICS.md`) rather than against FlashInfer.

Current Gate A coverage includes:

- exact top-k candidate IDs across FP16 and BF16,
- deterministic lowest-token-id tie resolution,
- exact handling of arbitrarily large boundary tie sets,
- exact top-p `keep` mask,
- FP32 probability arithmetic,
- renormalized probabilities within a bounded ULP tolerance,
- sampled tokens constrained to the reference nucleus,
- distributional sampling checks over repeated draws,
- uniform sampling for equal logits,
- deterministic explicit seed/offset behavior,
- RNG advancement under CUDA Graph replay.

The validation set reported, at commit `df3e3c0` (`results/SPIKE.md` §4):

```text
top-p keep mask:
0 mismatches / 195,840 elements
across 54 configurations

renormalized probabilities:
maximum observed error = 15 ULP
test bound = 64 ULP

test suite:
319 passed
2 skipped
```

Each claim is falsified as well as asserted: inverting the index half of the packed key fails 32 tests, forcing the old clamped tie path fails 4, relaxing the strict `<` at the cut fails 8, and restoring a host-side RNG counter fails the graph-replay test.

The sampler's draw is an inline splitmix64 over `(seed, offset, row)`; the offset is taken from PyTorch's generator via `PhiloxCudaState`, so CUDA Graph replay advances the RNG rather than freezing it. `SEMANTICS.md` does not require matching `torch.multinomial` draw-for-draw.

Gate B — the separate FP32 semantic-fidelity comparison defined in `benchmarks/SEMANTICS.md` — has not yet been run for the fused kernel.

## Current Status / Limitations

### Implemented and validated

- custom FP16/BF16 SM120 top-k selection,
- deterministic tie handling,
- fused candidate merge + top-p + sampling,
- PyTorch extension bindings,
- CUDA Graph-safe RNG behavior,
- Gate A correctness suite,
- benchmark ladder and phase attribution.

### Measured

- **2.28×–4.09×** speedup over `flashinfer_from_probs` across the tested grid,
- **22.5 µs** anchor B=1 latency,
- **34.8 µs** anchor B=32 latency,
- kernel floor and split-count sweeps,
- internal phase attribution,
- hot/cold L2 sensitivity.

### Experimental / future work

- make the row-split rule depend on `k` as well as batch size — at B=1, k=20 the rule picks 8 splits (19.96 µs) where 20 splits measures 16.40 µs, 21.7% faster (`results/raw/kernel_splits.csv`),
- validate and measure the register-resident warp merge now on the `register-merge` branch, which replaces the shared-memory bitonic but has not been run,
- investigate single-pass register-resident selection to reduce the three vocabulary traversals,
- complete Gate B semantic fidelity.

### Measurement limitations

- GPU clocks could not be locked; every row records `clocks_locked=false`. Median round-to-round spread is 0.2–0.5% per rung, but the fused-kernel rung's worst case is 18%.
- Nsight Compute hardware counters are unavailable because profiling requires administrator permission (`RmProfilingAdminOnly: 1`), so all attribution is wall-clock.
- Timings are amortized across repeated calls and therefore understate dependent-call launch latency.
- Input logits are synthetic Gaussian samples rather than logits captured from real models; the tie-fidelity result in particular should be re-run against a real logits capture.
- FlashInfer has different sampling semantics and is used only as a performance comparison.
- The measured improvement is an operator-level result, not a material end-to-end LLM decode speedup.

## References

- **FlashInfer** — production sampling performance baseline.
- **Hugging Face Transformers** — Phase 1 sampling contract and eager baseline.
- **PyTorch CUDA Extensions** — Python/C++/CUDA integration.
- **CUDA Graphs** — launch-overhead comparison baseline.
- **CUDA warp shuffle and cooperative reduction primitives** — used for low-overhead reductions and scans.
- **PyTorch Philox generator state** — CUDA Graph-safe RNG offsets for the kernel's sampler.

Detailed methodology, historical measurements, rejected optimizations, and full raw benchmark grids are kept under `docs/` and `results/` rather than duplicated here.
