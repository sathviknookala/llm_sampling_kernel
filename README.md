# Fused Top-K / Top-P CUDA Sampling Kernel

A custom CUDA implementation of decode-time **top-k + top-p sampling** for large-vocabulary LLMs. The kernel specializes the `[B, V] logits → [B] token ids` path for low decode batches and 128K–152K vocabularies, targeting the fixed-cost selection, sorting, and launch overhead left by general-purpose sampling implementations.

On an **NVIDIA RTX PRO 4000 Blackwell (SM120)**, the current implementation runs **2.50×–5.12× faster than FlashInfer's `top_k_top_p_sampling_from_probs`** across the measured parameter grid. At the anchor configuration (`V=151936`, `k=50`, `p=0.90`, BF16, hot L2), latency falls from **72.8 µs to 19.1 µs at B=1 — 3.81×**, in **2 kernel launches** per sampling call (FlashInfer uses 9). Latencies are amortized over back-to-back calls; see *Timing*.

Every number in this README comes from a committed artifact under `results/`, named at the point it is used. Current performance measurements are of the register-merge kernel (`*_regmerge` artifacts); the shared-memory bitonic kernel they replaced (commit `df3e3c0`) is kept alongside as the baseline.

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

The original Hugging Face eager path performs much of this work over the full vocabulary and issues **64–70 device operations per sampling call** — 47–51 kernels plus memcpy/memset (`results/raw/launch_counts.csv`, `results/raw/kernel_trace.csv`). A tighter PyTorch implementation and CUDA Graphs remove much of that overhead before any custom CUDA is written.

For this reason, the primary performance baseline is not Hugging Face eager. It is:

```text
FlashInfer 0.6.17
top_k_top_p_sampling_from_probs
```

The primary metric is **amortized device latency per sampling call in microseconds**.

The current result is an **operator-level optimization**. Sampling accounts for only approximately 0.16–1.2% of the measured end-to-end decode step at B=1 in the model probes in this repository (`results/raw/amdahl_probe.csv`), so the project does not claim a comparable end-to-end decode speedup.

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

The merge is a register-resident warp sort (`fs::warp_merge_sort`): warp 0 of a 128-thread block holds all `splits × K` packed candidates in registers and bitonic-sorts them with `__shfl_xor_sync`, with **zero barriers inside the sort** (4 in the whole kernel, down from 49) and **zero spill at every width**. It replaced a shared-memory bitonic and cut the sort phase 21–74% across the 15 measured (B, k) cells (`results/raw/kernel_phase_breakdown_regmerge.csv` against `kernel_phase_breakdown.csv`; `results/SPIKE.md` §8.2).

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
- Moving the merge into one warp's registers made the whole kernel 7.5–22.2% faster at every measured hot cell. The three row traversals are now the largest cost at B=32 (72%) and at B=16 for k ≤ 50; at B ≤ 8 with k ≥ 50 the merge kernel still leads.

Register-resident *selection* — collapsing the three row passes into one — remains the next optimization direction; it is not included in the measured result.

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

Captured per run in `results/raw/environment_regmerge.json` (and `environment_spike.json` for the bitonic baseline). The FlashInfer comparison is run in the same Python process and software environment as the custom kernel.

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

**custom fused kernel: 2.50×–5.12× faster than `flashinfer_from_probs`**

The top of that range is the noisiest cell in the grid (B=1, V=128256, k=20, whose three rounds sit one ~2.05 µs timing quantum apart); the anchor below is the number to quote.

At the anchor configuration:

```text
V = 151936
top_k = 50
top_p = 0.90
dtype = BF16
residency = hot
```

| Batch | FlashInfer from probs | Fused kernel | Speedup | Bitonic merge (`df3e3c0`) |
|---:|---:|---:|---:|---:|
| 1 | 72.8 µs | **19.1 µs** | **3.81×** | 22.5 µs (3.35×) |
| 4 | 73.8 µs | **19.7 µs** | **3.74×** | 22.5 µs (3.35×) |
| 8 | 74.3 µs | **20.3 µs** | **3.66×** | 22.6 µs (3.35×) |
| 16 | 74.3 µs | **24.6 µs** | **3.02×** | 26.6 µs (2.83×) |
| 32 | 101.2 µs | **30.7 µs** | **3.29×** | 34.8 µs (2.91×) |

Each speedup divides by FlashInfer measured in the same sweep; FlashInfer itself read 1.5% faster (median) in the register-merge sweep than in the bitonic one.

The low-batch latency remains almost flat through B=8 despite an 8× increase in total input work, consistent with the measured latency-bound behavior of this regime.

### Top-k sensitivity

For `V=151936`, BF16, hot L2:

| Configuration | Fused kernel | Speedup vs FlashInfer |
|---|---:|---:|
| B=1, k=20 | **16.3 µs** | **4.49×** |
| B=1, k=50 | **19.1 µs** | **3.81×** |
| B=1, k=100 | **24.1 µs** | **3.04×** |
| B=32, k=20 | **27.0 µs** | **3.81×** |
| B=32, k=50 | **30.7 µs** | **3.29×** |
| B=32, k=100 | **34.9 µs** | **2.95×** |

FP16 and BF16 were measured against each other at `V=151936`, `k=50`, `p=0.90`: they agree to within 0.7–1.2% at hot B ∈ {1, 4, 8, 32}, and the widest divergence across those ten cells is 9.8% (B=1, cold L2), inside this rung's round-to-round spread.

Cold-L2 measurements reduce the advantage, with measured speedups of **2.70×–3.95×**.

Full measurements are stored in `results/raw/spike_ladder_regmerge.csv` (1134 rows); the bitonic baseline is `results/raw/spike_ladder.csv`.

### Launches and kernel resources

`results/raw/kernel_trace.csv` — a CUPTI activity trace via `torch.profiler`, bf16 anchor, 50 steps:

| Path | B=1 | B=32 |
|---|---:|---:|
| **fused kernel** | **2 kernels**, 0 memcpy/memset, 17.8 µs device | **2 kernels**, 0 memcpy/memset, 28.0 µs device |
| `flashinfer_from_probs` | 9 kernels, 36.4 µs device | 9 kernels, 95.8 µs device |
| Hugging Face eager | 47 kernels + 17 memcpy/memset, 242.3 µs device | 51 kernels + 19 memcpy/memset, 2001.1 µs device |

Device time is the sum of kernel durations, so it excludes launch gaps and sits below the ladder latency. At B=1 the fused path's 17.8 µs splits 9.5 µs `topk_partial_kernel` / 8.4 µs `merge_sample_kernel`; at B=32 it is 21.6 / 6.4 µs.

`results/raw/kernel_attrs.csv` — `cudaFuncGetAttributes` and the CUDA occupancy API at the shipped launch configuration (the figure Nsight Compute reports as *theoretical occupancy*):

| Kernel | Block | Registers | Spill (local) | Static shared | Blocks/SM | Theoretical occupancy |
|---|---:|---:|---:|---:|---:|---:|
| `topk_partial_kernel` | 512 | 38 | **0 B** | 18.0 KB | 3 | **100%** |
| `merge_sample_kernel`, P=128 (B=32, k=20) | 128 | 48 | **0 B** | 2.0 KB | 10 | 83.3% |
| `merge_sample_kernel`, P=256 (k=20 at B ≤ 16; B=32, k=50) | 128 | 88 | **0 B** | 3.0 KB | 5 | 41.7% |
| `merge_sample_kernel`, P=512 (k=50 at B ≤ 16; B=32, k=100) | 128 | 130 | **0 B** | 5.0 KB | 3 | 25.0% |
| `merge_sample_kernel`, P=1024 (k=100, B ≤ 16) | 128 | 142 | **0 B** | 9.0 KB | 3 | 25.0% |

All 13 instantiations spill nothing — the 9 the measured grid dispatches, plus P=32 and P=64. The merge is register-limited by design — the candidates *are* the registers — and at B ≤ 16 its grid is B blocks on a 70-SM GPU, so no SM hosts a second block and its occupancy does not bind.

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

Phase ablation at B=1, `V=151936`, `k=50` (`results/raw/kernel_phase_breakdown_regmerge.csv`, derived from 9 rounds in `results/raw/kernel_phases_regmerge.csv`) attributes the 18.5 µs total approximately as, with the bitonic kernel alongside. The ablation is its own instrument and reads 0.6 µs below the ladder's 19.1 µs; its `full` control (`sample_fused` timed alongside) agrees with the last phase.

| Component | Latency | Share | Bitonic (`df3e3c0`) |
|---|---:|---:|---:|
| dispatch floor | 2.54 µs | 14% | 2.51 µs |
| high-byte histogram | 1.61 µs | 9% | 1.62 µs |
| high-byte search | 0.06 µs | <1% | 0.06 µs |
| low-byte histogram | 2.12 µs | 11% | 2.27 µs |
| threshold search | 0.28 µs | 2% | 0.15 µs |
| gather + tie resolution | 3.67 µs | 20% | 3.68 µs |
| candidate merge sort | **6.10 µs** | **33%** | 8.12 µs |
| softmax + top-p + draw | 2.09 µs | 11% | 4.09 µs |
| **Total** | **18.46 µs** | | 22.51 µs |

The merge kernel (last two rows together) fell from 12.21 to 8.19 µs; the partial kernel is unchanged, as it should be. At B=32 the three row traversals are now **72%** of the kernel, so register-resident selection is the next target there; at B=1 the merge kernel (44%) and the traversals (40%) are roughly even. For scale, `results/raw/kernel_floor.csv` puts one full pass over the vocabulary at **4.49 µs** at B=1, so the kernel is 4.3× above its own floor.

Because `ncu` is unavailable on this machine, this attribution is a wall-clock phase ablation — phases 1..n of the real kernel are run and the kernel then stops — not a hardware-counter profile. Individual sub-2 µs phase deltas at B ≤ 8 are not resolvable; a ~2.05 µs timing quantum migrates between adjacent phases across rounds — the softmax-tail row above moved by exactly one quantum, so only the merge kernel's total is resolvable at B=1.

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
│   ├── kernel_trace.py         # CUPTI per-kernel trace: launches + device time
│   ├── kernel_attrs.py         # registers, spills, shared memory, occupancy
│   ├── sanitize.py             # compute-sanitizer memcheck/racecheck/initcheck
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
│   ├── summary_regmerge.md     # summarized fused-kernel ladder (register merge)
│   ├── summary_spike.md        # summarized fused-kernel ladder (bitonic baseline)
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
$V -m benchmarks.kernel_phases --rounds 9 \
    --out results/raw/kernel_phases_regmerge.csv \
    --breakdown-out results/raw/kernel_phase_breakdown_regmerge.csv
```

Run the primary benchmark:

```bash
$V -m benchmarks.benchmark_sampling \
    --rounds 3 \
    --reps 5 \
    --out results/raw/spike_ladder_regmerge.csv \
    --env-out results/raw/environment_regmerge.json
```

Generate the summary:

```bash
$V -m benchmarks.summarize \
    --raw results/raw/spike_ladder_regmerge.csv \
    --out results/summary_regmerge.md
```

Trace launches, read kernel resources, and run the sanitizers:

```bash
$V -m benchmarks.kernel_trace
$V -m benchmarks.kernel_attrs
$V -m benchmarks.sanitize
```

Primary outputs:

```text
results/raw/spike_ladder_regmerge.csv
results/raw/environment_regmerge.json
results/raw/kernel_phases_regmerge.csv
results/raw/kernel_phase_breakdown_regmerge.csv
results/raw/kernel_trace.csv
results/raw/kernel_attrs.csv
results/raw/sanitizer.csv
results/raw/kernel_floor.csv
results/raw/kernel_splits.csv
results/summary_regmerge.md
```

The bitonic baseline (`spike_ladder.csv`, `kernel_phases.csv`, `kernel_phase_breakdown.csv`, `environment_spike.json`, `summary_spike.md`) came from the same commands with their default or `_spike` output names at `df3e3c0`.

Artifacts record `git_commit` as `6ddf1e0-dirty` (register merge) and `df3e3c0-dirty` (bitonic): in both cases the dirty state is uncommitted measurement scripts and, for the register merge, the additive `kernel_attrs()` debug binding — no kernel source differs from the named commit.

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

The validation set reported (`results/SPIKE.md` §4), on the bitonic kernel at `df3e3c0` and again, unchanged, on the register merge:

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

compute-sanitizer, 104 kernel tests (results/raw/sanitizer.csv):
memcheck   0 errors
racecheck  0 hazards
initcheck  0 errors
```

initcheck runs unfiltered: restricting it to this repo's kernels with `--kernel-name` stops it tracking writes by PyTorch's kernels, so every input tensor reads as uninitialized. A deliberately uninitialized `torch.empty` input confirms it still catches a real uninitialized read in `topk_partial_kernel` (the `initcheck_positive_control` row). The test-suite log is `results/raw/pytest_regmerge.txt`.

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
- sanitizer-clean under memcheck, racecheck and initcheck,
- benchmark ladder and phase attribution.

### Measured

- **2.50×–5.12×** speedup over `flashinfer_from_probs` across the tested grid (**3.81×** at the anchor),
- **19.1 µs** anchor B=1 latency (22.5 µs with the bitonic merge),
- **30.7 µs** anchor B=32 latency (34.8 µs),
- **2 kernel launches** per sampling call, against 9 for FlashInfer and 64–70 device operations for Hugging Face eager,
- **0 bytes of spill** in every production kernel instantiation; 100% theoretical occupancy for the selection kernel,
- kernel floor and split-count sweeps,
- internal phase attribution,
- hot/cold L2 sensitivity.

### Experimental / future work

- make the row-split rule depend on `k` as well as batch size — on the bitonic kernel, at B=1, k=20 the rule picked 8 splits (19.96 µs) where 20 splits measured 16.40 µs, 21.7% faster (`results/raw/kernel_splits.csv`); the register merge changed the merge cost that trade balances against, so it must be re-measured,
- investigate single-pass register-resident selection to reduce the three vocabulary traversals, now 40% of the kernel at B=1 and 72% at B=32,
- complete Gate B semantic fidelity.

### Measurement limitations

- GPU clocks could not be locked; every row records `clocks_locked=false`. The fused-kernel rung's round-to-round spread, (max − min) / median, is 1.3% median but 25.7% worst case, against 1.0% worst case for `flashinfer_from_probs` (`results/raw/spike_ladder_regmerge.csv`; 18.0% worst case on the bitonic sweep).
- Nsight Compute hardware counters are unavailable because profiling requires administrator permission (`RmProfilingAdminOnly: 1`), so all attribution is wall-clock. The unprivileged substitutes used instead are a CUPTI activity trace (launch counts, per-kernel device time), the CUDA occupancy API (theoretical occupancy), and `compute-sanitizer`; achieved occupancy and stall reasons remain unmeasured.
- Timings are amortized across repeated calls and therefore understate dependent-call launch latency. The fused kernel issues 2 launches per call and `flashinfer_from_probs` 9; the net bias between them is not measured.
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
