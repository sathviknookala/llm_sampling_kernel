# Prototype Spike — does a hand-written SM120 kernel beat the production sampler?

**Verdict: YES, and the project continues — but `DECISION.md` §6's "5–20× at low batch" is not
what this prototype delivers, and the gap is now measured rather than guessed.**

`results/DECISION.md` ended with one instruction: before building the full kernel, find out whether
a hand-written kernel can beat `flashinfer_from_probs` at 72.6 µs (B=1). It can. A first-pass fused
kernel runs the whole operation — `[B, V]` logits to one token id per sequence — in **20.6 µs at
B=1**, against **74.0 µs** for the best production sampler measured in the same process on the same
instrument.

All numbers below come from committed artifacts under `results/raw/`, on an idle GPU with
`clocks_locked=false`. Methodology: `docs/benchmark_methodology.md`.

---

## 1. The result

`results/raw/spike_ladder.csv` (1134 rows), `V=151936, k=50, p=0.90, bfloat16, hot`:

| B | `hf_eager` | `graph_compile` | `flashinfer_from_probs` | **`fused_kernel`** | vs the bar |
|---|---|---|---|---|---|
| 1 | 325.1 | 71.9 | 74.0 | **20.6** | **3.59×** |
| 4 | 598.8 | 95.1 | 74.5 | **20.8** | **3.59×** |
| 8 | 706.9 | 98.2 | 75.2 | **21.2** | **3.54×** |
| 16 | 1189.9 | 122.6 | 74.8 | **24.9** | **3.01×** |
| 32 | 2193.5 | 168.4 | 101.2 | **32.8** | **3.09×** |

Across the full parameter grid the win is **2.5×–4.3×**, widest at small `k` and low batch:

| | k=20 | k=50 | k=100 |
|---|---|---|---|
| B=1, V=151936 | 18.5 µs (**4.00×**) | 20.6 µs (3.59×) | 26.8 µs (2.75×) |
| B=32, V=151936 | 30.1 µs (3.41×) | 32.8 µs (3.09×) | 36.7 µs (2.81×) |

fp16 matches bf16 within noise. Cold L2 costs the kernel 1–33% (worst at B=32) and the win narrows
to 2.6×–3.6×. Round-to-round spread with rotated ordering is **0.4%**, against 2.2% for
`graph_compile` — the kernel is the most repeatable rung in the ladder.

**The comparison is deliberately unfavourable to this kernel.** `flashinfer_from_probs` is handed a
`[B, V]` probability tensor that someone else already softmaxed; the fused kernel consumes raw
logits and does the softmax itself. It still wins by 3.6×.

## 2. What it is not: the 5–20× in `DECISION.md` §6

§6 projected "single-digit to low tens of microseconds, i.e. a 5–20× operator win at low batch."
The prototype lands at the top of that latency range and the bottom of that ratio range. **The
claim should be restated as ~3–4× until a second kernel iteration says otherwise**, and the
headline should be the measured 3.59×, not the projection.

The reason is measured, not inferred. `results/raw/kernel_floor.csv` (135 rows) brackets what any
kernel can reach in this rig at `V=151936, bf16, hot`:

| probe | B=1 | B=8 | B=32 | what it bounds |
|---|---|---|---|---|
| `noop` — allocate `[B]`, launch, write | 2.57 | 2.57 | 2.57 | dispatch + launch floor |
| `scan_rowblock` — one full pass, one block/row | 10.23 | 10.23 | 10.23 | the simple grid shape |
| `scan_split` — one full pass, split grid | **4.56** | 6.38 | 10.38 | the split grid shape |
| `torch_argmax` — for calibration | 8.40 | 10.30 | 16.55 | |

So one full pass over the vocabulary costs **4.56 µs** at B=1 and the prototype costs 20.6 µs — it
is **4.5× above its own floor**, and that is where the missing speedup is. Three things account for
it, all structural rather than mysterious:

- **Three passes over the row.** Two 8-bit radix passes resolve the 16-bit key, a third gathers the
  candidates. A one-pass register-resident select (WarpSelect-style) would remove two of them.
- **~50 block-wide syncs in the selection kernel** plus a 55-stage bitonic merge. Replacing the
  merge's full sort with the same radix-select machinery would cut most of that.
- **Two launches** (~2.6 µs each), because the split shape needs a merge step.

The floor probe also settled a question that mattered more than expected: at B=1 the rig's dispatch
floor is 2.57 µs, not tens of microseconds. **The 74 µs bar is real device work, not instrument.**

## 3. Why the kernel is latency-bound, not bandwidth-bound

`results/raw/kernel_splits.csv` (255 rows), `k=20`, 8 splits: B=1 (8 blocks), B=4 (32 blocks) and
B=8 (64 blocks) all take **18.5 µs** — identical, despite 8× the total work. Each block scans the
same 18 992 elements either way, and adding blocks costs nothing until B=16. The kernel is bound by
its phase chain, not by throughput.

This is why more parallelism stops paying early. The same artifact sweeps how many slices the
vocabulary is cut into: the curve flattens by 4–8 splits and *rises* beyond that at B≥16, even
though the GPU has 70 SMs. The auto rule in `csrc/fused_sampling.cu` — `clamp(128/B, 4, 8)` — comes
from that artifact and is within ~1 µs of the measured optimum everywhere except `k=20, B=1`, where
20 splits would save a further 3 µs. That is leftover tuning headroom, not a design limit.

It is also the project's thesis showing up inside its own kernel: this operation is a latency
problem at every level.

## 4. Correctness

The kernel is **not Gate A / Gate B certified** — that is follow-on work. But the number above is
not being taken from a broken kernel:

- **Candidate selection is exact.** `topk_fused` reproduces `reference.py`'s `topk_ids` **exactly**
  in 40/40 configurations across `V ∈ {2000, 4001, 128256, 151936}`, `k ∈ {1, 20, 50, 100}`, fp16
  and bf16 — including bf16, where ~93% of rows at `k=50` have a k-boundary tie. The tie rule
  (lowest token id wins) is implemented, not approximated: it falls out of a single 64-bit compare
  on `(key << 32) | ~idx`.
- **The nucleus is right.** Every sampled token lands inside the reference's top-p cutoff, and the
  empirical support over 4000 draws equals the reference `cutoff` exactly.
- **The distribution is right.** Sampling frequencies match the renormalized reference distribution
  inside a 4-sigma binomial band over 20 000 draws; equal logits sample uniformly.
- 51 kernel tests, `tests/test_fused_kernel.py`. Suite total **266 passed, 2 skipped**.
- **The suite was falsified**: inverting the index half of the packed key — turning "lowest token id
  wins" into "highest wins" — fails 32 of the 51 tests.

### Known limitation, deliberate for the spike

Ties are collected into a per-split shared buffer of 2048. Beyond that the *retained values* are
still correct but which of several tied ids is emitted may differ from the reference. Measured tie
multiplicity is 4 (median) to 16 (max) per `results/raw/tie_fidelity.csv`, and the buffer is
per-split, so this needs >2048 tokens sharing the exact k-th value inside one slice. Pinned by
`test_tie_buffer_capacity_is_the_documented_spike_limit`. Gate A work must replace the clamp with a
radix pass on the index.

## 5. Design

One op, two kernels, `[B, V]` logits → `[B]` int64. `csrc/fused_sampling.cu`.

The specialization that pays: **fp16 and bf16 are both 16-bit sign-magnitude**, so a single
order-preserving map `key = b ^ (b & 0x8000 ? 0xFFFF : 0x8000)` makes descending-by-value identical
to descending-by-key for both dtypes, and selection needs **two 8-bit radix passes, not the four
fp32 would need**. Packing `(key, ~idx)` into 64 bits makes the tie rule a single comparison and
makes every packed value unique.

1. `topk_partial_kernel`, grid `(splits, B)` — high-byte histogram → low-byte histogram within the
   boundary bucket → exact threshold → gather, with the k-boundary tie resolved on index. Emits the
   exact top-K of its slice.
2. `merge_sample_kernel`, grid `B` — bitonic-sorts the `splits × K` candidates, decodes values back
   out of the keys (no second read of the logits), fp32 softmax over the survivors, top-p cut on the
   exclusive prefix, and an inverse-CDF draw against `u · Z_p`.

No normalized probability tensor is materialized: scaling `u` by the retained mass replaces
renormalization. The RNG is an inline splitmix64 on `(seed, offset, row)` — `SEMANTICS.md`
explicitly frees the kernel from matching `torch.multinomial` draw-for-draw, so no uniform tensor is
passed in and no extra launch is paid.

## 6. What this does not claim

- **Not an end-to-end decode win.** Sampling is 0.16–1.2% of a decode step
  (`results/raw/amdahl_probe.csv`). At 3.6×, the end-to-end saving is 0.12–0.86% for Qwen2-0.5B and
  ~0.1% for Mistral-7B. This remains an operator-specialization result, exactly as
  `DECISION.md` §9 required.
- **Not Gate A or Gate B certified.** Selection is exact and the distribution is verified, but
  `keep`/`renormed` are not asserted elementwise against the reference, and the fp32 fidelity gate
  has not been run on the kernel.
- **Not measured with hardware counters.** `ncu` is confirmed unavailable to this user:
  `/proc/driver/nvidia/params` reports `RmProfilingAdminOnly: 1`. Every attribution above is
  wall-clock, from isolated timing and the floor probes.
- **Amortized throughput timing.** Consecutive iterations overlap, so these numbers understate
  per-call launch latency in a real dependent decode loop. Two launches per call means the kernel
  benefits from that overlap somewhat more than a single-launch rung would.
- **FlashInfer's semantics differ** — it applies top-k/top-p to the full-vocabulary distribution
  where this repo renormalizes within the top-k survivors first. It is a performance rung, never a
  correctness target.
- **Clocks are not locked** (no permission). Round-to-round spread is 0.4% for this rung.

## 7. Next

1. Replace the tie clamp with a radix pass on the index, then run Gate A elementwise.
2. Collapse three row passes into one (register-resident warp select) and the bitonic merge into a
   radix select — the floor says ~4× remains.
3. Re-measure and decide whether §6's 5–20× is reachable or should be retired.

## Reproduction

```bash
export PATH=~/.venv_flashinfer/bin:/home/sathvik/cuda-12.9/bin:$PATH
V=~/.venv_flashinfer/bin/python
$V setup.py build_ext --inplace
$V -m pytest tests/ -q
$V -m benchmarks.kernel_floor                       # kernel_floor.csv + kernel_splits.csv
$V -m benchmarks.benchmark_sampling --rounds 3 --reps 5 \
    --out results/raw/spike_ladder.csv --env-out results/raw/environment_spike.json
$V -m benchmarks.summarize --raw results/raw/spike_ladder.csv --out results/summary_spike.md
```

The extension is built and benchmarked in `~/.venv_flashinfer` (torch 2.9.1+cu128), the only
environment where the kernel and FlashInfer can be timed in one process. `results/raw/environment_spike.json`
records the `.so` path and mtime that produced these numbers.
