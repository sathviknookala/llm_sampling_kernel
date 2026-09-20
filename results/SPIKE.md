# Prototype Spike — does a hand-written SM120 kernel beat the production sampler?

**Verdict: YES, and the project continues — but `DECISION.md` §6's "5–20× at low batch" is not
what this prototype delivers, and the gap is now measured rather than guessed.**

`results/DECISION.md` ended with one instruction: before building the full kernel, find out whether
a hand-written kernel can beat `flashinfer_from_probs` at 72.6 µs (B=1). It can. A first-pass fused
kernel runs the whole operation — `[B, V]` logits to one token id per sequence — in **22.5 µs at
B=1**, against **75.4 µs** for the best production sampler measured in the same process on the same
instrument.

All numbers below come from committed artifacts under `results/raw/`, on an idle GPU with
`clocks_locked=false`. Methodology: `docs/benchmark_methodology.md`.

---

## 1. The result

`results/raw/spike_ladder.csv` (1134 rows), `V=151936, k=50, p=0.90, bfloat16, hot`:

| B | `hf_eager` | `graph_compile` | `flashinfer_from_probs` | **`fused_kernel`** | vs the bar |
|---|---|---|---|---|---|
| 1 | 325.2 | 72.4 | 75.4 | **22.5** | **3.35×** |
| 4 | 598.7 | 95.2 | 75.4 | **22.5** | **3.35×** |
| 8 | 707.0 | 98.3 | 75.5 | **22.6** | **3.35×** |
| 16 | 1191.1 | 123.1 | 75.4 | **26.6** | **2.83×** |
| 32 | 2195.0 | 168.8 | 101.2 | **34.8** | **2.91×** |

Across the full parameter grid the win is **2.28×–4.09×**, widest at small `k` and low batch:

| | k=20 | k=50 | k=100 |
|---|---|---|---|
| B=1, V=151936 | 20.3 µs (**3.69×**) | 22.5 µs (3.35×) | 30.3 µs (2.48×) |
| B=32, V=151936 | 30.7 µs (3.35×) | 34.8 µs (2.91×) | 38.3 µs (2.69×) |

fp16 matches bf16 within noise. Cold L2 narrows the win to 2.53×–3.37×.

**These numbers are slower than the first spike measured, deliberately.** At commit `33802d7` the
kernel was 20.6 µs at B=1 / k=50 (3.59×) and 25.5 µs at k=100. Closing Gate A cost that: the cut
now normalizes before taking the prefix, exactly as `reference.py` does, which is K per-element
divisions the old `cum[i-1] >= top_p * z` form did not pay. The trade is **~9% at k=50 and ~19% at
k=100 in exchange for an exact `keep` mask and a tie path with no capacity clamp**. It is recorded
here rather than quietly absorbed, because the earlier 3.59× is in git history and will not
reproduce.

**The comparison is deliberately unfavourable to this kernel.** `flashinfer_from_probs` is handed a
`[B, V]` probability tensor that someone else already softmaxed; the fused kernel consumes raw
logits and does the softmax itself. It still wins by 3.6×.

## 2. What it is not: the 5–20× in `DECISION.md` §6

§6 projected "single-digit to low tens of microseconds, i.e. a 5–20× operator win at low batch."
The prototype lands at the top of that latency range and the bottom of that ratio range. **The
claim should be restated as ~3–4× until a second kernel iteration says otherwise**, and the
headline should be the measured 3.35×, not the projection.

The reason is measured, not inferred. `results/raw/kernel_floor.csv` (135 rows) brackets what any
kernel can reach in this rig at `V=151936, bf16, hot`:

| probe | B=1 | B=8 | B=32 | what it bounds |
|---|---|---|---|---|
| `noop` — allocate `[B]`, launch, write | 2.50 | 2.52 | 2.51 | dispatch + launch floor |
| `scan_rowblock` — one full pass, one block/row | 10.23 | 10.23 | 10.23 | the simple grid shape |
| `scan_split` — one full pass, best split | **4.49** | 6.18 | 10.38 | the split grid shape |
| `torch_argmax` — for calibration | 8.41 | 10.30 | 15.69 | |

So one full pass over the vocabulary costs **4.49 µs** at B=1 and the kernel costs 22.5 µs — it is
**5.0× above its own floor**, and that is where the missing speedup is.

The floor probe also settled a question that mattered more than expected: at B=1 the rig's dispatch
floor is 2.50 µs, not tens of microseconds. **The 74 µs bar is real device work, not instrument.**

### 2.1 Where the 22.5 µs actually goes — measured, superseding an earlier guess

An earlier revision of this section attributed the gap to three row passes, ~50 block syncs and the
bitonic merge, in roughly equal parts. **That was inference and it was wrong about the balance.**
`results/raw/kernel_phases.csv` (1215 rows, 9 rounds) runs phases 1..n of the real kernel and
stops, so each phase is a difference of two timings on the ladder's own instrument — the only
attribution available with `ncu` blocked. `results/raw/kernel_phase_breakdown.csv` holds the
derived table. At `V=151936, k=50, bf16, hot`:

| phase | B=1 | B=32 | |
|---|---|---|---|
| dispatch floor (`noop`) | 2.51 (11%) | 2.52 (7%) | launch + allocate |
| 1 high-byte histogram | 1.62 (7%) | 5.68 (16%) | traversal |
| 2 fold, suffix scan, bucket search | 0.06 (0%) | 0.14 (0%) | |
| 3 low-byte histogram | 2.27 (10%) | 8.04 (23%) | traversal |
| 4 fold, suffix scan, exact threshold | 0.15 (1%) | 0.01 (0%) | |
| 5 gather, tie resolve, store | 3.68 (16%) | 8.20 (24%) | traversal |
| 6 merge bitonic sort | **8.12 (36%)** | 5.21 (15%) | second launch |
| 7 softmax, cumsum, cut, draw | 4.09 (18%) | 5.01 (14%) | |
| **total** | **22.51** | **34.81** | |

Three things this changes:

- **At low batch the merge kernel is the largest single cost** — 12.2 µs of 22.5 at B=1, of which
  the bitonic sort is 36%. It launched `<<<B, 1024>>>`, so at B=1 it is one block on a 70-SM GPU.
  The three row passes together are 34%.
- **The mechanism is confirmed, not assumed.** Merge cost tracks the bitonic *stage count* across
  `k`: 6.13 / 8.12 / 11.92 µs for 36 / 45 / 55 stages at `k=20/50/100`, i.e. 0.170 / 0.180 / 0.217
  µs per stage. A sort whose cost is linear in its stage count is a sort, not a memory effect.
- **The two boundary searches are free** — 0.21 µs combined at B=1. The warp `suffix_sum_256` that
  collapsed 16 block syncs into 1 already took that phase off the table.
- **Phase 7 doubled, from 2.06 to 4.09 µs, and that is Gate A.** The per-element normalization the
  exact cut requires is visible directly in the ablation, which is how its cost was attributed.

At B=32 the balance inverts: the three traversals are 21.9 µs (63%) and the merge is 10.2 µs (29%).

**Resolution caveat.** Clocks cannot be locked on this machine. Within a round the rep spread is
0.0–0.1%, but at B≤8 the *cumulative* phase timings land on near-exact multiples of ~2.05 µs
(2, 2, 3, 3, 5, 9, 10 × 2.053 at B=1), so a ~2 µs quantum migrates between adjacent phases from
round to round. Individual sub-2 µs phase deltas at B≤8 are therefore not resolvable; their groups
are — phases 3–5 together cost 6.08 / 6.04 / 6.03 µs at B=1/4/8, stable to 0.8%. The quantum is
reproducible across 9 rounds and is **unexplained**; it does not affect the merge and traversal
numbers above, which are far larger than it. The `full` control column (`sample_fused` timed
alongside phase 7) agrees with phase 7 to within 0.04–0.39%, so the ablation is measuring the
production path and not a divergent copy — `tests/test_fused_kernel.py` pins that equality.

## 3. Why the kernel is latency-bound, not bandwidth-bound

`results/raw/kernel_splits.csv` (255 rows), `k=20`, 8 splits: B=1, B=4 and B=8 cost **19.96 /
20.48 / 20.48 µs** — within 2.6% of each other despite 8× the total work. Each block scans the
same ~19K elements either way, and adding rows costs almost nothing until B=16. The kernel is
bound by its phase chain, not by throughput.

**The auto split rule is now leaving 18% on the table at small `k`.** `make_plan` uses
`clamp(128/B, 4, 8)`, which was right when it was fitted. Re-measured on the current kernel:

| | best splits | at best | auto | auto cost |
|---|---|---|---|---|
| B=1, k=20 | **20** | 16.40 µs | 8 | 19.96 µs (+21.7%) |
| B=8, k=20 | **16** | 18.56 µs | 8 | 20.48 µs (+10.3%) |
| B=1, k=50 | 8 | 22.53 µs | 8 | — |
| B=1, k=100 | 8 | 30.48 µs | 8 | — |
| B=32, any k | 4 | — | 4 | — |

The rule is still right at `k=50` and `k=100` and at B=32; it is only wrong at `k=20`, where the
merge is small enough that more splits keep paying. A rule that depends on `k` as well as `B` is
the cheapest win currently on the table, and it is measurement, not redesign. It is deliberately
**not** applied in this revision: it would invalidate the ladder measured above, and the ladder
and the split sweep have to come from the same binary.

## 4. Correctness

**Gate A is closed. Gate B is not.**

- **Candidate selection is exact.** `topk_fused` reproduces `reference.py`'s `topk_ids` **exactly**
  across `V ∈ {2000, 4001, 128256, 151936}`, `k ∈ {1, 20, 50, 100}`, fp16 and bf16 — including
  bf16, where ~93% of rows at `k=50` have a k-boundary tie. The tie rule (lowest token id wins) is
  implemented, not approximated: it falls out of a single 64-bit compare on `(key << 32) | ~idx`.
- **Ties are exact at any multiplicity.** The old per-split buffer of 2048 clamped the answer; past
  it, retained values stayed right but the emitted tied id could differ. It now falls back to
  index bucketing plus a bitmap whose capacity is a proof, not a heuristic. Pinned on all-equal
  rows at four vocab/split combinations that each provably overflow the buffer.
- **`keep` is exact.** 0 mismatches over 195 840 elements across 54 configurations (both dtypes,
  `V ∈ {4000, 128256, 151936}`, `k ∈ {20, 50, 100}`, `top_p ∈ {0.9, 0.95, 1.0}`). The kernel cuts
  on the same expression `reference.py` uses — `(c_i - p_i) < top_p` over already-normalized
  probabilities — because `a/z < p` and `a < p*z` are not the same comparison in fp32.
- **`renormed` agrees to 15 ulp** (max abs 5.4e-07), not bitwise. It cannot be bitwise: torch sums
  the prefix with a scan, the kernel sums it serially. The test bound is 64 ulp.
- **The nucleus and the distribution are right.** Every sampled token lands inside the reference's
  cutoff; frequencies match the renormalized reference inside a 4-sigma binomial band over 20 000
  draws; equal logits sample uniformly.
- **Graph replay advances the RNG.** The offset comes from torch's generator via `PhiloxCudaState`,
  not a host counter that capture would freeze.
- **319 passed, 2 skipped.** Every claim above is falsified, not just asserted: inverting the index
  half of the packed key fails 32 tests; forcing the old clamped tie path fails 4; relaxing the
  strict `<` at the cut fails 8; restoring the host RNG counter fails the replay test.

One test here was worth recording as a near-miss. The general `keep` comparison passed unchanged
when the cut was perturbed by 1e-7 relative, because Gaussian logits never put the exclusive prefix
near `top_p` — it passed where the bug could not occur. It took a row of eight equal logits, where
every prefix is exactly `i/8` in fp32 and the cut lands *on* `top_p`, to make it discriminate.

## 5. Design

One op, two kernels, `[B, V]` logits → `[B]` int64. `csrc/fused_sampling.cu`.

The specialization that pays: **fp16 and bf16 are both 16-bit sign-magnitude**, so a single
order-preserving map `key = b ^ (b & 0x8000 ? 0xFFFF : 0x8000)` makes descending-by-value identical
to descending-by-key for both dtypes, and selection needs **two 8-bit radix passes, not the four
fp32 would need**. Packing `(key, ~idx)` into 64 bits makes the tie rule a single comparison and
makes every packed value unique.

1. `topk_partial_kernel`, grid `(splits, B)` — high-byte histogram → low-byte histogram within the
   boundary bucket → exact threshold → gather, with the k-boundary tie resolved on index. Emits the
   exact top-K of its slice. Tie sets too large for the shared buffer fall back to a second index
   bucketing plus a bitmap over the boundary bucket, which cannot overflow because the bucket is
   `2^shift` wide — so there is no capacity clamp on the answer.
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
- **Gate A is closed; Gate B is not.** `topk_ids` and `keep` are exact against `reference.py` --
  `keep` at 0 mismatches over 195 840 elements across 54 configurations -- and ties are now
  resolved exactly with no capacity clamp. `renormed` agrees to **15 ulp**, not bitwise, and cannot
  be bitwise: torch sums the prefix with a scan and the kernel sums it serially. The fp32 fidelity
  gate (Gate B) still has not been run on the kernel.
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

## 7. Measured non-results

Three changes were implemented, measured, and reverted. They are recorded so the next session does
not spend the same time rediscovering them.

- **Pair-indexing the bitonic stages.** `bitonic_ascending` guards with `if (ixj > i)`, so half the
  threads idle in each of the 36–55 stages. Indexing the pair instead of the element removes the
  predicate. Measured **neutral** — the before/after distributions overlapped at every batch. The
  idle lanes cost nothing because the block has spare warp slots, and halving the active warps
  loses as much memory-level parallelism as the predicate wastes.
- **Sorted runs per split, resuming the merge at `k = 2R`.** If each split emits its candidates
  sorted with alternating direction, the merge can skip every bitonic level below the run length.
  Merge stages fell 18–52% and total time did not move (except −3 to −7% at k=100). The stages
  *relocate* into the split kernel rather than disappearing, and on a latency-bound dependent chain
  21 stages cost 21 stages wherever they run. Worth revisiting only together with a merge that has
  no barriers at all.
- **A K-conditional normalization path.** Built on a three-round reading that the parallel divide
  was 2 µs worse at k=20. Nine rounds showed the opposite; the conditional was worse than either
  branch. See the resolution caveat above — at B=1 the ~2.05 µs quantum is larger than most of what
  was being compared.

`MERGE_BLOCK` is **1024** at the commit these numbers were measured on. 512 measured −7% at B=32
against the pre-Gate-A binary (`643d369`); re-measured after the Gate A work it was within noise of
1024 at every `k`, so `df3e3c0` put it back and every artifact here carries 1024. An earlier
revision of this line said 512, describing a value the tree no longer held. 128 and 256 were
clearly worse at B=1 — but that was measured when the block itself ran the bitonic sort, which is
no longer true (§8.2), so it does not transfer to the register merge.

## 8. Next

1. **Refit the split rule on `k`, not just `B`.** Measured 21.7% at B=1/k=20 and 10.3% at
   B=8/k=20 (§3). Cheapest item here by a wide margin.
2. **A barrier-free merge — implemented, NOT YET MEASURED.** The merge is the largest single cost
   at low batch and its price is 0.18 µs per bitonic stage — barrier and shared-memory round-trip,
   not arithmetic. `fs::warp_merge_sort` now holds the candidates in one warp's registers and does
   the whole sort with `__shfl_xor_sync` and zero `__syncthreads`. **Every number in this document
   predates it and describes the shared-memory bitonic**; the register merge has not been built,
   tested or timed, and nothing here may be quoted for it. Two departures from the sketch above,
   both deliberate: the layout is **blocked, not cyclic** — bitonic's inner loop runs `j = k/2 … 1`,
   so small `j` is the common case, and blocked makes exactly those stages register-local (15
   shuffle stages at P=512 against cyclic's 35); and `MERGE_BLOCK` drops 1024 → 128, because a
   1024-thread block caps ptxas at 64 registers per thread and the P=1024 candidate array alone
   wants 64. That block-size change is confounded with the merge change and has to be reported as
   one result, not two.

   What **is** measured is the build, which needs no GPU. `ptxas -v` at `sm_120`, production merge
   (`STOP=2`, bf16), against the same flags on the pre-change source:

   | | baseline (shared bitonic) | P=128 | P=256 | P=512 (k=50) | P=1024 (k=100) |
   |---|---|---|---|---|---|
   | registers | 30 | 48 | 88 | 130 | 142 |
   | spill stores / loads | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | **0 / 0** |
   | shared bytes | 9220 | 2052 | 3076 | 5124 | 9220 |
   | `__syncthreads` executed | 49 (P=512) | 4 | 4 | 4 | 4 |
   | SASS | 27 KB | 73 KB | 123 KB | 236 KB | **451 KB** |

   - **No spill at any width**, which was the design's main failure mode: every register index is a
     template parameter, so the candidate array cannot lower to local memory. Confirmed, not hoped.
   - **The block-size change was forced, and now has a number on it.** 130 registers × a
     1024-thread block is 133K registers against the SM's 65 536, so the old block could not have
     launched this kernel at all; ptxas would have capped it at 64 and spilled.
   - **Barriers collapse from 49 to 4, independent of `P`.** The baseline's 5 static `BAR.SYNC` sit
     inside the runtime `(k, j)` loops and execute once per bitonic stage — 45 at P=512 plus 4
     around them. The register sort's stages are unrolled and barrier-free, so the only 4 left are
     the one that publishes `buf` and the three in the softmax tail.
   - **The shuffle count confirms the layout choice exactly.** SASS shows 4 `SHFL` per 64-bit
     cross-lane exchange and 15 cross-lane stages at *every* `P` — with 32 lanes the lane-crossing
     levels are always the top five. Cyclic would make it 35 stages at P=512, 2.3× the shuffles.
   - **The risk this trades into is instruction footprint**: 236 KB of straight-line SASS at k=50
     and 451 KB at k=100, 9–17× the baseline, far past any L1 instruction cache. It may not matter
     — the stream is branch-free and perfectly sequential, which is the best case for a prefetcher
     — but if the merge does not get faster, this is the first thing to suspect, and the fix is
     targeted: only the *local* stages need compile-time slot indices, so the 15 cross-lane stages
     can be re-rolled into a runtime loop and cut the code size roughly in half.
3. **Run Gate B** — the fp32 fidelity gate, the one gate still not run on the kernel.
4. Collapse the three row passes into one register-resident warp select — 34% at B=1, 63% at B=32.
5. Re-measure and decide whether `DECISION.md` §6's 5–20× is reachable or should be retired.

## Reproduction

```bash
export PATH=~/.venv_flashinfer/bin:/home/sathvik/cuda-12.9/bin:$PATH
V=~/.venv_flashinfer/bin/python
$V setup.py build_ext --inplace
$V -m pytest tests/ -q
$V -m benchmarks.kernel_floor                       # kernel_floor.csv + kernel_splits.csv
$V -m benchmarks.kernel_phases --rounds 9           # kernel_phases.csv + kernel_phase_breakdown.csv
$V -m benchmarks.benchmark_sampling --rounds 3 --reps 5 \
    --out results/raw/spike_ladder.csv --env-out results/raw/environment_spike.json
$V -m benchmarks.summarize --raw results/raw/spike_ladder.csv --out results/summary_spike.md
```

The extension is built and benchmarked in `~/.venv_flashinfer` (torch 2.9.1+cu128), the only
environment where the kernel and FlashInfer can be timed in one process. `results/raw/environment_spike.json`
records the `.so` path and mtime that produced these numbers.
