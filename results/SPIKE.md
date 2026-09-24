# Prototype Spike — does a hand-written SM120 kernel beat the production sampler?

**Verdict: YES, and the project continues — but `DECISION.md` §6's "5–20× at low batch" is not
what this prototype delivers, and the gap is now measured rather than guessed.**

`results/DECISION.md` ended with one instruction: before building the full kernel, find out whether
a hand-written kernel can beat `flashinfer_from_probs` at 72.6 µs (B=1). It can. A first-pass fused
kernel runs the whole operation — `[B, V]` logits to one token id per sequence — in **19.1 µs at
B=1**, against **72.8 µs** for the best production sampler measured in the same process on the same
instrument: **3.81×**. That is the register-resident merge (§8.2); the shared-memory bitonic it
replaced ran 22.5 µs against 75.4 µs (3.35×), and both sweeps are committed.

All numbers below come from committed artifacts under `results/raw/`, on an idle GPU with
`clocks_locked=false`. Methodology: `docs/benchmark_methodology.md`.

---

## 1. The result

`results/raw/spike_ladder_regmerge.csv` (1134 rows, register merge), `V=151936, k=50, p=0.90,
bfloat16, hot`. The last column is the same cell in `results/raw/spike_ladder.csv` (shared bitonic,
`df3e3c0`):

| B | `hf_eager` | `graph_compile` | `flashinfer_from_probs` | **`fused_kernel`** | vs the bar | bitonic |
|---|---|---|---|---|---|---|
| 1 | 325.1 | 71.8 | 72.8 | **19.1** | **3.81×** | 22.5 (3.35×) |
| 4 | 598.5 | 95.1 | 73.8 | **19.7** | **3.74×** | 22.5 (3.35×) |
| 8 | 706.7 | 98.2 | 74.3 | **20.3** | **3.66×** | 22.6 (3.35×) |
| 16 | 1181.5 | 122.2 | 74.3 | **24.6** | **3.02×** | 26.6 (2.83×) |
| 32 | 2200.5 | 168.1 | 101.2 | **30.7** | **3.29×** | 34.8 (2.91×) |

Across the full parameter grid the win is **2.50×–5.12×** (was 2.28×–4.09×), widest at small `k`
and low batch, and the kernel is faster at every one of the 32 hot cells — 7.5–22.2%:

| | k=20 | k=50 | k=100 |
|---|---|---|---|
| B=1, V=151936 | 16.3 µs (**4.49×**) | 19.1 µs (3.81×) | 24.1 µs (3.04×) |
| B=32, V=151936 | 27.0 µs (3.81×) | 30.7 µs (3.29×) | 34.9 µs (2.95×) |

**Quote 3.81×, not the 5.12× top of the range.** The maximum is B=1, V=128256, k=20, where the
fused rung's three rounds read 16.05 / 12.36 / 14.34 µs — about one ~2 µs quantum apart (§2.1) —
while FlashInfer holds to 0.5%. Taking each cell's worst round instead of its median gives
2.48×–4.64×. The anchor's rounds are 19.16 / 18.42 / 19.11 µs.

FlashInfer read 3.5% faster at the B=1 anchor in this sweep than in the previous one (1.5% median
across the grid), so every ratio divides by FlashInfer from the *same* sweep. fp16 matches bf16 to
0.7–1.2% at hot B ∈ {1, 4, 8, 32}; B=16 hot and B=1 cold diverge 8.3% and 9.8%, inside this rung's
round-to-round spread. Cold L2 narrows the win to 2.70×–3.95× (was 2.53×–3.37×).

**Gate A cost time, and the register merge more than won it back.** At commit `33802d7` the kernel
was 20.6 µs at B=1 / k=50 (3.59×) and 25.5 µs at k=100. Closing Gate A took the bitonic kernel to
22.5 / 30.3 µs: the cut now normalizes before taking the prefix, exactly as `reference.py` does,
which is K per-element divisions the old `cum[i-1] >= top_p * z` form did not pay — **~9% at k=50
and ~19% at k=100 in exchange for an exact `keep` mask and a tie path with no capacity clamp**. The
register merge then brought it to 19.1 / 24.1 µs, below the pre-Gate-A kernel, with Gate A intact.
The 3.59× is in git history and is recorded here because it will not reproduce as such.

**The comparison is deliberately unfavourable to this kernel.** `flashinfer_from_probs` is handed a
`[B, V]` probability tensor that someone else already softmaxed; the fused kernel consumes raw
logits and does the softmax itself. It still wins by 3.8×.

## 2. What it is not: the 5–20× in `DECISION.md` §6

§6 projected "single-digit to low tens of microseconds, i.e. a 5–20× operator win at low batch."
The prototype lands at the top of that latency range and the bottom of that ratio range. **The
claim should be restated as ~3–4× until a second kernel iteration says otherwise**, and the
headline should be the measured 3.81×, not the projection.

The reason is measured, not inferred. `results/raw/kernel_floor.csv` (135 rows) brackets what any
kernel can reach in this rig at `V=151936, bf16, hot`:

| probe | B=1 | B=8 | B=32 | what it bounds |
|---|---|---|---|---|
| `noop` — allocate `[B]`, launch, write | 2.50 | 2.52 | 2.51 | dispatch + launch floor |
| `scan_rowblock` — one full pass, one block/row | 10.23 | 10.23 | 10.23 | the simple grid shape |
| `scan_split` — one full pass, best split | **4.49** | 6.18 | 10.38 | the split grid shape |
| `torch_argmax` — for calibration | 8.41 | 10.30 | 15.69 | |

So one full pass over the vocabulary costs **4.49 µs** at B=1 and the kernel costs 19.1 µs — it is
**4.3× above its own floor** (5.0× before the register merge), and that is where the missing
speedup is. The floor probes do not touch the merge, so the `df3e3c0` measurement still applies.

The floor probe also settled a question that mattered more than expected: at B=1 the rig's dispatch
floor is 2.50 µs, not tens of microseconds. **The 74 µs bar is real device work, not instrument.**

### 2.1 Where the time actually goes — measured, superseding an earlier guess

> **This subsection is the bitonic baseline** (`kernel_phases.csv`, `df3e3c0`), kept because §8.2's
> comparison is against it. The register merge's breakdown is §8.2.

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
`clamp(128/B, 4, 8)`, which was right when it was fitted. Re-measured on the bitonic kernel
(`df3e3c0`):

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
and the split sweep have to come from the same binary. **This table is the bitonic binary**
(`df3e3c0`): more splits trade traversal time against a larger merge, and the register merge
changed the price of the merge, so the refit has to start by re-running `kernel_floor`.

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
- **319 passed, 2 skipped** — on the bitonic merge and again, unchanged, on the register merge
  (`results/raw/pytest_regmerge.txt`).
  Every claim above is falsified, not just asserted: inverting the index half of the packed key
  fails 32 tests; forcing the old clamped tie path fails 4; relaxing the strict `<` at the cut
  fails 8; restoring the host RNG counter fails the replay test.
- **Sanitizer-clean.** `compute-sanitizer` memcheck, racecheck and initcheck over the 104 kernel
  tests: 0 errors, 0 hazards (`results/raw/sanitizer.csv`, logs under `results/raw/sanitizer/`).
  initcheck runs *unfiltered* — with `--kernel-name` it stops tracking writes by torch's kernels
  and reports every torch-produced input as uninitialized. A positive control proves the clean run
  discriminates: never-written `torch.empty` logits trip initcheck inside `topk_partial_kernel`
  (the `initcheck_positive_control` row, log `sanitizer/initcheck_control.log`). The suite log is
  `results/raw/pytest_regmerge.txt`.

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
2. `merge_sample_kernel`, grid `B`, 128 threads — warp 0 sorts the `splits × K` candidates in
   registers (§8.2), then the block decodes values back
   out of the keys (no second read of the logits), fp32 softmax over the survivors, top-p cut on the
   exclusive prefix, and an inverse-CDF draw against `u · Z_p`.

No normalized probability tensor is materialized: scaling `u` by the retained mass replaces
renormalization. The RNG is an inline splitmix64 on `(seed, offset, row)` — `SEMANTICS.md`
explicitly frees the kernel from matching `torch.multinomial` draw-for-draw, so no uniform tensor is
passed in and no extra launch is paid.

## 6. What this does not claim

- **Not an end-to-end decode win.** Sampling is 0.16–1.2% of a decode step at B=1
  (`results/raw/amdahl_probe.csv`, measured with the `flashinfer` rung), which is the ceiling on
  any end-to-end gain from a faster sampler. This remains an operator-specialization result, exactly as
  `DECISION.md` §9 required.
- **Gate A is closed; Gate B is not.** `topk_ids` and `keep` are exact against `reference.py` --
  `keep` at 0 mismatches over 195 840 elements across 54 configurations -- and ties are now
  resolved exactly with no capacity clamp. `renormed` agrees to **15 ulp**, not bitwise, and cannot
  be bitwise: torch sums the prefix with a scan and the kernel sums it serially. The fp32 fidelity
  gate (Gate B) still has not been run on the kernel.
- **Not measured with hardware counters.** `ncu` is confirmed unavailable to this user:
  `/proc/driver/nvidia/params` reports `RmProfilingAdminOnly: 1`. Every attribution above is
  wall-clock, from isolated timing and the floor probes. What *is* measured without counters:
  per-kernel device time from a CUPTI activity trace (`results/raw/kernel_trace.csv`) and
  theoretical occupancy from the CUDA occupancy API (`results/raw/kernel_attrs.csv`) — the same
  figure Nsight reports under that name. Achieved occupancy and stall reasons are not.
- **Amortized throughput timing.** Consecutive iterations overlap, so these numbers understate
  per-call launch latency in a real dependent decode loop. Both rungs benefit: the fused kernel
  issues 2 launches per call and `flashinfer_from_probs` 9 (`results/raw/kernel_trace.csv`), and
  neither the size nor the direction of the net bias between them is measured.
- **FlashInfer's semantics differ** — it applies top-k/top-p to the full-vocabulary distribution
  where this repo renormalizes within the top-k survivors first. It is a performance rung, never a
  correctness target.
- **Clocks are not locked** (no permission). In `spike_ladder_regmerge.csv` the fused rung's
  round-to-round spread, (max − min) / median, is 1.3% median and **25.7% worst case** (the B=1,
  k=20 cell in §1), against 0.3% / 1.0% for `flashinfer_from_probs`. On the same definition the
  bitonic sweep was 0.5% / 18.0%.

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

`MERGE_BLOCK` is **1024** in every bitonic-era artifact (`df3e3c0`) and **128** in every
`*_regmerge` artifact. 512 measured −7% at B=32 against the pre-Gate-A binary (`643d369`);
re-measured after the Gate A work it was within noise of 1024 at every `k`, so `df3e3c0` put it
back. 128 and 256 were clearly worse at B=1 — but that was measured when the block itself ran the
bitonic sort, which is no longer true (§8.2), so it did not transfer to the register merge.

## 8. Next

1. **Refit the split rule on `k`, not just `B`.** Measured 21.7% at B=1/k=20 and 10.3% at
   B=8/k=20 (§3) — on the bitonic binary; re-measure first.
2. **A barrier-free merge — MEASURED: 7.5–22.2% faster at every hot cell, landed.** The bitonic
   merge was the largest single cost at low batch at 0.18 µs per stage — barrier and shared-memory
   round-trip, not arithmetic. `fs::warp_merge_sort` holds the candidates in one warp's registers
   and does the whole sort with `__shfl_xor_sync` and zero `__syncthreads`. The layout is
   **blocked, not cyclic** — bitonic's inner loop runs `j = k/2 … 1`, so small `j` is the common
   case, and blocked makes exactly those stages register-local (15 shuffle stages at P=512 against
   cyclic's 35) — and `MERGE_BLOCK` drops 1024 → 128, because a 1024-thread block caps ptxas at 64
   registers per thread and the P=1024 candidate array alone wants 64. **The block-size change is
   confounded with the merge change; everything below is one result, not two.**

   **Build**, `ptxas -v` at `sm_120`, production merge (`STOP=2`, bf16), against the same flags on
   the pre-change source; the runtime API (`results/raw/kernel_attrs.csv`) agrees on every register
   and shared-memory figure:

   | | baseline (shared bitonic) | P=128 | P=256 | P=512 (k=50) | P=1024 (k=100) |
   |---|---|---|---|---|---|
   | registers | 30 | 48 | 88 | 130 | 142 |
   | spill stores / loads | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | **0 / 0** |
   | shared bytes | 9220 | 2052 | 3076 | 5124 | 9220 |
   | `__syncthreads` executed | 49 (P=512) | 4 | 4 | 4 | 4 |
   | SASS | 27 KB | 73 KB | 123 KB | 236 KB | **451 KB** |
   | theoretical occupancy | | 83.3% | 41.7% | 25.0% | 25.0% |

   - **No spill at any width** — 0 local bytes in all 13 instantiations (the 9 the measured grid
     dispatches, plus P=32/64), from
     `cudaFuncGetAttributes`, not just ptxas. Every register index is a template parameter, so the
     candidate array cannot lower to local memory.
   - **Barriers collapse from 49 to 4, independent of `P`.** The only 4 left are the one that
     publishes `buf` and the three in the softmax tail.
   - **Occupancy is register-limited and does not matter here.** 130 registers hold P=512 to 3
     blocks/SM (25%); at B ≤ 16 the merge grid is B blocks on 70 SMs, so no SM ever sees a second
     block. The partial kernel is 100% (38 registers, 512 threads).

   **Measured**, `results/raw/kernel_phase_breakdown_regmerge.csv` (9 rounds) against §2.1's
   `kernel_phase_breakdown.csv`, `V=151936, bf16, hot`:

   | | B=1 k=20 | B=1 k=50 | B=1 k=100 | B=32 k=20 | B=32 k=50 | B=32 k=100 |
   |---|---|---|---|---|---|---|
   | phase 6, sort | 6.13 → **4.10** | 8.12 → **6.10** | 11.92 → **8.12** | 4.14 → **1.07** | 5.21 → **2.92** | 7.46 → **5.62** |
   | phases 1–5, partial kernel | 10.25 → 10.25 | 10.30 → 10.27 | 10.75 → 10.35 | 24.53 → 24.42 | 24.59 → 24.58 | 25.04 → 24.91 |
   | whole kernel | 20.19 → **16.35** | 22.51 → **18.46** | 28.62 → **22.56** | 30.72 → **26.73** | 34.81 → **30.73** | 38.91 → **34.86** |

   - **The sort got cheaper and nothing else moved.** Phases 1–5 run the same code and hold within
     0–4%, which is the control: the change is in the merge kernel and nowhere else.
   - **The merge kernel (phases 6+7) fell 12.21 → 8.19 µs at B=1, k=50 and 10.22 → 6.15 at
     B=32.** At B=1 the sort and tail deltas individually land on multiples of the ~2.05 µs
     quantum (the tail read 4.09 → 2.09, exactly one quantum), so only the group is resolvable
     there. At B=32, which is off the quantum, the tail fell 5.01 → 3.22 alongside the sort — the
     128-thread block, not the sort, is the only candidate for that.
   - **The instruction-footprint risk did not materialize.** The 451 KB k=100 kernel gained as much
     as the 236 KB k=50 one (32% vs 25% on the sort at B=1), so there is no sign the straight-line
     SASS is paying for instruction fetch. Re-rolling the cross-lane stages is not needed. With
     `ncu` blocked this is wall-clock evidence, not a stall-reason measurement.
   - **The balance has moved toward the row passes, but not everywhere.** At B=32 the three
     traversals (phases 1, 3, 5) are 22.03 µs (72%) against the merge kernel's 6.15 µs (20%), and
     they also lead at B=16 for k ≤ 50. At B ≤ 8 with k ≥ 50 the merge kernel is still the larger
     cost — 8.19 against 7.40 µs at B=1, k=50 (`kernel_phase_breakdown_regmerge.csv`).

   **Launches**, `results/raw/kernel_trace.csv` (CUPTI activity trace, 50 steps, bf16 anchor):
   the fused path is **2 kernels and 0 memcpy/memset per step** at B=1 and B=32 — partial 9.48 µs
   + merge 8.36 µs of device time at B=1, 21.63 + 6.39 at B=32. `hf_eager` is 47 / 51 kernels plus
   17 / 19 memcpy and memset, which reconciles exactly with `launch_counts.csv`'s 64 / 70 device
   operations; `flashinfer_from_probs` is 9. Device time is kernel-busy time and excludes launch
   gaps, so it sits below the ladder's latency.
3. **Run Gate B** — the fp32 fidelity gate, the one gate still not run on the kernel.
4. **Collapse the three row passes into one register-resident warp select** — 40% of the kernel
   at B=1 and 72% at B=32; the largest cost at B=32 and at B=16 for k ≤ 50.
5. Re-measure and decide whether `DECISION.md` §6's 5–20× is reachable or should be retired.

## Reproduction

```bash
export PATH=~/.venv_flashinfer/bin:/home/sathvik/cuda-12.9/bin:$PATH
V=~/.venv_flashinfer/bin/python
$V setup.py build_ext --inplace
$V -m pytest tests/ -q
$V -m benchmarks.kernel_floor                       # kernel_floor.csv + kernel_splits.csv
$V -m benchmarks.kernel_phases --rounds 9 \
    --out results/raw/kernel_phases_regmerge.csv \
    --breakdown-out results/raw/kernel_phase_breakdown_regmerge.csv
$V -m benchmarks.benchmark_sampling --rounds 3 --reps 5 \
    --out results/raw/spike_ladder_regmerge.csv --env-out results/raw/environment_regmerge.json
$V -m benchmarks.summarize --raw results/raw/spike_ladder_regmerge.csv --out results/summary_regmerge.md
$V -m benchmarks.kernel_trace                       # kernel_trace.csv
$V -m benchmarks.kernel_attrs                       # kernel_attrs.csv
$V -m benchmarks.sanitize                           # sanitizer.csv + sanitizer/*.log
```

The bitonic-era artifacts (`spike_ladder.csv`, `kernel_phases.csv`, `kernel_phase_breakdown.csv`,
`environment_spike.json`, `summary_spike.md`) come from the same commands with their default or
`_spike` output names at `df3e3c0`; they are kept as the baseline §8.2 compares against.

**Provenance.** Every `*_regmerge` artifact, `kernel_trace.csv`, `kernel_attrs.csv`,
`sanitizer.csv` and `pytest_regmerge.txt` records `6ddf1e0-dirty`: the register merge at `0c9de95`
plus the additive `kernel_attrs()` debug binding and the new measurement scripts — no kernel source
differs from `0c9de95`. The bitonic baseline is `df3e3c0` (`spike_ladder.csv` records
`df3e3c0-dirty` for the same reason: uncommitted scripts, not kernel source).

The extension is built and benchmarked in `~/.venv_flashinfer` (torch 2.9.1+cu128), the only
environment where the kernel and FlashInfer can be timed in one process. `results/raw/environment_spike.json`
and `environment_regmerge.json` record the `.so` path and mtime that produced each sweep.
