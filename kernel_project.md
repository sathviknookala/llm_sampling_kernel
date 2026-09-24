# CLAUDE.md — fused_sampling_kernel

A CUDA/kernel-engineering project on the **token-sampling** stage of LLM decode — the path from
final-layer logits to a sampled token id. The unit under study is a **fused top-k + top-p sampling
kernel** for one autoregressive decode step, specialized for **NVIDIA SM120 (Blackwell)**, written
in CUDA C++ as a PyTorch extension and benchmarked against the generic PyTorch eager path. Scope is
evidence-driven: profiling decides what to optimize next, not a fixed plan.

## Git

- **Never add Claude as an author or co-author on commits or pushes.** The user is always the sole
  author — no `Co-Authored-By: Claude` trailer, no `Generated with Claude Code` line.

## Doc map

This file is the always-loaded hub — keep it thin. Pull the reference doc that fits the task:

- **`benchmarks/SEMANTICS.md`** — the locked operation definition: signature, target regime, stage
  order, tie policy, RNG/determinism, NaN/Inf policy, the Phase 1/2 reference contract, and the two
  conformance gates. *Read before writing any kernel, test, or benchmark.*
- **`docs/benchmark_methodology.md`** — the measurement rig: latency definition, amortized timing,
  ordering control, hot/cold L2 conditions, DRAM-floor formula, environment capture, reproduction
  commands, limitations. *Read before quoting or adding a benchmark number.*
- **`results/DECISION.md`** — the go/no-go gate and its nine answers, with the revised hypothesis.
  *Read before any kernel work.*
- **`results/SPIKE.md`** — the kernel's measured **2.28-4.09x** over `flashinfer_from_probs`, the
  floors that bound it, the measured phase breakdown, the closed Gate A, and §7's three measured
  non-results. *Read before touching `csrc/`.*
- **`results/summary_spike.md`** — generated summary over `results/raw/spike_ladder.csv`.
- **`benchmarks/kernel_phases.py`** — the phase ablation behind `SPIKE.md` §2.1: runs phases 1..n
  of the real kernel and stops, so the cost breakdown is measured rather than inferred.
- **`results/summary_ladder.md`** — generated summary tables over the raw sweep.
- **`benchmarks/regime.py`** — the single source for shapes, dtypes, and anchors. Do not hardcode
  these in a script; import them.

## The Operation

One decode step produces logits `[B, V]` — `B` sequences in flight, `V` the vocabulary (~128K–152K
for modern LLMs). The operation consumes that matrix and produces one token id per sequence:

```
[B, V] logits  ->  top-k  ->  softmax  ->  top-p  ->  renormalize  ->  sample  ->  [B] token ids
```

Per sequence, over logits `ℓ = [ℓ_1 … ℓ_V]`:

1. **Top-k** — keep the `K` largest logits (`R^V -> R^K`), in descending logit order, carrying each
   candidate's original vocabulary index. With `V ≈ 150K` and `K = 50`, 50 candidates survive.
2. **Softmax** — `p_i = exp(ℓ_i) / Σ_j exp(ℓ_j)` over the `K` survivors, so `Σ p_i = 1`. Use the
   numerically stable form: subtract the max retained logit before exponentiating.
3. **Top-p (nucleus)** — with candidates already in descending order, take cumulative sums
   `c_r = Σ_{i≤r} p_i` and cut at `r = min{ r : c_r ≥ p_top }`. Tokens past `r` are dropped.
4. **Renormalize** — the survivors no longer sum to 1, so divide by the retained mass
   `Z_p = Σ_{i≤r} p_i`, giving `p'_i = p_i / Z_p`.
5. **Sample** — draw `u ~ U(0,1)` and pick the token whose cumulative interval contains it; the
   candidate's original vocabulary index is the emitted token id.

**Renormalization is not a required step in the fused form.** Working with unnormalized weights
`w_i` and retained mass `Z_p = Σ_{i≤r} w_i`, drawing `u ~ U(0,1)` and selecting the first `i` with
`Σ_{j≤i} w_j ≥ u·Z_p` yields the identical categorical distribution with no normalized probability
tensor materialized.

## Why It Is a CUDA Target — corrected 2026-08-22 by measurement

The eager PyTorch path expresses the pipeline as separate tensor ops — top-k, softmax, cumsum,
mask, reduction + renormalize, multinomial — and pays for that structure. **What it actually pays
is now measured** (`results/raw/stage_profile.csv`, `results/raw/launch_counts.csv`):

- **A full-vocabulary sort.** HF sorts all ~152K entries to find a nucleus of ~50 — **65% of
  `hf_eager` at B=32**, the single largest cost at scale.
- **Per-op latency floors that dwarf the work.** At B=1 every stage costs 55–68 µs regardless of
  size; `torch.multinomial` over `[B, 50]` costs ~57 µs, essentially the same as over 151,936.
- **Kernel launches: 64–70 per decode step**, against FlashInfer's 9–11.
- Temporary probability / cumulative / mask / normalized tensors, and generic implementations
  covering shapes and runtime parameters this workload never uses.

**It does NOT pay for DRAM traffic, and the earlier framing of this section was wrong.** Reading
the full vocabulary is **0.17–0.80%** of `hf_eager` latency, and the `[B, V]` tensor fits in this
GPU's 48 MB L2 at every point in the regime. The old target of

```
read:   [B, V] input logits          <- ~0.5 us at B=1: not the problem
write:  [B]    sampled token ids
```

describes a bottleneck that does not exist here. The real target is to collapse the sort, the
fixed-cost sampler, and the launch chain into one pass with candidate state in registers — a
**latency and launch** win, not a bandwidth win.

## Target Regime

Deliberately narrower than a general-purpose sampling library — the specialization *is* the
project:

```
Architecture:   SM120 (Blackwell)
Logit dtype:    FP16 / BF16
Vocabulary:     ~128K-152K
Batch size:     low decode batches
Top-K:          small — 20 / 50 / 100
Top-P:          typical — 0.90 / 0.95
Output:         one token id per sequence
```

Production libraries must serve many architectures, arbitrary vocabularies, runtime top-k, per
request configuration, multiple sampling modes, and wide batch ranges. A kernel bound to this
regime can assume much more: fixed or template-specialized `K` lets the compiler optimize candidate
management aggressively, and a small candidate set can live largely in registers instead of a
generic global-memory structure. Recent SM120 experimentation around FlashInfer suggests simple
architecture-specific, register-heavy top-k implementations can beat more general sampling kernels
at low decode batch sizes — which is why SM120 specialization is an explicit design objective
rather than an afterthought.

## Core Hypothesis — REVISED 2026-08-22 on measurement

The original hypothesis led with global-memory traffic. **That part is rejected** by
`results/DECISION.md`: reading the whole vocabulary is 0.17-0.80% of `hf_eager` latency, and the
logits tensor fits in this GPU's 48 MB L2 at every point in the target regime. The operation is not
DRAM-bound and never was.

The surviving hypothesis, narrowed to what the measurements support:

> For **low-batch** LLM decoding on SM120, a specialized fused CUDA kernel can perform top-k
> filtering, top-p filtering, and categorical sampling with materially lower **operator latency**
> than the best available production sampler, by removing a materialized full-vocabulary softmax, a
> fixed-cost multinomial pass, and per-stage launch/latency floors — not by reducing DRAM traffic.

**It cannot claim decode-latency improvement.** Sampling is 0.16-1.2% of a measured decode step
(`results/raw/amdahl_probe.csv`), so even infinitely fast sampling is invisible end to end. The
deliverable is an operator-specialization result with an honest Amdahl ceiling attached.

**The bar is `flashinfer_from_probs`** — **75.4 / 75.5 / 101.2 µs at B=1/8/32**
(`results/raw/spike_ladder.csv`) — **not `hf_eager`**. The 72.6 / 73.9 µs in `DECISION.md` and
`summary_ladder.md` is the same rung measured in the pre-kernel sweep
(`results/raw/sampling_ladder.csv`); both are committed, quote whichever artifact you are citing.
A speedup against HF eager is not a result: `tight_eager`, ordinary eager PyTorch that collapses to
`[B, K]`, already gets 10x at B=32.

## Workflow Rules

### Plan Before Acting
- Enter plan mode for any non-trivial task (3+ steps or architectural decisions)
- If something goes sideways mid-task, stop and re-plan — don't push through
- Write a spec or checklist upfront to reduce ambiguity; verify with the user before implementing

### Subagent Strategy
- Use subagents to keep the main context window clean
- Offload research, exploration, and parallel analysis to subagents
- One focused task per subagent

### Self-Correction Loop
- After any correction: note the pattern so the same mistake doesn't recur
- Ruthlessly iterate on this until mistake rate drops

### Verification Before Done
- Never consider a task complete without demonstrating it works
- Check logs, run tests, or diff behavior when relevant
- **After touching `csrc/`, rebuild before testing**: `python setup.py build_ext --inplace`.
  `pytest` runs against the built `.so`, not the sources, so an unrebuilt edit tests the previous
  binary and passes. Declare headers as `depends` in `setup.py` so a `.cuh` edit triggers a
  rebuild — it still only happens when you actually run the build
- A test that passes where the bug cannot occur is not a test — confirm it fails without its fix
- Sampling is stochastic: a correctness test must pin the RNG seed, or assert on a distribution
  over enough draws to be meaningful. Top-k/top-p over a ~150K vocabulary involves ties,
  sorted-order dependence, and RNG stream alignment — an `allclose` against PyTorch will not catch
  a wrong-but-plausible distribution
- Ask: "Would a senior engineer approve this?"
- Before quoting a committed number, check the tree still reproduces it

### Demand Elegance
- For non-trivial changes: pause and ask "is there a more elegant solution?"
- If a fix feels hacky: "Knowing everything I know now, implement the clean version"
- Skip this for simple, obvious fixes — don't over-engineer

### Autonomous Bug Fixing
- Given a bug report: fix it; don't ask for hand-holding
- Point at logs, errors, failing tests — then resolve them

### Measurement Discipline
- Never quote a number that is not in a committed artifact under `results/`, and name the file
- State what a metric excludes when it is a floor rather than a measurement
- Don't pipe a long run through `grep` — the pipeline reports grep's exit code and can turn a
  crash that lost real rows into an apparent success
- A run that must outlive the session has to be launched detached (`setsid`), not backgrounded
- Check the GPU is actually free before a timed run; another process's residency shows up as an
  OOM or as inflated timings, not as a clear error

## Code Style: Comments

- No paragraph-style or multi-line block comments explaining what code does
- Comments only where intent isn't obvious from the code itself (e.g. non-obvious tradeoffs,
  gotchas, why not what)
- Max 1 line per comment; keep it tight
- No section dividers, no docstrings restating the function signature, no "this function does X"
  fluff
- If the code is self-explanatory, leave it uncommented

## Current Focus

**A register-resident warp merge has landed UNMEASURED and UNTESTED, and that is the first thing
to fix.** `fs::warp_merge_sort` (`csrc/fused_sampling.cuh`) replaces the shared-memory bitonic:
one warp, candidates in registers, `__shfl_xor_sync` across lanes, zero barriers inside the sort.
It **compiles clean with no spill at any width** and the build-side numbers are in `SPIKE.md` §8.2
-- but **nothing has been run**: no pytest, no ladder, no ablation. The GPU was occupied by an
unrelated RL job for the whole session. **Do not quote a speedup for it and do not trust it until
`pytest` passes.**

Every performance number in `results/` still describes the *previous* merge at `df3e3c0`. The
kernel there is **22.5 us at B=1** against **75.4 us** for `flashinfer_from_probs`
(`results/raw/spike_ladder.csv`, 1134 rows), **2.28x-4.09x** across the grid. Quote **3.35x**.

Next, in order:

0. **Build and run the gate.** `python setup.py build_ext --inplace` then `pytest tests/ -q`;
   321 tests collect, 319 passed + 2 skipped before this change. Nothing else matters until this
   is green
1. **Measure the register merge** -- `benchmarks.kernel_phases` for phase 6, then the full ladder.
   The block-size change (`MERGE_BLOCK` 1024 -> 128) is confounded with it and must be reported as
   one result. If phase 6 does not fall, suspect the 236 KB / 451 KB straight-line SASS footprint
   first; the fix is to re-roll the 15 cross-lane stages, which do not need compile-time indices
2. **Refit the split rule on `k`, not just `B`** -- `clamp(128/B, 4, 8)` is wrong at k=20: 20
   splits gives 16.40 us at B=1 against 19.96 (+21.7%). `results/raw/kernel_splits.csv`
3. **Run Gate B** -- the fp32 fidelity gate, the only gate still not run
4. Collapse the three row passes into one register-resident warp select -- 34% at B=1, 63% at
   B=32. This is Phase 2 of the register plan and is explicitly gated on Phase 1 measuring well

**Do not re-attempt these three** -- implemented, measured, reverted, written up in `SPIKE.md` §7:
pair-indexing the bitonic stages (neutral); sorted runs per split with the merge resuming at `2R`
(stages relocate, total flat); a K-conditional normalization path (built on 3-round noise).

## Last Session

**Session 10 -- the register merge, written and compiled but never run.** The GPU was busy with an
unrelated RL job start to finish, so this session produced code and build-side evidence only.

- **`fs::warp_merge_sort` replaces the shared bitonic.** Warp 0 holds all `P` packed candidates in
  registers and sorts them with `__shfl_xor_sync`. `P` is now a template parameter dispatched over
  {32,64,128,256,512,1024}; `merge_sample_kernel` and `merge_ids_kernel` are templated on it
- **The layout is blocked, not the cyclic one `SPIKE.md` §8 sketched.** Bitonic's inner loop runs
  `j = k/2 … 1`, so small `j` is the common case; blocked makes exactly those stages
  register-local. SASS confirms 15 cross-lane stages at every `P` against cyclic's 35 at P=512
- **No spill at any width, and the block-size change is forced, not chosen.** 0 spill bytes
  everywhere; 130 registers at P=512 against a 1024-thread block's 64-register cap is why
  `MERGE_BLOCK` had to drop 1024 -> 128. Barriers 49 -> 4, independent of `P`
- **The trade is a 9-17x instruction footprint** -- 236 KB of straight-line SASS at k=50, 451 KB
  at k=100. Recorded as the first suspect if the merge does not get faster
- **Three stale `MERGE_BLOCK` claims corrected.** `SPIKE.md` said 512 and `<<<B, 512>>>`; `643d369`
  set 512 and `df3e3c0` put it back to 1024, which is what every committed artifact was measured
  on. The artifacts were right; the prose was wrong

## Known Issues

- **`CLAUDE.md` is gitignored and therefore not recoverable from git.** It is the only
  record of scope and open decisions, and it lives in one place on this machine. Anything
  that must survive belongs in a tracked file under `docs/`.
- **The remote is HTTPS, not SSH, and this was forced.** This machine's only SSH key
  (`id_ed25519`) authenticates to GitHub as **`NeuralNookala`**, which has no write access to
  `sathviknookala/llm_sampling_kernel` — `git@github.com:` pushes are rejected. `origin` was
  switched to `https://github.com/sathviknookala/llm_sampling_kernel.git`, matching
  `decode_llm_kernel`, which pushes via the stored credential helper. To go back to SSH, that key
  has to be added to the `sathviknookala` account first.
- **Commit identity is set per-repo, not globally.** This repo's local config is
  `Sathvik Nookala <sathviknookala@gmail.com>`; the machine's global email is `@neuralads.ai` and
  global `user.name` is unset. A fresh clone or a new repo here needs the local config set again
  or commits land unattributed.
- **The remote repo name differs from the local directory** — `llm_sampling_kernel` vs
  `fused_sampling_kernel`. Intentional as far as this repo knows; noted so a future session does
  not read it as a wrong remote.
- **Gate A is closed; Gate B is not.** `results/raw/spike_ladder.csv` (1134 rows). Quote against
  `flashinfer_from_probs`, never `hf_eager`. `topk_ids`, ties and `keep` are exact; `renormed` is
  15 ulp and cannot be bitwise. **Gate B -- the fp32 fidelity gate -- has still not been run.**
- **Ties are exact at any multiplicity; the clamp is gone.** Past 2048 per split the kernel falls
  back to index bucketing plus a bitmap over the boundary bucket, which cannot overflow because
  the bucket is `2^shift` wide. Real logits never reach it (max multiplicity 14 in
  `results/raw/tie_fidelity.csv`), so the fast path is unchanged.
- **A device-side assert in `torch.multinomial` is process-fatal on CUDA.** Feeding NaN/inf with
  `check_inputs=False` poisons the CUDA context for the whole process — every later CUDA op fails,
  not just the offending call (verified 2026-08-22). A long sweep with validation off for timing
  loses the entire run to one bad row. Validate outside the timed region; never feed a known-bad
  tensor with checks off, and never write a test that does.
- **SM120 is confirmed and the repo's own `setup.py` builds for it.** RTX PRO 4000 Blackwell,
  `sm_120`, 25.2 GB, 48 MB L2, 70 SMs, driver 575.64.03; measured HBM **552.7 GB/s**. `nvcc` **12.9**
  at `/home/sathvik/cuda-12.9/bin/nvcc`. `TORCH_CUDA_ARCH_LIST` is unset and torch auto-emits the
  right gencode pair, so no env var is needed. `--use_fast_math` is deliberately **off**: `__expf`
  would put the kernel out of reach of the fp32 reference Gate A compares against.
- **`~/.venv_flashinfer` is the build and benchmark env, not `cu`.** CLAUDE.md used to call `cu`
  the main env; the committed results have always come from the venv, and it is the only env where
  the kernel and FlashInfer can be timed in one process. It has `ninja`; `cu` does not, and `cu`
  cannot host FlashInfer at all. **A `.so` is not portable across torch minor versions** — `cu` is
  torch 2.11, `qnt` 2.10, the venv 2.9.1 — so a build in one env will not load in another.
- **The tie-fidelity artifact uses Gaussian logits, not real model logits.** Real decode-step
  logits are heavy-tailed with a few dominant tokens, which changes tie density near the
  k-boundary. The 18/18 result is expected to hold — quantization noise does not become more
  faithful on real data — but it is measured on synthetic input and should be re-run once a real
  logits capture exists.
- **FlashInfer lives in `~/.venv_flashinfer`, not the main env, and needs PATH help.** Its wheel
  wants torch >=2.13/CUDA 13; this driver (575.64.03) caps at CUDA 12.9. The venv pins
  `torch 2.9.1+cu128` + `transformers 5.12.1`, and its JIT needs `ninja` and `nvcc` on `PATH`:
  `PATH=~/.venv_flashinfer/bin:/home/sathvik/cuda-12.9/bin:$PATH`. **vLLM is still not installed.**
- **FlashInfer's semantics differ from ours** — it applies top-k/top-p to the full-vocabulary
  distribution where we renormalize within the top-k survivors first. It is a performance rung
  only and is never gated against `reference.py`.
- **Clocks cannot be locked and persistence mode cannot be set** — `nvidia-smi -lgc` / `-pm` both
  return insufficient permissions. Every benchmark row carries `clocks_locked=false`. In
  `results/raw/spike_ladder.csv` the *median* round-to-round spread is 0.2-0.5% per rung, but the
  `fused_kernel` rung's **worst case is 18%** against 1.4% for `flashinfer_from_probs` — the same
  ~2.05 us quantum as the phase data, and the reason a 3-round sweep cannot settle a few-percent
  kernel change at B=1.
- **`reference.py` now does a full `[B, V]` sort**, not a `topk`, to get a portable tie-break. It
  is the semantic reference and is not the timed baseline, so this is deliberate — but it means
  the reference is no longer a "tight eager" *timing* rung. If such a rung is wanted in the ladder,
  it needs a separate topk-based implementation, explicitly labelled as not tie-exact.
- **A ~2.05 us timing quantum at B<=8 is real, reproducible, and unexplained.** In
  `results/raw/kernel_phases.csv` the cumulative phase timings land on near-exact integer
  multiples of it (2, 2, 3, 3, 5, 9, 10 x 2.053 at B=1, k=50), so a ~2 us step migrates between
  adjacent phases from round to round. Within-round rep spread is 0.0-0.1%, so it is not rep
  noise, and 9 rounds did not average it out. Consequence: **individual sub-2 us phase deltas at
  B<=8 are not resolvable** -- report groups (phases 3-5 together are stable to 0.8%). It does not
  touch the merge or traversal numbers, which are much larger. Unlocked clocks are the suspect but
  nothing has been measured to confirm it.
- **Profiling is confirmed blocked, not merely unconfirmed.** `/proc/driver/nvidia/params` reports
  `RmProfilingAdminOnly: 1`, the driver default, and nothing in `/etc/modprobe.d/` overrides it.
  `ncu` is installed (2025.2.1.0) but a non-root run returns `ERR_NVGPUCTRPERM`. Every attribution
  in `results/SPIKE.md` is therefore wall-clock. A fix needs
  `NVreg_RestrictProfilingToAdminUsers=0` plus a reboot, or `sudo ncu`. **The substitute is the
  `STOP`-templated phase ablation** (`benchmarks/kernel_phases.py`), which needs no permissions;
  `compute-sanitizer` and `nvcc -Xptxas -v` are also available unprivileged.
- **The register-residency premise is still borrowed, and the current kernel does not use it.**
  Candidates live in shared memory, not registers, and the selection is a three-pass radix rather
  than a one-pass register-resident warp select. The **5.0x** remaining against the measured floor
  (22.5 us against 4.49 us, `results/raw/kernel_floor.csv`) is the reason to try the register
  design — but with `ncu` blocked, occupancy and spill cannot be measured here, only inferred from
  wall-clock.

---

At session end, refresh Current Focus / Last Session / Known Issues here — overwrite in place, 3–5
bullets in Last Session on what was actually done, no appending, no changelogs (git log is for
history). When reference docs under `docs/` are created, add a doc map section back to this file
and move the detail out of it — this file is the always-loaded hub and should stay thin.
