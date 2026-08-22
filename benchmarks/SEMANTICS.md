# Operation Semantics — Target Regime

The locked definition of the operation the fused kernel must reproduce. Read this before writing
a kernel, a test, or a benchmark.

**The contract is the serving engine, not this repo's own reference.** `reference.py` is a tight
eager implementation of the stage order below; it exists to define the semantics precisely and to
be cheap to test against. It is not the thing to beat and it is not the authority — the serving
engine is. Where `reference.py` and the current contract engine disagree, the disagreement is
resolved in favor of the engine or recorded here as a deliberate, measured exception.

## Reference Contract

The engine the kernel is optimized against changes as the project matures. Exactly one engine is
the contract at a time, and this section says which.

### Phase 1 (current) — Hugging Face `transformers`

Pinned to **transformers 5.12.1**, the version installed and verified against on 2026-08-22.
`hf_baseline.py` calls the library's own `TopKLogitsWarper` and `TopPLogitsWarper` rather than
reimplementing them, so the baseline cannot silently drift from the library.

HF's decode-step sampling path, read from `generation/logits_process.py` and
`generation/utils.py::_sample`:

```
next_token_logits = logits[:, -1, :].to(dtype=torch.float32)   # HF casts to FP32
scores = TopKLogitsWarper(top_k, min_tokens_to_keep=1)(scores)  # mask to -inf, full [B,V] kept
scores = TopPLogitsWarper(top_p, min_tokens_to_keep=1)(scores)  # ascending sort over full [B,V]
probs  = softmax(scores, dim=-1)                                # full [B,V]
token  = multinomial(probs, num_samples=1)                      # over V, not K
```

Two consequences worth stating explicitly:

- **HF is semantically equivalent to the stage order below, not different from it.** Its top-k
  warper masks non-candidates to `-inf`, so the subsequent full-vocabulary softmax contributes
  exactly `0` for them and normalizes *within the k survivors* — the same distribution this
  document specifies. Its ascending-sort removal rule `cum_asc <= 1 - top_p` maps exactly onto the
  descending retention rule `c_{i-1} < top_p` used here. Both were verified empirically, not
  assumed: agreement to **3.7e-9** on FP32 logits across the regime grid.
- **HF is not shape-equivalent, and that is the whole optimization target.** Every stage stays
  `[B, V]`: a full-vocabulary sort, two full-vocabulary softmaxes, a scatter, and a multinomial
  over `V ≈ 128K-152K` to produce one integer. `reference.py` collapses to `[B, K]` after top-k.
  **The baseline to beat is `hf_baseline.py`, not `reference.py`** — quoting a speedup against the
  tight eager reference would flatter the kernel against a strawman nobody runs.

### Phase 2 (planned) — vLLM / FlashInfer

When the kernel is validated against Phase 1, the contract shifts to a production serving engine
and **this section gets rewritten before any number is quoted against it**.

**FlashInfer 0.6.17 is now installed and benchmarked**, but as a *performance* rung only — see
`docs/benchmark_methodology.md` and `results/DECISION.md`. It is **not** the contract yet: its
semantics differ (top-k/top-p applied to the full-vocabulary distribution, where this repo
renormalizes within the top-k survivors first), and it lives in an isolated venv because its wheel
requires a CUDA version this driver cannot run. `vllm` is still not installed, and nothing about
vLLM's internals is asserted anywhere in this repo.

Making FlashInfer the *contract* rather than a rung still requires answering, each verified against
installed source the same way HF was:

- Does the engine apply top-k/top-p to logits or to post-softmax probabilities, and in what order?
- Does it sample via `multinomial`, or via a Gumbel/exponential trick, or via rejection sampling
  that never materializes a sorted candidate list?
- Is its top-k count-limited (exactly k) or value-thresholded (all boundary ties, like HF)?
- Does it batch per-sequence `top_k`/`top_p` vectors, and does that change the candidate-management
  strategy the kernel is built around?
- What is its own kernel already doing — the honest baseline may already be a fused CUDA kernel,
  not an eager op sequence, which is a materially harder bar than Phase 1.

**A Phase 1 win does not transfer to Phase 2 by default.** The Phase 1 baseline is an unoptimized
eager path; a production engine's sampler is not. Any headline carried across phases must be
re-measured, and the write-up must name which engine a number is against.

### What conformance means — two gates, deliberately separate

Bit-exact agreement with the contract engine is **not** the gate, and in the target dtypes it is
not even available (see the tie policy). Conflating "is the kernel correct" with "does it match
HF" would make both questions unanswerable, so they are split.

**Gate A — kernel correctness. Kernel vs `reference.py`, exact.**

The kernel must reproduce the reference's candidate ids, nucleus, and renormalized probabilities
exactly, at the same dtype and on the same device. This is engine-independent and survives the
Phase 2 switch untouched. It is only possible because the tie-break is specified: with
`torch.topk`'s unspecified ordering, ~93% of BF16 rows at `k=50` could only be tested
distributionally, which is precisely where a weak gate would hide a real bug.

**Gate B — semantic fidelity. Measured against FP32, not against HF.**

The pass condition is *"fixed-`k` in low precision is at least as close to the FP32 result as the
contract engine is"* — not *"fixed-`k` matches the contract engine"*. HF is the contract for the
**shape and cost** of the operation, not the authority on its low-precision numerics, where it is
measurably the worse approximation. Evidence: `results/raw/tie_fidelity.csv`, 18/18 configurations,
regenerate with `python -m benchmarks.tie_fidelity`.

**TV distance from HF is reported, never gated.** It is carried in the artifact and guarded by a
loose tripwire so an unexplained jump is caught, but drifting from HF is not by itself a defect.
Any write-up quoting fidelity must give the FP32 comparison, not only the HF one.

Exact agreement with HF is still required on the paths where no tie and no RNG stream is involved
— `top_k=1`, one-hot logits — and on FP32 inputs, where the two agree to ~1e-8 across the regime.

`tests/test_hf_conformance.py` implements both gates for Phase 1.

## Signature

```
sample_eager(logits, top_k, top_p, *, generator=None, check_inputs=True, return_stages=False)
    logits : [B, V]  float16 | bfloat16 | float32   (device: cpu | cuda)
    top_k  : int     scalar, uniform across the batch, 1 <= top_k
    top_p  : float   scalar, uniform across the batch, 0 < top_p <= 1
    ->     : [B]     int64 vocabulary token ids
```

- **Scalar `top_k` / `top_p`, uniform across the batch.** Per-sequence parameters are the common
  production case but are out of the target regime; adding them later changes the kernel's
  candidate-management strategy, so the restriction is recorded rather than assumed away.
- **No in-place mutation.** `logits` is read-only; the operation allocates its own intermediates.
- **`top_k > V` is clamped to `V`.** Not an error — a small vocabulary is a legal edge case.
- **Output dtype is `int64`**, matching `torch.multinomial` and PyTorch indexing convention.

## Target Regime

```
Architecture:   SM120 (Blackwell)
Logit dtype:    FP16 / BF16          (FP32 accepted, used for reference checks)
Compute dtype:  FP32                 (all reductions and accumulation)
Vocabulary:     ~128K-152K           (128256 Llama-3, 151936 Qwen-2.5)
Batch size:     low decode batches   (1, 2, 4, 8, 16, 32)
Top-K:          1, 20, 50, 100
Top-P:          0.90, 0.95, 1.0
Output:         [B] int64 token ids
```

**Compute is FP32 regardless of input dtype.** Logits are cast to FP32 on entry and every stage —
max subtraction, exp, sum, cumsum, renormalization — runs in FP32. FP16 has ~3 decimal digits of
mantissa; a cumulative sum over 100 candidates in FP16 accumulates error comparable to the
probabilities being compared against `top_p`. The fused kernel must accumulate in FP32 too, and
this is a correctness requirement, not a tuning knob.

## Stage Order

```
[B,V] logits -> top-k -> softmax -> cumsum -> top-p mask -> renorm -> multinomial -> [B] ids
```

### 1. Top-k

`k = min(top_k, V)` largest logits per row, **sorted descending**, carrying original vocabulary
indices. Sorted order is required — every later stage depends on it.

**Tie policy: exactly `k` candidates; ties resolve to the lowest token id.**

Both halves are now specified, and neither is implementation-defined any more.

**Exactly `k`.** HF's top-k is *value-thresholded* (`scores < topk(scores, k).values[..., -1]`), so
every token tied with the k-th value survives and HF can retain **more than k** candidates. This
repo keeps exactly `k`. The divergence is real and, in the target dtypes, routine rather than
exceptional — measured at `V=128256, batch=512` on 2026-08-22:

| dtype | k | rows with a boundary tie | tie multiplicity (median / max) |
|---|---|---|---|
| FP16 | 20 / 50 / 100 | 13% / 31% / 48% | 1 / 4 |
| BF16 | 20 / 50 / 100 | 69% / 93% / 99% | 4 / 16 |

**Exactly `k` is not a concession — it is the more accurate choice.** Those tokens are not really
tied: in FP32 they hold distinct values and do not belong in the top `k`. Half-precision
quantization collapses them into one bucket, and HF's threshold formulation promotes that
quantization noise into extra candidates. Keeping exactly `k` discards the noise. Measured against
the FP32 result both are approximating, fixed-`k` is nearer in **18 of 18** swept configurations,
on mean and max TV alike — `results/raw/tie_fidelity.csv`. Temperature does not change this: BF16
ties are a *relative*-precision limit, so scaling logits leaves tie density essentially unchanged.

The information is destroyed in the input logits before the kernel sees them; no design choice
recovers the FP32 ordering. The only question is how to resolve an inherited ambiguity, and this
resolution is the one that stays closer to the intended operation.

**Lowest token id wins.** Among equal values the lower vocabulary index is retained and ordered
first. `reference.py` gets this from a stable descending sort rather than `torch.topk`, whose tie
order is unspecified and **differs between CPU and CUDA** — unusable as a target where ties are the
norm. The kernel gets the same semantics for free by comparing `(value, id)` lexicographically in
its candidate selection; it does not need a full sort, and the reference's sort is a
clarity-over-speed choice in the reference only.

There are two boundaries a tie can straddle, and both are covered by this rule:
- **the k-boundary** — which tied tokens enter the candidate set
- **the top-p cutoff** — which tied token is the last one kept inside the nucleus

### 2. Softmax — over the k survivors, not the vocabulary

`p_i = exp(ℓ_i - ℓ_max) / Σ_{j≤k} exp(ℓ_j - ℓ_max)`, so `Σ_{i≤k} p_i = 1`.

**This matches HF and is not a divergence.** HF softmaxes over the full vocabulary, but its
top-k warper has already masked non-candidates to `-inf`, which contribute exactly `0` — so its
normalization is over the k survivors too. Verified empirically to 3.7e-9; see Reference Contract.
The difference between the two is shape, not distribution.

The max subtraction is mandatory, not an optimization: it is what makes large logits finite.

### 3. Cumsum

`c_r = Σ_{i≤r} p_i` over the descending-sorted survivors.

### 4. Top-p mask

Nucleus cutoff `r = min{ r : c_r >= top_p }`, keeping candidates `1..r`. Implemented as the
equivalent exclusive-prefix test:

```
keep_i  <=>  c_{i-1} < top_p        (c_0 = 0)
```

- **The top-1 candidate is always kept**, since `c_0 = 0 < top_p` for any `top_p > 0`. There is no
  path that returns an empty nucleus.
- **The comparison is `>=` at the cutoff** (strict `<` on the exclusive prefix). **This matches
  HF exactly.** HF sorts ascending and removes where `cum_asc <= 1 - top_p`; substituting
  `cum_asc = 1 - c_{i-1}` gives removal when `c_{i-1} >= top_p`, i.e. retention when
  `c_{i-1} < top_p` — the identical rule. Older HF releases used a descending sort with
  `cumulative > top_p` and a right-shift, which keeps one extra token at exact equality; that
  formulation is **not** what 5.12.1 does and is not what the kernel implements.
- **`top_p >= 1.0` disables top-p filtering entirely** — all `k` survivors are retained. Under
  exact arithmetic `c_k = 1`, so `r = k`; in floating point `c_r` reaches `1.0` early and would
  drop tail tokens of negligible but nonzero probability. Special-casing restores the exact
  arithmetic semantics rather than departing from them.

### 5. Renormalization

`p'_i = p_i / Z_p` where `Z_p = Σ_{i≤r} p_i`, giving `Σ_{i≤r} p'_i = 1`.

Present in the eager reference because the specified flow includes it and `torch.multinomial` is
the sampler. **The fused kernel is expected to eliminate it**: drawing `u ~ U(0,1)` and selecting
the first `i` with `Σ_{j≤i} w_j >= u·Z_p` over unnormalized weights yields the identical
categorical distribution. That is an implementation freedom, not a semantic difference — the
distribution the kernel samples from must equal the one defined here.

### 6. Multinomial

One draw per row from the renormalized distribution; the winning candidate's original vocabulary
index is the emitted token id.

## RNG and Determinism

- **Randomness enters through an explicit `torch.Generator`.** No reliance on global RNG state.
  The generator's device must match the logits' device.
- **Determinism contract: same seed + same input + same shape + same device -> same output.** This
  is what the tests assert.
- **Candidate selection is fully deterministic and RNG-independent.** With the tie-break
  specified, the candidate ids, nucleus, and renormalized probabilities are a pure function of
  (logits, top_k, top_p) — identical on CPU and CUDA. Only the final draw consumes randomness.
- **Bit-exact agreement with `torch.multinomial` is explicitly NOT required of the fused kernel.**
  The kernel will consume its own RNG stream (likely Philox in-kernel) and cannot be expected to
  reproduce PyTorch's sampler draw-for-draw. The kernel's correctness gate is *distributional* —
  the empirical sampling frequencies must match the reference distribution — plus exactness on the
  deterministic paths (`top_k=1`, one-hot logits) where the sampler's stream is irrelevant.
- **Bit-exactness across batch size is not promised.** Reductions may be batched differently.

## NaN / Inf Policy

Checked on entry when `check_inputs=True`:

| Input | Policy |
|---|---|
| `NaN` anywhere in a row | **Reject** — `ValueError`. There is no defined sampling behavior. |
| `+inf` anywhere | **Reject** — `ValueError`. `ℓ - ℓ_max` gives `inf - inf = NaN`. |
| `-inf` in some entries | **Legal and supported.** This is the standard banned-token mask; `exp` gives exactly `0`, so the token has zero probability and can never be sampled. |
| entire row `-inf` | **Reject** — `ValueError`. Softmax is `0/0`; no token is samplable. |
| very large finite logits | **Supported.** Max subtraction bounds the exponent at `0`, so no overflow. The result degenerates to one-hot, which is correct. |
| very negative finite logits | **Supported.** `exp` underflows to `0`; the top-1 candidate always survives the mask, so the nucleus is never empty. |

**Passing an invalid input with `check_inputs=False` is undefined behavior, and on CUDA it is
process-fatal.** `torch.multinomial` guards its input with a device-side assert
(`probability tensor contains either inf, nan or element < 0`). On CPU that surfaces as a
`RuntimeError`; on CUDA the assert **poisons the CUDA context for the entire process** — verified
2026-08-22 — so every subsequent CUDA op in that process fails, not just the offending call. A long
sweep that turns validation off for timing therefore loses the whole run, not one unit, if a single
bad row reaches it. Validate once outside the timed region; never feed a known-bad tensor with
checks off.

**`check_inputs=True` costs a device-to-host sync** (`.any()` on a GPU tensor). It is the default
because silent NaN propagation into a token id is worse than a slow reference — but **it must be
`False` for any timed run**, and any benchmark that reports a number must record which setting it
used. This is exactly the kind of check a fused kernel gets to skip, so leaving it on would flatter
the kernel against its own baseline.

## Open Items

Not yet decided; nothing below is assumed by the current code.

**Resolved 2026-08-22 — the boundary-tie question.** Fixed-`k` with a lowest-id tie-break, and
the gate split into Gate A (exact vs the reference) and Gate B (fidelity vs FP32). Decided on
measurement, not preference: fixed-`k` is closer to FP32 truth than HF in 18/18 configurations, and
variable-`k` has no safe static bound — an all-equal-logits row makes HF retain all 128,256 tokens,
so a variable design needs a cap anyway and inherits the fixed-`K` constraint plus a spill path
that defeats register residency. Full reasoning in the tie policy above.

**Other open items:**

**Settled 2026-08-22 by the benchmark rig** (`results/DECISION.md`):

- **Temperature stays outside the timed operator.** HF runs it as a separate full-vocabulary
  warper; including it would inflate the baseline with work a fused kernel absorbs for free.
- **The baseline ladder is built and measured**: `hf_eager`, `ref_eager_fullsort`, `tight_eager`,
  `compile`, `graph_eager`, `graph_compile`, `flashinfer`, `flashinfer_from_probs`.
- **The performance anchor is `V=151936, k=50, p=0.90`** — `regime.ANCHOR_*`, the larger vocabulary
  and so the conservative point for a latency claim. The tie-fidelity artifact was measured at
  `V=128256` (`regime.FIDELITY_VOCAB`); every artifact records its own vocabulary.
- **The bar for the kernel is `flashinfer_from_probs`, not `hf_eager`.**

**Still open:**

- Whether per-sequence `top_k` / `top_p` vectors ever enter scope — likely forced by Phase 2
- When Phase 2 begins, and whether the contract moves to FlashInfer or vLLM. FlashInfer is
  measured as a rung today but is not the semantic contract; adopting it would change the top-p
  definition, which is a semantics change and needs its own decision.
