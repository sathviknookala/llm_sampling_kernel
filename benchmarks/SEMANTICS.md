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
and **this section gets rewritten before any number is quoted against it**. Neither `vllm` nor
`flashinfer` is installed, so nothing about their internals is asserted here. The questions the
shift has to answer, each verified against installed source the same way HF was:

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

### What conformance means

Bit-exact agreement with the contract engine is **not** the gate — see RNG and Determinism, and
see the tie policy for why it is not even available in the target dtypes. The gate is:

1. **Exact distributional agreement** where no tie straddles a decision boundary. On FP32 logits
   this is 100% of rows and holds to ~1e-8.
2. **Bounded total-variation distance** otherwise, with the bound measured and recorded, not
   assumed.
3. **Exact agreement on the deterministic paths** — `top_k=1`, one-hot logits — where the
   sampler's RNG stream is irrelevant.

`tests/test_hf_conformance.py` is that gate for Phase 1.

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

**Tie policy: exactly `k` candidates, values deterministic, index selection among ties
implementation-defined.** Which of several equal-valued tokens is retained is not specified, and
`torch.topk` does not document its tie-break. The kernel must match *retained logit values*, not
*retained indices*, and must be self-consistent (identical output for identical input, shape, and
seed on a given device). Tests assert on values and distributions, never on tie-broken indices.

**This is the one place the Phase 1 contract is knowingly broken, and it is not a corner case.**

HF's top-k is *value-thresholded*: `scores < topk(scores, k).values[..., -1]`. Every token tied
with the k-th value survives, so HF can retain **more than k** candidates. A fixed-`K`,
register-resident kernel — the entire premise of this project — structurally cannot. Measured at
`V=128256, k=50, batch=256` on 2026-08-22:

| dtype | rows with a boundary tie | HF candidates retained | rows differing from HF | max TV distance | mean TV |
|---|---|---|---|---|---|
| FP32 | ~0% | 50 | ~0% | ~1e-8 | ~1e-8 |
| FP16 | 24-30% | 50-52 | 13-14% | 0.013 | 0.0009 |
| BF16 | 95-96% | 51-55 | 62-71% | 0.037 | 0.006 |

Tie-free rows — no duplicate among the k candidates *and* no outside token equal to the k-th value
— are **100% of FP32 rows, ~2-4% of FP16 rows, and 0% of BF16 rows** at `V=4000, k=50`. Exact HF
conformance is therefore not available in the target dtypes at all; it is an FP32-only property.

There are two boundaries a tie can straddle, and both were observed:
- **the k-boundary** — HF keeps all tied tokens, this repo keeps exactly `k`
- **the top-p cutoff** — with equal probabilities inside the nucleus, HF's ascending sort and this
  repo's descending `topk` disagree about *which* tied token is the last one kept. Same nucleus
  size, same probability values, different token id.

The divergence is bounded and small because tied tokens at a boundary are by construction the
lowest-probability members of the nucleus. **The contract is therefore distributional, not
row-exact** — `test_half_precision_divergence_from_hf_stays_bounded` asserts TV < 0.05. Whether
that is the right resolution is an open decision, recorded below.

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

**Needs a decision before kernel work — the boundary-tie resolution.** Three options, and they
imply different kernel architectures:

1. **Exactly `k`, ties broken arbitrarily** (current behavior). Keeps the fixed-`K` register design
   intact. Accepts a bounded distributional gap from HF: TV up to 0.037 in BF16, affecting ~2/3 of
   rows. The gap shrinks to nothing in FP32.
2. **Match HF exactly** — retain every token tied at the k-boundary. Requires a variable-length
   candidate set, which is the specific thing a register-resident fixed-`K` design cannot do.
   Abandons the project's central performance premise to match a behavior that is arguably an
   artifact of HF's threshold formulation rather than an intended semantic.
3. **Declare the gap and gate on it** — keep option 1, but make the measured TV bound a committed
   artifact and a regression test, and state it in any write-up.

Option 3 is what the code currently does; option 1 vs 2 is the user's call, and it should be made
before the kernel's candidate-management strategy is fixed.

**Other open items:**

- Whether per-sequence `top_k` / `top_p` vectors ever enter scope — likely forced by Phase 2
- Whether temperature scaling is fused in or stays a caller-side multiply (HF applies it first, as
  a separate warper)
- The baseline ladder rungs beyond HF eager (`torch.compile`, CUDA graphs, the tight eager
  reference as a fusion-isolating rung, staged custom variants)
- The anchor model whose `V` the headline sweep quotes
- When Phase 2 begins, and which engine — this decides whether the Phase 1 headline survives
