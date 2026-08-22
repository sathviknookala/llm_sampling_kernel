# Decision Gate — Is an SM120 Fused Top-K/Top-P Kernel Justified?

**Verdict: GO, BUT REFRAME.**

Operator-level headroom is real and large. End-to-end decode headroom is not. The kernel is worth
building as an operator-specialization result; it is not worth building as a decode-latency result,
and the write-up must not claim to be one.

All numbers below are from committed artifacts under `results/raw/`, measured on an idle GPU with
`clocks_locked=false` (no permission on this machine). Methodology: `docs/benchmark_methodology.md`.

Anchor configuration unless stated: `V=151936, top_k=50, top_p=0.90, bfloat16, hot` — the residency
condition is discussed in §7.

---

## 1. Is HF eager bottlenecked by DRAM traffic? **No — decisively not.**

`results/raw/sampling_ladder.csv`, `floor_frac_of_latency`. Measured HBM bandwidth on this GPU is
**552.7 GB/s** (device-to-device copy at run time, `results/raw/environment.json`).

| B | logits (bf16) | DRAM read floor | hf_eager | floor as % of latency |
|---|---|---|---|---|
| 1 | 0.30 MB | 0.55 µs | 324.9 µs | **0.17%** |
| 8 | 2.43 MB | 4.40 µs | 706.6 µs | 0.62% |
| 32 | 9.72 MB | 17.6 µs | 2193.4 µs | 0.80% |

Reading the entire vocabulary costs under 1% of HF's runtime at every batch size. The whole logits
tensor also fits in this GPU's **48 MB L2** for every configuration in the regime (largest case
9.7 MB), so even that floor is rarely paid twice.

**The original memory-traffic hypothesis is rejected.** `CLAUDE.md`'s framing — "reducing
intermediate global-memory traffic" with a target of `read [B,V] / write [B]` — aims at a cost that
is not there. Any speedup this project measures comes from somewhere else, and the hypothesis text
should be rewritten before a number is quoted against it.

## 2. What actually dominates HF eager?

`results/raw/stage_profile.csv` (isolated amortized timing per stage; kernel counts from the
profiler). HF eager issues **64–70 CUDA kernels** per decode step's sampling.

| Stage | B=1 | B=32 |
|---|---|---|
| `topk_warper` | 79.2 µs (23.0%) | 185.3 µs (8.5%) |
| `topp_warper` | 146.2 µs (42.5%) | 1790.4 µs (**82.4%**) |
|  └ full `[B,V]` sort | 68.5 µs (19.9%) | 1406.3 µs (**64.7%**) |
|  └ softmax | 55.3 µs (16.1%) | 51.5 µs (2.4%) |
|  └ cumsum | 61.6 µs (17.9%) | 247.9 µs (11.4%) |
|  └ scatter + mask | 25.7 µs (7.5%) | 153.3 µs (7.1%) |
| `final_softmax` | 55.3 µs (16.1%) | 52.0 µs (2.4%) |
| `multinomial` | 68.0 µs (19.8%) | 113.9 µs (5.2%) |
| **TOTAL** | **344.3 µs** | **2173.6 µs** |

Two distinct regimes:

- **At B=32 it is the full-vocabulary sort**, at 65% of everything. HF sorts all 151,936 entries to
  find a nucleus of ~50.
- **At B=1 it is launch- and latency-bound.** No single stage dominates; every stage costs 55–68 µs
  regardless of how little work it does. `final_softmax` over `[1,V]` and `multinomial` over `[1,V]`
  each cost ~55–68 µs, and in the `tight_eager` path `multinomial` over `[1,50]` still costs
  **57.6 µs** — essentially the same as over 151,936. That is pure per-op overhead, not work.

## 3. What does `torch.compile` eliminate? **Almost nothing on its own.**

| B | hf_eager | tight_eager | compile | compile vs tight_eager |
|---|---|---|---|---|
| 1 | 324.9 | 136.1 (2.4×) | 157.6 (2.1×) | **1.16× slower** |
| 8 | 706.6 | 167.6 (4.2×) | 151.6 (4.7×) | 1.11× faster |
| 32 | 2193.4 | 219.2 (10.0×) | 219.1 (10.0×) | 1.00× — identical |

**The 2.4–10× is the algorithm, not the compiler.** `tight_eager` is plain eager PyTorch that uses
`torch.topk` to collapse to `[B,K]` and does every later stage at `K=50` instead of `V=151936`.
That single change captures the entire gain. Inductor on top of it is within noise at B=8/32 and
*slower* at B=1, where its extra prologue is not amortized. Kernel count only falls 43 → 35.

This matters for the project: a large part of what looked like fusion headroom is available from an
ordinary rewrite that any serving engine already does.

## 4. What does CUDA-graph replay add? **A real, modest win, concentrated at low batch.**

| B | tight_eager | graph_eager | graph_compile | graph_compile vs tight_eager |
|---|---|---|---|---|
| 1 | 136.1 | 81.7 | **72.3** | **1.88×** |
| 8 | 167.6 | 108.3 | 102.7 | 1.63× |
| 32 | 219.2 | 168.1 | 168.1 | 1.30× |

Graph replay leaves the device kernels untouched (`results/raw/launch_counts.csv`: still 37–45
kernels inside the graph) and removes only the host-side launch cost. The gain therefore shrinks as
batch grows and real work starts to dominate — exactly the signature of launch-bound behavior at
B=1 predicted by §2.

## 5. The fastest production baseline

FlashInfer 0.6.17, JIT-compiled for `sm_120`. **9–11 kernels**, against HF's 64–70.

| B | `flashinfer` (logits→token) | `flashinfer_from_probs` (probs→token) | best generic rung |
|---|---|---|---|
| 1 | 84.1 µs | **72.6 µs** | 72.3 µs (graph_compile) |
| 8 | 91.0 µs | **73.9 µs** | 102.7 µs |
| 32 | 177.3 µs | **101.2 µs** | 168.1 µs |

**`flashinfer_from_probs` is the bar to beat: 72.6 / 73.9 / 101.2 µs at B=1/8/32.** It is nearly
flat in batch, which is what a properly fused sampler should look like.

Two honest caveats. `flashinfer` (logits→token) includes a full-vocabulary softmax to match every
other rung's contract, and that softmax is most of its gap to `flashinfer_from_probs` — a fused
kernel consuming raw logits would absorb it. And at B=1 the best generic rung (`graph_compile`,
72.3 µs) *matches* FlashInfer; FlashInfer's advantage only opens up at B≥8, where it is 1.4–1.7×
ahead.

## 6. How much headroom remains?

Against the strongest baseline, expressed as achieved fraction of measured HBM bandwidth:

| Rung | B=1 | B=8 | B=32 |
|---|---|---|---|
| `graph_compile` | 0.8% | 4.3% | 10.5% |
| `flashinfer_from_probs` | 1.5% | 11.9% | **34.8%** |

**At B=1 the best available sampler runs at ~1.5% of hardware bandwidth** — 72 µs to consume
0.6 MB. The operation is nowhere near any hardware limit; it is latency- and occupancy-bound, with
a single row of ~152K elements spread across 70 SMs. A specialized kernel that keeps a small
candidate set in registers and makes one pass has a plausible target in the **single-digit to low
tens of microseconds**, i.e. a 5–20× operator win at low batch. That is genuine, identifiable
headroom and it is the strongest argument for the project.

**At B=32 the headroom is much thinner.** FlashInfer already achieves 34.8% of HBM bandwidth, so at
most ~3× remains before hitting a memory wall. The specialization thesis is a *low-batch* thesis,
which is consistent with the target regime but should be stated as a limit, not glossed.

Concretely, work the strongest baseline still performs that a fused kernel could remove: a
full-vocabulary softmax materialized to `[B,V]` before sampling; a separate multinomial pass whose
fixed cost (~57 µs even over `[B,50]`) dwarfs its work; and per-stage launch/latency floors that
`graph_compile` only partly hides.

## 7. Does hot vs cold L2 residency matter? **Slightly, and it does not change any conclusion.**

Cold = rotating through 4× L2 worth of distinct buffers so nothing is re-read from cache.

| Rung | median cold penalty | worst |
|---|---|---|
| `hf_eager` | −0.2% | +0.3% |
| `tight_eager` | +1.3% | +4.9% |
| `compile` | +1.8% | +4.9% |
| `graph_eager` | +6.9% | +14.8% |
| `graph_compile` | +7.1% | +15.5% |
| `flashinfer` | +3.0% | +9.7% |
| `flashinfer_from_probs` | +2.2% | +26.1% |

The slow rungs are indifferent — they are so far from memory-bound that residency is invisible. The
fast rungs feel it, which is the expected direction: the closer a rung gets to the memory floor, the
more the floor matters. The graph rungs' larger penalty is partly an artifact — CUDA-graph replay
needs a fixed input address, so their cold variant pays an extra `[B,V]` copy
(`cold_method=rotate_into_static`), and that copy is included in the number.

**Neither condition is "correct."** In real decode the LM head writes the logits immediately before
sampling reads them, so hot is arguably the more representative case. Nothing in the go/no-go turns
on the choice.

## 8. How much of real decode is sampling?

`results/raw/amdahl_probe.csv` — measured decode forward with a populated KV cache, prompt length
512, bf16, real weights.

| Model | V | decode step | vs weight-read floor | best sampler | sampling % of step |
|---|---|---|---|---|---|
| Mistral-7B-v0.1 | 32,000 | 28.18 ms (B=1) | 1.07× | 45.7 µs | **0.16%** |
| Mistral-7B-v0.1 | 32,000 | 33.13 ms (B=8) | 1.26× | 51.8 µs | 0.16% |
| Qwen2-0.5B | 151,936 | 7.27 ms (B=1) | 4.06× | 85.4 µs | **1.18%** |
| Qwen2-0.5B | 151,936 | 7.61 ms (B=8) | 4.25× | 98.5 µs | 1.29% |

Mistral-7B's decode step lands at 1.07× its weight-read floor, which validates the bandwidth model:
decode at low batch is bandwidth-bound on weights, and sampling is a rounding error beside it.

Amdahl ceiling on end-to-end decode, using the best available sampler as the starting point:

| Sampler speedup | Mistral-7B (B=1) | Qwen2-0.5B (B=1) |
|---|---|---|
| 2× | 0.08% | 0.59% |
| 5× | 0.13% | 0.94% |
| 10× | 0.15% | 1.06% |
| ∞ (free sampling) | **0.16%** | **1.18%** |

**Even an infinitely fast sampler saves 0.16–1.2% of a decode step.** Qwen2-0.5B is the favourable
end of the range — a 0.5B model with a 152K vocab is the most sampling-heavy realistic shape, and
its decode step sits 4× above its own bandwidth floor. A 7B-class model with a 152K vocab would put
sampling near ~0.3% of decode.

## 9. Recommendation: **GO, BUT REFRAME**

**Why not NO-GO.** The headroom is real and unusually clean: the best production sampler runs at
1.5% of hardware bandwidth at B=1, and 72 µs to process 0.6 MB is far from any limit. There is
identifiable work to remove — the materialized full-vocab softmax, the fixed-cost multinomial, the
per-stage latency floors. A 5–20× operator win at low batch is a defensible target, and the project
now has a measured, adversarial baseline (`flashinfer_from_probs`) to prove it against rather than
an eager strawman.

**Why not plain GO.** The hypothesis in `CLAUDE.md` is wrong in two ways that measurement settled:

1. **It is not a memory-traffic problem.** DRAM traffic is ≤0.8% of HF eager. The line
   "reducing full-vocabulary processing overhead, kernel launches, intermediate global-memory
   traffic" should drop the memory clause and lead with launch/latency floors and the avoided
   full-vocabulary sort.
2. **It cannot claim decode-latency improvement.** At 0.16–1.2% of a decode step, no sampler win is
   visible end to end. This is the same wall `decode_llm_kernel` hit, and it should be stated in the
   project's framing now rather than discovered in the write-up.

**Conditions for the reframe:**

- The headline metric is **operator latency vs `flashinfer_from_probs`**, not vs HF eager, and not
  end-to-end tokens/sec. A speedup against HF eager is not a result — `tight_eager`, 30 lines of
  ordinary PyTorch, already gets 10× at B=32.
- The claim is **low-batch specialization**. At B=32 FlashInfer is already at 34.8% of HBM and at
  most ~3× remains.
- Any decode-level number must be quoted with its Amdahl ceiling attached.

**Suggested next step before writing CUDA:** confirm the ~57 µs fixed cost of `torch.multinomial`
over `[B,50]` and the per-stage latency floors are what a single fused kernel actually removes, by
prototyping the sampling tail alone. If a hand-written kernel cannot beat 72 µs at B=1, the project
has its negative result cheaply.

---

## Limitations

- **Clocks could not be locked** (no permission); `clocks_locked=false` on every row. Round-to-round
  spread with rotated implementation ordering was 0.5–2.3%, so this is small but uncontrolled.
- **FlashInfer runs under torch 2.9.1+cu128 in an isolated venv**, because its wheel demands torch
  ≥2.13/CUDA 13 and this driver supports only CUDA 12.9. The entire ladder was re-run inside that
  venv so all rungs are compared under identical torch — no cross-environment comparison is made —
  but the repo's pinned torch is 2.11.0 and margins near 1× should be re-checked.
- **FlashInfer's semantics differ**: it applies top-k/top-p to the full-vocabulary distribution,
  where this repo renormalizes within the top-k survivors first. It is a performance rung only and
  is not gated against `reference.py`.
- **Synthetic Gaussian logits**, not real model logits. Heavy-tailed real logits change tie density
  and top-p cutoff positions, which can move per-stage costs.
- **No `ncu` counters.** DRAM figures are computed floors against a measured copy bandwidth, not
  measured traffic.
- **Amortized timing understates dependent-call launch latency** — consecutive iterations overlap.
  The Amdahl probe measures the dependent case and is the right source for end-to-end claims.
- **`torch.compile` at default mode only**; `max-autotune` was not swept.
