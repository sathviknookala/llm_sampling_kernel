#include <algorithm>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include "fused_sampling.cuh"

namespace {

constexpr int BLOCK = 512;
constexpr int MERGE_BLOCK = 512;  // measured: 512 beats 1024 by ~7% at B=32 and ties at B=1; 128 is far worse
constexpr int NSUB = 8;
constexpr int TIE_CAP = 2048;
constexpr int MERGE_CAP = 1024;
constexpr int K_CAP = 128;
// the last phase of each kernel; the ablation probes instantiate the lower values
constexpr int PARTIAL_ALL = 5;
constexpr int MERGE_ALL = 2;

#define SUF(b) suf[255 - (b)]

// one traversal of this block's slice; vector chunks first, then the ragged tail
template <typename F>
__device__ __forceinline__ void foreach_key(const uint16_t* __restrict__ row, int vlo, int vhi,
                                            int elo, int ehi, F f) {
  const uint4* rv = reinterpret_cast<const uint4*>(row);
  for (int i = vlo + threadIdx.x; i < vhi; i += blockDim.x) {
    uint4 v = rv[i];
    const uint16_t* h = reinterpret_cast<const uint16_t*>(&v);
    const int base = i * 8;
#pragma unroll
    for (int j = 0; j < 8; ++j) f(base + j, fs::mono_key(h[j]));
  }
  for (int i = elo + threadIdx.x; i < ehi; i += blockDim.x) f(i, fs::mono_key(row[i]));
}

// exact resolution of a k-boundary tie set too large to buffer. bucket the index into <=256 bins
// of 2^shift, find the bin holding the need-th smallest, take every tie below it whole, and order
// only within that bin -- via a bitmap that cannot overflow, since the bin is 2^shift wide.
// order inside cand does not matter: the merge sorts every candidate anyway.
template <typename F>
__device__ __forceinline__ void exact_ties(F foreach, uint32_t* hist, uint32_t* suf, uint32_t* bm,
                                           uint64_t* cand, uint32_t T, int shift, int n_gt,
                                           int need, int* s_bb, int* s_below, int* s_cnt) {
  const int t = threadIdx.x;
  const int sub = (t >> 6) & (NSUB - 1);
  for (int i = t; i < NSUB * 256; i += BLOCK) hist[i] = 0u;
  __syncthreads();
  foreach([&](int i, uint32_t k) {
    if (k == T) atomicAdd(&hist[sub * 256 + (i >> shift)], 1u);
  });
  __syncthreads();
  {
    uint32_t v = 0;
    if (t < 256) {
#pragma unroll
      for (int q = 0; q < NSUB; ++q) v += hist[q * 256 + t];
    }
    __syncthreads();
    if (t < 256) hist[t] = v;
  }
  __syncthreads();

  fs::suffix_sum_256(hist, suf);
  // ascending prefix from the descending scan: pre[b] = total - sum_{j>b}
  const uint32_t total = SUF(0);
  if (t < 256) {
    const uint32_t pre = total - ((t == 255) ? 0u : SUF(t + 1));
    const uint32_t prev = total - SUF(t);
    if (pre >= static_cast<uint32_t>(need) && prev < static_cast<uint32_t>(need)) {
      *s_bb = t;
      *s_below = static_cast<int>(prev);
    }
  }
  const int nwords = (1 << shift) >> 5;
  for (int i = t; i < nwords; i += BLOCK) bm[i] = 0u;
  if (t == 0) *s_cnt = 0;
  __syncthreads();

  const int bb = *s_bb, n_below = *s_below, base = bb << shift;
  foreach([&](int i, uint32_t k) {
    if (k != T) return;
    const int b = i >> shift;
    if (b < bb) {
      cand[n_gt + atomicAdd(s_cnt, 1)] = fs::pack(T, i);
    } else if (b == bb) {
      atomicOr(&bm[(i - base) >> 5], 1u << ((i - base) & 31));
    }
  });
  __syncthreads();

  if (t < 32) {
    const int lane = t;
    const int want = need - n_below;
    int written = 0;
    for (int w0 = 0; w0 < nwords && written < want; w0 += 32) {
      uint32_t word = (w0 + lane < nwords) ? bm[w0 + lane] : 0u;
      const int c = __popc(word);
      int incl = c;
#pragma unroll
      for (int off = 1; off < 32; off <<= 1) {
        const int o = __shfl_up_sync(0xFFFFFFFFu, incl, off);
        if (lane >= off) incl += o;
      }
      int slot = written + incl - c;
      while (word) {
        const int b = __ffs(word) - 1;
        word &= word - 1;
        if (slot < want) cand[n_gt + n_below + slot] = fs::pack(T, base + (w0 + lane) * 32 + b);
        ++slot;
      }
      written += __shfl_sync(0xFFFFFFFFu, incl, 31);
    }
  }
  __syncthreads();
}

// exact top-K of one slice of one row, by packed (key desc, index asc).
// two 8-bit radix passes resolve the 16-bit key; the k-boundary tie then resolves on index.
// STOP runs phases 1..STOP and sinks live state to `out`, so a phase costs a difference of two
// measured timings rather than an inference -- ncu is blocked on this box. PARTIAL_ALL is production.
template <int STOP>
__global__ void topk_partial_kernel(const uint16_t* __restrict__ x, uint64_t* __restrict__ partial,
                                    int vocab, int n_vec_total, int splits, int K, int shift) {
  // one allocation, so every STOP instantiation reserves the same shared memory -- otherwise the
  // compiler drops the arrays a truncated phase never reaches and the ablation compares occupancies
  constexpr int HIST_W = NSUB * 256;
  __shared__ __align__(16) uint32_t smem[HIST_W + 256 + TIE_CAP + K_CAP * 2];
  uint32_t* hist = smem;
  uint32_t* suf = hist + HIST_W;
  uint32_t* tie = suf + 256;
  uint64_t* cand = reinterpret_cast<uint64_t*>(tie + TIE_CAP);
  __shared__ int s_hb, s_lb, s_n_above, s_n_gt, s_n_out, s_n_tie, s_bb, s_below, s_cnt;

  const int t = threadIdx.x;
  const int s = blockIdx.x;
  const uint16_t* row = x + static_cast<size_t>(blockIdx.y) * vocab;
  const int sub = (t >> 6) & (NSUB - 1);

  int vlo = 0, vhi = 0, elo = 0, ehi = 0;
  if (n_vec_total > 0) {
    const int per = (n_vec_total + splits - 1) / splits;
    vlo = min(s * per, n_vec_total);
    vhi = min(vlo + per, n_vec_total);
    elo = (s == splits - 1) ? n_vec_total * 8 : vocab;
    ehi = vocab;
  } else {
    const int per = (vocab + splits - 1) / splits;
    elo = min(s * per, vocab);
    ehi = min(elo + per, vocab);
  }
  const int count = (vhi - vlo) * 8 + (ehi - elo);
  uint64_t* out = partial + (static_cast<size_t>(blockIdx.y) * splits + s) * K;
  // ceil-divided slices can leave a trailing split empty; padding sorts below every real packed
  // value, so an empty split simply contributes nothing to the merge
  if (count == 0) {
    for (int i = t; i < K; i += BLOCK) out[i] = 0ull;
    return;
  }
  const int keff = min(K, count);

  for (int i = t; i < HIST_W; i += BLOCK) hist[i] = 0u;
  __syncthreads();
  foreach_key(row, vlo, vhi, elo, ehi,
              [&](int, uint32_t k) { atomicAdd(&hist[sub * 256 + (k >> 8)], 1u); });
  __syncthreads();
  if constexpr (STOP == 1) {
    for (int i = t; i < K; i += BLOCK) out[i] = hist[(i & (NSUB - 1)) * 256 + i];
    return;
  }
  {
    uint32_t v = 0;
    if (t < 256) {
#pragma unroll
      for (int q = 0; q < NSUB; ++q) v += hist[q * 256 + t];
    }
    __syncthreads();
    if (t < 256) hist[t] = v;
  }
  __syncthreads();

  fs::suffix_sum_256(hist, suf);
  if (t < 256 && SUF(t) >= static_cast<uint32_t>(keff) &&
      (t == 255 || SUF(t + 1) < static_cast<uint32_t>(keff))) {
    s_hb = t;
    s_n_above = (t == 255) ? 0 : static_cast<int>(SUF(t + 1));
  }
  __syncthreads();
  const int hb = s_hb;
  const int n_above = s_n_above;
  if constexpr (STOP == 2) {
    for (int i = t; i < K; i += BLOCK)
      out[i] = (i == 0) ? static_cast<uint64_t>(hb)
                        : (i == 1 ? static_cast<uint64_t>(n_above) : 0ull);
    return;
  }

  for (int i = t; i < HIST_W; i += BLOCK) hist[i] = 0u;
  __syncthreads();
  foreach_key(row, vlo, vhi, elo, ehi, [&](int, uint32_t k) {
    if (static_cast<int>(k >> 8) == hb) atomicAdd(&hist[sub * 256 + (k & 0xFFu)], 1u);
  });
  __syncthreads();
  if constexpr (STOP == 3) {
    for (int i = t; i < K; i += BLOCK) out[i] = hist[(i & (NSUB - 1)) * 256 + i];
    return;
  }
  {
    uint32_t v = 0;
    if (t < 256) {
#pragma unroll
      for (int q = 0; q < NSUB; ++q) v += hist[q * 256 + t];
    }
    __syncthreads();
    if (t < 256) hist[t] = v;
  }
  __syncthreads();

  fs::suffix_sum_256(hist, suf);
  const int want = keff - n_above;
  if (t < 256 && SUF(t) >= static_cast<uint32_t>(want) &&
      (t == 255 || SUF(t + 1) < static_cast<uint32_t>(want))) {
    s_lb = t;
    s_n_gt = n_above + ((t == 255) ? 0 : static_cast<int>(SUF(t + 1)));
  }
  if (t == 0) {
    s_n_out = 0;
    s_n_tie = 0;
  }
  __syncthreads();

  const uint32_t T = (static_cast<uint32_t>(hb) << 8) | static_cast<uint32_t>(s_lb);
  const int n_gt = s_n_gt;
  const int need = keff - n_gt;
  if constexpr (STOP == 4) {
    for (int i = t; i < K; i += BLOCK)
      out[i] = (i == 0) ? static_cast<uint64_t>(T) : (i == 1 ? static_cast<uint64_t>(n_gt) : 0ull);
    return;
  }

  foreach_key(row, vlo, vhi, elo, ehi, [&](int i, uint32_t k) {
    if (k > T) {
      cand[atomicAdd(&s_n_out, 1)] = fs::pack(k, i);
    } else if (k == T) {
      const int slot = atomicAdd(&s_n_tie, 1);
      // boundary ties are the common case in bf16 but multiplicity is small (median 4, max 16 in
      // results/raw/tie_fidelity.csv). past TIE_CAP the retained *values* are still right, only
      // which tied id is emitted changes -- see results/SPIKE.md
      if (slot < TIE_CAP) tie[slot] = static_cast<uint32_t>(i);
    }
  });
  __syncthreads();

  if (need > 0) {
    if (s_n_tie <= TIE_CAP) {
      const int n_tie = s_n_tie;
      int p2 = 1;
      while (p2 < n_tie) p2 <<= 1;
      for (int i = t + n_tie; i < p2; i += BLOCK) tie[i] = 0xFFFFFFFFu;
      __syncthreads();
      if (p2 > 1) fs::bitonic_ascending<uint32_t>(tie, p2);
      for (int i = t; i < need; i += BLOCK) cand[n_gt + i] = fs::pack(T, tie[i]);
      __syncthreads();
    } else {
      // never seen on real logits (max tie multiplicity 14 in results/raw/tie_fidelity.csv), so
      // the two extra traversals cost nothing in practice and the answer is exact regardless
      auto foreach = [&](auto f) { foreach_key(row, vlo, vhi, elo, ehi, f); };
      exact_ties(foreach, hist, suf, tie, cand, T, shift, n_gt, need, &s_bb, &s_below, &s_cnt);
    }
  }

  for (int i = t; i < K; i += BLOCK) out[i] = (i < keff) ? cand[i] : 0ull;
}

__device__ __forceinline__ void merge_sort_shared(const uint64_t* __restrict__ src, uint64_t* buf,
                                                  int n, int P) {
  for (int i = threadIdx.x; i < P; i += blockDim.x) buf[i] = (i < n) ? src[i] : 0ull;
  __syncthreads();
  if (P > 1) fs::bitonic_ascending<uint64_t>(buf, P);
}

template <bool IS_BF16, int STOP>
__global__ void merge_sample_kernel(const uint64_t* __restrict__ partial,
                                    int64_t* __restrict__ out, int splits, int K, int P,
                                    float top_p, uint64_t seed, uint64_t offset) {
  __shared__ uint64_t buf[MERGE_CAP];
  __shared__ float w[K_CAP];
  __shared__ float cum[K_CAP];

  const int t = threadIdx.x;
  const int b = blockIdx.x;
  merge_sort_shared(partial + static_cast<size_t>(b) * splits * K, buf, splits * K, P);
  if constexpr (STOP == 1) {
    if (t == 0) out[b] = static_cast<int64_t>(fs::unpack_idx(buf[P - 1]));
    return;
  }

  // ascending sort, so rank r of the descending top-K sits at P-1-r
  const float m = fs::key_to_float<IS_BF16>(fs::unpack_key(buf[P - 1]));
  for (int i = t; i < K; i += blockDim.x) {
    w[i] = expf(fs::key_to_float<IS_BF16>(fs::unpack_key(buf[P - 1 - i])) - m);
  }
  __syncthreads();

  if (t == 0) {
    float z = 0.0f;
    for (int i = 0; i < K; ++i) {
      z += w[i];
      cum[i] = z;
    }
    // reference.py keeps i iff the exclusive prefix of the normalized probs is < top_p;
    // scaling by z avoids materializing them
    const float cut = top_p * z;
    int r = K - 1;
    if (top_p < 1.0f) {
      for (int i = 1; i < K; ++i) {
        if (cum[i - 1] >= cut) {
          r = i - 1;
          break;
        }
      }
    }
    const float u = fs::rng_uniform(seed, offset, static_cast<uint32_t>(b)) * cum[r];
    int pick = r;
    for (int i = 0; i <= r; ++i) {
      if (cum[i] >= u) {
        pick = i;
        break;
      }
    }
    out[b] = static_cast<int64_t>(fs::unpack_idx(buf[P - 1 - pick]));
  }
}

__global__ void merge_ids_kernel(const uint64_t* __restrict__ partial, int64_t* __restrict__ ids,
                                 int splits, int K, int P) {
  __shared__ uint64_t buf[MERGE_CAP];
  merge_sort_shared(partial + static_cast<size_t>(blockIdx.x) * splits * K, buf, splits * K, P);
  int64_t* out = ids + static_cast<size_t>(blockIdx.x) * K;
  for (int i = threadIdx.x; i < K; i += blockDim.x)
    out[i] = static_cast<int64_t>(fs::unpack_idx(buf[P - 1 - i]));
}

int next_pow2(int n) {
  int p = 1;
  while (p < n) p <<= 1;
  return p;
}

struct Plan {
  int batch, vocab, K, n_vec, splits, P, shift;
};

Plan make_plan(const torch::Tensor& logits, int64_t top_k, int64_t splits_override) {
  TORCH_CHECK(logits.is_cuda() && logits.dim() == 2 && logits.is_contiguous(),
              "logits must be a contiguous cuda [B, V] tensor");
  TORCH_CHECK(logits.scalar_type() == at::kBFloat16 || logits.scalar_type() == at::kHalf,
              "logits must be float16 or bfloat16");
  TORCH_CHECK(top_k >= 1, "top_k must be >= 1");
  Plan p;
  p.batch = static_cast<int>(logits.size(0));
  p.vocab = static_cast<int>(logits.size(1));
  p.K = static_cast<int>(std::min<int64_t>(top_k, p.vocab));
  TORCH_CHECK(p.K <= K_CAP, "top_k must be <= ", K_CAP, " (regime max is 100)");
  const bool aligned =
      (reinterpret_cast<uintptr_t>(logits.data_ptr()) % 16 == 0) && (p.vocab % 8 == 0);
  p.n_vec = aligned ? p.vocab / 8 : 0;
  // splits are bounded by the merge buffer and by how many blocks the batch already provides;
  // the second bound comes from results/raw/kernel_floor.csv
  // 2^shift-wide index buckets, at most 256 of them; the exact-tie bitmap is then 2^shift bits
  p.shift = 0;
  while ((p.vocab + (1 << p.shift) - 1) >> p.shift > 256) ++p.shift;
  TORCH_CHECK((1 << p.shift) <= TIE_CAP * 32, "vocabulary too large for the exact-tie bitmap");
  const int units = p.n_vec > 0 ? p.n_vec : p.vocab;
  // measured: results/raw/kernel_splits.csv. the kernel is latency-bound on its phase chain,
  // not throughput-bound, so more blocks stops paying well before the SMs are full
  const int autos = std::min(8, std::max(4, 128 / p.batch));
  p.splits = splits_override > 0 ? static_cast<int>(splits_override) : autos;
  p.splits = std::max(1, std::min({p.splits, MERGE_CAP / p.K, units}));
  p.P = next_pow2(p.splits * p.K);
  return p;
}

}  // namespace

#undef SUF

torch::Tensor sample_fused(torch::Tensor logits, int64_t top_k, double top_p, int64_t seed,
                           int64_t offset, int64_t splits_override) {
  TORCH_CHECK(top_p > 0.0 && top_p <= 1.0, "top_p must be in (0, 1]");
  const Plan p = make_plan(logits, top_k, splits_override);
  const bool is_bf16 = logits.scalar_type() == at::kBFloat16;

  auto out = torch::empty({p.batch}, logits.options().dtype(torch::kInt64));
  auto partial = torch::empty({p.batch, p.splits, p.K}, logits.options().dtype(torch::kInt64));
  auto stream = at::cuda::getCurrentCUDAStream();

  topk_partial_kernel<PARTIAL_ALL><<<dim3(p.splits, p.batch), BLOCK, 0, stream>>>(
      reinterpret_cast<const uint16_t*>(logits.data_ptr()),
      reinterpret_cast<uint64_t*>(partial.data_ptr()), p.vocab, p.n_vec, p.splits, p.K, p.shift);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const auto* pp = reinterpret_cast<const uint64_t*>(partial.data_ptr());
  if (is_bf16) {
    merge_sample_kernel<true, MERGE_ALL><<<p.batch, MERGE_BLOCK, 0, stream>>>(
        pp, out.data_ptr<int64_t>(), p.splits, p.K, p.P, static_cast<float>(top_p),
        static_cast<uint64_t>(seed), static_cast<uint64_t>(offset));
  } else {
    merge_sample_kernel<false, MERGE_ALL><<<p.batch, MERGE_BLOCK, 0, stream>>>(
        pp, out.data_ptr<int64_t>(), p.splits, p.K, p.P, static_cast<float>(top_p),
        static_cast<uint64_t>(seed), static_cast<uint64_t>(offset));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// debug entry point: the candidate set the kernel actually selected, for a Gate-A style check
torch::Tensor topk_fused(torch::Tensor logits, int64_t top_k, int64_t splits_override) {
  const Plan p = make_plan(logits, top_k, splits_override);
  auto ids = torch::empty({p.batch, p.K}, logits.options().dtype(torch::kInt64));
  auto partial = torch::empty({p.batch, p.splits, p.K}, logits.options().dtype(torch::kInt64));
  auto stream = at::cuda::getCurrentCUDAStream();

  topk_partial_kernel<PARTIAL_ALL><<<dim3(p.splits, p.batch), BLOCK, 0, stream>>>(
      reinterpret_cast<const uint16_t*>(logits.data_ptr()),
      reinterpret_cast<uint64_t*>(partial.data_ptr()), p.vocab, p.n_vec, p.splits, p.K, p.shift);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  merge_ids_kernel<<<p.batch, MERGE_BLOCK, 0, stream>>>(
      reinterpret_cast<const uint64_t*>(partial.data_ptr()), ids.data_ptr<int64_t>(), p.splits,
      p.K, p.P);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return ids;
}

// ablation probe: run phases 1..phase of the pipeline and stop. differences between adjacent
// phases give a measured cost breakdown, which is the only attribution available with ncu blocked.
torch::Tensor probe_phase(torch::Tensor logits, int64_t top_k, double top_p, int64_t phase,
                          int64_t splits_override) {
  TORCH_CHECK(phase >= 1 && phase <= PARTIAL_ALL + MERGE_ALL, "phase must be in [1, ",
              PARTIAL_ALL + MERGE_ALL, "]");
  const Plan p = make_plan(logits, top_k, splits_override);
  const bool is_bf16 = logits.scalar_type() == at::kBFloat16;

  // both tensors are allocated at every phase so the allocator's cost is constant across the sweep
  auto out = torch::empty({p.batch}, logits.options().dtype(torch::kInt64));
  auto partial = torch::empty({p.batch, p.splits, p.K}, logits.options().dtype(torch::kInt64));
  auto stream = at::cuda::getCurrentCUDAStream();
  const auto* x = reinterpret_cast<const uint16_t*>(logits.data_ptr());
  auto* pw = reinterpret_cast<uint64_t*>(partial.data_ptr());
  const dim3 grid(p.splits, p.batch);

  switch (phase) {
    case 1: topk_partial_kernel<1><<<grid, BLOCK, 0, stream>>>(x, pw, p.vocab, p.n_vec, p.splits, p.K, p.shift); break;
    case 2: topk_partial_kernel<2><<<grid, BLOCK, 0, stream>>>(x, pw, p.vocab, p.n_vec, p.splits, p.K, p.shift); break;
    case 3: topk_partial_kernel<3><<<grid, BLOCK, 0, stream>>>(x, pw, p.vocab, p.n_vec, p.splits, p.K, p.shift); break;
    case 4: topk_partial_kernel<4><<<grid, BLOCK, 0, stream>>>(x, pw, p.vocab, p.n_vec, p.splits, p.K, p.shift); break;
    default: topk_partial_kernel<PARTIAL_ALL><<<grid, BLOCK, 0, stream>>>(x, pw, p.vocab, p.n_vec, p.splits, p.K, p.shift); break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (phase <= PARTIAL_ALL) return out;

  const int ms = static_cast<int>(phase) - PARTIAL_ALL;
  const auto* pr = reinterpret_cast<const uint64_t*>(partial.data_ptr());
  auto* o = out.data_ptr<int64_t>();
  const float tp = static_cast<float>(top_p);
  if (is_bf16) {
    if (ms == 1) merge_sample_kernel<true, 1><<<p.batch, MERGE_BLOCK, 0, stream>>>(pr, o, p.splits, p.K, p.P, tp, 0ull, 0ull);
    else merge_sample_kernel<true, MERGE_ALL><<<p.batch, MERGE_BLOCK, 0, stream>>>(pr, o, p.splits, p.K, p.P, tp, 0ull, 0ull);
  } else {
    if (ms == 1) merge_sample_kernel<false, 1><<<p.batch, MERGE_BLOCK, 0, stream>>>(pr, o, p.splits, p.K, p.P, tp, 0ull, 0ull);
    else merge_sample_kernel<false, MERGE_ALL><<<p.batch, MERGE_BLOCK, 0, stream>>>(pr, o, p.splits, p.K, p.P, tp, 0ull, 0ull);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
