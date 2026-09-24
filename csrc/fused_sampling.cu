#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/detail/UnpackRaw.cuh>
#include <algorithm>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include "fused_sampling.cuh"

namespace {

constexpr int BLOCK = 512;
constexpr int MERGE_BLOCK = 128;
constexpr int NSUB = 8;
constexpr int TIE_CAP = 2048;
constexpr int MERGE_CAP = 1024;
constexpr int K_CAP = 128;
constexpr int PARTIAL_ALL = 5;
constexpr int MERGE_ALL = 2;

#define SUF(b) suf[255 - (b)]

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

template <int STOP>
__global__ void topk_partial_kernel(const uint16_t* __restrict__ x, uint64_t* __restrict__ partial,
                                    int vocab, int n_vec_total, int splits, int K, int shift) {
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
      auto foreach = [&](auto f) { foreach_key(row, vlo, vhi, elo, ehi, f); };
      exact_ties(foreach, hist, suf, tie, cand, T, shift, n_gt, need, &s_bb, &s_below, &s_cnt);
    }
  }

  for (int i = t; i < K; i += BLOCK) out[i] = (i < keff) ? cand[i] : 0ull;
}

template <bool IS_BF16, int STOP, int P>
__global__ void merge_sample_kernel(const uint64_t* __restrict__ partial,
                                    int64_t* __restrict__ out, int splits, int K,
                                    float top_p, uint64_t seed, uint64_t offset,
                                    at::PhiloxCudaState philox, bool from_gen,
                                    bool* __restrict__ keep_out, float* __restrict__ renormed_out) {
  __shared__ uint64_t buf[P];
  __shared__ float w[K_CAP];
  __shared__ float cum[K_CAP];
  __shared__ float s_z;

  const int t = threadIdx.x;
  const int b = blockIdx.x;
  fs::warp_merge_sort<P>(partial + static_cast<size_t>(b) * splits * K, buf, splits * K, K);
  if constexpr (STOP == 1) {
    if (t == 0) out[b] = static_cast<int64_t>(fs::unpack_idx(buf[P - 1]));
    return;
  }

  const float m = fs::key_to_float<IS_BF16>(fs::unpack_key(buf[P - 1]));
  for (int i = t; i < K; i += blockDim.x) {
    w[i] = expf(fs::key_to_float<IS_BF16>(fs::unpack_key(buf[P - 1 - i])) - m);
  }
  __syncthreads();

  if (t == 0) {
    float z = 0.0f;
    for (int i = 0; i < K; ++i) z += w[i];
    s_z = z;
  }
  __syncthreads();
  for (int i = t; i < K; i += blockDim.x) w[i] /= s_z;
  __syncthreads();

  if (t == 0) {
    float c = 0.0f;
    for (int i = 0; i < K; ++i) {
      c += w[i];
      cum[i] = c;
    }
    int r = K - 1;
    if (top_p < 1.0f) {
      for (int i = 1; i < K; ++i) {
        if (!((cum[i] - w[i]) < top_p)) {
          r = i - 1;
          break;
        }
      }
    }
    const float zp = cum[r];
    uint64_t rs = seed, ro_ = offset;
    if (from_gen) {
      const auto st = at::cuda::philox::unpack(philox);
      rs = std::get<0>(st);
      ro_ = std::get<1>(st);
    }
    const float u = fs::rng_uniform(rs, ro_, static_cast<uint32_t>(b)) * zp;
    int pick = r;
    for (int i = 0; i <= r; ++i) {
      if (cum[i] >= u) {
        pick = i;
        break;
      }
    }
    out[b] = static_cast<int64_t>(fs::unpack_idx(buf[P - 1 - pick]));
    if (keep_out != nullptr) {
      bool* ko = keep_out + static_cast<size_t>(b) * K;
      float* ro = renormed_out + static_cast<size_t>(b) * K;
      for (int i = 0; i < K; ++i) {
        ko[i] = (i <= r);
        ro[i] = (i <= r) ? w[i] / zp : 0.0f;
      }
    }
  }
}

template <int P>
__global__ void merge_ids_kernel(const uint64_t* __restrict__ partial, int64_t* __restrict__ ids,
                                 int splits, int K) {
  __shared__ uint64_t buf[P];
  fs::warp_merge_sort<P>(partial + static_cast<size_t>(blockIdx.x) * splits * K, buf, splits * K, K);
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
  p.shift = 0;
  while ((p.vocab + (1 << p.shift) - 1) >> p.shift > 256) ++p.shift;
  TORCH_CHECK((1 << p.shift) <= TIE_CAP * 32, "vocabulary too large for the exact-tie bitmap");
  const int units = p.n_vec > 0 ? p.n_vec : p.vocab;
  const int autos = std::min(8, std::max(4, 128 / p.batch));
  p.splits = splits_override > 0 ? static_cast<int>(splits_override) : autos;
  p.splits = std::max(1, std::min({p.splits, MERGE_CAP / p.K, units}));
  p.P = std::max(32, next_pow2(p.splits * p.K));
  return p;
}

template <typename F>
void dispatch_p(int P, F&& f) {
  switch (P) {
    case 32: f(std::integral_constant<int, 32>{}); break;
    case 64: f(std::integral_constant<int, 64>{}); break;
    case 128: f(std::integral_constant<int, 128>{}); break;
    case 256: f(std::integral_constant<int, 256>{}); break;
    case 512: f(std::integral_constant<int, 512>{}); break;
    case 1024: f(std::integral_constant<int, 1024>{}); break;
    default: TORCH_CHECK(false, "unsupported merge width ", P);
  }
}

template <typename F>
void dispatch_dtype(const torch::Tensor& logits, F&& f) {
  if (logits.scalar_type() == at::kBFloat16) {
    f(std::true_type{});
  } else {
    f(std::false_type{});
  }
}

struct MergeArgs {
  float top_p;
  uint64_t seed, offset;
  at::PhiloxCudaState philox;
  bool from_gen;
  bool* keep_out;
  float* renormed_out;
};

torch::Tensor empty_i64(const torch::Tensor& like, at::IntArrayRef shape) {
  return torch::empty(shape, like.options().dtype(torch::kInt64));
}

template <int STOP>
void launch_partial(const Plan& p, const torch::Tensor& logits, torch::Tensor& partial,
                    cudaStream_t stream) {
  topk_partial_kernel<STOP><<<dim3(p.splits, p.batch), BLOCK, 0, stream>>>(
      reinterpret_cast<const uint16_t*>(logits.data_ptr()),
      reinterpret_cast<uint64_t*>(partial.data_ptr()), p.vocab, p.n_vec, p.splits, p.K, p.shift);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <bool IS_BF16, int STOP>
void launch_merge_sample(const Plan& p, cudaStream_t stream, const torch::Tensor& partial,
                         torch::Tensor& out, const MergeArgs& a) {
  const auto* pp = reinterpret_cast<const uint64_t*>(partial.data_ptr());
  dispatch_p(p.P, [&](auto pv) {
    merge_sample_kernel<IS_BF16, STOP, decltype(pv)::value><<<p.batch, MERGE_BLOCK, 0, stream>>>(
        pp, out.data_ptr<int64_t>(), p.splits, p.K, a.top_p, a.seed, a.offset, a.philox,
        a.from_gen, a.keep_out, a.renormed_out);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void check_top_p(double top_p) {
  TORCH_CHECK(top_p > 0.0 && top_p <= 1.0, "top_p must be in (0, 1]");
}

torch::Tensor run_fused(const torch::Tensor& logits, const Plan& p, const MergeArgs& a) {
  auto out = empty_i64(logits, {p.batch});
  auto partial = empty_i64(logits, {p.batch, p.splits, p.K});
  auto stream = at::cuda::getCurrentCUDAStream();
  launch_partial<PARTIAL_ALL>(p, logits, partial, stream);
  dispatch_dtype(logits, [&](auto bf16) {
    launch_merge_sample<decltype(bf16)::value, MERGE_ALL>(p, stream, partial, out, a);
  });
  return out;
}

}

#undef SUF

static at::PhiloxCudaState philox_state(bool from_gen) {
  if (!from_gen) return at::PhiloxCudaState(0, 0);
  auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
      c10::nullopt, at::cuda::detail::getDefaultCUDAGenerator());
  std::lock_guard<std::mutex> lock(gen->mutex_);
  return gen->philox_cuda_state(4);
}

torch::Tensor sample_fused(torch::Tensor logits, int64_t top_k, double top_p, int64_t seed,
                           int64_t offset, int64_t splits_override) {
  check_top_p(top_p);
  const Plan p = make_plan(logits, top_k, splits_override);
  const bool from_gen = offset < 0;
  const MergeArgs a{static_cast<float>(top_p), static_cast<uint64_t>(seed),
                    static_cast<uint64_t>(from_gen ? 0 : offset), philox_state(from_gen),
                    from_gen, nullptr, nullptr};
  return run_fused(logits, p, a);
}

torch::Tensor topk_fused(torch::Tensor logits, int64_t top_k, int64_t splits_override) {
  const Plan p = make_plan(logits, top_k, splits_override);
  auto ids = empty_i64(logits, {p.batch, p.K});
  auto partial = empty_i64(logits, {p.batch, p.splits, p.K});
  auto stream = at::cuda::getCurrentCUDAStream();
  launch_partial<PARTIAL_ALL>(p, logits, partial, stream);
  const auto* pp = reinterpret_cast<const uint64_t*>(partial.data_ptr());
  dispatch_p(p.P, [&](auto pv) {
    merge_ids_kernel<decltype(pv)::value><<<p.batch, MERGE_BLOCK, 0, stream>>>(
        pp, ids.data_ptr<int64_t>(), p.splits, p.K);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return ids;
}

torch::Tensor probe_phase(torch::Tensor logits, int64_t top_k, double top_p, int64_t phase,
                          int64_t splits_override) {
  TORCH_CHECK(phase >= 1 && phase <= PARTIAL_ALL + MERGE_ALL, "phase must be in [1, ",
              PARTIAL_ALL + MERGE_ALL, "]");
  const Plan p = make_plan(logits, top_k, splits_override);
  auto out = empty_i64(logits, {p.batch});
  auto partial = empty_i64(logits, {p.batch, p.splits, p.K});
  auto stream = at::cuda::getCurrentCUDAStream();

  using PartialLaunch = void (*)(const Plan&, const torch::Tensor&, torch::Tensor&, cudaStream_t);
  constexpr PartialLaunch partial_at[] = {launch_partial<1>, launch_partial<2>, launch_partial<3>,
                                          launch_partial<4>, launch_partial<PARTIAL_ALL>};
  partial_at[std::min<int64_t>(phase, PARTIAL_ALL) - 1](p, logits, partial, stream);
  if (phase <= PARTIAL_ALL) return out;

  const MergeArgs a{static_cast<float>(top_p), 0, 0, at::PhiloxCudaState(0, 0), false, nullptr,
                    nullptr};
  dispatch_dtype(logits, [&](auto bf16) {
    constexpr bool IS_BF16 = decltype(bf16)::value;
    if (phase - PARTIAL_ALL == 1) {
      launch_merge_sample<IS_BF16, 1>(p, stream, partial, out, a);
    } else {
      launch_merge_sample<IS_BF16, MERGE_ALL>(p, stream, partial, out, a);
    }
  });
  return out;
}

std::vector<torch::Tensor> stages_fused(torch::Tensor logits, int64_t top_k, double top_p,
                                        int64_t seed, int64_t offset, int64_t splits_override) {
  check_top_p(top_p);
  const Plan p = make_plan(logits, top_k, splits_override);
  auto keep = torch::empty({p.batch, p.K}, logits.options().dtype(torch::kBool));
  auto renormed = torch::empty({p.batch, p.K}, logits.options().dtype(torch::kFloat32));
  const MergeArgs a{static_cast<float>(top_p), static_cast<uint64_t>(seed),
                    static_cast<uint64_t>(offset), at::PhiloxCudaState(0, 0), false,
                    keep.data_ptr<bool>(), renormed.data_ptr<float>()};
  return {run_fused(logits, p, a), keep, renormed};
}

using AttrRow = std::tuple<std::string, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;

template <typename F>
static void attr_row(std::vector<AttrRow>& rows, const char* name, int P, int block, F* fn) {
  cudaFuncAttributes a;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&a, fn));
  int blocks = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, fn, block, 0));
  rows.emplace_back(name, P, block, a.numRegs, static_cast<int64_t>(a.localSizeBytes),
                    static_cast<int64_t>(a.sharedSizeBytes), blocks);
}

std::vector<AttrRow> kernel_attrs() {
  std::vector<AttrRow> rows;
  attr_row(rows, "topk_partial_kernel", 0, BLOCK, topk_partial_kernel<PARTIAL_ALL>);
  for (int P : {32, 64, 128, 256, 512, 1024}) {
    dispatch_p(P, [&](auto pv) {
      constexpr int PV = decltype(pv)::value;
      attr_row(rows, "merge_sample_kernel<bf16>", PV, MERGE_BLOCK,
               merge_sample_kernel<true, MERGE_ALL, PV>);
      attr_row(rows, "merge_sample_kernel<fp16>", PV, MERGE_BLOCK,
               merge_sample_kernel<false, MERGE_ALL, PV>);
    });
  }
  return rows;
}
