#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include "fused_sampling.cuh"

namespace {

__global__ void noop_kernel(int64_t* __restrict__ out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = 0;
}

__device__ __forceinline__ uint64_t scan_range(const uint16_t* __restrict__ row, int lo, int hi,
                                               int n_vec_lo, int n_vec_hi) {
  uint64_t best = 0;
  const uint4* rv = reinterpret_cast<const uint4*>(row);
  for (int i = n_vec_lo + threadIdx.x; i < n_vec_hi; i += blockDim.x) {
    uint4 v = rv[i];
    const uint16_t* h = reinterpret_cast<const uint16_t*>(&v);
    const int base = i * 8;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      uint64_t p = fs::pack(fs::mono_key(h[j]), base + j);
      best = p > best ? p : best;
    }
  }
  for (int i = lo + threadIdx.x; i < hi; i += blockDim.x) {
    uint64_t p = fs::pack(fs::mono_key(row[i]), i);
    best = p > best ? p : best;
  }
  return best;
}

__global__ void scan_rowblock_kernel(const uint16_t* __restrict__ x, int64_t* __restrict__ out,
                                     int vocab, int n_vec) {
  __shared__ uint64_t smem[32];
  const uint16_t* row = x + static_cast<size_t>(blockIdx.x) * vocab;
  uint64_t best = scan_range(row, n_vec * 8, vocab, 0, n_vec);
  best = fs::block_max_u64(best, smem);
  if (threadIdx.x == 0) out[blockIdx.x] = static_cast<int64_t>(fs::unpack_idx(best));
}

// grid.x = split, grid.y = row; each block owns a contiguous slice of one row
__global__ void scan_split_kernel(const uint16_t* __restrict__ x, uint64_t* __restrict__ partial,
                                  int vocab, int n_vec, int splits) {
  __shared__ uint64_t smem[32];
  const int s = blockIdx.x;
  const uint16_t* row = x + static_cast<size_t>(blockIdx.y) * vocab;

  const int vec_per = (n_vec + splits - 1) / splits;
  const int vlo = min(s * vec_per, n_vec);
  const int vhi = min(vlo + vec_per, n_vec);

  // the ragged tail (vocab % 8 elements) goes entirely to the last split
  const int tlo = (s == splits - 1) ? n_vec * 8 : vocab;

  uint64_t best = scan_range(row, tlo, vocab, vlo, vhi);
  best = fs::block_max_u64(best, smem);
  if (threadIdx.x == 0) partial[static_cast<size_t>(blockIdx.y) * splits + s] = best;
}

__global__ void merge_partial_kernel(const uint64_t* __restrict__ partial,
                                     int64_t* __restrict__ out, int splits) {
  __shared__ uint64_t smem[32];
  const uint64_t* p = partial + static_cast<size_t>(blockIdx.x) * splits;
  uint64_t best = 0;
  for (int i = threadIdx.x; i < splits; i += blockDim.x) best = p[i] > best ? p[i] : best;
  best = fs::block_max_u64(best, smem);
  if (threadIdx.x == 0) out[blockIdx.x] = static_cast<int64_t>(fs::unpack_idx(best));
}

void check_logits(const torch::Tensor& x) {
  TORCH_CHECK(x.is_cuda(), "logits must be on cuda");
  TORCH_CHECK(x.dim() == 2, "logits must be [B, V]");
  TORCH_CHECK(x.is_contiguous(), "logits must be contiguous");
  TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
              "logits must be float16 or bfloat16");
}

// vectorised loads are only safe when every row start is 16B aligned
int vec_chunks(const torch::Tensor& x) {
  const int vocab = static_cast<int>(x.size(-1));
  const bool aligned = (reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0) && (vocab % 8 == 0);
  return aligned ? vocab / 8 : 0;
}

}  // namespace

torch::Tensor probe_noop(torch::Tensor logits) {
  check_logits(logits);
  const int batch = static_cast<int>(logits.size(0));
  auto out = torch::empty({batch}, logits.options().dtype(torch::kInt64));
  noop_kernel<<<(batch + 31) / 32, 32, 0, at::cuda::getCurrentCUDAStream()>>>(
      out.data_ptr<int64_t>(), batch);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor probe_scan_rowblock(torch::Tensor logits) {
  check_logits(logits);
  const int batch = static_cast<int>(logits.size(0));
  const int vocab = static_cast<int>(logits.size(-1));
  auto out = torch::empty({batch}, logits.options().dtype(torch::kInt64));
  scan_rowblock_kernel<<<batch, 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint16_t*>(logits.data_ptr()), out.data_ptr<int64_t>(), vocab,
      vec_chunks(logits));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor probe_scan_split(torch::Tensor logits, int64_t splits) {
  check_logits(logits);
  const int batch = static_cast<int>(logits.size(0));
  const int vocab = static_cast<int>(logits.size(-1));
  const int s = static_cast<int>(splits);
  TORCH_CHECK(s >= 1, "splits must be >= 1");
  auto out = torch::empty({batch}, logits.options().dtype(torch::kInt64));
  auto partial = torch::empty({batch, s}, logits.options().dtype(torch::kInt64));
  auto stream = at::cuda::getCurrentCUDAStream();
  scan_split_kernel<<<dim3(s, batch), 256, 0, stream>>>(
      reinterpret_cast<const uint16_t*>(logits.data_ptr()),
      reinterpret_cast<uint64_t*>(partial.data_ptr()), vocab, vec_chunks(logits), s);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  merge_partial_kernel<<<batch, 128, 0, stream>>>(
      reinterpret_cast<const uint64_t*>(partial.data_ptr()), out.data_ptr<int64_t>(), s);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
