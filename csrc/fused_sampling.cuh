#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace fs {

// fp16 and bf16 are both sign-magnitude 16-bit, so one order-preserving map covers both:
// flip all bits when negative, set the sign bit when positive
__device__ __forceinline__ uint32_t mono_key(uint16_t b) {
  return (b & 0x8000u) ? static_cast<uint32_t>(static_cast<uint16_t>(~b))
                       : static_cast<uint32_t>(static_cast<uint16_t>(b | 0x8000u));
}

// max over the packed form gives the largest key and, among equal keys, the lowest index --
// the repo's tie rule falls out of a single 64-bit comparison
__device__ __forceinline__ uint64_t pack(uint32_t key, uint32_t idx) {
  return (static_cast<uint64_t>(key) << 32) | static_cast<uint64_t>(0xFFFFFFFFu - idx);
}

__device__ __forceinline__ uint32_t unpack_key(uint64_t p) {
  return static_cast<uint32_t>(p >> 32);
}

__device__ __forceinline__ uint32_t unpack_idx(uint64_t p) {
  return 0xFFFFFFFFu - static_cast<uint32_t>(p & 0xFFFFFFFFull);
}

__device__ __forceinline__ uint64_t warp_max_u64(uint64_t v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    uint64_t o = __shfl_down_sync(0xFFFFFFFFu, v, off);
    v = o > v ? o : v;
  }
  return v;
}

__device__ __forceinline__ uint64_t block_max_u64(uint64_t v, uint64_t* smem) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int nwarps = (blockDim.x + 31) >> 5;
  v = warp_max_u64(v);
  if (lane == 0) smem[warp] = v;
  __syncthreads();
  if (warp == 0) {
    uint64_t t = lane < nwarps ? smem[lane] : 0ull;
    t = warp_max_u64(t);
    if (lane == 0) smem[0] = t;
  }
  __syncthreads();
  return smem[0];
}

}  // namespace fs

namespace fs {

// forward: positive -> b | 0x8000, negative -> ~b. both are involutions on their half.
__device__ __forceinline__ uint16_t key_to_bits(uint32_t key) {
  return (key & 0x8000u) ? static_cast<uint16_t>(key & 0x7FFFu)
                         : static_cast<uint16_t>(~key);
}

template <bool IS_BF16>
__device__ __forceinline__ float key_to_float(uint32_t key) {
  const uint16_t b = key_to_bits(key);
  if (IS_BF16) {
    const uint32_t u = static_cast<uint32_t>(b) << 16;
    return __int_as_float(static_cast<int>(u));
  }
  return __half2float(__ushort_as_half(b));
}

// splitmix64 on (seed, offset, row) -- the kernel is free to use its own stream, see SEMANTICS.md
__device__ __forceinline__ float rng_uniform(uint64_t seed, uint64_t offset, uint32_t row) {
  uint64_t z = seed + offset * 0x9E3779B97F4A7C15ull + static_cast<uint64_t>(row) * 0xD1B54A32D192ED03ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  z = z ^ (z >> 31);
  return static_cast<float>((z >> 40) & 0xFFFFFFull) * (1.0f / 16777216.0f);
}

template <typename T>
__device__ __forceinline__ void bitonic_ascending(T* s, int n) {
  for (int k = 2; k <= n; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int ixj = i ^ j;
        if (ixj > i) {
          const bool up = ((i & k) == 0);
          const T a = s[i], b = s[ixj];
          if ((a > b) == up) {
            s[i] = b;
            s[ixj] = a;
          }
        }
      }
      __syncthreads();
    }
  }
}

// suf[255-b] = sum_{j>=b} hist[j], done by one warp: 16 block syncs collapse to 1
__device__ __forceinline__ void suffix_sum_256(const uint32_t* hist, uint32_t* suf) {
  if (threadIdx.x < 32) {
    const int lane = threadIdx.x;
    uint32_t local[8], sum = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      local[j] = hist[255 - (lane * 8 + j)];
      sum += local[j];
    }
    uint32_t excl = sum;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
      const uint32_t o = __shfl_up_sync(0xFFFFFFFFu, excl, off);
      if (lane >= off) excl += o;
    }
    excl -= sum;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      excl += local[j];
      suf[lane * 8 + j] = excl;
    }
  }
  __syncthreads();
}

}  // namespace fs
