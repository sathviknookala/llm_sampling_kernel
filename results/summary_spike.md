# Sampling Ladder Summary

Source: `results/raw/spike_ladder.csv` (1134 rows). Median across rounds/reps; 
latency is amortized device time per sampling call, validation disabled in the timed region.


### Latency µs (speedup vs hf_eager) — V=151936, k=50, p=0.9, bfloat16, hot

| impl | B=1 | B=4 | B=8 | B=16 | B=32 |
|---|---|---|---|---|---|
| hf_eager | 325.1 (1.0x) | 598.8 (1.0x) | 706.9 (1.0x) | 1189.9 (1.0x) | 2193.5 (1.0x) |
| ref_eager_fullsort | 138.6 (2.3x) | 230.4 (2.6x) | 346.6 (2.0x) | 707.2 (1.7x) | 1538.1 (1.4x) |
| tight_eager | 136.5 (2.4x) | 163.7 (3.7x) | 167.1 (4.2x) | 183.0 (6.5x) | 219.2 (10.0x) |
| compile | 159.1 (2.0x) | 153.5 (3.9x) | 153.8 (4.6x) | 182.8 (6.5x) | 219.3 (10.0x) |
| graph_eager | 81.6 (4.0x) | 105.7 (5.7x) | 108.3 (6.5x) | 122.5 (9.7x) | 168.1 (13.0x) |
| graph_compile | 71.9 (4.5x) | 95.1 (6.3x) | 98.2 (7.2x) | 122.6 (9.7x) | 168.4 (13.0x) |
| flashinfer | 84.0 (3.9x) | 85.5 (7.0x) | 91.0 (7.8x) | 115.1 (10.3x) | 177.2 (12.4x) |
| flashinfer_from_probs | 74.0 (4.4x) | 74.5 (8.0x) | 75.2 (9.4x) | 74.8 (15.9x) | 101.2 (21.7x) |
| fused_kernel | 20.6 (15.8x) | 20.8 (28.8x) | 21.2 (33.3x) | 24.9 (47.8x) | 32.8 (66.9x) |

### Latency µs (speedup vs hf_eager) — V=151936, k=50, p=0.9, bfloat16, cold

| impl | B=1 | B=4 | B=8 | B=16 | B=32 |
|---|---|---|---|---|---|
| hf_eager | 325.4 (1.0x) | 598.9 (1.0x) | 705.7 (1.0x) | 1155.1 (1.0x) | 2190.4 (1.0x) |
| ref_eager_fullsort | 140.6 (2.3x) | 233.1 (2.6x) | 345.2 (2.0x) | 704.0 (1.6x) | 1536.8 (1.4x) |
| tight_eager | 137.4 (2.4x) | 165.7 (3.6x) | 169.7 (4.2x) | 186.6 (6.2x) | 230.3 (9.5x) |
| compile | 163.3 (2.0x) | 154.8 (3.9x) | 154.4 (4.6x) | 187.3 (6.2x) | 229.9 (9.5x) |
| graph_eager | 82.9 (3.9x) | 110.8 (5.4x) | 115.7 (6.1x) | 132.1 (8.7x) | 192.2 (11.4x) |
| graph_compile | 73.5 (4.4x) | 101.0 (5.9x) | 105.8 (6.7x) | 132.1 (8.7x) | 192.2 (11.4x) |
| flashinfer | 84.9 (3.8x) | 87.3 (6.9x) | 93.0 (7.6x) | 119.1 (9.7x) | 193.6 (11.3x) |
| flashinfer_from_probs | 75.0 (4.3x) | 74.5 (8.0x) | 75.3 (9.4x) | 78.3 (14.7x) | 127.6 (17.2x) |
| fused_kernel | 22.3 (14.6x) | 22.8 (26.3x) | 25.0 (28.3x) | 30.1 (38.3x) | 43.7 (50.2x) |

### Latency µs (speedup vs hf_eager) — V=151936, k=50, p=0.9, float16, hot

| impl | B=1 | B=4 | B=8 | B=16 | B=32 |
|---|---|---|---|---|---|
| hf_eager | 324.9 (1.0x) | 597.6 (1.0x) | 704.5 (1.0x) | 1188.0 (1.0x) | 2194.5 (1.0x) |
| ref_eager_fullsort | 141.7 (2.3x) | 236.0 (2.5x) | 354.7 (2.0x) | 697.8 (1.7x) | 1539.4 (1.4x) |
| tight_eager | 135.6 (2.4x) | 164.1 (3.6x) | 166.8 (4.2x) | 182.6 (6.5x) | 219.1 (10.0x) |
| compile | 159.3 (2.0x) | 154.3 (3.9x) | 154.6 (4.6x) | 157.5 (7.5x) | 192.1 (11.4x) |
| graph_eager | 81.6 (4.0x) | 104.8 (5.7x) | 108.0 (6.5x) | 122.4 (9.7x) | 167.4 (13.1x) |
| graph_compile | 71.6 (4.5x) | 95.0 (6.3x) | 98.1 (7.2x) | 110.4 (10.8x) | 151.5 (14.5x) |
| flashinfer | 84.0 (3.9x) | 85.3 (7.0x) | 90.5 (7.8x) | 115.0 (10.3x) | 176.6 (12.4x) |
| flashinfer_from_probs | 73.9 (4.4x) | 74.5 (8.0x) | 75.2 (9.4x) | 75.2 (15.8x) | 101.2 (21.7x) |
| fused_kernel | 20.6 (15.8x) | 20.8 (28.8x) | 20.9 (33.7x) | 23.3 (50.9x) | 31.0 (70.8x) |

### Latency µs (speedup vs hf_eager) — V=151936, k=50, p=0.9, float16, cold

| impl | B=1 | B=4 | B=8 | B=16 | B=32 |
|---|---|---|---|---|---|
| hf_eager | 325.4 (1.0x) | 598.8 (1.0x) | 705.6 (1.0x) | 1155.0 (1.0x) | 2190.3 (1.0x) |
| ref_eager_fullsort | 141.6 (2.3x) | 238.6 (2.5x) | 354.3 (2.0x) | 696.8 (1.7x) | 1541.6 (1.4x) |
| tight_eager | 137.4 (2.4x) | 165.5 (3.6x) | 169.3 (4.2x) | 187.8 (6.1x) | 229.3 (9.6x) |
| compile | 162.8 (2.0x) | 154.6 (3.9x) | 154.2 (4.6x) | 159.8 (7.2x) | 196.3 (11.2x) |
| graph_eager | 82.9 (3.9x) | 109.5 (5.5x) | 115.2 (6.1x) | 132.2 (8.7x) | 191.7 (11.4x) |
| graph_compile | 73.3 (4.4x) | 100.9 (5.9x) | 105.8 (6.7x) | 119.4 (9.7x) | 174.7 (12.5x) |
| flashinfer | 85.1 (3.8x) | 87.4 (6.9x) | 92.8 (7.6x) | 119.2 (9.7x) | 193.7 (11.3x) |
| flashinfer_from_probs | 75.5 (4.3x) | 74.9 (8.0x) | 75.2 (9.4x) | 78.3 (14.7x) | 127.6 (17.2x) |
| fused_kernel | 20.8 (15.6x) | 22.4 (26.7x) | 23.9 (29.5x) | 29.5 (39.2x) | 43.0 (51.0x) |

### Hot vs cold L2 residency

| impl | batch | dtype | hot µs | cold µs | cold penalty |
|---|---|---|---|---|---|
| compile | 1 | bfloat16 | 159.1 | 163.3 | +2.6% |
| compile | 1 | float16 | 159.3 | 162.8 | +2.2% |
| compile | 4 | bfloat16 | 153.5 | 154.8 | +0.8% |
| compile | 4 | float16 | 154.3 | 154.6 | +0.2% |
| compile | 8 | bfloat16 | 153.8 | 154.4 | +0.4% |
| compile | 8 | float16 | 154.6 | 154.2 | -0.3% |
| compile | 16 | bfloat16 | 182.8 | 187.3 | +2.5% |
| compile | 16 | float16 | 157.5 | 159.8 | +1.4% |
| compile | 32 | bfloat16 | 219.3 | 229.9 | +4.8% |
| compile | 32 | float16 | 192.1 | 196.3 | +2.2% |
| flashinfer | 1 | bfloat16 | 84.0 | 84.9 | +1.2% |
| flashinfer | 1 | float16 | 84.0 | 85.1 | +1.3% |
| flashinfer | 4 | bfloat16 | 85.5 | 87.3 | +2.1% |
| flashinfer | 4 | float16 | 85.3 | 87.4 | +2.4% |
| flashinfer | 8 | bfloat16 | 91.0 | 93.0 | +2.2% |
| flashinfer | 8 | float16 | 90.5 | 92.8 | +2.6% |
| flashinfer | 16 | bfloat16 | 115.1 | 119.1 | +3.5% |
| flashinfer | 16 | float16 | 115.0 | 119.2 | +3.6% |
| flashinfer | 32 | bfloat16 | 177.2 | 193.6 | +9.2% |
| flashinfer | 32 | float16 | 176.6 | 193.7 | +9.7% |
| flashinfer_from_probs | 1 | bfloat16 | 74.0 | 75.0 | +1.3% |
| flashinfer_from_probs | 1 | float16 | 73.9 | 75.5 | +2.1% |
| flashinfer_from_probs | 4 | bfloat16 | 74.5 | 74.5 | -0.0% |
| flashinfer_from_probs | 4 | float16 | 74.5 | 74.9 | +0.5% |
| flashinfer_from_probs | 8 | bfloat16 | 75.2 | 75.3 | +0.1% |
| flashinfer_from_probs | 8 | float16 | 75.2 | 75.2 | -0.0% |
| flashinfer_from_probs | 16 | bfloat16 | 74.8 | 78.3 | +4.7% |
| flashinfer_from_probs | 16 | float16 | 75.2 | 78.3 | +4.2% |
| flashinfer_from_probs | 32 | bfloat16 | 101.2 | 127.6 | +26.1% |
| flashinfer_from_probs | 32 | float16 | 101.2 | 127.6 | +26.1% |
| fused_kernel | 1 | bfloat16 | 20.6 | 22.3 | +8.5% |
| fused_kernel | 1 | float16 | 20.6 | 20.8 | +1.0% |
| fused_kernel | 4 | bfloat16 | 20.8 | 22.8 | +9.7% |
| fused_kernel | 4 | float16 | 20.8 | 22.4 | +7.9% |
| fused_kernel | 8 | bfloat16 | 21.2 | 25.0 | +17.7% |
| fused_kernel | 8 | float16 | 20.9 | 23.9 | +14.4% |
| fused_kernel | 16 | bfloat16 | 24.9 | 30.1 | +21.2% |
| fused_kernel | 16 | float16 | 23.3 | 29.5 | +26.3% |
| fused_kernel | 32 | bfloat16 | 32.8 | 43.7 | +33.1% |
| fused_kernel | 32 | float16 | 31.0 | 43.0 | +38.7% |
| graph_compile | 1 | bfloat16 | 71.9 | 73.5 | +2.2% |
| graph_compile | 1 | float16 | 71.6 | 73.3 | +2.3% |
| graph_compile | 4 | bfloat16 | 95.1 | 101.0 | +6.2% |
| graph_compile | 4 | float16 | 95.0 | 100.9 | +6.2% |
| graph_compile | 8 | bfloat16 | 98.2 | 105.8 | +7.7% |
| graph_compile | 8 | float16 | 98.1 | 105.8 | +7.8% |
| graph_compile | 16 | bfloat16 | 122.6 | 132.1 | +7.8% |
| graph_compile | 16 | float16 | 110.4 | 119.4 | +8.2% |
| graph_compile | 32 | bfloat16 | 168.4 | 192.2 | +14.1% |
| graph_compile | 32 | float16 | 151.5 | 174.7 | +15.3% |
| graph_eager | 1 | bfloat16 | 81.6 | 82.9 | +1.6% |
| graph_eager | 1 | float16 | 81.6 | 82.9 | +1.6% |
| graph_eager | 4 | bfloat16 | 105.7 | 110.8 | +4.9% |
| graph_eager | 4 | float16 | 104.8 | 109.5 | +4.5% |
| graph_eager | 8 | bfloat16 | 108.3 | 115.7 | +6.8% |
| graph_eager | 8 | float16 | 108.0 | 115.2 | +6.6% |
| graph_eager | 16 | bfloat16 | 122.5 | 132.1 | +7.8% |
| graph_eager | 16 | float16 | 122.4 | 132.2 | +8.0% |
| graph_eager | 32 | bfloat16 | 168.1 | 192.2 | +14.3% |
| graph_eager | 32 | float16 | 167.4 | 191.7 | +14.5% |
| hf_eager | 1 | bfloat16 | 325.1 | 325.4 | +0.1% |
| hf_eager | 1 | float16 | 324.9 | 325.4 | +0.1% |
| hf_eager | 4 | bfloat16 | 598.8 | 598.9 | +0.0% |
| hf_eager | 4 | float16 | 597.6 | 598.8 | +0.2% |
| hf_eager | 8 | bfloat16 | 706.9 | 705.7 | -0.2% |
| hf_eager | 8 | float16 | 704.5 | 705.6 | +0.2% |
| hf_eager | 16 | bfloat16 | 1189.9 | 1155.1 | -2.9% |
| hf_eager | 16 | float16 | 1188.0 | 1155.0 | -2.8% |
| hf_eager | 32 | bfloat16 | 2193.5 | 2190.4 | -0.1% |
| hf_eager | 32 | float16 | 2194.5 | 2190.3 | -0.2% |
| ref_eager_fullsort | 1 | bfloat16 | 138.6 | 140.6 | +1.4% |
| ref_eager_fullsort | 1 | float16 | 141.7 | 141.6 | -0.1% |
| ref_eager_fullsort | 4 | bfloat16 | 230.4 | 233.1 | +1.2% |
| ref_eager_fullsort | 4 | float16 | 236.0 | 238.6 | +1.1% |
| ref_eager_fullsort | 8 | bfloat16 | 346.6 | 345.2 | -0.4% |
| ref_eager_fullsort | 8 | float16 | 354.7 | 354.3 | -0.1% |
| ref_eager_fullsort | 16 | bfloat16 | 707.2 | 704.0 | -0.4% |
| ref_eager_fullsort | 16 | float16 | 697.8 | 696.8 | -0.2% |
| ref_eager_fullsort | 32 | bfloat16 | 1538.1 | 1536.8 | -0.1% |
| ref_eager_fullsort | 32 | float16 | 1539.4 | 1541.6 | +0.1% |
| tight_eager | 1 | bfloat16 | 136.5 | 137.4 | +0.7% |
| tight_eager | 1 | float16 | 135.6 | 137.4 | +1.3% |
| tight_eager | 4 | bfloat16 | 163.7 | 165.7 | +1.2% |
| tight_eager | 4 | float16 | 164.1 | 165.5 | +0.9% |
| tight_eager | 8 | bfloat16 | 167.1 | 169.7 | +1.5% |
| tight_eager | 8 | float16 | 166.8 | 169.3 | +1.5% |
| tight_eager | 16 | bfloat16 | 183.0 | 186.6 | +2.0% |
| tight_eager | 16 | float16 | 182.6 | 187.8 | +2.9% |
| tight_eager | 32 | bfloat16 | 219.2 | 230.3 | +5.1% |
| tight_eager | 32 | float16 | 219.1 | 229.3 | +4.7% |

### Memory-read floor vs measured latency (bfloat16, hot)

| impl | batch | logits MB | DRAM floor µs | latency µs | floor % of latency |
|---|---|---|---|---|---|
| compile | 1 | 0.30 | 0.55 | 159.1 | 0.35% |
| compile | 4 | 1.22 | 2.20 | 153.5 | 1.43% |
| compile | 8 | 2.43 | 4.40 | 153.8 | 2.86% |
| compile | 16 | 4.86 | 8.79 | 182.8 | 4.81% |
| compile | 32 | 9.72 | 17.59 | 219.3 | 8.02% |
| flashinfer | 1 | 0.30 | 0.55 | 84.0 | 0.66% |
| flashinfer | 4 | 1.22 | 2.20 | 85.5 | 2.57% |
| flashinfer | 8 | 2.43 | 4.40 | 91.0 | 4.83% |
| flashinfer | 16 | 4.86 | 8.79 | 115.1 | 7.64% |
| flashinfer | 32 | 9.72 | 17.59 | 177.2 | 9.92% |
| flashinfer_from_probs | 1 | 0.61 | 1.10 | 74.0 | 1.49% |
| flashinfer_from_probs | 4 | 2.43 | 4.40 | 74.5 | 5.90% |
| flashinfer_from_probs | 8 | 4.86 | 8.79 | 75.2 | 11.68% |
| flashinfer_from_probs | 16 | 9.72 | 17.59 | 74.8 | 23.50% |
| flashinfer_from_probs | 32 | 19.45 | 35.17 | 101.2 | 34.75% |
| fused_kernel | 1 | 0.30 | 0.55 | 20.6 | 2.67% |
| fused_kernel | 4 | 1.22 | 2.20 | 20.8 | 10.59% |
| fused_kernel | 8 | 2.43 | 4.40 | 21.2 | 20.71% |
| fused_kernel | 16 | 4.86 | 8.79 | 24.9 | 35.34% |
| fused_kernel | 32 | 9.72 | 17.59 | 32.8 | 53.62% |
| graph_compile | 1 | 0.30 | 0.55 | 71.9 | 0.77% |
| graph_compile | 4 | 1.22 | 2.20 | 95.1 | 2.31% |
| graph_compile | 8 | 2.43 | 4.40 | 98.2 | 4.48% |
| graph_compile | 16 | 4.86 | 8.79 | 122.6 | 7.17% |
| graph_compile | 32 | 9.72 | 17.59 | 168.4 | 10.44% |
| graph_eager | 1 | 0.30 | 0.55 | 81.6 | 0.67% |
| graph_eager | 4 | 1.22 | 2.20 | 105.7 | 2.08% |
| graph_eager | 8 | 2.43 | 4.40 | 108.3 | 4.06% |
| graph_eager | 16 | 4.86 | 8.79 | 122.5 | 7.18% |
| graph_eager | 32 | 9.72 | 17.59 | 168.1 | 10.46% |
| hf_eager | 1 | 0.30 | 0.55 | 325.1 | 0.17% |
| hf_eager | 4 | 1.22 | 2.20 | 598.8 | 0.37% |
| hf_eager | 8 | 2.43 | 4.40 | 706.9 | 0.62% |
| hf_eager | 16 | 4.86 | 8.79 | 1189.9 | 0.74% |
| hf_eager | 32 | 9.72 | 17.59 | 2193.5 | 0.80% |
| ref_eager_fullsort | 1 | 0.30 | 0.55 | 138.6 | 0.40% |
| ref_eager_fullsort | 4 | 1.22 | 2.20 | 230.4 | 0.95% |
| ref_eager_fullsort | 8 | 2.43 | 4.40 | 346.6 | 1.27% |
| ref_eager_fullsort | 16 | 4.86 | 8.79 | 707.2 | 1.24% |
| ref_eager_fullsort | 32 | 9.72 | 17.59 | 1538.1 | 1.14% |
| tight_eager | 1 | 0.30 | 0.55 | 136.5 | 0.40% |
| tight_eager | 4 | 1.22 | 2.20 | 163.7 | 1.34% |
| tight_eager | 8 | 2.43 | 4.40 | 167.1 | 2.63% |
| tight_eager | 16 | 4.86 | 8.79 | 183.0 | 4.80% |
| tight_eager | 32 | 9.72 | 17.59 | 219.2 | 8.02% |

### Parameter sensitivity (bfloat16, hot)

| impl | vocab | top_k | top_p | batch | median µs |
|---|---|---|---|---|---|
| compile | 128256 | 20 | 0.9 | 1 | 161.4 |
| compile | 128256 | 20 | 0.9 | 32 | 202.2 |
| compile | 128256 | 20 | 0.95 | 1 | 161.5 |
| compile | 128256 | 20 | 0.95 | 32 | 202.6 |
| compile | 128256 | 50 | 0.9 | 1 | 159.8 |
| compile | 128256 | 50 | 0.9 | 32 | 202.4 |
| compile | 128256 | 50 | 0.95 | 1 | 160.0 |
| compile | 128256 | 50 | 0.95 | 32 | 202.3 |
| compile | 128256 | 100 | 0.9 | 1 | 160.1 |
| compile | 128256 | 100 | 0.9 | 32 | 201.8 |
| compile | 128256 | 100 | 0.95 | 1 | 159.9 |
| compile | 128256 | 100 | 0.95 | 32 | 201.2 |
| compile | 151936 | 20 | 0.9 | 1 | 162.8 |
| compile | 151936 | 20 | 0.9 | 32 | 218.4 |
| compile | 151936 | 20 | 0.95 | 1 | 162.6 |
| compile | 151936 | 20 | 0.95 | 32 | 219.3 |
| compile | 151936 | 50 | 0.9 | 1 | 159.1 |
| compile | 151936 | 50 | 0.9 | 4 | 153.5 |
| compile | 151936 | 50 | 0.9 | 8 | 153.8 |
| compile | 151936 | 50 | 0.9 | 16 | 182.8 |
| compile | 151936 | 50 | 0.9 | 32 | 219.3 |
| compile | 151936 | 50 | 0.95 | 1 | 161.3 |
| compile | 151936 | 50 | 0.95 | 32 | 219.4 |
| compile | 151936 | 100 | 0.9 | 1 | 161.1 |
| compile | 151936 | 100 | 0.9 | 32 | 218.8 |
| compile | 151936 | 100 | 0.95 | 1 | 161.1 |
| compile | 151936 | 100 | 0.95 | 32 | 218.9 |
| flashinfer | 128256 | 20 | 0.9 | 1 | 83.1 |
| flashinfer | 128256 | 20 | 0.9 | 32 | 141.7 |
| flashinfer | 128256 | 20 | 0.95 | 1 | 83.3 |
| flashinfer | 128256 | 20 | 0.95 | 32 | 141.1 |
| flashinfer | 128256 | 50 | 0.9 | 1 | 83.1 |
| flashinfer | 128256 | 50 | 0.9 | 32 | 141.8 |
| flashinfer | 128256 | 50 | 0.95 | 1 | 83.1 |
| flashinfer | 128256 | 50 | 0.95 | 32 | 141.2 |
| flashinfer | 128256 | 100 | 0.9 | 1 | 82.9 |
| flashinfer | 128256 | 100 | 0.9 | 32 | 141.8 |
| flashinfer | 128256 | 100 | 0.95 | 1 | 83.0 |
| flashinfer | 128256 | 100 | 0.95 | 32 | 141.3 |
| flashinfer | 151936 | 20 | 0.9 | 1 | 84.2 |
| flashinfer | 151936 | 20 | 0.9 | 32 | 176.1 |
| flashinfer | 151936 | 20 | 0.95 | 1 | 84.1 |
| flashinfer | 151936 | 20 | 0.95 | 32 | 175.8 |
| flashinfer | 151936 | 50 | 0.9 | 1 | 84.0 |
| flashinfer | 151936 | 50 | 0.9 | 4 | 85.5 |
| flashinfer | 151936 | 50 | 0.9 | 8 | 91.0 |
| flashinfer | 151936 | 50 | 0.9 | 16 | 115.1 |
| flashinfer | 151936 | 50 | 0.9 | 32 | 177.2 |
| flashinfer | 151936 | 50 | 0.95 | 1 | 83.8 |
| flashinfer | 151936 | 50 | 0.95 | 32 | 176.0 |
| flashinfer | 151936 | 100 | 0.9 | 1 | 84.5 |
| flashinfer | 151936 | 100 | 0.9 | 32 | 177.7 |
| flashinfer | 151936 | 100 | 0.95 | 1 | 84.3 |
| flashinfer | 151936 | 100 | 0.95 | 32 | 177.2 |
| flashinfer_from_probs | 128256 | 20 | 0.9 | 1 | 74.2 |
| flashinfer_from_probs | 128256 | 20 | 0.9 | 32 | 84.4 |
| flashinfer_from_probs | 128256 | 20 | 0.95 | 1 | 74.3 |
| flashinfer_from_probs | 128256 | 20 | 0.95 | 32 | 83.6 |
| flashinfer_from_probs | 128256 | 50 | 0.9 | 1 | 74.0 |
| flashinfer_from_probs | 128256 | 50 | 0.9 | 32 | 82.7 |
| flashinfer_from_probs | 128256 | 50 | 0.95 | 1 | 74.2 |
| flashinfer_from_probs | 128256 | 50 | 0.95 | 32 | 82.2 |
| flashinfer_from_probs | 128256 | 100 | 0.9 | 1 | 74.1 |
| flashinfer_from_probs | 128256 | 100 | 0.9 | 32 | 84.5 |
| flashinfer_from_probs | 128256 | 100 | 0.95 | 1 | 74.2 |
| flashinfer_from_probs | 128256 | 100 | 0.95 | 32 | 83.8 |
| flashinfer_from_probs | 151936 | 20 | 0.9 | 1 | 73.7 |
| flashinfer_from_probs | 151936 | 20 | 0.9 | 32 | 102.8 |
| flashinfer_from_probs | 151936 | 20 | 0.95 | 1 | 73.9 |
| flashinfer_from_probs | 151936 | 20 | 0.95 | 32 | 102.0 |
| flashinfer_from_probs | 151936 | 50 | 0.9 | 1 | 74.0 |
| flashinfer_from_probs | 151936 | 50 | 0.9 | 4 | 74.5 |
| flashinfer_from_probs | 151936 | 50 | 0.9 | 8 | 75.2 |
| flashinfer_from_probs | 151936 | 50 | 0.9 | 16 | 74.8 |
| flashinfer_from_probs | 151936 | 50 | 0.9 | 32 | 101.2 |
| flashinfer_from_probs | 151936 | 50 | 0.95 | 1 | 73.9 |
| flashinfer_from_probs | 151936 | 50 | 0.95 | 32 | 100.7 |
| flashinfer_from_probs | 151936 | 100 | 0.9 | 1 | 73.8 |
| flashinfer_from_probs | 151936 | 100 | 0.9 | 32 | 103.0 |
| flashinfer_from_probs | 151936 | 100 | 0.95 | 1 | 74.2 |
| flashinfer_from_probs | 151936 | 100 | 0.95 | 32 | 102.3 |
| fused_kernel | 128256 | 20 | 0.9 | 1 | 17.3 |
| fused_kernel | 128256 | 20 | 0.9 | 32 | 26.6 |
| fused_kernel | 128256 | 20 | 0.95 | 1 | 17.3 |
| fused_kernel | 128256 | 20 | 0.95 | 32 | 26.7 |
| fused_kernel | 128256 | 50 | 0.9 | 1 | 20.4 |
| fused_kernel | 128256 | 50 | 0.9 | 32 | 28.8 |
| fused_kernel | 128256 | 50 | 0.95 | 1 | 20.7 |
| fused_kernel | 128256 | 50 | 0.95 | 32 | 28.8 |
| fused_kernel | 128256 | 100 | 0.9 | 1 | 25.5 |
| fused_kernel | 128256 | 100 | 0.9 | 32 | 33.2 |
| fused_kernel | 128256 | 100 | 0.95 | 1 | 26.7 |
| fused_kernel | 128256 | 100 | 0.95 | 32 | 33.7 |
| fused_kernel | 151936 | 20 | 0.9 | 1 | 18.5 |
| fused_kernel | 151936 | 20 | 0.9 | 32 | 30.1 |
| fused_kernel | 151936 | 20 | 0.95 | 1 | 18.5 |
| fused_kernel | 151936 | 20 | 0.95 | 32 | 30.3 |
| fused_kernel | 151936 | 50 | 0.9 | 1 | 20.6 |
| fused_kernel | 151936 | 50 | 0.9 | 4 | 20.8 |
| fused_kernel | 151936 | 50 | 0.9 | 8 | 21.2 |
| fused_kernel | 151936 | 50 | 0.9 | 16 | 24.9 |
| fused_kernel | 151936 | 50 | 0.9 | 32 | 32.8 |
| fused_kernel | 151936 | 50 | 0.95 | 1 | 20.8 |
| fused_kernel | 151936 | 50 | 0.95 | 32 | 32.8 |
| fused_kernel | 151936 | 100 | 0.9 | 1 | 26.8 |
| fused_kernel | 151936 | 100 | 0.9 | 32 | 36.7 |
| fused_kernel | 151936 | 100 | 0.95 | 1 | 27.8 |
| fused_kernel | 151936 | 100 | 0.95 | 32 | 37.1 |
| graph_compile | 128256 | 20 | 0.9 | 1 | 76.7 |
| graph_compile | 128256 | 20 | 0.9 | 32 | 150.9 |
| graph_compile | 128256 | 20 | 0.95 | 1 | 76.7 |
| graph_compile | 128256 | 20 | 0.95 | 32 | 150.0 |
| graph_compile | 128256 | 50 | 0.9 | 1 | 77.7 |
| graph_compile | 128256 | 50 | 0.9 | 32 | 150.7 |
| graph_compile | 128256 | 50 | 0.95 | 1 | 77.8 |
| graph_compile | 128256 | 50 | 0.95 | 32 | 150.3 |
| graph_compile | 128256 | 100 | 0.9 | 1 | 78.2 |
| graph_compile | 128256 | 100 | 0.9 | 32 | 151.9 |
| graph_compile | 128256 | 100 | 0.95 | 1 | 78.2 |
| graph_compile | 128256 | 100 | 0.95 | 32 | 151.8 |
| graph_compile | 151936 | 20 | 0.9 | 1 | 81.5 |
| graph_compile | 151936 | 20 | 0.9 | 32 | 166.3 |
| graph_compile | 151936 | 20 | 0.95 | 1 | 86.7 |
| graph_compile | 151936 | 20 | 0.95 | 32 | 166.5 |
| graph_compile | 151936 | 50 | 0.9 | 1 | 71.9 |
| graph_compile | 151936 | 50 | 0.9 | 4 | 95.1 |
| graph_compile | 151936 | 50 | 0.9 | 8 | 98.2 |
| graph_compile | 151936 | 50 | 0.9 | 16 | 122.6 |
| graph_compile | 151936 | 50 | 0.9 | 32 | 168.4 |
| graph_compile | 151936 | 50 | 0.95 | 1 | 81.6 |
| graph_compile | 151936 | 50 | 0.95 | 32 | 167.6 |
| graph_compile | 151936 | 100 | 0.9 | 1 | 82.1 |
| graph_compile | 151936 | 100 | 0.9 | 32 | 168.5 |
| graph_compile | 151936 | 100 | 0.95 | 1 | 82.1 |
| graph_compile | 151936 | 100 | 0.95 | 32 | 168.6 |
| graph_eager | 128256 | 20 | 0.9 | 1 | 76.7 |
| graph_eager | 128256 | 20 | 0.9 | 32 | 150.2 |
| graph_eager | 128256 | 20 | 0.95 | 1 | 76.7 |
| graph_eager | 128256 | 20 | 0.95 | 32 | 150.0 |
| graph_eager | 128256 | 50 | 0.9 | 1 | 77.8 |
| graph_eager | 128256 | 50 | 0.9 | 32 | 150.4 |
| graph_eager | 128256 | 50 | 0.95 | 1 | 77.9 |
| graph_eager | 128256 | 50 | 0.95 | 32 | 150.2 |
| graph_eager | 128256 | 100 | 0.9 | 1 | 78.3 |
| graph_eager | 128256 | 100 | 0.9 | 32 | 151.8 |
| graph_eager | 128256 | 100 | 0.95 | 1 | 78.3 |
| graph_eager | 128256 | 100 | 0.95 | 32 | 151.7 |
| graph_eager | 151936 | 20 | 0.9 | 1 | 81.5 |
| graph_eager | 151936 | 20 | 0.9 | 32 | 166.0 |
| graph_eager | 151936 | 20 | 0.95 | 1 | 81.5 |
| graph_eager | 151936 | 20 | 0.95 | 32 | 166.3 |
| graph_eager | 151936 | 50 | 0.9 | 1 | 81.6 |
| graph_eager | 151936 | 50 | 0.9 | 4 | 105.7 |
| graph_eager | 151936 | 50 | 0.9 | 8 | 108.3 |
| graph_eager | 151936 | 50 | 0.9 | 16 | 122.5 |
| graph_eager | 151936 | 50 | 0.9 | 32 | 168.1 |
| graph_eager | 151936 | 50 | 0.95 | 1 | 81.5 |
| graph_eager | 151936 | 50 | 0.95 | 32 | 167.3 |
| graph_eager | 151936 | 100 | 0.9 | 1 | 81.9 |
| graph_eager | 151936 | 100 | 0.9 | 32 | 168.3 |
| graph_eager | 151936 | 100 | 0.95 | 1 | 81.9 |
| graph_eager | 151936 | 100 | 0.95 | 32 | 168.4 |
| hf_eager | 128256 | 20 | 0.9 | 1 | 338.3 |
| hf_eager | 128256 | 20 | 0.9 | 32 | 1837.4 |
| hf_eager | 128256 | 20 | 0.95 | 1 | 338.2 |
| hf_eager | 128256 | 20 | 0.95 | 32 | 1835.4 |
| hf_eager | 128256 | 50 | 0.9 | 1 | 340.1 |
| hf_eager | 128256 | 50 | 0.9 | 32 | 1837.8 |
| hf_eager | 128256 | 50 | 0.95 | 1 | 340.1 |
| hf_eager | 128256 | 50 | 0.95 | 32 | 1837.5 |
| hf_eager | 128256 | 100 | 0.9 | 1 | 341.3 |
| hf_eager | 128256 | 100 | 0.9 | 32 | 1834.4 |
| hf_eager | 128256 | 100 | 0.95 | 1 | 341.4 |
| hf_eager | 128256 | 100 | 0.95 | 32 | 1833.2 |
| hf_eager | 151936 | 20 | 0.9 | 1 | 323.2 |
| hf_eager | 151936 | 20 | 0.9 | 32 | 2186.0 |
| hf_eager | 151936 | 20 | 0.95 | 1 | 323.1 |
| hf_eager | 151936 | 20 | 0.95 | 32 | 2186.4 |
| hf_eager | 151936 | 50 | 0.9 | 1 | 325.1 |
| hf_eager | 151936 | 50 | 0.9 | 4 | 598.8 |
| hf_eager | 151936 | 50 | 0.9 | 8 | 706.9 |
| hf_eager | 151936 | 50 | 0.9 | 16 | 1189.9 |
| hf_eager | 151936 | 50 | 0.9 | 32 | 2193.5 |
| hf_eager | 151936 | 50 | 0.95 | 1 | 325.0 |
| hf_eager | 151936 | 50 | 0.95 | 32 | 2189.8 |
| hf_eager | 151936 | 100 | 0.9 | 1 | 326.4 |
| hf_eager | 151936 | 100 | 0.9 | 32 | 2187.9 |
| hf_eager | 151936 | 100 | 0.95 | 1 | 326.7 |
| hf_eager | 151936 | 100 | 0.95 | 32 | 2188.7 |
| ref_eager_fullsort | 128256 | 20 | 0.9 | 1 | 137.0 |
| ref_eager_fullsort | 128256 | 20 | 0.9 | 32 | 1280.5 |
| ref_eager_fullsort | 128256 | 20 | 0.95 | 1 | 136.9 |
| ref_eager_fullsort | 128256 | 20 | 0.95 | 32 | 1280.0 |
| ref_eager_fullsort | 128256 | 50 | 0.9 | 1 | 136.6 |
| ref_eager_fullsort | 128256 | 50 | 0.9 | 32 | 1279.8 |
| ref_eager_fullsort | 128256 | 50 | 0.95 | 1 | 136.6 |
| ref_eager_fullsort | 128256 | 50 | 0.95 | 32 | 1280.1 |
| ref_eager_fullsort | 128256 | 100 | 0.9 | 1 | 136.7 |
| ref_eager_fullsort | 128256 | 100 | 0.9 | 32 | 1277.9 |
| ref_eager_fullsort | 128256 | 100 | 0.95 | 1 | 136.6 |
| ref_eager_fullsort | 128256 | 100 | 0.95 | 32 | 1277.9 |
| ref_eager_fullsort | 151936 | 20 | 0.9 | 1 | 138.7 |
| ref_eager_fullsort | 151936 | 20 | 0.9 | 32 | 1533.5 |
| ref_eager_fullsort | 151936 | 20 | 0.95 | 1 | 138.7 |
| ref_eager_fullsort | 151936 | 20 | 0.95 | 32 | 1534.0 |
| ref_eager_fullsort | 151936 | 50 | 0.9 | 1 | 138.6 |
| ref_eager_fullsort | 151936 | 50 | 0.9 | 4 | 230.4 |
| ref_eager_fullsort | 151936 | 50 | 0.9 | 8 | 346.6 |
| ref_eager_fullsort | 151936 | 50 | 0.9 | 16 | 707.2 |
| ref_eager_fullsort | 151936 | 50 | 0.9 | 32 | 1538.1 |
| ref_eager_fullsort | 151936 | 50 | 0.95 | 1 | 139.0 |
| ref_eager_fullsort | 151936 | 50 | 0.95 | 32 | 1535.5 |
| ref_eager_fullsort | 151936 | 100 | 0.9 | 1 | 138.9 |
| ref_eager_fullsort | 151936 | 100 | 0.9 | 32 | 1535.6 |
| ref_eager_fullsort | 151936 | 100 | 0.95 | 1 | 138.7 |
| ref_eager_fullsort | 151936 | 100 | 0.95 | 32 | 1535.7 |
| tight_eager | 128256 | 20 | 0.9 | 1 | 135.0 |
| tight_eager | 128256 | 20 | 0.9 | 32 | 202.3 |
| tight_eager | 128256 | 20 | 0.95 | 1 | 134.9 |
| tight_eager | 128256 | 20 | 0.95 | 32 | 202.0 |
| tight_eager | 128256 | 50 | 0.9 | 1 | 133.3 |
| tight_eager | 128256 | 50 | 0.9 | 32 | 201.6 |
| tight_eager | 128256 | 50 | 0.95 | 1 | 134.2 |
| tight_eager | 128256 | 50 | 0.95 | 32 | 201.5 |
| tight_eager | 128256 | 100 | 0.9 | 1 | 133.2 |
| tight_eager | 128256 | 100 | 0.9 | 32 | 201.3 |
| tight_eager | 128256 | 100 | 0.95 | 1 | 133.1 |
| tight_eager | 128256 | 100 | 0.95 | 32 | 201.2 |
| tight_eager | 151936 | 20 | 0.9 | 1 | 135.4 |
| tight_eager | 151936 | 20 | 0.9 | 32 | 217.4 |
| tight_eager | 151936 | 20 | 0.95 | 1 | 135.2 |
| tight_eager | 151936 | 20 | 0.95 | 32 | 217.0 |
| tight_eager | 151936 | 50 | 0.9 | 1 | 136.5 |
| tight_eager | 151936 | 50 | 0.9 | 4 | 163.7 |
| tight_eager | 151936 | 50 | 0.9 | 8 | 167.1 |
| tight_eager | 151936 | 50 | 0.9 | 16 | 183.0 |
| tight_eager | 151936 | 50 | 0.9 | 32 | 219.2 |
| tight_eager | 151936 | 50 | 0.95 | 1 | 134.1 |
| tight_eager | 151936 | 50 | 0.95 | 32 | 218.6 |
| tight_eager | 151936 | 100 | 0.9 | 1 | 134.6 |
| tight_eager | 151936 | 100 | 0.9 | 32 | 217.9 |
| tight_eager | 151936 | 100 | 0.95 | 1 | 135.4 |
| tight_eager | 151936 | 100 | 0.95 | 32 | 218.3 |

### Round-to-round spread (impl order rotated each round)

| impl | rounds | min median µs | max median µs | spread |
|---|---|---|---|---|
| compile | 3 | 159.0 | 159.4 | 0.2% |
| flashinfer | 3 | 91.9 | 91.9 | 0.0% |
| flashinfer_from_probs | 3 | 75.1 | 75.2 | 0.2% |
| fused_kernel | 3 | 23.0 | 23.1 | 0.4% |
| graph_compile | 3 | 103.4 | 105.7 | 2.2% |
| graph_eager | 3 | 112.7 | 113.1 | 0.3% |
| hf_eager | 3 | 702.1 | 706.5 | 0.6% |
| ref_eager_fullsort | 3 | 346.5 | 350.8 | 1.2% |
| tight_eager | 3 | 168.0 | 168.8 | 0.5% |
