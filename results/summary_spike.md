# Sampling Ladder Summary

Source: `results/raw/spike_ladder.csv` (1134 rows). Median across rounds/reps; 
latency is amortized device time per sampling call, validation disabled in the timed region.


### Latency µs (speedup vs hf_eager) — V=151936, k=50, p=0.9, bfloat16, hot

| impl | B=1 | B=4 | B=8 | B=16 | B=32 |
|---|---|---|---|---|---|
| hf_eager | 325.3 (1.0x) | 598.9 (1.0x) | 707.9 (1.0x) | 1192.0 (1.0x) | 2195.8 (1.0x) |
| ref_eager_fullsort | 138.9 (2.3x) | 230.7 (2.6x) | 347.6 (2.0x) | 708.2 (1.7x) | 1538.7 (1.4x) |
| tight_eager | 136.8 (2.4x) | 163.8 (3.7x) | 168.0 (4.2x) | 183.9 (6.5x) | 219.5 (10.0x) |
| compile | 157.3 (2.1x) | 152.3 (3.9x) | 152.3 (4.6x) | 183.7 (6.5x) | 220.3 (10.0x) |
| graph_eager | 81.6 (4.0x) | 106.0 (5.6x) | 108.7 (6.5x) | 123.2 (9.7x) | 169.1 (13.0x) |
| graph_compile | 77.4 (4.2x) | 95.2 (6.3x) | 98.4 (7.2x) | 123.3 (9.7x) | 169.2 (13.0x) |
| flashinfer | 84.0 (3.9x) | 85.6 (7.0x) | 91.1 (7.8x) | 116.0 (10.3x) | 177.8 (12.4x) |
| flashinfer_from_probs | 72.5 (4.5x) | 73.2 (8.2x) | 74.1 (9.6x) | 73.9 (16.1x) | 101.2 (21.7x) |
| fused_kernel | 22.5 (14.4x) | 22.5 (26.6x) | 22.6 (31.4x) | 26.6 (44.7x) | 34.8 (63.1x) |

### Latency µs (speedup vs hf_eager) — V=151936, k=50, p=0.9, bfloat16, cold

| impl | B=1 | B=4 | B=8 | B=16 | B=32 |
|---|---|---|---|---|---|
| hf_eager | 325.5 (1.0x) | 599.1 (1.0x) | 707.6 (1.0x) | 1156.6 (1.0x) | 2192.7 (1.0x) |
| ref_eager_fullsort | 139.1 (2.3x) | 232.9 (2.6x) | 346.3 (2.0x) | 705.0 (1.6x) | 1537.4 (1.4x) |
| tight_eager | 136.2 (2.4x) | 166.2 (3.6x) | 169.8 (4.2x) | 187.6 (6.2x) | 231.6 (9.5x) |
| compile | 161.8 (2.0x) | 153.6 (3.9x) | 153.2 (4.6x) | 187.6 (6.2x) | 230.9 (9.5x) |
| graph_eager | 83.0 (3.9x) | 111.0 (5.4x) | 116.2 (6.1x) | 132.6 (8.7x) | 192.7 (11.4x) |
| graph_compile | 73.5 (4.4x) | 101.1 (5.9x) | 106.1 (6.7x) | 132.7 (8.7x) | 192.8 (11.4x) |
| flashinfer | 84.2 (3.9x) | 87.4 (6.9x) | 93.0 (7.6x) | 120.0 (9.6x) | 194.3 (11.3x) |
| flashinfer_from_probs | 73.8 (4.4x) | 73.3 (8.2x) | 74.1 (9.6x) | 78.3 (14.8x) | 127.7 (17.2x) |
| fused_kernel | 24.4 (13.3x) | 24.6 (24.4x) | 26.6 (26.6x) | 31.1 (37.2x) | 45.5 (48.2x) |

### Latency µs (speedup vs hf_eager) — V=151936, k=50, p=0.9, float16, hot

| impl | B=1 | B=4 | B=8 | B=16 | B=32 |
|---|---|---|---|---|---|
| hf_eager | 325.0 (1.0x) | 598.6 (1.0x) | 705.6 (1.0x) | 1188.8 (1.0x) | 2195.8 (1.0x) |
| ref_eager_fullsort | 141.3 (2.3x) | 236.4 (2.5x) | 354.8 (2.0x) | 698.9 (1.7x) | 1540.6 (1.4x) |
| tight_eager | 137.1 (2.4x) | 164.3 (3.6x) | 167.1 (4.2x) | 183.3 (6.5x) | 219.3 (10.0x) |
| compile | 158.5 (2.0x) | 152.5 (3.9x) | 152.2 (4.6x) | 158.7 (7.5x) | 192.5 (11.4x) |
| graph_eager | 81.6 (4.0x) | 105.0 (5.7x) | 108.2 (6.5x) | 122.8 (9.7x) | 168.1 (13.1x) |
| graph_compile | 71.9 (4.5x) | 95.1 (6.3x) | 98.3 (7.2x) | 111.2 (10.7x) | 152.0 (14.5x) |
| flashinfer | 84.0 (3.9x) | 85.3 (7.0x) | 90.5 (7.8x) | 115.3 (10.3x) | 176.9 (12.4x) |
| flashinfer_from_probs | 72.2 (4.5x) | 73.3 (8.2x) | 73.8 (9.6x) | 73.9 (16.1x) | 101.2 (21.7x) |
| fused_kernel | 22.5 (14.4x) | 22.5 (26.6x) | 22.6 (31.3x) | 24.6 (48.2x) | 33.5 (65.6x) |

### Latency µs (speedup vs hf_eager) — V=151936, k=50, p=0.9, float16, cold

| impl | B=1 | B=4 | B=8 | B=16 | B=32 |
|---|---|---|---|---|---|
| hf_eager | 325.5 (1.0x) | 598.9 (1.0x) | 707.6 (1.0x) | 1156.5 (1.0x) | 2193.3 (1.0x) |
| ref_eager_fullsort | 141.6 (2.3x) | 239.3 (2.5x) | 355.0 (2.0x) | 698.3 (1.7x) | 1543.8 (1.4x) |
| tight_eager | 136.1 (2.4x) | 165.8 (3.6x) | 170.5 (4.1x) | 187.6 (6.2x) | 229.9 (9.5x) |
| compile | 161.2 (2.0x) | 153.2 (3.9x) | 153.1 (4.6x) | 160.1 (7.2x) | 196.6 (11.2x) |
| graph_eager | 82.9 (3.9x) | 109.5 (5.5x) | 115.4 (6.1x) | 132.6 (8.7x) | 192.6 (11.4x) |
| graph_compile | 73.3 (4.4x) | 101.0 (5.9x) | 105.9 (6.7x) | 119.7 (9.7x) | 175.2 (12.5x) |
| flashinfer | 84.2 (3.9x) | 87.4 (6.9x) | 92.7 (7.6x) | 119.2 (9.7x) | 194.4 (11.3x) |
| flashinfer_from_probs | 73.8 (4.4x) | 73.9 (8.1x) | 74.4 (9.5x) | 78.3 (14.8x) | 127.6 (17.2x) |
| fused_kernel | 22.5 (14.5x) | 24.3 (24.7x) | 25.1 (28.2x) | 30.8 (37.6x) | 45.0 (48.8x) |

### Hot vs cold L2 residency

| impl | batch | dtype | hot µs | cold µs | cold penalty |
|---|---|---|---|---|---|
| compile | 1 | bfloat16 | 157.3 | 161.8 | +2.9% |
| compile | 1 | float16 | 158.5 | 161.2 | +1.7% |
| compile | 4 | bfloat16 | 152.3 | 153.6 | +0.9% |
| compile | 4 | float16 | 152.5 | 153.2 | +0.4% |
| compile | 8 | bfloat16 | 152.3 | 153.2 | +0.6% |
| compile | 8 | float16 | 152.2 | 153.1 | +0.6% |
| compile | 16 | bfloat16 | 183.7 | 187.6 | +2.1% |
| compile | 16 | float16 | 158.7 | 160.1 | +0.9% |
| compile | 32 | bfloat16 | 220.3 | 230.9 | +4.8% |
| compile | 32 | float16 | 192.5 | 196.6 | +2.1% |
| flashinfer | 1 | bfloat16 | 84.0 | 84.2 | +0.1% |
| flashinfer | 1 | float16 | 84.0 | 84.2 | +0.2% |
| flashinfer | 4 | bfloat16 | 85.6 | 87.4 | +2.1% |
| flashinfer | 4 | float16 | 85.3 | 87.4 | +2.4% |
| flashinfer | 8 | bfloat16 | 91.1 | 93.0 | +2.1% |
| flashinfer | 8 | float16 | 90.5 | 92.7 | +2.4% |
| flashinfer | 16 | bfloat16 | 116.0 | 120.0 | +3.5% |
| flashinfer | 16 | float16 | 115.3 | 119.2 | +3.5% |
| flashinfer | 32 | bfloat16 | 177.8 | 194.3 | +9.3% |
| flashinfer | 32 | float16 | 176.9 | 194.4 | +9.9% |
| flashinfer_from_probs | 1 | bfloat16 | 72.5 | 73.8 | +1.8% |
| flashinfer_from_probs | 1 | float16 | 72.2 | 73.8 | +2.3% |
| flashinfer_from_probs | 4 | bfloat16 | 73.2 | 73.3 | +0.1% |
| flashinfer_from_probs | 4 | float16 | 73.3 | 73.9 | +0.8% |
| flashinfer_from_probs | 8 | bfloat16 | 74.1 | 74.1 | -0.0% |
| flashinfer_from_probs | 8 | float16 | 73.8 | 74.4 | +0.9% |
| flashinfer_from_probs | 16 | bfloat16 | 73.9 | 78.3 | +6.0% |
| flashinfer_from_probs | 16 | float16 | 73.9 | 78.3 | +5.9% |
| flashinfer_from_probs | 32 | bfloat16 | 101.2 | 127.7 | +26.1% |
| flashinfer_from_probs | 32 | float16 | 101.2 | 127.6 | +26.1% |
| fused_kernel | 1 | bfloat16 | 22.5 | 24.4 | +8.3% |
| fused_kernel | 1 | float16 | 22.5 | 22.5 | -0.1% |
| fused_kernel | 4 | bfloat16 | 22.5 | 24.6 | +9.0% |
| fused_kernel | 4 | float16 | 22.5 | 24.3 | +7.7% |
| fused_kernel | 8 | bfloat16 | 22.6 | 26.6 | +18.0% |
| fused_kernel | 8 | float16 | 22.6 | 25.1 | +11.1% |
| fused_kernel | 16 | bfloat16 | 26.6 | 31.1 | +16.6% |
| fused_kernel | 16 | float16 | 24.6 | 30.8 | +24.9% |
| fused_kernel | 32 | bfloat16 | 34.8 | 45.5 | +30.7% |
| fused_kernel | 32 | float16 | 33.5 | 45.0 | +34.3% |
| graph_compile | 1 | bfloat16 | 77.4 | 73.5 | -5.0% |
| graph_compile | 1 | float16 | 71.9 | 73.3 | +2.0% |
| graph_compile | 4 | bfloat16 | 95.2 | 101.1 | +6.2% |
| graph_compile | 4 | float16 | 95.1 | 101.0 | +6.2% |
| graph_compile | 8 | bfloat16 | 98.4 | 106.1 | +7.8% |
| graph_compile | 8 | float16 | 98.3 | 105.9 | +7.7% |
| graph_compile | 16 | bfloat16 | 123.3 | 132.7 | +7.6% |
| graph_compile | 16 | float16 | 111.2 | 119.7 | +7.6% |
| graph_compile | 32 | bfloat16 | 169.2 | 192.8 | +14.0% |
| graph_compile | 32 | float16 | 152.0 | 175.2 | +15.3% |
| graph_eager | 1 | bfloat16 | 81.6 | 83.0 | +1.7% |
| graph_eager | 1 | float16 | 81.6 | 82.9 | +1.6% |
| graph_eager | 4 | bfloat16 | 106.0 | 111.0 | +4.8% |
| graph_eager | 4 | float16 | 105.0 | 109.5 | +4.2% |
| graph_eager | 8 | bfloat16 | 108.7 | 116.2 | +6.9% |
| graph_eager | 8 | float16 | 108.2 | 115.4 | +6.7% |
| graph_eager | 16 | bfloat16 | 123.2 | 132.6 | +7.6% |
| graph_eager | 16 | float16 | 122.8 | 132.6 | +8.0% |
| graph_eager | 32 | bfloat16 | 169.1 | 192.7 | +14.0% |
| graph_eager | 32 | float16 | 168.1 | 192.6 | +14.6% |
| hf_eager | 1 | bfloat16 | 325.3 | 325.5 | +0.0% |
| hf_eager | 1 | float16 | 325.0 | 325.5 | +0.2% |
| hf_eager | 4 | bfloat16 | 598.9 | 599.1 | +0.0% |
| hf_eager | 4 | float16 | 598.6 | 598.9 | +0.1% |
| hf_eager | 8 | bfloat16 | 707.9 | 707.6 | -0.1% |
| hf_eager | 8 | float16 | 705.6 | 707.6 | +0.3% |
| hf_eager | 16 | bfloat16 | 1192.0 | 1156.6 | -3.0% |
| hf_eager | 16 | float16 | 1188.8 | 1156.5 | -2.7% |
| hf_eager | 32 | bfloat16 | 2195.8 | 2192.7 | -0.1% |
| hf_eager | 32 | float16 | 2195.8 | 2193.3 | -0.1% |
| ref_eager_fullsort | 1 | bfloat16 | 138.9 | 139.1 | +0.1% |
| ref_eager_fullsort | 1 | float16 | 141.3 | 141.6 | +0.2% |
| ref_eager_fullsort | 4 | bfloat16 | 230.7 | 232.9 | +1.0% |
| ref_eager_fullsort | 4 | float16 | 236.4 | 239.3 | +1.2% |
| ref_eager_fullsort | 8 | bfloat16 | 347.6 | 346.3 | -0.4% |
| ref_eager_fullsort | 8 | float16 | 354.8 | 355.0 | +0.0% |
| ref_eager_fullsort | 16 | bfloat16 | 708.2 | 705.0 | -0.5% |
| ref_eager_fullsort | 16 | float16 | 698.9 | 698.3 | -0.1% |
| ref_eager_fullsort | 32 | bfloat16 | 1538.7 | 1537.4 | -0.1% |
| ref_eager_fullsort | 32 | float16 | 1540.6 | 1543.8 | +0.2% |
| tight_eager | 1 | bfloat16 | 136.8 | 136.2 | -0.5% |
| tight_eager | 1 | float16 | 137.1 | 136.1 | -0.7% |
| tight_eager | 4 | bfloat16 | 163.8 | 166.2 | +1.5% |
| tight_eager | 4 | float16 | 164.3 | 165.8 | +0.9% |
| tight_eager | 8 | bfloat16 | 168.0 | 169.8 | +1.1% |
| tight_eager | 8 | float16 | 167.1 | 170.5 | +2.0% |
| tight_eager | 16 | bfloat16 | 183.9 | 187.6 | +2.0% |
| tight_eager | 16 | float16 | 183.3 | 187.6 | +2.3% |
| tight_eager | 32 | bfloat16 | 219.5 | 231.6 | +5.5% |
| tight_eager | 32 | float16 | 219.3 | 229.9 | +4.8% |

### Memory-read floor vs measured latency (bfloat16, hot)

| impl | batch | logits MB | DRAM floor µs | latency µs | floor % of latency |
|---|---|---|---|---|---|
| compile | 1 | 0.30 | 0.55 | 157.3 | 0.35% |
| compile | 4 | 1.22 | 2.20 | 152.3 | 1.44% |
| compile | 8 | 2.43 | 4.39 | 152.3 | 2.89% |
| compile | 16 | 4.86 | 8.79 | 183.7 | 4.79% |
| compile | 32 | 9.72 | 17.58 | 220.3 | 7.98% |
| flashinfer | 1 | 0.30 | 0.55 | 84.0 | 0.65% |
| flashinfer | 4 | 1.22 | 2.20 | 85.6 | 2.57% |
| flashinfer | 8 | 2.43 | 4.39 | 91.1 | 4.82% |
| flashinfer | 16 | 4.86 | 8.79 | 116.0 | 7.58% |
| flashinfer | 32 | 9.72 | 17.58 | 177.8 | 9.89% |
| flashinfer_from_probs | 1 | 0.61 | 1.10 | 72.5 | 1.52% |
| flashinfer_from_probs | 4 | 2.43 | 4.39 | 73.2 | 6.00% |
| flashinfer_from_probs | 8 | 4.86 | 8.79 | 74.1 | 11.86% |
| flashinfer_from_probs | 16 | 9.72 | 17.58 | 73.9 | 23.78% |
| flashinfer_from_probs | 32 | 19.45 | 35.16 | 101.2 | 34.73% |
| fused_kernel | 1 | 0.30 | 0.55 | 22.5 | 2.44% |
| fused_kernel | 4 | 1.22 | 2.20 | 22.5 | 9.75% |
| fused_kernel | 8 | 2.43 | 4.39 | 22.6 | 19.48% |
| fused_kernel | 16 | 4.86 | 8.79 | 26.6 | 32.98% |
| fused_kernel | 32 | 9.72 | 17.58 | 34.8 | 50.52% |
| graph_compile | 1 | 0.30 | 0.55 | 77.4 | 0.71% |
| graph_compile | 4 | 1.22 | 2.20 | 95.2 | 2.31% |
| graph_compile | 8 | 2.43 | 4.39 | 98.4 | 4.46% |
| graph_compile | 16 | 4.86 | 8.79 | 123.3 | 7.13% |
| graph_compile | 32 | 9.72 | 17.58 | 169.2 | 10.39% |
| graph_eager | 1 | 0.30 | 0.55 | 81.6 | 0.67% |
| graph_eager | 4 | 1.22 | 2.20 | 106.0 | 2.07% |
| graph_eager | 8 | 2.43 | 4.39 | 108.7 | 4.04% |
| graph_eager | 16 | 4.86 | 8.79 | 123.2 | 7.13% |
| graph_eager | 32 | 9.72 | 17.58 | 169.1 | 10.39% |
| hf_eager | 1 | 0.30 | 0.55 | 325.3 | 0.17% |
| hf_eager | 4 | 1.22 | 2.20 | 598.9 | 0.37% |
| hf_eager | 8 | 2.43 | 4.39 | 707.9 | 0.62% |
| hf_eager | 16 | 4.86 | 8.79 | 1192.0 | 0.74% |
| hf_eager | 32 | 9.72 | 17.58 | 2195.8 | 0.80% |
| ref_eager_fullsort | 1 | 0.30 | 0.55 | 138.9 | 0.40% |
| ref_eager_fullsort | 4 | 1.22 | 2.20 | 230.7 | 0.95% |
| ref_eager_fullsort | 8 | 2.43 | 4.39 | 347.6 | 1.26% |
| ref_eager_fullsort | 16 | 4.86 | 8.79 | 708.2 | 1.24% |
| ref_eager_fullsort | 32 | 9.72 | 17.58 | 1538.7 | 1.14% |
| tight_eager | 1 | 0.30 | 0.55 | 136.8 | 0.40% |
| tight_eager | 4 | 1.22 | 2.20 | 163.8 | 1.34% |
| tight_eager | 8 | 2.43 | 4.39 | 168.0 | 2.62% |
| tight_eager | 16 | 4.86 | 8.79 | 183.9 | 4.78% |
| tight_eager | 32 | 9.72 | 17.58 | 219.5 | 8.01% |

### Parameter sensitivity (bfloat16, hot)

| impl | vocab | top_k | top_p | batch | median µs |
|---|---|---|---|---|---|
| compile | 128256 | 20 | 0.9 | 1 | 159.9 |
| compile | 128256 | 20 | 0.9 | 32 | 203.5 |
| compile | 128256 | 20 | 0.95 | 1 | 160.0 |
| compile | 128256 | 20 | 0.95 | 32 | 202.3 |
| compile | 128256 | 50 | 0.9 | 1 | 158.1 |
| compile | 128256 | 50 | 0.9 | 32 | 203.0 |
| compile | 128256 | 50 | 0.95 | 1 | 157.9 |
| compile | 128256 | 50 | 0.95 | 32 | 202.5 |
| compile | 128256 | 100 | 0.9 | 1 | 158.0 |
| compile | 128256 | 100 | 0.9 | 32 | 203.1 |
| compile | 128256 | 100 | 0.95 | 1 | 158.3 |
| compile | 128256 | 100 | 0.95 | 32 | 202.8 |
| compile | 151936 | 20 | 0.9 | 1 | 161.0 |
| compile | 151936 | 20 | 0.9 | 32 | 218.8 |
| compile | 151936 | 20 | 0.95 | 1 | 160.9 |
| compile | 151936 | 20 | 0.95 | 32 | 218.7 |
| compile | 151936 | 50 | 0.9 | 1 | 157.3 |
| compile | 151936 | 50 | 0.9 | 4 | 152.3 |
| compile | 151936 | 50 | 0.9 | 8 | 152.3 |
| compile | 151936 | 50 | 0.9 | 16 | 183.7 |
| compile | 151936 | 50 | 0.9 | 32 | 220.3 |
| compile | 151936 | 50 | 0.95 | 1 | 159.6 |
| compile | 151936 | 50 | 0.95 | 32 | 219.4 |
| compile | 151936 | 100 | 0.9 | 1 | 159.3 |
| compile | 151936 | 100 | 0.9 | 32 | 221.1 |
| compile | 151936 | 100 | 0.95 | 1 | 159.3 |
| compile | 151936 | 100 | 0.95 | 32 | 223.2 |
| flashinfer | 128256 | 20 | 0.9 | 1 | 81.5 |
| flashinfer | 128256 | 20 | 0.9 | 32 | 141.8 |
| flashinfer | 128256 | 20 | 0.95 | 1 | 81.6 |
| flashinfer | 128256 | 20 | 0.95 | 32 | 141.1 |
| flashinfer | 128256 | 50 | 0.9 | 1 | 81.3 |
| flashinfer | 128256 | 50 | 0.9 | 32 | 141.8 |
| flashinfer | 128256 | 50 | 0.95 | 1 | 81.6 |
| flashinfer | 128256 | 50 | 0.95 | 32 | 141.3 |
| flashinfer | 128256 | 100 | 0.9 | 1 | 81.2 |
| flashinfer | 128256 | 100 | 0.9 | 32 | 141.9 |
| flashinfer | 128256 | 100 | 0.95 | 1 | 81.5 |
| flashinfer | 128256 | 100 | 0.95 | 32 | 141.3 |
| flashinfer | 151936 | 20 | 0.9 | 1 | 84.2 |
| flashinfer | 151936 | 20 | 0.9 | 32 | 177.0 |
| flashinfer | 151936 | 20 | 0.95 | 1 | 84.0 |
| flashinfer | 151936 | 20 | 0.95 | 32 | 176.6 |
| flashinfer | 151936 | 50 | 0.9 | 1 | 84.0 |
| flashinfer | 151936 | 50 | 0.9 | 4 | 85.6 |
| flashinfer | 151936 | 50 | 0.9 | 8 | 91.1 |
| flashinfer | 151936 | 50 | 0.9 | 16 | 116.0 |
| flashinfer | 151936 | 50 | 0.9 | 32 | 177.8 |
| flashinfer | 151936 | 50 | 0.95 | 1 | 83.9 |
| flashinfer | 151936 | 50 | 0.95 | 32 | 176.4 |
| flashinfer | 151936 | 100 | 0.9 | 1 | 84.5 |
| flashinfer | 151936 | 100 | 0.9 | 32 | 178.1 |
| flashinfer | 151936 | 100 | 0.95 | 1 | 84.3 |
| flashinfer | 151936 | 100 | 0.95 | 32 | 177.7 |
| flashinfer_from_probs | 128256 | 20 | 0.9 | 1 | 72.8 |
| flashinfer_from_probs | 128256 | 20 | 0.9 | 32 | 84.4 |
| flashinfer_from_probs | 128256 | 20 | 0.95 | 1 | 73.1 |
| flashinfer_from_probs | 128256 | 20 | 0.95 | 32 | 83.5 |
| flashinfer_from_probs | 128256 | 50 | 0.9 | 1 | 72.5 |
| flashinfer_from_probs | 128256 | 50 | 0.9 | 32 | 82.8 |
| flashinfer_from_probs | 128256 | 50 | 0.95 | 1 | 72.7 |
| flashinfer_from_probs | 128256 | 50 | 0.95 | 32 | 82.2 |
| flashinfer_from_probs | 128256 | 100 | 0.9 | 1 | 72.8 |
| flashinfer_from_probs | 128256 | 100 | 0.9 | 32 | 84.5 |
| flashinfer_from_probs | 128256 | 100 | 0.95 | 1 | 72.8 |
| flashinfer_from_probs | 128256 | 100 | 0.95 | 32 | 83.8 |
| flashinfer_from_probs | 151936 | 20 | 0.9 | 1 | 72.8 |
| flashinfer_from_probs | 151936 | 20 | 0.9 | 32 | 102.8 |
| flashinfer_from_probs | 151936 | 20 | 0.95 | 1 | 72.6 |
| flashinfer_from_probs | 151936 | 20 | 0.95 | 32 | 102.0 |
| flashinfer_from_probs | 151936 | 50 | 0.9 | 1 | 72.5 |
| flashinfer_from_probs | 151936 | 50 | 0.9 | 4 | 73.2 |
| flashinfer_from_probs | 151936 | 50 | 0.9 | 8 | 74.1 |
| flashinfer_from_probs | 151936 | 50 | 0.9 | 16 | 73.9 |
| flashinfer_from_probs | 151936 | 50 | 0.9 | 32 | 101.2 |
| flashinfer_from_probs | 151936 | 50 | 0.95 | 1 | 72.5 |
| flashinfer_from_probs | 151936 | 50 | 0.95 | 32 | 100.7 |
| flashinfer_from_probs | 151936 | 100 | 0.9 | 1 | 72.8 |
| flashinfer_from_probs | 151936 | 100 | 0.9 | 32 | 103.1 |
| flashinfer_from_probs | 151936 | 100 | 0.95 | 1 | 72.8 |
| flashinfer_from_probs | 151936 | 100 | 0.95 | 32 | 102.4 |
| fused_kernel | 128256 | 20 | 0.9 | 1 | 16.4 |
| fused_kernel | 128256 | 20 | 0.9 | 32 | 27.0 |
| fused_kernel | 128256 | 20 | 0.95 | 1 | 16.5 |
| fused_kernel | 128256 | 20 | 0.95 | 32 | 26.8 |
| fused_kernel | 128256 | 50 | 0.9 | 1 | 22.5 |
| fused_kernel | 128256 | 50 | 0.9 | 32 | 30.8 |
| fused_kernel | 128256 | 50 | 0.95 | 1 | 22.6 |
| fused_kernel | 128256 | 50 | 0.95 | 32 | 30.8 |
| fused_kernel | 128256 | 100 | 0.9 | 1 | 30.8 |
| fused_kernel | 128256 | 100 | 0.9 | 32 | 36.8 |
| fused_kernel | 128256 | 100 | 0.95 | 1 | 31.2 |
| fused_kernel | 128256 | 100 | 0.95 | 32 | 37.2 |
| fused_kernel | 151936 | 20 | 0.9 | 1 | 18.4 |
| fused_kernel | 151936 | 20 | 0.9 | 32 | 30.7 |
| fused_kernel | 151936 | 20 | 0.95 | 1 | 18.6 |
| fused_kernel | 151936 | 20 | 0.95 | 32 | 30.7 |
| fused_kernel | 151936 | 50 | 0.9 | 1 | 22.5 |
| fused_kernel | 151936 | 50 | 0.9 | 4 | 22.5 |
| fused_kernel | 151936 | 50 | 0.9 | 8 | 22.6 |
| fused_kernel | 151936 | 50 | 0.9 | 16 | 26.6 |
| fused_kernel | 151936 | 50 | 0.9 | 32 | 34.8 |
| fused_kernel | 151936 | 50 | 0.95 | 1 | 22.6 |
| fused_kernel | 151936 | 50 | 0.95 | 32 | 34.8 |
| fused_kernel | 151936 | 100 | 0.9 | 1 | 32.7 |
| fused_kernel | 151936 | 100 | 0.9 | 32 | 38.9 |
| fused_kernel | 151936 | 100 | 0.95 | 1 | 33.3 |
| fused_kernel | 151936 | 100 | 0.95 | 32 | 39.9 |
| graph_compile | 128256 | 20 | 0.9 | 1 | 77.0 |
| graph_compile | 128256 | 20 | 0.9 | 32 | 151.0 |
| graph_compile | 128256 | 20 | 0.95 | 1 | 76.8 |
| graph_compile | 128256 | 20 | 0.95 | 32 | 154.3 |
| graph_compile | 128256 | 50 | 0.9 | 1 | 77.8 |
| graph_compile | 128256 | 50 | 0.9 | 32 | 150.8 |
| graph_compile | 128256 | 50 | 0.95 | 1 | 77.9 |
| graph_compile | 128256 | 50 | 0.95 | 32 | 150.6 |
| graph_compile | 128256 | 100 | 0.9 | 1 | 78.3 |
| graph_compile | 128256 | 100 | 0.9 | 32 | 152.0 |
| graph_compile | 128256 | 100 | 0.95 | 1 | 78.3 |
| graph_compile | 128256 | 100 | 0.95 | 32 | 151.9 |
| graph_compile | 151936 | 20 | 0.9 | 1 | 86.9 |
| graph_compile | 151936 | 20 | 0.9 | 32 | 167.1 |
| graph_compile | 151936 | 20 | 0.95 | 1 | 81.5 |
| graph_compile | 151936 | 20 | 0.95 | 32 | 167.4 |
| graph_compile | 151936 | 50 | 0.9 | 1 | 77.4 |
| graph_compile | 151936 | 50 | 0.9 | 4 | 95.2 |
| graph_compile | 151936 | 50 | 0.9 | 8 | 98.4 |
| graph_compile | 151936 | 50 | 0.9 | 16 | 123.3 |
| graph_compile | 151936 | 50 | 0.9 | 32 | 169.2 |
| graph_compile | 151936 | 50 | 0.95 | 1 | 81.7 |
| graph_compile | 151936 | 50 | 0.95 | 32 | 168.1 |
| graph_compile | 151936 | 100 | 0.9 | 1 | 82.3 |
| graph_compile | 151936 | 100 | 0.9 | 32 | 169.5 |
| graph_compile | 151936 | 100 | 0.95 | 1 | 82.4 |
| graph_compile | 151936 | 100 | 0.95 | 32 | 170.0 |
| graph_eager | 128256 | 20 | 0.9 | 1 | 76.9 |
| graph_eager | 128256 | 20 | 0.9 | 32 | 150.9 |
| graph_eager | 128256 | 20 | 0.95 | 1 | 76.7 |
| graph_eager | 128256 | 20 | 0.95 | 32 | 150.6 |
| graph_eager | 128256 | 50 | 0.9 | 1 | 77.8 |
| graph_eager | 128256 | 50 | 0.9 | 32 | 150.7 |
| graph_eager | 128256 | 50 | 0.95 | 1 | 77.9 |
| graph_eager | 128256 | 50 | 0.95 | 32 | 150.5 |
| graph_eager | 128256 | 100 | 0.9 | 1 | 78.3 |
| graph_eager | 128256 | 100 | 0.9 | 32 | 151.9 |
| graph_eager | 128256 | 100 | 0.95 | 1 | 78.3 |
| graph_eager | 128256 | 100 | 0.95 | 32 | 151.9 |
| graph_eager | 151936 | 20 | 0.9 | 1 | 81.4 |
| graph_eager | 151936 | 20 | 0.9 | 32 | 166.9 |
| graph_eager | 151936 | 20 | 0.95 | 1 | 81.4 |
| graph_eager | 151936 | 20 | 0.95 | 32 | 167.2 |
| graph_eager | 151936 | 50 | 0.9 | 1 | 81.6 |
| graph_eager | 151936 | 50 | 0.9 | 4 | 106.0 |
| graph_eager | 151936 | 50 | 0.9 | 8 | 108.7 |
| graph_eager | 151936 | 50 | 0.9 | 16 | 123.2 |
| graph_eager | 151936 | 50 | 0.9 | 32 | 169.1 |
| graph_eager | 151936 | 50 | 0.95 | 1 | 81.6 |
| graph_eager | 151936 | 50 | 0.95 | 32 | 168.0 |
| graph_eager | 151936 | 100 | 0.9 | 1 | 82.1 |
| graph_eager | 151936 | 100 | 0.9 | 32 | 169.3 |
| graph_eager | 151936 | 100 | 0.95 | 1 | 82.2 |
| graph_eager | 151936 | 100 | 0.95 | 32 | 169.7 |
| hf_eager | 128256 | 20 | 0.9 | 1 | 338.4 |
| hf_eager | 128256 | 20 | 0.9 | 32 | 1839.1 |
| hf_eager | 128256 | 20 | 0.95 | 1 | 338.4 |
| hf_eager | 128256 | 20 | 0.95 | 32 | 1837.2 |
| hf_eager | 128256 | 50 | 0.9 | 1 | 340.3 |
| hf_eager | 128256 | 50 | 0.9 | 32 | 1840.4 |
| hf_eager | 128256 | 50 | 0.95 | 1 | 340.3 |
| hf_eager | 128256 | 50 | 0.95 | 32 | 1839.3 |
| hf_eager | 128256 | 100 | 0.9 | 1 | 341.4 |
| hf_eager | 128256 | 100 | 0.9 | 32 | 1836.2 |
| hf_eager | 128256 | 100 | 0.95 | 1 | 341.7 |
| hf_eager | 128256 | 100 | 0.95 | 32 | 1835.8 |
| hf_eager | 151936 | 20 | 0.9 | 1 | 323.1 |
| hf_eager | 151936 | 20 | 0.9 | 32 | 2188.7 |
| hf_eager | 151936 | 20 | 0.95 | 1 | 323.0 |
| hf_eager | 151936 | 20 | 0.95 | 32 | 2189.4 |
| hf_eager | 151936 | 50 | 0.9 | 1 | 325.3 |
| hf_eager | 151936 | 50 | 0.9 | 4 | 598.9 |
| hf_eager | 151936 | 50 | 0.9 | 8 | 707.9 |
| hf_eager | 151936 | 50 | 0.9 | 16 | 1192.0 |
| hf_eager | 151936 | 50 | 0.9 | 32 | 2195.8 |
| hf_eager | 151936 | 50 | 0.95 | 1 | 325.3 |
| hf_eager | 151936 | 50 | 0.95 | 32 | 2192.1 |
| hf_eager | 151936 | 100 | 0.9 | 1 | 326.6 |
| hf_eager | 151936 | 100 | 0.9 | 32 | 2192.4 |
| hf_eager | 151936 | 100 | 0.95 | 1 | 326.7 |
| hf_eager | 151936 | 100 | 0.95 | 32 | 2192.9 |
| ref_eager_fullsort | 128256 | 20 | 0.9 | 1 | 136.1 |
| ref_eager_fullsort | 128256 | 20 | 0.9 | 32 | 1281.5 |
| ref_eager_fullsort | 128256 | 20 | 0.95 | 1 | 136.0 |
| ref_eager_fullsort | 128256 | 20 | 0.95 | 32 | 1280.6 |
| ref_eager_fullsort | 128256 | 50 | 0.9 | 1 | 136.0 |
| ref_eager_fullsort | 128256 | 50 | 0.9 | 32 | 1280.3 |
| ref_eager_fullsort | 128256 | 50 | 0.95 | 1 | 135.4 |
| ref_eager_fullsort | 128256 | 50 | 0.95 | 32 | 1280.1 |
| ref_eager_fullsort | 128256 | 100 | 0.9 | 1 | 135.7 |
| ref_eager_fullsort | 128256 | 100 | 0.9 | 32 | 1278.8 |
| ref_eager_fullsort | 128256 | 100 | 0.95 | 1 | 135.7 |
| ref_eager_fullsort | 128256 | 100 | 0.95 | 32 | 1278.5 |
| ref_eager_fullsort | 151936 | 20 | 0.9 | 1 | 138.8 |
| ref_eager_fullsort | 151936 | 20 | 0.9 | 32 | 1535.0 |
| ref_eager_fullsort | 151936 | 20 | 0.95 | 1 | 138.8 |
| ref_eager_fullsort | 151936 | 20 | 0.95 | 32 | 1534.8 |
| ref_eager_fullsort | 151936 | 50 | 0.9 | 1 | 138.9 |
| ref_eager_fullsort | 151936 | 50 | 0.9 | 4 | 230.7 |
| ref_eager_fullsort | 151936 | 50 | 0.9 | 8 | 347.6 |
| ref_eager_fullsort | 151936 | 50 | 0.9 | 16 | 708.2 |
| ref_eager_fullsort | 151936 | 50 | 0.9 | 32 | 1538.7 |
| ref_eager_fullsort | 151936 | 50 | 0.95 | 1 | 138.9 |
| ref_eager_fullsort | 151936 | 50 | 0.95 | 32 | 1535.7 |
| ref_eager_fullsort | 151936 | 100 | 0.9 | 1 | 138.9 |
| ref_eager_fullsort | 151936 | 100 | 0.9 | 32 | 1536.5 |
| ref_eager_fullsort | 151936 | 100 | 0.95 | 1 | 138.8 |
| ref_eager_fullsort | 151936 | 100 | 0.95 | 32 | 1537.6 |
| tight_eager | 128256 | 20 | 0.9 | 1 | 132.4 |
| tight_eager | 128256 | 20 | 0.9 | 32 | 203.0 |
| tight_eager | 128256 | 20 | 0.95 | 1 | 132.4 |
| tight_eager | 128256 | 20 | 0.95 | 32 | 202.2 |
| tight_eager | 128256 | 50 | 0.9 | 1 | 136.5 |
| tight_eager | 128256 | 50 | 0.9 | 32 | 202.0 |
| tight_eager | 128256 | 50 | 0.95 | 1 | 136.4 |
| tight_eager | 128256 | 50 | 0.95 | 32 | 201.9 |
| tight_eager | 128256 | 100 | 0.9 | 1 | 136.4 |
| tight_eager | 128256 | 100 | 0.9 | 32 | 201.6 |
| tight_eager | 128256 | 100 | 0.95 | 1 | 136.4 |
| tight_eager | 128256 | 100 | 0.95 | 32 | 201.8 |
| tight_eager | 151936 | 20 | 0.9 | 1 | 133.3 |
| tight_eager | 151936 | 20 | 0.9 | 32 | 218.6 |
| tight_eager | 151936 | 20 | 0.95 | 1 | 133.0 |
| tight_eager | 151936 | 20 | 0.95 | 32 | 218.6 |
| tight_eager | 151936 | 50 | 0.9 | 1 | 136.8 |
| tight_eager | 151936 | 50 | 0.9 | 4 | 163.8 |
| tight_eager | 151936 | 50 | 0.9 | 8 | 168.0 |
| tight_eager | 151936 | 50 | 0.9 | 16 | 183.9 |
| tight_eager | 151936 | 50 | 0.9 | 32 | 219.5 |
| tight_eager | 151936 | 50 | 0.95 | 1 | 136.5 |
| tight_eager | 151936 | 50 | 0.95 | 32 | 219.1 |
| tight_eager | 151936 | 100 | 0.9 | 1 | 136.6 |
| tight_eager | 151936 | 100 | 0.9 | 32 | 219.5 |
| tight_eager | 151936 | 100 | 0.95 | 1 | 136.6 |
| tight_eager | 151936 | 100 | 0.95 | 32 | 219.9 |

### Round-to-round spread (impl order rotated each round)

| impl | rounds | min median µs | max median µs | spread |
|---|---|---|---|---|
| compile | 3 | 158.3 | 159.2 | 0.6% |
| flashinfer | 3 | 91.9 | 91.9 | 0.1% |
| flashinfer_from_probs | 3 | 73.9 | 74.2 | 0.5% |
| fused_kernel | 3 | 24.6 | 24.7 | 0.4% |
| graph_compile | 3 | 103.4 | 104.4 | 1.0% |
| graph_eager | 3 | 113.1 | 113.6 | 0.4% |
| hf_eager | 3 | 705.8 | 707.9 | 0.3% |
| ref_eager_fullsort | 3 | 350.2 | 352.0 | 0.5% |
| tight_eager | 3 | 168.4 | 169.7 | 0.7% |
