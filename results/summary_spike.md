# Sampling Ladder Summary

Source: `results/raw/spike_ladder.csv` (1134 rows). Median across rounds/reps; 
latency is amortized device time per sampling call, validation disabled in the timed region.


### Latency µs (speedup vs hf_eager) — V=151936, k=50, p=0.9, bfloat16, hot

| impl | B=1 | B=4 | B=8 | B=16 | B=32 |
|---|---|---|---|---|---|
| hf_eager | 325.2 (1.0x) | 598.7 (1.0x) | 707.0 (1.0x) | 1191.1 (1.0x) | 2195.0 (1.0x) |
| ref_eager_fullsort | 138.8 (2.3x) | 230.4 (2.6x) | 347.0 (2.0x) | 708.0 (1.7x) | 1538.5 (1.4x) |
| tight_eager | 136.5 (2.4x) | 164.3 (3.6x) | 168.5 (4.2x) | 183.0 (6.5x) | 219.3 (10.0x) |
| compile | 158.3 (2.1x) | 152.1 (3.9x) | 152.5 (4.6x) | 183.3 (6.5x) | 221.3 (9.9x) |
| graph_eager | 81.7 (4.0x) | 105.9 (5.7x) | 108.4 (6.5x) | 123.0 (9.7x) | 168.7 (13.0x) |
| graph_compile | 72.4 (4.5x) | 95.2 (6.3x) | 98.3 (7.2x) | 123.1 (9.7x) | 168.8 (13.0x) |
| flashinfer | 84.0 (3.9x) | 85.6 (7.0x) | 91.0 (7.8x) | 115.1 (10.3x) | 177.6 (12.4x) |
| flashinfer_from_probs | 75.4 (4.3x) | 75.4 (7.9x) | 75.5 (9.4x) | 75.4 (15.8x) | 101.2 (21.7x) |
| fused_kernel | 22.5 (14.4x) | 22.5 (26.6x) | 22.6 (31.4x) | 26.6 (44.7x) | 34.8 (63.1x) |

### Latency µs (speedup vs hf_eager) — V=151936, k=50, p=0.9, bfloat16, cold

| impl | B=1 | B=4 | B=8 | B=16 | B=32 |
|---|---|---|---|---|---|
| hf_eager | 325.4 (1.0x) | 599.1 (1.0x) | 707.3 (1.0x) | 1156.2 (1.0x) | 2192.6 (1.0x) |
| ref_eager_fullsort | 139.5 (2.3x) | 233.1 (2.6x) | 346.0 (2.0x) | 704.9 (1.6x) | 1537.6 (1.4x) |
| tight_eager | 137.4 (2.4x) | 166.2 (3.6x) | 170.0 (4.2x) | 188.2 (6.1x) | 230.7 (9.5x) |
| compile | 162.4 (2.0x) | 153.6 (3.9x) | 153.5 (4.6x) | 187.6 (6.2x) | 230.7 (9.5x) |
| graph_eager | 83.0 (3.9x) | 111.0 (5.4x) | 116.3 (6.1x) | 132.5 (8.7x) | 192.6 (11.4x) |
| graph_compile | 73.8 (4.4x) | 101.1 (5.9x) | 106.2 (6.7x) | 132.6 (8.7x) | 192.6 (11.4x) |
| flashinfer | 85.1 (3.8x) | 87.3 (6.9x) | 93.0 (7.6x) | 119.7 (9.7x) | 194.0 (11.3x) |
| flashinfer_from_probs | 75.8 (4.3x) | 75.2 (8.0x) | 75.6 (9.4x) | 78.3 (14.8x) | 127.6 (17.2x) |
| fused_kernel | 24.4 (13.4x) | 24.6 (24.4x) | 26.6 (26.6x) | 31.0 (37.3x) | 45.5 (48.2x) |

### Latency µs (speedup vs hf_eager) — V=151936, k=50, p=0.9, float16, hot

| impl | B=1 | B=4 | B=8 | B=16 | B=32 |
|---|---|---|---|---|---|
| hf_eager | 324.9 (1.0x) | 597.9 (1.0x) | 705.0 (1.0x) | 1188.4 (1.0x) | 2196.2 (1.0x) |
| ref_eager_fullsort | 141.7 (2.3x) | 236.6 (2.5x) | 354.9 (2.0x) | 699.1 (1.7x) | 1540.8 (1.4x) |
| tight_eager | 136.4 (2.4x) | 163.7 (3.7x) | 167.1 (4.2x) | 183.4 (6.5x) | 219.4 (10.0x) |
| compile | 158.6 (2.0x) | 152.5 (3.9x) | 152.3 (4.6x) | 158.6 (7.5x) | 192.4 (11.4x) |
| graph_eager | 81.6 (4.0x) | 105.0 (5.7x) | 108.1 (6.5x) | 122.8 (9.7x) | 168.3 (13.0x) |
| graph_compile | 72.0 (4.5x) | 95.1 (6.3x) | 98.3 (7.2x) | 110.1 (10.8x) | 152.0 (14.5x) |
| flashinfer | 84.0 (3.9x) | 85.3 (7.0x) | 90.5 (7.8x) | 115.2 (10.3x) | 177.1 (12.4x) |
| flashinfer_from_probs | 74.5 (4.4x) | 74.9 (8.0x) | 75.5 (9.3x) | 75.5 (15.7x) | 101.2 (21.7x) |
| fused_kernel | 22.5 (14.4x) | 22.5 (26.5x) | 22.5 (31.3x) | 24.6 (48.3x) | 33.6 (65.3x) |

### Latency µs (speedup vs hf_eager) — V=151936, k=50, p=0.9, float16, cold

| impl | B=1 | B=4 | B=8 | B=16 | B=32 |
|---|---|---|---|---|---|
| hf_eager | 325.4 (1.0x) | 599.2 (1.0x) | 707.3 (1.0x) | 1156.3 (1.0x) | 2192.2 (1.0x) |
| ref_eager_fullsort | 141.5 (2.3x) | 238.9 (2.5x) | 355.1 (2.0x) | 698.7 (1.7x) | 1543.5 (1.4x) |
| tight_eager | 137.1 (2.4x) | 165.8 (3.6x) | 169.4 (4.2x) | 187.5 (6.2x) | 230.3 (9.5x) |
| compile | 162.3 (2.0x) | 153.2 (3.9x) | 153.3 (4.6x) | 160.0 (7.2x) | 196.6 (11.2x) |
| graph_eager | 82.9 (3.9x) | 109.5 (5.5x) | 115.4 (6.1x) | 132.4 (8.7x) | 192.5 (11.4x) |
| graph_compile | 73.1 (4.5x) | 101.0 (5.9x) | 105.9 (6.7x) | 119.5 (9.7x) | 175.2 (12.5x) |
| flashinfer | 85.8 (3.8x) | 87.3 (6.9x) | 92.7 (7.6x) | 119.2 (9.7x) | 194.1 (11.3x) |
| flashinfer_from_probs | 75.7 (4.3x) | 75.2 (8.0x) | 75.4 (9.4x) | 78.3 (14.8x) | 127.6 (17.2x) |
| fused_kernel | 22.4 (14.5x) | 24.3 (24.7x) | 25.1 (28.2x) | 30.7 (37.6x) | 45.0 (48.7x) |

### Hot vs cold L2 residency

| impl | batch | dtype | hot µs | cold µs | cold penalty |
|---|---|---|---|---|---|
| compile | 1 | bfloat16 | 158.3 | 162.4 | +2.6% |
| compile | 1 | float16 | 158.6 | 162.3 | +2.3% |
| compile | 4 | bfloat16 | 152.1 | 153.6 | +1.0% |
| compile | 4 | float16 | 152.5 | 153.2 | +0.5% |
| compile | 8 | bfloat16 | 152.5 | 153.5 | +0.7% |
| compile | 8 | float16 | 152.3 | 153.3 | +0.7% |
| compile | 16 | bfloat16 | 183.3 | 187.6 | +2.3% |
| compile | 16 | float16 | 158.6 | 160.0 | +0.9% |
| compile | 32 | bfloat16 | 221.3 | 230.7 | +4.3% |
| compile | 32 | float16 | 192.4 | 196.6 | +2.2% |
| flashinfer | 1 | bfloat16 | 84.0 | 85.1 | +1.4% |
| flashinfer | 1 | float16 | 84.0 | 85.8 | +2.1% |
| flashinfer | 4 | bfloat16 | 85.6 | 87.3 | +2.1% |
| flashinfer | 4 | float16 | 85.3 | 87.3 | +2.4% |
| flashinfer | 8 | bfloat16 | 91.0 | 93.0 | +2.1% |
| flashinfer | 8 | float16 | 90.5 | 92.7 | +2.4% |
| flashinfer | 16 | bfloat16 | 115.1 | 119.7 | +4.0% |
| flashinfer | 16 | float16 | 115.2 | 119.2 | +3.5% |
| flashinfer | 32 | bfloat16 | 177.6 | 194.0 | +9.3% |
| flashinfer | 32 | float16 | 177.1 | 194.1 | +9.6% |
| flashinfer_from_probs | 1 | bfloat16 | 75.4 | 75.8 | +0.5% |
| flashinfer_from_probs | 1 | float16 | 74.5 | 75.7 | +1.6% |
| flashinfer_from_probs | 4 | bfloat16 | 75.4 | 75.2 | -0.2% |
| flashinfer_from_probs | 4 | float16 | 74.9 | 75.2 | +0.3% |
| flashinfer_from_probs | 8 | bfloat16 | 75.5 | 75.6 | +0.2% |
| flashinfer_from_probs | 8 | float16 | 75.5 | 75.4 | -0.2% |
| flashinfer_from_probs | 16 | bfloat16 | 75.4 | 78.3 | +3.9% |
| flashinfer_from_probs | 16 | float16 | 75.5 | 78.3 | +3.7% |
| flashinfer_from_probs | 32 | bfloat16 | 101.2 | 127.6 | +26.1% |
| flashinfer_from_probs | 32 | float16 | 101.2 | 127.6 | +26.1% |
| fused_kernel | 1 | bfloat16 | 22.5 | 24.4 | +8.1% |
| fused_kernel | 1 | float16 | 22.5 | 22.4 | -0.3% |
| fused_kernel | 4 | bfloat16 | 22.5 | 24.6 | +9.1% |
| fused_kernel | 4 | float16 | 22.5 | 24.3 | +7.7% |
| fused_kernel | 8 | bfloat16 | 22.6 | 26.6 | +18.0% |
| fused_kernel | 8 | float16 | 22.5 | 25.1 | +11.2% |
| fused_kernel | 16 | bfloat16 | 26.6 | 31.0 | +16.3% |
| fused_kernel | 16 | float16 | 24.6 | 30.7 | +24.8% |
| fused_kernel | 32 | bfloat16 | 34.8 | 45.5 | +30.6% |
| fused_kernel | 32 | float16 | 33.6 | 45.0 | +33.7% |
| graph_compile | 1 | bfloat16 | 72.4 | 73.8 | +1.9% |
| graph_compile | 1 | float16 | 72.0 | 73.1 | +1.5% |
| graph_compile | 4 | bfloat16 | 95.2 | 101.1 | +6.2% |
| graph_compile | 4 | float16 | 95.1 | 101.0 | +6.2% |
| graph_compile | 8 | bfloat16 | 98.3 | 106.2 | +8.0% |
| graph_compile | 8 | float16 | 98.3 | 105.9 | +7.7% |
| graph_compile | 16 | bfloat16 | 123.1 | 132.6 | +7.7% |
| graph_compile | 16 | float16 | 110.1 | 119.5 | +8.6% |
| graph_compile | 32 | bfloat16 | 168.8 | 192.6 | +14.1% |
| graph_compile | 32 | float16 | 152.0 | 175.2 | +15.3% |
| graph_eager | 1 | bfloat16 | 81.7 | 83.0 | +1.6% |
| graph_eager | 1 | float16 | 81.6 | 82.9 | +1.7% |
| graph_eager | 4 | bfloat16 | 105.9 | 111.0 | +4.9% |
| graph_eager | 4 | float16 | 105.0 | 109.5 | +4.3% |
| graph_eager | 8 | bfloat16 | 108.4 | 116.3 | +7.2% |
| graph_eager | 8 | float16 | 108.1 | 115.4 | +6.8% |
| graph_eager | 16 | bfloat16 | 123.0 | 132.5 | +7.7% |
| graph_eager | 16 | float16 | 122.8 | 132.4 | +7.9% |
| graph_eager | 32 | bfloat16 | 168.7 | 192.6 | +14.2% |
| graph_eager | 32 | float16 | 168.3 | 192.5 | +14.4% |
| hf_eager | 1 | bfloat16 | 325.2 | 325.4 | +0.1% |
| hf_eager | 1 | float16 | 324.9 | 325.4 | +0.2% |
| hf_eager | 4 | bfloat16 | 598.7 | 599.1 | +0.1% |
| hf_eager | 4 | float16 | 597.9 | 599.2 | +0.2% |
| hf_eager | 8 | bfloat16 | 707.0 | 707.3 | +0.0% |
| hf_eager | 8 | float16 | 705.0 | 707.3 | +0.3% |
| hf_eager | 16 | bfloat16 | 1191.1 | 1156.2 | -2.9% |
| hf_eager | 16 | float16 | 1188.4 | 1156.3 | -2.7% |
| hf_eager | 32 | bfloat16 | 2195.0 | 2192.6 | -0.1% |
| hf_eager | 32 | float16 | 2196.2 | 2192.2 | -0.2% |
| ref_eager_fullsort | 1 | bfloat16 | 138.8 | 139.5 | +0.5% |
| ref_eager_fullsort | 1 | float16 | 141.7 | 141.5 | -0.1% |
| ref_eager_fullsort | 4 | bfloat16 | 230.4 | 233.1 | +1.1% |
| ref_eager_fullsort | 4 | float16 | 236.6 | 238.9 | +1.0% |
| ref_eager_fullsort | 8 | bfloat16 | 347.0 | 346.0 | -0.3% |
| ref_eager_fullsort | 8 | float16 | 354.9 | 355.1 | +0.1% |
| ref_eager_fullsort | 16 | bfloat16 | 708.0 | 704.9 | -0.4% |
| ref_eager_fullsort | 16 | float16 | 699.1 | 698.7 | -0.1% |
| ref_eager_fullsort | 32 | bfloat16 | 1538.5 | 1537.6 | -0.1% |
| ref_eager_fullsort | 32 | float16 | 1540.8 | 1543.5 | +0.2% |
| tight_eager | 1 | bfloat16 | 136.5 | 137.4 | +0.6% |
| tight_eager | 1 | float16 | 136.4 | 137.1 | +0.5% |
| tight_eager | 4 | bfloat16 | 164.3 | 166.2 | +1.2% |
| tight_eager | 4 | float16 | 163.7 | 165.8 | +1.2% |
| tight_eager | 8 | bfloat16 | 168.5 | 170.0 | +0.9% |
| tight_eager | 8 | float16 | 167.1 | 169.4 | +1.3% |
| tight_eager | 16 | bfloat16 | 183.0 | 188.2 | +2.8% |
| tight_eager | 16 | float16 | 183.4 | 187.5 | +2.2% |
| tight_eager | 32 | bfloat16 | 219.3 | 230.7 | +5.2% |
| tight_eager | 32 | float16 | 219.4 | 230.3 | +5.0% |

### Memory-read floor vs measured latency (bfloat16, hot)

| impl | batch | logits MB | DRAM floor µs | latency µs | floor % of latency |
|---|---|---|---|---|---|
| compile | 1 | 0.30 | 0.55 | 158.3 | 0.35% |
| compile | 4 | 1.22 | 2.20 | 152.1 | 1.45% |
| compile | 8 | 2.43 | 4.39 | 152.5 | 2.88% |
| compile | 16 | 4.86 | 8.79 | 183.3 | 4.80% |
| compile | 32 | 9.72 | 17.58 | 221.3 | 7.95% |
| flashinfer | 1 | 0.30 | 0.55 | 84.0 | 0.65% |
| flashinfer | 4 | 1.22 | 2.20 | 85.6 | 2.57% |
| flashinfer | 8 | 2.43 | 4.39 | 91.0 | 4.83% |
| flashinfer | 16 | 4.86 | 8.79 | 115.1 | 7.64% |
| flashinfer | 32 | 9.72 | 17.58 | 177.6 | 9.90% |
| flashinfer_from_probs | 1 | 0.61 | 1.10 | 75.4 | 1.46% |
| flashinfer_from_probs | 4 | 2.43 | 4.39 | 75.4 | 5.83% |
| flashinfer_from_probs | 8 | 4.86 | 8.79 | 75.5 | 11.65% |
| flashinfer_from_probs | 16 | 9.72 | 17.58 | 75.4 | 23.32% |
| flashinfer_from_probs | 32 | 19.45 | 35.16 | 101.2 | 34.75% |
| fused_kernel | 1 | 0.30 | 0.55 | 22.5 | 2.44% |
| fused_kernel | 4 | 1.22 | 2.20 | 22.5 | 9.76% |
| fused_kernel | 8 | 2.43 | 4.39 | 22.6 | 19.49% |
| fused_kernel | 16 | 4.86 | 8.79 | 26.6 | 33.02% |
| fused_kernel | 32 | 9.72 | 17.58 | 34.8 | 50.52% |
| graph_compile | 1 | 0.30 | 0.55 | 72.4 | 0.76% |
| graph_compile | 4 | 1.22 | 2.20 | 95.2 | 2.31% |
| graph_compile | 8 | 2.43 | 4.39 | 98.3 | 4.47% |
| graph_compile | 16 | 4.86 | 8.79 | 123.1 | 7.14% |
| graph_compile | 32 | 9.72 | 17.58 | 168.8 | 10.42% |
| graph_eager | 1 | 0.30 | 0.55 | 81.7 | 0.67% |
| graph_eager | 4 | 1.22 | 2.20 | 105.9 | 2.08% |
| graph_eager | 8 | 2.43 | 4.39 | 108.4 | 4.05% |
| graph_eager | 16 | 4.86 | 8.79 | 123.0 | 7.15% |
| graph_eager | 32 | 9.72 | 17.58 | 168.7 | 10.42% |
| hf_eager | 1 | 0.30 | 0.55 | 325.2 | 0.17% |
| hf_eager | 4 | 1.22 | 2.20 | 598.7 | 0.37% |
| hf_eager | 8 | 2.43 | 4.39 | 707.0 | 0.62% |
| hf_eager | 16 | 4.86 | 8.79 | 1191.1 | 0.74% |
| hf_eager | 32 | 9.72 | 17.58 | 2195.0 | 0.80% |
| ref_eager_fullsort | 1 | 0.30 | 0.55 | 138.8 | 0.40% |
| ref_eager_fullsort | 4 | 1.22 | 2.20 | 230.4 | 0.95% |
| ref_eager_fullsort | 8 | 2.43 | 4.39 | 347.0 | 1.27% |
| ref_eager_fullsort | 16 | 4.86 | 8.79 | 708.0 | 1.24% |
| ref_eager_fullsort | 32 | 9.72 | 17.58 | 1538.5 | 1.14% |
| tight_eager | 1 | 0.30 | 0.55 | 136.5 | 0.40% |
| tight_eager | 4 | 1.22 | 2.20 | 164.3 | 1.34% |
| tight_eager | 8 | 2.43 | 4.39 | 168.5 | 2.61% |
| tight_eager | 16 | 4.86 | 8.79 | 183.0 | 4.80% |
| tight_eager | 32 | 9.72 | 17.58 | 219.3 | 8.02% |

### Parameter sensitivity (bfloat16, hot)

| impl | vocab | top_k | top_p | batch | median µs |
|---|---|---|---|---|---|
| compile | 128256 | 20 | 0.9 | 1 | 161.3 |
| compile | 128256 | 20 | 0.9 | 32 | 203.0 |
| compile | 128256 | 20 | 0.95 | 1 | 161.0 |
| compile | 128256 | 20 | 0.95 | 32 | 202.4 |
| compile | 128256 | 50 | 0.9 | 1 | 159.8 |
| compile | 128256 | 50 | 0.9 | 32 | 202.8 |
| compile | 128256 | 50 | 0.95 | 1 | 159.7 |
| compile | 128256 | 50 | 0.95 | 32 | 202.2 |
| compile | 128256 | 100 | 0.9 | 1 | 159.6 |
| compile | 128256 | 100 | 0.9 | 32 | 201.6 |
| compile | 128256 | 100 | 0.95 | 1 | 159.7 |
| compile | 128256 | 100 | 0.95 | 32 | 201.5 |
| compile | 151936 | 20 | 0.9 | 1 | 162.8 |
| compile | 151936 | 20 | 0.9 | 32 | 218.4 |
| compile | 151936 | 20 | 0.95 | 1 | 162.7 |
| compile | 151936 | 20 | 0.95 | 32 | 218.6 |
| compile | 151936 | 50 | 0.9 | 1 | 158.3 |
| compile | 151936 | 50 | 0.9 | 4 | 152.1 |
| compile | 151936 | 50 | 0.9 | 8 | 152.5 |
| compile | 151936 | 50 | 0.9 | 16 | 183.3 |
| compile | 151936 | 50 | 0.9 | 32 | 221.3 |
| compile | 151936 | 50 | 0.95 | 1 | 161.4 |
| compile | 151936 | 50 | 0.95 | 32 | 219.3 |
| compile | 151936 | 100 | 0.9 | 1 | 160.7 |
| compile | 151936 | 100 | 0.9 | 32 | 220.5 |
| compile | 151936 | 100 | 0.95 | 1 | 161.0 |
| compile | 151936 | 100 | 0.95 | 32 | 219.9 |
| flashinfer | 128256 | 20 | 0.9 | 1 | 83.7 |
| flashinfer | 128256 | 20 | 0.9 | 32 | 141.7 |
| flashinfer | 128256 | 20 | 0.95 | 1 | 84.1 |
| flashinfer | 128256 | 20 | 0.95 | 32 | 141.1 |
| flashinfer | 128256 | 50 | 0.9 | 1 | 83.6 |
| flashinfer | 128256 | 50 | 0.9 | 32 | 141.8 |
| flashinfer | 128256 | 50 | 0.95 | 1 | 83.5 |
| flashinfer | 128256 | 50 | 0.95 | 32 | 141.2 |
| flashinfer | 128256 | 100 | 0.9 | 1 | 83.8 |
| flashinfer | 128256 | 100 | 0.9 | 32 | 141.8 |
| flashinfer | 128256 | 100 | 0.95 | 1 | 83.8 |
| flashinfer | 128256 | 100 | 0.95 | 32 | 141.3 |
| flashinfer | 151936 | 20 | 0.9 | 1 | 84.0 |
| flashinfer | 151936 | 20 | 0.9 | 32 | 175.7 |
| flashinfer | 151936 | 20 | 0.95 | 1 | 83.9 |
| flashinfer | 151936 | 20 | 0.95 | 32 | 175.5 |
| flashinfer | 151936 | 50 | 0.9 | 1 | 84.0 |
| flashinfer | 151936 | 50 | 0.9 | 4 | 85.6 |
| flashinfer | 151936 | 50 | 0.9 | 8 | 91.0 |
| flashinfer | 151936 | 50 | 0.9 | 16 | 115.1 |
| flashinfer | 151936 | 50 | 0.9 | 32 | 177.6 |
| flashinfer | 151936 | 50 | 0.95 | 1 | 83.8 |
| flashinfer | 151936 | 50 | 0.95 | 32 | 176.2 |
| flashinfer | 151936 | 100 | 0.9 | 1 | 84.5 |
| flashinfer | 151936 | 100 | 0.9 | 32 | 177.0 |
| flashinfer | 151936 | 100 | 0.95 | 1 | 84.4 |
| flashinfer | 151936 | 100 | 0.95 | 32 | 176.7 |
| flashinfer_from_probs | 128256 | 20 | 0.9 | 1 | 75.0 |
| flashinfer_from_probs | 128256 | 20 | 0.9 | 32 | 84.3 |
| flashinfer_from_probs | 128256 | 20 | 0.95 | 1 | 74.8 |
| flashinfer_from_probs | 128256 | 20 | 0.95 | 32 | 83.5 |
| flashinfer_from_probs | 128256 | 50 | 0.9 | 1 | 75.1 |
| flashinfer_from_probs | 128256 | 50 | 0.9 | 32 | 82.7 |
| flashinfer_from_probs | 128256 | 50 | 0.95 | 1 | 74.5 |
| flashinfer_from_probs | 128256 | 50 | 0.95 | 32 | 82.2 |
| flashinfer_from_probs | 128256 | 100 | 0.9 | 1 | 74.9 |
| flashinfer_from_probs | 128256 | 100 | 0.9 | 32 | 84.5 |
| flashinfer_from_probs | 128256 | 100 | 0.95 | 1 | 74.9 |
| flashinfer_from_probs | 128256 | 100 | 0.95 | 32 | 83.7 |
| flashinfer_from_probs | 151936 | 20 | 0.9 | 1 | 74.9 |
| flashinfer_from_probs | 151936 | 20 | 0.9 | 32 | 102.8 |
| flashinfer_from_probs | 151936 | 20 | 0.95 | 1 | 74.9 |
| flashinfer_from_probs | 151936 | 20 | 0.95 | 32 | 102.0 |
| flashinfer_from_probs | 151936 | 50 | 0.9 | 1 | 75.4 |
| flashinfer_from_probs | 151936 | 50 | 0.9 | 4 | 75.4 |
| flashinfer_from_probs | 151936 | 50 | 0.9 | 8 | 75.5 |
| flashinfer_from_probs | 151936 | 50 | 0.9 | 16 | 75.4 |
| flashinfer_from_probs | 151936 | 50 | 0.9 | 32 | 101.2 |
| flashinfer_from_probs | 151936 | 50 | 0.95 | 1 | 74.8 |
| flashinfer_from_probs | 151936 | 50 | 0.95 | 32 | 100.7 |
| flashinfer_from_probs | 151936 | 100 | 0.9 | 1 | 75.0 |
| flashinfer_from_probs | 151936 | 100 | 0.9 | 32 | 103.0 |
| flashinfer_from_probs | 151936 | 100 | 0.95 | 1 | 74.9 |
| flashinfer_from_probs | 151936 | 100 | 0.95 | 32 | 102.3 |
| fused_kernel | 128256 | 20 | 0.9 | 1 | 18.3 |
| fused_kernel | 128256 | 20 | 0.9 | 32 | 27.0 |
| fused_kernel | 128256 | 20 | 0.95 | 1 | 18.4 |
| fused_kernel | 128256 | 20 | 0.95 | 32 | 26.9 |
| fused_kernel | 128256 | 50 | 0.9 | 1 | 22.5 |
| fused_kernel | 128256 | 50 | 0.9 | 32 | 30.9 |
| fused_kernel | 128256 | 50 | 0.95 | 1 | 22.5 |
| fused_kernel | 128256 | 50 | 0.95 | 32 | 30.9 |
| fused_kernel | 128256 | 100 | 0.9 | 1 | 27.7 |
| fused_kernel | 128256 | 100 | 0.9 | 32 | 35.5 |
| fused_kernel | 128256 | 100 | 0.95 | 1 | 28.8 |
| fused_kernel | 128256 | 100 | 0.95 | 32 | 36.7 |
| fused_kernel | 151936 | 20 | 0.9 | 1 | 20.3 |
| fused_kernel | 151936 | 20 | 0.9 | 32 | 30.7 |
| fused_kernel | 151936 | 20 | 0.95 | 1 | 20.5 |
| fused_kernel | 151936 | 20 | 0.95 | 32 | 30.7 |
| fused_kernel | 151936 | 50 | 0.9 | 1 | 22.5 |
| fused_kernel | 151936 | 50 | 0.9 | 4 | 22.5 |
| fused_kernel | 151936 | 50 | 0.9 | 8 | 22.6 |
| fused_kernel | 151936 | 50 | 0.9 | 16 | 26.6 |
| fused_kernel | 151936 | 50 | 0.9 | 32 | 34.8 |
| fused_kernel | 151936 | 50 | 0.95 | 1 | 22.5 |
| fused_kernel | 151936 | 50 | 0.95 | 32 | 34.8 |
| fused_kernel | 151936 | 100 | 0.9 | 1 | 30.3 |
| fused_kernel | 151936 | 100 | 0.9 | 32 | 38.3 |
| fused_kernel | 151936 | 100 | 0.95 | 1 | 30.7 |
| fused_kernel | 151936 | 100 | 0.95 | 32 | 38.9 |
| graph_compile | 128256 | 20 | 0.9 | 1 | 76.8 |
| graph_compile | 128256 | 20 | 0.9 | 32 | 150.5 |
| graph_compile | 128256 | 20 | 0.95 | 1 | 76.8 |
| graph_compile | 128256 | 20 | 0.95 | 32 | 150.1 |
| graph_compile | 128256 | 50 | 0.9 | 1 | 77.8 |
| graph_compile | 128256 | 50 | 0.9 | 32 | 150.9 |
| graph_compile | 128256 | 50 | 0.95 | 1 | 78.0 |
| graph_compile | 128256 | 50 | 0.95 | 32 | 154.5 |
| graph_compile | 128256 | 100 | 0.9 | 1 | 78.4 |
| graph_compile | 128256 | 100 | 0.9 | 32 | 151.9 |
| graph_compile | 128256 | 100 | 0.95 | 1 | 78.3 |
| graph_compile | 128256 | 100 | 0.95 | 32 | 151.9 |
| graph_compile | 151936 | 20 | 0.9 | 1 | 81.4 |
| graph_compile | 151936 | 20 | 0.9 | 32 | 166.6 |
| graph_compile | 151936 | 20 | 0.95 | 1 | 81.4 |
| graph_compile | 151936 | 20 | 0.95 | 32 | 167.0 |
| graph_compile | 151936 | 50 | 0.9 | 1 | 72.4 |
| graph_compile | 151936 | 50 | 0.9 | 4 | 95.2 |
| graph_compile | 151936 | 50 | 0.9 | 8 | 98.3 |
| graph_compile | 151936 | 50 | 0.9 | 16 | 123.1 |
| graph_compile | 151936 | 50 | 0.9 | 32 | 168.8 |
| graph_compile | 151936 | 50 | 0.95 | 1 | 87.4 |
| graph_compile | 151936 | 50 | 0.95 | 32 | 168.0 |
| graph_compile | 151936 | 100 | 0.9 | 1 | 82.2 |
| graph_compile | 151936 | 100 | 0.9 | 32 | 169.0 |
| graph_compile | 151936 | 100 | 0.95 | 1 | 82.2 |
| graph_compile | 151936 | 100 | 0.95 | 32 | 169.2 |
| graph_eager | 128256 | 20 | 0.9 | 1 | 76.8 |
| graph_eager | 128256 | 20 | 0.9 | 32 | 150.4 |
| graph_eager | 128256 | 20 | 0.95 | 1 | 76.8 |
| graph_eager | 128256 | 20 | 0.95 | 32 | 150.1 |
| graph_eager | 128256 | 50 | 0.9 | 1 | 77.8 |
| graph_eager | 128256 | 50 | 0.9 | 32 | 150.6 |
| graph_eager | 128256 | 50 | 0.95 | 1 | 78.0 |
| graph_eager | 128256 | 50 | 0.95 | 32 | 150.5 |
| graph_eager | 128256 | 100 | 0.9 | 1 | 78.3 |
| graph_eager | 128256 | 100 | 0.9 | 32 | 151.8 |
| graph_eager | 128256 | 100 | 0.95 | 1 | 78.3 |
| graph_eager | 128256 | 100 | 0.95 | 32 | 151.8 |
| graph_eager | 151936 | 20 | 0.9 | 1 | 81.3 |
| graph_eager | 151936 | 20 | 0.9 | 32 | 166.3 |
| graph_eager | 151936 | 20 | 0.95 | 1 | 81.4 |
| graph_eager | 151936 | 20 | 0.95 | 32 | 166.7 |
| graph_eager | 151936 | 50 | 0.9 | 1 | 81.7 |
| graph_eager | 151936 | 50 | 0.9 | 4 | 105.9 |
| graph_eager | 151936 | 50 | 0.9 | 8 | 108.4 |
| graph_eager | 151936 | 50 | 0.9 | 16 | 123.0 |
| graph_eager | 151936 | 50 | 0.9 | 32 | 168.7 |
| graph_eager | 151936 | 50 | 0.95 | 1 | 81.6 |
| graph_eager | 151936 | 50 | 0.95 | 32 | 167.9 |
| graph_eager | 151936 | 100 | 0.9 | 1 | 82.0 |
| graph_eager | 151936 | 100 | 0.9 | 32 | 168.8 |
| graph_eager | 151936 | 100 | 0.95 | 1 | 82.1 |
| graph_eager | 151936 | 100 | 0.95 | 32 | 168.9 |
| hf_eager | 128256 | 20 | 0.9 | 1 | 338.4 |
| hf_eager | 128256 | 20 | 0.9 | 32 | 1837.6 |
| hf_eager | 128256 | 20 | 0.95 | 1 | 338.3 |
| hf_eager | 128256 | 20 | 0.95 | 32 | 1836.3 |
| hf_eager | 128256 | 50 | 0.9 | 1 | 340.2 |
| hf_eager | 128256 | 50 | 0.9 | 32 | 1838.8 |
| hf_eager | 128256 | 50 | 0.95 | 1 | 340.3 |
| hf_eager | 128256 | 50 | 0.95 | 32 | 1837.7 |
| hf_eager | 128256 | 100 | 0.9 | 1 | 341.4 |
| hf_eager | 128256 | 100 | 0.9 | 32 | 1834.6 |
| hf_eager | 128256 | 100 | 0.95 | 1 | 341.4 |
| hf_eager | 128256 | 100 | 0.95 | 32 | 1834.0 |
| hf_eager | 151936 | 20 | 0.9 | 1 | 322.7 |
| hf_eager | 151936 | 20 | 0.9 | 32 | 2187.2 |
| hf_eager | 151936 | 20 | 0.95 | 1 | 322.9 |
| hf_eager | 151936 | 20 | 0.95 | 32 | 2187.6 |
| hf_eager | 151936 | 50 | 0.9 | 1 | 325.2 |
| hf_eager | 151936 | 50 | 0.9 | 4 | 598.7 |
| hf_eager | 151936 | 50 | 0.9 | 8 | 707.0 |
| hf_eager | 151936 | 50 | 0.9 | 16 | 1191.1 |
| hf_eager | 151936 | 50 | 0.9 | 32 | 2195.0 |
| hf_eager | 151936 | 50 | 0.95 | 1 | 325.1 |
| hf_eager | 151936 | 50 | 0.95 | 32 | 2191.1 |
| hf_eager | 151936 | 100 | 0.9 | 1 | 326.5 |
| hf_eager | 151936 | 100 | 0.9 | 32 | 2190.2 |
| hf_eager | 151936 | 100 | 0.95 | 1 | 326.5 |
| hf_eager | 151936 | 100 | 0.95 | 32 | 2190.7 |
| ref_eager_fullsort | 128256 | 20 | 0.9 | 1 | 136.0 |
| ref_eager_fullsort | 128256 | 20 | 0.9 | 32 | 1281.3 |
| ref_eager_fullsort | 128256 | 20 | 0.95 | 1 | 136.3 |
| ref_eager_fullsort | 128256 | 20 | 0.95 | 32 | 1281.0 |
| ref_eager_fullsort | 128256 | 50 | 0.9 | 1 | 136.1 |
| ref_eager_fullsort | 128256 | 50 | 0.9 | 32 | 1280.3 |
| ref_eager_fullsort | 128256 | 50 | 0.95 | 1 | 136.0 |
| ref_eager_fullsort | 128256 | 50 | 0.95 | 32 | 1280.3 |
| ref_eager_fullsort | 128256 | 100 | 0.9 | 1 | 136.2 |
| ref_eager_fullsort | 128256 | 100 | 0.9 | 32 | 1278.8 |
| ref_eager_fullsort | 128256 | 100 | 0.95 | 1 | 135.9 |
| ref_eager_fullsort | 128256 | 100 | 0.95 | 32 | 1278.6 |
| ref_eager_fullsort | 151936 | 20 | 0.9 | 1 | 138.6 |
| ref_eager_fullsort | 151936 | 20 | 0.9 | 32 | 1534.4 |
| ref_eager_fullsort | 151936 | 20 | 0.95 | 1 | 138.7 |
| ref_eager_fullsort | 151936 | 20 | 0.95 | 32 | 1534.6 |
| ref_eager_fullsort | 151936 | 50 | 0.9 | 1 | 138.8 |
| ref_eager_fullsort | 151936 | 50 | 0.9 | 4 | 230.4 |
| ref_eager_fullsort | 151936 | 50 | 0.9 | 8 | 347.0 |
| ref_eager_fullsort | 151936 | 50 | 0.9 | 16 | 708.0 |
| ref_eager_fullsort | 151936 | 50 | 0.9 | 32 | 1538.5 |
| ref_eager_fullsort | 151936 | 50 | 0.95 | 1 | 138.8 |
| ref_eager_fullsort | 151936 | 50 | 0.95 | 32 | 1535.2 |
| ref_eager_fullsort | 151936 | 100 | 0.9 | 1 | 138.9 |
| ref_eager_fullsort | 151936 | 100 | 0.9 | 32 | 1536.2 |
| ref_eager_fullsort | 151936 | 100 | 0.95 | 1 | 138.8 |
| ref_eager_fullsort | 151936 | 100 | 0.95 | 32 | 1536.3 |
| tight_eager | 128256 | 20 | 0.9 | 1 | 134.7 |
| tight_eager | 128256 | 20 | 0.9 | 32 | 202.2 |
| tight_eager | 128256 | 20 | 0.95 | 1 | 134.3 |
| tight_eager | 128256 | 20 | 0.95 | 32 | 202.2 |
| tight_eager | 128256 | 50 | 0.9 | 1 | 133.3 |
| tight_eager | 128256 | 50 | 0.9 | 32 | 202.1 |
| tight_eager | 128256 | 50 | 0.95 | 1 | 135.3 |
| tight_eager | 128256 | 50 | 0.95 | 32 | 201.7 |
| tight_eager | 128256 | 100 | 0.9 | 1 | 134.7 |
| tight_eager | 128256 | 100 | 0.9 | 32 | 201.3 |
| tight_eager | 128256 | 100 | 0.95 | 1 | 133.5 |
| tight_eager | 128256 | 100 | 0.95 | 32 | 201.1 |
| tight_eager | 151936 | 20 | 0.9 | 1 | 135.2 |
| tight_eager | 151936 | 20 | 0.9 | 32 | 216.9 |
| tight_eager | 151936 | 20 | 0.95 | 1 | 135.1 |
| tight_eager | 151936 | 20 | 0.95 | 32 | 217.3 |
| tight_eager | 151936 | 50 | 0.9 | 1 | 136.5 |
| tight_eager | 151936 | 50 | 0.9 | 4 | 164.3 |
| tight_eager | 151936 | 50 | 0.9 | 8 | 168.5 |
| tight_eager | 151936 | 50 | 0.9 | 16 | 183.0 |
| tight_eager | 151936 | 50 | 0.9 | 32 | 219.3 |
| tight_eager | 151936 | 50 | 0.95 | 1 | 135.9 |
| tight_eager | 151936 | 50 | 0.95 | 32 | 218.6 |
| tight_eager | 151936 | 100 | 0.9 | 1 | 136.0 |
| tight_eager | 151936 | 100 | 0.9 | 32 | 218.7 |
| tight_eager | 151936 | 100 | 0.95 | 1 | 135.8 |
| tight_eager | 151936 | 100 | 0.95 | 32 | 218.8 |

### Round-to-round spread (impl order rotated each round)

| impl | rounds | min median µs | max median µs | spread |
|---|---|---|---|---|
| compile | 3 | 158.4 | 159.2 | 0.5% |
| flashinfer | 3 | 91.9 | 91.9 | 0.0% |
| flashinfer_from_probs | 3 | 75.5 | 75.6 | 0.1% |
| fused_kernel | 3 | 24.6 | 24.6 | 0.1% |
| graph_compile | 3 | 103.3 | 106.0 | 2.5% |
| graph_eager | 3 | 113.1 | 113.3 | 0.2% |
| hf_eager | 3 | 705.3 | 707.3 | 0.3% |
| ref_eager_fullsort | 3 | 350.1 | 351.2 | 0.3% |
| tight_eager | 3 | 168.5 | 169.6 | 0.6% |
