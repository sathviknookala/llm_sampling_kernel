# Sampling Ladder Summary

Source: `results/raw/spike_ladder_regmerge.csv` (1134 rows). Median across rounds/reps; 
latency is amortized device time per sampling call, validation disabled in the timed region.


### Latency µs (speedup vs hf_eager) — V=151936, k=50, p=0.9, bfloat16, hot

| impl | B=1 | B=4 | B=8 | B=16 | B=32 |
|---|---|---|---|---|---|
| hf_eager | 325.1 (1.0x) | 598.5 (1.0x) | 706.7 (1.0x) | 1181.5 (1.0x) | 2200.5 (1.0x) |
| ref_eager_fullsort | 138.7 (2.3x) | 230.0 (2.6x) | 344.3 (2.1x) | 702.6 (1.7x) | 1536.6 (1.4x) |
| tight_eager | 136.8 (2.4x) | 163.8 (3.7x) | 167.2 (4.2x) | 183.1 (6.5x) | 219.0 (10.0x) |
| compile | 158.5 (2.1x) | 152.7 (3.9x) | 152.2 (4.6x) | 182.6 (6.5x) | 219.4 (10.0x) |
| graph_eager | 81.5 (4.0x) | 105.5 (5.7x) | 108.2 (6.5x) | 122.1 (9.7x) | 168.0 (13.1x) |
| graph_compile | 71.8 (4.5x) | 95.1 (6.3x) | 98.2 (7.2x) | 122.2 (9.7x) | 168.1 (13.1x) |
| flashinfer | 84.0 (3.9x) | 85.6 (7.0x) | 91.0 (7.8x) | 115.0 (10.3x) | 176.9 (12.4x) |
| flashinfer_from_probs | 72.8 (4.5x) | 73.8 (8.1x) | 74.3 (9.5x) | 74.3 (15.9x) | 101.2 (21.7x) |
| fused_kernel | 19.1 (17.0x) | 19.7 (30.4x) | 20.3 (34.8x) | 24.6 (48.1x) | 30.7 (71.6x) |

### Latency µs (speedup vs hf_eager) — V=151936, k=50, p=0.9, bfloat16, cold

| impl | B=1 | B=4 | B=8 | B=16 | B=32 |
|---|---|---|---|---|---|
| hf_eager | 325.5 (1.0x) | 599.1 (1.0x) | 705.4 (1.0x) | 1168.7 (1.0x) | 2186.8 (1.0x) |
| ref_eager_fullsort | 138.6 (2.3x) | 232.0 (2.6x) | 345.0 (2.0x) | 702.1 (1.7x) | 1536.7 (1.4x) |
| tight_eager | 136.2 (2.4x) | 165.9 (3.6x) | 169.4 (4.2x) | 187.2 (6.2x) | 229.6 (9.5x) |
| compile | 162.4 (2.0x) | 153.6 (3.9x) | 154.0 (4.6x) | 186.7 (6.3x) | 230.2 (9.5x) |
| graph_eager | 82.9 (3.9x) | 110.9 (5.4x) | 114.7 (6.1x) | 132.0 (8.9x) | 190.8 (11.5x) |
| graph_compile | 74.0 (4.4x) | 101.5 (5.9x) | 105.0 (6.7x) | 132.1 (8.8x) | 191.0 (11.5x) |
| flashinfer | 84.2 (3.9x) | 87.3 (6.9x) | 92.9 (7.6x) | 119.1 (9.8x) | 193.4 (11.3x) |
| flashinfer_from_probs | 74.6 (4.4x) | 74.6 (8.0x) | 74.8 (9.4x) | 78.3 (14.9x) | 127.6 (17.1x) |
| fused_kernel | 20.9 (15.6x) | 21.8 (27.4x) | 24.3 (29.0x) | 29.0 (40.3x) | 41.8 (52.3x) |

### Latency µs (speedup vs hf_eager) — V=151936, k=50, p=0.9, float16, hot

| impl | B=1 | B=4 | B=8 | B=16 | B=32 |
|---|---|---|---|---|---|
| hf_eager | 324.9 (1.0x) | 598.8 (1.0x) | 704.5 (1.0x) | 1180.2 (1.0x) | 2200.2 (1.0x) |
| ref_eager_fullsort | 141.7 (2.3x) | 235.5 (2.5x) | 354.1 (2.0x) | 693.5 (1.7x) | 1539.0 (1.4x) |
| tight_eager | 137.1 (2.4x) | 163.4 (3.7x) | 167.0 (4.2x) | 183.1 (6.4x) | 219.0 (10.0x) |
| compile | 159.5 (2.0x) | 152.3 (3.9x) | 152.6 (4.6x) | 158.4 (7.5x) | 191.8 (11.5x) |
| graph_eager | 81.5 (4.0x) | 104.5 (5.7x) | 107.8 (6.5x) | 122.2 (9.7x) | 166.6 (13.2x) |
| graph_compile | 71.8 (4.5x) | 95.0 (6.3x) | 98.2 (7.2x) | 109.6 (10.8x) | 150.7 (14.6x) |
| flashinfer | 84.0 (3.9x) | 85.3 (7.0x) | 90.6 (7.8x) | 115.0 (10.3x) | 176.3 (12.5x) |
| flashinfer_from_probs | 73.0 (4.4x) | 74.0 (8.1x) | 74.4 (9.5x) | 74.0 (16.0x) | 101.2 (21.7x) |
| fused_kernel | 19.0 (17.1x) | 19.6 (30.6x) | 20.2 (34.9x) | 22.5 (52.4x) | 30.3 (72.5x) |

### Latency µs (speedup vs hf_eager) — V=151936, k=50, p=0.9, float16, cold

| impl | B=1 | B=4 | B=8 | B=16 | B=32 |
|---|---|---|---|---|---|
| hf_eager | 325.4 (1.0x) | 598.8 (1.0x) | 704.4 (1.0x) | 1168.7 (1.0x) | 2186.6 (1.0x) |
| ref_eager_fullsort | 141.7 (2.3x) | 237.4 (2.5x) | 354.7 (2.0x) | 694.7 (1.7x) | 1542.2 (1.4x) |
| tight_eager | 136.0 (2.4x) | 165.1 (3.6x) | 169.7 (4.2x) | 187.4 (6.2x) | 229.4 (9.5x) |
| compile | 161.7 (2.0x) | 153.5 (3.9x) | 153.6 (4.6x) | 160.0 (7.3x) | 196.1 (11.1x) |
| graph_eager | 82.9 (3.9x) | 109.7 (5.5x) | 114.3 (6.2x) | 132.2 (8.8x) | 190.4 (11.5x) |
| graph_compile | 74.0 (4.4x) | 101.4 (5.9x) | 104.9 (6.7x) | 119.5 (9.8x) | 173.7 (12.6x) |
| flashinfer | 84.2 (3.9x) | 87.4 (6.9x) | 92.7 (7.6x) | 119.3 (9.8x) | 193.4 (11.3x) |
| flashinfer_from_probs | 74.6 (4.4x) | 74.2 (8.1x) | 74.8 (9.4x) | 78.3 (14.9x) | 127.6 (17.1x) |
| fused_kernel | 18.9 (17.2x) | 21.3 (28.1x) | 22.5 (31.3x) | 28.6 (40.8x) | 40.9 (53.4x) |

### Hot vs cold L2 residency

| impl | batch | dtype | hot µs | cold µs | cold penalty |
|---|---|---|---|---|---|
| compile | 1 | bfloat16 | 158.5 | 162.4 | +2.5% |
| compile | 1 | float16 | 159.5 | 161.7 | +1.4% |
| compile | 4 | bfloat16 | 152.7 | 153.6 | +0.6% |
| compile | 4 | float16 | 152.3 | 153.5 | +0.8% |
| compile | 8 | bfloat16 | 152.2 | 154.0 | +1.2% |
| compile | 8 | float16 | 152.6 | 153.6 | +0.7% |
| compile | 16 | bfloat16 | 182.6 | 186.7 | +2.2% |
| compile | 16 | float16 | 158.4 | 160.0 | +1.0% |
| compile | 32 | bfloat16 | 219.4 | 230.2 | +4.9% |
| compile | 32 | float16 | 191.8 | 196.1 | +2.3% |
| flashinfer | 1 | bfloat16 | 84.0 | 84.2 | +0.2% |
| flashinfer | 1 | float16 | 84.0 | 84.2 | +0.2% |
| flashinfer | 4 | bfloat16 | 85.6 | 87.3 | +2.1% |
| flashinfer | 4 | float16 | 85.3 | 87.4 | +2.4% |
| flashinfer | 8 | bfloat16 | 91.0 | 92.9 | +2.1% |
| flashinfer | 8 | float16 | 90.6 | 92.7 | +2.4% |
| flashinfer | 16 | bfloat16 | 115.0 | 119.1 | +3.6% |
| flashinfer | 16 | float16 | 115.0 | 119.3 | +3.7% |
| flashinfer | 32 | bfloat16 | 176.9 | 193.4 | +9.3% |
| flashinfer | 32 | float16 | 176.3 | 193.4 | +9.7% |
| flashinfer_from_probs | 1 | bfloat16 | 72.8 | 74.6 | +2.5% |
| flashinfer_from_probs | 1 | float16 | 73.0 | 74.6 | +2.2% |
| flashinfer_from_probs | 4 | bfloat16 | 73.8 | 74.6 | +1.2% |
| flashinfer_from_probs | 4 | float16 | 74.0 | 74.2 | +0.2% |
| flashinfer_from_probs | 8 | bfloat16 | 74.3 | 74.8 | +0.7% |
| flashinfer_from_probs | 8 | float16 | 74.4 | 74.8 | +0.5% |
| flashinfer_from_probs | 16 | bfloat16 | 74.3 | 78.3 | +5.4% |
| flashinfer_from_probs | 16 | float16 | 74.0 | 78.3 | +5.9% |
| flashinfer_from_probs | 32 | bfloat16 | 101.2 | 127.6 | +26.1% |
| flashinfer_from_probs | 32 | float16 | 101.2 | 127.6 | +26.1% |
| fused_kernel | 1 | bfloat16 | 19.1 | 20.9 | +9.5% |
| fused_kernel | 1 | float16 | 19.0 | 18.9 | -0.4% |
| fused_kernel | 4 | bfloat16 | 19.7 | 21.8 | +10.8% |
| fused_kernel | 4 | float16 | 19.6 | 21.3 | +9.0% |
| fused_kernel | 8 | bfloat16 | 20.3 | 24.3 | +19.8% |
| fused_kernel | 8 | float16 | 20.2 | 22.5 | +11.7% |
| fused_kernel | 16 | bfloat16 | 24.6 | 29.0 | +18.0% |
| fused_kernel | 16 | float16 | 22.5 | 28.6 | +27.2% |
| fused_kernel | 32 | bfloat16 | 30.7 | 41.8 | +36.1% |
| fused_kernel | 32 | float16 | 30.3 | 40.9 | +34.9% |
| graph_compile | 1 | bfloat16 | 71.8 | 74.0 | +3.1% |
| graph_compile | 1 | float16 | 71.8 | 74.0 | +3.0% |
| graph_compile | 4 | bfloat16 | 95.1 | 101.5 | +6.8% |
| graph_compile | 4 | float16 | 95.0 | 101.4 | +6.7% |
| graph_compile | 8 | bfloat16 | 98.2 | 105.0 | +7.0% |
| graph_compile | 8 | float16 | 98.2 | 104.9 | +6.8% |
| graph_compile | 16 | bfloat16 | 122.2 | 132.1 | +8.1% |
| graph_compile | 16 | float16 | 109.6 | 119.5 | +9.0% |
| graph_compile | 32 | bfloat16 | 168.1 | 191.0 | +13.6% |
| graph_compile | 32 | float16 | 150.7 | 173.7 | +15.3% |
| graph_eager | 1 | bfloat16 | 81.5 | 82.9 | +1.7% |
| graph_eager | 1 | float16 | 81.5 | 82.9 | +1.7% |
| graph_eager | 4 | bfloat16 | 105.5 | 110.9 | +5.1% |
| graph_eager | 4 | float16 | 104.5 | 109.7 | +5.0% |
| graph_eager | 8 | bfloat16 | 108.2 | 114.7 | +6.1% |
| graph_eager | 8 | float16 | 107.8 | 114.3 | +6.0% |
| graph_eager | 16 | bfloat16 | 122.1 | 132.0 | +8.1% |
| graph_eager | 16 | float16 | 122.2 | 132.2 | +8.2% |
| graph_eager | 32 | bfloat16 | 168.0 | 190.8 | +13.6% |
| graph_eager | 32 | float16 | 166.6 | 190.4 | +14.3% |
| hf_eager | 1 | bfloat16 | 325.1 | 325.5 | +0.1% |
| hf_eager | 1 | float16 | 324.9 | 325.4 | +0.2% |
| hf_eager | 4 | bfloat16 | 598.5 | 599.1 | +0.1% |
| hf_eager | 4 | float16 | 598.8 | 598.8 | +0.0% |
| hf_eager | 8 | bfloat16 | 706.7 | 705.4 | -0.2% |
| hf_eager | 8 | float16 | 704.5 | 704.4 | -0.0% |
| hf_eager | 16 | bfloat16 | 1181.5 | 1168.7 | -1.1% |
| hf_eager | 16 | float16 | 1180.2 | 1168.7 | -1.0% |
| hf_eager | 32 | bfloat16 | 2200.5 | 2186.8 | -0.6% |
| hf_eager | 32 | float16 | 2200.2 | 2186.6 | -0.6% |
| ref_eager_fullsort | 1 | bfloat16 | 138.7 | 138.6 | -0.0% |
| ref_eager_fullsort | 1 | float16 | 141.7 | 141.7 | -0.0% |
| ref_eager_fullsort | 4 | bfloat16 | 230.0 | 232.0 | +0.9% |
| ref_eager_fullsort | 4 | float16 | 235.5 | 237.4 | +0.8% |
| ref_eager_fullsort | 8 | bfloat16 | 344.3 | 345.0 | +0.2% |
| ref_eager_fullsort | 8 | float16 | 354.1 | 354.7 | +0.2% |
| ref_eager_fullsort | 16 | bfloat16 | 702.6 | 702.1 | -0.1% |
| ref_eager_fullsort | 16 | float16 | 693.5 | 694.7 | +0.2% |
| ref_eager_fullsort | 32 | bfloat16 | 1536.6 | 1536.7 | +0.0% |
| ref_eager_fullsort | 32 | float16 | 1539.0 | 1542.2 | +0.2% |
| tight_eager | 1 | bfloat16 | 136.8 | 136.2 | -0.4% |
| tight_eager | 1 | float16 | 137.1 | 136.0 | -0.8% |
| tight_eager | 4 | bfloat16 | 163.8 | 165.9 | +1.3% |
| tight_eager | 4 | float16 | 163.4 | 165.1 | +1.0% |
| tight_eager | 8 | bfloat16 | 167.2 | 169.4 | +1.3% |
| tight_eager | 8 | float16 | 167.0 | 169.7 | +1.6% |
| tight_eager | 16 | bfloat16 | 183.1 | 187.2 | +2.3% |
| tight_eager | 16 | float16 | 183.1 | 187.4 | +2.3% |
| tight_eager | 32 | bfloat16 | 219.0 | 229.6 | +4.9% |
| tight_eager | 32 | float16 | 219.0 | 229.4 | +4.8% |

### Memory-read floor vs measured latency (bfloat16, hot)

| impl | batch | logits MB | DRAM floor µs | latency µs | floor % of latency |
|---|---|---|---|---|---|
| compile | 1 | 0.30 | 0.55 | 158.5 | 0.35% |
| compile | 4 | 1.22 | 2.19 | 152.7 | 1.44% |
| compile | 8 | 2.43 | 4.39 | 152.2 | 2.88% |
| compile | 16 | 4.86 | 8.77 | 182.6 | 4.80% |
| compile | 32 | 9.72 | 17.54 | 219.4 | 8.00% |
| flashinfer | 1 | 0.30 | 0.55 | 84.0 | 0.65% |
| flashinfer | 4 | 1.22 | 2.19 | 85.6 | 2.56% |
| flashinfer | 8 | 2.43 | 4.39 | 91.0 | 4.82% |
| flashinfer | 16 | 4.86 | 8.77 | 115.0 | 7.63% |
| flashinfer | 32 | 9.72 | 17.54 | 176.9 | 9.92% |
| flashinfer_from_probs | 1 | 0.61 | 1.10 | 72.8 | 1.51% |
| flashinfer_from_probs | 4 | 2.43 | 4.39 | 73.8 | 5.94% |
| flashinfer_from_probs | 8 | 4.86 | 8.77 | 74.3 | 11.81% |
| flashinfer_from_probs | 16 | 9.72 | 17.54 | 74.3 | 23.62% |
| flashinfer_from_probs | 32 | 19.45 | 35.09 | 101.2 | 34.67% |
| fused_kernel | 1 | 0.30 | 0.55 | 19.1 | 2.87% |
| fused_kernel | 4 | 1.22 | 2.19 | 19.7 | 11.13% |
| fused_kernel | 8 | 2.43 | 4.39 | 20.3 | 21.60% |
| fused_kernel | 16 | 4.86 | 8.77 | 24.6 | 35.72% |
| fused_kernel | 32 | 9.72 | 17.54 | 30.7 | 57.11% |
| graph_compile | 1 | 0.30 | 0.55 | 71.8 | 0.76% |
| graph_compile | 4 | 1.22 | 2.19 | 95.1 | 2.31% |
| graph_compile | 8 | 2.43 | 4.39 | 98.2 | 4.47% |
| graph_compile | 16 | 4.86 | 8.77 | 122.2 | 7.18% |
| graph_compile | 32 | 9.72 | 17.54 | 168.1 | 10.44% |
| graph_eager | 1 | 0.30 | 0.55 | 81.5 | 0.67% |
| graph_eager | 4 | 1.22 | 2.19 | 105.5 | 2.08% |
| graph_eager | 8 | 2.43 | 4.39 | 108.2 | 4.06% |
| graph_eager | 16 | 4.86 | 8.77 | 122.1 | 7.18% |
| graph_eager | 32 | 9.72 | 17.54 | 168.0 | 10.44% |
| hf_eager | 1 | 0.30 | 0.55 | 325.1 | 0.17% |
| hf_eager | 4 | 1.22 | 2.19 | 598.5 | 0.37% |
| hf_eager | 8 | 2.43 | 4.39 | 706.7 | 0.62% |
| hf_eager | 16 | 4.86 | 8.77 | 1181.5 | 0.74% |
| hf_eager | 32 | 9.72 | 17.54 | 2200.5 | 0.80% |
| ref_eager_fullsort | 1 | 0.30 | 0.55 | 138.7 | 0.40% |
| ref_eager_fullsort | 4 | 1.22 | 2.19 | 230.0 | 0.95% |
| ref_eager_fullsort | 8 | 2.43 | 4.39 | 344.3 | 1.27% |
| ref_eager_fullsort | 16 | 4.86 | 8.77 | 702.6 | 1.25% |
| ref_eager_fullsort | 32 | 9.72 | 17.54 | 1536.6 | 1.14% |
| tight_eager | 1 | 0.30 | 0.55 | 136.8 | 0.40% |
| tight_eager | 4 | 1.22 | 2.19 | 163.8 | 1.34% |
| tight_eager | 8 | 2.43 | 4.39 | 167.2 | 2.62% |
| tight_eager | 16 | 4.86 | 8.77 | 183.1 | 4.79% |
| tight_eager | 32 | 9.72 | 17.54 | 219.0 | 8.01% |

### Parameter sensitivity (bfloat16, hot)

| impl | vocab | top_k | top_p | batch | median µs |
|---|---|---|---|---|---|
| compile | 128256 | 20 | 0.9 | 1 | 160.0 |
| compile | 128256 | 20 | 0.9 | 32 | 202.1 |
| compile | 128256 | 20 | 0.95 | 1 | 160.8 |
| compile | 128256 | 20 | 0.95 | 32 | 203.0 |
| compile | 128256 | 50 | 0.9 | 1 | 158.1 |
| compile | 128256 | 50 | 0.9 | 32 | 202.3 |
| compile | 128256 | 50 | 0.95 | 1 | 158.1 |
| compile | 128256 | 50 | 0.95 | 32 | 201.4 |
| compile | 128256 | 100 | 0.9 | 1 | 158.3 |
| compile | 128256 | 100 | 0.9 | 32 | 202.0 |
| compile | 128256 | 100 | 0.95 | 1 | 158.1 |
| compile | 128256 | 100 | 0.95 | 32 | 201.3 |
| compile | 151936 | 20 | 0.9 | 1 | 160.9 |
| compile | 151936 | 20 | 0.9 | 32 | 218.4 |
| compile | 151936 | 20 | 0.95 | 1 | 161.4 |
| compile | 151936 | 20 | 0.95 | 32 | 217.6 |
| compile | 151936 | 50 | 0.9 | 1 | 158.5 |
| compile | 151936 | 50 | 0.9 | 4 | 152.7 |
| compile | 151936 | 50 | 0.9 | 8 | 152.2 |
| compile | 151936 | 50 | 0.9 | 16 | 182.6 |
| compile | 151936 | 50 | 0.9 | 32 | 219.4 |
| compile | 151936 | 50 | 0.95 | 1 | 160.6 |
| compile | 151936 | 50 | 0.95 | 32 | 218.8 |
| compile | 151936 | 100 | 0.9 | 1 | 159.0 |
| compile | 151936 | 100 | 0.9 | 32 | 218.0 |
| compile | 151936 | 100 | 0.95 | 1 | 159.0 |
| compile | 151936 | 100 | 0.95 | 32 | 218.7 |
| flashinfer | 128256 | 20 | 0.9 | 1 | 82.4 |
| flashinfer | 128256 | 20 | 0.9 | 32 | 141.7 |
| flashinfer | 128256 | 20 | 0.95 | 1 | 82.4 |
| flashinfer | 128256 | 20 | 0.95 | 32 | 141.1 |
| flashinfer | 128256 | 50 | 0.9 | 1 | 82.1 |
| flashinfer | 128256 | 50 | 0.9 | 32 | 141.7 |
| flashinfer | 128256 | 50 | 0.95 | 1 | 82.4 |
| flashinfer | 128256 | 50 | 0.95 | 32 | 141.2 |
| flashinfer | 128256 | 100 | 0.9 | 1 | 82.1 |
| flashinfer | 128256 | 100 | 0.9 | 32 | 141.8 |
| flashinfer | 128256 | 100 | 0.95 | 1 | 82.2 |
| flashinfer | 128256 | 100 | 0.95 | 32 | 141.2 |
| flashinfer | 151936 | 20 | 0.9 | 1 | 83.7 |
| flashinfer | 151936 | 20 | 0.9 | 32 | 174.5 |
| flashinfer | 151936 | 20 | 0.95 | 1 | 84.0 |
| flashinfer | 151936 | 20 | 0.95 | 32 | 174.3 |
| flashinfer | 151936 | 50 | 0.9 | 1 | 84.0 |
| flashinfer | 151936 | 50 | 0.9 | 4 | 85.6 |
| flashinfer | 151936 | 50 | 0.9 | 8 | 91.0 |
| flashinfer | 151936 | 50 | 0.9 | 16 | 115.0 |
| flashinfer | 151936 | 50 | 0.9 | 32 | 176.9 |
| flashinfer | 151936 | 50 | 0.95 | 1 | 83.9 |
| flashinfer | 151936 | 50 | 0.95 | 32 | 175.3 |
| flashinfer | 151936 | 100 | 0.9 | 1 | 84.6 |
| flashinfer | 151936 | 100 | 0.9 | 32 | 176.7 |
| flashinfer | 151936 | 100 | 0.95 | 1 | 84.4 |
| flashinfer | 151936 | 100 | 0.95 | 32 | 176.1 |
| flashinfer_from_probs | 128256 | 20 | 0.9 | 1 | 73.4 |
| flashinfer_from_probs | 128256 | 20 | 0.9 | 32 | 84.3 |
| flashinfer_from_probs | 128256 | 20 | 0.95 | 1 | 73.3 |
| flashinfer_from_probs | 128256 | 20 | 0.95 | 32 | 83.5 |
| flashinfer_from_probs | 128256 | 50 | 0.9 | 1 | 73.2 |
| flashinfer_from_probs | 128256 | 50 | 0.9 | 32 | 82.7 |
| flashinfer_from_probs | 128256 | 50 | 0.95 | 1 | 73.5 |
| flashinfer_from_probs | 128256 | 50 | 0.95 | 32 | 82.3 |
| flashinfer_from_probs | 128256 | 100 | 0.9 | 1 | 73.5 |
| flashinfer_from_probs | 128256 | 100 | 0.9 | 32 | 84.5 |
| flashinfer_from_probs | 128256 | 100 | 0.95 | 1 | 73.6 |
| flashinfer_from_probs | 128256 | 100 | 0.95 | 32 | 83.8 |
| flashinfer_from_probs | 151936 | 20 | 0.9 | 1 | 73.4 |
| flashinfer_from_probs | 151936 | 20 | 0.9 | 32 | 102.8 |
| flashinfer_from_probs | 151936 | 20 | 0.95 | 1 | 73.0 |
| flashinfer_from_probs | 151936 | 20 | 0.95 | 32 | 102.0 |
| flashinfer_from_probs | 151936 | 50 | 0.9 | 1 | 72.8 |
| flashinfer_from_probs | 151936 | 50 | 0.9 | 4 | 73.8 |
| flashinfer_from_probs | 151936 | 50 | 0.9 | 8 | 74.3 |
| flashinfer_from_probs | 151936 | 50 | 0.9 | 16 | 74.3 |
| flashinfer_from_probs | 151936 | 50 | 0.9 | 32 | 101.2 |
| flashinfer_from_probs | 151936 | 50 | 0.95 | 1 | 73.2 |
| flashinfer_from_probs | 151936 | 50 | 0.95 | 32 | 100.7 |
| flashinfer_from_probs | 151936 | 100 | 0.9 | 1 | 73.3 |
| flashinfer_from_probs | 151936 | 100 | 0.9 | 32 | 103.0 |
| flashinfer_from_probs | 151936 | 100 | 0.95 | 1 | 73.0 |
| flashinfer_from_probs | 151936 | 100 | 0.95 | 32 | 102.3 |
| fused_kernel | 128256 | 20 | 0.9 | 1 | 14.3 |
| fused_kernel | 128256 | 20 | 0.9 | 32 | 24.3 |
| fused_kernel | 128256 | 20 | 0.95 | 1 | 14.3 |
| fused_kernel | 128256 | 20 | 0.95 | 32 | 24.7 |
| fused_kernel | 128256 | 50 | 0.9 | 1 | 19.0 |
| fused_kernel | 128256 | 50 | 0.9 | 32 | 27.7 |
| fused_kernel | 128256 | 50 | 0.95 | 1 | 19.9 |
| fused_kernel | 128256 | 50 | 0.95 | 32 | 27.7 |
| fused_kernel | 128256 | 100 | 0.9 | 1 | 22.7 |
| fused_kernel | 128256 | 100 | 0.9 | 32 | 32.8 |
| fused_kernel | 128256 | 100 | 0.95 | 1 | 23.4 |
| fused_kernel | 128256 | 100 | 0.95 | 32 | 33.5 |
| fused_kernel | 151936 | 20 | 0.9 | 1 | 16.3 |
| fused_kernel | 151936 | 20 | 0.9 | 32 | 27.0 |
| fused_kernel | 151936 | 20 | 0.95 | 1 | 16.3 |
| fused_kernel | 151936 | 20 | 0.95 | 32 | 27.9 |
| fused_kernel | 151936 | 50 | 0.9 | 1 | 19.1 |
| fused_kernel | 151936 | 50 | 0.9 | 4 | 19.7 |
| fused_kernel | 151936 | 50 | 0.9 | 8 | 20.3 |
| fused_kernel | 151936 | 50 | 0.9 | 16 | 24.6 |
| fused_kernel | 151936 | 50 | 0.9 | 32 | 30.7 |
| fused_kernel | 151936 | 50 | 0.95 | 1 | 20.3 |
| fused_kernel | 151936 | 50 | 0.95 | 32 | 30.7 |
| fused_kernel | 151936 | 100 | 0.9 | 1 | 24.1 |
| fused_kernel | 151936 | 100 | 0.9 | 32 | 34.9 |
| fused_kernel | 151936 | 100 | 0.95 | 1 | 25.3 |
| fused_kernel | 151936 | 100 | 0.95 | 32 | 36.0 |
| graph_compile | 128256 | 20 | 0.9 | 1 | 77.1 |
| graph_compile | 128256 | 20 | 0.9 | 32 | 150.4 |
| graph_compile | 128256 | 20 | 0.95 | 1 | 76.7 |
| graph_compile | 128256 | 20 | 0.95 | 32 | 149.9 |
| graph_compile | 128256 | 50 | 0.9 | 1 | 77.8 |
| graph_compile | 128256 | 50 | 0.9 | 32 | 150.1 |
| graph_compile | 128256 | 50 | 0.95 | 1 | 77.9 |
| graph_compile | 128256 | 50 | 0.95 | 32 | 150.1 |
| graph_compile | 128256 | 100 | 0.9 | 1 | 78.3 |
| graph_compile | 128256 | 100 | 0.9 | 32 | 151.5 |
| graph_compile | 128256 | 100 | 0.95 | 1 | 78.3 |
| graph_compile | 128256 | 100 | 0.95 | 32 | 151.4 |
| graph_compile | 151936 | 20 | 0.9 | 1 | 81.4 |
| graph_compile | 151936 | 20 | 0.9 | 32 | 165.9 |
| graph_compile | 151936 | 20 | 0.95 | 1 | 81.4 |
| graph_compile | 151936 | 20 | 0.95 | 32 | 165.7 |
| graph_compile | 151936 | 50 | 0.9 | 1 | 71.8 |
| graph_compile | 151936 | 50 | 0.9 | 4 | 95.1 |
| graph_compile | 151936 | 50 | 0.9 | 8 | 98.2 |
| graph_compile | 151936 | 50 | 0.9 | 16 | 122.2 |
| graph_compile | 151936 | 50 | 0.9 | 32 | 168.1 |
| graph_compile | 151936 | 50 | 0.95 | 1 | 81.6 |
| graph_compile | 151936 | 50 | 0.95 | 32 | 166.2 |
| graph_compile | 151936 | 100 | 0.9 | 1 | 82.1 |
| graph_compile | 151936 | 100 | 0.9 | 32 | 167.7 |
| graph_compile | 151936 | 100 | 0.95 | 1 | 82.3 |
| graph_compile | 151936 | 100 | 0.95 | 32 | 167.9 |
| graph_eager | 128256 | 20 | 0.9 | 1 | 76.8 |
| graph_eager | 128256 | 20 | 0.9 | 32 | 149.9 |
| graph_eager | 128256 | 20 | 0.95 | 1 | 76.8 |
| graph_eager | 128256 | 20 | 0.95 | 32 | 149.9 |
| graph_eager | 128256 | 50 | 0.9 | 1 | 77.9 |
| graph_eager | 128256 | 50 | 0.9 | 32 | 150.1 |
| graph_eager | 128256 | 50 | 0.95 | 1 | 77.9 |
| graph_eager | 128256 | 50 | 0.95 | 32 | 149.9 |
| graph_eager | 128256 | 100 | 0.9 | 1 | 78.3 |
| graph_eager | 128256 | 100 | 0.9 | 32 | 151.3 |
| graph_eager | 128256 | 100 | 0.95 | 1 | 78.3 |
| graph_eager | 128256 | 100 | 0.95 | 32 | 151.2 |
| graph_eager | 151936 | 20 | 0.9 | 1 | 81.4 |
| graph_eager | 151936 | 20 | 0.9 | 32 | 165.5 |
| graph_eager | 151936 | 20 | 0.95 | 1 | 81.4 |
| graph_eager | 151936 | 20 | 0.95 | 32 | 165.5 |
| graph_eager | 151936 | 50 | 0.9 | 1 | 81.5 |
| graph_eager | 151936 | 50 | 0.9 | 4 | 105.5 |
| graph_eager | 151936 | 50 | 0.9 | 8 | 108.2 |
| graph_eager | 151936 | 50 | 0.9 | 16 | 122.1 |
| graph_eager | 151936 | 50 | 0.9 | 32 | 168.0 |
| graph_eager | 151936 | 50 | 0.95 | 1 | 81.6 |
| graph_eager | 151936 | 50 | 0.95 | 32 | 165.9 |
| graph_eager | 151936 | 100 | 0.9 | 1 | 81.9 |
| graph_eager | 151936 | 100 | 0.9 | 32 | 167.7 |
| graph_eager | 151936 | 100 | 0.95 | 1 | 82.0 |
| graph_eager | 151936 | 100 | 0.95 | 32 | 167.6 |
| hf_eager | 128256 | 20 | 0.9 | 1 | 338.3 |
| hf_eager | 128256 | 20 | 0.9 | 32 | 1833.2 |
| hf_eager | 128256 | 20 | 0.95 | 1 | 338.2 |
| hf_eager | 128256 | 20 | 0.95 | 32 | 1831.7 |
| hf_eager | 128256 | 50 | 0.9 | 1 | 340.1 |
| hf_eager | 128256 | 50 | 0.9 | 32 | 1832.2 |
| hf_eager | 128256 | 50 | 0.95 | 1 | 340.2 |
| hf_eager | 128256 | 50 | 0.95 | 32 | 1830.8 |
| hf_eager | 128256 | 100 | 0.9 | 1 | 340.1 |
| hf_eager | 128256 | 100 | 0.9 | 32 | 1827.0 |
| hf_eager | 128256 | 100 | 0.95 | 1 | 339.9 |
| hf_eager | 128256 | 100 | 0.95 | 32 | 1826.4 |
| hf_eager | 151936 | 20 | 0.9 | 1 | 322.9 |
| hf_eager | 151936 | 20 | 0.9 | 32 | 2193.7 |
| hf_eager | 151936 | 20 | 0.95 | 1 | 323.1 |
| hf_eager | 151936 | 20 | 0.95 | 32 | 2194.5 |
| hf_eager | 151936 | 50 | 0.9 | 1 | 325.1 |
| hf_eager | 151936 | 50 | 0.9 | 4 | 598.5 |
| hf_eager | 151936 | 50 | 0.9 | 8 | 706.7 |
| hf_eager | 151936 | 50 | 0.9 | 16 | 1181.5 |
| hf_eager | 151936 | 50 | 0.9 | 32 | 2200.5 |
| hf_eager | 151936 | 50 | 0.95 | 1 | 324.9 |
| hf_eager | 151936 | 50 | 0.95 | 32 | 2195.5 |
| hf_eager | 151936 | 100 | 0.9 | 1 | 326.6 |
| hf_eager | 151936 | 100 | 0.9 | 32 | 2193.8 |
| hf_eager | 151936 | 100 | 0.95 | 1 | 326.8 |
| hf_eager | 151936 | 100 | 0.95 | 32 | 2193.6 |
| ref_eager_fullsort | 128256 | 20 | 0.9 | 1 | 136.3 |
| ref_eager_fullsort | 128256 | 20 | 0.9 | 32 | 1279.4 |
| ref_eager_fullsort | 128256 | 20 | 0.95 | 1 | 136.2 |
| ref_eager_fullsort | 128256 | 20 | 0.95 | 32 | 1278.8 |
| ref_eager_fullsort | 128256 | 50 | 0.9 | 1 | 135.9 |
| ref_eager_fullsort | 128256 | 50 | 0.9 | 32 | 1278.6 |
| ref_eager_fullsort | 128256 | 50 | 0.95 | 1 | 136.2 |
| ref_eager_fullsort | 128256 | 50 | 0.95 | 32 | 1278.1 |
| ref_eager_fullsort | 128256 | 100 | 0.9 | 1 | 136.0 |
| ref_eager_fullsort | 128256 | 100 | 0.9 | 32 | 1276.1 |
| ref_eager_fullsort | 128256 | 100 | 0.95 | 1 | 136.1 |
| ref_eager_fullsort | 128256 | 100 | 0.95 | 32 | 1275.5 |
| ref_eager_fullsort | 151936 | 20 | 0.9 | 1 | 138.7 |
| ref_eager_fullsort | 151936 | 20 | 0.9 | 32 | 1532.8 |
| ref_eager_fullsort | 151936 | 20 | 0.95 | 1 | 138.3 |
| ref_eager_fullsort | 151936 | 20 | 0.95 | 32 | 1532.6 |
| ref_eager_fullsort | 151936 | 50 | 0.9 | 1 | 138.7 |
| ref_eager_fullsort | 151936 | 50 | 0.9 | 4 | 230.0 |
| ref_eager_fullsort | 151936 | 50 | 0.9 | 8 | 344.3 |
| ref_eager_fullsort | 151936 | 50 | 0.9 | 16 | 702.6 |
| ref_eager_fullsort | 151936 | 50 | 0.9 | 32 | 1536.6 |
| ref_eager_fullsort | 151936 | 50 | 0.95 | 1 | 138.6 |
| ref_eager_fullsort | 151936 | 50 | 0.95 | 32 | 1534.1 |
| ref_eager_fullsort | 151936 | 100 | 0.9 | 1 | 138.8 |
| ref_eager_fullsort | 151936 | 100 | 0.9 | 32 | 1533.2 |
| ref_eager_fullsort | 151936 | 100 | 0.95 | 1 | 138.4 |
| ref_eager_fullsort | 151936 | 100 | 0.95 | 32 | 1533.7 |
| tight_eager | 128256 | 20 | 0.9 | 1 | 132.9 |
| tight_eager | 128256 | 20 | 0.9 | 32 | 202.3 |
| tight_eager | 128256 | 20 | 0.95 | 1 | 133.0 |
| tight_eager | 128256 | 20 | 0.95 | 32 | 202.1 |
| tight_eager | 128256 | 50 | 0.9 | 1 | 136.4 |
| tight_eager | 128256 | 50 | 0.9 | 32 | 201.5 |
| tight_eager | 128256 | 50 | 0.95 | 1 | 136.3 |
| tight_eager | 128256 | 50 | 0.95 | 32 | 201.0 |
| tight_eager | 128256 | 100 | 0.9 | 1 | 136.4 |
| tight_eager | 128256 | 100 | 0.9 | 32 | 201.4 |
| tight_eager | 128256 | 100 | 0.95 | 1 | 136.5 |
| tight_eager | 128256 | 100 | 0.95 | 32 | 201.1 |
| tight_eager | 151936 | 20 | 0.9 | 1 | 134.3 |
| tight_eager | 151936 | 20 | 0.9 | 32 | 216.4 |
| tight_eager | 151936 | 20 | 0.95 | 1 | 133.3 |
| tight_eager | 151936 | 20 | 0.95 | 32 | 217.2 |
| tight_eager | 151936 | 50 | 0.9 | 1 | 136.8 |
| tight_eager | 151936 | 50 | 0.9 | 4 | 163.8 |
| tight_eager | 151936 | 50 | 0.9 | 8 | 167.2 |
| tight_eager | 151936 | 50 | 0.9 | 16 | 183.1 |
| tight_eager | 151936 | 50 | 0.9 | 32 | 219.0 |
| tight_eager | 151936 | 50 | 0.95 | 1 | 136.2 |
| tight_eager | 151936 | 50 | 0.95 | 32 | 217.8 |
| tight_eager | 151936 | 100 | 0.9 | 1 | 136.5 |
| tight_eager | 151936 | 100 | 0.9 | 32 | 216.9 |
| tight_eager | 151936 | 100 | 0.95 | 1 | 136.2 |
| tight_eager | 151936 | 100 | 0.95 | 32 | 216.6 |

### Round-to-round spread (impl order rotated each round)

| impl | rounds | min median µs | max median µs | spread |
|---|---|---|---|---|
| compile | 3 | 158.8 | 159.1 | 0.2% |
| flashinfer | 3 | 91.8 | 91.9 | 0.0% |
| flashinfer_from_probs | 3 | 74.5 | 74.5 | 0.1% |
| fused_kernel | 3 | 22.2 | 22.2 | 0.1% |
| graph_compile | 3 | 103.1 | 103.8 | 0.6% |
| graph_eager | 3 | 112.4 | 112.6 | 0.2% |
| hf_eager | 3 | 703.3 | 706.0 | 0.4% |
| ref_eager_fullsort | 3 | 349.0 | 349.7 | 0.2% |
| tight_eager | 3 | 168.3 | 168.6 | 0.2% |
