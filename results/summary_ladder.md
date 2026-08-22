# Sampling Ladder Summary

Source: `results/raw/sampling_ladder.csv` (1008 rows). Median across rounds/reps; 
latency is amortized device time per sampling call, validation disabled in the timed region.


### Latency µs (speedup vs hf_eager) — V=151936, k=50, p=0.9, bfloat16, hot

| impl | B=1 | B=4 | B=8 | B=16 | B=32 |
|---|---|---|---|---|---|
| hf_eager | 324.9 (1.0x) | 597.8 (1.0x) | 706.6 (1.0x) | 1187.8 (1.0x) | 2193.4 (1.0x) |
| ref_eager_fullsort | 138.7 (2.3x) | 229.9 (2.6x) | 346.3 (2.0x) | 707.3 (1.7x) | 1538.0 (1.4x) |
| tight_eager | 136.1 (2.4x) | 164.0 (3.6x) | 167.6 (4.2x) | 182.6 (6.5x) | 219.2 (10.0x) |
| compile | 157.6 (2.1x) | 151.1 (4.0x) | 151.6 (4.7x) | 182.8 (6.5x) | 219.1 (10.0x) |
| graph_eager | 81.7 (4.0x) | 105.6 (5.7x) | 108.3 (6.5x) | 122.2 (9.7x) | 168.1 (13.1x) |
| graph_compile | 72.3 (4.5x) | 95.1 (6.3x) | 102.7 (6.9x) | 122.2 (9.7x) | 168.1 (13.0x) |
| flashinfer | 84.1 (3.9x) | 85.5 (7.0x) | 91.0 (7.8x) | 115.0 (10.3x) | 177.3 (12.4x) |
| flashinfer_from_probs | 72.6 (4.5x) | 73.1 (8.2x) | 73.9 (9.6x) | 73.6 (16.1x) | 101.2 (21.7x) |

### Latency µs (speedup vs hf_eager) — V=151936, k=50, p=0.9, bfloat16, cold

| impl | B=1 | B=4 | B=8 | B=16 | B=32 |
|---|---|---|---|---|---|
| hf_eager | 325.5 (1.0x) | 598.8 (1.0x) | 705.0 (1.0x) | 1154.0 (1.0x) | 2190.8 (1.0x) |
| ref_eager_fullsort | 139.5 (2.3x) | 232.8 (2.6x) | 344.9 (2.0x) | 704.3 (1.6x) | 1537.1 (1.4x) |
| tight_eager | 136.4 (2.4x) | 165.7 (3.6x) | 169.7 (4.2x) | 187.2 (6.2x) | 229.9 (9.5x) |
| compile | 161.3 (2.0x) | 152.5 (3.9x) | 152.6 (4.6x) | 187.1 (6.2x) | 229.7 (9.5x) |
| graph_eager | 83.0 (3.9x) | 111.0 (5.4x) | 115.9 (6.1x) | 132.3 (8.7x) | 192.2 (11.4x) |
| graph_compile | 78.0 (4.2x) | 101.0 (5.9x) | 105.9 (6.7x) | 132.4 (8.7x) | 192.1 (11.4x) |
| flashinfer | 83.8 (3.9x) | 87.3 (6.9x) | 93.0 (7.6x) | 119.1 (9.7x) | 193.7 (11.3x) |
| flashinfer_from_probs | 74.2 (4.4x) | 73.7 (8.1x) | 74.2 (9.5x) | 78.3 (14.7x) | 127.6 (17.2x) |

### Latency µs (speedup vs hf_eager) — V=151936, k=50, p=0.9, float16, hot

| impl | B=1 | B=4 | B=8 | B=16 | B=32 |
|---|---|---|---|---|---|
| hf_eager | 325.2 (1.0x) | 597.4 (1.0x) | 705.6 (1.0x) | 1187.7 (1.0x) | 2194.7 (1.0x) |
| ref_eager_fullsort | 141.9 (2.3x) | 236.8 (2.5x) | 355.3 (2.0x) | 696.8 (1.7x) | 1538.9 (1.4x) |
| tight_eager | 136.8 (2.4x) | 163.6 (3.7x) | 167.3 (4.2x) | 184.0 (6.5x) | 219.0 (10.0x) |
| compile | 157.3 (2.1x) | 150.7 (4.0x) | 151.6 (4.7x) | 158.8 (7.5x) | 191.1 (11.5x) |
| graph_eager | 81.6 (4.0x) | 104.8 (5.7x) | 108.0 (6.5x) | 123.3 (9.6x) | 166.9 (13.2x) |
| graph_compile | 71.8 (4.5x) | 95.1 (6.3x) | 101.0 (7.0x) | 110.6 (10.7x) | 151.1 (14.5x) |
| flashinfer | 84.1 (3.9x) | 85.3 (7.0x) | 90.6 (7.8x) | 115.0 (10.3x) | 176.3 (12.4x) |
| flashinfer_from_probs | 72.7 (4.5x) | 73.1 (8.2x) | 73.9 (9.6x) | 73.8 (16.1x) | 101.2 (21.7x) |

### Latency µs (speedup vs hf_eager) — V=151936, k=50, p=0.9, float16, cold

| impl | B=1 | B=4 | B=8 | B=16 | B=32 |
|---|---|---|---|---|---|
| hf_eager | 325.6 (1.0x) | 599.1 (1.0x) | 704.1 (1.0x) | 1153.7 (1.0x) | 2189.9 (1.0x) |
| ref_eager_fullsort | 141.6 (2.3x) | 238.9 (2.5x) | 354.5 (2.0x) | 696.9 (1.7x) | 1542.0 (1.4x) |
| tight_eager | 137.9 (2.4x) | 165.8 (3.6x) | 169.5 (4.2x) | 186.9 (6.2x) | 229.1 (9.6x) |
| compile | 161.1 (2.0x) | 152.7 (3.9x) | 152.6 (4.6x) | 159.9 (7.2x) | 196.3 (11.2x) |
| graph_eager | 82.9 (3.9x) | 109.5 (5.5x) | 115.2 (6.1x) | 132.2 (8.7x) | 191.6 (11.4x) |
| graph_compile | 73.4 (4.4x) | 100.9 (5.9x) | 105.8 (6.7x) | 119.6 (9.6x) | 174.4 (12.6x) |
| flashinfer | 84.3 (3.9x) | 93.1 (6.4x) | 92.8 (7.6x) | 119.2 (9.7x) | 193.5 (11.3x) |
| flashinfer_from_probs | 74.3 (4.4x) | 74.2 (8.1x) | 74.2 (9.5x) | 78.3 (14.7x) | 127.7 (17.2x) |

### Hot vs cold L2 residency

| impl | batch | dtype | hot µs | cold µs | cold penalty |
|---|---|---|---|---|---|
| compile | 1 | bfloat16 | 157.6 | 161.3 | +2.3% |
| compile | 1 | float16 | 157.3 | 161.1 | +2.4% |
| compile | 4 | bfloat16 | 151.1 | 152.5 | +0.9% |
| compile | 4 | float16 | 150.7 | 152.7 | +1.3% |
| compile | 8 | bfloat16 | 151.6 | 152.6 | +0.7% |
| compile | 8 | float16 | 151.6 | 152.6 | +0.6% |
| compile | 16 | bfloat16 | 182.8 | 187.1 | +2.3% |
| compile | 16 | float16 | 158.8 | 159.9 | +0.7% |
| compile | 32 | bfloat16 | 219.1 | 229.7 | +4.9% |
| compile | 32 | float16 | 191.1 | 196.3 | +2.7% |
| flashinfer | 1 | bfloat16 | 84.1 | 83.8 | -0.3% |
| flashinfer | 1 | float16 | 84.1 | 84.3 | +0.3% |
| flashinfer | 4 | bfloat16 | 85.5 | 87.3 | +2.2% |
| flashinfer | 4 | float16 | 85.3 | 93.1 | +9.0% |
| flashinfer | 8 | bfloat16 | 91.0 | 93.0 | +2.2% |
| flashinfer | 8 | float16 | 90.6 | 92.8 | +2.5% |
| flashinfer | 16 | bfloat16 | 115.0 | 119.1 | +3.6% |
| flashinfer | 16 | float16 | 115.0 | 119.2 | +3.6% |
| flashinfer | 32 | bfloat16 | 177.3 | 193.7 | +9.2% |
| flashinfer | 32 | float16 | 176.3 | 193.5 | +9.7% |
| flashinfer_from_probs | 1 | bfloat16 | 72.6 | 74.2 | +2.1% |
| flashinfer_from_probs | 1 | float16 | 72.7 | 74.3 | +2.2% |
| flashinfer_from_probs | 4 | bfloat16 | 73.1 | 73.7 | +0.8% |
| flashinfer_from_probs | 4 | float16 | 73.1 | 74.2 | +1.5% |
| flashinfer_from_probs | 8 | bfloat16 | 73.9 | 74.2 | +0.5% |
| flashinfer_from_probs | 8 | float16 | 73.9 | 74.2 | +0.4% |
| flashinfer_from_probs | 16 | bfloat16 | 73.6 | 78.3 | +6.5% |
| flashinfer_from_probs | 16 | float16 | 73.8 | 78.3 | +6.2% |
| flashinfer_from_probs | 32 | bfloat16 | 101.2 | 127.6 | +26.1% |
| flashinfer_from_probs | 32 | float16 | 101.2 | 127.7 | +26.1% |
| graph_compile | 1 | bfloat16 | 72.3 | 78.0 | +7.9% |
| graph_compile | 1 | float16 | 71.8 | 73.4 | +2.3% |
| graph_compile | 4 | bfloat16 | 95.1 | 101.0 | +6.2% |
| graph_compile | 4 | float16 | 95.1 | 100.9 | +6.1% |
| graph_compile | 8 | bfloat16 | 102.7 | 105.9 | +3.1% |
| graph_compile | 8 | float16 | 101.0 | 105.8 | +4.7% |
| graph_compile | 16 | bfloat16 | 122.2 | 132.4 | +8.3% |
| graph_compile | 16 | float16 | 110.6 | 119.6 | +8.1% |
| graph_compile | 32 | bfloat16 | 168.1 | 192.1 | +14.3% |
| graph_compile | 32 | float16 | 151.1 | 174.4 | +15.5% |
| graph_eager | 1 | bfloat16 | 81.7 | 83.0 | +1.6% |
| graph_eager | 1 | float16 | 81.6 | 82.9 | +1.6% |
| graph_eager | 4 | bfloat16 | 105.6 | 111.0 | +5.1% |
| graph_eager | 4 | float16 | 104.8 | 109.5 | +4.5% |
| graph_eager | 8 | bfloat16 | 108.3 | 115.9 | +7.0% |
| graph_eager | 8 | float16 | 108.0 | 115.2 | +6.7% |
| graph_eager | 16 | bfloat16 | 122.2 | 132.3 | +8.3% |
| graph_eager | 16 | float16 | 123.3 | 132.2 | +7.2% |
| graph_eager | 32 | bfloat16 | 168.1 | 192.2 | +14.4% |
| graph_eager | 32 | float16 | 166.9 | 191.6 | +14.8% |
| hf_eager | 1 | bfloat16 | 324.9 | 325.5 | +0.2% |
| hf_eager | 1 | float16 | 325.2 | 325.6 | +0.1% |
| hf_eager | 4 | bfloat16 | 597.8 | 598.8 | +0.2% |
| hf_eager | 4 | float16 | 597.4 | 599.1 | +0.3% |
| hf_eager | 8 | bfloat16 | 706.6 | 705.0 | -0.2% |
| hf_eager | 8 | float16 | 705.6 | 704.1 | -0.2% |
| hf_eager | 16 | bfloat16 | 1187.8 | 1154.0 | -2.8% |
| hf_eager | 16 | float16 | 1187.7 | 1153.7 | -2.9% |
| hf_eager | 32 | bfloat16 | 2193.4 | 2190.8 | -0.1% |
| hf_eager | 32 | float16 | 2194.7 | 2189.9 | -0.2% |
| ref_eager_fullsort | 1 | bfloat16 | 138.7 | 139.5 | +0.5% |
| ref_eager_fullsort | 1 | float16 | 141.9 | 141.6 | -0.2% |
| ref_eager_fullsort | 4 | bfloat16 | 229.9 | 232.8 | +1.2% |
| ref_eager_fullsort | 4 | float16 | 236.8 | 238.9 | +0.9% |
| ref_eager_fullsort | 8 | bfloat16 | 346.3 | 344.9 | -0.4% |
| ref_eager_fullsort | 8 | float16 | 355.3 | 354.5 | -0.2% |
| ref_eager_fullsort | 16 | bfloat16 | 707.3 | 704.3 | -0.4% |
| ref_eager_fullsort | 16 | float16 | 696.8 | 696.9 | +0.0% |
| ref_eager_fullsort | 32 | bfloat16 | 1538.0 | 1537.1 | -0.1% |
| ref_eager_fullsort | 32 | float16 | 1538.9 | 1542.0 | +0.2% |
| tight_eager | 1 | bfloat16 | 136.1 | 136.4 | +0.2% |
| tight_eager | 1 | float16 | 136.8 | 137.9 | +0.8% |
| tight_eager | 4 | bfloat16 | 164.0 | 165.7 | +1.0% |
| tight_eager | 4 | float16 | 163.6 | 165.8 | +1.3% |
| tight_eager | 8 | bfloat16 | 167.6 | 169.7 | +1.3% |
| tight_eager | 8 | float16 | 167.3 | 169.5 | +1.3% |
| tight_eager | 16 | bfloat16 | 182.6 | 187.2 | +2.5% |
| tight_eager | 16 | float16 | 184.0 | 186.9 | +1.6% |
| tight_eager | 32 | bfloat16 | 219.2 | 229.9 | +4.9% |
| tight_eager | 32 | float16 | 219.0 | 229.1 | +4.6% |

### Memory-read floor vs measured latency (bfloat16, hot)

| impl | batch | logits MB | DRAM floor µs | latency µs | floor % of latency |
|---|---|---|---|---|---|
| compile | 1 | 0.30 | 0.55 | 157.6 | 0.35% |
| compile | 4 | 1.22 | 2.20 | 151.1 | 1.46% |
| compile | 8 | 2.43 | 4.40 | 151.6 | 2.90% |
| compile | 16 | 4.86 | 8.80 | 182.8 | 4.82% |
| compile | 32 | 9.72 | 17.61 | 219.1 | 8.04% |
| flashinfer | 1 | 0.30 | 0.55 | 84.1 | 0.65% |
| flashinfer | 4 | 1.22 | 2.20 | 85.5 | 2.58% |
| flashinfer | 8 | 2.43 | 4.40 | 91.0 | 4.84% |
| flashinfer | 16 | 4.86 | 8.80 | 115.0 | 7.66% |
| flashinfer | 32 | 9.72 | 17.61 | 177.3 | 9.93% |
| flashinfer_from_probs | 1 | 0.61 | 1.10 | 72.6 | 1.52% |
| flashinfer_from_probs | 4 | 2.43 | 4.40 | 73.1 | 6.02% |
| flashinfer_from_probs | 8 | 4.86 | 8.80 | 73.9 | 11.92% |
| flashinfer_from_probs | 16 | 9.72 | 17.61 | 73.6 | 23.93% |
| flashinfer_from_probs | 32 | 19.45 | 35.22 | 101.2 | 34.80% |
| graph_compile | 1 | 0.30 | 0.55 | 72.3 | 0.76% |
| graph_compile | 4 | 1.22 | 2.20 | 95.1 | 2.32% |
| graph_compile | 8 | 2.43 | 4.40 | 102.7 | 4.28% |
| graph_compile | 16 | 4.86 | 8.80 | 122.2 | 7.20% |
| graph_compile | 32 | 9.72 | 17.61 | 168.1 | 10.48% |
| graph_eager | 1 | 0.30 | 0.55 | 81.7 | 0.67% |
| graph_eager | 4 | 1.22 | 2.20 | 105.6 | 2.08% |
| graph_eager | 8 | 2.43 | 4.40 | 108.3 | 4.07% |
| graph_eager | 16 | 4.86 | 8.80 | 122.2 | 7.21% |
| graph_eager | 32 | 9.72 | 17.61 | 168.1 | 10.48% |
| hf_eager | 1 | 0.30 | 0.55 | 324.9 | 0.17% |
| hf_eager | 4 | 1.22 | 2.20 | 597.8 | 0.37% |
| hf_eager | 8 | 2.43 | 4.40 | 706.6 | 0.62% |
| hf_eager | 16 | 4.86 | 8.80 | 1187.8 | 0.74% |
| hf_eager | 32 | 9.72 | 17.61 | 2193.4 | 0.80% |
| ref_eager_fullsort | 1 | 0.30 | 0.55 | 138.7 | 0.40% |
| ref_eager_fullsort | 4 | 1.22 | 2.20 | 229.9 | 0.96% |
| ref_eager_fullsort | 8 | 2.43 | 4.40 | 346.3 | 1.27% |
| ref_eager_fullsort | 16 | 4.86 | 8.80 | 707.3 | 1.24% |
| ref_eager_fullsort | 32 | 9.72 | 17.61 | 1538.0 | 1.14% |
| tight_eager | 1 | 0.30 | 0.55 | 136.1 | 0.40% |
| tight_eager | 4 | 1.22 | 2.20 | 164.0 | 1.34% |
| tight_eager | 8 | 2.43 | 4.40 | 167.6 | 2.63% |
| tight_eager | 16 | 4.86 | 8.80 | 182.6 | 4.82% |
| tight_eager | 32 | 9.72 | 17.61 | 219.2 | 8.03% |

### Parameter sensitivity (bfloat16, hot)

| impl | vocab | top_k | top_p | batch | median µs |
|---|---|---|---|---|---|
| compile | 128256 | 20 | 0.9 | 1 | 161.3 |
| compile | 128256 | 20 | 0.9 | 32 | 203.0 |
| compile | 128256 | 20 | 0.95 | 1 | 161.1 |
| compile | 128256 | 20 | 0.95 | 32 | 202.8 |
| compile | 128256 | 50 | 0.9 | 1 | 158.8 |
| compile | 128256 | 50 | 0.9 | 32 | 202.3 |
| compile | 128256 | 50 | 0.95 | 1 | 159.0 |
| compile | 128256 | 50 | 0.95 | 32 | 200.8 |
| compile | 128256 | 100 | 0.9 | 1 | 159.8 |
| compile | 128256 | 100 | 0.9 | 32 | 201.5 |
| compile | 128256 | 100 | 0.95 | 1 | 158.4 |
| compile | 128256 | 100 | 0.95 | 32 | 201.5 |
| compile | 151936 | 20 | 0.9 | 1 | 163.0 |
| compile | 151936 | 20 | 0.9 | 32 | 217.0 |
| compile | 151936 | 20 | 0.95 | 1 | 163.0 |
| compile | 151936 | 20 | 0.95 | 32 | 217.1 |
| compile | 151936 | 50 | 0.9 | 1 | 157.6 |
| compile | 151936 | 50 | 0.9 | 4 | 151.1 |
| compile | 151936 | 50 | 0.9 | 8 | 151.6 |
| compile | 151936 | 50 | 0.9 | 16 | 182.8 |
| compile | 151936 | 50 | 0.9 | 32 | 219.1 |
| compile | 151936 | 50 | 0.95 | 1 | 161.9 |
| compile | 151936 | 50 | 0.95 | 32 | 218.1 |
| compile | 151936 | 100 | 0.9 | 1 | 161.0 |
| compile | 151936 | 100 | 0.9 | 32 | 218.0 |
| compile | 151936 | 100 | 0.95 | 1 | 161.4 |
| compile | 151936 | 100 | 0.95 | 32 | 218.6 |
| flashinfer | 128256 | 20 | 0.9 | 1 | 81.9 |
| flashinfer | 128256 | 20 | 0.9 | 32 | 141.7 |
| flashinfer | 128256 | 20 | 0.95 | 1 | 82.0 |
| flashinfer | 128256 | 20 | 0.95 | 32 | 141.1 |
| flashinfer | 128256 | 50 | 0.9 | 1 | 82.1 |
| flashinfer | 128256 | 50 | 0.9 | 32 | 141.8 |
| flashinfer | 128256 | 50 | 0.95 | 1 | 81.7 |
| flashinfer | 128256 | 50 | 0.95 | 32 | 141.2 |
| flashinfer | 128256 | 100 | 0.9 | 1 | 82.2 |
| flashinfer | 128256 | 100 | 0.9 | 32 | 141.8 |
| flashinfer | 128256 | 100 | 0.95 | 1 | 82.2 |
| flashinfer | 128256 | 100 | 0.95 | 32 | 141.2 |
| flashinfer | 151936 | 20 | 0.9 | 1 | 84.0 |
| flashinfer | 151936 | 20 | 0.9 | 32 | 175.8 |
| flashinfer | 151936 | 20 | 0.95 | 1 | 84.0 |
| flashinfer | 151936 | 20 | 0.95 | 32 | 175.5 |
| flashinfer | 151936 | 50 | 0.9 | 1 | 84.1 |
| flashinfer | 151936 | 50 | 0.9 | 4 | 85.5 |
| flashinfer | 151936 | 50 | 0.9 | 8 | 91.0 |
| flashinfer | 151936 | 50 | 0.9 | 16 | 115.0 |
| flashinfer | 151936 | 50 | 0.9 | 32 | 177.3 |
| flashinfer | 151936 | 50 | 0.95 | 1 | 83.9 |
| flashinfer | 151936 | 50 | 0.95 | 32 | 176.1 |
| flashinfer | 151936 | 100 | 0.9 | 1 | 84.4 |
| flashinfer | 151936 | 100 | 0.9 | 32 | 177.7 |
| flashinfer | 151936 | 100 | 0.95 | 1 | 84.3 |
| flashinfer | 151936 | 100 | 0.95 | 32 | 177.2 |
| flashinfer_from_probs | 128256 | 20 | 0.9 | 1 | 73.1 |
| flashinfer_from_probs | 128256 | 20 | 0.9 | 32 | 84.3 |
| flashinfer_from_probs | 128256 | 20 | 0.95 | 1 | 72.9 |
| flashinfer_from_probs | 128256 | 20 | 0.95 | 32 | 83.5 |
| flashinfer_from_probs | 128256 | 50 | 0.9 | 1 | 73.2 |
| flashinfer_from_probs | 128256 | 50 | 0.9 | 32 | 82.7 |
| flashinfer_from_probs | 128256 | 50 | 0.95 | 1 | 72.9 |
| flashinfer_from_probs | 128256 | 50 | 0.95 | 32 | 82.3 |
| flashinfer_from_probs | 128256 | 100 | 0.9 | 1 | 73.0 |
| flashinfer_from_probs | 128256 | 100 | 0.9 | 32 | 84.5 |
| flashinfer_from_probs | 128256 | 100 | 0.95 | 1 | 73.1 |
| flashinfer_from_probs | 128256 | 100 | 0.95 | 32 | 83.8 |
| flashinfer_from_probs | 151936 | 20 | 0.9 | 1 | 72.9 |
| flashinfer_from_probs | 151936 | 20 | 0.9 | 32 | 102.8 |
| flashinfer_from_probs | 151936 | 20 | 0.95 | 1 | 72.7 |
| flashinfer_from_probs | 151936 | 20 | 0.95 | 32 | 102.0 |
| flashinfer_from_probs | 151936 | 50 | 0.9 | 1 | 72.6 |
| flashinfer_from_probs | 151936 | 50 | 0.9 | 4 | 73.1 |
| flashinfer_from_probs | 151936 | 50 | 0.9 | 8 | 73.9 |
| flashinfer_from_probs | 151936 | 50 | 0.9 | 16 | 73.6 |
| flashinfer_from_probs | 151936 | 50 | 0.9 | 32 | 101.2 |
| flashinfer_from_probs | 151936 | 50 | 0.95 | 1 | 73.1 |
| flashinfer_from_probs | 151936 | 50 | 0.95 | 32 | 100.7 |
| flashinfer_from_probs | 151936 | 100 | 0.9 | 1 | 72.8 |
| flashinfer_from_probs | 151936 | 100 | 0.9 | 32 | 103.0 |
| flashinfer_from_probs | 151936 | 100 | 0.95 | 1 | 72.9 |
| flashinfer_from_probs | 151936 | 100 | 0.95 | 32 | 102.3 |
| graph_compile | 128256 | 20 | 0.9 | 1 | 76.7 |
| graph_compile | 128256 | 20 | 0.9 | 32 | 150.1 |
| graph_compile | 128256 | 20 | 0.95 | 1 | 83.7 |
| graph_compile | 128256 | 20 | 0.95 | 32 | 150.0 |
| graph_compile | 128256 | 50 | 0.9 | 1 | 77.7 |
| graph_compile | 128256 | 50 | 0.9 | 32 | 154.4 |
| graph_compile | 128256 | 50 | 0.95 | 1 | 77.8 |
| graph_compile | 128256 | 50 | 0.95 | 32 | 150.2 |
| graph_compile | 128256 | 100 | 0.9 | 1 | 82.6 |
| graph_compile | 128256 | 100 | 0.9 | 32 | 151.8 |
| graph_compile | 128256 | 100 | 0.95 | 1 | 78.2 |
| graph_compile | 128256 | 100 | 0.95 | 32 | 151.8 |
| graph_compile | 151936 | 20 | 0.9 | 1 | 81.5 |
| graph_compile | 151936 | 20 | 0.9 | 32 | 166.1 |
| graph_compile | 151936 | 20 | 0.95 | 1 | 81.5 |
| graph_compile | 151936 | 20 | 0.95 | 32 | 171.0 |
| graph_compile | 151936 | 50 | 0.9 | 1 | 72.3 |
| graph_compile | 151936 | 50 | 0.9 | 4 | 95.1 |
| graph_compile | 151936 | 50 | 0.9 | 8 | 102.7 |
| graph_compile | 151936 | 50 | 0.9 | 16 | 122.2 |
| graph_compile | 151936 | 50 | 0.9 | 32 | 168.1 |
| graph_compile | 151936 | 50 | 0.95 | 1 | 81.6 |
| graph_compile | 151936 | 50 | 0.95 | 32 | 167.3 |
| graph_compile | 151936 | 100 | 0.9 | 1 | 88.1 |
| graph_compile | 151936 | 100 | 0.9 | 32 | 168.3 |
| graph_compile | 151936 | 100 | 0.95 | 1 | 82.2 |
| graph_compile | 151936 | 100 | 0.95 | 32 | 168.5 |
| graph_eager | 128256 | 20 | 0.9 | 1 | 76.7 |
| graph_eager | 128256 | 20 | 0.9 | 32 | 150.2 |
| graph_eager | 128256 | 20 | 0.95 | 1 | 76.8 |
| graph_eager | 128256 | 20 | 0.95 | 32 | 150.0 |
| graph_eager | 128256 | 50 | 0.9 | 1 | 77.8 |
| graph_eager | 128256 | 50 | 0.9 | 32 | 150.1 |
| graph_eager | 128256 | 50 | 0.95 | 1 | 77.8 |
| graph_eager | 128256 | 50 | 0.95 | 32 | 150.1 |
| graph_eager | 128256 | 100 | 0.9 | 1 | 78.2 |
| graph_eager | 128256 | 100 | 0.9 | 32 | 151.6 |
| graph_eager | 128256 | 100 | 0.95 | 1 | 78.2 |
| graph_eager | 128256 | 100 | 0.95 | 32 | 151.7 |
| graph_eager | 151936 | 20 | 0.9 | 1 | 81.5 |
| graph_eager | 151936 | 20 | 0.9 | 32 | 165.9 |
| graph_eager | 151936 | 20 | 0.95 | 1 | 81.5 |
| graph_eager | 151936 | 20 | 0.95 | 32 | 165.9 |
| graph_eager | 151936 | 50 | 0.9 | 1 | 81.7 |
| graph_eager | 151936 | 50 | 0.9 | 4 | 105.6 |
| graph_eager | 151936 | 50 | 0.9 | 8 | 108.3 |
| graph_eager | 151936 | 50 | 0.9 | 16 | 122.2 |
| graph_eager | 151936 | 50 | 0.9 | 32 | 168.1 |
| graph_eager | 151936 | 50 | 0.95 | 1 | 81.6 |
| graph_eager | 151936 | 50 | 0.95 | 32 | 167.1 |
| graph_eager | 151936 | 100 | 0.9 | 1 | 82.1 |
| graph_eager | 151936 | 100 | 0.9 | 32 | 168.1 |
| graph_eager | 151936 | 100 | 0.95 | 1 | 82.1 |
| graph_eager | 151936 | 100 | 0.95 | 32 | 168.2 |
| hf_eager | 128256 | 20 | 0.9 | 1 | 338.2 |
| hf_eager | 128256 | 20 | 0.9 | 32 | 1836.9 |
| hf_eager | 128256 | 20 | 0.95 | 1 | 338.4 |
| hf_eager | 128256 | 20 | 0.95 | 32 | 1835.6 |
| hf_eager | 128256 | 50 | 0.9 | 1 | 339.9 |
| hf_eager | 128256 | 50 | 0.9 | 32 | 1838.2 |
| hf_eager | 128256 | 50 | 0.95 | 1 | 340.3 |
| hf_eager | 128256 | 50 | 0.95 | 32 | 1837.3 |
| hf_eager | 128256 | 100 | 0.9 | 1 | 341.2 |
| hf_eager | 128256 | 100 | 0.9 | 32 | 1833.6 |
| hf_eager | 128256 | 100 | 0.95 | 1 | 341.4 |
| hf_eager | 128256 | 100 | 0.95 | 32 | 1833.7 |
| hf_eager | 151936 | 20 | 0.9 | 1 | 323.2 |
| hf_eager | 151936 | 20 | 0.9 | 32 | 2186.2 |
| hf_eager | 151936 | 20 | 0.95 | 1 | 322.9 |
| hf_eager | 151936 | 20 | 0.95 | 32 | 2186.7 |
| hf_eager | 151936 | 50 | 0.9 | 1 | 324.9 |
| hf_eager | 151936 | 50 | 0.9 | 4 | 597.8 |
| hf_eager | 151936 | 50 | 0.9 | 8 | 706.6 |
| hf_eager | 151936 | 50 | 0.9 | 16 | 1187.8 |
| hf_eager | 151936 | 50 | 0.9 | 32 | 2193.4 |
| hf_eager | 151936 | 50 | 0.95 | 1 | 325.1 |
| hf_eager | 151936 | 50 | 0.95 | 32 | 2189.7 |
| hf_eager | 151936 | 100 | 0.9 | 1 | 326.9 |
| hf_eager | 151936 | 100 | 0.9 | 32 | 2189.1 |
| hf_eager | 151936 | 100 | 0.95 | 1 | 326.5 |
| hf_eager | 151936 | 100 | 0.95 | 32 | 2189.1 |
| ref_eager_fullsort | 128256 | 20 | 0.9 | 1 | 136.0 |
| ref_eager_fullsort | 128256 | 20 | 0.9 | 32 | 1280.2 |
| ref_eager_fullsort | 128256 | 20 | 0.95 | 1 | 136.6 |
| ref_eager_fullsort | 128256 | 20 | 0.95 | 32 | 1280.2 |
| ref_eager_fullsort | 128256 | 50 | 0.9 | 1 | 136.9 |
| ref_eager_fullsort | 128256 | 50 | 0.9 | 32 | 1278.7 |
| ref_eager_fullsort | 128256 | 50 | 0.95 | 1 | 136.7 |
| ref_eager_fullsort | 128256 | 50 | 0.95 | 32 | 1278.0 |
| ref_eager_fullsort | 128256 | 100 | 0.9 | 1 | 137.1 |
| ref_eager_fullsort | 128256 | 100 | 0.9 | 32 | 1276.0 |
| ref_eager_fullsort | 128256 | 100 | 0.95 | 1 | 136.9 |
| ref_eager_fullsort | 128256 | 100 | 0.95 | 32 | 1276.3 |
| ref_eager_fullsort | 151936 | 20 | 0.9 | 1 | 138.8 |
| ref_eager_fullsort | 151936 | 20 | 0.9 | 32 | 1533.3 |
| ref_eager_fullsort | 151936 | 20 | 0.95 | 1 | 138.8 |
| ref_eager_fullsort | 151936 | 20 | 0.95 | 32 | 1534.0 |
| ref_eager_fullsort | 151936 | 50 | 0.9 | 1 | 138.7 |
| ref_eager_fullsort | 151936 | 50 | 0.9 | 4 | 229.9 |
| ref_eager_fullsort | 151936 | 50 | 0.9 | 8 | 346.3 |
| ref_eager_fullsort | 151936 | 50 | 0.9 | 16 | 707.3 |
| ref_eager_fullsort | 151936 | 50 | 0.9 | 32 | 1538.0 |
| ref_eager_fullsort | 151936 | 50 | 0.95 | 1 | 138.7 |
| ref_eager_fullsort | 151936 | 50 | 0.95 | 32 | 1534.6 |
| ref_eager_fullsort | 151936 | 100 | 0.9 | 1 | 138.8 |
| ref_eager_fullsort | 151936 | 100 | 0.9 | 32 | 1535.5 |
| ref_eager_fullsort | 151936 | 100 | 0.95 | 1 | 138.6 |
| ref_eager_fullsort | 151936 | 100 | 0.95 | 32 | 1535.5 |
| tight_eager | 128256 | 20 | 0.9 | 1 | 134.1 |
| tight_eager | 128256 | 20 | 0.9 | 32 | 202.2 |
| tight_eager | 128256 | 20 | 0.95 | 1 | 133.6 |
| tight_eager | 128256 | 20 | 0.95 | 32 | 202.0 |
| tight_eager | 128256 | 50 | 0.9 | 1 | 134.6 |
| tight_eager | 128256 | 50 | 0.9 | 32 | 201.5 |
| tight_eager | 128256 | 50 | 0.95 | 1 | 134.6 |
| tight_eager | 128256 | 50 | 0.95 | 32 | 201.0 |
| tight_eager | 128256 | 100 | 0.9 | 1 | 134.1 |
| tight_eager | 128256 | 100 | 0.9 | 32 | 201.2 |
| tight_eager | 128256 | 100 | 0.95 | 1 | 133.9 |
| tight_eager | 128256 | 100 | 0.95 | 32 | 201.5 |
| tight_eager | 151936 | 20 | 0.9 | 1 | 135.5 |
| tight_eager | 151936 | 20 | 0.9 | 32 | 217.1 |
| tight_eager | 151936 | 20 | 0.95 | 1 | 135.5 |
| tight_eager | 151936 | 20 | 0.95 | 32 | 217.0 |
| tight_eager | 151936 | 50 | 0.9 | 1 | 136.1 |
| tight_eager | 151936 | 50 | 0.9 | 4 | 164.0 |
| tight_eager | 151936 | 50 | 0.9 | 8 | 167.6 |
| tight_eager | 151936 | 50 | 0.9 | 16 | 182.6 |
| tight_eager | 151936 | 50 | 0.9 | 32 | 219.2 |
| tight_eager | 151936 | 50 | 0.95 | 1 | 134.6 |
| tight_eager | 151936 | 50 | 0.95 | 32 | 218.4 |
| tight_eager | 151936 | 100 | 0.9 | 1 | 135.1 |
| tight_eager | 151936 | 100 | 0.9 | 32 | 218.3 |
| tight_eager | 151936 | 100 | 0.95 | 1 | 136.6 |
| tight_eager | 151936 | 100 | 0.95 | 32 | 217.9 |

### Round-to-round spread (impl order rotated each round)

| impl | rounds | min median µs | max median µs | spread |
|---|---|---|---|---|
| compile | 3 | 157.6 | 158.7 | 0.7% |
| flashinfer | 3 | 91.9 | 94.0 | 2.3% |
| flashinfer_from_probs | 3 | 73.7 | 74.3 | 0.7% |
| graph_compile | 3 | 103.8 | 104.3 | 0.5% |
| graph_eager | 3 | 112.7 | 114.2 | 1.3% |
| hf_eager | 3 | 698.9 | 705.5 | 0.9% |
| ref_eager_fullsort | 3 | 344.6 | 350.7 | 1.8% |
| tight_eager | 3 | 168.9 | 170.4 | 0.9% |
