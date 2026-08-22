import torch

VOCAB_SIZES = (128256, 151936)
BATCH_SIZES = (1, 2, 4, 8, 16, 32)
TOP_K_VALUES = (1, 20, 50, 100)
TOP_P_VALUES = (0.90, 0.95, 1.0)

LOGIT_DTYPES = (torch.float16, torch.bfloat16)
COMPUTE_DTYPE = torch.float32
INDEX_DTYPE = torch.int64

# k=1 and p=1.0 are semantics edge cases, not performance points; batch 2 is redundant with 1 and 4
BENCH_BATCH_SIZES = (1, 4, 8, 16, 32)
BENCH_TOP_K_VALUES = (20, 50, 100)
BENCH_TOP_P_VALUES = (0.90, 0.95)

# performance anchor: the larger vocabulary, the conservative point for a latency claim
ANCHOR_VOCAB = 151936
ANCHOR_TOP_K = 50
ANCHOR_TOP_P = 0.90

# the vocabulary results/raw/tie_fidelity.csv was measured at
FIDELITY_VOCAB = 128256
