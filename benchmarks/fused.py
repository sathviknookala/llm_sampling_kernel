"""Python side of the fused SM120 sampling kernel.

The kernel draws from its own RNG stream (SEMANTICS.md frees it from matching
torch.multinomial draw-for-draw). By default the stream comes from torch's
default CUDA generator so that cuda-graph replay advances it.
"""

import fused_sampling

DEFAULT_SEED = 0x5EED


def sample_fused(logits, top_k, top_p, seed=DEFAULT_SEED, offset=None, splits=0):
    # offset < 0 routes the draw through torch's default CUDA generator, which is the only
    # form that survives graph capture; pass an explicit offset for a reproducible stream
    off = -1 if offset is None else offset
    return fused_sampling.sample_fused(logits, int(top_k), float(top_p), int(seed), int(off), int(splits))


def topk_fused(logits, top_k, splits=0):
    return fused_sampling.topk_fused(logits, int(top_k), int(splits))
