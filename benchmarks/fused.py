"""Python side of the fused SM120 sampling kernel.

The kernel draws from its own RNG stream (SEMANTICS.md frees it from matching
torch.multinomial draw-for-draw), so the caller only advances a counter.
"""

import itertools

import fused_sampling

_counter = itertools.count()
DEFAULT_SEED = 0x5EED


def sample_fused(logits, top_k, top_p, seed=DEFAULT_SEED, offset=None, splits=0):
    off = next(_counter) if offset is None else offset
    return fused_sampling.sample_fused(logits, int(top_k), float(top_p), int(seed), int(off), int(splits))


def topk_fused(logits, top_k, splits=0):
    return fused_sampling.topk_fused(logits, int(top_k), int(splits))
