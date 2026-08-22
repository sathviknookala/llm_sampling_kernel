import torch
from transformers.generation.logits_process import TopKLogitsWarper, TopPLogitsWarper

from .regime import COMPUTE_DTYPE


def hf_scores(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    scores = logits.to(COMPUTE_DTYPE)
    scores = TopKLogitsWarper(top_k=top_k, min_tokens_to_keep=1)(None, scores)
    scores = TopPLogitsWarper(top_p=top_p, min_tokens_to_keep=1)(None, scores)
    return scores


def hf_probs(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    return torch.softmax(hf_scores(logits, top_k, top_p), dim=-1)


def sample_hf(
    logits: torch.Tensor,
    top_k: int,
    top_p: float,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    probs = hf_probs(logits, top_k, top_p)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(1)
