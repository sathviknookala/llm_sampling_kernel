from dataclasses import dataclass

import torch

from .regime import COMPUTE_DTYPE


@dataclass(frozen=True)
class Stages:
    topk_logits: torch.Tensor
    topk_ids: torch.Tensor
    probs: torch.Tensor
    cumsum: torch.Tensor
    keep: torch.Tensor
    cutoff: torch.Tensor
    renormed: torch.Tensor
    token_ids: torch.Tensor


def validate_logits(logits: torch.Tensor) -> None:
    if torch.isnan(logits).any():
        raise ValueError("logits contain NaN")
    if torch.isposinf(logits).any():
        raise ValueError("logits contain +inf")
    if torch.isneginf(logits).all(dim=-1).any():
        raise ValueError("logits contain a row that is entirely -inf")


def sample_eager(
    logits: torch.Tensor,
    top_k: int,
    top_p: float,
    *,
    generator: torch.Generator | None = None,
    check_inputs: bool = True,
    return_stages: bool = False,
):
    if logits.dim() != 2:
        raise ValueError(f"logits must be [B, V], got shape {tuple(logits.shape)}")
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")
    if not 0.0 < top_p <= 1.0:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")
    if check_inputs:
        validate_logits(logits)

    k = min(top_k, logits.shape[-1])
    x = logits.to(COMPUTE_DTYPE)

    topk_logits, topk_ids = torch.topk(x, k, dim=-1, sorted=True)
    probs = torch.softmax(topk_logits, dim=-1)
    cumsum = probs.cumsum(dim=-1)

    if top_p >= 1.0:
        keep = torch.ones_like(probs, dtype=torch.bool)
    else:
        # exclusive prefix: keep i iff c_{i-1} < top_p, so the top-1 candidate always survives
        keep = (cumsum - probs) < top_p

    filtered = torch.where(keep, probs, torch.zeros_like(probs))
    renormed = filtered / filtered.sum(dim=-1, keepdim=True)

    choice = torch.multinomial(renormed, num_samples=1, generator=generator)
    token_ids = topk_ids.gather(-1, choice).squeeze(-1)

    if not return_stages:
        return token_ids
    return Stages(
        topk_logits=topk_logits,
        topk_ids=topk_ids,
        probs=probs,
        cumsum=cumsum,
        keep=keep,
        cutoff=keep.sum(dim=-1),
        renormed=renormed,
        token_ids=token_ids,
    )
