import pytest
import torch

from benchmarks.hf_baseline import hf_probs, sample_hf
from benchmarks.reference import sample_eager

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.fixture(params=DEVICES)
def device(request):
    return request.param


def gen(device, seed=0):
    return torch.Generator(device=device).manual_seed(seed)


def scatter_to_vocab(stages, vocab, device):
    full = torch.zeros(stages.renormed.shape[0], vocab, device=device)
    return full.scatter_(-1, stages.topk_ids, stages.renormed)


def assert_matches_hf(x, top_k, top_p, device, atol=1e-6):
    s = sample_eager(x, top_k, top_p, generator=gen(device), return_stages=True)
    mine = scatter_to_vocab(s, x.shape[-1], device)
    torch.testing.assert_close(mine, hf_probs(x, top_k, top_p), rtol=0, atol=atol)
    return s


# --- the contract: identical sampling distribution ---------------------------

@pytest.mark.parametrize("top_k", [1, 20, 50, 100])
@pytest.mark.parametrize("top_p", [0.5, 0.90, 0.95, 1.0])
def test_distribution_matches_hf_across_the_regime(device, top_k, top_p):
    torch.manual_seed(0)
    assert_matches_hf(torch.randn(8, 4000, device=device), top_k, top_p, device)


def tie_free_rows(x, top_k):
    # a row is tie-free when no tie can straddle either decision boundary: no duplicate among the
    # k candidates (top-p cutoff) and no token outside them equal to the k-th value (k-boundary)
    xf = x.float()
    vals = xf.topk(min(top_k, x.shape[-1]), dim=-1).values
    distinct = torch.tensor(
        [len(v.unique()) == v.numel() for v in vals], device=x.device
    )
    return distinct & ((xf == vals[:, -1:]).sum(-1) == 1)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_half_precision_matches_hf_exactly_on_tie_free_rows(device, dtype):
    torch.manual_seed(0)
    x = (torch.randn(256, 4000, device=device) * 4).to(dtype)
    free = tie_free_rows(x, 50)
    if not free.any():
        pytest.skip(f"no tie-free rows in 256 {dtype} rows -- ties are the norm here")
    s = sample_eager(x, 50, 0.95, generator=gen(device), return_stages=True)
    mine = scatter_to_vocab(s, x.shape[-1], device)
    torch.testing.assert_close(
        mine[free], hf_probs(x, 50, 0.95)[free], rtol=0, atol=1e-6
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_tie_free_rows_are_rare_in_half_precision(device, dtype):
    # the premise behind the bounded-TV contract: exact HF conformance is not available in the
    # target dtypes, because almost no row is free of ties at a decision boundary
    torch.manual_seed(0)
    x = (torch.randn(256, 4000, device=device) * 4).to(dtype)
    assert tie_free_rows(x, 50).float().mean().item() < 0.5


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_half_precision_divergence_from_hf_stays_bounded(device, dtype):
    # boundary ties are routine in the target regime -- ~95% of bf16 rows at V=128K, k=50 -- so
    # the fixed-K contract cannot match HF row-for-row. What it must do is stay close: the tied
    # tokens are the lowest-probability members of the nucleus. Measured 2026-08-22 at V=128256:
    # bf16 max TV 0.037 / mean 0.006, fp16 max TV 0.013 / mean 0.001.
    torch.manual_seed(0)
    x = (torch.randn(64, 4000, device=device) * 4).to(dtype)
    s = sample_eager(x, 50, 0.95, generator=gen(device), return_stages=True)
    mine = scatter_to_vocab(s, x.shape[-1], device)
    tv = 0.5 * (mine - hf_probs(x, 50, 0.95)).abs().sum(-1)
    assert tv.max().item() < 0.05


@pytest.mark.parametrize("batch", [1, 2, 8, 32])
def test_distribution_matches_hf_across_batch_sizes(device, batch):
    torch.manual_seed(0)
    assert_matches_hf(torch.randn(batch, 2000, device=device), 50, 0.95, device)


def test_nucleus_size_matches_hf(device):
    torch.manual_seed(0)
    x = torch.randn(32, 4000, device=device)
    for top_p in (0.5, 0.7, 0.90, 0.95, 0.99):
        s = sample_eager(x, 100, top_p, generator=gen(device), return_stages=True)
        assert torch.equal(s.cutoff, (hf_probs(x, 100, top_p) > 0).sum(-1))


def test_cutoff_convention_matches_hf_at_exact_equality(device):
    # equal logits give exact 0.25 steps; HF's ascending `cum <= 1 - top_p` removal and this
    # repo's `c_{i-1} < top_p` retention must agree on where the boundary falls
    x = torch.zeros(1, 4, device=device)
    s = sample_eager(x, 4, 0.5, generator=gen(device), return_stages=True)
    assert s.cutoff.item() == (hf_probs(x, 4, 0.5) > 0).sum(-1).item() == 2


def test_top_k_one_matches_hf(device):
    torch.manual_seed(0)
    x = torch.randn(32, 5000, device=device)
    assert torch.equal(
        sample_eager(x, 1, 0.95, generator=gen(device)), sample_hf(x, 1, 0.95, generator=gen(device))
    )


def test_neg_inf_masking_matches_hf(device):
    torch.manual_seed(0)
    x = torch.randn(8, 2000, device=device)
    x[:, 1000:] = float("-inf")
    assert_matches_hf(x, 100, 0.95, device)


def test_dominant_logit_matches_hf(device):
    x = torch.zeros(4, 1000, device=device)
    x[:, 500] = 1e4
    assert_matches_hf(x, 100, 0.95, device)
    assert (sample_hf(x, 100, 0.95, generator=gen(device)) == 500).all()


# --- the one known divergence ------------------------------------------------

def test_hf_keeps_all_boundary_ties_and_this_repo_keeps_exactly_k(device):
    # HF's top-k is value-thresholded (`scores < kth_value`), so it retains every token tied at
    # the boundary -- more than k of them. A fixed-K register-resident kernel cannot, so the
    # contract holds only up to boundary ties. Recorded here so it stays a known divergence.
    x = torch.zeros(1, 20, device=device)
    x[0, :8] = 5.0
    s = sample_eager(x, 5, 1.0, generator=gen(device), return_stages=True)
    assert (hf_probs(x, 5, 1.0) > 0).sum(-1).item() == 8
    assert s.cutoff.item() == 5
    torch.testing.assert_close(s.renormed[0, 0], torch.tensor(0.2, device=device))


def test_no_boundary_tie_means_no_divergence(device):
    # continuous logits: P(tie at the boundary) is negligible, so the contract is exact
    torch.manual_seed(0)
    x = torch.randn(64, 8000, device=device)
    kth = x.topk(50, dim=-1).values[:, -1]
    assert ((x == kth.unsqueeze(-1)).sum(-1) == 1).all()
    assert_matches_hf(x, 50, 0.95, device)
