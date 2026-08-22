import math

import pytest
import torch

from benchmarks.reference import sample_eager, validate_logits

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.fixture(params=DEVICES)
def device(request):
    return request.param


def gen(device, seed=0):
    return torch.Generator(device=device).manual_seed(seed)


def uniform_logits(b, k, device, dtype=torch.float32):
    return torch.zeros(b, k, device=device, dtype=dtype)


def logits_from_probs(probs, device, dtype=torch.float32):
    p = torch.tensor(probs, device=device, dtype=torch.float32)
    return p.log().to(dtype).unsqueeze(0)


# --- top-k membership -------------------------------------------------------

def test_topk_retains_the_k_largest_values(device):
    x = torch.randn(8, 5000, device=device)
    s = sample_eager(x, 50, 0.95, generator=gen(device), return_stages=True)
    expected = x.sort(dim=-1, descending=True).values[:, :50]
    torch.testing.assert_close(s.topk_logits, expected)


def test_topk_ids_address_their_own_logits(device):
    x = torch.randn(8, 5000, device=device)
    s = sample_eager(x, 50, 0.95, generator=gen(device), return_stages=True)
    torch.testing.assert_close(x.gather(-1, s.topk_ids), s.topk_logits)


def test_sampled_token_is_always_in_the_topk_set(device):
    x = torch.randn(16, 2000, device=device)
    g = gen(device)
    for _ in range(50):
        s = sample_eager(x, 20, 0.95, generator=g, return_stages=True)
        assert (s.topk_ids == s.token_ids.unsqueeze(-1)).any(dim=-1).all()


def test_sampled_token_is_inside_the_nucleus(device):
    x = torch.randn(16, 2000, device=device)
    g = gen(device)
    for _ in range(50):
        s = sample_eager(x, 100, 0.90, generator=g, return_stages=True)
        rank = (s.topk_ids == s.token_ids.unsqueeze(-1)).float().argmax(dim=-1)
        assert (rank < s.cutoff).all()


def test_top_k_larger_than_vocab_is_clamped(device):
    x = torch.randn(4, 30, device=device)
    s = sample_eager(x, 500, 0.95, generator=gen(device), return_stages=True)
    assert s.topk_logits.shape == (4, 30)


# --- top-p cutoff index -----------------------------------------------------

def test_cutoff_matches_hand_computed_example(device):
    # 0.40 0.30 0.15 0.10 0.05 -> cumulative 0.40 0.70 0.85 0.95 1.00, cut at 0.90 -> r = 4
    x = logits_from_probs([0.40, 0.30, 0.15, 0.10, 0.05], device)
    s = sample_eager(x, 5, 0.90, generator=gen(device), return_stages=True)
    assert s.cutoff.item() == 4
    assert s.keep.tolist() == [[True, True, True, True, False]]


def test_cutoff_boundary_is_inclusive_at_exact_equality(device):
    # equal logits give exactly-representable 0.25 steps: c = .25 .50 .75 1.0
    # r = min{r : c_r >= 0.5} = 2, so the third candidate is dropped, not kept
    x = uniform_logits(1, 4, device)
    s = sample_eager(x, 4, 0.5, generator=gen(device), return_stages=True)
    assert s.cumsum.tolist() == [[0.25, 0.5, 0.75, 1.0]]
    assert s.cutoff.item() == 2


def test_cutoff_is_monotone_in_top_p(device):
    x = torch.randn(8, 2000, device=device)
    prev = None
    for top_p in (0.5, 0.7, 0.9, 0.95, 0.99):
        s = sample_eager(x, 100, top_p, generator=gen(device), return_stages=True)
        if prev is not None:
            assert (s.cutoff >= prev).all()
        prev = s.cutoff


def test_nucleus_is_a_prefix_and_never_empty(device):
    x = torch.randn(32, 2000, device=device)
    s = sample_eager(x, 100, 0.90, generator=gen(device), return_stages=True)
    assert (s.cutoff >= 1).all()
    assert s.keep[:, 0].all()
    # a prefix mask never goes False -> True along the row
    assert not (~s.keep[:, :-1] & s.keep[:, 1:]).any()


def test_cutoff_mass_brackets_top_p(device):
    x = torch.randn(32, 2000, device=device)
    top_p = 0.90
    s = sample_eager(x, 100, top_p, generator=gen(device), return_stages=True)
    idx = (s.cutoff - 1).unsqueeze(-1)
    at_cutoff = s.cumsum.gather(-1, idx).squeeze(-1)
    before_cutoff = (s.cumsum - s.probs).gather(-1, idx).squeeze(-1)
    assert (at_cutoff >= top_p).all()
    assert (before_cutoff < top_p).all()


# --- renormalized probabilities ---------------------------------------------

@pytest.mark.parametrize("top_p", [0.5, 0.90, 0.95, 1.0])
def test_renormalized_probabilities_sum_to_one(device, top_p):
    x = torch.randn(32, 3000, device=device)
    s = sample_eager(x, 100, top_p, generator=gen(device), return_stages=True)
    torch.testing.assert_close(
        s.renormed.sum(dim=-1), torch.ones(32, device=device), rtol=0, atol=1e-6
    )


def test_renormalization_zeroes_everything_outside_the_nucleus(device):
    x = torch.randn(16, 2000, device=device)
    s = sample_eager(x, 100, 0.90, generator=gen(device), return_stages=True)
    assert (s.renormed[~s.keep] == 0).all()
    assert (s.renormed[s.keep] > 0).all()


def test_renormalization_preserves_relative_odds(device):
    x = torch.randn(8, 2000, device=device)
    s = sample_eager(x, 100, 0.90, generator=gen(device), return_stages=True)
    ratio = s.renormed[:, 0] / s.renormed[:, 1]
    torch.testing.assert_close(ratio, s.probs[:, 0] / s.probs[:, 1], rtol=1e-5, atol=0)


# --- determinism ------------------------------------------------------------

def test_same_seed_gives_identical_tokens(device):
    x = torch.randn(64, 4000, device=device)
    a = sample_eager(x, 50, 0.95, generator=gen(device, 1234))
    b = sample_eager(x, 50, 0.95, generator=gen(device, 1234))
    assert torch.equal(a, b)


def test_different_seeds_give_different_tokens(device):
    x = torch.randn(64, 4000, device=device)
    a = sample_eager(x, 50, 0.95, generator=gen(device, 1))
    b = sample_eager(x, 50, 0.95, generator=gen(device, 2))
    assert not torch.equal(a, b)


def test_stages_are_independent_of_the_seed(device):
    x = torch.randn(16, 2000, device=device)
    a = sample_eager(x, 50, 0.95, generator=gen(device, 1), return_stages=True)
    b = sample_eager(x, 50, 0.95, generator=gen(device, 2), return_stages=True)
    torch.testing.assert_close(a.renormed, b.renormed)
    assert torch.equal(a.cutoff, b.cutoff)


def test_repeated_draws_from_one_generator_advance_the_stream(device):
    x = torch.randn(64, 4000, device=device)
    g = gen(device, 7)
    assert not torch.equal(
        sample_eager(x, 50, 0.95, generator=g), sample_eager(x, 50, 0.95, generator=g)
    )


# --- top_p = 1 --------------------------------------------------------------

def test_top_p_one_keeps_every_candidate(device):
    x = torch.randn(32, 2000, device=device)
    s = sample_eager(x, 100, 1.0, generator=gen(device), return_stages=True)
    assert s.keep.all()
    assert (s.cutoff == 100).all()


def test_top_p_one_keeps_the_tail_even_when_cumsum_saturates(device):
    # one dominant logit: cumsum hits exactly 1.0 at position 0 in fp32
    x = torch.full((1, 100), -50.0, device=device)
    x[0, 0] = 50.0
    s = sample_eager(x, 100, 1.0, generator=gen(device), return_stages=True)
    assert s.cumsum[0, 0].item() == 1.0
    assert s.keep.all()


def test_top_p_one_leaves_probabilities_untouched(device):
    x = torch.randn(16, 2000, device=device)
    s = sample_eager(x, 50, 1.0, generator=gen(device), return_stages=True)
    torch.testing.assert_close(s.renormed, s.probs)


# --- top_k = 1 --------------------------------------------------------------

def test_top_k_one_returns_the_argmax(device):
    x = torch.randn(32, 5000, device=device)
    out = sample_eager(x, 1, 0.95, generator=gen(device))
    assert torch.equal(out, x.argmax(dim=-1))


@pytest.mark.parametrize("top_p", [0.01, 0.5, 1.0])
def test_top_k_one_ignores_seed_and_top_p(device, top_p):
    x = torch.randn(32, 5000, device=device)
    a = sample_eager(x, 1, top_p, generator=gen(device, 11))
    b = sample_eager(x, 1, top_p, generator=gen(device, 99))
    assert torch.equal(a, b)
    assert torch.equal(a, x.argmax(dim=-1))


def test_top_k_one_probability_is_exactly_one(device):
    x = torch.randn(8, 2000, device=device)
    s = sample_eager(x, 1, 0.5, generator=gen(device), return_stages=True)
    assert (s.renormed == 1.0).all()


# --- ties / equal logits ----------------------------------------------------

def test_equal_logits_give_a_uniform_distribution(device):
    x = uniform_logits(4, 64, device)
    s = sample_eager(x, 64, 1.0, generator=gen(device), return_stages=True)
    torch.testing.assert_close(
        s.renormed, torch.full_like(s.renormed, 1 / 64), rtol=0, atol=1e-7
    )


def test_equal_logits_sample_close_to_uniform(device):
    k = 8
    x = uniform_logits(4096, k, device)
    s = sample_eager(x, k, 1.0, generator=gen(device, 3), return_stages=True)
    rank = (s.topk_ids == s.token_ids.unsqueeze(-1)).float().argmax(dim=-1)
    counts = torch.bincount(rank, minlength=k).float()
    assert (counts > 0).all()
    # 4-sigma band on a binomial with n=4096, p=1/8
    n, p = 4096, 1 / k
    sigma = math.sqrt(n * p * (1 - p))
    assert (counts - n * p).abs().max().item() < 4 * sigma


def test_tie_at_the_k_boundary_retains_the_tied_value(device):
    x = torch.zeros(1, 10, device=device)
    x[0, :3] = 5.0
    s = sample_eager(x, 5, 1.0, generator=gen(device), return_stages=True)
    assert s.topk_logits.tolist() == [[5.0, 5.0, 5.0, 0.0, 0.0]]


def test_full_tie_never_samples_outside_the_candidate_set(device):
    x = uniform_logits(256, 16, device)
    s = sample_eager(x, 4, 1.0, generator=gen(device), return_stages=True)
    assert (s.topk_ids == s.token_ids.unsqueeze(-1)).any(dim=-1).all()


# --- very large / very small logits -----------------------------------------

@pytest.mark.parametrize("scale", [1e3, 1e10, 1e30, 3e38])
def test_huge_logits_do_not_overflow(device, scale):
    x = torch.zeros(4, 128, device=device)
    x[:, 0] = scale
    x[:, 1] = scale / 2
    s = sample_eager(x, 64, 0.95, generator=gen(device), return_stages=True)
    assert torch.isfinite(s.renormed).all()
    torch.testing.assert_close(
        s.renormed.sum(-1), torch.ones(4, device=device), rtol=0, atol=1e-6
    )


def test_dominant_logit_is_sampled_deterministically(device):
    x = torch.zeros(8, 1000, device=device)
    x[:, 500] = 1e4
    g = gen(device)
    for _ in range(20):
        assert (sample_eager(x, 100, 0.95, generator=g) == 500).all()


@pytest.mark.parametrize("scale", [-1e3, -1e10, -1e30, -3e38])
def test_very_negative_logits_do_not_underflow_to_an_empty_nucleus(device, scale):
    x = torch.full((4, 128), scale, device=device)
    x[:, 0] = scale / 2
    s = sample_eager(x, 64, 0.95, generator=gen(device), return_stages=True)
    assert torch.isfinite(s.renormed).all()
    assert (s.cutoff >= 1).all()
    torch.testing.assert_close(
        s.renormed.sum(-1), torch.ones(4, device=device), rtol=0, atol=1e-6
    )


def test_wide_dynamic_range_within_one_row(device):
    x = torch.tensor([[1e30, 1.0, 0.0, -1.0, -1e30]], device=device)
    s = sample_eager(x, 5, 0.95, generator=gen(device), return_stages=True)
    assert torch.isfinite(s.renormed).all()
    assert s.cutoff.item() == 1


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_half_precision_inputs_compute_in_fp32(device, dtype):
    x = (torch.randn(16, 2000, device=device) * 10).to(dtype)
    s = sample_eager(x, 100, 0.95, generator=gen(device), return_stages=True)
    assert s.probs.dtype == torch.float32
    assert s.renormed.dtype == torch.float32
    assert s.token_ids.dtype == torch.int64
    torch.testing.assert_close(
        s.renormed.sum(-1), torch.ones(16, device=device), rtol=0, atol=1e-6
    )


# --- NaN / Inf policy -------------------------------------------------------

def test_nan_is_rejected(device):
    x = torch.randn(4, 100, device=device)
    x[2, 7] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        sample_eager(x, 50, 0.95, generator=gen(device))


def test_positive_inf_is_rejected(device):
    x = torch.randn(4, 100, device=device)
    x[1, 3] = float("inf")
    with pytest.raises(ValueError, match=r"\+inf"):
        sample_eager(x, 50, 0.95, generator=gen(device))


def test_all_neg_inf_row_is_rejected(device):
    x = torch.randn(4, 100, device=device)
    x[3, :] = float("-inf")
    with pytest.raises(ValueError, match="entirely -inf"):
        sample_eager(x, 50, 0.95, generator=gen(device))


def test_neg_inf_is_legal_and_gets_zero_probability(device):
    x = torch.randn(8, 200, device=device)
    x[:, 100:] = float("-inf")
    s = sample_eager(x, 150, 1.0, generator=gen(device), return_stages=True)
    banned = s.topk_ids >= 100
    assert (s.renormed[banned] == 0).all()
    assert torch.isfinite(s.renormed).all()


def test_neg_inf_tokens_are_never_sampled(device):
    x = torch.full((64, 500), float("-inf"), device=device)
    x[:, :10] = torch.randn(64, 10, device=device)
    g = gen(device)
    for _ in range(20):
        assert (sample_eager(x, 100, 1.0, generator=g) < 10).all()


def test_check_inputs_toggles_the_validation_call(device, monkeypatch):
    # never feed a real NaN with checks off: multinomial's device-side assert poisons
    # the CUDA context for the whole process, not just the failing call
    called = []
    monkeypatch.setattr(
        "benchmarks.reference.validate_logits", lambda t: called.append(True)
    )
    x = torch.randn(4, 100, device=device)
    sample_eager(x, 50, 0.95, generator=gen(device), check_inputs=False)
    assert not called
    sample_eager(x, 50, 0.95, generator=gen(device), check_inputs=True)
    assert called


def test_validate_logits_accepts_a_partially_masked_row(device):
    x = torch.randn(4, 100, device=device)
    x[:, 50:] = float("-inf")
    validate_logits(x)


# --- argument validation ----------------------------------------------------

@pytest.mark.parametrize("top_k", [0, -1])
def test_bad_top_k_is_rejected(device, top_k):
    with pytest.raises(ValueError, match="top_k"):
        sample_eager(torch.randn(2, 100, device=device), top_k, 0.95)


@pytest.mark.parametrize("top_p", [0.0, -0.1, 1.01])
def test_bad_top_p_is_rejected(device, top_p):
    with pytest.raises(ValueError, match="top_p"):
        sample_eager(torch.randn(2, 100, device=device), 50, top_p)


def test_non_2d_logits_are_rejected(device):
    with pytest.raises(ValueError, match=r"\[B, V\]"):
        sample_eager(torch.randn(100, device=device), 50, 0.95)


def test_logits_are_not_mutated(device):
    x = torch.randn(8, 1000, device=device)
    before = x.clone()
    sample_eager(x, 50, 0.95, generator=gen(device))
    assert torch.equal(x, before)


# --- distributional gate ----------------------------------------------------

def test_sampling_frequencies_match_the_renormalized_distribution(device):
    probs = [0.40, 0.30, 0.15, 0.10, 0.05]
    n = 20000
    x = logits_from_probs(probs, device).expand(n, 5).contiguous()
    s = sample_eager(x, 5, 1.0, generator=gen(device, 5), return_stages=True)
    rank = (s.topk_ids == s.token_ids.unsqueeze(-1)).float().argmax(dim=-1)
    freq = torch.bincount(rank, minlength=5).float() / n
    expected = torch.tensor(probs, device=device)
    # 4-sigma binomial band, widest at p=0.4
    assert (freq - expected).abs().max().item() < 4 * math.sqrt(0.4 * 0.6 / n)


# --- specified tie-break: lowest token id wins ------------------------------

def test_ties_resolve_to_the_lowest_token_id(device):
    x = torch.zeros(1, 64, device=device)
    x[0, ::2] = 1.0
    s = sample_eager(x, 8, 1.0, generator=gen(device), return_stages=True)
    assert s.topk_ids[0].tolist() == [0, 2, 4, 6, 8, 10, 12, 14]


def test_boundary_tie_keeps_the_lowest_ids_of_the_tied_group(device):
    x = torch.zeros(1, 20, device=device)
    x[0, 3:12] = 5.0
    s = sample_eager(x, 5, 1.0, generator=gen(device), return_stages=True)
    assert s.topk_ids[0].tolist() == [3, 4, 5, 6, 7]


def test_tie_break_is_identical_on_cpu_and_cuda():
    if not torch.cuda.is_available():
        pytest.skip("no cuda")
    torch.manual_seed(0)
    x = (torch.randn(64, 4000) * 4).to(torch.bfloat16)
    a = sample_eager(x, 50, 0.95, generator=gen("cpu"), return_stages=True)
    b = sample_eager(x.cuda(), 50, 0.95, generator=gen("cuda"), return_stages=True)
    assert torch.equal(a.topk_ids, b.topk_ids.cpu())
    assert torch.equal(a.cutoff, b.cutoff.cpu())


def test_tie_break_is_stable_across_repeated_calls(device):
    torch.manual_seed(0)
    x = (torch.randn(32, 4000, device=device) * 4).to(torch.bfloat16)
    a = sample_eager(x, 50, 0.95, generator=gen(device), return_stages=True)
    b = sample_eager(x, 50, 0.95, generator=gen(device), return_stages=True)
    assert torch.equal(a.topk_ids, b.topk_ids)


def test_full_tie_selects_the_k_lowest_ids(device):
    x = torch.zeros(4, 500, device=device)
    s = sample_eager(x, 50, 1.0, generator=gen(device), return_stages=True)
    expected = torch.arange(50, device=device).expand(4, 50)
    assert torch.equal(s.topk_ids, expected)
