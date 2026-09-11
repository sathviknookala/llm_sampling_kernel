import csv
import math
from pathlib import Path

import pytest
import torch

from benchmarks.reference import sample_eager

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="kernel is cuda-only")

fused = pytest.importorskip("benchmarks.fused", reason="extension not built")

DTYPES = [torch.float16, torch.bfloat16]


def gen(seed=0):
    return torch.Generator(device="cuda").manual_seed(seed)


def stages(x, top_k, top_p, seed=0):
    return sample_eager(x, top_k, top_p, generator=gen(seed), return_stages=True)


def rank_of(topk_ids, token_ids):
    return (topk_ids == token_ids.unsqueeze(-1)).float().argmax(dim=-1)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("top_k", [1, 20, 50, 100])
@pytest.mark.parametrize("vocab", [2000, 4001, 128256, 151936])
def test_candidate_set_matches_the_reference_exactly(vocab, top_k, dtype):
    # Gate A's selection half. Exact, not distributional: the tie-break is specified, so
    # bf16 rows -- ~93% of which have a k-boundary tie -- must still land on the same ids.
    torch.manual_seed(0)
    x = (torch.randn(16, vocab, device="cuda") * 4).to(dtype)
    assert torch.equal(fused.topk_fused(x, top_k), stages(x, top_k, 0.9).topk_ids)


@pytest.mark.parametrize("dtype", DTYPES)
def test_all_equal_logits_resolve_to_the_lowest_token_ids(dtype):
    # every token ties, so the tie rule alone decides the answer
    x = torch.zeros(4, 4000, device="cuda", dtype=dtype)
    expected = torch.arange(50, device="cuda").expand(4, 50)
    assert torch.equal(fused.topk_fused(x, 50), expected)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("top_p", [0.5, 0.90, 0.95, 1.0])
def test_sampled_token_is_inside_the_reference_nucleus(top_p, dtype):
    torch.manual_seed(0)
    x = (torch.randn(8, 151936, device="cuda") * 4).to(dtype)
    s = stages(x, 50, top_p)
    for off in range(32):
        tok = fused.sample_fused(x, 50, top_p, offset=off)
        hit = s.topk_ids == tok.unsqueeze(-1)
        assert hit.any(dim=-1).all(), "sampled a token outside the top-k candidates"
        assert (rank_of(s.topk_ids, tok) < s.cutoff).all(), "sampled past the top-p cutoff"


def test_top_k_one_is_the_argmax_and_ignores_the_rng():
    torch.manual_seed(0)
    x = (torch.randn(8, 4000, device="cuda") * 4).to(torch.bfloat16)
    expected = x.float().argmax(dim=-1)
    for off in range(8):
        assert torch.equal(fused.sample_fused(x, 1, 0.9, offset=off), expected)


def test_one_hot_logits_always_sample_the_spike():
    x = torch.full((8, 4000), -30.0, device="cuda", dtype=torch.bfloat16)
    x[:, 1234] = 30.0
    for off in range(8):
        assert (fused.sample_fused(x, 50, 0.9, offset=off) == 1234).all()


def test_sampling_frequencies_match_the_renormalized_distribution():
    probs = [0.40, 0.30, 0.15, 0.10, 0.05]
    n = 20000
    row = torch.log(torch.tensor(probs, device="cuda"))
    x = row.expand(n, 5).contiguous().to(torch.bfloat16)
    s = stages(x, 5, 1.0)
    tok = fused.sample_fused(x, 5, 1.0, offset=0)
    freq = torch.bincount(rank_of(s.topk_ids, tok).long(), minlength=5).float() / n
    expected = torch.tensor(probs, device="cuda")
    # 4-sigma binomial band, widest at p=0.4
    assert (freq - expected).abs().max().item() < 4 * math.sqrt(0.4 * 0.6 / n)


def test_equal_logits_sample_close_to_uniform():
    k, n = 8, 4096
    x = torch.zeros(n, k, device="cuda", dtype=torch.bfloat16)
    tok = fused.sample_fused(x, k, 1.0, offset=0)
    counts = torch.bincount(tok.long(), minlength=k).float()
    assert (counts > 0).all()
    sigma = math.sqrt(n * (1 / k) * (1 - 1 / k))
    assert (counts - n / k).abs().max().item() < 4 * sigma


def test_the_top_p_cutoff_matches_the_reference_cutoff():
    # drive the cut from the empirical support: with enough draws every kept rank appears and
    # no dropped one does, which pins the cutoff without exposing it from the kernel
    torch.manual_seed(0)
    x = (torch.randn(4, 4000, device="cuda") * 2).to(torch.bfloat16)
    s = stages(x, 50, 0.90)
    seen = torch.zeros(4, 50, dtype=torch.bool, device="cuda")
    for off in range(4000):
        seen.scatter_(1, rank_of(s.topk_ids, fused.sample_fused(x, 50, 0.90, offset=off)).long().unsqueeze(-1), True)
    assert torch.equal(seen.sum(dim=-1), s.cutoff), f"support {seen.sum(-1).tolist()} vs cutoff {s.cutoff.tolist()}"


def test_same_seed_and_offset_reproduce_and_different_offsets_advance():
    torch.manual_seed(0)
    x = (torch.randn(64, 4000, device="cuda") * 4).to(torch.bfloat16)
    a = fused.sample_fused(x, 50, 0.9, seed=7, offset=3)
    assert torch.equal(a, fused.sample_fused(x, 50, 0.9, seed=7, offset=3))
    assert not torch.equal(a, fused.sample_fused(x, 50, 0.9, seed=7, offset=4))
    assert not torch.equal(a, fused.sample_fused(x, 50, 0.9, seed=8, offset=3))


def test_rejects_shapes_and_dtypes_outside_the_regime():
    x = torch.randn(2, 4000, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="float16 or bfloat16"):
        fused.sample_fused(x.float(), 50, 0.9)
    with pytest.raises(RuntimeError, match=r"top_k must be <= 128"):
        fused.sample_fused(x, 200, 0.9)
    with pytest.raises(RuntimeError, match=r"top_p must be in \(0, 1\]"):
        fused.sample_fused(x, 50, 1.5)
    with pytest.raises(RuntimeError, match="contiguous cuda"):
        fused.sample_fused(x.t().contiguous().t(), 50, 0.9)


def test_tie_buffer_capacity_is_the_documented_spike_limit():
    # ties beyond TIE_CAP per split keep the right *values* but may emit a different tied id.
    # one split over a 152K all-equal row is far past the cap; the retained value is still right.
    x = torch.zeros(1, 151936, device="cuda", dtype=torch.bfloat16)
    ids = fused.topk_fused(x, 50, splits=1)
    assert ids.shape == (1, 50)
    assert ids.unique().numel() == 50
    assert (x[0, ids[0]] == 0).all(), "retained values must still be the top-k values"


def test_committed_floor_artifact_brackets_the_kernel():
    path = Path(__file__).resolve().parents[1] / "results/raw/kernel_floor.csv"
    rows = list(csv.DictReader(path.open()))
    assert rows, "floor artifact is empty"
    noop = [float(r["median_us"]) for r in rows if r["op"] == "noop"]
    scans = [float(r["median_us"]) for r in rows if r["op"].startswith("scan")]
    assert max(noop) < 5.0, "launch floor moved; the spike's headroom claim assumes ~2.5 us"
    assert min(scans) > max(noop), "a full pass cannot be cheaper than an empty launch"


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("batch", [1, 8])
def test_the_last_ablation_phase_is_the_production_path(batch, dtype):
    # results/raw/kernel_phases.csv differences two timings per phase, which only attributes the
    # real kernel if the final phase *is* the real kernel. probe_phase pins seed/offset to 0.
    import fused_sampling

    torch.manual_seed(0)
    x = (torch.randn(batch, 4000, device="cuda") * 4).to(dtype)
    for top_k in (20, 50):
        got = fused_sampling.probe_phase(x, top_k, 0.9, 7, 0)
        want = fused_sampling.sample_fused(x, top_k, 0.9, 0, 0, 0)
        assert torch.equal(got, want)


def test_early_ablation_phases_stop_before_the_answer_exists():
    # a phase that silently ran to completion would make every delta zero and the artifact a lie
    import fused_sampling

    torch.manual_seed(0)
    x = (torch.randn(4, 4000, device="cuda") * 4).to(torch.bfloat16)
    full = fused_sampling.probe_phase(x, 50, 0.9, 7, 0)
    for phase in (1, 2, 3, 4, 5, 6):
        early = fused_sampling.probe_phase(x, 50, 0.9, phase, 0)
        assert early.shape == full.shape
    assert not torch.equal(fused_sampling.probe_phase(x, 50, 0.9, 6, 0), full)


def test_ablation_rejects_phases_outside_the_pipeline():
    import fused_sampling

    x = torch.zeros(2, 4000, device="cuda", dtype=torch.bfloat16)
    for phase in (0, -1, 8):
        with pytest.raises(RuntimeError):
            fused_sampling.probe_phase(x, 50, 0.9, phase, 0)
