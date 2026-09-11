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


@pytest.mark.parametrize("vocab,splits", [(151936, 1), (151936, 8), (20000, 4), (5000, 1)])
def test_ties_past_the_buffer_are_still_resolved_exactly(vocab, splits):
    # every element ties, so one slice holds vocab/splits of them -- far past TIE_CAP, which used
    # to clamp and emit an arbitrary tied id. the index-bucket fallback makes it exact instead.
    x = torch.zeros(2, vocab, device="cuda", dtype=torch.bfloat16)
    assert (vocab + splits - 1) // splits > 2048, "this case must actually overflow the buffer"
    for top_k in (1, 50, 100):
        ids = fused.topk_fused(x, top_k, splits=splits)
        expected = torch.arange(top_k, device="cuda").expand(2, top_k)
        assert torch.equal(ids, expected)


@pytest.mark.parametrize("vocab", [20000, 151936])
def test_dense_tie_rows_match_the_reference_with_one_split(vocab):
    # a narrow logit range makes bf16 collapse many values onto the boundary key, driving the
    # tie set past the buffer on realistic-shaped (not all-equal) input
    torch.manual_seed(0)
    x = (torch.randn(8, vocab, device="cuda") * 0.05).to(torch.bfloat16)
    for top_k in (20, 50, 100):
        assert torch.equal(fused.topk_fused(x, top_k, splits=1), stages(x, top_k, 0.9).topk_ids)


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


def _stage_gap(x, top_k, top_p):
    import fused_sampling

    ref = stages(x, top_k, top_p)
    _, keep, renormed = fused_sampling.stages_fused(x, top_k, top_p, 0, 0, 0)
    ulp = (renormed[ref.keep].view(torch.int32)
           - ref.renormed.float()[ref.keep].view(torch.int32)).abs()
    return keep, ref.keep, renormed, ref.renormed.float(), (ulp.max().item() if ulp.numel() else 0)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("top_p", [0.9, 0.95, 1.0])
@pytest.mark.parametrize("top_k", [1, 20, 50, 100])
def test_gate_a_keep_mask_matches_the_reference_exactly(top_k, top_p, dtype):
    # the kernel cuts on the same expression reference.py does -- (c_i - p_i) < top_p, normalized
    # before the prefix -- so this is exact, not approximate. a/z < p and a < p*z are not.
    torch.manual_seed(0)
    x = (torch.randn(64, 151936, device="cuda") * 4).to(dtype)
    keep, ref_keep, _, _, _ = _stage_gap(x, top_k, top_p)
    assert torch.equal(keep, ref_keep)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("top_k", [20, 50, 100])
def test_gate_a_renormed_matches_the_reference_to_a_few_ulp(top_k, dtype):
    # not bitwise: torch sums the prefix with a scan and the kernel sums it serially. 15 ulp is
    # the worst observed over the regime grid; 64 is headroom, not a licence for a wrong formula.
    torch.manual_seed(0)
    x = (torch.randn(64, 151936, device="cuda") * 4).to(dtype)
    _, ref_keep, renormed, ref_renormed, ulp = _stage_gap(x, top_k, 0.9)
    assert ulp <= 64, f"renormed drifted {ulp} ulp from the reference"
    assert torch.allclose(renormed, ref_renormed, atol=2e-6, rtol=0)
    assert (renormed[~ref_keep] == 0).all(), "dropped candidates must carry zero mass"


def test_gate_a_renormed_sums_to_one_over_the_nucleus():
    torch.manual_seed(0)
    x = (torch.randn(32, 151936, device="cuda") * 4).to(torch.bfloat16)
    for top_p in (0.9, 0.95, 1.0):
        _, _, renormed, _, _ = _stage_gap(x, 50, top_p)
        assert torch.allclose(renormed.sum(-1), torch.ones(32, device="cuda"), atol=1e-5)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("top_p,expected_kept", [(0.25, 2), (0.5, 4), (0.75, 6), (0.875, 7)])
def test_gate_a_cut_lands_exactly_on_the_boundary(top_p, expected_kept, dtype):
    # eight equal logits make every prob exactly 1/8 and every prefix exactly i/8 in fp32, so the
    # exclusive prefix sits *on* top_p. the strict < in reference.py then drops the boundary token.
    # gaussian logits never land here, which is why the general keep test cannot catch a wrong
    # comparison on its own.
    import fused_sampling

    x = torch.full((4, 4000), -30.0, device="cuda", dtype=dtype)
    x[:, :8] = 10.0
    _, keep, _ = fused_sampling.stages_fused(x, 8, top_p, 0, 0, 0)
    assert keep.sum(-1).tolist() == [expected_kept] * 4
    assert torch.equal(keep, stages(x, 8, top_p).keep)


def _capture(x, top_k, top_p):
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fused.sample_fused(x, top_k, top_p)
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g):
        out = fused.sample_fused(x, top_k, top_p)
    return g, out


def test_graph_replay_advances_the_rng_stream():
    # a host-side offset counter is baked into the graph at capture time, so every replay would
    # return the identical token -- silently, and only in the deployment mode that matters
    torch.manual_seed(0)
    x = (torch.randn(8, 4000, device="cuda") * 2).to(torch.bfloat16)
    g, out = _capture(x, 50, 0.95)
    draws = []
    for _ in range(16):
        g.replay()
        torch.cuda.synchronize()
        draws.append(out.clone())
    assert not all(torch.equal(draws[0], d) for d in draws), "graph replay froze the RNG stream"
    assert len({tuple(d.tolist()) for d in draws}) > 8


def test_graph_replay_still_samples_inside_the_nucleus():
    torch.manual_seed(0)
    x = (torch.randn(8, 4000, device="cuda") * 2).to(torch.bfloat16)
    ref = stages(x, 50, 0.95)
    g, out = _capture(x, 50, 0.95)
    for _ in range(16):
        g.replay()
        torch.cuda.synchronize()
        assert (rank_of(ref.topk_ids, out) < ref.cutoff).all()


def test_an_explicit_offset_is_still_reproducible():
    import fused_sampling

    torch.manual_seed(0)
    x = (torch.randn(8, 4000, device="cuda") * 2).to(torch.bfloat16)
    a = fused_sampling.sample_fused(x, 50, 0.9, 1234, 7, 0)
    b = fused_sampling.sample_fused(x, 50, 0.9, 1234, 7, 0)
    c = fused_sampling.sample_fused(x, 50, 0.9, 1234, 8, 0)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
