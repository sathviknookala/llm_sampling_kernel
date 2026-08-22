import torch

from .reference import sample_eager

try:
    from .hf_baseline import sample_hf

    HAVE_HF = True
except Exception:
    HAVE_HF = False

try:
    import flashinfer

    HAVE_FLASHINFER = True
except Exception:
    HAVE_FLASHINFER = False


def sample_tight(logits, top_k, top_p):
    x = logits.float()
    vals, ids = torch.topk(x, top_k, dim=-1)
    probs = torch.softmax(vals, dim=-1)
    cumsum = probs.cumsum(dim=-1)
    keep = (cumsum - probs) < top_p
    filtered = torch.where(keep, probs, torch.zeros_like(probs))
    renormed = filtered / filtered.sum(dim=-1, keepdim=True)
    choice = torch.multinomial(renormed, num_samples=1)
    return ids.gather(-1, choice).squeeze(-1)


def _hf(logits, top_k, top_p):
    return sample_hf(logits, top_k, top_p)


def _ref(logits, top_k, top_p):
    return sample_eager(logits, top_k, top_p, check_inputs=False)


def _flashinfer(logits, top_k, top_p):
    # logits -> token id, matching every other rung's contract. The full-vocab softmax is
    # inside the timed region because the fused kernel would also consume raw logits.
    probs = torch.softmax(logits.float(), dim=-1)
    return flashinfer.sampling.top_k_top_p_sampling_from_probs(probs, top_k, top_p)


def _flashinfer_from_probs(probs, top_k, top_p):
    # probs -> token id: FlashInfer's kernel cost alone, the way vLLM calls it
    return flashinfer.sampling.top_k_top_p_sampling_from_probs(probs, top_k, top_p)


EAGER_FNS = {
    "hf_eager": _hf,
    "ref_eager_fullsort": _ref,
    "tight_eager": sample_tight,
}
if HAVE_FLASHINFER:
    EAGER_FNS["flashinfer"] = _flashinfer
    EAGER_FNS["flashinfer_from_probs"] = _flashinfer_from_probs

PROBS_INPUT_IMPLS = {"flashinfer_from_probs"}


class Impl:
    """A ladder rung bound to one (shape, params) configuration."""

    def __init__(self, name, call, cold_method, notes=""):
        self.name = name
        self.call = call
        self.cold_method = cold_method
        self.notes = notes


def _rotating(fn, buffers, top_k, top_p):
    state = {"i": 0}
    n = len(buffers)

    def run():
        b = buffers[state["i"]]
        state["i"] = (state["i"] + 1) % n
        return fn(b, top_k, top_p)

    return run


def build_eager(name, fn, buffers, top_k, top_p, hot):
    if hot:
        x = buffers[0]
        return Impl(name, lambda: fn(x, top_k, top_p), "n/a")
    return Impl(name, _rotating(fn, buffers, top_k, top_p), "rotate")


def build_compile(fn, buffers, top_k, top_p, hot, name="compile"):
    compiled = torch.compile(fn, dynamic=False)
    if hot:
        x = buffers[0]
        run = lambda: compiled(x, top_k, top_p)
    else:
        run = _rotating(compiled, buffers, top_k, top_p)
    for _ in range(5):
        run()
    torch.cuda.synchronize()
    return Impl(name, run, "n/a" if hot else "rotate")


def build_graph(fn, buffers, top_k, top_p, hot, name="graph_eager", compile_first=False):
    target = torch.compile(fn, dynamic=False) if compile_first else fn
    static_in = buffers[0].clone()

    for _ in range(11):
        target(static_in, top_k, top_p)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = target(static_in, top_k, top_p)
    torch.cuda.synchronize()

    if hot:
        return Impl(name, lambda: graph.replay(), "n/a")

    state = {"i": 0}
    n = len(buffers)

    def run():
        # graph replay needs a fixed input address, so the cold condition costs an extra
        # [B,V] copy from the rotating pool. Recorded as a distinct cold method.
        static_in.copy_(buffers[state["i"]])
        state["i"] = (state["i"] + 1) % n
        graph.replay()

    return Impl(name, run, "rotate_into_static")
