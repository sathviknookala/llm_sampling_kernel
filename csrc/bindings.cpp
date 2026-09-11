#include <torch/extension.h>

torch::Tensor probe_noop(torch::Tensor logits);
torch::Tensor probe_scan_rowblock(torch::Tensor logits);
torch::Tensor probe_scan_split(torch::Tensor logits, int64_t splits);
torch::Tensor topk_fused(torch::Tensor logits, int64_t top_k, int64_t splits_override);
torch::Tensor sample_fused(torch::Tensor logits, int64_t top_k, double top_p, int64_t seed,
                           int64_t offset, int64_t splits_override);
std::vector<torch::Tensor> stages_fused(torch::Tensor logits, int64_t top_k, double top_p,
                                        int64_t seed, int64_t offset, int64_t splits_override);
torch::Tensor probe_phase(torch::Tensor logits, int64_t top_k, double top_p, int64_t phase,
                          int64_t splits_override);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("probe_noop", &probe_noop, "launch floor: allocate [B] and write it");
  m.def("probe_scan_rowblock", &probe_scan_rowblock, "one full pass, one block per row");
  m.def("probe_scan_split", &probe_scan_split, "one full pass, grid split across SMs");
  m.def("sample_fused", &sample_fused, "fused top-k + top-p + sample, [B, V] -> [B]");
  m.def("topk_fused", &topk_fused, "debug: the selected candidate ids, descending");
  m.def("stages_fused", &stages_fused, "debug: (token_ids, keep, renormed) for Gate A");
  m.def("probe_phase", &probe_phase, "ablation: run phases 1..n of the pipeline and stop");
}
