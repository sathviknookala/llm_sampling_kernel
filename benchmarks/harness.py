import json
import platform
import subprocess

import torch


def nvidia_smi(query, extra=None):
    cmd = ["nvidia-smi", f"--query-{query[0]}={query[1]}", "--format=csv,noheader"]
    if extra:
        cmd += extra
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unavailable"


def gpu_is_busy(mem_mib_threshold=512):
    apps = nvidia_smi(("compute-apps", "pid,process_name,used_memory"))
    if apps in ("", "unavailable"):
        return False, apps
    used = 0
    for line in apps.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if parts and parts[-1].endswith("MiB"):
            used += int(parts[-1].split()[0])
    return used > mem_mib_threshold, apps


def try_lock_clocks():
    try:
        r = subprocess.run(["nvidia-smi", "-lgc", "3090"], capture_output=True, text=True)
        return r.returncode == 0
    except Exception:
        return False


def git_commit():
    try:
        h = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
        return h + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def environment(clocks_locked):
    p = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    busy, apps = gpu_is_busy()
    env = {
        "gpu_name": p.name if p else "none",
        "gpu_sm": f"{p.major}{p.minor}" if p else "none",
        "gpu_l2_bytes": p.L2_cache_size if p else 0,
        "gpu_sm_count": p.multi_processor_count if p else 0,
        "gpu_total_mem_bytes": p.total_memory if p else 0,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "driver": nvidia_smi(("gpu", "driver_version")).splitlines()[0] if p else "none",
        "python": platform.python_version(),
        "clocks_locked": clocks_locked,
        "clocks_max_sm_mhz": nvidia_smi(("gpu", "clocks.max.sm")).splitlines()[0] if p else "none",
        "persistence_mode": nvidia_smi(("gpu", "persistence_mode")).splitlines()[0] if p else "none",
        "gpu_busy_at_start": busy,
        "gpu_compute_apps_at_start": apps.replace("\n", "; ") or "none",
        "git_commit": git_commit(),
    }
    try:
        import transformers

        env["transformers_version"] = transformers.__version__
    except Exception:
        env["transformers_version"] = "absent"
    try:
        import flashinfer

        env["flashinfer_version"] = flashinfer.__version__
    except Exception:
        env["flashinfer_version"] = "absent"
    return env


def measured_dram_bandwidth(nbytes=512 << 20, iters=20):
    a = torch.empty(nbytes // 2, dtype=torch.float16, device="cuda")
    a.normal_()
    b = torch.empty_like(a)
    for _ in range(5):
        b.copy_(a)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        b.copy_(a)
    e.record()
    torch.cuda.synchronize()
    seconds = s.elapsed_time(e) / 1000 / iters
    return 2 * nbytes / seconds


def amortized_us(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000.0 / iters


def calibrate_iters(fn, target_ms=120.0, min_iters=20, max_iters=5000):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(10):
        fn()
    e.record()
    torch.cuda.synchronize()
    per_call_ms = s.elapsed_time(e) / 10
    if per_call_ms <= 0:
        return max_iters
    return int(min(max_iters, max(min_iters, target_ms / per_call_ms)))


def repeat_amortized(fn, reps, iters, warmup):
    return [amortized_us(fn, iters, warmup if r == 0 else max(2, warmup // 4)) for r in range(reps)]


def percentile(xs, q):
    if not xs:
        return float("nan")
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * q
    lo, hi = int(pos), min(int(pos) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def dump_env(path, clocks_locked):
    env = environment(clocks_locked)
    with open(path, "w") as f:
        json.dump(env, f, indent=2, sort_keys=True)
    return env
