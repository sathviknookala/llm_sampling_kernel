"""compute-sanitizer memcheck / racecheck / initcheck over the fused-kernel test grid.

Only racecheck is filtered to this repo's kernels. initcheck must not be: a filtered run stops
tracking writes by torch's kernels, so every torch-produced input reads as uninitialized.
"""

import argparse
import csv
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import harness

TOOLS = ("memcheck", "racecheck", "initcheck")
OURS = ("topk_partial_kernel", "merge_sample_kernel", "merge_ids_kernel")
FILTERED = {"racecheck"}
TOOL_ARGS = {"racecheck": ["--racecheck-report", "all"], "memcheck": [], "initcheck": []}
# positive control: never-written logits must trip initcheck, or a clean run proves nothing
CONTROL = ("import torch; import fused_sampling; "
           "x = torch.empty(4, 151936, device='cuda', dtype=torch.bfloat16); "
           "fused_sampling.sample_fused(x, 50, 0.9, 1, 0, 0); torch.cuda.synchronize()")
FIELDS = [
    "tool", "kernels_checked", "scope", "tests_passed", "tests_failed", "tests_skipped",
    "sanitizer_errors", "hazards", "clean", "wall_s", "log", "git_commit",
]


def count(pattern, text):
    m = re.search(pattern, text)
    return int(m.group(1)) if m else 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/raw/sanitizer.csv")
    ap.add_argument("--log-dir", default="results/raw/sanitizer")
    ap.add_argument("--tools", nargs="+", default=list(TOOLS))
    ap.add_argument("--tests", default="tests/test_fused_kernel.py")
    args = ap.parse_args(argv)

    cs = shutil.which("compute-sanitizer")
    if cs is None:
        print("! compute-sanitizer not on PATH")
        return 2
    logs = Path(args.log_dir)
    logs.mkdir(parents=True, exist_ok=True)
    commit = harness.git_commit()
    rows = []
    for tool in args.tools:
        log = logs / f"{tool}.log"
        filt = [a for k in OURS for a in ("--kernel-name", f"kns={k}")] if tool in FILTERED else []
        cmd = [cs, "--tool", tool, *TOOL_ARGS[tool], *filt, "--print-limit", "100",
               sys.executable, "-m", "pytest", args.tests, "-q", "-p", "no:cacheprovider"]
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        wall = time.time() - t0
        text = proc.stdout + proc.stderr
        log.write_text(" ".join(cmd) + "\n\n" + text)
        passed = count(r"(\d+) passed", text)
        failed = count(r"(\d+) failed", text) + count(r"(\d+) errors? in", text)
        errors = count(r"ERROR SUMMARY: (\d+) error", text)
        hazards = count(r"RACECHECK SUMMARY: (\d+) hazard", text)
        summary = "RACECHECK SUMMARY" if tool == "racecheck" else "ERROR SUMMARY"
        clean = passed > 0 and failed == 0 and errors == 0 and hazards == 0 and summary in text
        rows.append({
            "tool": tool, "kernels_checked": "ours" if tool in FILTERED else "all",
            "scope": args.tests, "tests_passed": passed, "tests_failed": failed,
            "tests_skipped": count(r"(\d+) skipped", text), "sanitizer_errors": errors,
            "hazards": hazards, "clean": clean, "wall_s": round(wall, 1), "log": str(log),
            "git_commit": commit,
        })
        print(f"  {tool:10s} passed={passed} failed={failed} errors={errors} hazards={hazards} "
              f"clean={clean} ({wall:.0f} s) -> {log}")

    log = logs / "initcheck_control.log"
    cmd = [cs, "--tool", "initcheck", "--print-limit", "5", sys.executable, "-c", CONTROL]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    text = proc.stdout + proc.stderr
    log.write_text(" ".join(cmd) + "\n\n" + text)
    errors = count(r"ERROR SUMMARY: (\d+) error", text)
    rows.append({
        "tool": "initcheck_positive_control", "kernels_checked": "all",
        "scope": "uninitialized logits", "tests_passed": "", "tests_failed": "",
        "tests_skipped": "", "sanitizer_errors": errors, "hazards": "",
        "clean": errors > 0 and "topk_partial_kernel" in text,
        "wall_s": round(time.time() - t0, 1), "log": str(log), "git_commit": commit,
    })
    print(f"  control    errors={errors} discriminates={rows[-1]['clean']} -> {log}")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} rows -> {args.out}")
    return 0 if all(r["clean"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
