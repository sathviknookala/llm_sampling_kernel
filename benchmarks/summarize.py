import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

LADDER_ORDER = [
    "hf_eager", "ref_eager_fullsort", "tight_eager", "compile",
    "graph_eager", "graph_compile", "flashinfer", "flashinfer_from_probs",
]


def load(path):
    return list(csv.DictReader(open(path)))


def med(rows, field="median_us"):
    return statistics.median(float(r[field]) for r in rows) if rows else float("nan")


def group(rows, *keys):
    out = defaultdict(list)
    for r in rows:
        out[tuple(r[k] for k in keys)].append(r)
    return out


def order_impls(names):
    return sorted(names, key=lambda n: (LADDER_ORDER.index(n) if n in LADDER_ORDER else 99, n))


def table(lines, title, header, body):
    lines.append(f"\n### {title}\n")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join("---" for _ in header) + "|")
    lines.extend(body)


def anchor_rows(rows, vocab, top_k, top_p):
    return [r for r in rows
            if r["vocab"] == str(vocab) and r["top_k"] == str(top_k) and r["top_p"] == str(top_p)]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="results/raw/sampling_ladder.csv")
    ap.add_argument("--out", default="results/summary_ladder.md")
    ap.add_argument("--vocab", default="151936")
    ap.add_argument("--top-k", default="50")
    ap.add_argument("--top-p", default="0.9")
    args = ap.parse_args(argv)

    rows = load(args.raw)
    L = ["# Sampling Ladder Summary", "",
         f"Source: `{args.raw}` ({len(rows)} rows). Median across rounds/reps; ",
         "latency is amortized device time per sampling call, validation disabled in the timed region.", ""]

    anch = anchor_rows(rows, args.vocab, args.top_k, args.top_p)

    # 1. ladder by batch, hot, bf16
    for dtype in ("bfloat16", "float16"):
        for residency in ("hot", "cold"):
            sel = [r for r in anch if r["dtype"] == dtype and r["residency"] == residency]
            if not sel:
                continue
            batches = sorted({int(r["batch"]) for r in sel})
            impls = order_impls({r["impl"] for r in sel})
            g = group(sel, "impl", "batch")
            hf = {b: med(g.get(("hf_eager", str(b)), [])) for b in batches}
            body = []
            for im in impls:
                cells = []
                for b in batches:
                    m = med(g.get((im, str(b)), []))
                    sp = hf[b] / m if m == m and hf[b] == hf[b] else float("nan")
                    cells.append(f"{m:.1f} ({sp:.1f}x)" if m == m else "—")
                body.append(f"| {im} | " + " | ".join(cells) + " |")
            table(L, f"Latency µs (speedup vs hf_eager) — V={args.vocab}, k={args.top_k}, "
                     f"p={args.top_p}, {dtype}, {residency}",
                  ["impl"] + [f"B={b}" for b in batches], body)

    # 2. hot vs cold delta
    g = group(anch, "impl", "batch", "dtype")
    body = []
    for (im, b, dt) in sorted(g, key=lambda t: (order_impls([t[0]])[0], int(t[1]), t[2])):
        hot = [r for r in g[(im, b, dt)] if r["residency"] == "hot"]
        cold = [r for r in g[(im, b, dt)] if r["residency"] == "cold"]
        if not hot or not cold:
            continue
        h, c = med(hot), med(cold)
        body.append(f"| {im} | {b} | {dt} | {h:.1f} | {c:.1f} | {100*(c-h)/h:+.1f}% |")
    if body:
        table(L, "Hot vs cold L2 residency",
              ["impl", "batch", "dtype", "hot µs", "cold µs", "cold penalty"], body)

    # 3. DRAM floor
    body = []
    for (im, b, dt), rs in sorted(group(anch, "impl", "batch", "dtype").items(),
                                  key=lambda t: (order_impls([t[0][0]])[0], int(t[0][1]))):
        rs = [r for r in rs if r["residency"] == "hot"]
        if not rs or dt != "bfloat16":
            continue
        body.append(f"| {im} | {b} | {int(rs[0]['logits_bytes'])/1e6:.2f} | "
                    f"{float(rs[0]['dram_floor_us']):.2f} | {med(rs):.1f} | "
                    f"{100*float(rs[0]['dram_floor_us'])/med(rs):.2f}% |")
    table(L, "Memory-read floor vs measured latency (bfloat16, hot)",
          ["impl", "batch", "logits MB", "DRAM floor µs", "latency µs", "floor % of latency"], body)

    # 4. parameter sensitivity
    sens = [r for r in rows if r["residency"] == "hot" and r["dtype"] == "bfloat16"]
    g = group(sens, "impl", "vocab", "top_k", "top_p", "batch")
    body = []
    for key in sorted(g, key=lambda t: (order_impls([t[0]])[0], t[1], int(t[2]), t[3], int(t[4]))):
        im, v, k, p, b = key
        body.append(f"| {im} | {v} | {k} | {p} | {b} | {med(g[key]):.1f} |")
    table(L, "Parameter sensitivity (bfloat16, hot)",
          ["impl", "vocab", "top_k", "top_p", "batch", "median µs"], body)

    # 5. round-to-round spread (ordering artifact check)
    body = []
    for (im,), rs in sorted(group(anch, "impl").items(), key=lambda t: order_impls([t[0][0]])[0]):
        by_round = group(rs, "round")
        meds = [med(v) for v in by_round.values()]
        if len(meds) > 1:
            body.append(f"| {im} | {len(meds)} | {min(meds):.1f} | {max(meds):.1f} | "
                        f"{100*(max(meds)-min(meds))/min(meds):.1f}% |")
    table(L, "Round-to-round spread (impl order rotated each round)",
          ["impl", "rounds", "min median µs", "max median µs", "spread"], body)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(L) + "\n")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
