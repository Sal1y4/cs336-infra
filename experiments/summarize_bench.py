"""把 benchmark 的 JSONL 结果整理成 Markdown 表格。

    uv run python experiments/summarize_bench.py results/ablation.jsonl
"""

import json
import sys


def describe(record):
    parts = [record["dtype"] + ("+tf32" if record["tf32"] else "")]
    for key, default in (("attention_impl", "naive"), ("loss_impl", "naive"), ("optimizer_impl", "naive")):
        if record.get(key, default) != default:
            parts.append(f"{key.removesuffix('_impl')}={record[key]}")
    if record.get("compile"):
        parts.append("compile")
    return " ".join(parts)


def main(path):
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    print("| 配置 | ctx | batch | step ms (±std) | fwd / bwd / opt ms | tok/s | 峰值显存 GiB |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for r in rows:
        if r["oom"]:
            print(f"| {describe(r)} | {r['context_length']} | {r['batch_size']} | OOM | | | |")
            continue
        print(
            f"| {describe(r)} | {r['context_length']} | {r['batch_size']} | {r['step_ms']:.1f} (±{r['step_ms_std']:.1f}) "
            f"| {r['forward_ms']:.1f} / {r['backward_ms']:.1f} / {r['optimizer_ms']:.1f} "
            f"| {r['tokens_per_second']:.0f} | {r['peak_allocated_gib'] or 0:.2f} |"
        )


if __name__ == "__main__":
    main(sys.argv[1])
