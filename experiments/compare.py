"""交替对比几个 benchmark 配置。规则见 docs/measurement-rules.md。

例子（项目根目录下）：
    uv run python experiments/compare.py --rounds 5 --out results/compare_compile.jsonl \\
        --config "fused=--dtype bf16 --attention-impl sdpa --loss-impl torch --optimizer-impl torch_fused" \\
        --config "fused+compile=--dtype bf16 --attention-impl sdpa --loss-impl torch --optimizer-impl torch_fused --compile" \\
        -- --preset tinystories --context-length 256 --batch-size 32 --memory-fraction 0.8 --measure-steps 50

`--` 后面的参数原样传给每一次 experiments/benchmark.py。
每次运行都起一个新进程：上一个配置的显存、编译缓存和分配器状态不会带到下一个里。
"""

import argparse
import json
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cs336_basics.benchmarking import append_jsonl  # noqa: E402
from cs336_basics.comparison import format_markdown, interleaved_order, summarize  # noqa: E402

BENCHMARK_SCRIPT = Path(__file__).resolve().parent / "benchmark.py"


def parse_config(text):
    name, separator, flags = text.partition("=")
    if not separator or not name.strip():
        raise argparse.ArgumentTypeError('expected NAME="benchmark flags", for example fused="--dtype bf16"')
    return name.strip(), shlex.split(flags)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=parse_config, action="append", required=True, help='NAME="benchmark.py 的参数"')
    parser.add_argument("--rounds", type=int, default=5, help="每个配置跑几轮（规则要求至少 5）")
    parser.add_argument("--metric", default="step_ms")
    parser.add_argument("--baseline", help="作为比较对象的配置名，默认是第一个 --config")
    parser.add_argument("--out", required=True, help="所有运行的原始记录（JSONL）")
    parser.add_argument("common", nargs="*", help="-- 之后的参数，传给每一次 benchmark.py")
    return parser.parse_args()


def run_once(flags, common):
    """跑一次 benchmark.py，返回它写出的记录（一个 batch size 一条）。"""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "bench.jsonl"
        command = [sys.executable, str(BENCHMARK_SCRIPT), *common, *flags, "--out", str(out)]
        subprocess.run(command, check=True)
        return [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    args = parse_args()
    configs = dict(args.config)
    if len(configs) != len(args.config):
        raise SystemExit("config names must be unique")

    order = interleaved_order(list(configs), args.rounds)
    records = []
    for position, (round_index, name) in enumerate(order):
        print(f"== round {round_index + 1}/{args.rounds}  config {name}  ({position + 1}/{len(order)})", flush=True)
        results = run_once(configs[name], args.common)
        if len(results) != 1:
            raise SystemExit("compare.py expects exactly one batch size and one context length per run")
        record = {"config": name, "round": round_index, "position": position, **results[0]}
        records.append(record)
        append_jsonl(args.out, record)  # 每跑完一次就落盘：中途断了也不丢已有结果

    print()
    print(format_markdown(summarize(records, metric=args.metric, baseline=args.baseline), metric=args.metric))


if __name__ == "__main__":
    main()
