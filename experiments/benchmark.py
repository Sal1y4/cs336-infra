"""端到端训练步 benchmark。

例子（项目根目录下）：
    uv run python experiments/benchmark.py --preset tinystories --context-length 256 --batch-size 8 16 32
    uv run python experiments/benchmark.py --preset tinystories --dtype bf16 --peak-tflops 20.6 --out results/bench.jsonl
"""

import argparse
from dataclasses import asdict

import torch

from cs336_basics.benchmarking import (
    PRESETS,
    ModelConfig,
    TransformerLM,
    append_jsonl,
    count_parameters,
    environment_info,
    run_benchmark,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=PRESETS, default="tinystories")
    parser.add_argument("--vocab-size", type=int)
    parser.add_argument("--d-model", type=int)
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--num-heads", type=int)
    parser.add_argument("--d-ff", type=int)
    parser.add_argument("--context-length", type=int, nargs="+", default=[256])
    parser.add_argument("--batch-size", type=int, nargs="+", default=[32])
    parser.add_argument("--mode", choices=["forward", "forward_backward", "train"], default="train")
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=10)
    parser.add_argument("--peak-tflops", type=float, help="本机实测的 matmul 峰值，用来算 MFU")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", help="把每组结果追加到这个 JSONL 文件")
    return parser.parse_args()


def fmt(value, digits=1):
    return "-" if value is None else f"{value:.{digits}f}"


def main():
    args = parse_args()
    model_args = dict(PRESETS[args.preset])
    for key in ("vocab_size", "d_model", "num_layers", "num_heads", "d_ff"):
        if getattr(args, key) is not None:
            model_args[key] = getattr(args, key)

    env = environment_info(args.device)
    print(env)
    print("ctx  batch  fwd_ms  bwd_ms  opt_ms  step_ms(±std)  tok/s  TFLOP/s  MFU  peak_alloc_GiB")

    for context_length in args.context_length:
        config = ModelConfig(context_length=context_length, **model_args)
        num_params = count_parameters(TransformerLM(**asdict(config)))
        for batch_size in args.batch_size:
            result = run_benchmark(
                config,
                batch_size,
                mode=args.mode,
                dtype=args.dtype,
                tf32=args.tf32,
                warmup_steps=args.warmup_steps,
                measure_steps=args.measure_steps,
                device=args.device,
                peak_tflops=args.peak_tflops,
            )
            if result.oom:
                print(f"{context_length:<4} {batch_size:<6} OOM")
            else:
                print(
                    f"{context_length:<4} {batch_size:<6} {fmt(result.forward_ms):>6}  {fmt(result.backward_ms):>6}  "
                    f"{fmt(result.optimizer_ms):>6}  {fmt(result.step_ms)}(±{fmt(result.step_ms_std)})  "
                    f"{fmt(result.tokens_per_second, 0)}  {fmt(result.achieved_tflops, 2)}  "
                    f"{fmt(result.mfu, 3)}  {fmt(result.peak_allocated_gib, 2)}"
                )
            if args.out:
                append_jsonl(
                    args.out,
                    {"preset": args.preset, "num_params": num_params, **asdict(config), **asdict(result), **env},
                )


if __name__ == "__main__":
    main()
