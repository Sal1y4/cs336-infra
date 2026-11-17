import gc
import json
import math
import timeit
from contextlib import nullcontext
from dataclasses import asdict, dataclass

import torch

from cs336_basics.ablations import build_optimizer, get_loss_fn
from cs336_basics.model import TransformerLM
from cs336_basics.nn_utils import clip_grad_norm_
from cs336_basics.runinfo import gpu_snapshot, run_metadata


# 讲义 TinyStories 配置（7.2.1）和 Assignment 2 表 1 的 small 配置。
PRESETS = {
    "tiny": dict(vocab_size=1000, d_model=128, num_layers=2, num_heads=4, d_ff=512),
    "tinystories": dict(vocab_size=10000, d_model=512, num_layers=4, num_heads=16, d_ff=1344),
    "small": dict(vocab_size=10000, d_model=768, num_layers=12, num_heads=12, d_ff=3072),
}


@dataclass
class ModelConfig:
    vocab_size: int
    context_length: int
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int
    rope_theta: float = 10000.0


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def training_flops_per_token(config):
    # 只数矩阵乘：前向每个乘加算 2 FLOP，反向约为前向 2 倍，所以训练是前向 x3。
    d = config.d_model
    per_layer = 4 * d * d + 3 * d * config.d_ff
    matmul_params = config.num_layers * per_layer + d * config.vocab_size  # 加 lm_head；embedding 是查表
    # QK^T 和 (softmax)V：每个 token 每层 2 * 2 * T * d（causal mask 没有省掉计算，按满矩阵算）
    attention = config.num_layers * 4 * config.context_length * d
    return 3 * (2 * matmul_params + attention)


@dataclass
class BenchmarkResult:
    mode: str
    batch_size: int
    context_length: int
    dtype: str
    tf32: bool
    oom: bool
    forward_ms: float | None = None
    backward_ms: float | None = None
    optimizer_ms: float | None = None
    step_ms: float | None = None
    step_ms_std: float | None = None
    step_ms_p50: float | None = None
    step_ms_p99: float | None = None
    tokens_per_second: float | None = None
    achieved_tflops: float | None = None
    mfu: float | None = None
    peak_allocated_gib: float | None = None
    peak_reserved_gib: float | None = None
    # 测量开始前和结束后各取一次 nvidia-smi：时钟或温度在测量期间变了，结果就要打折看。
    gpu_before: dict | None = None
    gpu_after: dict | None = None


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _mean_std(values):
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, variance ** 0.5


def percentile(values, q):
    # 最近秩法：排序后取第 ceil(q/100 * n) 个。样本少时 P99 就是最大值，
    # 所以想看长尾要把 measure_steps 设到 100 以上。
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100 * len(ordered)))
    return ordered[rank - 1]


def run_benchmark(
    config,
    batch_size,
    mode="train",          # "forward" | "forward_backward" | "train"
    dtype="fp32",          # "fp32" | "bf16"（bf16 用 autocast，参数仍是 fp32）
    tf32=False,
    warmup_steps=5,
    measure_steps=10,
    device="cuda",
    peak_tflops=None,      # 用本机实测 matmul 峰值算 MFU；不给就不算
    max_l2_norm=1.0,
    seed=0,
    memory_fraction=None,  # 限制 PyTorch 可用显存比例，超出直接 OOM
    attention_impl="naive",
    loss_impl="naive",
    optimizer_impl="naive",
    compile=False,
):
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        # set_per_process_memory_fraction 等接口要求带编号的设备，"cuda" 不行。
        device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(seed)
    if device.type == "cuda":
        # Windows/WSL 驱动默认会在显存不够时溢出到共享系统内存（sysmem fallback），
        # 程序不报 OOM，但会慢 10 倍以上。给一个上限，让超显存的配置直接报 OOM。
        if memory_fraction is not None:
            torch.cuda.set_per_process_memory_fraction(memory_fraction, device)
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32

    result = BenchmarkResult(mode, batch_size, config.context_length, dtype, tf32, oom=False)

    model = optimizer = None
    try:
        model = TransformerLM(**asdict(config), attention_impl=attention_impl).to(device)
        optimizer = build_optimizer(optimizer_impl, model.parameters(), lr=1e-3)
        loss_fn = get_loss_fn(loss_impl)
        if compile:
            # 编译发生在第一次前向，会落在 warmup 里，不计入测量
            model = torch.compile(model)

        # 随机数据就够了：这里只测速度和显存，不看 loss。
        x = torch.randint(0, config.vocab_size, (batch_size, config.context_length), device=device)
        y = torch.randint(0, config.vocab_size, (batch_size, config.context_length), device=device)

        if dtype == "bf16":
            autocast = torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        else:
            autocast = nullcontext()

        def one_step():
            # 每个阶段前后都同步：CUDA 是异步的，不同步只会量到 kernel 发射的时间。
            times = {}
            start = timeit.default_timer()
            with autocast:
                logits = model(x)
                loss = loss_fn(logits, y) if mode != "forward" else None
            _sync(device)
            times["forward"] = timeit.default_timer() - start

            if mode == "forward":
                return times

            start = timeit.default_timer()
            loss.backward()
            _sync(device)
            times["backward"] = timeit.default_timer() - start

            start = timeit.default_timer()
            if mode == "train":
                clip_grad_norm_(model.parameters(), max_l2_norm)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            _sync(device)
            times["optimizer"] = timeit.default_timer() - start
            return times

        if mode == "forward":
            model.eval()
            grad_context = torch.no_grad()
        else:
            grad_context = nullcontext()

        with grad_context:
            for _ in range(warmup_steps):
                one_step()

            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                result.gpu_before = gpu_snapshot(device.index)

            records = [one_step() for _ in range(measure_steps)]

            if device.type == "cuda":
                result.gpu_after = gpu_snapshot(device.index)

        def phase_ms(name):
            if name not in records[0]:
                return None
            return 1000 * sum(record[name] for record in records) / len(records)

        step_seconds = [sum(record.values()) for record in records]
        step_mean, step_std = _mean_std(step_seconds)

        result.forward_ms = phase_ms("forward")
        result.backward_ms = phase_ms("backward")
        result.optimizer_ms = phase_ms("optimizer")
        result.step_ms = 1000 * step_mean
        result.step_ms_std = 1000 * step_std
        result.step_ms_p50 = 1000 * percentile(step_seconds, 50)
        result.step_ms_p99 = 1000 * percentile(step_seconds, 99)
        tokens = batch_size * config.context_length
        result.tokens_per_second = tokens / step_mean

        if mode == "train":
            result.achieved_tflops = training_flops_per_token(config) * result.tokens_per_second / 1e12
            if peak_tflops:
                result.mfu = result.achieved_tflops / peak_tflops

        if device.type == "cuda":
            result.peak_allocated_gib = torch.cuda.max_memory_allocated(device) / 2**30
            result.peak_reserved_gib = torch.cuda.max_memory_reserved(device) / 2**30
    except torch.OutOfMemoryError:
        result.oom = True
    finally:
        # 释放这一组的显存，避免影响下一组的峰值统计。
        model = optimizer = None
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return result


def environment_info(device):
    # 保留旧名字；内容见 cs336_basics/runinfo.py
    return run_metadata(device)


def append_jsonl(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
