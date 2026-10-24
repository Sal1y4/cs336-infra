import json
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch

from cs336_basics.data import get_batch
from cs336_basics.model import TransformerLM
from cs336_basics.nn_utils import clip_grad_norm_, cross_entropy
from cs336_basics.optimizers import AdamW, get_lr_cosine_schedule


@dataclass
class TrainConfig:
    train_data: str
    val_data: str
    out_dir: str = "runs/default"
    # 模型（默认是讲义 TinyStories 配置）
    vocab_size: int = 10000
    context_length: int = 256
    d_model: int = 512
    num_layers: int = 4
    num_heads: int = 16
    d_ff: int = 1344
    rope_theta: float = 10000.0
    # 优化
    batch_size: int = 32
    max_steps: int = 5000
    max_lr: float = 1e-3
    min_lr: float = 1e-4
    warmup_steps: int = 200
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    max_grad_norm: float = 1.0
    # 数值与设备
    dtype: str = "fp32"  # "fp32" | "bf16"
    tf32: bool = False
    device: str = "cuda"
    seed: int = 0
    # 评估、日志、checkpoint（单位都是 step）
    eval_interval: int = 250
    eval_batches: int = 20
    log_interval: int = 10
    checkpoint_interval: int = 1000

    @classmethod
    def from_dict(cls, values):
        known = {field.name for field in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**values)

    def model_kwargs(self):
        return dict(
            vocab_size=self.vocab_size,
            context_length=self.context_length,
            d_model=self.d_model,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            rope_theta=self.rope_theta,
        )


def load_tokens(path):
    # memmap：不把整份数据读进内存，按需从磁盘取。
    return np.load(path, mmap_mode="r")


def fixed_val_starts(num_tokens, context_length, num_windows, seed):
    # 验证集的窗口位置固定下来，不同 step、不同 run 之间的 val loss 才可比。
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, num_tokens - context_length, (num_windows,), generator=generator)


@torch.no_grad()
def evaluate(model, val_tokens, starts, config, device, autocast):
    model.eval()
    losses = []
    for batch_starts in starts.split(config.batch_size):
        x = torch.stack([torch.from_numpy(val_tokens[s : s + config.context_length].astype(np.int64)) for s in batch_starts])
        y = torch.stack(
            [torch.from_numpy(val_tokens[s + 1 : s + 1 + config.context_length].astype(np.int64)) for s in batch_starts]
        )
        x, y = x.to(device), y.to(device)
        with autocast:
            losses.append(cross_entropy(model(x), y).float().item())
    model.train()
    return sum(losses) / len(losses)


def save_training_checkpoint(path, model, optimizer, step, config):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": step,
        "config": asdict(config),
        # 保存随机数状态：续训后抽到的 batch 与不中断时完全一样。
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    tmp = Path(str(path) + ".tmp")
    torch.save(checkpoint, tmp)
    tmp.replace(path)  # 先写临时文件再改名，中途被杀不会留下半个 checkpoint


def load_training_checkpoint(path, model, optimizer):
    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    torch.set_rng_state(checkpoint["torch_rng_state"])
    if checkpoint["cuda_rng_state"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
    return checkpoint["iteration"]


def train(config, resume_from=None, max_steps_this_run=None):
    device = torch.device(config.device)
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(config), indent=2))

    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = config.tf32
        torch.backends.cudnn.allow_tf32 = config.tf32

    train_tokens = load_tokens(config.train_data)
    val_tokens = load_tokens(config.val_data)
    val_starts = fixed_val_starts(len(val_tokens), config.context_length, config.eval_batches * config.batch_size, seed=1234)

    model = TransformerLM(**config.model_kwargs()).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=config.max_lr,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
        weight_decay=config.weight_decay,
    )

    step = 0
    if resume_from is not None:
        step = load_training_checkpoint(resume_from, model, optimizer)

    if config.dtype == "bf16":
        autocast = torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    else:
        autocast = nullcontext()

    end_step = config.max_steps if max_steps_this_run is None else min(config.max_steps, step + max_steps_this_run)
    tokens_per_step = config.batch_size * config.context_length
    log_path = out_dir / "log.jsonl"
    history = []

    def log(record):
        history.append(record)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    window_start = time.perf_counter()
    window_steps = 0

    while step < end_step:
        # 调度器按“即将执行的这一步”取学习率，cosine 在 max_steps 处降到 min_lr。
        lr = get_lr_cosine_schedule(step, config.max_lr, config.min_lr, config.warmup_steps, config.max_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = get_batch(train_tokens, config.batch_size, config.context_length, device)
        with autocast:
            loss = cross_entropy(model(x), y)
        loss.backward()
        grad_norm = clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        window_steps += 1

        if step % config.log_interval == 0 or step == end_step:
            # .item() 会同步 GPU，所以只在打日志时调用。
            elapsed = time.perf_counter() - window_start
            record = {
                "step": step,
                "tokens": step * tokens_per_step,
                "train_loss": loss.float().item(),
                "lr": lr,
                "grad_norm": float(grad_norm),
                "tokens_per_second": window_steps * tokens_per_step / elapsed,
            }
            if device.type == "cuda":
                record["peak_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
            if step % config.eval_interval == 0 or step == end_step:
                record["val_loss"] = evaluate(model, val_tokens, val_starts, config, device, autocast)
            log(record)
            print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in record.items()))
            window_start = time.perf_counter()
            window_steps = 0

        if step % config.checkpoint_interval == 0 or step == end_step:
            save_training_checkpoint(out_dir / "checkpoint.pt", model, optimizer, step, config)

    return history
