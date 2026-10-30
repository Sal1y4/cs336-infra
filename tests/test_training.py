import json
import sys
from pathlib import Path

import numpy as np
import torch

from cs336_basics.bpe import Tokenizer, train_bpe
from cs336_basics.training import TrainConfig, train

from .common import FIXTURES_PATH

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from prepare_data import SPECIAL_TOKEN, split_into_chunks  # noqa: E402


def _write_tokens(path, tokens):
    np.save(path, np.asarray(tokens, dtype=np.uint16))
    return str(path)


def _tiny_config(tmp_path, **overrides):
    # 一段重复的短序列：模型应该能很快记住，loss 明显下降。
    pattern = list(range(50)) * 200
    values = dict(
        train_data=_write_tokens(tmp_path / "train.npy", pattern),
        val_data=_write_tokens(tmp_path / "val.npy", pattern[:2000]),
        out_dir=str(tmp_path / "run"),
        vocab_size=64,
        context_length=16,
        d_model=32,
        num_layers=2,
        num_heads=2,
        d_ff=64,
        batch_size=8,
        max_steps=60,
        max_lr=3e-3,
        min_lr=3e-4,
        warmup_steps=5,
        eval_interval=20,
        eval_batches=2,
        log_interval=10,
        checkpoint_interval=1000,
        device="cpu",
    )
    values.update(overrides)
    return TrainConfig(**values)


def test_train_loss_decreases_and_logs(tmp_path):
    config = _tiny_config(tmp_path)
    history = train(config)

    assert history[-1]["train_loss"] < history[0]["train_loss"] - 1.0
    val_losses = [record["val_loss"] for record in history if "val_loss" in record]
    assert len(val_losses) == 3 and val_losses[-1] < val_losses[0]

    log_lines = (Path(config.out_dir) / "log.jsonl").read_text().splitlines()
    assert len(log_lines) == len(history)
    assert json.loads((Path(config.out_dir) / "config.json").read_text())["max_steps"] == 60
    assert (Path(config.out_dir) / "checkpoint.pt").exists()


def test_lr_schedule_reaches_min_lr_at_max_steps(tmp_path):
    history = train(_tiny_config(tmp_path, max_steps=20, log_interval=1, eval_interval=1000))
    assert history[0]["lr"] == 0.0  # warmup 从 0 开始
    assert abs(history[-1]["lr"] - 3e-4) < 1e-4


def test_resume_matches_uninterrupted_run(tmp_path):
    (tmp_path / "a").mkdir()
    straight = _tiny_config(tmp_path / "a", max_steps=20, log_interval=1, eval_interval=1000)
    straight_history = train(straight)

    (tmp_path / "b").mkdir()
    resumed = _tiny_config(tmp_path / "b", max_steps=20, log_interval=1, eval_interval=1000)
    train(resumed, max_steps_this_run=10)
    resumed_history = train(resumed, resume_from=Path(resumed.out_dir) / "checkpoint.pt")

    # 续训的后 10 步应与不中断运行的后 10 步逐位一致（CPU fp32 是确定性的）。
    assert [r["train_loss"] for r in resumed_history] == [r["train_loss"] for r in straight_history[10:]]


def test_split_into_chunks_preserves_text_and_encoding():
    text = (FIXTURES_PATH / "tinystories_sample.txt").read_text()
    chunks = split_into_chunks(text, 5)
    assert "".join(chunks) == text and len(chunks) > 1

    vocab, merges = train_bpe(FIXTURES_PATH / "tinystories_sample.txt", 300, [SPECIAL_TOKEN])
    tokenizer = Tokenizer(vocab, merges, [SPECIAL_TOKEN])
    chunked = [token for chunk in chunks for token in tokenizer.encode(chunk)]
    assert chunked == tokenizer.encode(text)


def test_bf16_training_runs_on_cpu(tmp_path):
    history = train(_tiny_config(tmp_path, dtype="bf16", max_steps=10, eval_interval=10))
    assert torch.isfinite(torch.tensor(history[-1]["val_loss"]))


def test_training_with_ablation_switches_matches_naive(tmp_path):
    # 开关只换实现、不换数学：前几步的 loss 应与自实现版本几乎一致。
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    common = dict(max_steps=10, log_interval=1, eval_interval=1000)
    naive = train(_tiny_config(tmp_path / "a", **common))
    swapped = train(
        _tiny_config(
            tmp_path / "b", attention_impl="sdpa", loss_impl="torch", optimizer_impl="torch_foreach", **common
        )
    )
    for a, b in zip(naive, swapped):
        assert abs(a["train_loss"] - b["train_loss"]) < 1e-3
