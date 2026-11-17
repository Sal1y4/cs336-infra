"""运行环境记录：每次 benchmark 或训练都把“这是在什么代码、什么环境、GPU 什么状态下跑的”写进结果。

为什么需要：同一份代码在不同时钟频率、温度、后台占用下测出的速度可以差 10% 以上。
只留下一个 tok/s 数字，事后就没法判断两次结果的差异来自代码还是来自环境。
做法参考 modded-nanogpt 的 run_log.py（日志开头写入源码、版本和 nvidia-smi 输出）。
"""

import platform
import subprocess
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]

# nvidia-smi --query-gpu 的字段。clocks.sm 是核心时钟：笔记本顶着功耗墙时它会掉，速度跟着掉。
SMI_FIELDS = (
    "name",
    "driver_version",
    "temperature.gpu",
    "power.draw",
    "utilization.gpu",
    "memory.used",
    "clocks.sm",
)
# 这些字段是数字，其余保留为字符串。
SMI_NUMERIC_FIELDS = ("temperature.gpu", "power.draw", "utilization.gpu", "memory.used", "clocks.sm")


def _run(command, cwd=None, timeout=10):
    try:
        completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def git_revision(repo_root=REPO_ROOT):
    """当前 commit，以及工作区有没有未提交的改动（dirty=True 说明结果对应不上任何一个 commit）。"""
    commit = _run(["git", "rev-parse", "--short", "HEAD"], cwd=repo_root)
    if commit is None:
        return {"git_commit": None, "git_dirty": None}
    status = _run(["git", "status", "--porcelain"], cwd=repo_root)
    return {"git_commit": commit, "git_dirty": None if status is None else bool(status)}


def parse_smi_csv(fields, line):
    """解析 `nvidia-smi --query-gpu=... --format=csv,noheader,nounits` 的一行输出。

    不支持的字段 nvidia-smi 会输出 "[N/A]" 之类的文字，这里统一记成 None。
    """
    values = [value.strip() for value in line.split(",")]
    if len(values) != len(fields):
        raise ValueError(f"expected {len(fields)} values, got {len(values)}: {line!r}")
    record = {}
    for field, value in zip(fields, values):
        if field in SMI_NUMERIC_FIELDS:
            try:
                record[field] = float(value)
            except ValueError:
                record[field] = None
        else:
            record[field] = None if value.startswith("[") else value
    return record


def gpu_snapshot(index=0):
    """此刻的 GPU 状态。没有 nvidia-smi（比如 CPU 机器）时返回 None。"""
    output = _run(
        [
            "nvidia-smi",
            f"--id={index}",
            "--query-gpu=" + ",".join(SMI_FIELDS),
            "--format=csv,noheader,nounits",
        ]
    )
    if not output:
        return None
    try:
        return parse_smi_csv(SMI_FIELDS, output.splitlines()[0])
    except ValueError:
        return None


def run_metadata(device):
    """一次运行开始时记录的全部环境信息。"""
    device = torch.device(device)
    info = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": str(device),
        **git_revision(),
    }
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(device)
        info["gpu_snapshot"] = gpu_snapshot(device.index or 0)
    return info
