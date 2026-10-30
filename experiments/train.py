"""训练入口。

例子（项目根目录下）：
    uv run python experiments/train.py --config experiments/configs/tinystories_40m.json --out-dir runs/fp32_seed0
    uv run python experiments/train.py --config experiments/configs/tinystories_40m.json --dtype bf16 --out-dir runs/bf16_seed0
    uv run python experiments/train.py --config experiments/configs/tinystories_40m.json --out-dir runs/fp32_seed0 --resume runs/fp32_seed0/checkpoint.pt
"""

import argparse
import json
from dataclasses import fields

import torch

from cs336_basics.training import TrainConfig, train


def parse_bool(value):
    return value.lower() in ("1", "true", "yes")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="JSON 配置文件，命令行参数会覆盖其中的同名项")
    parser.add_argument("--resume", help="从这个 checkpoint 续训")
    # 每个 TrainConfig 字段都可以从命令行覆盖，比如 --dtype bf16 --seed 1
    for field in fields(TrainConfig):
        kind = parse_bool if field.type is bool else field.type
        parser.add_argument("--" + field.name.replace("_", "-"), type=kind)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, encoding="utf-8") as f:
        values = json.load(f)
    for field in fields(TrainConfig):
        override = getattr(args, field.name)
        if override is not None:
            values[field.name] = override
    values.setdefault("device", "cuda" if torch.cuda.is_available() else "cpu")
    train(TrainConfig.from_dict(values), resume_from=args.resume)


if __name__ == "__main__":
    main()
