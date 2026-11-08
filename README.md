# Transformer Training Efficiency on a Single 6 GB GPU

Self-study of [Stanford CS336](https://stanford-cs336.github.io/spring2025/) Assignment 1 (I was not enrolled in the course). I implemented the BPE tokenizer, a 22.7M-parameter Transformer LM, AdamW and the training loop myself, then profiled and optimized training on an RTX 3060 Laptop GPU (6 GB, WSL2). The assignment scaffold, handout and tests come from the course (MIT license, see [LICENSE](./LICENSE)).

## At a glance

- **Training step 352.6 ms to 85.2 ms (4.1x throughput)** by stacking TF32, bf16 autocast, SDPA, fused cross-entropy, fused AdamW and `torch.compile`; peak memory 3.49 GiB to 1.83 GiB. Each step is ablated separately (tables below).
- **Same model quality:** 5,000-step runs on TinyStories with 2 seeds; final validation loss matches the fp32 baseline within 0.001 at every 1,000-step checkpoint.
- **Found and fixed a bf16 bug:** RoPE angles computed in bf16 had a max error of 1.20 at T=256; they are now computed in fp32, with tests in `tests/test_mixed_precision.py`.
- **Honest limits:** part of the end-to-end speedup comes from different SM clocks (the GPU sits at its 80 W power cap; clocks are recorded with every result), and only 2 seeds were run.

Reproduce: see the commands under "复现" below (`experiments/run_ablation.sh`, `experiments/train.py`). The detailed write-up below is in Chinese.

---

# 6GB 单卡上的 Transformer 训练效率分析

在一块 RTX 3060 Laptop（6GB，WSL2）上，对自学 [Stanford CS336](https://stanford-cs336.github.io/spring2025/) Assignment 1 时自己实现的 Transformer LM（22.7M 参数）做训练效率分析和优化：分阶段 benchmark、bf16 数值修复、逐项消融，最后用端到端收敛实验验证。

这是自学项目，没有选这门课。作业框架、讲义和测试来自课程（MIT 协议，见 [LICENSE](./LICENSE)），模型和训练组件是我自己按作业要求写的。本 README 下面的部分和 `cs336_basics/benchmarking.py`、`training.py`、`ablations.py`、`experiments/` 是在此基础上新增的。

## 结果

测量条件：讲义 TinyStories 配置（d_model 512，4 层，16 头），ctx 256，batch 32。GPU 一直顶着 80 W 功耗墙，所以每次结果都同时记录 SM 时钟。

**单步消融**（`experiments/run_ablation.sh`，跑了两遍，两遍之间各项差异 ≤ 2.8%）

| 配置（逐项累加） | step ms | tok/s | 峰值显存 GiB |
|---|---|---|---|
| fp32（自实现 attention / CE / AdamW） | 352.6 | 23,236 | 3.49 |
| + TF32 | 265.6 | 30,844 | |
| bf16 autocast | 202.6 | 40,436 | 2.96 |
| + SDPA | 152.5 | 53,731 | 2.25 |
| + torch cross_entropy | 140.9 | 58,151 | |
| + fused AdamW | 135.0 | 60,671 | |
| + torch.compile | **85.2** | **96,107** | **1.83** |

逐项去除（从最终配置出发，tok/s）：去掉 bf16（改用 TF32）50,435；去掉 SDPA 84,457；去掉 torch CE 86,857；去掉 fused AdamW 90,249。

**端到端收敛**（`experiments/train.py`，5000 步，40.96M token）

| run | 最终 val_loss | 平均 tok/s | 峰值显存 GiB | 墙钟 s |
|---|---|---|---|---|
| fp32 seed0 | 1.645 | 23,086 | 3.18 | 1831.6 |
| 最终配置 seed0 | 1.645 | 106,205 | 1.68 | 412.7 |
| fp32 seed1 | 1.639 | 24,080 | 3.18 | 1753.3 |
| 最终配置 seed1 | 1.639 | 109,550 | 1.68 | 395.9 |

同一个 seed 下，两种配置每 1000 步检查点的 val_loss 差 ≤ 0.001，比两个 seed 之间约 0.006 的差小得多。

**局限**
- 端到端 4.4 倍的加速里混入了时钟差异：最终配置的 run 约 1420 MHz，fp32 的 run 只有 1042–1207 MHz。
- 只跑了 2 个 seed。
- 开 compile 后 optimizer 阶段反而变慢（约 9–10 ms，不开时 6.8 ms），原因还没查清。

## 改了什么

- **Benchmark**（`cs336_basics/benchmarking.py`）：前向、反向、optimizer 每个阶段后都同步再计时；记录峰值显存，计算 MFU。WSL 下显存不够时，驱动会把数据溢出到系统内存，程序不报 OOM，但会慢 10–60 倍。`--memory-fraction` 给显存设上限，让这种情况直接报 OOM。
- **bf16 数值**（`model.py`）：RoPE 原来用输入 dtype 计算角度，bf16 下 T=256 时最大误差达到 1.20，现在改为 fp32 计算。RMSNorm 先升到 fp32 再计算。对应的测试在 `tests/test_mixed_precision.py`。
- **消融开关**（`cs336_basics/ablations.py`）：attention 可选 naive / sdpa，loss 可选 naive / torch，optimizer 可选 naive / torch_foreach / torch_fused，另外可开 compile。都通过配置切换，并有测试保证换实现前后 loss 一致。
- **训练流水线**（`cs336_basics/training.py`）：数据用 memmap 读取，验证窗口固定；checkpoint 一并保存配置和 RNG 状态，先写临时文件再改名；测试保证续训与不中断运行逐位一致。

## 复现

```sh
uv run python experiments/prepare_data.py --train-text data/TinyStoriesV2-GPT4-train.txt \
    --valid-text data/TinyStoriesV2-GPT4-valid.txt --out-dir data/tinystories \
    --vocab-size 10000 --bpe-sample-mb 50 --train-mb 200
bash experiments/run_ablation.sh results/ablation.jsonl
uv run python experiments/train.py --config experiments/configs/tinystories_40m.json --out-dir runs/fp32_seed0
uv run python experiments/train.py --config experiments/configs/tinystories_40m.json --out-dir runs/final_seed0 \
    --dtype bf16 --attention-impl sdpa --loss-impl torch --optimizer-impl torch_fused --compile true
```

---

## 原作业 README

### CS336 Spring 2025 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_assignment1_basics.pdf](./cs336_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv#installation) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

