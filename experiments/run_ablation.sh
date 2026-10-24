#!/usr/bin/env bash
# 累加式消融：每一行在上一行基础上多打开一个开关；最后几行是“最终配置去掉一项”。
# 用法：bash experiments/run_ablation.sh results/ablation.jsonl
set -euo pipefail
OUT=${1:-results/ablation.jsonl}
mkdir -p "$(dirname "$OUT")"
COMMON="--preset tinystories --context-length 256 --batch-size 32 --memory-fraction 0.8 --warmup-steps 5 --measure-steps 20 --out $OUT"

run() { echo "== $*"; uv run python experiments/benchmark.py $COMMON "$@"; }

run                                                      # 0 基线 fp32
run --tf32                                               # 1
run --dtype bf16                                         # 2
run --dtype bf16 --attention-impl sdpa                   # 3
run --dtype bf16 --attention-impl sdpa --loss-impl torch # 4
FINAL="--dtype bf16 --attention-impl sdpa --loss-impl torch --optimizer-impl torch_fused"
run $FINAL                                               # 5
run $FINAL --compile                                     # 6 最终配置
# 留一法：最终配置各去掉一项
run --attention-impl sdpa --loss-impl torch --optimizer-impl torch_fused --compile --tf32  # 去 bf16（保留 tf32）
run --dtype bf16 --loss-impl torch --optimizer-impl torch_fused --compile                  # 去 SDPA
run --dtype bf16 --attention-impl sdpa --optimizer-impl torch_fused --compile              # 去 torch CE
run --dtype bf16 --attention-impl sdpa --loss-impl torch --compile                         # 去 fused AdamW
