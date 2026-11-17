import json
import subprocess
import sys
from pathlib import Path

import pytest

from cs336_basics.benchmarking import PRESETS, ModelConfig, percentile, run_benchmark
from cs336_basics.comparison import format_markdown, interleaved_order, mean_std, summarize
from cs336_basics.runinfo import SMI_FIELDS, git_revision, parse_smi_csv, run_metadata

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_parse_smi_csv():
    line = "NVIDIA GeForce RTX 3060 Laptop GPU, 591.59, 71, 79.84, 100, 3412, 1207"
    record = parse_smi_csv(SMI_FIELDS, line)
    assert record["name"] == "NVIDIA GeForce RTX 3060 Laptop GPU"
    assert record["driver_version"] == "591.59"
    assert record["temperature.gpu"] == 71.0
    assert record["power.draw"] == pytest.approx(79.84)
    assert record["clocks.sm"] == 1207.0


def test_parse_smi_csv_unsupported_field_is_none():
    line = "Some GPU, 591.59, 60, [N/A], 5, 100, [Not Supported]"
    record = parse_smi_csv(SMI_FIELDS, line)
    assert record["power.draw"] is None and record["clocks.sm"] is None
    assert record["utilization.gpu"] == 5.0


def test_parse_smi_csv_wrong_field_count():
    with pytest.raises(ValueError):
        parse_smi_csv(SMI_FIELDS, "only, two")


def test_run_metadata_cpu():
    info = run_metadata("cpu")
    assert info["device"] == "cpu" and info["torch"]
    assert "gpu_snapshot" not in info
    json.dumps(info)  # 必须能直接写进 JSONL


def test_git_revision_outside_a_repo(tmp_path):
    assert git_revision(tmp_path) == {"git_commit": None, "git_dirty": None}


def test_interleaved_order_rotates_start():
    order = interleaved_order(["a", "b", "c"], rounds=4)
    assert [name for _, name in order] == list("abc" "bca" "cab" "abc")
    # 每一轮每个配置恰好出现一次
    for round_index in range(4):
        assert sorted(name for r, name in order if r == round_index) == ["a", "b", "c"]


def test_interleaved_order_rejects_bad_input():
    with pytest.raises(ValueError):
        interleaved_order(["a", "a"], rounds=2)
    with pytest.raises(ValueError):
        interleaved_order(["a"], rounds=0)


def test_mean_std_uses_sample_std():
    mean, std = mean_std([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
    assert mean == 5.0
    assert std == pytest.approx(2.13809, abs=1e-5)  # 除以 n-1
    assert mean_std([3.0]) == (3.0, None)


def test_summarize_pairs_by_round():
    # 第 1 轮整体比第 0 轮慢 10%（模拟机器变热），但 b 每一轮都恰好是 a 的 2 倍快。
    records = [
        {"config": "a", "round": 0, "step_ms": 100.0},
        {"config": "b", "round": 0, "step_ms": 50.0},
        {"config": "a", "round": 1, "step_ms": 110.0},
        {"config": "b", "round": 1, "step_ms": 55.0},
        {"config": "b", "round": 2, "step_ms": None},  # OOM 的记录被跳过
    ]
    rows = {row["config"]: row for row in summarize(records)}
    assert rows["a"]["mean"] == 105.0 and rows["a"]["ratio_mean"] == 1.0
    assert rows["b"]["n"] == 2
    assert rows["b"]["ratio_mean"] == pytest.approx(2.0)
    assert rows["b"]["ratio_std"] == pytest.approx(0.0, abs=1e-12)  # 按轮配对后，环境漂移被抵消
    assert "| b | 2 |" in format_markdown(list(rows.values()))


def test_summarize_unknown_baseline():
    with pytest.raises(ValueError):
        summarize([{"config": "a", "round": 0, "step_ms": 1.0}], baseline="missing")


def test_percentile_nearest_rank():
    values = [5.0, 1.0, 4.0, 2.0, 3.0]
    assert percentile(values, 50) == 3.0
    assert percentile(values, 99) == 5.0
    assert percentile([7.0], 50) == 7.0


def test_benchmark_reports_percentiles_cpu():
    config = ModelConfig(context_length=16, **PRESETS["tiny"])
    result = run_benchmark(config, batch_size=2, warmup_steps=1, measure_steps=3, device="cpu")
    assert result.step_ms_p50 <= result.step_ms_p99
    assert result.gpu_before is None and result.gpu_after is None


def test_compare_script_end_to_end_cpu(tmp_path):
    out = tmp_path / "compare.jsonl"
    command = [
        sys.executable,
        str(REPO_ROOT / "experiments" / "compare.py"),
        "--rounds", "2",
        "--out", str(out),
        "--config", "naive=--loss-impl naive",
        "--config", "torch=--loss-impl torch",
        "--",
        "--preset", "tiny", "--context-length", "16", "--batch-size", "2",
        "--warmup-steps", "1", "--measure-steps", "2", "--device", "cpu",
    ]  # fmt: skip
    completed = subprocess.run(command, capture_output=True, text=True, cwd=REPO_ROOT)
    assert completed.returncode == 0, completed.stderr
    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert [(r["round"], r["config"]) for r in records] == [(0, "naive"), (0, "torch"), (1, "torch"), (1, "naive")]
    assert all(r["loss_impl"] == r["config"] for r in records)
    assert "相对基线" in completed.stdout
