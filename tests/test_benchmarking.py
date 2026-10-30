from dataclasses import asdict

from cs336_basics.benchmarking import (
    PRESETS,
    ModelConfig,
    TransformerLM,
    count_parameters,
    run_benchmark,
    training_flops_per_token,
)


def test_tinystories_parameter_count():
    config = ModelConfig(context_length=256, **PRESETS["tinystories"])
    # 4 层 x (4*512^2 + 3*512*1344 + 2*512) + 2 * 10000*512 + 512
    assert count_parameters(TransformerLM(**asdict(config))) == 22_696_448


def test_training_flops_per_token():
    config = ModelConfig(context_length=256, **PRESETS["tinystories"])
    matmul_params = 4 * (4 * 512 * 512 + 3 * 512 * 1344) + 512 * 10000
    attention = 4 * 4 * 256 * 512
    assert training_flops_per_token(config) == 3 * (2 * matmul_params + attention)


def test_run_benchmark_cpu_smoke():
    config = ModelConfig(context_length=16, **PRESETS["tiny"])
    for mode in ("forward", "forward_backward", "train"):
        result = run_benchmark(config, batch_size=2, mode=mode, warmup_steps=1, measure_steps=2, device="cpu")
        assert not result.oom
        assert result.step_ms > 0
        assert result.tokens_per_second > 0
        assert (result.backward_ms is None) == (mode == "forward")


def test_run_benchmark_bf16_autocast_cpu():
    config = ModelConfig(context_length=16, **PRESETS["tiny"])
    result = run_benchmark(config, batch_size=2, dtype="bf16", warmup_steps=1, measure_steps=1, device="cpu")
    assert not result.oom and result.achieved_tflops > 0
