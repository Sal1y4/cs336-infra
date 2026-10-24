import torch

from cs336_basics.model import MultiHeadSelfAttention, RMSNorm, rope


def test_rope_bf16_matches_fp32_reference():
    torch.manual_seed(0)
    x = torch.randn(1, 16, 2048, 32)
    positions = torch.arange(2048)
    reference = rope(x, positions, 10000.0)

    output = rope(x.to(torch.bfloat16), positions, 10000.0)
    assert output.dtype == torch.bfloat16

    # 误差应该只来自 bf16 的输入/输出舍入，而不是位置和角度的精度。
    # 修复前这个误差在 T=2048 时约为 8。
    error = (output.float() - reference).abs().max().item()
    assert error < 0.05, error


def test_rope_fp32_unchanged_dtype():
    x = torch.randn(2, 4, 8, 16)
    assert rope(x, torch.arange(8), 10000.0).dtype == torch.float32


def test_rmsnorm_fp16_large_inputs_do_not_overflow():
    norm = RMSNorm(64)
    x = torch.full((2, 3, 64), 300.0)  # 300^2 = 90000 超出 fp16 上限 65504

    output = norm(x.to(torch.float16))
    assert output.dtype == torch.float16
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output.float(), norm(x), atol=1e-3, rtol=1e-3)


def test_attention_under_bf16_autocast_stays_close_to_fp32():
    torch.manual_seed(0)
    attention = MultiHeadSelfAttention(512, 16, theta=10000.0)
    x = torch.randn(1, 2048, 512)
    reference = attention(x)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = attention(x)
    # 修复前相对误差随序列长度增长（T=256/512/2048 约 0.007/0.018/0.048），
    # 修复后与长度无关（约 0.004）。
    relative = (output.float() - reference).norm() / reference.norm()
    assert relative < 0.01, relative.item()
