import pytest
import torch

from cs336_basics.ablations import build_optimizer, get_loss_fn
from cs336_basics.benchmarking import PRESETS, ModelConfig, run_benchmark
from cs336_basics.model import MultiHeadSelfAttention


def test_sdpa_matches_naive_attention():
    torch.manual_seed(0)
    naive = MultiHeadSelfAttention(64, 4, theta=10000.0)
    sdpa = MultiHeadSelfAttention(64, 4, theta=10000.0, attention_impl="sdpa")
    sdpa.load_state_dict(naive.state_dict())
    x = torch.randn(2, 33, 64, requires_grad=True)

    out_naive = naive(x)
    out_sdpa = sdpa(x)
    torch.testing.assert_close(out_sdpa, out_naive, atol=1e-5, rtol=1e-5)

    grad_naive = torch.autograd.grad(out_naive.sum(), naive.q_proj.weight)[0]
    grad_sdpa = torch.autograd.grad(out_sdpa.sum(), sdpa.q_proj.weight)[0]
    torch.testing.assert_close(grad_sdpa, grad_naive, atol=1e-4, rtol=1e-4)


def test_torch_loss_matches_naive():
    torch.manual_seed(0)
    logits = torch.randn(4, 16, 100) * 5
    targets = torch.randint(0, 100, (4, 16))
    torch.testing.assert_close(get_loss_fn("torch")(logits, targets), get_loss_fn("naive")(logits, targets))


@pytest.mark.parametrize("impl", ["torch_foreach"])
def test_torch_adamw_matches_naive(impl):
    torch.manual_seed(0)
    weights = [torch.nn.Parameter(torch.randn(8, 8)) for _ in range(2)]
    copies = [torch.nn.Parameter(w.detach().clone()) for w in weights]
    ours = build_optimizer("naive", weights, lr=1e-2, weight_decay=0.1)
    theirs = build_optimizer(impl, copies, lr=1e-2, weight_decay=0.1)
    for step in range(5):
        grads = [torch.randn(8, 8) for _ in weights]
        for p, q, g in zip(weights, copies, grads):
            p.grad, q.grad = g.clone(), g.clone()
        ours.step()
        theirs.step()
    for p, q in zip(weights, copies):
        torch.testing.assert_close(p, q, atol=1e-6, rtol=1e-5)


def test_benchmark_with_all_switches_cpu():
    config = ModelConfig(context_length=16, **PRESETS["tiny"])
    result = run_benchmark(
        config,
        batch_size=2,
        warmup_steps=1,
        measure_steps=1,
        device="cpu",
        attention_impl="sdpa",
        loss_impl="torch",
        optimizer_impl="torch_foreach",
    )
    assert not result.oom and result.step_ms > 0
