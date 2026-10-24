"""消融开关：把自实现组件换成 PyTorch 现成实现，一次只换一个，方便归因。

默认值都是 "naive"（课程里自己写的版本），所以不打开开关时行为与原来完全一致。
"""

import torch
import torch.nn.functional as F

from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizers import AdamW

LOSS_IMPLS = ("naive", "torch")
OPTIMIZER_IMPLS = ("naive", "torch_foreach", "torch_fused")
ATTENTION_IMPLS = ("naive", "sdpa")


def get_loss_fn(loss_impl):
    if loss_impl == "naive":
        return cross_entropy
    if loss_impl == "torch":
        # F.cross_entropy 用融合的 log_softmax + nll，不另外物化 exp(logits)
        def torch_cross_entropy(logits, targets):
            return F.cross_entropy(logits.flatten(0, -2), targets.flatten())

        return torch_cross_entropy
    raise ValueError(f"unknown loss_impl: {loss_impl}")


def build_optimizer(optimizer_impl, parameters, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
    if optimizer_impl == "naive":
        return AdamW(parameters, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    # 两者的数学和你的 AdamW 相同（解耦 weight decay），区别只在实现：
    # foreach 把同一操作批量发给所有参数，fused 把整个更新写成一个 CUDA kernel。
    options = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    if optimizer_impl == "torch_foreach":
        return torch.optim.AdamW(parameters, foreach=True, **options)
    if optimizer_impl == "torch_fused":
        return torch.optim.AdamW(parameters, fused=True, **options)
    raise ValueError(f"unknown optimizer_impl: {optimizer_impl}")
