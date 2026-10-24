import torch
import math

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                grad = parameter.grad
                state = self.state[parameter]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                state["step"] += 1
                step = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(
                    grad,
                    grad,
                    value=1 - beta2,
                )
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                step_size = lr / bias_correction1
                denominator = (
                    exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)
                ) + eps
                parameter.mul_(1 - lr * weight_decay)
                parameter.addcdiv_(
                    exp_avg,
                    denominator,
                    value=-step_size,
                )
        return loss


def get_lr_cosine_schedule(
    it,                      # 当前训练到第几轮
    max_learning_rate,       # 最大学习率
    min_learning_rate,       # 最小学习率
    warmup_iters,            # 前多少轮做 warmup
    cosine_cycle_iters,      # 到第几轮降到最低学习率
):
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    
    if it > cosine_cycle_iters:
        return min_learning_rate
    
    progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)

    cosine_value = math.cos(math.pi * progress)

    return min_learning_rate + 0.5 * (
        max_learning_rate - min_learning_rate
    ) * (1 + cosine_value)

