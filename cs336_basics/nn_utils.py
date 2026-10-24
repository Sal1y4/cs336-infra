import torch


def softmax(in_features, dim):
    max_values = in_features.max(dim=dim, keepdim=True).values
    shifted = in_features - max_values

    exp_values = torch.exp(shifted)

    return exp_values / exp_values.sum(dim=dim, keepdim=True)

def cross_entropy(inputs, targets):
    max_values = inputs.max(dim=-1, keepdim=True).values
    shifted = inputs - max_values
    log_sum_exp = torch.log(torch.exp(shifted).sum(dim=-1))
    target_logits = inputs.gather(
        dim=-1,
        index=targets.unsqueeze(-1),
    ).squeeze(-1)
    loss = log_sum_exp - target_logits + max_values.squeeze(-1)

    return loss.mean()

def clip_grad_norm_(parameters, max_l2_norm):
    parameters = [
        parameter
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not parameters:
        return None
    total_norm_squared = sum(
        parameter.grad.pow(2).sum()
        for parameter in parameters
    )
    total_norm = torch.sqrt(total_norm_squared)
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + 1e-6)

        for parameter in parameters:
            parameter.grad.mul_(scale)

    return total_norm
