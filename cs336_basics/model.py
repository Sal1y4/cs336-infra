import torch
from torch import nn


class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(vocab_size, d_model) * 0.02
        )
    
    def forward(self, token_ids):
        return self.weight[token_ids]
    
class Linear(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(d_out, d_in) * 0.02
        )

    def forward(self, in_features):
        return in_features @ self.weight.T
    
def silu(in_features):
    return in_features * torch.sigmoid(in_features)

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()

        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, in_features):
        gate = silu(self.w1(in_features))
        content = self.w3(in_features)
        hidden = gate * content

        return self.w2(hidden)

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]

    scores = Q @ K.transpose(-2, -1)
    scores = scores / (d_k ** 0.5)

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    weights = torch.softmax(scores, dim=-1)

    return weights @ V

def rope(in_query_or_key, token_positions, theta):
    d_k = in_query_or_key.shape[-1]

    assert d_k % 2 == 0

    # 角度和旋转都在 fp32 里算：bf16 只有 8 位有效位，位置 257 起就不能精确表示，
    # 角度误差会达到弧度级。最后再转回输入的 dtype。
    in_dtype = in_query_or_key.dtype
    in_query_or_key = in_query_or_key.float()

    frequency_indices = torch.arange(
        0,
        d_k,
        2,
        device=in_query_or_key.device,
        dtype=torch.float32,
    )

    inv_freq = 1.0 / (theta ** (frequency_indices / d_k))

    angles = token_positions.to(torch.float32).unsqueeze(-1) * inv_freq

    while angles.ndim < in_query_or_key.ndim:
        angles = angles.unsqueeze(-3)

    cos_angles = torch.cos(angles)
    sin_angles = torch.sin(angles)

    even = in_query_or_key[..., 0::2]
    odd = in_query_or_key[..., 1::2]

    rotated_even = even * cos_angles - odd * sin_angles
    rotated_odd = even * sin_angles + odd * cos_angles

    return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2, -1).to(in_dtype)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, theta=None):
        super().__init__()

        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)
        self.theta = theta

    def forward(self, in_features, token_positions=None):
        Q = self.q_proj(in_features)
        K = self.k_proj(in_features)
        V = self.v_proj(in_features)
        
        Q = Q.unflatten(-1, (self.num_heads, self.d_head))
        K = K.unflatten(-1, (self.num_heads, self.d_head))
        V = V.unflatten(-1, (self.num_heads, self.d_head))

        Q = Q.transpose(-3, -2)
        K = K.transpose(-3, -2)
        V = V.transpose(-3, -2)

        sequence_length = in_features.shape[-2]

        if self.theta is not None:
            if token_positions is None:
                token_positions = torch.arange(
                    sequence_length,
                    device=in_features.device,
                )
            Q = rope(Q, token_positions, self.theta)
            K = rope(K, token_positions, self.theta)

        mask = torch.tril(
            torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=in_features.device,
            )
        )

        attention_output = scaled_dot_product_attention(Q, K, V, mask)

        attention_output = attention_output.transpose(-3, -2)
        attention_output = attention_output.flatten(-2, -1)

        return self.output_proj(attention_output)
    
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, in_features):
        # 讲义 3.4.1：先 upcast 到 fp32 再求平方均值，fp16 下 x^2 很容易溢出。
        in_dtype = in_features.dtype
        in_features = in_features.float()

        mean_square = in_features.pow(2).mean(dim=-1, keepdim=True)
        rms = torch.sqrt(mean_square + self.eps)

        return ((in_features / rms) * self.weight).to(in_dtype)
    

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, theta):
        super().__init__()
    
    
        self.ln1 = RMSNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, theta)
        
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, in_features, token_positions=None):
        in_features = in_features + self.attn(
            self.ln1(in_features), 
            token_positions,
        )
        
        return in_features + self.ffn(self.ln2(in_features))


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta):
        super().__init__()

        self.context_length = context_length

        self.token_embeddings = Embedding(vocab_size, d_model)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, rope_theta)
            for _ in range(num_layers)
        ])

        self.ln_final = RMSNorm(d_model)

        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, token_ids):
        x = self.token_embeddings(token_ids)

        for layer in self.layers:
            x = layer(x)

        x = self.ln_final(x)

        return self.lm_head(x)