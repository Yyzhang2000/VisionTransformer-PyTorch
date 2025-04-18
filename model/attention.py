import torch
import torch.nn as nn
import torch.nn.functional as F

from config import AttentionConfig


class MultiHeadAttention(nn.Module):
    def __init__(self, config: AttentionConfig):
        super().__init__()

        self.num_heads = config.num_heads
        self.hidden_states = config.hidden_states

        assert (
            self.hidden_states % self.num_heads == 0
        ), "hidden_states must be divisible by num_heads"

        self.head_dim = self.hidden_states // self.num_heads

        self.qkv_proj = nn.Linear(
            self.hidden_states, 3 * self.hidden_states, bias=config.use_bias
        )
        self.out_proj = nn.Linear(
            self.hidden_states, self.hidden_states, bias=config.use_bias
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor):
        B, S, D = x.shape

        q, k, v = map(
            lambda x: x.view(B, S, self.num_heads, self.head_dim).transpose(1, 2),
            self.qkv_proj(x).chunk(3, dim=-1),
        )

        score = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)

        attn = F.softmax(score, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, S, D)
        out = self.out_proj(out)

        return out
