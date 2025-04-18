import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ViTConfig
from .attention import MultiHeadAttention


class PatchEmbedding(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()

        self.hidden_states = config.hidden_states
        self.patch_size = config.patch_size
        self.num_channels = config.num_channels

        self.linear_projection = nn.Conv2d(
            in_channels=self.num_channels,
            out_channels=self.hidden_states,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, x: torch.Tensor):
        x = self.linear_projection(x)

        x = x.flatten(2).transpose(1, 2).contiguous()  # (B, C, H, W) -> (B, H*W, C)

        return x


class PositionalEncoding(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()

        self.hidden_states = config.hidden_states
        self.patch_size = config.patch_size
        self.image_size = config.image_size

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.hidden_states))
        self.num_patches = (self.image_size // self.patch_size) ** 2 + 1

        pe = torch.zeros(self.num_patches, self.hidden_states)
        position = torch.arange(0, self.num_patches, dtype=torch.float).unsqueeze(
            1
        )  # (1, N)

        div_term = torch.exp(
            torch.arange(0, self.hidden_states, 2).float()
            * -(torch.log(torch.tensor(10000.0)) / self.hidden_states)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # (1, N, D)

    def forward(self, x: torch.Tensor):
        tokens_batch = self.cls_token.expand(x.size(0), -1, -1)

        x = torch.cat((tokens_batch, x), dim=1)

        x += self.pe

        return x


class EncoderBlock(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()

        self.hidden_state = config.hidden_states

        self.ln1 = nn.LayerNorm(self.hidden_state)
        self.attention = MultiHeadAttention(config.attention_config)

        self.ln2 = nn.LayerNorm(self.hidden_state)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_state, self.hidden_state * 4),
            nn.GELU(),
            nn.Linear(self.hidden_state * 4, self.hidden_state),
        )

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x


class MLPHead(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()

        self.hidden_state = config.hidden_states
        self.num_classes = config.num_classes

        self.head = nn.Sequential(
            nn.LayerNorm(self.hidden_state),
            nn.Linear(self.hidden_state, self.hidden_state),
            nn.GELU(),
            nn.Linear(self.hidden_state, self.num_classes),
        )

    def forward(self, x):
        x = x[:, 0]  # Extract the CLS token
        x = self.head(x)
        return x


class ViT(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()

        self.patch_embedding = PatchEmbedding(config)
        self.positional_encoding = PositionalEncoding(config)

        self.encoder_blocks = nn.ModuleList(
            [EncoderBlock(config) for _ in range(config.num_layers)]
        )

        self.mlp_head = MLPHead(config)

    def forward(self, x: torch.Tensor):
        x = self.patch_embedding(x)
        x = self.positional_encoding(x)

        for block in self.encoder_blocks:
            x = block(x)

        x = self.mlp_head(x)

        return x


if __name__ == "__main__":
    config = ViTConfig()
    model = ViT(config)

    # Dummy input
    B = 8
    x = torch.randn(
        B, config.num_channels, config.image_size, config.image_size
    )  # (batch_size, channels, height, width)
    output = model(x)

    assert output.shape == (B, config.num_classes), "Output shape mismatch"
    print("Output shape:", output.shape)  # Should be (batch_size, num_classes)
