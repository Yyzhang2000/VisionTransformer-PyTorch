from dataclasses import dataclass, field


@dataclass
class AttentionConfig:
    num_heads: int = 8
    dropout: float = 0.1
    use_bias: bool = True
    hidden_states: int = 512


@dataclass
class ViTConfig:
    # Image Information
    num_channels = 3
    image_size: int = 224
    num_classes: int = 1000

    # ViT Configuration
    hidden_states: int = 768
    attention_config: AttentionConfig = field(init=False)
    patch_size: int = 8
    num_layers: int = 12
    attention_dropout: float = 0.1
    dropout: float = 0.1
    use_bias: bool = True

    def __post_init__(self):
        self.attention_config = AttentionConfig(
            dropout=self.attention_dropout,
            use_bias=self.use_bias,
            hidden_states=self.hidden_states,
        )
