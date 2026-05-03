import torch
import torch.nn as nn


class SinePositionalEncoding2D(nn.Module):
    """2D sine-cosine positional encoding for spatial feature maps."""

    def __init__(self, d_model: int = 256, temperature: float = 10000.0):
        super().__init__()
        assert d_model % 2 == 0, 'd_model must be divisible by 2'
        self.d_model = d_model
        self.temperature = temperature

    def forward(self, x):
        """Returns positional encoding with same shape as x: (N, d_model, H, W)."""
        N, C, H, W = x.shape
        half_C = C // 2

        dim_t = torch.arange(half_C, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / half_C)

        y_pos = torch.arange(H, dtype=torch.float32, device=x.device).unsqueeze(1).expand(H, W)
        x_pos = torch.arange(W, dtype=torch.float32, device=x.device).unsqueeze(0).expand(H, W)

        pos_y = y_pos.unsqueeze(-1) / dim_t  # (H, W, half_C)
        pos_x = x_pos.unsqueeze(-1) / dim_t  # (H, W, half_C)

        # Interleave sin/cos across even/odd dimension indices
        pos_y = torch.stack([pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()], dim=-1).flatten(-2)
        pos_x = torch.stack([pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()], dim=-1).flatten(-2)

        pos = torch.cat([pos_y, pos_x], dim=-1).permute(2, 0, 1)  # (C, H, W)
        return pos.unsqueeze(0).expand(N, -1, -1, -1)              # (N, C, H, W)


class DETRNeck(nn.Module):
    """Transformer encoder neck with 2D sine positional encoding.

    Adds positional encoding to the flattened feature sequence before passing
    through the encoder.  Returns a (memory, pos_embed) tuple consumed by
    DETRHead:
      memory    – encoder output  (HW, N, d_model)
      pos_embed – source positions (HW, N, d_model) for decoder cross-attention
    """

    def __init__(self,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_encoder_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1):
        super().__init__()

        self.pos_encoding = SinePositionalEncoding2D(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
        )
        # enable_nested_tensor requires batch_first=True; disable to avoid the warning
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers, enable_nested_tensor=False
        )

    def forward(self, x):
        """Forward pass.

        Args:
            x: backbone output (N, d_model, H, W)

        Returns:
            (memory, pos_embed) where both are (HW, N, d_model)
        """
        pos_embed = self.pos_encoding(x)                   # (N, d_model, H, W)
        src = x.flatten(2).permute(2, 0, 1)               # (HW, N, d_model)
        pos_flat = pos_embed.flatten(2).permute(2, 0, 1)  # (HW, N, d_model)
        memory = self.encoder(src + pos_flat)              # (HW, N, d_model)
        return memory, pos_flat


class DETRTransformerEncoder(DETRNeck):
    def __init__(self, args):
        super().__init__(
            d_model=args.d_model,
            nhead=args.nhead,
            num_encoder_layers=args.num_encoder_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
        )
