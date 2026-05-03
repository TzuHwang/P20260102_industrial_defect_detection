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
        """Returns positional encoding with the same shape as x: (N, d_model, H, W)."""
        N, C, H, W = x.shape
        half_C = C // 2

        dim_t = torch.arange(half_C, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / half_C)

        y_pos = torch.arange(H, dtype=torch.float32, device=x.device).unsqueeze(1).expand(H, W)
        x_pos = torch.arange(W, dtype=torch.float32, device=x.device).unsqueeze(0).expand(H, W)

        pos_y = y_pos.unsqueeze(-1) / dim_t  # (H, W, half_C)
        pos_x = x_pos.unsqueeze(-1) / dim_t  # (H, W, half_C)

        pos_y = torch.stack([pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()], dim=-1).flatten(-2)
        pos_x = torch.stack([pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()], dim=-1).flatten(-2)

        pos = torch.cat([pos_y, pos_x], dim=-1).permute(2, 0, 1)  # (C, H, W)
        return pos.unsqueeze(0).expand(N, -1, -1, -1)              # (N, C, H, W)


class PatchEmbed(nn.Module):
    """Image to patch embeddings via strided convolution with layer norm."""

    def __init__(self, in_channels: int = 3, patch_size: int = 16, d_model: int = 256):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """Returns (tokens, Hp, Wp) where tokens is (N, HpWp, d_model)."""
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]
        return self.norm(x.flatten(2).permute(0, 2, 1)), Hp, Wp


class DETRViTBackbone(nn.Module):
    """Vision Transformer backbone for DETR.

    Splits the image into non-overlapping patches, embeds each patch with a
    linear projection, adds 2D sine positional encoding, and encodes the
    resulting sequence with transformer encoder layers.

    Outputs a spatial feature map (N, d_model, H/patch_size, W/patch_size)
    that is compatible with DETRTransformerEncoder neck.
    """

    def __init__(self,
                 patch_size: int = 16,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1):
        super().__init__()

        self.patch_size = patch_size
        self.d_model = d_model

        self.patch_embed = PatchEmbed(3, patch_size, d_model)
        self.pos_encoding = SinePositionalEncoding2D(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """Forward pass.

        Args:
            x: input images (N, 3, H, W); H and W must be divisible by patch_size

        Returns:
            spatial feature map (N, d_model, H/patch_size, W/patch_size)
        """
        N = x.size(0)
        tokens, Hp, Wp = self.patch_embed(x)  # (N, HpWp, d_model)

        # Build positional encoding using a spatial view of the current tokens
        feat_spatial = tokens.permute(0, 2, 1).reshape(N, self.d_model, Hp, Wp)
        pos = self.pos_encoding(feat_spatial)                   # (N, d_model, Hp, Wp)
        pos_flat = pos.flatten(2).permute(0, 2, 1)             # (N, HpWp, d_model)

        out = self.norm(self.encoder(tokens + pos_flat))        # (N, HpWp, d_model)

        # Reshape back to spatial for the neck
        return out.permute(0, 2, 1).reshape(N, self.d_model, Hp, Wp)


class DETRViTSmall(DETRViTBackbone):
    """ViT-Small backbone for DETR."""

    def __init__(self, args):
        super().__init__(
            patch_size=args.patch_size,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
        )


class DETRViTBase(DETRViTBackbone):
    """ViT-Base backbone for DETR."""

    def __init__(self, args):
        super().__init__(
            patch_size=args.patch_size,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
        )
