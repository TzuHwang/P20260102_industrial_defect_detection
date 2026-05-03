import torch
import torch.nn as nn


class MLP(nn.Module):
    """Multi-layer perceptron with ReLU activations on hidden layers."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers)])

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = torch.relu(layer(x)) if i < len(self.layers) - 1 else layer(x)
        return x


class DETRDecoderLayer(nn.Module):
    """Single DETR decoder layer: self-attention → cross-attention → FFN.

    Positional encodings are added to Q and K at each attention step,
    matching the original DETR formulation.
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.drop3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, query_pos, pos_embed):
        """
        Args:
            tgt:       decoder queries       (num_queries, N, d_model)
            memory:    encoder output        (HW, N, d_model)
            query_pos: query pos embedding   (num_queries, N, d_model)
            pos_embed: source pos embedding  (HW, N, d_model)
        """
        # Self-attention with query positional encoding added to Q and K
        q = k = tgt + query_pos
        tgt = self.norm1(tgt + self.drop1(self.self_attn(q, k, tgt)[0]))

        # Cross-attention: Q = tgt + query_pos, K = memory + pos_embed, V = memory
        q = tgt + query_pos
        tgt = self.norm2(tgt + self.drop2(self.cross_attn(q, memory + pos_embed, memory)[0]))

        # Feed-forward
        tgt = self.norm3(tgt + self.drop3(self.ffn(tgt)))
        return tgt


class DETRHead(nn.Module):
    """DETR detection head: transformer decoder + classification and box FFNs.

    Consumes the (memory, pos_embed) tuple from DETRNeck and produces per-query
    class logits and normalised box predictions.
    """

    def __init__(self,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_decoder_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 num_queries: int = 100,
                 num_classes: int = 91):
        super().__init__()

        self.query_embed = nn.Embedding(num_queries, d_model)
        self.query_pos_embed = nn.Embedding(num_queries, d_model)

        self.decoder_layers = nn.ModuleList([
            DETRDecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_decoder_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # Prediction heads
        self.class_embed = nn.Linear(d_model, num_classes + 1)
        self.bbox_embed = MLP(d_model, d_model, 4, num_layers=3)

    def forward(self, x):
        """Forward pass.

        Args:
            x: (memory, pos_embed) tuple from DETRNeck, each (HW, N, d_model)

        Returns:
            dict with keys:
              'pred_logits': (N, num_queries, num_classes + 1)
              'pred_boxes':  (N, num_queries, 4) – normalised [cx, cy, w, h] in [0, 1]
        """
        memory, pos_embed = x
        N = memory.size(1)

        # Expand learned embeddings to batch dimension
        tgt = self.query_embed.weight.unsqueeze(1).expand(-1, N, -1)       # (Q, N, d)
        query_pos = self.query_pos_embed.weight.unsqueeze(1).expand(-1, N, -1)

        for layer in self.decoder_layers:
            tgt = layer(tgt, memory, query_pos, pos_embed)

        tgt = self.norm(tgt).permute(1, 0, 2)  # (N, Q, d_model)

        return {
            'pred_logits': self.class_embed(tgt),           # (N, Q, num_classes + 1)
            'pred_boxes': self.bbox_embed(tgt).sigmoid(),   # (N, Q, 4) ∈ [0, 1]
        }


class DETRTransformerDecoder(DETRHead):
    def __init__(self, args):
        super().__init__(
            d_model=args.d_model,
            nhead=args.nhead,
            num_decoder_layers=args.num_decoder_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            num_queries=args.num_queries,
            num_classes=args.num_classes,
        )
