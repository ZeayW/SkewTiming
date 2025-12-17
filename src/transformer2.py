import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class PathTransformer(nn.Module):
    """
    Inputs:
      x: float tensor [B, L, d_in]   -- padded sequences of node features per path
      lengths: long tensor [B]       -- true lengths (<= L) for masking
    Output:
      path_emb: float tensor [B, d_model] -- one embedding per path
    """
    def __init__(
        self,
        d_in: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
        base_pe:str = "learned",
    ):
        super().__init__()

        self.input_proj = nn.Linear(d_in, d_model) if d_in != d_model else nn.Identity()

        if base_pe == "learned":
            self.pos_encoding = LearnedPositionalEncoding(d_model)
        elif base_pe == "sinusoidal":
            self.pos_encoding = SinusoidalPositionalEncoding(d_model)
        else:
            self.pos_encoding = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape

        h = self.input_proj(x)  # [B, L, d_model]

        if self.pos_encoding is not None:
            h = self.pos_encoding(h)

        # Build key padding mask: True for padded positions
        device = x.device
        arange = torch.arange(L, device=device).unsqueeze(0)  # [1, L]
        mask = arange >= lengths.unsqueeze(1)                 # [B, L], bool
        # Pass through Transformer
        h = self.encoder(h, src_key_padding_mask=mask)  # [B, L, d_model]

        path_emb = h[:, 0]

        return self.norm_out(path_emb)


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        self.pos = nn.Embedding(max_len, d_model)
        nn.init.trunc_normal_(self.pos.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        return x + self.pos(positions)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)  # [max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        return x + self.pe[:L].unsqueeze(0)

def pad_paths(paths: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    paths: list of [Li, d_in] float tensors
    Returns:
      x: [B, Lmax, d_in], lengths: [B]
    """
    B = len(paths)
    d_in = paths[0].size(-1)
    lengths = torch.tensor([p.size(0) for p in paths], dtype=torch.long,device=paths[0].device)
    Lmax = int(lengths.max().item())
    x = torch.zeros(B, Lmax, d_in, dtype=paths[0].dtype, device=paths[0].device)
    for i, p in enumerate(paths):
        x[i, :p.size(0)] = p
    return x, lengths

