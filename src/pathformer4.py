import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        return x + self.pe[:L].unsqueeze(0)


class CorrPositionalEncodingStrong(nn.Module):
    def __init__(self, d_model: int, base: str = "sinusoidal",
                 use_rbf: bool = True, n_rbf: int = 8,
                 dropout: float = 0.1, gain_init: float = 0.5):
        super().__init__()
        self.use_rbf = use_rbf
        self.n_rbf = n_rbf if use_rbf else 0
        self.dropout = nn.Dropout(dropout)
        self.gain = nn.Parameter(torch.tensor(gain_init))
        self.norm = nn.LayerNorm(d_model)

        if base == "sinusoidal":
            self.base = SinusoidalPositionalEncoding(d_model)
        elif base == "none":
            self.base = None
        else:
            raise ValueError("base must be 'sinusoidal' or 'none'")

        feat_dim = 0
        if self.base is not None:
            feat_dim += d_model

        raw_dim = 2
        feat_dim += raw_dim

        if self.n_rbf > 0:
            feat_dim += 2 * self.n_rbf
            centers = torch.linspace(0, 1, steps=self.n_rbf)
            widths = torch.full((self.n_rbf,), 0.20)
            self.register_buffer("rbf_centers", centers)
            self.register_buffer("rbf_widths", widths)

        self.proj = nn.Linear(feat_dim, d_model)

    def rbf(self, v: torch.Tensor) -> torch.Tensor:
        x = v.unsqueeze(-1)
        diff = (x - self.rbf_centers) / (self.rbf_widths + 1e-6)
        return torch.exp(-0.5 * diff.pow(2))

    def forward(self, x, c_local, c_sink, mask):
        B, L, D = x.shape
        feats = []

        if self.base is not None:
            feats.append(self.base(torch.zeros_like(x)))

        c_local = c_local.clamp(0, 1)
        c_sink = c_sink.clamp(0, 1)

        raw = torch.stack([c_local, c_sink], dim=-1)  # [B,L,4]
        feats.append(raw)

        if self.n_rbf > 0:
            feats.append(self.rbf(c_local))
            feats.append(self.rbf(c_sink))

        z = torch.cat(feats, dim=-1)
        pe = self.proj(z)
        pe = self.norm(pe)
        pe = self.gain * self.dropout(pe)
        if mask is not None:
            pe = pe.masked_fill(mask.unsqueeze(-1), 0.0)
        return x + pe


# ----------------- Delay Encoder -----------------

class DelayFeatureEncoder(nn.Module):
    """
    Projects scalar input delay [B] to [B, 1, d_model] and adds to features.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, input_delay: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, D]
        input_delay: [B] or [B, 1]
        """
        if input_delay.dim() == 1:
            input_delay = input_delay.unsqueeze(-1)  # [B, 1]

        # Project delay to embedding dimension
        delay_emb = self.proj(input_delay.float())  # [B, D]
        delay_emb = self.norm(delay_emb).unsqueeze(1)  # [B, 1, D]

        # Add to all tokens in the sequence
        return x + delay_emb


class NeighborCorrBias(nn.Module):
    def __init__(self, alpha: float = 0.7):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))

    def forward(self, c_local, mask, H: int):
        B, L = c_local.shape
        device = c_local.device
        idx = torch.arange(L, device=device)

        neigh = (idx[None, :] - idx[:, None]).abs() == 1  # [L,L]
        neigh = neigh.unsqueeze(0).expand(B, L, L)        # [B,L,L]

        c_edge = torch.zeros(B, L, L, device=device)
        c_edge[:, 1:, :-1] = c_local[:, 1:].unsqueeze(-1)
        c_edge[:, :-1, 1:] = c_local[:, 1:].unsqueeze(-2)

        bias = self.alpha * (neigh * c_edge)  # [B,L,L]

        if mask is not None:
            pad = mask.unsqueeze(-1)
            bias = bias.masked_fill(pad, 0.0)
            bias = bias.masked_fill(pad.transpose(-1, -2), 0.0)

        return bias.unsqueeze(1).expand(B, H, L, L)  # [B,H,L,L]


class SinkNodeAttentionBias(nn.Module):
    """
    Additional bias only for sink token (index 0) attending to nodes.
    """
    def __init__(self, beta: float = 1.0, floor: float = 0.1, gamma: float = 0.7):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(beta))
        self.floor = nn.Parameter(torch.tensor(floor))
        self.gamma = nn.Parameter(torch.tensor(gamma))

    def forward(self, c_sink, mask, H: int):
        """
        c_sink: [B,L]
        mask: [B,L]
        Returns:
          bias_sink: [B,H,L,L], non-zero only at row 0 (sink query)
        """
        B, L = c_sink.shape
        c = c_sink.clamp(0, 1)
        c_remap = torch.pow(c + 1e-6, self.gamma)
        c_remap = self.floor + (1.0 - self.floor) * c_remap
        if mask is not None:
            c_remap = c_remap.masked_fill(mask, 0.0)  # [B,L]

        # base bias for sink query: [B, Lk]
        sink_row = self.beta * c_remap  # [B,L]

        # expand to [B,H,1,L] and place at row 0
        bias = torch.zeros(B, H, L, L, device=c.device)
        bias[:, :, 0, :] = sink_row.unsqueeze(1)  # only query=0 row gets bias
        return bias


def safe_mask_scores(scores, key_padding_mask, large_neg=1e4, clamp=20.0):
    if key_padding_mask is not None:
        B, H, Lq, Lk = scores.shape
        if key_padding_mask.shape != (B, Lk):
            raise RuntimeError(f"mask shape {key_padding_mask.shape} != (B={B},Lk={Lk})")
        pad = key_padding_mask.unsqueeze(1).unsqueeze(2).float()  # [B,1,1,Lk]
        scores = scores - large_neg * pad
    scores = scores - scores.amax(dim=-1, keepdim=True)
    scores = scores.clamp(min=-clamp, max=clamp)
    return scores


class BiasedMultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, large_neg=1e4, clamp=20.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.large_neg = large_neg
        self.clamp = clamp

    def forward(self, x, key_padding_mask=None, attn_bias=None):
        B, L, D = x.shape
        H = self.mha.num_heads
        d = D // H

        W = self.mha.in_proj_weight
        b = self.mha.in_proj_bias
        q = F.linear(x, W[:D, :], b[:D])
        k = F.linear(x, W[D:2*D, :], b[D:2*D])
        v = F.linear(x, W[2*D:, :], b[2*D:])

        q = q.view(B, L, H, d).transpose(1, 2)
        k = k.view(B, L, H, d).transpose(1, 2)
        v = v.view(B, L, H, d).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / (d ** 0.5)  # [B,H,L,L]

        if attn_bias is not None:
            if attn_bias.shape != scores.shape:
                raise RuntimeError(f"attn_bias {attn_bias.shape} != scores {scores.shape}")
            scores = scores + attn_bias

        scores = safe_mask_scores(scores, key_padding_mask, self.large_neg, self.clamp)
        attn = scores.softmax(dim=-1)
        attn = F.dropout(attn, p=self.mha.dropout, training=self.training)

        out = attn @ v
        out = out.transpose(1, 2).reshape(B, L, D)
        out = self.mha.out_proj(out)
        return out

class PathTransformerW(nn.Module):
    """
    One-sink-per-path transformer:
      - token 0 is sink
      - c_local neighbor bias in node-node attention
      - c_sink bias only for sink (token 0) attending to nodes
    """
    def __init__(self,
                 d_in: int,
                 d_model: int = 128,
                 n_heads: int = 4,
                 n_layers: int = 3,
                 d_ff: int = 256,
                 dropout: float = 0.1,
                 alpha: float = 10,
                 beta: float = 10,
                 base_pe: str = "sinusoidal",
                 use_corr_pe: bool = True,
                 use_attn_bias: bool = True):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        self.input_proj = nn.Linear(d_in, d_model) if d_in != d_model else nn.Identity()
        self.use_corr_pe = use_corr_pe
        self.use_corr_bias = use_attn_bias

        self.delay_enc = DelayFeatureEncoder(d_model)

        if use_corr_pe:
            self.corr_pe = CorrPositionalEncodingStrong(
                d_model=d_model, base=base_pe,
                use_rbf=True, n_rbf=8, dropout=dropout, gain_init=0.5
            )
        else:
            self.base_pe = SinusoidalPositionalEncoding(d_model)

        if self.use_corr_bias:
            self.neigh_bias = NeighborCorrBias(alpha=alpha)
            self.sink_bias = SinkNodeAttentionBias(beta=beta, floor=0.1, gamma=0.7)

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict(dict(
                attn=BiasedMultiheadAttention(d_model, n_heads, dropout=dropout),
                ln1=nn.LayerNorm(d_model),
                ln2=nn.LayerNorm(d_model),
                ff=nn.Sequential(
                    nn.Linear(d_model, d_ff),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_ff, d_model),
                    nn.Dropout(dropout),
                ),
            )))

        self.norm_out = nn.LayerNorm(d_model)


    def forward(self, x, lengths, input_delay: torch.Tensor | None = None,c_local: torch.Tensor | None = None, c_sink: torch.Tensor | None = None):
        """
        x: [B,L,d_in], lengths: [B]
        c_local: [B,L] (c_local[:,0]=0), c_sink: [B,L]
        """
        B, L, _ = x.shape
        device = x.device

        ar = torch.arange(L, device=device).unsqueeze(0)
        mask = ar >= lengths.unsqueeze(1)  # [B,L]

        h = self.input_proj(x)
        if input_delay is not None:
            h = self.delay_enc(h, input_delay)

        if c_local is None:
            c_local = torch.zeros(B, L, device=device, dtype=h.dtype)
        if c_sink is None:
            c_sink = torch.zeros(B, L, device=device, dtype=h.dtype)

        if self.use_corr_pe:
            h = self.corr_pe(h, c_local=c_local, c_sink=c_sink, mask=mask)
        else:
            h = self.base_pe(h)

        for layer in self.layers:
            if self.use_corr_bias:
                # neighbor bias (all nodes)
                bias_nn = self.neigh_bias(c_local=c_local, mask=mask, H=self.n_heads)  # [B,H,L,L]
                # sink-only bias (row 0)
                bias_sink = self.sink_bias(c_sink=c_sink, mask=mask, H=self.n_heads)   # [B,H,L,L]
                # total bias: neighbor + sink row
                attn_bias = bias_nn + bias_sink
            else:
                attn_bias = None

            h_res = h
            h = layer.ln1(h)
            h = h_res + layer.attn(h, key_padding_mask=mask, attn_bias=attn_bias)
            h = h + layer.ff(layer.ln2(h))

        sink_emb = self.norm_out(h[:, 0, :])  # [B,D]
        return sink_emb


def pad_paths(paths: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    B = len(paths)
    d_in = paths[0].size(-1)
    lengths = torch.tensor([p.size(0) for p in paths], dtype=torch.long, device=paths[0].device)
    Lmax = int(lengths.max().item())
    x = torch.zeros(B, Lmax, d_in, dtype=paths[0].dtype, device=paths[0].device)
    for i, p in enumerate(paths):
        x[i, :p.size(0)] = p
    return x, lengths
