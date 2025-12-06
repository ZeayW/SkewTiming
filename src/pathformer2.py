import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

# ---------- Base PEs ----------
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
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        return x + self.pe[:L].unsqueeze(0)


# ---------- Stronger Corr PE ----------
class CorrPositionalEncodingStrong(nn.Module):
    """
    Strong, gated correlation-based PE:
      - stats over c_local, c_sink
      - RBF features
      - LayerNorm + gain
    """
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
        elif base == "learned":
            self.base = LearnedPositionalEncoding(d_model)
        elif base == "none":
            self.base = None
        else:
            raise ValueError("base must be 'sinusoidal' | 'learned' | 'none'")

        feat_dim = 0
        if self.base is not None:
            feat_dim += d_model  # base PE channel

        # raw corr features: c_local, c_sink, product, abs diff
        raw_dim = 4
        feat_dim += raw_dim

        if self.n_rbf > 0:
            # RBF for c_local and c_sink means
            feat_dim += 2 * self.n_rbf
            centers = torch.linspace(0, 1, steps=self.n_rbf)
            widths = torch.full((self.n_rbf,), 0.20)
            self.register_buffer("rbf_centers", centers)
            self.register_buffer("rbf_widths", widths)

        self.proj = nn.Linear(feat_dim, d_model)

    def rbf(self, v: torch.Tensor) -> torch.Tensor:
        # v: [B, L] in [0,1]
        x = v.unsqueeze(-1)  # [B,L,1]
        diff = (x - self.rbf_centers) / (self.rbf_widths + 1e-6)
        return torch.exp(-0.5 * diff.pow(2))  # [B,L,K]

    def forward(self,
                x: torch.Tensor,
                c_local: torch.Tensor,
                c_sink: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """
        x: [B,L,D] after input proj (before Transformer)
        c_local, c_sink: [B,L] in [0,1]
        mask: [B,L] bool, True = pad
        """
        B, L, D = x.shape
        feats = []

        if self.base is not None:
            feats.append(self.base(torch.zeros_like(x)))  # [B,L,D]

        c_local = c_local.clamp(0, 1)
        c_sink = c_sink.clamp(0, 1)
        prod = c_local * c_sink
        diff = (c_local - c_sink).abs()
        raw = torch.stack([c_local, c_sink, prod, diff], dim=-1)  # [B,L,4]
        feats.append(raw)

        if self.n_rbf > 0:
            # mean over non-pad positions as a global scalar per token
            c_local_mean = c_local
            c_sink_mean = c_sink
            feats.append(self.rbf(c_local_mean))  # [B,L,K]
            feats.append(self.rbf(c_sink_mean))   # [B,L,K]

        z = torch.cat(feats, dim=-1)        # [B,L,F]
        pe = self.proj(z)                   # [B,L,D]
        pe = self.norm(pe)
        pe = self.gain * self.dropout(pe)

        if mask is not None:
            pe = pe.masked_fill(mask.unsqueeze(-1), 0.0)
        return x + pe


# ---------- Stronger Corr Attention Bias ----------
class CorrAttentionBiasStrong(nn.Module):
    """
    Stronger bias combining:
      - neighbor corr via c_local
      - sink alignment via c_sink
      - correlation floor so low-corr tokens still get some bias
    """
    def __init__(self,
                 alpha_neighbor: float = 0.7,
                 beta_sink: float = 0.7,
                 corr_floor: float = 0.1):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha_neighbor))
        self.beta = nn.Parameter(torch.tensor(beta_sink))
        self.corr_floor = nn.Parameter(torch.tensor(corr_floor))

    def forward(self,
                c_local: torch.Tensor,
                c_sink: torch.Tensor,
                mask: torch.Tensor,
                H: int) -> torch.Tensor:
        """
        c_local, c_sink: [B,L]
        mask: [B,L] (True=pad)
        returns bias: [B,H,L,L]
        """
        B, L = c_local.shape
        device = c_local.device
        idx = torch.arange(L, device=device)

        # Neighbor bias
        neigh = (idx[None, :] - idx[:, None]).abs() == 1  # [L,L]
        neigh = neigh.unsqueeze(0).expand(B, L, L)        # [B,L,L]

        # use c_local of the later index along the edge
        c_edge = torch.zeros(B, L, L, device=device)
        c_edge[:, 1:, :-1] = c_local[:, 1:].unsqueeze(-1)  # (i,j=i-1)
        c_edge[:, :-1, 1:] = c_local[:, 1:].unsqueeze(-2)  # (i,j=i+1)

        neigh_bias = self.alpha * (neigh * c_edge)  # [B,L,L]

        # Sink alignment bias with floor to lift low-corr tokens
        cs = c_sink.clamp(0, 1)
        cs_floor = self.corr_floor + (1.0 - self.corr_floor) * cs  # [B,L]
        sink_prod = cs_floor.unsqueeze(-1) * cs_floor.unsqueeze(-2)  # [B,L,L]
        sink_bias = self.beta * sink_prod

        bias = neigh_bias + sink_bias  # [B,L,L]

        # mask out pads in keys and queries
        if mask is not None:
            pad = mask.unsqueeze(-1)   # [B,L,1]
            bias = bias.masked_fill(pad, 0.0)
            bias = bias.masked_fill(pad.transpose(-1, -2), 0.0)

        # broadcast to heads
        bias = bias.unsqueeze(1).expand(B, H, L, L)  # [B,H,L,L]
        return bias


# ---------- Biased MHA (stable masking) ----------
def safe_mask_scores(scores: torch.Tensor,
                     key_padding_mask: torch.Tensor | None,
                     large_neg: float = 1e4,
                     clamp: float = 20.0) -> torch.Tensor:
    # scores: [B,H,L,L]
    if key_padding_mask is not None:
        B, H, L, _ = scores.shape
        if key_padding_mask.size(0) != B or key_padding_mask.size(1) != L:
            raise RuntimeError(f"key_padding_mask shape {key_padding_mask.shape} incompatible with scores {scores.shape}")
        pad = key_padding_mask.unsqueeze(1).unsqueeze(2).float()  # [B,1,1,L]
        scores = scores - large_neg * pad

    scores = scores - scores.amax(dim=-1, keepdim=True)
    scores = scores.clamp(min=-clamp, max=clamp)
    return scores

class BiasedMultiheadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1,
                 large_neg: float = 1e4, clamp: float = 20.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.large_neg = large_neg
        self.clamp = clamp

    def forward(self,
                x: torch.Tensor,
                key_padding_mask: torch.Tensor | None = None,
                attn_bias: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: [B,L,D]
        key_padding_mask: [B,L] bool
        attn_bias: [B,H,L,L] or None
        """
        B, L, D = x.shape
        H = self.mha.num_heads
        d = D // H

        # standard projections
        W = self.mha.in_proj_weight
        b = self.mha.in_proj_bias
        q = F.linear(x, W[:D, :], b[:D])
        k = F.linear(x, W[D:2*D, :], b[D:2*D])
        v = F.linear(x, W[2*D:, :], b[2*D:])

        q = q.view(B, L, H, d).transpose(1, 2)  # [B,H,L,d]
        k = k.view(B, L, H, d).transpose(1, 2)  # [B,H,L,d]
        v = v.view(B, L, H, d).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / (d ** 0.5)  # [B,H,L,L]

        if attn_bias is not None:
            if attn_bias.shape != scores.shape:
                raise RuntimeError(f"attn_bias must be {scores.shape}, got {attn_bias.shape}")
            scores = scores + attn_bias

        scores = safe_mask_scores(scores, key_padding_mask, self.large_neg, self.clamp)

        attn = scores.softmax(dim=-1)
        attn = F.dropout(attn, p=self.mha.dropout, training=self.training)

        out = attn @ v  # [B,H,L,d]
        out = out.transpose(1, 2).reshape(B, L, D)
        out = self.mha.out_proj(out)
        return out


# ---------- Improved PathTransformerCorr ----------
class PathTransformerW(nn.Module):
    """
    Strengthened version:
      - strong CorrPositionalEncodingStrong
      - strong CorrAttentionBiasStrong
    Inputs:
      x: [B,L,d_in]
      lengths: [B]
      c_local, c_sink: [B,L] in [0,1] or None
    Output:
      path_emb: [B,d_model]
    """
    def __init__(self,
                 d_in: int,
                 d_model: int = 128,
                 n_heads: int = 4,
                 n_layers: int = 3,
                 d_ff: int = 256,
                 dropout: float = 0.1,
                 use_cls_token: bool = True,
                 base_pe: str = "sinusoidal",
                 use_corr_pe: bool = True,
                 use_attn_bias: bool = True):
        super().__init__()
        self.use_cls = use_cls_token
        self.use_corr_pe = use_corr_pe
        self.use_attn_bias = use_attn_bias

        self.input_proj = nn.Linear(d_in, d_model) if d_in != d_model else nn.Identity()

        if self.use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.corr_pe = CorrPositionalEncodingStrong(
            d_model=d_model, base=base_pe,
            use_rbf=True, n_rbf=8, dropout=dropout, gain_init=0.5
        ) if use_corr_pe else None

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict(dict(
                attn=BiasedMultiheadAttention(d_model, n_heads, dropout=dropout),
                ff=nn.Sequential(
                    nn.Linear(d_model, d_ff),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_ff, d_model),
                    nn.Dropout(dropout),
                ),
                ln1=nn.LayerNorm(d_model),
                ln2=nn.LayerNorm(d_model),
            )))

        self.corr_bias = CorrAttentionBiasStrong() if use_attn_bias else None
        self.norm_out = nn.LayerNorm(d_model)
        self.n_heads = n_heads

    def forward(self,
                x: torch.Tensor,
                lengths: torch.Tensor,
                c_local: torch.Tensor | None = None,
                c_sink: torch.Tensor | None = None) -> torch.Tensor:
        B, L, _ = x.shape
        device = x.device

        # key padding mask
        ar = torch.arange(L, device=device).unsqueeze(0)
        mask = ar >= lengths.unsqueeze(1)  # [B,L]

        h = self.input_proj(x)  # [B,L,D]

        if self.use_cls:
            cls = self.cls_token.expand(B, 1, -1)
            h = torch.cat([cls, h], dim=1)
            lengths = lengths + 1
            L = L + 1
            mask = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=device), mask], dim=1)

            if c_local is not None:
                c_local = torch.cat([torch.zeros(B, 1, device=device, dtype=h.dtype), c_local], dim=1)
            if c_sink is not None:
                c_sink = torch.cat([torch.zeros(B, 1, device=device, dtype=h.dtype), c_sink], dim=1)

        # default correlations
        if c_local is None:
            c_local = torch.zeros(B, L, device=device, dtype=h.dtype)
        if c_sink is None:
            c_sink = torch.zeros(B, L, device=device, dtype=h.dtype)

        # strong correlation PE
        if self.use_corr_pe:
            h = self.corr_pe(h, c_local=c_local, c_sink=c_sink, mask=mask)
        else:
            # fallback: simple sinusoidal
            h = SinusoidalPositionalEncoding(h.size(-1))(h)

        # transformer layers
        for layer in self.layers:
            # attention with strong bias
            attn_bias = None
            if self.use_attn_bias:
                attn_bias = self.corr_bias(c_local=c_local, c_sink=c_sink, mask=mask, H=self.n_heads)

            h_res = h
            h = layer.ln1(h)
            h = h_res + layer.attn(h, key_padding_mask=mask, attn_bias=attn_bias)

            ff_res = h
            h = layer.ln2(h)
            h = ff_res + layer.ff(h)

        if self.use_cls:
            path_emb = h[:, 0]
        else:
            valid = (~mask).float().unsqueeze(-1)
            summed = (h * valid).sum(dim=1)
            denom = valid.sum(dim=1).clamp_min(1.0)
            path_emb = summed / denom

        return self.norm_out(path_emb)

def pad_paths(paths: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    B = len(paths)
    d_in = paths[0].size(-1)
    lengths = torch.tensor([p.size(0) for p in paths], dtype=torch.long, device=paths[0].device)
    Lmax = int(lengths.max().item())
    x = torch.zeros(B, Lmax, d_in, dtype=paths[0].dtype, device=paths[0].device)
    for i, p in enumerate(paths):
        x[i, :p.size(0)] = p
    return x, lengths
