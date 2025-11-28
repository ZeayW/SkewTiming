import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

class PathTransformer(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
        use_cls_token: bool = True,
        base_pe: str = "learned",   #  'none' | 'learned' | 'sinusoidal'
        use_corr_pe: bool = True,      # NEW: toggle additive correlation PE
        use_rbf: bool = True,
        n_rbf: int = 8,
        use_attn_bias: bool = True,
        pos_dropout: float = 0.0,
        alpha=0,
        beta=1,
        norm_first: bool = True
    ):
        super().__init__()
        self.use_cls = use_cls_token
        self.use_attn_bias = use_attn_bias
        self.use_corr_pe = use_corr_pe
        self.base_pe_name = base_pe

        self.input_proj = nn.Linear(d_in, d_model) if d_in != d_model else nn.Identity()

        if self.use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.alpha = alpha
        self.beta = beta
        # self.alpha = nn.Parameter(torch.tensor(alpha))
        # self.beta = nn.Parameter(torch.tensor(beta))



        # If Corr PE is enabled, construct it; otherwise optional base PE only
        if use_corr_pe:
            self.corr_pe = CorrPositionalEncoding(
                d_model=d_model, alpha=self.alpha,beta=self.beta,base=base_pe, use_rbf=use_rbf, n_rbf=n_rbf, dropout=pos_dropout
            )
            self.base_pe = None  # handled inside corr_pe
        else:
            if base_pe == "sinusoidal":
                self.base_pe = SinusoidalPositionalEncoding(d_model)
            elif base_pe == "learned":
                self.base_pe = LearnedPositionalEncoding(d_model)
            elif base_pe == "none":
                self.base_pe = None
            else:
                raise ValueError("base_pe must be 'sinusoidal' | 'learned' | 'none'")

        encoder_layers = []
        for _ in range(n_layers):
            attn = BiasedMultiheadAttention(d_model, n_heads, dropout=dropout)
            ff = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout),
            )
            norm1 = nn.LayerNorm(d_model)
            norm2 = nn.LayerNorm(d_model)
            encoder_layers.append(nn.ModuleDict(dict(attn=attn, ff=ff, norm1=norm1, norm2=norm2)))
        self.layers = nn.ModuleList(encoder_layers)

        if use_attn_bias:
            self.corr_bias = CorrAttentionBias(alpha=self.alpha,beta=self.beta)

        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor,
                c_local: torch.Tensor = None, c_sink: torch.Tensor = None) -> torch.Tensor:
        B, L, _ = x.shape
        device = x.device

        ar = torch.arange(L, device=device).unsqueeze(0)
        mask = ar >= lengths.unsqueeze(1)  # [B, L]

        h = self.input_proj(x)

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

        # Default correlations if not provided
        if c_local is None:
            c_local = torch.zeros(B, L, device=device, dtype=h.dtype)
        if c_sink is None:
            c_sink = torch.zeros(B, L, device=device, dtype=h.dtype)

        # Apply positional signals
        if self.use_corr_pe:
            h = self.corr_pe(h, c_local=c_local, c_sink=c_sink, mask=mask)
        else:
            if self.base_pe is not None:
                h = self.base_pe(h)
        # Transformer layers
        for layer in self.layers:
            residual = h
            h = layer.norm1(h)
            attn_bias = None
            if self.use_attn_bias:
                attn_bias = self.corr_bias(
                    attn_scores=torch.zeros(h.size(0), layer.attn.mha.num_heads, h.size(1), h.size(1), device=h.device),
                    c_local=c_local, c_sink=c_sink, mask=mask
                )
            h_attn = layer.attn(h, key_padding_mask=mask, attn_bias=attn_bias)
            h = residual + h_attn

            residual = h
            h = layer.norm2(h)
            h_ff = layer.ff(h)
            h = residual + h_ff

        if self.use_cls:
            path_emb = h[:, 0]
        else:
            valid = (~mask).float().unsqueeze(-1)
            summed = (h * valid).sum(dim=1)
            denom = valid.sum(dim=1).clamp_min(1.0)
            path_emb = summed / denom

        return self.norm_out(path_emb)

# --------- Base PEs from previous answer ---------
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

# --------- Correlation-aware PE (additive) ---------
class CorrPositionalEncoding(nn.Module):
    """
    Add correlation-aware features onto token embeddings.
    Inputs to forward:
      base_x: [B, L, D] token embeddings (after input proj, before Transformer)
      c_local: [B, L] in [0,1], c_local[:,0]=0 for start tokens
      c_sink:  [B, L] in [0,1]
      mask:    [B, L] bool, True if padded
    """
    def __init__(self, d_model: int, alpha, beta,base: str = "sinusoidal",
                 use_rbf: bool = True, n_rbf: int = 8, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.use_rbf = use_rbf
        self.n_rbf = n_rbf
        self.alpha = alpha
        self.beta = beta
        # self.alpha = nn.Parameter(torch.tensor(alpha))
        # self.beta = nn.Parameter(torch.tensor(beta))

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
            feat_dim += d_model
        raw_dim = 3  # c_local, c_sink, c_local*c_sink
        rbf_dim = 2 * n_rbf if use_rbf else 0
        self.proj = nn.Linear(feat_dim + raw_dim + rbf_dim, d_model)

        if use_rbf:
            centers = torch.linspace(0, 1, steps=n_rbf)
            widths = torch.full((n_rbf,), 0.15)
            self.register_buffer("rbf_centers", centers)
            self.register_buffer("rbf_widths", widths)

    def rbf(self, c: torch.Tensor) -> torch.Tensor:
        # c: [B, L]
        x = c.unsqueeze(-1)  # [B, L, 1]
        diff = (x - self.rbf_centers) / (self.rbf_widths + 1e-6)
        return torch.exp(-0.5 * diff.pow(2))  # [B, L, n_rbf]

    def forward(self, base_x: torch.Tensor, c_local: torch.Tensor,
                c_sink: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, L, D = base_x.shape
        feats = []

        if self.base is not None:
            # Feed zeros to get pure base PE signal, independent of base_x magnitude
            feats.append(self.base(torch.zeros_like(base_x)))

        c_local = self.alpha*c_local.clamp(0, 1)
        c_sink = self.beta*c_sink.clamp(0, 1)
        inter = c_local * c_sink
        raw = torch.stack([c_local, c_sink, inter], dim=-1)  # [B, L, 3]
        feats.append(raw)

        if self.use_rbf:
            feats.append(self.rbf(c_local))  # [B, L, n_rbf]
            feats.append(self.rbf(c_sink))   # [B, L, n_rbf]

        z = torch.cat(feats, dim=-1)  # [B, L, F]
        pe = self.proj(z)             # [B, L, D]
        pe = self.dropout(pe)

        if mask is not None:
            pe = pe.masked_fill(mask.unsqueeze(-1), 0.0)
        return base_x + pe

def apply_attn_mask(scores: torch.Tensor, key_padding_mask: torch.Tensor, clamp: float = 20.0):
    # scores: [B,H,L,L], key_padding_mask: [B,L] (True=pad)
    # mask columns (keys) only
    pad = key_padding_mask.unsqueeze(1).unsqueeze(2)            # [B,1,1,L]
    scores = scores.masked_fill(pad, float("-inf"))

    # If an entire row is -inf, replace with zeros to make softmax well-defined
    all_inf = torch.isinf(scores).all(dim=-1, keepdim=True)     # [B,H,L,1]
    scores = torch.where(all_inf, torch.zeros_like(scores), scores)

    # Optional: clamp to avoid overflow
    #scores = scores.clamp(min=-clamp, max=clamp)

    return scores

# --------- Attention bias from correlations ---------
class CorrAttentionBias(nn.Module):
    """
    Adds correlation-based bias to attention logits.
    Use by injecting into a custom MultiheadAttention wrapper.
    """
    def __init__(self, alpha, beta):
        super().__init__()
        self.alpha  = alpha
        self.beta = beta
        # self.alpha = nn.Parameter(torch.tensor(alpha))
        # self.beta = nn.Parameter(torch.tensor(beta))

    def forward(self, attn_scores: torch.Tensor, c_local: torch.Tensor,
                c_sink: torch.Tensor, mask: torch.Tensor):
        # attn_scores: [B, H, L, L] pre-softmax
        B, H, L, _ = attn_scores.shape
        device = attn_scores.device
        idx = torch.arange(L, device=device)
        neigh = (idx[None, :] - idx[:, None]).abs() == 1  # [L, L]
        neigh = neigh.unsqueeze(0).unsqueeze(0).expand(B, H, L, L)

        c_edge = torch.zeros(B, L, L, device=device)
        # map local corr to (i,j=i-1) and (i,j=i+1)
        # using c_local at later index along the path
        c_edge[:, 1:, :-1] = c_local[:, 1:].unsqueeze(-1)
        c_edge[:, :-1, 1:] = c_local[:, 1:].unsqueeze(-2)
        c_edge = c_edge.unsqueeze(1).expand(B, H, L, L)

        attn_scores = attn_scores + self.alpha * (neigh * c_edge).float()

        sink_prod = (c_sink.unsqueeze(-1) * c_sink.unsqueeze(-2))  # [B, L, L]
        attn_scores = attn_scores + self.beta * sink_prod.unsqueeze(1)

        if mask is not None:
            pad = mask.unsqueeze(1).unsqueeze(2)  # key padding on columns
            attn_scores = attn_scores.masked_fill(pad, -100000)
            attn_scores = attn_scores.masked_fill(pad.transpose(-1, -2), -100000)
        return attn_scores

# --------- MHA with external bias ---------
class BiasedMultiheadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

    def forward(self, x, key_padding_mask=None, attn_bias=None):
        # x: [B, L, D], attn_bias: [B, H, L, L] to add pre-softmax
        # nn.MultiheadAttention does not natively accept biases, so we emulate via hooks.
        # We compute attention with custom bias by bypassing MHA internals using functional call.
        # Simpler approach: use attn_mask as [B*H*L, L], but PyTorch's attn_mask merges heads.
        # Here we fall back to a manual scaled dot-product attention per head.

        B, L, D = x.shape
        H = self.mha.num_heads
        head_dim = D // H
        qkv_proj = self.mha.in_proj_weight
        in_bias = self.mha.in_proj_bias
        Wq, Wk, Wv = qkv_proj[:D, :], qkv_proj[D:2*D, :], qkv_proj[2*D:, :]
        bq, bk, bv = in_bias[:D], in_bias[D:2*D], in_bias[2*D:]

        q = torch.addmm(self.mha.in_proj_bias[:D], x.reshape(-1, D), Wq.t()).reshape(B, L, D)
        k = torch.addmm(self.mha.in_proj_bias[D:2*D], x.reshape(-1, D), Wk.t()).reshape(B, L, D)
        v = torch.addmm(self.mha.in_proj_bias[2*D:], x.reshape(-1, D), Wv.t()).reshape(B, L, D)

        q = q.view(B, L, H, head_dim).transpose(1, 2)  # [B,H,L,d]
        k = k.view(B, L, H, head_dim).transpose(1, 2)
        v = v.view(B, L, H, head_dim).transpose(1, 2)

        scale = 1.0 / (head_dim ** 0.5)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B,H,L,L]

        if attn_bias is not None:
            attn_scores = attn_scores + attn_bias  # [B,H,L,L]

        if key_padding_mask is not None:
            pad = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [B,1,1,L]
            attn_scores = attn_scores.masked_fill(pad, -100000)

        attn_weights = attn_scores.softmax(dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.mha.dropout, training=self.training)

        out = torch.matmul(attn_weights, v)  # [B,H,L,d]
        out = out.transpose(1, 2).reshape(B, L, D)
        out = self.mha.out_proj(out)
        return out


def pad_paths(paths: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    B = len(paths)
    d_in = paths[0].size(-1)
    lengths = torch.tensor([p.size(0) for p in paths], dtype=torch.long, device=paths[0].device)
    Lmax = int(lengths.max().item())
    x = torch.zeros(B, Lmax, d_in, dtype=paths[0].dtype, device=paths[0].device)
    for i, p in enumerate(paths):
        x[i, :p.size(0)] = p
    return x, lengths


