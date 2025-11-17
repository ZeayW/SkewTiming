import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================
# Correlation-aware node encoding (single graph)
# =============================
class CorrNodeEncoding1G(nn.Module):
    """
    Additive correlation-aware encoding per node using stats over sinks.
    Inputs:
      x: [N, D] projected node features
      C: [N, S] correlations in [0,1]
      mask_nodes: [N] bool (True=pad), optional
    """
    def __init__(self, d_model: int, use_rbf: bool = True, n_rbf: int = 8, dropout: float = 0.1, gate_init: float = 0.1):
        super().__init__()
        self.use_rbf = use_rbf
        self.n_rbf = n_rbf if use_rbf else 0
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Parameter(torch.tensor(gate_init))
        self.norm = nn.LayerNorm(d_model)

        feat_dim = 4  # mean, max, min, std across sinks
        if self.n_rbf > 0:
            feat_dim += 2 * self.n_rbf  # RBF for mean and max
            centers = torch.linspace(0, 1, steps=self.n_rbf)
            widths = torch.full((self.n_rbf,), 0.15)
            self.register_buffer("rbf_centers", centers)
            self.register_buffer("rbf_widths", widths)

        self.proj = nn.Linear(feat_dim, d_model)

    def rbf(self, v: torch.Tensor) -> torch.Tensor:
        # v: [N] in [0,1]
        x = v.unsqueeze(-1)  # [N,1]
        diff = (x - self.rbf_centers) / (self.rbf_widths + 1e-6)
        return torch.exp(-0.5 * diff.pow(2))  # [N,K]

    def forward(self, x: torch.Tensor, C: torch.Tensor, mask_nodes: torch.Tensor | None = None) -> torch.Tensor:
        C = C.clamp(0, 1)                 # [N,S]
        C_mean = C.mean(dim=-1)           # [N]
        C_max, _ = C.max(dim=-1)          # [N]
        C_min, _ = C.min(dim=-1)          # [N]
        C_std = C.std(dim=-1, unbiased=False)  # [N]

        feats = [C_mean.unsqueeze(-1), C_max.unsqueeze(-1), C_min.unsqueeze(-1), C_std.unsqueeze(-1)]
        if self.n_rbf > 0:
            feats.append(self.rbf(C_mean))  # [N,K]
            feats.append(self.rbf(C_max))   # [N,K]

        z = torch.cat(feats, dim=-1)  # [N,F]
        pe = self.proj(z)             # [N,D]
        pe = self.norm(pe)
        pe = self.gate * self.dropout(pe)

        if mask_nodes is not None:
            pe = pe.masked_fill(mask_nodes.unsqueeze(-1), 0.0)
        return x + pe


# =============================
# Correlation-based attention biases (single graph)
# =============================
class CorrAttentionBiasGraph1G(nn.Module):
    """
    Builds attention biases:
      - node-node bias: [N, N] using similarity of sink correlation profiles
      - sink-to-node bias: [S, N] using columns of C (each sink's correlation over all nodes)
    Assumption: C's columns are ordered to match sink_idx order.
    """
    def __init__(self, alpha_nn: float = 0.3, beta_sn: float = 0.8):
        super().__init__()
        self.alpha_nn = nn.Parameter(torch.tensor(alpha_nn))
        self.beta_sn = nn.Parameter(torch.tensor(beta_sn))

    def node_node_bias(self, C: torch.Tensor) -> torch.Tensor:
        # C: [N,S] -> normalized similarity in sink space -> [N,N]
        eps = 1e-6
        Cn = C / (C.norm(dim=-1, keepdim=True) + eps)
        sim = Cn @ Cn.t()  # [N,N]
        return self.alpha_nn * sim

    def sink_node_bias(self, C: torch.Tensor) -> torch.Tensor:
        """
        Return [S, N], row s is correlation of sink s to every node.
        Given C is [N, S] (node x sink), C.t() is [S, N].
        Assumes C's columns match sink_idx order externally.
        """
        return self.beta_sn * C.t()  # [S, N]


# =============================
# Safe masking helper (single graph)
# =============================
def safe_mask_scores(scores: torch.Tensor, key_padding_mask: torch.Tensor | None, large_neg: float = 1e4, clamp: float = 20.0) -> torch.Tensor:
    """
    scores: [1, H, Lq, Lk]
    key_padding_mask: [Lk] (True=pad)
    Applies column-only masking with large negative (no -inf), stabilizes softmax.
    Includes assertions to prevent shape explosions.
    """
    # Basic sanity checks to catch unintended broadcasts early
    if scores.dim() != 4:
        raise RuntimeError(f"scores must be 4D [1,H,Lq,Lk], got {tuple(scores.shape)}")
    if scores.size(0) != 1:
        raise RuntimeError(f"Batch must be 1 in single-graph mode, got B={scores.size(0)}")
    _, _, Lq, Lk = scores.shape

    if key_padding_mask is not None:
        if key_padding_mask.dim() != 1 or key_padding_mask.size(0) != Lk:
            raise RuntimeError(f"key_padding_mask must be [Lk={Lk}], got {tuple(key_padding_mask.shape)}")
        pad = key_padding_mask.view(1, 1, 1, Lk).to(dtype=scores.dtype)  # [1,1,1,Lk]
        scores = scores - large_neg * pad

    # Row-wise stabilization and clamp
    scores = scores - scores.amax(dim=-1, keepdim=True)
    return scores.clamp(min=-clamp, max=clamp)


# =============================
# Biased multihead attention with per-query key bias (single graph)
# =============================
class BiasedMHA_Bipartite1G(nn.Module):
    """
    Q: [Lq, D], K,V: [Lk, D]
    per_query_key_bias: [Lq, Lk], added pre-softmax per head.
    key_padding_mask: [Lk] (True=pad), applied to columns only via large negative.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, large_neg: float = 1e4, clamp: float = 20.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.large_neg = large_neg
        self.clamp = clamp

    def forward(self,
                Q: torch.Tensor,
                K: torch.Tensor,
                V: torch.Tensor,
                key_padding_mask: torch.Tensor | None = None,
                per_query_key_bias: torch.Tensor | None = None) -> torch.Tensor:
        # Lift single graph to batch=1
        Q = Q.unsqueeze(0)  # [1,Lq,D]
        K = K.unsqueeze(0)  # [1,Lk,D]
        V = V.unsqueeze(0)  # [1,Lk,D]
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(0)  # [1,Lk]

        B, Lq, D = Q.shape
        Lk = K.size(1)
        H = self.mha.num_heads
        d = D // H

        # In-projections
        W = self.mha.in_proj_weight
        b = self.mha.in_proj_bias
        q = F.linear(Q, W[:D, :], b[:D])             # [1,Lq,D]
        k = F.linear(K, W[D:2*D, :], b[D:2*D])       # [1,Lk,D]
        v = F.linear(V, W[2*D:, :], b[2*D:])         # [1,Lk,D]

        # Split heads
        q = q.view(B, Lq, H, d).transpose(1, 2)      # [1,H,Lq,d]
        k = k.view(B, Lk, H, d).transpose(1, 2)      # [1,H,Lk,d]
        v = v.view(B, Lk, H, d).transpose(1, 2)      # [1,H,Lk,d]

        # Scaled dot-product scores
        scores = (q @ k.transpose(-2, -1)) / (d ** 0.5)  # [1,H,Lq,Lk]

        # Add per-query bias with strict shape checking
        if per_query_key_bias is not None:
            if not (per_query_key_bias.dim() == 2 and per_query_key_bias.size(0) == Lq and per_query_key_bias.size(1) == Lk):
                raise RuntimeError(f"per_query_key_bias must be [Lq={Lq}, Lk={Lk}], got {tuple(per_query_key_bias.shape)}")
            scores = scores + per_query_key_bias.view(1, 1, Lq, Lk)  # broadcast to heads, no repeat

        # Key mask (columns only) with large negative sentinel and stabilization
        scores = safe_mask_scores(scores, key_padding_mask.squeeze(0) if key_padding_mask is not None else None,
                                  large_neg=self.large_neg, clamp=self.clamp)

        # Softmax and dropout
        attn = scores.softmax(dim=-1)
        attn = F.dropout(attn, p=self.mha.dropout, training=self.training)

        # Weighted sum
        out = attn @ v                                  # [1,H,Lq,d]
        out = out.transpose(1, 2).reshape(B, Lq, D)     # [1,Lq,D]
        out = self.mha.out_proj(out)                    # [1,Lq,D]
        return out.squeeze(0)                           # [Lq,D]



# =============================
# Graph transformer (single graph, sink_idx authoritative)
# =============================
class PathGraphFormer(nn.Module):
    """
    Single-graph correlation-aware transformer.
    Inputs:
      x: [N, d_in] node features
      C: [N, S] node-to-sink correlations (columns aligned with sink_idx)
      sink_idx: [S] long tensor of node indices designating sinks (required)
      mask_nodes: [N] bool (True=pad), optional
    Output:
      sink_emb: [S, D] embeddings of the sink nodes (order matches sink_idx)
    """
    def __init__(self,
                 d_in: int,
                 d_model: int = 128,
                 n_heads: int = 4,
                 n_layers: int = 3,
                 d_ff: int = 256,
                 dropout: float = 0.1,
                 use_corr_pe: bool = True,
                 use_corr_bias: bool = True,
                 use_sink_query: bool = True):
        super().__init__()
        self.use_corr_pe = use_corr_pe
        self.use_corr_bias = use_corr_bias
        self.use_sink_query = use_sink_query

        self.proj_in = nn.Linear(d_in, d_model) if d_in != d_model else nn.Identity()
        self.node_corr_pe = CorrNodeEncoding1G(d_model) if use_corr_pe else None
        self.bias_builder = CorrAttentionBiasGraph1G() if use_corr_bias else None

        if use_sink_query:
            self.sink_query_proj = nn.Linear(d_model, d_model)

        # Attention stacks
        self.sn_attn = nn.ModuleList([BiasedMHA_Bipartite1G(d_model, n_heads, dropout=dropout) for _ in range(n_layers)]) if use_sink_query else None
        self.nn_attn = nn.ModuleList([BiasedMHA_Bipartite1G(d_model, n_heads, dropout=dropout) for _ in range(n_layers)])

        self.ln_nodes_1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.ln_nodes_2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.ln_sinks = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)]) if use_sink_query else None

        self.ff_nodes = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(d_ff, d_model), nn.Dropout(dropout)
            ) for _ in range(n_layers)
        ])
        if use_sink_query:
            self.ff_sinks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
                    nn.Linear(d_ff, d_model), nn.Dropout(dropout)
                ) for _ in range(n_layers)
            ])

        self.ln_out = nn.LayerNorm(d_model)

    def forward(self,
                x: torch.Tensor,
                C: torch.Tensor,
                sink_idx: torch.Tensor,
                mask_nodes: torch.Tensor | None = None) -> torch.Tensor:
        # Basic checks and shapes
        if not (sink_idx is not None and sink_idx.dim() == 1):
            raise RuntimeError("sink_idx must be a 1D LongTensor of node indices")
        N = x.size(0)
        S = sink_idx.size(0)
        if C.size(0) != N or C.size(1) != S:
            raise ValueError(f"C must be [N={N}, S={S}] matching x and sink_idx; got {tuple(C.shape)}")

        # Project node features
        h = self.proj_in(x)  # [N,D]
        if mask_nodes is None:
            mask_nodes = torch.zeros(N, dtype=torch.bool, device=h.device)

        # Correlation-aware additive encoding on nodes
        if self.use_corr_pe:
            h = self.node_corr_pe(h, C=C, mask_nodes=mask_nodes)  # [N,D]

        # Optional sink queries initialized from designated sink nodes (detached init)
        if self.use_sink_query:
            sink_q = h.index_select(0, sink_idx).detach()         # [S,D]
            sink_q = self.sink_query_proj(sink_q)                  # [S,D]
        else:
            sink_q = None

        # Precompute biases
        if self.use_corr_bias:
            bias_nn = self.bias_builder.node_node_bias(C)         # [N,N]
            # IMPORTANT: bias_sn must be [S, N]; C.t() assumes columns align with sink_idx
            bias_sn = self.bias_builder.sink_node_bias(C) if self.use_sink_query else None  # [S,N]
        else:
            bias_nn = None
            bias_sn = None

        L = len(self.nn_attn)
        for ell in range(L):
            # Sink <- Nodes: Q=sink_q [S,D], K/V=h [N,D], bias_sn [S,N]
            if self.use_sink_query:
                sq_res = sink_q
                sink_q = self.ln_sinks[ell](sink_q)
                sink_q = sq_res + self.sn_attn[ell](
                    Q=sink_q, K=h, V=h,
                    key_padding_mask=mask_nodes,          # [N]
                    per_query_key_bias=bias_sn             # [S,N]
                )
                sink_q = sink_q + self.ff_sinks[ell](sink_q)

            # Nodes <- Nodes: self-attn among nodes with bias_nn [N,N]
            h_res = h
            h = self.ln_nodes_1[ell](h)
            h = h_res + self.nn_attn[ell](
                Q=h, K=h, V=h,
                key_padding_mask=mask_nodes,              # [N]
                per_query_key_bias=bias_nn                # [N,N]
            )
            h = self.ln_nodes_2[ell](h + self.ff_nodes[ell](h))

        h = self.ln_out(h)  # [N,D]

        # Final outputs are the embeddings of the designated sink nodes.
        sink_emb = h.index_select(0, sink_idx)  # [S,D]
        if self.use_sink_query:
            sink_emb = sink_emb + sink_q        # fuse node state with aggregated sink query
        return sink_emb  # [S,D]


# =============================
# Minimal single-graph example
# =============================
if __name__ == "__main__":
    torch.manual_seed(0)

    N, S = 12, 4
    d_in, d_model = 32, 64

    # Node features [N, d_in]
    x = torch.randn(N, d_in)

    # Sink indices (authoritative)
    sink_idx = torch.tensor([2, 5, 7, 11], dtype=torch.long)

    # Correlation matrix C aligned with sink_idx columns: [N, S]
    # IMPORTANT: Column s corresponds to sink at sink_idx[s]
    C = torch.rand(N, S)

    # Optional padding mask; all valid here
    mask_nodes = torch.zeros(N, dtype=torch.bool)

    model = GraphTransformerWithSinks1G(
        d_in=d_in,
        d_model=d_model,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        dropout=0.1,
        use_corr_pe=True,
        use_corr_bias=True,
        use_sink_query=True
    )

    sink_emb = model(x, C=C, sink_idx=sink_idx, mask_nodes=mask_nodes)  # [S, d_model]
    print("Sink embedding shape:", sink_emb.shape)
    print("First sink embedding (first 5 dims):", sink_emb[0, :5])

    # Example supervised loss over sink embeddings (regression sketch)
    y = torch.randn(S, d_model)
    loss = F.mse_loss(sink_emb, y)
    loss.backward()
    print("Loss:", loss.item())
