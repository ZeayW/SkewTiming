import torch


def segment_softmax_aggregate(z, score, dst_pos, num_dst):
    """Softmax-normalize edge scores by destination and aggregate edge values."""
    if dst_pos.numel() == 0:
        return z.new_zeros((num_dst, z.shape[1])), score

    max_score = torch.full((num_dst, 1), -float('inf'), dtype=score.dtype, device=score.device)
    max_score.scatter_reduce_(0, dst_pos.unsqueeze(1), score, reduce='amax', include_self=True)
    score_exp = torch.exp(score - max_score[dst_pos])
    score_sum = torch.zeros((num_dst, 1), dtype=score.dtype, device=score.device)
    score_sum.index_add_(0, dst_pos, score_exp)
    alpha = score_exp / score_sum[dst_pos]

    aggregated = torch.zeros((num_dst, z.shape[1]), dtype=z.dtype, device=z.device)
    aggregated.index_add_(0, dst_pos, alpha * z)
    return aggregated, alpha


class DAGCorrelationFunction(torch.autograd.Function):
    """Level-ordered correlation propagation with a compact custom backward."""

    @staticmethod
    def forward(ctx, initial_hp, edge_weight, topo_edges):
        hp = initial_hp.clone()
        for src, dst, eid in topo_edges:
            if eid.numel() == 0:
                continue
            msg = hp[src] * edge_weight[eid]
            hp[dst] = 0
            hp.index_add_(0, dst, msg)

        ctx.topo_edges = topo_edges
        ctx.save_for_backward(hp, edge_weight)
        return hp

    @staticmethod
    def backward(ctx, grad_output):
        hp, edge_weight = ctx.saved_tensors
        grad_hp = grad_output.clone()
        grad_weight = torch.zeros_like(edge_weight)

        for src, dst, eid in reversed(ctx.topo_edges):
            if eid.numel() == 0:
                continue
            grad_dst = grad_hp[dst]
            grad_weight.index_add_(
                0,
                eid,
                torch.sum(hp[src] * grad_dst, dim=1, keepdim=True),
            )
            grad_hp.index_add_(0, src, grad_dst * edge_weight[eid])

        # initial_hp is a constant endpoint seed in NUA-Timer.
        return None, grad_weight, None


def dag_correlation(initial_hp, edge_weight, topo_edges):
    if edge_weight.dim() != 2 or edge_weight.shape[1] != 1:
        raise ValueError('edge_weight must have shape [num_edges, 1]')
    return DAGCorrelationFunction.apply(initial_hp, edge_weight, topo_edges)
