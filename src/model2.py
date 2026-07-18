"""Torch Module for TimeConv layer"""
import dgl
import torch
import torch as th
from torch import nn

from utils import *
from options import get_options
from transformer import PathTransformer
from pathformer4 import *
from dag_ops import dag_correlation, segment_softmax_aggregate
from training_utils import normalize_endpoint_correlation
from time import time

options = get_options()


TCAD6_MODEL_CONFIG = {
    'hidden_dim': 128,
    'flag_noTPE': False,
    'flag_noFSE': False,
    'flag_residual': True,
    'use_pathgnn': True,
    'path_feat_choice': 3,
    'path_corr_choice': 1,
    'path_delay_choice': 3,
    'use_corr_pe': True,
    'use_attn_bias': True,
    'flag_gt': False,
    'flag_transformer': 1,
    'flag_rawpath': True,
    'flag_singlepath': False,
    'flag_delay': False,
    'flag_degree': False,
    'flag_width': True,
    'flag_path_supervise': True,
    'flag_reverse': True,
    'flag_splitfeat': False,
    'agg_choice': 0,
    'attn_choice': 0,
    'alpha': 5,
    'beta': 5,
    'base_pe': 'sinusoidal',
}


def require_tcad6_model_config(name, value):
    expected = TCAD6_MODEL_CONFIG[name]
    if value != expected:
        raise ValueError(
            '{}={} is no longer supported in the active BPN path. '
            'Use src/scripts/run_train_tcad6.sh as the canonical configuration '
            '({}={}).'.format(name, value, name, expected)
        )


# device = th.device("cuda:" + str(options.gpu) if th.cuda.is_available() and options.gpu!=-1 else "cpu")
# from transformers import BertTokenizer, MobileBertForSequenceClassification, MobileBertConfig
class VanillaDirGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers=3):
        super(VanillaDirGNN, self).__init__()
        self.num_layers = num_layers
        self.mlp_in = MLP(in_dim, hidden_dim, hidden_dim)
        # combine self, forward, reverse messages
        self.mlp_update = MLP(3 * hidden_dim, hidden_dim, hidden_dim)
        self.activation = nn.LeakyReLU(negative_slope=0.1)

    def _msg(self, edges):
        return {'mg': edges.src['feat_gnn']}

    def _reduce_mean_f(self, nodes):
        if 'mg' not in nodes.mailbox:
            # no incoming edges of this type
            return {'neigh_f': th.zeros_like(nodes.data['feat_gnn'])}
        m = nodes.mailbox['mg'].mean(dim=1)
        return {'neigh_f': m}

    def _reduce_mean_r(self, nodes):
        if 'mg' not in nodes.mailbox:
            # no incoming edges of this type
            return {'neigh_r': th.zeros_like(nodes.data['feat_gnn'])}
        m = nodes.mailbox['mg'].mean(dim=1)
        return {'neigh_r': m}
    def _forward_impl(self, graph, feat, use_builtin):
        with graph.local_scope():
            graph.ndata['feat_gnn'] = self.mlp_in(feat)

            for _ in range(self.num_layers):
                if use_builtin:
                    graph.update_all(
                        fn.copy_u('feat_gnn', 'mg'), fn.mean('mg', 'neigh_f'), etype='forward')
                    graph.update_all(
                        fn.copy_u('feat_gnn', 'mg'), fn.mean('mg', 'neigh_r'), etype='reverse')
                else:
                    graph.update_all(self._msg, self._reduce_mean_f, etype='forward')
                    graph.update_all(self._msg, self._reduce_mean_r, etype='reverse')

                h_cat = th.cat((
                    graph.ndata['feat_gnn'],
                    graph.ndata['neigh_f'],
                    graph.ndata['neigh_r'],
                ), dim=1)
                graph.ndata['feat_gnn'] = self.activation(self.mlp_update(h_cat))

            return graph.ndata['feat_gnn']

    @staticmethod
    def _assert_close(name, value, reference):
        if not th.allclose(value, reference, rtol=1e-4, atol=1e-5):
            max_abs = th.max(th.abs(value - reference)).item()
            raise ValueError('FSE GNN {} mismatch: max_abs={:.6g}'.format(name, max_abs))

    def forward(self, graph, feat, impl='udf'):
        if impl == 'udf':
            return self._forward_impl(graph, feat, use_builtin=False)
        if impl == 'builtin':
            return self._forward_impl(graph, feat, use_builtin=True)
        if impl != 'compare':
            raise ValueError('Unknown FSE GNN implementation: {}'.format(impl))

        udf_output = self._forward_impl(graph, feat, use_builtin=False)
        builtin_output = self._forward_impl(graph, feat, use_builtin=True)
        self._assert_close('output', builtin_output.detach(), udf_output.detach())

        if th.is_grad_enabled():
            node_scale = ((th.arange(len(udf_output), device=udf_output.device) % 7) + 1).unsqueeze(1)
            col_scale = ((th.arange(udf_output.shape[1], device=udf_output.device) % 5) + 1).unsqueeze(0)
            projection = (node_scale * col_scale).to(dtype=udf_output.dtype)
            projection = projection / projection.max()
            parameters = tuple(parameter for parameter in self.parameters() if parameter.requires_grad)
            udf_grads = th.autograd.grad(
                th.sum(udf_output * projection), parameters, allow_unused=True)
            builtin_grads = th.autograd.grad(
                th.sum(builtin_output * projection), parameters, retain_graph=True, allow_unused=True)
            for idx, (builtin_grad, udf_grad) in enumerate(zip(builtin_grads, udf_grads)):
                if builtin_grad is None or udf_grad is None:
                    if builtin_grad is not udf_grad:
                        raise ValueError('FSE GNN parameter {} gradient presence mismatch'.format(idx))
                    continue
                self._assert_close(
                    'parameter {} gradient'.format(idx), builtin_grad, udf_grad)

        return builtin_output


def extract_endpoint_metadata(graph, nodesprob, PIsmask):
    """
    Extract metadata for each endpoint using parallel tensor operations.
    Compatible with older PyTorch versions (manual nan-stats calculation).
    """
    device = nodesprob.device
    num_endpoints = nodesprob.shape[1]

    # Get PI indices
    PIsmask_1d = PIsmask.squeeze(-1)  # [num_nodes]

    # Extract PI correlations: [num_PIs, num_endpoints]
    pi_corr = nodesprob[PIsmask_1d, :]

    # Create mask for non-zero correlations (Valid Inputs)
    # [num_PIs, num_endpoints]
    nonzero_mask = pi_corr > 0
    mask_float = nonzero_mask.float()

    # Calculate counts (N) for each endpoint
    counts = mask_float.sum(dim=0)  # [num_endpoints]
    valid_endpoints = counts > 0  # Mask for endpoints that have at least 1 input

    # ===== Helper: Safe Mean & Std Calculation =====
    def compute_masked_stats(values, mask, counts):
        # 1. Mean
        # Zero out invalid values so sum is correct
        safe_values = torch.where(mask, values, torch.zeros_like(values))
        val_sum = safe_values.sum(dim=0)

        # Avoid division by zero
        safe_counts = counts.clamp(min=1)
        mean = val_sum / safe_counts

        # 2. Std
        # Subtract mean (broadcasted)
        diff = (values - mean.unsqueeze(0))
        # Zero out diffs where mask is False
        diff_masked = torch.where(mask, diff, torch.zeros_like(diff))
        sq_diff_sum = diff_masked.pow(2).sum(dim=0)

        # Sample variance (N-1)
        denom = (counts - 1).clamp(min=0)
        # Avoid div by zero for N=0 or N=1
        safe_denom = denom.clamp(min=1)

        var = sq_diff_sum / safe_denom
        std = torch.sqrt(var)

        # Zero out stats where N=0 (mean) or N<=1 (std)
        mean = torch.where(counts > 0, mean, torch.zeros_like(mean))
        std = torch.where(counts > 1, std, torch.zeros_like(std))

        return mean, std

    # ===== 1) Input Correlation Distribution =====
    corr_mean, corr_std = compute_masked_stats(pi_corr, nonzero_mask, counts)

    # Max/Min
    # Initialize with small/large values
    neg_inf = torch.tensor(float('-inf'), device=device)
    pos_inf = torch.tensor(float('inf'), device=device)

    # Use where to put -inf/inf in invalid spots so max/min ignores them
    # But if a column is ALL invalid, max/min will return -inf/inf, we fix that later
    masked_max_input = torch.where(nonzero_mask, pi_corr, neg_inf)
    masked_min_input = torch.where(nonzero_mask, pi_corr, pos_inf)

    corr_max = masked_max_input.max(dim=0).values
    corr_min = masked_min_input.min(dim=0).values

    # Fix results for endpoints with no inputs (0 correlation)
    corr_max = torch.where(valid_endpoints, corr_max, torch.zeros_like(corr_max))
    corr_min = torch.where(valid_endpoints, corr_min, torch.zeros_like(corr_min))

    num_nonzero = counts

    # Calculate the entropy of the correlation
    PIs_prob = th.transpose(nodesprob[PIsmask_1d], 0, 1)
    nodes_prob_tr = th.transpose(nodesprob, 0, 1)
    etp_all = -th.sum(nodes_prob_tr * th.log(nodes_prob_tr.clamp(min=1e-10)), dim=1).unsqueeze(1)
    etp_all_norm = etp_all.squeeze(1)/ th.sum(nodes_prob_tr,dim=1).clamp(min=1)
    etp_pi = -th.sum(PIs_prob * th.log(PIs_prob.clamp(min=1e-10)), dim=1).unsqueeze(1)

    # ===== 2) Input Arrival Time Variance =====
    # Get raw arrival times: [num_PIs] -> [num_PIs, 1] -> [num_PIs, num_endpoints]
    pi_arrivals = graph.ndata['delay'][PIsmask_1d]  # [num_PIs, 1]
    pi_arrivals_expanded = pi_arrivals.expand(-1, num_endpoints)

    # Use the SAME mask (correlation > 0) to define relevant inputs
    arrival_mean_tensor, arrival_std_tensor = compute_masked_stats(
        pi_arrivals_expanded, nonzero_mask, counts
    )


    # ===== 5) Fanin Size =====
    # Create mask for non-zero correlations (Valid Nodes)
    # [num_nodes, num_endpoints]
    nonzero_mask_all = nodesprob > 0
    mask_float_all = nonzero_mask_all.float()
    # Calculate counts (N) for each endpoint
    fanin_size = mask_float_all.sum(dim=0)  # [num_endpoints]

    # Package into dict
    metadata = {
        'corr_min': corr_min,
        'corr_max': corr_max,
        'corr_std': corr_std,
        'corr_entropy_input': etp_pi,
        'corr_entropy_all': etp_all,
        'corr_entropy_all_norm': etp_all_norm,
        'num_corr_inputs': num_nonzero,
        'fanin_size':fanin_size,
        'arrival_time_std': arrival_std_tensor,
        'arrival_time_mean': arrival_mean_tensor,
    }

    return metadata


def row_softmax_on_nonzero(x: torch.Tensor) -> torch.Tensor:
    """
    x: 2D tensor of shape (N, D)
    Returns a tensor of same shape where, for each row,
    softmax is applied only to non-zero elements.
    """
    # Boolean mask of non-zero elements
    mask = x != 0
    if len(x[mask]) == 0:
        return x
    # For numerical stability, subtract max per row over non-zero entries
    # Fill zeros with very large negative so they don't affect max
    neg_inf = torch.finfo(x.dtype).min
    x_masked_for_max = torch.where(mask, x, torch.full_like(x, neg_inf))
    row_max = x_masked_for_max.max(dim=1, keepdim=True).values

    # Compute exp only on non-zero, zero elsewhere
    exps = torch.where(mask, torch.exp(x - row_max), torch.zeros_like(x))

    # Row-wise sum over non-zero entries
    row_sum = exps.sum(dim=1, keepdim=True)

    # Avoid division by zero when a row is all zeros
    out = torch.where(row_sum > 0, exps / row_sum, torch.zeros_like(exps))
    return out


def cat_tensor(t1, t2):
    if t1 is None:
        return t2
    elif t2 is None:
        return t1
    else:
        return th.cat((t1, t2), dim=0)


def get_nodename(nodes_name, nid):
    if nodes_name[nid][1] is not None:
        return nodes_name[nid][1]
    else:
        return nodes_name[nid][0]


# The multiplayer perceptron model
class MLP(th.nn.Module):
    def __init__(self, *sizes, negative_slope=0.1, batchnorm=False, dropout=False):
        super().__init__()
        fcs = []
        for i in range(1, len(sizes)):
            fcs.append(th.nn.Linear(sizes[i - 1], sizes[i]))
            if i < len(sizes) - 1:
                fcs.append(th.nn.LeakyReLU(negative_slope=negative_slope))
                if dropout: fcs.append(th.nn.Dropout(p=0.01))
                if batchnorm: fcs.append(th.nn.BatchNorm1d(sizes[i]))
        self.layers = th.nn.Sequential(*fcs)

    def forward(self, x):
        return self.layers(x)


class BPN(nn.Module):

    def __init__(self,
                 infeat_dim1,
                 infeat_dim2,
                 hidden_dim,
                 device,
                 alpha=5,
                 beta=5,
                 base_pe='sinusoidal',
                 flag_path_supervise=True,
                 flag_reverse=True,
                 ):
        super(BPN, self).__init__()

        for name, value in {
            'hidden_dim': hidden_dim,
            'flag_path_supervise': flag_path_supervise,
            'flag_reverse': flag_reverse,
            'alpha': alpha,
            'beta': beta,
            'base_pe': base_pe,
        }.items():
            require_tcad6_model_config(name, value)

        self.device = device
        self.flag_path_supervise = flag_path_supervise
        self.flag_reverse = flag_reverse
        self.infeat_dim = infeat_dim1
        self.hidden_dim = hidden_dim
        self.mlp_pi = MLP(4, int(hidden_dim / 2), hidden_dim)

        tf_dim = hidden_dim
        gnn_outdim = int(tf_dim / 2)
        cpe_base_pe = base_pe
        if getattr(options, 'cpe_depth_encoding', 'absolute') == 'correlation_only':
            cpe_base_pe = 'none'
        self.pathformer = PathTransformerW(
            d_in=tf_dim,
            d_model=tf_dim,
            n_heads=4,
            n_layers=3,
            alpha=alpha,
            beta=beta,
            base_pe=cpe_base_pe,
            use_corr_pe=True,
            use_attn_bias=True,
        )
        self.mlp_out_residual = MLP(tf_dim, self.hidden_dim, self.hidden_dim, 1, negative_slope=0.1)
        self.gnn_pathfeat = VanillaDirGNN(in_dim=infeat_dim1 + infeat_dim2, hidden_dim=gnn_outdim)
        self.proj_pathfeat = nn.Linear(2, int(tf_dim / 2))
        self.linear_delay = nn.Linear(1, tf_dim)

        self.mlp_w2 = MLP(2, hidden_dim, 1)

        self.probinfo_dim = 32
        self.mlp_probinfo = MLP(1, hidden_dim, self.probinfo_dim)

        # TPE + FSE + CPE(path embedding) + path-delay embedding for path_delay_choice=3.
        new_out_dim = hidden_dim + gnn_outdim + self.probinfo_dim + tf_dim + tf_dim
        self.mlp_out_new = MLP(new_out_dim, self.hidden_dim, self.hidden_dim, 1, negative_slope=0.1)
        self.new_out_dim = new_out_dim

        self.feat_name1 = 'feat'
        self.feat_name2 = 'feat'
        self.infeat_dim2 = infeat_dim2 + infeat_dim1
        self.infeat_dim1 = infeat_dim2 + infeat_dim1

        feat_dim_m = self.infeat_dim2 + 1
        feat_dim_g = self.infeat_dim1

        feat_dim_m += 1

        neigh_dim_m = self.hidden_dim + feat_dim_m
        neigh_dim_g = self.hidden_dim + feat_dim_g

        self.linear_neigh_module = th.nn.Linear(neigh_dim_m, hidden_dim)
        self.linear_neigh_gate = th.nn.Linear(neigh_dim_g, hidden_dim)

        atnn_dim_m = hidden_dim
        atnn_dim_g = hidden_dim
        self.attention_vector_g = nn.Parameter(th.randn(atnn_dim_g, 1), requires_grad=True)
        self.attention_vector_m = nn.Parameter(th.randn(atnn_dim_m, 1), requires_grad=True)

        self.activation = th.nn.LeakyReLU(negative_slope=0)
        self.activation2 = th.nn.LeakyReLU(negative_slope=0.2)

    def nodes_func_module(self, nodes):
        h = self.activation(nodes.data['neigh'])
        # print('m',th.sum(h[h<0]))
        return {'h': h, 'attn_sum': nodes.data['attn_sum'], 'attn_max': nodes.data['attn_max']}

    def nodes_func_gate(self, nodes):
        # mask = nodes.data['is_po'].squeeze() != 1
        h = self.activation(nodes.data['neigh'])
        # print('g', th.sum(h[h < 0]))
        return {'h': h, 'attn_sum': nodes.data['attn_sum'], 'attn_max': nodes.data['attn_max']}

    def edge_msg_module_weight(self, edges):

        normalized_attn_e = th.exp(edges.data['attn_e'] - edges.dst['attn_max']) / edges.dst['attn_sum'].squeeze(2)

        return {'weight': normalized_attn_e}

    def edge_msg_module(self, edges):

        # h_dst = th.cat((edges.data['bit_position']/edges.dst['width'],edges.dst[self.feat_name2]),dim=1)
        h_dst = th.cat((edges.data['bit_position'], edges.dst[self.feat_name2]), dim=1)
        # print(th.sum(edges.data['bit_position']/edges.dst['width'])/len(edges.data['bit_position']))

        h_dst = th.cat((h_dst, edges.dst['width2']), dim=1)


        z = th.cat((edges.src['h'], h_dst), dim=1)

        z = self.linear_neigh_module(z)
        e = th.matmul(z, self.attention_vector_m)
        e = self.activation2(e)
        return {'attn_e': e, 'z': z}

    def message_func_module(self, edges):
        # m = th.cat((edges.src['h'], edges.data['bit_position'].unsqueeze(1)), dim=1)
        m = edges.data['z']
        rst = {'m': m, 'attn_e': edges.data['attn_e']}
        return rst

    def reduce_func_attn_m(self, nodes):
        # h_pos = th.mean(nodes.mailbox['pos'],dim=1)
        attn_e = nodes.mailbox['attn_e']
        max_attn_e = th.max(attn_e, dim=1).values
        attn_e = attn_e - max_attn_e.unsqueeze(1)
        attn_e_exp = th.exp(attn_e)
        attn_exp_sum = th.sum(attn_e_exp, dim=1).unsqueeze(1)
        alpha = attn_e_exp / attn_exp_sum
        h = th.sum(alpha * nodes.mailbox['m'], dim=1)
        return {'neigh': h, 'attn_sum': attn_exp_sum, 'attn_max': max_attn_e}

    def reduce_func_attn_g(self, nodes):

        attn_e = nodes.mailbox['attn_e']
        max_attn_e = th.max(attn_e, dim=1).values
        attn_e = attn_e - max_attn_e.unsqueeze(1)
        attn_e_exp = th.exp(attn_e)
        attn_exp_sum = th.sum(attn_e_exp, dim=1).unsqueeze(1)
        alpha = attn_e_exp / attn_exp_sum
        h = th.sum(alpha * nodes.mailbox['m'], dim=1)
        return {'neigh': h, 'attn_sum': attn_exp_sum, 'attn_max': max_attn_e}

    def reduce_func_mean(self, nodes):
        return {'neigh': th.mean(nodes.mailbox['m'], dim=1), 'pos': th.mean(nodes.mailbox['pos'], dim=1)}

    def edge_msg_gate(self, edges):
        h_dst = edges.dst[self.feat_name2]

        # h_dst = self.linear_feat_gate(h_dst)

        z = th.cat((edges.src['h'], h_dst), dim=1)
        z = self.linear_neigh_gate(z)
        e = th.matmul(z, self.attention_vector_g)
        e = self.activation2(e)

        return {'attn_e': e, 'z': z}

    def edge_msg_gate_weight(self, edges):

        weight = th.exp(edges.data['attn_e'] - edges.dst['attn_max']) / edges.dst['attn_sum'].squeeze(2)

        return {'weight': weight}

    def message_func_gate(self, edges):
        m = edges.data['z']
        return {'m': m, 'attn_e': edges.data['attn_e']}

    def message_func_reverse(self, edges):

        prob = edges.src['hp'] * edges.data['weight']

        return {'mp': prob}

    def reduce_fun_reverse(self, nodes):
        return {'hp': th.sum(nodes.mailbox['mp'], dim=1)}

    def reduce_func_maxprob(self, nodes):

        max_prob = th.max(nodes.mailbox['p'], dim=1).values
        # print('ee', max_prob.shape, len(nodes))
        return {'max_prob': max_prob}

    def edge_msg_critical(self, edges):
        is_critical = th.logical_and(edges.src['is_critical'], edges.src['max_prob'] == edges.dst['hp'])
        is_critical = th.logical_and(is_critical, edges.dst['hp'] != 0)
        return {'is_critical': is_critical}

    def message_func_maxprob_r(self, edges):
        is_critical = th.logical_and(edges.src['is_critical'], edges.src['max_prob'] == edges.dst['hp'])
        is_critical = th.logical_and(is_critical, edges.dst['hp'] != 0)

        w = is_critical * edges.data['weight']
        return {'is_critical': is_critical, 'w': w}

    def reduce_func_maxprob_r(self, nodes):

        is_critical = th.any(nodes.mailbox['is_critical'], dim=1)
        corr_local = th.sum(nodes.mailbox['w'], dim=1)
        return {'is_critical': is_critical, 'cl': corr_local}

    def message_func_prob(self, edges):
        msg = th.gather(edges.src['hp'], dim=1, index=edges.dst['id'])
        pi_feat = edges.src['delay']
        return {'mp': msg, 'mi': pi_feat, 'mw': edges.data['w']}

    def nodes_func_pi(self, nodes):
        h = th.cat((nodes.data['delay'], nodes.data['value']), dim=1)
        h = self.mlp_pi(h)
        h = self.activation(h)
        # mask = nodes.data['is_po'].squeeze() != 1
        # h[mask] = self.activation(h[mask])
        return {'h': h}

    def reduce_func_prob(self, nodes):
        prob_sum = th.sum(nodes.mailbox['mp'], dim=1)
        # prob_sum = th.sum(nodes.mailbox['mp']*nodes.mailbox['mi'], dim=1)
        # prob_max = th.max(nodes.mailbox['mp'],dim=1).values
        # print(nodes.mailbox['mw'].shape,nodes.mailbox['mp'].shape)
        prob_target = th.softmax(nodes.mailbox['mw'], dim=1).reshape(nodes.mailbox['mp'].shape)
        # print(prob_target)
        #prob_dev = F.cosine_similarity(prob_target,nodes.mailbox['mp'],dim=1)
        prob_ce = th.sum(prob_target * th.log(prob_target / (nodes.mailbox['mp'] + 1e-10)), dim=1)

        prob_dev = F.cosine_similarity(prob_target,nodes.mailbox['mp'],dim=1)
        #prob_dev = F.kl_div(th.log_softmax(nodes.mailbox['mp'],dim=0),prob_target,reduction='none').sum(dim=1)

        # print(prob_target,nodes.mailbox['mp'])
        #print(prob_dev)

        # union_prob = 1 / nodes.mailbox['mp'].shape[1]
        #
        # prob_ce = th.sum(union_prob*th.log(union_prob/ (nodes.mailbox['mp']+1e-10) ),dim=1)

        # print(nodes.mailbox['mp'].shape,union_prob)
        # print(prob_dev,prob_ce)
        # exit()
        #prob_dev = th.sum(nodes.mailbox['mp']*th.log(nodes.mailbox['mp']+1e-10),dim=1)

        # prob_dev = th.sum(th.pow(nodes.mailbox['ml'] - prob_mean,2), dim=1)

        return {'prob_ce': prob_ce, 'prob_sum': prob_sum, 'prob_dev': prob_dev}

    def prop_backward_dgl(self, graph, graph_info):
        topo_r = graph_info['topo_r']
        for l, nodes in enumerate(topo_r[1:]):
            graph.pull(nodes, self.message_func_reverse, self.reduce_fun_reverse, etype='reverse')

        return graph.ndata['hp']

    def prop_backward_scatter(self, graph, graph_info):
        hp = graph.ndata['hp']
        edge_weight = graph.edges['reverse'].data['weight'].reshape(-1, 1)
        topo_r_edges = self.get_topo_r_edges(graph, graph_info)

        for src, dst, eid in topo_r_edges:
            if eid.numel() == 0:
                continue
            msg = hp[src] * edge_weight[eid]
            if th.is_grad_enabled():
                hp_next = hp.clone()
                hp_next[dst] = 0
                hp = hp_next.index_add(0, dst, msg)
            else:
                hp[dst] = 0
                hp.index_add_(0, dst, msg)

        graph.ndata['hp'] = hp
        return hp

    def get_topo_r_edges(self, graph, graph_info):
        topo_r_edges = graph_info.get('topo_r_in_edges')
        if topo_r_edges is not None:
            return topo_r_edges
        return [
            graph.in_edges(nodes, form='all', etype='reverse')
            for nodes in graph_info['topo_r'][1:]
        ]

    def prop_backward_custom(self, graph, graph_info):
        hp = dag_correlation(
            graph.ndata['hp'],
            graph.edges['reverse'].data['weight'].reshape(-1, 1),
            self.get_topo_r_edges(graph, graph_info),
        )
        graph.ndata['hp'] = hp
        return hp

    def prop_backward(self, graph, graph_info):
        mtde_backward_impl = getattr(options, 'mtde_backward_impl', 'dgl')
        if mtde_backward_impl == 'dgl':
            return self.prop_backward_dgl(graph, graph_info)
        if mtde_backward_impl == 'scatter':
            return self.prop_backward_scatter(graph, graph_info)
        if mtde_backward_impl == 'custom':
            if not th.is_grad_enabled():
                return self.prop_backward_scatter(graph, graph_info)
            return self.prop_backward_custom(graph, graph_info)
        if mtde_backward_impl == 'compare':
            initial_hp = graph.ndata['hp'].clone()
            with th.no_grad():
                graph.ndata['hp'] = initial_hp.clone()
                dgl_hp = self.prop_backward_dgl(graph, graph_info).clone()
                graph.ndata['hp'] = initial_hp.clone()
                scatter_hp = self.prop_backward_scatter(graph, graph_info).clone()
                graph.ndata['hp'] = initial_hp.clone()
                custom_hp = self.prop_backward_custom(graph, graph_info).clone()
            if not th.allclose(dgl_hp, scatter_hp, rtol=1e-4, atol=1e-5):
                max_abs = th.max(th.abs(dgl_hp - scatter_hp)).item()
                raise ValueError('MTDE scatter backward mismatch: max_abs={:.6g}'.format(max_abs))
            if not th.allclose(dgl_hp, custom_hp, rtol=1e-4, atol=1e-5):
                max_abs = th.max(th.abs(dgl_hp - custom_hp)).item()
                raise ValueError('MTDE custom backward mismatch: max_abs={:.6g}'.format(max_abs))
            edge_weight = graph.edges['reverse'].data['weight']
            if th.is_grad_enabled() and edge_weight.requires_grad:
                node_scale = ((th.arange(initial_hp.shape[0], device=initial_hp.device) % 7) + 1).reshape(-1, 1)
                col_scale = ((th.arange(initial_hp.shape[1], device=initial_hp.device) % 5) + 1).reshape(1, -1)

                graph.ndata['hp'] = initial_hp.clone()
                scatter_grad_hp = self.prop_backward_scatter(graph, graph_info)
                scatter_probe = th.sum(scatter_grad_hp * node_scale * col_scale)
                scatter_grad, = th.autograd.grad(scatter_probe, edge_weight)

                graph.ndata['hp'] = initial_hp.clone()
                custom_grad_hp = self.prop_backward_custom(graph, graph_info)
                custom_probe = th.sum(custom_grad_hp * node_scale * col_scale)
                custom_grad, = th.autograd.grad(custom_probe, edge_weight)

                if not th.allclose(scatter_grad, custom_grad, rtol=1e-4, atol=1e-5):
                    max_abs = th.max(th.abs(scatter_grad - custom_grad)).item()
                    raise ValueError('MTDE custom gradient mismatch: max_abs={:.6g}'.format(max_abs))
            graph.ndata['hp'] = initial_hp
            if th.is_grad_enabled():
                return self.prop_backward_custom(graph, graph_info)
            return self.prop_backward_scatter(graph, graph_info)
        raise ValueError('Unknown MTDE backward implementation: {}'.format(mtde_backward_impl))

    def fse_components(self, nodes_prob, nodes_emb, graph_info):
        nodes_prob_tr = th.transpose(nodes_prob, 0, 1)
        aggregation = getattr(options, 'fse_aggregation', 'raw_sum')
        if aggregation == 'endpoint_mean':
            fse_weights = normalize_endpoint_correlation(nodes_prob_tr)
        elif aggregation == 'raw_sum':
            fse_weights = nodes_prob_tr
        else:
            raise ValueError('Unknown FSE aggregation: {}'.format(aggregation))
        h_global = th.matmul(fse_weights, nodes_emb)
        probinfo_etp = -th.sum(
            nodes_prob_tr * th.log(nodes_prob_tr + 1e-10), dim=1).unsqueeze(1)
        weight_etp = -th.sum(
            nodes_prob_tr * th.log(nodes_prob_tr + 1e-10), dim=1).unsqueeze(1)
        minmax = (
            th.max(nodes_prob_tr, dim=1).values - th.min(nodes_prob_tr, dim=1).values).unsqueeze(1)
        return h_global, probinfo_etp, weight_etp, minmax

    def fse_node_embeddings(self, graph, graph_info):
        impl = getattr(options, 'fse_gnn_impl', 'udf')
        cache_mode = getattr(options, 'fse_eval_cache', 'off')
        can_cache = not self.training and not th.is_grad_enabled()

        if not can_cache or cache_mode == 'off':
            return self.gnn_pathfeat(graph, graph.ndata['feat'], impl=impl)

        cache_key = '_fse_nodes_emb'
        cached = graph_info.get(cache_key)
        if cached is None:
            cached = self.gnn_pathfeat(graph, graph.ndata['feat'], impl=impl).detach()
            graph_info[cache_key] = cached
            return cached

        if cache_mode == 'compare':
            reference = self.gnn_pathfeat(graph, graph.ndata['feat'], impl=impl)
            VanillaDirGNN._assert_close('eval cache', cached, reference)
        return cached

    def append_path_context(self, path_feat, distance, corr):
        depth_encoding = getattr(options, 'cpe_depth_encoding', 'absolute')
        if depth_encoding == 'correlation_only':
            distance = th.zeros_like(distance)
        elif depth_encoding != 'absolute':
            raise ValueError('Unknown CPE depth encoding: {}'.format(depth_encoding))
        feat_raw = self.proj_pathfeat(th.cat((distance, corr), dim=1))
        return th.cat((path_feat, feat_raw), dim=1)


    def init_path_buffers(self, graph, graph_info, nodes_emb):
        feat_p = nodes_emb
        po_count = len(graph_info['POs'])
        po_feat = feat_p[graph_info['POs']]
        distance = th.zeros((po_count, 1), dtype=th.float, device=graph.device)
        corr = th.ones((po_count, 1), dtype=th.float, device=graph.device)
        po_feat = self.append_path_context(po_feat, distance, corr)

        max_path_len = len(graph_info['topo_r'])
        path_feat = th.zeros((po_count, max_path_len, po_feat.shape[1]),
                             dtype=po_feat.dtype, device=graph.device)
        path_feat[:, 0, :] = po_feat
        path_lengths = th.zeros(po_count, dtype=th.float, device=graph.device)
        path_inputdelay = th.zeros((po_count, 1), dtype=th.float, device=graph.device)

        c_local = torch.zeros(po_count, max_path_len, device=graph.device)
        c_sink = torch.zeros(po_count, max_path_len, device=graph.device)
        c_sink[:, 0] = 1

        return feat_p, po_count, path_feat, path_lengths, path_inputdelay, c_sink, c_local

    def encode_path_inputs(self, path_feat, path_lengths, path_inputdelay, c_sink, c_local, stage_time=None):
        timed = stage_time is not None
        if timed:
            start = time()
        path_emb = self.pathformer(path_feat, path_lengths, c_sink=c_sink, c_local=c_local)
        if timed:
            stage_time['CPE_encode'] += time() - start

        return path_emb, path_lengths, path_inputdelay

    def path_search_dense(self, graph, graph_info, nodes_emb, stage_time=None):

        timed = stage_time is not None
        if timed:
            start = time()
        feat_p, po_count, path_feat, path_lengths, path_inputdelay, c_sink, c_local = \
            self.init_path_buffers(graph, graph_info, nodes_emb)
        is_ended = th.zeros(po_count, dtype=th.bool, device=graph.device)
        debug_counts = [] if getattr(options, 'log_level', 0) >= 2 else None
        debug_pairs = [] if getattr(options, 'log_level', 0) >= 2 else None
        if timed:
            stage_time['CPE_preparefeat'] += time() - start

        with th.no_grad():
            #graph.ndata['hp'] = graph_info['nodes_prob']
            pre_nodes = graph_info['POs']
            _, nodes = graph.out_edges(pre_nodes, etype='reverse')
            nodes = th.unique(nodes)
            l = 0

            while True:
                if timed:
                    start = time()
                if len(pre_nodes) !=0:
                    graph.pull(pre_nodes, fn.copy_u('hp', 'p'), self.reduce_func_maxprob, etype='forward')

                graph.pull(nodes, self.message_func_maxprob_r, self.reduce_func_maxprob_r, etype='reverse')

                critical_mask = th.transpose(graph.ndata['is_critical'][nodes], 0, 1)
                critical_counts = th.sum(critical_mask, dim=1)
                if debug_counts is not None:
                    debug_counts.append(critical_counts.detach().cpu())
                    debug_ep, debug_pos = critical_mask.nonzero(as_tuple=True)
                    debug_pairs.append((debug_ep.detach().cpu(), nodes[debug_pos].detach().cpu()))

                # if th.sum(critical_mask) == 0:
                #     break

                is_ended_mask = th.logical_and(critical_counts == 0, ~is_ended)
                is_ended[is_ended_mask] = True
                path_lengths[is_ended_mask] = l + 1
                if timed:
                    stage_time['CPE_pathfind'] += time() - start

                if th.sum(critical_mask) == 0:
                    break

                if timed:
                    start = time()
                nodes_feat_l = feat_p[nodes]
                path_feat_l = th.matmul(critical_mask.float(), nodes_feat_l)
                num_critical = critical_counts.unsqueeze(1)
                path_feat_l = path_feat_l / num_critical.clamp(min=1)

                nodes_delay_l = graph.ndata['is_critical'][nodes] * graph.ndata['delay'][nodes]
                nodes_delay_l = th.max(th.transpose(nodes_delay_l,0,1),dim=1).values.unsqueeze(1)
                path_inputdelay = th.maximum(path_inputdelay,nodes_delay_l)

                distance = (l + 1) * th.ones((path_feat_l.shape[0], 1), dtype=th.float, device=graph.device)
                nodes_prob_l = graph.ndata['hp'][nodes]
                nodes_prob_l_tr = th.transpose(nodes_prob_l, 0, 1)  # N_ep * N_l
                corr = th.sum(nodes_prob_l_tr * critical_mask, dim=1) / critical_counts.clamp(min=1)
                corr = corr.unsqueeze(1)
                path_feat_l = self.append_path_context(path_feat_l, distance, corr)

                path_feat[:, l + 1, :] = path_feat_l

                row_max = nodes_prob_l_tr.max(dim=1, keepdim=True).values
                nodes_prob_l_tr = nodes_prob_l_tr / row_max.clamp(min=1e-8)
                cs = th.sum(nodes_prob_l_tr * critical_mask, dim=1) / critical_counts.clamp(min=1)

                c_sink[:, l + 1] = cs
                cl = th.matmul(critical_mask.float(), graph.ndata['cl'][nodes]) / critical_counts.clamp(min=1)
                cl = cl.diagonal()
                c_local[:, l + 1] = cl

                if timed:
                    stage_time['CPE_preparefeat'] += time() - start
                    start = time()
                filtered_mask = th.sum(graph.ndata['is_critical'][nodes], dim=1) >= 1
                #pre_nodes = nodes
                pre_nodes = nodes[filtered_mask]
                _, nodes = graph.out_edges(pre_nodes, etype='reverse')
                nodes = th.unique(nodes)
                l += 1
                if timed:
                    stage_time['CPE_pathfind'] += time() - start

        path_feat = path_feat[:, :l + 1, :]
        c_sink = c_sink[:, :l + 1]
        c_local = c_local[:, :l + 1]
        if debug_counts is not None:
            graph_info['_cpe_dense_counts'] = debug_counts
            graph_info['_cpe_dense_pairs'] = debug_pairs

        return path_feat, path_lengths, path_inputdelay, c_sink, c_local

    def path_search_sparse(self, graph, graph_info, nodes_emb, stage_time=None):
        timed = stage_time is not None
        if timed:
            start = time()
        feat_p, po_count, path_feat, path_lengths, path_inputdelay, c_sink, c_local = \
            self.init_path_buffers(graph, graph_info, nodes_emb)
        is_ended = th.zeros(po_count, dtype=th.bool, device=graph.device)
        debug_counts = [] if getattr(options, 'log_level', 0) >= 2 else None
        if timed:
            stage_time['CPE_preparefeat'] += time() - start

        with th.no_grad():
            pre_nodes = graph_info['POs']
            _, nodes = graph.out_edges(pre_nodes, etype='reverse')
            nodes = th.unique(nodes)
            l = 0

            while True:
                if timed:
                    start = time()
                if len(pre_nodes) != 0:
                    graph.pull(pre_nodes, fn.copy_u('hp', 'p'), self.reduce_func_maxprob, etype='forward')

                graph.pull(nodes, self.message_func_maxprob_r, self.reduce_func_maxprob_r, etype='reverse')

                critical = graph.ndata['is_critical'][nodes]
                node_pos, endpoint_idx = critical.nonzero(as_tuple=True)
                counts = th.bincount(endpoint_idx, minlength=po_count).to(dtype=feat_p.dtype)
                if debug_counts is not None:
                    debug_counts.append(counts.detach().cpu())

                is_ended_mask = th.logical_and(counts == 0, ~is_ended)
                is_ended[is_ended_mask] = True
                path_lengths[is_ended_mask] = l + 1
                if timed:
                    stage_time['CPE_pathfind'] += time() - start

                if endpoint_idx.numel() == 0:
                    break

                if timed:
                    start = time()
                critical_nodes = nodes[node_pos]
                counts_safe = counts.clamp(min=1)

                path_feat_l = th.zeros((po_count, feat_p.shape[1]), dtype=feat_p.dtype, device=graph.device)
                path_feat_l.index_add_(0, endpoint_idx, feat_p[critical_nodes])
                path_feat_l = path_feat_l / counts_safe.unsqueeze(1)

                delay_values = graph.ndata['delay'][critical_nodes].squeeze(1)
                nodes_delay_l = th.zeros((po_count,), dtype=delay_values.dtype, device=graph.device)
                nodes_delay_l.scatter_reduce_(0, endpoint_idx, delay_values, reduce='amax', include_self=True)
                path_inputdelay = th.maximum(path_inputdelay, nodes_delay_l.unsqueeze(1))

                distance = (l + 1) * th.ones((po_count, 1), dtype=th.float, device=graph.device)
                prob_values = graph.ndata['hp'][critical_nodes, endpoint_idx]
                corr_sum = th.zeros((po_count,), dtype=prob_values.dtype, device=graph.device)
                corr_sum.index_add_(0, endpoint_idx, prob_values)
                corr = (corr_sum / counts_safe).unsqueeze(1)
                path_feat_l = self.append_path_context(path_feat_l, distance, corr)

                path_feat[:, l + 1, :] = path_feat_l

                row_max = graph.ndata['hp'][nodes].max(dim=0).values.clamp(min=1e-8)
                cs_values = prob_values / row_max[endpoint_idx]
                cs_sum = th.zeros((po_count,), dtype=cs_values.dtype, device=graph.device)
                cs_sum.index_add_(0, endpoint_idx, cs_values)
                c_sink[:, l + 1] = cs_sum / counts_safe

                cl_values = graph.ndata['cl'][critical_nodes, endpoint_idx]
                cl_sum = th.zeros((po_count,), dtype=cl_values.dtype, device=graph.device)
                cl_sum.index_add_(0, endpoint_idx, cl_values)
                c_local[:, l + 1] = cl_sum / counts_safe

                if timed:
                    stage_time['CPE_preparefeat'] += time() - start
                    start = time()
                filtered_mask = critical.any(dim=1)
                pre_nodes = nodes[filtered_mask]
                _, nodes = graph.out_edges(pre_nodes, etype='reverse')
                nodes = th.unique(nodes)
                l += 1
                if timed:
                    stage_time['CPE_pathfind'] += time() - start

        path_feat = path_feat[:, :l + 1, :]
        c_sink = c_sink[:, :l + 1]
        c_local = c_local[:, :l + 1]
        if debug_counts is not None:
            graph_info['_cpe_sparse_counts'] = debug_counts

        return path_feat, path_lengths, path_inputdelay, c_sink, c_local

    def path_search_frontier(self, graph, graph_info, nodes_emb, stage_time=None):
        timed = stage_time is not None
        if timed:
            start = time()
        feat_p, po_count, path_feat, path_lengths, path_inputdelay, c_sink, c_local = \
            self.init_path_buffers(graph, graph_info, nodes_emb)
        is_ended = th.zeros(po_count, dtype=th.bool, device=graph.device)
        debug_counts = [] if getattr(options, 'log_level', 0) >= 2 else None
        debug_pairs = [] if getattr(options, 'log_level', 0) >= 2 else None
        if timed:
            stage_time['CPE_preparefeat'] += time() - start

        with th.no_grad():
            num_nodes = graph.number_of_nodes()
            frontier_ep = th.arange(po_count, dtype=th.long, device=graph.device)
            frontier_nodes = graph_info['POs']
            critical_key = frontier_ep * num_nodes + frontier_nodes
            critical_ep = frontier_ep.clone()
            critical_nodes = frontier_nodes.clone()
            critical_max = graph.ndata['hp'].new_zeros(critical_key.numel())
            l = 0

            while True:
                if timed:
                    start = time()

                if frontier_nodes.numel() == 0:
                    next_ep = frontier_ep.new_empty(0)
                    next_nodes = frontier_nodes.new_empty(0)
                    next_cl = graph.ndata['hp'].new_empty(0)
                    candidate_nodes = frontier_nodes.new_empty(0)
                else:
                    active_nodes = th.unique(frontier_nodes)
                    _, candidate_nodes = graph.out_edges(active_nodes, etype='reverse')
                    candidate_nodes = th.unique(candidate_nodes)

                    current_key = frontier_ep * num_nodes + frontier_nodes
                    current_pos = th.searchsorted(critical_key, current_key)
                    current_max = critical_max[current_pos].clone()

                    degrees = graph.out_degrees(frontier_nodes, etype='reverse')
                    edge_src, edge_dst, edge_eid = graph.out_edges(frontier_nodes, form='all', etype='reverse')
                    if edge_dst.numel() != 0:
                        current_pair_idx = th.arange(frontier_nodes.numel(), dtype=th.long,
                                                     device=graph.device).repeat_interleave(degrees)
                        if debug_counts is not None and not th.equal(edge_src, frontier_nodes[current_pair_idx]):
                            print('CPE frontier debug: DGL out_edges order is not aligned with repeated frontier nodes',
                                  flush=True)
                        current_ep = frontier_ep[current_pair_idx]
                        pred_prob = graph.ndata['hp'][edge_dst, current_ep]
                        recomputed_max = th.full((frontier_nodes.numel(),), -float('inf'),
                                                 dtype=pred_prob.dtype, device=graph.device)
                        recomputed_max.scatter_reduce_(0, current_pair_idx, pred_prob,
                                                       reduce='amax', include_self=True)
                        has_fanin = degrees > 0
                        current_max = th.where(has_fanin, recomputed_max, current_max)
                        critical_max[current_pos] = current_max

                    if candidate_nodes.numel() == 0:
                        next_ep = frontier_ep.new_empty(0)
                        next_nodes = frontier_nodes.new_empty(0)
                        next_cl = graph.ndata['hp'].new_empty(0)
                    else:
                        candidate_mask = th.zeros((num_nodes,), dtype=th.bool, device=graph.device)
                        candidate_mask[candidate_nodes] = True

                        critical_degrees = graph.out_degrees(critical_nodes, etype='reverse')
                        critical_edge_src, critical_edge_dst, critical_edge_eid = graph.out_edges(
                            critical_nodes, form='all', etype='reverse')
                        critical_pair_idx = th.arange(critical_nodes.numel(), dtype=th.long,
                                                      device=graph.device).repeat_interleave(critical_degrees)
                        if debug_counts is not None and critical_edge_dst.numel() != 0 and not th.equal(
                                critical_edge_src, critical_nodes[critical_pair_idx]):
                            print('CPE frontier debug: DGL out_edges order is not aligned with repeated critical nodes',
                                  flush=True)

                        edge_ep = critical_ep[critical_pair_idx]
                        edge_max = critical_max[critical_pair_idx]
                        pred_prob = graph.ndata['hp'][critical_edge_dst, edge_ep]
                        in_candidate = candidate_mask[critical_edge_dst]
                        is_critical_edge = th.logical_and(in_candidate, pred_prob == edge_max)
                        is_critical_edge = th.logical_and(is_critical_edge, pred_prob != 0)
                        selected_ep = edge_ep[is_critical_edge]
                        if selected_ep.numel() == 0:
                            next_ep = frontier_ep.new_empty(0)
                            next_nodes = frontier_nodes.new_empty(0)
                            next_cl = graph.ndata['hp'].new_empty(0)
                        else:
                            critical_dst = critical_edge_dst[is_critical_edge]
                            critical_weight = graph.edges['reverse'].data['weight'][
                                critical_edge_eid[is_critical_edge]].reshape(-1)

                            pair_key = selected_ep * num_nodes + critical_dst
                            unique_key, inverse = th.unique(pair_key, return_inverse=True)
                            next_ep = unique_key // num_nodes
                            next_nodes = unique_key % num_nodes
                            next_cl = th.zeros((unique_key.numel(),), dtype=critical_weight.dtype, device=graph.device)
                            next_cl.index_add_(0, inverse, critical_weight)

                counts = th.bincount(next_ep, minlength=po_count).to(dtype=feat_p.dtype)
                if debug_counts is not None:
                    debug_counts.append(counts.detach().cpu())
                    debug_pairs.append((next_ep.detach().cpu(), next_nodes.detach().cpu()))
                is_ended_mask = th.logical_and(counts == 0, ~is_ended)
                is_ended[is_ended_mask] = True
                path_lengths[is_ended_mask] = l + 1
                if timed:
                    stage_time['CPE_pathfind'] += time() - start

                if next_ep.numel() == 0:
                    break

                if timed:
                    start = time()
                counts_safe = counts.clamp(min=1)

                path_feat_l = th.zeros((po_count, feat_p.shape[1]), dtype=feat_p.dtype, device=graph.device)
                path_feat_l.index_add_(0, next_ep, feat_p[next_nodes])
                path_feat_l = path_feat_l / counts_safe.unsqueeze(1)

                delay_values = graph.ndata['delay'][next_nodes].squeeze(1)
                nodes_delay_l = th.zeros((po_count,), dtype=delay_values.dtype, device=graph.device)
                nodes_delay_l.scatter_reduce_(0, next_ep, delay_values, reduce='amax', include_self=True)
                path_inputdelay = th.maximum(path_inputdelay, nodes_delay_l.unsqueeze(1))

                distance = (l + 1) * th.ones((po_count, 1), dtype=th.float, device=graph.device)
                prob_values = graph.ndata['hp'][next_nodes, next_ep]
                corr_sum = th.zeros((po_count,), dtype=prob_values.dtype, device=graph.device)
                corr_sum.index_add_(0, next_ep, prob_values)
                corr = (corr_sum / counts_safe).unsqueeze(1)
                path_feat_l = self.append_path_context(path_feat_l, distance, corr)

                path_feat[:, l + 1, :] = path_feat_l

                row_max = graph.ndata['hp'][candidate_nodes].max(dim=0).values.clamp(min=1e-8)
                cs_values = prob_values / row_max[next_ep]
                cs_sum = th.zeros((po_count,), dtype=cs_values.dtype, device=graph.device)
                cs_sum.index_add_(0, next_ep, cs_values)
                c_sink[:, l + 1] = cs_sum / counts_safe

                cl_sum = th.zeros((po_count,), dtype=next_cl.dtype, device=graph.device)
                cl_sum.index_add_(0, next_ep, next_cl)
                c_local[:, l + 1] = cl_sum / counts_safe

                if timed:
                    stage_time['CPE_preparefeat'] += time() - start

                frontier_ep = next_ep
                frontier_nodes = next_nodes
                next_key = frontier_ep * num_nodes + frontier_nodes
                if next_key.numel() != 0:
                    merged_key = th.cat((critical_key, next_key))
                    merged_max = th.cat((critical_max, critical_max.new_zeros(next_key.numel())))
                    critical_key, inverse = th.unique(merged_key, return_inverse=True)
                    critical_max_new = th.full((critical_key.numel(),), -float('inf'),
                                               dtype=critical_max.dtype, device=graph.device)
                    critical_max_new.scatter_reduce_(0, inverse, merged_max, reduce='amax', include_self=True)
                    critical_max = th.where(th.isinf(critical_max_new),
                                            th.zeros_like(critical_max_new),
                                            critical_max_new)
                    critical_ep = critical_key // num_nodes
                    critical_nodes = critical_key % num_nodes
                l += 1

        path_feat = path_feat[:, :l + 1, :]
        c_sink = c_sink[:, :l + 1]
        c_local = c_local[:, :l + 1]
        if debug_counts is not None:
            graph_info['_cpe_frontier_counts'] = debug_counts
            graph_info['_cpe_frontier_pairs'] = debug_pairs

        return path_feat, path_lengths, path_inputdelay, c_sink, c_local

    def assert_same_path_inputs(self, dense_inputs, candidate_inputs, name):
        names = ('path_feat', 'path_lengths', 'path_inputdelay', 'c_sink', 'c_local')
        mismatches = []
        for tensor_name, dense_value, candidate_value in zip(names, dense_inputs, candidate_inputs):
            if dense_value.shape != candidate_value.shape:
                mismatches.append(
                    '{} shape {} != {}'.format(tensor_name, tuple(dense_value.shape), tuple(candidate_value.shape))
                )
                continue
            if not th.allclose(dense_value, candidate_value, rtol=1e-4, atol=1e-5):
                max_abs = th.max(th.abs(dense_value - candidate_value)).item()
                detail = '{} max_abs={:.6g}'.format(tensor_name, max_abs)
                if tensor_name == 'path_lengths':
                    diff = candidate_value - dense_value
                    bad_idx = th.nonzero(diff != 0).flatten()[:5]
                    if bad_idx.numel() != 0:
                        detail += ' sample={} dense={} candidate={}'.format(
                            bad_idx.detach().cpu().numpy().tolist(),
                            dense_value[bad_idx].detach().cpu().numpy().tolist(),
                            candidate_value[bad_idx].detach().cpu().numpy().tolist(),
                        )
                mismatches.append(detail)

        if mismatches:
            raise ValueError('CPE {} implementation mismatch: {}'.format(name, '; '.join(mismatches)))

    def reset_path_search_state(self, graph, initial_is_critical):
        graph.ndata['is_critical'] = initial_is_critical.clone()
        if 'max_prob' in graph.ndata:
            graph.ndata['max_prob'] = th.zeros_like(graph.ndata['hp'])
        if 'cl' in graph.ndata:
            graph.ndata['cl'] = th.zeros_like(graph.ndata['hp'])

    def path_embedding(self, graph, graph_info, nodes_emb, stage_time=None):
        cpe_impl = getattr(options, 'cpe_impl', 'dense')

        if cpe_impl == 'dense':
            path_inputs = self.path_search_dense(graph, graph_info, nodes_emb, stage_time)
        elif cpe_impl == 'sparse':
            path_inputs = self.path_search_sparse(graph, graph_info, nodes_emb, stage_time)
        elif cpe_impl == 'frontier':
            path_inputs = self.path_search_frontier(graph, graph_info, nodes_emb, stage_time)
        elif cpe_impl == 'compare':
            initial_is_critical = graph.ndata['is_critical'].clone()
            dense_inputs = self.path_search_dense(graph, graph_info, nodes_emb, None)
            self.reset_path_search_state(graph, initial_is_critical)
            sparse_inputs = self.path_search_sparse(graph, graph_info, nodes_emb, stage_time)
            self.assert_same_path_inputs(dense_inputs, sparse_inputs, 'sparse')
            self.reset_path_search_state(graph, initial_is_critical)
            frontier_inputs = self.path_search_frontier(graph, graph_info, nodes_emb, None)
            if getattr(options, 'log_level', 0) >= 2 and not th.equal(dense_inputs[1], frontier_inputs[1]):
                bad_idx = th.nonzero(frontier_inputs[1] != dense_inputs[1]).flatten()[:5]
                dense_counts = graph_info.get('_cpe_dense_counts', [])
                frontier_counts = graph_info.get('_cpe_frontier_counts', [])
                dense_pairs = graph_info.get('_cpe_dense_pairs', [])
                frontier_pairs = graph_info.get('_cpe_frontier_pairs', [])
                def nodes_for_endpoint(pairs, endpoint):
                    seq = []
                    for eps, nodes in pairs:
                        seq.append(nodes[eps == endpoint].numpy().tolist())
                    return seq
                for idx in bad_idx.detach().cpu().numpy().tolist():
                    dense_seq = [int(c[idx].item()) for c in dense_counts]
                    frontier_seq = [int(c[idx].item()) for c in frontier_counts]
                    print('CPE frontier debug endpoint {} dense_len={} frontier_len={} dense_counts={} frontier_counts={} dense_nodes={} frontier_nodes={}'.format(
                        idx,
                        int(dense_inputs[1][idx].item()),
                        int(frontier_inputs[1][idx].item()),
                        dense_seq,
                        frontier_seq,
                        nodes_for_endpoint(dense_pairs, idx),
                        nodes_for_endpoint(frontier_pairs, idx),
                    ), flush=True)
            self.assert_same_path_inputs(dense_inputs, frontier_inputs, 'frontier')
            path_inputs = sparse_inputs
        else:
            raise ValueError('Unknown CPE implementation: {}'.format(cpe_impl))

        return self.encode_path_inputs(*path_inputs, stage_time=stage_time)

    def mtde_forward_level_data(self, graph, graph_info, i, nodes, use_cache):
        if use_cache:
            cache = graph_info.get('mtde_forward_cache')
            if cache is None:
                raise ValueError('mtde_forward_cache was requested, but graph_info has no cached edge data')
            return cache[i]

        isModule_mask = graph.ndata['is_module'][nodes] == 1
        isGate_mask = graph.ndata['is_module'][nodes] == 0
        nodes_gate = nodes[isGate_mask]
        nodes_module = nodes[isModule_mask]
        return {
            'nodes_gate': nodes_gate,
            'gate_eids': graph.in_edges(nodes_gate, form='eid', etype='intra_gate'),
            'gate_reverse_eids': graph.out_edges(nodes_gate, form='eid', etype='reverse'),
            'nodes_module': nodes_module,
            'module_eids': graph.in_edges(nodes_module, form='eid', etype='intra_module'),
            'module_reverse_eids': graph.out_edges(nodes_module, form='eid', etype='reverse'),
        }

    def mtde_forward_once(self, graph, graph_info, use_cache=False):
        topo = graph_info['topo']
        PO_mask = graph_info['POs']

        graph.edges['intra_module'].data['bit_position'] = graph.edges['intra_module'].data[
            'bit_position'].unsqueeze(1)

        reverse_weight = None
        if self.flag_reverse or self.flag_path_supervise:
            reverse_weight = th.zeros((graph.number_of_edges('reverse'), 1), dtype=th.float).to(self.device)
            graph.edges['reverse'].data['weight'] = reverse_weight

        # propagate messages in the topological order, from PIs to POs
        for i, nodes in enumerate(topo):
            # for PIs
            if i == 0:
                graph.apply_nodes(self.nodes_func_pi, nodes)
                continue

            level_data = self.mtde_forward_level_data(graph, graph_info, i, nodes, use_cache)
            nodes_gate = level_data['nodes_gate']
            nodes_module = level_data['nodes_module']

            if len(nodes_gate) != 0:
                eids = level_data['gate_eids']
                graph.apply_edges(self.edge_msg_gate, eids, etype='intra_gate')
                graph.pull(nodes_gate, self.message_func_gate, self.reduce_func_attn_g, self.nodes_func_gate,
                           etype='intra_gate')

                eids_r = level_data['gate_reverse_eids']
                graph.apply_edges(self.edge_msg_gate_weight, eids, etype='intra_gate')
                graph.edges['reverse'].data['weight'][eids_r] = graph.edges['intra_gate'].data['weight'][eids]
            if len(nodes_module) != 0:
                eids = level_data['module_eids']
                graph.pull(nodes_module, fn.copy_e('bit_position', 'pos'), fn.mean('pos', 'width2'),
                           etype='intra_module')
                graph.apply_edges(self.edge_msg_module, eids, etype='intra_module')
                graph.pull(nodes_module, self.message_func_module, self.reduce_func_attn_m,
                           self.nodes_func_module, etype='intra_module')

                graph.apply_edges(self.edge_msg_module_weight, eids, etype='intra_module')
                eids_r = level_data['module_reverse_eids']
                graph.edges['reverse'].data['weight'][eids_r] = graph.edges['intra_module'].data['weight'][eids]

        h_gnn = graph.ndata['h'][PO_mask]
        if self.flag_reverse or self.flag_path_supervise:
            reverse_weight = graph.edges['reverse'].data['weight']
        return h_gnn, reverse_weight

    def mtde_scatter_attention(self, z, score, dst_pos, num_dst):
        aggregated, alpha = segment_softmax_aggregate(z, score, dst_pos, num_dst)
        return self.activation(aggregated), alpha

    def mtde_forward_scatter_once(self, graph, graph_info):
        cache = graph_info.get('mtde_forward_cache')
        if cache is None:
            raise ValueError('MTDE forward scatter requires cached topology data')

        h = graph.ndata['h']
        reverse_weight = th.zeros((graph.number_of_edges('reverse'), 1),
                                  dtype=th.float, device=self.device)

        for i, nodes in enumerate(graph_info['topo']):
            if i == 0:
                pi_input = th.cat((graph.ndata['delay'][nodes], graph.ndata['value'][nodes]), dim=1)
                h[nodes] = self.activation(self.mlp_pi(pi_input))
                continue

            level_data = cache[i]
            nodes_gate = level_data['nodes_gate']
            if len(nodes_gate) != 0:
                gate_src = level_data['gate_src']
                gate_dst = level_data['gate_dst']
                gate_z = self.linear_neigh_gate(th.cat((h[gate_src], graph.ndata[self.feat_name2][gate_dst]), dim=1))
                gate_score = self.activation2(th.matmul(gate_z, self.attention_vector_g))
                gate_h, gate_alpha = self.mtde_scatter_attention(
                    gate_z, gate_score, level_data['gate_dst_pos'], len(nodes_gate))
                h[nodes_gate] = gate_h
                reverse_weight[level_data['gate_reverse_eids']] = gate_alpha

            nodes_module = level_data['nodes_module']
            if len(nodes_module) != 0:
                module_src = level_data['module_src']
                module_dst = level_data['module_dst']
                module_bit_position = graph.edges['intra_module'].data['bit_position'][
                    level_data['module_eids']].reshape(-1, 1)
                module_dst_feat = th.cat((
                    module_bit_position,
                    graph.ndata[self.feat_name2][module_dst],
                    level_data['module_width2'][level_data['module_dst_pos']],
                ), dim=1)
                module_z = self.linear_neigh_module(th.cat((h[module_src], module_dst_feat), dim=1))
                module_score = self.activation2(th.matmul(module_z, self.attention_vector_m))
                module_h, module_alpha = self.mtde_scatter_attention(
                    module_z, module_score, level_data['module_dst_pos'], len(nodes_module))
                h[nodes_module] = module_h
                reverse_weight[level_data['module_reverse_eids']] = module_alpha

        graph.ndata['h'] = h
        graph.edges['reverse'].data['weight'] = reverse_weight
        return h[graph_info['POs']], reverse_weight

    def mtde_forward_dgl(self, graph, graph_info):
        mtde_forward_cache = getattr(options, 'mtde_forward_cache', 'off')
        if mtde_forward_cache == 'off':
            return self.mtde_forward_once(graph, graph_info, use_cache=False)
        if mtde_forward_cache == 'cache':
            return self.mtde_forward_once(graph, graph_info, use_cache=True)
        if mtde_forward_cache == 'compare':
            with graph.local_scope():
                ref_h, ref_weight = self.mtde_forward_once(graph, graph_info, use_cache=False)
                ref_h = ref_h.clone()
                ref_weight = ref_weight.clone() if ref_weight is not None else None

            cached_h, cached_weight = self.mtde_forward_once(graph, graph_info, use_cache=True)
            if not th.allclose(ref_h, cached_h, rtol=1e-4, atol=1e-5):
                max_abs = th.max(th.abs(ref_h - cached_h)).item()
                raise ValueError('MTDE forward cache mismatch: h_gnn max_abs={:.6g}'.format(max_abs))
            if ref_weight is not None and not th.allclose(ref_weight, cached_weight, rtol=1e-4, atol=1e-5):
                max_abs = th.max(th.abs(ref_weight - cached_weight)).item()
                raise ValueError('MTDE forward cache mismatch: reverse.weight max_abs={:.6g}'.format(max_abs))
            return cached_h, cached_weight
        raise ValueError('Unknown MTDE forward cache mode: {}'.format(mtde_forward_cache))

    def mtde_forward(self, graph, graph_info):
        mtde_forward_impl = getattr(options, 'mtde_forward_impl', 'dgl')
        if mtde_forward_impl == 'dgl':
            return self.mtde_forward_dgl(graph, graph_info)[0]
        if mtde_forward_impl == 'scatter':
            return self.mtde_forward_scatter_once(graph, graph_info)[0]
        if mtde_forward_impl == 'compare':
            with th.no_grad(), graph.local_scope():
                ref_h, ref_weight = self.mtde_forward_dgl(graph, graph_info)
                ref_h = ref_h.clone()
                ref_weight = ref_weight.clone()
            scatter_h, scatter_weight = self.mtde_forward_scatter_once(graph, graph_info)
            if not th.allclose(ref_h, scatter_h, rtol=1e-4, atol=1e-5):
                max_abs = th.max(th.abs(ref_h - scatter_h)).item()
                raise ValueError('MTDE forward scatter mismatch: h_gnn max_abs={:.6g}'.format(max_abs))
            if not th.allclose(ref_weight, scatter_weight, rtol=1e-4, atol=1e-5):
                max_abs = th.max(th.abs(ref_weight - scatter_weight)).item()
                raise ValueError('MTDE forward scatter mismatch: reverse.weight max_abs={:.6g}'.format(max_abs))
            return scatter_h
        raise ValueError('Unknown MTDE forward implementation: {}'.format(mtde_forward_impl))

    def mtde_forward_eval_cached(self, graph, graph_info):
        case_key = graph_info.get('eval_case_key')
        can_cache = case_key is not None and not self.training and not th.is_grad_enabled()
        cache = graph_info.get('_mtde_eval_case_cache') if can_cache else None
        if cache is not None and cache['key'] == case_key:
            graph.ndata['h'] = cache['h']
            if cache['reverse_weight'] is not None:
                graph.edges['reverse'].data['weight'] = cache['reverse_weight']
            return cache['h'][graph_info['POs']]

        h_gnn = self.mtde_forward(graph, graph_info)
        if can_cache:
            reverse_weight = None
            if 'reverse' in graph.etypes and 'weight' in graph.edges['reverse'].data:
                reverse_weight = graph.edges['reverse'].data['weight'].detach().clone()
            graph_info['_mtde_eval_case_cache'] = {
                'key': case_key,
                'h': graph.ndata['h'].detach().clone(),
                'reverse_weight': reverse_weight,
            }
        return h_gnn

    def forward(self, graph, graph_info,flag_meta=False,stage_time=None):
        timed = stage_time is not None

        with (graph.local_scope()):
            # stage 1: MDTE
            if timed:
                start = time()
            h_gnn = self.mtde_forward_eval_cached(graph, graph_info)
            if timed:
                stage_time['MTDE_forward'] += time() - start

            h = h_gnn

            rst = None
            prob_sum, prob_dev, prob_ce = th.tensor([0.0]), th.tensor([0.0]), th.tensor([0.0])
            POs_criticalprob = None
            rst_residual, path_inputdelay = None, None
            metadata = None

            if timed:
                start = time()
            nodes_emb = self.fse_node_embeddings(graph, graph_info)
            if timed:
                stage_time['FSE'] += time() - start

            POs = graph_info['POs']

            if timed:
                start = time()
            nodes_prob = self.prop_backward(graph, graph_info)
            if timed:
                stage_time['MTDE_backward'] += time() - start

            graph.ndata['hp'] = nodes_prob
            if self.flag_path_supervise:
                graph.ndata['id'] = th.zeros((graph.number_of_nodes(), 1), dtype=th.int64).to(self.device)
                graph.ndata['id'][POs] = graph_info['PO_cols'].unsqueeze(-1)
                empty_prob = nodes_prob.new_zeros((graph.number_of_nodes(), 1))
                graph.ndata['prob_sum'] = empty_prob
                graph.ndata['prob_dev'] = empty_prob.clone()
                graph.ndata['prob_ce'] = empty_prob.clone()
                graph.pull(POs, self.message_func_prob, self.reduce_func_prob, etype='pi2po')
                POs_criticalprob = graph.ndata['prob_sum'][POs]
                prob_sum = graph.ndata['prob_sum'][POs]
                prob_dev = graph.ndata['prob_dev'][POs]
                prob_ce = graph.ndata['prob_ce'][POs]

            # stage 2: generate the FSE
            if timed:
                start = time()
            h_global, probinfo_etp, weight_etp, minmax = self.fse_components(
                nodes_prob, nodes_emb, graph_info)
            h_etp = self.mlp_probinfo(probinfo_etp)
            h_global = th.cat((h_global, h_etp), dim=1)

            w = self.mlp_w2(th.cat((weight_etp, minmax), dim=1))
            h = th.cat((h, w * h_global), dim=1)
            if timed:
                stage_time['FSE'] += time() - start

            # stage 3: generate CPE
            path_emb, path_lengths, path_inputdelay = self.path_embedding(
                graph, graph_info, nodes_emb, stage_time if timed else None)
            rst_residual = self.mlp_out_residual(path_emb)
            delay_emb = self.linear_delay(path_inputdelay)
            h = th.cat((h, path_emb, delay_emb), dim=1)

            # stage 4: projection
            if timed:
                start = time()
            rst = self.mlp_out_new(h)
            if timed:
                stage_time['proj'] += time() - start

            if flag_meta:
                PIs_mask = graph.ndata['is_pi'] == 1
                metadata = extract_endpoint_metadata(graph, nodes_prob, PIs_mask)
                metadata['critical_input_arrival'] = path_inputdelay.squeeze(1)
                metadata['critical_path_length'] = path_lengths
                for key, value in metadata.items():
                    metadata[key] = value.reshape((len(value), 1))

            output = (rst, rst_residual, path_inputdelay, prob_sum, prob_dev, prob_ce, POs_criticalprob, metadata)
            return output + (stage_time,) if timed else output

class ACCNN(nn.Module):

    def __init__(self,
                 infeat_dim,
                 hidden_dim,
                 flag_homo=False):
        super(ACCNN, self).__init__()

        self.flag_homo = flag_homo

        if self.flag_homo:
            self.mlp_agg = MLP(hidden_dim + infeat_dim, int(hidden_dim / 2), hidden_dim)
        else:
            self.mlp_agg_module = MLP(hidden_dim + infeat_dim, int(hidden_dim / 2), hidden_dim)
            self.mlp_agg_gate = MLP(hidden_dim + infeat_dim, int(hidden_dim / 2), hidden_dim)
        self.mlp_pi = MLP(4, int(hidden_dim / 2), hidden_dim)
        self.mlp_out = MLP(hidden_dim, hidden_dim, 1)

    def nodes_func(self, nodes):
        m_self = nodes.data['feat']
        h = th.cat((nodes.data['neigh'], m_self), dim=1)
        h = self.mlp_agg(h)

        return {'h': h}

    def nodes_func_module(self, nodes):

        m_self = nodes.data['feat']
        h = th.cat((nodes.data['neigh'], m_self), dim=1)
        h = self.mlp_agg_module(h)

        return {'h': h}

    def nodes_func_gate(self, nodes):

        m_self = nodes.data['feat']
        h = th.cat((nodes.data['neigh'], m_self), dim=1)
        h = self.mlp_agg_gate(h)

        return {'h': h}

    def nodes_func_pi(self, nodes):
        h = th.cat((nodes.data['delay'], nodes.data['value']), dim=1)
        h = self.mlp_pi(h)

        return {'h': h}

    def forward(self, graph, graph_info):

        topo = graph_info['topo']
        PO_mask = graph_info['POs']
        prob_sum, prob_dev = th.tensor([0.0]), th.tensor([0.0])
        POs_criticalprob = None

        with (graph.local_scope()):
            # propagate messages in the topological order, from PIs to POs
            for i, nodes in enumerate(topo):
                # for PIs
                if i == 0:
                    graph.apply_nodes(self.nodes_func_pi, nodes)
                elif graph_info['is_heter']:
                    isModule_mask = graph.ndata['is_module'][nodes] == 1
                    isGate_mask = graph.ndata['is_module'][nodes] == 0
                    nodes_gate = nodes[isGate_mask]
                    nodes_module = nodes[isModule_mask]
                    if len(nodes_gate) != 0: graph.pull(nodes_gate, fn.copy_u('h', 'm'), fn.mean('m', 'neigh'),
                                                        self.nodes_func_gate, etype='intra_gate')
                    if len(nodes_module) != 0: graph.pull(nodes_module, fn.copy_u('h', 'm'), fn.mean('m', 'neigh'),
                                                          self.nodes_func_module, etype='intra_module')
                else:
                    graph.pull(nodes, fn.copy_u('h', 'm'), fn.mean('m', 'neigh'), self.nodes_func)

            h = graph.ndata['h'][PO_mask]
            rst = self.mlp_out(h)

            return rst, prob_sum, prob_dev, POs_criticalprob


class PathModel(nn.Module):
    def __init__(self, infeat_dim, hidden_dim, device, impl_choice=0):
        super(PathModel, self).__init__()
        self.impl_choice = impl_choice
        self.device = device
        if impl_choice == 0:
            self.model = MLP(infeat_dim, hidden_dim, hidden_dim, 1)

    def forward(self, POs_feat):
        rst = th.zeros((len(POs_feat), 1), dtype=th.float).to(self.device)
        for i, feat in enumerate(POs_feat):
            path_delays = self.model(feat)
            max_delay = th.max(path_delays)
            rst[i] = max_delay
        return rst


class Graphormer(nn.Module):

    def __init__(self,
                 infeat_dim,
                 feat_dim=512,
                 hidden_dim=1024,
                 num_heads=4):
        super(Graphormer, self).__init__()

        self.layers = th.nn.ModuleList([
            dgl.nn.GraphormerLayer(
                feat_size=feat_dim,  # the dimension of the input node features
                hidden_size=hidden_dim,  # the dimension of the hidden layer
                num_heads=num_heads,  # the number of attention heads
                dropout=0.1,  # the dropout rate
                activation=th.nn.ReLU(),  # the activation function
                norm_first=False,  # whether to put the normalization before attention and feedforward
            )
            for _ in range(6)
        ])

        self.mlp_n = MLP(infeat_dim, int(feat_dim / 2), feat_dim)
        self.mlp_out = MLP(feat_dim, int(feat_dim / 2), 1)

        self.degree_encoder = dgl.nn.DegreeEncoder(
            max_degree=8,  # the maximum degree to cut off
            embedding_dim=feat_dim  # the dimension of the degree embedding
        )
        self.spatial_encoder = dgl.nn.SpatialEncoder(
            max_dist=5,  # the maximum distance between two nodes
            num_heads=num_heads,  # the number of attention heads
        )

    def forward(self, graphs_info):
        deg_emb = self.degree_encoder(th.stack((graphs_info['in_degree'], graphs_info['out_degree'])))
        node_feat = self.mlp_n(graphs_info['node_feat_new'])
        num_graphs, max_num_nodes, _ = node_feat.shape
        # node feature + degree encoding as input
        x = node_feat + deg_emb

        # spatial encoding and path encoding serve as attention bias

        spatial_encoding = self.spatial_encoder(graphs_info['dist'])
        attn_bias = th.rand(num_graphs, max_num_nodes, max_num_nodes, self.num_heads)
        # attn_bias = spatial_encoding
        for layer in self.layers:
            x = layer(
                x,
                attn_mask=graphs_info['attn_mask'],
                attn_bias=attn_bias,
            )

        res = None
        for i in range(x.shape[0]):
            res = cat_tensor(res, x[i][graphs_info['POs_mask'][i]])
            # print(graphs_info['POs_mask'][i].shape,x[i][graphs_info['POs_mask'][i]].shape)
        res = self.mlp_out(res)
        return res
