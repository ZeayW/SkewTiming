import dgl
import torch as th
from dgl import function as fn
from torch import nn


class GraphProp(nn.Module):
    def __init__(self, featname, flag_distance):
        super().__init__()
        self.featname = featname
        self.flag_distance = flag_distance

    def nodes_func_distance(self, nodes):
        return {self.featname: nodes.data[self.featname] + nodes.data['delay']}

    def message_func_distance(self, edges):
        return {'m': edges.src[self.featname] + edges.src['intra_delay']}

    def forward(self, graph, topo):
        with graph.local_scope():
            for nodes in topo[1:]:
                if self.flag_distance:
                    graph.pull(
                        nodes,
                        self.message_func_distance,
                        fn.max('m', self.featname),
                        self.nodes_func_distance,
                    )
                else:
                    graph.pull(nodes, fn.copy_u(self.featname, 'm'), fn.max('m', self.featname))
            return graph.ndata[self.featname]


def is_heter(graph):
    return len(graph.etypes) > 1 or len(graph.ntypes) > 1


def heter2homo(graph):
    src_module, dst_module = graph.edges(etype='intra_module', form='uv')
    src_gate, dst_gate = graph.edges(etype='intra_gate', form='uv')
    homo_graph = dgl.graph(
        (th.cat([src_module, src_gate]), th.cat([dst_module, dst_gate])),
        num_nodes=graph.num_nodes(),
        device=graph.device,
    )
    for key, value in graph.ndata.items():
        homo_graph.ndata[key] = value
    return homo_graph


def gen_topo(graph, flag_reverse=False):
    graph = heter2homo(graph) if is_heter(graph) else graph
    return dgl.topological_nodes_generator(graph, reverse=flag_reverse)


def edge_source_level_ratings(src, dst, node_levels):
    """Rank unique source levels per destination from deepest to shallowest."""
    if src.numel() == 0:
        return th.zeros((0, 1), dtype=th.float, device=src.device)

    src_levels = node_levels.reshape(-1)[src].to(dtype=th.int64)
    dst = dst.to(dtype=th.int64)
    max_level = int(src_levels.max().item())
    sort_key = dst * (max_level + 1) + (max_level - src_levels)
    order = th.argsort(sort_key, stable=True)
    sorted_dst = dst[order]
    sorted_levels = src_levels[order]

    new_destination = th.ones_like(sorted_dst, dtype=th.bool)
    new_destination[1:] = sorted_dst[1:] != sorted_dst[:-1]
    new_level = new_destination.clone()
    new_level[1:] |= sorted_levels[1:] != sorted_levels[:-1]

    cumulative_levels = th.cumsum(new_level.to(dtype=th.int64), dim=0)
    destination_ids = th.cumsum(new_destination.to(dtype=th.int64), dim=0) - 1
    destination_offsets = cumulative_levels[new_destination] - 1
    sorted_ratings = cumulative_levels - destination_offsets[destination_ids]

    ratings = th.empty_like(sorted_ratings)
    ratings[order] = sorted_ratings
    return ratings.to(dtype=th.float).unsqueeze(1)


def reverse_graph(graph):
    src, dst = graph.edges()
    reverse = dgl.graph((dst, src), num_nodes=graph.num_nodes(), device=graph.device)
    for key, value in graph.ndata.items():
        reverse.ndata[key] = value
    for key, value in graph.edata.items():
        reverse.edata[key] = value
    return reverse


def graph_filter(graph):
    reverse = reverse_graph(heter2homo(graph))
    topo = dgl.topological_nodes_generator(reverse)
    reverse.ndata['temp'] = th.zeros(
        (graph.number_of_nodes(), 1), dtype=th.float, device=graph.device
    )
    reverse.ndata['temp'][graph.ndata['is_po'] == 1] = 1
    keep_mask = GraphProp('temp', False)(reverse, topo).squeeze(1).bool()
    nodes = th.arange(graph.number_of_nodes(), device=graph.device)
    return nodes[keep_mask], nodes[~keep_mask]


def find_fanin_pis(graph, graph_info):
    device = graph.device
    reverse = reverse_graph(heter2homo(graph))
    node_delays = th.ones(
        (graph.number_of_nodes(), 1), dtype=th.float, device=device
    )
    reverse.ndata['intra_delay'] = node_delays
    topo = dgl.topological_nodes_generator(reverse)
    nodes = th.arange(graph.number_of_nodes(), device=device)
    pos = nodes[graph.ndata['is_po'] == 1]

    reverse.ndata['po_onehot'] = th.full(
        (reverse.number_of_nodes(), len(pos)), -10000.0, dtype=th.float, device=device
    )
    for index, po in enumerate(pos.tolist()):
        reverse.ndata['po_onehot'][po, index] = 0
    distances = GraphProp('po_onehot', True)(reverse, topo)
    distances[distances < -1000] = 0

    result = {}
    for index, po in enumerate(pos.tolist()):
        po_distances = distances[:, index]
        pi_mask = (po_distances != 0) & (graph.ndata['is_pi'] == 1)
        pis = nodes[pi_mask]
        if pis.numel() == 0:
            result[po] = (0, [])
            continue
        pi_distances = po_distances[pi_mask]
        max_distance = th.max(pi_distances)
        critical_pis = pis[pi_distances == max_distance].tolist()
        result[po] = (max_distance.item(), critical_pis)
    return result


def get_pi2po_edges(graph, graph_info):
    del graph_info  # Kept for compatibility with the historical call site.
    edges = ([], [], [])
    for po, (distance, pis) in find_fanin_pis(graph, {}).items():
        edges[0].extend(pis)
        edges[1].extend([po] * len(pis))
        edges[2].extend([distance] * len(pis))
    return edges
