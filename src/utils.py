import torch as th
import dgl
from torch import nn
from dgl import function as fn
from options import get_options
import matplotlib.pyplot as plt
import pickle
import numpy as np
import matplotlib as mpl
mpl.rcParams['font.size'] = 11          # 全局默认文字
mpl.rcParams['axes.titlesize'] = 12     # 标题
mpl.rcParams['axes.labelsize'] = 11     # 轴标签
mpl.rcParams['xtick.labelsize'] = 11    # x刻度
mpl.rcParams['ytick.labelsize'] = 11    # y刻度
mpl.rcParams['legend.fontsize'] = 11    # 图例

device = th.device("cuda:0" if th.cuda.is_available()  else "cpu")

class GraphProp(nn.Module):

    def __init__(self,featname,flag_distance):
        super(GraphProp, self).__init__()
        self.featname = featname
        self.flag_distance = flag_distance
    #
    def nodes_func_distance(self,nodes):
        h = nodes.data[self.featname] + nodes.data['delay']
        return {self.featname:h}
    #
    def message_func_distance(self,edges):
        return {'m':edges.src[self.featname] + edges.src['intra_delay']}
    def forward(self, graph,topo):
        with graph.local_scope():
            #propagate messages in the topological order, from PIs to POs
            for i, nodes in enumerate(topo[1:]):
                if self.flag_distance:
                    graph.pull(nodes, self.message_func_distance, fn.max('m', self.featname),self.nodes_func_distance)
                else:
                    graph.pull(nodes, fn.copy_u(self.featname, 'm'), fn.max('m', self.featname))

            return graph.ndata[self.featname]



def is_heter(graph):
    return len(graph._etypes)>1 or len(graph._ntypes)>1

def heter2homo(graph):
    src_module, dst_module = graph.edges(etype='intra_module', form='uv')
    src_gate, dst_gate = graph.edges(etype='intra_gate', form='uv')
    homo_g = dgl.graph((th.cat([src_module, src_gate]), th.cat([dst_module, dst_gate])))


    for key, data in graph.ndata.items():
        homo_g.ndata[key] = graph.ndata[key]

    return homo_g

def gen_topo(graph,flag_reverse=False):
    if is_heter(graph):
        src_module, dst_module = graph.edges(etype='intra_module', form='uv')
        src_gate, dst_gate = graph.edges(etype='intra_gate', form='uv')
        g = dgl.graph((th.cat([src_module,src_gate]), th.cat([dst_module,dst_gate])))
    else:
        g = graph
    topo = dgl.topological_nodes_generator(g,reverse=flag_reverse)

    return topo


def gen_level(graph):

    levels = []
    nodes_list = th.tensor(range(graph.number_of_nodes()),device=graph.device)
    POs_nid = nodes_list[graph.ndata['is_po'] == 1]

    levels.append(POs_nid)
    cur_nodes = POs_nid
    #pre_nid = po_id.item()
    #print(len(cur_nodes))
    while True:
        _,cur_nodes = graph.out_edges(cur_nodes,etype='reverse')

        cur_nodes = th.unique(cur_nodes)
        #print(len(cur_nodes))
        if len(cur_nodes)==0:
            break
        levels.append(cur_nodes)

    return levels
    # #exit()
    # new_levels = []
    # for nodes in levels:
    #     isModule_mask = graph.ndata['is_module'][nodes] == 1
    #     isGate_mask = graph.ndata['is_module'][nodes] == 0
    #     nodes_gate = nodes[isGate_mask]
    #     nodes_module = nodes[isModule_mask]
    #     new_levels.append([nodes_gate,nodes_module,nodes])
    # return new_levels

def add_newEtype(graph,new_etype,new_edges,new_edge_feats):
    graph = graph.to(th.device('cpu'))
    edges_dict = {}
    for etype in graph.etypes:
        if etype == new_etype:
            continue
        edges_dict[('node', etype, 'node')] = graph.edges(etype=etype)
    edges_dict[('node', new_etype, 'node')] = new_edges
    new_graph = dgl.heterograph(edges_dict)

    for key, value in graph.ndata.items():
        new_graph.ndata[key] = value
    for etype in graph.etypes:
        if etype == new_etype:
            continue
        for key, value in graph.edges[etype].data.items():
            new_graph.edges[etype].data[key] = value

    for key,value in new_edge_feats.items():
        new_graph.edges[new_etype].data[key] = value

    return new_graph

def get_pi2po_edges(graph,graph_info):
    new_edges = ([], [],[])
    edges_weight = []
    po2pis = find_faninPIs(graph, graph_info)

    for po, (distance,pis) in po2pis.items():
        new_edges[0].extend(pis)
        new_edges[1].extend([po] * len(pis))
        # if len(pis) != 0:
        #     edges_weight.extend([1 / len(pis)] * len(pis))

    return new_edges

def add_pi2po_edges(graph,graph_info):

    new_edges,edges_weight = get_pi2po_edges(graph,graph_info)
    new_edges_feat = {
        'prob':th.tensor(edges_weight, dtype=th.float).unsqueeze(1)
    }
    new_graph = add_newEtype(graph,'pi2po',new_edges,new_edges_feat)

    return new_graph



def add_reverse_edges(graph):
    if is_heter(graph):
        edges_g = graph.edges(etype='intra_gate')
        edges_m = graph.edges(etype='intra_module')
        reverse_edges = (th.cat((edges_g[1],edges_m[1])), th.cat((edges_g[0],edges_m[0])))
        new_graph = add_newEtype(graph,'reverse',reverse_edges,{})

        forward_edges = (reverse_edges[1], reverse_edges[0])
        new_graph = add_newEtype(new_graph, 'forward', forward_edges, {})

    else:
        new_graph = dgl.heterograph(
            {
                ('node', 'edge', 'node'): graph.edges(),
                ('node', 'edge_r', 'node'): (graph.edges()[1],graph.edges()[0]),
            }
        )
        for key, value in graph.ndata.items():
            new_graph.ndata[key] = value
        for key, value in graph.edata.items():
            new_graph.edges['edge'].data[key] = value
            new_graph.edges['edge_r'].data[key] = value

    return new_graph



def reverse_graph(g):
    edges = g.edges()
    reverse_edges = (edges[1], edges[0])

    rg = dgl.graph(reverse_edges, num_nodes=g.num_nodes())
    for key, value in g.ndata.items():
        # print(key,value)
        rg.ndata[key] = value
    for key, value in g.edata.items():
        # print(key,value)
        rg.edata[key] = value
    return rg


def graph_filter(graph):
    homo_graph = heter2homo(graph)
    homo_graph_r = reverse_graph(homo_graph)
    topo_r = dgl.topological_nodes_generator(homo_graph_r)
    graphProp_model = GraphProp('temp',False)
    homo_graph_r.ndata['temp'] = th.zeros((graph.number_of_nodes(), 1), dtype=th.float)
    homo_graph_r.ndata['temp'][graph.ndata['is_po'] == 1] = 1
    fitler_mask = graphProp_model(homo_graph_r, topo_r).squeeze(1)
    # print(fitler_mask.shape,th.sum(fitler_mask))
    remove_nodes = th.tensor(range(graph.number_of_nodes()))[fitler_mask == 0]
    remain_nodes = th.tensor(range(graph.number_of_nodes()))[fitler_mask == 1]
    # print('\t filtering: remove {} useless nodes'.format(len(remove_nodes)))

    return remain_nodes, remove_nodes

def get_intranode_delay(ntype):
    return 1
    if ntype == 'mux':
        return 0.5
    elif ntype=='add':
        return 1.5
    elif ntype in ['eq','lt','ne','decoder','encoder']:
        return 2
    else:
        return 1

def find_faninPIs(graph,graph_info):

    nodes_type = graph_info['ntype']
    nodes_name = graph_info['nodes_name']
    nodes_intradelay = [get_intranode_delay(t) for t in nodes_type]

    fanin_pis = {}
    homo_graph = heter2homo(graph)
    homo_graph_r = reverse_graph(homo_graph)
    homo_graph_r.ndata['intra_delay'] = th.tensor(nodes_intradelay,dtype=th.float).unsqueeze(1).to(device)
    topo_r = dgl.topological_nodes_generator(homo_graph_r)
    topo_r = [l.to(device) for l in topo_r]
    graphProp_model = GraphProp('po_onehot',True).to(device)
    nodes_list = th.tensor(range(graph.number_of_nodes())).to(device)
    POs = nodes_list[graph.ndata['is_po'] == 1].cpu().numpy().tolist()


    homo_graph_r.ndata['po_onehot'] = -10000*th.ones((homo_graph_r.number_of_nodes(), len(POs)), dtype=th.float).to(device)
    for i, po in enumerate(POs):
        homo_graph_r.ndata['po_onehot'][po][i] = 0
    nodes2POs_distance = graphProp_model(homo_graph_r, topo_r).squeeze(1)
    nodes2POs_distance[nodes2POs_distance < -1000] = 0
    if len(POs)==1:
        nodes2POs_distance =  nodes2POs_distance.unsqueeze(1)
    for i, po in enumerate(POs):

        pi_mask = th.logical_and((nodes2POs_distance[:, [i]] !=0).squeeze(1), graph.ndata['is_pi'] == 1)
        pis = nodes_list[pi_mask].cpu().numpy().tolist()
        #print('#PI:', len(pis))
        if len(pis) == 0:
            fanin_pis[po] = (0,[])
        else:
            pis_distance = nodes2POs_distance[:,[i]][pi_mask].squeeze(1)
            max_distance = th.max(pis_distance)
            critical_pis = th.tensor(pis).to(device)[pis_distance==max_distance].cpu().numpy().tolist()
            #fanin_pis[po] =  nodes_list[pi_mask].numpy().tolist()
            fanin_pis[po] = (max_distance,critical_pis)
            #print(nodes_name[po])
            #print('\t',[(nodes_name[pi][0],dst.item()) for pi,dst in zip(pis,pis_distance)])
            # print('\t',pis_distance)

    return fanin_pis


def filter_list(l, idxs):
    new_l = []
    for i in idxs:
        new_l.append(l[i])

    return new_l



def draw_bar():
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    with open('predict2.pkl','rb') as f:
        labels, labels_hat, ratio = pickle.load(f)
    print('----Loaded')
    error = [(r-1)*100 for r in ratio]
    error = th.tensor(error,device=device)
    x = []
    y = []
    indexs = list(range(-25, 30,5))
    for i, e in enumerate(indexs):
        if i == 0:
            num = len(error[error <= e])
        elif i == len(indexs) - 1:
            num = len(error[error >= e])
        else:
            num = len(error[th.logical_and(error >= e-2.5, error < e + 2.5)])
        x.append(e)
        y.append(num / len(error))
    # # plt.bar(x,y)
    # plt.xlabel('error（%）')
    # plt.ylabel('percentage（%）')
    # plt.bar(x, y, width=0.03)
    # print(x)
    # print(y)
    x = np.array(x)
    y = np.array(y)
    plt.figure(figsize=(7.5, 4.8), dpi=150)
    bar_width = 3
    colors = plt.cm.Blues(0.6 + 0.4 * (y - y.min()) / (y.max() - y.min() + 1e-12))
    plt.bar(x, y, width=bar_width, color=colors, edgecolor='black', linewidth=0.6)

    # 仅设置 x/y 轴字体为 11pt
    ax = plt.gca()
    ax.set_xlabel("误差 x", fontsize=13)
    ax.set_ylabel("占比 y", fontsize=13)
    ax.tick_params(axis='x', labelsize=13)
    ax.tick_params(axis='y', labelsize=13)

    # 柱顶标注
    for xi, yi in zip(x, y):
        plt.text(xi, yi + max(y) * 0.01, f"{yi * 100:.2f}%", ha='center', va='bottom',fontsize=14)

    # y 轴百分比
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 100:.0f}%"))

    # plt.title("误差-占比柱状分布")
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.xticks(x)
    plt.margins(x=0.02)
    plt.tight_layout()
    plt.show()
    print(list(zip(x,y)))

    plt.savefig('bar5.png')


if __name__ == "__main__":
    draw_bar()