"""Torch Module for TimeConv layer"""
import dgl
import torch as th
from torch import nn
from dgl import function as fn

from utils import *
from options import get_options
options = get_options()
#device = th.device("cuda:" + str(options.gpu) if th.cuda.is_available() and options.gpu!=-1 else "cpu")

def cat_tensor(t1,t2):
    if t1 is None:
        return t2
    elif t2 is None:
        return t1
    else:
        return th.cat((t1,t2),dim=0)

def get_nodename(nodes_name,nid):
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
                 global_cat_choice=0,
                 global_info_choice=0,
                 agg_choice=0,
                 attn_choice=1,
                 flag_delay=False,
                 flag_degree=False,
                 flag_width=False,
                 flag_path_supervise=False,
                 flag_reverse=False,
                 flag_splitfeat=False,
                 ):
        super(BPN, self).__init__()

        self.device = device
        self.global_cat_choice = global_cat_choice
        self.global_info_choice = global_info_choice
        self.flag_delay = flag_delay
        self.flag_degree = flag_degree
        self.flag_width = flag_width
        self.flag_path_supervise = flag_path_supervise
        self.flag_reverse = flag_reverse
        self.hidden_dim = hidden_dim
        self.agg_choice = agg_choice
        self.attn_choice = attn_choice
        self.mlp_pi = MLP(4, int(hidden_dim / 2), hidden_dim)
        #self.linear_pi = th.nn.Linear(4, hidden_dim)
        self.mlp_agg = MLP(hidden_dim, int(hidden_dim / 2), hidden_dim)

        if self.global_cat_choice==8: self.mlp_w = MLP(hidden_dim, int(hidden_dim / 2),1)
        if self.global_cat_choice in [9,11]: self.mlp_w2 = MLP(1, hidden_dim, 1)
        if self.global_cat_choice == 10: self.mlp_w2 = MLP(2, hidden_dim, 1)

        new_out_dim = 0
        if self.global_info_choice in [0, 1]:
            new_out_dim += self.hidden_dim
        elif self.global_info_choice==2:
            new_out_dim += 2*self.hidden_dim
        elif self.global_info_choice in [3, 4, 5,7]:
            new_out_dim += self.hidden_dim + 1
        elif self.global_info_choice in [6,9]:
            new_out_dim += self.hidden_dim + 2
        elif  self.global_info_choice in [8]:
            new_out_dim += 2*hidden_dim + 1
        if self.global_cat_choice in [0, 3, 4]:
            new_out_dim += 1
        elif self.global_cat_choice in [1, 5, 7,8,9,10,11]:
            new_out_dim += self.hidden_dim



        if new_out_dim != 0: self.mlp_out_new = MLP(new_out_dim, self.hidden_dim, self.hidden_dim, 1, negative_slope=0.1)


        out_dim = hidden_dim
        if flag_splitfeat:
            self.feat_name1 = 'feat_gate'
            self.feat_name2 = 'feat_module'

            self.mlp_self_gate = MLP(infeat_dim1, int(hidden_dim / 2), hidden_dim)
            self.mlp_self_module = MLP(infeat_dim2, int(hidden_dim / 2), hidden_dim)
            self.infeat_dim2 = infeat_dim2
            self.infeat_dim1 = infeat_dim1
        else:
            self.feat_name1 = 'feat'
            self.feat_name2 = 'feat'
            self.infeat_dim2 = infeat_dim2 + infeat_dim1
            self.infeat_dim1 = infeat_dim2 + infeat_dim1
            self.mlp_self = MLP(self.infeat_dim1, int(hidden_dim / 2), hidden_dim)
            self.mlp_self_module = self.mlp_self
            self.mlp_self_gate = self.mlp_self

        feat_dim_m =  self.infeat_dim2 + 1
        feat_dim_g = self.infeat_dim1
        if self.flag_degree:
            feat_dim_m += 1
            feat_dim_g += 1
        if self.flag_width:
            feat_dim_m +=1

        neigh_dim_m = self.hidden_dim + feat_dim_m
        neigh_dim_g = self.hidden_dim + feat_dim_g

        if self.flag_delay:
            neigh_dim_g += 1
            neigh_dim_m += 1
        # feat_outdim_m = 32
        # feat_outdim_g = 32
        # neigh_dim_m = self.hidden_dim + feat_outdim_m
        # neigh_dim_g = self.hidden_dim + feat_outdim_g
        # self.linear_feat_module = th.nn.Linear(feat_dim_m,feat_outdim_m)
        # self.linear_feat_gate = th.nn.Linear(feat_dim_g, feat_outdim_g)

        self.mlp_neigh_module = MLP(neigh_dim_m, int(hidden_dim / 2), hidden_dim)
        self.mlp_neigh_gate = MLP(neigh_dim_g, int(hidden_dim / 2), hidden_dim)
        self.linear_neigh_module = th.nn.Linear(neigh_dim_m, hidden_dim)
        self.linear_neigh_gate = th.nn.Linear(neigh_dim_g, hidden_dim)

        atnn_dim_m = hidden_dim
        atnn_dim_g = hidden_dim
        self.attention_vector_g = nn.Parameter(th.randn(atnn_dim_g, 1), requires_grad=True)
        self.attention_vector_m = nn.Parameter(th.randn(atnn_dim_m,1),requires_grad=True)

        # self.linear_neigh_gate = self.linear_neigh_module
        # self.mlp_neigh_gate = self.mlp_neigh_module
        # self.attention_vector_g = self.attention_vector_m

        self.mlp_out = MLP(out_dim,hidden_dim,1)
        self.activation = th.nn.LeakyReLU(negative_slope=0)
        self.activation2 = th.nn.LeakyReLU(negative_slope=0.2)


    def nodes_func_module(self,nodes):
        h = self.activation(nodes.data['neigh'])
        #print('m',th.sum(h[h<0]))
        if self.flag_reverse or self.flag_path_supervise:
            return {'h':h,'attn_sum':nodes.data['attn_sum'],'attn_max': nodes.data['attn_max']}
        else:
            return {'h': h}

    def nodes_func_gate(self,nodes):
        #mask = nodes.data['is_po'].squeeze() != 1
        h = self.activation(nodes.data['neigh'])
        #print('g', th.sum(h[h < 0]))
        if (self.flag_reverse or self.flag_path_supervise) and self.attn_choice == 1:
            return {'h': h, 'exp_src_sum': nodes.data['exp_src_sum'], 'exp_src_max': nodes.data['exp_src_max']}
        elif (self.flag_reverse or self.flag_path_supervise) and self.attn_choice == 0:
            return {'h': h, 'attn_sum': nodes.data['attn_sum'],'attn_max': nodes.data['attn_max']}
        else:
            return {'h': h}


    def edge_msg_module_weight(self,edges):

        normalized_attn_e = th.exp(edges.data['attn_e'] - edges.dst['attn_max']) / edges.dst['attn_sum'].squeeze(2)

        return {'weight': normalized_attn_e}

    def edge_msg_module(self, edges):

        #h_dst = th.cat((edges.data['bit_position']/edges.dst['width'],edges.dst[self.feat_name2]),dim=1)
        h_dst = th.cat((edges.data['bit_position'] , edges.dst[self.feat_name2]), dim=1)
        #print(th.sum(edges.data['bit_position']/edges.dst['width'])/len(edges.data['bit_position']))
        if self.flag_degree:
            h_dst = th.cat((h_dst,edges.dst['degree']),dim=1)
        if self.flag_width:
            h_dst = th.cat((h_dst,edges.dst['width2']),dim=1)

        #h_dst = self.linear_feat_module(h_dst)

        z = th.cat((edges.src['h'],h_dst),dim=1)
        if self.flag_delay:
            z = th.cat((z,edges.src['delay']),dim=1)

        #z = self.mlp_neigh_module(z)
        z = self.linear_neigh_module(z)
        e = th.matmul(z, self.attention_vector_m)
        e = self.activation2(e)
        return {'attn_e': e,'z':z}


    def message_func_module(self,edges):
        #m = th.cat((edges.src['h'], edges.data['bit_position'].unsqueeze(1)), dim=1)
        m = edges.data['z']
        rst = {'m':m,'attn_e':edges.data['attn_e']}
        return rst

    def reduce_func_attn_m(self,nodes):
        #h_pos = th.mean(nodes.mailbox['pos'],dim=1)
        if self.flag_reverse or self.flag_path_supervise:
            attn_e = nodes.mailbox['attn_e']
            max_attn_e = th.max(attn_e, dim=1).values
            attn_e = attn_e - max_attn_e.unsqueeze(1)
            attn_e_exp = th.exp(attn_e)
            attn_exp_sum = th.sum(attn_e_exp, dim=1).unsqueeze(1)
            alpha = attn_e_exp / attn_exp_sum
            h = th.sum(alpha * nodes.mailbox['m'], dim=1)
            return {'neigh':h,'attn_sum':attn_exp_sum,'attn_max':max_attn_e}
        else:
            alpha = th.softmax(nodes.mailbox['attn_e'], dim=1)
            h = th.sum(alpha * nodes.mailbox['m'], dim=1)
            #print(th.mean(nodes.mailbox['pos'],dim=1),h.shape)

            return {'neigh': h}

    def reduce_func_attn_g(self,nodes):

        if self.flag_reverse or self.flag_path_supervise:
            attn_e = nodes.mailbox['attn_e']
            max_attn_e = th.max(attn_e, dim=1).values
            attn_e = attn_e - max_attn_e.unsqueeze(1)
            attn_e_exp = th.exp(attn_e)
            attn_exp_sum = th.sum(attn_e_exp, dim=1).unsqueeze(1)
            alpha = attn_e_exp / attn_exp_sum
            h = th.sum(alpha * nodes.mailbox['m'], dim=1)
            return {'neigh':h,'attn_sum':attn_exp_sum,'attn_max':max_attn_e}
        else:
            alpha = th.softmax(nodes.mailbox['attn_e'], dim=1)
            h = th.sum(alpha * nodes.mailbox['m'], dim=1)
            return {'neigh': h}

    def reduce_func_mean(self,nodes):
        return {'neigh':th.mean(nodes.mailbox['m'],dim=1),'pos':th.mean(nodes.mailbox['pos'],dim=1)}


    def edge_msg_gate(self, edges):
        h_dst = edges.dst[self.feat_name2]
        if self.flag_degree:
            h_dst = th.cat((h_dst,edges.dst['degree']),dim=1)

        #h_dst = self.linear_feat_gate(h_dst)

        z = th.cat((edges.src['h'], h_dst),dim=1)
        if self.flag_delay:
            z = th.cat((z,edges.src['delay']),dim=1)
        #z = self.mlp_neigh_gate(z)
        z = self.linear_neigh_gate(z)
        e = th.matmul(z, self.attention_vector_g)
        e = self.activation2(e)

        return {'attn_e': e,'z':z}

    def edge_msg_gate_weight(self,edges):

        if self.attn_choice == 0:
            weight = th.exp(edges.data['attn_e'] - edges.dst['attn_max']) / edges.dst['attn_sum'].squeeze(2)
        elif self.attn_choice == 1:
            msg = edges.src['h'] - edges.dst['exp_src_max']
            weight = th.mean(th.exp(msg) / edges.dst['exp_src_sum'], dim=1)
        else:
            assert False

        return {'weight': weight}

    def message_func_gate(self,edges):
        m = edges.data['z']
        return {'m':m,'attn_e':edges.data['attn_e']}

    def reduce_func_smoothmax(self, nodes):
        msg = nodes.mailbox['m']
        #msg = msg - th.max(msg)
        # msg = msg - 1000
        weight = th.softmax(msg, dim=1)
        #criticality = th.mean(weight,dim=2)
        exp_src_max = th.max(msg,dim=1).values
        exp_src_sum = th.sum(th.exp(msg-exp_src_max.unsqueeze(1)),dim=1)

        if self.flag_reverse or self.flag_path_supervise:
            return {'neigh': (msg * weight).sum(1),'exp_src_sum':exp_src_sum,'exp_src_max':exp_src_max}
        else:
            return {'neigh': (msg * weight).sum(1)}

    def message_func_reverse(self,edges):

        prob = edges.src['hp'] * edges.data['weight']
        dst = edges.src['hd'] + 1

        return {'mp': prob, 'dst': dst}

    def reduce_fun_reverse(self,nodes):
        return {'hp':th.sum(nodes.mailbox['mp'],dim=1),'hd':th.max(nodes.mailbox['dst'],dim=1).values}

    def message_func_delay(self,edges):
        return {'md': edges.src['delay'],'w':edges.data['weight']}
        #return {'md':edges.src['delay']}

    def reduce_func_delay_g(self,nodes):

        #input_delay = th.max(nodes.mailbox['md'],dim=1).values
        delay = th.sum(nodes.mailbox['md'] * nodes.mailbox['w'], dim=1)
        return {'delay':delay}

    def reduce_func_delay_m(self,nodes):

        #input_delay = th.max(nodes.mailbox['md'], dim=1).values
        delay = th.sum(nodes.mailbox['md']*nodes.mailbox['w'],dim=1)
        return {'delay':delay}


    def message_func_prob(self, edges):
        msg = th.gather(edges.src['hp'], dim=1, index=edges.dst['id'])
        pi_feat = edges.src['delay']
        return {'mp': msg,'mi':pi_feat}

    def nodes_func_pi(self,nodes):
        h = th.cat((nodes.data['delay'],nodes.data['value']),dim=1)
        h = self.mlp_pi(h)
        #h = self.linear_pi(h)

        h = self.activation(h)
        #mask = nodes.data['is_po'].squeeze() != 1
        #h[mask] = self.activation(h[mask])

        return {'h':h}

    def reduce_func_prob(self,nodes):
        prob_sum = th.sum(nodes.mailbox['mp'],dim=1)
        #prob_sum = th.sum(nodes.mailbox['mp']*nodes.mailbox['mi'], dim=1)
        prob_max = th.max(nodes.mailbox['mp'],dim=1).values
        prob_mean = th.mean(nodes.mailbox['mp'], dim=1).unsqueeze(1)
        prob_dev = th.sum(th.abs(nodes.mailbox['mp']-prob_mean),dim=1)
        #prob_dev = th.sum(nodes.mailbox['mp']*th.log(nodes.mailbox['mp']+1e-10),dim=1)

        #prob_dev = th.sum(th.pow(nodes.mailbox['ml'] - prob_mean,2), dim=1)

        return {'prob_max':prob_max,'prob_sum':prob_sum,'prob_dev':prob_dev}

    def prop_delay(self,graph,graph_info):
        topo = graph_info['topo']
        for i, nodes in enumerate(topo[1:]):
                isModule_mask = graph.ndata['is_module'][nodes] == 1
                isGate_mask = graph.ndata['is_module'][nodes] == 0
                nodes_gate = nodes[isGate_mask]
                nodes_module = nodes[isModule_mask]
                if len(nodes_gate)!=0:
                    #eids = graph.in_edges(nodes_gate, form='eid', etype='intra_gate')
                    graph.pull(nodes_gate, self.message_func_delay, self.reduce_func_delay_g, etype='intra_gate')
                    #graph.apply_edges(self.edge_msg_delay_ratio_g, eids, etype='intra_gate')
                if len(nodes_module) != 0:
                    #eids = graph.in_edges(nodes_module, form='eid', etype='intra_module')
                    graph.pull(nodes_module, self.message_func_delay, self.reduce_func_delay_m, etype='intra_module')
                    #graph.apply_edges(self.edge_msg_delay_ratio_m,eids,etype='intra_module')
        return graph.ndata['delay'], graph.ndata['input_delay']

    def prop_backward(self,graph,graph_info):
        topo_r = graph_info['topo_r']
        with graph.local_scope():
            for i, nodes in enumerate(topo_r[1:]):
                graph.pull(nodes, self.message_func_reverse, self.reduce_fun_reverse, etype='reverse')
            return graph.ndata['hp'], graph.ndata['hd']

    def forward(self,graph,graph_info):

        topo = graph_info['topo']
        PO_mask = graph_info['POs']

        with (graph.local_scope()):
            graph.edges['intra_module'].data['bit_position'] = graph.edges['intra_module'].data['bit_position'].unsqueeze(1)

            if self.flag_reverse or self.flag_path_supervise:
                graph.edges['reverse'].data['weight'] = th.zeros((graph.number_of_edges('reverse'), 1),
                                                                 dtype=th.float).to(self.device)
            #propagate messages in the topological order, from PIs to POs
            for i, nodes in enumerate(topo):
                isModule_mask = graph.ndata['is_module'][nodes] == 1
                isGate_mask = graph.ndata['is_module'][nodes] == 0

                # for PIs
                if i==0:
                    graph.apply_nodes(self.nodes_func_pi,nodes)
                # for other nodes
                else:
                    nodes_gate = nodes[isGate_mask]
                    nodes_module = nodes[isModule_mask]
                    message_func_gate = self.message_func_gate if self.attn_choice == 0 else fn.copy_src('h', 'm')
                    reduce_func_gate = self.reduce_func_attn_g if self.attn_choice == 0 else self.reduce_func_smoothmax

                    if len(nodes_gate) != 0:
                        if self.attn_choice == 0:
                            eids = graph.in_edges(nodes_gate, form='eid', etype='intra_gate')
                            graph.apply_edges(self.edge_msg_gate, eids, etype='intra_gate')
                        graph.pull(nodes_gate, message_func_gate, reduce_func_gate, self.nodes_func_gate,
                                   etype='intra_gate')

                        if self.flag_reverse or self.flag_path_supervise:
                            eids = graph.in_edges(nodes_gate, form='eid', etype='intra_gate')
                            eids_r = graph.out_edges(nodes_gate, form='eid', etype='reverse')
                            graph.apply_edges(self.edge_msg_gate_weight, eids, etype='intra_gate')
                            if self.attn_choice == 0:
                                graph.edges['reverse'].data['weight'][eids_r] = \
                                graph.edges['intra_gate'].data['weight'][eids]
                            elif self.attn_choice == 1:
                                graph.edges['reverse'].data['weight'][eids_r] = \
                                graph.edges['intra_gate'].data['weight'][eids].unsqueeze(1)

                            if self.flag_delay: graph.pull(nodes_gate, self.message_func_delay,
                                                           self.reduce_func_delay_g,
                                                           etype='intra_gate')
                    if len(nodes_module) != 0:
                        eids = graph.in_edges(nodes_module, form='eid', etype='intra_module')
                        graph.pull(nodes_module, fn.copy_e('bit_position', 'pos'), fn.mean('pos', 'width2'),
                                   etype='intra_module')
                        graph.apply_edges(self.edge_msg_module, eids, etype='intra_module')
                        graph.pull(nodes_module, self.message_func_module, self.reduce_func_attn_m,
                                   self.nodes_func_module, etype='intra_module')

                        if self.flag_reverse or self.flag_path_supervise:
                            graph.apply_edges(self.edge_msg_module_weight, eids, etype='intra_module')
                            eids_r = graph.out_edges(nodes_module, form='eid', etype='reverse')
                            graph.edges['reverse'].data['weight'][eids_r] = graph.edges['intra_module'].data['weight'][
                                eids]

                            if self.flag_delay: graph.pull(nodes_module, self.message_func_delay,
                                                           self.reduce_func_delay_m, etype='intra_module')

            h_gnn = graph.ndata['h'][PO_mask]
            h  = h_gnn
            rst = self.mlp_out(h_gnn)

            prob_sum, prob_dev = th.tensor([0.0]),th.tensor([0.0])
            POs_criticalprob = None

            if self.flag_reverse or self.flag_path_supervise:

                POs_criticalprob = None
                POs = graph_info['POs']
                nodes_prob,nodes_dst = self.prop_backward(graph,graph_info)
                nodes_dst[nodes_dst < -100] = 0

                if self.flag_path_supervise or self.global_cat_choice in [3,4,5,7]:
                    graph.ndata['hp'] = nodes_prob
                    graph.ndata['id'] = th.zeros((graph.number_of_nodes(), 1), dtype=th.int64).to(self.device)
                    graph.ndata['id'][POs] = th.tensor(range(len(POs)), dtype=th.int64).unsqueeze(-1).to(self.device)
                    graph.pull(POs, self.message_func_prob, self.reduce_func_prob, etype='pi2po')
                    POs_criticalprob = graph.ndata['prob_sum'][POs]
                    prob_sum = graph.ndata['prob_sum'][POs]
                    prob_dev = graph.ndata['prob_dev'][POs]

                if not self.flag_reverse:
                    return rst, prob_sum,prob_dev,POs_criticalprob


                nodes_emb = graph.ndata['h']

                PIs_mask = graph.ndata['is_pi'] == 1
                PIs_prob = th.transpose(nodes_prob[PIs_mask], 0, 1)

                nodes_prob_tr = th.transpose(nodes_prob, 0, 1)
                #print(nodes_dst.shape,th.max(nodes_dst,dim=1).values.shape)
                #nodes_dst = nodes_dst / th.max(nodes_dst,dim=1).values
                #nodes_prob_tr = th.transpose(nodes_prob*nodes_dst, 0, 1)

                h_global = th.matmul(nodes_prob_tr, nodes_emb)

                #h_global = self.activation(h_global)

                #h_global = (1 / th.sum(nodes_prob_tr, dim=1)).unsqueeze(1) * h_global


                if self.global_info_choice == 2:
                    h_pi = th.matmul(PIs_prob, graph.ndata['h'][PIs_mask])
                    h_global = th.cat((h_global,h_pi),dim=1)
                elif self.global_info_choice in [3]:
                    h_pi = th.matmul(PIs_prob, graph.ndata['delay'][PIs_mask])
                    h_global = th.cat((h_global, h_pi), dim=1)
                elif self.global_info_choice in [4]:
                    nodes_delay, nodes_inputDelay = self.prop_delay(graph, graph_info)
                    h_d = nodes_inputDelay[POs]
                    h_global = th.cat((h_global, h_d), dim=1)
                elif self.global_info_choice == 5:
                    nodes_delay, nodes_inputDelay = self.prop_delay(graph, graph_info)
                    h_d = th.matmul(nodes_prob_tr,nodes_inputDelay)
                    h_global = th.cat((h_global, h_d), dim=1)
                elif self.global_info_choice == 6:
                    nodes_delay, nodes_inputDelay = self.prop_delay(graph, graph_info)
                    h_d = nodes_inputDelay[POs]

                    h_pi = th.matmul(PIs_prob, graph.ndata['delay'][PIs_mask])
                    h_global = th.cat((h_global, h_d,h_pi), dim=1)
                elif self.global_info_choice == 7:
                    maxPI_idx = th.argmax(PIs_prob,dim=1)
                    maxPI_delay =graph.ndata['delay'][PIs_mask][maxPI_idx]
                    h_global = th.cat((h_global, maxPI_delay), dim=1)

                elif self.global_info_choice == 8:
                    maxPI_idx = th.argmax(PIs_prob,dim=1)
                    maxPI_delay =graph.ndata['delay'][PIs_mask][maxPI_idx]
                    h_pi = th.matmul(PIs_prob, graph.ndata['h'][PIs_mask])
                    h_global = th.cat((h_global, h_pi,maxPI_delay), dim=1)

                elif self.global_info_choice == 9:
                    maxPI_idx = th.argmax(PIs_prob,dim=1)
                    maxPI_delay =graph.ndata['delay'][PIs_mask][maxPI_idx]
                    nodes_delay, nodes_inputDelay = self.prop_delay(graph, graph_info)
                    h_d = nodes_inputDelay[POs]
                    h_global = th.cat((h_global, h_d,maxPI_delay), dim=1)


                if self.global_cat_choice == 0:
                    h = th.cat((rst,h_global),dim=1)
                elif self.global_cat_choice == 1:
                    h = th.cat((h,h_global),dim=1)
                elif self.global_cat_choice == 2:
                    h = h_global
                elif self.global_cat_choice == 3:
                    h = th.cat((rst,(1-prob_sum)*h_global),dim=1)
                elif self.global_cat_choice == 4:
                    h = th.cat((rst,self.mlp_w(1-prob_sum)*h_global),dim=1)
                elif self.global_cat_choice == 5:
                    h = th.cat((h,(1-prob_sum)*h_global),dim=1)
                elif self.global_cat_choice == 6:
                    h = th.cat((h,self.mlp_w(1-prob_sum)*h_global),dim=1)
                elif self.global_cat_choice == 7:
                    prob_max = th.max(nodes_prob_tr, dim=1).values.unsqueeze(1)
                    h = th.cat((h, (1 - prob_max) * h_global), dim=1)
                elif self.global_cat_choice == 8:
                    w = self.mlp_w(h)
                    h = th.cat((h, w * h_global), dim=1)
                elif self.global_cat_choice == 9:
                    etp = -th.sum(nodes_prob_tr*th.log(nodes_prob_tr+1e-10),dim=1).unsqueeze(1)
                    w = self.mlp_w2(etp)
                    h = th.cat((h, w * h_global), dim=1)
                elif self.global_cat_choice == 9:
                    etp = -th.sum(nodes_prob_tr * th.log(nodes_prob_tr + 1e-10), dim=1).unsqueeze(1)
                    minmax = (th.max(nodes_prob_tr,dim=1).values - th.min(nodes_prob_tr,dim=1).values).unsqueeze(1)
                    w = self.mlp_w2(th.cat((etp,minmax),dim=1))
                    h = th.cat((h, w * h_global), dim=1)
                elif self.global_cat_choice == 11:
                    minmax = (th.max(nodes_prob_tr,dim=1).values - th.min(nodes_prob_tr,dim=1).values).unsqueeze(1)
                    w = self.mlp_w2(minmax)
                    h = th.cat((h, w * h_global), dim=1)


                rst = self.mlp_out_new(h)

                return  rst,prob_sum, prob_dev,POs_criticalprob

            return rst,prob_sum, prob_dev,POs_criticalprob


class ACCNN(nn.Module):

    def __init__(self,
                 infeat_dim,
                 hidden_dim,
                 flag_homo=False):
        super(ACCNN, self).__init__()

        self.flag_homo = flag_homo

        if self.flag_homo:
            self.mlp_agg = MLP(hidden_dim+ infeat_dim, int(hidden_dim / 2), hidden_dim)
        else:
            self.mlp_agg_module = MLP(hidden_dim + infeat_dim, int(hidden_dim / 2), hidden_dim)
            self.mlp_agg_gate = MLP(hidden_dim + infeat_dim, int(hidden_dim / 2), hidden_dim)
        self.mlp_pi = MLP(4, int(hidden_dim / 2), hidden_dim)
        self.mlp_out = MLP( hidden_dim,hidden_dim,1)


    def nodes_func(self,nodes):
        m_self = nodes.data['feat']
        h = th.cat((nodes.data['neigh'], m_self), dim=1)
        h = self.mlp_agg(h)

        return {'h':h}

    def nodes_func_module(self,nodes):

        m_self = nodes.data['feat']
        h = th.cat((nodes.data['neigh'], m_self), dim=1)
        h = self.mlp_agg_module(h)

        return {'h': h}

    def nodes_func_gate(self,nodes):

        m_self = nodes.data['feat']
        h = th.cat((nodes.data['neigh'], m_self), dim=1)
        h = self.mlp_agg_gate(h)

        return {'h': h}

    def nodes_func_pi(self,nodes):
        h = th.cat((nodes.data['delay'],nodes.data['value']),dim=1)
        h = self.mlp_pi(h)

        return {'h':h}


    def forward(self, graph,graph_info):

        topo = graph_info['topo']
        PO_mask = graph_info['POs']
        prob_sum, prob_dev = th.tensor([0.0]), th.tensor([0.0])
        POs_criticalprob = None

        with (graph.local_scope()):
            #propagate messages in the topological order, from PIs to POs
            for i, nodes in enumerate(topo):
                # for PIs
                if i==0:
                    graph.apply_nodes(self.nodes_func_pi,nodes)
                elif graph_info['is_heter']:
                    isModule_mask = graph.ndata['is_module'][nodes] == 1
                    isGate_mask = graph.ndata['is_module'][nodes] == 0
                    nodes_gate = nodes[isGate_mask]
                    nodes_module = nodes[isModule_mask]
                    if len(nodes_gate)!=0: graph.pull(nodes_gate, fn.copy_u('h', 'm'),fn.mean('m', 'neigh'), self.nodes_func_gate, etype='intra_gate')
                    if len(nodes_module)!=0: graph.pull(nodes_module, fn.copy_u('h', 'm'), fn.mean('m', 'neigh'), self.nodes_func_module, etype='intra_module')
                else:
                    graph.pull(nodes, fn.copy_u('h', 'm'), fn.mean('m', 'neigh'), self.nodes_func)

            h = graph.ndata['h'][PO_mask]
            rst = self.mlp_out(h)

            return rst,prob_sum, prob_dev,POs_criticalprob


class PathModel(nn.Module):
    def __init__(self,infeat_dim,hidden_dim,device,impl_choice=0):
        super(PathModel, self).__init__()
        self.impl_choice = impl_choice
        self.device = device
        if impl_choice == 0:
            self.model = MLP(infeat_dim,hidden_dim,hidden_dim,1)

    def forward(self,POs_feat):
        rst = th.zeros((len(POs_feat),1),dtype=th.float).to(self.device)
        for i,feat in enumerate(POs_feat):
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

        self.mlp_n =  MLP(infeat_dim, int(feat_dim/2), feat_dim)
        self.mlp_out = MLP(feat_dim, int(feat_dim/2), 1)

        self.degree_encoder = dgl.nn.DegreeEncoder(
                max_degree=8,  # the maximum degree to cut off
                embedding_dim=feat_dim  # the dimension of the degree embedding
            )
        self.spatial_encoder = dgl.nn.SpatialEncoder(
                max_dist=5,  # the maximum distance between two nodes
                num_heads=num_heads,  # the number of attention heads
            )

    def forward(self,graphs_info):
        deg_emb = self.degree_encoder(th.stack((graphs_info['in_degree'],graphs_info['out_degree'])))
        node_feat = self.mlp_n(graphs_info['node_feat_new'])
        num_graphs, max_num_nodes,_ = node_feat.shape
        # node feature + degree encoding as input
        x = node_feat + deg_emb

        # spatial encoding and path encoding serve as attention bias


        spatial_encoding = self.spatial_encoder(graphs_info['dist'])
        attn_bias = th.rand(num_graphs, max_num_nodes, max_num_nodes, self.num_heads)
        #attn_bias = spatial_encoding
        for layer in self.layers:
            x = layer(
                x,
                attn_mask=graphs_info['attn_mask'],
                attn_bias=attn_bias,
            )

        res = None
        for i in range(x.shape[0]):
            res = cat_tensor(res,x[i][graphs_info['POs_mask'][i]])
            #print(graphs_info['POs_mask'][i].shape,x[i][graphs_info['POs_mask'][i]].shape)
        res = self.mlp_out(res)
        return res
