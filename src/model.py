"""Torch Module for TimeConv layer"""
import dgl
import torch as th
from torch import nn
from dgl import function as fn
from xgboost import XGBRegressor
from utils import *
from options import get_options
options = get_options()
device = th.device("cuda:" + str(options.gpu) if th.cuda.is_available() else "cpu")

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


class TimeConv(nn.Module):

    def __init__(self,
                 infeat_dim1,
                 infeat_dim2,
                 hidden_dim,
                 global_out_choice=0,
                 global_cat_choice=0,
                 global_info_choice=0,
                 pi_choice=0,
                 agg_choice=0,
                 attn_choice=1,
                 inv_choice=-1,
                 flag_degree=False,
                 flag_width=False,
                 flag_delay_pd=False,
                 flag_delay_m=False,
                 flag_delay_g=False,
                 flag_delay_pi=False,
                 flag_ntype_g=False,
                 flag_train=True,
                 flag_path_supervise=False,
                 flag_filter=False,
                 flag_reverse=False,
                 flag_splitfeat=False,
                 flag_homo=False,
                 flag_global=True,
                 flag_attn=False):
        super(TimeConv, self).__init__()

        self.global_out_choice = global_out_choice
        self.global_cat_choice = global_cat_choice
        self.global_info_choice = global_info_choice
        self.inv_choice = inv_choice
        self.flag_degree = flag_degree
        self.flag_width = flag_width
        self.flag_delay_m = flag_delay_m
        self.flag_delay_g = flag_delay_g
        self.flag_delay_pi = flag_delay_pi
        self.flag_delay_pd = flag_delay_pd
        self.flag_ntype_g = flag_ntype_g
        self.pi_choice = pi_choice
        self.flag_train = flag_train
        self.flag_path_supervise = flag_path_supervise
        self.flag_filter = flag_filter
        self.flag_reverse = flag_reverse
        self.flag_global = flag_global
        self.flag_attn = flag_attn
        self.hidden_dim = hidden_dim
        self.agg_choice = agg_choice
        self.attn_choice = attn_choice
        self.mlp_pi = MLP(4, int(hidden_dim / 2), hidden_dim)
        self.mlp_agg = MLP(hidden_dim, int(hidden_dim / 2), hidden_dim)

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
            if self.agg_choice==0:
                self.mlp_self = MLP(self.infeat_dim1, int(hidden_dim / 2), hidden_dim)
                self.mlp_self_module = self.mlp_self
                self.mlp_self_gate = self.mlp_self
            else:
                feat_self_g_dim = self.infeat_dim1
                feat_self_m_dim = self.infeat_dim2+1
                if self.flag_delay_pi:
                    feat_self_g_dim += 1
                    feat_self_m_dim += 1
                if self.flag_delay_pd:
                    feat_self_g_dim += 1
                    feat_self_m_dim += 1
                self.mlp_self_gate = MLP(feat_self_g_dim, int(hidden_dim / 2), hidden_dim)
                self.mlp_self_module = MLP(feat_self_m_dim, int(hidden_dim / 2), hidden_dim)
        if flag_homo:
            self.mlp_neigh = MLP(hidden_dim, int(hidden_dim / 2), hidden_dim)
        else:
            if self.agg_choice==0:
                neigh_dim_m = hidden_dim + self.infeat_dim2+1
                neigh_dim_g = hidden_dim + self.infeat_dim1
            else:
                neigh_dim_m = hidden_dim
                neigh_dim_g = hidden_dim
            if self.flag_delay_pi:
                neigh_dim_m += 1
                neigh_dim_g += 1
            if self.flag_degree:
                neigh_dim_m += 1
                neigh_dim_g += 1
            if self.flag_delay_pd:
                neigh_dim_m += 1
                neigh_dim_g += 1
            if self.inv_choice in [0,2]:
                neigh_dim_m += 1
                neigh_dim_g += 1
            self.mlp_neigh_module = MLP(neigh_dim_m, int(hidden_dim / 2), hidden_dim)
            self.mlp_neigh_gate = MLP(neigh_dim_g, int(hidden_dim / 2), hidden_dim)
        if flag_global:
            self.mlp_global = MLP(1, int(hidden_dim / 2), hidden_dim)
            out_dim = hidden_dim * 2
        if flag_attn:
            hidden_dim_attn = int(hidden_dim/8)
            atnn_dim_m = hidden_dim + hidden_dim_attn *2
            atnn_dim_g = hidden_dim
            self.mlp_type = MLP(self.infeat_dim2, hidden_dim_attn, hidden_dim_attn)
            if self.flag_width:
                self.mlp_pos = MLP(2, hidden_dim_attn, hidden_dim_attn)
            else:
                self.mlp_pos = MLP(1, hidden_dim_attn, hidden_dim_attn)
            if self.flag_delay_m:
                atnn_dim_m += hidden_dim_attn
                self.mlp_level_m = MLP(2, hidden_dim_attn, hidden_dim_attn)
            if self.flag_delay_g:
                atnn_dim_g += hidden_dim_attn
                self.mlp_level = MLP(2, hidden_dim_attn, hidden_dim_attn)
            if self.flag_ntype_g:
                atnn_dim_g += hidden_dim_attn
                self.mlp_type_g = MLP(self.infeat_dim1, hidden_dim_attn, hidden_dim_attn)

            if self.inv_choice in [ 1,2]:
                atnn_dim_m += hidden_dim_attn
                self.mlp_inv = MLP(1, hidden_dim_attn, hidden_dim_attn)
                atnn_dim_g += hidden_dim_attn


            self.attention_vector_g = nn.Parameter(th.randn(atnn_dim_g, 1), requires_grad=True)
            self.attention_vector_m = nn.Parameter(th.randn(atnn_dim_m,1),requires_grad=True)


        self.mlp_out = MLP(out_dim,hidden_dim,1)
        self.activation = nn.ReLU()
        self.activation2 = th.nn.LeakyReLU(negative_slope=0.2)


    def nodes_func(self,nodes):
        if self.flag_attn:
            #h = self.mlp_neigh(nodes.data['neigh']) + self.mlp_self(nodes.data['feat'])
            h = self.mlp_neigh(nodes.data['neigh'])
        else:
            h = self.mlp_neigh(nodes.data['neigh']) + self.mlp_self(nodes.data[self.feat_name1])
        # apply activation except the POs
        mask = nodes.data['is_po'].squeeze() != 1
        h[mask] = self.activation(h[mask])

        return {'h':h}

    def nodes_func_module(self,nodes):

        mask = nodes.data['is_po'].squeeze() != 1
        m_self = th.cat((nodes.data['pos'], nodes.data[self.feat_name2]), dim=1)
        #m_self = nodes.data[self.feat_name2]
        # print(th.sum(th.abs(nodes.data['width']-nodes.data['pos'])))
        # # exit()
        if self.flag_delay_pi:
            m_self = th.cat((m_self, nodes.data['input_delay']), dim=1)
        if self.flag_degree:
            m_self = th.cat((m_self, nodes.data['degree']), dim=1)
        if self.agg_choice ==0:
            h = th.cat((nodes.data['neigh'], m_self), dim=1)
            h = self.mlp_neigh_module(h)
        else:
            h = self.mlp_neigh_module(nodes.data['neigh']) + self.mlp_self_module(m_self)

        h[mask] = self.activation(h[mask])

        if self.flag_reverse or self.flag_path_supervise:
            return {'h':h,'attn_sum':nodes.data['attn_sum'],'attn_max': nodes.data['attn_max']}
        else:
            return {'h': h}

    def nodes_func_gate(self,nodes):

        mask = nodes.data['is_po'].squeeze() != 1
        m_self = nodes.data[self.feat_name1]

        if self.flag_delay_pi:
            m_self = th.cat((m_self,nodes.data['input_delay']),dim=1)
        if self.flag_degree:
            m_self = th.cat((m_self, nodes.data['degree']), dim=1)
        if self.agg_choice ==0:

            h = th.cat((nodes.data['neigh'], m_self), dim=1)

            h = self.mlp_neigh_gate(h)
        else:
            h = self.mlp_neigh_gate(nodes.data['neigh']) + self.mlp_self_gate(m_self)
        h[mask] = self.activation(h[mask])
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
        m_type = self.mlp_type(edges.dst[self.feat_name2])
        m_pos = edges.data['bit_position']
        if self.flag_width:
            m_pos = th.cat((m_pos,edges.dst['width']),dim=1)
        m_pos = self.mlp_pos(m_pos)
        if self.flag_delay_m:
            m_level = self.mlp_level_m(th.cat((edges.src['delay'],edges.dst['delay']),dim=1))
            # m_pos = self.mlp_pos(th.cat((edges.data['bit_position'].unsqueeze(1),edges.src['level']),dim=1))
            z = th.cat((edges.src['h'], m_pos,m_level,m_type), dim=1)
        else:
            z = th.cat((edges.src['h'], m_pos, m_type), dim=1)
        if self.inv_choice in [1,2]:
            z = th.cat((z, self.mlp_inv(edges.data['is_inv'])), dim=1)
        e = th.matmul(z, self.attention_vector_m)
        e = self.activation2(e)
        return {'attn_e': e}

    def message_func_module(self,edges):
        #m = th.cat((edges.src['h'], edges.data['bit_position'].unsqueeze(1)), dim=1)
        m = edges.src['h']

        #pos = edges.data['bit_position'].unsqueeze(1)
        if self.flag_delay_pd:
            m = th.cat((m,self.mlp_out(m)),dim=1)
        if self.inv_choice in [0,2]:
            m = th.cat((m, edges.data['is_inv']), dim=1)
        rst = {'m':m,'pos':edges.data['bit_position']}
        if self.flag_attn:
            rst['attn_e'] = edges.data['attn_e']
        return rst


    def reduce_func_attn_m(self,nodes):

        if self.flag_reverse or self.flag_path_supervise:
            attn_e = nodes.mailbox['attn_e']
            max_attn_e = th.max(attn_e, dim=1).values
            attn_e = attn_e - max_attn_e.unsqueeze(1)
            attn_e_exp = th.exp(attn_e)
            attn_exp_sum = th.sum(attn_e_exp, dim=1).unsqueeze(1)
            alpha = attn_e_exp / attn_exp_sum
            h = th.sum(alpha * nodes.mailbox['m'], dim=1)
            return {'neigh':h,'pos':th.mean(nodes.mailbox['pos'],dim=1),'attn_sum':attn_exp_sum,'attn_max':max_attn_e}
        else:
            alpha = th.softmax(nodes.mailbox['attn_e'], dim=1)
            h = th.sum(alpha * nodes.mailbox['m'], dim=1)
            #print(th.mean(nodes.mailbox['pos'],dim=1),h.shape)

            return {'neigh': h,'pos':th.mean(nodes.mailbox['pos'],dim=1)}

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

        z = edges.src['h']
        if self.flag_delay_g:
            m_level = self.mlp_level(th.cat((edges.src['delay'],edges.dst['delay']),dim=1))
            z = th.cat((edges.src['h'],m_level), dim=1)
        if self.flag_ntype_g:
            m_type = self.mlp_type_g(edges.dst[self.feat_name2])
            z = th.cat((z, m_type), dim=1)
        if self.inv_choice in [1,2]:
            z = th.cat((z, self.mlp_inv(edges.data['is_inv'])), dim=1)
        e = th.matmul(z, self.attention_vector_g)
        e = self.activation2(e)
        return {'attn_e': e}

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
        m = edges.src['h']
        if self.flag_delay_pd:
            m = th.cat((m,self.mlp_out(m)),dim=1)
        if self.inv_choice in [0,2]:
            m = th.cat((m, edges.data['is_inv']), dim=1)
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
        delay = th.max(nodes.mailbox['md'],dim=1).values+0.3
        #input_delay = th.max(nodes.mailbox['md'],dim=1).values
        input_delay = th.sum(nodes.mailbox['md'] * nodes.mailbox['w'], dim=1)
        return {'delay':delay,'input_delay':input_delay}

    def reduce_func_delay_m(self,nodes):
        delay = th.max(nodes.mailbox['md'],dim=1).values+0.6
        #input_delay = th.max(nodes.mailbox['md'], dim=1).values
        input_delay = th.sum(nodes.mailbox['md']*nodes.mailbox['w'],dim=1)
        return {'delay':delay,'input_delay':input_delay}


    def message_func_prob(self, edges):
        msg = th.gather(edges.src['hp'], dim=1, index=edges.dst['id'])
        return {'mp': msg}

    def nodes_func_pi(self,nodes):
        #h = nodes.data['delay']
        h = th.cat((nodes.data['delay'],nodes.data['value']),dim=1)
        h = self.mlp_pi(h)
        #mask = nodes.data['is_po'].squeeze() != 1
        #h[mask] = self.activation(h[mask])

        return {'h':h}

    def reduce_func_prob(self,nodes):
        prob_sum = th.sum(nodes.mailbox['mp'],dim=1)
        prob_max = th.max(nodes.mailbox['mp'],dim=1).values
        #prob_sum = th.sum(nodes.mailbox['ml'] * nodes.mailbox['w'], dim=1)
        prob_mean = th.mean(nodes.mailbox['mp'], dim=1).unsqueeze(1)
        prob_dev = th.sum(th.abs(nodes.mailbox['mp']-prob_mean),dim=1)
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

    def forward(self, graph,graph_info):

        topo = graph_info['topo']
        PO_mask = graph_info['POs_mask']
        PO_feat = graph_info['POs_feat']

        with (graph.local_scope()):
            graph.edges['intra_module'].data['bit_position'] = graph.edges['intra_module'].data['bit_position'].unsqueeze(1)
            #graph.ndata['pos'] = th.zeros((graph.number_of_nodes(),1),dtype=th.float).to(device)
            if self.flag_delay_pi or self.flag_delay_g or self.flag_delay_m:
                nodes_delay,nodes_inputDelay = self.prop_delay(graph,graph_info)

                # print(graph.ndata['level'][PO_mask].squeeze(1))
                # print(nodes_delay[PO_mask].squeeze(1))
                # print(graph.ndata['label'][PO_mask].squeeze(1))
                # exit()
                graph.ndata['level'] = nodes_delay
                graph.ndata['input_delay'] = nodes_inputDelay
                #graph.ndata['level'] = nodes_delay
                #print(graph.edges['intra_module'].data['ratio'])


            if self.flag_reverse or self.flag_path_supervise:
                graph.edges['reverse'].data['weight'] = th.zeros((graph.number_of_edges('reverse'), 1),
                                                                 dtype=th.float).to(device)
            #propagate messages in the topological order, from PIs to POs
            for i, nodes in enumerate(topo):
                isModule_mask = graph.ndata['is_module'][nodes] == 1
                isGate_mask = graph.ndata['is_module'][nodes] == 0

                # for PIs
                if i==0:
                    graph.apply_nodes(self.nodes_func_pi,nodes)
                # for other nodes
                elif self.flag_attn:
                    if graph_info['is_heter']:
                        nodes_gate = nodes[isGate_mask]
                        nodes_module = nodes[isModule_mask]
                        message_func_gate = self.message_func_gate if self.attn_choice==0 else fn.copy_src('h', 'm')
                        reduce_func_gate = self.reduce_func_attn_g if self.attn_choice==0 else self.reduce_func_smoothmax

                        if len(nodes_gate)!=0:
                            if self.attn_choice == 0:
                                eids = graph.in_edges(nodes_gate, form='eid', etype='intra_gate')
                                graph.apply_edges(self.edge_msg_gate, eids, etype='intra_gate')
                            graph.pull(nodes_gate, message_func_gate, reduce_func_gate, self.nodes_func_gate, etype='intra_gate')
                            if self.flag_reverse or self.flag_path_supervise:
                                eids = graph.in_edges(nodes_gate, form='eid', etype='intra_gate')
                                eids_r = graph.out_edges(nodes_gate, form='eid', etype='reverse')
                                graph.apply_edges(self.edge_msg_gate_weight, eids, etype='intra_gate')
                                if self.attn_choice==0:
                                    graph.edges['reverse'].data['weight'][eids_r] = graph.edges['intra_gate'].data['weight'][eids]
                                elif self.attn_choice==1:
                                    graph.edges['reverse'].data['weight'][eids_r] = graph.edges['intra_gate'].data['weight'][eids].unsqueeze(1)
                        if len(nodes_module)!=0:
                            eids = graph.in_edges(nodes_module, form='eid', etype='intra_module')
                            graph.apply_edges(self.edge_msg_module, eids, etype='intra_module')
                            graph.pull(nodes_module, self.message_func_module, self.reduce_func_attn_m, self.nodes_func_module, etype='intra_module')
                            if self.flag_reverse or self.flag_path_supervise:
                                graph.apply_edges(self.edge_msg_module_weight, eids, etype='intra_module')
                                eids_r = graph.out_edges(nodes_module, form='eid', etype='reverse')
                                graph.edges['reverse'].data['weight'][eids_r] = graph.edges['intra_module'].data['weight'][eids]
                    else:
                        graph.pull(nodes, self.message_func_attn, self.reduce_func_attn_g, self.nodes_func)
                else:
                    if graph_info['is_heter']:
                        nodes_gate = nodes[isGate_mask]
                        nodes_module = nodes[isModule_mask]
                        if len(nodes_gate)!=0: graph.pull(nodes_gate, fn.copy_src('h', 'm'), self.reduce_func_smoothmax, self.nodes_func_gate, etype='intra_gate')
                        if len(nodes_module)!=0: graph.pull(nodes_module, self.message_func_module, self.reduce_func_mean, self.nodes_func_module, etype='intra_module')
                    else:
                        graph.pull(nodes, fn.copy_src('h', 'm'), fn.mean('m', 'neigh'), self.nodes_func)


            h_gnn = graph.ndata['h'][PO_mask]


            if self.flag_global:
                h_global = self.mlp_global(PO_feat)

                h = th.cat([h_gnn,h_global],dim=1)
            else:
                h = h_gnn

            rst = self.mlp_out(h)


            prob_sum, prob_dev = th.tensor([0.0]),th.tensor([0.0])
            POs_criticalprob = None

            # if not self.flag_train and self.flag_path_supervise:
            #     return rst,prob_sum, prob_dev,POs_criticalprob

            #print("aaaa")

            if self.flag_reverse or self.flag_path_supervise:
                critical_po_mask = rst.squeeze(1) > 10
                # if self.flag_filter:
                #     POs = POs[critical_po_mask]
                POs_criticalprob = None
                POs = graph_info['POs']
                nodes_prob,nodes_dst = self.prop_backward(graph,graph_info)
                nodes_dst[nodes_dst < -100] = 0
                # graph.ndata['hp'] = nodes_prob
                # graph.ndata['id'] = th.zeros((graph.number_of_nodes(), 1), dtype=th.int64).to(device)
                # graph.ndata['id'][POs] = th.tensor(range(len(POs)), dtype=th.int64).unsqueeze(-1).to(device)
                # graph.pull(POs, self.message_func_prob, fn.sum('mp', 'prob'), etype='pi2po')
                # POs_criticalprob = graph.ndata['prob'][POs]

                if self.flag_path_supervise or self.global_cat_choice in [3,4,5]:

                    graph.ndata['hp'] = nodes_prob
                    graph.ndata['id'] = th.zeros((graph.number_of_nodes(), 1), dtype=th.int64).to(device)
                    graph.ndata['id'][POs] = th.tensor(range(len(POs)), dtype=th.int64).unsqueeze(-1).to(device)
                    #print(graph.number_of_edges(etype='pi2po'),len(POs))
                    graph.pull(POs, self.message_func_prob, self.reduce_func_prob, etype='pi2po')
                    POs_criticalprob = None

                    POs_criticalprob = graph.ndata['prob_sum'][POs]

                    # graph.pull(POs, fn.copy_src('delay','md'), fn.mean('md', 'di'), etype='pi2po')
                    #
                    # nodes_dst += graph.ndata['delay']
                    #PIs_mask = graph.ndata['is_pi'] == 1
                    # PIs_dst = th.transpose(nodes_dst[PIs_mask], 0, 1)
                    # POs_maxDst_idx = th.argmax(PIs_dst, dim=1)
                    # POs_delay_d = graph.ndata['delay'][POs_maxDst_idx]
                    #
                    #PIs_prob = th.transpose(nodes_prob[PIs_mask], 0, 1)
                    #POs_maxProb = th.max(PIs_prob, dim=1).values.unsqueeze(1)
                    #POs_criticalprob = POs_maxProb
                    # POs_maxProb_idx = th.argmax(PIs_prob, dim=1)
                    # POs_delay_p = graph.ndata['delay'][POs_maxProb_idx]
                    #
                    # POs_name = [graph_info['nodes_name'][n] for n in POs]
                    # POname2idx = {n: i for i, n in enumerate(POs_name)}
                    # cur_PIs_dst = PIs_dst[POname2idx['do_15[1]']]
                    # mask = cur_PIs_dst >= 0
                    # nodes_list = th.tensor(range(graph.number_of_nodes())).to(device)
                    # PIs_nid = nodes_list[PIs_mask][mask]
                    # PIs_nid2idx = {nid:i for i,nid in enumerate(PIs_nid.detach().cpu().numpy().tolist())}
                    # PIs_name = [graph_info['nodes_name'][n] for n in PIs_nid]
                    # idxs = []
                    # for j,pi in enumerate(PIs_name):
                    #     if pi in ['di_13[0]']:
                    #         idxs.append(j)
                    # #print(idxs)
                    # #PIs_prob = PIs_prob[:,idxs]
                    # for j, po in enumerate(POs):
                    #     prob = PIs_prob[j]
                    #     prob_list = prob.detach().cpu().numpy().tolist()
                    #     po_name = graph_info['nodes_name'][po]
                    #     critical_PIs = graph.in_edges(po, etype='pi2po')[
                    #         0].detach().cpu().numpy().tolist()
                    #     predicted_critical_PI = th.argmax(prob).item()
                    #     flag = predicted_critical_PI in  critical_PIs
                    #     critical_PIs.append(predicted_critical_PI)
                    #     print(po_name,flag,{graph_info['nodes_name'][n]:prob_list[PIs_nid2idx[n]] for n in critical_PIs})
                    #
                    # reverse_eids = th.tensor(range(graph.number_of_edges(etype='reverse'))).to(device)
                    #
                    # noncritical_eids = reverse_eids[graph.edges['reverse'].data['weight'].squeeze(1)<0.1]
                    #
                    # graph.remove_edges(noncritical_eids,etype='reverse')
                    # for j, po in enumerate(POs):
                    #     po_name = graph_info['nodes_name'][po]
                    #     PI_prob = PIs_prob[j]
                    #     if PI_prob.item()<0.5:
                    #         continue
                    #     print('{} --> {}'.format('di_13[0]',po_name))
                    #     print('\t',graph_info['nodes_name'][po],'\t',graph_info['ntype'][po])
                    #     cur_nid = po
                    #     while True:
                    #         preds = graph.successors(cur_nid,etype='reverse')
                    #         if len(preds)==0:
                    #             break
                    #         preds_prob = nodes_prob[preds][:,j]
                    #         cur_nid = preds[th.argmax(preds_prob)]
                    #         prob = preds_prob[th.argmax(preds_prob)]
                    #         #assert  len(cur_nid)==1, "{}".format(cur_nid)
                    #         print('\t',graph_info['nodes_name'][cur_nid],'\t',graph_info['ntype'][cur_nid])
                    # exit()


                    # POs_delay_w = th.matmul(PIs_prob, graph.ndata['delay'][PIs_mask])
                    #
                    # cur_PIs_dst = PIs_dst[POname2idx['do_10[2]']]
                    # mask = cur_PIs_dst>=0
                    # mask = cur_PIs_dst>-1000
                    # cur_Pis_delay = nodes_delay[PIs_mask][mask].detach().cpu().numpy().tolist()
                    # cur_PIs_dst = cur_PIs_dst[mask].detach().cpu().numpy().tolist()
                    # cur_PIs_prob = PIs_prob[POname2idx['do_10[2]']][mask].detach().cpu().numpy().tolist()
                    # cur_PIs_prob = [round(v, 3) for v in cur_PIs_prob]
                    # PIs_nid = nodes_list[PIs_mask][mask]
                    # PIs_name = [graph_info['nodes_name'][n] for n in PIs_nid]
                    #
                    # critical_PIs = graph.in_edges(POs[POname2idx['do_10[2]']],etype='pi2po')[0].detach().cpu().numpy().tolist()
                    # critical_PIs_name = [graph_info['nodes_name'][n] for n in critical_PIs]
                    # print(critical_PIs_name)
                    # #print(PIs_name[33:])
                    # cur_PIs_dst = [round(v, 3) for v in cur_PIs_dst]
                    #
                    # #print(list(zip(PIs_name,cur_Pis_delay,cur_PIs_dst,cur_PIs_prob)))
                    # exit()

                    # print([get_nodename(graph_info['nodes_name'],po) for po in POs])
                    # PIs_mask = graph.ndata['is_pi'] == 1
                    # PIs = th.tensor(range(graph.number_of_nodes())).to(device)[graph.ndata['is_pi'] == 1]
                    # PIs = PIs.detach().cpu().numpy().tolist()
                    # print('#PI:',len(PIs))
                    # PIs_prob = th.transpose(nodes_prob[PIs_mask], 0, 1)
                    # print(PIs_prob.shape)
                    # for i, po in enumerate(POs):
                    #     pis_prob = PIs_prob[i]
                    # print(get_nodename(graph_info['nodes_name'],po))
                    # print('\t',[(graph_info['nodes_name'][pi][0],prob.item()) for pi,prob in zip(PIs,pis_prob)])
                    # print('\t',th.max(pis_prob))
                    # print('\t',th.sum(pis_prob))
                    # POs_argmaxPI = th.argmax(PIs_prob, dim=1).cpu().numpy().tolist()

                    # print([get_nodename(graph_info['nodes_name'],i) for i in POs_argmaxPI])

                    # exit()
                    #path_loss = th.mean(graph.ndata['loss'][POs])
                    prob_sum = graph.ndata['prob_sum'][POs]
                    prob_dev = graph.ndata['prob_dev'][POs]


                    #return rst, prob_sum,prob_dev,POs_criticalprob
                    #return rst, path_loss,POs_delay_d,POs_delay_p,POs_delay_m,POs_delay_w,POs_criticalprob




                #PIs_mask = graph.ndata['is_pi'] == 1
                #PIs_mask = th.logical_or(graph.ndata['is_pi'] == 1, graph.ndata['is_module'] == 1)
                #module_mask = graph.ndata['is_module'] == 1
                #modules_prob = nodes_prob[module_mask]
                #POs_moduleprob = th.transpose(modules_prob, 0, 1)
                #h_module = th.matmul(POs_moduleprob, graph.ndata['h'][module_mask])
                #print((graph.ndata['value'][:,[2]]==0).shape, (graph.ndata['is_pi'] == 1).shape)

                #PIs_mask = th.logical_or(graph.ndata['is_pi'] == 1,(graph.ndata['value'][:,[2]]==0).squeeze(1))

                #nodes_attn = th.softmax(th.transpose(nodes_prob,0,1),dim=1)
                nodes_emb = graph.ndata['h']
                if self.global_info_choice in [1]:
                    nodes_prob = nodes_prob[graph.ndata['is_po']==0]
                    nodes_emb = graph.ndata['h'][graph.ndata['is_po']==0]


                nodes_prob_tr = th.transpose(nodes_prob,0,1)

                h_global = th.matmul(nodes_prob_tr,nodes_emb)

                PIs_mask = graph.ndata['is_pi'] == 1
                PIs_prob = th.transpose(nodes_prob[PIs_mask], 0, 1)

                if self.global_info_choice == 2:
                    h_pi = th.matmul(PIs_prob, graph.ndata['h'][PIs_mask])
                    h_global =h_global+h_pi
                elif self.global_info_choice == 3:
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
                elif self.global_info_choice in [7,8]:
                    nodes_delay, nodes_inputDelay = self.prop_delay(graph, graph_info)
                    h_d = nodes_inputDelay[POs]
                    h_p =  th.matmul(nodes_prob_tr,graph.ndata['PE'])
                    if self.global_info_choice == 8:
                        h_p = self.mlp_pe(h_p)
                    h_global = th.cat((h_global, h_d,h_p), dim=1)

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
                    h = th.cat((h,self.mlp_w(1-prob_sum)*h_global),dim=1)

                if self.global_out_choice == 0:
                    rst = self.mlp_out_new(h)
                else:
                    mask =prob_sum.squeeze(1)>0.5
                    rst = self.mlp_out_new(h)
                    #print(h[mask].shape)
                    rst[mask] = self.mlp_out_new2(h[mask])

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
        PO_mask = graph_info['POs_mask']
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
                    if len(nodes_gate)!=0: graph.pull(nodes_gate, fn.copy_src('h', 'm'),fn.mean('m', 'neigh'), self.nodes_func_gate, etype='intra_gate')
                    if len(nodes_module)!=0: graph.pull(nodes_module, fn.copy_src('h', 'm'), fn.mean('m', 'neigh'), self.nodes_func_module, etype='intra_module')
                else:
                    graph.pull(nodes, fn.copy_src('h', 'm'), fn.mean('m', 'neigh'), self.nodes_func)

            h = graph.ndata['h'][PO_mask]
            rst = self.mlp_out(h)

            return rst,prob_sum, prob_dev,POs_criticalprob


class PathModel(nn.Module):
    def __init__(self,infeat_dim,hidden_dim,impl_choice=0):
        super(PathModel, self).__init__()
        self.impl_choice = impl_choice
        if impl_choice == 0:
            self.model = MLP(infeat_dim,hidden_dim,hidden_dim,1)
        elif impl_choice==1:
            self.model = XGBRegressor(n_estimators=500, max_depth=100, nthread=25)
    def forward(self,POs_feat):
        rst = th.zeros((len(POs_feat),1),dtype=th.float).to(device)
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
