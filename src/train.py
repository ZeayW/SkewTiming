import os
import sys

if '--cuda_blocking' in sys.argv:
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import matplotlib.pyplot as plt
import torch
from queue import Queue
from options import *
#from model import *
from model2 import *
import pickle
import numpy as np
import copy
from random import shuffle
import random
import torch as th
from torch.utils.data import DataLoader,Dataset
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler,Sampler
from torch.nn.functional import softmax
import torch.nn as nn
import datetime
from torchmetrics import R2Score
import dgl
import tee
from utils import *
from time import time
import itertools
from contextlib import nullcontext
from training_utils import (
    ModelEMA,
    filter_endpoint_rows,
    replace_case_50_with_case_0,
    supervision_loss_weights,
    valid_endpoint_mask,
)


options = get_options()
device = th.device("cuda:" + str(options.gpu) if th.cuda.is_available() and options.gpu !=-1 else "cpu")
print(device)
R2_score = R2Score().to(device)
Loss = nn.MSELoss()
Loss = nn.L1Loss()

def enable_runtime_stats():
    return getattr(options, 'log_level', 0) >= 1


def new_stage_time():
    return {
        'MTDE_forward': 0,
        'MTDE_backward': 0,
        'FSE': 0,
        'CPE_preparefeat': 0,
        'CPE_pathfind': 0,
        'CPE_encode': 0,
        'proj': 0,
        'other': 0,
        'all': 0,
    }


def new_train_profile():
    profile = new_stage_time()
    profile.update({
        'batch_graph': 0,
        'gather_data': 0,
        'model_forward': 0,
        'loss_metric': 0,
        'backward_step': 0,
        'graph_mutation': 0,
    })
    return profile


def sync_timer_start():
    if device.type == 'cuda':
        th.cuda.synchronize(device)
    return time()


def sync_timer_add(profile, key, start):
    if device.type == 'cuda':
        th.cuda.synchronize(device)
    profile[key] += time() - start


def add_stage_time(profile, stage_time):
    for key, value in stage_time.items():
        if key in profile:
            profile[key] += value


def format_profile(profile):
    train_keys = ('batch_graph', 'gather_data', 'model_forward', 'loss_metric',
                  'backward_step', 'graph_mutation')
    model_keys = ('MTDE_forward', 'MTDE_backward', 'FSE', 'CPE_preparefeat',
                  'CPE_pathfind', 'CPE_encode', 'proj', 'other')
    train_total = sum(profile[key] for key in train_keys)
    model_total = sum(profile[key] for key in model_keys)

    train_parts = []
    for key in train_keys:
        value = profile[key]
        pct = 100 * value / train_total if train_total > 0 else 0
        train_parts.append('{}={:.2f}s/{:.1f}%'.format(key, value, pct))

    model_parts = []
    for key in model_keys:
        value = profile[key]
        pct = 100 * value / model_total if model_total > 0 else 0
        model_parts.append('{}={:.2f}s/{:.1f}%'.format(key, value, pct))

    return 'Profile train: {}, train_total={:.2f}s\nProfile model: {}, model_stage_total={:.2f}s'.format(
        ', '.join(train_parts),
        train_total,
        ', '.join(model_parts),
        model_total,
    )


TCAD6_OPTION_CONFIG = {
    'hidden_dim': 128,
    'base_pe': 2,
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
    'split_feat': False,
    'agg_choice': 0,
    'attn_choice': 0,
    'alpha': 5.0,
    'beta': 5.0,
    'global_info_choice': 12,
    'global_cat_choice': 10,
    'flag_filter': True,
    'pretrain_dir': None,
    'flag_continue_trainpath': False,
}


def validate_tcad6_options(opts):
    for name, expected in TCAD6_OPTION_CONFIG.items():
        value = getattr(opts, name)
        if value != expected:
            raise ValueError(
                '{}={} is no longer supported by the refactored main model. '
                'Use src/scripts/run_train_tcad6.sh as the canonical configuration '
                '({}={}).'.format(name, value, name, expected)
            )


with open(options.ntype_file, 'rb') as f:
    ntype2id, ntype2id_gate, ntype2id_module = pickle.load(f)
num_gate_types = len(ntype2id_gate)
num_gate_types -= 3
num_module_types = len(ntype2id_module)
print(ntype2id)




def cat_tensor(t1,t2):
    if t1 is None:
        return t2
    elif t2 is None:
        return t1
    else:
        return th.cat((t1,t2),dim=0)


# print(split_list)
# exit()
def load_data(usage,options):
    assert usage in ['train','val','test']

    designs_group = None
    needs_design_groups = usage == 'test' and (options.test_iter or options.flag_group) and not options.flag_meta
    data_path = options.data_savepath
    if data_path.endswith('/'):
        data_path = data_path[:-1]
    data_file = os.path.join(data_path, 'data.pkl')
    if 'round7' in data_path:
        split_file = os.path.join(data_path, 'split.pkl')
    else:
        split_file = os.path.join(os.path.split(data_path)[0], 'split_new.pkl')
        if needs_design_groups:
            with open('designs_group_new.pkl', 'rb') as f:
                designs_group = pickle.load(f)

    #print(designs_group)



    #designs_group = None
    #split_file = os.path.join(data_path, 'split_new.pkl')
    with open(data_file, 'rb') as f:
        data_all = pickle.load(f)
        if '_case' in data_all[0][1]['design_name']:
            design_names = [d[1]['design_name'][:d[1]['design_name'].rfind('_case')] for d in data_all]
        else:
            design_names = [d[1]['design_name'] for d in data_all]


    with open(split_file, 'rb') as f:
        split_list = pickle.load(f)


    # for case_id in [110,220,183,185,319,320,329,371,383,392,399]:
    #     split_list['train'].append('random_logic_00{}'.format(case_id))
    # with open(os.path.join(os.path.split(data_path)[0], 'split_new2.pkl'),'wb') as f:
    #     pickle.dump(split_list,f)

    target_list = split_list[usage]
    target_list = [n for n in target_list]

    #print(split_list,target_list)

    data = [d for i,d in enumerate(data_all) if design_names[i] in target_list]
    case_range = (0, 100)
    if options.quick:
        if usage == 'train':
            case_range = (0,20)
        else:
            case_range = (0, 20)

    print("------------Loading {}_data #{} {}-------------".format(usage,len(data),case_range))

    loaded_data = []
    if (options.test_iter or usage=='test') and options.flag_group:
        loaded_data = {}
    for graph,graph_info in data:
        #print(graph_info['design_name'])
        #if int(graph_info['design_name'].split('_')[-1]) in [54, 96, 131, 300, 327, 334, 397]:
        #    continue

        if usage == 'test' and designs_group is None:
            if len( graph_info['delay-label_pairs'][0][1]) <= 150:
                continue
            if graph_info['design_name'] in ['s15850','s5378','tv80', 'sha3', 'ldpcenc', 'mc6809']: continue

            if not options.test_iter and graph_info['design_name'] in ['aes128','ecg']: continue

        name2nid = {graph_info['nodes_name'][i]:i for i in range(len(graph_info['nodes_name']))}
        #print(graph_info['design_name'],len(graph_info['delay-label_pairs'][0][1]))

        if options.flag_homo:
            graph = heter2homo(graph)
        if options.remove01:
            nodes_list = th.tensor(range(graph.number_of_nodes()))
            mask = th.logical_or(graph.ndata['value'][:,0]==1, graph.ndata['value'][:,1]==1)
            constant_list = nodes_list[mask]
            graph.remove_nodes(constant_list)

        if options.inv_choice!=-1:
            graph.edges['intra_module'].data['is_inv'] = graph.edges['intra_module'].data['is_inv'].unsqueeze(1)
            graph.edges['intra_gate'].data['is_inv'] = graph.edges['intra_gate'].data['is_inv'].unsqueeze(1)
        graph.ndata['feat'] = graph.ndata['ntype']
        graph.ndata['feat'] = graph.ndata['ntype'][:,3:]

        graph.ndata['feat_module'] = graph.ndata['ntype_module']
        graph.ndata['feat_gate'] = graph.ndata['ntype_gate'][:,3:]
        graph.ndata['h'] = th.ones((graph.number_of_nodes(), options.hidden_dim), dtype=th.float)


        #print(graph_info['design_name'],len(graph_info['delay-label_pairs']))
        if len(graph_info['delay-label_pairs'][0][0])!= len(graph.ndata['is_pi'][graph.ndata['is_pi'] == 1]):
            print('skip',graph_info['design_name'])
            continue

        if options.flag_reverse or options.flag_path_supervise:
            graph = add_reverse_edges(graph)

        if options.flag_path_supervise or options.global_cat_choice in [3,4,5]:
            graph = add_newEtype(graph,'pi2po',([],[]),{})

        graph_info['graph'] = graph
        replace_case_50_with_case_0(graph_info)
        graph_info['delay-label_pairs'] = graph_info['delay-label_pairs'][case_range[0]:case_range[1]]

        if options.flag_filter:
            for i in range(len(graph_info['delay-label_pairs'])):
                k = 20
                PIs_delay, POs_label, POs_baselabel, pi2po_edges = graph_info['delay-label_pairs'][i]
                pi2po_edges = filter_criticalPIs(pi2po_edges,k)
                graph_info['delay-label_pairs'][i] = (PIs_delay, POs_label, POs_baselabel, pi2po_edges)

        if (options.test_iter or usage=='test') and options.flag_group:
            if designs_group is None:
                loaded_data[graph_info['design_name']] = loaded_data.get(graph_info['design_name'],[])
                loaded_data[graph_info['design_name']].append(graph_info)
            else:
                group_id = designs_group[graph_info['design_name']]
                loaded_data[group_id] = loaded_data.get(group_id,[])
                loaded_data[group_id].append(graph_info)
        else:
            loaded_data.append(graph_info)

    return loaded_data

def get_idx_loader(data,batch_size,flag_train):
    drop_last =  flag_train and len(data) % batch_size < max(batch_size/2,8)

    sampler = SubsetRandomSampler(th.arange(len(data)))
    idx_loader = DataLoader([i for i in range(len(data))], sampler=sampler, batch_size=batch_size,
                            drop_last=drop_last)
    return idx_loader

def init_model(options):
    if options.flag_baseline == -1:
        validate_tcad6_options(options)
        if options.base_pe == 0:
            base_pe = 'none'
        elif options.base_pe == 1:
            base_pe = 'learned'
        elif options.base_pe == 2:
            base_pe = 'sinusoidal'
        else:
            assert False, 'wrong positional encoding'
        model = BPN(
                infeat_dim1=num_gate_types,
                infeat_dim2=num_module_types,
                hidden_dim=options.hidden_dim,
                device=device,
                alpha = options.alpha,
                beta = options.beta,
                base_pe=base_pe,
                flag_path_supervise=options.flag_path_supervise,
                flag_reverse=options.flag_reverse,
            ).to(device)
    elif  options.flag_baseline == 0:
        model = ACCNN(infeat_dim=num_gate_types+num_module_types,
                      hidden_dim=options.hidden_dim,
                      flag_homo=options.flag_homo)
    print("creating model:")
    print(model)

    return model

def filter_criticalPIs(pi2po_edges,k=5):
    res = ([],[],[])
    count = {}
    for i in range(len(pi2po_edges[0])):
        src = pi2po_edges[0][i]
        dst = pi2po_edges[1][i]
        w = pi2po_edges[2][i]
        count[dst] = count.get(dst,0) + 1
        if count[dst] > k:
            continue
        res[0].append(src)
        res[1].append(dst)
        res[2].append(w)
    return res

def init(seed):
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def gather_data(sampled_data,sampled_graphs,graphs_info,idx,flag_addedge):

    POs_label_chunks, PIs_delay_chunks = [], []
    start_idx = 0
    new_edges = ([], [])
    new_edges_weight_chunks = []
    for data in sampled_data:
        PIs_delay, POs_label, _, pi2po_edges = data['delay-label_pairs'][idx][:4]

        POs_label_chunks.append(th.as_tensor(POs_label, dtype=th.float, device=device).unsqueeze(-1))
        PIs_delay_chunks.append(th.as_tensor(PIs_delay, dtype=th.float, device=device).unsqueeze(-1))

        graph = data['graph']
        # collect the new edges from critical PIs to PO
        if flag_addedge:
            new_edges[0].extend([nid + start_idx for nid in pi2po_edges[0]])
            new_edges[1].extend([nid + start_idx for nid in pi2po_edges[1]])
            if len(pi2po_edges)==3:
                new_edges_weight_chunks.append(th.as_tensor(pi2po_edges[2], dtype=th.float, device=device))

        start_idx += graph.number_of_nodes()

    POs_label_all = th.cat(POs_label_chunks, dim=0)
    PIs_delay_all = th.cat(PIs_delay_chunks, dim=0)

    if flag_addedge:
        new_edges_feat = {}
        if new_edges_weight_chunks:
            new_edges_feat = {'w': th.cat(new_edges_weight_chunks, dim=0)}
        sampled_graphs.add_edges(th.as_tensor(new_edges[0], device=device), th.as_tensor(new_edges[1], device=device),
                                 data=new_edges_feat,etype='pi2po')


    sampled_graphs.ndata['delay'] = th.zeros((sampled_graphs.number_of_nodes(), 1), dtype=th.float, device=device)
    sampled_graphs.ndata['delay'][sampled_graphs.ndata['is_pi'] == 1] = PIs_delay_all
    # sampled_graphs.ndata['input_delay'] = th.zeros((sampled_graphs.number_of_nodes(), 1), dtype=th.float).to(device)
    # sampled_graphs.ndata['input_delay'][sampled_graphs.ndata['is_pi'] == 1] = PIs_delay_all

    return POs_label_all, PIs_delay_all, sampled_graphs,graphs_info



    
def get_batched_data(graphs,po_batch_size=2048):
    # po_batch_size = 4096

    sampled_graphs = dgl.batch(graphs)
    sampled_graphs = sampled_graphs.to(device)
    nodes_list = th.arange(sampled_graphs.number_of_nodes(), device=device)
    POs = nodes_list[sampled_graphs.ndata['is_po'] == 1]

    POs_batches = []
    num_pos = len(POs)
    num_po_batches = max(1, (num_pos + po_batch_size - 1) // po_batch_size)
    for batch_idx in range(num_po_batches):
        start = batch_idx * num_pos // num_po_batches
        end = (batch_idx + 1) * num_pos // num_po_batches
        POs_batches.append((POs[start:end], th.arange(start, end, device=device)))
    
    
    graphs_info = {}
    topo_levels = gen_topo(sampled_graphs)
    graphs_info['is_heter'] = is_heter(sampled_graphs)
    graphs_info['topo'] = [l.to(device) for l in topo_levels]
    cache_mtde_forward = (
        getattr(options, 'mtde_forward_cache', 'off') in ['cache', 'compare']
        or getattr(options, 'mtde_forward_impl', 'dgl') in ['scatter', 'compare']
    )
    if cache_mtde_forward:
        def compact_dst_positions(dst, dst_nodes):
            if dst.numel() == 0:
                return dst.new_empty(0)
            sorted_nodes, original_pos = th.sort(dst_nodes)
            sorted_pos = th.searchsorted(sorted_nodes, dst)
            if not th.equal(sorted_nodes[sorted_pos], dst):
                raise ValueError('MTDE cache contains an edge outside its destination level')
            return original_pos[sorted_pos]

        mtde_forward_cache = []
        for i, nodes in enumerate(graphs_info['topo']):
            entry = {'nodes': nodes}
            if i != 0:
                isModule_mask = sampled_graphs.ndata['is_module'][nodes] == 1
                isGate_mask = sampled_graphs.ndata['is_module'][nodes] == 0
                nodes_gate = nodes[isGate_mask]
                nodes_module = nodes[isModule_mask]
                gate_eids = sampled_graphs.in_edges(nodes_gate, form='eid', etype='intra_gate')
                gate_src, gate_dst = sampled_graphs.find_edges(gate_eids, etype='intra_gate')
                gate_dst_pos = compact_dst_positions(gate_dst, nodes_gate)
                module_eids = sampled_graphs.in_edges(nodes_module, form='eid', etype='intra_module')
                module_src, module_dst = sampled_graphs.find_edges(module_eids, etype='intra_module')
                module_dst_pos = compact_dst_positions(module_dst, nodes_module)

                module_bit_position = sampled_graphs.edges['intra_module'].data['bit_position'][module_eids]
                module_bit_position = module_bit_position.reshape(-1, 1)
                module_width2 = th.zeros((len(nodes_module), 1), dtype=module_bit_position.dtype, device=device)
                module_width2.index_add_(0, module_dst_pos, module_bit_position)
                module_counts = th.bincount(module_dst_pos, minlength=len(nodes_module)).clamp(min=1)
                module_width2 = module_width2 / module_counts.to(module_width2.dtype).unsqueeze(1)
                entry.update({
                    'nodes_gate': nodes_gate,
                    'gate_eids': gate_eids,
                    'gate_src': gate_src,
                    'gate_dst': gate_dst,
                    'gate_dst_pos': gate_dst_pos,
                    'gate_reverse_eids': sampled_graphs.out_edges(nodes_gate, form='eid', etype='reverse'),
                    'nodes_module': nodes_module,
                    'module_eids': module_eids,
                    'module_src': module_src,
                    'module_dst': module_dst,
                    'module_dst_pos': module_dst_pos,
                    'module_width2': module_width2,
                    'module_reverse_eids': sampled_graphs.out_edges(nodes_module, form='eid', etype='reverse'),
                })
            mtde_forward_cache.append(entry)
        graphs_info['mtde_forward_cache'] = mtde_forward_cache
    topo_r = gen_topo(sampled_graphs, flag_reverse=True)
    graphs_info['topo_r'] = [l.to(device) for l in topo_r]
    if 'reverse' in sampled_graphs.etypes and options.mtde_backward_impl in ['scatter', 'custom', 'compare']:
        graphs_info['topo_r_in_edges'] = [
            tuple(t.to(device) for t in sampled_graphs.in_edges(nodes, form='all', etype='reverse'))
            for nodes in graphs_info['topo_r'][1:]
        ]
    # level = gen_level(sampled_graphs)
    # graphs_info['level'] = [l.to(device) for l in level]
    #graphs_info['level'] = [[l[0].to(device),l[1].to(device),l[2].to(device)] for l in level]
    graphs_info['POs_batches'] = POs_batches
    
    return sampled_graphs,graphs_info


def configure_endpoint_batch(graphs_info, POs):
    graphs_info['POs'] = POs
    graphs_info['POs_origin'] = POs
    graphs_info['PO_cols'] = th.arange(len(POs), dtype=th.long, device=device)
    graphs_info['num_endpoint_cols'] = len(POs)

def select_train_po_batch_size(num_nodes, num_pos):
    if options.po_batch_size > 0:
        return options.po_batch_size

    po_bs = 1200
    if num_nodes > 180000 or (num_nodes > 145000 and num_pos > 800):
        po_bs = min(po_bs, max(256, int(num_pos / 2)))

    node_budget = getattr(options, 'po_batch_node_budget', 0)
    if node_budget > 0 and num_nodes > 0:
        po_bs = min(po_bs, max(256, int(node_budget / num_nodes)))

    if num_pos > 0:
        po_bs = min(po_bs, num_pos)
    return max(1, int(po_bs))

def init_criticality_matrix(graph, POs, graph_info):

    graph.ndata['hp'] = th.zeros(
        (graph.number_of_nodes(), graph_info['num_endpoint_cols']), dtype=th.float, device=device)
    graph.ndata['hp'][POs, graph_info['PO_cols']] = 1
    graph.ndata['is_critical'] = graph.ndata['hp'].bool()

    return graph

def cal_metrics(labels_hat,labels):
    valid_mask = valid_endpoint_mask(labels)
    labels_hat = labels_hat[valid_mask]
    labels = labels[valid_mask]
    if labels.numel() == 0:
        raise ValueError('cannot calculate metrics without any labeled endpoints')
    labels_hat_flat = labels_hat.reshape(-1)
    labels_flat = labels.reshape(-1)
    ss_res = th.sum((labels_flat - labels_hat_flat) ** 2)
    ss_tot = th.sum((labels_flat - th.mean(labels_flat)) ** 2)
    r2 = (1 - ss_res / ss_tot.clamp(min=1e-12)).item()
    mape = th.mean(th.abs(labels_hat[labels != 0] - labels[labels != 0]) / labels[labels != 0])
    ratio = labels_hat[labels != 0] / labels[labels != 0]
    min_ratio = th.min(ratio)
    max_ratio = th.max(ratio)

    return r2,mape,ratio,min_ratio,max_ratio

def inference(model,test_data,batch_size,usage,save_path,flag_save=False):
    prob_file = os.path.join(save_path, 'POs_criticalprob3_{}.pkl'.format(usage))
    labels_file = os.path.join(save_path, 'labels_hat3_{}.pkl'.format(usage))
    labels_file2 = os.path.join(save_path, 'labels_{}.pkl'.format(usage))

    with ((th.no_grad())):
        labels,labels_hat = None,None
        POs_criticalprob = None

        for i in range(0,len(test_data),batch_size):
            idxs = list(range(i,min(i+batch_size,len(test_data))))
            sampled_data = []
            num_cases = 100
            graphs = []
            for idx in idxs:
                data = test_data[idx]
                num_cases = min(num_cases,len(data['delay-label_pairs']))
                sampled_data.append(test_data[idx])
                graphs.append(data['graph'])
                #print(data['design_name'])

            # print(num_cases)
            flag_r = options.flag_reverse or options.flag_path_supervise
            po_batchsize =2048
            sampled_graphs, graphs_info = get_batched_data(graphs,po_batchsize)

            for POs,POs_mask in graphs_info['POs_batches']:
                POs_mask = POs_mask.to(device) if th.is_tensor(POs_mask) else th.tensor(POs_mask, device=device)
                #print(len(POs),len(POs_mask))
                # if len(POs)<2000:
                #     exit()
                configure_endpoint_batch(graphs_info, POs)
                #graphs_info['batched_POs_mask'] = POs_mask
                if flag_r:
                    sampled_graphs = init_criticality_matrix(sampled_graphs, POs, graphs_info)
                

                graphs_info['nodes_name'] = data['nodes_name']
                #print(data['design_name'])
                for j in range(num_cases):
                    #torch.cuda.empty_cache()
                    flag_addedge = options.flag_path_supervise or options.global_cat_choice in [3,4,5]
                    POs_label, PIs_delay, sampled_graphs, graphs_info = gather_data(sampled_data, sampled_graphs,
                                                                                    graphs_info, j, flag_addedge)
                    POs_label = POs_label[POs_mask]

                    cur_labels_hat, _,_,prob_sum,prob_dev,prob_ce,cur_POs_criticalprob,_ = model(sampled_graphs, graphs_info)[:8]

                    if flag_addedge:
                        sampled_graphs.remove_edges(sampled_graphs.edges('all', etype='pi2po')[2], etype='pi2po')

                    valid_mask = valid_endpoint_mask(POs_label)
                    if not th.any(valid_mask):
                        continue
                    POs_label = POs_label[valid_mask]
                    cur_labels_hat = cur_labels_hat[valid_mask]
                    cur_POs_criticalprob = filter_endpoint_rows(
                        cur_POs_criticalprob, valid_mask
                    )

                    labels_hat = cat_tensor(labels_hat,cur_labels_hat)

                    labels = cat_tensor(labels,POs_label)
                    POs_criticalprob = cat_tensor(POs_criticalprob,cur_POs_criticalprob)

                #     data['delay-label_pairs'][j] = (
                #         PIs_delay, POs_label, None, new_edges, cur_POs_criticalprob.detach().cpu().numpy().tolist())
                #
                # new_dataset.append((data['graph'], data))



        labels_hat[labels_hat>30] = 30
        labels_hat[labels_hat <0] = 0


        test_loss = Loss(labels_hat, labels).item()
        test_r2, test_mape, ratio,min_ratio, max_ratio = cal_metrics(labels_hat,labels)

        flag_save =False
        if flag_save:
            # with open(data_file,'wb') as f:
            #     pickle.dump(new_dataset,f)
            with open(prob_file, 'wb') as f:
                pickle.dump(POs_criticalprob.detach().cpu().numpy().tolist(), f)
            with open(labels_file, 'wb') as f:
                pickle.dump(labels_hat.detach().cpu().numpy().tolist(), f)
            with open(labels_file2, 'wb') as f:
                pickle.dump(labels.detach().cpu().numpy().tolist(), f)


        if POs_criticalprob is not None:
            mask1 = POs_criticalprob.squeeze(1) <= 0.05
            mask2 = POs_criticalprob.squeeze(1) > 0.5
            mask3 = th.logical_and(POs_criticalprob.squeeze(1) > 0.05,POs_criticalprob.squeeze(1) <=0.5)
            mask_l = labels.squeeze(1) != 0

            #print(th.mean(POs_criticalprob))
            print(th.mean(POs_criticalprob),len(labels[mask1]) / len(labels), len(labels[mask2]) / len(labels))
            # if len(labels_hat[mask1])>=2:
            #     temp_r2 = R2_score(labels_hat[mask1], labels[mask1]).item()
            #     temp_mape = th.mean(
            #         th.abs(labels_hat[th.logical_and(mask1, mask_l)] - labels[th.logical_and(mask1, mask_l)]) / labels[
            #             th.logical_and(mask1, mask_l)])
            #     #print(labels_hat[mask1],labels[mask1])
            #     print(temp_r2, temp_mape)
            # if len(labels_hat[mask2]) >= 2:
            #     temp_r2 = R2_score(labels_hat[mask2], labels[mask2]).item()
            #     temp_mape = th.mean(
            #         th.abs(labels_hat[th.logical_and(mask2, mask_l)] - labels[th.logical_and(mask2, mask_l)]) /
            #         labels[th.logical_and(mask2, mask_l)])
            #     print(temp_r2, temp_mape)
            # if len(labels_hat[mask3]) >= 2:
            #     temp_r3 = R2_score(labels_hat[mask3], labels[mask3]).item()
            #     temp_mape = th.mean(
            #         th.abs(labels_hat[th.logical_and(mask3, mask_l)] - labels[th.logical_and(mask3, mask_l)]) /
            #         labels[th.logical_and(mask3, mask_l)])
            #     print(temp_r3, temp_mape)

        x = []
        y = []
        indexs = list(range(9,40))
        for i,r in enumerate(indexs):
            r = r / 20
            if i==0:
                num = len(ratio[ratio<r+0.05])
            elif i== len(indexs)-1:
                num = len(ratio[ratio >= r])
            else:
                num = len(ratio[th.logical_and(ratio>=r,ratio<r+0.05)])
            x.append(r)
            y.append(num/len(ratio))
        #plt.bar(x,y)
        plt.xlabel('ratio')
        plt.ylabel('percent')
        plt.bar(x,y,width=0.03)
        #print(list(zip(x,y)))

        plt.savefig('bar2.png')
        max_label = max(th.max(labels_hat).item(),th.max(labels).item())
        plt.xlim(0, max_label)
        plt.ylim(0, max_label)
        plt.xlabel('predict')
        plt.ylabel('label')
        plt.scatter(labels_hat.detach().cpu().numpy().tolist(),labels.detach().cpu().numpy().tolist(),s=0.2)
        plt.savefig('scatter2.png')

        return labels_hat, labels,test_loss, test_r2,test_mape,min_ratio,max_ratio
    #model.flag_train = True

def test(model,test_data,flag_reverse,batch_size,po_bs=2048):
    flag_meta = options.flag_meta
    runtime_stats = enable_runtime_stats()
    stage_time = new_stage_time() if runtime_stats else None
    prediction_file = getattr(options, 'test_prediction_file', None)
    prediction_endpoint_chunks = [] if prediction_file else None
    prediction_case_position_chunks = [] if prediction_file else None
    prediction_case_index_chunks = [] if prediction_file else None

    num_cases = 100
    #num_cases = 10
    #batch_size = options.batch_size if flag_reverse else len(test_data)
    test_idx_loader = get_idx_loader(test_data, batch_size,False)
    model.flag_train = False
    model.eval()
    # print(len(test_data))
    # print(test_data[0])
    design_name = 'unknown'
    with (th.no_grad()):
        labels_chunks, labels_hat_chunks = [], []
        # for i in range(0,len(test_data),batch_size):
        #     idxs = list(range(i,min(i+batch_size,len(test_data))))
        corr_sim_avg = 0
        num_edp = 0
        metadata_all = None
        for batch, idxs in enumerate(test_idx_loader):
            if runtime_stats:
                start= time()
            idxs = idxs.numpy().tolist()
            sampled_data = []

            graphs = []
            for idx in idxs:
                data = test_data[idx]
                num_cases = min(num_cases,len(data['delay-label_pairs']))
                sampled_data.append(test_data[idx])
                graphs.append(data['graph'])

            flag_r = flag_reverse or options.flag_path_supervise


            sampled_graphs, graphs_info = get_batched_data(graphs,po_batch_size=po_bs)
            graphs_info['nodes_name'] = data['nodes_name']
            design_name = data['design_name']
            if runtime_stats:
                stage_time['other'] += time()-start

            for POs, POs_mask in graphs_info['POs_batches']:
                POs_mask = POs_mask.to(device) if th.is_tensor(POs_mask) else th.tensor(POs_mask, device=device)
                # print(len(POs),len(POs_mask))
                # if len(POs)<2000:
                #     exit()
                configure_endpoint_batch(graphs_info, POs)
                # graphs_info['batched_POs_mask'] = POs_mask
                if flag_r:
                    sampled_graphs = init_criticality_matrix(sampled_graphs, POs, graphs_info)


                for j in range(num_cases):
                    #if j in [0,50]: continue

                    #torch.cuda.empty_cache()
                    flag_addedge = options.flag_path_supervise or options.global_cat_choice in [3,4,5]
                    POs_label, PIs_delay, sampled_graphs,graphs_info = gather_data(sampled_data,sampled_graphs,graphs_info,j,flag_addedge)

                    POs_label = POs_label[POs_mask]

                    model_out = model(sampled_graphs, graphs_info, flag_meta, stage_time if runtime_stats else None)
                    cur_labels_hat,cur_labels_hat_residual,path_inputdelay,prob_sum,prob_dev,prob_ce,_,cur_metadata = model_out[:8]

                    if flag_addedge:
                        sampled_graphs.remove_edges(sampled_graphs.edges('all', etype='pi2po')[2], etype='pi2po')

                    valid_mask = valid_endpoint_mask(POs_label)
                    if not th.any(valid_mask):
                        continue
                    if prediction_file:
                        if len(sampled_data) != 1:
                            raise ValueError(
                                '--test_prediction_file requires test batch_size=1'
                            )
                        num_valid = int(valid_mask.sum().item())
                        case_indices = sampled_data[0].get('case_indices')
                        case_index = case_indices[j] if case_indices is not None else j
                        prediction_endpoint_chunks.append(
                            POs_mask[valid_mask].detach().cpu()
                        )
                        prediction_case_position_chunks.append(
                            th.full((num_valid,), j, dtype=th.long)
                        )
                        prediction_case_index_chunks.append(
                            th.full((num_valid,), case_index, dtype=th.long)
                        )
                    POs_label = POs_label[valid_mask]
                    cur_labels_hat = cur_labels_hat[valid_mask]
                    cur_labels_hat_residual = cur_labels_hat_residual[valid_mask]
                    path_inputdelay = path_inputdelay[valid_mask]
                    prob_sum = filter_endpoint_rows(prob_sum, valid_mask)
                    prob_dev = filter_endpoint_rows(prob_dev, valid_mask)
                    prob_ce = filter_endpoint_rows(prob_ce, valid_mask)
                    if cur_metadata is not None:
                        cur_metadata = {
                            key: (
                                value[valid_mask]
                                if th.is_tensor(value)
                                and value.ndim > 0
                                and value.shape[0] == valid_mask.shape[0]
                                else value
                            )
                            for key, value in cur_metadata.items()
                        }

                    if runtime_stats:
                        num_edp += len(prob_dev)
                        corr_sim_avg += th.sum(prob_dev)
                        #print(corr_sim_avg/num_edp)
                    cur_labels_hat = cur_labels_hat.clamp(min=0, max=30)

                    if flag_meta:
                        cur_labels_hat2 = cur_labels_hat_residual + path_inputdelay
                        cur_labels_hat2 = cur_labels_hat2.clamp(min=0, max=30)
                        cur_metadata['labels_hat1'] = cur_labels_hat
                        cur_metadata['labels_hat2'] = cur_labels_hat2
                        cur_metadata['labels_gt'] = POs_label

                        for key, value in cur_metadata.items():
                            cur_metadata[key] = value.squeeze(1).detach().cpu().numpy().tolist()
                        cur_metadata['case_idx'] = [j] * len(POs_label)
                        if metadata_all is None:
                            metadata_all = cur_metadata
                        else:
                            for key in metadata_all.keys():
                                metadata_all[key].extend(cur_metadata[key])

                    labels_hat_chunks.append(cur_labels_hat.detach().cpu())
                    labels_chunks.append(POs_label.detach().cpu())


        if runtime_stats:
            corr_sim_avg = corr_sim_avg / num_edp if num_edp else 0
            if th.is_tensor(corr_sim_avg):
                corr_sim_avg = corr_sim_avg.item()

        labels_hat = th.cat(labels_hat_chunks, dim=0)
        labels = th.cat(labels_chunks, dim=0)
        test_loss = Loss(labels_hat, labels).item()
        test_r2, test_mape, ratio, min_ratio, max_ratio = cal_metrics(labels_hat, labels)

        if prediction_file:
            prediction_file = prediction_file.replace('{design}', design_name)
            prediction_dir = os.path.dirname(prediction_file)
            if prediction_dir:
                os.makedirs(prediction_dir, exist_ok=True)
            th.save({
                'design_name': design_name,
                'labels': labels.squeeze(-1),
                'predictions': labels_hat.squeeze(-1),
                'endpoint_cols': th.cat(prediction_endpoint_chunks),
                'case_positions': th.cat(prediction_case_position_chunks),
                'case_indices': th.cat(prediction_case_index_chunks),
            }, prediction_file)
            print('Saved test predictions to {}'.format(prediction_file))

        if flag_meta:
            df = load_metadata(None,metadata_all)
            save_dir = os.path.join(options.checkpoint,options.predict_path,design_name)
            os.makedirs(save_dir,exist_ok=True)
            for key in metadata_all.keys():
                if key not in ['labels_hat1','labels_hat2','labels_gt']:
                    plot_feature_vs_error(df,key,save_dir=save_dir)

            for key in metadata_all.keys():
                metadata_all[key] = th.tensor(metadata_all[key],dtype=th.float)
                #print(key,metadata_all[key].shape)

        if runtime_stats:
            for key,value in stage_time.items():
                if key!='all':
                    stage_time['all'] += value

            runtime_str = 'Runtime: '
            for key,value in stage_time.items():
                runtime_str += '{}={}, '.format(key,round(value/num_cases,3))
            print(runtime_str)

        result = (labels_hat, labels,test_loss, test_r2,test_mape,min_ratio,max_ratio)
        return result + (corr_sim_avg, metadata_all) if runtime_stats else result + (metadata_all,)


def test_case_first(model, test_data, flag_reverse, batch_size, po_bs=2048):
    del batch_size  # Case-first evaluation intentionally processes one design at a time.
    flag_meta = options.flag_meta
    runtime_stats = enable_runtime_stats()
    stage_time = new_stage_time() if runtime_stats else None
    prediction_file = getattr(options, 'test_prediction_file', None)
    if prediction_file and len(test_data) != 1:
        raise ValueError('--test_prediction_file requires one design in case-first evaluation')

    prediction_endpoint_chunks = [] if prediction_file else None
    prediction_case_position_chunks = [] if prediction_file else None
    prediction_case_index_chunks = [] if prediction_file else None
    labels_chunks, labels_hat_chunks = [], []
    corr_sim_sum = 0
    num_edp = 0
    metadata_all = None
    design_name = 'unknown'
    total_cases = 0

    model.flag_train = False
    model.eval()
    with th.no_grad():
        for design_position, data in enumerate(test_data):
            if runtime_stats:
                start = time()
            sampled_graphs, graphs_info = get_batched_data([data['graph']], po_batch_size=po_bs)
            graphs_info['nodes_name'] = data['nodes_name']
            design_name = data['design_name']
            all_nodes = th.arange(sampled_graphs.number_of_nodes(), device=device)
            all_pos = all_nodes[sampled_graphs.ndata['is_po'] == 1]
            if runtime_stats:
                stage_time['other'] += time() - start

            num_cases = min(100, len(data['delay-label_pairs']))
            if getattr(options, 'eval_case_limit', 0) > 0:
                num_cases = min(num_cases, options.eval_case_limit)
            flag_r = flag_reverse or options.flag_path_supervise
            flag_addedge = options.flag_path_supervise or options.global_cat_choice in [3, 4, 5]
            for case_position in range(num_cases):
                total_cases += 1
                pos_labels, _, sampled_graphs, graphs_info = gather_data(
                    [data], sampled_graphs, graphs_info, case_position, flag_addedge)
                valid_cols = th.nonzero(
                    valid_endpoint_mask(pos_labels), as_tuple=False).squeeze(1)
                if valid_cols.numel() == 0:
                    if flag_addedge:
                        edge_ids = sampled_graphs.edges('all', etype='pi2po')[2]
                        if edge_ids.numel() != 0:
                            sampled_graphs.remove_edges(edge_ids, etype='pi2po')
                    continue

                num_po_batches = max(1, (len(valid_cols) + po_bs - 1) // po_bs)
                case_indices = data.get('case_indices')
                case_index = case_indices[case_position] if case_indices is not None else case_position
                if getattr(options, 'eval_mtde_cache', 'off') == 'cache':
                    graphs_info['eval_case_key'] = (design_position, case_position)
                else:
                    graphs_info.pop('eval_case_key', None)
                    graphs_info.pop('_mtde_eval_case_cache', None)

                for po_batch_index in range(num_po_batches):
                    start = po_batch_index * len(valid_cols) // num_po_batches
                    end = (po_batch_index + 1) * len(valid_cols) // num_po_batches
                    endpoint_cols = valid_cols[start:end]
                    pos = all_pos[endpoint_cols]
                    configure_endpoint_batch(graphs_info, pos)
                    if flag_r:
                        sampled_graphs = init_criticality_matrix(sampled_graphs, pos, graphs_info)

                    model_out = model(
                        sampled_graphs,
                        graphs_info,
                        flag_meta,
                        stage_time if runtime_stats else None,
                    )
                    (cur_labels_hat, cur_labels_hat_residual, path_inputdelay,
                     prob_sum, prob_dev, prob_ce, _, cur_metadata) = model_out[:8]
                    batch_labels = pos_labels[endpoint_cols]

                    if prediction_file:
                        num_valid = len(endpoint_cols)
                        prediction_endpoint_chunks.append(endpoint_cols.detach().cpu())
                        prediction_case_position_chunks.append(
                            th.full((num_valid,), case_position, dtype=th.long))
                        prediction_case_index_chunks.append(
                            th.full((num_valid,), case_index, dtype=th.long))

                    if runtime_stats:
                        num_edp += len(prob_dev)
                        corr_sim_sum += th.sum(prob_dev)
                    cur_labels_hat = cur_labels_hat.clamp(min=0, max=30)

                    if flag_meta:
                        cur_labels_hat2 = (cur_labels_hat_residual + path_inputdelay).clamp(min=0, max=30)
                        cur_metadata['labels_hat1'] = cur_labels_hat
                        cur_metadata['labels_hat2'] = cur_labels_hat2
                        cur_metadata['labels_gt'] = batch_labels
                        for key, value in cur_metadata.items():
                            cur_metadata[key] = value.squeeze(1).detach().cpu().numpy().tolist()
                        cur_metadata['case_idx'] = [case_position] * len(batch_labels)
                        if metadata_all is None:
                            metadata_all = cur_metadata
                        else:
                            for key in metadata_all:
                                metadata_all[key].extend(cur_metadata[key])

                    labels_hat_chunks.append(cur_labels_hat.detach().cpu())
                    labels_chunks.append(batch_labels.detach().cpu())

                if flag_addedge:
                    edge_ids = sampled_graphs.edges('all', etype='pi2po')[2]
                    if edge_ids.numel() != 0:
                        sampled_graphs.remove_edges(edge_ids, etype='pi2po')

            graphs_info.pop('eval_case_key', None)
            graphs_info.pop('_mtde_eval_case_cache', None)

        if not labels_chunks:
            raise ValueError('case-first evaluation found no labeled endpoints')
        corr_sim_avg = corr_sim_sum / num_edp if runtime_stats and num_edp else 0
        if th.is_tensor(corr_sim_avg):
            corr_sim_avg = corr_sim_avg.item()

        labels_hat = th.cat(labels_hat_chunks, dim=0)
        labels = th.cat(labels_chunks, dim=0)
        test_loss = Loss(labels_hat, labels).item()
        test_r2, test_mape, ratio, min_ratio, max_ratio = cal_metrics(labels_hat, labels)

        if prediction_file:
            prediction_file = prediction_file.replace('{design}', design_name)
            prediction_dir = os.path.dirname(prediction_file)
            if prediction_dir:
                os.makedirs(prediction_dir, exist_ok=True)
            th.save({
                'design_name': design_name,
                'labels': labels.squeeze(-1),
                'predictions': labels_hat.squeeze(-1),
                'endpoint_cols': th.cat(prediction_endpoint_chunks),
                'case_positions': th.cat(prediction_case_position_chunks),
                'case_indices': th.cat(prediction_case_index_chunks),
            }, prediction_file)
            print('Saved test predictions to {}'.format(prediction_file))

        if flag_meta:
            df = load_metadata(None, metadata_all)
            save_dir = os.path.join(options.checkpoint, options.predict_path, design_name)
            os.makedirs(save_dir, exist_ok=True)
            for key in metadata_all:
                if key not in ['labels_hat1', 'labels_hat2', 'labels_gt']:
                    plot_feature_vs_error(df, key, save_dir=save_dir)
            for key in metadata_all:
                metadata_all[key] = th.tensor(metadata_all[key], dtype=th.float)

        if runtime_stats:
            for key, value in stage_time.items():
                if key != 'all':
                    stage_time['all'] += value
            runtime_str = 'Runtime: '
            for key, value in stage_time.items():
                runtime_str += '{}={}, '.format(key, round(value / max(total_cases, 1), 3))
            print(runtime_str)

        result = (labels_hat, labels, test_loss, test_r2, test_mape, min_ratio, max_ratio)
        return result + (corr_sim_avg, metadata_all) if runtime_stats else result + (metadata_all,)


def use_case_first_eval(data, po_bs):
    eval_impl = getattr(options, 'eval_impl', 'legacy')
    if eval_impl != 'auto':
        return eval_impl == 'case_first'

    for graph_info in data:
        diagnostics = graph_info.get('parser_diagnostics', {})
        if diagnostics.get('missing_po_labels', 0) > 0:
            return True
        graph = graph_info['graph']
        num_pos = int(graph.ndata['is_po'].sum().item())
        if graph.number_of_nodes() >= 300000 and num_pos > po_bs:
            return True
    return False


def test_all(test_data,model,batch_size,flag_reverse,po_bs=2048,usage='test',flag_group=False,flag_infer=False,flag_save=False,save_file_dir=None):
    metadata_all = {}
    runtime_stats = enable_runtime_stats()
    print('Testing...')
    if flag_group:
        labels_hat_all_chunks, labels_all_chunks = [], []
        batch_sizes = [64, 32, 17, 8]
        batch_sizes = [16, 32]
        if len(test_data)!=2:
            batch_sizes = [1]*len(test_data)
        r2_list  = []
        mape_list = []
        for i, (name, data) in enumerate(test_data.items()):
            #torch.cuda.empty_cache()

            if options.test_po_batch_size > 0:
                po_bs = options.test_po_batch_size
            elif name in ['aes128']:
                po_bs = 3500
            else:
                po_bs = 10000

            test_out = None
            if flag_infer:
                labels_hat, labels, test_loss, test_r2, test_mape, test_min_ratio, test_max_ratio = inference(model, data,batch_sizes[i], usage,save_file_dir,flag_save)
            else:
                test_fn = test_case_first if use_case_first_eval(data, po_bs) else test
                test_out = test_fn(model, data,flag_reverse,batch_size,po_bs=po_bs)
                if runtime_stats:
                    labels_hat,labels,test_loss, test_r2, test_mape, test_min_ratio, test_max_ratio,corr_sim,metadata = test_out
                else:
                    labels_hat,labels,test_loss, test_r2, test_mape, test_min_ratio, test_max_ratio,metadata = test_out
                if options.flag_meta:
                    metadata_all[name] = metadata
            message = '\t{} {},\t#endpoints:{}\t loss={:.3f}\tr2={:.3f}\tmape={:.3f}\tmin_ratio={:.2f}\tmax_ratio={:.2f}'.format(
                usage,name, len(labels),test_loss, test_r2, test_mape, test_min_ratio, test_max_ratio)
            if runtime_stats:
                message += ', \tcorr_sim={:.3f}'.format(corr_sim)
            print(message)
            r2_list.append(test_r2)
            mape_list.append(test_mape)
            labels_hat_all_chunks.append(labels_hat.detach().cpu())
            labels_all_chunks.append(labels.detach().cpu())
            del labels_hat, labels
            if test_out is not None:
                del test_out
            if device.type == 'cuda':
                th.cuda.empty_cache()




        labels_hat_all = th.cat(labels_hat_all_chunks, dim=0)
        labels_all = th.cat(labels_all_chunks, dim=0)
        test_r2, test_mape, ratio, min_ratio, max_ratio = cal_metrics(labels_hat_all, labels_all)
        print(
            '\t{} overall\tr2={:.3f}\tmape={:.3f}\tmin_ratio={:.2f}\tmax_ratio={:.2f}'.format(
                usage,test_r2, test_mape, test_min_ratio, test_max_ratio))
        print(
            '\t{} avg\tr2={:.3f}\tmape={:.3f}'.format(
                usage, sum(r2_list)/len(r2_list),sum(mape_list)/len(mape_list)))
        # labels = labels_all.detach().cpu().numpy().tolist()
        # labels_hat = labels_hat_all.detach().cpu().numpy().tolist()
        # ratio = ratio.detach().cpu().numpy().tolist()
        # with open('predict2.pkl', 'wb') as f:
        #     pickle.dump((labels, labels_hat, ratio), f)
    else:
        if flag_infer:
            _, _, test_loss, test_r2, test_mape, test_min_ratio, test_max_ratio = inference(model, test_data,batch_size, usage,save_file_dir, flag_save)
        else:
            test_fn = test_case_first if use_case_first_eval(test_data, po_bs) else test
            test_out = test_fn(model, test_data,flag_reverse,batch_size,po_bs=po_bs)
            if runtime_stats:
                labels_hat_all, labels_all, test_loss, test_r2, test_mape, test_min_ratio, test_max_ratio,corr_sim,metadata = test_out
            else:
                labels_hat_all, labels_all, test_loss, test_r2, test_mape, test_min_ratio, test_max_ratio,metadata = test_out
            if options.flag_meta:
                metadata_all['all'] = metadata
            test_r2, test_mape, ratio, min_ratio, max_ratio = cal_metrics(labels_hat_all, labels_all)
            labels = labels_all.detach().cpu().numpy().tolist()
            labels_hat =  labels_hat_all.detach().cpu().numpy().tolist()
            ratio = ratio.detach().cpu().numpy().tolist()
            with open('predict.pkl','wb') as f:
                pickle.dump((labels,labels_hat,ratio),f)
        message = '\t{}: loss={:.3f}\tr2={:.3f}\tmape={:.3f}\tmin_ratio={:.2f}\tmax_ratio={:.2f}'.format(
            usage,test_loss, test_r2,test_mape,test_min_ratio,test_max_ratio)
        if runtime_stats:
            message += ', \tcorr_sim={:.3f}'.format(corr_sim)
        print(message)

    if options.flag_meta:
        save_dir = os.path.join(options.checkpoint, options.predict_path)
        os.makedirs(save_dir,exist_ok=True)
        save_file = os.path.join(save_dir,'metadata.pkl')
        with open(os.path.join(save_file),'wb') as f:
            pickle.dump(metadata_all,f)
        print(metadata_all)

    return test_r2,test_mape

def train(model):
    print(options)
    th.multiprocessing.set_sharing_strategy('file_system')

    train_data_savepath = options.data_savepath
    train_data = load_data('train',options)
    val_data, test_data = None, None

    print("Data successfully loaded")

    def ensure_val_data():
        nonlocal val_data
        if val_data is not None:
            return
        val_options = copy.copy(options)
        val_options.data_savepath = train_data_savepath
        val_options.flag_group = False
        val_data = load_data('test', val_options)
        print("Validation data successfully loaded")

    def ensure_test_data():
        nonlocal test_data
        if test_data is not None:
            return
        test_options = copy.copy(options)
        test_options.data_savepath = '../datasets/cases_round7_v5/heter_removepiPO_new_full/'
        test_options.flag_group = True
        test_data = load_data('test', test_options)
        for extra_path in getattr(options, 'extra_test_data_path', []):
            extra_options = copy.copy(test_options)
            extra_options.data_savepath = extra_path
            extra_options.quick = False
            extra_data = load_data('test', extra_options)
            for design_name, design_data in extra_data.items():
                if design_name in test_data:
                    raise ValueError('duplicate test design {!r} from {}'.format(
                        design_name, extra_path))
                test_data[design_name] = design_data
        print("Test data successfully loaded: {}".format(', '.join(test_data.keys())))

    train_idx_loader = get_idx_loader(train_data,options.batch_size,True)

    optim = th.optim.Adam(
        model.parameters(), options.learning_rate, weight_decay=options.weight_decay
    )
    scheduler = None
    if options.lr_scheduler:
        scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(
            optim,
            mode='min',
            factor=options.lr_scheduler_factor,
            patience=options.lr_scheduler_patience,
            min_lr=options.min_learning_rate,
        )
    ema = ModelEMA(model, options.ema_decay) if options.ema_decay > 0 else None
    if options.ema_start_epoch < 0:
        raise ValueError('--ema_start_epoch must be non-negative')
    if options.smooth_ccal and options.flag_alternate:
        raise ValueError('--smooth_ccal replaces --flag_alternate; do not enable both')

    print('Optimization controls: scheduler={} factor={} patience={} min_lr={} '
          'ema_decay={} ema_start_epoch={} ema_scheduler_source={} smooth_ccal={}'.format(
        options.lr_scheduler,
        options.lr_scheduler_factor,
        options.lr_scheduler_patience,
        options.min_learning_rate,
        options.ema_decay,
        options.ema_start_epoch,
        options.ema_scheduler_source,
        options.smooth_ccal,
    ))

    model.train()



    print("----------------Start training----------------")

    cur_time = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    num_traindata = len(train_data)
    for epoch in range(options.num_epoch):

        model.train()
        model.flag_train = True
        if ema is not None and epoch == options.ema_start_epoch:
            print('EMA updates begin after {} warmup epoch(s)'.format(options.ema_start_epoch), flush=True)

        flag_path = options.flag_path_supervise
        flag_reverse = options.flag_reverse
        #Loss = nn.L1Loss()
        if options.smooth_ccal:
            flag_path = True
        elif options.flag_alternate:
            if epoch%3!=0:
                flag_path = False

        ccal_weight, residual_weight = supervision_loss_weights(flag_path, options.smooth_ccal)

        model.flag_path_supervise = flag_path
        model.flag_reverse = flag_reverse

        #train_idx_loader.batch_size = options.batch_size if epoch%2==0 else options.batch_size*2

        print('Epoch {} ------------------------------------------------------------'.format(epoch+1))
        print('Supervision: path={} ccal_weight={:.6f} residual_weight={:.6f}'.format(
            flag_path, ccal_weight, residual_weight), flush=True)
        total_num,total_loss, total_r2 = 0,0.0,0

        for batch, idxs in enumerate(train_idx_loader):
            batch_start_time = time()
            batch_profile = new_train_profile() if enable_runtime_stats() else None
            if batch_profile is not None and device.type == 'cuda':
                th.cuda.reset_peak_memory_stats(device)
            #torch.cuda.empty_cache()
            sampled_data = []

            idxs = idxs.numpy().tolist()
            num_cases = 1000
            graphs = []
            num_nodes,num_pos = 0,0
            for idx in idxs:
                data = train_data[idx]
                num_cases = min(num_cases,len(data['delay-label_pairs']))
                shuffle(train_data[idx]['delay-label_pairs'])
                sampled_data.append(train_data[idx])
                graphs.append(data['graph'])
                num_nodes += data['graph'].number_of_nodes()
                num_pos += len(train_data[idx]['delay-label_pairs'][0][1])
            if options.debug_case_limit > 0:
                num_cases = min(num_cases, options.debug_case_limit)

            flag_r = flag_reverse or flag_path
            num_POs = 0
            total_labels, total_labels_hat, total_prob = [], [], []


            po_bs = select_train_po_batch_size(num_nodes, num_pos)
            if enable_runtime_stats():
                print('Batch graph: num_nodes={} num_pos={} po_bs={} node_po_product={}'.format(
                    num_nodes, num_pos, po_bs, num_nodes * po_bs), flush=True)
            #po_bs = 896
            if batch_profile is not None:
                profile_start = sync_timer_start()
            sampled_graphs, graphs_info = get_batched_data(graphs,po_batch_size=po_bs)
            if batch_profile is not None:
                sync_timer_add(batch_profile, 'batch_graph', profile_start)
            #print(len(graphs_info['POs_batches'][0][0]),sampled_graphs.number_of_nodes())

            for POs, POs_mask in graphs_info['POs_batches']:
                POs_mask = POs_mask.to(device) if th.is_tensor(POs_mask) else th.tensor(POs_mask, device=device)
                configure_endpoint_batch(graphs_info, POs)

                if flag_r:
                    sampled_graphs = init_criticality_matrix(sampled_graphs, POs, graphs_info)

                for i in range(num_cases):
                    #torch.cuda.empty_cache()
                    flag_addedge = flag_path or options.global_cat_choice in [3,4,5]
                    if batch_profile is not None:
                        profile_start = sync_timer_start()
                    POs_label, PIs_delay, sampled_graphs, graphs_info = gather_data(sampled_data, sampled_graphs,
                                                                                    graphs_info, i, flag_addedge)
                    if batch_profile is not None:
                        sync_timer_add(batch_profile, 'gather_data', profile_start)

                    POs_label = POs_label[POs_mask]
                    if batch_profile is not None:
                        stage_time = new_stage_time()
                        profile_start = sync_timer_start()
                        model_out = model(sampled_graphs, graphs_info, stage_time=stage_time)
                        sync_timer_add(batch_profile, 'model_forward', profile_start)
                        add_stage_time(batch_profile, stage_time)
                    else:
                        model_out = model(sampled_graphs, graphs_info)
                    labels_hat, labels_hat_residual,path_inputdelay,prob_sum,prob_dev,prob_ce,_,_ = model_out[:8]

                    valid_mask = valid_endpoint_mask(POs_label)
                    if not th.any(valid_mask):
                        if flag_addedge:
                            sampled_graphs.remove_edges(
                                sampled_graphs.edges('all', etype='pi2po')[2],
                                etype='pi2po',
                            )
                        continue
                    POs_label = POs_label[valid_mask]
                    labels_hat = labels_hat[valid_mask]
                    labels_hat_residual = labels_hat_residual[valid_mask]
                    path_inputdelay = path_inputdelay[valid_mask]
                    prob_sum = filter_endpoint_rows(prob_sum, valid_mask)
                    prob_dev = filter_endpoint_rows(prob_dev, valid_mask)
                    prob_ce = filter_endpoint_rows(prob_ce, valid_mask)
                    total_num += len(POs_label)
                    if batch_profile is not None:
                        profile_start = sync_timer_start()
                    train_loss = Loss(labels_hat, POs_label)
                    #print(th.any(th.isnan(train_loss)))
                    if flag_path:
                        #path_loss = th.mean(prob_sum )
                        #path_loss = th.mean(prob_sum-1*prob_dev)
                        #train_loss += -path_loss
                        #path_loss = prob_sum
                        # path_loss = prob_sum - 1 * prob_dev
                        # train_loss = th.mean((th.exp(1 - path_loss)) * th.abs(labels_hat-POs_label))
                        #train_loss = th.mean(th.abs(labels_hat-POs_label)) + th.mean(th.exp(1 - path_loss))
                        #train_loss = th.mean(th.abs(labels_hat-POs_label)) + th.mean(prob_ce)
                        #train_loss = th.mean(th.pow(labels_hat - POs_label,2)) + th.mean(prob_ce)
                        train_loss += ccal_weight * th.mean(prob_ce)
                        if options.flag_residual:
                            train_loss += residual_weight * Loss(
                                labels_hat_residual, POs_label-path_inputdelay)
                            #print(th.cat((labels_hat_residual+path_inputdelay-labels_hat,labels_hat-POs_label),dim=1))
                        #train_loss = th.mean((th.exp(1 - path_loss)) * th.pow(labels_hat - POs_label,2))
                    # if options.flag_residual:
                    #     train_loss += Loss(labels_hat_residual, POs_label - path_inputdelay)
                    num_POs += len(prob_sum)

                    total_prob.append(prob_sum.detach())
                    total_labels.append(POs_label.detach())
                    total_labels_hat.append(labels_hat.detach())

                    if i==num_cases-1:
                        with th.no_grad():
                            metric_labels = th.cat(total_labels, dim=0)
                            metric_labels_hat = th.cat(total_labels_hat, dim=0)
                            train_r2, train_mape, ratio, min_ratio, max_ratio = cal_metrics(metric_labels_hat, metric_labels)
                            path_loss_avg = 0.0
                            prob_avg = th.cat(total_prob, dim=0).mean().item() if total_prob else 0.0
                        batch_time = time() - batch_start_time
                        print('{}/{} train_loss:{:.3f}, {:.3f} {:.3f}\ttrain_r2:{:.3f}\ttrain_mape:{:.3f}, ratio:{:.2f}-{:.2f}\tbatch_time:{:.2f}s'.format((batch+1)*options.batch_size,num_traindata,train_loss.item(),path_loss_avg,prob_avg,train_r2,train_mape,min_ratio,max_ratio,batch_time), flush=True)
                    if batch_profile is not None:
                        sync_timer_add(batch_profile, 'loss_metric', profile_start)

                    if len(labels_hat) ==0:
                        continue

                    if batch_profile is not None:
                        profile_start = sync_timer_start()
                    optim.zero_grad()
                    train_loss.backward()
                    optim.step()
                    if ema is not None and epoch >= options.ema_start_epoch:
                        ema.update(model)
                    if batch_profile is not None:
                        sync_timer_add(batch_profile, 'backward_step', profile_start)
                    #torch.cuda.empty_cache()

                    if flag_addedge:
                        if batch_profile is not None:
                            profile_start = sync_timer_start()
                        sampled_graphs.remove_edges(sampled_graphs.edges('all',etype='pi2po')[2],etype='pi2po')
                        if batch_profile is not None:
                            sync_timer_add(batch_profile, 'graph_mutation', profile_start)

            if batch_profile is not None:
                print(format_profile(batch_profile), flush=True)
                if device.type == 'cuda':
                    print('Peak CUDA memory: allocated={:.2f}GiB reserved={:.2f}GiB'.format(
                        th.cuda.max_memory_allocated(device) / (1024 ** 3),
                        th.cuda.max_memory_reserved(device) / (1024 ** 3),
                    ), flush=True)
            if options.max_train_batches > 0 and batch + 1 >= options.max_train_batches:
                print('Reached max_train_batches={} at epoch {}'.format(options.max_train_batches, epoch), flush=True)
                break

        torch.cuda.empty_cache()
        print('End of epoch {}'.format(epoch))
        should_eval = options.eval_every > 0 and (epoch + 1) % options.eval_every == 0
        test_every = options.test_every if options.test_every > 0 else options.eval_every
        should_test = test_every > 0 and (epoch + 1) % test_every == 0
        val_r2 = val_mape = test_r2 = test_mape = float('nan')
        raw_val_r2 = raw_val_mape = float('nan')

        # Keep the scheduler trajectory identical to the raw Scheduler baseline.
        # A second validation pass is needed only after EMA has received updates.
        use_raw_scheduler_metric = (
            scheduler is not None
            and ema is not None
            and options.ema_scheduler_source == 'raw'
        )
        if should_eval and use_raw_scheduler_metric and ema.num_updates > 0:
            ensure_val_data()
            po_bs = 2048
            raw_val_r2, raw_val_mape = test_all(
                val_data,
                model,
                options.batch_size,
                po_bs=po_bs,
                usage='val',
                flag_reverse=flag_reverse or flag_path,
            )
            print('Raw validation for scheduler: r2={:.3f}, mape={:.3f}'.format(
                raw_val_r2, raw_val_mape), flush=True)

        eval_parameters = ema.average_parameters(model) if ema is not None else nullcontext()
        with eval_parameters:
            if should_eval:
                ensure_val_data()
                po_bs = 2048
                val_r2,val_mape = test_all(val_data,model,options.batch_size,po_bs=po_bs,usage='val',flag_reverse=flag_reverse or flag_path)
                if use_raw_scheduler_metric and ema.num_updates == 0:
                    raw_val_r2, raw_val_mape = val_r2, val_mape
            else:
                print('Skipping validation at epoch {} (eval_every={})'.format(epoch, options.eval_every))

            if should_test:
                ensure_test_data()
                po_bs = 2048
                test_r2, test_mape = test_all(test_data, model, 1, po_bs=po_bs, usage='test',
                                              flag_reverse=flag_reverse or flag_path, flag_group=True)
            else:
                print('Skipping test at epoch {} (test_every={})'.format(epoch, test_every))

            # Save the same parameters that produced the recorded metrics.
            if options.checkpoint:
                save_path = '../checkpoints/{}'.format(options.checkpoint)
                th.save(model.state_dict(), os.path.join(save_path,"{}.pth".format(epoch)))
                with open(os.path.join(checkpoint_path,'res.txt'),'a') as f:
                    if should_eval:
                        result = 'Epoch {}, val: {:.3f},{:.3f}'.format(epoch, val_r2, val_mape)
                        if should_test:
                            result += '; test: {:.3f},{:.3f}'.format(test_r2, test_mape)
                        else:
                            result += '; test skipped'
                        if ema is not None:
                            result += '; raw val: {:.3f},{:.3f}; ema_updates: {}'.format(
                                raw_val_r2, raw_val_mape, ema.num_updates)
                        f.write(result + '\n')
                    else:
                        result = 'Epoch {}, validation skipped'.format(epoch)
                        if should_test:
                            result += '; test: {:.3f},{:.3f}'.format(test_r2, test_mape)
                        else:
                            result += '; test skipped'
                        f.write(result + '\n')

        if scheduler is not None and should_eval:
            scheduler_mape = raw_val_mape if use_raw_scheduler_metric else val_mape
            scheduler_source = 'raw' if use_raw_scheduler_metric else ('ema' if ema is not None else 'raw')
            old_lr = optim.param_groups[0]['lr']
            scheduler.step(scheduler_mape)
            new_lr = optim.param_groups[0]['lr']
            print('Learning rate ({} val MAPE {:.6f}): {:.6g} -> {:.6g}'.format(
                scheduler_source, scheduler_mape, old_lr, new_lr), flush=True)


if __name__ == "__main__":




    seed = random.randint(1, 10000)
    seed = 5201
    init(seed)
    if options.test_iter:

        assert options.checkpoint, 'no checkpoint dir specified'
        model_save_path = '../checkpoints/{}/{}.pth'.format(options.checkpoint, options.test_iter)
        assert os.path.exists(model_save_path), 'start_point {} of checkpoint {} does not exist'.\
            format(options.test_iter, options.checkpoint)
        input_options = options
        options = th.load('../checkpoints/{}/options.pkl'.format(options.checkpoint))

        # options = merge_with_loaded(input_options, options)
        # options.use_pathgnn = True
        # options.path_feat_choice = 0
        # options.flag_rawpath = True
        # th.save(options,'../checkpoints/{}/options.pkl'.format(options.checkpoint))
        # exit()
        options.checkpoint = input_options.checkpoint
        options.data_savepath = input_options.data_savepath
        options.test_iter = input_options.test_iter
        options.quick = input_options.quick
        options.batch_size = input_options.batch_size
        options.gpu = input_options.gpu
        options.flag_group = input_options.flag_group
        options.predict_path = input_options.predict_path
        options.flag_meta = input_options.flag_meta
        options.log_level = input_options.log_level
        options.test_po_batch_size = input_options.test_po_batch_size
        options.eval_impl = input_options.eval_impl
        options.eval_mtde_cache = input_options.eval_mtde_cache
        options.eval_case_limit = input_options.eval_case_limit
        options.test_prediction_file = input_options.test_prediction_file


        options = merge_with_loaded(input_options,options)
        options.log_level = input_options.log_level
        

        logs_files = [f for f in os.listdir('../checkpoints/{}'.format(options.checkpoint)) if f.startswith('test') and '-' not in f and '_' not in f]
        logs_idx = [int(f[4:].split('.')[0]) for f in logs_files]
        log_idx = 1 if len(logs_idx)==0 else max(logs_idx)+1
        stdout_f = '../checkpoints/{}/test{}.log'.format(options.checkpoint,log_idx)
        with tee.StdoutTee(stdout_f):
            print(options)

            model = init_model(options)
            model.flag_train = True
            flag_inference = False
            po_bs = (
                options.test_po_batch_size
                if getattr(options, 'test_po_batch_size', 0) > 0
                else 2048
            )

            model = model.to(device)
            model.load_state_dict(th.load(model_save_path,map_location='cuda:{}'.format(options.gpu) if th.cuda.is_available() else "cpu" ))
            #usages = ['test','train']
            usages = ['test']
            #usages = ['train']

            for usage in usages:
                flag_save = True
                flag_infer = False
                save_file_dir = options.checkpoint
                test_data = load_data(usage,options)

                if len(test_data)==0:
                    continue
                #test_loss, test_r2, test_mape, test_min_ratio, test_max_ratio = test(model, test_data,options.flag_reverse)
                test_all(test_data,model,options.batch_size,options.flag_reverse,po_bs,'test',options.flag_group,flag_infer,flag_save,save_file_dir)

    elif options.checkpoint:
        print('saving logs and models to ../checkpoints/{}'.format(options.checkpoint))
        checkpoint_path = '../checkpoints/{}'.format(options.checkpoint)
        stdout_f = '../checkpoints/{}/stdout.log'.format(options.checkpoint)
        stderr_f = '../checkpoints/{}/stderr.log'.format(options.checkpoint)
        os.makedirs(checkpoint_path)  # exist not ok
        th.save(options, os.path.join(checkpoint_path, 'options.pkl'))
        with open(os.path.join(checkpoint_path,'res.txt'),'w') as f:
            pass
        with tee.StdoutTee(stdout_f), tee.StderrTee(stderr_f):

            if options.pretrain_dir is not None:
                #pretrain_options = th.load('../checkpoints/{}/options.pkl'.format(os.path.split(options.pretrain_dir)[0]))

                model = init_model(options)
                model.load_state_dict(th.load(options.pretrain_dir,map_location='cuda:{}'.format(options.gpu) if th.cuda.is_available() else "cpu"))

                if options.flag_continue_trainpath:
                    model.flag_transformer = True
                    d_in = model.infeat_dim1 if options.flag_rawpath else model.hidden_dim
                    model.pathformer = PathTransformer(d_in=d_in, d_model=model.hidden_dim, n_heads=4, n_layers=3, use_cls_token=True)
                    model.mlp_out_new = MLP(model.new_out_dim+model.hidden_dim, model.hidden_dim, model.hidden_dim, 1, negative_slope=0.1)
            else:
                model = init_model(options)
            model = model.to(device)

            print('seed:', seed)
            train(model)

    else:
        print('No checkpoint is specified. abandoning all model checkpoints and logs')
        model = init_model(options)
        if options.pretrain_dir is not None:
            model.load_state_dict(th.load(options.pretrain_dir,map_location='cuda:{}'.format(options.gpu) if th.cuda.is_available() else "cpu"))
        if options.flag_reverse:
            if options.pi_choice == 0: model.mlp_global_pi = MLP(2, int(options.hidden_dim / 2), options.hidden_dim)
            model.mlp_out_new = MLP(options.out_dim, options.hidden_dim, 1)
        model = model.to(device)
        train(model)
