import matplotlib.pyplot as plt
import torch
from queue import Queue
from options import get_options
#from model import *
from model2 import *
import pickle
import numpy as np
import os
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



options = get_options()
device = th.device("cuda:" + str(options.gpu) if th.cuda.is_available() else "cpu")
R2_score = R2Score().to(device)
Loss = nn.MSELoss()
Loss = nn.L1Loss()

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

    data_path = options.data_savepath
    if data_path.endswith('/'):
        data_path = data_path[:-1]
    data_file = os.path.join(data_path, 'data.pkl')
    if 'round7' in data_path:
        split_file = os.path.join(data_path, 'split.pkl')
        designs_group = None
    else:
        split_file = os.path.join(os.path.split(data_path)[0], 'split_new.pkl')
        with open('designs_group.pkl', 'rb') as f:
            designs_group = pickle.load(f)
    with open(data_file, 'rb') as f:
        data_all = pickle.load(f)
        design_names = [d[1]['design_name'].split('_')[-1] for d in data_all]

    with open(split_file, 'rb') as f:
        split_list = pickle.load(f)

    target_list = split_list[usage]
    target_list = [n.split('_')[-1] for n in target_list]
    #print(target_list[:10])


    data = [d for i,d in enumerate(data_all) if design_names[i] in target_list]
    case_range = (0, 100)
    if options.quick:
        if usage == 'train':
            case_range = (0,20)
        else:
            case_range = (0, 40)
    print("------------Loading {}_data #{} {}-------------".format(usage,len(data),case_range))

    loaded_data = []
    if options.test_iter or usage=='test' and options.flag_group:
        loaded_data = {}
    for  graph,graph_info in data:
        #print(graph_info['design_name'])
        #if int(graph_info['design_name'].split('_')[-1]) in [54, 96, 131, 300, 327, 334, 397]:
        #    continue
        name2nid = {graph_info['nodes_name'][i]:i for i in range(len(graph_info['nodes_name']))}
        if usage == 'test' and designs_group is None:

            if len(graph_info['delay-label_pairs'][0][1]) < 150:
                continue
            if graph_info['design_name'] in ['tv80', 'sha3', 'ldpcenc', 'mc6809']: continue
        #if graph_info['design_name'] not in ['s15850']: continue
        
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


        # print(th.sum(graph.ndata['value'][:,0])+th.sum(graph.ndata['value'][:,1])+th.sum(graph.ndata['is_pi'])+th.sum(graph.ndata['feat']),th.sum(graph.ndata['ntype']))

        #graph.ndata['degree'] = graph.ndata['width']

        graph.ndata['feat_module'] = graph.ndata['ntype_module']
        graph.ndata['feat_gate'] = graph.ndata['ntype_gate'][:,3:]
        graph_info['POs_feat'] = graph_info['POs_level_max'].unsqueeze(-1)
        graph.ndata['h'] = th.ones((graph.number_of_nodes(), options.hidden_dim), dtype=th.float)
        graph.ndata['PO_feat'] = th.zeros((graph.number_of_nodes(), 1), dtype=th.float)
        graph.ndata['PO_feat'][graph.ndata['is_po']==1] = graph_info['POs_feat']

        #print(graph_info['design_name'],len(graph_info['delay-label_pairs']))
        if len(graph_info['delay-label_pairs'][0][0])!= len(graph.ndata['is_pi'][graph.ndata['is_pi'] == 1]):
            print('skip',graph_info['design_name'])
            continue

        if options.flag_reverse or options.flag_path_supervise:
            graph = add_reverse_edges(graph)

        if options.flag_path_supervise or options.global_cat_choice in [3,4,5]:
            graph = add_newEtype(graph,'pi2po',([],[]),{})

        graph_info['graph'] = graph
        #graph_info['PI_mask'] = PI_mask
        graph_info['delay-label_pairs'] = graph_info['delay-label_pairs'][case_range[0]:case_range[1]]
        #if options.flag_group:

        if options.test_iter or usage=='test' and options.flag_group:
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

def get_idx_loader(data,batch_size):
    drop_last = False
    sampler = SubsetRandomSampler(th.arange(len(data)))
    idx_loader = DataLoader([i for i in range(len(data))], sampler=sampler, batch_size=batch_size,
                            drop_last=drop_last)
    return idx_loader

def init_model(options):
    if options.flag_baseline == -1:
        model = TimeConv(
                infeat_dim1=num_gate_types,
                infeat_dim2=num_module_types,
                hidden_dim=options.hidden_dim,
                global_out_choice=options.global_out_choice,
                global_cat_choice=options.global_cat_choice,
                global_info_choice= options.global_info_choice,
                inv_choice= options.inv_choice,
                flag_degree=options.flag_degree,
                flag_width=options.flag_width,
                flag_delay_pd=options.flag_delay_pd,
                flag_delay_m=options.flag_delay_m,
                flag_delay_g=options.flag_delay_g,
                flag_delay_pi=options.flag_delay_pi,
                flag_ntype_g=options.flag_ntype_g,
                flag_path_supervise=options.flag_path_supervise,
                flag_filter = options.flag_filter,
                flag_reverse=options.flag_reverse,
                flag_splitfeat=options.split_feat,
                agg_choice=options.agg_choice,
                attn_choice=options.attn_choice,
                flag_homo=options.flag_homo,
                flag_global=options.flag_global,
                flag_attn=options.flag_attn
            ).to(device)
    elif  options.flag_baseline == 0:
        model = ACCNN(infeat_dim=num_gate_types+num_module_types,
                      hidden_dim=options.hidden_dim,
                      flag_homo=options.flag_homo)
    print("creating model:")
    print(model)

    return model


def init(seed):
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def gather_data(sampled_data,sampled_graphs,graphs_info,idx,flag_addedge):

    POs_label_all, PIs_delay_all,POs_criticalprob_all = None, None,None
    start_idx = 0
    new_edges, new_edges_weight = ([], []), []
    for data in sampled_data:
        PIs_delay, POs_label, POs_baselabel, pi2po_edges = data['delay-label_pairs'][idx][:4]

        POs_label_all = cat_tensor(POs_label_all, th.tensor(POs_label, dtype=th.float).unsqueeze(-1).to(device))
        PIs_delay_all = cat_tensor(PIs_delay_all, th.tensor(PIs_delay, dtype=th.float).unsqueeze(-1).to(device))

        graph = data['graph']
        # collect the new edges from critical PIs to PO
        if flag_addedge:
            new_edges[0].extend([nid + start_idx for nid in pi2po_edges[0]])
            new_edges[1].extend([nid + start_idx for nid in pi2po_edges[1]])
            if len(pi2po_edges)==3:
                new_edges_weight.extend(pi2po_edges[2])

        if len(data['delay-label_pairs'][idx]) == 5:
            POs_criticalprob = data['delay-label_pairs'][idx][4]
            POs_criticalprob_all = cat_tensor(POs_criticalprob_all, th.tensor(POs_criticalprob, dtype=th.float).to(device))

        start_idx += graph.number_of_nodes()

    if flag_addedge:
        new_edges_feat = {}
        if len(new_edges_weight) > 0:
            new_edges_feat = {'w': th.tensor(new_edges_weight, dtype=th.float).unsqueeze(1).to(device)}
        sampled_graphs.add_edges(th.tensor(new_edges[0]).to(device), th.tensor(new_edges[1]).to(device),
                                 etype='pi2po')


    if POs_criticalprob_all is not None:
        prob_mask = POs_criticalprob_all.squeeze(1) <= 0.5
        prob_mask = th.logical_and(POs_criticalprob_all.squeeze(1) >=0.1,POs_criticalprob_all.squeeze(1) < 0.5)
        POs_label_all = POs_label_all[prob_mask]
        new_POs = graphs_info['POs_origin'][prob_mask]
        graphs_info['POs'] = new_POs
        sampled_graphs.ndata['is_po'] = th.zeros((sampled_graphs.number_of_nodes(), 1)).to(device)
        sampled_graphs.ndata['is_po'][new_POs] = 1
        graphs_info['POs_mask'] = (sampled_graphs.ndata['is_po'] == 1).squeeze(-1).to(device)
        #POs_label = POs_label[prob_mask]
        sampled_graphs.ndata['hd'] = -1000 * th.ones((sampled_graphs.number_of_nodes(), len(new_POs)),
                                                     dtype=th.float).to(device)
        sampled_graphs.ndata['hp'] = th.zeros((sampled_graphs.number_of_nodes(), len(new_POs)),
                                              dtype=th.float).to(device)
        for j, po in enumerate(new_POs):
            sampled_graphs.ndata['hp'][po][j] = 1
            sampled_graphs.ndata['hd'][po][j] = 0

    #POs_label_all = POs_label_all[graphs_info['batched_POs_mask']]
    #graphs_info['label'] = POs_label_all

    sampled_graphs.ndata['delay'] = th.zeros((sampled_graphs.number_of_nodes(), 1), dtype=th.float).to(device)
    sampled_graphs.ndata['delay'][sampled_graphs.ndata['is_pi'] == 1] = PIs_delay_all
    sampled_graphs.ndata['input_delay'] = th.zeros((sampled_graphs.number_of_nodes(), 1), dtype=th.float).to(device)
    sampled_graphs.ndata['input_delay'][sampled_graphs.ndata['is_pi'] == 1] = PIs_delay_all

    return POs_label_all, PIs_delay_all, sampled_graphs,graphs_info



    
def get_batched_data(graphs,po_batch_size=2048):


    sampled_graphs = dgl.batch(graphs)
    sampled_graphs = sampled_graphs.to(device)
    nodes_list = th.tensor(range(sampled_graphs.number_of_nodes())).to(device)
    POs = nodes_list[sampled_graphs.ndata['is_po'] == 1]
    POs_list = POs.cpu().numpy().tolist()

    POs_batches = []
    if len(POs)>po_batch_size:
        num_batch = int(len(POs)/po_batch_size)+1
        for b in range(num_batch):
            batched_POs_list = POs_list[b*po_batch_size:(b+1)*po_batch_size]
            POs_mask = list(range(b*po_batch_size,min((b+1)*po_batch_size,len(POs_list))))
            POs_batches.append((th.tensor(batched_POs_list).to(device),POs_mask))
    else:
        POs_mask = list(range(len(POs)))
        POs_batches.append((POs,POs_mask))
    
    
    graphs_info = {}
    topo_levels = gen_topo(sampled_graphs)
    graphs_info['is_heter'] = is_heter(sampled_graphs)
    graphs_info['topo'] = [l.to(device) for l in topo_levels]
    topo_r = gen_topo(sampled_graphs, flag_reverse=True)
    graphs_info['topo_r'] = [l.to(device) for l in topo_r]
    graphs_info['POs_batches'] = POs_batches
    
    return sampled_graphs,graphs_info


def init_criticality_matrix(graph,POs):

    graph.ndata['hd'] = -1000 * th.ones((graph.number_of_nodes(), len(POs)), dtype=th.float).to(
        device)
    graph.ndata['hp'] = th.zeros((graph.number_of_nodes(), len(POs)), dtype=th.float).to(device)
    for k, po in enumerate(POs.detach().cpu().numpy().tolist()):
        graph.ndata['hp'][po][k] = 1
        graph.ndata['hd'][po][k] = 0

    return graph

def cal_metrics(labels_hat,labels):
    r2 = R2_score(labels_hat, labels).item()
    mape = th.mean(th.abs(labels_hat[labels != 0] - labels[labels != 0]) / labels[labels != 0])
    ratio = labels_hat[labels != 0] / labels[labels != 0]
    min_ratio = th.min(ratio)
    max_ratio = th.max(ratio)

    return r2,mape,ratio,min_ratio,max_ratio

def inference(model,test_data,batch_size,usage,save_path,flag_save=False):
    prob_file = os.path.join(save_path, 'POs_criticalprob3_{}.pkl'.format(usage))
    labels_file = os.path.join(save_path, 'labels_hat3_{}.pkl'.format(usage))
    labels_file2 = os.path.join(save_path, 'labels_{}.pkl'.format(usage))
    data_file = os.path.join(save_path, 'data_{}.pkl'.format(usage))

    new_dataset = []

    #model.flag_train = False
    with ((th.no_grad())):
        total_num, total_loss, total_r2 = 0, 0.0, 0
        labels,labels_hat = None,None
        POs_criticalprob = None
        temp_labels, temp_labels_hat = None, None
        POs_topo = None

        for i in range(0,len(test_data),batch_size):
            #print(i,len(test_data),batch_size)
            idxs = list(range(i,min(i+batch_size,len(test_data))))
        # for batch, idxs in enumerate(test_idx_loader):
        #     idxs = idxs.numpy().tolist()
            abnormal_POs = {}
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
            po_batchsize =1024
            sampled_graphs, graphs_info = get_batched_data(graphs,po_batchsize)

            for POs,POs_mask in graphs_info['POs_batches']:
                POs_mask = th.tensor(POs_mask).to(device)
                #print(len(POs),len(POs_mask))
                # if len(POs)<2000:
                #     exit()
                graphs_info['POs'] = POs
                graphs_info['POs_origin'] = POs
                #graphs_info['batched_POs_mask'] = POs_mask
                if flag_r:
                    sampled_graphs = init_criticality_matrix(sampled_graphs,POs)
                

                graphs_info['nodes_name'] = data['nodes_name']
                #print(data['design_name'])
                for j in range(num_cases):
                    torch.cuda.empty_cache()
                    flag_addedge = options.flag_path_supervise or options.global_cat_choice in [3,4,5]
                    POs_label, PIs_delay, sampled_graphs, graphs_info = gather_data(sampled_data, sampled_graphs,
                                                                                    graphs_info, j, flag_addedge)
                    POs_label = POs_label[POs_mask]

                    cur_labels_hat, prob_sum,prob_dev,cur_POs_criticalprob = model(sampled_graphs, graphs_info)

                    ratio = cur_labels_hat[POs_label!=0] / POs_label[POs_label!=0]
                    # error_mask = ratio>10
                    # error_pos = POs[POs_label.squeeze(1)!=0][error_mask].cpu().numpy().tolist()
                    # erro_pos_name = [data['nodes_name'][n] for n in error_pos]
                    # error_l = POs_label[POs_label!=0][error_mask].cpu().numpy().tolist()
                    # error_l = [n for n in error_l]
                    # error_lh = cur_labels_hat[POs_label!=0][error_mask].cpu().numpy().tolist()
                    # error_lh = [round(n, 2) for n in error_lh]
                    # if len(error_pos)!=0:
                    #     print(j)
                    #     print(list(zip(error_pos,erro_pos_name,error_l,error_lh)))


                        # nodes_name = data['nodes_name']
                        # po_nid = error_pos[0]
                        # cur_nids = Queue()
                        # cur_nids.put(po_nid)
                        # all = []
                        # homo_g = sampled_graphs
                        #
                        # visited = {}
                        # while cur_nids.qsize()>0:
                        #     cur_nid = cur_nids.get()
                        #     if not visited.get(cur_nid,False):
                        #         visited[cur_nid] = True
                        #
                        #         preds = homo_g.predecessors(cur_nid)
                        #         all.append(cur_nid)
                        #         #print(cur_nid,nodes_name[cur_nid])
                        #         for n in preds:
                        #             if not visited.get(n,False):
                        #                 cur_nids.put(n.item())
                        #
                        # fanin_nodes = [nodes_name[n] for n in all]
                        # print(set(fanin_nodes))
                        # print(set(POs_name).intersection(fanin_nodes))
                    # if j==20: exit()

                    labels_hat = cat_tensor(labels_hat,cur_labels_hat)

                    labels = cat_tensor(labels,POs_label)
                    POs_criticalprob = cat_tensor(POs_criticalprob,cur_POs_criticalprob)

                    if not options.flag_path_supervise:
                        continue

                    new_edges = [[], []]
                    new_edges[0] = sampled_graphs.edges(etype='pi2po')[0].detach().cpu().numpy().tolist()
                    new_edges[1] = sampled_graphs.edges(etype='pi2po')[1].detach().cpu().numpy().tolist()
                    if flag_addedge:
                        sampled_graphs.remove_edges(sampled_graphs.edges('all', etype='pi2po')[2], etype='pi2po')
                #     data['delay-label_pairs'][j] = (
                #         PIs_delay, POs_label, None, new_edges, cur_POs_criticalprob.detach().cpu().numpy().tolist())
                #
                # new_dataset.append((data['graph'], data))



        # labels_hat[labels_hat>30] = 30
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

def test(model,test_data,flag_reverse,batch_size):

    #batch_size = options.batch_size if flag_reverse else len(test_data)
    test_idx_loader = get_idx_loader(test_data, batch_size)
    model.flag_train = False
    # print(len(test_data))
    # print(test_data[0])
    with (th.no_grad()):
        labels,labels_hat = None,None
        # for i in range(0,len(test_data),batch_size):
        #     idxs = list(range(i,min(i+batch_size,len(test_data))))
        for batch, idxs in enumerate(test_idx_loader):
            idxs = idxs.numpy().tolist()
            sampled_data = []
            num_cases = 100
            graphs = []
            for idx in idxs:
                data = test_data[idx]
                num_cases = min(num_cases,len(data['delay-label_pairs']))
                sampled_data.append(test_data[idx])
                graphs.append(data['graph'])

            flag_r = flag_reverse or options.flag_path_supervise

            sampled_graphs, graphs_info = get_batched_data(graphs)

            for POs, POs_mask in graphs_info['POs_batches']:
                POs_mask = th.tensor(POs_mask).to(device)
                # print(len(POs),len(POs_mask))
                # if len(POs)<2000:
                #     exit()
                graphs_info['POs'] = POs
                graphs_info['POs_origin'] = POs
                # graphs_info['batched_POs_mask'] = POs_mask
                if flag_r:
                    sampled_graphs = init_criticality_matrix(sampled_graphs, POs)

                for j in range(num_cases):
                    torch.cuda.empty_cache()
                    flag_addedge = options.flag_path_supervise or options.global_cat_choice in [3,4,5]
                    POs_label, PIs_delay, sampled_graphs,graphs_info = gather_data(sampled_data,sampled_graphs,graphs_info,j,flag_addedge)


                    POs_label = POs_label[POs_mask]

                    cur_labels_hat, prob_sum,prob_dev,_ = model(sampled_graphs, graphs_info)


                    labels_hat = cat_tensor(labels_hat,cur_labels_hat)
                    labels = cat_tensor(labels,POs_label)

                    if flag_addedge:
                        sampled_graphs.remove_edges(sampled_graphs.edges('all', etype='pi2po')[2], etype='pi2po')

        test_loss = Loss(labels_hat, labels).item()
        test_r2, test_mape, ratio, min_ratio, max_ratio = cal_metrics(labels_hat, labels)

        return labels_hat, labels,test_loss, test_r2,test_mape,min_ratio,max_ratio


def test_all(test_data,model,batch_size,flag_reverse,usage='test',flag_group=False,flag_infer=False,flag_save=False,save_file_dir=None):
    print('Testing...')
    if flag_group:
        labels_hat_all, labels_all = None, None
        batch_sizes = [64, 32, 17, 8]
        batch_sizes = [8, 32]
        if len(test_data)!=2:
            batch_sizes = [1]*len(test_data)
        for i, (name, data) in enumerate(test_data.items()):
            torch.cuda.empty_cache()
            # print(len(test_data))
            # continue
            if flag_infer:
                labels_hat, labels, test_loss, test_r2, test_mape, test_min_ratio, test_max_ratio = inference(model, data,batch_sizes[i], usage,save_file_dir,flag_save)
            else:
                labels_hat,labels,test_loss, test_r2, test_mape, test_min_ratio, test_max_ratio = test(model, data,flag_reverse,batch_size)
            print(
                '\t{} {},\t#endpoints:{}\t loss={:.3f}\tr2={:.3f}\tmape={:.3f}\tmin_ratio={:.2f}\tmax_ratio={:.2f}'.format(
                    usage,name, len(labels),test_loss, test_r2, test_mape, test_min_ratio, test_max_ratio))
            labels_hat_all = cat_tensor(labels_hat_all, labels_hat)
            labels_all = cat_tensor(labels_all, labels)
        test_r2, test_mape, ratio, min_ratio, max_ratio = cal_metrics(labels_hat_all, labels_all)
        print(
            '\t{} overall\tr2={:.3f}\tmape={:.3f}\tmin_ratio={:.2f}\tmax_ratio={:.2f}'.format(
                usage,test_r2, test_mape, test_min_ratio, test_max_ratio))
    else:
        if flag_infer:
            _, _, test_loss, test_r2, test_mape, test_min_ratio, test_max_ratio = inference(model, test_data,batch_size, usage,save_file_dir, flag_save)
        else:
            _, _, test_loss, test_r2, test_mape, test_min_ratio, test_max_ratio = test(model, test_data,flag_reverse,batch_size)
        print(
            '\t{}: loss={:.3f}\tr2={:.3f}\tmape={:.3f}\tmin_ratio={:.2f}\tmax_ratio={:.2f}'.format(usage,test_loss, test_r2,test_mape,test_min_ratio,test_max_ratio))

    return test_r2,test_mape

def train(model):
    print(options)
    th.multiprocessing.set_sharing_strategy('file_system')

    train_data = load_data('train',options)
    val_data = load_data('val',options)
    test_data = load_data('test',options)
    print("Data successfully loaded")

    train_idx_loader = get_idx_loader(train_data,options.batch_size)

    optim = th.optim.Adam(
        model.parameters(), options.learning_rate, weight_decay=options.weight_decay
    )

    model.train()



    print("----------------Start training----------------")

    cur_time = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    num_traindata = len(train_data)
    for epoch in range(options.num_epoch):


        model.flag_train = True

        flag_path = options.flag_path_supervise
        flag_reverse = options.flag_reverse
        Loss = nn.L1Loss()
        if options.flag_alternate:
            if epoch%2!=0:
                flag_path = False
            # if epoch%3==0:
            #     flag_reverse = False
            #     Loss = nn.L1Loss()
            # else:
            #     flag_path = False
            #     Loss = nn.MSELoss()
            #     if options.flag_reverse:
            #         train_idx_loader = get_idx_loader(train_data, options.batch_size )
            #     else:
            #         train_idx_loader = get_idx_loader(train_data, options.batch_size*2)

        model.flag_path_supervise = flag_path
        model.flag_reverse = flag_reverse

        #train_idx_loader.batch_size = options.batch_size if epoch%2==0 else options.batch_size*2

        print('Epoch {} ------------------------------------------------------------'.format(epoch+1))
        total_num,total_loss, total_r2 = 0,0.0,0

        for batch, idxs in enumerate(train_idx_loader):
            torch.cuda.empty_cache()
            sampled_data = []

            idxs = idxs.numpy().tolist()
            num_cases = 1000
            graphs = []

            for idx in idxs:
                data = train_data[idx]
                num_cases = min(num_cases,len(data['delay-label_pairs']))
                shuffle(train_data[idx]['delay-label_pairs'])
                sampled_data.append(train_data[idx])
                graphs.append(data['graph'])

            flag_r = flag_reverse or flag_path
            num_POs, totoal_path_loss,total_prob = 0,0,0
            total_labels,total_labels_hat = None,None

            sampled_graphs, graphs_info = get_batched_data(graphs)
            for POs, POs_mask in graphs_info['POs_batches']:
                POs_mask = th.tensor(POs_mask).to(device)
                graphs_info['POs'] = POs
                graphs_info['POs_origin'] = POs

                if flag_r:
                    sampled_graphs = init_criticality_matrix(sampled_graphs, POs)

                for i in range(num_cases):
                    torch.cuda.empty_cache()
                    flag_addedge = flag_path or options.global_cat_choice in [3,4,5]
                    POs_label, PIs_delay, sampled_graphs, graphs_info = gather_data(sampled_data, sampled_graphs,
                                                                                    graphs_info, i, flag_addedge)
                    POs_label = POs_label[POs_mask]
                    labels_hat,prob_sum,prob_dev,_ = model(sampled_graphs, graphs_info)
                    total_num += len(POs_label)
                    train_loss = Loss(labels_hat, POs_label)
                    path_loss =th.tensor(0.0)

                    if flag_path:
                        #path_loss = th.mean(prob_sum )
                        #path_loss = th.mean(prob_sum-1*prob_dev)
                        #train_loss += -path_loss
                        #path_loss = prob_sum
                        path_loss = prob_sum - 1 * prob_dev
                        train_loss = th.mean((th.exp(1 - path_loss)) * th.abs(labels_hat-POs_label))
                        #train_loss = th.mean((th.exp(1 - path_loss)) * th.pow(labels_hat - POs_label,2))

                    num_POs += len(prob_sum)

                    totoal_path_loss += th.mean(path_loss).item()*len(prob_sum)
                    total_prob += th.sum(prob_sum)
                    total_labels = cat_tensor(total_labels, POs_label)
                    total_labels_hat = cat_tensor(total_labels_hat, labels_hat)

                    if i==num_cases-1:
                        train_r2, train_mape, ratio, min_ratio, max_ratio = cal_metrics(total_labels_hat, total_labels)
                        path_loss_avg = totoal_path_loss / num_POs
                        prob_avg = total_prob / num_POs
                        print('{}/{} train_loss:{:.3f}, {:.3f} {:.3f}\ttrain_r2:{:.3f}\ttrain_mape:{:.3f}, ratio:{:.2f}-{:.2f}'.format((batch+1)*options.batch_size,num_traindata,train_loss.item(),path_loss_avg,prob_avg,train_r2,train_mape,min_ratio,max_ratio))

                    if len(labels_hat) ==0:
                        continue

                    optim.zero_grad()
                    train_loss.backward()
                    optim.step()
                    torch.cuda.empty_cache()

                    if flag_addedge:
                        sampled_graphs.remove_edges(sampled_graphs.edges('all',etype='pi2po')[2],etype='pi2po')

        torch.cuda.empty_cache()
        model.flag_train = False
        # _,_,val_loss, val_r2,val_mape,val_min_ratio,val_max_ratio = test(model, val_data,flag_reverse or flag_path)
        # _,_,test_loss, test_r2,test_mape,test_min_ratio,test_max_ratio = test(model,test_data,flag_reverse or flag_path)

        model.flag_train = True
        torch.cuda.empty_cache()
        print('End of epoch {}'.format(epoch))
        val_r2,val_mape = test_all(val_data,model,options.batch_size,usage='val',flag_reverse=flag_reverse or flag_path)
        test_r2, test_mape = test_all(test_data,model,options.batch_size,usage='test',flag_reverse=flag_reverse or flag_path,flag_group='True')
        model.flag_train = True
        torch.cuda.empty_cache()
        if options.checkpoint:
            save_path = '../checkpoints/{}'.format(options.checkpoint)
            th.save(model.state_dict(), os.path.join(save_path,"{}.pth".format(epoch)))
            with open(os.path.join(checkpoint_path,'res.txt'),'a') as f:
                f.write('Epoch {}, val: {:.3f},{:.3f}; test:{:.3f},{:.3f}\n'.format(epoch,val_r2,val_mape,test_r2,test_mape))


if __name__ == "__main__":




    seed = random.randint(1, 10000)
    init(seed)
    if options.test_iter:

        assert options.checkpoint, 'no checkpoint dir specified'
        model_save_path = '../checkpoints/{}/{}.pth'.format(options.checkpoint, options.test_iter)
        assert os.path.exists(model_save_path), 'start_point {} of checkpoint {} does not exist'.\
            format(options.test_iter, options.checkpoint)
        input_options = options
        options = th.load('../checkpoints/{}/options.pkl'.format(options.checkpoint))
        options.data_savepath = input_options.data_savepath
        options.checkpoint = input_options.checkpoint
        options.test_iter = input_options.test_iter
        options.quick = input_options.quick
        options.batch_size = input_options.batch_size
        options.gpu = input_options.gpu
        options.flag_path_supervise = input_options.flag_path_supervise
        #options.flag_reverse = input_options.flag_reverse

        options.inv_choice = input_options.inv_choice
        options.remove01 = input_options.remove01
        options.flag_baseline = input_options.flag_baseline
        options.global_out_choice = input_options.global_out_choice
        options.flag_group = input_options.flag_group



        logs_files = [f for f in os.listdir('../checkpoints/{}'.format(options.checkpoint)) if f.startswith('test') and '-' not in f and '_' not in f]
        logs_idx = [int(f[4:].split('.')[0]) for f in logs_files]
        log_idx = 1 if len(logs_idx)==0 else max(logs_idx)+1
        stdout_f = '../checkpoints/{}/test{}.log'.format(options.checkpoint,log_idx)
        with tee.StdoutTee(stdout_f):
            print(options)
            # exit()
            model = init_model(options)
            model.flag_train = True
            flag_inference = True

            #if True:
            # if options.flag_reverse and not options.flag_path_supervise:
            #     if options.pi_choice == 0: model.mlp_global_pi = MLP(2, int(options.hidden_dim / 2), options.hidden_dim)
            #     model.mlp_out_new = MLP(options.out_dim, options.hidden_dim, 1)
            model = model.to(device)
            model.load_state_dict(th.load(model_save_path,map_location='cuda:{}'.format(options.gpu) if th.cuda.is_available() else "cpu" ))
            usages = ['train','test']
            #usages = ['test']

            for usage in usages:
                flag_save = True
                flag_infer = True
                save_file_dir = options.checkpoint
                test_data = load_data(usage,options)
                if len(test_data)==0:
                    continue
                #test_loss, test_r2, test_mape, test_min_ratio, test_max_ratio = test(model, test_data,options.flag_reverse)
                test_all(test_data,model,options.batch_size,options.flag_reverse,'test',options.flag_group,flag_infer,flag_save,save_file_dir)
                # if options.flag_group:
                #     labels_hat_all,labels_all = None,None
                #     for i,data in enumerate(test_data):
                #         labels_hat,labels,test_loss, test_r2, test_mape, test_min_ratio, test_max_ratio = inference(model, data,batch_sizes[i], usage,save_file_dir, flag_save)
                #         #labels_hat,labels,test_loss, test_r2, test_mape, test_min_ratio, test_max_ratio = test(model, data,options.flag_reverse)
                #         print(
                #             '\tgroup:{},\t loss={:.3f}\tr2={:.3f}\tmape={:.3f}\tmin_ratio={:.2f}\tmax_ratio={:.2f}'.format(
                #                 i,test_loss, test_r2, test_mape, test_min_ratio, test_max_ratio))
                #         labels_hat_all = cat_tensor(labels_hat_all,labels_hat)
                #         labels_all = cat_tensor(labels_all, labels)
                #     test_r2, test_mape, ratio, min_ratio, max_ratio = cal_metrics(labels_hat_all, labels_all)
                #     print(
                #         '\toverall\tr2={:.3f}\tmape={:.3f}\tmin_ratio={:.2f}\tmax_ratio={:.2f}'.format(
                #             test_r2, test_mape, test_min_ratio, test_max_ratio))
                # else:
                #     _,_,test_loss, test_r2,test_mape,test_min_ratio,test_max_ratio = inference(model, test_data,options.batch_size,usage,save_file_dir,flag_save)
                #     print(
                #         '\ttest: loss={:.3f}\tr2={:.3f}\tmape={:.3f}\tmin_ratio={:.2f}\tmax_ratio={:.2f}'.format(test_loss, test_r2,
                #                                                                                                  test_mape,test_min_ratio,test_max_ratio))

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
            model = init_model(options)

            if options.pretrain_dir is not None:
                model.load_state_dict(th.load(options.pretrain_dir,map_location='cuda:{}'.format(options.gpu) if th.cuda.is_available() else "cpu"))


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