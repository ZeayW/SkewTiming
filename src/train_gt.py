import matplotlib.pyplot as plt
import torch
import dgl
print(dgl.__version__)
print(torch.__version__)


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
import itertools

from gtmodel import GTModel
from dgl import LapPE
from dgl.nn import LapPosEncoder
import dgl.sparse as dglsp
from options_gt import get_options


options = get_options()
device = th.device("cuda:" + str(options.gpu) if th.cuda.is_available() else "cpu")
#device = th.device("cpu")
R2_score = R2Score().to(device)
Loss = nn.MSELoss()
Loss = nn.L1Loss()


with open(os.path.join(options.data_savepath, 'ntype2id.pkl'), 'rb') as f:
    ntype2id,ntype2id_gate,ntype2id_module = pickle.load(f)
num_gate_types = len(ntype2id_gate)
num_gate_types -= 3
num_module_types = len(ntype2id_module)
# print(num_gate_types,num_module_types)
# # print(ntype2id)
# print(ntype2id,ntype2id_gate,ntype2id_module)
# exit()

data_path = options.data_savepath
if data_path.endswith('/'):
    data_path = data_path[:-1]
data_file = os.path.join(data_path, 'data.pkl')
#split_file = os.path.join(data_path, 'split.pkl')

with open(data_file, 'rb') as f:
    data_all = pickle.load(f)
    design_names = [d[1]['design_name'].split('_')[-1] for d in data_all]

if 'round7' in data_path:
    split_file = os.path.join(data_path, 'split.pkl')
    designs_group = None
else:
    split_file = os.path.join(os.path.split(data_path)[0], 'split_new.pkl')
    with open('designs_group.pkl', 'rb') as f:
        designs_group = pickle.load(f)

with open(split_file, 'rb') as f:
    split_list = pickle.load(f)


def cat_tensor(t1,t2):
    if t1 is None:
        return t2
    elif t2 is None:
        return t1
    else:
        return th.cat((t1,t2),dim=0)
# print(split_list)
# exit()
def load_data(usage,flag_quick=True,flag_grouped=False):

    assert usage in ['train','val','test']

    target_list = split_list[usage]
    target_list = [n.split('_')[-1] for n in target_list]

    data = [d for i,d in enumerate(data_all) if design_names[i] in target_list]
    case_range = (0, 100)
    if flag_quick:
        if usage == 'train':
            case_range = (0,20)
        else:
            case_range = (0, 40)
    print("------------Loading {}_data #{} {}-------------".format(usage,len(data),case_range))

    loaded_data = []
    if flag_grouped:
        loaded_data = {}
    for  i,(graph,graph_info) in enumerate(data):
        #
        # graph_homo = heter2homo(graph)
        # graph.ndata['PE'] = dgl.laplacian_pe(graph_homo, k=options.pe_size, padding=True)
        # loaded_data.append((graph,graph_info))
        # continue
        if usage == 'test' and designs_group is None:
            if len(graph_info['delay-label_pairs'][0][1]) <= 150:
                continue
            if graph_info['design_name'] in ['tv80', 'sha3', 'ldpcenc', 'mc6809']: continue

        graph = heter2homo(graph)
        graph.ndata['feat_i'] = graph.ndata['ntype'][:,3:]

        if options.feat_choice ==1:
            graph.ndata['feat_i'] = th.cat((graph.ndata['feat_i'],graph.ndata['degree']),dim=1)
        elif options.feat_choice in [2,3]:
            nodes_topo = th.zeros((graph.number_of_nodes(), 1), dtype=th.float)
            topo = gen_topo(graph)
            for i,nids in enumerate(topo):
                nodes_topo[nids] = i
            graph.ndata['feat_i'] = th.cat((graph.ndata['feat_i'],nodes_topo),dim=1)
            if options.feat_choice == 3:
                graph.ndata['feat_i'] = th.cat((graph.ndata['feat_i'], graph.ndata['degree']), dim=1)
        elif options.feat_choice ==4:
            nodes_topo = th.zeros((graph.number_of_nodes(), 1), dtype=th.float)
            topo = gen_topo(graph)
            for i, nids in enumerate(topo):
                nodes_topo[nids] = i
            graph.ndata['feat_i'] = nodes_topo
        elif options.feat_choice == 5:
            graph.ndata['feat_i'] = graph.ndata['degree']

        if len(graph_info['delay-label_pairs'][0][0])!= len(graph.ndata['is_pi'][graph.ndata['is_pi'] == 1]):
            print('skip',graph_info['design_name'])
            continue

        graph_info['graph'] = graph
        graph_info['delay-label_pairs'] = graph_info['delay-label_pairs'][case_range[0]:case_range[1]]

        if flag_grouped:
            if designs_group is None:
                loaded_data[graph_info['design_name']] = loaded_data.get(graph_info['design_name'],[])
                loaded_data[graph_info['design_name']].append(graph_info)
            else:
                group_id = designs_group[graph_info['design_name']]
                loaded_data[group_id] = loaded_data.get(group_id,[])
                loaded_data[group_id].append(graph_info)
        else:
            loaded_data.append(graph_info)


    # with open('data_{}.pkl'.format(usage),'wb') as f:
    #     pickle.dump(loaded_data,f)
    #exit()
    return loaded_data

def get_idx_loader(data,batch_size):
    drop_last = True
    sampler = SubsetRandomSampler(th.arange(len(data)))
    idx_loader = DataLoader([i for i in range(len(data))], sampler=sampler, batch_size=batch_size,
                            drop_last=drop_last)
    return idx_loader


def init_model(options):
    # model = Graphormer(infeat_dim=num_gate_types+num_module_types+1,
    #                    feat_dim=128,
    #                    hidden_dim=256)
    in_size = 1
    if options.feat_choice in [0,1,2,3]:
        in_size += num_gate_types+num_module_types
    if options.feat_choice in [1,2,4,5]:
        in_size += 1
    elif options.feat_choice == 3:
        in_size += 2
    model = GTModel(in_size=in_size,
                    out_size=1,
                    hidden_size=options.hidden_dim,
                    pos_enc_size=options.pe_size)
    print("creating model:")
    print(model)

    return model


def init(seed):
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# def get_pos_encoding(g):
#     # transform = LapPE(k=128, feat_name='eigvec', eigval_name='eigval', padding=True)
#     # g = transform(g)
#     # eigvals, eigvecs = g.ndata['eigval'], g.ndata['eigvec']
#     # transformer_encoder = LapPosEncoder(
#     #     model_type="Transformer", num_layer=3, k=128, dim=64, n_head=4
#     # )
#     # pos_encoding=transformer_encoder(eigvals, eigvecs)
#     # deepset_encoder = LapPosEncoder(
#     #     model_type="DeepSet", num_layer=3, k=5, dim=16, num_post_layer=2
#     # )
#     # pos_encoding=deepset_encoder(eigvals, eigvecs)
#
#     return pos_encoding


def gather_data(sampled_data,sampled_graphs,idx):

    POs_label_all, PIs_delay_all = None, None
    for data in sampled_data:
        PIs_delay, POs_label, POs_baselabel, pi2po_edges = data['delay-label_pairs'][idx][:4]

        POs_label_all = cat_tensor(POs_label_all, th.tensor(POs_label, dtype=th.float).unsqueeze(-1).to(device))
        PIs_delay_all = cat_tensor(PIs_delay_all, th.tensor(PIs_delay, dtype=th.float).unsqueeze(-1).to(device))


    #graphs_info['label'] = POs_label_all
    feat_delay = th.zeros((sampled_graphs.number_of_nodes(), 1), dtype=th.float).to(device)
    feat_delay[sampled_graphs.ndata['is_pi'] == 1] = PIs_delay_all
    sampled_graphs.ndata['feat'] = th.cat((sampled_graphs.ndata['feat_i'],feat_delay),dim=1)


    return POs_label_all, sampled_graphs




def test(model,test_data,batch_size):

    with (th.no_grad()):
        labels_all,labels_hat_all = None,None
        for i in range(0,len(test_data),batch_size):
            idxs = list(range(i,min(i+batch_size,len(test_data))))
            sampled_data = []
            num_cases = 100
            graphs = []
            for idx in idxs:
                data = test_data[idx]
                num_cases = min(num_cases,len(data['delay-label_pairs']))
                sampled_data.append(test_data[idx])
                #data['graph'].ndata['PE'] = dgl.laplacian_pe(data['graph'], k=options.pe_size, padding=True)
                graphs.append(data['graph'])

            sampled_graphs = dgl.batch(graphs)
            sampled_graphs = sampled_graphs.to(device)
            indices = torch.stack(sampled_graphs.edges())
            N = sampled_graphs.num_nodes()
            A = dglsp.spmatrix(indices, shape=(N, N))

            for i in range(num_cases):
                node_delay = []
                labels = None
                torch.cuda.empty_cache()
                labels, sampled_graphs = gather_data(sampled_data, sampled_graphs, i)

                labels_hat = model(sampled_graphs,A)

                labels_hat_all = cat_tensor(labels_hat_all,labels_hat)
                labels_all = cat_tensor(labels_all,labels)

        test_loss = Loss(labels_hat_all, labels_all).item()
        test_r2 = R2_score(labels_hat_all, labels_all).item()
        test_mape = th.mean(th.abs(labels_hat_all[labels_all != 0] - labels_all[labels_all != 0]) / labels_all[labels_all != 0])
        ratio = labels_hat_all[labels_all != 0] / labels_all[labels_all != 0]
        min_ratio = th.min(ratio)
        max_ratio = th.max(ratio)

        return labels_hat,labels,test_loss, test_r2,test_mape,min_ratio,max_ratio

def cal_metrics(labels_hat,labels):
    r2 = R2_score(labels_hat, labels).item()
    mape = th.mean(th.abs(labels_hat[labels != 0] - labels[labels != 0]) / labels[labels != 0])
    ratio = labels_hat[labels != 0] / labels[labels != 0]
    min_ratio = th.min(ratio)
    max_ratio = th.max(ratio)

    return r2,mape,ratio,min_ratio,max_ratio

def test_all(test_data,model,batch_size,usage='test',flag_group=False):
    if flag_group:
        labels_hat_all, labels_all = None, None
        batch_sizes = [64, 32, 17, 8]
        if len(test_data)!=4:
            batch_sizes = [1]*len(test_data)
        for i, (name, data) in enumerate(test_data.items()):
            torch.cuda.empty_cache()
            labels_hat,labels,test_loss, test_r2, test_mape, test_min_ratio, test_max_ratio = test(model, data,batch_size)
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
        _, _, test_loss, test_r2, test_mape, test_min_ratio, test_max_ratio = test(model, test_data,batch_size)
        print(
            '\t{}: loss={:.3f}\tr2={:.3f}\tmape={:.3f}\tmin_ratio={:.2f}\tmax_ratio={:.2f}'.format(usage,test_loss, test_r2,test_mape,test_min_ratio,test_max_ratio))

    return test_r2,test_mape
def train(model):
    print(options)
    th.multiprocessing.set_sharing_strategy('file_system')

    train_data = load_data('train',options.quick)
    val_data = load_data('val',options.quick)
    test_data = load_data('test',options.quick,flag_grouped=options.flag_group)
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

                #data['graph'].ndata['PE'] = get_pos_encoding(data['graph'])
                #print(data['graph'].ndata['PE'].shape)
                #data['graph'].ndata['PE'] = dgl.laplacian_pe(data['graph'], k=options.pe_size, padding=True)
                graphs.append(data['graph'])

            sampled_graphs = dgl.batch(graphs)
            sampled_graphs = sampled_graphs.to(device)
            indices = torch.stack(sampled_graphs .edges())
            N = sampled_graphs .num_nodes()
            A = dglsp.spmatrix(indices, shape=(N, N))

            total_labels,total_labels_hat = None,None

            #num_cases = 1
            for i in range(num_cases):
                torch.cuda.empty_cache()
                labels, sampled_graphs = gather_data(sampled_data, sampled_graphs, i)
                labels_hat = model(sampled_graphs,A)
                train_loss = Loss(labels_hat, labels)
                total_labels = cat_tensor(total_labels, labels)
                total_labels_hat = cat_tensor(total_labels_hat, labels_hat)

                if i==num_cases-1:
                    train_r2 = R2_score(total_labels_hat, total_labels)
                    train_mape = th.mean(th.abs(total_labels_hat[total_labels != 0] - total_labels[total_labels != 0]) / total_labels[total_labels != 0])
                    ratio = total_labels_hat[total_labels != 0] / total_labels[total_labels != 0]
                    min_ratio = th.min(ratio)
                    max_ratio = th.max(ratio)
                    #print(data['design_name'],len(total_labels),num_POs)
                    #print(model.attention_vector_g[0].item(),model.attention_vector_m[0].item())
                    print('{}/{} train_loss:{:.3f}\ttrain_r2:{:.3f}\ttrain_mape:{:.3f}, ratio:{:.2f}-{:.2f}'.format((batch+1)*options.batch_size,num_traindata,train_loss.item(),train_r2.item(),train_mape.item(),min_ratio,max_ratio))

                if len(labels_hat) ==0:
                    continue

                torch.cuda.empty_cache()
                optim.zero_grad()
                train_loss.backward()
                optim.step()
                torch.cuda.empty_cache()

        torch.cuda.empty_cache()
        val_r2, val_mape = test_all(val_data, model, options.batch_size, usage='val',)
        test_r2, test_mape = test_all(test_data, model, options.batch_size, usage='test',
                                      flag_group='True')

        torch.cuda.empty_cache()
        print('End of epoch {}'.format(epoch))
        print('\tval:  loss={:.3f}\tr2={:.3f}\tmape={:.3f}\tmin_ratio={:.2f}\tmax_ratio={:.2f}'.format(val_loss,val_r2,val_mape,val_min_ratio,val_max_ratio))
        print('\ttest: loss={:.3f}\tr2={:.3f}\tmape={:.3f}\tmin_ratio={:.2f}\tmax_ratio={:.2f}'.format(test_loss,test_r2,test_mape,test_min_ratio,test_max_ratio))
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

        options.batch_size = input_options.batch_size
        options.gpu = input_options.gpu
        options.quick = input_options.quick
        options.flag_group = input_options.flag_group

        logs_files = [f for f in os.listdir('../checkpoints/{}'.format(options.checkpoint)) if f.startswith('test')]
        logs_idx = [int(f[4:].split('.')[0]) for f in logs_files]
        log_idx = 1 if len(logs_idx)==0 else max(logs_idx)+1
        stdout_f = '../checkpoints/{}/test{}.log'.format(options.checkpoint, log_idx)
        with tee.StdoutTee(stdout_f):
            print(options)

            model = init_model(options)

            model = model.to(device)
            model.load_state_dict(th.load(model_save_path,map_location='cuda:{}'.format(options.gpu)))
            usages = ['train','test','val']
            usages = ['test']
            for usage in usages:
                flag_save = True
                save_file_dir = options.checkpoint
                test_data = load_data(usage,options.quick,flag_grouped=options.flag_group)

                test_all(test_data, model, options.batch_size, usage='test',flag_group=options.flag_group)

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
                model.load_state_dict(th.load(options.pretrain_dir,map_location='cuda:{}'.format(options.gpu)))

            model = model.to(device)

            print('seed:', seed)
            train(model)

    else:
        print('No checkpoint is specified. abandoning all model checkpoints and logs')
        model = init_model(options)
        if options.pretrain_dir is not None:
            model.load_state_dict(th.load(options.pretrain_dir,map_location='cuda:{}'.format(options.gpu)))

        model = model.to(device)
        train(model)