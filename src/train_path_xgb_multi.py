import matplotlib.pyplot as plt
import torch
import dgl
print(dgl.__version__)
print(torch.__version__)

from model import PathModel
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

from options import get_options
import pandas as pd
# from preprocess import *
import sklearn
import xgboost as xgb



options = get_options()
device = th.device("cuda:" + str(options.gpu) if th.cuda.is_available() else "cpu")
#device = th.device("cpu")
R2_score = R2Score().to(device)
Loss = nn.MSELoss()
Loss = nn.L1Loss()


data_path = options.data_savepath
if data_path.endswith('/'):
    data_path = data_path[:-1]
data_file = os.path.join(data_path, 'data.pkl')

with open(data_file, 'rb') as f:
    max_len,data_all = pickle.load(f)
    design_names = [d['design_name'].split('_')[-1] for d in data_all]


max_len = 81


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


def gather_data(data,index,flag_train):

    wrong_pi = set()
    all_pi = set()
    random_paths = data['random_paths']
    critical_paths, pi2delay, POs_label = data['critical_path'][index]
    POs_feat = []
    for po_idx, rand_paths_info in enumerate(random_paths):
        feat_po = []
        critical_path_info = critical_paths[po_idx]
        feat_global = []
        feat_global.append(critical_path_info['rank'])
        feat_global.append(critical_path_info['rank_ratio'])
        feat_global.append(rand_paths_info['num_nodes'])
        # feat_global.append(rand_paths_info['num_seq'])
        # feat_global.append(rand_paths_info['num_cmb'])
        feat_global.append(rand_paths_info['num_reg'])

        paths_info = [critical_path_info]
        if not flag_train:
            shuffle(rand_paths_info['paths_rd'])
            paths_info.extend(rand_paths_info['paths_rd'])

        # if len(paths_info)>5:
        #     print(data['design_name'],len(paths_info))
        for p in paths_info:
            feat_path = []
            pi = p['path'][0]
            p_len = len(p['path'])
            all_pi.add(pi)
            if pi2delay.get(pi,None) is None:
                wrong_pi.add(pi)
                input_delay = 0
            else:
                input_delay = pi2delay[pi]

            feat_path.append(input_delay)
            feat_path.append(p_len)
            feat_path.extend(p['path_ntype'])
            #print('\t',len(feat_global),len(feat_path),len(p['path_degree']))
            feat_path.extend(p['path_degree'])

            feat_path.extend([0]*(max_len-p_len))

            feat = feat_global.copy()

            feat.extend(feat_path)
            feat_po.append(feat)

        if flag_train:
            POs_feat.append(feat_po[0])
        else:
            POs_feat.append(feat_po)


    if index == 0 and len(wrong_pi) != 0: print( data['design_name'],len(wrong_pi),len(all_pi))

    return POs_label,POs_feat




# print(split_list)
# exit()
def load_data(usage,flag_grouped=False,flag_quick=True):

    assert usage in ['train','val','test']

    target_list = split_list[usage]
    target_list = [n.split('_')[-1] for n in target_list]
    # print(target_list)
    # print(design_names)
    dataset = [d for i,d in enumerate(data_all) if design_names[i] in target_list]
    case_range = (0, 100)
    if flag_quick:
        if usage == 'train':
            case_range = (0,20)
        else:
            case_range = (0, 40)
    print("------------Loading {}_data #{} {}-------------".format(usage,len(dataset),case_range))

    labels,feats = {}, {}

    labels, feat = [],[]
    if usage=='train' or not flag_grouped:
        loaded_data = []
    else:
        loaded_data = {}

    for data in dataset:
        if usage == 'test' and designs_group is None:
            if len(gather_data(data,0,usage=='train')[0]) <= 150:
                continue
            if data['design_name'] in [ 'tv80', 'sha3', 'ldpcenc', 'mc6809']: continue
        # if data['design_name'] not in ['y_quantizer']: continue

        print(data['design_name'])
        end_idx = min(case_range[1],len(data['critical_path']))
        for i in range(case_range[0],end_idx):
            cur_label,cur_feat = gather_data(data,i,usage=='train')
            # labels.extend(cur_label)
            # feat.extend(cur_feat)
            #
            if usage=='train':
                for j in range(len(cur_feat)):
                    labels.extend(cur_label)
                    feat.extend(cur_feat)
            elif flag_grouped:
                if designs_group is None:
                    loaded_data[data['design_name']] = loaded_data.get(data['design_name'],[[],[]])
                    loaded_data[data['design_name']][0].extend(cur_label)
                    loaded_data[data['design_name']][1].extend(cur_feat)
                else:
                    group_id = designs_group[data['design_name']]
                    loaded_data[group_id] = loaded_data.get(group_id,[[],[]])
                    loaded_data[group_id][0].extend(cur_label)
                    loaded_data[group_id][1].extend(cur_feat)
            else:
                labels.extend(cur_label)
                feat.extend(cur_feat)

    if usage=='train' or not flag_grouped:
        loaded_data = [np.array(labels),np.array(feat)]
    # else:
    #     for group,data in loaded_data.items():
    #         loaded_data[group] = (np.array(data[0]),np.array(data[1]))

    return loaded_data

    # feat = np.array(feat)
    # labels = np.array(labels)
    # return feat,labels



def init(seed):
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def cal_metrics(labels_hat,labels):
    r2 = R2_score(labels_hat, labels).item()
    mape = th.mean(th.abs(labels_hat[labels != 0] - labels[labels != 0]) / labels[labels != 0])
    smape = th.mean(th.abs(labels_hat - labels) / (th.abs(labels)+th.abs(labels_hat)))
    ratio = labels_hat[labels != 0] / labels[labels != 0]
    min_ratio = th.min(ratio)
    max_ratio = th.max(ratio)

    return r2,mape,smape,ratio,min_ratio,max_ratio

def test(data,model):
    labels, feats = data
    labels_hat = []
    #feat, labels = load_data(usage, options.quick)
    for feat in feats:
        label_hat = model.predict(np.array(feat))
        label_hat = max(label_hat)
        labels_hat.append(label_hat)

    #     exit()

    # labels_hat = model.predict(feat)
    labels = th.tensor(labels).to(device)
    labels_hat = th.tensor(labels_hat).to(device)

    labels_hat[labels_hat<0] = 0

    test_r2, test_mape, test_smape, ratio, min_ratio, max_ratio = cal_metrics(labels_hat,labels)


    # l = labels[ratio>5].cpu().numpy().tolist()
    # lh = labels_hat[ratio > 5].cpu().numpy().tolist()
    # lh = [round(n,2) for n in lh]
    # print(list(zip(l,lh)))

    return labels_hat,labels,test_r2,test_mape,test_smape,min_ratio,max_ratio

def test_all(test_data,model,usage='test',flag_group=False):
    if flag_group:
        labels_hat_all, labels_all = None, None
        for i, (name, data) in enumerate(test_data.items()):
            labels_hat,labels, test_r2, test_mape, test_smape,test_min_ratio, test_max_ratio = test(data,model)
            print(
                '\t{} {},\t #ep:{}\tr2={:.3f}\tmape={:.3f}\tsmape={:.3f}\tmin_ratio={:.2f}\tmax_ratio={:.2f}'.format(
                    usage,name, len(labels), test_r2, test_mape, test_smape,test_min_ratio, test_max_ratio))
            labels_hat_all = cat_tensor(labels_hat_all, labels_hat)
            labels_all = cat_tensor(labels_all, labels)
        test_r2, test_mape, test_smape,ratio, min_ratio, max_ratio = cal_metrics(labels_hat_all, labels_all)
        print(
            '\t{} overall\tr2={:.3f}\tmape={:.3f}\tsmape={:.3f}\tmin_ratio={:.2f}\tmax_ratio={:.2f}'.format(
                usage,test_r2, test_mape, test_smape,test_min_ratio, test_max_ratio))
    else:
        _, _, test_r2, test_mape, test_smape,test_min_ratio, test_max_ratio = test(test_data['0'],model)
        print(
            '\t{}: \tr2={:.3f}\tmape={:.3f}\tmape={:.3f}\tmin_ratio={:.2f}\tmax_ratio={:.2f}'.format(usage,test_loss, test_r2,test_mape,test_smape,test_min_ratio,test_max_ratio))

    return test_r2,test_mape

def train():
    print(options)
    th.multiprocessing.set_sharing_strategy('file_system')

    train_label,train_feat = load_data('train',False,options.quick)
    test_data = load_data('test',options.flag_group,options.quick)

    print(train_feat.shape)
    train_feat = pd.DataFrame(train_feat)
    train_label = pd.DataFrame(train_label)
    print('Training ...')
    xgbr = xgb.XGBRegressor(n_estimators=100, max_depth=45, nthread=25)
    #xgbr = xgb.XGBRegressor(n_estimators=45, max_depth=8, nthread=25)
    xgbr.fit(train_feat, train_label)

    save_dir = '../checkpoints/{}'.format(options.checkpoint)
    with open(f"{save_dir}/ep_model.pkl", "wb") as f:
        pickle.dump(xgbr, f)
    print('Finish!')

    print('Testing ...')
    test(test_data,xgbr)

if __name__ == "__main__":
    seed = random.randint(1, 10000)
    init(seed)
    if options.test_iter:

        assert options.checkpoint, 'no checkpoint dir specified'
        model_save_path = '../checkpoints/{}/ep_model.pkl'.format(options.checkpoint)
        assert os.path.exists(model_save_path)

        with open(model_save_path,'rb') as f:
            model = pickle.load(f)

        logs_files = [f for f in os.listdir('../checkpoints/{}'.format(options.checkpoint)) if f.startswith('test') and '_' not in f]
        logs_idx = [int(f[4:].split('.')[0]) for f in logs_files]
        log_idx = 1 if len(logs_idx)==0 else max(logs_idx)+1
        stdout_f = '../checkpoints/{}/test{}.log'.format(options.checkpoint,log_idx)
        with tee.StdoutTee(stdout_f):

            usages = ['train','test','val']
            usages = ['test']
            for usage in usages:
                data = load_data(usage, options.flag_group, options.quick)
                test_all(data,model,usage,options.flag_group)

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
            train()

    else:
        print('No checkpoint is specified. abandoning all model checkpoints and logs')

        train()