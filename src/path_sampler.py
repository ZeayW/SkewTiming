import torch as th
from utils import *
from torch import nn
from options import get_options
import os
import pickle
from dgl import node_subgraph
from random import shuffle
import tee
options = get_options()

data_path = options.data_savepath
if data_path.endswith('/'):
    data_path = data_path[:-1]
data_file = os.path.join(data_path, 'data.pkl')



with open(data_file, 'rb') as f:
    data_all = pickle.load(f)
    #design_names = [d[1]['design_name'].split('_')[-1] for d in data_all]
    design_names = [d[1]['design_name'] for d in data_all]

device = th.device("cuda:0" if th.cuda.is_available() else "cpu")


class PathFinder(nn.Module):
    def __init__(self):
        super(PathFinder,self).__init__()

    def message_func_r(self, edges):
        return {'mdr': edges.src['hdr'] + 1}

    def reduce_func_r(self,nodes):
        return {'hdr':th.max(nodes.mailbox['mdr'],dim=1).values}

    def forward_r(self,graph,topo_r):
        with graph.local_scope():
            for i, nodes in enumerate(topo_r[1:]):
                graph.pull(nodes, self.message_func_r, self.reduce_func_r, etype='edge_r')
            #graph.ndata['hd'] = graph.ndata['hd'] + graph.ndata['delay']
            return graph.ndata['hdr']

model = PathFinder()


def get_nodes_delay(graph,topo_r):

    nodes_list = th.tensor(range(graph.number_of_nodes())).to(device)
    POs = nodes_list[graph.ndata['is_po'] == 1].cpu().numpy().tolist()
    PIs = nodes_list[graph.ndata['is_pi'] == 1].cpu().numpy().tolist()

    graph.ndata['hdr'] = -1000 * th.ones((graph.number_of_nodes(), len(POs)), dtype=th.float).to(device)
    for k, po in enumerate(POs):
        graph.ndata['hdr'][po][k] = 0

    nodes_delay = model.forward_r(graph, topo_r)

    return nodes_delay

def collect_nodes_feat(graph):
    nodes_ntype = graph.ndata['ntype'][:,3:]
    nodes_degree = graph.out_degrees()

    return nodes_ntype,nodes_degree


def sample_randompaths(graph,data):
    max_len = 0
    nodes_type = data['nodes_type']
    nodes_degree = data['nodes_degree']

    nodes_list = th.tensor(range(graph.number_of_nodes())).to(device)
    POs = nodes_list[graph.ndata['is_po'] == 1].cpu().numpy().tolist()
    nodes_delay = data['nodes_delay']


    num_nodes = graph.number_of_nodes()
    num_seq = len(nodes_list[th.logical_and(graph.ndata['is_po'] == 1,graph.ndata['is_pi'] == 1)])
    num_cmb = num_nodes - num_seq

    paths_all = []
    for j in range(len(POs)):
        nodes_delay_j = nodes_delay[:,j]


        drive_PIs_mask = th.logical_and(graph.ndata['is_pi']==1,nodes_delay_j>0)
        drive_PIs = nodes_list[drive_PIs_mask].cpu().numpy().tolist()
        num_reg = min(len(drive_PIs),50)

        # drive_PI_names = [data['nodes_name'][n] for n in drive_PIs]
        # drive_PI_names = [n for n in drive_PI_names if '[' in n]
        # drive_registers = [n.split('[')[0] for n in drive_PI_names]
        # drive_registers = list(set(drive_registers))
        # num_reg = len(drive_registers)
        #print(data['nodes_name'][POs[j]],[data['nodes_name'][n] for n in drive_PIs],drive_PI_names,drive_registers,num_reg )


        # PIs_delay = nodes_delay_j[drive_PIs_mask]
        # PIs_delay = PIs_delay.cpu().numpy().tolist()
        # delay_dict = {drive_PIs[i]:PIs_delay[i] for i in range(len(PIs_delay))}
        # sorted_pis = sorted(delay_dict.items(),key = lambda kv:(kv[1], kv[0]))
        # sorted_pis.reverse()
        # PIs_rank = {sorted_pis[i][0]:i+1 for i in range(len(sorted_pis))}


        nodes_delay_j = nodes_delay_j.cpu().numpy().tolist()
        shuffle(drive_PIs)
        sampled_PIs = drive_PIs[:num_reg]
        paths = []

        #print(data['nodes_name'][POs[j]])
        for pi in sampled_PIs:

            path = []
            cur_nid = pi
            path.append(cur_nid)
            cur_delay = nodes_delay_j[pi]
            while True:
                successors = graph.successors(cur_nid, etype='edge')
                if len(successors) == 0:
                    break
                # print(cur_delay,data['nodes_name'][cur_nid],[ (data['nodes_name'][n],nodes_delay_j[n]) for n in successors])
                successors = [n for n in successors if nodes_delay_j[n] == cur_delay - 1]
                cur_nid = successors[0]
                path.append(cur_nid.item())
                cur_delay = cur_delay - 1
            #print([data['nodes_name'][n] for n in path])

            max_len = max(len(path),max_len)
            paths.append({
                'path_degree': nodes_degree[path].numpy().tolist(),
                'path_ntype': th.sum(nodes_type[path], dim=0).numpy().tolist(),
                'path':path
            })

        paths_all.append({
            'num_nodes':num_nodes,
            'num_seq':num_seq,
            'num_cmb':num_cmb,
            'num_reg':num_reg,
            'paths_rd':paths
        })
        #print(len(paths))
        #exit()

    return paths_all,max_len

def sample_criticalpath(graph,data):
    max_len = 0
    nodes_type = data['nodes_type']
    nodes_degree = data['nodes_degree']

    path_label_pairs = []

    nodes_delay_base = data['nodes_delay']

    nodes_list = th.tensor(range(graph.number_of_nodes())).to(device)
    POs = nodes_list[graph.ndata['is_po'] == 1].cpu().numpy().tolist()
    PIs = nodes_list[graph.ndata['is_pi'] == 1].cpu().numpy().tolist()



    for i, (PIs_delay, POs_label, _, _ ) in enumerate(data['delay-label_pairs']):
        #print(i)
        pi2delay = {PIs[n] : PIs_delay[n] for n in range(len(PIs))}
        POs_rank = {}
        label2po = {}
        for k,po in enumerate(POs):
            label = POs_label[k]
            label2po[label] = label2po.get(label,[])
            label2po[label].append(po)
        label2po_sorted = sorted(label2po.items(),key = lambda kv:(kv[0], kv[1]))
        label2po_sorted.reverse()
        rank = 1
        for _, pos in label2po_sorted:
            for po in pos:
                POs_rank[po] = rank
            rank += len(pos)


        graph.ndata['delay'] = th.zeros((graph.number_of_nodes(), 1), dtype=th.float).to(device)
        graph.ndata['delay'][PIs] = th.tensor(PIs_delay,dtype=th.float).unsqueeze(1).to(device)
        nodes_delay =nodes_delay_base + graph.ndata['delay']
        nodes_delay_base_t = th.transpose(nodes_delay_base,0,1)
        #nodes_delay_PIs = nodes_delay_tmp[graph.ndata['is_pi']==1]
        nodes_delay_t = th.transpose(nodes_delay,0,1)
        maxdelay_PIs = th.argmax(nodes_delay_t,dim=1)

        critical_paths = []

        for j, po in enumerate(POs):
            critical_path = []
            nodes_delay_j = nodes_delay_base_t[j]
            drive_PIs_mask = th.logical_and(graph.ndata['is_pi'] == 1, nodes_delay_j > 0)
            drive_PIs = nodes_list[drive_PIs_mask].cpu().numpy().tolist()
            nodes_delay_j = nodes_delay_j.cpu().numpy().tolist()

            critical_pi = maxdelay_PIs[j].item()
            input_delay = graph.ndata['delay'][critical_pi]
            if pi2delay.get(critical_pi,None) is None:
                print(i,data['design_name'],data['nodes_name'][critical_pi],data['nodes_name'][po])


            cur_nid = critical_pi
            critical_path.append(cur_nid)
            cur_delay = nodes_delay_j[critical_pi]
            idx = 0
            while True:
                idx+=1
                successors = graph.successors(cur_nid,etype='edge')
                if len(successors)==0:
                    break
                #print(cur_delay,data['nodes_name'][cur_nid],[ (data['nodes_name'][n],nodes_delay_j[n]) for n in successors])
                successors = [n for n in successors if nodes_delay_j[n]==cur_delay-1]
                cur_nid = successors[0]
                critical_path.append(cur_nid.item())
                cur_delay = cur_delay-1

                if idx>1000:
                    print(i,j)

            max_len = max(len(critical_path), max_len)

            critical_paths.append({
                    'rank':POs_rank[po],
                    'rank_ratio': POs_rank[po]/len(POs_label),
                    'level':len(critical_path)-1,
                    'path':critical_path,
                    'path_degree': nodes_degree[critical_path].numpy().tolist(),
                    'path_ntype': th.sum(nodes_type[critical_path],dim=0).numpy().tolist(),
                })
        #print({data['nodes_name'][k]:v for (k,v) in pi2delay.items()})

        path_label_pairs.append((critical_paths,pi2delay,POs_label))

        #if i>10:
        #    exit()
    return path_label_pairs,max_len

if __name__ == "__main__":
    new_dataset = []
    max_len_all = 0
    stdout_f = './stdout.log'

    with tee.StdoutTee(stdout_f):

        for i,(graph,graph_info) in enumerate(data_all):
            new_data = {'design_name': graph_info['design_name']}
            topo_r = gen_topo(graph, flag_reverse=True)
            topo_r = [l.to(device) for l in topo_r]

            graph = heter2homo(graph)
            nodes_ntype,nodes_degree = collect_nodes_feat(graph)
            # new_data['nodes_type'] = nodes_ntype.numpy().tolist()
            # new_data['nodes_degree'] = nodes_degree.numpy().tolist()
            # print([(graph_info['nodes_name'][i],nodes_degree[i].item()) for i in range(len(nodes_degree))][:50])
            # exit()
            graph = add_reverse_edges(graph)
            graph = graph.to(device)

            # if 'ss_pcm' not in graph_info['design_name']:
            #     continue
            # if graph_info['design_name'] not in ['y_quantizer']:
            #     continue
            print(i,graph_info['design_name'])
            #if graph_info['design_name'] not in ['i2c']: continue


            graph_info['nodes_delay'] = get_nodes_delay(graph,topo_r)
            graph_info['nodes_type'] = nodes_ntype
            graph_info['nodes_degree'] = nodes_degree
            random_paths,max_len = sample_randompaths(graph,graph_info)
            print('\t extracted random paths')
            max_len_all = max(max_len_all,max_len)
            new_data['random_paths'] = random_paths
            critical_paths,max_len = sample_criticalpath(graph,graph_info)
            print('\t extracted critical paths')
            max_len_all = max(max_len_all, max_len)
            new_data['critical_path'] = critical_paths

            new_dataset.append(new_data)

            # if i>=20:
            #     break

        print(max_len_all)
        with open('path_data_open_aes.pkl','wb') as f:
            pickle.dump((max_len_all,new_dataset),f)
        #

