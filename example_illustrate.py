import torch as th
from dgl import function as fn

    def reduce_func_maxcor(nodes):

        max_cor = th.max(nodes.mailbox['p'], dim=1).values
        return {'max_cor': max_cor}

    def message_func_maxcor_r(edges):
        is_critical = th.logical_and(edges.src['is_critical'], edges.src['max_cor'] == edges.dst['hp'])
        is_critical = th.logical_and(is_critical, edges.dst['hp'] != 0)
        w = is_critical * edges.data['weight']
        return {'is_critical': is_critical, 'w': w}


    def reduce_func_maxcor_r( nodes):

        is_critical = th.any(nodes.mailbox['is_critical'], dim=1)
        corr_local = th.sum(nodes.mailbox['w'], dim=1)
        return {'is_critical': is_critical, 'cl': corr_local}


    def path_embedding( graph, graph_info):

        graph_info['timing_correlation_matrix'] = graph.ndata['hp'] # shape: N \times N_e, where N is total number of nodes, and N_e is number of endpoints in this batch
        #initial the input feature for the nodes
        feat_p = gnn_vanilla(graph,graph.ndata['feat'])

        path_feat = feat_p[graph_info['POs']]
        path_feat = path_feat.reshape(path_feat.shape[0], 1, path_feat.shape[1])
        path_lengths = th.zeros(len(graph_info['POs']), dtype=th.float, device=graph.device)
        path_inputdelay = th.zeros((len(graph_info['POs']),1), dtype=th.float, device=graph.device)
        path_numPI = th.zeros((len(graph_info['POs']),1), dtype=th.float, device=graph.device)
        is_ended = th.zeros(len(graph_info['POs']), dtype=th.bool, device=graph.device)

        # record the two types of correlation that will be feed into the transformer
        c_local = th.zeros(len(graph_info['POs']), len(graph_info['topo_r']),device=graph.device)  
        c_sink = th.zeros(len(graph_info['POs']), len(graph_info['topo_r']), device=graph.device)
        c_sink[:, 0] = 1

        # Parallel Critical Path Extraction
        with th.no_grad():
            k = 0
            cur_nodes = graph_info['POs'] # nodes in the current level L_k,
            _, pre_nodes = graph.out_edges(cur_nodes, etype='reverse') # nodes in the next level L_{k+1},
            pre_nodes = th.unique(pre_nodes)

            while True:
                # for nodes in the current level L_k, collect information (maximum correlation score) from their immediate predecessors in level L_{k+1},
                if len(cur_nodes) !=0:
                    graph.pull(cur_nodes, fn.copy_u('hp', 'p'), self.reduce_func_maxcor, etype='forward')
                
                # identify the critical nodes in the next level L_{k+1}
                graph.pull(pre_nodes, self.message_func_maxcor_r, self.reduce_func_maxcor_r, etype='reverse')
                critical_mask = th.transpose(graph.ndata['is_critical'][pre_nodes], 0, 1) # shape: N_e \times N, for each endpoint, record the critical nodes in level L{k+1}

                # terminate if no nodes in next level is marked as critical to any endpoint
                if th.sum(critical_mask) == 0:
                    break

                # is_ended_mask records the endpoints whose paths terminate at current level
                is_ended_mask = th.logical_and(th.sum(critical_mask, dim=1) == 0, ~is_ended)
                is_ended[is_ended_mask] = True
                path_lengths[is_ended_mask] = k + 1

                # obtain the critical-path node features at next level L_{k+1}, then stack to form the input to the transformer model
                # for an endpoint, if multiple critical nodes exist in level L_{k+1}, then we use the mean feature
                nodes_feat_l = feat_p[pre_nodes]
                path_feat_l = th.matmul(critical_mask.float(), nodes_feat_l)
                num_critical = th.sum(critical_mask, dim=1, keepdim=True)
                path_feat_l = path_feat_l / num_critical.clamp(min=1)
                #stack
                path_feat_l = path_feat_l.reshape(path_feat_l.shape[0], 1, path_feat_l.shape[1])
                path_feat = th.cat((path_feat, path_feat_l), dim=1)

                # compute the two types of correlation that will be feed into the transformer
                
                nodes_cor_l = graph.ndata['hp'][pre_nodes]
                nodes_cor_l_tr = th.transpose(nodes_cor_l, 0, 1)  # N_ep * N_l
                row_max = nodes_cor_l_tr.max(dim=1, keepdim=True).values
                row_max_safe = row_max.clamp(min=1e-8)
                nodes_cor_l_tr = nodes_cor_l_tr / row_max_safe
                cs = th.sum(nodes_cor_l_tr * critical_mask, dim=1) / th.sum(critical_mask, dim=1).clamp(min=1)
                c_sink[:, k + 1] = cs
                cl = th.matmul(critical_mask.float(), graph.ndata['cl'][nodes]) / th.sum(critical_mask,
                                                                                      dim=1).clamp(min=1)
                cl = cl.diagonal()
                c_local[:, k + 1] = cl

                # filter out the inactive nodes
                filtered_mask = th.sum(graph.ndata['is_critical'][pre_nodes], dim=1) >= 1
                cur_nodes = pre_nodes[filtered_mask]
                _, pre_nodes = graph.out_edges(cur_nodes, etype='reverse')
                pre_nodes = th.unique(pre_nodes)
                k += 1

        path_inputdelay = path_inputdelay / path_numPI.clamp(min=1)

        if self.use_corr_pe or self.use_corr_bias:
            c_sink = c_sink[:, :k + 1]
            c_local = c_local[:, :k + 1]
            input_delay =  path_inputdelay if self.path_delay_choice == 0 else None
            path_emb = self.pathformer(path_feat, path_lengths, input_delay = input_delay,c_sink=c_sink, c_local=c_local)
        else:
            path_emb = self.pathformer(path_feat, path_lengths)

        return path_emb

