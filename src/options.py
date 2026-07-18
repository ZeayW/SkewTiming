import argparse

def get_options(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=10.0)
    parser.add_argument('--beta', type=float, default=10.0)
    parser.add_argument('--base_pe', type=int, default=1)
    parser.add_argument('--log_level', type=int, default=0,
                        help='0: no runtime/statistics logging; 1: enable runtime breakdown and extra test statistics')
    parser.add_argument('--cuda_blocking', action='store_true',
                        help='debug only: set CUDA_LAUNCH_BLOCKING=1 before CUDA initialization')
    parser.add_argument('--cpe_impl', type=str, default='frontier', choices=['dense', 'sparse', 'frontier', 'compare'],
                        help='CPE path-search implementation: dense keeps the original behavior; sparse uses sparse aggregation; frontier uses sparse endpoint-node frontiers; compare checks all implementations on the same batch')
    parser.add_argument('--mtde_backward_impl', type=str, default='custom',
                        choices=['dgl', 'scatter', 'custom', 'compare'],
                        help='MTDE backward correlation propagation: dgl keeps the original DGL pull path; scatter uses cached reverse edges; custom adds a compact autograd path; compare checks all implementations')
    parser.add_argument('--mtde_forward_cache', type=str, default='cache', choices=['off', 'cache', 'compare'],
                        help='MTDE forward topology-edge lookup cache: off keeps original graph queries; cache reuses cached eids; compare checks cached and original outputs')
    parser.add_argument('--mtde_forward_impl', type=str, default='scatter', choices=['dgl', 'scatter', 'compare'],
                        help='MTDE forward propagation: dgl uses graph message passing; scatter uses cached tensor segment operations; compare checks both')
    parser.add_argument('--fse_gnn_impl', type=str, default='builtin', choices=['udf', 'builtin', 'compare'],
                        help='FSE structure GNN aggregation: udf keeps Python message/reduce functions; builtin uses DGL built-ins; compare checks outputs and parameter gradients')
    parser.add_argument('--fse_eval_cache', type=str, default='cache', choices=['off', 'cache', 'compare'],
                        help='cache case-invariant FSE node embeddings only during eval/no_grad; compare recomputes and checks the cached value')
    parser.add_argument('--fse_aggregation', choices=('raw_sum', 'endpoint_mean'), default='raw_sum',
                        help='aggregate FSE node embeddings by raw correlation sum or endpoint-wise normalized mean')
    parser.add_argument('--cpe_depth_encoding', choices=('absolute', 'correlation_only'), default='absolute',
                        help='use absolute path distance/position in CPE or retain only correlation-aware path context')
    parser.add_argument('--flag_noTPE', action='store_true')
    parser.add_argument('--flag_noFSE', action='store_true')
    parser.add_argument('--use_corr_pe',action='store_true')
    parser.add_argument('--use_attn_bias', action='store_true')
    parser.add_argument('--ntype_file',type=str,default='./ntype2id.pkl')
    parser.add_argument('--flag_split', action='store_true')
    parser.add_argument('--flag_group',action='store_true')
    parser.add_argument('--flag_baseline',type=int,default=-1,help='choose the model, -1: ours; 0:ACCNN; 1:Graph Transformer; 2: Path-based')
    parser.add_argument('--flag_alternate',action='store_true')
    parser.add_argument('--smooth_ccal', action='store_true',
                        help='apply 1/3 CCAL and 1/6 residual supervision every epoch instead of hard three-epoch alternation')
    parser.add_argument('--lr_scheduler', action='store_true',
                        help='reduce learning rate when validation MAPE plateaus')
    parser.add_argument('--lr_scheduler_factor', type=float, default=0.25)
    parser.add_argument('--lr_scheduler_patience', type=int, default=3)
    parser.add_argument('--min_learning_rate', type=float, default=1e-5)
    parser.add_argument('--ema_decay', type=float, default=0.0,
                        help='parameter EMA decay; 0 disables EMA evaluation/checkpoints')
    parser.add_argument('--ema_start_epoch', type=int, default=5,
                        help='number of full warmup epochs before EMA updates begin')
    parser.add_argument('--ema_scheduler_source', type=str, default='raw', choices=['raw', 'ema'],
                        help='validation parameters used to drive the LR scheduler when EMA is enabled')
    parser.add_argument('--global_out_choice', type=int, default=0,
                        help='choose the way to implement the global attention')
    parser.add_argument('--global_cat_choice', type=int, default=1, help='choose the way to implement the global attention')
    parser.add_argument('--global_info_choice',  type=int,default=3,help='choose the way to implement the global attention')
    parser.add_argument('--remove01',action='store_true')
    parser.add_argument('--inv_choice',type=int,default=-1)
    parser.add_argument('--quick',action='store_true')
    parser.add_argument('--flag_residual', action='store_true')
    parser.add_argument('--flag_continue_trainpath', action='store_true')
    parser.add_argument('--flag_gt', action='store_true')
    parser.add_argument('--flag_meta', action='store_true')
    parser.add_argument('--flag_transformer', type=int,default=1,help='valid value in [1,2,3]')
    parser.add_argument('--flag_rawpath', action='store_true')
    parser.add_argument('--use_pathgnn', action='store_true')
    parser.add_argument('--path_feat_choice', type=int,default=0,help='')
    parser.add_argument('--path_corr_choice', type=int, default=0)
    parser.add_argument('--path_delay_choice', type=int, default=-1)
    parser.add_argument('--flag_singlepath', action='store_true')
    parser.add_argument('--flag_degree', action='store_true')
    parser.add_argument('--flag_width', action='store_true')
    parser.add_argument('--flag_delay', action='store_true')
    parser.add_argument('--flag_delay_g', action='store_true')
    parser.add_argument('--flag_delay_pi', action='store_true')
    parser.add_argument('--flag_delay_pd', action='store_true')
    parser.add_argument('--flag_ntype_g', action='store_true')
    parser.add_argument('--flag_path_supervise', action='store_true')
    parser.add_argument('--pretrain_dir', type=str)
    parser.add_argument('--pi_choice',type=int,default=0)
    parser.add_argument('--agg_choice', type=int,default=0)
    parser.add_argument('--split_feat',action='store_true')
    parser.add_argument('--attn_choice', type=int,default=0,help='choose the way to implement the attention')
    parser.add_argument('--flag_homo', action='store_true')
    parser.add_argument('--flag_reverse', action='store_true')
    parser.add_argument('--flag_filter', action='store_true')
    parser.add_argument('--flag_global', action='store_true')
    parser.add_argument('--flag_attn',action='store_true')
    parser.add_argument('--target_residual', action='store_true', help=('set the prediction target as the redisual delay'))
    parser.add_argument('--num_fold',type=int,default=5,help=('number of folds (only vaild for train_kfold)'))
    parser.add_argument('--rawdata_path', type=str, help='the directory that contains the raw dataset. Type: str')
    parser.add_argument('--min_cases_per_design', type=int, default=100,
                        help='minimum valid timing cases required by the raw-data parser')
    parser.add_argument('--parser_seed', type=int, default=0,
                        help='deterministic seed for parser dataset ordering and splits')
    parser.add_argument('--parser_workers', type=int, default=1,
                        help='number of processes used to parse cases within each design')
    parser.add_argument('--parser_constant_impl', choices=('scan', 'worklist', 'compare'),
                        default='worklist', help='constant propagation implementation')
    parser.add_argument("--checkpoint",type=str,help= "checkpoint to save the results and logs")
    parser.add_argument("--test_iter", type=str, default=None,help="iter to test the model")
    parser.add_argument("--learning_rate", type=float, help = 'the learning rate for training. Type: float.',default=1e-3)
    parser.add_argument("--batch_size", type=int, help = 'the number of graphs in each training batch. Type: int',default=64)
    # parser.add_argument("--batch_size_po", type=int, help='the maximum number of PO in each batch. Type: int',
    #                     default=64)
    parser.add_argument("--num_epoch", type=int, help='Type: int; number of epoches that the training procedure runs. Type: int',default=2000)
    parser.add_argument("--eval_every", type=int, default=1,
                        help='run validation/test every N epochs; <=0 disables periodic evaluation')
    parser.add_argument("--test_every", type=int, default=0,
                        help='run test every N epochs; <=0 follows eval_every while validation remains unchanged')
    parser.add_argument('--extra_test_data_path', action='append', default=[],
                        help='additional serialized dataset directory to append to the grouped test set')
    parser.add_argument("--debug_case_limit", type=int, default=0,
                        help='debug only: cap timing cases per design when >0')
    parser.add_argument("--max_train_batches", type=int, default=0,
                        help='debug only: stop each epoch after this many train batches when >0')
    parser.add_argument("--po_batch_size", type=int, default=0,
                        help='training PO batch size override; <=0 keeps the original dynamic TCAD6 setting')
    parser.add_argument("--test_po_batch_size", type=int, default=0,
                        help='evaluation PO batch size override; <=0 keeps the design-specific default')
    parser.add_argument('--eval_impl', choices=('auto', 'legacy', 'case_first'), default='auto',
                        help='evaluation loop: auto selects case_first for sparse-label or large PO-batched designs')
    parser.add_argument('--eval_mtde_cache', choices=('off', 'cache'), default='cache',
                        help='reuse case-specific MTDE forward state across PO batches during case-first evaluation')
    parser.add_argument('--eval_case_limit', type=int, default=0,
                        help='debug only: cap cases per design during evaluation when >0')
    parser.add_argument("--test_prediction_file", type=str, default=None,
                        help='optional .pt file for labeled test predictions and endpoint/case indices')
    parser.add_argument("--po_batch_node_budget", type=int, default=200000000,
                        help='training memory guard: cap num_nodes * po_batch_size when >0; set 0 to disable')
    parser.add_argument("--in_dim", type=int, help='the dimension of the input feature. Type: int',default=9)
    parser.add_argument("--out_dim", type=int, help='the dimension of the output embedding. Type: int', default=128)
    parser.add_argument("--hidden_dim", type=int, help='the dimension of the intermediate GNN layers. Type: int',default=128)
    parser.add_argument("--weight_decay", type=float, help='weight decay. Type: float',default=0)
    parser.add_argument("--gpu",type=int,help='index of gpu. Type: int')
    parser.add_argument('--data_savepath',type=str,help='the directory that contains the dataset. Type: str',default='../data/arith_blocks')
    parser.add_argument('--testdata_path', type=str, help='the directory that contains the testing dataset. Type: str',
                        default='../data/arith_blocks')
    parser.add_argument('--predict_path',type=str,help='the directory used to save the prediction result. Type: str',default='../prediction/example')
    parser.add_argument('--flag_split01',action='store_true',help="control whether use seperate node for each constant (1'b0,1'b1)")
    options = parser.parse_args(args)

    return options


def merge_with_loaded(default_ns: argparse.Namespace, loaded: argparse.Namespace) -> argparse.Namespace:
    d = vars(default_ns)
    d.update(vars(loaded) or {})
    return argparse.Namespace(**d)
