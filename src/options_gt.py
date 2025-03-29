import argparse


def get_options(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--flag_group',action='store_true')
    parser.add_argument('--feat_choice',type=int,default=0)
    parser.add_argument('--pe_size',type=int,default=128)
    parser.add_argument('--quick',action='store_true')
    parser.add_argument('--pretrain_dir', type=str)
    parser.add_argument('--rawdata_path', type=str, help='the directory that contains the raw dataset. Type: str')
    parser.add_argument("--checkpoint",type=str,help= "checkpoint to save the results and logs")
    parser.add_argument("--test_iter", type=str, default=None,help="iter to test the model")
    parser.add_argument("--learning_rate", type=float, help = 'the learning rate for training. Type: float.',default=1e-3)
    parser.add_argument("--batch_size", type=int, help = 'the number of samples in each training batch. Type: int',default=64)
    parser.add_argument("--num_epoch", type=int, help='Type: int; number of epoches that the training procedure runs. Type: int',default=2000)
    parser.add_argument("--hidden_dim", type=int, help='the dimension of the intermediate GNN layers. Type: int',default=256)
    parser.add_argument("--weight_decay", type=float, help='weight decay. Type: float',default=0)
    parser.add_argument("--gpu",type=int,help='index of gpu. Type: int')
    parser.add_argument('--data_savepath',type=str,help='the directory that contains the dataset. Type: str',default='../data/arith_blocks')
    parser.add_argument('--predict_path',type=str,help='the directory used to save the prediction result. Type: str',default='../prediction/example')
    options = parser.parse_args(args)


    return options
