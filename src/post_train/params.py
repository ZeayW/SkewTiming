import argparse
def get_args():
    parser = argparse.ArgumentParser(description="Train Confidence Fusion Model")

    # Path Arguments
    # parser.add_argument('--train_data_path', type=str, required=True,
    #                     help='Path to the metadata.pkl file for training')
    #
    # parser.add_argument('--test_data_path', type=str, required=True,
    #                     help='Path to the metadata.pkl file for testing')
    # parser.add_argument('--val_data_path', type=str, required=True,
    #                     help='Path to the metadata.pkl file for validation')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the metadata.pkl files')
    parser.add_argument('--checkpoint', type=str, default='checkpoints_fusion',
                        help='Directory to save the trained model')

    # Training Arguments
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=10000, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument("--gpu",type=int,help='index of gpu. Type: int')
    parser.add_argument("--feature_mask", nargs='+', type=int, help='the feature selection mask. Type: int',default=[1,5,6,7,10])
    parser.add_argument("--use_union_norm", action='store_true', help='decide whether to apply union normalization across train/val/test or not')
    parser.add_argument('--norm_strategy', type=str, default='standard', help='feature normalization strategy')
    parser.add_argument("--flag_perdesignNorm", action='store_true',
                        help='decide whether to apply normalization per-design')
    return parser.parse_args()