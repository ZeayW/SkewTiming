import torch
from torch.utils.data import Dataset
import pickle
import numpy as np
import os

def get_norm_config(strategy):
    """根据策略返回特征配置"""
    feature_names = [
        'corr_min', 'corr_max', 'corr_std',
        'corr_entropy_input', 'corr_entropy_all', 'corr_entropy_all_norm',
        'num_corr_inputs', 'fanin_size',
        'arrival_time_std', 'arrival_time_mean', 'critical_input_arrival',
        'critical_path_length'
    ]

    if strategy == 'standard':
        return {k: 'standard' for k in feature_names}

    elif strategy == 'robust':
        return {k: 'robust' for k in feature_names}
    elif strategy == 'minmax':
        return {k: 'minmax' for k in feature_names}
    elif strategy == 'log':
        return {k: 'log' for k in feature_names}
    elif strategy == 'none':
        return {k: 'none' for k in feature_names}
    elif strategy == 'hybrid':
        # === 精心设计的混合策略 ===
        return {
            # Correlation Stats: [0, 1]区间，可能有偏
            'corr_min': 'standard',
            'corr_max': 'standard',
            'corr_std': 'standard',

            # Entropy: 分布特征，Robust 更好
            'corr_entropy_input': 'standard',
            'corr_entropy_all': 'standard',
            'corr_entropy_all_norm': 'standard',

            # Count/Size: 典型的长尾分布 -> Log 变换
            'num_corr_inputs': 'log',
            'fanin_size': 'log',

            # Timing: 物理量，可能有异常大值 -> Robust
            'arrival_time_std': 'log',
            'arrival_time_mean': 'log',
            'critical_input_arrival': 'log',

            # Depth/Length: 整数计数，长尾 -> Log 或 Standard
            'critical_path_length': 'log'
        }
    elif strategy == 'hybrid2':
        # === 精心设计的混合策略 ===
        return {
            # Correlation Stats: [0, 1]区间，可能有偏
            'corr_min': 'none',
            'corr_max': 'none',
            'corr_std': 'none',

            # Entropy: 分布特征，Robust 更好
            'corr_entropy_input': 'none',
            'corr_entropy_all': 'none',
            'corr_entropy_all_norm': 'none',

            # Count/Size: 典型的长尾分布 -> Log 变换
            'num_corr_inputs': 'log',
            'fanin_size': 'log',

            # Timing: 物理量，可能有异常大值 -> Robust
            'arrival_time_std': 'log',
            'arrival_time_mean': 'log',
            'critical_input_arrival': 'log',

            # Depth/Length: 整数计数，长尾 -> Log 或 Standard
            'critical_path_length': 'log'
        }
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def masked_list(list,mask):
    res = []
    for i,v in enumerate(list):
        if i in mask:
            res.append(v)
    return res

class MetadataDataset(Dataset):

    def __init__(self, pkl_path, feature_select_mask):
        """
        初始化：仅加载数据并提取原始特征，不进行标准化。
        """
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Metadata file not found at: {pkl_path}")

        # print(f"Loading raw metadata from {pkl_path}...")
        with open(pkl_path, 'rb') as f:
            self.raw_data = pickle.load(f)

        self.design_names = list(self.raw_data.keys())
        self.flat_index_map = []

        self.FEATURE_NAMES = [
            'corr_min',  # 0
            'corr_max',  # 1
            'corr_std',  # 2
            'corr_entropy_input',  # 3
            'corr_entropy_all',  # 4
            'corr_entropy_all_norm',  # 5
            'num_corr_inputs',  # 6
            'fanin_size',  # 7
            'arrival_time_std',  # 8
            'arrival_time_mean',  # 9
            'critical_input_arrival',  # 10
            'critical_path_length',  # 11
        ]
        self.FEATURE_NAMES = masked_list(self.FEATURE_NAMES,feature_select_mask)
        #print(self.FEATURE_NAMES)

        # 统计样本数
        total_samples = 0

        for design in self.design_names:
            d_data = self.raw_data[design]
            num_samples = d_data['labels_gt'].shape[0]
            total_samples += num_samples

            # 1. 提取原始特征 (Feature Stack Order MUST be consistent)

            # 构建特征矩阵 (Feature Matrix)
            feats = []
            for key in self.FEATURE_NAMES:
                feats.append(d_data[key].float())
            feats = torch.stack(feats, dim=1)

            # 存储原始特征
            d_data['features_raw'] = feats
            # 初始化 normalized 特征为 raw (直到调用 normalize)
            d_data['features_norm'] = feats.clone()

            # 建立索引
            for i in range(num_samples):
                self.flat_index_map.append((design, i))

        #print(f"Loaded {len(self.design_names)} designs, {total_samples} samples.")




    def get_all_features(self):
        """获取所有样本的原始特征矩阵 (用于计算统计量)"""
        all_feats = []
        for design in self.design_names:
            all_feats.append(self.raw_data[design]['features_raw'])
        return torch.cat(all_feats, dim=0)


    def apply_normalization(self, norm_config, flag_perdesign):
        """
        应用混合 Normalization 策略

        Args:
            norm_config: Dict[feature_name, method]
                         例: {'corr_entropy': 'robust', 'fanin_size': 'log'}
            precomputed_stats: 外部传入的统计量 (从训练集计算)，
                              格式: {'feature_name': {'mean': ..., 'std': ...}}
        """

        all_feats = self.get_all_features()  # [N, num_features]


        # 应用变换到每个 design
        for design in self.design_names:
            stats = {}
            raw = self.raw_data[design]['features_raw']
            norm = torch.zeros_like(raw)

            for i, feat_name in enumerate(self.FEATURE_NAMES):

                method = norm_config.get(feat_name, 'none')
                if flag_perdesign:
                    all_feats = self.raw_data[design]['features_raw']
                stats[feat_name] = self._compute_stats(all_feats[:, i], method)

                col_raw = raw[:, i]
                col_norm = self._transform_feature(col_raw, method, stats[feat_name])
                norm[:, i] = col_norm

            self.raw_data[design]['features_norm'] = norm

    def _compute_stats(self, col_data, method):
        """计算单个特征列的统计量"""
        epsilon = 1e-6

        if method == 'standard':
            return {
                'mean': col_data.mean().item(),
                'std': (col_data.std().item() + epsilon)
            }
        elif method == 'robust':
            q25 = torch.quantile(col_data, 0.25).item()
            q50 = torch.quantile(col_data, 0.50).item()
            q75 = torch.quantile(col_data, 0.75).item()
            iqr = max(q75 - q25, epsilon)
            return {'median': q50, 'iqr': iqr}

        elif method == 'minmax':
            return {
                'min': col_data.min().item(),
                'max': col_data.max().item()
            }
        elif method == 'log':
            # Log 变换不需要统计量
            return {}
        elif method == 'none':
            return {}
        else:
            raise ValueError(f"Unknown method: {method}")

    def _transform_feature(self, col_raw, method, stats):
        """对单个特征列应用变换"""
        epsilon = 1e-6

        if method == 'standard':
            return (col_raw - stats['mean']) / stats['std']

        elif method == 'robust':
            return (col_raw - stats['median']) / stats['iqr']

        elif method == 'minmax':
            rng = max(stats['max'] - stats['min'], epsilon)
            return (col_raw - stats['min']) / rng

        elif method == 'log':
            # log1p: log(1 + x), 适合非负特征
            return torch.log1p(torch.relu(col_raw))

        elif method == 'none':
            return col_raw
        else:
            raise ValueError(f"Unknown method: {method}")

    def __len__(self):
        return len(self.flat_index_map)

    def __getitem__(self, idx):
        design, local_idx = self.flat_index_map[idx]
        data = self.raw_data[design]

        return (
            data['features_norm'][local_idx],  # Normalized Features
            data['labels_hat1'][local_idx].float().unsqueeze(0),  # Pred 1
            data['labels_hat2'][local_idx].float().unsqueeze(0),  # Pred 2
            data['labels_gt'][local_idx].float().unsqueeze(0)  # Label
        )

def load_datasets(data_path, feat_select_mask,norm_strategy='standard',flag_norm_perdesign=False):
    res = []
    norm_config = get_norm_config(norm_strategy)
    for usage in ['train','val','test']:
        # 1. 实例化数据集 (此时包含原始数据)
        #print("=== Initializing Datasets for {}===".format(usage))
        ds = MetadataDataset(os.path.join(data_path, usage,'metadata.pkl'),feat_select_mask)

        # 3. 应用统一的标准化
        #print("=== Applying Normalization for {}===".format(usage))
        ds.apply_normalization(norm_config,flag_norm_perdesign)
        res.append(ds)

    return res
