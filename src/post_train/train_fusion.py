import sys
sys.path.append('../')
import os.path

import torch
import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_utils2 import load_datasets
from params import get_args
from torchmetrics import R2Score
import tee

args = get_args()
device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() and args.gpu !=-1 else "cpu")

R2_score = R2Score().to(device)

def cal_metrics(labels_hat,labels):
    r2 = R2_score(labels_hat, labels).item()
    mape = th.mean(th.abs(labels_hat[labels != 0] - labels[labels != 0]) / labels[labels != 0])
    ratio = labels_hat[labels != 0] / labels[labels != 0]
    min_ratio = th.min(ratio)
    max_ratio = th.max(ratio)

    return r2,mape,ratio,min_ratio,max_ratio

def cat_tensor(t1,t2):
    if t1 is None:
        return t2
    elif t2 is None:
        return t1
    else:
        return th.cat((t1,t2),dim=0)

class ConfidenceFusionNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # 输入: Meta Features (6) + Pred1 (1) + Pred2 (1) = 8
        self.net = nn.Sequential(
            nn.Linear(input_dim+2, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output weight w in [0, 1]
        )

    def forward(self, meta_feats, p1, p2):
        # Concatenate features with raw predictions
        # p1, p2 shape: [Batch, 1]
        inputs = torch.cat([meta_feats, p1, p2], dim=1)
        #inputs = meta_feats
        w = self.net(inputs)

        # Weighted Average
        final_pred = w * p1 + (1 - w) * p2
        return final_pred, w


def train_fusion():
    # === Configuration ===
    print(args)
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr



    # === Data Loading ===
    feature_select_mask = [1, 2, 3, 5, 6, 7, 10, 11]
    feature_select_mask = args.feature_mask
    if isinstance(feature_select_mask, int):
        feature_select_mask = [feature_select_mask]
    input_dim = len(feature_select_mask)

    train_set, val_set, test_set = load_datasets(args.data_path, feature_select_mask,norm_strategy=args.norm_strategy,flag_norm_perdesign=args.flag_perdesignNorm)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,drop_last=True)
    val_loader = DataLoader(val_set, batch_size=len(val_set), shuffle=False)
    test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=False)

    print('#train: {}, #val: {}, #test: {}'.format(len(train_set),len(val_set),len(test_set)))

    # === Model Setup ===
    model = ConfidenceFusionNet(input_dim=input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()  # MAE Loss
    criterion = nn.MSELoss()  # MSE Loss

    print(f"\nStart Training Fusion Model on {device}...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        print('Epoch {}'.format(epoch))

        total_labels_hat, total_labels_hat_original,total_labels = None,None,None
        for feats, p1, p2, label in train_loader:
            feats, p1, p2, label = feats.to(device), p1.to(device), p2.to(device), label.to(device)

            optimizer.zero_grad()
            pred, w = model(feats, p1, p2)
            total_labels = cat_tensor(total_labels,label)
            total_labels_hat = cat_tensor(total_labels_hat,pred)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()


        train_r2, train_mape, ratio, min_ratio, max_ratio = cal_metrics(total_labels_hat, total_labels)
        print('\t train: \tr2={:.3f}\tmape={:.3f}'.format(train_r2, train_mape))

        # === Validation ===
        model.eval()
        total_labels_hat, total_labels_hat_original,total_labels = None,None,None
        with torch.no_grad():
            for feats, p1, p2, label in val_loader:
                feats, p1, p2, label = feats.to(device), p1.to(device), p2.to(device), label.to(device)
                pred, w = model(feats, p1, p2)
                total_labels = cat_tensor(total_labels, label)
                total_labels_hat = cat_tensor(total_labels_hat, pred)

            val_r2, val_mape, ratio, min_ratio, max_ratio = cal_metrics(total_labels_hat, total_labels)
            print('\t val:   \tr2={:.3f}\tmape={:.3f}'.format(val_r2, val_mape))

        total_labels_hat, total_labels_hat_original,total_labels = None,None,None

        with torch.no_grad():
            for feats, p1, p2, label in test_loader:
                feats, p1, p2, label = feats.to(device), p1.to(device), p2.to(device), label.to(device)

                pred, w = model(feats, p1, p2)

                total_labels = cat_tensor(total_labels, label)
                total_labels_hat = cat_tensor(total_labels_hat, pred)

            test_r2, test_mape, ratio, min_ratio, max_ratio = cal_metrics(total_labels_hat, total_labels)
            print('\t test:  \tr2={:.3f}\tmape={:.3f}'.format(test_r2, test_mape))


        # === Save Model ===
        torch.save(model.state_dict(), os.path.join(args.checkpoint,'{}.pth'.format(epoch)))


if __name__ == "__main__":
    os.makedirs(args.checkpoint)
    stdout_f = '{}/stdout.log'.format(args.checkpoint)
    with tee.StdoutTee(stdout_f):
        train_fusion()
