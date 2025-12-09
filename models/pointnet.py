import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
class TNet(nn.Module):
    """
    学一个 k×k 的仿射矩阵，用来对齐输入（k=3）或中间特征（k=64）。
    输入:  x 形状 (B, k, N)
    输出:  transform 形状 (B, k, k)
    """
    def __init__(self, k=3):
        super().__init__()
        self.k = k

        # 1x1 conv 相当于对每个点做 MLP
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.relu = nn.ReLU()

        # 把最后一层初始化成“接近恒等变换”
        nn.init.constant_(self.fc3.weight, 0)
        nn.init.constant_(self.fc3.bias, 0)

    def forward(self, x):
        # x: (B, k, N)
        batch_size = x.size(0)

        x = self.relu(self.bn1(self.conv1(x)))   # (B, 64, N)
        x = self.relu(self.bn2(self.conv2(x)))   # (B,128, N)
        x = self.relu(self.bn3(self.conv3(x)))   # (B,1024,N)

        # max pooling over N
        x = torch.max(x, 2)[0]                  # (B,1024)

        x = self.relu(self.bn4(self.fc1(x)))    # (B,512)
        x = self.relu(self.bn5(self.fc2(x)))    # (B,256)
        x = self.fc3(x)                         # (B,k*k)

        # 加上单位阵偏置，让初始值接近 I
        iden = torch.eye(self.k, device=x.device).view(1, self.k * self.k)
        x = x + iden
        x = x.view(batch_size, self.k, self.k)  # (B,k,k)
        return x


class PointNetEncoder(nn.Module):
    """
    PointNet backbone，用于提取全局特征。
    输入:  x 形状 (B, 3, N)
    输出:  global_feat 形状 (B,1024)
           trans:      输入 T-Net 的 3x3 矩阵 (B,3,3)
           trans_feat: 特征 T-Net 的 64x64 矩阵 (B,64,64) 或 None
    """
    def __init__(self, global_feat=True, feature_transform=False):
        super().__init__()
        self.global_feat = global_feat
        self.feature_transform = feature_transform

        self.input_transform = TNet(k=3)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)

        if self.feature_transform:
            self.feature_transform_net = TNet(k=64)

        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (B,3,N)
        batch_size = x.size(0)
        n_pts = x.size(2)

        # Input T-Net
        trans = self.input_transform(x)          # (B,3,3)
        x_t = x.transpose(2, 1)                  # (B,N,3)
        x_t = torch.bmm(x_t, trans)              # (B,N,3)
        x = x_t.transpose(2, 1)                  # (B,3,N)

        # 第一段共享 MLP
        x = self.relu(self.bn1(self.conv1(x)))   # (B,64,N)

        # Feature T-Net
        if self.feature_transform:
            trans_feat = self.feature_transform_net(x)  # (B,64,64)
            x_t = x.transpose(2, 1)                     # (B,N,64)
            x_t = torch.bmm(x_t, trans_feat)            # (B,N,64)
            x = x_t.transpose(2, 1)                     # (B,64,N)
        else:
            trans_feat = None

        # 第二段共享 MLP
        x = self.relu(self.bn2(self.conv2(x)))   # (B,128,N)
        x = self.bn3(self.conv3(x))              # (B,1024,N)

        # 全局 max pooling
        x = torch.max(x, 2)[0]                   # (B,1024)

        if self.global_feat:
            return x, trans, trans_feat
        else:
            # 如果以后做分割，可以返回 per-point 特征 + 全局特征
            raise NotImplementedError("这里先只实现 global feature 用于分类")


class PointNetCls(nn.Module):
    """
    PointNet 分类网络，对应论文的 classification network。
    输入:  x 形状 (B,3,N)
    输出:  logits 形状 (B,k)
    """
    def __init__(self, k=40, feature_transform=False):
        super().__init__()
        self.feature_transform = feature_transform

        self.feat = PointNetEncoder(global_feat=True,
                                    feature_transform=feature_transform)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (B,3,N)
        x, trans, trans_feat = self.feat(x)      # x: (B,1024)
        x = self.relu(self.bn1(self.fc1(x)))     # (B,512)
        x = self.relu(self.bn2(self.fc2(x)))     # (B,256)
        x = self.dropout(x)
        x = self.fc3(x)                          # (B,k)
        return x, trans, trans_feat


def feature_transform_regularizer(trans):
    """
    论文里 feature T-Net 的正则项：
        L_reg = || I - A A^T ||_F^2
    trans: (B,k,k)
    """
    if trans is None:
        return 0.0
    k = trans.size(1)
    batch_size = trans.size(0)
    I = torch.eye(k, device=trans.device).unsqueeze(0).expand(batch_size, -1, -1)
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss


