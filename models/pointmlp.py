import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
class PointMLP(nn.Module):
    def __init__(self, point_hidden=64, global_hidden=256, out_dim=1024, num_classes=40):
        super().__init__()
        # 对每个点做 MLP: 3 → 64 → 64
        self.point_mlp = nn.Sequential(
            nn.Linear(3, point_hidden),
            nn.ReLU(),
            nn.Linear(point_hidden, point_hidden),
            nn.ReLU(),
        )
        # 对全局池化后向量做 MLP: 64 → 256 → 1024
        self.global_mlp = nn.Sequential(
            nn.Linear(point_hidden, global_hidden),
            nn.ReLU(),
            nn.Linear(global_hidden, out_dim),
        )
        self.classifier = nn.Linear(out_dim, num_classes)

    def forward(self, x):
        B, N, C = x.shape  # C=3
        # 1. 先对每个点做MLP
        x = self.point_mlp(x.view(B*N, 3)).view(B, N, -1)
        # 2. 最大池化
        x = torch.max(x, dim=1).values   # (B, 64)
        # 3. 全局 MLP
        x = self.global_mlp(x)           # (B, 1024)
        # 4. 分类
        x = self.classifier(x)           # (B, 40)
        return x
