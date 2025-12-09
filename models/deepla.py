import torch
import torch.nn as nn
import torch.nn.functional as F
from knn_cuda import KNN


# =========================================
# 1. DropPath（Stochastic Depth）
# =========================================
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.dim() - 1)
        random_tensor = keep + torch.rand(shape, device=x.device)
        random_tensor.floor_()
        return x / keep * random_tensor


# =========================================
# 2. 正式的 batched KNN（你的 CUDA 版）
# =========================================
class BatchedKNN(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.knn = KNN(k=k, transpose_mode=True)

    def forward(self, xyz):
        """
        xyz: [B, N, 3]
        return: idx [B, N, k]
        """
        dist, idx = self.knn(xyz, xyz)
        return idx.long()


# =========================================
# 3. index_points（支持 batch）
# =========================================
def index_points(points, idx):
    """
    points: [B, N, C]
    idx:    [B, N, K]
    return: [B, N, K, C]
    """
    B, N, C = points.shape
    _, _, K = idx.shape

    idx_expanded = idx.unsqueeze(-1).expand(-1, -1, -1, C)
    points_expanded = points.unsqueeze(1).expand(-1, N, -1, -1)

    return torch.gather(points_expanded, 2, idx_expanded)


# =========================================
# 4. VFR（EdgeConv）模块
# =========================================
class VFR(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, x, knn_idx):
        """
        x: [B, N, C]
        knn_idx: [B, N, K]
        """
        x = self.linear(x)
        B, N, C = x.shape

        x_neighbors = index_points(x, knn_idx)
        x_center = x.unsqueeze(2)
        edge_feat = x_neighbors - x_center    # [B,N,K,C]

        x_agg = edge_feat.max(dim=2)[0]       # [B,N,C]
        return self.bn(x_agg.reshape(B * N, C)).reshape(B, N, C)


# =========================================
# 5. FFN（通道方向 MLP）
# =========================================
class FFN(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()
        hid = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hid),
            nn.ReLU(inplace=True),
            nn.Linear(hid, dim),
            nn.BatchNorm1d(dim),
        )

    def forward(self, x):
        B, N, C = x.shape
        return self.net(x.reshape(B*N, C)).reshape(B, N, C)


# =========================================
# 6. ResLFE Block（DeepLA 核心）
# =========================================
class ResLFEBlock(nn.Module):
    def __init__(self, dim, depth=2, drop_path=0.1):
        super().__init__()
        self.depth = depth

        self.mlp0 = FFN(dim)
        self.vfrs = nn.ModuleList([VFR(dim, dim) for _ in range(depth)])
        self.ffns = nn.ModuleList([FFN(dim) for _ in range(depth)])
        self.dp = nn.ModuleList([DropPath(drop_path) for _ in range(depth)])

    def forward(self, x, knn_idx):
        x = x + self.mlp0(x)
        for i in range(self.depth):
            x = x + self.dp[i](self.vfrs[i](x, knn_idx))
            x = x + self.dp[i](self.ffns[i](x))
        return x


# =========================================
# 7. DeepLA Backbone
# =========================================
class DeepLA_Backbone(nn.Module):
    def __init__(self, in_dim=3, embed_dim=64,
                 num_blocks=4, depth_per_block=2, k=16):
        super().__init__()
        self.knn_module = BatchedKNN(k)
        self.proj = nn.Linear(in_dim, embed_dim)

        self.blocks = nn.ModuleList([
            ResLFEBlock(embed_dim,
                        depth=depth_per_block,
                        drop_path=0.1)
            for _ in range(num_blocks)
        ])

    def forward(self, xyz):
        """
        xyz:  [B, N, 3]
        """
        x = self.proj(xyz)
        knn_idx = self.knn_module(xyz)   # [B,N,K]

        for blk in self.blocks:
            x = blk(x, knn_idx)

        return x


# =========================================
# 8. PointMLP 分类头（输入维度可控）
# =========================================
class PointMLP_Head(nn.Module):
    def __init__(self, in_dim,
                 point_hidden=64, global_hidden=256,
                 out_dim=1024, num_classes=40):
        super().__init__()

        self.pt_mlp = nn.Sequential(
            nn.Linear(in_dim, point_hidden),
            nn.ReLU(),
            nn.Linear(point_hidden, point_hidden),
            nn.ReLU(),
        )

        self.global_mlp = nn.Sequential(
            nn.Linear(point_hidden, global_hidden),
            nn.ReLU(),
            nn.Linear(global_hidden, out_dim)
        )

        self.classifier = nn.Linear(out_dim, num_classes)

    def forward(self, x):
        B, N, C = x.shape

        x = self.pt_mlp(x.reshape(B*N, C)).reshape(B, N, -1)
        x = x.max(dim=1)[0]      # [B, point_hidden]
        x = self.global_mlp(x)   # [B, out_dim]
        return self.classifier(x)


# =========================================
# 9. DeepLA + PointMLP 完整模型
# =========================================
class DeepLA_PointMLP_Model(nn.Module):
    def __init__(self,
                 num_classes=40,
                 embed_dim=64,
                 k=16,
                 num_blocks=4,
                 depth_per_block=2):
        super().__init__()

        self.backbone = DeepLA_Backbone(
            in_dim=3,
            embed_dim=embed_dim,
            num_blocks=num_blocks,
            depth_per_block=depth_per_block,
            k=k
        )

        self.classifier = PointMLP_Head(
            in_dim=embed_dim,
            num_classes=num_classes
        )

    def forward(self, xyz):
        feat = self.backbone(xyz)
        return self.classifier(feat)


# =========================================
# 10. Quick Test
# =========================================
if __name__ == "__main__":
    xyz = torch.randn(4, 1024, 3).cuda()
    model = DeepLA_PointMLP_Model(embed_dim=64).cuda()
    out = model(xyz)
    print("output:", out.shape)
