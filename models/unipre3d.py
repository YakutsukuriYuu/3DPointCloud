import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# import or define UniPre3D backbone
# （从前面我们写好的 backbone 直接粘贴即可）
# ============================================================

# ========== FPS ==========
def fps(xyz, M):
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, M, dtype=torch.long, device=xyz.device)
    distance = torch.ones(B, N, device=xyz.device) * 1e10
    farthest = torch.randint(0, N, (B,), device=xyz.device)

    batch = torch.arange(B, device=xyz.device)
    for i in range(M):
        centroids[:, i] = farthest
        centroid = xyz[batch, farthest].view(B, 1, 3)
        dist = ((xyz - centroid) ** 2).sum(dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = distance.max(-1)[1]

    new_xyz = xyz.gather(1, centroids.unsqueeze(-1).repeat(1, 1, 3))
    return new_xyz, centroids


# ========== KNN ==========
try:
    from knn_cuda import KNN
    USE_KNN_CUDA = True
except:
    USE_KNN_CUDA = False
    print("⚠️ knn_cuda not found, fallback to torch.cdist (slow).")


def knn_query(xyz, new_xyz, k):
    B, N, _ = xyz.shape
    _, M, _ = new_xyz.shape

    if USE_KNN_CUDA:
        knn = KNN(k=k, transpose_mode=True)
        _, idx = knn(new_xyz, xyz)
        return idx.long()

    dist = torch.cdist(new_xyz, xyz)
    _, idx = torch.topk(dist, k, dim=-1, largest=False)
    return idx


# ========== index points ==========
def index_points(points, idx):
    B, N, C = points.shape
    _, M, K = idx.shape
    idx_expand = idx.unsqueeze(-1).expand(B, M, K, C)
    pts_expand = points.unsqueeze(1).expand(B, M, N, C)
    return torch.gather(pts_expand, 2, idx_expand)


# ========== Patch Embed ==========
class PatchEmbed(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, grouped_points):
        x = self.mlp(grouped_points)
        return x.max(dim=2)[0]


# ========== Transformer Block ==========
class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=4, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

        hidden = int(dim * mlp_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim)
        )

    def forward(self, x):
        x2 = self.norm1(x)
        attn_out, _ = self.attn(x2, x2, x2)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


# ============================================================
#             UniPre3D BACKBONE (High-fidelity)
# ============================================================
class UniPre3D_Backbone(nn.Module):
    def __init__(self,
                 in_dim=3,
                 embed_dims=[64, 128, 256, 512],
                 k=16):
        super().__init__()

        self.embed_dims = embed_dims
        self.k = k

        self.patch_layers = nn.ModuleList()
        self.tf_layers = nn.ModuleList()

        prev_dim = in_dim
        for dim in embed_dims:
            self.patch_layers.append(PatchEmbed(prev_dim + 3, dim))
            self.tf_layers.append(TransformerBlock(dim))
            prev_dim = dim

    def forward(self, xyz):
        B, N, _ = xyz.shape

        features = []
        current_xyz = xyz
        current_feat = xyz  # 初始特征 = xyz 本身

        for i, dim in enumerate(self.embed_dims):
            M = N // (4 ** i)

            new_xyz, fps_idx = fps(current_xyz, M)

            knn_idx = knn_query(current_xyz, new_xyz, self.k)
            grouped_xyz = index_points(current_xyz, knn_idx)
            grouped_feat = index_points(current_feat, knn_idx)

            patch_input = torch.cat([grouped_xyz, grouped_feat], dim=-1)

            embed = self.patch_layers[i](patch_input)
            feat = self.tf_layers[i](embed)

            features.append(feat)
            current_xyz, current_feat = new_xyz, feat

        global_feat = features[-1].mean(1)

        return {
            "x1": features[0],
            "x2": features[1],
            "x3": features[2],
            "x4": features[3],
            "global": global_feat
        }


# ============================================================
#                    POINTMLP CLASSIFIER
# ============================================================
class PointMLP_Head(nn.Module):
    def __init__(self, in_dim, point_hidden=64,
                 global_hidden=256, out_dim=1024,
                 num_classes=40):
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
        x = x.max(dim=1)[0]
        x = self.global_mlp(x)
        return self.classifier(x)


# ============================================================
#      UniPre3D + PointMLP → 完整分类模型
# ============================================================
class UniPre3D_PointMLP_Classifier(nn.Module):
    def __init__(self, num_classes=40,
                 embed_dims=[64, 128, 256, 512],
                 k=16):
        super().__init__()

        self.backbone = UniPre3D_Backbone(
            in_dim=3,
            embed_dims=embed_dims,
            k=k
        )

        self.classifier = PointMLP_Head(
            in_dim=embed_dims[0],  # x1 的 channel
            num_classes=num_classes
        )

    def forward(self, xyz):
        feats = self.backbone(xyz)
        x1 = feats["x1"]
        return self.classifier(x1)


# ============================================================
# Quick test
# ============================================================
if __name__ == "__main__":
    xyz = torch.randn(4, 2048, 3).cuda()
    model = UniPre3D_PointMLP_Classifier().cuda()

    out = model(xyz)
    print(out.shape)   # [4, 40]
