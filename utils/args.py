import torch

ckpt_path = "../checkpoints/deepla/best.pth"

# 1. 加载原 checkpoint
ckpt = torch.load(ckpt_path, map_location="cpu")

# 2. 添加你想要的新字段
ckpt["val_acc"] = ckpt["best_val_acc"]

torch.save(ckpt, ckpt_path)

print("✅ 成功添加新的字段 new_field")
