import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
def load_h5(h5_name):
    with h5py.File(h5_name,'r') as f:
        data=f['data'][:]
        label=f['label'][:]
    # print(data.shape, label.shape)
    # print("数据类型:", data.dtype, label.dtype)
    return data,label

def load_h5_dir(h5_dir,split="train"):
    all_data=[]
    all_label=[]
    import os
    list_file=os.path.join(h5_dir,f"{split}_files.txt")
    with open(list_file,"r") as f:
        h5_names=[line.strip() for line in f if line.strip()]
    for h5_name in h5_names:
        h5_path=os.path.join(h5_dir,h5_name)
        # print(h5_dir)
        # print(h5_name)
        data,label=load_h5(h5_path)
        all_data.append(data)
        all_label.append(label)
    all_data=np.concatenate(all_data,axis=0)
    all_label=np.concatenate(all_label,axis=0).squeeze()
    return all_data,all_label


# # modelnet40_ply_hdf5_2048.zip
# # 这个数据集已经归一化了 
# class ModelNet40Dataset(Dataset):
#     def __init__(self,data,labels,num_points=1024,train=True):
#         super().__init__()
#         self.data=data
#         self.labels=labels
#         self.num_points=num_points
#         self.train=train
#     def __getitem__(self, index):
#         points=self.data[index]
#         labels=self.labels[index]

#         if self.train:
#             # 1024 采样 平均
#             choice=np.random.choice(points.shape[0],self.num_points,replace=False) # 不放回采样
#         else:
#             # 测试无所吊谓
#             choice=np.arange(self.num_points)
            
#         points=points[choice,:]

#         # 显式转张量
#         points=torch.tensor(points).float()
#         labels=torch.tensor(labels).long()
#         return points,labels
#     def __len__(self):
#         return self.data.shape[0]
  


class ModelNet40Dataset(Dataset):
    def __init__(
        self,
        data,
        labels,
        num_points=1024,
        train=True,
        noise_sigma=0.0,        # 高斯噪声标准差
        occlusion_ratio=0.0,    # 删除点比例（随机遮挡）
        block_occlusion=False,  # True = 局部区域遮挡(球形/方块)
    ):
        super().__init__()
        self.data = data
        self.labels = labels
        self.num_points = num_points
        self.train = train

        # ===== 新增鲁棒性控制参数 =====
        self.noise_sigma = noise_sigma
        self.occlusion_ratio = occlusion_ratio
        self.block_occlusion = block_occlusion

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        points = self.data[index]
        label = self.labels[index]

        # ======================
        # 1) 固定点数随机采样
        # ======================
        if self.train:
            choice = np.random.choice(points.shape[0], self.num_points, replace=False)
        else:
            # test 无所谓，只取前 num_points
            choice = np.arange(self.num_points)

        points = points[choice, :]

        # ======================
        # 2) 高斯噪声（Noise）
        # ======================
        if self.noise_sigma > 0:
            noise = np.random.normal(0, self.noise_sigma, size=points.shape)
            points = points + noise

        # ======================
        # 3) 遮挡（Occlusion）
        # ======================

        # --- A) 随机删除点 ---
        if self.occlusion_ratio > 0 and not self.block_occlusion:
            keep = int(self.num_points * (1 - self.occlusion_ratio))
            choice = np.random.choice(self.num_points, keep, replace=False)
            points = points[choice, :]

        # --- B) 局部区域遮挡（球形区域） ---
        if self.block_occlusion:
            # 随机遮挡区域
            center = np.random.uniform(-0.2, 0.2, size=(3,))
            radius = 0.3
            dist = np.linalg.norm(points - center, axis=1)
            points = points[dist > radius]

        # ======================
        # 4) 遮挡后补齐点数（保证固定点数）
        # ======================
        if points.shape[0] < self.num_points:
            pad = np.random.choice(points.shape[0], self.num_points - points.shape[0], replace=True)
            points = np.concatenate([points, points[pad]], axis=0)

        # ======================
        # 5) 转换为 Tensor
        # ======================
        points = torch.tensor(points).float()
        label = torch.tensor(label).long()

        return points, label

    

def init_data(data_path="./dataset/modelnet40_ply_hdf5_2048"):
    #data_path="./dataset/modelnet40_ply_hdf5_2048"
    train_data,train_label=load_h5_dir(data_path,"train")
    test_data,test_label=load_h5_dir(data_path,"test")

    train_set_mild = ModelNet40Dataset(
        train_data, train_label,
        train=True,
        noise_sigma=0.005,       # 很小的轻噪声
        occlusion_ratio=0.1,     # 删10%点
        block_occlusion=False
    )
    train_set_moderate = ModelNet40Dataset(
        train_data, train_label,
        train=True,
        noise_sigma=0.01,        # 中等噪声
        occlusion_ratio=0.2,     # 随机删掉20%
        block_occlusion=False
    )

    train_set_heavy = ModelNet40Dataset(
        train_data, train_label,
        train=True,
        noise_sigma=0.02,        # 强噪声
        occlusion_ratio=0.4,     # 删除 40% 点
        block_occlusion=True     # 加局部区域遮挡
    )


    train_set=ModelNet40Dataset(data=train_data,labels=train_label,num_points=1024,train=True)


    # 测试数据集 做鲁棒性测试
    test_set=ModelNet40Dataset(data=test_data,labels=test_label,num_points=1024,train=False) # 干净数据
    
    test_noise = ModelNet40Dataset(test_data, test_label, num_points=1024, train=False, noise_sigma=0.02)  #噪声 σ=0.02
    test_occ = ModelNet40Dataset(test_data, test_label,num_points=1024, train=False, occlusion_ratio=0.5)  #随机遮挡 50%
    test_block = ModelNet40Dataset(test_data, test_label,num_points=1024, train=False, block_occlusion=True) #区域遮挡


    train_loader=DataLoader(train_set,batch_size=32,shuffle=True,num_workers=8,drop_last=True)
    train_loader_mild = DataLoader(train_set_mild, batch_size=32, shuffle=True)
    train_loader_mod  = DataLoader(train_set_moderate, batch_size=32, shuffle=True)
    train_loader_heavy = DataLoader(train_set_heavy, batch_size=32, shuffle=True)


     # ====== 测试数据集（干净数据）======
    test_loader = DataLoader(
        test_set,
        batch_size=32,
        shuffle=False,
        num_workers=8,
        drop_last=False
    )

    # ====== 噪声测试（Noise Robustness）======
    test_loader_noise = DataLoader(
        test_noise,
        batch_size=32,
        shuffle=False,
        num_workers=8,
        drop_last=False
    )

    # ====== 遮挡测试（Random Occlusion）======
    test_loader_occ = DataLoader(
        test_occ,
        batch_size=32,
        shuffle=False,
        num_workers=8,
        drop_last=False
    )

    # ====== 局部区域遮挡（Block Occlusion）======
    test_loader_block = DataLoader(
        test_block,
        batch_size=32,
        shuffle=False,
        num_workers=8,
        drop_last=False
    )
    #return train_loader,test_loader_noise
    return train_loader, test_loader, test_loader_noise, test_loader_occ, test_loader_block