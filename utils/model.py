import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import numpy as np
from tqdm import tqdm



def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic=True #让 cuDNN 选择确定性的算法
    # torch.backends.cudnn.benchmark=False #不启用 cuDNN 的“自动选择最快算法”的功能。

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
        


def one_epoch_train(model,criterion,optimizer,dataloader:DataLoader,device):
    all_loss=0.0
    right_num=0
    total_samples=0
    model.train()
    for x,y in tqdm(dataloader):
        x=x.to(device)
        y=y.to(device)
        y_hat=model(x)
        loss=criterion(y_hat,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        all_loss+=loss.item()*x.size(0)
        right_num+=(y_hat.argmax(dim=1)==y).sum().item()
        batch_size=x.size(0)
        total_samples+=batch_size
    train_loss=all_loss/total_samples
    train_acc=right_num/total_samples
    return train_loss,train_acc

def one_epoch_eval(model,criterion,dataloader:DataLoader,device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x,y in tqdm(dataloader):
            x=x.to(device)
            y=y.to(device)
            y_hat=model(x)
            loss=criterion(y_hat,y)
            batch_size=x.size(0)
            total_loss+=loss.item()*batch_size
            pred=y_hat.argmax(dim=1)
            correct+=(pred==y).sum().item()
            total+=batch_size
        avg_loss=total_loss/total
        acc=correct/total
        return avg_loss,acc


def save(path,epoch,model,optimizer,val_acc,best_val_acc):
    ckpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_val_acc":best_val_acc,
        "val_acc": val_acc,
    }
    torch.save(ckpt, path)
    print(f"Checkpoint saved to {path}")

 

def loadlook(path,pth,weights_only=True):
    checkpoint=torch.load(f"{path}/{pth}",weights_only=weights_only)
    epoch=checkpoint["epoch"]
    val_acc=checkpoint["val_acc"]
    best_val_acc=checkpoint["best_val_acc"]
    print(f"epoch:{epoch} | val acc:{val_acc} | best val acc:{best_val_acc}")
# # ============= 推理（单张/批量） =============
# def predict(x):
#     model.eval()
#     with torch.no_grad():
#         x = x.to(self.device)
#         out = self.model(x)
#         return out
