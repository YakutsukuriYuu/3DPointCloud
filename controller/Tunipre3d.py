import torch
from utils.data import init_data
#from models.deepla import DeepLA_PointMLP_Model
from models.unipre3d import UniPre3D_PointMLP_Classifier
from utils.model import *

def resume_train(lr=1e-3,epochs=250,path="./checkpoints/unipre3d"):
    set_seed()
    device=get_device()
    train_loader,test_loader=init_data()
    checkpoint = torch.load(f"{path}/best.pth", map_location=device)
    model=UniPre3D_PointMLP_Classifier().to(device)
    model.load_state_dict(checkpoint["model"])
    optimizer=torch.optim.Adam(params=model.parameters(),lr=lr,weight_decay=1e-4)
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_epoch=checkpoint['epoch']+1
    criterion=torch.nn.CrossEntropyLoss()
    best_val_acc = 0.0
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=20,gamma=0.5)
    for epoch in range(start_epoch,epochs):
        _,_=one_epoch_train(model=model,criterion=criterion,optimizer=optimizer,dataloader=train_loader,device=device)
        avg_loss,acc=one_epoch_eval(model=model,criterion=criterion,dataloader=test_loader,device=device)
        # 每个 epoch 结束后更新学习率
        scheduler.step()
        # 看一下当前 lr
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"epoch:{epoch+1} | lr={current_lr:.6f} | avg_loss{avg_loss:.4f} | acc:{acc:.4f}")

        if epoch+1==epochs:
            #def save(path,epoch,model,optimizer,val_acc,best_val_acc):
            save(f"{path}/last_{epoch+1}.pth",epoch+1,model,optimizer,acc,best_val_acc)
        if acc>best_val_acc:
            best_val_acc=acc
            save(f"{path}/best.pth",epoch+1,model,optimizer,acc,best_val_acc)
def train(lr=1e-3,epochs=250,path="./checkpoints/unipre3d"):
    set_seed()
    device=get_device()
    train_loader,test_loader=init_data()
    model=UniPre3D_PointMLP_Classifier().to(device)
    criterion=torch.nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(params=model.parameters(),lr=lr,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=20,gamma=0.5)


    best_val_acc = 0.0

    for epoch in range(epochs):
        _,_=one_epoch_train(model=model,criterion=criterion,optimizer=optimizer,dataloader=train_loader,device=device)
        avg_loss,acc=one_epoch_eval(model=model,criterion=criterion,dataloader=test_loader,device=device)
        # 每个 epoch 结束后更新学习率
        scheduler.step()
        # 看一下当前 lr
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"epoch:{epoch+1} | lr={current_lr:.6f} | avg_loss{avg_loss:.4f} | acc:{acc:.4f}")

        if epoch+1==epochs:
            #def save(path,epoch,model,optimizer,val_acc,best_val_acc):
            save(f"{path}/last_{epoch+1}.pth",epoch+1,model,optimizer,acc,best_val_acc)
        if acc>best_val_acc:
            best_val_acc=acc
            save(f"{path}/best.pth",epoch+1,model,optimizer,acc,best_val_acc)
def eval(path="./checkpoints/unipre3d"):
    # ----------- 加载模型权重 -----------
    checkpoint = torch.load(f"{path}/best.pth")
    loadlook(path, "best.pth")
    set_seed()
    device = get_device()

    # ----------- 加载四种测试集 -----------
    train_loader, test_loader_clean, test_loader_noise, test_loader_occ, test_loader_block = init_data()

    # ----------- 创建模型 -----------
    model=UniPre3D_PointMLP_Classifier().to(device)
    model.load_state_dict(checkpoint["model"])
    criterion = torch.nn.CrossEntropyLoss()

    print("====== Running Robustness Evaluation ======")

    # ========== Clean test ==========
    loss_clean, acc_clean = one_epoch_eval(model, criterion, test_loader_clean, device)
    print(f"[Clean]      loss={loss_clean:.4f} | acc={acc_clean:.4f}")

    # ========== Noise test ==========
    loss_noise, acc_noise = one_epoch_eval(model, criterion, test_loader_noise, device)
    print(f"[Noise σ=0.02] loss={loss_noise:.4f} | acc={acc_noise:.4f}")

    # ========== Occlusion test ==========
    loss_occ, acc_occ = one_epoch_eval(model, criterion, test_loader_occ, device)
    print(f"[Occlusion 50%] loss={loss_occ:.4f} | acc={acc_occ:.4f}")

    # ========== Block Occlusion test ==========
    loss_block, acc_block = one_epoch_eval(model, criterion, test_loader_block, device)
    print(f"[Block Occlusion] loss={loss_block:.4f} | acc={acc_block:.4f}")

    print("============================================")


    