import math
import argparse
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from model import effnet1d
from model import testmodel
from model import rac_net
from model import sscnn
from model import convnext_1d
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from utils import initialize_weights, train_one_epoch, evaluate,final_evaluate,seed_torch
import matplotlib.pyplot as plt
import numpy as np
from model.muti_scale_pc import MSPCNet
from cfg.cfg import cfg
import time
import random

seed_torch()
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

# num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# print(f"The model has {num_params} parameters.")
# 1. 读取文件名和标签
# base_data_path="./data/fold_1/"
# train_label = base_data_path+'train_labels.csv'  # 替换为实际路径
# train_data_dir = base_data_path+"train_data"  # 替换为实际路径
# eval_label=base_data_path+'val_labels.csv'
# eval_data_dir = base_data_path+"val_data"
base_data_path="./data/"
# fold="fold_1_"
# train_label = base_data_path+fold+'train_labels.csv'  # 替换为实际路径
# data_dir = base_data_path+"data/"  # 替换为实际路径
# eval_label=base_data_path+fold+'val_labels.csv'

# label_df = pd.read_csv(train_label)
# eval_label_df = pd.read_csv(eval_label)
# 2. 定义数据集类
class CustomDataset(Dataset):
    def __init__(self, label_df, data_dir):
        self.label_df = label_df
        self.data_dir = data_dir

    def __len__(self):
        return len(self.label_df)

    def __getitem__(self, idx):
        file_name = self.label_df.iloc[idx, 0]
        label = self.label_df.iloc[idx, 1:6].values.astype('float32')

        # 读取csv文件中的数据
        file_path = os.path.join(self.data_dir, str(file_name)+'.csv')
        data = pd.read_csv(file_path, header=None).values.astype('float32').flatten()

        # 转化为PyTorch张量
        data = torch.tensor(data, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        data=torch.unsqueeze(data, 0)
        #label=torch.unsqueeze(label, 0)

        return data, label

# 3. 创建数据加载器

# traindataset = CustomDataset(label_df, data_dir)
# evaldataset = CustomDataset(eval_label_df, data_dir)
# data_loader = DataLoader(traindataset, batch_size=64, shuffle=True)
# eval_data_loader = DataLoader(evaldataset, batch_size=64)



result=np.array([[0,0,0,0,0,0,0,0,0,0]])
# model.apply(initialize_weights)
# for fold in ["fold_1_"]:
for fold in ["fold_1_","fold_2_","fold_3_","fold_4_","fold_5_"]:
    # model=cyh.efficientnetv2_s()
    # model=testmodel.SPNet()
    model=convnext_1d.CONVNEXT1D(in_channel=1,out_channel=5,spectrum_size=3844)
    # model = sscnn.SSCNN(in_channel=1,out_channel=5,spectrum_size=3844)
    # model=MSPCNet(num_classes=5,spectrum_length=3844,**cfg["pc2"])
    # model = testmodel.AFGK_star()
    # model = testmodel.MultiModalModel()
    # model = rac_net.RACNET(in_channel=1,out_channel=5,spectrum_size=3834)
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(pg, lr=0.0001, weight_decay=1E-3)# momentum=0.9,
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / 100)) / 2) * (1 - 0.01) + 0.01  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    model.to(device)
    train_label = base_data_path+fold+'train_labels.csv'  # 替换为实际路径
    data_dir = base_data_path+"data/"  # 替换为实际路径
    eval_label=base_data_path+fold+'val_labels.csv'
    label_df = pd.read_csv(train_label)
    eval_label_df = pd.read_csv(eval_label)
    traindataset = CustomDataset(label_df, data_dir)
    evaldataset = CustomDataset(eval_label_df, data_dir)
    data_loader = DataLoader(traindataset, batch_size=64, shuffle=True)
    eval_data_loader = DataLoader(evaldataset, batch_size=64)
    trainloss= []
    evalloss = []
    minloss=100
    for epoch in range(100):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                    optimizer=optimizer,
                                                    data_loader=data_loader,
                                                    device=device,
                                                    epoch=epoch)
        trainloss.append(train_loss)
        scheduler.step()

        # validate
        val_loss, val_acc = evaluate(model=model,
                                        data_loader=eval_data_loader,
                                        device=device,
                                        epoch=epoch)
        evalloss.append(val_loss)
        if val_loss<minloss:
            minloss=val_loss
            torch.save(model.state_dict(), "./weights/bestmodel.pth")
            pass

    model_weight_path = "./weights/bestmodel.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.to(device)
    data=final_evaluate(model=model,
                                        data_loader=eval_data_loader,
                                        device=device)
    data=data[1:,:]
    result=np.concatenate([result, data], axis=0)
    # print(result.shape)

    #torch.save(model.state_dict(), "model-{}.pth".format(epoch))


# print(f"minloss={minloss}")
# load model weights

np.savetxt('final_evaluate.csv', result, delimiter=',')
data=result[1:,:]
print(f"persenterr={np.sum(np.abs(data[:,5:10]-data[:,0:5])/data[:,0:5],axis=0)/len(data[:,0])}")
print(f"abserr={np.sum(np.abs(data[:,5:10]-data[:,0:5]),axis=0)/len(data[:,0])}")
# 创建折线图
plt.plot(trainloss, label='trainloss')
plt.plot(evalloss, label='evalloss')

# 添加标题和标签
plt.title('Line Plot of Two Arrays')
plt.xlabel('Index')
plt.ylabel('Value')

# 显示图例
plt.legend()

# 保存图像
plt.savefig('line_plot.png')

print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))