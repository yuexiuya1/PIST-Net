import math
import argparse
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from model import multimodel
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from utils import read_split_data, multi_train_one_epoch, multi_evaluate,multi_final_evaluate,seed_torch
import matplotlib.pyplot as plt
import numpy as np
# from model.muti_scale_pc import MSPCNet
from cfg.cfg import cfg
import time
from astropy.io import fits

seed_torch()
torch.set_default_dtype(torch.float32)

base_data_path="./data/"

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
# model = multimodel.multi(num_classes=5,variant='s1')

# num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# print(f"The model has {num_params} parameters.")
# 1. 读取文件名和标签
# base_data_path="./data/"
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
        label = self.label_df.iloc[idx, 1:].values.astype('float32')

        # 读取csv文件中的数据
        spec_file_path = os.path.join(self.data_dir, str(file_name)+'.csv')
        spec_data = pd.read_csv(spec_file_path, header=None).values.astype('float32').flatten()

        # 转化为PyTorch张量
        spec_data = torch.tensor(spec_data, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        spec_data=torch.unsqueeze(spec_data, 0)
        pic_file_path = os.path.join("./data/pic72/", str(file_name)+'.fits')
        hdu=fits.open(pic_file_path)
        #data =  np.transpose(hdu[0].data.astype(np.float32), (1, 2, 0))
        pic_data =  hdu[0].data.astype(np.float32)


        # 转化为PyTorch张量
        pic_data = torch.tensor(pic_data, dtype=torch.float32)
        #label=torch.unsqueeze(label, 0)

        return spec_data, pic_data,label

# 3. 创建数据加载器
result=np.array([[0,0,0,0,0,0,0,0,0,0]])
# model.apply(initialize_weights)
# for fold in ["fold_1_"]:
for fold in ["fold_1_","fold_2_","fold_3_","fold_4_","fold_5_"]:

    # model=multimodel.multi()
    model = multimodel.multiTransformer()
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
    data_loader = DataLoader(traindataset, batch_size=32, shuffle=True)
    eval_data_loader = DataLoader(evaldataset, batch_size=64)
    trainloss= []
    evalloss = []
    minloss=100
    for epoch in range(100):
        # train
        train_loss, train_acc = multi_train_one_epoch(model=model,
                                                    optimizer=optimizer,
                                                    data_loader=data_loader,
                                                    device=device,
                                                    epoch=epoch)
        trainloss.append(train_loss)
        scheduler.step()

        # validate
        val_loss, val_acc = multi_evaluate(model=model,
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
    data=multi_final_evaluate(model=model,
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
# traindataset = CustomDataset(label_df, data_dir)
# evaldataset = CustomDataset(eval_label_df, data_dir)
# data_loader = DataLoader(traindataset, batch_size=64, shuffle=True)
# eval_data_loader = DataLoader(evaldataset, batch_size=64)
# pg = [p for p in model.parameters() if p.requires_grad]
# optimizer = optim.Adam(pg, lr=0.0001, weight_decay=1E-4)# momentum=0.9,
# # Scheduler https://arxiv.org/pdf/1812.01187.pdf
# lf = lambda x: ((1 + math.cos(x * math.pi / 100)) / 2) * (1 - 0.01) + 0.01  # cosine
# scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
# model.to(device)
# trainloss= []
# evalloss = []
# minloss=100
# for epoch in range(100):
#     # train
#     train_loss, train_acc = multi_train_one_epoch(model=model,
#                                                 optimizer=optimizer,
#                                                 data_loader=data_loader,
#                                                 device=device,
#                                                 epoch=epoch)
#     trainloss.append(train_loss)
#     scheduler.step()

#     # validate
#     val_loss, val_acc = multi_evaluate(model=model,
#                                      data_loader=eval_data_loader,
#                                      device=device,
#                                      epoch=epoch)
#     evalloss.append(val_loss)
#     if val_loss<minloss:
#         minloss=val_loss
#         torch.save(model.state_dict(), "./weights/bestmodel.pth")
#         pass
#     #torch.save(model.state_dict(), "model-{}.pth".format(epoch))


# print(f"minloss={minloss}")
# # model = MSPCNet(num_classes=5,spectrum_length=3844,**cfg["pc"])
# # load model weights
# model_weight_path = "./weights/bestmodel.pth"
# model.load_state_dict(torch.load(model_weight_path, map_location=device))
# model.to(device)
# data=multi_final_evaluate(model=model,
#                                      data_loader=eval_data_loader,
#                                      device=device)
# np.savetxt('final_evaluate.csv', data, delimiter=',')
# data=data[1:,:]
# print(f"persenterr={np.sum(np.abs(data[:,5:10]-data[:,0:5])/data[:,0:5],axis=0)/len(data[:,0])}")
# print(f"abserr={np.sum(np.abs(data[:,5:10]-data[:,0:5]),axis=0)/len(data[:,0])}")

# # 创建折线图
# plt.plot(trainloss, label='trainloss')
# plt.plot(evalloss, label='evalloss')

# # 添加标题和标签
# plt.title('Line Plot of Two Arrays')
# plt.xlabel('Index')
# plt.ylabel('Value')

# # 显示图例
# plt.legend()

# # 保存图像
# plt.savefig('line_plot.png')

# print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))