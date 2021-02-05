import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from data_provider import MovingMNIST,Henan,KTH,SRAD,Tianjin
from model.convlstm import ConvLSTM
from model.predrnn import PredRNN
from model.predrnnpp import PredRNNpp
from model.tc_lstm import TC_LSTM
from model.DFN import DFN
from model.CDNA import CDNA
from model.e3d_lstm import E3DLSTM

from parameter import get_args
from train_test import train, test

# 获取参数
args = get_args()
# 创建数据集
# data_train = SRAD(r'D:\SRAD\train.npy',args)
# data_test = SRAD(r'D:\SRAD\test.npy',args)
data_train = MovingMNIST(r'D:\MovingMnist\train.npy',args)
data_test = MovingMNIST(r'D:\MovingMnist\test.npy',args)
# data_train = KTH(r'D:\KTH\train.npy',args)
# data_test = KTH(r'D:\KTH\test.npy',args)
# data_train = Henan(r'D:\henan100\train',args)
# data_test = Henan(r'D:\henan100\test',args)
# data_train = Tianjin(r'D:\tianjin\downsample\haihe\six\train.npy',args)
# data_test = Tianjin(r'D:\tianjin\downsample\haihe\six\test.npy',args)
# 加载数据集
data_train = DataLoader(data_train, batch_size=args.batch_size, shuffle=True)
data_test = DataLoader(data_test, batch_size=64, shuffle=False)

net = TC_LSTM(args)

# 加载网络
if args.pre_trained == 1 or args.is_training == 0:
    net = torch.load(args.save_dir+'\\model.pth',map_location='cpu')
    net.configs = args
    # stats = torch.load(args.save_dir + "/model.pkl")
    # net.load_state_dict(stats)
net = net.to(args.device)

# 损失函数
criterion = []
criterion.append(nn.MSELoss().to(args.device))
criterion.append(nn.L1Loss().to(args.device))
# # 优化器
optimizer = optim.Adam(net.parameters(), lr=args.lr)

# 训练 测试
if args.is_training == 1:
    train(args, data_train,data_test,net,criterion, optimizer)
else:
    test(args,data_test,net)



