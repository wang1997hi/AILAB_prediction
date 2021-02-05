import numpy as np
import torch
from torch.utils.data import Dataset
from skimage import transform
import os
from math import ceil

"""
data  set
"""
class MovingMNIST(Dataset):
    def __init__(self, path,args):
        self.img = np.load(path)  # 20,10000,64,64
        self.args = args
        self.args.input_length = 10
        self.args.output_length = 10
        self.args.total_length = 20
        self.args.img_width = 64
        self.args.img_channel = 1
        self.args.normalized_coe = 255

    def __getitem__(self, index):
        return self.img[:self.args.total_length, index, np.newaxis] / self.args.normalized_coe
        # b,l,c,h,w

    def __len__(self):
        return self.img.shape[1]

class UCF101(Dataset):
    def __init__(self, path,args):
        self.img = np.load(path)  # 20,10000,64,64
        self.args = args
        self.args.input_length = 10
        self.args.output_length = 10
        self.args.total_length = 20
        self.args.img_width = 64
        self.args.img_channel = 1
        self.args.normalized_coe = 255

    def __getitem__(self, index):
        return self.img[:self.args.total_length, index, np.newaxis] / self.args.normalized_coe
        # b,l,c,h,w

    def __len__(self):
        return self.img.shape[1]

class KTH(Dataset):
    def __init__(self, path,args):
        self.img = np.load(path)  # 20,10000,128,128
        self.args = args
        self.args.input_length = 10
        self.args.output_length = 10
        self.args.total_length = 20
        self.args.img_width = 128
        self.args.img_channel = 1
        self.args.normalized_coe = 255

    def __getitem__(self, index):
        return self.img[:self.args.total_length, index, np.newaxis] / self.args.normalized_coe

    def __len__(self):
        return self.img.shape[1]


class SRAD(Dataset):
    def __init__(self, path,args):
        self.img = np.load(path)  # 20,10000,64,64
        self.args = args
        self.args.input_length = 10
        self.args.output_length = 10
        self.args.total_length = 20
        self.args.img_width = 100
        self.args.img_channel = 1
        self.args.normalized_coe = 80

    def __getitem__(self, index):
        return self.img[:self.args.total_length, index, np.newaxis] / 80

    def __len__(self):
        return self.img.shape[1]


class Henan(Dataset):
    def __init__(self, path,args):
        self.data_list = []  # 4662,1,100,100
        self.data_length = []
        self.stride = 1
        self.args = args
        self.args.input_length = 10
        self.args.output_length = 10
        self.args.total_length = 20
        self.args.img_width = 100
        self.args.img_channel = 1
        self.args.normalized_coe = 80
        for root, dirs, files in os.walk(path):
            if files:
                for file in files:
                    data = np.load(root + '/' + file)[:,:self.args.img_channel]
                    self.data_list.append(data)
                    self.data_length.append((data.shape[0] - self.args.total_length+1) // self.stride)

    def __getitem__(self, index):
        for i in range(len(self.data_list)):
            if index - self.data_length[i] < 0:
                img = self.data_list[i][index * self.stride:index * self.stride + self.args.total_length] / self.args.normalized_coe
                return img  # l c h w
            else:
                index -= self.data_length[i]

    def __len__(self):
        return sum(self.data_length)


class Tianjin(Dataset):
    def __init__(self, path,args):
        self.data_list = []  # time,channel,width,height     降水 湿度 最高气温 最低气温 平均气温 风速
        self.stride = 1
        self.args = args
        self.args.input_length = 10
        self.args.output_length = 10
        self.args.total_length = 20
        self.args.img_width = 40
        self.args.img_channel = 1
        self.args.normalized_coe = 45
        self.data = np.load(path)[:,2:3]
        self.data_length = (self.data.shape[0] - self.args.total_length+1) // self.stride


    def __getitem__(self, index):
        img = self.data[index * self.stride:index * self.stride + self.args.total_length] / self.args.normalized_coe
        return img  # l c h w

    def __len__(self):
        return self.data_length




