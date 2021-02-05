import numpy as np
from torch import nn
import torch


class DFN(nn.Module):
    def __init__(self, configs):
        super(DFN, self).__init__()
        self.configs = configs
        self.configs.patch_size = 1
        ###############################
        #  filter-generating network  #
        ###############################
        self.w_x = nn.Sequential(
                                nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding=3//2),
                                nn.LeakyReLU(),
                                nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,padding=3//2,stride=2),
                                nn.LeakyReLU(),
                                nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=3//2),
                                nn.LeakyReLU(),
                                nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding=3//2),
                                nn.LeakyReLU(),
                                nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=3//2),
                                nn.LeakyReLU())
        self.w_s = nn.Sequential(
                                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=3 // 2),
                                nn.LeakyReLU(),
                                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=3 // 2),
                                nn.LeakyReLU())
        self.w_o = nn.Sequential(
                                nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=3 // 2),
                                nn.LeakyReLU(),
                                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2),
                                nn.LeakyReLU(),
                                nn.Upsample(scale_factor=2),
                                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2),
                                nn.LeakyReLU(),
                                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2),
                                nn.LeakyReLU(),
                                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1),
                                nn.LeakyReLU())
        self.last = nn.Conv2d(in_channels=128, out_channels=82, kernel_size=1)

    def forward(self, frames, mask_true):
        # [batch, length, channel, height, width]
        next_frames = []
        batch = frames.shape[0]
        height = self.configs.img_width // self.configs.patch_size
        width = self.configs.img_width // self.configs.patch_size
        state = torch.zeros([batch, 128, height//2, width//2]).to(self.configs.device)
        for t in range(self.configs.total_length - 1):
            if t < self.configs.input_length:
                input_frame_a = frames[:, t]
            else:
                input_frame_a = mask_true[:, t - self.configs.input_length] * frames[:, t] + \
                              (1 - mask_true[:, t - self.configs.input_length]) * x_gen
            state = self.w_s(state) + self.w_x(input_frame_a)
            filters = self.w_o(state)
            filters = self.last(filters)[:,:81]
            #########################ewa
            #  transformer network  #
            #########################
            filters = filters.permute(0,2,3,1).contiguous()
            filters = filters.reshape(-1,81)
            filters = nn.functional.softmax(filters,dim=1)
            filters = filters.reshape(batch,height,width, 81)
            filters = filters.permute(0,3,1,2).contiguous()
            filter_localexpand = np.reshape(np.eye(81, 81),(81, 1, 9, 9))
            filter_localexpand = torch.from_numpy(filter_localexpand).float().to(self.configs.device)
            input_frame_b = torch.nn.functional.conv2d(input=input_frame_a,weight=filter_localexpand,padding=9//2)
            output = input_frame_b * filters
            output = torch.sum(output,dim=1,keepdim=True)
            x_gen = output
            # if t >= self.configs.input_length - 1:
            next_frames.append(x_gen)
        # [length, batch, channel, height, width] -> [batch, length,height, width,channel ]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 2, 3, 4).contiguous()
        return next_frames
