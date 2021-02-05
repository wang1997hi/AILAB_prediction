import numpy as np
from torch import nn
import torch


class DFN(nn.Module):
    def __init__(self, configs):
        super(DFN, self).__init__()
        self.configs = configs
        self.w_x = nn.Sequential(
                                nn.Conv2d(in_channels=configs.patch_size,out_channels=32,kernel_size=3,padding=3//2),
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
        self.last = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=82, kernel_size=1),
                                  nn.Conv2d(in_channels=82, out_channels=1, kernel_size=1))

    def forward(self, frames):
        # [batch, length, channel, height, width]
        next_frames = []
        batch = frames.shape[0]
        height = self.configs.img_width // self.configs.patch_size
        width = self.configs.img_width // self.configs.patch_size
        state = torch.zeros([batch, 128, height//2, width//2]).to(self.configs.device)
        for t in range(self.configs.total_length - 1):
            if t < self.configs.input_length:
                input_frame = frames[:, t]
            else:
                input_frame = x_gen
            state = self.w_s(state) + self.w_x(input_frame)
            filters = self.w_o(state)
            filters = self.last(filters)
            x_gen = filters
            if t >= self.configs.input_length - 1:
                next_frames.append(x_gen)
        # [length, batch, channel, height, width] -> [batch, length,height, width,channel ]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 2, 3, 4).contiguous()
        return next_frames
