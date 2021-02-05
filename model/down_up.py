import numpy as np
from torch import nn
import torch

class Down_sample(nn.Module):
    def __init__(self):
        super(Down_sample, self).__init__()
        self.downsample = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1))

    def forward(self, frames):
        result = self.downsample(frames)

        return result


class Up_sample(nn.Module):
    def __init__(self):
        super(Up_sample, self).__init__()

        self.upsample = nn.Sequential(nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2,
                                                         padding=1, output_padding=1),
                                      nn.ReLU(inplace=True),

                                      nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2,
                                                         padding=1, output_padding=1),
                                      nn.ReLU(inplace=True),

                                      nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2,
                                                         padding=1, output_padding=1),
                                      nn.ReLU(inplace=True),

                                      nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2,
                                                         padding=1, output_padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1))

    def forward(self, frames):
        result = self.upsample(frames)
        return result