import numpy as np
from torch import nn
import torch

class ConvLSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, width, filter_size, stride, layer_norm):
        super(ConvLSTMCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self.conv_x = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden * 4, width, width])
        )
        self.conv_h = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden * 4, width, width])
        )
        self.conv_c = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 2, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden * 2, width, width])
        )
        self.conv_o = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden, width, width])
        )


    def forward(self, x_t, h_t, c_t):
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        c_concat = self.conv_c(c_t)
        i_x, f_x, g_x, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        i_c, f_c = torch.split(c_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h + i_c)
        f_t = torch.sigmoid(f_x + f_h + f_c)
        g_t = torch.tanh(g_x + g_h)

        c_new = f_t * c_t + i_t * g_t
        mem = torch.cat((x_t, h_t), 1)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(c_new))
        h_new = o_t * torch.tanh(c_new)
        return h_new, c_new

class ConvLSTM(nn.Module):
    def __init__(self, configs):
        super(ConvLSTM, self).__init__()
        self.configs = configs
        self.str_hidden = '64,64,64,64'
        self.num_hidden = [int(x) for x in self.str_hidden.split(',')]
        self.num_layers = len(self.num_hidden)
        self.frame_channel = configs.patch_size * configs.patch_size
        cell_list = []
        self.conv_num = 0
        width = configs.img_width // configs.patch_size

        for i in range(self.num_layers):
            in_channel = self.frame_channel if i == 0 else self.num_hidden[i-1]
            # in_channel = 64 if i == 0 else num_hidden[i-1]
            cell_list.append(
                ConvLSTMCell(in_channel, self.num_hidden[i], width//(2**self.conv_num), configs.filter_size,
                                       configs.stride, configs.layer_norm)
            )
        self.cell_list = nn.ModuleList(cell_list)

        self.conv_last = nn.Conv2d(self.num_hidden[self.num_layers-1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)


    def forward(self, frames):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        # frames = frames.permute(0, 1, 4, 2, 3).contiguous()
        batch = frames.shape[0]
        height = frames.shape[3]//(2**self.conv_num)
        width = frames.shape[4]//(2**self.conv_num)

        next_frames = []
        h_t = []
        c_t = []

        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height, width]).to(self.configs.device)
            h_t.append(zeros)
            c_t.append(zeros)


        for t in range(self.configs.total_length-1):
            if t < self.configs.input_length:
                net = frames[:,t]
            else:
                net = x_gen
            # net = self.conv_first(net)
            h_t[0], c_t[0] = self.cell_list[0](net, h_t[0], c_t[0])
            for i in range(1, self.num_layers):
                h_t[i], c_t[i] = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i])
            x_gen = self.conv_last(h_t[self.num_layers-1])
            # x_gen = self.transpose(h_t[self.num_layers-1])

            if t >= self.configs.input_length-1:
                next_frames.append(x_gen)
        # [length, batch, channel, height, width] -> [batch, length,height, width,channel ]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 2, 3, 4).contiguous()
        return next_frames