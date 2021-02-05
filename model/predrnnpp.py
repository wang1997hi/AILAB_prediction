import numpy as np
from torch import nn
import torch


class CasualLSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, width, filter_size, stride):
        super(CasualLSTMCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self.conv_x = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden * 7, width, width])
        )
        self.conv_h = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden * 3, width, width])
        )
        self.conv_m = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden * 4, width, width])
        )
        self.conv_c = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden * 3, width, width])
        )
        self.conv_c_new = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden * 3, width, width])
        )
        self.conv_o = nn.Sequential(
            nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden, width, width])
        )
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0)

    def forward(self, x_t, h_t, c_t, m_t):
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        m_concat = self.conv_m(m_t)
        c_concat = self.conv_c(c_t)
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h = torch.split(h_concat, self.num_hidden, dim=1)
        i_m_prime, f_m_prime, g_m_prime, m_m = torch.split(m_concat, self.num_hidden, dim=1)
        i_c, f_c, g_c = torch.split(c_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h + i_c)
        f_t = torch.sigmoid(f_x + f_h + f_c)
        g_t = torch.tanh(g_x + g_h + g_c)

        c_new = f_t * c_t + i_t * g_t

        c_concat_new = self.conv_c_new(c_new)
        i_c_new, f_c_new, g_c_new = torch.split(c_concat_new, self.num_hidden, dim=1)

        i_t_prime = torch.sigmoid(i_x_prime + i_m_prime + i_c_new)
        f_t_prime = torch.sigmoid(f_x_prime + f_m_prime + f_c_new)
        g_t_prime = torch.tanh(g_x_prime + g_m_prime + g_c_new)

        m_new = f_t_prime * torch.tanh(m_m) + i_t_prime * g_t_prime

        mem = torch.cat((c_new, m_new), 1)
        o_t = torch.tanh(o_x + self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))
        return h_new, c_new, m_new


class GHU(nn.Module):
    def __init__(self, in_channel, num_hidden, width, filter_size, stride):
        super(GHU, self).__init__()
        self.padding = filter_size // 2
        self.num_hidden = num_hidden
        self.conv_x = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden * 2, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden * 2, width, width])
        )
        self.conv_z = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden * 2, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden * 2, width, width])
        )

    def forward(self, x_t, z_t):
        x_concat = self.conv_x(x_t)
        z_concat = self.conv_z(z_t)
        p_x, s_x = torch.split(x_concat, self.num_hidden, dim=1)
        p_z, s_z = torch.split(z_concat, self.num_hidden, dim=1)

        p_t = torch.tanh(p_x + p_z)
        s_t = torch.sigmoid(s_x + s_z)
        z_new = s_t * p_t + (1 - s_t) * z_t
        return z_new


class PredRNNpp(nn.Module):
    def __init__(self, configs):
        super(PredRNNpp, self).__init__()
        self.configs = configs
        self.str_hidden = '128,128,128,128'
        self.num_hidden = [int(x) for x in self.str_hidden.split(',')]
        self.num_layers = len(self.num_hidden)
        self.frame_channel = configs.patch_size * configs.patch_size * configs.img_channel
        self.filter_size = 5
        self.stride = 1
        cell_list = []
        width = configs.img_width // configs.patch_size

        for i in range(self.num_layers):
            in_channel = self.frame_channel if i == 0 else self.num_hidden[i - 1]
            cell_list.append(
                CasualLSTMCell(in_channel, self.num_hidden[i], width, self.filter_size,
                               self.stride))
        self.cell_list = nn.ModuleList(cell_list)
        self.GHU = GHU(self.num_hidden[0], self.num_hidden[0], width, self.filter_size,
                       self.stride)
        self.conv_last = nn.Sequential(nn.Conv2d(self.num_hidden[-1],self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False))

    def forward(self, frames, mask_true):
        # [batch, length, channel, height, width]
        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]
        next_frames = []
        h_t = []
        c_t = []


        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height, width]).to(self.configs.device)
            h_t.append(zeros)
            c_t.append(zeros)

        z = torch.zeros([batch, self.num_hidden[0], height, width]).to(self.configs.device)
        memory = torch.zeros([batch, self.num_hidden[0], height, width]).to(self.configs.device)

        for t in range(self.configs.total_length - 1):
            if t < self.configs.input_length:
                input_frame = frames[:, t]
            else:
                input_frame = mask_true[:, t - self.configs.input_length] * frames[:, t] + \
                      (1 - mask_true[:, t - self.configs.input_length]) * x_gen

            h_t[0], c_t[0], memory = self.cell_list[0](input_frame, h_t[0], c_t[0], memory)
            z = self.GHU(h_t[0], z)
            h_t[1], c_t[1], memory = self.cell_list[1](z, h_t[1], c_t[1], memory)
            for i in range(3, self.num_layers):
                h_t[i], c_t[i], memory = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)
            x_gen = self.conv_last(h_t[self.num_layers - 1])
            # if t >= self.configs.input_length - 1:
            #     next_frames.append(x_gen)
            next_frames.append(x_gen)
        # [length, batch, channel, height, width] -> [batch, length,channel ,height, width]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 2, 3, 4).contiguous()
        return next_frames
