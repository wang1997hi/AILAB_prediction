import torch
import torch.nn as nn
from torch.nn import functional as F


class ConvLSTMCell(nn.Module):
    # def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, groups=1, bias=True):
    def __init__(self, in_channel, num_hidden, width, filter_size, stride, layer_norm):
        super(ConvLSTMCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self.bais_i = 0.001
        self.bais_f = 0.001
        self.bais_g = 0.001
        self.bais_o = 0.001
        self.conv_x = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden * 2, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden * 2, width, width])
        )
        self.conv_h = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 2, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden * 2, width, width])
        )
        self.conv_c = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden, width, width])
        )

    def forward(self, x_t, h_t, c_t):
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        f_c = self.conv_c(c_t)
        f_x, g_x = torch.split(x_concat, self.num_hidden, dim=1)
        f_h, g_h = torch.split(h_concat, self.num_hidden, dim=1)

        f_t = torch.sigmoid(f_x + f_h + f_c)
        g_t = torch.tanh(g_x + g_h)

        c_new = f_t * c_t + f_t * g_t
        h_new = f_t * torch.tanh(c_new)
        return h_new, (h_new, c_new)


class RgcLSTM(nn.Module):
    def __init__(self, configs):
        super(RgcLSTM, self).__init__()
        self.configs = configs
        self.str_R_channels = str(configs.patch_size*configs.patch_size*configs.img_channel)+', 64, 64, 64, 0'
        self.str_A_channels = str(configs.patch_size*configs.patch_size*configs.img_channel)+', 64, 64, 64'
        self.num_R_channels = [int(x) for x in self.str_R_channels.split(',')]
        self.num_A_channels = [int(x) for x in self.str_A_channels.split(',')]
        self.num_layers = len(self.num_R_channels) - 1
        self.frame_channel = configs.patch_size * configs.patch_size
        convcell_list = []
        conv_list = []
        update_A_list = []
        self.conv_num = 0
        width = configs.img_width // configs.patch_size
        # R
        for i in range(self.num_layers):
            convcell_list.append(ConvLSTMCell(2 * self.num_A_channels[i] + self.num_R_channels[i + 1], self.num_R_channels[i],
                                width // (2 ** i), configs.filter_size, configs.stride, configs.layer_norm))

        # A_hat
        for i in range(self.num_layers):
            conv = nn.Sequential(nn.Conv2d(self.num_R_channels[i], self.num_A_channels[i], 5, padding=2), nn.ReLU())
            if i == 0:
                conv.add_module('satlu', SatLU())
            conv_list.append(conv)

        self.upsample = nn.Upsample(scale_factor=2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # E,A
        for l in range(self.num_layers-1):
            update_A = nn.Sequential(nn.Conv2d(2 * self.num_A_channels[l], self.num_A_channels[l + 1], 5, padding=2),
                                     self.maxpool)
            update_A_list.append(update_A)

        self.convcell_list = nn.ModuleList(convcell_list)
        self.conv_list = nn.ModuleList(conv_list)
        self.update_A_list = nn.ModuleList(update_A_list)

    def forward(self, input):
        next_frames = []
        batch = input.shape[0]
        width = input.shape[4]//(2**self.conv_num)

        R_seq = [None] * self.num_layers   # R = h
        H_seq = [None] * self.num_layers   # H = (h,c)
        E_seq = [None] * self.num_layers

        for i in range(self.num_layers):
            E_seq[i] = torch.zeros(batch, 2 * self.num_A_channels[i], width, width).cuda()
            R_seq[i] = torch.zeros(batch, self.num_R_channels[i], width, width).cuda()
            width = width // 2

        for t in range(self.configs.total_length):
        # for t in range(self.configs.output_length):
            # A = input[:, t]
            # R
            for i in reversed(range(self.num_layers)):
                cell = self.convcell_list[i]
                if t == 0:
                    E = E_seq[i]
                    R = R_seq[i]
                    hx = (R, R)   # hx=(h,c)
                else:
                    E = E_seq[i]
                    R = R_seq[i]
                    hx = H_seq[i]
                if i == self.num_layers - 1:
                    R, hx = cell(E, hx[0], hx[1])
                else:
                    tmp = torch.cat((E, self.upsample(R_seq[i + 1])), 1)
                    R, hx = cell(tmp, hx[0], hx[1])
                R_seq[i] = R
                H_seq[i] = hx

            # A
            for i in range(self.num_layers):
                conv = self.conv_list[i]
                A_hat = conv(R_seq[i])
                if i == 0:
                    frame_prediction = A_hat
                    A = input[:, t]
                    # if t < self.configs.input_length:
                    #     A = input[:, t]
                    # else:
                    #     A = frame_prediction
                pos = F.relu(A_hat - A)
                neg = F.relu(A - A_hat)
                E = torch.cat([pos, neg], 1)
                E_seq[i] = E
                if i < self.num_layers - 1:
                    update_A = self.update_A_list[i]
                    A = update_A(E)
            if t >= self.configs.input_length:
                next_frames.append(frame_prediction)
            # next_frames.append(frame_prediction)

        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 2, 3, 4).contiguous()
        return next_frames


class SatLU(nn.Module):

    def __init__(self, lower=0, upper=255, inplace=False):
        super(SatLU, self).__init__()
        self.lower = lower
        self.upper = upper
        self.inplace = inplace

    def forward(self, input):
        return F.hardtanh(input, self.lower, self.upper, self.inplace)

    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else ''
        return self.__class__.__name__ + ' (' \
               + 'min_val=' + str(self.lower) \
               + ', max_val=' + str(self.upper) \
               + inplace_str + ')'
