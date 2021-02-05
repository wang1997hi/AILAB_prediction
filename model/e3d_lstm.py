import numpy as np
from torch import nn
import torch


class E3DLSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, window_length,width):
        super(E3DLSTMCell, self).__init__()

        self.num_hidden = num_hidden
        filter_size = (2,5,5)
        padding = (0,2,2)
        self.conv_x = nn.Sequential(
            nn.Conv3d(in_channel, num_hidden * 7, kernel_size=filter_size, padding=padding),
            nn.LayerNorm([num_hidden * 7, window_length,width, width])
        )
        self.conv_h = nn.Sequential(
            nn.Conv3d(num_hidden, num_hidden * 4, kernel_size=filter_size, padding=padding),
            nn.LayerNorm([num_hidden * 4, window_length,width, width])
        )
        self.conv_m = nn.Sequential(
            nn.Conv3d(num_hidden, num_hidden * 3, kernel_size=filter_size, padding=padding),
            nn.LayerNorm([num_hidden * 3, window_length,width, width])
        )
        self.conv_o = nn.Sequential(
            nn.Conv3d(num_hidden * 2, num_hidden, kernel_size=filter_size, padding=padding),
            nn.LayerNorm([num_hidden, window_length,width, width])
        )
        self.conv_last = nn.Conv3d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0)

        self.layer_norm = nn.LayerNorm([num_hidden,window_length,width,width])
    # def self_attention(self, r, c_history):
    #     batch_size = r.size(0)
    #     channels = r.size(1)
    #
    #     r_flatten = r.view(batch_size,channels,-1)
    #     c_history_flatten = c_history.view(batch_size,channels,-1)
    #     # b,t,x * b,tao,y = b,t,tao
    #     recall = torch.einsum('bcx,bcy->bxy',r_flatten,c_history_flatten)
    #     recall = torch.nn.functional.softmax(recall,dim=2)
    #     # b,t,tao * b,c,tao = b,c,t
    #     recall = torch.einsum('bxy,bcy->bxc',recall,c_history_flatten)
    #     recall = recall.view(r.shape)
    #     return recall
    def self_attention(self, r, c_history):
        batch_size = r.size(0)
        channels = r.size(1)
        shape = r.shape           # b,c,t,h,w
        r = r.view(batch_size,channels,-1)           # b,c,x
        c_history = c_history.view(batch_size,channels,-1)           # b,c,y
        recall = torch.bmm(r.permute(0,2,1).contiguous(),c_history)   # b,x,c * b,c,y -> b,x,y
        recall = torch.nn.functional.softmax(recall, dim=2)
        recall = torch.bmm(recall, c_history.permute(0, 2, 1).contiguous())      # b,x,y * b,y,c = b,x,c
        recall = recall.permute(0, 2, 1).contiguous()    # b,x,c->b,c,x
        recall = recall.view(shape)
        return recall


    def forward(self, x_t, h_t, c_t, m_t, c_history):
        x_t = torch.nn.functional.pad(x_t, [0, 0, 0, 0, 0, 1])
        h_t = torch.nn.functional.pad(h_t, [0, 0, 0, 0, 0, 1])
        m_t_p = torch.nn.functional.pad(m_t, [0, 0, 0, 0, 0, 1])

        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        m_concat = self.conv_m(m_t_p)

        i_x, r_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, r_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        i_m_prime, f_m_prime, g_m_prime = torch.split(m_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        r_t = torch.sigmoid(r_x + r_h)
        g_t = torch.tanh(g_x + g_h)

        recall = self.self_attention(r_t,c_history)
        c_new = self.layer_norm(c_t+recall) + i_t * g_t
        # c_new = self.layer_norm(r_t*c_t) + i_t * g_t
        i_t_prime = torch.sigmoid(i_x_prime + i_m_prime)
        f_t_prime = torch.sigmoid(f_x_prime + f_m_prime)
        g_t_prime = torch.tanh(g_x_prime + g_m_prime)

        m_new = f_t_prime * m_t + i_t_prime * g_t_prime

        mem = torch.cat((c_new, m_new), 1)
        mem_p = torch.nn.functional.pad(mem, [0, 0, 0, 0, 0, 1])
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem_p))
        h_new = o_t * torch.tanh(self.conv_last(mem))
        return h_new, c_new, m_new


class E3DLSTM(nn.Module):
    def __init__(self, configs):
        super(E3DLSTM, self).__init__()
        self.configs = configs
        self.str_hidden = '64,64,64,64'
        self.num_hidden = [int(x) for x in self.str_hidden.split(',')]
        self.num_layers = len(self.num_hidden)
        self.frame_channel = configs.patch_size * configs.patch_size * configs.img_channel
        cell_list = []
        self.window_length = 2
        self.window_stride = 1
        width = configs.img_width // configs.patch_size

        for i in range(self.num_layers):
            in_channel = self.frame_channel if i == 0 else self.num_hidden[i - 1]
            cell_list.append(E3DLSTMCell(in_channel, self.num_hidden[i], self.window_length,width))
        self.cell_list = nn.ModuleList(cell_list)
        # p = (out-1) * S + F - W

        self.decoder = nn.Conv3d(self.num_hidden[self.num_layers - 1], self.frame_channel,
                                 kernel_size=(self.window_length, 1, 1), stride=(self.window_length, 1, 1),padding=0)

    def forward(self, frames, mask_true):
        # frames shape:batch,length,channel,h,w
        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []
        # c_history[i] shape:batch,channel,length,h,w
        c_history = []

        input_list = []

        for time_step in range(self.window_length - 1):
            input_list.append(torch.zeros([batch, self.frame_channel, height, width]).to(self.configs.device))
        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], self.window_length, height, width]).to(self.configs.device)
            h_t.append(zeros)
            c_t.append(zeros)
            c_history.append(zeros)

        memory = torch.zeros([batch, self.num_hidden[0], self.window_length, height, width]).to(self.configs.device)

        for t in range(self.configs.total_length - 1):
            if t < self.configs.input_length:
                input_frame = frames[:, t]
            else:
                input_frame = mask_true[:, t - self.configs.input_length] * frames[:, t] + \
                              (1 - mask_true[:, t - self.configs.input_length]) * x_gen

            input_list.append(input_frame)
            # if t % (self.window_length - self.window_stride) == 0:
            input_frames = torch.stack(input_list[t:])
            input_frames = input_frames.permute(1, 2, 0, 3, 4).contiguous()
            h_t[0], c_t[0], memory = self.cell_list[0](input_frames, h_t[0], c_t[0], memory, c_history[0])

            c_history[0] = torch.cat([c_history[0], c_t[0]], 2)
            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory, c_history[i])
                c_history[i] = torch.cat([c_history[i], c_t[i]], 2)

            # x_gen shape: batch,channel,length,h,w
            x_gen = self.decoder(h_t[self.num_layers - 1])
            x_gen = torch.squeeze(x_gen,dim=2)
            next_frames.append(x_gen)

            # if t >= self.configs.input_length - 1:
            #     next_frames.append(x_gen)
        # [length, batch, channel, height, width] -> [batch, length,height, width,channel ]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 2, 3, 4).contiguous()
        return next_frames
