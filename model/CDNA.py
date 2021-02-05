import math
import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F


class ConvLSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, filter_size,configs,stride=1, padding=0):
        super(ConvLSTMCell, self).__init__()
        self.num_hidden = num_hidden
        self.configs = configs
        self.padding = padding if padding else filter_size // 2
        self.conv_x = nn.Conv2d(in_channel, num_hidden * 4, kernel_size=filter_size, stride=stride,
                                padding=self.padding)
        self.conv_h = nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride,
                                padding=self.padding)
        self.conv_o = nn.Conv2d(num_hidden, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding)

    def forward(self, x_t, states):
        if states is None:
            states = (
                torch.zeros([x_t.shape[0], self.num_hidden, x_t.shape[2], x_t.shape[2]]).to(self.configs.device),
                torch.zeros([x_t.shape[0], self.num_hidden, x_t.shape[2], x_t.shape[2]]).to(self.configs.device)
            )
        c_t, h_t = states
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        i_x, f_x, g_x, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h)
        g_t = torch.tanh(g_x + g_h)
        o_t = torch.sigmoid(o_x + o_h)
        c_new = f_t * c_t + i_t * g_t
        h_new = o_t * torch.tanh(c_new)

        return h_new, (c_new, h_new)


class CDNA(nn.Module):
    def __init__(self, configs):
        super(CDNA, self).__init__()
        self.configs = configs
        self.str_hidden = '32, 32, 64, 64, 128, 64, 32'
        self.lstm_size = [int(x) for x in self.str_hidden.split(',')]
        self.num_layers = len(self.lstm_size)
        self.frame_channel = configs.patch_size * configs.patch_size * configs.img_channel
        width = configs.img_width // configs.patch_size
        self.num_masks = 10
        self.is_robot_state_used = False  # 是否使用机械臂的状态：action & states
        self.KERNEL_SIZE = 5

        if not self.is_robot_state_used:
            self.POS_DIM = 0
            self.ACTION_DIM = 0
        else:
            self.POS_DIM = 3
            self.ACTION_DIM = 4

        '''
        Define each layer of the model.
        '''
        self.conv1 = nn.Conv2d(in_channels=self.frame_channel, out_channels=self.lstm_size[0], kernel_size=5, stride=2,
                               padding=2)
        self.conv1_norm = nn.LayerNorm([self.lstm_size[0], width // 2, width // 2])

        self.lstm1 = ConvLSTMCell(in_channel=self.lstm_size[0], configs=self.configs,num_hidden=self.lstm_size[0], filter_size=5,
                                  stride=1, padding=2)
        self.lstm1_norm = nn.LayerNorm([self.lstm_size[0], width // 2, width // 2])

        self.lstm2 = ConvLSTMCell(in_channel=self.lstm_size[0], configs=self.configs,num_hidden=self.lstm_size[1], filter_size=5,
                                  stride=1, padding=2)
        self.lstm2_norm = nn.LayerNorm([self.lstm_size[1], width // 2, width // 2])

        self.stride1 = nn.Conv2d(in_channels=self.lstm_size[1], out_channels=self.lstm_size[1], kernel_size=3, stride=2,
                                 padding=1)

        self.lstm3 = ConvLSTMCell(in_channel=self.lstm_size[1], configs=self.configs,num_hidden=self.lstm_size[2], filter_size=5,
                                  stride=1, padding=2)
        self.lstm3_norm = nn.LayerNorm([self.lstm_size[2], width // 4, width // 4])

        self.lstm4 = ConvLSTMCell(in_channel=self.lstm_size[2], configs=self.configs,num_hidden=self.lstm_size[3], filter_size=5,
                                  stride=1, padding=2)
        self.lstm4_norm = nn.LayerNorm([self.lstm_size[3], width // 4, width // 4])

        self.stride2 = nn.Conv2d(in_channels=self.lstm_size[3], out_channels=self.lstm_size[3], kernel_size=3, stride=2,
                                 padding=1)
        self.robotConcat = nn.Conv2d(in_channels=self.lstm_size[3] + self.POS_DIM + self.ACTION_DIM,
                                     out_channels=self.lstm_size[3], kernel_size=1, stride=1)

        self.lstm5 = ConvLSTMCell(in_channel=self.lstm_size[3], configs=self.configs,num_hidden=self.lstm_size[4], filter_size=5,
                                  stride=1, padding=2)
        self.lstm5_norm = nn.LayerNorm([self.lstm_size[4], width // 8, width // 8])

        self.deconv1 = nn.ConvTranspose2d(in_channels=self.lstm_size[4], out_channels=self.lstm_size[4], kernel_size=3,
                                          stride=2, output_padding=1, padding=1)

        self.lstm6 = ConvLSTMCell(in_channel=self.lstm_size[4],configs=self.configs, num_hidden=self.lstm_size[5], filter_size=5,
                                  stride=1, padding=2)
        self.lstm6_norm = nn.LayerNorm([self.lstm_size[5], width // 4, width // 4])

        self.deconv2 = nn.ConvTranspose2d(in_channels=self.lstm_size[5] + self.lstm_size[1],
                                          out_channels=self.lstm_size[5] + self.lstm_size[1], kernel_size=3, stride=2,
                                          output_padding=1, padding=1)

        self.lstm7 = ConvLSTMCell(in_channel=self.lstm_size[5] + self.lstm_size[1],configs=self.configs, num_hidden=self.lstm_size[6],
                                  filter_size=5, stride=1, padding=2)
        self.lstm7_norm = nn.LayerNorm([self.lstm_size[6], width // 2, width // 2])

        self.deconv3 = nn.ConvTranspose2d(in_channels=self.lstm_size[6] + self.lstm_size[0],
                                          out_channels=self.lstm_size[6], kernel_size=3, stride=2, output_padding=1,
                                          padding=1)
        self.deconv3_norm = nn.LayerNorm([self.lstm_size[6], width, width])

        self.fc = nn.Linear(int(self.lstm_size[4] * width * width / 64),
                            self.KERNEL_SIZE * self.KERNEL_SIZE * self.num_masks)
        self.maskout = nn.ConvTranspose2d(self.lstm_size[6], self.num_masks + 1, kernel_size=1, stride=1)
        self.deconv4 = nn.ConvTranspose2d(in_channels=self.lstm_size[6], out_channels=self.frame_channel, kernel_size=1,
                                          stride=1)

    def forward(self, frames, mask_true, action=None, init_pos=None):
        lstm_state1, lstm_state2, lstm_state3, lstm_state4, lstm_state5, lstm_state6, lstm_state7 = None, None, None, None, None, None, None
        next_frames = []

        for t in range(self.configs.total_length - 1):
            if t < self.configs.input_length:
                input_frame = frames[:, t]
            else:
                input_frame = mask_true[:, t - self.configs.input_length] * frames[:, t] + \
                              (1 - mask_true[:, t - self.configs.input_length]) * x_gen

            conv1 = self.conv1(input_frame)  # (b,3,64,64)->(b,32,32,32)
            conv1_norm = self.conv1_norm(conv1)

            lstm1, lstm_state1 = self.lstm1(conv1_norm, lstm_state1)  # (b,3,32,32)->(b,32,32,32)
            lstm1_norm = self.lstm1_norm(lstm1)

            lstm2, lstm_state2 = self.lstm2(lstm1_norm, lstm_state2)  # (b,32,32,32)->(b,32,32,32)
            lstm2_norm = self.lstm2_norm(lstm2)

            stride1 = self.stride1(lstm2_norm)  # (b,32,32,32)->(b,32,16,16)

            lstm3, lstm_state3 = self.lstm3(stride1, lstm_state3)  # (b,32,16,16)->(b,64,16,16)
            lstm3_norm = self.lstm3_norm(lstm3)

            lstm4, lstm_state4 = self.lstm4(lstm3_norm, lstm_state4)  # (b,64,16,16)->(b,64,16,16)
            lstm4_norm = self.lstm4_norm(lstm4)

            stride2 = self.stride2(lstm4_norm)  # (b,64,16,16)->(b,64,8,8)

            if self.is_robot_state_used:
                state_action = torch.cat([action, init_pos], dim=1)
                smear = torch.reshape(state_action, list(state_action.shape) + [1, 1])
                smear = smear.repeat(1, 1, 8, 8)
                stride2 = torch.cat([stride2, smear], dim=1)

            robotconcat = self.robotConcat(stride2)  # (b,64,8,8)

            lstm5, lstm_state5 = self.lstm5(robotconcat, lstm_state5)  # (b,64,8,8)->(b,128,8,8)
            lstm5_norm = self.lstm5_norm(lstm5)

            deconv1 = self.deconv1(lstm5_norm)  # (b,128,8,8)->(b,128,16,16)

            lstm6, lstm_state6 = self.lstm6(deconv1, lstm_state6)  # (b,128,16,16)->(b,64,16,16)
            lstm6_norm = self.lstm6_norm(lstm6)

            # skip connection
            lstm6_norm = torch.cat([lstm6_norm, torch.relu(stride1)], dim=1)  # (b,64+32,16,16)
            deconv2 = self.deconv2(lstm6_norm)  # (b,64+32,16,16)->(b,64+32,32,32)

            lstm7, lstm_state7 = self.lstm7(deconv2, lstm_state7)  # (b,64+32,32,32)->(b,32,32,32)
            lstm7_norm = self.lstm7_norm(lstm7)
            # skip connection
            lstm7_norm = torch.cat([lstm7_norm, conv1_norm], dim=1)  # (b,32+32,32,32)

            deconv3 = self.deconv3(lstm7_norm)  # (b,32+32,32,32)->(b,32,64,64)
            deconv3_norm = self.deconv3_norm(deconv3)

            maskout = self.maskout(torch.relu(deconv3_norm))  # (b,32,64,64)->(b,10+1,64,64)
            masks = torch.softmax(maskout, dim=1)
            mask_list = torch.split(masks, split_size_or_sections=1, dim=1)

            deconv4 = self.deconv4(torch.relu(deconv3_norm))
            transformed = [torch.sigmoid(deconv4)]

            kernel = lstm5_norm.view(lstm5_norm.shape[0], -1)
            transformed += self.cdna_transformation(input_frame, kernel)

            x_gen = mask_list[0] * input_frame
            for layer, mask in zip(transformed, mask_list[1:]):
                x_gen += mask * layer
            next_frames.append(x_gen)
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 2, 3, 4).contiguous()
        return next_frames

    def robot_process(robot_actions, robot_pos, shape=[8, 8]):
        '''
        description: Concat the robot_action vector and the robot_pos vector.

        param {robot_actions:list(shape=[batchbatch, 4]), robot_pos:list(shape=[batch, 3]), shape = [8, 8]}

        return {torch.tensor in shape of [batch_size, dim(robot_action)+dim(robot_pos), shape]}
        '''
        state_action = torch.cat([robot_actions, robot_pos], dim=1)
        smear = torch.reshape(state_action, list(state_action.shape) + [1, 1])
        smear = smear.repeat(1, 1, shape[0], shape[1])
        return smear

    def cdna_transformation(self, image, cdna_input):
        '''
        description:
        param {type}
        return {type}
        '''
        batch_size, height, width = image.shape[0], image.shape[2], image.shape[3]
        cdna_kerns = self.fc(cdna_input)
        cdna_kerns = cdna_kerns.view(batch_size, 10, 1, 5, 5)
        cdna_kerns = torch.relu(cdna_kerns - 1e-12) + 1e-12
        norm_factor = torch.sum(cdna_kerns, dim=[2, 3, 4], keepdim=True)
        cdna_kerns /= norm_factor

        cdna_kerns = cdna_kerns.view(batch_size * 10, 1, 5, 5)
        image = image.permute([1, 0, 2, 3])

        # image shape 3,8,h,w
        # kerns shape out,in kh,kw   80,1,5,5
        # out shape 3,80,h,w
        transformed = torch.conv2d(image, cdna_kerns, stride=1, padding=[2, 2], groups=batch_size)

        transformed = transformed.view(self.frame_channel, batch_size, 10, height, width)
        transformed = transformed.permute([1, 0, 3, 4, 2])
        transformed = torch.unbind(transformed, dim=-1)
        return transformed
