import torch
import matplotlib as mpl
import matplotlib.colors as colors
import numpy as np
import cv2
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from torch.nn.functional import binary_cross_entropy


def colormap():
    return mpl.colors.LinearSegmentedColormap.from_list('cmap', ['#FFFFFF', '#C0C0FE', '#7A72EE', '#1E26D0', '#A6FCA8',
                                                                 '#00EA00', '#10921A', '#FCF464', '#C8C802', '#8C8C00',
                                                                 '#FEACAC', '#FE645C', '#EE0230', '#D48EFE', '#AA24FA'],
                                                        80)


# 80.18015215089319
def get_MSE(pred, label):
    pred = np.maximum(pred, 0)
    pred = np.minimum(pred, 1)
    pred = pred.reshape(-1, 1)
    label = label.reshape(-1, 1)
    MSE = np.square(label - pred).sum()
    return MSE


def get_SSIM(pred, label):
    return structural_similarity(pred, label, data_range=1)


def get_PSNR(pred, label):
    return peak_signal_noise_ratio(pred, label, data_range=1)


def get_CSI(pred, label, threshold=20):
    width = pred.shape[0]
    CSI = 0
    FAR = 0
    POD = 0
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    for i in range(width):
        for j in range(width):
            if label[i, j] > threshold and pred[i, j] > threshold:
                TP += 1
            elif label[i, j] > threshold and pred[i, j] < threshold:
                FN += 1
            elif label[i, j] < threshold and pred[i, j] > threshold:
                FP += 1
            elif label[i, j] < threshold and pred[i, j] < threshold:
                TN += 1

    CSI += TP / (TP + FN + FP + 1e-6)
    POD += TP / (TP + FN + 1e-6)
    FAR += FP / (TP + FP + 1e-6)

    return CSI, POD, FAR


def gray2RGB(gray):
    zeros = np.zeros(256, np.dtype('uint8'))
    lut = np.dstack((zeros, zeros, zeros))
    gray = gray[:, :, np.newaxis]
    gray = np.concatenate((gray, gray, gray), axis=-1)
    for dbz in range(256):
        if dbz <= 0:
            lut[0, dbz] = [255, 255, 255]
        elif dbz <= 5:
            lut[0, dbz] = [192, 192, 254]
        elif dbz <= 10:
            lut[0, dbz] = [122, 114, 238]
        elif dbz <= 15:
            lut[0, dbz] = [30, 38, 208]
        elif dbz <= 20:
            lut[0, dbz] = [166, 252, 168]
        elif dbz <= 25:
            lut[0, dbz] = [0, 234, 0]
        elif dbz <= 30:
            lut[0, dbz] = [16, 146, 26]
        elif dbz <= 35:
            lut[0, dbz] = [252, 244, 100]
        elif dbz <= 40:
            lut[0, dbz] = [200, 200, 2]
        elif dbz <= 45:
            lut[0, dbz] = [140, 140, 0]
        elif dbz <= 50:
            lut[0, dbz] = [254, 172, 172]
        elif dbz <= 55:
            lut[0, dbz] = [254, 100, 92]
        elif dbz <= 60:
            lut[0, dbz] = [238, 2, 48]
        elif dbz <= 65:
            lut[0, dbz] = [212, 142, 254]
        elif dbz <= 70:
            lut[0, dbz] = [170, 36, 250]
    image_RGB = cv2.LUT(gray, lut)
    image_RGB = cv2.cvtColor(image_RGB, cv2.COLOR_BGR2RGB)
    return image_RGB

def preciptation(gray):
    zeros = np.zeros(256, np.dtype('uint8'))
    lut = np.dstack((zeros, zeros, zeros))  # 1,256,3
    gray = gray[:, :, np.newaxis]
    gray = np.concatenate((gray, gray, gray), axis=-1)  # h,w,3
    for dbz in range(256):
        if dbz <= 1:
            lut[0, dbz] = [255, 255, 255]
        elif dbz <= 10:
            lut[0, dbz] = [166, 242, 143]
        elif dbz <= 25:
            lut[0, dbz] = [61, 186, 61]
        elif dbz <= 50:
            lut[0, dbz] = [97, 184, 255]
        elif dbz <= 100:
            lut[0, dbz] = [0, 0, 255]
        elif dbz <= 250:
            lut[0, dbz] = [250, 0, 250]
        else:
            lut[0, dbz] = [128, 0, 64]
    image_RGB = cv2.LUT(gray, lut)
    image_RGB = cv2.cvtColor(image_RGB, cv2.COLOR_BGR2RGB)
    return image_RGB

def temperature(gray):
    zeros = np.zeros(256, np.dtype('uint8'))
    lut = np.dstack((zeros, zeros, zeros))  # 1,256,3
    gray = gray[:, :, np.newaxis]
    gray = np.concatenate((gray, gray, gray), axis=-1)  # h,w,3
    for dbz in range(-60,50,1):
        if dbz <= -30:
            lut[0, dbz] = [2, 12, 100]
        elif dbz <= -28:
            lut[0, dbz] = [7, 30, 120]
        elif dbz <= -26:
            lut[0, dbz] = [17, 49, 139]
        elif dbz <= -24:
            lut[0, dbz] = [27 , 68, 159]
        elif dbz <= -22:
            lut[0, dbz] = [38 , 87, 179]
        elif dbz <= -20:
            lut[0, dbz] = [48 , 106, 199]
        elif dbz <= -18:
            lut[0, dbz] = [59 , 126, 219]
        elif dbz <= -16:
            lut[0, dbz] = [78 , 138, 221]
        elif dbz <= -14:
            lut[0, dbz] = [97, 150, 224]
        elif dbz <= -12:
            lut[0, dbz] = [116, 163, 226]
        elif dbz <= -10:
            lut[0, dbz] = [135 , 175, 229]
        elif dbz <= -8:
            lut[0, dbz] = [155 , 188, 232]
        elif dbz <= -6:
            lut[0, dbz] = [154 , 196, 220]
        elif dbz <= -4:
            lut[0, dbz] = [153 , 205, 208]
        elif dbz <= -2:
            lut[0, dbz] = [152 , 214, 196]
        elif dbz <= 0:
            lut[0, dbz] = [151 , 232, 173]
        elif dbz <= 2:
            lut[0, dbz] = [215 , 222, 126]
        elif dbz <= 4:
            lut[0, dbz] = [234 , 219, 112]
        elif dbz <= 6:
            lut[0, dbz] = [244 , 217, 99]
        elif dbz <= 8:
            lut[0, dbz] = [250 , 204, 79]
        elif dbz <= 10:
            lut[0, dbz] = [247 , 180, 48]
        elif dbz <= 12:
            lut[0, dbz] = [242 , 155, 0]
        elif dbz <= 14:
            lut[0, dbz] = [241 , 147, 3]
        elif dbz <= 16:
            lut[0, dbz] = [240 , 132, 10]
        elif dbz <= 18:
            lut[0, dbz] = [239 , 117, 17]
        elif dbz <= 20:
            lut[0, dbz] = [238 , 102, 24]
        elif dbz <= 22:
            lut[0, dbz] = [238 , 88, 31]
        elif dbz <= 24:
            lut[0, dbz] = [231 , 75, 26]
        elif dbz <= 26:
            lut[0, dbz] = [224 , 63, 22]
        elif dbz <= 28:
            lut[0, dbz] = [217 , 51, 18]
        elif dbz <= 30:
            lut[0, dbz] = [208 , 36, 14]
        elif dbz <= 32:
            lut[0, dbz] = [194 , 0, 3]
        elif dbz <= 34:
            lut[0, dbz] = [187 , 1, 9]
        elif dbz <= 35:
            lut[0, dbz] = [169 , 2, 16]
        elif dbz <= 37:
            lut[0, dbz] = [138 , 5, 25]
        elif dbz <= 40:
            lut[0, dbz] = [111 , 0, 21]
        else:
            lut[0, dbz] = [80, 0, 15]
    image_RGB = cv2.LUT(gray, lut)
    image_RGB = cv2.cvtColor(image_RGB, cv2.COLOR_BGR2RGB)
    return image_RGB

def schedule_sampling(args,batch_size):
    zeros = np.zeros((batch_size,
                      args.total_length - args.input_length - 1,
                      args.patch_size ** 2 * args.img_channel,
                      args.img_width // args.patch_size,
                      args.img_width // args.patch_size))
    if not args.scheduled_sampling:
        return zeros

    if args.sampling_value > 0:
        args.sampling_value -= args.sampling_changing_rate

    random_flip = np.random.random_sample(
        (batch_size, args.total_length - args.input_length - 1))
    true_token = (random_flip < args.sampling_value)
    ones = np.ones((args.patch_size ** 2 * args.img_channel,
                    args.img_width // args.patch_size,
                    args.img_width // args.patch_size))
    zeros = np.zeros((args.patch_size ** 2 * args.img_channel,
                      args.img_width // args.patch_size,
                      args.img_width // args.patch_size))
    real_input_flag = []
    for i in range(batch_size):
        for j in range(args.total_length - args.input_length - 1):
            if true_token[i, j]:
                real_input_flag.append(ones)
            else:
                real_input_flag.append(zeros)
    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag,
                                 (batch_size,
                                  args.total_length - args.input_length - 1,
                                  args.patch_size ** 2 * args.img_channel,
                                  args.img_width // args.patch_size,
                                  args.img_width // args.patch_size))
    return real_input_flag

def max_pooling_forward(z, pooling=(2,2), strides=(2, 2)):
    """
    最大池化前向过程
    :param z: 卷积层矩阵,形状(N,C,H,W)，N为batch_size，C为通道数
    :param pooling: 池化大小(k1,k2)
    :param strides: 步长
    :param padding: 0填充
    :return:
    """
    N, C,H, W = z.shape

    # 输出的高度和宽度
    out_h = H // strides[0]
    out_w = W // strides[1]
    pool_z = np.zeros((N,C,out_h, out_w))
    for n in np.arange(N):
        for c in np.arange(C):
            for i in np.arange(out_h):
                for j in np.arange(out_w):
                    pool_z[n,c, i, j] = np.mean(z[n,c,strides[0] * i:strides[0] * i + pooling[0],strides[1] * j:strides[1] * j + pooling[1]])

    return pool_z