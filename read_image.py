from PIL import Image
import numpy as np
import os
from pycwr.io import read_auto
import matplotlib.pyplot as plt
from pycwr.draw.RadarPlot import Graph, GraphMap
import cv2
from utils import gray2RGB, get_MSE, get_CSI


def bz2_to_numpy(file):
    PRD = read_auto(file)
    fig, ax = plt.subplots()
    graph = Graph(PRD)
    graph.plot_crf(ax)
    product = graph.Radar.product
    CR = product['CR'].values
    CR = np.where(np.isnan(CR), 0, CR)
    return CR


def numpy_to_png(CR):
    print(CR[200:250, 200:250])
    file_name = 'pycwr_test.png'
    img_gt = np.uint8(CR)
    img_gt = gray2RGB(img_gt)
    cv2.imwrite(file_name, img_gt)


def generate_input(bz2_dir, numpy_dir):
    data_all = None
    count = 0
    if os.path.exists(numpy_dir) is False:
        os.makedirs(numpy_dir)
    for root, dirs, files in os.walk(bz2_dir):
        if files:
            for file in files:
                data = bz2_to_numpy(bz2_dir + "/" + file)
                data = data[np.newaxis, :, :, np.newaxis]
                if data_all is None:
                    data_all = data
                else:
                    data_all = np.concatenate((data_all, data), axis=0)  # sqe,batch,h,w
            np.save(numpy_dir + "/input.npy", data_all)


generate_input(r"D:\bz2_dir", r"D:\numpy_dir")


def video_to_numpy(dir_path, out_path, window_width=20, stride=10):
    all_video_numpy = []
    for root, dirs, files in os.walk(dir_path):
        if files:
            print(files)
            for file in files:
                one_video_numpy = getFrame(os.path.join(root, file))
                one_video_len = one_video_numpy.shape[0]
                for i in range(0, one_video_len - window_width + 1, stride):
                    one_squ_numpy = one_video_numpy[i:i + window_width]
                    all_video_numpy.append(one_squ_numpy)
                print(len(all_video_numpy))
    all_video_numpy = np.stack(all_video_numpy, axis=1)
    np.save(out_path, all_video_numpy)


def getFrame(videoPath):
    cap = cv2.VideoCapture(videoPath)
    numFrame = 0
    rval = cap.isOpened()
    one_video_numpy = []
    while rval:
        numFrame += 1
        rval, frame = cap.read()
        if rval:
            frame = cv2.resize(frame, (128, 128))
            one_video_numpy.append(frame[:, :, 0])
        else:
            break
    return np.stack(one_video_numpy)  # sqe,h,w


def video2dataset(indice_path, video_dir, output_path, window_width=20, stride=5):
    f = open(indice_path, "r")
    txt_lines = f.readlines()
    dataset = []
    for line in txt_lines: # 一行数据
        name_indice = line.split('\t')
        if name_indice[0].isspace() is False:
            filename = name_indice[0].strip() + '_uncomp.avi'
            print(filename)
            indice_list = name_indice[-1].split(',')
            video_numpy = getFrame(os.path.join(video_dir, filename))
            for indice in indice_list: # 一个视频片段
                begin = int(indice.split('-')[0]) - 1
                end = int(indice.split('-')[1])
                if end > video_numpy.shape[0]:
                    end = video_numpy.shape[0]
                sub_video_numpy = video_numpy[begin:end]
                sub_video_len = end - begin
                for i in range(0, sub_video_len - window_width + 1, stride): # 窗口滑动
                    one_squ_numpy = sub_video_numpy[i:i + window_width]
                    if one_squ_numpy.shape[0] != 20:
                        print(one_squ_numpy.shape)
                    dataset.append(one_squ_numpy)
                # print(len(dataset))
    all_video_numpy = np.stack(dataset, axis=1)
    np.save(output_path, all_video_numpy)


video2dataset(r'D:\KTH\train_indice.txt', r'D:\KTH\train', r'D:\KTH\train.npy')
