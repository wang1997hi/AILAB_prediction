import numpy as np
import torch
from preprocess import reshape_patch, reshape_patch_back
from utils import gray2RGB, get_MSE, get_CSI, get_SSIM, get_PSNR,preciptation,temperature
import cv2
import os
from skimage.measure import compare_ssim
from utils import schedule_sampling
import time


def down(img):
    maxpool = torch.nn.MaxPool2d(kernel_size=4, stride=4)
    # maxpool = torch.nn.AvgPool2d(4)
    img_list = []
    for i in range(img.shape[1]):
        img_list.append(maxpool(img[:, i]))
    result = torch.stack(img_list, dim=0).permute(1, 0, 2, 3, 4).contiguous()
    # result = result[:,:,0:1,0:400,0:400]
    return result


# 训练
def train(args, data_train, data_test, net, criterion, optimizer):
    # 训练过程中的损失值
    loss_list = []
    min_loss = 9999


    for epoch in range(args.max_iterations):
        # 进度
        batch = 0

        # 是否下采样
        for img in data_train:
            batch_size = img.shape[0]
            if args.is_down == 1:
                img = down(img)


            # 切割
            img = reshape_patch(img, args.patch_size)

            # 存到GPU,CPU
            img = img.to(args.device).float()
            # 梯度清零
            optimizer.zero_grad()

            mask_true = schedule_sampling(args,batch_size)
            mask_true = torch.Tensor(mask_true).to(args.device).float()

            # 前向计算
            out = net(img, mask_true)
            # 计算损失值
            loss = criterion[0](out, img[:, 1:args.total_length])
            # loss += criterion[1](out, img[:, 1:args.total_length])
            # print(loss)
            # 反向计算梯度
            loss.backward()
            # 更新参数
            optimizer.step()

            batch += args.batch_size
            # 每10个batch
            if batch % (args.batch_size *200) == 0:
                # 测试并保存,显示结果
                # 输出进度
                print("epoch:" + str(epoch) + ",batch:" + str(batch) + "," +
                      str(batch / data_train.dataset.__len__() * 100) + "%", end=" ")
                loss = test(args, data_test, net)
                print('min_loss='+str(min_loss))
                loss_list.append(loss)
                np.save(args.save_dir + '/loss.npy', loss_list)

                # 保存模型
                if min_loss > loss:
                    # torch.save(net.state_dict(), args.save_dir + '/model.pkl')
                    min_loss = loss
                    torch.save(net, args.save_dir + '/model.pth')


# 测试
def test(args, data, net):
    with torch.no_grad():
        MSE = np.zeros(args.output_length)
        CSI = np.zeros(args.output_length)
        POD = np.zeros(args.output_length)
        FAR = np.zeros(args.output_length)
        SSIM = np.zeros(args.output_length)
        PSNR = np.zeros(args.output_length)
        has_showed = 0
        for img in data:  # b,l,c,h,w
            batch_size = img.shape[0]

            # 是否下采样
            if args.is_down == 1:
                img = down(img)
            # 切割
            img = reshape_patch(img, args.patch_size)

            # 存到CPU或GPU
            img = img.to(args.device).float()

            # 计划采样
            mask_true = np.zeros((batch_size, args.output_length - 1, args.img_channel * args.patch_size ** 2,
                                 args.img_width // args.patch_size, args.img_width // args.patch_size))
            mask_true = torch.FloatTensor(mask_true).to(args.device)



            # 得到输出out
            out = net(img, mask_true)
            out = out[:, args.input_length - 1:]


            # 存到CPU
            out = out.cpu().detach().numpy()
            img = img.cpu().detach().numpy()

            # 还原切割
            out = reshape_patch_back(out, args.patch_size)
            img = reshape_patch_back(img, args.patch_size)

            # 可视化,测试frame-wise结果
            for i in range(args.output_length):
                for b in range(img.shape[0]):
                    # 评价指标
                    # CSI_one,POD_one,FAR_one = get_CSI(out[b, i, 0]*80, img[b, i+args.input_length, 0]*80,args.threshold)
                    # CSI[i] += CSI_one
                    # POD[i] += POD_one
                    # FAR[i] += FAR_one

                    # SSIM[i] += get_SSIM(out[b, i,0], img[b, i+args.input_length, 0])
                    # PSNR[i] += get_PSNR(out[b, i,0], img[b, i+args.input_length, 0])
                    MSE[i] += get_MSE(out[b, i, 0], img[b, i+args.input_length, 0])

                    # 可视化结果
                    if b < args.show_sample and has_showed == 0:

                        if os.path.exists(args.save_dir + '\\results\\' + str(b)) is False:
                            os.makedirs(args.save_dir + '\\results\\' + str(b))

                        # 预测
                        name = 'pd' + str(i + 1) + '.png'
                        file_name = args.save_dir + '\\results\\' + str(b) + '\\' + name
                        img_pd = out[b, i, 0, :, :]
                        img_pd = np.maximum(img_pd, 0)
                        img_pd = np.minimum(img_pd, 1)
                        img_pd = np.uint8(img_pd * args.normalized_coe)
                        # 是否彩图
                        # img_pd = gray2RGB(img_pd)
                        cv2.imwrite(file_name, img_pd)

                        # 真值
                        name = 'gt' + str(i + 1) + '.png'
                        file_name = args.save_dir + '\\results\\' + str(b) + "\\" + name
                        img_gt = img[b, i+args.input_length, 0, :, :]
                        img_gt = np.uint8(img_gt * args.normalized_coe)
                        # img_gt = gray2RGB(img_gt)
                        cv2.imwrite(file_name, img_gt)

                        # 输入
                        # name = 'img' + str(i + 1) + '.png'
                        # file_name = args.save_dir + '\\results\\' + str(b) + "\\" + name
                        # img_img = img[b, i, 0, :, :]
                        # img_img = np.uint8(img_img * args.normalized_coe)
                        # # img_img = gray2RGB(img_img)
                        # cv2.imwrite(file_name, img_img)
            has_showed = 1

        # 取平均值
        MSE = MSE / data.dataset.__len__()
        PSNR = PSNR / data.dataset.__len__()
        SSIM = SSIM / data.dataset.__len__()
        CSI = CSI / data.dataset.__len__()
        POD = POD / data.dataset.__len__()
        FAR = FAR / data.dataset.__len__()
        print("MSE:"+str(MSE.mean()))
        # print("PSNR:" + str(PSNR.mean()))
        # print("SSIM:" + str(SSIM.mean()))
        # print("CSI:" + str(CSI.mean()))
        # print("POD:" + str(POD.mean()))
        # print("FAR:" + str(FAR.mean()))
        # np.save((args.save_dir + '/CSI'+str(args.threshold)+'.npy'), CSI)
        # np.save((args.save_dir + '/POD'+str(args.threshold)+'.npy'), POD)
        # np.save((args.save_dir + '/FAR'+str(args.threshold)+'.npy'), FAR)
        np.save((args.save_dir + '/MSE.npy'), MSE)
        # np.save((args.save_dir + '/PSNR.npy'), PSNR)
        # np.save((args.save_dir + '/SSIM.npy'), SSIM)

        return MSE.mean()
