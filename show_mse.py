import numpy as np
import matplotlib.pyplot as plt

# MSE1 = np.load(r'E:\pycharm\project\pred\experiments\MOVING_MNIST\PredRNN++\MSE.npy')
# plt.plot(np.arange(1, 11), MSE1, marker='o',label='PredRNN++')
MSE2 = np.load(r'E:\pycharm\project\pred\experiments\MOVING_MNIST\T-PredRNN++\MSE.npy')
plt.plot(np.arange(1, 2), np.arange(1, 2), marker='o',label='TC-LSTM')
# MSE3 = np.load(r'E:\pycharm\project\pred\experiments\MOVING_MNIST\CDNA\MSE.npy')
# plt.plot(np.arange(1, 11), MSE3, marker='o',label='CDNA')
# MSE4 = np.load(r'E:\pycharm\project\pred\experiments\MOVING_MNIST\DFN\MSE.npy')
# plt.plot(np.arange(1, 11), MSE4, marker='o',label='DFN')
# MSE5 = np.load(r'E:\pycharm\project\pred\experiments\MOVING_MNIST\PredRNN\MSE.npy')
# plt.plot(np.arange(1, 11), MSE5, marker='o',label='PredRNN')
# print('predrnn++',MSE1.mean())
# print('TC-LSTM++',MSE2.mean())
# print('CDNA',MSE3.mean())
# print('DFN',MSE4.mean())
# print('predrnn',MSE5.mean())
# PSNR35 = np.load('E:/pycharm/project/pred/experiments/35/PSNR.npy')
# plt.plot(np.arange(1, 11), PSNR31, marker='o',label='PredRNN++ patch4',linestyle='--')
# plt.plot(np.arange(1, 11), PSNR32, marker='o',label='T-PredRNN++ 64-64')
# plt.plot(np.arange(1, 11), PSNR33, marker='o',label='maxpool and transpose_conv')
# plt.plot(np.arange(1, 11), PSNR34, marker='o',label='pool and interpolation')
# plt.plot(np.arange(1, 11), PSNR35, marker='o',label='conv and interpolation')

# threshold = 30
# metric = 'POD'
# for threshold in (10,20,30):
#     for metric in ('CSI','POD','FAR'):
#         plt.cla()
#         # model20 = np.load('E:/pycharm/project/pred/experiments/10/'+metric+str(threshold)+'.npy')      # deep flow
#         model11 = np.load('E:/pycharm/project/pred/experiments/11/'+metric+str(threshold)+'.npy')    # PredRNN++
#         model14 = np.load('E:/pycharm/project/pred/experiments/14/'+metric+str(threshold)+'.npy')    # T-Predrnn++
#         # model15 = np.load('E:/pycharm/project/pred/experiments/15/'+metric+str(threshold)+'.npy')    # PCA flow
#
#         # plt.plot(np.arange(1, 11), model20, marker='o',label='Deep flow',linestyle='--')
#         plt.plot(np.arange(1, 11), model11, marker='o',label='PredRNN++',linestyle='--')
#         plt.plot(np.arange(1, 11), model14, marker='o',label='T-PredRNN++')
#         # plt.plot(np.arange(1, 11), model15, marker='o',label='PCA Flow',linestyle='--')
#
#         plt.legend()
#         plt.ylabel(metric)
#         plt.xlabel('step')
#         plt.ylim(0,1)
#         plt.title('threshold='+str(threshold))
#         fig = plt.gcf()
#         fig.savefig('Jiangsu_'+metric+str(threshold)+'.eps')
plt.xlabel('step')
plt.ylabel('MSE')
plt.legend()
# plt.show()
plt.savefig('temp.pdf')
plt.show()