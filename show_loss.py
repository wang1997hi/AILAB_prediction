import numpy as np
import matplotlib.pyplot as plt




loss1 = np.load(r'E:\pycharm\project\pred\experiments\Henan\PredRNN++\loss.npy')
plt.plot(np.arange(1, loss1.shape[0]+1)*0.2, loss1, label='PredRNN++')
# loss2 = np.load(r'E:\pycharm\project\pred\experiments\MOVING_MNIST\T-PredRNN++\loss.npy')
# plt.plot(np.arange(1, loss2.shape[0]+1)*0.2, loss2, label='Causal LSTM')
# loss3 = np.load(r'E:\pycharm\project\pred\experiments\MOVING_MNIST\PredRNN\loss.npy')
# plt.plot(np.arange(1, loss3.shape[0]+1)*0.2, loss3, label='PredRNN')
plt.xlabel('epoch')
plt.ylabel('MSE')
plt.legend()
plt.show()
