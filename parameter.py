import argparse
"""
set parameter
"""
def get_args():
    parser = argparse.ArgumentParser(description='PyTorch video prediction')

    # -------------------------------以下参数系统请手动配置---------------------------------------------------------------
    parser.add_argument('--save_dir', type=str, default=r'experiments\MOVING_MNIST\TC-LSTM')    # 本次实验路径
    parser.add_argument('--is_training', type=int, default=0)                                   # 训练or测试
    parser.add_argument('--pre_trained', type=int, default=0)                                   # 继续训练
    parser.add_argument('--device', type=str, default='cuda')                                   # gpu or cpu
    parser.add_argument('--show_sample', type=int, default=30)                                  # 展示的个例数
    parser.add_argument('--threshold', type=int, default=20)                                    # 评价指标CSI的阈值

    # optimization
    parser.add_argument('--lr', type=float, default=0.001)                                      # 学习率
    parser.add_argument('--batch_size', type=int, default=8)                                    # 批处理尺寸
    parser.add_argument('--patch_size', type=int, default=1)                                    # patch下采样
    parser.add_argument('--max_iterations', type=int, default=20)                               # 学习轮数

    # scheduled sampling
    parser.add_argument('--scheduled_sampling', type=int, default=0)                            # 是否计划采样
    parser.add_argument('--sampling_value', type=float, default=1)                              # 计划采样起始值
    parser.add_argument('--sampling_changing_rate', type=float, default=0.000015)               # 每代减小值

    # -------------------------------以下参数系统自动配置----------------------------------------------------------------
    # data
    parser.add_argument('--is_down', type=int, default=0)                                       # 数据是否需要额外下采样
    parser.add_argument('--input_length', type=int, default=10)                                 # 序列输入长度
    parser.add_argument('--output_length', type=int, default=10)                                # 序列输出长度
    parser.add_argument('--total_length', type=int, default=20)                                 # 序列总长度
    parser.add_argument('--img_width', type=int, default=100)                                    # 数据空间尺寸
    parser.add_argument('--img_channel', type=int, default=1)                                   # 图像通道
    parser.add_argument('--normalized_coe', type=int, default=40)                              # 雷达80  图像255

    args = parser.parse_args()
    return args
