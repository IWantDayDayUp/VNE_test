import time
from substrate import Substrate
from maker import simulate_events_one
from analysis import Analysis

import os


def main():

    # Step1: 读取底层网络和虚拟网络请求文件
    basepath = os.path.dirname(__file__)
    network_files_dir = '/networks/'
    sub_filename = 'subts.txt'
    sub = Substrate(basepath + network_files_dir, sub_filename)
    event_queue1 = simulate_events_one(basepath + '/VNRequest/', 2000)

    # Step2: 选择映射算法
    algorithm = 'rl'
    arg = 10  # 训练轮次

    # Step3: 处理虚拟网络请求事件
    start = time.time()
    sub.handle(event_queue1, algorithm, arg)
    time_cost = time.time() - start
    print(time_cost)

    # Step4: 输出映射结果文件
    tool = Analysis()
    tool.save_result(sub, '%s-VNE-0409-%s.txt' % (algorithm, arg))


if __name__ == '__main__':
    main()
