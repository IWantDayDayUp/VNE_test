# -*- coding: utf-8 -*-

"""
配置文件
"""

import os

###################################################################
# 数据文件
FILENAME = "VNEModel"
SnFile = "d:/VNE_TEST/data/default/maprecord.txt" # 物理网络数据
VnFile = "d:/VNE_TEST/data/default/virtualnetworkTP.txt" # 虚拟网络数据
ResultFile = "d:/VNE_TEST/data/result" # 结果文件
DATAFILE = "d:/VNE_TEST/data/" # 文件

# print(DATAFILE)
###################################################################
# 设置时间，理论上时间长度为无穷

time_unit = 1

TIMES = 200000
STEP = 1
###################################################################
Max_Jump = 6
Min_Jump = 2
INF = 9999999999
###################################################################
# 是否全部测试
full_request = True
FULL_REQUEST = True
# 随机训练次数
iter_time = 200
ITER_TIMES = 200
# 抽样数量
batch_size = 60
BATCH_SIZE = 60
###################################################################
# 是否开启多对一模式
MANY_TO_ONE = False
###################################################################
# 是否开启GPU模式
IS_GPU = True
###################################################################
import numpy as np
# 是否开启平滑
IS_FLOODING = False
FLOODING_LOSS = 0.1 # (0.1~0.5)
###################################################################
# 尽量充满SN网络
max_try_numbers = 1
MAX_TRY_NUMBERS = 1
# 充满后每次训练的VN个数
new_try_numbers = 1
NEW_TRY_NUMBERS = 1
###################################################################
IS_GCN = False
IS_GAT = True
###################################################################
NCLASS = 3
###################################################################
TEST = False