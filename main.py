from util import config
from data_loader import predata
import numpy as np
from model.LVne import NetModel as LModel
from network.LNetModel import LNet

import torch

# 测试函数-用于测试数据
# test.test_func()

def read_data():
    """
    solution:
    SN_Link:(s1,s2,sbandwidth)
    SN_Node:(snode)
    VN_Link:(v1,v2,vbandwidth)
    VN_Node:(vnode)
    VN_Life:{时序key:[[序号num,生命周期period,开始时间start_time,结束时间end_time],]}
    :return:
    """
    (solution, SN_Link, SN_Node, VN_Link, VN_Node, VN_Life) = predata.read_SN_VN(config.SnFile, config.VnFile)
    return (solution, SN_Link, SN_Node, VN_Link, VN_Node, VN_Life)

def run():
    # 读取数据e
    (solution, SN_Link, SN_Node, VN_Link, VN_Node, VN_Life) = predata.read_SN_VN(config.SnFile, config.VnFile)

    # model = LModel(solution, SN_Link, SN_Node, VN_Link, VN_Node, VN_Life)

    # data = {
    #     "SN_Link": SN_Link,
    #     "SN_Node": SN_Node,
    #     "VN_Node": VN_Node,
    #     "VN_Link": VN_Link,
    #     "VN_Life": VN_Life,
    #     "solution": solution
    # }

    # net = LNet(nhidden=128, batch_size=10, nembedding=128, dropout=.5)

    # model.experience(net, data)
    # model.experience(data)  # 原版的
    
    for i in range(10):
        print(VN_Life[0][i])

if __name__ == "__main__":
    run()