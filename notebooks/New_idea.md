# New Idea

## Anomal-E: A self-supervised network intrusion detection system based on graph neural networks(SCI 一区)

1. 文中提到, 传统的入侵检测都使用的是 "节点特征", 并没有利用 "链路特征", 导致检测准确率的下降
2. 传统的入侵检测都使用 "监督学习", 但现实情况下并没有那么多的带标签的数据集

所以, 作者提出:

1. 改进GCN, 由原来聚合周围节点的特征, 改为聚合周围链路的特征
2. 使用DGI算法作 "自监督学习"

启发:

1. 提供了另外一种学习 "链路特征" 的方式(即改进GCN), 那么能否使用两个GCN分别提取节点和链路的特征, 然后再拼接?? 这个做法区别于 "Co-embedding of Nodes and Edges with Graph Neural Networks"
2. DGI是自监督学习, 而VNE节点映射也是没有标准数据集, 能否嫁接过去? (有人使用交叉熵, 是否就是另类的DGI?)

## Co-embedding of Nodes and Edges with Graph Neural Networks

现有的基于GCN的算法都是基于 "节点特征" 的, 却忽略 "链路特征"

受 图理论中 "LINE Graph" 的启发, 作者提出了一种新的框架:

1. 原有的GCN需要使用两个矩阵: 节点特征矩阵, 节点邻接矩阵, 现在交换节点和链路的角色, 生成两个新的矩阵: 链路特征矩阵和链路邻接矩阵
2. 使用序列模型, 交替更新网络的 node embedding 和 egde embedding, 以此来同时学习节点和链路的特征表示

启发:

1. 是否可以使用两个GCN, 同样交换节点和链路的角色, 分别提取节点和链路的特征(并联), 然后再拼接成最终的特征的表示(这里需要注意维度, 链路数量比节点数量要多, 需要降维), 但是并联结构无法像序列结构那样体现节点和链路特征之间的联系(还是觉得序列结构好一点)

## Resource Fragmentation-Aware Embedding in Dynamic Network Virtualization Environments

1. 分别定义了节点和链路的 "RFD(资源碎片度)矩阵"
2. 一个节点(或链路)的资源碎片状态, 可以用它周围节点或链路的剩余资源比例表示

启发:

1. 关键词: "周围节点或链路", 那是否可以使用 GCN 来聚合节点和链路的剩余资源信息?
2. 如何使用 GCN 聚合周围链路的剩余资源状态(暂时: 改进GCN 或者 交换节点和链路角色)
3. 论文中说的是最短路径的最小带宽, 如果采用聚合的方式, 得到的是求和, 如何体现最小带宽资源?

## Optical Network Traffic Prediction Based on Graph Convolutional Neural Networks

1. 使用 GCN 和 GRU 串联, 捕捉流量的时空依赖性, 并应用到多节点的时间序列预测

启发:

1. 是否可以将这一整个结构包装成一个 "生成器", 应用到 GAN 结构里, 进行预测?

## Learning Graph Embedding With Adversarial Training Methods

启发:

1. 文中提到了能应用到 "聚类任务", 是否可以改进师兄的 k-means 算法??

## 额外人参果
