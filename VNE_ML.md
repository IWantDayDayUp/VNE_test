# 关于机器学习在VNE领域的应用

## 1. 张培颖

使用 `GCN` 与 `RL`

自定义适应性矩阵与适应性值, 减少资源碎片化问题

state:

- 可用cpu资源
- 可用带宽资源
- 虚拟节点cpu需求
- 虚拟链路带宽需求
- 已映射虚拟节点
- 已映射虚拟链路
- 未映射虚拟节点
- 未映射虚拟链路

action:

- 节点映射策略
- 链路映射策略

reward:

- $R(G_i)\bullet R/C$

## 2. Automatic Virtual Network Embedding - A DRL Approach with GCN

[Automatic Virtual Network Embedding - A DRL Approach with GCN](https://geminilight.cn/2020/07/13/RP%20-%20%E7%A7%91%E7%A0%94%E8%AE%BA%E6%96%87/paper-nfv-vne-dlr-gcn/)

## 3. 赵丽媛硕士论文 2020(Reinforcement Learning for Resource Mapping in 5G Network Slicing)

state:

- 节点资源能力CPU
- 链路带宽和
- 度
- 节点连通性: 到其他节点距离之和的倒数

reward: `node + link`

## 4. 李蒙 2020(没看到英文论文)

节点映射模块state:

- 节点资源能力CPU
- 链路带宽和
- 度
- 接近中心性(联通性)
- `到已映射节点的平均距离`
- `中介中心性`
- `特征向量中心性`

reward: `R/C`

`链路映射模块state`

- 带宽
- 中介中心性

## 5. 姚海鹏 2018(英文: A Novel Reinforcement Learning Algorithm for Virtual Network Embedding)

state:

- 节点资源能力CPU
- 度
- 带宽和
- `到已映射节点的平均距离`
