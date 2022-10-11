# Graph Convolutional Networks

## 1. Graph Convolutional Networks

![Multi-layer Graph Convolutional Network (GCN) with first-order filters.](https://tkipf.github.io/graph-convolutional-networks/images/gcn_web.png)

[推荐链接: GRAPH CONVOLUTIONAL NETWORKS](https://tkipf.github.io/graph-convolutional-networks/)

The goal of **Graph Convolutional Networks(GCNs)** is to learn a function of signals/features on a Graph $G = (V, E)$ which takes as input:

- A feature description $x_i$ for every node $i$, summarized in a $N × D$ feature matrix $X$ ($N$: number of nodes, $D$: number of input features)
- A representative description of the graph structure in matrix form; typically in the form of an adjacency matrix $A$ (or some function thereof)

and produces a node-level output $Z$ (an $N × F$ feature matrix, where $F$ is the number of output features per node). Graph-level outputs can be modeled by introducing some form of pooling operation.

As an example, let's consider the following very simple form of a layer-wise propagation rule:

$$f(H^{(l)}, A) = \sigma (AH^{(l)}W^{(l)})$$

where $W^{(l)}$ is a weight matrix for the $l$-th neural network layer and $\sigma(\cdot)$ is a non-linear activation function like the **ReLU**.

## 2. Graph Attention Networks

[相关论文: Graph Attention Networks](https://arxiv.org/abs/1710.10903)

图卷积网络(GCN)告诉我们将 `局部` 的图结构和节点特征结合可以在节点分类任务中获得不错的表现

但, 美中不足的是GCN结合邻近节点特征的方式和图的结构依依相关, 这局限了训练所得模型在其他图结构上的泛化能力

图注意力机制(GAT)提出了用注意力机制对邻近节点特征加权求和, 邻近节点特征的权重完全取决于节点特征, 独立于图结构
