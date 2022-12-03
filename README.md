# 虚拟网络嵌入

## 0. 模型框架(原作者的工作)

一种基于指针网络模型的虚拟网络嵌入方法

通过指针网络模型（Encoder-Decoder）求解组合优化问题，并在编码层引入self-attention"# VirtualNetworkEmbedding"

torch版本号: 1.7.1, 下载地址: <https://download.pytorch.org/whl/torch_stable.html>

```cmd
pip install -r requirements.txt

<!-- PyTorch Geometric (for pytorch 1.7.0, 1.7.1, ...) -->

pip install --no-index torch-scatter -f <https://pytorch-geometric.com/whl/torch-1.7.0+cpu.html>
pip install --no-index torch-sparse -f <https://pytorch-geometric.com/whl/torch-1.7.0+cpu.html>
pip install --no-index torch-cluster -f <https://pytorch-geometric.com/whl/torch-1.7.0+cpu.html>
pip install --no-index torch-spline-conv -f <https://pytorch-geometric.com/whl/torch-1.7.0+cpu.html>
pip install torch-geometric
```

直接在model里面运行

## 1. 文件功能描述

* data_loader: 用于数据导入以及预处理
* main: 用于存放训练、测试、预测代码文件的入口
* model: 存放神经网络模型或者虚拟网络嵌入框架
* util: 用于存放一些功能文件，比如配置文件，日志文件，功能函数文件
* main.py: 作为程序启动入口

## 2. 总结

作者提出的方法(基于编码器), 看不懂

作者这里提供的 GR-VNE 和 GR-VNE2, 也不是我想找的 "基于GCN的VNE", 看样子还是基于编码器的, 看不懂目前

但VNE这个流程的实现, 还是可以节点的, 可以仿照这个项目总结出自己的一套框架, 为以后新的论文提供方便
