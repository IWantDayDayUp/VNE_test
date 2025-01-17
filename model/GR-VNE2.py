import torch

from util import utils, config
from data_loader import predata
from model.base import BaseModel
import pandas as pd
from network.GRNetModel2 import GRNet
from network.gcn import GCN


class NetModel(BaseModel):
    def __init__(self, solution, SN_Link, SN_Node, VN_Link, VN_Node, VN_Life):
        """
        :param solution: 求解
        :param SN_Link: 底层链路/三元组(si,sj,bandwidth)
        :param SN_Node: 底层结点/cpu可利用资源
        :param VN_Link: 虚拟链路
        :param VN_Node: 虚拟节点
        :param VN_Life: 虚拟请求生存时间
        """
        super().__init__(solution, SN_Link, SN_Node, VN_Link, VN_Node, VN_Life)

    def get_solution(self, net, s_nodes, v_nodes, s_links, v_links,
                     original_s_nodes, original_s_links, batch_size=10,
                     dropout=.5, alpha=.5, iter_time=config.ITER_TIMES, b=-1, gamma=0.9, eps=1e-6,longterm_rc=0.5):
        """
        :param s_nodes:
        :param v_nodes:
        :param s_links:
        :param v_links:
        :param args:
        :param kwargs:
        :return:
        """

        device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
        if config.IS_GPU:
            net.to(device=device)

        best_mapping_solution = {
            'code': False,
            'mapping_nodes': [],
            'mapping_links': dict(),
            'link_consumption': config.INF
        }

        ceiterion = torch.optim.Adadelta(net.parameters(), lr=1e-4, eps=1e-2)
        # ceiterion = torch.optim.SGD(net.parameters(), lr=1e-4, momentum=.912, weight_decay=0)

        for i in range(iter_time):
            node_mapping_solutions, shuffled_node_mapping_solutions, output_weights = net.get_node_mapping(
                s_nodes=s_nodes,
                s_links=s_links,
                v_nodes=v_nodes,
                v_links=v_links,
            )

            # 检测node mapping solutions是否符合，若符合则进行链路映射
            embedding_successes, link_mapping_solutions, link_consumptions, hops = utils.get_hops_and_link_consumptions(
                s_nodes=s_nodes,
                s_links=s_links,
                origin_v_nodes=v_nodes,
                origin_v_links=v_links,
                origin_node_mapping=node_mapping_solutions
            )
            if config.IS_GPU:
                link_consumptions = link_consumptions.to(device=device)

            link_consumptions = link_consumptions - link_consumptions.mean()

            # 记录下最优
            j = torch.argmin(link_consumptions)
            if link_consumptions[j] < best_mapping_solution['link_consumption']:
                best_mapping_solution['mapping_nodes'] = node_mapping_solutions[j]
                best_mapping_solution['mapping_links'] = link_mapping_solutions[j]
                best_mapping_solution['link_consumption'] = link_consumptions[j]
                best_mapping_solution['code'] = embedding_successes[j]

            if b == -1:
                b = link_consumptions.min()

            link_consumptions = (1 - longterm_rc + eps) * link_consumptions
            # link_consumptions = 1/(node_ut+link_ut+eps)*link_consumptions
            reward = (b - link_consumptions)
            if config.IS_GPU:
                reward = reward.cuda()
            # 计算loss
            cross_entropy_loss = net.get_CrossEntropyLoss(output_weights, shuffled_node_mapping_solutions)
            loss = torch.dot(cross_entropy_loss,reward)

            if config.IS_FLOODING:
                b_loss = config.FLOODING_LOSS
                loss = (loss - b_loss).abs() + b_loss

            net.zero_grad()
            loss.backward(retain_graph=True)
            ceiterion.step()

            b = b * alpha + (1 - alpha) * link_consumptions.mean()

        return {
            'net': net,
            'best_mapping_solution': best_mapping_solution,
            'b': b
        }

def run():
    # 读取数据
    (solution, SN_Link, SN_Node, VN_Link, VN_Node, VN_Life) = predata.read_SN_VN(config.SnFile, config.VnFile)

    model = NetModel(solution, SN_Link, SN_Node, VN_Link, VN_Node, VN_Life)

    data = {
        "SN_Link": SN_Link,
        "SN_Node": SN_Node,
        "VN_Node": VN_Node,
        "VN_Link": VN_Link,
        "VN_Life": VN_Life,
        "solution": solution
    }
    # 网络模型
    net = GRNet(hidden_dim=128, batch_size=config.BATCH_SIZE, embedding_dim=128, dropout=.2)
    print(net)
    model.experience(net, data, load_model_path="GR-VNE2", full_request=config.FULL_REQUEST)

# run()