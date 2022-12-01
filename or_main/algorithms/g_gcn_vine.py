import os
import copy
import torch
import torch_geometric
import numpy as np

from or_main.common import utils
from or_main.common import config
from or_main.common.config import model_save_path
from or_main.algorithms.Agent_BaseLine import BaselineVNEAgent
from or_main.algorithms.model.A3C import A3C_Model
from or_main.main.a3c_gcn_train.vne_env_a3c_train import A3C_GCN_Action


class Action:
    """
    Action 类
    """
    def __init__(self):
        # 暂时未成功映射的VN
        self.vnrs_postponement = None
        
        # 成功映射的VN: (vn对象, 节点映射结果, 链路映射结果)
        self.vnrs_embedding = None
        
        # 节点映射阶段失败的VN数量
        self.num_node_embedding_fails = 0
        # 链路映射阶段失败的VN数量
        self.num_link_embedding_fails = 0

    def __str__(self):
        action_str = "[{0:2} VNR POST.] [{1:2} VNR EMBED.]".format(
            len(self.vnrs_postponement),
            len(self.vnrs_embedding),
        )

        return action_str


class A3C_GCN_VNEAgent(BaselineVNEAgent):
    
    # def __init__(
    #         self, logger, time_window_size, agent_type, type_of_virtual_node_ranking,
    #         allow_embedding_to_same_substrate_node, max_embedding_path_length
    # ):
    #     super(A3C_GCN_VNEAgent, self).__init__(
    #         logger, time_window_size, agent_type, type_of_virtual_node_ranking,
    #         allow_embedding_to_same_substrate_node, max_embedding_path_length
    #     )
    def __init__(self, logger, time_window_size, agent_type, agent_mode, type_of_virtual_node_ranking, 
                 allow_embedding_to_same_substrate_node, max_embedding_path_length):
        super().__init__(logger, time_window_size, agent_type, agent_mode, type_of_virtual_node_ranking, 
                         allow_embedding_to_same_substrate_node, max_embedding_path_length)
    
        self.initial_s_CPU = []
        self.initial_s_bandwidth = []
        
        self.count_node_mapping = 0
        self.action_count = 0
        
        # useless
        self.eligibility_trace = np.zeros(shape=(config.SUBSTRATE_NODES,))
        
        # TODO: 修改成GCN
        self.model = A3C_Model(
            chev_conv_state_dim=config.NUM_SUBSTRATE_FEATURES, action_dim=config.SUBSTRATE_NODES
        )
        
        # load model
        self.agent_mode = agent_mode
        self.new_model_path = os.path.join(model_save_path, "A3C_model_0421.pth")
        if self.agent_mode:
            # TODO: model path
            self.model.load_state_dict(torch.load(self.new_model_path))
    
    def save_model(self, ):
        """
        save model
        """
        if self.agent_mode:
            torch.save(self.model, self.new_model_path)

    def get_state_information(self, copied_substrate, vnr, already_embedding_s_nodes, current_v_node, current_v_cpu_demand, v_pending):
        """
        底层网络特征矩阵
        """
        # 初始资源
        s_cpu_capacity = copied_substrate.initial_s_cpu_capacity
        s_bandwidth_capacity = copied_substrate.initial_s_node_total_bandwidth

        # 每个底层节点目前CPU + 周围可用带宽和
        s_cpu_remaining = []
        s_bandwidth_remaining = []
        for s_node, s_node_data in copied_substrate.net.nodes(data=True):
            s_cpu_remaining.append(s_node_data['CPU'])

            total_node_bandwidth = 0.0
            for link_id in copied_substrate.net[s_node]:
                total_node_bandwidth += copied_substrate.net[s_node][link_id]['bandwidth']

            s_bandwidth_remaining.append(total_node_bandwidth)

        # 生成底层网络特征矩阵
        substrate_features = []
        substrate_features.append(s_cpu_capacity)
        substrate_features.append(s_bandwidth_capacity)
        substrate_features.append(s_cpu_remaining)
        substrate_features.append(s_bandwidth_remaining)
        substrate_features.append(already_embedding_s_nodes)  # 已映射的底层节点

        # Convert to the torch.tensor
        substrate_features = torch.tensor(substrate_features)
        substrate_features = torch.transpose(substrate_features, 0, 1)  # 转置T
        substrate_features = substrate_features.view(1, config.SUBSTRATE_NODES, config.NUM_SUBSTRATE_FEATURES)  # reshape
        # substrate_features.size() --> (100, 5)

        # 从networkX类型获取Data类型
        substrate_geometric_data = torch_geometric.utils.from_networkx(copied_substrate.net)

        vnr_features = []
        vnr_features.append(current_v_cpu_demand)
        vnr_features.append(sum((vnr.net[current_v_node][link_id]['bandwidth'] for link_id in vnr.net[current_v_node])))
        vnr_features.append(v_pending)  # 待定虚拟节点数量
        vnr_features = torch.tensor(vnr_features).view(1, 1, 3)
        # substrate_features.size() --> (100, 5)
        # vnr_features.size()) --> (1, 3)

        return substrate_features, substrate_geometric_data.edge_index, vnr_features

    def find_substrate_nodes(self, copied_substrate, vnr):
        """
        重写父类方法: 给定VN请求, 计算所有虚拟节点的映射结果
        """
        subset_S_per_v_node = {}  # 候选节点集合
        embedding_s_nodes = {}  # 节点映射结果
        already_embedding_s_nodes = []  # 已经被占用的底层节点集合
        
        # 底层节点被使用情况: 0 未使用, 1 已使用
        current_embedding_s_nodes = [0] * len(copied_substrate.net.nodes)

        # 虚拟节点排序
        sorted_v_nodes_with_node_ranking = utils.get_sorted_v_nodes_with_node_ranking(
            vnr=vnr, type_of_node_ranking=self.type_of_virtual_node_ranking
        )

        num_remain_v_node = 1

        for v_node_id, v_node_data, _ in sorted_v_nodes_with_node_ranking:

            v_cpu_demand = v_node_data['CPU']
            v_node_location = v_node_data['LOCATION']
            
            # 还未映射的虚拟节点数量(待定数量)
            v_pending = len(sorted_v_nodes_with_node_ranking) - num_remain_v_node

            # 候选底层节点集合
            subset_S_per_v_node[v_node_id] = utils.find_subset_S_for_virtual_node(
                copied_substrate, v_cpu_demand, v_node_location, already_embedding_s_nodes
            )

            # 获得网络tensor特征矩阵, 邻接矩阵, 虚拟网络特征矩阵
            substrate_features, edge_index, vnr_features = self.get_state_information(
                copied_substrate,
                vnr,
                current_embedding_s_nodes,
                v_node_id,
                v_cpu_demand,
                v_pending
            )

            # TODO: 修改成自己的模型
            selected_s_node_id = self.model.select_node(substrate_features, edge_index, vnr_features)

            if copied_substrate.net.nodes[selected_s_node_id]['CPU'] <= v_cpu_demand or \
                    selected_s_node_id in already_embedding_s_nodes:
                self.num_node_embedding_fails += 1
                msg = "VNR {0} REJECTED ({1}): 'no suitable SUBSTRATE NODE for nodal constraints: {2}' {3}".format(
                    vnr.id, self.num_node_embedding_fails, v_cpu_demand, vnr
                )
                self.logger.info("{0} {1}".format(utils.step_prefix(self.time_step), msg))
                
                return None

            # 更新节点映射结果
            assert selected_s_node_id != -1
            embedding_s_nodes[v_node_id] = (selected_s_node_id, v_cpu_demand)

            # 不允许共用
            if not self.allow_embedding_to_same_substrate_node:
                already_embedding_s_nodes.append(selected_s_node_id)

            # 更新底层网络副本的资源
            # !: 每成功映射一个节点就立马更新底层资源, 如果后面有个虚拟节点失败了, 怎么拿回已分配下去的资源???
            assert copied_substrate.net.nodes[selected_s_node_id]['CPU'] >= v_cpu_demand
            copied_substrate.net.nodes[selected_s_node_id]['CPU'] -= v_cpu_demand
            
            # 1: 该节点被占用  0: 没占用
            current_embedding_s_nodes[selected_s_node_id] = 1
            num_remain_v_node += 1

        return embedding_s_nodes

    def get_action(self, state):
        """
        根据当前state做出一个action
        
        state: 传入的当前环境的状态
        """
        self.time_step += 1

        # 构造一个行动类的对象
        action = Action()

        # 
        # if self.time_step < self.next_embedding_epoch:
        #     action.num_node_embedding_fails = self.num_node_embedding_fails
        #     action.num_link_embedding_fails = self.num_link_embedding_fails
        #     return action

        # 失败和成功的VN集合
        action.vnrs_postponement = {}
        action.vnrs_embedding = {}

        # 底层网络副本
        COPIED_SUBSTRATE = copy.deepcopy(state.substrate)
        # 当前时刻等待被映射的VN集合
        VNRs_COLLECTED = state.vnrs_collected

        self.embedding(VNRs_COLLECTED, COPIED_SUBSTRATE, action)

        assert len(action.vnrs_postponement) + len(action.vnrs_embedding) == len(VNRs_COLLECTED)

        # 下一个映射的窗口(将这一窗口里的VN按收益排序, 依次映射)
        self.next_embedding_epoch += self.time_window_size

        action.num_node_embedding_fails = self.num_node_embedding_fails
        action.num_link_embedding_fails = self.num_link_embedding_fails
        
        # TODO: loss
        # self.model.zero_grad()
        
        return action

