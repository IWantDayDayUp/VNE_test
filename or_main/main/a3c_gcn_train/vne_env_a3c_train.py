import copy

import gym
import networkx as nx
import torch
import torch_geometric

from or_main.common import config, utils
from or_main.environments.vne_env import Substrate, VNR


class A3C_GCN_State:
    def __init__(self, substrate_features, substrate_edge_index, current_v_node, vnr_features):
        self.substrate_features = substrate_features
        self.substrate_edge_index = substrate_edge_index
        self.current_v_node = current_v_node
        self.vnr_features = vnr_features

    def __str__(self):
        substrate_features_str = str(self.substrate_features)
        substrate_edge_index_str = str(self.substrate_edge_index)
        current_v_node_str = str(self.current_v_node)
        vnr_features_str = str(self.vnr_features)

        state_str = " ".join([substrate_features_str, substrate_edge_index_str, current_v_node_str, vnr_features_str])

        return state_str

    def __repr__(self):
        return self.__str__()


class A3C_GCN_Action:
    """
    A3C动作类: 将虚拟节点n_v映射到底层节点n_s上
    """
    
    def __init__(self):
        self.v_node = None
        self.s_node = None

    def __str__(self):
        action_str = "[V_NODE {0:2}] [S_NODE {1:2}]".format(
            self.v_node, self.s_node
        )

        return action_str


class A3C_GCN_TRAIN_VNEEnvironment(gym.Env):
    
    def __init__(self, logger):
        self.logger = logger

        self.time_step = None
        self.episode_reward = None
        self.num_reset = 0

        # 相关评价指标
        self.revenue = None
        self.acceptance_ratio = None
        self.rc_ratio = None
        self.link_embedding_fails_against_total_fails_ratio = None

        # VN总数, 成功总数
        self.total_arrival_vnrs = 0
        self.total_embedded_vnrs = 0

        # 底层网络
        self.substrate = Substrate()
        # self.copied_substrate = copy.deepcopy(self.substrate)
        
        self.vnrs = []  # 虚拟网络集合
        self.vnr_idx = 0
        self.vnr = None
        self.v_node_embedding_success = []
        self.vnr_embedding_success_count = []
        self.already_embedded_v_nodes = []
        self.embedding_s_nodes = None

        # 上一时刻的收益与成本
        self.previous_step_revenue = None
        self.previous_step_cost = None

        self.egb_trace = None
        self.decay_factor_for_egb_trace = 0.99

        self.current_embedding = None
        self.sorted_v_nodes = None
        self.current_v_node = None
        self.current_v_cpu_demand = None

    def reset(self):
        """
        环境重置
        """
        self.time_step = 0

        self.episode_reward = 0.0

        self.revenue = 0.0
        self.acceptance_ratio = 0.0
        self.rc_ratio = 0.0
        self.link_embedding_fails_against_total_fails_ratio = 0.0

        self.num_reset += 1
        # if self.num_reset % 1 == 0:
        # self.substrate = copy.deepcopy(self.copied_substrate)
        self.substrate = Substrate()
        self.vnr_idx = 0
        self.vnrs = []
        for idx in range(config.NUM_VNR_FOR_TRAIN):
            self.vnrs.append(
                VNR(
                    id=idx,
                    vnr_duration_mean_rate=config.VNR_DURATION_MEAN_RATE,
                    delay=config.VNR_DELAY,
                    time_step_arrival=0
                )
            )
        # self.vnrs = sorted(
        #     self.vnrs, key=lambda vnr: vnr.revenue, reverse=True
        # )
        self.vnr = self.vnrs[self.vnr_idx]
        self.v_node_embedding_success = []
        self.vnr_embedding_success_count = []

        self.already_embedded_v_nodes = []

        self.embedding_s_nodes = {}
        self.num_processed_v_nodes = 0
        self.previous_step_revenue = 0.0
        self.previous_step_cost = 0.0

        self.egb_trace = [1] * len(self.substrate.net.nodes)
        self.current_embedding = [0] * len(self.substrate.net.nodes)

        self.sorted_v_nodes = utils.get_sorted_v_nodes_with_node_ranking(
            vnr=self.vnr, type_of_node_ranking=config.TYPE_OF_VIRTUAL_NODE_RANKING.TYPE_2
        )

        self.current_v_node, current_v_node_data, _ = self.sorted_v_nodes[self.num_processed_v_nodes]
        self.current_v_cpu_demand = current_v_node_data['CPU']

        substrate_features, substrate_edge_index, vnr_features = self.get_state_information(
            self.current_v_node, self.current_v_cpu_demand
        )
        initial_state = A3C_GCN_State(substrate_features, substrate_edge_index, self.current_v_node, vnr_features)

        return initial_state

    def step(self, action: A3C_GCN_Action):
        """
        执行action, 返回(下一状态, reward, 是否结束, (收益, 接受率, R/C, 链路失败率))
        """
        self.time_step += 1
        
        # 当前VN
        self.vnr = self.vnrs[self.vnr_idx]

        embedding_success = True
        v_cpu_demand = None

        node_embedding_fail_conditions = [
            self.substrate.net.nodes[action.s_node]['CPU'] < self.vnr.net.nodes[action.v_node]['CPU'],
            self.current_embedding[action.s_node] == 1
        ]

        sum_v_bandwidth_demand = 0.0  # for r_c calculation
        sum_s_bandwidth_embedded = 0.0  # for r_c calculation

        # 资源约束不满足 或 当前底层节点已经被占用
        if any(node_embedding_fail_conditions):
            embedding_success = False
        else:
            # Success for node embedding
            # 当前虚拟节点的需求
            v_cpu_demand = self.vnr.net.nodes[action.v_node]['CPU']
            # 更新映射关系
            self.embedding_s_nodes[action.v_node] = action.s_node

            # 链路映射: Start to try link embedding
            for already_embedded_v_node in self.already_embedded_v_nodes:
                if self.vnr.net.has_edge(action.v_node, already_embedded_v_node):
                    # 邻接虚拟链路需求
                    v_bandwidth_demand = self.vnr.net[action.v_node][already_embedded_v_node]['bandwidth']
                    sum_v_bandwidth_demand += v_bandwidth_demand

                    # 将满足链路需求的底层链路挑出来
                    subnet = nx.subgraph_view(
                        self.substrate.net,
                        filter_edge=lambda node_1_id, node_2_id: \
                            True if self.substrate.net.edges[(node_1_id, node_2_id)]['bandwidth'] >= v_bandwidth_demand else False
                    )

                    # 该虚拟链路的两个端节点
                    src_s_node = self.embedding_s_nodes[already_embedded_v_node]
                    dst_s_node = self.embedding_s_nodes[action.v_node]
                    
                    # 寻找虚拟路径
                    if len(subnet.edges) == 0 or not nx.has_path(subnet, source=src_s_node, target=dst_s_node):
                        # 不存在虚拟路径
                        embedding_success = False
                        del self.embedding_s_nodes[action.v_node]
                        break
                    else:
                        # 找出10条最短路径
                        MAX_K = 10
                        shortest_s_path = utils.k_shortest_paths(subnet, source=src_s_node, target=dst_s_node, k=MAX_K)[0]
                        
                        # 最短路径大于最大映射长度
                        if len(shortest_s_path) > config.MAX_EMBEDDING_PATH_LENGTH:
                            embedding_success = False
                            break
                        else:
                            # 成功找到一条底层链路
                            s_links_in_path = []
                            for node_idx in range(len(shortest_s_path) - 1):
                                s_links_in_path.append((shortest_s_path[node_idx], shortest_s_path[node_idx + 1]))
                            # 更新底层资源
                            for s_link in s_links_in_path:
                                assert self.substrate.net.edges[s_link]['bandwidth'] >= v_bandwidth_demand
                                self.substrate.net.edges[s_link]['bandwidth'] -= v_bandwidth_demand
                                sum_s_bandwidth_embedded += v_bandwidth_demand

        # calculate r_s
        if embedding_success:
            r_s = self.substrate.net.nodes[action.s_node]['CPU'] / self.substrate.initial_s_cpu_capacity[
                action.s_node]
        else:
            r_s = 1.0

        if embedding_success:
            # ALL SUCCESS --> EMBED VIRTUAL NODE!
            assert self.substrate.net.nodes[action.s_node]['CPU'] >= v_cpu_demand
            # 更新节点资源
            self.substrate.net.nodes[action.s_node]['CPU'] -= v_cpu_demand
            # 该底层节点被占用
            self.current_embedding[action.s_node] = 1
            # 以映射虚拟节点
            self.already_embedded_v_nodes.append(action.v_node)
            # 当前虚拟节点映射结果
            self.v_node_embedding_success.append(embedding_success)
        else:
            self.v_node_embedding_success.append(embedding_success)
            
            # 节点映射成功, 但链路映射失败, 回收该节点的映射方案
            if action.v_node in self.embedding_s_nodes:
                del self.embedding_s_nodes[action.v_node]


        # 이 지점에서 self.num_processed_v_nodes += 1 매우 중요: 이후 next_state 및 reward 계산에 영향을 줌
        # 已处理的虚拟节点数
        self.num_processed_v_nodes += 1

        # 计算reward
        reward = self.get_reward(
            embedding_success, v_cpu_demand, sum_v_bandwidth_demand, sum_s_bandwidth_embedded, action, r_s
        )

        done = False
        # 是否处理完所有虚拟节点
        # if not embedding_success or self.num_processed_v_nodes == len(self.vnr.net.nodes):
        if self.num_processed_v_nodes == len(self.vnr.net.nodes):
            # 是否全部虚拟节点都映射成功
            if sum(self.v_node_embedding_success) == len(self.vnr.net.nodes):
                self.vnr_embedding_success_count.append(1)
            else:
                self.vnr_embedding_success_count.append(0)

            # 是否结束整个VNE
            if self.vnr_idx == len(self.vnrs) - 1 or sum(self.vnr_embedding_success_count[-3:]) == 0:
            # if self.vnr_idx == len(self.vnrs) - 1:
                # print(self.vnr_embedding_success_count)
                print("The number of embedded success vnr: ", sum(self.vnr_embedding_success_count))
                done = True
                next_state = A3C_GCN_State(None, None, None, None)
            else:
                # 不结束
                
                # 下一个VN的id
                self.vnr_idx += 1
                self.vnr = self.vnrs[self.vnr_idx]
                # 虚拟节点排序
                self.sorted_v_nodes = utils.get_sorted_v_nodes_with_node_ranking(
                    vnr=self.vnr, type_of_node_ranking=config.TYPE_OF_VIRTUAL_NODE_RANKING.TYPE_2
                )
                self.num_processed_v_nodes = 0  # 已处理的虚拟节点数量
                # 当前要映射的虚拟节点及其CPU需求
                self.current_v_node, current_v_node_data, _ = self.sorted_v_nodes[self.num_processed_v_nodes]
                self.current_v_cpu_demand = current_v_node_data['CPU']

                # 特征矩阵
                substrate_features, substrate_edge_index, vnr_features = self.get_state_information(
                    self.current_v_node, self.current_v_cpu_demand
                )
                next_state = A3C_GCN_State(substrate_features, substrate_edge_index, self.current_v_node, vnr_features)
                self.already_embedded_v_nodes = []
                self.current_embedding = [0] * len(self.substrate.net.nodes)
                self.v_node_embedding_success = []
        else:
            # 下一个需要处理的节点
            self.current_v_node, current_v_node_data, _ = self.sorted_v_nodes[self.num_processed_v_nodes]
            # 该节点的CPU需求
            self.current_v_cpu_demand = current_v_node_data['CPU']
            # 获取特征矩阵
            substrate_features, substrate_edge_index, vnr_features = self.get_state_information(
                self.current_v_node, self.current_v_cpu_demand
            )
            next_state = A3C_GCN_State(substrate_features, substrate_edge_index, self.current_v_node, vnr_features)

        info = {}

        return next_state, reward, done, info

    def get_state_information(self, current_v_node, current_v_cpu_demand):
        """
        获取特征矩阵
        """
        # Substrate Initial State
        s_cpu_capacity = self.substrate.initial_s_cpu_capacity
        s_bandwidth_capacity = self.substrate.initial_s_node_total_bandwidth

        s_cpu_remaining = []
        s_bandwidth_remaining = []

        # S_cpu_Free, S_bw_Free
        for s_node, s_node_data in self.substrate.net.nodes(data=True):
            s_cpu_remaining.append(s_node_data['CPU'])

            total_node_bandwidth = 0.0
            for link_id in self.substrate.net[s_node]:
                total_node_bandwidth += self.substrate.net[s_node][link_id]['bandwidth']

            s_bandwidth_remaining.append(total_node_bandwidth)

        assert len(s_cpu_capacity) == len(s_bandwidth_capacity) == len(s_cpu_remaining) == len(s_bandwidth_remaining) == len(self.current_embedding)

        # Generate substrate feature matrix
        substrate_features = []
        substrate_features.append(s_cpu_capacity)
        substrate_features.append(s_bandwidth_capacity)
        substrate_features.append(s_cpu_remaining)
        substrate_features.append(s_bandwidth_remaining)
        substrate_features.append(self.current_embedding)

        # Convert to the torch.tensor
        substrate_features = torch.tensor(substrate_features)
        substrate_features = torch.transpose(substrate_features, 0, 1)
        substrate_features = substrate_features.view(1, config.SUBSTRATE_NODES, config.NUM_SUBSTRATE_FEATURES)
        # substrate_features.size() --> (1, 100, 5)

        # GCN for Feature Extract
        substrate_geometric_data = torch_geometric.utils.from_networkx(self.substrate.net)

        vnr_features = []
        vnr_features.append(current_v_cpu_demand)
        vnr_features.append(sum((self.vnr.net[current_v_node][link_id]['bandwidth'] for link_id in self.vnr.net[current_v_node])))
        vnr_features.append(len(self.sorted_v_nodes) - self.num_processed_v_nodes)
        vnr_features = torch.tensor(vnr_features).view(1, 1, 3)

        # substrate_features.size() --> (1, 100, 5)
        # vnr_features.size()) --> (1, 3)

        return substrate_features, substrate_geometric_data.edge_index, vnr_features

    def get_reward(self, embedding_success, v_cpu_demand, sum_v_bandwidth_demand, sum_s_bandwidth_embedded, action, r_s):
        """
        计算奖励reward
        """
        # 已处理虚拟节点数量占总数量的比例
        gamma_action = self.num_processed_v_nodes / len(self.vnr.net.nodes)
        r_a = 100 * gamma_action if embedding_success else -100 * gamma_action

        # calculate r_c
        if embedding_success:
            # 该节点映射应该花费的成本
            step_revenue = v_cpu_demand + sum_v_bandwidth_demand
            # 实际花费的成本
            step_cost = v_cpu_demand + sum_s_bandwidth_embedded
            
            # 增量
            delta_revenue = step_revenue - self.previous_step_revenue
            delta_cost = step_cost - self.previous_step_cost
            
            # 分母是否为0
            if delta_cost == 0.0:
                r_c = 1.0
            else:
                r_c = delta_revenue / delta_cost
                
            # 更新
            self.previous_step_revenue = step_revenue
            self.previous_step_cost = step_cost
        else:
            r_c = 1.0

        # calculate eligibility trace(资格追踪??)
        for s_node in self.substrate.net.nodes:
            if action.s_node == s_node:
                self.egb_trace[s_node] = self.decay_factor_for_egb_trace * (self.egb_trace[s_node] + 1)
            else:
                self.egb_trace[s_node] = self.decay_factor_for_egb_trace * self.egb_trace[s_node]

        reward = r_a * r_c * r_s / (self.egb_trace[action.s_node] + 1e-6)

        return reward
