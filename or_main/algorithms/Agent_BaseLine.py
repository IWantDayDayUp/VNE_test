import networkx as nx
import copy
from or_main.common import utils

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

class BaselineVNEAgent:
    def __init__(
            self, logger, time_window_size, agent_type, agent_mode, type_of_virtual_node_ranking,
            allow_embedding_to_same_substrate_node, max_embedding_path_length
    ):
        self.logger = logger
        
        self.num_node_embedding_fails = 0
        self.num_link_embedding_fails = 0
        
        self.time_step = 0
        self.time_window_size = time_window_size
        self.next_embedding_epoch = time_window_size
        
        self.agent_type = agent_type
        self.agent_mode = agent_mode
        
        self.type_of_virtual_node_ranking = type_of_virtual_node_ranking
        self.allow_embedding_to_same_substrate_node = allow_embedding_to_same_substrate_node
        self.max_embedding_path_length = max_embedding_path_length

    def find_substrate_nodes(self, copied_substrate, vnr):
        '''
        计算给定vn的节点映射结果
        '''
        subset_S_per_v_node = {}
        embedding_s_nodes = {}
        already_embedding_s_nodes = []

        # 虚拟节点排序
        sorted_v_nodes_with_node_ranking = utils.get_sorted_v_nodes_with_node_ranking(
            vnr=vnr, type_of_node_ranking=self.type_of_virtual_node_ranking
        )
        
        for v_node_id, v_node_data, _ in sorted_v_nodes_with_node_ranking:
            v_cpu_demand = v_node_data['CPU']
            v_node_location = v_node_data['LOCATION']

            # 候选节点集合
            subset_S_per_v_node[v_node_id] = utils.find_subset_S_for_virtual_node(
                copied_substrate, v_cpu_demand, v_node_location, already_embedding_s_nodes
            )

            # 计算候选节点CPU以及邻接链路带宽和, 挑选最大的作为目标节点
            selected_s_node_id = max(
                subset_S_per_v_node[v_node_id],
                key=lambda s_node_id: self.calculate_H_value(
                    copied_substrate.net.nodes[s_node_id]['CPU'],
                    copied_substrate.net[s_node_id]
                ),
                default=None
            )

            if selected_s_node_id is None:
                self.num_node_embedding_fails += 1
                msg = "VNR {0} REJECTED ({1}): 'no suitable SUBSTRATE NODE for nodal constraints: {2}' {3}".format(
                    vnr.id, self.num_node_embedding_fails, v_cpu_demand, vnr
                )
                self.logger.info("{0} {1}".format(utils.step_prefix(self.time_step), msg))
                return None

            assert selected_s_node_id != -1
            embedding_s_nodes[v_node_id] = (selected_s_node_id, v_cpu_demand)

            if not self.allow_embedding_to_same_substrate_node:
                already_embedding_s_nodes.append(selected_s_node_id)

            # !: 每成功映射一个节点就立马更新底层资源, 如果后面有个虚拟节点失败了, 怎么拿回已分配下去的资源???
            assert copied_substrate.net.nodes[selected_s_node_id]['CPU'] >= v_cpu_demand
            copied_substrate.net.nodes[selected_s_node_id]['CPU'] -= v_cpu_demand

        return embedding_s_nodes

    def find_substrate_path(self, copied_substrate, vnr, embedding_s_nodes):
        """
        为给定vn寻找最短底层路径, 返回映射结果
        """
        embedding_s_paths = {}

        # mapping the virtual nodes and substrate_net nodes
        for src_v_node, dst_v_node, edge_data in vnr.net.edges(data=True):
            v_link = (src_v_node, dst_v_node)
            src_s_node = embedding_s_nodes[src_v_node][0]
            dst_s_node = embedding_s_nodes[dst_v_node][0]
            v_bandwidth_demand = edge_data['bandwidth']

            if src_s_node == dst_s_node:
                embedding_s_paths[v_link] = ([], v_bandwidth_demand)
            else:
                # 删选出底层链路中满足带宽需求的链路
                subnet = nx.subgraph_view(
                    copied_substrate.net,
                    filter_edge=lambda node_1_id, node_2_id: \
                        True if copied_substrate.net.edges[(node_1_id, node_2_id)]['bandwidth'] >= v_bandwidth_demand else False
                )

                if len(subnet.edges) == 0 or not nx.has_path(subnet, source=src_s_node, target=dst_s_node):
                    self.num_link_embedding_fails += 1
                    msg = "VNR {0} REJECTED ({1}): 'no suitable LINK for bandwidth demand: {2} {3}".format(
                        vnr.id, self.num_link_embedding_fails, v_bandwidth_demand, vnr
                    )
                    self.logger.info("{0} {1}".format(utils.step_prefix(self.time_step), msg))
                    return None

                # 计算最短路径
                MAX_K = 1
                shortest_s_path = utils.k_shortest_paths(subnet, source=src_s_node, target=dst_s_node, k=MAX_K)[0]

                # Check the path length
                if len(shortest_s_path) > self.max_embedding_path_length:
                    self.num_link_embedding_fails += 1
                    msg = "VNR {0} REJECTED ({1}): 'no suitable LINK for bandwidth demand: {2} {3}".format(
                        vnr.id, self.num_link_embedding_fails, v_bandwidth_demand, vnr
                    )
                    self.logger.info("{0} {1}".format(utils.step_prefix(self.time_step), msg))
                    return None

                s_links_in_path = []
                for node_idx in range(len(shortest_s_path) - 1):
                    s_links_in_path.append((shortest_s_path[node_idx], shortest_s_path[node_idx + 1]))

                # 更新底层网络链路资源
                for s_link in s_links_in_path:
                    assert copied_substrate.net.edges[s_link]['bandwidth'] >= v_bandwidth_demand
                    copied_substrate.net.edges[s_link]['bandwidth'] -= v_bandwidth_demand

                embedding_s_paths[v_link] = (s_links_in_path, v_bandwidth_demand)

        return embedding_s_paths

    def calculate_H_value(self, s_cpu_capacity, adjacent_links):
        """
        calculate the H value(该节点CPU以及邻接链路带宽和)
        """
        total_node_bandwidth = sum((adjacent_links[link_id]['bandwidth'] for link_id in adjacent_links))

        # total_node_bandwidth = 0.0
        #
        # for link_id in adjacent_links:
        #     total_node_bandwidth += adjacent_links[link_id]['bandwidth']

        return s_cpu_capacity * total_node_bandwidth

    def node_mapping(self, VNRs_COLLECTED, COPIED_SUBSTRATE, action):
        """
        将等待映射的VN按收益降序排序, 依次映射, 并返回节点映射结果
        """
        sorted_vnrs = sorted(
            VNRs_COLLECTED.values(), key=lambda vnr: vnr.revenue, reverse=True
        )
        
        sorted_vnrs_and_node_embedding = []
        for vnr in sorted_vnrs:
            embedding_s_nodes = self.find_substrate_nodes(COPIED_SUBSTRATE, vnr)

            if embedding_s_nodes is None:
                action.vnrs_postponement[vnr.id] = vnr
            else:
                sorted_vnrs_and_node_embedding.append((vnr, embedding_s_nodes))

        return sorted_vnrs_and_node_embedding

    def link_mapping(self, sorted_vnrs_and_node_embedding, COPIED_SUBSTRATE, action):
        """
        按照节点映射结果进行链路映射
        """
        for vnr, embedding_s_nodes in sorted_vnrs_and_node_embedding:
            embedding_s_paths = self.find_substrate_path(COPIED_SUBSTRATE, vnr, embedding_s_nodes)

            if embedding_s_paths is None:
                action.vnrs_postponement[vnr.id] = vnr
            else:
                action.vnrs_embedding[vnr.id] = (vnr, embedding_s_nodes, embedding_s_paths)
    
    def save_model(self, ):
        """
        
        """
        pass

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
        
        return action

    def embedding(self, VNRs_COLLECTED, COPIED_SUBSTRATE, action):
        """
        将收集好的一系列VN映射到底层网络中
        """
        #####################################
        # step 1 - Greedy Node Mapping      #
        #####################################
        sorted_vnrs_and_node_embedding = self.node_mapping(VNRs_COLLECTED, COPIED_SUBSTRATE, action)

        #####################################
        # step 2 - Link Mapping             #
        #####################################
        self.link_mapping(sorted_vnrs_and_node_embedding, COPIED_SUBSTRATE, action)
