import gym
import networkx as nx
from random import randint, expovariate

from or_main.algorithms.a_baseline import Action
from or_main.common import utils, config
import enum

class EnvMode(enum.Enum):
    plain = 0
    a3c = 1


class Substrate:
    """
    
    """

    def __init__(self):
        all_connected = False
        while not all_connected:
            self.net = nx.gnm_random_graph(n=config.SUBSTRATE_NODES, m=config.SUBSTRATE_LINKS)
            all_connected = nx.is_connected(self.net)

        self.initial_s_cpu_capacity = []
        self.initial_s_bw_capacity = []
        self.initial_s_node_total_bandwidth = []
        self.initial_total_cpu_capacity = 0
        self.initial_total_bandwidth_capacity = 0

        # corresponding CPU and bandwidth resources of it are real numbers uniformly distributed from 50 to 100
        self.min_cpu_capacity = 1.0e10
        self.max_cpu_capacity = 0.0
        for node_id in self.net.nodes:
            self.net.nodes[node_id]['CPU'] = randint(config.SUBSTRATE_NODE_CAPACITY_MIN, config.SUBSTRATE_NODE_CAPACITY_MAX)
            self.net.nodes[node_id]['LOCATION'] = randint(0, config.NUM_LOCATION)
            self.initial_s_cpu_capacity.append(self.net.nodes[node_id]['CPU'])
            self.initial_total_cpu_capacity += self.net.nodes[node_id]['CPU']
            if self.net.nodes[node_id]['CPU'] < self.min_cpu_capacity:
                self.min_cpu_capacity = self.net.nodes[node_id]['CPU']
            if self.net.nodes[node_id]['CPU'] > self.max_cpu_capacity:
                self.max_cpu_capacity = self.net.nodes[node_id]['CPU']

        self.min_bandwidth_capacity = 1.0e10
        self.max_bandwidth_capacity = 0.0
        for edge_id in self.net.edges:
            self.net.edges[edge_id]['bandwidth'] = randint(config.SUBSTRATE_LINK_CAPACITY_MIN, config.SUBSTRATE_LINK_CAPACITY_MAX)
            self.initial_s_bw_capacity.append(self.net.edges[edge_id]['bandwidth'])
            self.initial_total_bandwidth_capacity += self.net.edges[edge_id]['bandwidth']
            if self.net.edges[edge_id]['bandwidth'] < self.min_bandwidth_capacity:
                self.min_bandwidth_capacity = self.net.edges[edge_id]['bandwidth']
            if self.net.edges[edge_id]['bandwidth'] > self.max_bandwidth_capacity:
                self.max_bandwidth_capacity = self.net.edges[edge_id]['bandwidth']

        for s_node_id in range(len(self.net.nodes)):
            total_node_bandwidth = 0.0
            for link_id in self.net[s_node_id]:
                total_node_bandwidth += self.net[s_node_id][link_id]['bandwidth']
            self.initial_s_node_total_bandwidth.append(total_node_bandwidth)

    def __eq__(self, other):
        remaining_cpu_resource = sum([node_data['CPU'] for _, node_data in self.net.nodes(data=True)])
        remaining_bandwidth_resource = sum([link_data['bandwidth'] for _, _, link_data in self.net.edges(data=True)])

        other_remaining_cpu_resource = sum([node_data['CPU'] for _, node_data in other.net.nodes(data=True)])
        other_remaining_bandwidth_resource = sum([link_data['bandwidth'] for _, _, link_data in other.net.edges(data=True)])

        equal_conditions = [
            len(self.net.nodes) == len(other.net.nodes),
            len(self.net.edges) == len(other.net.edges),
            remaining_cpu_resource == other_remaining_cpu_resource,
            remaining_bandwidth_resource == other_remaining_bandwidth_resource
        ]

        if all(equal_conditions):
            return True
        else:
            return False

    def __str__(self):
        remaining_cpu_resource = sum([node_data['CPU'] for _, node_data in self.net.nodes(data=True)])
        remaining_bandwidth_resource = sum([link_data['bandwidth'] for _, _, link_data in self.net.edges(data=True)])

        substrate_str = "[SUBST. CPU: {0:6.2f}%, BAND: {1:6.2f}%]".format(
            100 * remaining_cpu_resource / self.initial_total_cpu_capacity,
            100 * remaining_bandwidth_resource / self.initial_total_bandwidth_capacity,
        )

        return substrate_str

    def __repr__(self):
        remaining_cpu_resource = sum([node_data['CPU'] for _, node_data in self.net.nodes(data=True)])
        remaining_bandwidth_resource = sum([link_data['bandwidth'] for _, _, link_data in self.net.edges(data=True)])

        substrate_str = "[SUBSTRATE cpu: {0:4}/{1:4}={2:6.2f}% ({3:2}~{4:3}), " \
                        "bandwidth: {5:4}/{6:4}={7:6.2f}% ({8:2}~{9:3})]".format(
            remaining_cpu_resource, self.initial_total_cpu_capacity, 100 * remaining_cpu_resource / self.initial_total_cpu_capacity,
            self.min_cpu_capacity, self.max_cpu_capacity,
            remaining_bandwidth_resource, self.initial_total_bandwidth_capacity, 100 * remaining_bandwidth_resource / self.initial_total_bandwidth_capacity,
            self.min_bandwidth_capacity, self.max_bandwidth_capacity
        )
        return substrate_str


class VNR:
    """
    
    """
    
    def __init__(self, id, vnr_duration_mean_rate, delay, time_step_arrival):
        self.id = id

        self.duration = int(expovariate(vnr_duration_mean_rate) + 1.0)

        self.delay = delay

        self.num_nodes = randint(config.VNR_NODES_MIN, config.VNR_NODES_MAX)

        all_connected = False
        while not all_connected:
            self.net = nx.gnp_random_graph(n=self.num_nodes, p=config.VNR_LINK_PROBABILITY, directed=True)
            all_connected = nx.is_weakly_connected(self.net)

        self.num_of_edges = len(self.net.edges)
        self.num_of_edges_complete_graph = int(self.num_nodes * (self.num_nodes - 1) / 2)

        self.min_cpu_demand = 1.0e10
        self.max_cpu_demand = 0.0
        for node_id in self.net.nodes:
            self.net.nodes[node_id]['CPU'] = randint(
                config.VNR_CPU_DEMAND_MIN, config.VNR_CPU_DEMAND_MAX
            )
            self.net.nodes[node_id]['LOCATION'] = randint(0, config.NUM_LOCATION)
            if self.net.nodes[node_id]['CPU'] < self.min_cpu_demand:
                self.min_cpu_demand = self.net.nodes[node_id]['CPU']
            if self.net.nodes[node_id]['CPU'] > self.max_cpu_demand:
                self.max_cpu_demand = self.net.nodes[node_id]['CPU']

        self.min_bandwidth_demand = 1.0e10
        self.max_bandwidth_demand = 0.0
        for edge_id in self.net.edges:
            self.net.edges[edge_id]['bandwidth'] = randint(
                config.VNR_BANDWIDTH_DEMAND_MIN, config.VNR_BANDWIDTH_DEMAND_MAX
            )
            if self.net.edges[edge_id]['bandwidth'] < self.min_bandwidth_demand:
                self.min_bandwidth_demand = self.net.edges[edge_id]['bandwidth']
            if self.net.edges[edge_id]['bandwidth'] > self.max_bandwidth_demand:
                self.max_bandwidth_demand = self.net.edges[edge_id]['bandwidth']

        self.time_step_arrival = time_step_arrival
        self.time_step_leave_from_queue = self.time_step_arrival + self.delay

        self.time_step_serving_started = None
        self.time_step_serving_completed = None

        self.revenue = utils.get_revenue_VNR(self)

        self.cost = None

    def __lt__(self, other_vnr):
        return 1.0 / self.revenue < 1.0 / other_vnr.revenue

    def __str__(self):
        vnr_stat_str = "nodes: {0:>2}, edges: {1:>2}|{2:>3}, revenue: {3:6.1f}({4:1}~{5:2}, {6:1}~{7:2})".format(
            self.num_nodes, self.num_of_edges, self.num_of_edges_complete_graph,
            self.revenue,
            self.min_cpu_demand, self.max_cpu_demand,
            self.min_bandwidth_demand, self.max_bandwidth_demand
        )

        vnr_str = "[id: {0:2}, {1:>2}, arrival: {2:>4}, expired out: {3:>4}, " \
                  "duration: {4:>4}, started: {5:>4}, completed out: {6:>4}]".format(
            self.id, vnr_stat_str,
            self.time_step_arrival, self.time_step_leave_from_queue, self.duration,
            self.time_step_serving_started if self.time_step_serving_started else "N/A",
            self.time_step_serving_completed if self.time_step_serving_completed else "N/A"
        )

        return vnr_str


class State:
    """
    
    """
    
    def __init__(self):
        self.substrate = None
        self.vnrs_collected = None
        self.vnrs_serving = None

    def __str__(self):
        state_str = str(self.substrate)
        vnrs_collected_str = "[{0:2} VNR COLLECTED]".format(len(self.vnrs_collected))
        vnrs_serving_str = "[{0:2} VNR SERVING]".format(len(self.vnrs_serving))

        state_str = " ".join([state_str, vnrs_collected_str, vnrs_serving_str])

        return state_str

    def __repr__(self):
        state_str = repr(self.substrate)
        vnrs_collected_str = "[{0:2} VNR COLLECTED]".format(len(self.vnrs_collected))
        vnrs_serving_str = "[{0:2} VNR SERVING]".format(len(self.vnrs_serving))

        state_str = " ".join([state_str, vnrs_collected_str, vnrs_serving_str])

        return state_str


class VNEEnvironment(gym.Env):
    """
    整个算法的模型: 底层网络 + 虚拟网络
    """
    
    def __init__(self, logger):
        self.logger = logger

        self.SUBSTRATE = Substrate()
        self.VNRs_INFO = {}

        time_step = 0
        vnr_id = 0

        while True:
            next_arrival = int(expovariate(config.VNR_INTER_ARRIVAL_RATE))

            time_step += next_arrival

            if time_step >= config.GLOBAL_MAX_STEPS and not config.GLOBAL_GENERATION_NUMBERS:
                break

            if vnr_id >= config.GLOBAL_MAX_NUMBERS and config.GLOBAL_GENERATION_NUMBERS:
                break

            vnr = VNR(
                id=vnr_id,
                vnr_duration_mean_rate=config.VNR_DURATION_MEAN_RATE,
                delay=config.VNR_DELAY,
                time_step_arrival=time_step
            )

            self.VNRs_INFO[vnr.id] = vnr
            vnr_id += 1

        # 目前存活的已映射VN(正在服务中的VN)
        self.VNRs_SERVING = None
        # 目前队列里等待映射的VN
        self.VNRs_COLLECTED = None

        self.time_step = None

        # 到达的VN总数
        self.total_arrival_vnrs = None
        # 到达的VN里成功映射的VN总数
        self.total_embedded_vnrs = None

        self.episode_reward = None
        
        # 累计奖励
        self.revenue = None
        self.acceptance_ratio = None
        self.rc_ratio = None
        self.link_embedding_fails_against_total_fails_ratio = None

    def reset(self):
        """
        环境重置
        """
        
        self.VNRs_SERVING = {}
        self.VNRs_COLLECTED = {}

        self.time_step = 0

        self.total_arrival_vnrs = 0
        self.total_embedded_vnrs = 0

        self.episode_reward = 0.0
        self.revenue = 0.0
        self.acceptance_ratio = 0.0
        self.rc_ratio = 0.0
        self.link_embedding_fails_against_total_fails_ratio = 0.0

        # 收集新到达的等待映射的VN
        self.collect_vnrs_new_arrival()

        initial_state = State()
        initial_state.substrate = self.SUBSTRATE
        initial_state.vnrs_collected = self.VNRs_COLLECTED
        initial_state.vnrs_serving = self.VNRs_SERVING

        return initial_state

    def step(self, action: Action):
        """
        执行action, 并返回(下一状态, 奖励, 是否终止, 其他信息)
        """
        self.time_step += 1

        # 生命周期结束, 需要释放资源的VN
        vnrs_left_from_queue = self.release_vnrs_expired_from_collected(
            action.vnrs_embedding if action.vnrs_postponement is not None and action.vnrs_embedding is not None else []
        )
        # 
        vnrs_serving_completed = self.complete_vnrs_serving()

        # processing of embedding & postponement
        if action.vnrs_postponement is not None and action.vnrs_embedding is not None:
            
            # 对于成功映射的VN, 落实到底层网络中
            for vnr, embedding_s_nodes, embedding_s_paths in action.vnrs_embedding.values():
                assert vnr not in vnrs_left_from_queue
                assert vnr not in vnrs_serving_completed

                self.starting_serving_for_a_vnr(vnr, embedding_s_nodes, embedding_s_paths)

        # 收集新的VN
        self.collect_vnrs_new_arrival()

        reward = 0.0
        cost = 0.0

        # 累加当前正在服务中的VN的收益与成本
        for vnr, _, embedding_s_paths in self.VNRs_SERVING.values():
            reward += vnr.revenue
            cost += vnr.cost

        # 是否结束整个VNE流程
        if self.time_step >= config.GLOBAL_MAX_STEPS:
            done = True
        else:
            done = False

        # 下一状态
        next_state = State()
        next_state.substrate = self.SUBSTRATE
        next_state.vnrs_collected = self.VNRs_COLLECTED
        next_state.vnrs_serving = self.VNRs_SERVING

        self.episode_reward += reward
        self.revenue = self.episode_reward / self.time_step
        self.acceptance_ratio = self.total_embedded_vnrs / self.total_arrival_vnrs if self.total_arrival_vnrs else 0.0
        print("self.total_embedded_vnrs:{},self.total_arrival_vnrs:{}".format(self.total_embedded_vnrs,self.total_arrival_vnrs))
        self.rc_ratio = reward / cost if cost else 0.0
        self.link_embedding_fails_against_total_fails_ratio = \
            action.num_link_embedding_fails / (action.num_node_embedding_fails + action.num_link_embedding_fails) \
            if action and action.num_link_embedding_fails + action.num_node_embedding_fails else 0.0

        info = {
            "revenue": self.revenue,
            "acceptance_ratio": self.acceptance_ratio,
            "rc_ratio": self.rc_ratio,
            "link_embedding_fails_against_total_fails_ratio": self.link_embedding_fails_against_total_fails_ratio
        }

        return next_state, reward, done, info

    def release_vnrs_expired_from_collected(self, vnrs_embedding):
        '''
        需要从等待队列里离开的VN集合
        '''
        vnrs_left_from_queue = []
        for vnr in self.VNRs_COLLECTED.values():
            # 最大等待时间到了
            if vnr.time_step_leave_from_queue <= self.time_step and vnr.id not in vnrs_embedding:
                vnrs_left_from_queue.append(vnr)

        for vnr in vnrs_left_from_queue:
            del self.VNRs_COLLECTED[vnr.id]
            if self.logger:
                self.logger.info("{0} VNR LEFT OUT {1}".format(utils.step_prefix(self.time_step), vnr))

        return vnrs_left_from_queue

    def starting_serving_for_a_vnr(self, vnr, embedding_s_nodes, embedding_s_paths):
        """
        将VN落实到底层网络, 服务正式开启
        """
        for s_node_id, v_cpu_demand in embedding_s_nodes.values():
            self.SUBSTRATE.net.nodes[s_node_id]['CPU'] -= v_cpu_demand

        for s_links_in_path, v_bandwidth_demand in embedding_s_paths.values():
            for s_link in s_links_in_path:
                self.SUBSTRATE.net.edges[s_link]['bandwidth'] -= v_bandwidth_demand

        # 服务开启时刻
        vnr.time_step_serving_started = self.time_step
        # 服务结束时刻
        vnr.time_step_serving_completed = self.time_step + vnr.duration
        # 实际映射成本
        vnr.cost = utils.get_cost_VNR(vnr, embedding_s_paths)

        self.VNRs_SERVING[vnr.id] = (vnr, embedding_s_nodes, embedding_s_paths)
        if self.logger:
            self.logger.info("{0} VNR SERVING STARTED {1}".format(utils.step_prefix(self.time_step), vnr))
        self.total_embedded_vnrs += 1

        del self.VNRs_COLLECTED[vnr.id]

    def complete_vnrs_serving(self):
        '''
        服务到期, 需要离开底层网络的VN集合
        '''
        vnrs_serving_completed = []
        for vnr, embedding_s_nodes, embedding_s_paths in self.VNRs_SERVING.values():
            # 到期
            if vnr.time_step_serving_completed and vnr.time_step_serving_completed <= self.time_step:
                vnrs_serving_completed.append(vnr)

                # 释放资源
                for s_node_id, v_cpu_demand in embedding_s_nodes.values():
                    self.SUBSTRATE.net.nodes[s_node_id]['CPU'] += v_cpu_demand
                for s_links_in_path, v_bandwidth_demand in embedding_s_paths.values():
                    for s_link in s_links_in_path:
                        self.SUBSTRATE.net.edges[s_link]['bandwidth'] += v_bandwidth_demand

        for vnr in vnrs_serving_completed:
            assert vnr.id in self.VNRs_SERVING
            del self.VNRs_SERVING[vnr.id]
            if self.logger:
                self.logger.info("{0} VNR SERVING COMPLETED {1}".format(utils.step_prefix(self.time_step), vnr))

        return vnrs_serving_completed

    def collect_vnrs_new_arrival(self):
        """
        收集当前时刻新到达的VN请求集合
        """
        for vnr in self.VNRs_INFO.values():
            if vnr.time_step_arrival == self.time_step:
                self.VNRs_COLLECTED[vnr.id] = vnr
                self.total_arrival_vnrs += 1
                if self.logger:
                    self.logger.info("{0} NEW VNR ARRIVED {1}".format(utils.step_prefix(self.time_step), vnr))
