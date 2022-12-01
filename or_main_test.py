import copy
from msilib.schema import Class
import os
import sys
from random import expovariate

import numpy as np
import pandas as pd
import time
import pandas as pd
from data_loader import predata


current_path = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(current_path, os.pardir))
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

from or_main.common.logger import get_logger
# from or_main.algorithms.a_baseline import BaselineVNEAgent
from or_main.algorithms.Agent_BaseLine import BaselineVNEAgent
from or_main.algorithms.g_a3c_gcn_vine import A3C_GCN_VNEAgent
from or_main.common import utils, config
from or_main.common.utils import draw_performance
from or_main.environments.vne_env import VNEEnvironment

# print(PROJECT_HOME)
PROJECT_HOME = ""
logger = get_logger("vne", PROJECT_HOME)

class VNE:
    
    def __init__(self, agent_name, agent_mode) -> None:
        self.performance_revenue = np.zeros(shape=(1, config.GLOBAL_MAX_STEPS + 1))
        self.performance_acceptance_ratio = np.zeros(shape=(1, config.GLOBAL_MAX_STEPS + 1))
        self.performance_rc_ratio = np.zeros(shape=(1, config.GLOBAL_MAX_STEPS + 1))
        self.performance_link_fail_ratio = np.zeros(shape=(1, config.GLOBAL_MAX_STEPS + 1))
        
        self.agent_name = agent_name
        self.agent_mode = agent_mode
        
        self.agent = None
        self.agent_label = None
        
        self.env = VNEEnvironment(logger)
    
    def get_agents(self, agent_name, agent_mode=False):
        """
        给每个算法设置一个智能体, 返回智能体集合
        """
        self.agent = BaselineVNEAgent(
                        logger=logger,
                        time_window_size=config.TIME_WINDOW_SIZE,
                        agent_type=config.ALGORITHMS.BASELINE,
                        agent_mode=self.agent_mode,
                        type_of_virtual_node_ranking=config.TYPE_OF_VIRTUAL_NODE_RANKING.TYPE_2,
                        allow_embedding_to_same_substrate_node=config.ALLOW_EMBEDDING_TO_SAME_SUBSTRATE_NODE,
                        max_embedding_path_length=config.MAX_EMBEDDING_PATH_LENGTH
                    )
        self.agent_label = config.ALGORITHMS.BASELINE.value
        
        # agent = A3C_GCN_VNEAgent(
        #                 local_model=None,
        #                 beta=0.3,
        #                 logger=logger,
        #                 time_window_size=config.TIME_WINDOW_SIZE,
        #                 agent_type=config.ALGORITHMS.BASELINE,
        #                 type_of_virtual_node_ranking=config.TYPE_OF_VIRTUAL_NODE_RANKING.TYPE_2,
        #                 allow_embedding_to_same_substrate_node=config.ALLOW_EMBEDDING_TO_SAME_SUBSTRATE_NODE,
        #                 max_embedding_path_length=config.MAX_EMBEDDING_PATH_LENGTH
        #             )
        # agent_label = config.ALGORITHMS.A3C_GCN.value
        pass
    
    def reset(self, ):
        """
        
        """
        self.performance_revenue = np.zeros(shape=(1, config.GLOBAL_MAX_STEPS + 1))
        self.performance_acceptance_ratio = np.zeros(shape=(1, config.GLOBAL_MAX_STEPS + 1))
        self.performance_rc_ratio = np.zeros(shape=(1, config.GLOBAL_MAX_STEPS + 1))
        self.performance_link_fail_ratio = np.zeros(shape=(1, config.GLOBAL_MAX_STEPS + 1))
        
        self.env = VNEEnvironment(logger)

        # print("当前物理网络节点数为: {}, 物理链路数量为: {}; "
        #       "虚拟网络请求数为: {}, 虚拟网络生存周期服从指数分布, 其中lambda为: {}".format(config.SUBSTRATE_NODES,config.SUBSTRATE_LINKS,config.GLOBAL_MAX_NUMBERS,
        #                                                        config.VNR_DURATION_MEAN_RATE))

        # print("虚拟网络节点数量: {}-{}, 服从均匀分布; 到达时间服从泊松分布, 即每100个时间单位, 平均有5次VNR到达".format(config.VNR_NODES_MIN,config.VNR_NODES_MAX))

        # print("当前物理网络最小cpu资源为: {}, 最大cpu资源为: {}, 最小带宽资源为: {}, 最大带宽资源为: {}, 服从均匀分布".format(config.SUBSTRATE_NODE_CAPACITY_MIN,
        #                                                                    config.SUBSTRATE_NODE_CAPACITY_MAX,config.SUBSTRATE_LINK_CAPACITY_MIN,
        #                                                                    config.SUBSTRATE_LINK_CAPACITY_MAX))
        # print("当前虚拟网络最小cpu资源为: {}, 最大cpu资源为: {}, 最小带宽资源为: {}, 最大带宽资源为: {}, 服从均匀分布".format(config.VNR_CPU_DEMAND_MIN,
        #                                                                    config.VNR_CPU_DEMAND_MAX,
        #                                                                    config.VNR_BANDWIDTH_DEMAND_MIN,
        #                                                                    config.VNR_BANDWIDTH_DEMAND_MAX))

        # 底层节点资源数据: CPU + bw
        self.SN_Link = [[v, u, b] for v, u, b in self.env.SUBSTRATE.net.edges.data("bandwidth")]
        self.SN_Node = [node[1] for node in self.env.SUBSTRATE.net.nodes.data("CPU")]

        # 所有虚拟网络的链路和节点信息
        self.VN_Link = {}
        self.VN_Node = {}
        self.VN_Life = []
        
        # 生成泊松分布的到达时间 + 生命周期
        self.VN_Arrive_Time = predata.arrive_time(len(self.env.VNRs_INFO))
        self.duration = predata.create_virtual_network_lift_time(size=len(self.env.VNRs_INFO),lambda_=1/config.VNR_DURATION_MEAN_RATE)
        
        # 记录所有VN的相关信息
        for i in range(len(self.env.VNRs_INFO)):
            vnr = self.env.VNRs_INFO[i]
            vn_node = [node[1] for node in vnr.net.nodes.data("CPU")]
            vn_link = vnr.net.edges.data("bandwidth")

            # vn_life = vnr.duration # 生存周期
            # VN_Life.append([i, vn_life[i], 0, 0])

            self.env.VNRs_INFO[i].duration = self.duration[i]  # 生存周期
            self.VN_Life.append([i, self.duration[i], 0, 0])

            self.VN_Node.update({i: vn_node})
            self.VN_Link.update({i: list(vn_link)})

            self.env.VNRs_INFO[i].time_step_arrival = self.VN_Arrive_Time[i] # 到达时间

        self.data = {
            "SN_Link": self.SN_Link,
            "SN_Node": self.SN_Node,
            "VN_Node": self.VN_Node,
            "VN_Link": self.VN_Link,
            "VN_Life": {0: self.VN_Life},  # 0时刻的虚拟网络??
            "VN_Arrive_Time": self.VN_Arrive_Time,
        }
        # predata.save_data(self.data, len(self.VN_Link))

        self.state = self.env.reset()

        # 是否结束整个VNE流程
        self.done = False
        self.time_step = 0
        
        self.start_ts = time.time()
        self.run_start_ts = time.time()
    
    def save_result(self, ):
        """
        保存智能体的评价指标为 "CSV"
        """
        df = pd.DataFrame()
        df["r/c"] = self.performance_rc_ratio
        df["revenue"] = self.performance_revenue
        df["accept_ratio"] = self.performance_acceptance_ratio
        
        # TODO: change the path
        df.to_csv("D:/VNE_test/data/result/{}_{}.csv".format(self.agent_label, len(self.env.VNRs_INFO)))
        # time.sleep(100)
        
    def show_result(self, ):
        """
        show result
        """
        draw_performance(
                    agents=[self.agent], agent_labels=[self.agent_label], run=1, time_step=self.time_step,
                    performance_revenue=self.performance_revenue / (1),
                    performance_acceptance_ratio=self.performance_acceptance_ratio / (1),
                    performance_rc_ratio=self.performance_rc_ratio / (1),
                    performance_link_fail_ratio=self.performance_link_fail_ratio / (1),
                )
        
    def run(self, ):
        """
        VNE process
        """
        
        self.reset()
        
        while not self.done:
            self.time_step += 1
            
            before_action_msg = "state {0} | ".format(repr(self.state))
            before_action_simple_msg = "state {0} | ".format(self.state)
            logger.info("{0} {1}".format(
                utils.run_agent_step_prefix(1, self.agent_label, self.time_step), before_action_msg
            ))

            # action: 每个智能体的单次动作
            # 包括:
            # 1. 暂时未成功映射的vn 2. 成功映射的vn的节点和链路映射结果 3. 节点映射阶段失败数量 4. 链路映射阶段失败数量
            action = self.agent.get_action(self.state)

            action_msg = "act. {0:30} |".format(
                str(action) if action.vnrs_embedding is not None and action.vnrs_postponement is not None else " - "
            )
            logger.info("{0} {1}".format(
                utils.run_agent_step_prefix(1, self.agent_label, self.time_step), action_msg
            ))

            # 执行action, 返回(下一状态, 奖励, 是否结束, (收益, 接受率, R/C, 链路失败比例))
            next_state, reward, self.done, info = self.env.step(action)

            elapsed_time = time.time() - self.run_start_ts
            after_action_msg = "reward {0:6.1f} | revenue {1:6.1f} | acc. ratio {2:4.2f} | " \
                                "r/c ratio {3:4.2f} | {4}".format(
                reward, info['revenue'], info['acceptance_ratio'], info['rc_ratio'],
                time.strftime("%Hh %Mm %Ss", time.gmtime(elapsed_time)),
            )
            after_action_msg += " | {0:3.1f} steps/sec.".format(self.time_step / elapsed_time)
            logger.info("{0} {1}".format(
                utils.run_agent_step_prefix(1, self.agent_label, self.time_step), after_action_msg
            ))

            print("{0} {1} {2} {3}".format(
                utils.run_agent_step_prefix(1, self.agent_label, self.time_step),
                before_action_simple_msg,
                action_msg,
                after_action_msg
            ))

            self.state = next_state
            
            # 记录相关评价指标
            # TODO: resource util_rate
            self.performance_revenue[0, self.time_step] += info['revenue']
            self.performance_acceptance_ratio[0, self.time_step] += info['acceptance_ratio']
            self.performance_rc_ratio[0, self.time_step] += info['rc_ratio']
            self.performance_link_fail_ratio[0, self.time_step] += \
                info['link_embedding_fails_against_total_fails_ratio']

            if self.time_step > config.FIGURE_START_TIME_STEP - 1 and self.time_step % 1000 == 0:
                self.show_result()
        
        msg = "RUN: {0} FINISHED - ELAPSED TIME: {1}".format(
            1, time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - self.start_ts))
        )
        logger.info(msg), print(msg)
        
    def vne(self, ):
        if self.agent_mode:
            # for i in range(config.MaxEpisode):
            self.get_agents(agent_name=self.agent_name, agent_mode=self.agent_mode)
            for _ in range(5):
                self.get_agents(agent_name=self.agent_name, agent_mode=self.agent_mode)
                
                # self.run()
            
            # self.agent.save_model()
        
        # test
        self.get_agents(agent_name=self.agent_name)
        self.run()


if __name__ == "__main__":
    
    target_algorithms = [
        "A3C_GCN",               
        "TOPOLOGY_AWARE_NODE_RANKING", 
        "BASELINE", 
        "TOPOLOGY_AWARE_DEGREE", 
        "EGO_NETWORK", 
        "DETERMINISTIC_VINE", 
        "RANDOMIZED_VINE", 
        "TOPOLOGY_AWARE_NODE_RANKING", 
        "A3C_GCN"
    ]
    
    # main(target_algorithms[0], mode=True)
    vne = VNE(target_algorithms[0], agent_mode=False)
    vne.vne()
