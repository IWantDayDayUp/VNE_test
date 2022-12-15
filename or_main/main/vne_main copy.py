import copy
import os
import sys
from random import expovariate

import numpy as np
import pandas as pd
import time

current_path = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(current_path, os.pardir))

# 根目录
root_dir = os.path.abspath(os.path.join(PROJECT_HOME, ".."))
if root_dir not in sys.path:
    sys.path.append(root_dir)

# print(current_path)
# print(PROJECT_HOME)

if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

from or_main.algorithms.a_baseline import BaselineVNEAgent
from or_main.algorithms.b_topology_aware_baseline import TopologyAwareBaselineVNEAgent
from or_main.algorithms.c_ego_network_baseline import EgoNetworkBasedVNEAgent
from or_main.algorithms.f_node_rank_baseline import TopologyAwareNodeRankingVNEAgent
from or_main.algorithms.g_a3c_gcn_vine import A3C_GCN_VNEAgent

from or_main.common.logger import get_logger

from or_main.common import utils, config
from or_main.common.utils import draw_performance
from or_main.environments.vne_env import VNEEnvironment

logger = get_logger("vne", PROJECT_HOME)


def main():
    # 给每个算法设置一个智能体
    agents, agent_labels = get_agents()

    # 相关评价指标: 收益, 接受率, R/C, 链路失败率
    # GLOBAL_MAX_STEPS = 20000, 总的实验时间
    performance_revenue = np.zeros(shape=(len(agents), config.GLOBAL_MAX_STEPS + 1))
    performance_acceptance_ratio = np.zeros(shape=(len(agents), config.GLOBAL_MAX_STEPS + 1))
    performance_rc_ratio = np.zeros(shape=(len(agents), config.GLOBAL_MAX_STEPS + 1))
    performance_link_fail_ratio = np.zeros(shape=(len(agents), config.GLOBAL_MAX_STEPS + 1))

    start_ts = time.time()
    
    # config.NUM_RUNS = 1: 将算数平均值作为最终结果
    for run in range(1):
        run_start_ts = time.time()

        env = VNEEnvironment(logger)

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
        SN_Link = [[v, u, b] for v, u, b in env.SUBSTRATE.net.edges.data("bandwidth")]
        SN_Node = [node[1] for node in env.SUBSTRATE.net.nodes.data("CPU")]

        VN_Link = {}
        VN_Node = {}
        VN_Life = []
        
        # 生成泊松分布的到达时间 + 生命周期
        import data_loader.predata as predata
        VN_Arrive_Time = predata.arrive_time(len(env.VNRs_INFO))
        duration = predata.create_virtual_network_lift_time(size=len(env.VNRs_INFO), lambda_=1/config.VNR_DURATION_MEAN_RATE)
        
        # 记录所有VN的相关信息
        for i in range(len(env.VNRs_INFO)):
            vnr = env.VNRs_INFO[i]
            vn_link = vnr.net.edges.data("bandwidth")
            vn_node = [node[1] for node in vnr.net.nodes.data("CPU")]

            # vn_life = vnr.duration # 生存周期
            # VN_Life.append([i, vn_life[i], 0, 0])

            env.VNRs_INFO[i].duration = duration[i]  # 生存周期
            VN_Life.append([i, duration[i], 0, 0])

            VN_Node.update({i: vn_node})
            VN_Link.update({i: list(vn_link)})

            env.VNRs_INFO[i].time_step_arrival = VN_Arrive_Time[i] # 到达时间

        # print("虚拟网络请求总数量: {}".format(len(env.VNRs_INFO)))
        data = {
            "SN_Link": SN_Link,
            "SN_Node": SN_Node,
            "VN_Node": VN_Node,
            "VN_Link": VN_Link,
            "VN_Life": {0: VN_Life},
            "VN_Arrive_Time": VN_Arrive_Time,
        }
        from data_loader import predata
        predata.save_data(data, len(VN_Link))

        # 每个智能体的环境
        envs = []
        for agent_id in range(len(agents)):
            if agent_id == 0:
                envs.append(env)
            else:
                envs.append(copy.deepcopy(env))

        msg = "RUN: {0} STARTED".format(run + 1)
        logger.info(msg), print(msg)

        # 每个智能体的初始状态
        states = []
        for agent_id in range(len(agents)):
            states.append(envs[agent_id].reset())

        # 是否结束整个VNE流程
        done = False
        
        time_step = 0

        while not done:
            time_step += 1
            
            # 
            for agent_id in range(len(agents)):
                before_action_msg = "state {0} | ".format(repr(states[agent_id]))
                before_action_simple_msg = "state {0} | ".format(states[agent_id])
                logger.info("{0} {1}".format(
                    utils.run_agent_step_prefix(run + 1, agent_labels[agent_id], time_step), before_action_msg
                ))

                # action: 每个智能体的单次动作
                # 包括:
                # 1. 暂时未成功映射的vn 2. 成功映射的vn的节点和链路映射结果 3. 节点映射阶段失败数量 4. 链路映射阶段失败数量
                action = agents[agent_id].get_action(states[agent_id])

                action_msg = "act. {0:30} |".format(
                    str(action) if action.vnrs_embedding is not None and action.vnrs_postponement is not None else " - "
                )
                logger.info("{0} {1}".format(
                    utils.run_agent_step_prefix(run + 1, agent_labels[agent_id], time_step), action_msg
                ))

                # 执行action, 返回(下一状态, 奖励, 是否结束, (收益, 接受率, R/C, 链路失败比例))
                next_state, reward, done, info = envs[agent_id].step(action)

                elapsed_time = time.time() - run_start_ts
                after_action_msg = "reward {0:6.1f} | revenue {1:6.1f} | acc. ratio {2:4.2f} | " \
                                   "r/c ratio {3:4.2f} | {4}".format(
                    reward, info['revenue'], info['acceptance_ratio'], info['rc_ratio'],
                    time.strftime("%Hh %Mm %Ss", time.gmtime(elapsed_time)),
                )

                after_action_msg += " | {0:3.1f} steps/sec.".format(time_step / elapsed_time)
                logger.info("{0} {1}".format(
                    utils.run_agent_step_prefix(run + 1, agent_labels[agent_id], time_step), after_action_msg
                ))

                print("{0} {1} {2} {3}".format(
                    utils.run_agent_step_prefix(run + 1, agent_labels[agent_id], time_step),
                    before_action_simple_msg,
                    action_msg,
                    after_action_msg
                ))

                states[agent_id] = next_state
                
                # 记录相关评价指标
                performance_revenue[agent_id, time_step] += info['revenue']
                performance_acceptance_ratio[agent_id, time_step] += info['acceptance_ratio']
                performance_rc_ratio[agent_id, time_step] += info['rc_ratio']
                performance_link_fail_ratio[agent_id, time_step] += \
                    info['link_embedding_fails_against_total_fails_ratio']

            # if time_step > config.FIGURE_START_TIME_STEP - 1 and time_step % 1000 == 0:
            #     draw_performance(
            #         agents=agents, agent_labels=agent_labels, run=run, time_step=time_step,
            #         performance_revenue=performance_revenue / (run + 1),
            #         performance_acceptance_ratio=performance_acceptance_ratio / (run + 1),
            #         performance_rc_ratio=performance_rc_ratio / (run + 1),
            #         performance_link_fail_ratio=performance_link_fail_ratio / (run + 1),
            #     )

        time.sleep(5)
        
        # 保存智能体的评价指标为 "CSV"
        import pandas as pd
        for agent_id in range(len(agents)):
            df = pd.DataFrame()
            df["r/c"] = performance_rc_ratio[agent_id]
            df["revenue"] = performance_revenue[agent_id]/ (run + 1)
            df["accept_ratio"] = performance_acceptance_ratio[agent_id]
            df.to_csv("or_main/data/result/{}_{}.csv".format(agent_labels[agent_id], len(env.VNRs_INFO)))
            time.sleep(100)


        # 尝试画图
        try:
            draw_performance(
                agents=agents, agent_labels=agent_labels, run=run, time_step=time_step,
                performance_revenue=performance_revenue / (run + 1),
                performance_acceptance_ratio=performance_acceptance_ratio / (run + 1),
                performance_rc_ratio=performance_rc_ratio / (run + 1),
                performance_link_fail_ratio=performance_link_fail_ratio / (run + 1),
                send_image_to_slack=True
            )
        except:
            pass

        msg = "RUN: {0} FINISHED - ELAPSED TIME: {1}".format(
            run + 1, time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_ts))
        )
        logger.info(msg), print(msg)

def get_agents():
    """
    给每个算法设置一个智能体, 返回智能体集合
    """
    agents = []
    agent_labels = []

    for target_algorithm in config.target_algorithms:
        # if target_algorithm == config.ALGORITHMS.BASELINE.name:
        #     agents.append(
        #         BaselineVNEAgent(
        #             logger=logger,
        #             time_window_size=config.TIME_WINDOW_SIZE,
        #             agent_type=config.ALGORITHMS.BASELINE,
        #             type_of_virtual_node_ranking=config.TYPE_OF_VIRTUAL_NODE_RANKING.TYPE_2,
        #             allow_embedding_to_same_substrate_node=config.ALLOW_EMBEDDING_TO_SAME_SUBSTRATE_NODE,
        #             max_embedding_path_length=config.MAX_EMBEDDING_PATH_LENGTH
        #         )
        #     )
        #     agent_labels.append(config.ALGORITHMS.BASELINE.value)

        # if target_algorithm == config.ALGORITHMS.TOPOLOGY_AWARE_DEGREE.name:
        #     agents.append(
        #         TopologyAwareBaselineVNEAgent(
        #             beta=0.3,
        #             logger=logger,
        #             time_window_size=config.TIME_WINDOW_SIZE,
        #             agent_type=config.ALGORITHMS.BASELINE,
        #             type_of_virtual_node_ranking=config.TYPE_OF_VIRTUAL_NODE_RANKING.TYPE_1,
        #             allow_embedding_to_same_substrate_node=config.ALLOW_EMBEDDING_TO_SAME_SUBSTRATE_NODE,
        #             max_embedding_path_length=config.MAX_EMBEDDING_PATH_LENGTH
        #         )
        #     )
        #     agent_labels.append(config.ALGORITHMS.TOPOLOGY_AWARE_DEGREE.value)

        # if target_algorithm == config.ALGORITHMS.EGO_NETWORK.name:
        #     agents.append(
        #         EgoNetworkBasedVNEAgent(
        #             beta=0.9,
        #             logger=logger,
        #             time_window_size=config.TIME_WINDOW_SIZE,
        #             agent_type=config.ALGORITHMS.BASELINE,
        #             type_of_virtual_node_ranking=config.TYPE_OF_VIRTUAL_NODE_RANKING.TYPE_1,
        #             allow_embedding_to_same_substrate_node=config.ALLOW_EMBEDDING_TO_SAME_SUBSTRATE_NODE,
        #             max_embedding_path_length=config.MAX_EMBEDDING_PATH_LENGTH
        #         )
        #     )
        #     agent_labels.append(config.ALGORITHMS.EGO_NETWORK.value)

        # if target_algorithm == config.ALGORITHMS.DETERMINISTIC_VINE.name:
        #     agents.append(
        #         DeterministicVNEAgent(
        #             logger=logger,
        #             time_window_size=config.TIME_WINDOW_SIZE,
        #             agent_type=config.ALGORITHMS.BASELINE,
        #             type_of_virtual_node_ranking=config.TYPE_OF_VIRTUAL_NODE_RANKING.TYPE_2,
        #             allow_embedding_to_same_substrate_node=config.ALLOW_EMBEDDING_TO_SAME_SUBSTRATE_NODE,
        #             max_embedding_path_length=config.MAX_EMBEDDING_PATH_LENGTH
        #         )
        #     )
        #     agent_labels.append(config.ALGORITHMS.DETERMINISTIC_VINE.value)

        # if target_algorithm == config.ALGORITHMS.RANDOMIZED_VINE.name:
        #     agents.append(
        #         RandomizedVNEAgent(
        #             logger=logger,
        #             time_window_size=config.TIME_WINDOW_SIZE,
        #             agent_type=config.ALGORITHMS.BASELINE,
        #             type_of_virtual_node_ranking=config.TYPE_OF_VIRTUAL_NODE_RANKING.TYPE_2,
        #             allow_embedding_to_same_substrate_node=config.ALLOW_EMBEDDING_TO_SAME_SUBSTRATE_NODE,
        #             max_embedding_path_length=config.MAX_EMBEDDING_PATH_LENGTH
        #         )
        #     )
        #     agent_labels.append(config.ALGORITHMS.RANDOMIZED_VINE.value)

        # if target_algorithm == config.ALGORITHMS.TOPOLOGY_AWARE_NODE_RANKING.name:
        #     agents.append(
        #         TopologyAwareNodeRankingVNEAgent(
        #             beta=0.3,
        #             logger=logger,
        #             time_window_size=config.TIME_WINDOW_SIZE,
        #             agent_type=config.ALGORITHMS.BASELINE,
        #             type_of_virtual_node_ranking=config.TYPE_OF_VIRTUAL_NODE_RANKING.TYPE_2,
        #             allow_embedding_to_same_substrate_node=config.ALLOW_EMBEDDING_TO_SAME_SUBSTRATE_NODE,
        #             max_embedding_path_length=config.MAX_EMBEDDING_PATH_LENGTH
        #         )
        #     )
        #     agent_labels.append(config.ALGORITHMS.TOPOLOGY_AWARE_NODE_RANKING.value)

        # A3C_GCN算法
        if target_algorithm == config.ALGORITHMS.A3C_GCN.name:
            agents.append(
                A3C_GCN_VNEAgent(
                    local_model=None,
                    beta=0.3,
                    logger=logger,
                    time_window_size=config.TIME_WINDOW_SIZE,  # default = 1
                    agent_type=config.ALGORITHMS.BASELINE,  # default = "BL"
                    type_of_virtual_node_ranking=config.TYPE_OF_VIRTUAL_NODE_RANKING.TYPE_2,  # 1
                    allow_embedding_to_same_substrate_node=config.ALLOW_EMBEDDING_TO_SAME_SUBSTRATE_NODE,  # default = false
                    max_embedding_path_length=config.MAX_EMBEDDING_PATH_LENGTH  # default = 4
                )
            )
            agent_labels.append(config.ALGORITHMS.A3C_GCN.value)
            
        # if target_algorithm == config.ALGORITHMS.MCTS.name:
        #     agents.append(
        #         MCST_VNEAgent(
        #             beta=0.3,
        #             logger=logger,
        #             time_window_size=config.TIME_WINDOW_SIZE,
        #             agent_type=config.ALGORITHMS.BASELINE,
        #             type_of_virtual_node_ranking=config.TYPE_OF_VIRTUAL_NODE_RANKING.TYPE_2,
        #             allow_embedding_to_same_substrate_node=config.ALLOW_EMBEDDING_TO_SAME_SUBSTRATE_NODE,
        #             max_embedding_path_length=config.MAX_EMBEDDING_PATH_LENGTH
        #         )
        #     )
        #     agent_labels.append(config.ALGORITHMS.MCTS.value)
        
        
    return agents, agent_labels

def test():
    # agents, agent_labels = get_agents()
    # print(len(agents))
    # print(agent_labels)
    
    import pandas as pd
    
    df = pd.DataFrame()
    
    df["test"] = [i for i in range(100)]
    # df.to_csv("test.csv")
    df.to_csv("or_main/data/result/test.csv")
    

if __name__ == "__main__":
    main()
    
    # test()
    

