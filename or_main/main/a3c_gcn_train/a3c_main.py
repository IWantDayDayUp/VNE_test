import os, sys
import torch.multiprocessing as mp
import wandb

current_path = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(current_path, os.pardir))
root_dir = os.path.abspath(os.path.join(PROJECT_HOME, "../.."))

if root_dir not in sys.path:
    sys.path.append(root_dir)

if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

from a3c_worker import Worker
from or_main.algorithms.model.A3C import A3C_Model
from or_main.algorithms.model.utils import SharedAdam, draw_rl_train_performance, set_wandb
from common import config

WANDB = False

def main():
    global_net = A3C_Model(
        chev_conv_state_dim=config.NUM_SUBSTRATE_FEATURES,  # 5
        action_dim=config.SUBSTRATE_NODES  # 100
    )
    global_net.share_memory()  # share the global parameters in multiprocessing
    optimizer = SharedAdam(global_net.parameters(), lr=2e-4, betas=(0.92, 0.999))  # global optimizer
    
    # 该模式下, 主进程的参数要通过传参, 才可把主进程的参数传入子进程中
    # 子进程拷贝主进程的参数进行使用, 不改变主进程参数
    mp.set_start_method('spawn')

    # 共享变量
    global_episode = mp.Value('i', 0)  # 'i' == signed int
    global_episode_reward = mp.Value('d', 0.0)  # 'd' = double
    
    # 多线程队列
    message_queue = mp.Queue()

    # 完美入门轻量级可视化工具
    if WANDB:
        set_wandb(global_net)

    # 创建并启动线程, 默认有3个线程
    workers = [
        Worker(
            global_net, optimizer, global_episode, global_episode_reward, message_queue, idx
        ) for idx in range(config.NUM_WORKERS)  # default = 3
    ]
    for w in workers:
        w.start()

    global_episode_rewards = []  # record episode reward to plot
    critic_losses = []
    actor_objectives = []
    train_info_dict = {}
    while True:
        message = message_queue.get()
        if message is not None:
            global_episode_reward_from_worker, critic_loss, actor_objective = message
            global_episode_rewards.append(global_episode_reward_from_worker)
            critic_losses.append(critic_loss)
            actor_objectives.append(actor_objective)

            if WANDB:
                train_info_dict["train global episode reward"] = global_episode_reward_from_worker
                train_info_dict["train critic loss"] = critic_loss
                train_info_dict["train actor objectives"] = actor_objective

                wandb.log(train_info_dict)
        else:
            break

    # 使所有线程进入 '阻塞' 状态, 待所有线程都执行结束, 再一起继续执行
    for w in workers:
        w.join()

    draw_rl_train_performance(
        # config.MAX_EPISODES,  # 50000
        300,  # 50000
        global_episode_rewards,
        critic_losses,
        actor_objectives,
        config.rl_train_graph_save_path,  # "or_main/out/rl_train_graphs"
        period=10
    )


if __name__ == "__main__":
    main()
