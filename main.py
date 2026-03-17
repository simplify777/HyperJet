import os

import gymnasium as gym
import torch
from tianshou.data import VectorReplayBuffer
from tianshou.env import BaseVectorEnv
from tianshou.policy import PPOPolicy
from tianshou.utils.net.common import Net, ActorCritic

from environment.config.config import *
from environment.env_collect import EnvCollector
from environment.offloading_env import make_env

from environment.config import config
from utils.utils import is_float, get_mean
from utils.utils import get_file_paths, get_gpu

utilization_rates = []
end_times = []
makespans = []


def collect_test(policy, envs, name=config.GNN_TYPE, random=False, env_func=None):
    # _env = get_env_from_vector_envs(envs)
    # _env.reset()
    # policy.eval()
    if env_func:
        utilization_rates.append([])
        end_times.append([])
    collector = EnvCollector(policy, envs, VectorReplayBuffer(BUFFER_SIZE, len(envs)), env_func=env_func)
    collector_stats = collector.collect(random=random, random_way=name, n_episode=BATCH_SIZE)
    print(f"{name}:")
    for k, v in collector_stats.items():
        if not k.endswith("s"):
            print(f"{k}: {v}", end=' ')
    print()
    if env_func:
        mean_utilization_rate = get_mean(utilization_rates[-1])
        mean_end_time = get_mean(end_times[-1])
        print(f"Mean utilization rate: {mean_utilization_rate}")
        print(f"Mean end time: {mean_end_time}")
        print(f"Makespan: {makespans[-1]}")

def init_env(env, gnn_type=GNN_TYPE):
    use_graph_state = USE_GRAPH_STATE
    if isinstance(env, gym.Env):
        setattr(env, "use_graph_state", use_graph_state)
        setattr(env, "_init_state", None)
        env.reset()
    elif isinstance(env, BaseVectorEnv):
        env.set_env_attr("use_graph_state", use_graph_state)
        env.set_env_attr("_init_state", None)
        env.set_env_attr("_gnn_type", gnn_type)
        env.set_env_attr("_HG", None)

def set_env_attr(env, key, value):
    if isinstance(env, gym.Env):
        setattr(env, key, value)
    elif isinstance(env, BaseVectorEnv):
        env.set_env_attr(key, value)

def env_func(env_infos):
    utilizations = []
    _end_times = []
    sum_makespan = 0
    for env_info in env_infos:
        acts = env_info["acts"]
        exec_time = env_info["exec_time"]
        end_time = list(env_info["end_time"].values())
        makespan = env_info["episode_time"]
        sum_makespan += makespan
        assert makespan == max(end_time), f"makespan: {makespan} != end_time: {max(end_time)}"
        utilization = [0] * NUM_RESOURCE_CLUSTER
        mx_end_time = env_info["curr_time"]

        for task_id, task_exec_time in exec_time.items():
            utilization[acts[task_id]] += task_exec_time
        for i in range(NUM_RESOURCE_CLUSTER):
            utilization[i] /= mx_end_time
        utilizations.append(utilization)
        _end_times.append(sorted(end_time))
    utilization_rates[-1].append(get_mean(utilizations))
    end_times[-1].append(get_mean(_end_times))
    makespans.append(sum_makespan / len(env_infos))


if __name__ == "__main__":
    from tianshou.utils.net.discrete import Actor, Critic
    from tianshou.trainer import onpolicy_trainer
    from models import HyperJetNet  # 导入我们手写的超图网络

    training_paths, test_paths = get_file_paths()

    env, train_envs, test_envs = make_env("HG", training_paths, test_paths, GNN_TYPE, use_graph_state=USE_GRAPH_STATE,
                                          device=get_gpu(), use_cache=False)

    print("[INFO] Constructing HyperJet neural network (HGNN + Seq2Seq)...")
    # 实例化我们的超图网络，由于目前 PyTorch 对 RTX 50 系不支持，先强制用 cpu
    net_a = HyperJetNet(state_shape=env.observation_space.shape, device="cpu")
    net_c = HyperJetNet(state_shape=env.observation_space.shape, device="cpu")
    
    actor = Actor(net_a, env.action_space.n, device="cpu").to("cpu")
    critic = Critic(net_c, device="cpu").to("cpu")
    actor_critic = ActorCritic(actor, critic)
    
    optim = torch.optim.Adam(actor_critic.parameters(), lr=1e-3)
    dist = torch.distributions.Categorical

    policy = PPOPolicy(actor=actor, critic=critic, optim=optim, dist_fn=dist, 
                       action_space=env.action_space, observation_space=env.observation_space.shape)

    # 跑一下传统的沙袋算法做个对比
    collect_test(policy, train_envs, name="heft", random=True, env_func=env_func)
    collect_test(policy, train_envs, name="greedy", random=True, env_func=env_func)
    collect_test(policy, train_envs, name="random", random=True, env_func=env_func)

    # =============== 核心：开始训练我们的 AI ===============
    print("\n[START] 开始训练 PPO-HyperJet (等待进度条跑完)...")
    train_collector = EnvCollector(policy, train_envs, VectorReplayBuffer(BUFFER_SIZE, len(train_envs)))
    test_collector = EnvCollector(policy, test_envs, VectorReplayBuffer(BUFFER_SIZE, len(test_envs)))

    test_rewards_history = []
    original_collect = test_collector.collect
    def my_collect(*args, **kwargs):
        res = original_collect(*args, **kwargs)
        test_rewards_history.append(res["rew"])
        return res
    test_collector.collect = my_collect

    result = onpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        max_epoch=50,         # 修改为要求的 50 轮
        step_per_epoch=1000,
        repeat_per_collect=2,
        episode_per_test=10,
        batch_size=16,
        step_per_collect=100,
    )
    
    print("\n[DONE] 训练完成！有请 AI 完全体登场测试：")
    collect_test(policy, train_envs, name="PPO-AI-HyperJet", random=False, env_func=env_func)

    import matplotlib.pyplot as plt

    # ----- 图1：训练收敛曲线 -----
    plt.figure(figsize=(10, 5))
    epochs = range(1, len(test_rewards_history) + 1)
    plt.plot(epochs, test_rewards_history, marker='o', linestyle='-', color='#d62728', markersize=4, linewidth=2)
    plt.title('PPO Learning Curve (Test Reward)', fontsize=14, fontweight='bold')
    plt.xlabel('Evaluation Steps', fontsize=12)
    plt.ylabel('Average Reward', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('learning_curve.png', dpi=300)
    print("\n[INFO] 收敛曲线已保存至 learning_curve.png")

    # ----- 图2：Makespan 对比柱状图 -----
    plt.figure(figsize=(8, 6))
    labels = ['HEFT', 'Greedy', 'Random', 'PPO-HyperJet']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # 提取 makespans 列表中的前四个分别是 heft, greedy, random, PPO
    algo_makespans = makespans[:4]
    
    bars = plt.bar(labels, algo_makespans, color=colors, edgecolor='black', linewidth=1.2)
    plt.title('Makespan Comparison (Lower is Better)', fontsize=14, fontweight='bold')
    plt.ylabel('Average Makespan', fontsize=12)
    
    # 为柱状图添加数值标签
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + (max(algo_makespans)*0.01), f'{yval:.4f}', 
                 va='bottom', ha='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig('makespan_comparison.png', dpi=300)
    print("[INFO] 对比柱状图已保存至 makespan_comparison.png")
