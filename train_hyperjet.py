import os
import torch
from torch.utils.tensorboard import SummaryWriter
from tianshou.data import VectorReplayBuffer
from tianshou.policy import PPOPolicy
from tianshou.utils.net.common import Net, ActorCritic
from tianshou.trainer import onpolicy_trainer

from environment.config.config import *
from environment.env_collect import EnvCollector
from environment.offloading_env import make_env
from utils.utils import get_file_paths, get_gpu
# 引入作者写好的日志工具！
from utils.logger import DataLogger

if __name__ == "__main__":
    print("🚀 初始化训练环境与模型...")
    # 1. 准备数据和环境
    training_paths, test_paths = get_file_paths()
    env, train_envs, test_envs = make_env("HG", training_paths, test_paths, GNN_TYPE, use_graph_state=USE_GRAPH_STATE,
                                          device="cpu", use_cache=False)

    # 2. 构建 Actor-Critic 模型
    # actor = Net(env.observation_space.shape, hidden_sizes=[128, 64], device="cpu")
    # critic = Net(env.observation_space.shape, hidden_sizes=[128, 64], device="cpu")
    # actor_critic = ActorCritic(actor, critic)
    from tianshou.utils.net.discrete import Actor, Critic  # 👈 直接导在这里，清晰明了

    base_actor = Net(env.observation_space.shape, hidden_sizes=[128, 64], device="cpu")
    base_critic = Net(env.observation_space.shape, hidden_sizes=[128, 64], device="cpu")

    actor = Actor(base_actor, env.action_space.n, device="cpu")
    critic = Critic(base_critic, device="cpu")

    optim = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=1e-3)
    # optim = torch.optim.Adam(actor_critic.parameters(), lr=1e-3)
    dist = torch.distributions.Categorical

    # 3. 实例化 PPO 策略 (AI 大脑)
    policy = PPOPolicy(actor=actor, critic=critic, optim=optim, dist_fn=dist,
                       action_space=env.action_space, observation_space=env.observation_space.shape)

    # 4. 配置收集器 (注意：这里不再传入 random=True，强制使用神经网络做决策)
    train_collector = EnvCollector(policy, train_envs, VectorReplayBuffer(BUFFER_SIZE, len(train_envs)))
    test_collector = EnvCollector(policy, test_envs)

    # 5. 配置 TensorBoard 日志记录器 (核心！用来画图的！)
    log_path = os.path.join("logs", "ppo_hyperjet")
    writer = SummaryWriter(log_path)
    logger = DataLogger(writer)  # 使用原作者自定义的 Logger

    print("🔥 引擎点火，开始 300 轮强化学习训练！")
    # 6. 启动 Tianshou 训练主循环
    result = onpolicy_trainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=EPOCH,  # 300轮
        step_per_epoch=STEP_PER_EPOCH,
        repeat_per_collect=REPEAT_PER_COLLECT,
        episode_per_test=NUM_TEST_HG,
        batch_size=BATCH_SIZE,
        step_per_collect=BATCH_SIZE,
        logger=logger  # 实时记录画图数据
    )

    print(f"\n🎉 训练彻底完成！")
    print(result)

    # 7. 保存训练好的最强大脑
    model_save_path = "hyperjet_best_policy.pth"
    torch.save(policy.state_dict(), model_save_path)
    print(f"💾 模型已保存至: {model_save_path}")