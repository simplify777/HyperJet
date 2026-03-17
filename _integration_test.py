"""
集成测试：验证 HyperJetNet 与真实 HyperJet 环境无缝对接。
运行：python _integration_test.py
"""
import torch
from tianshou.utils.net.common import ActorCritic
from tianshou.utils.net.discrete import Actor, Critic
from tianshou.policy import PPOPolicy

from environment.offloading_env import make_env
from environment.config.config import GNN_TYPE, USE_GRAPH_STATE, BUFFER_SIZE, BATCH_SIZE
from utils.utils import get_file_paths, get_gpu
from models import HyperJetNet

def main():
    device = get_gpu()
    print(f"[INFO] Using device: {device}")

    # 1. 构建真实环境
    training_paths, test_paths = get_file_paths()
    env, train_envs, test_envs = make_env(
        "HG", training_paths, test_paths,
        GNN_TYPE, use_graph_state=USE_GRAPH_STATE,
        device=device, use_cache=False
    )

    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n
    print(f"[INFO] obs_shape = {obs_shape}, n_actions = {n_actions}")

    # 2. 实例化 HyperJetNet
    net_a = HyperJetNet(state_shape=obs_shape, device=device)
    net_c = HyperJetNet(state_shape=obs_shape, device=device)

    actor  = Actor(net_a, n_actions, device=device).to(device)
    critic = Critic(net_c, device=device).to(device)
    actor_critic = ActorCritic(actor, critic)

    optim = torch.optim.Adam(actor_critic.parameters(), lr=1e-3)
    dist  = torch.distributions.Categorical

    policy = PPOPolicy(
        actor=actor, critic=critic, optim=optim, dist_fn=dist,
        action_space=env.action_space,
        observation_space=env.observation_space.shape
    )

    # 3. 采样一个 batch 做一次 forward 验证
    obs, info = env.reset()
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, hidden = net_a(obs_tensor, state=None, info=info)

    print(f"[PASS] Forward OK  features: {logits.shape}, hidden: {hidden.shape}")
    assert logits.shape[-1] == net_a.embed_dim, f"features dim mismatch: {logits.shape}"
    print("[PASS] Integration test passed!")

if __name__ == "__main__":
    main()
