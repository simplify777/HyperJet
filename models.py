"""
models.py — HyperJet 核心网络模块
=====================================
包含：
  - HGNNLayer       : 单层超图神经网络（对应论文公式 20/21）
  - HGNNEncoder     : 多层 HGNN 编码器（Fig. 7 完整流程）
  - Seq2SeqDecoder  : GRU + 公式 22 注意力的序列调度解码器（Fig. 9）
  - HyperJetNet     : 完整 Actor-Critic 网络（兼容 tianshou PPOPolicy）

论文对应：
  HyperJet: Joint Communication and Computation Scheduling
  for Hypergraph Tasks in Distributed Edge Computing

硬件目标：RTX 5060（CUDA 12.9）—— 所有张量均通过 .to(device) 灵活挂载。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from environment.config.config import (
    EMBED_SIZE,
    NUM_HG_TASKS,
    NUM_HG,
    NUM_HG_EDGES,
    NUM_RESOURCE_CLUSTER,
    USE_GRAPH_STATE,
)

# ============================================================
#  全局尺寸常量（与 offloading_env.py 中 init_state 一致）
# ============================================================
# 超图关联矩阵（incidence matrix）列数 = NUM_HG × NUM_HG_EDGES
_INCIDENCE_COLS: int = NUM_HG * NUM_HG_EDGES   # 默认 40
# 单步观测的节点数
_N_TASKS: int = NUM_HG * NUM_HG_TASKS           # 默认 20
# 完整状态特征维度 M = incidence_cols + 2*resource + 8
_STATE_DIM: int = _INCIDENCE_COLS + NUM_RESOURCE_CLUSTER * 2 + 8  # 默认 56


# ─────────────────────────────────────────────────────────────
#  辅助：安全求逆向量（避免除以零）
# ─────────────────────────────────────────────────────────────
def _safe_inv(vec: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """对一维向量逐元素求倒数，对接近零的项用 eps 保护。"""
    return 1.0 / (vec + eps)


# ─────────────────────────────────────────────────────────────
#  辅助：安全求逆平方根向量
# ─────────────────────────────────────────────────────────────
def _safe_inv_sqrt(vec: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """对一维向量逐元素求 x^{-1/2}，对接近零的项用 eps 保护。"""
    return 1.0 / (torch.sqrt(vec) + eps)


# ╔══════════════════════════════════════════════════════════════╗
# ║               核心模块一：HGNNLayer                         ║
# ║  严格对照公式 20/21：                                        ║
# ║  X_y = σ( D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2} X Θ )   ║
# ╚══════════════════════════════════════════════════════════════╝
class HGNNLayer(nn.Module):
    """
    单层超图神经网络卷积层（HGNN Layer）。

    对应论文公式 (20) 和 (21)：
        X_y^h = σ( D_v^{-1/2} · H^h · W^h · D_e^{-1} · H^{hT} · D_v^{-1/2} · X_{y-1}^h · Θ_{y-1} )

    参数说明：
        in_features  : 输入节点特征维度（X 的列数）
        out_features : 输出节点特征维度（即 Θ 的输出维度）
        use_bias     : 是否在线性变换 Θ 中加偏置项
    """

    def __init__(self, in_features: int, out_features: int, use_bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # ── Θ（论文中的可学习投影矩阵）──────────────────────────
        # 对应公式中的 "X_{y-1} · Θ_{y-1}"
        # 先做线性投影再传播，等价于先传播再投影（但先投影更节省显存）
        self.theta = nn.Linear(in_features, out_features, bias=use_bias)

        # ── W^h（超边权重矩阵）────────────────────────────────
        # 对应公式中的 "W^h"，这里用对角近似（学习每条超边的标量权重）
        # 注：full W^h 为 E×E 矩阵，但实践中常退化为每超边一个标量
        # 此处用 nn.Parameter 形式保持可训练性
        self.W = None  # 延迟初始化（在 forward 中根据 E 动态创建）

        nn.init.xavier_uniform_(self.theta.weight)

    # ────────────────────────────────────────────────────────
    def _get_W(self, E: int, device: torch.device) -> torch.Tensor:
        """
        懒初始化超边权重矩阵 W（对角矩阵，shape [E, E]）。
        若已存在且大小匹配则直接复用，否则重新创建。
        """
        if self.W is None or self.W.shape[0] != E:
            # W^h 初始化为单位矩阵（不改变超边信息流）
            w_diag = torch.ones(E, device=device, dtype=torch.float32)
            # 注册为 Parameter 使其参与反向传播
            self.W = nn.Parameter(w_diag)
            self.register_parameter(f"W_{E}", self.W)
        return torch.diag(self.W.to(device))  # shape: [E, E]

    # ────────────────────────────────────────────────────────
    def forward(
        self,
        X: torch.Tensor,      # 节点特征矩阵，shape: [N, in_features]
        H: torch.Tensor,      # 超图关联矩阵，shape: [N, E]，H[i,j]=1 表示节点 i 属于超边 j
    ) -> torch.Tensor:
        """
        前向传播，严格按公式 (21) 步骤展开，每步注释对应公式项。

        输入：
            X : [N, F_in]  — 当前层节点特征
            H : [N, E]     — 超图关联矩阵（incidence matrix）

        输出：
            X_out : [N, F_out] — 经超图卷积后的节点特征
        """
        device = X.device
        N, F_in = X.shape
        _, E = H.shape

        H = H.to(device=device, dtype=torch.float32)   # 确保类型对齐

        # ── 步骤 1：计算顶点度向量 d_v = H · 1_E ─────────────
        # d_v[i] = 该节点 i 参与的超边数量
        # 对应公式中 D_v 的对角元素
        d_v = H.sum(dim=1)          # shape: [N]
        # D_v^{-1/2}_{ii} = 1 / sqrt(d_v[i])
        d_v_invsqrt = _safe_inv_sqrt(d_v)   # shape: [N]

        # ── 步骤 2：计算超边度向量 d_e = H^T · 1_N ───────────
        # d_e[j] = 第 j 条超边包含的节点数量
        # 对应公式中 D_e 的对角元素
        d_e = H.sum(dim=0)          # shape: [E]
        # D_e^{-1}_{jj} = 1 / d_e[j]
        d_e_inv = _safe_inv(d_e)    # shape: [E]

        # ── 步骤 3：构造 W^h（超边权重对角矩阵）─────────────
        # 对应公式中的 W^h，shape: [E, E]
        W = self._get_W(E, device)  # shape: [E, E]

        # ── 步骤 4：计算 Θ（线性投影）────────────────────────
        # 对应公式末尾的 "X_{y-1} · Θ_{y-1}"
        # shape: [N, F_out]
        X_theta = self.theta(X)     # [N, F_in] → [N, F_out]

        # ── 步骤 5：D_v^{-1/2} · X_theta（左乘顶点度缩放）──
        # 等价于对每行 i 乘以 d_v_invsqrt[i]
        # shape: [N, F_out]
        X_scaled = d_v_invsqrt.unsqueeze(1) * X_theta   # [N, F_out]

        # ── 步骤 6：H^T · (D_v^{-1/2} X Θ)（超边聚合）────
        # 将各节点特征聚合到所属超边
        # shape: [E, F_out]
        E_agg = H.t() @ X_scaled   # [E, N] × [N, F_out] = [E, F_out]

        # ── 步骤 7：D_e^{-1} · E_agg（超边归一化）────────
        # 等价于对每行 j 乘以 d_e_inv[j]
        # shape: [E, F_out]
        E_norm = d_e_inv.unsqueeze(1) * E_agg   # [E, F_out]

        # ── 步骤 8：W^h · E_norm（超边权重加权）──────────
        # shape: [E, F_out]
        E_weighted = W @ E_norm     # [E, E] × [E, F_out] = [E, F_out]

        # ── 步骤 9：H · E_weighted（节点特征更新/广播）──
        # 将超边信息广播回各节点
        # shape: [N, F_out]
        X_agg = H @ E_weighted      # [N, E] × [E, F_out] = [N, F_out]

        # ── 步骤 10：D_v^{-1/2} · X_agg（右乘顶点度缩放）─
        # 再次用顶点度归一化，对应公式最外层的 D_v^{-1/2}
        # shape: [N, F_out]
        X_out_pre = d_v_invsqrt.unsqueeze(1) * X_agg   # [N, F_out]

        # ── 步骤 11：激活函数 σ（论文用 ReLU）─────────────
        # 对应公式中的 σ(·)
        X_out = F.relu(X_out_pre)   # [N, F_out]

        return X_out


# ╔══════════════════════════════════════════════════════════════╗
# ║               核心模块一（续）：HGNNEncoder                 ║
# ║  将多个 HGNNLayer 堆叠，输出节点嵌入（Fig. 7 完整流程）     ║
# ╚══════════════════════════════════════════════════════════════╝
class HGNNEncoder(nn.Module):
    """
    多层超图神经网络编码器。

    对应 Fig. 7 所示流程：
        Graph Feature → Node Feature Transform
                      → Edge Feature Gathering
                      → Node Feature Aggregating
                      → Vertex Feature（输出）

    通过堆叠 `num_layers` 个 HGNNLayer 实现逐层精炼节点表示。

    参数：
        in_dim    : 输入节点特征维度（等于状态矩阵的特征数 M）
        hid_dim   : 中间隐层维度
        out_dim   : 最终节点嵌入维度（EMBED_SIZE）
        num_layers: HGNN 层数（论文未指定，默认 2 层足够）
        dropout   : Dropout 正则化概率
    """

    def __init__(
        self,
        in_dim: int,
        hid_dim: int = EMBED_SIZE,
        out_dim: int = EMBED_SIZE,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        # 构建 HGNN 层序列
        layers = []
        for i in range(num_layers):
            _in  = in_dim  if i == 0 else hid_dim
            _out = out_dim if i == num_layers - 1 else hid_dim
            layers.append(HGNNLayer(_in, _out))
        self.layers = nn.ModuleList(layers)

        # 输出层归一化（稳定训练）
        self.layer_norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        X: torch.Tensor,    # 节点特征，shape: [N, in_dim]
        H: torch.Tensor,    # 超图关联矩阵，shape: [N, E]
    ) -> torch.Tensor:
        """
        多层前向传播。

        输出：
            Z : [N, out_dim] — 每个任务节点的超图嵌入向量
        """
        Z = X
        for i, layer in enumerate(self.layers):
            Z = layer(Z, H)                        # 超图卷积
            if i < self.num_layers - 1:
                Z = self.dropout(Z)                # 中间层 Dropout
        Z = self.layer_norm(Z)                     # 最终归一化
        return Z  # [N, out_dim]


# ╔══════════════════════════════════════════════════════════════╗
# ║           核心模块二：Seq2SeqDecoder                        ║
# ║  对应 Fig. 9（Encoder→Context Vector→Decoder）              ║
# ║  + 公式 22（注意力权重 μ_{ji}）                             ║
# ╚══════════════════════════════════════════════════════════════╝
class Seq2SeqDecoder(nn.Module):
    """
    序列到序列调度解码器，含 GRU 编码器和公式 22 注意力机制。

    架构（对应 Fig. 9 右侧部分）：
        [e_1, e_2, ..., e_L]  ←── HGNN 输出的节点嵌入序列
              ↓ GRU Encoder
          Context Vector (h_L)
              ↓ GRU Decoder（逐步解码每个任务的卸载动作）
          d_{j-1} → 公式22注意力 → 加权上下文 → 动作 logits

    公式 22：
        μ_{ji} = exp(Dist(d_{j-1}, e_i)) / Σ_{k=1}^{L} exp(Dist(d_{j-1}, e_k))
    其中 Dist 用点积实现（Scaled Dot-Product Attention）。

    参数：
        embed_dim   : 节点嵌入维度（= HGNN 输出维度 = EMBED_SIZE）
        hidden_dim  : GRU 隐状态维度
        n_actions   : 动作空间大小（= NUM_RESOURCE_CLUSTER = 4）
    """

    def __init__(
        self,
        embed_dim: int = EMBED_SIZE,
        hidden_dim: int = EMBED_SIZE,
        n_actions: int = NUM_RESOURCE_CLUSTER,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_actions = n_actions

        # ── GRU Encoder（对应 Fig. 9 中的 "Encoder"）────────
        # 将 L 个节点嵌入序列编码为 Context Vector
        self.gru_encoder = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,    # 输入 shape: [B, L, embed_dim]
        )

        # ── GRU Decoder（对应 Fig. 9 中的 "Decoder"）────────
        # 每步以"注意力加权上下文 + 当前任务嵌入"作为输入
        self.gru_decoder = nn.GRUCell(
            input_size=embed_dim * 2,  # context_vec + current_node_embed
            hidden_size=hidden_dim,
        )

        # ── 注意力评分层（用于公式 22 的 Dist 计算）──────────
        # 将 d_{j-1}（decoder 隐状态）投影到 embed_dim 空间以便与 e_i 做点积
        self.attn_proj = nn.Linear(hidden_dim, embed_dim, bias=False)

        # ── 输出特征直接供 Tianshou 的 Actor/Critic 线性层处理 ──
        # Tianshou 会接手最后的 Linear 层，所以这里不再需要 self.output_layer

    # ────────────────────────────────────────────────────────
    def _attention(
        self,
        d_prev: torch.Tensor,    # decoder 上一步隐状态，shape: [B, hidden_dim]
        encoder_outs: torch.Tensor,  # encoder 所有时步输出，shape: [B, L, embed_dim]
    ) -> torch.Tensor:
        """
        计算公式 (22) 注意力权重 μ_{ji}。

        公式：
            μ_{ji} = exp(Dist(d_{j-1}, e_i)) / Σ_k exp(Dist(d_{j-1}, e_k))

        这里 Dist(d, e) = (W_q · d)^T · e / sqrt(embed_dim)（Scaled Dot-Product）

        输出：
            context : [B, embed_dim] — 加权上下文向量 Σ_i μ_{ji} · e_i
            weights : [B, L]         — 注意力权重（可用于可视化/分析）
        """
        B, L, D = encoder_outs.shape

        # 将 decoder 隐状态投影到 embed_dim 空间 → query 向量
        # shape: [B, embed_dim]
        query = self.attn_proj(d_prev)

        # Scaled Dot-Product：query · encoder_out^T / sqrt(D)
        # query unsqueeze → [B, 1, embed_dim]
        # encoder_outs   → [B, L, embed_dim]
        # scores         → [B, 1, L] → squeeze → [B, L]
        scores = torch.bmm(
            query.unsqueeze(1),          # [B, 1, embed_dim]
            encoder_outs.transpose(1, 2) # [B, embed_dim, L]
        ).squeeze(1) / (D ** 0.5)        # [B, L]，对应论文中 Dist(d_{j-1}, e_i)

        # Softmax 归一化，得到公式 (22) 中的 μ_{ji}
        weights = F.softmax(scores, dim=-1)  # [B, L]

        # 计算加权上下文向量 Σ_i μ_{ji} · e_i
        # weights unsqueeze → [B, 1, L]
        # encoder_outs      → [B, L, embed_dim]
        # context           → [B, embed_dim]
        context = torch.bmm(
            weights.unsqueeze(1),   # [B, 1, L]
            encoder_outs            # [B, L, embed_dim]
        ).squeeze(1)                # [B, embed_dim]

        return context, weights

    # ────────────────────────────────────────────────────────
    def forward(
        self,
        node_embeds: torch.Tensor,  # HGNN 输出，shape: [B, N, embed_dim]
        task_idx: int = 0,          # 当前需要决策的任务序号（topsort 顺序）
        hidden: torch.Tensor = None # decoder GRU 隐状态（跨步传递）
    ):
        """
        前向传播（单步解码一个任务的动作 logits）。

        输入：
            node_embeds : [B, N, embed_dim] — 所有节点嵌入（来自 HGNN）
            task_idx    : int              — 当前决策任务在序列中的位置
            hidden      : [B, hidden_dim]  — 上一步 decoder GRU 隐状态

        输出：
            features : [B, hidden_dim] — 任务的上下文表征（送给 Tianshou Actor）
            hidden   : [B, hidden_dim] — 更新后的 GRU 隐状态（传递给下一步）
        """
        B, N, D = node_embeds.shape
        device = node_embeds.device

        # ── Encoder：将全部节点嵌入编码为上下文序列 ──────────
        # 对应 Fig. 9 中 e_1…e_i 经 Encoder 得到 Context Vector
        # encoder_outs shape: [B, N, hidden_dim]
        # h_n          shape: [1, B, hidden_dim]
        encoder_outs, h_n = self.gru_encoder(node_embeds)

        # 初始化 decoder 隐状态（首步用 encoder 最终隐状态）
        if hidden is None:
            hidden = h_n.squeeze(0)   # [B, hidden_dim]
        hidden = hidden.to(device)

        # ── 公式 22：计算注意力权重 μ_{ji} ──────────────────
        context, attn_weights = self._attention(hidden, encoder_outs)
        # context : [B, embed_dim]

        # ── 拼接上下文与当前任务嵌入 ─────────────────────────
        # 对应 Fig. 9 中 Decoder 输入 = context vector + 当前节点特征
        cur_embed = node_embeds[:, task_idx, :]     # [B, embed_dim]
        decoder_input = torch.cat([context, cur_embed], dim=-1)  # [B, embed_dim*2]

        # ── GRUCell：更新 decoder 隐状态 ─────────────────────
        hidden = self.gru_decoder(decoder_input, hidden)  # [B, hidden_dim]

        # ── 不需要输出 logits，直接输出特征供 tianshou 使用 ──
        features = hidden                   # [B, hidden_dim]

        return features, hidden


# ╔══════════════════════════════════════════════════════════════╗
# ║           完整网络：HyperJetNet                             ║
# ║  将 HGNN + Seq2Seq 组合，兼容 tianshou Net 接口             ║
# ║  可直接用于 tianshou Actor / Critic 的 preprocess_net       ║
# ╚══════════════════════════════════════════════════════════════╝
class HyperJetNet(nn.Module):
    """
    HyperJet 完整网络（Fig. 9 全流程）。

    流水线：
        obs [B, N, M]
          → 切分：X = obs[..., incidence_cols:]    （节点特征）
                  H = obs[..., :incidence_cols]     （关联矩阵）
          → HGNNEncoder(X, H)                       （超图编码）
          → Seq2SeqDecoder(node_embeds, task_idx)   （序列解码）
          → features [B, embed_dim]

    与 tianshou 的对接：
        - tianshou 的 Net 接口要求 forward(obs, state, info) → (logits/values, state)
        - Actor(HyperJetNet, n_actions) : 自动在 features 后接 Linear 到 n_actions 并 softmax
        - Critic(HyperJetNet)           : 自动在 features 后接 Linear 到 1 输出 V(s)

    参数：
        state_shape : 观测空间 shape，元组 (N, M) 或 (N*M,)
        device      : 计算设备（'cuda:0' / 'cpu'）
        embed_dim   : 节点嵌入维度（默认 EMBED_SIZE=64）
        num_hgnn_layers : HGNN 层数（默认 2）
    """

    # output_dim 属性供 tianshou Critic 读取
    output_dim: int = EMBED_SIZE

    def __init__(
        self,
        state_shape,
        device: str = "cpu",
        embed_dim: int = EMBED_SIZE,
        num_hgnn_layers: int = 2,
    ):
        super().__init__()
        self.device = device
        self.embed_dim = embed_dim

        # ── 解析观测维度 ──────────────────────────────────────
        if isinstance(state_shape, (tuple, list)):
            if len(state_shape) == 2:
                self.n_tasks, self.state_feat_dim = int(state_shape[0]), int(state_shape[1])
            else:
                # 扁平化输入：(N*M,) → 自动推断
                total = int(state_shape[0])
                self.n_tasks = _N_TASKS
                self.state_feat_dim = total // _N_TASKS
        else:
            self.n_tasks = _N_TASKS
            self.state_feat_dim = int(state_shape) // _N_TASKS

        # 超图关联矩阵列数（前 incidence_cols 列为 H，其余为节点特征）
        self.incidence_cols = _INCIDENCE_COLS if USE_GRAPH_STATE else 0
        # 纯节点特征维度（去掉关联矩阵部分）
        self.node_feat_dim = self.state_feat_dim - self.incidence_cols

        # ── 子网络实例化 ───────────────────────────────────────
        # 1. HGNN 编码器（对应 Fig. 7）
        self.hgnn_encoder = HGNNEncoder(
            in_dim=self.node_feat_dim,
            hid_dim=embed_dim,
            out_dim=embed_dim,
            num_layers=num_hgnn_layers,
        )

        # 2. Seq2Seq 解码器（对应 Fig. 9）
        self.seq2seq = Seq2SeqDecoder(
            embed_dim=embed_dim,
            hidden_dim=embed_dim,
            n_actions=NUM_RESOURCE_CLUSTER,
        )

        # Tianshou Actor/Critic 直接接收 embed_dim，内部会自动做后续 Linear 映射。

        # 将模型挂载到指定设备
        self.to(device)

    # ────────────────────────────────────────────────────────
    def _parse_obs(self, obs: torch.Tensor):
        """
        将观测张量解析为关联矩阵 H 和节点特征 X。

        输入  obs : [B, N, M] 或 [N, M]（二维时自动扩 batch 维）
        输出  H   : [B, N, E]  超图关联矩阵
              X   : [B, N, F]  节点特征矩阵
        """
        if obs.dim() == 2:
            obs = obs.unsqueeze(0)    # [N, M] → [1, N, M]

        # obs 前 incidence_cols 列是关联矩阵
        if self.incidence_cols > 0:
            H = obs[:, :, :self.incidence_cols]       # [B, N, E]
            X = obs[:, :, self.incidence_cols:]       # [B, N, F]
        else:
            # 若未使用超图状态，H 退化为全1矩阵（每节点属于同一超边）
            B, N, _ = obs.shape
            H = torch.ones(B, N, 1, device=obs.device, dtype=obs.dtype)
            X = obs                                   # [B, N, M]

        return H, X

    # ────────────────────────────────────────────────────────
    def forward(
        self,
        obs,                        # numpy array 或 Tensor，shape: [B, N, M] / [B, N*M]
        state=None,                 # GRU 隐状态（跨时步传递），tianshou 约定
        info: dict = None,          # tianshou 传入的环境 info（含 HG 对象）
    ):
        """
        tianshou 兼容的前向接口。

        tianshou 要求：forward(obs, state, info) → (logits_or_value, state)

        返回：
            features : [B, embed_dim] — 特征向量（给 Actor/Critic 的最后 Linear 层使用）
            state    : [B, hidden_dim] — 更新后的 GRU 隐状态
        """
        # ── 类型转换 ──────────────────────────────────────────
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32)
        obs = obs.to(self.device, dtype=torch.float32)

        # ── 处理扁平化输入（tianshou 有时会将 obs 做 flatten）─
        if obs.dim() == 2 and obs.shape[-1] == self.n_tasks * self.state_feat_dim:
            # [B, N*M] → [B, N, M]
            obs = obs.view(obs.shape[0], self.n_tasks, self.state_feat_dim)
        elif obs.dim() == 1:
            # [N*M] → [1, N, M]
            obs = obs.view(1, self.n_tasks, self.state_feat_dim)

        B = obs.shape[0]

        # ── 解析 H 和 X ───────────────────────────────────────
        H, X = self._parse_obs(obs)
        # H : [B, N, E]，X : [B, N, F]

        # ── HGNN 编码（逐 sample 处理，因 H 可能不同）─────────
        # 注：若批次内所有 sample 的 H 相同可进一步向量化，
        #     但此处为保安全逐条处理
        node_embeds_list = []
        for b in range(B):
            H_b = H[b]   # [N, E]
            X_b = X[b]   # [N, F]
            Z_b = self.hgnn_encoder(X_b, H_b)   # [N, embed_dim]
            node_embeds_list.append(Z_b)
        # 堆叠为批次张量
        node_embeds = torch.stack(node_embeds_list, dim=0)  # [B, N, embed_dim]

        # ── 确定当前决策任务序号（默认从最后一个任务获取全局表示）
        # 实际逐步调度时 tianshou 每步传入完整 obs，task_idx 由 env 内部决定
        # 这里输出所有节点的平均 logits 作为 forward 输出（适合 batch rollout）
        # 另外也获取单步解码结果（task_idx=0 为默认，训练时 info 可以覆盖）
        task_idx = 0
        if info is not None and isinstance(info, dict) and "id" in info:
            task_idx = int(info["id"]) % self.n_tasks

        # ── Seq2Seq 解码（含公式 22 注意力）──────────────────
        features, new_hidden = self.seq2seq(node_embeds, task_idx=task_idx, hidden=state)
        # features   : [B, hidden_dim]
        # new_hidden : [B, hidden_dim]

        return features, new_hidden


# ────────────────────────────────────────────────────────────
#  便捷工厂函数：快速构建完整的 Actor-Critic 对
# ────────────────────────────────────────────────────────────
def build_actor_critic(state_shape, device: str = "cpu"):
    """
    构建两个独立的 HyperJetNet 实例，分别用作 Actor 和 Critic 的 preprocess_net。

    使用示例（在 main.py 中）：
        from models import HyperJetNet, build_actor_critic
        from tianshou.utils.net.discrete import Actor, Critic

        device = get_gpu()
        net_a, net_c = build_actor_critic(env.observation_space.shape, device)
        actor  = Actor(net_a, env.action_space.n, device=device).to(device)
        critic = Critic(net_c, device=device).to(device)

    返回：
        net_a : HyperJetNet（用于 Actor）
        net_c : HyperJetNet（用于 Critic）
    """
    net_a = HyperJetNet(state_shape=state_shape, device=device)
    net_c = HyperJetNet(state_shape=state_shape, device=device)
    return net_a, net_c


# ────────────────────────────────────────────────────────────
#  快速冒烟测试（直接运行此文件时执行）
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("HyperJetNet 冒烟测试 (Smoke Test)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 模拟观测维度（与 offloading_env.py 一致）
    N, M = _N_TASKS, _STATE_DIM        # 20, 56
    B = 4                              # batch size

    print(f"\n观测维度: batch={B}, N={N}, M={M}")
    print(f"超图关联矩阵列数: {_INCIDENCE_COLS}")
    print(f"节点特征维度: {M - _INCIDENCE_COLS}")
    print(f"动作空间大小: {NUM_RESOURCE_CLUSTER}")

    # 构建网络
    net = HyperJetNet(state_shape=(N, M), device=device).to(device)
    print(f"\n网络参数量: {sum(p.numel() for p in net.parameters() if p.requires_grad):,}")

    # 测试前向传播
    obs = torch.randn(B, N, M).to(device)
    with torch.no_grad():
        features, hidden = net(obs, state=None, info={})

    print(f"\n前向传播结果：")
    print(f"  features shape : {features.shape}    期望: [{B}, {EMBED_SIZE}]")
    print(f"  hidden shape   : {hidden.shape}    期望: [{B}, {EMBED_SIZE}]")

    assert features.shape == (B, EMBED_SIZE), \
        f"❌ features 维度错误: {features.shape}"
    assert hidden.shape == (B, EMBED_SIZE), \
        f"❌ hidden 维度错误: {hidden.shape}"

    # 测试 HGNNLayer 单独运行
    print("\nHGNNLayer 单元测试：")
    layer = HGNNLayer(in_features=16, out_features=64).to(device)
    x_test = torch.randn(10, 16).to(device)
    h_test = torch.randint(0, 2, (10, 5)).float().to(device)   # N=10, E=5
    out = layer(x_test, h_test)
    print(f"  输入 X: {x_test.shape}, H: {h_test.shape}")
    print(f"  输出: {out.shape}    期望: [10, 64]")
    assert out.shape == (10, 64), f"❌ HGNNLayer 输出维度错误: {out.shape}"

    print("\n[PASS] All tests passed! HyperJetNet is ready.")
