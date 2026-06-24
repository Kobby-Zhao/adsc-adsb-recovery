请你作为一名熟悉 PyTorch、Mamba/Selective State Space Model、轨迹恢复和时间序列建模的高级工程师，帮我实现一个用于跨洋 ADS-C 稀疏轨迹恢复任务的 Anchor-conditioned Bi-Mamba 主干网络代码。

请注意：我不是要普通 Bi-Mamba，而是要一个针对 ADS-C 稀疏锚点恢复任务定制的锚点条件双向 Mamba 主干。请你生成结构清晰、模块化、可训练、便于后续接入高度恢复分支 SAVCA 的 PyTorch 代码。

一、任务背景

研究任务是跨洋巡航阶段稀疏 ADS-C 轨迹恢复。输入是相邻两个 ADS-C 锚点及 gap 内的分钟级时间步，输出是锚点之间的分钟级轨迹特征，用于后续经纬度恢复和高度恢复。

该任务不是普通时间序列预测，而是 hard-anchor gap recovery：

1. 左右 ADS-C 锚点已知；
2. 中间分钟级轨迹缺失；
3. 模型需要利用左右锚点约束恢复中间轨迹；
4. 高度分支后续将采用 SAVCA，即状态引导的锚点归一化高度变化分配；
5. 当前代码重点实现 Anchor-conditioned Bi-Mamba 主干，输出每个时间步的双向上下文特征 h_t。

二、输入张量说明

假设 batch 输入已经整理成固定长度序列，padding 部分通过 seq_mask 标识。

请实现 forward 接口支持以下输入：

1. q: Tensor, shape [B, T, obs_dim]
   当前时间步观测输入。
   - 锚点处为真实 ADS-C 观测值；
   - 缺失处可以为 0 或占位值；
   - obs_dim 通常为 3，对应 lat/lon/alt，或者为归一化后的 x/y/z。

2. obs_mask: Tensor, shape [B, T, 1]
   当前时间步是否为 ADS-C 锚点观测。
   - 锚点处为 1；
   - 缺失处为 0。

3. d_left: Tensor, shape [B, T, 1]
   当前时间步距离左锚点的时间距离，可以是分钟数，也可以是归一化距离。

4. d_right: Tensor, shape [B, T, 1]
   当前时间步距离右锚点的时间距离。

5. tau: Tensor, shape [B, T, 1]
   当前时间步在 gap 内的相对位置，范围 [0,1]。

6. y_left: Tensor, shape [B, state_dim]
   左锚点状态，通常为 [lat, lon, alt] 或 [x, y, z]。

7. y_right: Tensor, shape [B, state_dim]
   右锚点状态。

8. delta_y: Tensor, shape [B, state_dim]
   左右锚点状态差，delta_y = y_right - y_left。

9. gap_len: Tensor, shape [B, 1]
   gap 长度，可以是分钟数，也可以是归一化长度。

10. c_gap: Tensor or None, shape [B, graph_dim]
    可选，全航班锚点图上下文，由 AGCE 模块输出。
    如果没有 AGCE，则允许 c_gap=None。

11. seq_mask: Tensor, shape [B, T, 1] or [B, T]
    有效时间步 mask。
    - 有效位置为 1；
    - padding 位置为 0。

三、Anchor-conditioned 输入构造

对于每个时间步 t，需要构造：

x_t^ac = [
    q_t,
    obs_mask_t,
    d_left_t,
    d_right_t,
    tau_t,
    y_left,
    y_right,
    delta_y,
    gap_len,
    c_gap
]

其中 y_left, y_right, delta_y, gap_len, c_gap 都是 gap-level 特征，需要 repeat 到 [B, T, *] 后与 time-step 特征拼接。

如果 c_gap=None，则不拼接该项。

请实现一个 AnchorConditionedInputEncoder 类：

输入上述特征；
输出嵌入序列 e，shape [B, T, d_model]。

嵌入层建议为：

Linear(input_dim, d_model)
LayerNorm(d_model)
SiLU 或 GELU
Dropout

四、Bi-Mamba 主干

请实现 AnchorConditionedBiMambaBackbone 类。

结构包括：

1. input_encoder: AnchorConditionedInputEncoder

2. forward_mamba:
   对输入序列 e 按时间正向扫描。

3. backward_mamba:
   对输入序列 reverse 后扫描，再 reverse 回原时间顺序。

4. 双向融合：
   不要只简单拼接，先实现 anchor-distance selective fusion。

融合权重根据以下特征预测：

fuse_input_t = [
    d_left_t,
    d_right_t,
    tau_t,
    delta_y,
    gap_len,
    c_gap
]

其中 delta_y, gap_len, c_gap 需要 repeat 到 [B,T,*]。

计算：

omega = softmax(MLP_fuse(fuse_input_t), dim=-1)

omega shape [B,T,2]

然后：

h_t = omega_f * h_t^f + omega_b * h_t^b

其中 h_t^f 和 h_t^b shape 均为 [B,T,d_model]。

也请保留一个选项 fusion_mode：
- "gated": 使用上述锚点距离选择性融合；
- "concat": 直接拼接 [h_f, h_b] 后通过 Linear 映射回 d_model；
- "sum": h_f + h_b。

默认使用 "gated"。

五、Mamba 模块实现要求

优先使用 mamba_ssm 库中的 Mamba：

from mamba_ssm import Mamba

如果环境没有安装 mamba_ssm，请提供一个 fallback 模块，例如使用 GRU 或简单的 nn.TransformerEncoderLayer 占位，但代码结构要保持一致，便于后续替换为真正的 Mamba。

请写成：

try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False

如果 HAS_MAMBA=True：
    使用 Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

如果 HAS_MAMBA=False：
    使用 FallbackSequenceBlock，例如：
    - nn.GRU(batch_first=True)
    或
    - depthwise conv + feed-forward
请在注释中说明 fallback 只是为了代码可运行，不代表最终实验模型。

六、输出

AnchorConditionedBiMambaBackbone.forward 返回一个 dict：

{
    "h": h,                     # [B,T,d_model] 融合后的上下文特征
    "h_forward": h_f,            # [B,T,d_model]
    "h_backward": h_b,           # [B,T,d_model]
    "fusion_weight": omega,      # [B,T,2]，如果 fusion_mode="gated"，否则 None
    "seq_mask": seq_mask         # [B,T,1]
}

其中 h 将作为后续：
1. 经纬度恢复头的输入；
2. SAVCA 高度恢复分支的输入。

七、请同时实现一个简单的 SAVCAHeightHead 接口占位

请实现一个简化版 SAVCAHeightHead，方便测试主干输出是否能接高度分支。

输入：
- h: [B,T,d_model]
- d_left, d_right, tau: [B,T,1]
- z_left: [B,1]
- z_right: [B,1]
- gap_len: [B,1]
- c_gap: optional [B,graph_dim]
- seq_mask: [B,T,1]

输出：
{
    "z_hat": [B,T,1],
    "r": [B,T,1],
    "p": [B,T,1],
    "a": [B,T,1]
}

SAVCA 逻辑：

u_t^z = [h_t, d_left_t, d_right_t, tau_t, delta_z, gap_len, c_gap]

r_t = sigmoid(MLP_state(u_t^z))

a_raw_t = softplus(MLP_alloc(u_t^z))

a_t = a_raw_t * (alpha + (1 - alpha) * r_t)

p_t = (a_t + eps) / sum_valid(a_t + eps)

注意：
- p_t 要只在有效 seq_mask 上归一化；
- padding 位置 p_t 应为 0；
- p_t 在有效时间步上和为 1；
- z_hat_t = z_left + (z_right - z_left) * cumulative_sum(p_t)
- 理论上最后一个有效时间步应等于 z_right；
- 如果序列中包含左锚点和右锚点，注意 p_t 更严格地应定义在时间间隔上。为了简化测试，可以先定义在 T 个有效步上；但请在注释中指出正式版本建议将 p 定义在 T_interval 个时间间隔上。

八、代码风格要求

1. 使用 PyTorch。
2. 代码模块化，至少包括：
   - AnchorConditionedInputEncoder
   - MambaBlock 或 SequenceBlock
   - AnchorDistanceFusion
   - AnchorConditionedBiMambaBackbone
   - SAVCAHeightHead
3. 每个类都要有清楚注释，说明输入输出 shape。
4. forward 中要做基本 shape 检查或注释。
5. 不要写死 obs_dim、state_dim、graph_dim，要通过 __init__ 参数传入。
6. 支持 c_gap=None。
7. 支持 seq_mask。
8. 提供一个最小 demo，用随机张量测试 forward 是否能跑通。
9. demo 中打印：
   - h shape
   - fusion_weight shape
   - z_hat shape
   - p.sum(dim=1) 检查是否接近 1
10. 代码应尽量能直接复制运行。

九、请注意几个关键实现细节

1. gap-level 特征 repeat 到时间维：
   y_left.unsqueeze(1).expand(-1,T,-1)

2. seq_mask 统一处理为 [B,T,1]。

3. reverse 序列时，如果存在 padding，可以先简单 flip 整个 T 维；后续正式版本可按有效长度 reverse。请在代码注释中说明这一点。
   如果你能实现基于 seq_mask 的 per-sample reverse 更好。

4. fusion weight 对 padding 位置可以不关心，但最终输出 h 建议乘 seq_mask。

5. SAVCA 中归一化时：
   denom = ((a + eps) * seq_mask).sum(dim=1, keepdim=True)
   p = ((a + eps) * seq_mask) / (denom + eps)

6. cumulative sum:
   cdf = torch.cumsum(p, dim=1)
   z_hat = z_left[:,None,:] + delta_z[:,None,:] * cdf

7. 如果 seq_mask 存在 padding，最后 padding 部分 z_hat 可以保持不使用，或者乘 mask。

十、请输出内容

请直接输出完整 Python 代码，不要只给伪代码。
代码后请简要说明：
1. Anchor-conditioned Bi-Mamba 与普通 Bi-Mamba 的区别；
2. AGCE 输出 c_gap 如何接入；
3. h 如何送入 SAVCA；
4. 当前代码中哪些地方是简化实现，正式实验需要进一步完善。