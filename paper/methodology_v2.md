# 三、高度感知的分钟级轨迹恢复方法

本章系统阐述本文提出的高度感知级联恢复框架。3.1 节给出问题形式化定义；3.2 节分析 ENU 坐标系下三维联合预测的梯度失衡问题并给出分离头方案；3.3 节介绍阶段一的几何锚定基线；3.4 节介绍阶段二的 DMS 时序精炼模块；3.5 节介绍高度辅助监督机制；3.6 节给出总损失函数与训练策略。

---

## 3.1 问题形式化与动机

### 3.1.1 稀疏观测轨迹恢复

给定航班 $f$ 在时间窗 $[1, T]$ 内的稀疏观测序列 $\mathcal{O} = \{\mathbf{o}_t\}_{t=1}^{T}$，其中 $\mathbf{o}_t = (\lambda_t, \phi_t, h_t)$ 为 ADS-C 报文提供的经度、纬度和气压高度。观测掩码 $\mathbf{m} \in \{0,1\}^{T}$ 标记每个时间步是否存在有效观测：$m_t = 1$ 表示位置 $t$ 有观测（锚点），$m_t = 0$ 表示该位置缺失。锚点集合记为 $\mathcal{A} = \{t \mid m_t = 1\}$。

连续缺失位置构成若干个 gap 区间。形式化地，序列中存在一组不重叠的连续缺失段 $\{\mathcal{G}_k\}_{k=1}^{K}$，其中 $\mathcal{G}_k = [l_k, r_k]$ 满足：

$$\forall t \in [l_k, r_k]: m_t = 0, \quad m_{l_k-1} = 1, \quad m_{r_k+1} = 1$$

即每个 gap 的左右端点紧邻锚点。恢复目标为：给定 $\mathcal{O}$ 和 $\mathbf{m}$，预测完整分钟级轨迹 $\hat{\mathbf{Y}} = \{\hat{\mathbf{y}}_t\}_{t=1}^{T}$，其中 $\hat{\mathbf{y}}_t = (\hat{\lambda}_t, \hat{\phi}_t, \hat{h}_t)$。

### 3.1.2 为什么高度需要独立处理

在巡航阶段，水平轨迹和高度轨迹具有本质不同的动力学特征：

- **水平运动**受大圆航线和航路点约束，在分钟尺度上呈现平滑、渐进的曲线，相邻点的位移矢量高度相关。ENU 坐标系下，水平增量仅在 $10^1$-$10^2$ 米量级。
- **高度变化**以阶梯爬升（step climb）为主要形式——飞机在燃油消耗减轻重量后，以约 305-610 m 为单位跳跃至更高巡航高度层。这些变化具有突发性、局部性，在 gap 内部可能仅发生一次。此外，对流层顶附近的晴空颠簸（CAT）也会引入无规律的短时高度波动。ENU 坐标系下，锚点相对高度可在 $10^1$-$10^2$ m 量级变化。

两者的差异不仅是数值量级上的，更是**信息结构上的**：水平运动的强时序相关性使前向/后向 LSTM 即便在稀疏观测下也能较好地建模趋势；而高度变化的稀疏性和突发性使其需要额外的几何约束和独立的学习路径。

这一差异在训练中表现为梯度失衡。设 backbone 共享参数 $\boldsymbol{\theta}$，主干三维输出为 $\hat{\mathbf{y}}_t = \mathbf{W}_{\text{out}} \mathbf{h}_t(\boldsymbol{\theta})$，Smooth L1 损失对各维度的梯度近似为：

$$\frac{\partial \mathcal{L}}{\partial \boldsymbol{\theta}} \approx \sum_{t} \sum_{d \in \{E,N,U\}} \frac{\partial \mathcal{L}_\delta(\hat{y}_t^{(d)}, y_t^{(d)})}{\partial \hat{y}_t^{(d)}} \cdot \frac{\partial \hat{y}_t^{(d)}}{\partial \boldsymbol{\theta}}$$

在 ENU 空间中，$\partial \mathcal{L} / \partial \hat{y}_t^{(E)} \sim O(10^4)$，$\partial \mathcal{L} / \partial \hat{y}_t^{(U)} \sim O(10^1)$，量级差达 $10^3$。Adam 优化器虽通过二阶矩估计进行逐参数自适应缩放，但**共享参数的梯度方向**仍由水平分量主导——backbone 优化路径的 99.7% 由 E/N 分量决定，高度分量对参数更新的贡献被淹没。

一个自然的疑问是：能否通过对高度维度施加更大的损失权重来纠正这一失衡？假设将高度损失权重设为 $\lambda_U$，则高度梯度变为：

$$\frac{\partial \mathcal{L}}{\partial \boldsymbol{\theta}}\bigg|_U \propto \lambda_U \cdot O(10^1)$$

要使高度梯度与水平梯度等量级，需 $\lambda_U \approx 500$。然而，如此极端的权重会导致：(1) 训练初期高度预测不稳定时产生巨大的梯度振荡；(2) 权重值本身成为需要调优的超参数，消融实验中难以公平对比；(3) 审稿人易质疑"调参优化结果"。因此，损失重加权无法从根本上解决该问题。

本文的核心思路是**从架构层面入手**，通过分离高度预测头、构建几何锚定基线和设计时序精炼模块，使高度通道获得独立的特征表达和学习路径。

---

## 3.2 分离高度预测头

### 3.2.1 从共享投影到独立投影

在标准的多维回归架构中，序列编码器的隐状态经单一线性层投影至三维输出空间：

$$\hat{\mathbf{y}}_t = \mathbf{W}_{\text{out}} \mathbf{h}_t + \mathbf{b}_{\text{out}}$$

其中 $\mathbf{W}_{\text{out}} \in \mathbb{R}^{d_h \times 3} = [\mathbf{w}_E, \mathbf{w}_N, \mathbf{w}_U]$ 的三列分别对应 E、N、U 三个维度。三个维度的预测共享同一隐表示 $\mathbf{h}_t$，梯度通过该权重矩阵的所有元素回传。由于 $\mathbf{w}_U$ 列仅贡献约 0.3% 的梯度范数，其更新幅度远小于 $\mathbf{w}_E$ 和 $\mathbf{w}_N$ 列。换言之，**高度预测的头层参数几乎处于停滞状态**，主干的隐表示 $\mathbf{h}_t$ 中高度相关信息长期处于欠训练状态。

本文的分离策略是将单一投影头拆分为两个独立头：

$$\hat{\mathbf{y}}_t^{\text{horiz}} = \mathbf{W}_{\text{horiz}} \mathbf{h}_t + \mathbf{b}_{\text{horiz}} \in \mathbb{R}^2$$

$$\hat{y}_t^{\text{alt}} = \mathbf{W}_{\text{alt}} \mathbf{h}_t + b_{\text{alt}} \in \mathbb{R}^1$$

$$\hat{\mathbf{y}}_t = [\hat{\mathbf{y}}_t^{\text{horiz}}; \hat{y}_t^{\text{alt}}] \in \mathbb{R}^3$$

这一拆分具有三重意义：(1) $\mathbf{W}_{\text{alt}} \in \mathbb{R}^{d_h \times 1}$ 拥有完全独立的高度梯度路径——3.5 节引入的高度辅助损失仅通过 $\mathbf{W}_{\text{alt}}$ 回传梯度，不影响水平头；(2) 高度头的参数量与拆分前一致，不增加模型容量；(3) 前向和后向 LSTM 各自输出的 $\boldsymbol{\mu}_t^{f}[\text{alt}]$ 和 $\boldsymbol{\mu}_t^{b}[\text{alt}]$ 为零冗余的高度表示，直接服务于后续的几何锚定和 DMS 精炼。

### 3.2.2 双向时序编码与位置感知融合

本文的双向预测主干由前向 LSTM 和后向 LSTM 构成，两个方向共享相同的输入编码器结构但具有独立的隐状态和参数。前向 LSTM 从 $t=1$ 至 $t=T$ 顺序处理，在 gap 区间内靠近左锚点的位置具有信息优势；后向 LSTM 从 $t=T$ 至 $t=1$ 逆序处理，在靠近右锚点的位置具有信息优势。两个方向的输出 $\boldsymbol{\mu}_t^{f}$ 和 $\boldsymbol{\mu}_t^{b}$ 通过位置感知融合模块进行自适应组合。

位置感知融合的核心思想是：融合权重应反映每个时间步相对于其最近锚点的距离。对于 gap 内的时间步 $t$，设 $\alpha_t = d_t^{\text{prev}} / (d_t^{\text{prev}} + d_t^{\text{next}})$ 为其归一化位置，位置先验权重定义为：

$$w_t^{f,\text{prior}} = 1 - \alpha_t, \quad w_t^{b,\text{prior}} = \alpha_t$$

直观上，靠近左锚点（$\alpha_t \to 0$）时前向预测更可靠（$w_t^{f} \to 1$），靠近右锚点（$\alpha_t \to 1$）时后向预测更可靠（$w_t^{b} \to 1$）。MLP 在该先验基础上学习有界偏移：

$$w_t^{f} = \text{clamp}\big(w_t^{f,\text{prior}} + \tanh(\text{MLP}(\mathbf{z}_t)) \cdot \delta_{\max}, \; \epsilon, \; 1-\epsilon\big)$$

$$w_t^{b} = 1 - w_t^{f}$$

其中 $\delta_{\max} = 0.30$ 控制 MLP 可偏离先验的最大幅度，$\epsilon = 10^{-8}$ 保证数值稳定。这种设计保证了**位置信号永远不被 MLP 完全覆盖**——无论训练如何，靠近左锚点时前向分支的主导地位在结构上得到保证。

融合后的初步预测为 $\mathbf{p}_t = w_t^{f} \boldsymbol{\mu}_t^{f} + w_t^{b} \boldsymbol{\mu}_t^{b}$，构成后续高度分支的输入。

---

## 3.3 阶段一：几何锚定基线

阶段一的设计原则是：**在模型从数据中学习任何模式之前，先通过几何约束给出最小信息下的合理估计**。这一基线具有零可学习参数，仅仅利用锚点位置提供的高度边界条件。

### 3.3.1 锚点前向/后向填充

定义填充操作 $\text{FFill}$（Forward Fill）和 $\text{BFill}$（Backward Fill）如下：

$$h_t^{\text{fwd}} = \text{FFill}(\mathbf{h}, \mathbf{m})_t = \begin{cases} h_t & \text{if } m_t = 1 \\ h_{t'}^{\text{fwd}} & \text{else, where } t' = \max\{\tau < t \mid m_\tau = 1\} \end{cases}$$

直观地，$h_t^{\text{fwd}}$ 是每个 gap 位置沿前向方向"看到"的最近锚点高度——gap 中的所有点共享同一个来自左锚点的高度值。$h_t^{\text{bwd}}$ 同理沿后向方向从右锚点反向填充。

计算上通过 cummax 算子高效实现。构造 $\tilde{h}_t = h_t \cdot m_t + (-\infty) \cdot (1 - m_t)$，则 $\{h_t^{\text{fwd}}\} = \text{cummax}(\tilde{h}_1, \ldots, \tilde{h}_T)$。反向填充通过对反转序列应用 cummax 后再次反转实现。序列首尾边界情况（第一个锚点之前和最后一个锚点之后的位置）通过向首个/末个锚点值夹紧处理。

### 3.3.2 线性插值基线

对于 gap $\mathcal{G}_k = [l_k, r_k]$ 内的任意时间步 $t$，基于两端锚点的线性插值基线为：

$$h_t^{\text{base}} = h_t^{\text{fwd}} + \alpha_t \cdot (h_t^{\text{bwd}} - h_t^{\text{fwd}})$$

其中 $\alpha_t = d_t^{\text{prev}} / (d_t^{\text{prev}} + d_t^{\text{next}}) \in [0,1]$。当 $\alpha_t = 0$（gap 首点），$h_t^{\text{base}} = h_t^{\text{fwd}}$（锚点值）；当 $\alpha_t = 1$（gap 末点），$h_t^{\text{base}} = h_t^{\text{bwd}}$。

线性插值是最大熵假设——在没有任何 gap 内部信息的情况下，高度从 $h^{\text{fwd}}$ 到 $h^{\text{bwd}}$ 的最无偏估计即为匀速线性过渡。在跨洋巡航场景中，这一假设对于大部分时段是准确的：飞行高度通常在恒定或接近恒定的巡航高度层，线性趋势捕捉了主体变化。

然而，线性基线无法表达两类偏离：(1) gap 内部的非线性变化（如阶梯爬升发生在 gap 中点附近）；(2) gap 边界处的局部波动（如飞机刚完成爬升进入巡航时的微量高度调整）。前者需要阶段二的 DMS 序列精炼来捕捉，后者通过后续的边界增强模块处理。

### 3.3.3 有界学习残差

在几何基线之上，允许主干网络学习一个残差修正项。主干网络的高度头输出 $\hat{y}_t^{\text{alt}}$ 经 $\tanh$ 映射至 $[-1,1]$ 后再与动态界限相乘：

$$\Delta_t^{\text{main}} = \tanh(\hat{y}_t^{\text{alt}}) \cdot r_t^{\text{dyn}}$$

动态界限 $r_t^{\text{dyn}}$ 根据 gap 长度自适应调整：

$$r_t^{\text{dyn}} = \text{clamp}\big(91.44 + 4.572 \cdot \min(d_t^{\text{prev}} + d_t^{\text{next}}, 60), \; 91.44, \; 365.76\big) \quad (\text{m})$$

设计理念为：短 gap（如 5 分钟）中不确定性低，残差上限收紧至约 91 m，防止模型在信息充足时过度修正；长 gap（如 60 分钟以上）中不确定性高，残差上限放宽至约 366 m，允许模型表达较大的高度变化（如阶梯爬升的约 610 m 跨度可被分阶段建模）。$\tanh$ 函数天然提供平滑的饱和特性——当主干输出趋近 $\pm\infty$ 时，修正量趋近 $\pm r_t^{\text{dyn}}$，不会超出预设界限。

**阶段一最终输出：**

$$\hat{h}_t^{(1)} = h_t^{\text{base}} + \Delta_t^{\text{main}}$$

---

## 3.4 阶段二：DMS 高度时序精炼

阶段一产生的 $\hat{h}_t^{(1)}$ 在 gap 边界处天然与锚点连续（$\hat{h}_{l_k}^{(1)} = h_{l_k-1}$，$\hat{h}_{r_k}^{(1)} = h_{r_k+1}$），但其内部缺乏时序一致性——每个时间步的修正 $\Delta_t^{\text{main}}$ 是独立计算的，彼此之间没有约束。一个 gap 内可能出现相邻两步修正量符号相反、幅度差异大的情况，导致局部锯齿状波动。

DMS 模块的任务是在保持边界连续性的前提下，对 gap 内部的高度曲线进行序列级平滑精炼。

### 3.4.1 设计原则：为什么 DMS 不吃水平特征

直觉上，向 DMS 提供完整的 3D ENU 特征似乎能给模型更多信息。但实验表明（见 4.3 节消融），包含水平维度（E、N）的 9 维输入（$\boldsymbol{\mu}_t^{f}$、$\boldsymbol{\mu}_t^{b}$、$\mathbf{p}_t$ 各 3 维）并不优于纯高度输入。原因有二：

首先，DMS 的任务是修正高度预测，它需要判断"当前的高度估计是否合理、是否与序列中的其他位置自洽"，这主要依赖于高度本身的模式——如"gap 中点的高度应平滑过渡于两端锚点之间"、"相邻点的高度差不应超过巡航飞行物理极限"等——而非水平位置信息。E、N 分量对高度精炼的边际信息增益极小。

其次，更重要的是，包含水平维度会引入**虚假相关性风险**。DMS 的注意力机制可能在训练中学习到"某些特定水平位置模式 → 特定高度修正"的伪关联，这种关联在训练集上降低 loss 但在测试集上不成立（例如，训练集中"北大西洋航路西行航班"恰好多为特定巡航高度层，模型可能将经度范围与高度修正绑定，而非学习真正的高度模式）。

因此，DMS 的时序输入被严格限定为 4 维纯高度特征：

$$\mathbf{x}_t^{\text{dms}} = \big[ \boldsymbol{\mu}_t^{f}[\text{alt}],\; \boldsymbol{\mu}_t^{b}[\text{alt}],\; h_t^{\text{base}},\; \Delta_t^{\text{main}} \big] \in \mathbb{R}^4$$

四维通道各自编码不同语义：前向高度推断、后向高度推断、几何边界约束、阶段一已学得的修正。这组特征不包含任何水平维度信息，确保 DMS 的所有注意力权重和学习到的修正模式严格基于高度维度，在训练集与测试集之间具有更好的泛化性。

### 3.4.2 稀疏上下文特征

除 4 维时序特征外，DMS 的注意力模块还接收 13 维稀疏上下文特征 $\mathbf{s}_t$，描述每个时间步与锚点和 gap 的关系：

$$\mathbf{s}_t = \big[ m_t,\; d_t^{\text{prev}},\; d_t^{\text{next}},\; d_t^{\text{prev}}+d_t^{\text{next}},\; \alpha_t,\; \mathbb{1}[d_t^{\text{prev}}>0],\; \mathbb{1}[d_t^{\text{next}}>0],\; \tau_t^{\text{prev}},\; \tau_t^{\text{next}},\; h_t^{\text{prev}},\; h_t^{\text{next}},\; h_t^{\text{next}}-h_t^{\text{prev}},\; h_t^{\text{interp}} \big]$$

其中 $\tau_t^{\text{prev}} = 1/(1+d_t^{\text{prev}})$ 和 $\tau_t^{\text{next}} = 1/(1+d_t^{\text{next}})$ 为锚点接近度（值域 $(0,1]$，距离越近值越大），$h_t^{\text{prev}}$、$h_t^{\text{next}}$ 为前向/后向锚点高度，$\Delta h_t^{\text{anchor}} = h_t^{\text{next}} - h_t^{\text{prev}}$ 为锚点间高度差，$h_t^{\text{interp}} = h_t^{\text{prev}} + \alpha_t \cdot \Delta h_t^{\text{anchor}}$ 为线性插值。这些特征使注意力机制能够感知：(1) 每个时间步与锚点的远近（决定该位置的信息充分程度）；(2) gap 整体的高度跨度（决定修正的幅度预期）；(3) 锚点是否存在（边界 gap 可能只有单侧锚点）。

### 3.4.3 Anchor-Aware 时序注意力

DMS 的注意力机制是**内在的（intra-sequence）**而非跨序列的——每个时间步对同一 gap 内的所有其他步计算注意力权重，不引入跨样本信息。这与 Transformer 的 self-attention 类似，但增加了稀疏上下文调制。

拼接时序特征与稀疏上下文后，通过评分 MLP 计算注意力得分：

$$\mathbf{z}_t = [\mathbf{x}_t^{\text{dms}}; \mathbf{s}_t] \in \mathbb{R}^{4+13}$$

$$e_t = \text{MLP}_{\text{score}}(\mathbf{z}_t) \in \mathbb{R}$$

$\text{MLP}_{\text{score}}$ 采用两层结构：Linear($d_{\text{in}}$, $d_a$) → SiLU → Dropout → Linear($d_a$, 1)，其中 $d_a = 64$。Dropout 正则化（$p=0.1$）防止注意力过度集中于少数点。

注意力权重经 softmax 归一化：

$$\alpha_t^{\text{attn}} = \frac{\exp(e_t)}{\sum_{\tau \in \mathcal{G}_k} \exp(e_\tau)}$$

注意分母仅在当前 gap $\mathcal{G}_k$ 内求和，不同 gap 之间注意力独立——因为不同 gap 的锚点约束和内部结构互不相关。

加权序列和全局上下文分别为：

$$\tilde{\mathbf{x}}_t = \alpha_t^{\text{attn}} \cdot \mathbf{x}_t^{\text{dms}}, \quad \mathbf{c} = \sum_{t \in \mathcal{G}_k} \tilde{\mathbf{x}}_t$$

全局上下文 $\mathbf{c}$ 是 gap 内所有位置信息的加权聚合，为解码器提供 gap 整体的高度状态感知。

### 3.4.4 DMS 解码器与全局精炼

注意力加权后的序列与全局上下文和稀疏特征拼接，进入解码器：

$$\mathbf{l}_t = \text{MLP}_{\text{decode}}\big([\tilde{\mathbf{x}}_t; \mathbf{c}; \mathbf{s}_t]\big) \in \mathbb{R}^{d_l}$$

解码器为三层 MLP，将注意力输出映射至 $d_l = 32$ 维隐空间。

解码后的隐序列 $\{\mathbf{l}_t\}_{t \in \mathcal{G}_k}$ 随后进入单层 Transformer 编码器进行全局一致性精炼：

$$\mathbf{L}' = \text{TransformerEncoder}(\mathbf{L}; \mathbf{M})$$

其中 $\mathbf{L} = [\mathbf{l}_{l_k}, \ldots, \mathbf{l}_{r_k}] \in \mathbb{R}^{|\mathcal{G}_k| \times d_l}$，$\mathbf{M}$ 为可选的 padding mask。Transformer 编码器采用 Pre-LN（LayerNorm 在前）和 GELU 激活。多头自注意力（$h=2$ 个头）使每个时间步的隐状态能够直接与 gap 内任意其他位置交互，从而捕捉跨步的一致性模式——例如，若 gap 中部某个位置的隐表示异常偏离相邻点，自注意力会将其拉回与周围一致的状态。

精炼后的隐表示经线性投影输出标量修正：

$$\delta_t^{\text{dms}} = \mathbf{w}_{\text{out}}^T \mathbf{l}'_t + b_{\text{out}}$$

**初始化策略**是训练稳定性的关键。若 $\mathbf{w}_{\text{out}}$ 采用零初始化（$\mathbf{w}_{\text{out}} = \mathbf{0}, b_{\text{out}} = 0$），则 DMS 在训练初始阶段对高度预测的贡献精确为零。虽然这在理论上允许模型"先学好基础再叠加修正"，但实践中零初始化导致 DMS 在有限的训练迭代内无法积累足够的梯度来"推开"权重——消融实验证实，24 epoch 训练后零初始化的 DMS 输出权重范数仅从 $0$ 增长至 $2 \times 10^{-2}$，实际修正量接近零。

本文采用小方差随机初始化：$\mathbf{w}_{\text{out}} \sim \mathcal{N}(0, 0.01^2)$，偏置为零。这使得 DMS 从训练第一步即产生微小但非零的修正（标准差约 $10^{-2}$ m），梯度回传通路在训练初期即被激活，后续凭借 $\tanh$ 限幅等下游约束自然控制修正幅度。

**阶段二最终输出：**

$$\hat{h}_t^{(2)} = \hat{h}_t^{(1)} + \delta_t^{\text{dms}}$$

水平坐标（E、N）在阶段二中保持不变：$\hat{\mathbf{y}}_t^{\text{horiz},(2)} = \hat{\mathbf{y}}_t^{\text{horiz},(1)}$。

---

## 3.5 高度辅助监督

### 3.5.1 动机

尽管分离头为高度提供了独立的投影权重，但传递给这些权重的梯度信号仍然依赖于主损失中高度分量的贡献。在 ENU 空间的主损失中，高度分量占比约 $0.3\%$——即便有了独立头，$\mathbf{W}_{\text{alt}}$ 接收到的梯度仍然微弱，导致前向/后向 LSTM 输出的高度特征 $\boldsymbol{\mu}_t^{f}[\text{alt}]$ 和 $\boldsymbol{\mu}_t^{b}[\text{alt}]$ 具有区分的不足。DMS 以这些特征为核心输入，若其质量不佳，DMS 的精炼能力受到根本性限制。

因此，需要一种机制直接增强 $\mathbf{W}_{\text{alt}}$ 的梯度信号，迫使主干网络产生高质量的高度隐表示。

### 3.5.2 位置加权辅助损失

在训练阶段，我们对前向和后向 LSTM 各自的高度预测施加独立的监督信号：

$$\mathcal{L}_{\text{aux}} = \frac{1}{|\mathcal{G}|} \sum_{t \in \mathcal{G}} \Big[ w_t^{f,\text{aux}} \cdot \mathcal{L}_\delta\big(\boldsymbol{\mu}_t^{f}[\text{alt}], h_t^{\text{gt}}\big) + w_t^{b,\text{aux}} \cdot \mathcal{L}_\delta\big(\boldsymbol{\mu}_t^{b}[\text{alt}], h_t^{\text{gt}}\big) \Big]$$

其中位置权重按各方向 LSTM 的信息充分程度分配：

$$w_t^{f,\text{aux}} = (1 - \alpha_t) \cdot \mathbb{1}[m_t = 0]$$

$$w_t^{b,\text{aux}} = \alpha_t \cdot \mathbb{1}[m_t = 0]$$

设计逻辑为：前向 LSTM 在靠近左锚点处（$\alpha_t \to 0$，$w_t^{f,\text{aux}} \to 1$）具有充足的历史上下文信息，其对高度的预测应当准确，因此承担更大的监督责任；而在靠近右锚点处（$\alpha_t \to 1$），前向 LSTM 已远离最近锚点，其预测不确定性增大，相应地降低权重。后向 LSTM 同理对称。

这一设计避免了"在信息不足的位置强制要求高精度预测"的不合理监督。加权机制的自适应性来源于 gap 内部位置本身——不引入额外超参数。

辅助损失仅作用于 gap 内部（$m_t = 0$），不对锚点位置施加额外约束（锚点由主损失中的锚点权重 $w_{\text{anc}}$ 充分监督）。

### 3.5.3 关键特性

高度辅助监督具有以下特性：(1) **仅训练时存在**——推理阶段完全移除，不增加推理计算量或参数量；(2) **梯度路径独立**——辅助损失的梯度仅流经 $\mathbf{W}_{\text{alt}}$ 和 LSTM 隐状态，不经过水平头或融合模块；(3) **无需额外标注**——监督信号 $h_t^{\text{gt}}$ 来自 ADS-B 分钟级真值，与主损失使用同一数据源。

实验表明（见 4.3 节），引入辅助损失后，A1→A2（DMS 阶段）的高度改善从 $0.35$ m 提升至 $3.47$ m，验证了"改善主干高度特征质量 → DMS 精炼能力增强"的因果链条。

---

## 3.6 总损失函数与训练策略

### 3.6.1 多分量损失函数

完整的训练目标由以下分量加权组合：

**主回归损失**在 ENU 坐标空间计算三维 Smooth L1 损失：

$$\mathcal{L}_{\text{main}} = \frac{1}{T} \sum_{t=1}^{T} \sum_{d \in \{E,N,U\}} \mathcal{L}_\delta(\hat{y}_t^{(d)}, y_t^{(d)}) \cdot \big[w_{\text{anc}} \cdot \mathbb{1}[m_t=1] + w_{\text{gap}} \cdot \mathbb{1}[m_t=0] \big]$$

其中 $w_{\text{anc}} = 1.0$，$w_{\text{gap}} = 2.0$。对缺失区间施加更高权重是因为 gap 内部才是轨迹恢复的核心难点——锚点位置的预测可通过简单的恒等映射（$\hat{\mathbf{y}}_t = \mathbf{o}_t$）实现完美恢复，而 gap 内部需要模型真正理解运动规律。

**一阶平滑损失**约束相邻预测点的变化量一致性：

$$\mathcal{L}_{\text{smooth}} = \frac{1}{T-1} \sum_{t=1}^{T-1} \big\| (\hat{\mathbf{y}}_{t+1} - \hat{\mathbf{y}}_t) - (\mathbf{y}_{t+1}^{\text{gt}} - \mathbf{y}_t^{\text{gt}}) \big\|_1$$

**高度垂直平滑损失**仅作用于 gap 内部的相邻点对，约束高度差分的一致性：

$$\mathcal{L}_{\text{vert}} = \frac{1}{|\mathcal{G}_{\text{pair}}|} \sum_{(t, t+1) \in \mathcal{G}_{\text{pair}}} \mathcal{L}_\delta(\hat{h}_{t+1} - \hat{h}_t,\; h_{t+1}^{\text{gt}} - h_t^{\text{gt}})$$

**高度辅助损失** $\mathcal{L}_{\text{aux}}$ 如 3.5 节定义。

**总损失：**

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{main}} + 0.1 \cdot \mathcal{L}_{\text{smooth}} + 0.02 \cdot \mathcal{L}_{\text{vert}} + 10.0 \cdot \mathcal{L}_{\text{aux}}$$

各系数由验证集上的网格搜索确定，并在所有消融实验中保持一致。

### 3.6.2 课程学习策略

为引导模型逐步适应稀疏观测条件，采用三阶段课程学习。训练集按观测缺失率划分为三个难度级别：阶段一（stage 1）包含短 gap（平均 <5 分钟）的样本，缺失率低；阶段二（stage 2）包含中等 gap（5-30 分钟）；阶段三（stage 3）按真实 ADS-C 间隔分布采样，缺失率最高、gap 最长。

每个 epoch 从三个阶段按动态权重加权采样 800 个样本。权重按照以下 schedule 从易到难过渡：

$$\begin{aligned} \text{Epoch 1-6}: &\quad w_1 = 0.70, \; w_2 = 0.25, \; w_3 = 0.05 \\ \text{Epoch 7-12}: &\quad w_1 = 0.20, \; w_2 = 0.60, \; w_3 = 0.20 \\ \text{Epoch 13-18}: &\quad w_1 = 0.15, \; w_2 = 0.35, \; w_3 = 0.50 \\ \text{Epoch 19-24}: &\quad w_1 = 0.05, \; w_2 = 0.20, \; w_3 = 0.75 \end{aligned}$$

这一 schedule 的直观含义是：训练早期模型在简单样本上建立基本的运动模式理解，中后期逐步过渡至真实分布下的困难样本，避免训练初期因复杂稀疏模式导致的梯度不稳定。验证集固定为阶段三样本，以真实场景的性能作为模型选择的依据。

其余训练设置为：Adam 优化器，初始学习率 $10^{-3}$，权重衰减 $10^{-4}$，梯度裁剪阈值 $1.0$，batch size 32，teacher forcing 比例从 $0.8$ 指数衰减至 $0.25$，共训练 24 epoch。所有输入特征经训练集统计量进行 Z-score 标准化后送入模型。

---

[后续章节：4. 实验设计 / 5. 结果分析 / 6. 结论]
