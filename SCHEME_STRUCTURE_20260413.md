# data-0313 方案结构梳理（基于 2026-04-13 代码现状）

## 1. 当前代码库的真实定位

`data-0313` 当前不是单纯的数据抓取仓库，也不是只做一个基础版轨迹恢复模型，而是一个三层系统：

1. 数据构建层  
   从 ADS-B 原始轨迹和 ADS-C 解码文本构建分钟级巡航轨迹、稀疏观测样本和真实回放输入。

2. 监督训练层  
   使用分钟级 ADS-B 轨迹作为监督真值，模拟 ADS-C 稀疏观测，训练轨迹恢复模型。

3. 真实回放评估层  
   在真实 ADS-C 回放场景下，不依赖 gap 内分钟级真值，而以边界一致性、形状合理性、参考一致性为核心进行评估。

这意味着方案主线应该写成：

**“面向真实 ADS-C 稀疏轨迹恢复的分阶段训练与边界一致性增强方案”**

而不应继续写成早期 README 那种“OpenSky 拉数与 ADS-C/ADS-B 融合分析脚本集合”。

---

## 2. README 与代码现状的偏差

### 2.1 `readme.md` 明显过时

`readme.md` 仍然描述的是早期脚本流水线：

- `01_fetch_airports.py`
- `02_fetch_flights_list.py`
- `03_flights_process.py`
- `05_fetch_points.py`
- `10_merge_adsc_adsb.py`

这些并不是当前主干训练/评估闭环。

### 2.2 `readme2.md` 比 `readme.md` 更接近现状，但仍偏“第一版规范”

`readme2.md` 描述的是：

- 分钟级聚合
- 稀疏模拟
- Gap-aware LSTM
- 双向融合
- 基础训练评估

但当前代码已经新增并实际使用了：

- 标准化与目标归一化
- ENU/相对坐标建模
- anchor-relative 高度目标
- altitude governance
- risk-aware segment metadata
- curriculum 式分阶段样本调度
- targeted altitude loss / local spike loss
- left-edge directional constraint
- replay 侧 left-edge projection / smoothing
- reference consistency audit

因此，当前方案不应停留在“基础版 BiLSTM 恢复”，而应强调：

**“围绕高度边界稳定性和真实回放一致性的多阶段增强方案”**

---

## 3. 当前代码主干

### 3.1 数据准备主流程

主入口：

- `scripts/prepare_data.py`

核心步骤：

1. 读取 ADS-B 原始 CSV
2. 通过 `src/preprocessing/adsb_aggregate.py` 做分钟级聚合
3. 通过 `src/preprocessing/cruise_filter.py` 提取巡航片段
4. 通过 `src/preprocessing/adsc_parse.py` 解析真实 ADS-C
5. 通过 `src/preprocessing/adsc_pattern_stats.py` 统计真实 ADS-C 间隔分布
6. 生成 stage1/stage2/stage3 不同稀疏模式的观测掩码
7. 通过 `src/preprocessing/sample_builder.py` 切窗生成样本
8. 通过 `src/preprocessing/feature_builder.py` 构造外生特征与质量代理特征

### 3.2 训练主流程

主入口：

- `scripts/train.py`

实际训练时额外加入了：

- anchor altitude features
- vertical v2 features
- anchor gate
- alt label governance
- 标准化
- target normalization
- risk-aware sample meta
- failure-mode reweighting
- curriculum mixing

### 3.3 推理与测试主流程

主入口：

- `scripts/evaluate.py`
- `scripts/infer.py`

主要支持：

- 标准测试集定量评估
- 坐标系恢复
- 高度目标反变换
- 可视化输出

### 3.4 真实 ADS-C 回放评估

主入口：

- `scripts/real_adsc_replay_eval.py`

这是当前仓库非常关键的一层，说明项目目标不只是“测试集 RMSE 更低”，而是：

- 在真实 ADS-C 稀疏场景中进行回放
- 关注边界连续性
- 关注左边界方向错误
- 关注 overshoot
- 关注 reference consistency

---

## 4. 当前模型叙事应该怎么写

从代码看，最合理的技术叙事不是“单一网络恢复轨迹”，而是四层结构：

1. 稀疏观测建模层  
   用真实 ADS-C 间隔统计驱动模拟采样，并通过 stage1/stage2/stage3 构造不同难度样本。

2. 双向轨迹恢复层  
   以前向/后向时序编码为主体，融合观测点、时间间隔、运动学代理特征和质量特征，恢复分钟级轨迹。

3. 高度治理层  
   对高度分量单独增强，引入 anchor-relative 表达、anchor altitude 特征、局部边界约束与 targeted loss，专门处理边界跳变和局部尖峰。

4. 真实回放校核层  
   在真实 ADS-C 输入下，用边界一致性、参考一致性和形状合理性替代无法直接获得的 gap 内真值误差。

---

## 5. 适合正式方案的推荐目录结构

下面这个结构适合直接写成正式“技术方案/实施方案”：

## 一、项目背景与问题定义

- ADS-C 报文稀疏，分钟级连续轨迹缺失
- 真实跨洋/远距离航段中缺少高频连续观测
- 传统插值方法难以保证边界连续性与高度稳定性
- 目标是恢复分钟级轨迹，并保证真实回放中的边界合理性

## 二、总体技术路线

- 数据来源：ADS-B 高频轨迹 + ADS-C 稀疏报文
- 训练思路：用 ADS-B 构造监督真值，用真实 ADS-C 分布模拟稀疏观测
- 推理思路：仅使用真实 ADS-C 稀疏输入进行恢复
- 评估思路：离线监督误差 + 真实回放边界一致性双轨评估

## 三、数据处理与样本构建方案

### 3.1 ADS-B 分钟级聚合

- 原始轨迹清洗
- 时间对齐
- 每分钟聚合
- 航向圆周均值处理

### 3.2 巡航片段筛选

- 垂直速度约束
- 速度变化约束
- 航向变化约束
- 最小连续巡航时长约束

### 3.3 ADS-C 解析与统计建模

- Tag 7 位置与时间提取
- Tag13/14/15/16 存在性提取
- 位置精度字段解析
- 真实间隔分布统计

### 3.4 稀疏观测模拟与多阶段样本生成

- stage1：短缺口、低缺失率
- stage2：中等缺口、局部连续缺失
- stage3：按真实 ADS-C 分布采样
- 滑窗切分、尾段补齐、时间断裂切段

### 3.5 特征工程

- 观测掩码特征
- `dt_prev / dt_next / gap_len / gap_pos_ratio`
- 垂直速度、转弯率、局部波动统计
- tag 存在性与位置精度质量特征
- anchor altitude 衍生特征

## 四、模型设计方案

### 4.1 基础双向恢复框架

- 前向时序编码
- 后向时序编码
- 双向融合输出

### 4.2 多坐标表达与目标空间设计

- `latlon` 或 `enu` 坐标模式
- 绝对高度与 anchor-relative 高度两种目标方式
- 目标标准化与反变换

### 4.3 高度专项增强模块

- anchor altitude feature
- vertical v2 feature
- altitude baseline / residual 建模
- DMS refiner / altitude refiner 变体
- left-edge directional constraint

### 4.4 风险感知与样本重加权

- short / medium / long segment bucket
- two-anchor / asymmetric / sparse-context 模式
- risk flag / teacher scale
- failure-mode reweighting

## 五、训练策略设计

### 5.1 数据划分与训练闭环

- 按 flight_id 划分 train / val / test
- 避免同航班泄漏
- 保存标准化器与目标统计量

### 5.2 标签治理与样本过滤

- anchor gate
- no-anchor 样本附录统计
- alt label governance

### 5.3 分阶段课程学习

- 先易后难的 curriculum schedule
- 多阶段样本混合采样
- 逐步提升稀疏难度

### 5.4 损失函数设计

- anchor/gap 分区损失
- smooth loss
- multi-scale loss
- altitude residual / boundary / first-step / second-step 专项损失
- local spike / targeted rightstep2 loss

## 六、推理与后处理方案

- 标准推理流程
- 坐标反变换
- 高度后处理
- replay 侧 left-edge projection
- replay 侧局部 smoothing

## 七、评估体系设计

### 7.1 离线监督评估

- MAE / RMSE
- gap 区域误差
- 水平误差与垂直误差
- 分 gap 长度、分 anchor 模式统计

### 7.2 真实 ADS-C 回放评估

- boundary consistency
- left-edge wrong-direction ratio
- overshoot rate
- reference consistency audit
- qualitative replay case study

## 八、实验设计与迭代路线

- baseline 对比：UniLSTM / CNNLSTM / Transformer / BiLSTM
- exp1-exp12 系列边界与高度增强实验
- 回放复核与 reference consistency 审计
- 形成最终生产链路配置

## 九、系统输出与交付物

- 训练 checkpoint
- 标准化器与 target scaler
- 测试集评估表
- replay 边界一致性报告
- case plots
- reference consistency 审计文件

## 十、风险点与后续优化方向

- 真实 ADS-C gap 内缺少分钟级真值
- 高度边界稳定性仍是核心难点
- replay 指标与离线 RMSE 不完全一致
- 后续可继续做更强的边界约束与 reference-aware 融合

---

## 6. 如果你要写“简版方案”，建议只保留这 6 章

如果你现在先写一个短版，可压缩成：

1. 问题背景与目标
2. 数据处理与样本构建
3. 模型设计与高度治理
4. 训练策略与分阶段学习
5. 评估体系与真实回放验证
6. 实验计划与预期成果

---

## 7. 一句话结论

当前 `data-0313` 最适合包装成：

**“基于 ADS-B 监督和 ADS-C 真实间隔统计的分钟级轨迹恢复系统，并通过边界一致性增强与真实回放审计解决高度恢复不稳定问题。”**
