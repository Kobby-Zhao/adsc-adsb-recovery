# 项目版本梳理报告

## 1. 当前最终主线结论

- 当前代码层面的最终候选模型为 `C_bimamba_context_xyaux_zlinear_zadapter_gapaware_small`，对应 `backbone_type=bimamba_context_xyaux_zlinear_zadapter_gapaware_small`，实现位于 [src/models/full_model.py](/home/jj/workspace/data-0313/src/models/full_model.py)。
- 当前正式训练配置为 [formal_24ep_gapaware_small.yaml](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/configs/formal_24ep_gapaware_small.yaml)，其课程数据引用的是 `final_s123_curriculum/pools/S1_train.parquet`、`S2_medium_train.parquet`、`S3_train.parquet`。
- 当前正式训练入口为 [scripts/train.py](/home/jj/workspace/data-0313/scripts/train.py)，正式评估入口为 [scripts/evaluate.py](/home/jj/workspace/data-0313/scripts/evaluate.py)。
- 当前最终候选 checkpoint 已存在： [formal_24ep_gapaware_small/best.pt](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/formal_24ep_gapaware_small/best.pt)。
- `xyaux_zlinear` 基线和 `zadapter` 基线仍保留为 24 epoch 可复现实验，分别对应 [ab24_xyaux_zlinear.yaml](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/configs/ab24_xyaux_zlinear.yaml) 和 [ab24_xyaux_zlinear_zadapter.yaml](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/configs/ab24_xyaux_zlinear_zadapter.yaml)。
- 当前正式 baseline 体系以 `final_s123_curriculum/configs/formal_24ep_*` 为主，包括 `unilstm_proto`、`bilstm_proto`、`cnnlstm_proto`、`transformer_proto`、`mamba_proto`、`bimamba_context_xyaux_zlinear`。
- 历史上的 `coarsetrend`、`vprogaux`、`vprog_resaux`、A1/A2/A3、SAVCA、旧 S2 比较目录、旧 sanity 目录、旧 proto baseline 目录已在本轮清理，不再视为项目保留对象。
- 当前正式课程数据应理解为 `final S1 / S2_medium / S3 curriculum`。历史 `stage2_clean`、`S2_new_train.parquet` 等旧过渡版本已清理，但仍有少量旧脚本保留这些路径的文字引用。
- [run_stagewise_eval_proto_compare_24e.py](/home/jj/workspace/data-0313/scripts/run_stagewise_eval_proto_compare_24e.py) 的 `S2` 评估口径已在本轮改成 `stage2_medium_clean/samples.parquet`；但现有 `outputs/analysis/stagewise_eval_proto_compare_24e` 汇总结果若未重跑，仍可能反映旧口径。
- `formal_24ep_gapaware_small` 目录中 **未发现 `main_task_metrics_test_summary_dim.csv`**，且本轮全局搜索也未发现该 run 的 test summary 落在别处，需人工确认是否尚未正式跑 test。

## 2. 当前最终模型与 baseline 清单

### 表 1：当前最终主线文件清单

| 类别 | 文件或目录路径 | 对应方案名称 | 是否当前最终版 | 作用说明 | 判断依据 | 建议操作 |
| --- | --- | --- | --- | --- | --- | --- |
| 模型代码 | [src/models/full_model.py](/home/jj/workspace/data-0313/src/models/full_model.py) | ACT-BiMamba / `gapaware_small` | 是 | 定义 `bimamba_context_xyaux_zlinear_zadapter_gapaware_small`、双向隐藏态对齐、xy/z 解耦、z-adapter、gapaware_small 条件输入 | `backbone_type` 注册、forward 分支和 z-adapter 实现均在此文件 | 保留 |
| 模型代码 | [src/models/sequence_baselines.py](/home/jj/workspace/data-0313/src/models/sequence_baselines.py) | Mamba/LSTM/Transformer/BiMamba 序列骨干 | 是 | 提供 MambaEncoderSequencePredictor、reverse/restore、anchor condition features 等共享基础组件 | 被 `full_model.py` 和训练入口直接引用 | 保留 |
| 训练脚本 | [scripts/train.py](/home/jj/workspace/data-0313/scripts/train.py) | 正式训练入口 | 是 | 读取 config，构造 `TrajectoryRecoveryModel`，执行训练与 checkpoint 保存 | 当前所有正式 `formal_24ep_*` 配置均通过该入口训练 | 保留 |
| 评估脚本 | [scripts/evaluate.py](/home/jj/workspace/data-0313/scripts/evaluate.py) | 正式评估入口 | 是 | 加载 config 与 checkpoint，执行 val/test 评估并生成 summary | 当前 stagewise 评估、正式单模型评估均基于该入口 | 保留 |
| 配置文件 | [formal_24ep_gapaware_small.yaml](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/configs/formal_24ep_gapaware_small.yaml) | 当前最终候选 ACT-BiMamba | 是 | 24 epoch 正式训练配置，引用 final S1/S2_medium/S3 pools | `backbone_type` 为 `...gapaware_small`，`experiment_note` 标注 final candidate | 保留 |
| 配置文件 | [ab24_xyaux_zlinear.yaml](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/configs/ab24_xyaux_zlinear.yaml) | `xyaux_zlinear` 基线 | 是 | 当前主线中的结构基线 | 当前 24 epoch 基线配置，输出目录完整 | 保留 |
| 配置文件 | [ab24_xyaux_zlinear_zadapter.yaml](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/configs/ab24_xyaux_zlinear_zadapter.yaml) | `zadapter` 基线 | 是 | 当前主线中的高度增强对照版本 | 当前 24 epoch 对照配置，输出目录完整 | 保留 |
| 数据集 | [final_s123_curriculum/pools/S1_train.parquet](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/pools/S1_train.parquet) | Final S1 train pool | 是 | 正式课程式训练第一阶段数据池 | 被 `formal_24ep_*` 系列配置直接引用 | 保留 |
| 数据集 | [final_s123_curriculum/pools/S2_medium_train.parquet](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/pools/S2_medium_train.parquet) | Final S2 train pool | 是 | 正式课程式训练第二阶段数据池 | `formal_24ep_gapaware_small.yaml` 等正式配置直接引用；由 `finalize_s123_curriculum.py` 生成 | 保留 |
| 数据集 | [final_s123_curriculum/pools/S3_train.parquet](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/pools/S3_train.parquet) | Final S3 train pool | 是 | 正式课程式训练第三阶段数据池 | 被正式配置直接引用 | 保留 |
| 数据集源 | [stage1_clean/samples.parquet](/home/jj/workspace/data-0313/outputs/mvp_merged_250_20260514_clean/stage1_clean/samples.parquet) | Final stage1 clean source | 是 | S1 原始清洗样本来源 | `finalize_s123_curriculum.py` 使用此源构建 final pool | 保留 |
| 数据集源 | [stage2_medium_clean/samples.parquet](/home/jj/workspace/data-0313/outputs/mvp_merged_250_20260514_clean/stage2_medium_clean/samples.parquet) | Final stage2_medium clean source | 是 | S2_medium 原始样本来源 | `finalize_s123_curriculum.py` 明确引用该路径 | 保留 |
| 数据集源 | [stage3_clean/samples.parquet](/home/jj/workspace/data-0313/outputs/mvp_merged_250_20260514_clean/stage3_clean/samples.parquet) | Final stage3 clean source | 是 | S3 原始样本来源 | `finalize_s123_curriculum.py` 明确引用该路径 | 保留 |
| 实验结果 | [formal_24ep_gapaware_small](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/formal_24ep_gapaware_small) | 当前最终候选正式实验目录 | 是 | 保存 24 epoch 正式训练结果、诊断、历史记录、checkpoint | `train_run_signature.json` 显示 backbone 为 `...gapaware_small`；`best.pt` 与 `history.json` 存在 | 保留 |
| checkpoint | [formal_24ep_gapaware_small/best.pt](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/formal_24ep_gapaware_small/best.pt) | 当前最终候选最佳权重 | 是 | 当前候选模型的正式 best checkpoint | 配置与 run signature 一致；已用于后续质性对比脚本 | 保留 |
| 主结果表 | [stagewise_proto_compare_summary.csv](/home/jj/workspace/data-0313/outputs/analysis/stagewise_eval_proto_compare_24e/stagewise_proto_compare_summary.csv) | 当前阶段化正式 baseline 主结果表 | 是 | 汇总 `bimamba` 与 proto baseline 的 S1/S2/S3 stagewise test 结果 | 文件已存在且内容完整 | 保留 |
| long-gap 诊断 | [longgap_failure_diagnosis_20260531](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/longgap_failure_diagnosis_20260531) | 当前主线 failure diagnosis | 是 | 保存 boundary jump / shape failure 诊断结果 | 含 `failure_mode_summary.csv` 与 `failure_mode_summary_fixed.md` | 保留 |

### 表 2：正式 baseline 版本清单

| 模型名称 | model_key | 代码实现路径 | 配置文件路径 | 训练命令或脚本 | 输出目录 | best checkpoint 路径 | S1/S2/S3 最终结果文件 | 是否已经完成正式结果 | 是否存在旧版重复结果 | 建议保留的最终版本 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Kalman Filter | `kalman_filter` | [scripts/eval_rts_kalman_baseline.py](/home/jj/workspace/data-0313/scripts/eval_rts_kalman_baseline.py) | [kalman_filter_clean_absolute.yaml](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/clean_baselines_gaponly_v1/configs/kalman_filter_clean_absolute.yaml) | 独立基线脚本；stagewise 由 [run_stagewise_eval_proto_compare_24e.py](/home/jj/workspace/data-0313/scripts/run_stagewise_eval_proto_compare_24e.py) 复制已有结果 | [clean_baselines_gaponly_v1/kalman_filter_clean_absolute](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/clean_baselines_gaponly_v1/kalman_filter_clean_absolute) | [best.pt](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/clean_baselines_gaponly_v1/kalman_filter_clean_absolute/best.pt) | `outputs/analysis/stagewise_eval_proto_compare_24e/kalman_filter_S1~S3`；汇总见 [stagewise_proto_compare_summary.csv](/home/jj/workspace/data-0313/outputs/analysis/stagewise_eval_proto_compare_24e/stagewise_proto_compare_summary.csv) 与 [linear_kalman_stagewise_summary.csv](/home/jj/workspace/data-0313/outputs/analysis/stagewise_simple_baselines/linear_kalman_stagewise_summary.csv) | 是 | 旧重复目录已基本清理完毕 | 保留 `clean_baselines_gaponly_v1 + stagewise_eval_proto_compare_24e` |
| UniLSTM-proto | `unilstm_proto` | [src/models/full_model.py](/home/jj/workspace/data-0313/src/models/full_model.py) / [src/models/sequence_baselines.py](/home/jj/workspace/data-0313/src/models/sequence_baselines.py) | [formal_24ep_unilstm_proto.yaml](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/configs/formal_24ep_unilstm_proto.yaml) | `python scripts/train.py --config .../formal_24ep_unilstm_proto.yaml` | [formal_24ep_unilstm_proto](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/formal_24ep_unilstm_proto) | [best.pt](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/formal_24ep_unilstm_proto/best.pt) | `outputs/analysis/stagewise_eval_proto_compare_24e/lstm_proto_S1~S3` | 是 | 旧重复目录已基本清理完毕 | 保留 `formal_24ep_unilstm_proto` |
| BiLSTM-proto | `bilstm_proto` | 同上 | [formal_24ep_bilstm_proto.yaml](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/configs/formal_24ep_bilstm_proto.yaml) | `python scripts/train.py --config .../formal_24ep_bilstm_proto.yaml` | [formal_24ep_bilstm_proto](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/formal_24ep_bilstm_proto) | [best.pt](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/formal_24ep_bilstm_proto/best.pt) | `outputs/analysis/stagewise_eval_proto_compare_24e/bilstm_proto_S1~S3` | 是 | 已清理大部分旧版本 | 保留 `formal_24ep_bilstm_proto` |
| CNN-LSTM-proto | `cnnlstm_proto` | 同上 | [formal_24ep_cnnlstm_proto.yaml](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/configs/formal_24ep_cnnlstm_proto.yaml) | `python scripts/train.py --config .../formal_24ep_cnnlstm_proto.yaml` | [formal_24ep_cnnlstm_proto](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/formal_24ep_cnnlstm_proto) | [best.pt](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/formal_24ep_cnnlstm_proto/best.pt) | `outputs/analysis/stagewise_eval_proto_compare_24e/cnnlstm_proto_S1~S3` | 是 | 旧重复目录已基本清理完毕 | 保留 `formal_24ep_cnnlstm_proto` |
| Transformer-proto | `transformer_proto` | 同上 | [formal_24ep_transformer_proto.yaml](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/configs/formal_24ep_transformer_proto.yaml) | `python scripts/train.py --config .../formal_24ep_transformer_proto.yaml` | [formal_24ep_transformer_proto](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/formal_24ep_transformer_proto) | [best.pt](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/formal_24ep_transformer_proto/best.pt) | `outputs/analysis/stagewise_eval_proto_compare_24e/transformer_proto_S1~S3` | 是 | 旧重复目录已基本清理完毕 | 保留 `formal_24ep_transformer_proto` |
| Mamba-proto | `mamba_proto` | 同上 | [formal_24ep_mamba_proto.yaml](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/configs/formal_24ep_mamba_proto.yaml) | `python scripts/train.py --config .../formal_24ep_mamba_proto.yaml` | [formal_24ep_mamba_proto](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/formal_24ep_mamba_proto) | [best.pt](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/formal_24ep_mamba_proto/best.pt) | `outputs/analysis/stagewise_eval_proto_compare_24e/mamba_proto_S1~S3` | 是 | 旧重复目录已基本清理完毕 | 保留 `formal_24ep_mamba_proto` |
| BiMamba（当前 plain baseline） | `bimamba_context_xyaux_zlinear` | [src/models/full_model.py](/home/jj/workspace/data-0313/src/models/full_model.py) | [formal_24ep_bimamba.yaml](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/configs/formal_24ep_bimamba.yaml) | `python scripts/train.py --config .../formal_24ep_bimamba.yaml` | [formal_24ep_bimamba](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/formal_24ep_bimamba) | [best.pt](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/formal_24ep_bimamba/best.pt) | `outputs/analysis/stagewise_eval_proto_compare_24e/bimamba_S1~S3` | 是 | 旧 `proposed_curriculum_24e` 已清理 | 保留 `formal_24ep_bimamba` |
| ACT-BiMamba（本文最终候选） | `bimamba_context_xyaux_zlinear_zadapter_gapaware_small` | [src/models/full_model.py](/home/jj/workspace/data-0313/src/models/full_model.py) | [formal_24ep_gapaware_small.yaml](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/configs/formal_24ep_gapaware_small.yaml) | `python scripts/train.py --config .../formal_24ep_gapaware_small.yaml` | [formal_24ep_gapaware_small](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/formal_24ep_gapaware_small) | [best.pt](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/formal_24ep_gapaware_small/best.pt) | **待确认**：当前未发现 `S1/S2/S3 stagewise` 结果目录，也未发现该 run dir 的 `main_task_metrics_test_summary_dim.csv` | 部分完成：训练完成，val 结果齐全；test summary 待确认 | 仍有 `xyaux_zlinear` / `zadapter` 结构对照目录 | 保留 `formal_24ep_gapaware_small` |

## 3. 数据集版本清单

### 表 4：数据集版本清单

| 数据集名称 | 路径 | 样本数 | 对应 split/阶段 | 是否 final 版本 | 是否存在派生列重算 bug 风险 | 是否与当前训练脚本匹配 | 建议操作 |
| --- | --- | ---: | --- | --- | --- | --- | --- |
| stage1_clean | [stage1_clean/samples.parquet](/home/jj/workspace/data-0313/outputs/mvp_merged_250_20260514_clean/stage1_clean/samples.parquet) | 3591 个 sample / 182013 行 | S1 原始样本源 | 是 | 字段 `obs_mask/dt_prev/dt_next/gap_len/gap_pos_ratio` 存在，但是否重算正确无单独校验，待人工复核 | 是，被 `finalize_s123_curriculum.py` 使用 | 保留 |
| stage2_medium_clean | [stage2_medium_clean/samples.parquet](/home/jj/workspace/data-0313/outputs/mvp_merged_250_20260514_clean/stage2_medium_clean/samples.parquet) | 3591 / 182013 行 | Final S2 原始样本源 | 是 | 字段完整，但针对“仅替换 obs_mask 未重算派生列”的历史 bug，本轮无法仅凭表面字段完全排除，**待人工复核** | 是，被 `finalize_s123_curriculum.py` 和正式 final config 间接引用 | 保留 |
| stage3_clean | [stage3_clean/samples.parquet](/home/jj/workspace/data-0313/outputs/mvp_merged_250_20260514_clean/stage3_clean/samples.parquet) | 3591 / 182013 行 | S3 原始样本源，正式 data.samples_path/test split 来源 | 是 | 字段完整；未发现异常证据，但仍建议人工复核关键派生列 | 是 | 保留 |
| S1_train | [final_s123_curriculum/pools/S1_train.parquet](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/pools/S1_train.parquet) | 2883 / 145089 行 | final curriculum stage1 train pool | 是 | 派生列字段存在；由 finalization 脚本生成，低于源级风险，但仍建议人工 spot check | 是，正式训练直接使用 | 保留 |
| S2_medium_train | [final_s123_curriculum/pools/S2_medium_train.parquet](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/pools/S2_medium_train.parquet) | 2883 / 145089 行 | final curriculum stage2 train pool | 是 | **重点待人工复核**：历史上 S2 相关派生列曾出过 bug；本轮仅确认字段存在，未重算校验 | 是，正式训练直接使用 | 保留 |
| S3_train | [final_s123_curriculum/pools/S3_train.parquet](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/pools/S3_train.parquet) | 2883 / 145089 行 | final curriculum stage3 train pool | 是 | 派生列字段存在；建议人工抽样复核 | 是 | 保留 |
| curriculum stats: S1 | [S1_per_sample_stats.csv](/home/jj/workspace/data-0313/outputs/analysis/curriculum_distribution/S1_per_sample_stats.csv) | 3591 | 统计文件 | 是 | 不适用 | 间接匹配 | 保留 |
| curriculum stats: S2 | [S2_per_sample_stats.csv](/home/jj/workspace/data-0313/outputs/analysis/curriculum_distribution/S2_per_sample_stats.csv) | 3591 | 统计文件 | 待确认 | 不适用 | 与当前 final S2_medium 不完全等价 | 待确认 |
| curriculum stats: S3/test | [S3_per_sample_stats.csv](/home/jj/workspace/data-0313/outputs/analysis/curriculum_distribution/S3_per_sample_stats.csv) | 3591 | 统计文件 | 是 | 不适用 | 间接匹配 | 保留 |

## 4. 实验目录与 checkpoint 清单

### 表 5：当前关键实验目录

| 实验目录 | 实验名称 | 对应模型 | epoch 数 | 是否 sanity / debug / 正式 24 epoch | best.pt 是否存在 | history.json 是否存在 | test 结果是否存在 | 是否建议保留 | 备注 |
| --- | --- | --- | ---: | --- | --- | --- | --- | --- | --- |
| [final_s123_curriculum/formal_24ep_gapaware_small](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/formal_24ep_gapaware_small) | 当前最终候选 | ACT-BiMamba | 24 | 正式 24 epoch | 是 | 是 | **待确认** | 保留 | 当前主线；test summary 目前全局未找到 |
| [final_s123_curriculum/formal_24ep_bimamba](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/formal_24ep_bimamba) | plain BiMamba | `bimamba_context_xyaux_zlinear` | 24 | 正式 24 epoch | 是 | 是 | 是 | 保留 | 当前正式 plain baseline |
| [final_s123_curriculum/ab24_xyaux_zlinear](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/ab24_xyaux_zlinear) | 结构基线 | `xyaux_zlinear` | 24 | 正式 24 epoch | 是 | 是 | 是 | 保留 | 当前主线消融基线 |
| [final_s123_curriculum/ab24_xyaux_zlinear_zadapter](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/ab24_xyaux_zlinear_zadapter) | 高度增强基线 | `zadapter` | 24 | 正式 24 epoch | 是 | 是 | 是 | 保留 | 当前主线消融基线 |
| [final_s123_curriculum/formal_24ep_unilstm_proto](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/formal_24ep_unilstm_proto) | 正式 baseline | UniLSTM-proto | 24 | 正式 24 epoch | 是 | 是 | 是 | 保留 | 正式 baseline |
| [final_s123_curriculum/formal_24ep_bilstm_proto](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/formal_24ep_bilstm_proto) | 正式 baseline | BiLSTM-proto | 24 | 正式 24 epoch | 是 | 是 | 是 | 保留 | 正式 baseline |
| [final_s123_curriculum/formal_24ep_cnnlstm_proto](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/formal_24ep_cnnlstm_proto) | 正式 baseline | CNN-LSTM-proto | 24 | 正式 24 epoch | 是 | 是 | 是 | 保留 | 正式 baseline |
| [final_s123_curriculum/formal_24ep_transformer_proto](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/formal_24ep_transformer_proto) | 正式 baseline | Transformer-proto | 24 | 正式 24 epoch | 是 | 是 | 是 | 保留 | 正式 baseline |
| [final_s123_curriculum/formal_24ep_mamba_proto](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/formal_24ep_mamba_proto) | 正式 baseline | Mamba-proto | 24 | 正式 24 epoch | 是 | 是 | 是 | 保留 | 正式 baseline |
| [clean_baselines_gaponly_v1/kalman_filter_clean_absolute](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/clean_baselines_gaponly_v1/kalman_filter_clean_absolute) | 正式 Kalman baseline | Kalman Filter | 待确认 | 基线目录 | 是 | 否 | stagewise 有 | 保留 | 不走共享神经框架；结果主要见 stagewise 目录 |
| [longgap_failure_diagnosis_20260531](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/longgap_failure_diagnosis_20260531) | long-gap diagnosis | 诊断目录 | - | 失败模式诊断 | 否 | 否 | 是 | 保留 | 当前分析结论依赖此目录 |

## 5. 历史旧版与疑似可清理文件

### 表 3：当前仍可继续清理的候选项

| 文件或目录路径 | 可能对应的实验/模型 | 为什么判断为旧版或中间版 | 是否仍被当前代码引用 | 是否包含重要结果 | 删除风险 | 建议操作 |
| --- | --- | --- | --- | --- | --- | --- |
| [backbones_transition_balanced_v1](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/backbones_transition_balanced_v1) | 旧 backbones 对比 | 目录名明确为旧 balanced 对比实验 | 未见当前正式配置引用 | 中 | 中 | 先归档 / 待确认 |
| [bidirectional_fusion_paper_experiments_20260519](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/bidirectional_fusion_paper_experiments_20260519) | 旧 fusion paper 实验 | 面向早期 fusion 对比，不是当前最终主线 | 未见当前正式配置引用 | 中 | 中 | 先归档 / 待确认 |
| [bidirectional_global_bins_physical_time_20260519](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/bidirectional_global_bins_physical_time_20260519) | 旧 physical-time 分箱实验 | 早期分析目录 | 未见当前正式配置引用 | 中 | 中 | 先归档 / 待确认 |
| [bidirectional_global_bins_physical_time_20260519_smoke](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/bidirectional_global_bins_physical_time_20260519_smoke) | smoke 版本 | 目录名含 `smoke` | 否 | 低 | 低 | 可删除但需人工确认 |
| [bidirectional_mechanism_analysis_20260518](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/bidirectional_mechanism_analysis_20260518) | 旧机制分析 | 分析型目录 | 未见当前正式配置引用 | 中 | 中 | 先归档 / 待确认 |
| [bidirectional_mechanism_analysis_physical_time_20260518](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/bidirectional_mechanism_analysis_physical_time_20260518) | 旧机制分析 physical time | 同上 | 未见当前正式配置引用 | 中 | 中 | 先归档 / 待确认 |
| [bidirectional_mechanism_case_scan_20260519](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/bidirectional_mechanism_case_scan_20260519) | 旧 case scan | 诊断型目录 | 未见当前正式配置引用 | 中 | 中 | 先归档 / 待确认 |
| [bimamba_context_xyzh_5ep_v1](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/bimamba_context_xyzh_5ep_v1) | 旧 xyzh 5ep | 5 epoch 旧分支 | 否 | 低到中 | 低 | 可删除但需人工确认 |
| [bimamba_longgap_batch_20260531](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/bimamba_longgap_batch_20260531) | 旧 long-gap batch 图/清单 | 批量分析输出，已被正式 failure diagnosis 替代 | 未见当前正式流程直接引用 | 中 | 中 | 待确认 |
| [curriculum_distribution_audit_20260528](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/curriculum_distribution_audit_20260528) | 旧 curriculum audit | 审计型目录 | 未见当前正式配置引用 | 中 | 中 | 先归档 / 待确认 |
| [mamba_recurrent_gapfusion_normfix_v2](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/mamba_recurrent_gapfusion_normfix_v2) | 旧 recurrent/gapfusion 路线 | 非当前最终主线 | 未见正式配置引用 | 中 | 中 | 先归档 / 待确认 |
| [obscons_gaponly_bilstm_vs_backbone_v1](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_bilstm_vs_backbone_v1) | 旧 backbone 对比 | 早期对比目录 | 未见当前正式配置引用 | 中 | 中 | 先归档 / 待确认 |
| [obscons_gaponly_bimamba_clean_v1](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_bimamba_clean_v1) | 旧 clean BiMamba | 旧主线 | 未见当前正式配置引用 | 中 | 中 | 先归档 / 待确认 |
| [obscons_gaponly_bimamba_clean_v2](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_bimamba_clean_v2) | 旧 clean BiMamba v2 | 同上 | 未见当前正式配置引用 | 中 | 中 | 先归档 / 待确认 |
| [obscons_gaponly_bimamba_hiddenfusion_anchorinit_v1](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_bimamba_hiddenfusion_anchorinit_v1) | 旧 hiddenfusion 分支 | 旧结构路线 | 未见当前正式配置引用 | 中 | 中 | 先归档 / 待确认 |
| [obscons_gaponly_bimamba_hiddenfusion_anchorloss_v2](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_bimamba_hiddenfusion_anchorloss_v2) | 旧 anchorloss 分支 | 同上 | 未见当前正式配置引用 | 中 | 中 | 先归档 / 待确认 |
| [obscons_gaponly_bimamba_hiddenfusion_anchorrelative_v3](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_bimamba_hiddenfusion_anchorrelative_v3) | 旧 anchorrelative 分支 | 同上 | 未见当前正式配置引用 | 中 | 中 | 先归档 / 待确认 |
| [obscons_gaponly_bimamba_recurrent_anchorfix_v4](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_bimamba_recurrent_anchorfix_v4) | 旧 recurrent 分支 | 同上 | 未见当前正式配置引用 | 中 | 中 | 先归档 / 待确认 |
| [obscons_gaponly_bimamba_recurrent_clean_v3](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_bimamba_recurrent_clean_v3) | 旧 recurrent clean 分支 | 同上 | 未见当前正式配置引用 | 中 | 中 | 先归档 / 待确认 |
| [obscons_gaponly_curriculum_a3_v1](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_curriculum_a3_v1) | 旧 curriculum ablation | 旧课程对比目录 | 未见当前正式配置引用 | 中 | 中 | 先归档 / 待确认 |
| [obscons_gaponly_fltp_v1](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_fltp_v1) | 旧 FLTP 路线 | 非当前主线 | 未见当前正式配置引用 | 中 | 中 | 先归档 / 待确认 |
| [obscons_gaponly_physical_time_ablation_v1](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_physical_time_ablation_v1) | 旧 physical-time ablation | 历史对比目录 | 未见当前正式配置引用 | 中 | 中 | 先归档 / 待确认 |
| [obscons_gaponly_physical_time_v1](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_physical_time_v1) | 旧 physical-time 路线 | 同上 | 未见当前正式配置引用 | 中 | 中 | 先归档 / 待确认 |
| [obscons_gaponly_ssvr_v1](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_ssvr_v1) | 旧 SSVR 路线 | 非当前主线 | 未见当前正式配置引用 | 中 | 中 | 先归档 / 待确认 |
| [obscons_gaponly_ssvr_v2](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_ssvr_v2) | 旧 SSVR 路线 v2 | 同上 | 未见当前正式配置引用 | 中 | 中 | 先归档 / 待确认 |
| [obscons_gaponly_ssvr_v3b_rho0](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/obscons_gaponly_ssvr_v3b_rho0) | 旧 SSVR 路线 v3 | 同上 | 未见当前正式配置引用 | 中 | 中 | 先归档 / 待确认 |
| [paper_combined_tables_20260518](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/paper_combined_tables_20260518) | 旧论文汇总表 | 可能被新汇总替代 | 未见当前正式训练流程引用 | 中 | 中 | 待确认 |
| [replanned_curriculum_audit_20260528](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/replanned_curriculum_audit_20260528) | 旧 replanned curriculum audit | 审计型目录 | 未见当前正式配置引用 | 中 | 中 | 先归档 / 待确认 |
| [rts_kalman_smoother_baseline_v1](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/rts_kalman_smoother_baseline_v1) | 旧独立 Kalman baseline 目录 | 当前正式 Kalman 结果已集中到 `clean_baselines_gaponly_v1` | 待确认 | 中 | 中 | 待确认 |
| [selected_recovery_compare_all_models_20260601](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/selected_recovery_compare_all_models_20260601) | 旧选样多模型对比图 | 若当前只保留 top10/正式图，则可视为旧展示目录 | 否 | 低到中 | 中 | 待确认 |
| [single_gap_forward_backward_fusion_20260519](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/single_gap_forward_backward_fusion_20260519) | 旧单 gap fusion 分析 | 分析型目录 | 否 | 中 | 中 | 先归档 / 待确认 |
| [single_gap_forward_backward_fusion_20260519_clean](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/single_gap_forward_backward_fusion_20260519_clean) | 旧单 gap fusion clean 分析 | 同上 | 否 | 中 | 中 | 先归档 / 待确认 |
| [single_gap_forward_backward_fusion_20260519_clean_wide](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/single_gap_forward_backward_fusion_20260519_clean_wide) | 旧单 gap fusion wide 分析 | 同上 | 否 | 中 | 中 | 先归档 / 待确认 |
| [single_gap_forward_backward_fusion_20260519_cruise](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/single_gap_forward_backward_fusion_20260519_cruise) | 旧单 gap fusion cruise 分析 | 同上 | 否 | 中 | 中 | 先归档 / 待确认 |
| [stage_dataset_condition_stats_20260519](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/stage_dataset_condition_stats_20260519) | 旧 stage 条件统计 | 已有 `outputs/analysis/curriculum_distribution` 更接近当前口径 | 否 | 中 | 中 | 先归档 / 待确认 |

## 6. 代码引用关系

### 6.1 当前正式训练/评估实际引用链

1. 正式训练入口  
   [scripts/train.py](/home/jj/workspace/data-0313/scripts/train.py)  
   - 读取 YAML 配置；
   - 根据 `cfg["model"]["backbone_type"]` 构造 `TrajectoryRecoveryModel`；
   - 实际模型实现入口为 [src/models/full_model.py](/home/jj/workspace/data-0313/src/models/full_model.py)。

2. 正式评估入口  
   [scripts/evaluate.py](/home/jj/workspace/data-0313/scripts/evaluate.py)  
   - 优先使用命令行传入的 `--checkpoint`；
   - 若未传入，则默认读取 `Path(cfg["outputs"]["run_dir"]) / cfg["outputs"]["checkpoint_name"]`；
   - 当前配置普遍将 `checkpoint_name` 设为 `best.pt`，因此正式评估默认使用 best，而不是 last。

3. 当前最终候选配置  
   [formal_24ep_gapaware_small.yaml](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/configs/formal_24ep_gapaware_small.yaml)  
   实际引用：
   - `backbone_type: bimamba_context_xyaux_zlinear_zadapter_gapaware_small`
   - `training.curriculum.stage_paths.stage1 -> final_s123_curriculum/pools/S1_train.parquet`
   - `training.curriculum.stage_paths.stage2 -> final_s123_curriculum/pools/S2_medium_train.parquet`
   - `training.curriculum.stage_paths.stage3 -> final_s123_curriculum/pools/S3_train.parquet`

### 6.2 gapaware_small、coarsetrend、vprog 引用关系

- `gapaware_small`
  - 代码中存在：`src/models/full_model.py`
  - 当前正式配置引用：**是**
  - 正式训练目录：`formal_24ep_gapaware_small`

- `coarsetrend`
  - 代码中存在：`src/models/full_model.py`
  - 当前正式配置引用：**否**
  - 相关实验目录已删除

- `vprogaux`
  - 代码中存在：`src/models/full_model.py`
  - 当前正式配置引用：**否**
  - 相关实验目录已删除

- `vprog_resaux`
  - 代码中存在：`src/models/full_model.py`
  - 当前正式配置引用：**否**
  - 相关实验目录已删除

### 6.3 old S2 / S2_medium / S2_new 引用关系

- 当前**正式主线**引用的是：
  - `final_s123_curriculum/pools/S2_medium_train.parquet`
  - 上游源为 `outputs/mvp_merged_250_20260514_clean/stage2_medium_clean/samples.parquet`

- 仍在旧脚本或辅助分析代码中出现旧 S2 路径的例子：
  - `scripts/eval_obs_conditioned_stagewise.sh`
  - `scripts/analyze_curriculum_distribution.py`
  - `scripts/diagnose_gaponly_anchor_stability.py`
  - `scripts/retrain_clean_baselines_gaponly.sh`
  - `scripts/retrain_backbones_transition_balanced_20260523.sh`

结论：
- **正式训练主线不再使用旧 `stage2_clean` 或 `S2_new_train.parquet`。**
- `stagewise_eval_proto_compare_24e` 已在本轮切换到 `S2_medium`，但其历史输出需要重新执行后才会完全更新。
- 上述其余脚本若继续保留，后续也建议统一切换到 `S2_medium`；若不再使用，可列入下一轮删除候选。

### 6.4 baseline 脚本是否引用最终数据集

- 正式 `formal_24ep_*_proto` 训练配置：**是**，使用 `final_s123_curriculum/pools/S1_train.parquet`、`S2_medium_train.parquet`、`S3_train.parquet`
- `stagewise_eval_proto_compare_24e` 当前仍在用旧 `S2` 口径，需要后续改为 `S2_medium`
- 本轮全局搜索未找到 `formal_24ep_gapaware_small` 的 `main_task_metrics_test_summary_dim.csv`

## 7. 建议清理策略

### 立即保留

- `src/models/full_model.py`
- `src/models/sequence_baselines.py`
- `scripts/train.py`
- `scripts/evaluate.py`
- `outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/`
- `outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/configs/formal_24ep_gapaware_small.yaml`
- `outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/pools/S1_train.parquet`
- `outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/pools/S2_medium_train.parquet`
- `outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/pools/S3_train.parquet`
- `outputs/analysis/stagewise_eval_proto_compare_24e/`
- `outputs/experiments/obs_conditioned_gaponly/longgap_failure_diagnosis_20260531/`
- `outputs/experiments/obs_conditioned_gaponly/clean_baselines_gaponly_v1/`
- `outputs/runs/real_adsc_anchor_only_top10_all_models_20260601`
- `outputs/runs/adsb_reference_anchor3_all_models_20260601`

### 建议归档

- `backbones_transition_balanced_v1`
- `bidirectional_fusion_paper_experiments_20260519`
- `bidirectional_global_bins_physical_time_20260519`
- `bidirectional_mechanism_analysis_20260518`
- `bidirectional_mechanism_analysis_physical_time_20260518`
- `bidirectional_mechanism_case_scan_20260519`
- `curriculum_distribution_audit_20260528`
- `mamba_recurrent_gapfusion_normfix_v2`
- `obscons_gaponly_bilstm_vs_backbone_v1`
- `obscons_gaponly_bimamba_clean_v1`
- `obscons_gaponly_bimamba_clean_v2`
- `obscons_gaponly_bimamba_hiddenfusion_anchorinit_v1`
- `obscons_gaponly_bimamba_hiddenfusion_anchorloss_v2`
- `obscons_gaponly_bimamba_hiddenfusion_anchorrelative_v3`
- `obscons_gaponly_bimamba_recurrent_anchorfix_v4`
- `obscons_gaponly_bimamba_recurrent_clean_v3`
- `obscons_gaponly_curriculum_a3_v1`
- `obscons_gaponly_fltp_v1`
- `obscons_gaponly_physical_time_ablation_v1`
- `obscons_gaponly_physical_time_v1`
- `obscons_gaponly_ssvr_v1`
- `obscons_gaponly_ssvr_v2`
- `obscons_gaponly_ssvr_v3b_rho0`
- `paper_combined_tables_20260518`
- `replanned_curriculum_audit_20260528`
- `rts_kalman_smoother_baseline_v1`
- `single_gap_forward_backward_fusion_20260519*`
- `stage_dataset_condition_stats_20260519`

### 可删除但需人工确认

- `bidirectional_global_bins_physical_time_20260519_smoke`
- `bimamba_context_xyzh_5ep_v1`
- `bimamba_longgap_batch_20260531`
- `selected_recovery_compare_all_models_20260601`

### 不建议删除

- `longgap_failure_diagnosis_20260531`
- `final_s123_curriculum`
- `clean_baselines_gaponly_v1`
- `outputs/analysis/stagewise_eval_proto_compare_24e`
- 所有 `formal_24ep_*` 正式实验目录
- `outputs/runs/real_adsc_anchor_only_top10_all_models_20260601`
- `outputs/runs/adsb_reference_anchor3_all_models_20260601`

## 8. 待我确认的问题

1. [formal_24ep_gapaware_small](/home/jj/workspace/data-0313/outputs/experiments/obs_conditioned_gaponly/final_s123_curriculum/formal_24ep_gapaware_small) 中未发现 `main_task_metrics_test_summary_dim.csv`。  
   - 当前全局搜索也未发现该 run 的 test summary 落在别处。待确认是否尚未正式跑 test，或只做了 val 与质性可视化。

2. `stagewise_eval_proto_compare_24e` 当前 S2 评估口径仍使用旧 `stage2_clean`。  
   - 该脚本已在本轮改为 `S2_medium` 口径，但历史输出目录 `outputs/analysis/stagewise_eval_proto_compare_24e` 需要重跑后才会与新口径一致。

3. `curriculum_distribution/S2_per_sample_stats.csv` 是否仍需要保留。  
   - 它与当前 final `S2_medium` 口径不完全一致，若后续重做统计图，可能应以 `S2_medium` 重新生成。

4. 多个旧辅助脚本仍保留着已删除目录或旧 `stage2_clean` 路径的文字引用。  
   - 当前已确认仍存在引用的脚本包括：`scripts/eval_obs_conditioned_stagewise.sh`、`scripts/analyze_curriculum_distribution.py`、`scripts/diagnose_gaponly_anchor_stability.py`、`scripts/retrain_clean_baselines_gaponly.sh`、`scripts/retrain_backbones_transition_balanced_20260523.sh`、`scripts/run_proposed_curriculum_24e.sh`、`scripts/audit_curriculum_stage_distributions_20260528.py`、`scripts/audit_replanned_curriculum_20260528.py`、`scripts/analyze_fusion_and_gap_segments.py`、`scripts/finalize_s123_curriculum.py`。  
   - 其中 `scripts/finalize_s123_curriculum.py` 还保留了已删除的 `sanity_5ep` 和旧 `bimamba_xyaux_zlinear_24e_v1` 文字引用，若后续仍需用于再生成 final pools，应先修正；若不再使用，可列入下一轮删除候选。  
   - 这些脚本若后续不再使用，可列入下一轮删除候选；若需保留，则应统一修正路径。

## 确认后可执行的清理命令草案

> 以下仅为草案，**本轮未执行**。

```bash
# 1. 先建归档目录
mkdir -p outputs/archive/obs_conditioned_gaponly_legacy

# 2. 建议先归档的旧分析/旧路线目录
mv outputs/experiments/obs_conditioned_gaponly/backbones_transition_balanced_v1 outputs/archive/obs_conditioned_gaponly_legacy/
mv outputs/experiments/obs_conditioned_gaponly/bidirectional_fusion_paper_experiments_20260519 outputs/archive/obs_conditioned_gaponly_legacy/
mv outputs/experiments/obs_conditioned_gaponly/bidirectional_global_bins_physical_time_20260519 outputs/archive/obs_conditioned_gaponly_legacy/
mv outputs/experiments/obs_conditioned_gaponly/bidirectional_mechanism_analysis_20260518 outputs/archive/obs_conditioned_gaponly_legacy/
mv outputs/experiments/obs_conditioned_gaponly/bidirectional_mechanism_analysis_physical_time_20260518 outputs/archive/obs_conditioned_gaponly_legacy/
mv outputs/experiments/obs_conditioned_gaponly/bidirectional_mechanism_case_scan_20260519 outputs/archive/obs_conditioned_gaponly_legacy/
mv outputs/experiments/obs_conditioned_gaponly/curriculum_distribution_audit_20260528 outputs/archive/obs_conditioned_gaponly_legacy/
mv outputs/experiments/obs_conditioned_gaponly/replanned_curriculum_audit_20260528 outputs/archive/obs_conditioned_gaponly_legacy/

# 3. 若确认无用，可删除低风险剩余目录
rm -rf outputs/experiments/obs_conditioned_gaponly/bidirectional_global_bins_physical_time_20260519_smoke
rm -rf outputs/experiments/obs_conditioned_gaponly/bimamba_context_xyzh_5ep_v1
rm -rf outputs/experiments/obs_conditioned_gaponly/bimamba_longgap_batch_20260531
```
