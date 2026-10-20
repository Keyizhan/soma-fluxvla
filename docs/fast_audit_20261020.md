# FAST 离线诊断与配方消融（2026-10-20）

本次结果支持“训练样本记忆增强、验证泛化未改善”的判断。没有发现 FAST
编码或 token 映射使损失无法下降的证据。这个结论不等于排除了所有实现问题，
也不能把 teacher-forced FAST 指标直接解释为真机任务成功率。

本次只新增离线诊断工具，没有更改生产训练目标、部署参数或权重，没有运行真机。

## 数据与可比性

- 原始日志：Stage1 1,264 次更新、4 次验证；Stage2 2,528 次更新、8 次验证。
- 原训练缓存共 23,224 条记录；按 `(source, episode)` 核对，93 个训练 episode
  和 13 个验证 episode 无交集。缓存中 103 条没有动作目标，4,721 条动作长度
  为 1–49，18,400 条长度为 50。
- 编解码与权重对照：固定 128 条原缓存记录，训练/验证各 64 条，每个划分
  各含 32 条短 horizon、32 条完整 50-step horizon，抽样 seed 为 20261020。
- 缓存标识与部署包元数据一致：
  `7dddfd240b01b28835919125c8e462347710e47f5391488446eea24fa6f851b1`。
  原 `samples.json` SHA256：
  `6a944794baf02cf769cba73688c433ecf18c07b2d3fc4150122d6b2e0318c9ac`。
- 128 条是分层诊断样本，短 horizon 被过采样；以下结果不是全验证集指标。
  图像、状态、动作和 token 均来自原训练缓存，不是现场新采样。
- 本次模型、prompt 和数据准备代码基线：
  `e9c0bdd5175b545d8383293e8feac90f2a6e38da`。

## 1. 编解码没有出现本次怀疑的错位

128/128 条动作重新编码后，FAST token 与缓存完全一致；解码系数数量全部匹配
原有效 horizon × 14 维。未发现低于 tokenizer 最小值而被裁剪的 DCT 系数。

以下是逐样本误差再取组均值。关节误差单位 rad，夹爪误差单位 mm。
这些是“真实动作 → FAST → 重建动作”的量化误差，不是模型预测误差。

| 划分 | horizon | 样本数 | 关节 RMSE | 夹爪 MAE |
|---|---|---:|---:|---:|
| 训练 | 1–49 | 32 | 0.00985 | 0.457 |
| 训练 | 50 | 32 | 0.00803 | 0.357 |
| 验证 | 1–49 | 32 | 0.00947 | 0.510 |
| 验证 | 50 | 32 | 0.00861 | 0.334 |

这里没有出现足以说明整体 FAST 失效的重建异常。小量化误差对具体抓取的影响，
仍需要结合动作误差和任务容忍度判断。

## 2. 单独统计 FAST 后，训练与验证差距更清楚

原日志的 `fast_ce` 对整个 action 区域求均值，包含 `Action:` 等结构文本、
FAST token 和结束符。本次把纯 FAST 与结构 token 分开统计；纯 FAST CE
仍使用完整模型词表作为 softmax 分母，没有改变分类问题。

| 权重 | 训练纯 FAST CE ↓ | 训练 token 准确率 ↑ | 验证纯 FAST CE ↓ | 验证 token 准确率 ↑ |
|---|---:|---:|---:|---:|
| step-750 | 2.888 | 35.23% | 4.154 | 15.95% |
| step-1500 | 1.595 | 68.24% | 4.523 | 14.81% |

两份权重在完全相同的 128 条样本上评估。继续训练后，训练样本明显变好，
验证样本反而变差。

step-750 的结构 token CE 在训练/验证分别约 0.0007 / 0.0170，
会拉低混合统计的 `fast_ce`。模型分给 FAST 词表的概率质量约为 99.3% / 99.2%：
它基本知道这里应输出 FAST，困难主要在具体 FAST 符号的选择。

step-750 验证短/完整 horizon CE 分别为 4.175 / 4.144，本次没有发现
短 horizon 是主要异常来源。

**这些都是 teacher forcing 指标**：预测每个 token 时提供真实前缀，
不等于自由生成完整动作序列的准确率，更不等于 subtask 句级准确率。

## 3. 16 样本、四组匹配起点消融

四组全部从同一个 step-750 完整权重启动，重新建立 AdamW 优化器。
训练 16 条、独立验证 16 条，各自包含 8 条短 horizon 和 8 条完整 horizon。
四组的训练顺序、样本、随机种子和初始评估完全一致。

- 32 次更新；microbatch 1、累积 4，effective batch 4。
- LR 2e-5，AdamW weight decay 0.01，gradient clip 1。
- BF16 参数和 Adam 状态；seed 1702；固定输入尺寸 224。
- flow 评估使用固定噪声/时间随机种子和有效动作 mask。
- 四组均正常完成 32 次更新，loss/梯度全部有限；不保存新权重。

| 配方 | subtask / FAST / flow 权重 | 隔离 flow→prefix |
|---|---|---|
| FAST-only | 0 / 1 / 0 | 不训练 flow |
| Stage1 目标 | 10 / 1 / 0 | 不训练 flow |
| 原混合目标 | 0.15 / 0.15 / 1 | 否 |
| 混合 + 梯度隔离 | 0.15 / 0.15 / 1 | 是 |

“Stage1 目标”只是换成 Stage1 的损失权重，**不是原 Stage1 初始化重训**。
所有配方训练时 FAST loss 都沿用原 action 区域定义，评估才单独拆出纯 FAST。

| 配方 | 训练纯 FAST CE | 验证纯 FAST CE | 训练 token 准确率 | 验证 token 准确率 | 验证 flow MSE |
|---|---:|---:|---:|---:|---:|
| 共同初始值 | 2.913 | 4.029 | 33.40% | 15.57% | 0.031769 |
| FAST-only，32 updates | 0.103 | 4.369 | 100% | 15.27% | 0.032158 |
| Stage1 目标，32 updates | 0.105 | 4.365 | 100% | 15.42% | 0.032077 |
| 原混合目标，32 updates | 0.136 | 4.350 | 100% | 14.54% | 0.032212 |
| 混合 + 梯度隔离，32 updates | 0.140 | 4.351 | 100% | 15.27% | 0.032156 |

可得出的结论：

1. FAST 分支可以快速学会这批目标，CE 没有被实现固定在 3 附近。
2. 不训练 flow 时，训练/验证分化仍然出现；flow 梯度干扰不是这个现象的必要条件。
3. 这次隔离 flow 梯度没有挽救验证 FAST CE。

本实验有意用小样本检查“能否学会”和配方差异。它不能证明全量训练时隔离梯度无用，
也不足以从 16 条验证样本、单个随机种子中选定最终训练配方。
FAST-only 的 flow expert 虽冻结，共享前缀仍会变化，因此 flow MSE 不必保持不变。

## 4. epoch 阶梯仍需要进一步定位

原始日志更新序号连续，记录的 epoch 一致，LR 与配方的连续调度计算完全吻合；
绘图平滑没有制造原日志不存在的阶梯。

按原配置重放 sampler，每轮训练样本恰好出现一次，轮间 shuffle 不同。
各轮首尾 1,280 个样本的平均有效 horizon 约为 44，未见大幅长度组成变化。
这是采样算法重放，原日志没有逐 step 样本 ID，因此不能替代真实执行轨迹。

Stage2 后续七个 epoch 边界，后 20 步减前 20 步的 FAST CE 均值分别为：
−0.167、−0.354、−0.502、−0.434、−0.302、−0.163、−0.017。
小样本记忆实验与过拟合解释相容，但还不能据此确定阶梯形成的唯一原因。

额外 16 条探针上，同权重的 train/eval 模式 CE 差值为 0；
交换图像会改变预测，说明模型并非完全忽略图像。探针样本少，不据此推断
完整验证集的视觉泛化能力。

## 后续优先级

1. 在训练监控中同时记录纯 FAST CE、token accuracy 和结构 token CE，
   保留原混合 loss 用于复现，不再只看一个混合数字。
2. 将 step-750 保留为后续离线比较基线。对完整验证集做自由生成、解码后
   关节/夹爪误差及 flow 动作误差评估，再决定部署权重。
3. 检查按 episode 划分后的场景、物体、动作覆盖，做保持动作标签一致的视觉增强
   和训练长度对照。比较多个种子，避免继续仅压低训练 CE。
4. 若要定位 epoch 阶梯，下一轮增加逐 step 样本 ID 和分组损失记录；
   本次没有足够证据去修改 sampler 或默认训练目标。

## 复现工具

先按仓库安装说明准备环境。`FULL` 是原缓存目录，包含 `samples.json`、
`valid_lengths.npy`、`fast_tokens.npz`；`SUBSET` 是该缓存的行切片，保留
`normalization` 和 records 的 split/source/episode/valid_length，并增加
原行号 `source_index`，数组采用现有 `Fixtures` 的 `arrays/*.npy` 布局。
抽样必须同时切片图像、状态、动作、mask、长度和 FAST token，保持行一一对应。

`SOURCE` 包含 `stage1.jsonl`、`stage2.jsonl` 及对应 `_val.jsonl`；
采样顺序重放针对原运行的 8 个 rank。`BUNDLE` 是本地完整部署包路径。

```bash
python scripts/diagnostics/audit_fast.py logs \
  --source "$SOURCE" --full "$FULL" --output artifacts/fast/logs.json
python scripts/diagnostics/audit_fast.py codec \
  --subset "$SUBSET" --full "$FULL" --output artifacts/fast/codec.json
python scripts/diagnostics/audit_fast.py score \
  --bundle "$BUNDLE" --subset "$SUBSET" --output artifacts/fast/score.json
python scripts/diagnostics/ablate_fast.py \
  --bundle "$BUNDLE" --subset "$SUBSET" --arm hybrid_insulated \
  --updates 32 --batch-size 4 --lr 2e-5 --output artifacts/fast/insulated
```

其他 arm 为 `fast_only`、`stage1_objective`、`hybrid`，输出目录必须不存在。
GPU 模型评分和小训练在 96GB RTX PRO 6000 Blackwell 上完成；
本次小训练显存峰值约 36GiB。数据、图像、原始日志和权重均不随代码发布。
