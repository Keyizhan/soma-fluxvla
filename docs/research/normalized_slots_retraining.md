# 归一化 + slot 重训练建议（2026-10-23）

推荐配置：`configs/subtask/loss_examples/recommended_retrain.json`。按用户最新要求部署并重启 Stage 1；运行命令必须带 `--stop-after-stage stage1`。Stage 2 暂不启动。

## 已实现

- Separate 模式 FAST/Subtask 原始比例从 1:1 线性变化至 10:1，逐步自动归一化。
- Subtask 的 key/template 原始比例 1:0.1，归一化为 10/11:1/11。
- 初始目标为 `0.5*FAST + (5/11)*key + (0.5/11)*template`；末步为 `(10/11)*FAST + (10/121)*key + (1/121)*template`，系数合计 1。
- 动态模式也先插值、后归一化；本次按用户要求启用 1→10。
- Unified+split+normalize 用加权 token 数作分母；它与 Separate 的区域均值目标不同，原始未加权 CE 日志保留。
- 不归一化 Flow 与 CE 的组合。继续使用现有独立 Flow 系数；不能把不同单位的损失简单当成同一尺度。

显式通用模板示例：`{action} fabric {object} in the {destination} basket using the {arm} arm`。模板在配置层描述业务语义，解析器不写死业务句式。复合句同时标记两个动作及并行分隔符，避免只强化对象而漏掉动作类别/并行结构。

真实训练 tokenizer 的 CPU 掩码检查：15/15 标签对齐、无 fallback；key token 比例 train=38.2883%、validation=38.0610%。来源 `D:/pi05_subtask/audits/loss_diagnostics_20261023/explicit_slot_probe.json`。该结果只证明覆盖与对齐，不证明模型已经学好关键词。加载器仍有上游 regex 警告，本次未更换 tokenizer，以免改变当前 token 合约。

## 推荐的短程重训练

1. 从原始 pi05_base 重新初始化，optimizer 重置；保留现有 train/validation/test 划分、动作缓存、seed、batch 和归一化数据统计。不要从当前可能已泛化退化的后期权重开始。
2. Stage 1 训练 3 epochs，峰值 LR 1e-4、warmup 5%；FAST 原始系数在本阶段总更新内从1线性增加到10，Subtask保持1，再归一化。slot key/template 原始比例10:1。每 epoch 保存并验证，保留全部3个 epoch 检查点，另导出按 FAST 验证 CE 选出的 final.deploy。
3. 当前 train anchors=155515、global batch=64，对应每 epoch 2430 updates，总计7290；检查点位于2430、4860、7290。检查点为部署权重包，不能当作包含 optimizer 的精确续训快照。
4. 使用 `python -m fluxvla.training.subtask.run --config configs/subtask/loss_examples/recommended_retrain.json --stop-after-stage stage1`。Stage 2 配置保留用于未来评估，但本次调度在 Stage 1 receipt 写入后退出，不运行 Stage 2。
5. 三个 epoch 完成后比较未加权 FAST/key/template 验证 CE、关键字段自主生成准确率和动作解码误差；以验证结果选 checkpoint，再决定 Stage 2。动态总 loss 不适合跨步直接比较。建议同预算统一 CE 对照；本次运行本身不构成因果对照。

## 为什么不直接加大 FAST 权重

既有记录的 FAST 验证 CE 从 epoch1=3.2087 升至 epoch2=3.6430，同时末50步训练 CE 从2.6089降至1.9361。Subtask 验证也从0.03780略升至0.03897。这支持泛化退化的初步判断，不支持“FAST仍欠拟合，所以强加权”。只有两个验证点，仍需更多证据。

若验证 FAST 继续上升，优先选更早模型、检查数据覆盖和有效样本多样性；不要用更大 FAST 权重去追训练 CE。若 key 验证表现差而模板已好，调整 key/FAST 的相对投入；slot 只能改变监督位置，不能消除数据捷径。

## 归一化的边界

系数合计1仅消除系数总和变化导致的直接倍率变化，不能保证真实梯度范数恒定、两个任务不冲突或 Flow/CE 的有效梯度比例恒定。空 key/template 区域贡献0，不将空区域预算重新分给其他区域。动态系数归一化不是 GradNorm，也不依据验证集自动调权。
