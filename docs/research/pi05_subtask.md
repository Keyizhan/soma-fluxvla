# π0.5 Subtask 调研与复现增量

更新日期：2026-10-17。研究基准：π0.5 原论文 arXiv:2504.16054v1；代码基准：本仓库当前文件，见下方代码来源。

目标是在已有 π0.5 动作训练基础上增加显式 subtask 预测与执行。本文区分论文证据、本库实际行为和 AI 提出的实现设计。本文是调研与实现规格，尚未实现、运行或验证所提方案，不能作为论文全部训练配方的复现声明。

## 1. 信息来源与可信边界

### 1.1 标记规则

| 标记 | 谁提供的信息 | 可作何种使用 |
|---|---|---|
| `[官方论文 Fxx]` | Physical Intelligence 作者在论文正文、附录或图中给出 | 作为论文方法事实；不自动证明公开权重或本库具备该能力 |
| `[本库代码 Cxx]` | AI 阅读当前本地代码后观察到的行为 | 作为静态代码事实；不是官方声明，也不是运行验证结果 |
| `[AI 推导 Dxx]` | AI 从公式、代码行为推导的约束或风险 | 必须经过实现和验证；不能归因于论文作者 |
| `[AI 建议 Axx]` | AI 为复现提出的接口、模板、参数策略或流程 | 可实施方案，未经验证，不代表作者原实现 |
| `[未披露 Uxx]` | 本次检查的论文相关章节和图中未找到明确规定 | 表示本次证据不足，不表示官方在其他渠道从未说明 |
| `[AI 读图 E01]` | AI 从官方图像估读数值 | 近似值，不是作者发布的数值表 |

每个实现小节的来源声明覆盖该节的表格和伪代码；混合来源逐条标记。伪代码均由 AI 编写，不是论文或官方仓库摘录。代码链接记录本次阅读位置，后续修改可能使行号变化；函数名可用于重新定位。

### 1.2 官方原始资料

论文：*π0.5: a Vision-Language-Action Model with Open-World Generalization*，Physical Intelligence，2025-04-22。

- [论文 HTML v1](https://arxiv.org/html/2504.16054v1)
- [论文 PDF v1](https://arxiv.org/pdf/2504.16054v1)
- [IV-A 架构](https://arxiv.org/html/2504.16054v1#S4.SS1)
- [IV-B 离散与连续动作](https://arxiv.org/html/2504.16054v1#S4.SS2)
- [IV-C 预训练](https://arxiv.org/html/2504.16054v1#S4.SS3)
- [IV-D 后训练](https://arxiv.org/html/2504.16054v1#S4.SS4)
- [IV-E 机器人系统](https://arxiv.org/html/2504.16054v1#S4.SS5)
- [V-E 高层推理消融](https://arxiv.org/html/2504.16054v1#S5.SS5)
- [附录 A-E 技术细节](https://arxiv.org/html/2504.16054v1#A1.SS5)
- [图 4 训练任务示例](https://arxiv.org/html/2504.16054v1/pretraining-finetuning-tasks.png)
- [图 13 高层消融](https://arxiv.org/html/2504.16054v1/figures/performance_vs_hl_data.png)
- [图 17 逐任务消融](https://arxiv.org/html/2504.16054v1/figures/HL_breakdowns.png)
- [图 18 注意力掩码](https://arxiv.org/html/2504.16054v1/figures/attention_mask.png)

### 1.3 官方事实索引

本表只压缩记录复现所需的事实；后面的具体接口、mask 构造和调度不是作者公开代码。

| ID | 官方论文明确的信息 | 原始位置 |
|---|---|---|
| F01 | 同一模型做高层文本与低层动作；低层条件是观察和子任务 | IV-A；II |
| F02 | 子任务文本自回归生成；动作用 10 步去噪 | IV-B |
| F03 | 状态以离散文本 token 输入 | IV-A；A-E |
| F04 | PaliGemma 主干与较小 action expert 通过 attention 交互 | A-E |
| F05 | prefix 双向；FAST 因果；expert 不读 FAST；主干不读 expert | A-E；图 18 |
| F06 | 预训练用离散 token，后训练联合 CE 与 flow matching | IV-B/D |
| F07 | 后训练加入随机初始化 expert；组合损失 α=10 | IV-D |
| F08 | expert 使用时间 MLP 与 adaptive RMSNorm；动作块为 50 | A-E |
| F09 | 高层四相机，低层使用前方及双腕相机；高层频率较低 | IV-E；II |
| F10 | HL 使用人工子任务及相关目标框；VI 使用在线语言示范 | IV-C/D；图 4 |
| F11 | implicit HL 保留训练高层监督，运行时不显式生成子任务 | V-E |

### 1.4 本库代码来源

以下不是“官方说的”，而是 AI 对本地实现的静态检查。

| ID | 本库事实 | 代码位置 |
|---|---|---|
| C01 | `PI05FlowMatching` 继承 PI0；本类主要覆盖动作 suffix 嵌入和时间条件 | [pi05_flowmatching.py](../../fluxvla/models/vlas/pi05_flowmatching.py) |
| C02 | `embed_prefix()` 把全部输入语言置于双向 prefix；语言 embedding 乘 `sqrt(hidden_size)` | [pi0_flowmatching.py:204](../../fluxvla/models/vlas/pi0_flowmatching.py#L204) |
| C03 | `forward_model()` 对外只返回 suffix 与 prefix KV；语言 hidden states 未对外暴露 | [pi0_flowmatching.py:441](../../fluxvla/models/vlas/pi0_flowmatching.py#L441) |
| C04 | 当前训练 `forward()` 计算动作 MSE，再使用动作 mask 与样本权重归约 | [pi0_flowmatching.py:537](../../fluxvla/models/vlas/pi0_flowmatching.py#L537) |
| C05 | 当前加噪约定是 `x_t=t*noise+(1-t)*actions`，目标为 `noise-actions` | 同 C04 |
| C06 | prompt 为 `Task: ..., State: ...;\nAction: `，状态离散化为文本 | [prompters.py:14](../../fluxvla/transforms/prompters.py#L14) |
| C07 | `ProcessPrompts` 按长度直接截断；`with_labels` 按动作维度计算监督区间 | [transform_prompts.py:23](../../fluxvla/transforms/transform_prompts.py#L23) |
| C08 | mask 工具根据一维标志的 cumsum 构造块注意力 | [model_utils.py:77](../../fluxvla/engines/utils/model_utils.py#L77) |
| C09 | 语言主干返回最终 hidden states 与 KV，不在该 forward 中输出词表 logits | [condition_gemma.py](../../fluxvla/models/backbones/llms/condition_gemma.py)，`ConditionGemmaModel.forward` |
| C10 | 当前配置词表 257152、主干宽 2048、expert 宽 1024；embedding 映射涉及基础权重 lm_head | [pi05_piper.py](../../configs/pi05_piper.py)，`model` |
| C11 | 当前 collator 未配置高层目标字段；已有 RTC 训练配置 | [pi05_piper.py](../../configs/pi05_piper.py)，`runner.collator` 与 `rtc_training_config` |

未检查基础 checkpoint 内各矩阵的逐值相等关系，未验证输出 head 的绑权、logits 标度或公开权重是否已有可直接使用的子任务生成能力。

## 2. 复现范围与条件关系

### 2.1 目标分解

来源：`[官方论文 F01]`；变量简写和接口示意：`[AI 建议 A01]`。

```text
g = 总体任务
s = 当前子任务文本
o = 当前观察，包括图像和状态
a = 动作块

高层：p(s | o, g)
低层：p(a | o, s)
联合：p(a, s | o, g) = p(a | o, s) p(s | o, g)
```

`[AI 推导 D01]` 要遵循这个低层条件关系，低层输入和融合 KV 中不能保留总体任务信息。把 `g` 与 `s` 同时送入低层是可研究的变体，但不能称为上述分解的直接实现。

`[AI 建议 A01]` 第一版采用共享参数、两次独立前向：

```python
subtask = model.generate_subtask(observation, overall_task)
actions = model.sample_actions(observation, subtask)
```

### 2.2 范围边界

来源：`[AI 建议 A02]`；论文训练配方依据：`[官方论文 F06/F10]`。

第一版目标是基于已有 π0.5 增加显式 subtask。保留现有动作 expert、动作归一化和 flow 时间约定；增加高层文本训练、生成与条件调度。第一版不补全 FAST 共训练、目标框预测、原始多源数据混合等全部配方。完成后只能称“π0.5 subtask 扩展”；不能称论文全配方复现。

本文没有证明哪一项技术改动单独贡献最高性能。论文的完整方案与 implicit HL 对比，不能进一步分解为模板、缓存或某种 mask 实现的单项增益。

## 3. 数据进入模型时的两条分支

来源：以下字段、模板、拆分方式全部为 `[AI 建议 A03]`。依据是 F01 与本库 C06；不是官方提示模板。

同一时刻原始记录：

```text
image/state：当前观察
overall_task：总体任务
subtask：当前时刻的监督子任务
actions：未来动作块
```

高层分支：

```text
条件：图像 + 总体任务 + 状态 + 输出提示
目标：子任务 token + EOS
```

低层分支：

```text
条件：图像 + 子任务 + 状态
目标：连续动作块
```

示例：

```text
高层条件：
Task: put the shoes on the rack, State: ...;
Subtask:

高层目标：
pick up the left shoe<EOS>

低层条件：
Task: pick up the left shoe, State: ...;
Action:
```

保持低层 `Task/State/Action` 格式是为了减少相对本库输入格式的改动；其是否最优尚未验证。不要给 `Subtask:` 随意注册一个新特殊 token；先用现有 tokenizer 的普通文本编码，避免新增词表参数。高层/低层必须使用同一套状态归一化、离散化及文本清理约定。

`[AI 推导 D02]` 高层条件中不能放入真实子任务答案，再把生成正确率当作高层能力。低层训练使用真实子任务不构成这种泄漏，因为低层的任务本来就是在给定子任务时预测动作。

## 4. Batch 张量与 tokenizer 合同

来源：本节接口和处理规则全部为 `[AI 建议 A04]`；当前截断风险依据 `[本库代码 C07]`。

| 字段 | 形状 | 含义 |
|---|---|---|
| `hl_input_ids` | `[B, Lh]` | 高层条件与 teacher-forcing 目标 |
| `hl_valid_mask` | `[B, Lh]` | 文本有效位置 |
| `hl_ar_mask` | `[B, Lh]` | 进入新因果块的位置 |
| `hl_labels` | `[B, Lh]` | 条件/padding 为 -100，目标为 ID |
| `ll_input_ids` | `[B, Ll]` | 低层子任务条件 |
| `ll_valid_mask` | `[B, Ll]` | 低层文本有效位置 |
| `actions` | `[B, T, D]` | 现有动作目标 |
| `action_masks` | 沿用当前广播合同 | 有效动作监督位置 |
| `has_hl` | `[B]` | 有无高层目标 |
| `has_action` | `[B]` | 有无动作目标 |

图像及图像有效 mask 沿用图像管线；如果高低层视角不同，单独表达视角选择，不可依赖位置猜测。

必须执行的规则：

1. 有效 mask、因果块 mask、loss mask 分别构造。
2. 条件参与 attention，不参与 CE。
3. EOS 参与 CE；padding 不参与 CE。
4. 显式读取 tokenizer 的 BOS/EOS/PAD，核实自动添加行为，避免重复 BOS/EOS。
5. 不默认“字符串片段分别编码后拼接”等于“整串编码”。选择一种固定分段合同；用最终 token 序列确定目标边界，不能以字符长度代替 token 长度。
6. 保证条件末尾有有效提示 token，可预测第一个目标 token。
7. 禁止静默截断目标或 EOS；超长样本报错或按明确、记录在配置中的规则处理。
8. Padding ID 使用真实 PAD，不以布尔 False 隐式替代。
9. 记录模板版本、文本大小写/空白清理、状态维度和离散化版本。
10. 输出 side 的状态字符串和输入 side 完全一致；不得在新路径重复归一化。

## 5. 高层注意力 mask 的逐 token 实现

来源：自回归需求 `[官方论文 F02]`；本库工具 `[本库代码 C08]`；以下具体块矩阵是 `[AI 建议 A05]`，不是图 18 的逐格抄录。

设 P 为图像、总体任务、状态及输出提示，S 为子任务目标。

| Query / Key | P | S |
|---|---|---|
| P | 双向允许 | 禁止 |
| S | 允许 | 因果允许 |

目标为 `s0 s1 s2 EOS` 时，每个目标输入位置只读取 P 和不晚于自身的目标输入。预测 label 再向后移一位，见第 7 节。

本库工具可表达此结构：

```python
# AI 伪代码；True 表示允许读取。
block_ids = torch.cumsum(ar_mask, dim=1)
allowed = block_ids[:, None, :] <= block_ids[:, :, None]
allowed &= valid[:, None, :] & valid[:, :, None]
```

对应一维标志：

```text
P: 0 0 0 0 ...
S: 1 1 1 1 ...
```

`[AI 推导 D03]` 如果 P 能读取未来子任务，未来答案可经 P 的 hidden states 回流至当前预测位置。仅把 S 内部改为三角形仍不够。

`[本库代码 C02]` 当前所有语言位置填 0，不能把目标 S 原样塞入现有双向 prefix。

`[官方论文 F05]` 图 18 展示 prefix、FAST、expert 的关系；`[未披露 U01]` 该图没有单独画 subtask 目标 token 区域，不能用它声称已经获得作者完整高层 mask。

实现细节 `[AI 建议 A05]`：布尔 mask 转加性 mask 时检查当前 attention API 对 True 的语义；padding query 的全屏蔽行需验证不产生 NaN；有效输出不得依赖 padding 内容。高层训练、prefill、增量 decode 必须采用一致的可见关系。

## 6. 语言 hidden states 与词表投影

来源：代码现状 `[本库代码 C03/C09/C10]`；以下接法 `[AI 建议 A06]`。

```text
主干 hidden states：[B, Nh, 2048]
词表 logits：       [B, Nh, V]
当前配置 V：         257152
```

新增返回接口可返回字典，避免破坏旧 tuple 调用者：

```python
# AI 接口示意
{
    "prefix_hidden_states": ...,
    "action_hidden_states": ...,
    "prefix_key_values": ...,
}
```

可选绑权方案：

```python
# 只有确认 checkpoint 的输出投影合同后才能采用。
logits = F.linear(text_hidden_states, llm_backbone.embed_tokens.weight)
```

不能直接把这个候选方案当作已验证事实。实施前核对：

- 原 checkpoint 输入 embedding、输出 head 的权重关系。
- 当前名称映射实际加载的张量、形状与词表顺序。
- 输出前是否已做最终 norm；不得漏掉或重复 norm。
- 是否存在额外 logits 标度或处理。
- tokenizer vocabulary 与输出 V 完全对应。
- 保存和重载后绑权关系保持；优化器不重复管理同一参数。

`[本库代码 C02]` 输入语言 embedding 已乘 `sqrt(hidden_size)`。`[AI 建议 A06]` 新高层训练和 decode 路径必须保持一致；输入 scaling 不意味着输出 logits 再乘同一系数。直接调用主干传 `inputs_embeds` 时尤其需要核对。

显存优化 `[AI 建议 A06]`：正确性基线通过后，只对负责预测目标的 hidden states 投影到 V，避免给全部图像位置分配词表 logits。不要在未确认索引前优化。

## 7. CE 对齐到每个 token

来源：本节索引、伪代码和归一化均为 `[AI 建议 A07]`；语言生成监督需求依据 F02/F06。

```text
位置：0 ... c-1 | c    c+1 c+2  c+3
输入：条件      | pick up  shoe EOS
标签：-100 ...  | pick up  shoe EOS
```

正确预测关系：

```text
logits[c-1] → pick
logits[c]   → up
logits[c+1] → shoe
logits[c+2] → EOS
```

```python
# AI 伪代码；logits 与 labels 必须已经位于同一序列坐标系。
shift_logits = logits[:, :-1, :]
shift_labels = labels[:, 1:]
loss_sum = F.cross_entropy(
    shift_logits.float().reshape(-1, vocab_size),
    shift_labels.reshape(-1),
    ignore_index=-100,
    reduction="sum",
)
num_targets = (shift_labels != -100).sum()
loss_hl = loss_sum / num_targets.clamp_min(1)
```

若 logits 包含图像位置，labels 必须前置对应数量的 -100；也可先取文本 hidden states，再对纯文本 labels 做 shift，二选一并固定合同。不能把纯文本 labels 直接对齐全模态序列。

`[AI 推导 D04]` 用同位置 hidden state 预测同位置 token 会看到答案，不能作为 next-token 训练。EOS 必须受监督，否则停止行为没有对应目标。

零目标 batch 的处理 `[AI 建议 A07]`：不要直接使用对全 -100 label 做 mean CE 的结果；返回连接正确计算图的零损失或在明确分支中跳过，并验证 DDP unused-parameter 行为。

## 8. 两分支训练与梯度

来源：以下具体训练组织为 `[AI 建议 A08]`；论文组合训练依据 F06/F07。

```python
loss_hl = forward_high_level(o, overall_task, target_subtask)
loss_action = forward_low_level(o, ground_truth_subtask, actions)
loss = lambda_hl * loss_hl + lambda_action * loss_action
```

第一版低层使用真实子任务；部署使用高层预测子任务。两次前向共享主干参数，不是训练两个独立 VLM。第一版不自动使用高层 argmax 结果训练低层，以便分离高层错误和动作错误；是否后续采用预测条件训练是额外实验。

`[AI 推导 D05]` 此方案存在真实子任务训练与预测子任务部署的分布差异，需闭环评估。attention 单向不等于梯度 stop-gradient；两次前向也不会自动隔离共享参数梯度。

实现要求 `[AI 建议 A08]`：

- CE 按有效目标 token 归约，动作沿用本库 mask 与归约。
- 单独记录两个 loss、有效数量及权重。
- 论文 α=10 不自动等于删减配方后的最佳权重。
- 无高层目标/无动作目标的样本分别屏蔽相应监督。
- 冻结策略写入配置；若希望主干适应高层任务，不能在训练前向外层误用 no_grad。
- 不默认加 detach 或 knowledge insulation；若添加，明确作为不同训练方案。
- DDP 中使用全局有效数量。若期望全局 token 均值，在常规 DDP 梯度按 world_size 平均的条件下，本地 loss 可用 `world_size * local_loss_sum / global_count`，global_count 为无梯度聚合值；须按实际训练器归约行为验证。
- 一次总 loss backward，或正确累积后一次 optimizer step；不得无意把两分支变成两次不同参数状态的更新。
- 新 checkpoint 保存训练权重、模板、tokenizer 和 loss 合同；resume 时检查兼容。

## 9. 动作 flow 的时间方向与保留项

来源：公式是 `[本库代码 C05]`；积分示意为 `[AI 推导 D06]`，不是已运行的采样器。

```python
x_t = t * noise + (1 - t) * actions
target_v = noise - actions
```

此约定下 t=1 是噪声，t=0 是干净动作。因此：

```python
# AI 示意，实际采样边界、mask 和数值实现需与现有约定核对。
x = noise
t = 1.0
dt = -1.0 / num_steps
for _ in range(num_steps):
    v = predict_velocity(x, t, low_level_context)
    x = x + dt * v
    t += dt
```

`[AI 建议 A09]` subtask 接入时保留现有动作归一化、维度裁剪、action mask、adaptive RMSNorm、时间 MLP 和 flow 方向。基础 checkpoint 上不重新随机初始化已有 action expert；F07 指的是论文原始后训练阶段，不是本次增量微调要求。

`[本库代码 C11]` 当前启用 RTC 训练。`[AI 建议 A09]` 首轮 subtask 对比需固定 RTC 设置，或单独记录关闭后的基线，不把其变化混入 subtask 增益。

## 10. 自回归生成的完整循环

来源：自回归需求 F02；以下解码器和参数选择均为 `[AI 建议 A10]`。

第一版用 greedy，作为确定性基线；不是论文给出的采样参数。

1. 构造不含答案的高层条件。
2. 编码图像、状态、总体任务，使用双向条件 prefix mask。
3. 获取最后一个有效条件位置 hidden state，投影得到下一 token logits。
4. argmax 产生第一个 token；EOS 则结束。
5. 非 EOS token 输入主干，因果读取先前条件与输出，得到下一个 token。
6. 循环直到 EOS 或 max_new_tokens。
7. 去除控制 token，按固定规则解析子任务。
8. 对空输出、格式错误、无 EOS 截断分别返回明确状态，不能当正常结果静默执行。
9. 用结果重新构造低层上下文。

必须保存的配置：max_new_tokens、特殊 token ID、greedy/temperature/top-p、解析规则、异常结果策略。max_new_tokens 的具体数值待根据实际目标 token 长度分布确定；这里没有虚构论文默认值。

Batch 与缓存细节 `[AI 建议 A10]`：

- 每个样本独立停止；结束后不追加有效文本。
- 右 padding 时按有效长度取末位，不能用统一 `hidden[:, -1]`。
- RoPE position_ids 按有效序列位置；cache 的物理槽位与逻辑位置分别核对。
- 首版可无缓存重算，但模型须 eval、关闭 dropout；后续缓存版逐 token 对齐。
- 同时测试单样本、不同条件长度 batch、提前 EOS、长度上限。
- 不允许对一次 prefix forward 的所有位置 argmax 后直接 decode，冒充递归生成。
- tokenizer 的 decode 可从底层 tokenizer 调用；当前包装器没有显式定义 decode 接口，新增封装时要验证行为。

## 11. 高层转低层与缓存隔离

来源：以下调用和缓存方案为 `[AI 建议 A11]`；条件关系 F01，视角差异 F09。

```python
subtask = generate_subtask(high_level_observation, overall_task)
low_context = build_low_level_context(
    images=low_level_images,
    state=current_state,
    task=subtask,
)
actions = sample_actions(low_context)
```

第一版不复用高层融合语言 KV。`[AI 推导 D07]` 高层图像、状态等 prefix hidden states 已可能经双向 attention 混入总体任务；删除显式总体任务 token 的缓存槽位，不等于移除总体任务信息。

| 缓存对象 | AI 建议 |
|---|---|
| 同一帧、同样预处理的独立视觉编码输出 | 可研究复用，注意训练时梯度与图像增强一致性 |
| 同一个低层动作块中不变的 prefix KV | 可用于多个去噪步 |
| 高层含总体任务的融合 KV | 第一版不复用给低层 |
| 不同时间或不同子任务的低层 KV | 失效并重建 |
| 随去噪步累积追加动作的可变 cache | 不允许作为不变 prefix 使用 |

每个动作去噪步的 noisy action 均不同，不应把它们永久累计为历史 token。若使用会原地修改的 cache 对象，必须控制截断、复制或独立缓存，验证长度不随去噪步异常增长。

`[未披露 U02]` 论文没有给出完整高低层 KV 复用协议，也没有披露其精确工程缓存生命周期。

## 12. 高低层调度与动作队列

来源：高层频率较低来自 F09；本节具体协议全部为 `[AI 建议 A12]`。

```text
开始：高层生成子任务，构造低层条件。
每个动作块边界：
    获取最新观察。
    满足配置的刷新条件时，生成新子任务。
    在块边界提交子任务并使旧低层缓存失效。
    生成动作块，执行配置长度的动作前缀。
```

必须配置并记录：

- 高层刷新间隔及计时单位。
- 动作块长度与实际执行长度，二者分开。
- 高层使用的观察时间戳、生成开始/结束时间。
- 子任务提交时刻与版本号。
- 切换时旧动作队列清空、截断或执行完的规则。
- 空结果、生成失败时保持当前子任务还是暂停提交的策略。
- 整任务完成和子任务文本 EOS 的不同语义。
- 异步时旧版本生成结果能否被提交，避免迟到结果覆盖新条件。

`[未披露 U03]` 论文没有给出精确刷新周期、完整完成判定、终止 token 语义和队列切换协议。固定周期、事件触发或完成检测均是待验证选择，不能标成作者默认实现。

## 13. 文件级修改计划

来源：全部为 `[AI 建议 A13]`，尚未执行。

| 位置 | 增量 |
|---|---|
| 新增 `fluxvla/models/vlas/pi05_subtask.py` | `forward_high_level`、`forward_low_level`、总 loss、`generate_subtask` |
| `pi0_flowmatching.py` | 可选暴露语言 hidden states，保持原动作调用兼容 |
| 新增或扩展 prompt transform | 分离条件/目标、构造 label、mask 与低层子任务 prompt |
| `prompters.py` | 明确高低层格式；保留低层现有格式合同 |
| collator 配置 | 新字段及可选监督 mask |
| 新增 subtask 配置 | 长度、权重、冻结、模板版本、生成参数 |
| checkpoint 加载保存 | 词表投影、绑权、模板及 resume 兼容 |
| 新增推理入口 | 两阶段调用、子任务状态、低层采样及队列调度 |

优先顺序：张量合同 → mask → hidden states/输出投影 → CE → 高层无缓存生成 → 真实子任务驱动低层 → 两阶段串接 → 调度 → 缓存优化。

## 14. 验证标准

来源：全部为 `[AI 建议 A14]`，不是已通过的测试结果。

| 检查 | 必须观察到的行为 |
|---|---|
| 未来隔离 | 只改较晚目标 token，较早位置 logits 不变 |
| label shift | 条件末位预测首 token，前一目标位置预测 EOS |
| padding | 修改无效 ID 不影响有效输出，不产生 NaN |
| 微型拟合 | 能生成目标子任务并在 EOS 停止 |
| 缓存一致性 | 有/无 KV cache 的逐 token 结果一致，容差内 logits 对齐 |
| 总体任务隔离 | 固定观察、子任务和噪声，只改总体任务，低层结果不变 |
| 子任务进入计算 | 改子任务确实改变低层 token 与上下文；行为效果需另外评估 |
| 梯度 | CE 到达预期主干参数，动作 loss 到达预期动作/共享参数 |
| loss 归约 | 不同目标长度与 DDP 切分下符合选定全局归约合同 |
| 保存重载 | 同输入、同噪声、eval 条件下结果一致 |
| 队列切换 | 无旧条件动作意外跨版本提交 |
| 闭环分解 | 分别报告高层质量、真实子任务驱动动作、预测子任务驱动动作 |

单靠“换子任务后动作变化”不能证明语义正确；高层文本合理也不能证明低层执行成功。

## 15. 完整方案与 implicit HL 的证据

来源：定义 `[官方论文 F11]`；下表是 `[AI 读图 E01]`，非精确数值表。来源为图 13、图 17。

| 指标/任务 | 完整方案约值 | implicit HL 约值 | AI 计算差值 |
|---|---:|---:|---:|
| 平均任务进度 | 78.5% | 71% | +7.5 个百分点 |
| 物品放入抽屉 | 85% | 79% | +6 个百分点 |
| 餐具放入水槽 | 91% | 77% | +14 个百分点 |
| 衣物放入篮子 | 80% | 83% | -3 个百分点 |
| 整理床铺 | 58% | 46% | +12 个百分点 |

`[官方图 13]` 完整与 implicit HL 对应 p=0.144。`[AI 推导 D08]` 该对比未达到常用 0.05 门槛，不能据此承诺显式生成具有统计确认的固定收益。指标是平均任务进度，不是整任务成功率；逐任务差值不能用于推断逐任务显著性。

`[AI 建议 A15]` 评估至少区分不含高层监督、含监督但不显式生成、含监督且显式生成三组。不能默认本库现有动作-only 微调就是论文 implicit HL；需确认基础权重来源及训练监督后再命名。

## 16. 尚未闭合的证据与更正

| ID | 未闭合问题 | 当前可执行处理 | 来源属性 |
|---|---|---|---|
| U01 | subtask 目标的作者完整 mask 未单独画出 | 使用第 5 节因果方案并验证 | AI 设计 |
| U02 | 高低层缓存协议未完整披露 | 两次独立前向，先不跨层复用 KV | AI 设计 |
| U03 | 精确刷新、完成和终止协议未完整披露 | 配置化、记录时间戳与版本 | AI 设计 |
| U04 | 作者完整提示与目标序列格式未知 | 保存本文模板版本 | AI 设计 |
| U05 | 解码温度、长度、采样设置未知 | greedy 正确性基线，长度依数据确定 | AI 设计 |
| U06 | 公开权重的语言输出能力、head 绑权未验证 | 检查 checkpoint 并做生成验证 | 待检查 |
| U07 | 标签边界、失败恢复、跨边界动作块规范未明确 | 后续单独制定数据合同 | 待设计 |
| U08 | 本文增量的最佳 loss 权重与冻结策略未知 | 记录配置并做受控比较 | 待实验 |

更正对话中可能造成误读的表述：

- `[AI 更正]` “论文未披露”在本文只表示本次检查范围内没有明确规定，不作官方所有资料范围的断言。
- `[AI 更正]` 两次前向、模板、字段名、绑权投影、greedy、DDP 归约和块边界调度都是 AI 提案，不是作者原实现。
- `[AI 更正]` 本库确有基础能力和配置，但尚未运行 subtask 路径；本文没有实施模型修改。
- `[AI 更正]` 本文不将论文 attention 规则解释为 stop-gradient，也不把其他论文/社区方案的 knowledge insulation 归给本论文。
- `[AI 更正]` 附录 v1 的头数文字与本库配置存在差异；本次增量不按该文字重建主干，保持已加载 checkpoint 的配置。具体差异需单独核验，不能为接入 subtask 擅自改结构。

## 17. 实施状态

- 已完成：原论文相关段落与图的阅读；本库关键训练路径静态检查；复现增量和来源归类；官方及社区开源调研，见第 18–27 节。
- 未完成：基础权重输出头核验；新增训练/生成实现；任何所列测试；GPU/真机实验。
- 本文只修改调研文档，不改变模型、配置或训练任务。

## 18. 开源复现调研范围与来源标记

调研日期：2026-10-17。以下内容来自官方及社区公开源码、issue、模型卡和数据卡的静态阅读，未运行这些仓库的训练、模型生成或机器人评估。

部分源码仅取得 `main` 页面，未将全部文件统一锁定到同一个 commit。跨文件问题是当前所读页面组合下的风险，不能表述为已复现的运行错误。网页可访问不等于仓库能完整安装，也不等于权重可用。未找到某项说明只表示此次检查范围内的证据缺口。

在第 1 节标记基础上增加：

| 标记 | 来源与含义 |
|---|---|
| `[官方说明 Oxx]` | Physical Intelligence 的 README 或明确身份的贡献者回复 |
| `[官方代码 Oxx]` | AI 静态阅读官方 OpenPI 源码得到的事实，不是模型运行结果 |
| `[社区说明 Rxx]` | 社区作者的 README、issue、模型卡或数据卡声明，未由官方背书 |
| `[社区代码 Rxx]` | AI 静态阅读第三方实现得到的事实 |
| `[AI 判断 Jxx]` | AI 对可复用性、差异、风险的推导，未经运行验证 |
| `[AI 建议 Nxx]` | 面向本库的技术取舍与修改建议 |

后文一个段落或代码块前的标记覆盖该段内容。代码片段均为缩写或说明性改写，不代表已直接移植到本库。

## 19. 仓库与权重索引

| ID | 来源 | 值得提取的内容 | 证据边界 |
|---|---|---|---|
| O01 | [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | 原始动作模型、embedding、attention、归一化、加载和 FAST 自回归范例 | 官方公开 π0.5 入口主要提供 flow 动作，不等于论文完整高层系统 |
| R01 | [LisavilaLee/openpi_with_subtask](https://github.com/LisavilaLee/openpi_with_subtask) | 最小 CE 增量、tokenizer、首 token 修复和生成入口 | identity 示例；prompt-only 缓存不适合直接承担长任务推进 |
| R02 | [Ke-Wang1017/openpi_subtask](https://github.com/Ke-Wang1017/openpi_subtask) | 独立 π0.5、三损失、区域 mask、梯度隔离、数据及服务 | 完整度较高但仍需核验 mask、KV 和服务调用链 |
| R03 | [BrunoFANG1/openpi_subtask_generation](https://github.com/BrunoFANG1/openpi_subtask_generation) | 早期模型、测试及固定容量 KV | 作者明确说训练代码过时且不正确 |
| R04 | [Hebbian-Robotics/openpi-flash](https://github.com/Hebbian-Robotics/openpi-flash) | JIT 解码、词表约束、服务包装 | 工程实现，不等于论文完整配方；存在位置与输入合同风险 |
| R05 | [hebbianrobotics/pi05_subtask](https://huggingface.co/hebbianrobotics/pi05_subtask) | 小规模 CE 微调与公开权重线索 | identity 训练不能证明阶段分解 |
| R06 | [dgrachev/pi05_subtask_base](https://huggingface.co/dgrachev/pi05_subtask_base) | LeRobot/PyTorch 预处理器合同线索 | 作者说基础权重不变，仅改元数据；关联插件源码未核实 |
| R07 | [KeWangRobotics/libero_10_subtasks](https://huggingface.co/datasets/KeWangRobotics/libero_10_subtasks) | 带 subtask 与 overall_task 的公开数据结构 | 未逐 episode 验证标注来源、质量、阶段变化或划分 |

`[AI 判断 J01]` 最适合组合借鉴的是官方基础模块、R01 的最小增量与 R02 的完整训练/推理细节。不是把三个仓库直接拼接；它们的 mask、低层条件和缓存实现有实质差异。

版本记录：R01 的提交历史页面可见 [99c738a322300166499efa0b614ffa95dbb9b21e](https://github.com/LisavilaLee/openpi_with_subtask/commit/99c738a322300166499efa0b614ffa95dbb9b21e)，但本轮未成功取得所有文件在该 commit 的统一快照。因此本文仍以所读公开页面为证据，不宣称所有检查已固定到该版本。其他仓库也待完整固定版本。

## 20. 官方库可以直接帮助复现的细节

### 20.1 官方发布范围

`[官方说明 O01]` 官方 README 说明仓库中的 π0.5 训练与推理支持 flow-matching head；贡献者 `kpertsch` 在 #647 确认当时只支持 action decoding。

来源：[官方 README](https://github.com/Physical-Intelligence/openpi/blob/main/README.md)、[#647 官方贡献者回复](https://github.com/Physical-Intelligence/openpi/issues/647)。回复时间为 2025-09-11，不能将历史回复独立当作未来所有版本的保证；本轮同时检查了公开主分支相关文件。

`[AI 判断 J02]` 这说明有 π0.5 动作训练并不代表已有显式 subtask 生成。#647 中其他用户关于如何添加高层的建议不属于官方实现说明。

### 20.2 输出投影已存在

`[官方代码 O02]` 官方 JAX `gemma.py` 的 Embedder 使用相同输入表完成 encode/decode：

```python
# 官方实现关系的缩写
embedding = embedding_table[token_ids] * sqrt(hidden_size)
logits = hidden_states @ embedding_table.T
```

来源：[gemma.py，Embedder.encode/decode](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/gemma.py)。主干模块没有同样方便的 subtask 对外生成入口，社区增加了调用包装。

`[AI 判断 J03]` 第 6 节绑权方案现在有官方 JAX 源码依据，不再只是一般性的 AI 候选接法；但本库转换后的 checkpoint、名称映射、最终 norm 和词表对应仍未验证。所需增量优先是暴露 hidden states 与调用输出投影，不应先新增大型随机语言 head。

### 20.3 FAST 自回归路径可借鉴但不可原样复制

`[官方代码 O03]` 官方 `pi0_fast.py` 包含 token_ar_mask、token_loss_mask、next-token shift、文本目标投影、prefix prefill、AR 解码和 EOS；使用固定容量缓存的相关接口。

来源：[pi0_fast.py](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0_fast.py)。

`[AI 判断 J04]` FAST 的语言模块与 π0.5 模块接口不同，外层解码循环不能不改底层 KV 就复制过去。复制时必须同时核对缓存结构、容量、写入位置、attention 的 Key 长度及 RoPE。

### 20.4 官方代码与原论文的时间范围

`[官方说明 O04]` 当前官方 README 将公开 π0.5 模型与 knowledge insulation 关联。来源同 O01。

`[AI 判断 J05]` 此说明与第 1 节采用的 2025-04 原论文不是同一个时间点的实现证据。不能把公开权重的后续训练说明直接补成原论文未披露的梯度代码，也不能把 R02 的隔离实现归给原论文。

## 21. LisavilaLee 的最小 subtask 增量

本节代码来源为 R01，不是官方方案。

### 21.1 模型增量

`[社区代码 R01]` 新增 `Pi05Subtask(Pi0)`，保持原主干/expert 结构，通过 compute_loss 增加 CE，并提供 generate_subtask 与 build_full_observation。

来源：[pi0.py](https://github.com/LisavilaLee/openpi_with_subtask/blob/main/src/openpi/models/pi0.py)。

### 21.2 Tokenizer 合同

`[社区代码 R01]` tokenizer 返回 tokens、valid mask、AR mask、loss mask。训练格式为：

```text
[BOS] Task: 总体任务. Subtask: 子任务 [EOS] [PAD...]
条件：AR=0，loss=False
子任务与 EOS：AR=1，loss=True
```

训练 pair 与推理 prefix 有独立方法；文本做小写、空白/下划线清理和末尾标点处理。subtask pair 方法未传入状态字符串，并允许超长截断。

来源：[tokenizer.py，tokenize_high_low_prompt / tokenize_high_level_prefix](https://github.com/LisavilaLee/openpi_with_subtask/blob/main/src/openpi/models/tokenizer.py)。

`[AI 判断 J06]` 可借鉴分离合同，但 Piper 不能照搬无状态路径；截断时必须检查目标和 EOS。不能因为通用 tokenizer 的 tokenize 方法支持 state，就认为 subtask 专用方法也支持。

### 21.3 联合前向与 CE

`[社区代码 R01]` compute_loss 使用图像 AR=0、tokenizer 提供的文本 AR 和双向动作块，联合前向得到 prefix_out 与 suffix_out。CE 对齐关系为：

```python
text_hidden = prefix_out[:, num_image_tokens : num_image_tokens + num_text - 1]
targets = tokenized_prompt[:, 1:]
loss_mask = token_loss_mask[:, 1:]
# 每个样本按有效目标数归约
return subtask_loss[:, None] + flow_loss
```

来源：[Pi05Subtask.compute_loss](https://github.com/LisavilaLee/openpi_with_subtask/blob/main/src/openpi/models/pi0.py#L296)。

`[AI 判断 J07]` 这里 CE 是逐样本 token 均值，与全 batch token 均值不完全相同；没有显式照搬论文 α=10；动作读取总体任务与子任务共同所在 prefix。这与第 2 节建议的两次独立输入方案不同，不能混称一种实现。

### 21.4 首 token 物理索引修复

`[社区代码 R01]` generate_subtask 查找最后有效位置的物理索引，而不是以有效 token 数减一当索引；后续 RoPE 从有效 token 数开始。

来源：[generate_subtask](https://github.com/LisavilaLee/openpi_with_subtask/blob/main/src/openpi/models/pi0.py#L362)。

`[AI 解释 J08]`：

```text
valid mask = [1, 1, 0, 0, 1, 1]
最后有效位置物理索引 = 5 → 取首 token 的 hidden state
有效 token 总数      = 4 → 下一 token 的逻辑 RoPE 位置
```

缺失相机和 padding hole 都可能使这两个量不同。该修复可作为本库解码实现的优先参照。

### 21.5 Batch EOS 与回填

`[社区代码 R01]` 生成循环判断当前一步是否所有样本都输出 EOS，没有单独累计每个样本的 finished。build_full_observation 按文本 prefix 有效长度把生成 ID 填入 padding。

来源：[同一文件 generate_subtask / build_full_observation](https://github.com/LisavilaLee/openpi_with_subtask/blob/main/src/openpi/models/pi0.py#L406)。

`[AI 判断 J09]` B>1 需逐样本 finished、首 EOS 位置及输出有效 mask；回填要核对剩余容量、各样本生成长度及截断。不能因 B=1 可用就宣布一般 batch 可用。

### 21.6 Prompt-only 子任务缓存与计时

`[社区代码 R01]` Policy.infer 按原始 prompt 缓存 subtask；相同 prompt 再请求时复用文本。infer_ms 计时在高层生成之后开始。

来源：[policy.py](https://github.com/LisavilaLee/openpi_with_subtask/blob/main/src/openpi/policies/policy.py#L54)。

`[AI 判断 J10]` 固定总体指令、场景持续变化的长任务会复用旧决策；不能直接用于阶段推进。该 infer_ms 也不是两阶段端到端耗时。

### 21.7 Identity 示例的边界

`[社区说明 R01]` 作者在 #701 说明示例 high_prompt=low_prompt=prompt，尚未使用人工子任务标注。

来源：[#701，LisavilaLee 2026-03-19 回复](https://github.com/Physical-Intelligence/openpi/issues/701)。

`[AI 判断 J11]` identity 示例适合链路验证，不能证明视觉驱动的任务分解。关于 state 的 issue 解释还须以实际 embed_suffix 和 transform 路径核对，不能把评论中的类比当作确定代码事实。

## 22. Ke-Wang1017 的训练与推理细节

本节来源 R02；模型入口是独立的 `pi05.py` 与 `gemma_05.py`，不是普通 pi0.py。

### 22.1 三种损失与区域 mask

`[社区代码 R02]` 配置独立的 subtask_loss_weight、fast_token_loss_weight、flow_matching_loss_weight；通过 subtask_region_mask 与 action_region_mask 分开归约两类 CE。flow 用 real_action_dim 控制真实动作维度。

来源：[pi05.py，compute_loss](https://github.com/Ke-Wang1017/openpi_subtask/blob/main/src/openpi/models/pi05.py#L184)、[pi05_config.py](https://github.com/Ke-Wang1017/openpi_subtask/blob/main/src/openpi/models/pi05_config.py)。

`[AI 判断 J12]` 如果区域 mask 缺失后 fallback 到总 loss mask，两类损失可能作用同一目标区域；实现时应显式检查，而不是静默 fallback。Piper 要明确真实 14 维动作与内部 32 维 padding，不能沿用 LIBERO 假设。

### 22.2 训练格式与预测段

`[社区代码 R02]` 三段分别为总体任务+状态、子任务、可选 FAST 动作。无 FAST 时子任务预测段包括 `;\nAction: ` 与 EOS；有 FAST 时结尾不同。

来源：[tokenizer.py，tokenize_high_low_prompt](https://github.com/Ke-Wang1017/openpi_subtask/blob/main/src/openpi/models/tokenizer.py#L120)。

`[AI 判断 J13]` 生成长度要包含格式 token；语义评估须去除格式部分；进入动作时防止重复添加 Action；有/无 FAST 的目标结尾不能不加区分地混用。

### 22.3 全部文本因果与其他方案的区别

`[社区代码 R02]` 图像 AR=False，全部文本（包括总体任务、状态）AR=True；不同于 R01 的条件文本双向、目标文本因果。

来源：[pi05.py，embed_prefix](https://github.com/Ke-Wang1017/openpi_subtask/blob/main/src/openpi/models/pi05.py#L123)、[tokenizer.py](https://github.com/Ke-Wang1017/openpi_subtask/blob/main/src/openpi/models/tokenizer.py#L194)。

`[AI 判断 J14]` 这是模型信息流差异，不是代码风格。结果比较须记录实际 mask，不把两个仓库视作相同结构的独立验证。

### 22.4 从 flow 上下文移除 FAST 答案

`[社区代码 R02]` hybrid 训练中把 action_region_mask 对应 ID 置 padding、有效 mask 置 False，再独立构建 flow prefix KV。

来源：[pi05.py，flow 分支](https://github.com/Ke-Wang1017/openpi_subtask/blob/main/src/openpi/models/pi05.py#L247)。

`[AI 判断 J15]` 仅将 FAST token 的 CE loss mask 设 False 不能防止动作答案泄漏；必须阻止 flow attention 读取它们。这是后续补 FAST 时应优先迁移的修复。

### 22.5 梯度路径与冻结不是一回事

`[社区代码 R02]` stop_gradient_flow_to_prefix 开关对 flow prefix KV 使用 stop_gradient；action-expert 配置同时有参数 freeze_filter。

来源：[模型 KV 梯度处理](https://github.com/Ke-Wang1017/openpi_subtask/blob/main/src/openpi/models/pi05.py#L285)、[训练配置](https://github.com/Ke-Wang1017/openpi_subtask/blob/main/src/openpi/training/config.py#L649)。

`[AI 判断 J16]` 应分别验证梯度路径与优化器参数集合。冻结规则主要按 llm 路径匹配，不能只按 README 的“冻结 VLM”推断视觉塔也被冻结。打印参数名、trainable 状态和梯度覆盖后再下结论。

### 22.6 固定容量 KV 的底层机制

`[社区代码 R02]` gemma_05 的缓存包括 idx、K、V；prefill 按 attention Key 容量预留空间，文本单步用 dynamic_update_slice 写入。动作读取缓存与当前动作 KV，不把每次动作去噪都当历史文本累计。

来源：[gemma_05.py](https://github.com/Ke-Wang1017/openpi_subtask/blob/main/src/openpi/models/gemma_05.py)。

`[AI 判断 J17]` #701 的 mask/KV 长度问题说明，固定容量 attention mask 不能配普通未预留容量的 KV。移植必须成套核对 cache 结构、容量、写入索引、mask Key 长度、RoPE。该实现以单 token 长度等条件区分路径，也需验证动作 horizon 等边界。

### 22.7 数据字段与加载映射

`[社区数据 R07]` 数据卡列出 500 episodes、138090 frames、10 fps，包含 subtask、overall_task、state、actions 和图像。

来源：[数据卡](https://huggingface.co/datasets/KeWangRobotics/libero_10_subtasks)。

`[社区代码 R02]` repack 示例使用 task 与 subtask 字段。来源：[LeRobotLiberoSubtaskDataConfig](https://github.com/Ke-Wang1017/openpi_subtask/blob/main/src/openpi/training/config.py#L445)。

`[AI 判断 J18]` 数据卡 overall_task 与配置 task 的映射要沿实际 loader 核实；不能仅凭名称差异断言运行失败，也不能默认兼容。仍需逐 episode 检查标注变化、边界、来源和划分。

### 22.8 同步服务二次生成风险

`[社区代码 R02]` 同步 server 先生成子任务，再带该文本请求动作；模型 sample_actions 内又调用 sample_low_level_task。

来源：[sync server](https://github.com/Ke-Wang1017/openpi_subtask/blob/main/scripts/async_pi05/sync_pi05_websocket_server.py#L296)、[sample_actions](https://github.com/Ke-Wang1017/openpi_subtask/blob/main/src/openpi/models/pi05.py#L401)、[同步引擎](https://github.com/Ke-Wang1017/openpi_subtask/blob/main/scripts/async_pi05/sync_pi05_inference.py)。

`[AI 判断 J19，待统一版本运行]` 该组合可能生成两次，在含已有子任务的序列后继续生成，使返回子任务与动作实际条件不一致。建议明确拆成 generate_subtask 与 sample_actions_given_subtask。尚未运行，不写成已复现故障。

### 22.9 超参数是社区配置，不是官方推荐

`[社区代码 R02]` hybrid 示例使用子任务/FAST/flow 权重 0.15/0.15/1.0；配置与 README 提供不同训练模式及不同 horizon、长度。来源：[训练配置](https://github.com/Ke-Wang1017/openpi_subtask/blob/main/src/openpi/training/config.py)、[README_subtask](https://github.com/Ke-Wang1017/openpi_subtask/blob/main/README_subtask.md)。

`[AI 判断 J20]` 这些权重只能视为社区实验起点。内部路径、norm stats 位置、训练步数和 horizon 需统一检查，不能把 README 示例全部当可直接执行的本机命令。

## 23. Bruno 的模型与测试参考

`[社区说明 R03]` 作者在 #701 明确指出训练代码未发布，仓库内训练代码过时且不正确；可参考模型和测试。作者称基础权重直接生成不完美，小规模微调能改善。

来源：[#701 作者 2025-10-05 回复](https://github.com/Physical-Intelligence/openpi/issues/701)。

`[社区代码 R03]` 可研究 CE 与 flow 的连接、自定义 Gemma、固定容量缓存、deembed 和 next-token shift。

来源：[pi05.py](https://github.com/BrunoFANG1/openpi_subtask_generation/blob/main/src/openpi/models/pi05.py)、[测试入口](https://github.com/BrunoFANG1/openpi_subtask_generation/blob/main/tmp_test/test_pi05_subtask_generation.py)。

`[AI 判断 J21]` 适合解释后续社区实现的底层来源，不适合作为现成训练基线。issue 关闭、作者说语言变正常或成功率提高，都不能替代受控评估。

## 24. Hebbian 的推理工程及位置风险

### 24.1 可提取的工程内容

`[社区代码 R04]` openpi-flash 提供 prefill+AR 的 JIT 展开、绑权词表投影、EOS 状态、可打印 ASCII 约束及 planner/action 包装。

来源：[subtask_generator.py](https://github.com/Hebbian-Robotics/openpi-flash/blob/main/src/hosting/subtask_generator.py)、[架构说明](https://github.com/Hebbian-Robotics/openpi-flash/blob/main/ARCHITECTURE.md)。

`[AI 判断 J22]` 其注释报告的延迟不是本机测量；JAX unrolled decode 的编译与缓存策略不能直接当 PyTorch 最佳实现。

### 24.2 物理索引与逻辑 RoPE 仍混用

`[社区代码 R04]` prefill RoPE 使用 cumsum(valid)-1；last_position 是最后有效位置的物理索引，下一生成位置却使用 last_position+1。

来源：[prefix 与 AR 实现](https://github.com/Hebbian-Robotics/openpi-flash/blob/main/src/hosting/subtask_generator.py#L205)。

`[AI 判断 J23]` padding/hole 存在时不一致。必须分开 last_valid_index 和 next_rope_position；后者从有效 token 数开始。该判断为静态风险，需固定版本后以有/无 padding、有/无 cache 对齐测试复核。

### 24.3 ASCII 不是标准解码要求

`[社区代码 R04]` 非可打印 ASCII 的词表项被屏蔽，保留 PAD/EOS；SentencePiece 的词边界标记替换为空格后检查。

来源：[词表 mask](https://github.com/Hebbian-Robotics/openpi-flash/blob/main/src/hosting/subtask_generator.py#L54)。

`[AI 判断 J24]` 该手段可约束英文形式，不证明语义选阶段正确；会排除中文等文本。目标若含换行，允许集合也要核对。它是社区补救措施，不能作为论文标准默认值；保留 PAD 为可生成项也应单独审计。

### 24.4 状态、相机与低层条件不同

`[社区代码 R04]` planner 构造零状态；缺失相机填零且 mask=True。动作包装器默认将总体任务与子任务共同拼入 prompt。

来源：[Observation 构造](https://github.com/Hebbian-Robotics/openpi-flash/blob/main/src/hosting/subtask_generator.py#L164)、[动作包装器](https://github.com/Hebbian-Robotics/openpi-flash/blob/main/src/hosting/subtask_policy.py#L73)。

`[AI 判断 J25]` 这些均不是 Piper 当前数据合同；照搬会改变输入分布及第 2 节低层条件。工程接口可参考，不能直接宣称论文完整方案复现。

## 25. 公开权重与 LeRobot 线索的证据边界

### 25.1 Hebbian 小规模权重

`[社区说明 R05]` 模型卡报告 100 步、30 个 LIBERO episodes 的训练，使用 high_prompt=low_prompt=task_prompt。

来源：[模型卡](https://huggingface.co/hebbianrobotics/pi05_subtask)。

`[AI 判断 J26]` 这是微调与生成链路的线索，不能证明场景驱动的阶段切换。训练 loss 下降不能当作长任务闭环收益。未下载权重或复核模型卡报告。

### 25.2 dgrachev 基础权重元数据

`[社区说明 R06]` 作者明确说权重与 lerobot/pi05_base 相同，只改变配置与预处理元数据；context 与模型预测段分离。使用 subtask 字段时优先读取，否则退回 task。

来源：[模型卡](https://huggingface.co/dgrachev/pi05_subtask_base)。

`[AI 判断 J27]` 这是处理器兼容入口，不是额外训练好的 subtask checkpoint。值得借鉴的是保存 processor 注册与条件/预测段合同，防止加载基础 checkpoint 后静默使用普通 π0.5 tokenizer。关联插件源码尚未核实，不能列作已审计的 PyTorch 训练实现。

## 26. 面向本库的增量优先级

来源：全部为 `[AI 建议 N01]`，结合本库 C01–C11 与上述来源，不是官方优先级。

| 优先级 | 需要落实的增量 | 主要参照 |
|---|---|---|
| P0 | 暴露语言 hidden states，复用经核验的词表投影 | O02、R01 |
| P0 | 条件、目标、有效、因果、loss 区域分离 | R01 tokenizer |
| P0 | 首 token 物理索引与 RoPE 逻辑位置分离 | R01 修复；J23 风险 |
| P0 | EOS、逐样本 finished、容量与超长处理 | 补齐社区边界 |
| P0 | 选定低层只读 s 或读取 g+s，不能混称同一方案 | F01 与 R01/R02/R04 差异 |
| P0 | 高层生成与给定子任务的动作采样分离 | J19 |
| P1 | 独立 loss 权重、有效动作维度及梯度合同 | R02 |
| P1 | 基于最新观察刷新子任务，记录队列版本 | 不使用 prompt-only 缓存 |
| P1 | 有/无缓存逐 token 与 padding 回归验证 | R01/R02/R04 |
| P2 | FAST 区域彻底隔离出 flow 上下文 | R02 |
| P2 | 固定容量或编译解码优化 | R03/R02/R04 |
| P2 | 词表限制作为独立实验开关 | R04，禁止默认为官方行为 |

`[AI 建议 N02]` 第 14 节验证应新增或细化：masked 相机 hole 的末位索引测试、逻辑 position 测试、batch 异步 EOS、子任务只生成一次、返回文本与动作条件一致、同一总体任务随场景变化的刷新、FAST 目标泄漏、完整端到端计时。

## 27. 本轮结论与后续证据待办

`[AI 判断 J28]` 已有开源代码足以具体参考 CE、生成及缓存连接；真正需要本库严格定义的是信息条件、mask、位置、缓存生命周期和高层刷新。不同社区实现不能未经选择就混合，也不能称为相同架构的复现结果。

| 待办 | 为什么仍未闭合 |
|---|---|
| 统一各仓库 commit 与文件快照 | 当前部分页面来自不同抓取时间的 main |
| 核验本库基础权重输出表与语言能力 | 只读代码不能确认 checkpoint 的实际能力 |
| 运行末位索引、RoPE、EOS 和缓存测试 | 本轮仅静态检查 |
| 核验 R02 服务是否重复生成 | 需固定文件版本并跟踪真实调用 |
| 核验 R07 数据字段映射与标注质量 | 已读结构，不等于逐条数据审计 |
| 查实 R06 关联插件源码 | 当前仅有模型卡线索 |
| 复核闭环性能及官方 PR 合入情况 | 未取得可证明完整 subtask 已合入的官方 PR；不作全仓永久缺失断言 |

本轮只将用户确认的窗口版调研纳入本文，没有修改模型、训练配置或启动任何训练任务。
