# Subtask / FAST 损失门控

默认不配置 `text_loss`，保持原有统一 token CE 及旧 checkpoint 身份。新增门控仅影响明确启用它们的新实验，不会修改正在运行的训练。

2026-10-23 更新：示例配置已启用 `text_loss.normalize_weights: true` 并使用显式 `{slot}` 模板。旧配置省略此字段时仍保留原行为，避免偷偷改变历史实验。以下基本公式描述未归一化形式；启用后的公式见下一节。

## 自动归一化与显式槽位

Separate 模式先调度原始权重，再归一化：`a=w_subtask/(w_subtask+w_fast)`、`b=1-a`。Subtask 内部固定权重也归一化：`k=w_key/(w_key+w_template)`、`t=1-k`。

`L_text = a × (k × L_key + t × L_template) + b × L_FAST`。

不开启 Subtask 拆分时，括号替换为原始 Subtask 平均 CE。1:1 对应 0.5:0.5；FAST:Subtask=10:1 对应 10/11:1/11；key:template=1:0.1 对应 10/11:1/11。动态模式对原始比例线性插值后再归一化，因此有效系数不一定随步数线性变化。

Unified 模式本身按 token 归一化；开启拆分与自动归一化时改为加权 token 均值：`(w_key*S_key + w_template*S_template + S_FAST)/(w_key*N_key + w_template*N_template + N_FAST)`。FAST 系数为 1，任务间比例仍由 token 数决定；不能把它与 Separate 的层次归一化混为一谈。

Flow 保持外部原权重，不与 CE/MSE 混合做无依据的单位归一化。系数总和稳定不意味着梯度范数、梯度方向或实际 Flow/CE 梯度比例不变。空区域贡献为零，分配给它的预算不转移给其他区域。

当前示例模板显式标注 action、object、source/destination、arm、location/orientation，以及并行分隔符；复合句模板先于单句。通用匹配算法不含这些业务短语，业务定义仅存在 JSON 中。新句式未匹配时仍全 key，并非保证自动理解新句子。

## 三个开关

`text_loss.mode`：`unified_token` 或 `separate`。后者表示同一模型、同一批次中分别归一化后求加权和，不是两个模型或轮流更新。

`text_loss.weighting`：`fixed` 或 `linear`。仅 separate 支持动态任务权重。它是预定步数调度，不是根据验证结果自动调整，也不是 GradNorm。

`text_loss.split_subtask`：是否将 Subtask 进一步拆成 key 和 template。`key_weight`、`template_weight` 固定，默认分别为 1 和 0.1，可配置非负值。原始未加权指标始终保留。

记每组 NLL 总和为 S，有效 token 数为 N：

| 模式 | 文本优化目标 |
|---|---|
| unified，不拆 Subtask | (S_subtask + S_fast) / (N_subtask + N_fast) |
| separate，不拆 Subtask | w_subtask × S_subtask/N_subtask + w_fast × S_fast/N_fast |
| unified，拆 Subtask | (w_key × S_key + w_template × S_template + S_fast) / (N_key + N_template + N_fast) |
| separate，拆 Subtask | w_subtask × (w_key × S_key/N_key + w_template × S_template/N_template) + w_fast × S_fast/N_fast |

空区域贡献为零，不产生 NaN；诊断中无有效 token 的 CE 为 null，不伪报为零。统一模式的分母保持有效 token 数，不改成权重之和。因此 split+unified 是位置加权 CE，而 split+separate 是区域均值加权，两者含义不同。

Stage 2 在上述文本项外加既有 `joint_flow_weight × Flow MSE`。action_expert/flow_only 保持原有仅优化动作分支的行为。

## 配置示例

分开归一化、固定 1:1：

```json
"text_loss": {
  "mode": "separate",
  "weighting": "fixed",
  "subtask_weight": 1.0,
  "fast_weight": 1.0,
  "split_subtask": false
}
```

分开归一化、FAST 从 1 线性升到 10、关键词与模板为 1:0.1：

```json
"text_loss": {
  "mode": "separate",
  "weighting": "linear",
  "subtask_weight": 1.0,
  "fast_weight": 1.0,
  "end_subtask_weight": 1.0,
  "end_fast_weight": 10.0,
  "start_update": 1,
  "split_subtask": true,
  "key_weight": 1.0,
  "template_weight": 0.1
}
```

`end_update` 省略时使用当前 stage 的总 optimizer updates；指定时，达到该步后固定最终权重。第一个 stage 和第二个 stage 各自从自己的 update 计数调度；同 stage 恢复使用保存的 update，不从头重新爬升。

例如一个 100 步 stage，start=1、end=100，则第 1 步 FAST=1，第 100 步 FAST=10。实际权重按步记录在训练日志中。配置不接受拼错字段、NaN、负数以及统一模式下的任务权重调度。

**旧配置省略 normalize_weights 或设为 false 时不归一化。** `10 × FAST + 1 × Subtask` 不等于 `10/11 × FAST + 1/11 × Subtask`。新示例已启用归一化，但不能据此承诺 FAST 会更快泛化。

## 通用模板掩码

自动方式仅用训练 split 的去重文本，确定性提取共同词元子序列作为固定骨架。跨句式的变化位置保留为 key，不使用当前布料任务的词表、绝对位置或硬编码句子。只有一个不同训练文本、未知文本或无法可靠对齐 tokenizer offsets 时，保守地将相应内容视为 key。

掩码在实际生成的完整 Subtask segment 上通过真实 tokenizer 的 offset mapping 对齐，并核对 token ID 完全相同；跨越 key/template 边界的 token 归入 key。attention mask、输入文本和原 token 序列不变。它们与标签一起做自回归 shift。

自动推断的 key 是“相对训练语料发生变化的位置”，**不等于已经验证的语义关键词**。常量也可能具有语义，字段间相关性也可能形成捷径。需要更精确的监督时，可使用通用 `{slot}` 模板接口明确标记字段；不应把这种人工语义信息伪装为自动发现。

可在配置顶层加入 `"subtask_templates": ["{action} {object} to {destination}"]`，将花括号包围的字段标为 key。该例仅演示语法，不是当前数据集的完整模板。未匹配文本仍保守处理为全 key。也可提供序列化的 `subtask_mask_spec`；训练导出记录实际 spec 和 fingerprint，评估时核对训练与评估的规范一致。

## 指标与验证

保留 `subtask_ce`、`fast_ce`、`token_ce`、`flow_mse`，新增 `subtask_key_ce`、`subtask_template_ce`、`fast_code_ce` 和各自有效计数。FAST 原指标仍包含 Action 分隔符和结束标记；fast_code_ce 只包含动作码。

优化继续使用既有“各 microbatch 均值，梯度累积后 DDP rank 均值”的约定；日志/验证用所有 rank 的 NLL 和有效计数池化。因此变长序列下优化均值和日志池化 CE 不必完全相等。日志中的 raw token contribution 也不是独立加权模式的实际梯度贡献。

`optimization_weighted_subtask_term` / `optimization_weighted_fast_term` 记录实际加权目标的两项，其对应 fraction 是目标数值占比，仍不是梯度贡献占比。旧 `optimization_*loss_fraction` 保留兼容，并带有 `raw_unified_token_ce_not_weighted_objective` 口径说明。

训练 CE 低不足以证明过拟合或学会视觉判断。至少检查同一 checkpoint 的固定 train/validation 样本、key/template 分组 CE、自由生成正确率；多 checkpoint 的验证趋势才能判断泛化退化。可结合真实图像替换对照检查输入依赖，但错配图像测试本身不是完整的因果证明。

## 实验与恢复边界

新配置使用独立 `output` 和 `shared_stage1_root`，不可复用正在训练的目录。门控和模板规范参与训练记录及导出校验。改变 loss 是新实验，不是原实验的严格续训；原有中间 checkpoint 仍然只有权重，不会因新门控而获得 optimizer 精确恢复能力。

建议先对照统一 CE 与 separate 固定权重，再单独启用 key/template 拆分，最后测试权重调度。一次同时打开三个开关，无法辨别收益来自哪一项。

可运行示例在 `configs/subtask/loss_examples/`：`separate_fixed.json`、`separate_split_fixed.json`、`separate_split_linear.json`、`unified_split_fixed.json`。它们继承当前双卡服务器路径与 Flow 权重 10，但各有独立输出目录和 Stage 1 目录。它们尚未部署或启动，需在包含新代码的训练环境运行。
