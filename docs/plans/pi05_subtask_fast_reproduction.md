> 历史实验计划，保留研究过程。当前入口与已落实的条件语义见 [subtask_workflow.md](../subtask_workflow.md)。

# π0.5 subtask + FAST 复现待办（对齐 Ke-Wang1017/openpi_subtask）

参考实现：`experiments/pi05_subtask_architecture/references/kewang/`
commit：**`385d0e8330397ecfa29e9b675042b424c4d5386c`**（`Merge pull request #8 from TheWisp/fix/inference-bugs`）
本地仓库：`/home/ylhp-e-ai/DATA2TB/kyz/soma-fluxvla`（非 git，改动另存快照）
目标：**把 FAST 打开**，从"subtask CE + flow"升级为完整 KI 配方（subtask CE + FAST CE + flow MSE + flow 前缀 stop-gradient）。

状态约定：`[ ]` 未做 / `[~]` 进行中 / `[x]` 完成（完成后附证据路径）。

---

## 0. 参考实现事实基线（已逐行核对，勿凭记忆改）

| 项 | 参考实现（Ke-Wang @385d0e8） | 位置 |
|---|---|---|
| 单一 prefix 拼接 | `Task: g; State: <256bin>; Subtask: s;\nAction: <FAST>|<EOS>` | `models/tokenizer.py:133-303` |
| 文本 attention | **全部因果**（`ar_mask` 全 True，含 Task/State 段） | `models/pi05.py:161-163` |
| 三条 loss 权重 | `subtask_loss_weight` / `fast_token_loss_weight` / `flow_matching_loss_weight` | `models/pi05_config.py:38-40` |
| 文本 logits | 一次前向 + `deembed`（embedding 转置） | `models/pi05.py:232-236` |
| region mask shift | `subtask_region_mask[:,1:]`、`action_region_mask[:,1:]`，**各自按自身计数归一化** | `models/pi05.py:245-270` |
| flow 防泄漏 | 把 action 区 token 置 0、mask 置 False，**再**建 flow 专用 prefix KV | `models/pi05.py:284-311` |
| flow 隔离 | `jax.tree.map(stop_gradient, kv_cache)` | `models/pi05.py:321-322` |
| 去噪 | `x += dt*v`，`dt=-1/num_steps`，t 从 1→0，去噪期间 **cache 不更新** | `models/pi05.py:452-511` |
| FAST token 映射 | `vocab_size - 1 - 128 - token_id` | `models/tokenizer.py:307-310` |
| FAST 编解码 | DCT(axis=timesteps, ortho) → ×10 → round → chr → BPE；解码反向 idct | HF `physical-intelligence/fast` `processing_action_tokenizer.py` |
| 三档预设 | `libero_pi05_subtask_flow`(1/0/1) / `libero_pi05_subtask_fast`(10/1/0) / `libero_pi05_subtask_hybrid`(0.15/0.15/1) | `training/config.py:634/672/757` |

我们现状 vs 参考实现的差异（已知，需在本轮显式处理）：

| 项 | 我们（`pi05_subtask.py`） | 参考 | 本轮决策 |
|---|---|---|---|
| 文本 mask | 条件双向 + 目标因果 | 全因果 | 保留我们的（更贴论文），但要在文档写清差 |
| 低层是否含 `g` | 不含（`o+s`） | 含（prefix 里有 Task） | **待定**，见 T3.4 |
| FAST CE | **无** | 有 | 本轮核心 |
| flow 前缀 | 只 `o+s` | `o+s`+（Task），且 action 区被清零 | 见 T3.4 |
| 时间采样 | `sample_beta` 近似 | `jax.random.beta(1.5,1)` | 换成 `torch.distributions.Beta` |
| flow loss 归一化 | 全局 token 均值 + mask | `mean` over `real_action_dim` | 保留我们的，报告注明不可比 |

---

## 1. P0 资产与阻塞项

**实测结论（2026-10-17 18:0x）：词表已落盘并验证可用。HF 直连不通，走本机 clash（`127.0.0.1:7897`）可通。**

- [x] **T1.1 FAST tokenizer 落盘**
  - `assets/fast/`（5 个文件，已下载，见 `assets/fast/SHA256SUMS`）
    ```
    6f021ca1...  processing_action_tokenizer.py   (6145 B)
    f40cfbb1...  processor_config.json            (253 B)
    ca3d163b...  special_tokens_map.json          (3 B)
    b4030e2a...  tokenizer_config.json            (322 B)
    6507dd70...  tokenizer.json                   (686974 B)
    ```
  - 取数方式（可复现）：
    ```bash
    P=http://127.0.0.1:7897   # clash-verge mixed port，实测 huggingface.co 直连超时
    curl -sSL -x $P -o <file> https://huggingface.co/physical-intelligence/fast/resolve/main/<file>
    ```
  - **不要**用 `HF_ENDPOINT=hf-mirror` 那条路（也能通，但 clash 更快且已跑通）
- [x] **T1.2 加载验证（部分）**
  - 已在 `/home/ylhp-e-ai/miniconda3/bin/python`（py3.13 / transformers 5.13.1 / tokenizers 0.22.2 / scipy 1.17.1）通过：
    `AutoProcessor.from_pretrained(dir, trust_remote_code=True)` → `UniversalActionProcessor`
  - **仓库 venv（py3.12 / transformers 5.3.0）缺 scipy** → 见 T1.3 的决策
  - 两条无害告警（可忽略）：`Could not load tokenizer from subfolder bpe_tokenizer`、`Ignoring clean_up_tokenization_spaces=True`
- [x] **T1.3 依赖决策：离线预计算，训练环境不新增依赖**
  - 决定：FAST 编码放进**数据准备脚本**（一次性），把 `fast_tokens` 写进 fixtures；训练 venv 只需要 token id（`int64` 数组），**不装 scipy/不加载 remote code**
  - 理由：仓库 pyproject 硬钉版本（`numpy==1.26.4 / transformers==5.3.0`），往训练环境塞 scipy 有污染风险；token 化只依赖 (tokenizer 摘要, 归一化, horizon, dim)，全部可冻结
  - 需新增：`pyproject.toml` 里加 `fast = ["scipy"]` 可选 extra + `scripts/architecture/prepare_fast_tokens.py`
  - 仍需在**真实训练 venv** 里验证 `transformers 5.3.0 + trust_remote_code` 能否加载一次（用于数据准备脚本跑在同一解释器时的兜底）；不能则数据准备用 miniconda 解释器
- [x] **T1.4 实测 tokenizer 超参（与默认值不同，注意）**
  ```json
  {"scale": 10, "vocab_size": 2048, "min_token": -354, "action_dim": null, "time_horizon": null}
  ```
  - 编码：`round(dct(actions, axis=时间, norm=ortho) * 10)` → `+354` 平移到非负 → `chr` → BPE
  - token id 范围实测 `[106, 2035]`（在 2048 词表内）→ PaliGemma 映射 `pg = 257152 - 1 - 128 - id` 落在 `[255728, 256717]`，**不越界**
- [x] **T1.4b 真实动作的 token 长度实测（决定 `max_token_len`，关键）**
  | 动作形态（T=50, D=32, 14 维有效） | FAST token 数 | 解码 MSE（14 维） |
  |---|---:|---:|
  | 平滑正弦（各维同相） | 59–114 | 1.4e-2 |
  | 平滑正弦（各维异相） | 74 | 1.2e-2 |
  | 分段线性路点 | 73–86 | 1.3e-2 |
  | 均匀随机（最坏） | **927–943** | 2.3e-2 |
  | horizon=10 / 25 / 50 | 24 / 49 / 74 | 1.6e-2 / 1.3e-2 / 1.2e-2 |
  - 结论 1：`max_token_len=200` 对**平滑真实数据**够用但很紧（Task+State 段就吃掉 ~40–100 token）；参考实现 LIBERO 用 256 → **我们至少 256**
  - 结论 2：**FAST 是有损表征**，`scale=10` 的量化底噪约 **1.2e-2 MSE**（14 维归一化空间）。它是训练监督的表征，不是部署动作路径——报告里必须写清，别把 FAST 解码误差当成 flow 的性能
  - 结论 3：截断必须 fail-fast（见 T2.4）——horizon 若保持 50，随机/抖动数据轻松破 200
- [ ] **T1.5 冻结参考快照**
  - `references/kewang` 已是 commit `385d0e8` 的完整 clone；补 `MANIFEST.json`（commit + 用到文件的 sha256）防止后续漂移
- [x] **T1.6 训练侧可见性（已打通，两跳）**
  - 实测：仓库与 fixtures 是**两份**——本机 `/home/ylhp-e-ai/DATA2TB/kyz/soma-fluxvla`（代码/报告），
    pro6000-vm `/home/ubuntu/2xPro6000/...`（真正跑实验的地方，含 `fixtures/samples.npz`），
    assets 在 pro6000 的 `/home/ubuntu/e-ai-shared/soma-training/.../pi05_base`
  - **根因不是网络**：TCP/`tailscale ping` 都通；pro6000 的 `ubuntu` 现在只接受**公钥**（`ssh -o BatchMode=yes` 0.3s 成功），
    而本机 `rsh.py` 原来强制 `PubkeyAuthentication=no` + 密码 → 服务端不提供该方式时客户端静默卡在认证阶段。
    本机（pi 主机）`~/.ssh` 无私钥，`ylhp-e-ai` 那台有 key。
  - 方案（不改对方 authorized_keys）：两跳 —— `RSH_TARGET=ubuntu@100.78.78.46` 先密码登 `ylhp-e-ai`，再由它用 key 跳 pro6000
  - 已验证：`hostname=pro6000-vm`、`shoes_expansion_20261017/fixtures/samples.npz` 可见、两张卡空闲（各 2 MiB / 97887 MiB）
  - 待办：写 `scripts/architecture/sync_to_training_host.sh`（本机→pro6000 代码 + `assets/fast` + fixtures；pro6000→本机 结果）

## 2. P0 数据契约（`fluxvla/transforms/subtask_fast_prompts.py` + `scripts/architecture/`）

**已实现并测试通过（2026-10-17）：24 项 pytest 通过、ruff 通过、真 tokenizer 端到端布局验证 0 失败，
详见 `experiments/pi05_subtask_architecture/fast_stage1_20261017/REPORT.md`。**

- [x] **T2.1 六个并行数组的契约**
  `tokens / valid(prefix_mask) / ar_mask / loss_mask / subtask_region_mask / action_region_mask`，全部定长 `max_len=200`
  - 语义：`*_region_mask` 描述的是**输入位置**；CE 时统一 `[:,1:]` 做 next-token shift
  - 断言：`subtask_region_mask & action_region_mask == 0`，两者并集 == `loss_mask`（fast 模式）
- [x] **T2.2 拼接格式（严格照抄，不要自创）**（差一点：清洗函数不折叠连续空格，与参考一致）
  ```
  Task: {clean(g)}; State: {32×0..255}; Subtask: {clean(s)};
  \nAction: {<fast_token_pg>...}|{EOS}
  ```
  - `clean()`：lower / strip / `_`→空格 / 去末尾标点 / 补 `.`（参考实现 `tokenizer.py:201-233`）
  - **fast 模式下 subtask 段不带 EOS**（EOS 在 `|` 之后）；flow 模式仍是 `s;\nAction: ` + EOS —— **两种模式 token 序列不同，不能混用一个 pack 函数**
- [x] **T2.3 token id 映射**（`fast_to_model`/`model_to_fast` + 双向断言）
  `pg_id = vocab_size - 1 - 128 - fast_id`；断言 `pg_id >= text_vocab_start` 且 `pg_id < vocab_size`，并**用实际 embedding 行数**（不是 tokenizer 声明的 vocab）重算偏移——我们已经在 `<image>` 占位 token 上踩过这个坑
- [x] **T2.4 截断/溢出策略（与参考实现不同，必须 fail-fast）**
  参考实现在 `len > max_len` 时静默截断，**会切掉尾部 FAST token 与 EOS**，等于给模型错误监督。
  实测：平滑动作 horizon=50 只需 60–115 token，但随机/抖动数据可达 900+ → 溢出是真实风险。
  我们要：`len > max_len` → 直接报错并输出诊断（task 长度 / state 长度 / FAST token 数）；首选把 `max_token_len` 提到 256，并把 state 段从 32 个 bin 压到 16 个 bin 作为可选压缩策略（记录每档的 token 分布）。
- [x] **T2.5 fixtures**（`fast_stage1_20261017/fixtures/`：actions/fast_tokens/labels/layout）
  固定 3–5 条样本（正常 / 最长 / 边界）→ 保存 tokens+6 mask 的 npz 快照 + sha256，用于回归
- [x] **T2.6 一致性自检（脚本内）**（`prepare_fast_tokens.py` 记录 tokenizer sha256 + 往返 MSE）
  `decode(tokens[action_region]) → actions_hat`，与输入动作比重建 MSE；记录 `scale=10` 的量化误差上界

---

### 2.7 实测结果（真 tokenizer，max_len=256）

| 段 | token 数 |
|---|---:|
| prefix（`Task:` + `State:` 32 bin + `Subtask:`） | **150** |
| subtask 文本 + `;` | 6–9 |
| action 区（`\nAction: ` + FAST + `|` + EOS） | 69–77 |
| 合计 valid | **225–236**（余量仅 20–31） |

**`State` 段是最大开销，且 32 维里有 18 维是补零：**

| state 编码 | prefix token |
|---|---:|
| `state_dim=32`（14 真实 + 18 零） | **150** |
| `state_dim=32`（全 32 真实） | 139 |
| `state_dim=14`（只编真实维） | **73** |

→ **决策：`state_dim=14`**（省 ~77 token，256 预算下 FAST 可容纳到 ~140 token）。
否则 `max_len` 必须 ≥320。

## 3. P0 模型改动（`fluxvla/models/vlas/pi05_subtask.py`）

- [ ] **T3.1 FAST CE（与 subtask CE 共用一次前向）**
  - 在 `forward_high_level` 中接受 `action_region_mask`，按 `[:,1:]` shift 后**独立归一化**（`global_weighted_mean`，与 subtask 项分开计数）
  - 复用 `project_text`（embedding 转置），不新建随机 head
  - 断言：FAST 目标位置 ⊆ `valid & ar_mask`（沿用现有校验风格）
- [ ] **T3.2 三损失总装**
  `loss = w_subtask*CE_sub + w_fast*CE_fast + w_flow*MSE_flow`，三个权重进 config，默认 **10 / 1 / 0（stage 1）**
- [ ] **T3.3 flow 防泄漏（关键，当前完全缺失）**
  - 构建 flow prefix **之前**，把 action 区 token 置 0、`ll_valid_mask` 置 False（参考实现 `pi05.py:291-306`）
  - 验收：`flow` 分支的 prefix KV 与"不含 action 区"的序列逐元素一致；梯度不得从 flow 流入 action 区 token 的 embedding 行
- [ ] **T3.4 低层上下文决策（需你拍板，二选一）**
  - (a) 保持现状：`p(a|o,s)`，不含 `g` → 更贴论文公式，但与参考实现/文献数字不同源
  - (b) 对齐参考实现：低层 prefix 含 `Task: g; State; Subtask` → 可直接对比 Ke-Wang
  - 建议：**默认 (b)**（复现优先），把 (a) 作为一个显式开关 `low_level_sees_task`，两种都留测试
- [ ] **T3.5 `stop_gradient_flow_to_prefix` 保持不变**，与 T3.3 正交；验收时两组都跑（`cached` vs `insulated`）
- [ ] **T3.6 采样侧**
  连续动作部署**不生成 FAST**（FAST 只是训练时的动作表征监督）；`sample_actions_given_subtask` 不加新路径，但要加一条断言/文档说明
- [ ] **T3.7 修时间采样**
  用 `torch.distributions.Beta(1.5,1.0)` 替换 `sample_beta` 近似（或至少在 fast/hybrid 预设里显式传入真 Beta）

---

## 4. P1 配置与训练

- [ ] **T4.1 三档预设（新建 `configs/pi05_subtask_fast.py`）**
  | 预设 | w_sub | w_fast | w_flow | 用途 |
  |---|---|---|---|---|
  | `stage1_fast` | 10 | 1 | 0 | 让 VLM 学会动作表征（完整 KI 的前置） |
  | `flow_only` | 0 | 0 | 1 | 基线（我们已有） |
  | `hybrid` | 0.15 | 0.15 | 1 | 最终配方 |
- [ ] **T4.2 超参对齐参考**（LIBERO 值，我们数据要重算，但要记录推导）
  `lr 2.5e-5 cosine → 2.5e-6`、`warmup 3000`、`bs 32`、`ema 0.999`、`max_token_len 256`、`action_horizon 25(fast)/10(flow)`
- [ ] **T4.3 分阶段训练脚本**：stage1 → hybrid；每阶段独立 checkpoint + `execution_status.json`
- [ ] **T4.4 数据侧准备（含 FAST 预计算，见 T1.3）**：`prepare_shoes_subtasks.py` 增加 FAST token 列（或训练时在线编码，需固定 tokenizer 摘要）

---

## 5. P0 验收测试（写进 `tests/`，无测试不算完成）

- [ ] **T5.1 region/shift 正确性**：构造可判定样本，断言 subtask 与 FAST 两段 CE 的 target 位置精确
- [ ] **T5.2 区域隔离**：只改 FAST 目标文本 → subtask CE 不变（反之亦然）
- [ ] **T5.3 防泄漏**：flow 分支 prefix 中 action 区被清零 + mask False；给出"改动 FAST token 不影响 flow 输出"的证据
- [ ] **T5.4 与参考实现逐 token 对齐**：同一 prompt/action，比较 token ids + 6 个 mask（允许 state bin 差异，需列明）
- [ ] **T5.5 编解码往返**：`actions → FAST → decode → actions_hat` 的 MSE 上界；边界值（±1、0、常数序列）
- [ ] **T5.6 三档权重 loss 健全性**：无 NaN、零监督不炸、DDP 全局归一化正确
- [ ] **T5.7 现有 26 项测试不许回归**

---

## 6. P1/P2 实验（复现顺序）

- [ ] **T6.1 stage1 在 shoes 99 点跑通**：报 subtask 生成准确率 + FAST token top-1 + 解码 MSE
- [ ] **T6.2 hybrid vs flow_only vs stage1-only** 的三方对照（同一验证集/噪声/预算）
- [ ] **T6.3 重跑"隔离 vs 不隔离"**：这次才是完整 KI 的对照（之前缺 FAST，结论不成立）
- [ ] **T6.4 报告口径固定**：区分"格式正确"与"语义正确"；给多数类基线 + 混淆矩阵；`zero_state` 消融必须进结论
- [ ] **T6.5 明确不做**：RTC、A2 联合前向、异步调度、真机/仿真成功率

---

## 7. 已知坑（从两套代码里核出来的，照抄会中招）

1. **Ke-Wang `sample_actions` 复用生成后的 `kv_cache`，但 mask 只用 prefill 的 `prefix_mask`** → 逐样本变长/裁剪时有越界隐患。我们保持"重新 prefill flow 前缀"的做法，不照抄。
2. **token 映射偏移必须用 embedding 实际行数**，不是 tokenizer 声明 vocab（`<image>` 占位 token 事件）。
3. **fast 模式下 subtask 段没有 EOS** → 推理停止条件与 v1 不同，`generate_subtask` 的 EOS/`|` 处理要单独写测试。
4. **截断**会静默切掉 FAST 尾部和 `|` → 必须 fail-fast。
5. `transformers 5.3` + `trust_remote_code` 的风险 → 优先 vendor。
6. FAST `scale=10` 与我们的 quantile(q01/q99) 归一化耦合：**换归一化就等于换词表分布**，一换必须重测 T5.5。
7. `State` bin 串在 FAst 段之前的同一因果链里；若 state 已经决定动作，FAST CE 会显得"太容易" → 必须报 `no_state` / `shuffled_state` 消融。
8. 参考实现的 flow loss 是 `mean over real_action_dim` 无 mask；我们是 mask 化全局均值 → **数字不可直接对比**，报告里写死。

---

## 8. 建议执行顺序

1. T1.1–T1.3（解锁 tokenizer，半天，最大风险在这里）
2. T2.1–T2.5（数据契约 + fixtures，1 天）
3. T3.1–T3.3（FAST CE + 防泄漏，1–2 天）
4. T5.1–T5.5（验收测试，与 3 同步写）
5. T4.1–T4.3（配置 + stage1 训练）
6. T6.1–T6.3（实验对照）

**完成定义**：三档预设可跑、防泄漏有测试证据、与参考实现逐 token 对齐、stage1 与 hybrid 的数字齐全且口径写清。
