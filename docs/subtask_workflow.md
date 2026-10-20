# Subtask 训练、导出与推理

在仓库根目录执行下面的命令。先安装 `pip install -e '.[cache,test]'`，
准备本地基础权重和 tokenizer；数据与权重不会由这些命令自动下载。
Python 3.12、PyTorch 2.8.0、Transformers 5.3.0 是本次整合使用的环境。

## 代码边界

| 目录 | 职责 |
|---|---|
| `fluxvla/models/vlas/pi05_subtask.py` | 高层文本、FAST 辅助监督、低层 flow、KV cache |
| `fluxvla/transforms/subtask_fast_prompts.py` | 训练与推理共用的 prompt/token/mask |
| `fluxvla/training/presets/` | 普通 PI0.5 与 subtask 模型配置 |
| `fluxvla/training/subtask/` | 数据准备、两阶段训练、DDP、checkpoint 与导出 |
| `fluxvla/inference/` | 完整权重加载、图像/状态处理、subtask 稳定策略和动作解码 |
| `scripts/architecture/` | 保留的历史实验，不是生产训练/推理入口 |

`configs/pi05_piper.py` 和 `configs/pi05_subtask_fast.py` 是兼容入口，
每次读取生成独立配置，避免不同实验互相修改配置。无 subtask 训练仍用 `scripts/train.py`。
历史 `SubtaskRuntime` 接收归一化张量，仅用于架构实验；实际数据评估和服务端均调用 `SubtaskPolicy`。

## 模型与条件语义

三路 RGB 图像、总体任务和离散化状态进入高层，生成 subtask 与后续 FAST token。
Stage1 训练 subtask/FAST 的 CE；Stage2 同时训练 CE 和 flow matching。
模型参数名称和结构保留，完整 safetensors 可以严格加载。

**训练的 flow 分支读取重新编码的 text 条件，prefix 是双向注意力。**
高层生成文本使用因果注意力。两者不能因为共用 VLM 就视为同一个 KV 条件。
`generation` 模式将高层生成 KV 直接交给动作分支，保留用于显式消融；新导出默认 `text`。
FAST 是训练辅助监督，实际连续动作由 flow sampler 生成。

- 默认完整生成上限为 64 token；日志保留所有生成 token 和原始文本。
- `Action:` 后的内容只从动作条件中排除，不截断日志中的生成输出。
- text 模式重新编码当前图像/状态/总体任务/选定 subtask。
- text 稳定策略默认为连续两次新预测确认切换；复用旧标签不累计投票。
- 失败的动作推理不提交标签切换。每个 episode 应调用 `policy.reset()`。
- `SubtaskPolicy` 是单流有状态对象，每个会话使用独立实例。

## 训练

编辑 `configs/subtask/fabric106.json` 的数据、缓存、基础权重、tokenizer 和输出目录。
该配方固定 106 episodes（93 训练、13 验证）、anchor stride 8、horizon 50、全局 batch 64、
8 GPU，每卡 micro batch 4；Stage1 4 epochs，在 epoch3/4 中挑选；Stage2 hybrid 8 epochs。
改变数据集时也要调整 episode 数与校验哈希，使用新缓存/输出目录。

```bash
# 轻量环境检查；不启动训练，也不验证完整基础权重哈希。
python scripts/train_subtask.py --config configs/subtask/fabric106.json --check

# 仅检查或准备数据（不需要 GPU）
python -m fluxvla.training.subtask.prepare scan --config configs/subtask/fabric106.json
python -m fluxvla.training.subtask.prepare prepare --config configs/subtask/fabric106.json
python -m fluxvla.training.subtask.prepare verify --config configs/subtask/fabric106.json

# 完整流程：校验基础权重、缓存、存储容量，随后运行两阶段。
python scripts/train_subtask.py --config configs/subtask/fabric106.json
```

数据需为 LeRobot v2.1，包含 `meta/info.json`、`meta/episodes.jsonl`、parquet、三路图像或视频。
每帧需要 `state`/`observation.state`、`actions`/`action`、`timestamp`、
`overall_task`、`subtask`、`subtask_valid`；状态/动作均为 14 维。
训练动作是 **下一帧起的 12 个关节相对 anchor 的增量，以及 2 个夹爪的绝对米值**。
有效 action horizon 不跨标注区间末端；不足部分置零并用 mask 排除。
归一化只拟合训练轨迹，验证轨迹不会进入统计量。

中间 checkpoint 保存部署权重，不含优化器；Stage2 最终 checkpoint 另保存优化器和 RNG。
中途仅有权重时，默认拒绝声称精确续训。明确接受新优化器才使用：

```bash
python scripts/train_subtask.py --config configs/subtask/fabric106.json --warm-start-stage2
```

## 导出

训练流程自动生成 `*.deploy/`；通常直接复制整个目录即可。
从完整 `.pt` 再导出，需要同目录的 `.pt.ready.json` 和原始训练缓存：

```bash
python scripts/export_checkpoint.py \
  --checkpoint work_dirs/fabric106_ae8/stage2/step-00002528.pt \
  --output checkpoints/step-00002528.deploy
# 迁移机器后可以添加 --cache <原缓存新路径> --tokenizer <原tokenizer新路径>
```

部署包包含完整模型、原生配置、tokenizer、归一化统计、条件合同和 SHA256 清单；
不依赖另一份基础权重。已有输出目录不会被覆盖。权重、统计量与 tokenizer 必须成套迁移。
仅有 `model_delta*.pt` 的历史产物不能传给这个入口。

旧云端包的合同写了 generation，但训练代码实际使用 text。核对来源后，可显式使用
`--allow-legacy-contract` 以 text 加载；日志会记录该覆盖。不要据此自动信任任意旧包。
新包明确记录双向 text flow 条件；状态量化使用训练的 float64 quantile 运算再转 float32。
旧包继续使用部署基线的 float32 quantile 运算，保证既有输出回归；临界离散化点可能有微小舍入差异。

## 离线评估

输入 NPZ：`states [N,14]` 为物理状态（关节 rad，夹爪 m），
`images [N,9,H,W]` 为按前、左腕、右腕排列的 RGB uint8。
可选 `actions [N,T,14]` 是**绝对物理 GT**，可选同形状 `action_masks` 排除无效目标。
不要直接把训练缓存中的归一化 states/actions 当成此格式。
JSON 的 `records` 按 episode/frame 排序，每行包含 `episode`、`frame`、`overall_task`，
可选 `source`、`subtask`。GT subtask 只用于评分，不作为推理条件输入。

```bash
python scripts/evaluate_subtask.py \
  --bundle checkpoints/step-00000750.deploy --allow-legacy-contract \
  --observations data/offline/observations.npz --records data/offline/records.json \
  --output artifacts/offline-step750 --seed 1702 --confirmations 1
```

默认 confirmations=1 测原始预测；改成 2 测部署稳定策略。对比不同权重应保持样本顺序、seed 与参数一致。
输出逐帧 subtask/完整 token/切换原因 JSONL、绝对动作 NPZ，以及准确率、GT 连续区间内跳变数、
关节 RMSE、夹爪 MAE。没有 GT 时相应指标为空。这些指标不同于 teacher-forced CE 与训练 flow loss，
也不能直接代表真机成功率。

## 推理服务与真机的边界

```bash
# 安装另一个仓库提供的 soma_deploy serving 与 pi05_subtask 客户端协议。
# 也可以用 PYTHONPATH=/path/to/soma-deploy/src 指向已有框架。
pip install -e '.[serve]'
python scripts/serve_policy.py \
  --bundle checkpoints/step-00000750.deploy --allow-legacy-contract \
  --config configs/deploy/subtask_text.json --port 10093
```

这里只启动 loopback 推理服务，不初始化 CAN、相机、机械臂或急停。
连接方需支持 `soma.policy_server.v1` 和 `pi05_subtask` client adapter。
服务端单实例只供一个连续任务流使用；新任务/新会话应重启服务，避免继承上一会话的标签确认状态。
远程传输、执行频率、RTC、滤波、速度限制、碰撞/急停和归位由外部部署框架负责。
本次整合没有修改现场运行配置，也没有提供自动使能机械臂的入口。

Python API 的 `actions` 是绝对目标；`semantic_actions` 是反归一化后的关节增量/夹爪米值。
SOMA 协议的 `actions` 必须发送 `semantic_actions`，由现有客户端加一次 `state_used`。
夹爪索引为 6、13，连续控制，scale=1；不能额外乘 0.07 或再次加状态。
50 帧 horizon、30 Hz 模型动作采样间隔与请求推理的 Hz 是不同参数。
每次请求都重建当前观测的动作条件；标签保持不会复用旧图像的 KV。

## 验证

```bash
OMP_NUM_THREADS=1 pytest -q
ruff check fluxvla scripts tests

# 两进程 CPU 小模型端到端检查（不等于完整 8 GPU 训练复现）
SUBTASK_SMOKE_DIR=/tmp/subtask-smoke pytest -q tests/test_subtask_pipeline.py
PIPELINE_SMOKE_CONFIG=/tmp/subtask-smoke/config.json \
  pytest -q tests/test_pipeline_final_only_integration.py
```

端到端检查包括 cache/FAST、两阶段训练、模型导出、权重内容一致性、显式 warm start、
优化器仅最终保存和失败后重试。完整权重的数值回归见 `docs/validation_20261020.json`。
