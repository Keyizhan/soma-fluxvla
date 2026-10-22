# soma-fluxvla

PI0.5 Piper 与 subtask + FAST + flow 的模型、训练、导出和推理代码。

Subtask 完整流程见 [训练、导出与推理说明](docs/subtask_workflow.md)。
FAST 损失检查、编解码验证和四组离线对照见 [2026-10-20 诊断报告](docs/fast_audit_20261020.md)。
官方论文对照、关闭 FAST 的补充实验和后续架构方案见 [小数据适配研究](docs/research/pi05_small_data_plan.md)。
本仓库提供模型与推理服务；真机控制由独立部署框架负责。

新增入口：`scripts/train_subtask.py`、`scripts/export_checkpoint.py`、
`scripts/evaluate_subtask.py` 和 `scripts/serve_policy.py`。
统一推理位于 `fluxvla/inference/`，默认 text 条件、64 token 上限、BF16 GPU 推理。
原有无 subtask 训练入口仍保留。数据、权重、凭据和机器运行日志不在仓库中。
从旁边的 `fluxvla-training` 快照整理，Python 包名保留 `fluxvla`，以兼容原有模型和配置。

## 安装

建议 Linux、Python 3.12。先安装与 CUDA 匹配的 PyTorch 2.8.0 和 torchvision 0.23.0，
再安装本项目；云端当前训练使用这两个版本。项目安装不编译推理专用 CUDA 扩展。

```bash
python -m venv .venv
source .venv/bin/activate
# 先按本机 CUDA 环境安装 torch==2.8.0 和 torchvision==0.23.0。
pip install -e '.[cache,benchmark,test]'
```

训练与缓存读取不要求 Decord；构建、原视频像素校验要求 `cache` extra。
构建器使用与云端实验相同的 `decord2==3.4.0`（导入名仍为 `decord`）。
压缩测试要求 `benchmark` extra。依赖不包含仿真、ROS、其他 VLA 或奖励模型环境。

## 目录

```text
configs/pi05_piper.py           云端 shoes 运行参数的可移植版本
scripts/train.py                无 subtask 训练入口
scripts/train.sh                torchrun 启动器
scripts/build_video_cache.py    缓存构建
scripts/verify_video_cache.py   对比原视频像素
scripts/benchmark_video_cache.py 压缩实验（不接入训练）
fluxvla/datasets/cache/         Zstd 块读取、构建和索引管理
experiments/                   本地实验结果（不提交 Git）
tests/                         缓存、分片和恢复兼容性检查
```

## 数据、权重和配置

默认相对路径均相对于项目根目录：

- 数据：`data/piper_shoes_v21`，包含 `meta/`、`data/`、`videos/`。
- 基础权重和 tokenizer：`checkpoints/pi05_base/`。
- 解码缓存：`cache/piper_shoes_v21/`。
- 输出：`work_dirs/pi05_piper/`。

这些大文件不复制进源码。可使用软链接，或修改配置中的相应路径。
更改数据目录时，同时更新 `data_root_path` 和 `video_cache.data_root`。
配置中的归一化统计来自此次 shoes 数据；换数据集时必须替换或重新计算统计。

默认参数与云端此次训练一致：5 epochs、每卡 batch 8、每卡 12 workers、
梯度累积 1、学习率 1.6e-4、BF16、RTC 训练 max_delay 7、每 3000 步及每轮保存。
8 卡时全局 batch 为 64；改变卡数会改变全局 batch 和每轮步数。

## 构建和检查缓存

```bash
python scripts/build_video_cache.py \
  --data-root data/piper_shoes_v21 --cache-dir cache/piper_shoes_v21 \
  --workers 4 --chunk-frames 16

python scripts/verify_video_cache.py \
  --data-root data/piper_shoes_v21 --cache-dir cache/piper_shoes_v21
```

缓存保存 RGB、224×224、uint8 帧以及真实帧时间戳，**每 8 帧独立用 Zstd level 3 无损压缩**。
索引记录每块的字节偏移和长度；读取随机帧只解压其所在块，不需要解码 MP4。
所有 GPU/worker mmap 同一份压缩文件，
由操作系统共享文件页。每个 worker 限制打开映射数量；返回给训练的是独立数组。
构建按小批帧写入，支持验证后续建、原子发布索引、来源/尺寸变更识别。
不同版本数据或尺寸建议使用不同缓存目录；旧版本文件不会自动删除。

默认缺缓存或缓存过期会报错，不会静默切回慢速视频解码。禁用缓存可将
`ProcessParquetInputs.video_cache` 设为 `None`。

读取器也兼容旧 `.npy` 文件，以及旧云端 `index.json`（绝对路径、fps/n 格式）。这类索引没有来源校验和
真实时间戳，沿用旧路径、按 FPS 匹配；搬迁后建议用新构建器重建。

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 15 \
  python scripts/benchmark_video_cache.py \
  --cache-dir cache/piper_shoes_v21 --data-root data/piper_shoes_v21 \
  --output-dir experiments/compression_local
```

基准默认测试 8 帧 Zstd level 3，并测量训练实际使用的缓存读取器。
添加 `--compare` 可比较 LZ4、Zstd、差分 Zstd 的 1/8/32 帧块；所有变体逐字节验证无损。
随机读取测试针对文件页已缓存的情况，不代表冷盘吞吐或完整压缩训练性能。
新项目的训练读取默认采用 8 帧 Zstd 缓存。云端原训练仍使用旧 `.npy` 缓存，
此次本地整理没有切换云端格式。压缩基准也保留其他块大小作对照。

## 训练与断点恢复

```bash
# 单卡启动检查；完整训练建议使用原来 8 卡的资源规模。
NPROC_PER_NODE=1 bash scripts/train.sh configs/pi05_piper.py work_dirs/pi05_piper

# 8 卡，与云端此次训练的 batch 设置一致。
NPROC_PER_NODE=8 bash scripts/train.sh configs/pi05_piper.py work_dirs/pi05_piper

# 使用包含训练状态的 .pt 恢复，不能用仅权重的 .safetensors 替代。
NPROC_PER_NODE=8 bash scripts/train.sh configs/pi05_piper.py work_dirs/pi05_piper \
  --resume-from work_dirs/pi05_piper/checkpoints/latest-checkpoint.pt
```

启动器支持 `PYTHON`、`NNODES`、`NODE_RANK`、`MASTER_ADDR`、`MASTER_PORT`。
使用 nohup/终端复用器可让训练独立于 SSH；启动器本身在前台运行，方便显示错误。
恢复沿用云端模型、优化器、调度器、step/epoch 恢复逻辑；不承诺复现中断前的
worker 预取队列和随机数状态。不同卡数恢复也不保证相同样本顺序。
恢复完整 `.pt` 时跳过重新加载基础权重，并严格检查模型 state_dict 一致性；
tokenizer 文件和训练数据仍需就位。架构不同或不完整的权重文件不能作为完整训练断点。

数据由 `DistributedRepeatingDataset` 按 rank×worker 分片和打乱，配置保持
`sampler=None`；不要额外给这个 IterableDataset 设置 DistributedSampler。

## 保留和移除的范围

保留 PI0.5、它继承的 PI0/BaseVLA、ConditionGemma、SigLIP、线性投影器、
Piper Parquet 数据处理、训练器、优化器、日志和 RTC **训练**。
不包含 GR00T、DreamZero、OpenVLA、LLaVA、SmolVLA、X-VLA、ARM/SARM、
真机/ROS/远程推理、RTC 推理加速、仿真评估、其他机器人示例及上游测试资源。
部分通用基类仍包含训练共享接口；不会为了删掉名字中的上游标识而改变模型结构。

原目录和云端训练不受本项目创建影响。来源清单见 `source-manifest.json`，
上游版权及许可见 `NOTICE` 和 `LICENSE`。

## 验证

```bash
python -m pytest -q
ruff check .
python -m compileall -q fluxvla scripts configs
```

本地 CPU 检查用于验证依赖、注册、数据/缓存和微型模型的训练状态恢复；
完整 PI0.5 8 卡训练仍需目标 GPU、基础权重和 Piper 数据。


## Controlled subtask comparison (2026-10-22)

New runs resolve `configs/subtask/common.json` with three small method overrides.
`runtime.json` supplies the shared seed, data/cache/base paths and hardware/precision settings.
A run file can extend a method file and override runtime settings; resolution happens only
after the full inheritance chain is merged.
The fixed `goal_text_v1_split.json` specifies 89 train / 11 validation / 15 test episodes.
All paths are repository-working-directory relative except `extends`, which is relative
to its configuration file. Set the common incoming path for the target Linux host.
Historical fabric106 filenames are aliases; the data identity is dataset_goal_text_v1.

```bash
python -m fluxvla.training.subtask.comparison --configs configs/subtask/fabric106.json configs/subtask/fabric106_hybrid_insulated.json configs/subtask/fabric106_two_stage.json --check
```

Omit --check on the configured GPU host to execute. Resolved configs reject differences
outside method/preset, output, notes and run identifier. Learning rates, epochs, runtime,
precision and deployment settings must match. Hybrid starts directly from the official
base and runs one joint 8-epoch phase. Hybrid_insulated and two_stage share a 4-epoch
subtask+FAST stage1 checkpoint, then run 8 epochs of insulated joint training or
frozen-VLM action-only training, respectively. Shared stage1 selects validation
subtask CE at epochs 3/4 and pins actual SHA256, cache and configuration identity.
Both staged methods reset optimizer and schedule at the transition. Comparison
rejects differing stage1 recipes between these two methods.

Hybrid has 8 total epochs; hybrid_insulated and two_stage have 12.
Hybrid versus hybrid_insulated is not a stop-gradient-only ablation: initialization
also differs. KI motivates the second-stage gradient design, not the necessity of
our preliminary FAST stage. A same-stage1 joint-versus-insulated ablation is a
separate experiment. Metadata records initialization and update budgets.

| Method | Stage2 subtask / FAST / flow | VLM gradients |
| --- | --- | --- |
| hybrid | .15 / .15 / 1 | CE and flow |
| hybrid_insulated | .15 / .15 / 1 | CE; flow KV detached |
| two_stage | 0 / 0 / 1 | vision/language/projector frozen; KV detached |

All methods use Task + State + Subtask through its semicolon as the low-level prefix,
segmented tokenization, causal text and bidirectional image tokens. No low-level Action,
EOS or FAST tokens are included. Deployment and text-reconstruction inference share that
contract; historical bundles retain recorded legacy compatibility. High-level FAST labels
remain available to evaluate every method. Batch 64, BF16, no EMA and no augmentation match.

Rebuild old caches. Scan checks actual Parquet and trajectory hashes and rejects duplicate
trajectories crossing fixed splits. Only train fits normalization. Test is recorded only
in split audit metadata and never enters training caches, FAST encoding, prompt budgeting,
batching or model selection. READY records data/labels, split and normalization identities;
verification rescans source files. Every valid frame is sampled; next-frame windows
cross label boundaries and repeat the final action at episode end, with repeated
tail positions excluded from flow loss. FAST still encodes the fixed padded window.

Validation computes unweighted subtask CE, FAST CE and flow MSE even when training weights
are zero. Flow noise and Beta time use sample-ID/seed/stream/update keys. Validation uses
a fixed independent stream and identical samples across methods; shuffle is independent.

```bash
python -m fluxvla.training.subtask.evaluate --config configs/subtask/fabric106_two_stage.json --bundle <final.deploy> --device cuda
```

This independent evaluator currently supports validation metrics only; a separate final test evaluator is not implemented. CPU tests use torch 2.5.1 and
transformers 5.3.0; target torch 2.8 GPU/DDP and real robot rollouts remain separate checks.
Durable checkpoint transactions require Linux statvfs and directory fsync; Windows CPU
validation does not certify production checkpoint durability. No full training is launched.


Validation recorded during this implementation: the latest affected configuration/cache
regression has 20 passed; the preceding model/prefix regression has 38 passed. A wider
53-test group also passed before the final refinements. Linux-only checkpoint transaction
tests fail on Windows at missing os.statvfs; persistence guarantees were not weakened.
Direct train validates cache configuration and fixed split identity; source/cache file
hashing is performed by run verification. The recorded base SHA is configured identity;
run preflight verifies the actual base file. No full training, production cache rebuild
or final test-set run occurred.


### Shared PI05 training policy alignment (2026-10-22)

All three subtask methods now use the ordinary PI05 warmup/cosine schedule:
5% warmup with floor-rounded steps, first optimizer update at LR zero, cosine
scheduler ending at zero. Peak learning rates and method stages are unchanged.
AdamW reuses the ordinary parameter grouping (no decay on bias or 1-D parameters).
RTC uses the ordinary exponential delay sampler (max_delay=7 exclusive,
temperature=1), clean prefix time zero, and masks delay positions from flow loss.
Sample-keyed RTC draws preserve independent training and validation randomness.

Training losses are local valid-element means, equally averaged over accumulation
microbatches, then averaged by DDP. Validation separately aggregates actual valid
numerators/counts, including RTC and episode-tail masks. Correct Beta time sampling
is retained; it was not one of the requested alignment changes.

Caches now use `next-frame-fixed-horizon-masked-tail-v3` and
`controlled-goal-text-v2`, at `cache/goal_text_v1_pi05_aligned` by default.
Rebuild caches and train a new recipe; old caches/checkpoints must not be relabeled.
The shared-stage1 root also has a new aligned name. Existing output directories
with an old run contract are rejected; use a new output directory in that case.

### Required flow-time sampling

Ordinary PI0.5 and all subtask recipes MUST use `time_sampler="beta"`, with default alpha=1.5 and beta=1.0. Configure `time_beta_alpha` and `time_beta_beta` in `configs/subtask/common.json` (or a method override); both must be finite and positive. These values control actual keyed training/validation draws and the saved model configuration for all three methods. Validation uses the configured law too, so validation losses across different Beta configurations are not directly comparable. The bundle evaluator enforces matching configuration; a separate common-distribution evaluation override is not provided. The corrected local FluxVLA Beta branch is shared by the models. Subtask keyed sampling uses the equivalent inverse CDF when beta=1, otherwise PyTorch Beta with isolated CPU RNG. Default draws remain unchanged. The old uniform-power ratio and `exact_beta_time=False` are rejected; missing sampler fields default to correct Beta. Existing `exact_beta_time=True` bundle fields are accepted for compatibility. New ordinary runs therefore use a different time distribution from historical buggy runs.
