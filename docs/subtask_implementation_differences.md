# Subtask训练实现差异与当前受控比较

更新／重新核查：2026-10-22。本文以 `ours/code/soma-fluxvla-main` 当前实际代码为准。已实现低层task条件隔离与full-prefix attention；此前低层保留整体task、全causal文本、106条数据及随机93/13划分的描述均不再适用于新训练。

## 1. 当前三种方法

令Ls为subtask CE、Lf为FAST CE、La为flow MSE；Ns、Nf为有效监督token数，Ltoken=(Ns·Ls+Nf·Lf)/(Ns+Nf)。统一目标在每个rank/microbatch内按全部有效token归一化；空监督批次返回可微零。联合flow系数λ由共同配置joint_flow_weight控制，默认1.0，尚未经调参，不视为官方α=10的数值等价。hybrid直接从官方base联合训练8 epochs；hybrid_insulated与two_stage先执行共享的4 epochs FAST Stage1，再分别进行8 epochs隔离梯度联合训练与动作专家训练。保留历史stage2目录名作为联合／动作阶段标识，不代表hybrid存在Stage1。

| 方法 | 初始阶段 | 联合／动作阶段 | flow到VLM | VLM训练 |
|---|---|---|---|---|
| hybrid | 官方base | Ltoken+λLa，8 epochs | 允许 | CE与flow |
| hybrid_insulated | Ltoken，4 epochs | Ltoken+λLa，8 epochs | prefix KV detach | CE |
| two_stage | Ltoken，4 epochs | La，8 epochs | prefix KV detach | 视觉／语言／projector冻结 |

hybrid_insulated与two_stage共享同一Stage1 checkpoint，总预算均为12 epochs；hybrid直接base训练8 epochs。因此hybrid与hybrid_insulated同时改变初始化与梯度隔离，不能称作仅stop-gradient不同的消融。若验证隔离效果，应另设从同一Stage1出发的联合训练消融，不改变这三个方法的定义。metadata记录初始化来源、总优化更新预算与额外Stage1更新数。共同batch64、原有学习率、BF16、无EMA、无augmentation保持不变。

配置入口为 `configs/subtask/common.json`（共同训练配方）、`runtime.json`（共享seed／路径／硬件与精度）和三个fabric106方法小覆盖。文件名fabric106仅保留命令兼容，不表示仍使用106条数据。继承链先合并、后统一解析，支持运行文件extends方法后覆盖seed；方法文件不复制共同stage数值。

`python -m fluxvla.training.subtask.comparison --configs <三个方法配置> --check` 对解析结果执行allowlist比较。允许方法、输出、运行标识以及方法规定的阶段结构／preset差异；共同联合／动作阶段的epochs、LR、运行时、数据、提示或其它参数不同会拒绝同组比较。

## 2. 固定新数据及身份

当前数据固定为 `ours/dataset_goal_text_v1`。实际扫描与独立审计均得到：

| 集合 | episodes | frames | valid anchors |
|---|---:|---:|---:|
| train | 89 | 180145 | 155515 |
| validation | 11 | 22028 | 19032 |
| test | 15 | 31291 | 27125 |

固定episode manifest为 `configs/subtask/goal_text_v1_split.json`。训练不再随机重划episode。扫描对实际Parquet取SHA256（包括标签），并对float32 state+float32 action+float64 timestamp计算轨迹hash，拒绝重复轨迹跨集合。当前未发现跨集合精确重复；这不等于排除近重复或语义重复。

归一化只由train拟合。test仅出现在split审计manifest与身份记录，不进入训练缓存、FAST编码、prompt长度预算、epoch batching或选模。train与validation缓存可跨训练seed复用。READY记录Parquet／标签、轨迹、split和归一化身份；run的verify重扫来源及缓存文件hash，direct train执行轻量cache配置与固定split身份校验。旧缓存必须重建。

## 3. 提示、动作与评估契约

所有三方法使用同一低层 `State: …; Subtask: …;`，无整体Task、Action delimiter、EOS或FAST答案。高层仍根据图像、整体task和state预测subtask。prepare、train与独立evaluate共用canonical prompt，低层训练使用真值subtask，推理使用生成文本重新编码；禁止复用包含整体task的高层generation KV，新模型底层API也拒绝外部prefix cache。

高层图像、task与state条件构成双向full-prefix，subtask／FAST目标仍自回归；低层图像、state与已给定subtask组成双向full-prefix。action tokens互相可见且可读取低层prefix，prefix不能看action，连续动作分支不能看FAST答案。two_stage仍保留高层FAST labels供独立评估。

新提示契约为`pi05-subtask-paper-v1`，导出显式记录`state_subtask`格式、task-free属性及attention规则。训练、导出和部署保持一致；缺失／未知的新契约或模型与prompt矛盾会报错，不能降级为旧格式。真legacy bundle保留原有兼容路径，历史行为不代表新训练默认。

动作窗口从当前有效帧的t+1开始，固定horizon50，跨subtask边界；episode末尾重复最后动作但flow mask=0。每个valid帧均采样（最后一帧无有效未来动作，flow监督为零）；FAST继续编码完整补齐窗口。12关节维度相对anchor state取差、2夹爪维度保持绝对值，flow仅监督真实14维。

独立验证器始终真实计算不加权subtask CE、FAST CE、flow MSE，zero training loss weight不替代为零指标。三方法使用相同验证样本与数量，flow noise与time通过sample ID、seed和stream固定；数据shuffle、noise、time和validation分离随机流，validation不改变训练随机状态。

`python -m fluxvla.training.subtask.evaluate --config <方法配置> --bundle <stage2/step-00019440.deploy>` 提供validation三指标评估，并检查bundle与cache／共同配置身份。bundle路径以该阶段`COMPLETE.json`的`deployment`字段为准；当前155515个train anchors、batch64对应每epoch2430次更新，Stage2最终为19440步（Stage1预算9720步），`final.deploy`仅用于Stage1选模导出。当前不提供独立最终test evaluator；test闭环成功率和真机评测仍未覆盖。共同联合／动作阶段主要比较固定最终step，逐epoch验证仅作诊断；hybrid_insulated与two_stage的共享Stage1保留epochs3／4按validation subtask CE选模。

## 4. 初始化、恢复与导出

hybrid_insulated与two_stage的共享Stage1存于shared_stage1_root/seed-<seed>/stage1，选定receipt固定实际model.safetensors SHA256、cache身份和共同配置。复用不仅检查COMPLETE，还校验bundle全部文件hash和选定权重。动作阶段重置optimizer与schedule；新运行、显式warm-start、最终full恢复和COMPLETE复用均保留并校验Stage1 provenance。输出目录变化不会触发另一套Stage1。hybrid不生成或读取Stage1，直接加载官方base；run preflight核验实际base hash，direct train记录的base hash属于配置声明值。

当前checkpoint策略为`optimizer_save=final_only`，不能将中途续训理解为精确恢复：

- Stage1仅在epoch3／4保存权重bundle，结束后按subtask CE选择并导出`final.deploy`，删除这两个候选目录；全程不保存optimizer／RNG。选中的权重可能来自epoch3，最终metadata中的`update=9720`表示已完成训练预算，实际选中位置见`picked_eval_epoch`／`picked_eval_update`。
- Stage2每750次optimizer更新保存`step-XXXXXXXX.deploy`，最终19440步同时保存完整`.pt`（模型、optimizer、各rank RNG及进度）和同名`.deploy`。中间没有optimizer快照，也不按每个epoch额外保存；停止参数不会强制产生额外快照。
- 遇到中间weight-only快照，默认拒绝续训；需显式使用`run --warm-start-stage2`或直接`train --warm-start`。该路径沿用快照进度及其LR位置，但重新初始化optimizer、不恢复RNG，不能宣称逐位恢复。最终full只支持最后保存／导出收尾的恢复，不提供中途精确断点恢复。

依据：[训练与保存实现](D:/pi05_subtask/ours/code/soma-fluxvla-main/fluxvla/training/subtask/train.py:223)、[checkpoint策略校验](D:/pi05_subtask/ours/code/soma-fluxvla-main/fluxvla/training/subtask/checkpoints.py:59)。

持久化事务仍要求Linux statvfs与目录fsync；未为Windows测试降低持久性保证。所有新bundle低层契约一致，旧bundle兼容与新契约明确分开。

## 5. 与kewang及π0.5官方论文的逐项对照

### 5.1 ours与本地kewang快照

本表比较实际代码路径，而非把README、博客与配置视为同一配方。数值、窗口与损失名称相同不代表训练样本、物理控制量或梯度归一化相同。下列链接指向当前本地文件和一基行号；数据与checkpoint不复制到源码仓库。

| 维度 | ours当前实现 | kewang本地快照 | 代码依据 |
|---|---|---|---|
| 低层条件与尾部 | 三方法均State＋Subtask截至分号，不含整体task或Action/EOS/FAST。 | 保留整体task/state/subtask；hybrid移除action区域；该快照flow-only tokenizer分支仍添加Action/EOS，属于当前参考差异，未移植到ours。 | [ours](D:/pi05_subtask/ours/code/soma-fluxvla-main/fluxvla/training/subtask/config.py:78)；[kewang](D:/pi05_subtask/kewang/code/openpi_subtask-main/src/openpi/models/tokenizer.py:216)；[kewang](D:/pi05_subtask/kewang/code/openpi_subtask-main/src/openpi/models/pi05.py:273) |
| 分段tokenization | condition与subtask分开编码；prepare/train/evaluate共用canonical构造。 | tokenize_high_low_prompt分开编码condition、subtask与FAST片段。 | [ours](D:/pi05_subtask/ours/code/soma-fluxvla-main/fluxvla/transforms/subtask_fast_prompts.py:253)；[kewang](D:/pi05_subtask/kewang/code/openpi_subtask-main/src/openpi/models/tokenizer.py:216) |
| 文本／图像attention | 高层图像／条件文本full-prefix、输出causal；低层图像／state／subtask full-prefix。 | embed_prefix图像ar_mask=False、语言True，形成图像双向／文本causal。 | [ours](D:/pi05_subtask/ours/code/soma-fluxvla-main/fluxvla/models/vlas/pi05_subtask.py:97)；[kewang](D:/pi05_subtask/kewang/code/openpi_subtask-main/src/openpi/models/pi05.py:135) |
| 训练动作条件与答案隔离 | 低层使用当前样本的真值subtask文本；高层保留完整FAST targets，但低层泄漏守卫禁止FAST。 | 训练使用tokenized真值subtask；hybrid另建专用flow prefix，剔除FAST真实答案。 | [ours](D:/pi05_subtask/ours/code/soma-fluxvla-main/fluxvla/training/subtask/fixtures.py:97)；[ours](D:/pi05_subtask/ours/code/soma-fluxvla-main/fluxvla/models/vlas/pi05_subtask.py:105)；[kewang](D:/pi05_subtask/kewang/code/openpi_subtask-main/src/openpi/models/pi05.py:273) |
| 推理KV与文本重建 | 新契约强制从生成subtask文本重建task-free低层prefix，拒绝generation KV；旧bundle单独兼容。 | sample_low_level_task返回生成后的KV，再供动作路径使用；不等价于重新编码文本。 | [ours](D:/pi05_subtask/ours/code/soma-fluxvla-main/fluxvla/inference/subtask.py:191)；[kewang](D:/pi05_subtask/kewang/code/openpi_subtask-main/src/openpi/models/pi05.py:355) |
| 动作窗口 | 下一帧t+1起，固定50步，跨subtask边界；episode尾重复但flow不监督。 | loader时间偏移从0开始，按model.action_horizon取固定动作序列；不是按标签边界截断。 | [ours](D:/pi05_subtask/ours/code/soma-fluxvla-main/fluxvla/training/subtask/common.py:47)；[kewang](D:/pi05_subtask/kewang/code/openpi_subtask-main/src/openpi/training/data_loader.py:143) |
| 采样／质量过滤 | 每个subtask_valid有效anchor一次，不再有stride。 | 标准dataset loader按样本读取；未等同采用ours的subtask_valid过滤政策。 | [ours](D:/pi05_subtask/ours/code/soma-fluxvla-main/fluxvla/training/subtask/common.py:43)；[kewang](D:/pi05_subtask/kewang/code/openpi_subtask-main/src/openpi/training/data_loader.py:131) |
| hybrid初始化与梯度 | 直接base联合8epochs；Ltoken+λLa，无detach，VLM由CE和flow更新。 | hybrid配置直接加载base，.15/.15/1，40ksteps；不是先FAST Stage1。 | [ours](D:/pi05_subtask/ours/code/soma-fluxvla-main/fluxvla/training/subtask/config.py:25)；[ours](D:/pi05_subtask/ours/code/soma-fluxvla-main/fluxvla/training/presets/pi05_subtask_fast.py:77)；[kewang](D:/pi05_subtask/kewang/code/openpi_subtask-main/src/openpi/training/config.py:755) |
| hybrid_insulated | FAST Stage1后联合8epochs；保留三损失，prefix KV detach，VLM仍由CE训练。 | 该快照没有与ours同名且完整一致的独立受控配方；detach能力存在于模型。 | [ours](D:/pi05_subtask/ours/code/soma-fluxvla-main/fluxvla/training/presets/pi05_subtask_fast.py:78)；[kewang](D:/pi05_subtask/kewang/code/openpi_subtask-main/src/openpi/models/pi05.py:321) |
| two_stage与freeze | 4epoch FAST适配后8epoch expert；Stage1冻结动作分支；expert阶段冻结vision/language/projector并detach。 | FAST配置预算20k、expert预算8k，但expert的weight_loader实际指向FAST第12k步checkpoint；expert detach，freeze_filter匹配语言主干并排除expert，未显式将vision纳入同一冻结过滤器。 | [ours](D:/pi05_subtask/ours/code/soma-fluxvla-main/fluxvla/training/subtask/train.py:24)；[kewang](D:/pi05_subtask/kewang/code/openpi_subtask-main/src/openpi/training/config.py:675)；[kewang](D:/pi05_subtask/kewang/code/openpi_subtask-main/src/openpi/training/config.py:748) |
| CE归一化 | 本rank/microbatch将有效subtask与FAST token合并归一化；累积microbatch等权、DDP平均梯度。原始分项CE仅作诊断。 | 每样本按对应有效token数归一化，训练器再对样本loss平均。 | [ours](D:/pi05_subtask/ours/code/soma-fluxvla-main/fluxvla/models/vlas/pi05_subtask.py:154)；[kewang](D:/pi05_subtask/kewang/code/openpi_subtask-main/src/openpi/models/pi05.py:245)；[kewang](D:/pi05_subtask/kewang/code/openpi_subtask-main/scripts/train.py:150) |
| flow维度与归一化 | 本rank有效元素均值；仅14维，尾部与RTC延迟位置不计MSE。 | compute_loss默认real_action_dim=32；train调用未覆盖该参数，因此默认包含补齐维度，不能将其MSE与ours直接横比。 | [ours](D:/pi05_subtask/ours/code/soma-fluxvla-main/fluxvla/models/vlas/pi05_subtask.py:214)；[ours](D:/pi05_subtask/ours/code/soma-fluxvla-main/fluxvla/training/subtask/prepare.py:251)；[kewang](D:/pi05_subtask/kewang/code/openpi_subtask-main/src/openpi/models/pi05.py:208)；[kewang](D:/pi05_subtask/kewang/code/openpi_subtask-main/scripts/train.py:150) |
| FAST编码与超长 | 离线缓存完整固定窗口FAST；roundtrip检查；超budget报错，不静默截断。 | transform在线调用tokenizer产生FAST，后续按max_len处理。即便BPE相同，也不能假定输入归一化或接受样本完全相同。 | [ours](D:/pi05_subtask/ours/code/soma-fluxvla-main/fluxvla/training/subtask/prepare.py:176)；[kewang](D:/pi05_subtask/kewang/code/openpi_subtask-main/src/openpi/models/tokenizer.py:254)；[kewang](D:/pi05_subtask/kewang/code/openpi_subtask-main/src/openpi/transforms.py:286) |
| 动作表示与统计 | Piper12关节对anchor取delta、2夹爪绝对值；train-only差分目标窗口q01/q99。 | LIBERO控制量及配置资产统计，启用quantile normalization；不应把Piper关节delta当作同一物理目标。 | [ours](D:/pi05_subtask/ours/code/soma-fluxvla-main/fluxvla/training/subtask/common.py:47)；[ours](D:/pi05_subtask/ours/code/soma-fluxvla-main/fluxvla/training/subtask/prepare.py:119)；[kewang](D:/pi05_subtask/kewang/code/openpi_subtask-main/src/openpi/training/config.py:729) |
| 数据规模与split | dataset_goal_text_v1，115episodes／233464frames，固定89/11/15；train拟合，test隔离。 | 本地LIBERO快照500episodes／138090frames／10tasks，info声明0:500全为train；不是ours固定holdout。 | [ours元数据](D:/pi05_subtask/ours/dataset_goal_text_v1/meta/info.json:1)；[kewang元数据](D:/pi05_subtask/kewang/dataset/libero_10_subtasks/meta/info.json:1) |
| 本体／相机／horizon | 14state、14action、前／左腕／右腕3视角；50步@30Hz。 | 8state、7action、外部／腕部2视角；hybrid20步、FAST/expert25步@10Hz。 | [ours元数据](D:/pi05_subtask/ours/dataset_goal_text_v1/meta/info.json:1)；[kewang元数据](D:/pi05_subtask/kewang/dataset/libero_10_subtasks/meta/info.json:16)；[kewang](D:/pi05_subtask/kewang/code/openpi_subtask-main/src/openpi/training/config.py:760) |
| 优化器／batch／LR | AdamW，全局batch64；Stage1峰值1e-4，联合／expert7.5e-5；按epoch真实样本数产生更新，阶段切换重置optimizer/schedule。 | AdamW配置；峰值2.5e-5，FAST/expert batch512、hybrid64；按20k/8k/40k steps配方，不能据同epoch数量比较预算。 | [ours](D:/pi05_subtask/ours/code/soma-fluxvla-main/fluxvla/training/subtask/train.py:37)；[ours](D:/pi05_subtask/ours/code/soma-fluxvla-main/configs/subtask/common.json:1)；[kewang](D:/pi05_subtask/kewang/code/openpi_subtask-main/src/openpi/training/config.py:557)；[kewang](D:/pi05_subtask/kewang/code/openpi_subtask-main/src/openpi/training/config.py:735)；[kewang](D:/pi05_subtask/kewang/code/openpi_subtask-main/src/openpi/training/config.py:779) |
| 框架／精度 | PyTorch DDP＋梯度累积；新运行模型和attention统一BF16；CPU smoke FP32。 | JAX/Flax主训练，model dtype默认BF16；同时存在PyTorch精度配置字段；相同名称不保证attention数值实现一致。 | [ours](D:/pi05_subtask/ours/code/soma-fluxvla-main/fluxvla/training/subtask/fixtures.py:130)；[ours](D:/pi05_subtask/ours/code/soma-fluxvla-main/configs/subtask/runtime.json:1)；[kewang](D:/pi05_subtask/kewang/code/openpi_subtask-main/src/openpi/models/pi0_config.py:20)；[kewang](D:/pi05_subtask/kewang/code/openpi_subtask-main/src/openpi/training/config.py:555) |
| EMA／增强 | 专用受控路径无EMA、无随机augmentation；显式拒绝开启；缓存仅resize。 | 示例三配方ema=.999；train预处理有crop/rotate/color jitter（腕部不做外部相机crop/rotate）。 | [ours](D:/pi05_subtask/ours/code/soma-fluxvla-main/fluxvla/training/subtask/config.py:25)；[ours](D:/pi05_subtask/ours/code/soma-fluxvla-main/fluxvla/training/subtask/prepare.py:144)；[kewang](D:/pi05_subtask/kewang/code/openpi_subtask-main/src/openpi/training/config.py:789)；[kewang](D:/pi05_subtask/kewang/code/openpi_subtask-main/src/openpi/models/model.py:175) |
| 独立评估与随机性 | 始终计算三项无权重指标；验证sample ID/seed固定noise和time，二者component generator独立；不消耗训练RNG。 | compute_loss按启用loss分支计算，未见与ours同一受控三指标独立验证器；此处不据训练loss推断闭环成功率。 | [ours](D:/pi05_subtask/ours/code/soma-fluxvla-main/fluxvla/training/subtask/train.py:81)；[ours](D:/pi05_subtask/ours/code/soma-fluxvla-main/fluxvla/training/subtask/randomness.py:5)；[kewang](D:/pi05_subtask/kewang/code/openpi_subtask-main/src/openpi/models/pi05.py:219) |
| 选模／导出／恢复 | 两个分阶段方法共享Stage1按ep3/4 subtaskCE选择；联合／expert最终step主评估；bundle含模型、tokenizer、normalization、prompt契约及hash；Stage1 receipt固定权重身份；Stage2每750步导出权重，仅最终步保存optimizer，中途需显式warm-start。 | 参考训练配置按save_interval保存，推理走自己的policy/model资产路径；没有自动共享ours的缓存身份、固定split与bundle契约。 | [ours](D:/pi05_subtask/ours/code/soma-fluxvla-main/fluxvla/training/subtask/run.py:82)；[ours](D:/pi05_subtask/ours/code/soma-fluxvla-main/fluxvla/training/subtask/checkpoints.py:108)；[kewang](D:/pi05_subtask/kewang/code/openpi_subtask-main/src/openpi/training/config.py:741) |

前向答案隔离与反向梯度隔离必须分别看待：禁止action expert读取真实FAST答案是前向约束，detach prefix KV是反向约束，freeze参数又是优化器约束。一个成立不意味着另外两个自动成立。尤其two_stage的视觉冻结范围与参考不同；不能只按方法名宣称数值等价。

### 5.2 三种方法与官方π0.5的通用对照：已对齐项与剩余差异

本节仅比较我们的实现与官方信息，kewang对照独立保留在5.1。先分析三种方法共有的条件构造、attention、表示与监督差异，再在5.3分析各方法的损失、梯度和冻结策略。阶段数量、初始化来源及由具体数据规模决定的训练预算不作为本节的差异维度。

| 维度 | 官方π0.5论文 | 三种方法的共同实现及差异 | 官方依据 |
|---|---|---|---|
| 高低层条件 | 低层动作分布依赖observation与预测subtask，不依赖整体task。 | 已对齐：低层仅使用图像、state与subtask，并重新编码自己的prefix；整体task只进入高层，禁止高层KV复用。 | [论文原文](D:/pi05_subtask/pi05/pi05.txt:439) |
| attention与答案隔离 | 图像、prompt和离散state使用full-prefix；FAST自回归，action expert不能读取FAST答案。 | 已对齐：条件prefix内部双向，subtask／FAST输出保持causal；低层subtask为已知条件，纳入full-prefix，连续动作与FAST答案隔离。 | [论文原文](D:/pi05_subtask/pi05/pi05.txt:1951) |
| 状态／动作表示 | 状态离散为文本，动作包含FAST与连续flow两种表示；跨本体动作向量按最大空间padding。 | 均具备离散状态、FAST标签与flow构件；Piper适配为12关节delta、2夹爪绝对值，flow仅监督14物理维。本体适配本身不代表算法偏离，论文的padding描述也不足以推出相同的loss分母。本项保留现有本体适配，不为形式一致而监督补齐维度；各方法是否启用FAST训练损失另见5.3。 | [状态与动作表示](D:/pi05_subtask/pi05/pi05.txt:494)；[padding](D:/pi05_subtask/pi05/pi05.txt:670) |
| 监督类型覆盖 | 结合机器人、高层任务与web数据，高层数据含bbox，另加入语言遥操作指令示范。 | 当前本地训练均使用任务／subtask文本与动作监督，没有覆盖官方完整的web、bbox及语言遥操作监督流程。这里比较监督类型，不以episode数量或训练步数判定方法差异。 | [数据组成](D:/pi05_subtask/pi05/pi05.txt:604)；[高层监督](D:/pi05_subtask/pi05/pi05.txt:652)；[语言指令示范](D:/pi05_subtask/pi05/pi05.txt:683) |
| 评估证据 | 报告跨环境任务、长期任务及语言跟随的闭环行为。 | 当前三方法的文档证据均为CPU实现测试及离线三指标评估能力，尚无最终test-set／真机成功率证据。这是验证覆盖的差异，不能据此推断方法性能高低。 | [论文原文](D:/pi05_subtask/pi05/pi05.txt:1887) |

前向答案隔离与反向梯度隔离是不同约束。原始π0.5论文的attention说明不能单独用来判定是否应detach全部prefix KV或冻结VLM。反向梯度机制另以官方后续[π0.5＋Knowledge Insulation介绍](https://www.pi.website/research/knowledge_insulation)为明确参照：VLM继续接受FAST及语言监督，action expert学习连续动作，同时阻断其梯度回传VLM。下面分别标明原始论文与后续KI的比较依据。

### 5.3 各方法的特异性差异

以下集中比较联合／动作训练时的目标与参数更新机制；5.2中的共同差异不再重复。Ls、Lf、La分别为本实现独立报告的subtask CE、FAST CE与flow MSE；训练CE采用统一Ltoken。共享Stage1可按目标类型对照官方离散token预训练，两个joint阶段均可对照联合后训练；这不是完整数据、预算或数值配方的复现，也不是hybrid与hybrid_insulated分别对应预训练和后训练。two_stage的动作阶段保留为冻结VLM的对照。

#### 5.3.1 hybrid

| 维度 | 我们的实现 | 与官方信息的比较 |
|---|---|---|
| 训练目标 | Ltoken＋λLa，同时保留文本、FAST与连续动作监督。 | 与原始π0.5联合next-token prediction＋flow的目标形式相似；已改为统一有效token CE，但flow reduction及监督覆盖仍不同，λ不能视为论文Eq.1中α=10的数值等价。[论文原文](D:/pi05_subtask/pi05/pi05.txt:683) |
| 梯度路径 | prefix KV不detach，VLM同时接收CE与flow梯度。 | 与后续官方KI明确阻断action expert到VLM梯度的机制不同；hybrid是无梯度隔离的联合训练。 |
| VLM更新 | 视觉、语言与projector参与联合优化。 | 与KI一样保持VLM可训练，但其更新信号包含flow；“VLM可训练”并不足以说明采用了KI。 |

hybrid的特异性在于保留联合监督且允许flow更新VLM；与KI的核心差别是反向梯度路径。

#### 5.3.2 hybrid_insulated

| 维度 | 我们的实现 | 与官方信息的比较 |
|---|---|---|
| 训练目标 | 联合训练使用Ltoken＋λLa，持续保留CE与flow监督。 | 与原始π0.5的联合目标形式相似，也符合后续KI持续以离散动作／语言监督训练VLM的机制；本地损失归一化和权重不能直接等价于论文α=10。[论文原文](D:/pi05_subtask/pi05/pi05.txt:683) |
| 梯度路径 | detach prefix KV，阻断flow到VLM的梯度。 | 与后续官方KI的核心反向隔离机制相符；这一判断来自KI说明，不能仅从原始论文的前向attention mask推导。 |
| VLM更新 | VLM继续由CE更新，action expert由flow更新。 | 与KI强调的“隔离flow梯度，同时继续学习VLM表征”相符；不属于冻结VLM。 |

hybrid_insulated的联合训练机制与KI相符；5.2所列条件构造与attention已对齐，监督覆盖差异仍然存在，机制相符不代表所有实现细节相同。

#### 5.3.3 two_stage

| 维度 | 我们的实现 | 与官方信息的比较 |
|---|---|---|
| 训练目标 | 动作专家训练时仅La，subtask CE和FAST CE权重均为0。 | 与原始π0.5继续next-token prediction＋flow的联合目标不同，也缺少KI要求的持续离散动作／语言监督。[论文原文](D:/pi05_subtask/pi05/pi05.txt:683) |
| 梯度路径 | detach prefix KV，阻断flow到VLM的梯度。 | 单看这一约束与KI相符，但不能据此认定实现了完整KI机制。 |
| VLM更新 | 冻结vision、language与projector，仅训练动作分支。 | 与KI保持VLM通过CE继续学习的机制不同；这是冻结VLM的动作专家训练。独立验证计算CE，不代表训练时仍优化CE。 |

two_stage的特异性在于动作训练时停止CE并冻结VLM；这与hybrid_insulated“隔离flow但持续训练VLM”的机制有实质区别。

### 5.4 与本地普通π0.5训练路径对齐的六项公共行为

本节的“普通π0.5”指本仓库已有的普通训练路径，不表示下述参数均来自官方论文。按既定公共实现，三方法均采用5% warmup（向下取整、首更新LR=0）、cosine目标下限为0（使用update−1，最后一次实际更新LR仍略大于0）；复用AdamW参数分组（bias及一维参数不衰减）；RTC开启（exponential、max_delay=7不含上界、temperature=1）；窗口从t+1开始；尾部重复动作但flow屏蔽；训练loss为本rank/microbatch有效元素均值，累积等权后由DDP平均梯度。验证仍独立按实际有效计数汇总。

因此窗口起点与尾部监督现在不再沿用kewang的当前帧策略。三种方法的阶段及梯度隔离定义保留，CE目标已改为统一token归一化；正确Beta时间采样也保留。FAST编码完整补齐窗口，不额外引入变长FAST目标。不满足当前提示／窗口契约的旧缓存必须重建。当前窗口契约为`next-frame-fixed-horizon-masked-tail-v3`，缓存版本为`controlled-goal-text-paper-v3`，READY另记录`pi05-subtask-paper-v1`提示契约。默认cache为`cache/goal_text_v1_paper_prefix`，共享Stage1为`work_dirs/shared_goal_text_v1_paper_prefix_token_ce_v1`，三方法输出目录同时带`paper_prefix_token_ce_v1`，避免混用旧产物。损失目标版本为`unified-token-ce-v1`，纳入Stage1身份及部署元数据；本次仅改变损失，可复用已满足paper提示契约的缓存，无需重新做FAST编码。

### 5.5 如何查看Ltoken内部的Ls与Lf

CE部分只用Ltoken参与训练，但每阶段的`metrics.jsonl`保留分项指标，验证写入`validation.jsonl`，独立evaluate也返回相同诊断字段。

| 关注点 | 日志字段 | 含义 |
|---|---|---|
| 分项难度 | `subtask_ce`、`fast_ce` | 分别按有效token数平均的Ls、Lf |
| 监督数量 | `subtask_token_count`、`fast_token_count` | Ns、Nf；排除padding及无效样本 |
| token数量份额 | `subtask_token_fraction`、`fast_token_fraction` | Ns/(Ns+Nf)、Nf/(Ns+Nf) |
| 全局池化CE贡献 | `token_subtask_contribution`、`token_fast_contribution` | Cs=Ns·Ls/(Ns+Nf)、Cf=Nf·Lf/(Ns+Nf)，二者相加等于`token_ce` |
| 全局池化loss份额 | `subtask_loss_fraction`、`fast_loss_fraction` | Cs/(Cs+Cf)、Cf/(Cs+Cf) |
| 实际优化CE及分项 | `optimization_token_ce`、`optimization_token_subtask_contribution`、`optimization_token_fast_contribution` | 按训练采用的microbatch等权及DDP rank平均计算；两项贡献相加等于优化CE |
| 实际优化loss份额 | `optimization_subtask_loss_fraction`、`optimization_fast_loss_fraction` | 上一行分项贡献各自除以优化CE；要问训练目标内部的占比，优先查看这两个字段 |

前五行是跨当前更新的所有microbatch和rank汇总真实NLL和计数的诊断（验证则覆盖整个validation）；最后两行对应实际优化权重。不同microbatch有效token数不等时，两种口径通常不相等。日志以`metric_scope`和`optimization_scope`明确区分，不用全局token池化偷偷改变训练梯度。

例如Ns=20、Nf=80、Ls=2、Lf=1，则Ltoken=1.2，Cs=0.4、Cf=0.8：token数量份额是20%/80%，loss份额是33.3%/66.7%。这些都是损失值的分解，不是参数梯度影响或任务重要性的份额；无需增加第二次反向传播。无监督或零总CE导致未定义的比例记为`null`。two_stage动作训练不执行CE，因此训练日志CE标为未计算，而验证仍真实报告两项CE。

## 6. 最终独立验收与限制

2026-10-22由三个子Agent分别实现统一token目标、训练诊断及配置／产物契约，主Agent独立审查并执行验收。此前已实现的低层task隔离和full-prefix attention继续纳入回归；本次保留三方法的梯度隔离、冻结及Piper动作表示，改变CE训练目标。

合并回归为 **186 passed，1 skipped（51.05s）**，见[合并验收JUnit](D:/pi05_subtask/audits/unified_token_20261022/acceptance.xml)。主Agent独立构造的验收文件另行完整执行为 **8 passed（28.65s）**，见[独立验收JUnit](D:/pi05_subtask/audits/unified_token_20261022/independent_acceptance.xml)；其中5项与合并回归重叠，不应简单相加。三方法配置及系数见[配置核验](D:/pi05_subtask/audits/unified_token_20261022/config_verification.json)。

独立验收覆盖：

- 从真实模型logits手算所有有效token的CE，验证统一损失及梯度；覆盖不同subtask／FAST长度、sample weight、无效样本和空监督。
- 两个CPU/Gloo进程实际执行DDP及no_sync梯度累积，包含一个空microbatch；与独立逐microbatch损失梯度计算一致。
- 同一双进程测试核对跨rank汇总的真实NLL／count、全局token池化诊断及实际优化口径贡献／份额，避免rank0局部指标冒充全局指标。
- 新统一目标下，hybrid允许flow更新VLM；hybrid_insulated隔离flow但CE仍更新VLM；two_stage冻结视觉／语言／projector，仅训练动作分支。额外使用λ=2.5核对joint组合公式。
- 有效监督token必须被两个互斥region完整覆盖；拒绝新统一目标混入旧独立CE系数。旧Stage1身份、目标不一致的导出和不同λ的同组比较被拒绝。
- 真实模型／tokenizer导出和重载、低层task不变性、full-prefix答案隔离、旧bundle兼容及此前提示契约门禁继续通过回归。

跳过项依赖当前环境未安装的soma_deploy。合并测试的65条warning属于yapf/lib2to3弃用、preset模块重复runpy加载及已有BF16 cache/recompute差异提示；没有将这些提示当作数值一致性证明。独立验收仅有yapf弃用warning。Windows序列化测试仅替换测试内的POSIX持久化钩子，未降低生产Linux持久化保证。

本轮未运行完整GPU训练、Linux持久化事务／final-only集成测试、最终test-set评估或真机rollout。已有paper提示契约缓存可复用；更早的旧提示缓存仍需重建。新目标必须重新训练共享Stage1，旧checkpoint不能视为使用统一token目标训练。第2节数据数量沿用本次会话前一轮实际重扫结果；本次没有重新扫描生产数据或启动训练。
