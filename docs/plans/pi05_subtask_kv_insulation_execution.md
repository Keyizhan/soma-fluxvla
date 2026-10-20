> 历史实验计划，保留研究过程。当前入口与已落实的条件语义见 [subtask_workflow.md](../subtask_workflow.md)。

# π0.5 梯度隔离与 KV 缓存实验

2026-10-17。用户明确授权对齐 Ke-Wang 的梯度隔离、文本 KV、动作 prefix KV 并运行一组实验。实现和实验设计为AI工作；参考社区 commit `385d0e8330397ecfa29e9b675042b424c4d5386c`，不称为官方内部实现。

## 固定范围

- 保留高层 p(s|o,g)、低层 p(a|o,s)；不引入总体任务到低层，不复用高层融合KV。
- 借鉴 `gemma_05.py` 的位置/K/V、预留容量、增量写入和动作只读缓存，以及 flow prefix stop-gradient。
- 保留实际 valid mask 表达相机/PAD空洞、逐样本EOS；显式区分append/read，不按token长度猜模式。
- 缓存仅存在单次请求内；观察或subtask变化时重新prefill，不做跨请求命中。
- 不增加鞋子数据，不改FAST、RTC和正式训练器；完整续训另列未完成。

## 验证与预先声明

1. 微型模型：缓存/无缓存logits、动作和有效梯度；隔离后flow的视觉/语言梯度为零，CE仍可更新；不同EOS、相机及PAD空洞、horizon=1、缓存长度不增长、子任务变化刷新。
2. 完整公开基座：FP32和BF16下同输入/噪声的logits、8-token固定预算生成、flow预测、10步采样对齐；完整梯度覆盖和确定性梯度采样对照。
3. 未通过完整正确性不启动训练组。预设FP32 rtol=2e-4/atol=5e-5；BF16 rtol=.03/atol=.125，同时单列实际最大差值并要求生成token一致。BF16阈值较宽，不以“通过”代替数值报告。梯度数值比较每参数最多1024个确定性元素，相对L2≤.03；微型测试比较全部梯度元素。

## 三组实验

| 组 | 文本缓存 | 动作prefix缓存 | flow独立prefix | flow梯度隔离 |
|---|---|---|---|---|
| baseline | 否 | 否 | 否，原组合路径 | 否 |
| cached | 是 | 是 | 是 | 否 |
| insulated | 是 | 是 | 是 | 是 |

缓存前后主要比较数值等价和计算成本；cached/insulated比较梯度隔离。不会将微小浮点差异下长期优化轨迹视为必须逐位一致。

全部使用同一公开基座、种子1704、120次AdamW更新、lr1e-5、权重1:1、microbatch1累积2、FP32参数+BF16 autocast、真实32维架构动作。视觉编码器冻结，语言/projector/expert按代码参与训练；正确性梯度检查另行解冻视觉以验证路径。噪声独立CUDA generator seed1705；真实Beta时间由NumPy seed1706预生成，绕过原近似helper。

输入为三个二元因素（程序颜色图像、状态正负、normal/reversed规则指令），子任务由三者XOR决定为move left/right；8种组合中6训练、2保留组合验证，类别均衡。规则动作同时由状态和子任务决定，包含50步×32维。这是AI构造的条件学习探针，没有机器人动作语义，也不用于报告机器人性能。固定种子、单次小实验，不声称显著性。

记录分项学习曲线、完整文本/EOS、合成动作MSE、同观察改subtask的动作变化；记录显存、训练耗时、预热与6次稳定推理的prefill/每token/动作prefix/去噪延迟。benchmark强制8-token预算时显式关闭EOS停止；正常准确率评估仍按真实EOS。

## 产物

入口：`scripts/architecture/run_pi05_kv_experiment.py`。

本地证据：`experiments/pi05_subtask_architecture/kv_insulation_20261017/`。

远端：`/home/ubuntu/2xPro6000/experiments/pi05_subtask_architecture/kv_insulation_20261017/`，独立源码快照、运行日志和模型参数delta。参数delta不含optimizer/RNG，不标记P12精确续训完成。

状态：本地/远端26项测试通过；约定精度完整模型验证、三组120步训练完成；用户已要求停止后续测试；baseline/cached重载完成，insulated重载未完成。


## 实测后的协议修订 v3（不覆盖失败）

第一次完整模型FP32验证通过，BF16文本logits未通过预先声明容差。修正了文本路径RoPE计算dtype与原语言路径不一致的问题；逐层检查证明修正后的prefill隐藏状态可完全一致，但BF16增量生成与整段重算仍在第三个token发生argmax差异。原失败日志保留在correctness、correctness_v2，未放宽其阈值，也不标记BF16文本通过。

为隔离本轮研究变量，新增显式`text_decode_fp32=True`，三组文本推理（含图像prefill）统一禁用autocast，训练CE/flow和动作推理仍使用BF16 autocast。完整模型重新验收这一明确的精度组合后再训练。FP32文本缓存/无缓存对齐不意味着BF16自由解码已验收。v3源码独立归档，初始source_manifest另存保留。

模型沿用基座构造时的少量norm参数dtype，不强制改权重精度；报告记录各dtype参数数目。FP32是主体参数及文本计算模式的简称。微型模型全部梯度元素对照通过；完整模型采用每参数采样检查并保留全部参数梯度覆盖清单。


本轮低层目标只依赖state与subtask，训练覆盖这两个因素的四种组合；保留输入仍包含未见的图像/state/subtask组合。动作MSE不等于全新动作规则泛化。三组训练时可能共享宿主机CPU/NFS，训练耗时仅作实测描述；主要推理性能对照另起新进程，在全部训练退出后同一GPU顺序测量，每组2次预热+6次记录。


用户后续纠正：知识隔离作为论文复现约束保留；本轮缺FAST动作CE，不是完整KI。高层推理固定标准增量AR/KV链路。后续测试已停止，不自动补跑。


## 用户明确补充的目标架构：FAST动作监督为必做

本节是用户后续指定的架构要求，取代本轮小实验中FAST暂缓的范围；历史运行未使用FAST这一事实保持不变。停止进一步测试的指令仍有效。

目标包含三项独立记录的监督：

1. 高层subtask CE：输入观察o和总体任务g，预测子任务s。
2. FAST动作CE：在既定低层p(a|o,s)合同下，观察o和标注子任务s作为条件，将真实连续动作经FAST tokenizer编码为目标token；条件区双向，FAST目标因果。复用经过核验的绑权词表投影，不默认新建随机大head。
3. Flow动作MSE：action expert读取仅由o/s建立的低层prefix KV；对该KV执行stop-gradient，阻断flow到视觉/语言主干的梯度，保留expert及动作/时间投影的训练。

总目标为lambda_subtask * CE_subtask + lambda_FAST * CE_FAST + lambda_flow * MSE_flow。权重尚未指定，不将旧小实验1:1或社区0.15/0.15/1当作官方结论。

FAST监督必须落实tokenizer版本/动作归一化/真实维度/token映射及特殊ID/截断策略/独立labels与区域mask。不能只在loss函数里增加一个名字，也不能把普通子任务文本CE当成动作token监督。

Flow路径不能读取标注FAST动作token，包括不能读取已融合这些token的prefix缓存；应从干净的o/s条件构建KV。高层含g的融合KV也不进入本方案低层。FAST CE和flow的输入/梯度边界须分别表达。

推理时高层标准AR生成子任务并消费/追加高层KV；随后为o/s独立建立低层KV，flow多步只读生成连续动作。连续动作部署路径不要求再生成整段FAST动作token；FAST在这里承担训练时动作表示监督。

当前实现进度：subtask CE、flow、flow prefix stop-gradient、文本KV、动作prefix KV已有代码；FAST tokenizer接入、动作token训练数据、FAST CE及其防泄漏串接尚未实现，完整KI未完成。当前代码配置是已有实验配置，不冒充完整目标配方。

论文依据：[KI官方说明](https://www.pi.website/research/knowledge_insulation)、[论文§5.1–5.2](https://www.pi.website/download/pi05_KI.pdf)。低层只读o/s和具体工程拆分沿用本项目合同，属于本项目实现选择，不声称作者完整内部subtask实现已公开。
