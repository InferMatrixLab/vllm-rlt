# vLLM-RLT 插件化开发记录

- 工作分支为 `vllm-rlt-plugin2`，起点为主仓 `4821d47`。
- 设计目标为 vLLM `0.34.0rc1.dev1149+g3146a9df4`，本地实际验证使用同一 Git revision `3146a9df4140`。该构建环境的生成版本标签为 `0.26.1rc1.dev6849+g3146a9df4`。
- 每轮记录本轮范围、复现命令、实际结果和待核实项。改动在当前工作区留为 unstaged，供 VS Code 审阅。
- [Loop 调度策略规格](#loop-调度策略规格) 记录三阶段队列、优先级、合批规则和验收例子。

## 阶段一 功能实现

每个开发 loop 负责总图的一部分。先写可观察的通过条件，再实现、跑最小用例、定位失败并复测。共享契约可以用 CPU 替身先验收，真实模型达到对应接入点后再跑端到端回归。

| Loop | 范围 | 验收 |
|---|---|---|
| 1 | 引擎进程和接入契约 | 插件可发现，EngineCore 创建 `LoopRequest`，逐 op 计划和回执能走同步和异步入口。 |
| 2 | 调度 | `LoopScheduler` 实现上游接口，`LoopAsyncScheduler` 继承它。验收选批、span 进度、迟到回执和取消。 |
| 3 | KV | 复用 vLLM 分配和块表，验收深度隔离、发布条件及早退补写。 |
| 4 | Worker | vLLM 根据配置装配插件 worker 和 runner。 |
| 5 | ModelRunner | 逐 span 铺行，hidden 跨轮留在设备，只有到采样边界的行交付 token。 |
| 6 | 模型 | 先接 Inkling-2-Mini VarLoop，再用 Ouro 验证真实 gate 早退。 |
| 7 | 执行器 | 逐 op 计划和结果通过 vLLM 执行器跨进程交接。 |
| 8 | 服务入口 | `vllm serve` 真实请求验证固定深度、请求级 1/3/默认轮数及混深度。 |
| 9 | 投机 | 验证浅层 draft、深层 target、接受和拒绝回滚。 |
| 10 | 算子和并行 | 复用首批模型所需的 vLLM 算子，验证 DP op 选择和 dummy 接口。 |

Inkling 的固定深度对照先用 W1 C1。请求级轮数用同一实例收 `rlt_target_loops=1`、`3` 和省略参数的请求，按请求和 token 位置统计实际 CORE 次数。混深度另用确定的调度顺序触发同一个 CORE 批中的不同深度。首轮按已确认的 decode 口径验收，Inkling prefill 和首枚输出 token 的轮数在模型接入时单列验证。Ouro 沿用满深度 prefill，后续 decode 对照参考 gate 判定、退出轮 hidden、`LAST_EXITED` 补写和下一 token 的读取。GSM8K 阶段回归沿用固定题集，记录正确率和平均 decode 轮数。

## 阶段二 性能优化

### 基本优化

依次优化异步调度，CUDA Graph 与 mixed batch、chunked prefill、packed prefill，以及 Loop Scheduler 和 KV 管理。正式对照双方使用同一 Inkling VarLoop checkpoint、固定三轮、同卡、同一请求回放、相同精度和可用 KV 容量。先预热，关闭 profiler 做配对测量。W1 C1 用于快速检查，W3 C16 和 W7 C32 用于代表性对照，W1–W7 用于阶段收尾。

基本优化全部打开后，输出吞吐 `rlt / vLLM` 和 TTFT P95 `vLLM / rlt` 各至少达到 `0.95`，目标为 `0.97`，进一步目标为 `0.98` 以上。记录 TTFT P50、TPOT 和 ITL，用于解释差异。Ouro 的验收以早退正确性为准。

### Loop 特性

固定已通过的 rlt 基本优化配置，逐项对照 WCPB、自投机，以及在自投机上的 Decode pipeline。Coda 合批先用 trace 判断瓶颈，再决定是否做开关对照。增益需高于同条件的基线波动，并能复测。最后用组合配置回放代表负载，与 rlt 基本优化组和原生 vLLM 分别对照。

### W1–W7 与请求级深度

保留标准 W1–W7 作为固定三轮的原生对照。W1/W2 是 128→128、C1/C32，W3/W4 是
4096→128、C1/C32，W5 是 1024→512、C8，W6/W7 是相同的 ShareGPT 请求集、自然
EOS 且最多输出 512、C8/C32。前面迭代用的 `W3 C16` 是长 prompt 的诊断档，报告上
单独注明并发。原生和 RLT 同卡配对、同输入顺序、同精度、同实际 KV 容量、同 token
预算和 EOS 策略。做完整 warmup 后关 trace/profiler 跑正式波。

请求级深度作为额外维度。先用 W1 输入做服务验收，同实例并发发送显式 1、2、3 和
省略参数的请求，校验 prefill 实际三轮、首枚 token 一致、后续 decode 每个位置
分别执行 1、2、3、3 次 CORE。请求通过 `vllm_xargs.rlt_target_loops` 指定。在当前
Inkling-2-Mini VarLoop 上，prefill 和首枚 token 固定使用服务配置的三轮，缩短深度
从第二枚输出 token 开始体现，所以 TTFT 用于看排队和合批影响，TPOT/ITL 与实际深度
分布用于看 decode 变化。

正式矩阵给每个负载加一组可变深度回放，复用原 W 编号和负载定义。

| 负载 | 输入 / 输出与并发 | 混深度验证重点 |
|---|---|---|
| W1 可变深度 | 128→128，C1 | 单请求短输入，按深度测延迟和吞吐，给并发组提供参考。 |
| W2 混深度 | 128→128，C32 | 短 prompt、高并发 decode；验证真实跨深度 CORE 合批和总吞吐。 |
| W3 可变深度 | 4096→128，C1 | 长 prompt 的首 token 与后续 decode，检查深度变动时的分段耗时。 |
| W4 混深度 | 4096→128，C32 | 长 prompt 下的 prefill/decode 同批；分别看首批、续入和每种深度的 TTFT。 |
| W5 混深度 | 1024→512，C8 | 中等并发、长生成；验证稳态、物理批大小变化及 TPOT/ITL 尾部。 |
| W6 混深度 | ShareGPT，自然 EOS，上限 512，C8 | 真实请求的中等并发和批大小变化，按深度看请求延迟。 |
| W7 混深度 | ShareGPT，自然 EOS，上限 512，C32 | 实际长度与结束时点变化；看请求吞吐、端到端延迟、各深度输出长度和质量变化。 |

每个负载按固定的请求序列给 1/2/3 轮分配 25%/25%/50%，前 32 条为 8/8/16 条，C8
热身及首批为 2/2/4 条。三轮组内显式指定和省略参数各占一半。深度绑定到请求，真实
语料按长度分层分配，避免浅层请求恰好集中在短输入上。W4 的首批和续入沿用同一比例。
RLT 的两个测量组使用相同 prompt、
到达顺序、目标深度和采样配置，只切换 `rlt_mixed_core`。以短 trace 验证实际同一次
Worker CORE 包含不同深度，检查物理 forward 和 FULL/PIECEWISE/eager 覆盖。packed
prefill 在含 prompt 的批次中也会合批，trace 将其与稳态纯 decode 分开统计。

正式测量每组运行三轮固定请求回放。W1–W4 每轮各 64 条，W5 每轮 32 条固定长度语料，
忽略 EOS。W6/W7 从同一 256 条 ShareGPT 语料中固定回放前 64 条，按 EOS 结束。长输出
负载统一使用固定子集控制测试时长，首轮 W5 的 64 条探针单独保留。每组先热身 8 条或
两倍并发量，取较大值且不超过语料大小。四个运行组为原生固定三轮、RLT 固定三轮并
开启 `rlt_mixed_core`、RLT 可变深度关闭 `rlt_mixed_core`、RLT 可变深度开启
`rlt_mixed_core`。RLT 组均开启 packed prefill 和已验证的 `rlt_async_prefill_lookahead`，
这轮测基础与请求深度，WCPB、自投机和 Decode pipeline 留作另行配对。
此前批内状态交接、FULL 缓冲复用和 prefill 提前入队的正式 W1/W3 A/B 使用
`rlt_mixed_core=false` 和 `rlt_packed_prefill=true`。本轮固定三轮 RLT 按要求显式
开启 `rlt_mixed_core`，重新测原生比值。
本轮在 GPU 7 串行启动四个服务配置，每组 21 个正式波。启动参数、实际 GPU 进程与
原始输出按组保留。
[本轮矩阵启动记录](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/w1w7-depth/matrix-run.json)

按优先级暂停全量矩阵时保留了 35/84 波，关闭混深度的可变深度组完成 21 波，开启
组完成 W1/W2/W3/W5 和两波 W6。W4 复测、W6 剩余一波、W7 及两组固定三轮正式
回放留在暂停记录中。下述 W1/W3 专项与这份全量回放独立保存。

报告总输出 tok/s、请求/s、每种深度的 TTFT/TPOT/ITL 及输出长度，并保留三轮请求的
尾延迟，防止低深度的平均收益遮住三轮请求的回退。固定输出的负载按实际 CORE 次数和
token 输出验收。W6/W7 另记录 EOS 和长度，解释因生成长度变化产生的吞吐差异。

原生 vLLM 在这个 checkpoint 上通过服务级配置选轮数，W1–W7 的三轮组可作为对照。
混深度主 A/B 采用同一个请求级深度序列的 RLT 两组，避免将减少 CORE 次数带来的收益
计入混深度合批本身。应用侧比较使用均开启 `rlt_mixed_core` 的 RLT 固定三轮与可变
深度版本，同一任务的输出变化随深度一并记录。Ouro 继续承担真实 gate 早退语义验收。

当前优化版在 GPU 7 用正常异步与 Graph 配置完成四档服务烟测。开关 `rlt_mixed_core`
的两组均通过，四条请求各返回八枚 token，prompt 每个位置都实际提交了三轮 CORE，
七个后续位置按各自档位执行。分别观测到 4 次和 0 次纯 decode 的混深度物理提交，
两组各有 2 次 P/D 混深度物理提交。显式四轮均返回 HTTP 400，随后合法请求继续成功。
该结果验证请求与执行路径，正式性能以本轮矩阵的无 profiler 数据为准。
[混深度开启烟测](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/independent-candidates/depth-smoke/mixed-on-gpu7/summary.json) ·
[混深度关闭烟测](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/independent-candidates/depth-smoke/mixed-off-gpu7/summary.json)

独立图路径探针在 GPU 4 对每个档位连续生成 8 枚 token，只计执行路径。1 轮请求的
第 1 次 CORE 和 2 轮请求的第 2 次 CORE 在七个后续 token 都走 eager，其他逐 op
图可复用。三轮和默认请求的 35 次后续逐 op 执行都走图。W2 C32 短探针里，开启
混深度时纯 decode 有 12 个 CORE 走 `execute_mixed_core` 的 eager 路径，其中 6 个
物理批跨实际深度、6 个在同一深度混合终轮和非终轮。正式吞吐和延迟仍取 GPU 7 的
同卡数据。
[实际 CUDA Graph 路径](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/w1w7-depth/terminal-core-probe/gpu4-mixed-on/summary.json)

### 固定三轮开启混深度的 W1/W3 专项

集中核对原生 vLLM、RLT 开启混深度 CORE、RLT 关闭混深度 CORE 三组。全部固定三轮，
RLT 保持 packed prefill 与提前入队。GPU 7 同卡使用 W1 C1、标准 W3 C1，以及前面
用来定位并发问题的 `W3 C16`，每波分别回放 8/8/32 条，每组同样热身后测三波。C1
核对单请求开销，C16 核对开启开关后长 prompt 混批和 decode 的实际执行差异。

| 负载 | 原生 / RLT 开 / RLT 关 tok/s | 开启后吞吐比 | 开启后 TTFT P50 / P95 比 |
|---|---:|---:|---:|
| W1 C1 | 262.38 / 255.38 / 255.52 | 97.33% | 93.11% / 92.21% |
| W3 C1 | 229.32 / 223.94 / 224.22 | 97.65% | 99.02% / 99.61% |
| W3 C16 | 1015.58 / 985.67 / 987.48 | 97.05% | 95.96% / 99.05% |

TTFT 比为原生耗时除以 RLT 耗时。27 个正式窗口均采到独占和 2032 MHz，三组实际
KV 容量均为 291018 token。开启合批的 W1 TTFT P50/P95 为 12.26/12.95 ms，原生为
11.41/11.94 ms。两个 C1 负载三个运行组均逐 token 一致。W3 C16 开关合批对照 92/96
逐 token 一致；开启组对原生为 86/96，差异集中在先前已有差异的续入请求子集。
[固定三轮专项无 profiler 数据](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/w1w7-depth/fixed3-focus/summary.json)

固定三轮、开启合批的独立 W3 C16 路径探针记录了 1640 次物理执行，其中 984 次为
CORE。全部 CORE 每次只有一个深度，包括 378 次 prefill/decode 同批 CORE。共 15 次
eager 回退集中在三批 prompt 尾段各自的五个 op，逻辑大小为 465、75、859 个 token，
均带 15 行 decode，且缺少对应的预热图大小。这个探针用于核对路径，时间以 GPU 7
无插桩的同卡对照为准。
[W3 C16 物理执行路径](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/w1w7-depth/terminal-core-probe/fixed-w3-gpu4-mixed-on/summary.json) ·
[eager 回退批大小](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/w1w7-depth/terminal-core-probe/fixed-w3-gpu4-mixed-on/fallback-shapes.json)

### 请求级可变深度的 W1/W3 合批对照

全量矩阵继续暂停。专项继续使用 W1 C1、W3 C1、W3 C16 三档，三波各回放 8、8、32
条，强制生成 128 枚 token。按固定序列请求 1/2/3/默认，开关组源代码、warmup、语料、
深度和其它开关一致。GPU 7 无 profiler 测吞吐、TTFT、TPOT、ITL 和逐 token 对照。
独立路径探针记录 FULL、PIECEWISE 图、普通 eager 回退和混深度 CORE 的 eager
执行，区分未捕获终轮、未配置批大小及 metadata 或地址校验失败。同时按请求验证
每个 prompt 位置的三轮、后续各位置的目标轮数。路径探针的 C1 请求各生成 16 枚
token，C16 保留完整 128 枚，计数用于归因。

W1 C1 与 W3 C1 在开关两组各核对 8 条，整个 decode 分别有 60 次短深度终轮 CORE 因
缺少预热变体走 eager。1 轮的第 1 轮、2 轮的第 2 轮各 30 次回退；2 轮的第 1 轮和
3 轮的三轮都 replay FULL。prefill 仍使用三轮，两个 C1 探针的开关前后路径计数一致。

W3 C16 的 32 条请求逐位置均达到约定深度。关闭合批时，纯 decode 的 CORE 为 1179 次
物理提交，717 次 FULL，462 次普通 eager 回退。后者包括 331 次变体缺失、109 次批大小
未配置和 22 次图键布局未匹配。开启后提交减为 622 次，212 次 FULL，16 次缺变体回退，
394 次进入混深度分支并直接走 eager。在这 394 次中，270 次实际轮次相同，触发条件是
这一轮有的请求结束、有的继续。两组都有 300 次 prefill/decode 同批 CORE 直接走合批
eager，该路径也受 packed prefill 启用。两个探针均未观测到 Graph 地址校验失败。
[CUDA Graph 路径与回退原因](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/w1w7-depth/variable-focus/graph-paths.json)

同卡正式波里，W3 C16 开启后吞吐 216.91→205.70 tok/s（−5.17%），TTFT P50/P95
569.44/4337.00→511.79/3954.65 ms，TPOT P50 54.52→60.45 ms，ITL P95
105.06→115.85 ms。三个请求深度的 TPOT P50 都上升，其中三轮为 65.08→67.67 ms。
开关输出 90/96 逐 token 一致，首 token 全部一致，差异出现在 2/3 轮请求的 decode。

W1 C1 关闭组首测/换序复测为 92.41/107.06 tok/s，开启组 97.78 tok/s，变化方向随
运行顺序翻转，路径一致。W3 C1 关闭组首测前两波采到外部 GPU 进程，原始结果保留但
正式对照换用独占复测，关闭/开启分别为 100.46/97.04 tok/s。两个 C1 负载逐 token
一致。W1 开启组按请求 1/2/3 轮的 TPOT P50 分别为 13.02/19.69/4.02 ms。
[可变深度专项无 profiler 数据](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/w1w7-depth/variable-focus/summary.json)

以实际 token 计，W3 C16 两组都处理了 9144 个 decode token 的 CORE 访问，覆盖
纯 decode 与 P/D。关闭/开启组分别有 4398/912 个访问走图，其中三轮请求由 3707
降到 905 个。`execute_mixed_core` 分支也触发 Worker 的相同输入布局复用限制，
实际跨轮或同轮中存在不同终轮状态时，该次 CORE 按混合状态路径构造输入。

## Loop 1 引擎进程和接入契约

本轮范围是 general plugin 发现、`LoopRequest` 绑定和逐 op 的 EngineCore 计划/回执。验收使用锁定版上游 EngineCore 的同步和批队列路径，模拟 runner 完成一轮 CORE 后才在 Coda 产生 token。另检查安装后的 entry point 可发现，以及请求参数经原有 `SamplingParams.extra_args` 传到 `LoopRequest`。

### 实现

- 在 `pyproject.toml` 注册 `rlt` general plugin。`vllm_rlt` 包入口延迟加载原有导出，插件发现阶段保持轻量。旧导出已随独立引擎迁出。
- 安装 `LoopRequest` 时按原类身份更新已加载的引用，并自检 EngineCore 使用的绑定。请求级参数沿 `vllm_xargs` 和 `SamplingParams.extra_args` 进入请求。格式错误走上游 `VLLMValidationError`。
- `LoopSchedulerOutput` 保留每个 span 的位置和深度，按请求汇总本步工作量供 EngineCore 使用。同一请求的两个 span 可出现在一份计划中。

### 验收结果

- CPU 回归 **41 passed**，包含实际调用 EngineCore 的同步和批队列 step、计划序列化、HTTP 请求格式到 `LoopRequest` 的转换，以及旧引擎契约和调度选择回归。逐 op 测试使用 scheduler 和 executor 替身，验证中间 op 完成后能继续，只有 Coda 交付 token。
- 在临时目录构建并安装 wheel。用 `VLLM_PLUGINS=rlt` 调用 vLLM 实际加载器，确认发现安装后的 entry point 及 `Request` 替换。用 `VLLM_PLUGINS=` 验证原生路径保留上游 `Request`。
- 新增和修改的 Python 文件通过 `ruff check` 与 `ruff format --check`。`git diff --check` 通过。

本次 CPU 测试使用本地已构建依赖的 vLLM checkout。Git revision 为 `3146a9df4140`，测试涉及的 EngineCore、请求、输出协议和加载器源码与该 revision 一致。该 checkout 的生成版本标签是 `0.26.1rc1.dev6849+g3146a9df4`。后续模型验收使用该 checkout 构建的 CUDA 扩展和记录的运行配置。

```bash
rlt_vllm_src=/home/zjy/code/david/tmp/inkling2-mini-perf-20261123/vllm-source
rlt_vllm_deps=/dev/shm/zjy-inkling2-perf/vllm-deps
env CUDA_VISIBLE_DEVICES= VLLM_PLUGINS= \
  PYTHONPATH="$PWD:$rlt_vllm_src:$rlt_vllm_deps" \
  /home/zjy/code/david/.venv/bin/python -m pytest -q \
  tests/plugin tests/test_engine_core_contract.py tests/test_scheduler_selection.py
```

本轮接入契约通过。后续 loop 接入真实调度器与 GPU 模型。

## Loops 2–5 调度、KV、Worker 和 ModelRunner

- `LoopScheduler` 直接实现上游接口。`LoopAsyncScheduler` 继承同步类，对已知后继的 op 提前提交。请求通过各自的 span 记录深度和逻辑进度，完成回执仍用原计划结算。
- 调度器复用 vLLM 的块分配和前缀缓存。分配时延迟发布，终轮完成后再提交有效前缀。取消时保留在途 KV，等最后一份回执后释放。CPU 用例使用真实 vLLM KV 管理器。
- `LoopWorker` 通过上游工厂装配 MRv2 子类。功能阶段用 eager，CORE 内的不同深度和终轮路径分组执行。Runner 复用上游输入、metadata 和采样器，在设备持有 loop 状态。
- 已通过的 CPU 用例覆盖逐请求深度、同批混深度计划、异步取消和 Coda 去重，以及在终轮之前查不到可复用前缀。模型端到端结果随 GPU 请求验收更新。

## Loops 6–8 模型和服务验收

- Inkling 复用上游的参数、层与 KV 注册，按逐 op 续跑 AttnRes，CORE 的非终轮恢复 Prelude 状态，终轮把状态交给 Coda。真实并发测试发现 Coda 合批会收到有 bank 和无 bank 的两种行。Coda 不再保留 CORE 的 bank，该问题修复后重跑请求级轮数用例。
- Ouro 复用 vLLM 的 dense 组件。按参考实现保留每层四处归一化，并建设各深度的注意力缓存。提前退出时，用 vLLM 的槽位更新通路将退出轮的 K/V 写入被跳过的深度。配置范围错误通过上游请求错误回执返回。
- 接入验证使用单卡 B300、vLLM revision `3146a9df4140`、标准 `vllm serve` 和本地 checkpoint。Inkling 默认以三轮 prefill 产生首枚 token，`rlt_target_loops` 作用于后续 decode。GPU 测试使用 eager，开启 `--additional-config '{"rlt_trace_steps":true}'` 可查执行记录。

### GPU 验收记录

| 验收 | 实际结果 | 原始记录 |
|---|---|---|
| Inkling 异步请求级轮数 | 同一服务并发 `1`、`3`、默认三轮。三条请求各生成八个 token，首枚来自三轮 prefill，七个 decode token 的实际 CORE 次数逐请求为 `1/3/3`。观察到 17 个混深度 CORE 计划。并发结果与各自 solo 相同，固定三轮短例与原生相同。超范围 `4` 返回 HTTP 400，之后正常请求继续成功。 | `/tmp/rlt-inkling-functional.XR1nvW/` |
| Inkling 同步、独立 Worker、WCPB 和 prefix cache | 上游 `MultiprocExecutor` 启动独立 Worker，128-token prompt 按 16-token chunk 分段。同一请求有 8 个跨深度 CORE 计划，重复请求从 64-token 前缀命中继续。冷执行、命中后和原生 vLLM 都生成相同的 8 个 token。 | `/tmp/rlt-inkling-wavefront.0uoTX0/` |
| Ouro 早退 | 4 个阈值分别生成 10 枚 token。阈值 0 的 9 个 decode token 都在深度 0 退出。阈值 0.35 的零基退出深度为 `1,2,2,2,3,1,2,1,2`，实际经历了较浅退出后继续读更深历史并生成。GSM8K-87 固定四轮回归随后达到 59/87。 | `/tmp/rlt-ouro-trace.BpCIxW/` |

## Loops 9–10 自投机与并行接口

- Inkling 的单候选 greedy 自投机以 `rlt_self_speculation` 启用。第一个 CORE 保留一份浅层 Coda 状态。浅层 Coda 提出候选，target 继续完成深层 CORE 和 Coda。候选只在与 target 一致时接受，拒绝时交付 target token。Target Coda 在递交请求状态前写回本位置。跨 token 的 Decode pipeline 与在途投机 KV 回收留给性能阶段。
- GPU 短例覆盖 6 条请求，每条生成 12 枚 token。自投机与关闭自投机的插件逐 token 相同。真实执行记录有 58 次接受和 8 次拒绝。原生 vLLM 并发对照有 5/6 条完全一致，另一条从第 7 枚 token 发生分歧，三个路径单请求复跑全部一致。仍待隔离批形状下分歧的具体数值来源。原始记录在 `/tmp/rlt-inkling-spec.JssSVj/`。
- DP 提供确定的 op 选择以及空闲 rank 的 dummy 描述。CPU 验证投票聚合、平票和空闲行为。实际 DP+EP collective 与多卡运行留待后续，当前 worker 限定 DP=1、PP=1。

W1 C1 固定三轮复跑使用一条 128-token prompt 并强制生成 128 枚 token，插件和原生 vLLM 逐 token 一致。此处记录功能性对照。正式吞吐和 TTFT 仍在阶段二测量。


## 插件基础能力清单

| 能力 | 入口与当前证据 | 验收缺口 |
|---|---|---|
| 安装、插件启用和原生对照 | `pyproject.toml` 注册入口，使用锁定的 vLLM revision。Python wheel 和 CUDA wheel 均已构建。原生对照用 `VLLM_PLUGINS=`，启用时 API 进程、EngineCore 和独立 Worker 均可加载。 | CI 已切换到固定的上游源码，远端 workflow 尚未运行。 |
| CLI 和进程级配置 | `entrypoints/cli.py` 使用上游 serve parser、校验和启动逻辑。`entrypoints/runtime_args.py` 将 RLT flags 合入 `additional_config`。类型、拼写和正数约束收口到 `config.py`。同步、异步类和 Worker 由 CLI 选默认值。 | 每个新增启动选项需同时覆盖项目 CLI 与 `vllm serve --additional-config` 两条入口。 |
| 请求级参数与线上 API | `core/request.py` 校验 `vllm_xargs`，completions、流式 chat 和离线 `vllm.LLM` 已通过真实请求。线上及离线生成 logprobs 已透传。线上验了停止字符串、流式断连和错误请求后的恢复。 | prompt logprobs 和结构化输出在当前功能边界明确拒绝。待扩展批量输出字段和参数组合用例。 |
| 新模型 | 在 `models/` 添加 adapter、在 `plugin.py` 注册，再补全 `core/loop.py` 的布局声明、每轮 KV 注册和对应模型用例。需要定制 batch 输入时可使用上游 MRv2 的 `get_model_state_cls`。Inkling 和 Ouro 已有真实运行证据。 | 第三个模型加入前补一份 adapter 合约用例，覆盖缺少 KV 层或模型配置不匹配时的启动错误。 |
| 自有算子 | `ops/` 放 Python 包装和 Torch reference，`csrc/` 放 CUDA 算子。SwiGLU 的投影和权重加载沿用 vLLM；Ouro 选用 `--rlt-silu-and-mul` 时替换 MLP 中的 activation。 | 与 vLLM packed kernel 的性能和数值差异要在相同输入上配对检查。 |
| 请求与 KV 生命周期 | CPU 覆盖在途取消、Coda 结算、前缀提交、早退状态和 cache reset。独立 Worker 的 WCPB 已验证前缀命中。实际回归验证 Ouro 早退后继续生成。 | 流式断连后运行请求数归零，后续请求成功。待补低 KV 容量下抢占恢复、固定与自适应请求并发的端到端覆盖。 |
| 排障和回归 | 复用 vLLM 的健康检查和进程日志。可打开 `rlt_trace_steps` 查看请求 ID、op、span、depth、Ouro 提交和 draft 判定。CPU 接口测试及 GSM8K-87 评分脚本已迁入插件路径。 | W1 和请求级轮数已固化为两条 GPU 服务烟测。还需度量和报告按请求实际深度、前缀命中和投机接受率。 |
| 设备和并行能力 | 同步、异步和单卡独立 Worker 已验，DP op 共识与 dummy 描述有 CPU 回归。当前 worker 限定 DP=1、PP=1。 | TP 至少补一个真实多卡 smoke；DP/EP collective、跨 rank dummy 和 CUDA Graph 进入对应功能实现后验收。 |

启动参数按服务生效，例如 `--rlt-prefill-chunk-size 16`。逐请求参数继续走
`vllm_xargs`，例如 `rlt_target_loops`。扩展启动参数时，从 `config.py`
声明字段，在 `entrypoints/runtime_args.py` 暴露 CLI，再由实际使用的组件读取。
若只需调 RLT 的模型算子，在适配器选择该算子并保留上游默认路径。

## 清理后复验

- 移除独立引擎、服务、PD、旧模型实现及其专用测试。保留并迁移 GSM8K-87
  回归和 RLT 的 SiLU-and-Mul 算子、原生构建与专项测试。旧架构的文档按 Git
  revision 查阅。
- 删除后的 wheel 重新启动 Inkling。W1 C1 `128/128` 输出与原生 vLLM
  逐 token 相同，三个不同轮数的并发请求成功。另有一个逐请求设为一轮的
  `/v1/chat/completions` 流式请求正常结束。原始记录在
  `/tmp/rlt-plugin-clean-wheel.7qQGVX/`。
- Ouro 通过锁定的 GSM8K-87 协议完成固定四轮，结果 **59/87**，与记录的
  Transformers 分数一致，质量门槛通过。使用阈值 `0.5` 和最少一轮跑同一 87 题仍是 **59/87**，
  平均 decode 深度由 **4.000** 降到 **3.026**。记录在
  `/tmp/rlt-plugin-eval.QI0HFo/`。
- 在 B300 编译并安装 CUDA wheel。原 SiLU-and-Mul 数值、尾部、对齐、流和捕图
  专项子集通过。使用 `vllm-rlt serve --rlt-silu-and-mul` 启动 Ouro，早退短例
  10 个 token 与使用 vLLM activation 的服务逐 token 一致。用上游 `vllm.LLM`
  离线启动同一套配置，请求的 8 个输出 token 及其 logprobs 均返回。
  记录在 `/tmp/rlt-plugin-native-wheel.VS3mcI/`。

### 重跑 Inkling 文件级 GPU 验收

按 [README](../README.md) 启动插件服务并加 `--rlt-trace-steps`，将日志重定向
到文件。用匹配的模型、dtype、执行方式及 KV 配置另启原生 `vllm serve`，
并设置 `VLLM_PLUGINS=`。用以下命令运行固定深度和同实例混深度两个用例。

```bash
RLT_SMOKE_PLUGIN_URL=http://localhost:8000 \
RLT_SMOKE_NATIVE_URL=http://localhost:8001 \
RLT_SMOKE_STEP_LOG=/path/to/inkling.log \
python -m pytest -q tests/plugin/test_serving_smoke.py
```


### 最终复验记录

- 当前源码的调度、入口、配置、原生构建与算子 CPU 用例 **69 passed**。
  锁定的评测环境运行 GSM8K 和 tokenizer 用例 **143 passed**，
  三项缺 Thinking tokenizer 资源的用例跳过。
- 已安装 CUDA wheel 的 SiLU-and-Mul 专项子集 **113 passed**。
  最后两条 Inkling 服务验收 **2 passed**。真实服务还验证了生成 logprobs、
  prompt logprobs 返回 HTTP 400、停止字符串生效，以及流式断连后请求回收。
- 对早先 Inkling 并发分歧的 6 个输入，用一批请求携带生成 logprobs 复跑，
  原生和插件逐 token **6/6** 一致。原先独立并发请求的分歧尚未定位。
  复跑记录为 `/tmp/rlt-plugin-native-wheel.VS3mcI/inkling-parity-logprobs.json`。
- 197 个旧框架文件从工作区删除，分支的 Git 暂存区为空。开发记录和代码
  留在 `vllm-rlt-plugin2` 工作区供 VS Code 审阅。阶段二吞吐、TTFT 和
  Loop 特性收益的正式对照尚未开始。


## 阶段二执行路径接入

这一轮先实现缺失路径，并用 B300 上的 Inkling VarLoop 做功能对照。
各路径保留开关，供后续逐项配对测量。

| 功能 | 实现与验收 | 当前边界 |
|---|---|---|
| 混深度 CORE | `--rlt-mixed-core` 让 Inkling 的投影与 MoE 共用整批权重计算。各深度使用自己的 attention 和短卷积缓存。Ouro 也共享 QKV、输出投影和 MLP，按实际退出的行补写。 | Inkling 稳定合成并发输入 6/6 逐 token 一致，自然文本测试仍出现并发差异。Ouro 混阈值也观察到分歧。默认保持分组路径。 |
| Packed prefill | `--rlt-packed-prefill` 通过 MRv2 的 ragged token/offset 表把 chunk 放进同一 forward。WCPB 同请求不同深度也能合并。每个深度写自己的 slot，并按各 chunk 的实际起点处理短卷积。两条 128-token 输入在冷请求和前缀命中时均与对照逐 token 相同，前缀命中从位置 64 继续。 | 多形状和压力下的配对性能待测。 |
| 逐 op CUDA Graph | 复用 vLLM `CUDAGraphMode`、FULL wrapper 和 breakable PIECEWISE wrapper。RLT 为 op、深度、行布局及 metadata 保存稳定输入。公开 CLI 接受 `-cc`。 | Inkling 的均匀 decode 走 FULL，prefill 动态 attention 走 PIECEWISE。混深度 CORE、尺寸未命中和片段超出图缓存容量走 eager。当前 GPU 验收覆盖 Inkling。 |
| 跨 token Decode pipeline | `--rlt-self-speculation --rlt-decode-pipeline` 启动一个未确认的下一 token。KV 块复用 vLLM。候选命中时保留进度，拒绝时删除该位置的 hidden 并用真实 token 原地重算。CPU 覆盖接纳和回滚。 | Inkling 单请求对照 6/6 token 与 logprobs 一致，稳定合成并发输入 6/6 一致。自然提示并发输出仍需定位。 |
| Draft/target Coda 合批 | 同一 Coda forward 支持 draft 和 target 两类行。只允许 target 修改已确认的 worker 采样状态，draft 只返回候选。实际运行记录有 28 个混合 Coda 计划。 | 尚未测量合批带来的收益。 |

Graph 对照的 3 条短例都与 eager 逐 token 相同。W1 固定三轮跑完 128 个输出
token，首次 capture 和后续 replay 均与 eager 逐 token 一致。开启 packed prefill
或 Decode pipeline 会改变实际的批形状。第一处分歧的 logprobs 已保存，多个
案例出现了候选排序翻转，尚未隔离具体算子与舍入来源。

对应原始记录位于 `/tmp/rlt-phase2-graph.xeGbY9/`、
`/tmp/rlt-phase2-packed.G1ieoi/`、`/tmp/rlt-phase2-coda.jKlEbE/`、
`/tmp/rlt-phase2-mixed.hxiP4u/`、`/tmp/rlt-phase2-pipeline.lDc9v2/`
及 `/tmp/rlt-phase2-ouro.RPMbpU/`。

通过新 CLI 同时开启 Graph、混深度 CORE、WCPB、packed prefill 及自投机
pipeline。三条并发 128-token prompt 的输出都与 eager 对照一致。另一组
6 条 16-token 输出全部正常结束，记录到 **39 次保留**和 **6 次回滚**。
组合验收记录见 `/tmp/rlt-phase2-combined.763tW1/`。

相关 CPU 回归 **71 passed**，GPU 服务项与算子 GPU 用例按环境配置跳过。
Ruff 和 `git diff --check` 通过，Git 暂存区为空。
吞吐和 TTFT 的 95% 门槛仍待正式配对验收。


## 首轮基础优化性能对照

本轮数据采集于 CUDA Graph 启动预热修正前。
使用同一张 NVIDIA B300 GPU 2、同一个 Inkling-2-Mini VarLoop checkpoint、固定三轮
BF16、TP1、异步调度、每步 1024 token、最大 32 请求。双方采用
`FULL_AND_PIECEWISE`，启用 chunked prefill、关闭前缀缓存。共同 KV 预算为 80 GiB，
双方实际报告 291,018 可用 token。RLT 额外启用 `rlt_mixed_core` 和
`rlt_packed_prefill`。WCPB、自投机和跨 token pipeline 关闭。

每档先回放一整波热身，再测三波固定请求。W1 C1 为 8 条 128→128，W3 C16
为 32 条 4096→128，均忽略 EOS。W7 C32 回放 64 条 ShareGPT，允许自然 EOS、
上限 512。实际这两组 W7 请求均生成到 512-token 上限。计时取客户端 SSE token ID
到达时标，吞吐计入最后一条请求的排空时间。pmon 在计时窗内未采样到其他 GPU 2
进程的 SM 活动。

配置对齐后，执行覆盖仍有差异。原生 vLLM 按 batch 捕图，RLT 按逐 op 和输入布局
捕图并在混深度 CORE 或未命中时走 eager。RLT 的两轮 W3 正式窗口包括新增捕图，
W3 与 W7 的数据应作为初测，整表不包含 profiler 开销。

| 负载 | 吞吐 vLLM / RLT（tok/s） | RLT / vLLM 吞吐 | TTFT P95 vLLM / RLT（ms） | TTFT 达成率 | TPOT P50 vLLM / RLT（ms） | 完整输出 ID 一致 |
|---|---:|---:|---:|---:|---:|---:|
| W1 C1 | 260.6 / 136.8 | 52.5% | 12.61 / 16.32 | 77.3% | 3.78 / 7.24 | 24/24 |
| W3 C16 | 1012.1 / 151.2 | 14.9% | 1233.78 / 1316.90 | 93.7% | 13.41 / 97.34 | 86/96 |
| W7 C32 | 3723.8 / 314.5 | 8.4% | 253.99 / 441.12 | 57.6% | 8.36 / 101.36 | 8/192 |

这里的吞吐和延迟分位数均取三波的中位数。TTFT 达成率为原生 P95 除以 RLT P95。
三档的吞吐和 TTFT 达成率都低于 95%。RLT 的 W3 吞吐范围为 114.4–179.1 tok/s，
W7 为 233.2–316.9 tok/s。W7 三波 TTFT P95 分别为 435.85、4483.81 和
441.12 ms。W3 和 W7 的输出分歧最早发生在第一个 token，W7 分歧请求的
首次分歧中位数为第 4 个 token。这两档速度只作诊断对照。

原始测量及启动配置保存在 [性能测量目录](../artifacts/plugin2-perf.9pulvo/)。
[汇总 JSON](../artifacts/plugin2-perf.9pulvo/summary.json) 含各档 TPOT、ITL、
输出及 GPU 采样。用 `summarize.py` 可重算已保存的波次。


## Host / Device 时间轴

[交互时间轴](../artifacts/plugin2-perf.9pulvo/timeline/host-device.html) 可切换 W1
稳态 decode、W3 prefill、prefill 到 decode 的过渡及后续 decode。每个片段用相同
时间刻度，分别对齐双方的阶段锚点。Nsight 独立采样，未计入上面的吞吐和 TTFT。

W1 采到原生 36 个、RLT 39 个 decode token。以下按各自 token 数归一化。

| 每输出 token | 原生 vLLM | RLT |
|---|---:|---:|
| Host CUDA Graph replay | 1 | 5 |
| Device Graph 或 kernel 覆盖时间 | 3.72 ms | 4.01 ms |
| 首尾计算之间无 Graph 或 kernel 覆盖的时间 | 0.24 ms | 4.70 ms |
| Host scheduler plan 累计时间 | 0.07 ms | 0.13 ms |

W1 的主要差距出现在 Prelude、三次 CORE、Coda 分开提交后的 GPU 计算间隔。
无 Graph 或 kernel 覆盖的区间仍可能有拷贝。W3 的采样含相同的 16 条输入，
原生输出上限 128，RLT 为 12，只用于观察对齐的局部阶段。在 RLT 的整段 W3
记录里，纯 decode 的 Prelude、CORE、Coda 每次 forward 的请求数中位数分别为
8、15.5、8，CORE 范围为 1–16。[采样汇总](../artifacts/plugin2-perf.9pulvo/timeline/profile-summary.json)
中 W3 数字限定在首次纯 decode 后 260 ms。


## Inkling CUDA Graph 启动预热

逐 op 捕图接入 vLLM `Worker.compile_or_warm_up_model()` 调用的 `capture_model()`。
Warmup 复用 MRv2 dummy 输入、attention metadata、slot mapping 和 vLLM 的 Graph
wrapper，依次预热 Prelude、每个 CORE 深度、Coda。配置的同深度 decode 批优先
覆盖小批形状，单行 prefill 预留捕图额度。未预热的布局在请求中走 eager。

在与首轮测速相同的 B300 和 80 GiB KV 配置下，服务就绪前创建 60 张逐 op 图。
没有额外回放请求热身，直接发送 W1 的首波 8 条请求，输出与改动前逐 token
一致，日志无运行时新增捕图，吞吐 135.0 tok/s。旧版三波中位数是 136.8 tok/s。
另一个使用可扩展 KV 的 GPU 启动验证保留了默认 JIT kernel warmup，同样在就绪前
完成 60 张图。自动分配固定 KV 的验证先捕图估算显存并释放临时图，再用实际 KV
捕图。首条请求的 128-token prefill 命中五个 PIECEWISE 图条目，随后的 decode
命中并 replay 了五个 FULL 图。首条请求的输出 token ID 与改动前逐个相同。
修正后的 W3/W7 尚未重新配对测量。
原始数据见 [启动日志与首波请求](../artifacts/plugin2-perf.9pulvo/startup-warmup/)。


## 异步优化方案

保留映射 A，每个 EngineCore step 执行一个 loop op。`LoopAsyncScheduler` 继承
`LoopScheduler`，复用 vLLM 的 batch queue、executor、MRv2 输入准备和异步输出。
第一轮用 Inkling 固定三轮验证基础异步，关闭 WCPB、自投机和 Decode pipeline。

### 当前开销和依赖

对修正启动预热前的 W1 trace 按 decode token 归一化。在同一进程内用 CUPTI
`correlationId` 把 GPU graph 区间关联到 Host `cudaGraphLaunch` 及 Worker op。

| 指标 | 原生 vLLM | RLT |
|---|---:|---:|
| Graph replay / token | 1 | 5 |
| MRv2 `prepare_inputs` Host 累计时间 / token | 0.26 ms | 0.91 ms |
| H2D memcpy / token | 2 | 10 |
| D2H memcpy / token | 2 | 2 |
| D2D memcpy / token | 0 | 约 286 |

RLT 相邻 graph 区间的中位间隔为 Prelude → CORE **1.02 ms**、CORE → CORE
**0.55 ms**、CORE → Coda **0.23 ms**、Coda → 下一 Prelude **2.68 ms**。
这些间隔含图外 kernel、采样和拷贝。拷贝按 CUPTI memcpy 事件统计，受控复测时
需用 NVTX 把 D2D 归因到 metadata、hidden 和算子内部。W1 样本里两边每个 token
都回读采样结果。RLT 的 scheduler 目前直到 Coda 回执处理后才创建下一 token
的普通 span，因而回执位于下一 Prelude 的提交路径上。
[Host / Device 时间轴](../artifacts/plugin2-perf.9pulvo/timeline/host-device.html)

当前 `LoopAsyncScheduler._on_submit` 已能在提交后立即推进已知的后继 op。
固定三轮可以把下一 CORE 排进上游队列。Ouro 的 gate 决定是否排下一轮，草稿
Coda 的 token 也有独立的回读依赖。`LoopModelRunner` 仍为每个 op 调用完整的
MRv2 `execute_model`，重复输入准备、块表 gather 和 slot mapping。普通 Coda
已走原生 `AsyncOutput`，draft/target 合批与 Ouro 的特殊分支在 Worker 内提前
调用 `get_output()` 或从 GPU `.tolist()`/`.item()`。

锁定版 vLLM 的 TP1 默认异步队列深度为 2。`EngineCore.step_with_batch_queue`
先填队列，再按顺序回收最旧结果。`UniProcExecutor` 的 `non_block=True` 仍在当前
CPU 线程调用 Worker，`Future` 将 GPU 输出的等待推到回收时。MRv2 把真实采样 token
留在 `RequestState.last_sampled_tokens`，`postprocess_sampled` 更新设备状态，
下一批 `prepare_inputs` 在 GPU 读取它。单独的 D2H stream 同时把前端所需输出送回
CPU。输入和输出的所有权由两条路径分别管理。

### 优化一 让真实 token 在设备上进入下一 Prelude

#### Scheduler 和 Worker 交换哪些字段

设 `p` 为当前 decode span 的输入位置。`Coda(p)` 消费位置 `p`，采样得到位置
`p+1` 的真实 token ID。首版选用以下数据结构，覆盖 Inkling 固定轮数和请求级
固定 1/3 轮的稳定 decode。

| 数据结构 | 字段及其含义 |
|---|---|
| Scheduler 的 `LoopRequest` | `num_computed_tokens=p` 表示 CPU 已确认的计算前缀终点，`num_tokens=p+1` 表示 CPU 已知的 token 数。提交 `Coda(p)` 将 `num_output_placeholders` 从 0 增到 1。CPU 回收 Coda 后才推进前缀、追加实际 token ID 并减回 0。 |
| Scheduler 的 `_RequestState.spans` | 继续以 `token_start` 为键。`spans[p]` 等 Coda 回执。预排 `spans[p+1]` 时增加 `_SpanProgress.awaiting_coda_at: int \| None`，值为 `p`。每个请求至多保留一个这样的后继。该后继在等待期间只可发 Prelude，成功确认后清掉依赖并开放 CORE。现有 `tentative` 和 `input_token` 继续表示自投机路径的候选。 |
| `LoopSchedulerOutput` 与 `LoopSpan` | 后继计划发 `op=PRELUDE` 和 `LoopSpan(token_start=p+1, token_count=1, depth=0, prefill=False, sample=False, input_token=None)`。保留原有 `request_id`。普通 decode 的 `input_token=None` 沿用 MRv2 的设备输入来源，调度计划无需携带采样 ID。 |
| 原生 `CachedRequestData` | 该行 `req_ids` 仍指向已注册的请求，`num_computed_tokens=p+1` 是 **本次 Worker 输入的 CPU 上界**。它根据本次 span 的起点构造，独立于 `LoopRequest.num_computed_tokens`。`new_block_ids` 只携带 `allocate_slots` 返回的各 KV group 增量，`num_output_tokens` 填 CPU 已确认的计数，TP1 的 `new_token_ids` 为空。 |
| MRv2 Worker 的 `RequestState` | `req_id_to_index[request_id]` 得到持久槽 `s`。`last_sampled_tokens[s, 0]` 保存 `Coda(p)` 生成的真实 ID，`num_computed_tokens.gpu[s]` 由 MRv2 `postprocess_sampled` 推到 `p+1`。`update_requests` 将上面的 CPU 上界写到 `num_computed_tokens_np[s]`。`prepare_pos_seq_lens` 和 `combine_sampled_and_draft_tokens` 读取设备状态，填 `InputBuffers.positions` 和 `input_ids`。 |
| 原生 `ModelRunnerOutput` 与 `AsyncOutput` | 通过 `req_ids` 对齐 `sampled_token_ids` 和 `logprobs`，原生 D2H stream 和 `copy_event` 驱动输出 Future。中间 Prelude/CORE 继续用 `_loop_completion` 返回 CPU 回执，表示该 op 已入流。它的 `sampled_token_ids` 为空行，EngineCore 仍通过 `sample_tokens` 收取。 |

示例中若 CPU 前缀终点仍为 `p`，为后继申请的覆盖范围到 `p+2` 独占，
`allocate_slots` 的 `num_new_tokens` 参数按 `p+2 - request.num_computed_tokens`
计算，即 2。已有块会被 allocator 复用。用 `delay_cache_blocks=True` 保留发布时机。
GPU 真实进度来自原生 `postprocess_sampled`，CPU 借用 `CachedRequestData` 发上界。
读取 token 和生成位置仍走 MRv2 现成的 kernel。

```text
GPU compute      Coda(p) → sample/postprocess → Prelude(p+1) → CORE(p+1)
GPU D2H copy                    └→ token ID / logprobs → copy_event
CPU scheduler    提交 Coda(p) → 预排 Prelude(p+1) → 回收 Coda → 开放后继 CORE
```

#### 提交、确认和回收

1. `LoopAsyncScheduler` 在 `Coda(p)` 已入队、请求仍可生成、KV 充足时允许预排
   `Prelude(p+1)`。要求 `num_output_tokens + num_output_placeholders < max_tokens`
   且 `p+2 < max_model_len`，即 `Coda(p)` 生成 token 后仍能再生成。尊重暂停状态。
   预留失败时等正常回执。
2. EngineCore 按现有顺序调用 `execute_model(Coda)`、`sample_tokens()`，再提交后继。
   D2H stream 从采样结果拷贝，compute stream 上的设备状态更新先于下一 Prelude
   的输入生成。以上顺序允许下一 Prelude 在 Coda 的 `Future.result()` 被调用前入流。
3. `update_from_output` 用现有 `_in_flight[id(scheduler_output)]` 对应 `_Submission`，
   先处理 Coda 的 sampler 结果和结束条件。继续生成时解除 `awaiting_coda_at`。
   完成、取消或收到文本 stop 触发的前端 abort 时，禁止该后继继续和产生输出。
   已经入流的 Prelude 可以完成，相关位置的 KV 保持私有。
4. Scheduler 用 `_Submission` 持有的 `_RequestState` 和 span 引用归属回执，状态表
   仍指向同一实例时才允许更新。保留 `in_flight` 计数，所有 CPU 回执收齐后再走已有
   `finished_req_ids` 清理。Worker 的 KV/hidden 使用和复用在同一 compute stream
   按序执行。任何新增跨流消费者需用对应 event 退役，CPU 入流回执仅用于调度记账。

首版验收通过后再扩展到首个输出 token 后的 prefill → decode 衔接。前缀命中和文本
stop 的异步 abort 单独验收。浅层草稿的接受与拒绝继续由 Decode pipeline 处理。

### 优化二 复用 MRv2 输入状态和按数据类型选择存放位置

#### 一个续轮批次复用哪些对象

`LoopModelRunner` 首版维护一个 `_PreparedDecodeBatch`，存的是最近一次普通 decode
的 MRv2 结果和设备视图，供下一次完全相同的有序批次使用。`B` 是行数，`T` 是本步
有效 token 数，`G` 是 KV group 数，`R=max_num_reqs`。

```python
@dataclass(slots=True)
class _PreparedDecodeBatch:
    rows: tuple[tuple[str, int, int, int], ...]  # request_id, MRv2 slot, token_start, token_count
    input_epoch: int
    batch_desc: BatchExecutionDescriptor
    input_batch: InputBatch
    block_tables: tuple[torch.Tensor, ...]  # 每个 KV group 一张已 gather 的块表
    slot_mappings: torch.Tensor  # [G, T_padded]
```

`rows` 用 MRv2 `BatchReqState.req_ids` 和 `idx_mapping_np` 的顺序构造，位置和长度取
`LoopSpan`。在首次 `prepare_inputs` 后保存原生 `InputBatch`。它的 `req_ids`、
`idx_mapping_np`、`query_start_loc_np` 和 `seq_lens_cpu_upper_bound` 留在 CPU，
`input_ids`、`positions`、`query_start_loc`、`seq_lens` 等是 `InputBuffers` 上的
GPU 视图，`idx_mapping`、`logits_indices` 保留同一批准备时生成的设备 tensor。
`prepare_attn` 首次调用原生 `BlockTables.gather_block_tables` 和
`compute_slot_mappings`，此时将张量和 `InputBatch` 一起保存到 Runner 的记录里。

下一 op 仍走 `execute_model` 和 `gather_batch_req_state` 检查实际批顺序。只有
`rows`、`input_epoch`、`batch_desc` 都相同，并且当前 op 的通用输入语义相同，才在
Runner 的 `prepare_inputs`、`prepare_attn` 钩子复用上述对象。`LoopExecution.ordered`
每步依据当前 `LoopSpan` 重新绑定，因此 `depth`、`last_core` 及 op 使用各自当前值。
`model_state.prepare_attn` 第一轮照常调用现有 builder。若它也形成可观测瓶颈，再用
Inkling 的 `get_model_state_cls` 钩子保存经验证可复用的 backend metadata。

`input_epoch` 是 Runner 的失效计数。请求槽位分配/回收、KV 块 append/overwrite、
copy-on-write、KV 清零、输入设备缓冲内容被重写或采样推进设备位置时使记录失效。
遇到新位置、新批成员、重排行、dummy/profile、ragged 或混深度首版用原生准备路径
并使旧记录失效。为不同深度生成 mask 时使用独立视图或 scratch，避免覆盖共享的
group slot mapping。这个单条缓存的命中率和收益先在 W1 与稳定 C16 decode 测量。

#### 各字段的存放位置及更新方式

| 数据 | 存放与更新 |
|---|---|
| `RequestState.prompt_len`、`prefill_len`，`BlockTables.num_blocks` | 直接用 MRv2 `UvaBackedTensor`。前两个是 `[R]` 的 `int32`，`num_blocks` 是 `[G, R]` 的 `int32`。CPU 更新 `.np`，消费 kernel 读 `.gpu` 对应的 UVA 快照。 |
| `BlockTables.block_tables[g].gpu` 及条目增量 | 每组设备表为 `[R, max_blocks[g]] int32`。`CachedRequestData.new_block_ids` 进入原生 `append_block_ids`，以 `StagedWriteTensor` 暂存 row index、start、累计长度和新 block IDs。通过 UVA 交给 `apply_staged_writes` 的 fused writer，只改新增位置。 |
| `InputBatch.idx_mapping`、`query_start_loc` | 在 MRv2 已选定行序后生成。首版在同一 span 续轮复用 `[B]` 的设备索引和 `[B+1]` offsets，有 padding 时使用实际捕图尺寸。变批时沿用原生 H2D。后续按实测采用 pinned/UVA 描述后写入固定设备地址。 |
| 混深度需要传给 GPU 的路由字段 | CPU 以 MRv2 行序生成紧凑 `int32[B, 3]`，列为 `req_slot, depth, flags`。明确参与 GPU 路由时用 `UvaBufferPool.copy_to_uva()` 供一次性分组/掩码 kernel 读取，重复读取的路由结果放到设备 scratch。位置继续使用 `InputBatch.positions`。 |
| `input_ids`、`positions`、`seq_lens`、已 gather 的块表与 `slot_mappings` | GPU 常驻。设备侧生成，保持 FULL Graph 可绑定的地址。`RequestState.all_token_ids` 延续上游对长 token 历史的 UVA 策略。每步采样的 `last_sampled_tokens[R, 1]` 和设备进度继续常驻 GPU。 |
| `_InklingState.stack/prelude/inputs/bank`、KV | GPU 常驻。它们供后续 op 和 attention kernel 读取。 |
| 提交给 CPU 的 token 和 logprobs | 沿用 `AsyncOutput` 的 D2H stream 和 `copy_event`，保留源 tensor 到拷贝完成。 |

UVA 使用 pinned host memory 的设备视图。它省去显式 DMA 的场合仍有 CPU staging
和跨总线访问。多次读或被 Graph 捕获的数据优先放稳定的设备 buffer。槽位容量从
实际在途 **消费者数** 推导，多子批时也要计算。消费者未退役时不得覆盖或扩容对应
槽位。沿用 `UvaBufferPool` 的非 UVA 显式拷贝回退。只有单独输入准备流能证明降低
关键路径时才引入它，并用 per-buffer event 从 H2D 到 compute 建依赖。

#### 逐 op Graph 如何共享设备地址

输入复用验收后再调整 `worker/loop_graph.py`。现有 `_GraphEntry.inputs`、
`metadata_buffers` 和 `padding` 是按图复制的。目标是让同一 uniform decode bucket
的各 op 绑定同一组稳定设备 views，包括 `positions[T_padded]`、`is_padding[T_padded]`、
各 KV group 的 `block_table`/`slot_mapping` 和 Inkling `_InklingState` 中布局一致的
`stack`、`prelude`、`inputs`、`bank`。`stack` 的形状为
`[max_loop_blocks + 1, T_padded, hidden_size]`。`prelude_blocks` 和 `num_blocks` 仍是
当前 op 的 CPU 布局参数。`tau` 和 backend 私有 scratch 按可共享性分别绑定，现有
图 key 保留 op、深度、布局和后端约束。

由 Runner 管理一套设备 workspace 和当前 `rows` 的关联，Graph entry 只持有设备
视图与原生 wrapper。同一批连续执行时，各 op 就地交接 `_InklingState`，省去
`stage_inputs` 和整块 stack 回拷。旧批仍有未完成 span 且新批要使用该 workspace 时，
先把旧批活跃行写回现有 `LoopExecution.state[request_id][token_start]` 的独立
`_InklingState`，再用设备 gather/scatter 装载新批。首版允许保留当前逐 entry staging
作为变批回退。buffer 与 graph 录制时绑定，所有覆盖先按同一 compute stream 排序。
这项用 NVTX 和 CUPTI 分别归因 metadata、跨轮 hidden 与算子内部的 D2D，保留逐 op
图和 eager 对照。

### 优化三 延后特性路径的 CPU 等待

普通 Coda 直接保留 MRv2 的异步输出。合批 Coda 用同一批采样结果区分 draft 与
target。Worker 把目标行照常交给 MRv2，草稿保持设备侧候选，通过异步回执带回
scheduler 真正需要的部分。多个 subgroup 聚合时分别保留各自输出及 event，到
`get_output()` 再聚合，移除 Worker forward 中的同步 `.tolist()` 和 `get_output()`。
GPU sampler 状态只由目标行更新，CPU 只提交目标回执中的 token。自投机拒绝时按原有位置回滚。

Ouro 在 GPU 生成逐行早退 mask，并按 mask 完成相应深度的 KV 补写。只异步回读
scheduler 决定后继需要的退出信息。这条回执仍是该行继续或退出的真实依赖，等待
期间允许提交其他已就绪的行。Ouro 按 gate、退出深度、KV 和输出验收。

### 实施顺序与验收

1. 在启动预热修正后的同一配置上重录 W1。追踪 `schedule → worker → graph launch`、
   Coda 输出 Future 入队/就绪/回收、`prepare_inputs`、graph input staging 及 copy
   类型。对照异步深度 2，确认 graph 命中和设备执行基线。
2. 单独实现真实 token 向前传递。用 trace 验证下一 Prelude 已提交时，Coda 的 CPU
   结果尚未回收，并测 Coda → Prelude 间隔。逐 token 对照 greedy 输出和 logprobs，
   覆盖 EOS、stop、长度上限、取消、抢占、slot 复用和同期混合 1/3/默认轮数。
3. 单独测输入复用和 graph staging。W1 的同 token 后继 op 应能复用输入，观察每
   token 的输入准备次数、H2D/D2D 次数和字节、graph 间隔、TPOT 和 ITL。用 W3
   及变批的 C16/C32 验证缓存失效、ragged、混深度、KV 一致和 pinned slot 复用。
   队列深度增加需要单独 A/B，比较 TTFT 尾延迟、KV 和 pinned memory 占用。
4. 基础路径独立验收后再接草稿和早退。W3/W7 先重查基线的自然文本输出分歧，
   固定回放、精度、可用 KV 和优化开关，分开记录 correctness 与 profiling。
   正式测速关闭 profiler，仍以 Inkling 吞吐与 TTFT P95 达到原生 95% 为门槛，
   WCPB、自投机及 Decode pipeline 的收益另测。


### 异步交接与输入复用的首轮实现

- `LoopAsyncScheduler` 允许普通 Inkling decode 在上一 Coda 等待 CPU 回执时预排一个
  Prelude。复用 MRv2 `last_sampled_tokens`、原生 output Future 和已有 KV 块分配。
  预排缺少 KV 时等待上一回执。CPU 用例覆盖正常确认、stop token、取消、长度上限
  和过 KV block 边界。
- `LoopModelRunner` 保存一条 `_PreparedDecodeBatch`。首版只有普通单 token decode
  才可命中，缓存 key 用 `(request_id, MRv2 slot, token_start)` 和 batch descriptor。
  单一记录直接清空表示失效。请求、KV 或采样状态变化走原 MRv2 准备路径。首次准备
  继续使用 MRv2 的 H2D 与块表增量更新，续轮复用 `InputBatch` 和已准备的 attention
  输入。现有模型 metadata builder 仍每步调用。UVA 复用上游长度和块表增量缓冲。

同一张 B300 GPU 7，固定三轮 BF16，80 GiB KV，原有基本优化全开。各组先热身，再跑
三波 W1 C1，每波 8 条 128→128。吞吐和 TPOT 取三波中位数。计时窗内的 GPU 进程
采样显示前三组只有各自的 RLT Worker。

| 改动 | W1 吞吐 tok/s | TPOT P50 | 与修改前逐 token 一致 |
|---|---:|---:|---:|
| RLT 启动预热版本 | 137.4 | 7.21 ms | 24/24 |
| 加设备侧 token 交接 | 177.9 | 5.54 ms | 24/24 |
| 再加 MRv2 输入和块表复用 | 198.8 | 4.95 ms | 24/24 |

NVTX 诊断把 `host.loop.plan.prelude` 的位置与 `AsyncOutput.get_output()` 对齐。
保留实现的 W1 trace 采到 38/38 次在上一 Coda 开始回收前已提交下一 Prelude，
每 token 一次 MRv2 输入准备、两次 H2D、五次 Graph replay。Coda → Prelude
的 Graph 间隔中位数为 0.97 ms，早先基线 trace 为 2.68 ms。此处是 profile 诊断，
吞吐使用独立的无 profiler 波次。

共享 Graph 输入地址的第一版 W1 为 177.9 tok/s，减少 Host metadata 装配后为
192.1 tok/s，仍低于输入复用组。两版均从工作区撤回。保留逐 op Graph staging，
同一 metadata 对象的 tensor 字段只枚举一次。直接共享 Graph 设备工作区仍需达到
吞吐与 TTFT 对照再接回。W3 短测各组均完成，但相对修改前仍有请求输出分歧，
正式 95% 性能门槛尚未重新配对验收。

保留实现另跑 GPU 对照检查。缓存命中时用原 MRv2 路径重建输入和 attention
准备结果，比较 `input_ids`、`positions`、`seq_lens`、offsets、logits 索引、
各组块表与 slot mapping。超过 1,000 次输入及 attention 对照一致，包含多请求批。
CPU 回归 191 通过、195 跳过。Ruff 和 `git diff --check` 通过。

[W1 配对数据及 trace](../artifacts/plugin2-perf.9pulvo/async-opt/summary.json)


### 异步优化后的剩余间隔

复用已有 Nsight 采样。W1 从一个完整 decode token 的首张图开始，到下一 token
首张图为止，按 CUPTI correlation ID 关联 Host CUDA launch 和 Device graph。
原生有 35 个完整间隔，优化前、后 RLT 各 38 个。单位为 ms/token。

| Device 区间 | 原生 | RLT 优化前 | RLT 当前 |
|---|---:|---:|---:|
| Graph 或 kernel 覆盖 | 3.72 | 4.01 | 3.92 |
| Graph 与 kernel 均未覆盖 | 0.20 | 4.73 | 2.11 |

当前 RLT 与原生的 profile 差约 2.10 ms/token，其中约 1.90 ms 在 Graph 和 kernel
覆盖区间之外。未覆盖区间仍可包含 memcpy。图级 trace 的覆盖时间含框架 kernels。

| 相邻 Graph 中位间隔 | RLT 优化前 | RLT 当前 |
|---|---:|---:|
| Prelude → CORE 1 | 1.02 | 0.78 |
| CORE 1 → CORE 2 | 0.76 | 0.40 |
| CORE 2 → CORE 3 | 0.32 | 0.10 |
| CORE 3 → Coda | 0.23 | 0.09 |
| Coda → 下一 Prelude | 2.68 | 0.98 |

前三处值得进一步拆的边是 Coda → Prelude、Prelude → CORE 1 和 CORE 1 → CORE 2。
在这些边上，Host 调用下一次 `cudaGraphLaunch` 的中位时间分别比上一图结束晚
0.97、0.78、0.40 ms。后两条 CORE 边已观测到提交与 GPU 执行重叠。

`_GraphEntry.stage_metadata()` 的 Host NVTX 区间累计为 1.83 ms/token，包含 Python
组装与 D2D 提交。用 CUPTI 关联到这一区间的 memcpy 约 240 次/token，GPU copy
用时合计约 0.24 ms/token。另有约 21 次 Graph 输入 D2D、30 次 staging 区间外 D2D。
Host 区间可与 GPU 执行重叠。原生 MRv2 输入生成和块表准备各一次，分别约 0.18 和
0.06 ms/token。采样 Host 区间约 0.41 ms/token，原生约 0.40 ms。Coda 回执的
Host `get_output` 约 0.009 ms/token。Graph 间隔本身还包含采样、输入准备及小拷贝。
[W1 分解数据](../artifacts/plugin2-perf.9pulvo/async-opt/w1-gap-breakdown.json)

旧 W3 trace 还有合批和 Graph 覆盖差距。RLT 每次 prefill op 只有一行。44 次纯
decode CORE 中，11 次单行 replay，22 次 16 行混深度走 eager，11 次 15 行同深度没有
命中捕图档位。后两类区分基于 per-op NVTX 和 CUDA launch，15 行不在该轮配置中。
这是异步改动前的 trace，当前 W3 的占比尚未重抓。
[旧 W3 Host / Device 时间轴](../artifacts/plugin2-perf.9pulvo/timeline/host-device.html)


## 当前 RLT 与原生 vLLM 配对对照

使用固定 revision `3146a9df4140` 和同一份当前 RLT 源码快照。两边同一张 B300
GPU 2 顺序运行，Inkling 固定三轮 BF16、TP1、异步调度、1024 token 预算、32
请求上限、80 GiB KV，实际都报告 291,018 个 KV token。基础开关与首轮对照相同，
每档先热身再用相同请求回放，无 profiler。W1 取三波中位数。W3 的统计取同时完成
的前两波，原生第三波为 1013.4 tok/s，RLT 第三波 CUDA illegal memory access。

| 负载 | 原生 / RLT 吞吐 tok/s | RLT / 原生吞吐 | TTFT P50 原生 / RLT | TTFT P95 原生 / RLT | TPOT P50 原生 / RLT | 完整输出 ID 一致 |
|---|---:|---:|---:|---:|---:|---:|
| W1 C1 | 260.6 / 208.0 | 79.8% | 11.68 / 17.25 ms | 11.94 / 20.96 ms | 3.78 / 4.62 ms | 24/24 |
| W3 C16 | 1012.6 / 179.7 | 17.7% | 154.0 / 1273.5 ms | 1232.7 / 1297.0 ms | 13.42 / 82.74 ms | 55/64 |

W3 的 RLT 两波为 207.8、151.6 tok/s。第三波报错在下一次 `gather_block_tables`
CUDA 提交时上报，具体触发 kernel 待定位。以相同配置重新启动进行一波独立重试，
完成全部请求，吞吐 145.6 tok/s。重试独立保存，表格仍用原始完成的配对波次。
[原始跑分与失败记录](../artifacts/plugin2-perf.9pulvo/native-pair/timed/)

### W1 的差距在 Host 重复准备和逐 op 提交

GPU 2 独立采集双方各 38 个完整的稳态 decode token。原生一次 `execute_model`
构建一次 metadata，随即 replay 整模型图。RLT 主输入与块表准备已降到一次，
五个 op 仍各调用一次 metadata builder 和逐图 staging。

| 每输出 token | 原生 vLLM | 当前 RLT |
|---|---:|---:|
| CUDA Graph replay | 1 | 5 |
| GPU Graph 或 kernel 覆盖 | 3.74 ms | 3.89 ms |
| GPU 没有 Graph 或 kernel 覆盖 | 0.23 ms | 2.00 ms |
| Host `prepare_inputs` 加块表与 slot 准备 | 0.29 ms | 0.24 ms |
| Host MRv2 metadata builder | 0.23 ms | 1.09 ms |
| Host RLT `stage_metadata` | 无对应调用 | 1.76 ms |
| Host scheduler plan | 0.069 ms | 0.124 ms |
| Host 采样 | 0.41 ms | 0.37 ms |
| Host `AsyncOutput.get_output` | 约 0.009 ms | 约 0.009 ms |
| CUPTI 记录的 D2D memcpy | 0 | 291 次 |

CUDA API 相关性把约 239 次 D2D 归到 RLT `stage_metadata`。Host 区间可以与 GPU
工作重叠，按区间拆解归因。图或 kernel 覆盖差 0.15 ms/token，间隔差 1.78 ms/token。
W1 的 prefill 短 trace 同样显示执行切分，原生一次 forward，RLT 五次。从 Worker
开始执行到首 token CPU 回收，分别为 15.25 ms 和 18.75 ms。

### W3 的差距在 prefill 与 decode 的合批及图覆盖

GPU 7 独立采集相同的 16 条 4096→128 请求。原生在含 prefill 的 65 个 forward 中
同时处理已进入 decode 的请求，活跃请求数中位数 9。RLT 的 64 个 prefill chunk
经过 320 个逐 op forward，每次一行。原生可以更早交付已完成 prefill 的请求，
对应无 profiler W3 的 TTFT P50 为 154 ms，RLT 为 1273 ms。

原生的 127 次纯 decode forward 全部提交 Graph，批大小中位数 16。RLT 有 510 次
纯 decode CORE，只有 129 次单行提交 Graph。其余 252 次是 16 行混深度 eager，
129 次是 15 行同深度未命中图档位。原生 `CudaGraphManager.dispatch` 会选择覆盖
真实批量的已捕图档位。RLT 当前逐 op 图检查实际 token 数是否在配置列表里。
采样取双方第一波完整 W3。原生在 profiler 下的第二波停顿，独立无 profiler 的
三波均完成。

[最新分段时间轴](../artifacts/plugin2-perf.9pulvo/native-pair/trace/host-device.html)
和 [配对 trace 汇总](../artifacts/plugin2-perf.9pulvo/native-pair/trace/summary.json)
记录 W1 首 token、稳态 decode 和 W3 prefill、过渡及 decode 中段。


## Metadata 与高并发 Graph 优化方案

先用现有 W1 捕图尺寸压低 metadata builder 和 staging，再用 W3 分别验证同深度
padding 和混深度执行。每步维持一个 loop op 的 EngineCore 调度边界。W1 每 token
的 builder 和 staging Host 区间分别是 1.09 ms 和 1.76 ms，两项都值得独立测量。

### 1. 同一批输入的 metadata 只建一次

在现有 `_PreparedDecodeBatch` 上追加一份 backend metadata 引用。该记录已有实际
行顺序、请求槽、token 位置、`InputBatch`、gather 后的块表和 slot mapping。通过
`LoopExecution` 把当前准备记录交给 Inkling `get_model_state_cls()` 注册的模型状态。
模型状态继承上游 `DefaultModelState`，首次调用原生 `prepare_attn()`，把它返回的
layer → backend metadata 存进记录。续轮命中时返回同一份映射。`LoopSpan.depth`
仍由每次提交提供，对应层照常读取当前轮的 KV 和短卷积缓存。

有效范围沿用单条准备记录。同一份 `InputBatch`、块表和 slot mapping、相同的
descriptor 及 backend 构建模式才能复用。换位置、变批、slot 重用、KV 块变更、
采样推进、dummy/profile 或其他路径覆写 builder scratch 都使记录失效。首版覆盖
普通同深度单 token decode。ragged prefill 的 FA4 CPU 长度计划和混深度 per-group
mask 仍按实际布局构建。W1 目标从每 token 5 次 builder 降到 1 次，配对检查
各 KV group 的 offsets、slot mapping 和 SConv metadata。

### 2. Graph 直接消费稳定的 metadata 地址

按字段来源处理当前 `_GraphEntry.stage_metadata()` 的约 239 次 D2D/token。

| 字段 | 处理方式 |
|---|---|
| `query_start_loc`、`seq_lens`、gather 后的块表、`slot_mapping` | 这批已由 MRv2 写进常驻设备缓冲。warmup 与 replay 使用同一地址及相同布局时，让同档位的各 op 直接引用这些设备视图。 |
| SConv 的 `seq_idx`、`query_start` | `InklingSconvMetadataBuilder` 已持有常驻 buffer。用真实 batch 的 offsets 更新一次。只在相同 builder 实例、相同地址和消费者生命周期均确认后直连。 |
| 重新生成的 backend 临时 tensor | 使用固定设备目标，以 capture 时登记的字段绑定更新。按各字段的实际更新频率做一次 staging，多个只读 op 共用同一更新结果。格式或地址不兼容时用已有保守路径。 |
| PIECEWISE attention 仍要用的 CPU 标量和实际 ragged 布局 | 保留 runtime metadata，由 breakpoints 使用。FULL Graph 从预先绑定的 GPU 地址读取。 |

`_GraphEntry` 在 capture 时确定字段和地址。FULL replay 只校验可复用的绑定并更新
仍需复制的字段，省去每个 op 全量遍历 dataclass、按层重建字典和逐字段 `copy_`。
仍需复制的共享字段由 Runner 在同一 compute stream 生产，Graph 按入流顺序读取。
首轮把 metadata 直连和 `_InklingState` 工作区改动分开验证。之前同时试验共享
缓冲区的两个版本 W1 为 177.9/192.1 tok/s，低于输入复用组 198.8 tok/s。这里以
builder 次数、metadata D2D 次数和无 profiler 的吞吐、TTFT 共同决定是否保留改动。

### 3. 同深度 decode 使用可覆盖的 FULL 档位

RLT 的精确匹配来自当前接入位置。`LoopCudaGraphs.run()` 在模型层收到已经按真实
行数准备好的 tensors，再按 `tokens in self.sizes`、每个 span 长度和 tensor layout
查图。此处把 15 改成 16 时，attention metadata、hidden 和输出行的形状需要一起变。

在 Runner 准备输入前，使用 MRv2 `BatchExecutionDescriptor` 的形状语义选档。
当前 `1/2/4/8/16/24/32` 已覆盖 1–32 的单 token decode。预热按 op、CORE 变体及
档位捕图，运行时取能容纳实际批量的最小兼容图。例如 15 行使用 16 行档。图 key
保留 op、loop depth、`last_core` 引起的实际执行变体、物理档位和 backend 布局。
真实请求数、请求 ID 和位置通过输入数据传入。

| 15 行使用 16 行图 | 真实和物理形状 |
|---|---|
| Scheduler 和 `LoopExecution.ordered` | 15 条真实行。只有它们参与 KV 分配、hidden 归属、采样和输出提交。 |
| MRv2 `InputBatch` | `num_reqs=num_tokens=15`，`num_reqs_after_padding=num_tokens_after_padding=16`。复用 MRv2 的 `prepare_inputs`、`BlockTables`、`PAD_SLOT_ID` 和 `is_padding`。 |
| Padding attention 行 | `query_start_loc=[0, 1, …, 15, 15]`，尾行 `seq_lens=0`，块表按 MRv2 空块规则准备，slot 为 `PAD_SLOT_ID`。验证 SConv 也跳过尾行，MoE 使用 `is_padding`。 |
| Graph 的 Inkling 输入及输出 | `positions`、stack、bank 等设备 view 按物理档位准备，尾部处理遵循各 backend 的 padding 契约。保存 hidden 和采样时沿用实际行映射与 MRv2 `logits_indices`。 |

在固定接口下，Runner 的 `prepare_inputs` 接收所选物理 descriptor，模型状态针对
该逐 op descriptor 构建 padding-aware metadata。上游 `execute_model` 继续进入
RLT adapter，由 RLT 在 op 层 replay。复用原生 `CudaGraphManager.dispatch` 的
兼容规则和 `prepare_inputs_to_capture` 的 dummy 形状。实际捕图的变体、地址绑定
及 replay 属于逐 op 执行器。增加档位由捕图显存、启动延迟和实际 padding 代价决定。
W3 可先加 15 档做单独诊断，并保留 W1 与 W3 prefill 的既有档位。覆盖版验收时显式
观察 15→16、回缩及变批，并用 eager 比对输出、各深度 KV 和 SConv 写入。

### 4. 混深度和 ragged 分别验证 Graph 覆盖

当前 W3 的 129 次同深度 15 行 miss 属于档位问题。252 次混深度 CORE 已经有 16 行，
`execute_mixed_core` 按 `(depth, last_core, propose_draft)` 动态分组并构造 GPU
行索引和各组 metadata。先记录深度组合。用相同流量对照按深度拆批进 FULL 图和
现有共享投影/MoE 的混深度 eager，测 TPOT、吞吐及同请求输出。长期用 PIECEWISE
先捕稳定的整批 projection/MoE，再把随深度和组大小变化的 attention/SConv 调用
留在 breakpoints。固定设备侧路由和分支形状经过验收后，再考虑混深度 FULL。

Ragged prefill 和 prefill/decode 同 op 合批另用 PIECEWISE 检查。物理 token 档位
与真实 `query_start_loc` 分开，attention 的 CPU plan 仍看真实 seq/query lengths。
W3 的 TTFT P50 还要通过缩短早到请求从 prefill 到首次 decode 的等待来改善。并发
回归同时检查多波请求、缺图回退和上轮出现的 CUDA illegal memory access。

### 分阶段实现与实测

Inkling 模型状态通过 `get_model_state_cls()` 接入，首次走原生 metadata builder。
`PreparedDecodeBatch` 持有这次准备的输入、块表、slot mapping 及 metadata，后续同批
op 共用它。相同物理档位的 FULL 图绑定一份共享的 metadata 缓冲，每份新准备记录
只刷新一次。保留已有 PIECEWISE metadata staging 和 hidden 状态交接。

GPU 2 独占顺序运行 W1，每组三波、每波 8 条 128→128。各组统计为三波中位数。

| 阶段 | 吞吐 | TTFT P95 | TPOT P50 | 与基线逐 token 一致 |
|---|---:|---:|---:|---:|
| 修改前 RLT | 221.64 tok/s | 15.28 ms | 4.43 ms | 基线 |
| Metadata builder 复用 | 232.27 tok/s | 15.62 ms | 4.22 ms | 24/24 |
| 加 FULL metadata 共享 staging | 247.42 tok/s | 17.79 ms | 3.94 ms | 24/24 |

单独 NVTX 采集的 38 个稳态 token 显示 builder 从 5 次降到 1 次，Host 区间从约
1.09 降到 0.22 ms/token。`stage_metadata` 的 D2D 从约 239 降到 47 次/token，
Host 区间约 0.26 ms/token。五张逐 op 图每 token 仍分别 replay。
[W1 测速](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/timed/w1-summary.json) ·
[Metadata trace](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/trace/staging-summary.json)

Runner 对已完整捕图的 FULL decode 档位使用容量匹配。实际行数留在原生 `InputBatch`
和 `LoopSpan`，在准备 GPU 输入时复用 MRv2 的 padding 视图。Inkling 的 per-op hidden
按物理档位计算，保存状态和采样沿用真实行。在独立的 GPU 覆盖采样里，受控 3→4、
15→16、回缩 3→4 各出现 155 次逐 op replay。单波 W3 采到 381 次 15→16 replay。
混深度 CORE 仍沿用现有路径。
[Padding 覆盖](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/trace/padding-summary.json)

之前的 W3 trace 中，RLT 有 123/1336 次块表 gather 在 GPU 开始前，已有至少两个
后继 forward 完成 Host 提交，最多七个。原生样本的 192 次都未出现。MRv2 的 UVA
缓冲按两槽轮换，使用者必须先让旧的 GPU 读取退役。RLT 现已按实际 MRv2 forward
轮换 CUDA events，多个 subgroup 分别计数，在复用槽前等待对应事件。CPU 回执和
GPU 完成仍各自追踪。该保护独立于 metadata 和 padding，使用同配置 W1、W3 复测。
[UVA 提交顺序](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/trace/uva-order-before.json)

GPU 6 受控同批验证比较了容量匹配、真实行数匹配和 eager。`3→4、15→16、3→4`
的 21 条请求共 672 个 token，在两组对照中逐 token 一致。Padding 行的 slot
均为 `PAD_SLOT_ID`，采样索引全部指向实际请求。
[Padding 正确性对照](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/trace/padding-id-parity.json)

### UVA 保护后的同卡复测

GPU 2 使用相同 CPU affinity 和 warmup 流程。W3 的原生 vLLM 和 RLT 默认混深度
路径各完成 32 条热身及三波正式请求，每波 32 条。RLT 三波均未复现非法访存。

| W3 配置 | 吞吐中位数 | TTFT P50 | TTFT P95 |
|---|---:|---:|---:|
| 原生 vLLM | 1013.22 tok/s | 153.79 ms | 1230.98 ms |
| RLT 混深度 CORE | 381.16 tok/s | 1281.11 ms | 1299.44 ms |
| RLT 按深度拆开 CORE | 690.99 tok/s | 1277.61 ms | 1295.60 ms |

三波 RLT 吞吐分别为 382.27、381.16、244.89 tok/s，采样到的 GPU 进程和频率
一致，尾波差距保留在结果中。RLT 对原生逐请求的输出分别有 27/32、29/32、29/32
完全一致。W3 的前 16 条逐 token 一致，差异出现在后补的请求。按深度拆批同样完成
全部三波，吞吐较混深度提高 81.3%，达到原生的 68.2%，TPOT P50 从 33.39 降至
14.31 ms。原生可以在 prefill 过程中陆续返回首 token。RLT 的两批请求大多集中到约
1.3 s 返回，TTFT P50 的差距在拆批后仍存在。W3 的性能和逐 token 验收尚未通过。
[UVA 保护后的同卡对照](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/timed/uva-event-summary.json)

GPU 2 独占的默认配置 W1 复测中，RLT 的吞吐中位数为 237.37 tok/s，原生为
260.75 tok/s。RLT 达到原生的 91.0%，相较本轮修改前的 221.64 tok/s 提升 7.1%。
TTFT P50/P95 为 14.06/14.67 ms，原生为 11.69/12.09 ms。24 条请求的输出逐 token
一致。较早的 247.42 tok/s 是加 UVA 保护前的阶段结果。

### W3 首 token 的分布

W3 的并发上限为 16，客户端在一条请求生成完 128 个 token 后才补入下一条。
取第三波的请求级结果，各个 16 条请求组的首 token 数据如下。整波 P50 用原始的
32 条请求直接计算。

| W3 配置 | 首批 16 条最早 / P50 | 后补 16 条最早 / P50 | 整波 P50 |
|---|---:|---:|---:|
| 原生 vLLM | 87.1 / 723.6 ms | 112.5 / 152.5 ms | 153.8 ms |
| RLT 混深度 CORE | 1183.1 / 1277.5 ms | 1185.5 / 1288.6 ms | 1283.0 ms |
| RLT 按深度拆开 CORE | 1143.1 / 1273.9 ms | 997.1 / 1281.4 ms | 1277.6 ms |

每个 4096-token prompt 拆成四个 1024-token chunk。原生 trace 显示已有的 decode
与尚在跑的 chunk 可进入同一次 forward。RLT Scheduler 按 `_order` 扫描请求，提交后
把该请求移到队尾，由每步选中的第一个 op 决定该步还能接收哪些 span。一个 chunk
需要先后经过 `PRELUDE → CORE×3 → CODA`，最后一个 chunk 的 CODA 才会出首 token。
RLT 的原始 W3 trace 中，64 个 chunk 共提交 320 次单请求 forward。按计划的
`token_start` 对齐到 W3 的第一条 worker 记录，首个 `token_start=3072` 的 CODA
到约 1363 ms 才提交。调度在不同请求的中间 chunk 之间轮转，最后一个 chunk 集中
排到后面。关掉混深度 CORE 主要改善首 token 之后的生成速度，两组 TTFT 分布相近。
[W3 分段时间轴](../artifacts/plugin2-perf.9pulvo/native-pair/trace/host-device.html)

用旧版 RLT 的 CUPTI/NVTX 记录对齐首次 `token_start=3072` 的 CODA。它之前已经
执行 48 个中间 CODA，正好是 16 条请求各三个非末尾 chunk。期间有 48 次
`sample_tokens()` 和 48 次 `AsyncOutput.get_output()`。对应 Host 记录分别累计
18.56 ms 和 104.82 ms。`get_output()` 会等此前提交的 GPU 工作，区间可能重叠，
累计时间不能直接当成删掉采样路径后会得到的收益。当前 Worker 仍让普通 CODA
调用 MRv2 sampler，最终版配对 trace 需要核对这条路径及新增的 UVA 等待。
[旧版 RLT Nsight Systems 记录](../artifacts/plugin2-perf.9pulvo/native-pair/trace/rlt-basic/inkling.nsys-rep)

### 当前版本 W3 CUPTI/NVTX

GPU 3 上先跑 32 条热身，再采正式 32 条 W3，最大并发 16。原生的 CUDA profiler
启动接口在热身后未返回，后续使用常驻 Nsight，按请求 ID 提取正式波次。性能门槛
使用独立的无 profiler 数据。

表中的 mixed CORE 对应 `rlt_mixed_core`，对比的是 decode CORE 是否按 loop depth
拆批。三组 W3 都用同一个实例处理 prefill 和 decode。下文的 prefill/decode 合批
指同一次物理 forward 同时有两类 token。

当前默认 mixed CORE 的 RLT 已采到全部 32 条的入队、首次调度和首 token。首批
16 条的请求级中位数为入队到首次 Worker 23.73 ms，首次 Worker 到最后一个
prefill CODA 1283.87 ms，末段 CODA 提交到 EngineCore 产出首 token 5.58 ms。
正式波的 640 次含 prefill 的物理 forward 都是单请求；第一个可能返回首 token
的 CODA 前，执行了 48 个中间 CODA，分别走过 sampler 和结果回收。两个 Host
区间累计 17.50 ms 和 0.42 ms。在首条入队至首 token 期间，`_submit_forward()`
内 UVA 退役事件的 `cudaEventSynchronize` 共 291 次，Host 区间覆盖 274.09 ms，
其中包含等待已提交 GPU 工作的时间。
[当前 RLT Nsight 报告](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/trace/w3-final/mixed-on-continuous/inkling.nsys-rep)

同卡原生 vLLM 的常驻采集保留了全部 32 条的首 token 标记，末尾四条请求在
已有首 token 后停止推进。正式窗口没有导出 CUPTI 设备事件，以下原生数据取
Host 调度标记。首批 16 条请求逐请求计算，各列独立取中位数。

| 配置 | 入队至首次 Worker | 首次 Worker 至最后一段 prefill 开始 | 最后一段 prefill 开始至 EngineCore 首 token | 入队至 EngineCore 首 token |
|---|---:|---:|---:|---:|
| 原生 vLLM | 600.97 ms | 84.69 ms | 41.86 ms | 727.40 ms |
| RLT 混深度 CORE | 23.73 ms | 1283.87 ms | 5.58 ms | 1313.21 ms |
| RLT 按深度拆开 CORE | 28.36 ms | 1362.45 ms | 6.64 ms | 1401.25 ms |

原生计划中有 127 次含 prefill 和 decode 的 forward。RLT 的 640 次含 prefill
的 forward 均没有混入 decode。RLT 把首批请求很快分发到 GPU，但多数请求直到
约 1.3 s 才从第一段推进到末段。原生逐个开始，开始后的请求较快推进到最后一段。
在 GPU 4 上补抓原生 NVTX，正式 32 条请求全部完成，首批的首次 Worker 至最后
一段 prefill 开始中位数为 83.57 ms，观察到相同的 127 次 prefill/decode 合批。
关闭 mixed CORE 后，GPU 3 上依然在首 token 之前执行 48 个中间 CODA，正式波的
640 次含 prefill forward 也仍单请求。
[原生 Nsight 报告](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/trace/w3-final/native-continuous/inkling.nsys-rep) ·
[原生完整 NVTX](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/trace/w3-final/native-nvtx-gpu4/inkling.nsys-rep) ·
[按深度拆批 Nsight 报告](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/trace/w3-final/mixed-off-continuous/inkling.nsys-rep) ·
[请求级区间与合批记录](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/trace/w3-final/request-summary.json)

### Prefill/decode 同批定点验证

当前 Inkling2-mini TP1 路径中，A 先生成 8 个 token，再提交 B 的 128-token prompt。
Worker 和 Inkling 模型在同一次实际 forward 分别收到 A 的 1 个 decode token 和 B
的 128 个 prefill token。`PRELUDE`、三个深度的 `CORE` 和 `CODA` 共五次这样执行，
物理 token 数都是 129，均走 eager。与相同请求独立运行比较，B 的 24 个 token
逐 token 一致，A 的第 39 个输出发生分歧，当时混批 forward 已经结束。此例确认了
同 op 的 P/D 合批执行路径，输出一致性仍需定位。W3 的两个 RLT 配置均观察到
640 次单请求 prefill forward，P/D 同批次数为零。
[两请求 GPU 验证](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/trace/pd-forward-probe/results.json)

### 先推进已开始请求，再接新请求

`LoopScheduler.schedule()` 先扫描 `RUNNING`，再扫描 `WAITING` 和 `PREEMPTED`。
各组按请求到达顺序扫描。发出一个 op 后，请求继续保有原来的顺序。一个计划仍由
第一个可运行 span 决定 op，后续同 op 的 span 共享剩余 token 预算。新请求用
现有 KV 余量入场，正在运行的请求扩展 KV 时仍可按原有条件抢占。

运行中的请求如果要等待 CPU 或 GPU 结果，调度器可以先提交其他就绪工作。
`max_num_active_seqs` 限制新请求入场，`max_num_scheduled_tokens` 约束调度预算，
上限为 Runner 的 `max_num_batched_tokens`。WCPB 继续按前一 chunk 的已完成深度
开放工作，自投机与 Decode pipeline 继续按 draft 和 target 的依赖决定何时推进、
保留或回滚。同步和异步都覆盖了长 prompt 先完成已有 chunk、decode 先取 1 个
token、后续请求填余量的场景。

GPU 4 同卡独占 A/B 保持 `rlt_mixed_core=true` 和 `rlt_packed_prefill=true`，每组满波
热身后跑三波，以下取中位数。CPU 插件回归 47 通过、2 跳过。

| 负载 | 旧 / 新吞吐 | 旧 / 新 TTFT P50 | 旧 / 新 TTFT P95 | 旧 / 新 TPOT P50 |
|---|---:|---:|---:|---:|
| W1 C1 | 236.05 / 232.82 tok/s | 14.92 / 14.29 ms | 15.58 / 15.08 ms | 4.15 / 4.22 ms |
| W3 C16 | 178.95 / 597.81 tok/s | 1284.44 / 495.91 ms | 1305.73 / 2376.31 ms | 80.27 / 20.22 ms |

旧版 W3 吞吐波动较大，三波为 178.95、385.01、178.45 tok/s。新版本相较旧版
最快一波仍提升 55.3%。W3 首批 16 条的 TTFT P50 从 1276.81 升到 1397.80 ms，
续入请求从 1292.13 降到 395.04 ms，整波 P50 的改善主要来自续入，P95 有明显退化。
W1 输出 24/24 与旧版逐 token 一致。W3 新版三波自身逐 token 一致，对旧版的三波
分别为 31/32、30/32、31/32，差异出现在续入组。新旧开启 WCPB、自投机和流水线的
两个 smoke 请求均完成，输出逐 token 一致，分别出现 6 次同请求混深度 CORE、
8 次 draft CODA 和 7 次使用候选输入的 PRELUDE。
[GPU 4 速度和输出对照](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/scheduler-running/summary.json) ·
[Loop 功能回归](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/scheduler-running/smoke-summary.json)

GPU 7 额外关闭混深度 CORE、保留 packed prefill，旧版和新版各跑两波。W3 吞吐从
780.75 降到 586.94 tok/s，TTFT P50 从 1273.20 降到 510.50 ms，P95 从 1291.21
升到 2447.46 ms。新旧两波均为 30/32 一致，差异仍在续入组。两档有相同方向的
TTFT 变化，吞吐收益取决于 CORE 路径。性能和输出验收尚未通过。
[关闭混深度 CORE 的对照](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/scheduler-running/mixed-off-summary.json)

#### W3 TPOT、吞吐和尾延迟的归因

GPU 7 在两份调度快照上用相同旧 UVA event 保护，关闭混深度 CORE，保留 packed
prefill。同卡 CUDA/NVTX 各覆盖 32 条热身与 32 条正式请求。旧调度有 3815 次物理
forward，3175 次纯 decode 走 FULL、640 次单请求 prefill 走 PIECEWISE。新调度
减少到 2004 次，其中 1334 次纯 decode 走 FULL、40 次 prefill 走 PIECEWISE，另
有 630 次 P/D 混批全部走 eager。630 次里有 598 次总 token 数等于已捕图的 1024。
典型布局为 1013 个 prefill token 和 11 个 decode token，无法命中单请求 1024-token
prefill 的图 key。32 条请求中 30 条从四个 prefill chunk 变为五个。126 次混批
CODA 含 1208 个 decode 行，占正式波 decode CODA 行的 29.7%。这些 decode 要等
同批的 prefill 计算。trace 中 mixed CORE 的 Worker Host 区间 P50 是 12.34 ms，
旧单请求 prefill CORE 为 3.25 ms。CUPTI 的正式窗口缺完整 Graph 活动，Host 与
执行模式可用于归因，设备记录的累计覆盖不直接作为各图的 GPU 耗时。

另有已就绪的长 span 因 token 预算反复跳过的情况。首批请求 1 的首段
1024-token prefill 从最后 CORE 的 Host 提交结束到 CODA 开始间隔 3275.76 ms。
续入请求 16 的首段 1023-token chunk 同样有 2417.06 ms 间隔。先选到的 decode
消费预算后，已经固定长度的 span 不能放入剩余位置。后续调度机会持续重复这一
选择。这个等待独立于 Graph 的覆盖，需对已就绪、因预算落选的 span 保证服务。
op 选择优先考虑能够交付或解锁后续工作的已就绪 span。组批时，已预留且因预算
被跳过的长 span 在下一次匹配的 op 有机会先取预算。这个规则针对已观察到的等待，
也为 CORE 等中间工作保留推进机会。
[W3 物理 batch 与图模式](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/trace/w3-final/scheduler-profile-comparison.json) ·
[长 span 的 CORE 到 CODA 间隔](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/trace/w3-final/scheduler-prefill-coda-gaps.json)

### 缩短 UVA 槽的占用时间

MRv2 的 `UvaBufferPool` 在 CPU 上轮换写入 pinned 槽。`BlockTables.num_blocks`
由 `apply_staged_writes()` 发布，`_gather_block_tables_kernel` 读取它后把结果写入
设备侧块表。后续的 slot mapping 和模型 attention 消费 GPU 侧表。`prefill_len`
除了供输入准备读取，还在 `Sampler` 中用于计算采样和拒绝数量，因此准备输入后立即
释放所有池的槽不安全。

首个优化针对单卡 Inkling 的 PRELUDE/CORE 无采样路径。`LoopInklingModelState`
复用 MRv2 准备模型输入，在返回前通过 `LoopExecution` 通知 Runner 记录槽位 event。
此时相关 GPU 读操作已在原 compute stream 入队，后续模型计算仍可以异步执行。
环形队列复用槽时等待同一个 event，下一次提交保留已记录的时点。CODA、投机和其他
未做生命周期验证的路径等到原来的退役点。另一个可以独立测的改动是在分组块计数
发生变化时才刷新 `BlockTables.num_blocks`，避免每个 op 轮换一次相同数据。

用同一 workload 比较槽复用等待、首 token 前 GPU 空档、吞吐和 TTFT 分位数，确认
在变批、释放与复用 KV 块、多波请求及投机回滚时的输出。把 event Host 时间与无
profiler 变化一起看，避免把仍需完成的 GPU 工作计为可消除的等待。

GPU 4 独占无 profiler A/B，保持新版调度、关闭混深度 CORE、开启 packed prefill。
每个配置完整热身后跑三波，下表取中位数。固定 vLLM 依赖的插件回归 50 通过、2 跳过。

| 负载 | 原退役点 / 提前退役吞吐 | TTFT P50 | TTFT P95 | TPOT P50 | 与原退役点逐 token 一致 |
|---|---:|---:|---:|---:|---:|
| W1 C1 | 236.35 / 246.45 tok/s | 14.73 / 14.62 ms | 15.18 / 15.58 ms | 4.15 / 3.98 ms | 24/24 |
| W3 C16 | 596.05 / 605.20 tok/s | 496.51 / 492.26 ms | 2395.77 / 2362.61 ms | 20.33 / 20.03 ms | 96/96 |

WCPB、自投机和 Decode pipeline 的两个 smoke 请求与原退役点逐 token 一致。提前
退役改善了 W1 的吞吐，W3 的收益较小。W3 的 P/D eager 路径与长 span 的预算等待
仍需单独处理。
[UVA 退役点 A/B](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/uva-early-retire/summary.json)

GPU 7 单独采集 W3 每份快照的正式波，均有 2004 次物理 forward。新路径有 1604 次
在模型输入准备后记录 event，PRELUDE 401 次，CORE 1203 次。每次标记包住 CUPTI
的 `cudaEventRecord`。Worker 调用前的 fence 等待累计从 367.78 降到 25.45 ms。
设备侧在这些 Host 等待期间仍可计算，所以用无 profiler 的 1.53% W3 吞吐收益
衡量端到端影响。
[UVA event 位置及等待统计](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/uva-early-retire/event-record-summary.json)

### P/D 混批 PIECEWISE 图

普通 Inkling 的 `PRELUDE`、同执行变体的 `CORE` 和 `CODA` 按总 token 数和稳定的
计算输入形状复用 PIECEWISE 图。启动时单条 1024-token prefill 捕的图可以供同样
1024 个 token 的 P/D 混批使用。vLLM 的 attention breakpoint 在每次 replay 时读取
真实请求的 offset、seq len、KV 表和 prefill/decode 切分。图内 SConv 使用常驻
缓冲区，每次刷新真实行的块表以及 per-token `seq_idx`、`query_start` 和 slot。
图按 op、模型执行变体和设备输入形状选择。无匹配物理 token 档位或元数据形状时
走 eager，均匀 decode 使用已有 FULL 路径。

GPU 7 定点验证保持同一调度和真实物理布局。`1023P+1D → 1022P+2D → 1023P+1D`
连续复用纯 1024P 的五个 capture，分别回放 PRELUDE、CORE 三轮和 CODA 的实际
CUDA Graph 段。合计九条请求与原混批 eager 逐 token 一致，包含 129-token 残批的
eager 回退。独立纯 1024P 也与原图逐 token 一致。
[混批 Graph 定点对照](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/trace/pd-graph-check/summary.json)

### 预算落选后的调度顺序

一个 span 是单个请求里的一段连续 token。比如 4096-token prompt 的首段可以是
位置 0 到 1023。该段先预留 KV，依次完成 PRELUDE、CORE 的各轮和 CODA。中间
prompt chunk 的 CODA 完成后推进已计算位置。最后一个 prompt chunk 的 CODA 才
交付首 token。decode 通常每段只有一个模型输入 token。还未预留的 prompt 可以按
当步剩余容量切段，首次提交后沿用该段起点和长度，保证后续 op 能接续相同的 hidden
state、KV 和提交记录。

1. 先看已开始的请求，检查各段依赖。WCPB 用前一 chunk 的已完成轮次判断后续
   chunk 哪一轮已可运行。Decode pipeline 用已经产生的候选及当前位置依赖判断
   哪个下一 token 工作可先提交。
2. 已经预留 KV 且就绪的段，如果曾因 token 或批槽容量落选，等得最久的优先。
   其次优先可产出正式 token 的工作，再考虑可结束中间 chunk 的工作。没有这些
   候选时按运行顺序选第一个可执行的段。该段决定此次 op 并先使用预算。
3. 先用其他已开始的同 op 工作填批。预算、在途请求上限、KV 都允许时，再接入
   等待中的请求。新入场的 prefill 可以用剩余预算切段；已开始执行的段沿用原长度。
4. 同一个物理 forward 的工作共用 op，按所需模型执行变体分组。Worker 准备实际
   token、KV 和 metadata，并按图覆盖及执行变体选择 FULL、PIECEWISE 或 eager。
5. 处理 Worker 回执并推进每段的状态。正常末段交付采样结果；下一 token 的投机
   状态等正式结果验证，命中提交，猜错回滚。请求结束或取消后等待在途工作退役，
   再收回 KV 和 Worker 状态。

用预算为 8 的小例子。A 正在 decode，就绪 `CODA` 需要 1 个 token；B 需要完成它
首段的 `CODA`，该段已预留 8 个 token；C 尚未入场。第一次 CODA 回收 A 的结果，
B 因余量只有 7 而落选。第二次先给 B 足够的预算，完成它的第一个 prompt chunk。
下一次 PRELUDE 可以排 A 的 1 个 decode token 和 B 新切的 7 个 prefill token。
三轮 CORE 及 CODA 继续推进这两段。再下一轮 PRELUDE，若 B 只剩 1 个 prompt
token，可排 A 的 1 个、B 的 1 个和 C 的 6 个。这个顺序让已开始的大段有机会收尾，
也可能让正生成的请求多等一个 op。衡量影响时同时看 TPOT/ITL 和 TTFT P95。

#### 同卡跑分、Graph 覆盖和预算回补

GPU 4 无 profiler，关闭混深度 CORE、保留 packed prefill。旧配置已经使用提前退役
UVA event。每个配置完整热身后取三波中位数。

| W3 配置 | 吞吐 tok/s | TTFT P50 | TTFT P95 | TPOT P50 |
|---|---:|---:|---:|---:|
| RUNNING 优先 | 605.20 | 492.26 ms | 2362.61 ms | 20.03 ms |
| 加 P/D PIECEWISE | 851.19 | 317.99 ms | 1443.06 ms | 14.00 ms |
| 再加预算回补与完成优先 | 911.42 | 244.91 ms | 1349.83 ms | 14.48 ms |

只补图的 W3 三波对旧版 96/96 逐 token 一致。调度也修改后，三波各有 30/32 与
旧版一致，差异在两条续入请求，新版三波彼此逐 token 一致。W1 新版中位数为
246.75 tok/s，TPOT 未观察到大幅变化。插件回归 52 通过、2 跳过。组合功能 smoke
覆盖同请求 WCPB、draft CODA 和使用候选输入的 PRELUDE。
[只改 Graph 对照](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/pd-piecewise-fair/graph-only-vs-stable.json) ·
[调度增量对照](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/pd-piecewise-fair/combined-vs-graph-only.json)

最初轮转配置也在 GPU 4 复跑三波，吞吐分别为 635.81、739.09、726.78 tok/s。
当前配置中位吞吐较该中位数提高 25.4%，TTFT P50 从约 1298 降至 245 ms，P95
从 1315.71 升至 1349.83 ms，TPOT P50 从 13.06 升至 14.48 ms。这个端到端对照
包含早退 UVA、Graph 覆盖和调度策略的累计变化。

GPU 7 独立 trace 中有 635 次 P/D 物理 forward，其中 610 次实际提交 PIECEWISE
的 CUDA Graph。另有 25 次 eager。调度标记记录 12 次预算落选和 12 次恢复。其中
请求 1 的两个 1024-token CODA 都在落选后的下一步获得预算，分别对应步骤
1729→1730、1744→1745，每次都观察到 7 个 CUDA Graph 段实际提交。
[预算回补与实际执行 trace](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/pd-piecewise-fair/fairness-summary.json)

#### 当前 RLT 与原生 vLLM 基础档

GPU 4 独占、相同 CPU affinity 0–31。两边使用同一 Inkling2-mini 三轮模型、同一
vLLM 版本、1024-token 预算、80 GiB KV（实际 291018 token）、异步调度与
FULL_AND_PIECEWISE。RLT 开 packed prefill，关闭混深度 CORE、WCPB、自投机和
Decode pipeline。每边完整热身后无 profiler 各跑三波，取三波中位数。

| 负载 | 原生 / RLT 吞吐 | 吞吐比 | 原生 / RLT TTFT P50 | 原生 / RLT TTFT P95 | 原生 / RLT TPOT P50 | 原生 / RLT ITL P95 |
|---|---:|---:|---:|---:|---:|---:|
| W1 C1 | 260.40 / 247.15 tok/s | 94.91% | 11.70 / 14.46 ms | 12.32 / 14.57 ms | 3.78 / 3.96 ms | 3.86 / 4.04 ms |
| W3 C16 | 1013.66 / 912.62 tok/s | 90.03% | 153.43 / 245.12 ms | 1229.27 / 1348.77 ms | 13.40 / 14.46 ms | 22.10 / 23.65 ms |

按原生 TTFT 除以 RLT TTFT，W1 的 P50/P95 分别为 80.96%/84.52%，W3 为
62.59%/91.14%。这两组的吞吐和 TTFT 均未达到 95% 目标。W3 首批 16 条 TTFT P50
为 721.89/802.74 ms，续入 16 条为 152.06/231.06 ms。W1 三波 24/24 与原生逐 token
一致。W3 三波各 27/32 一致，差异始终来自相同的五条续入请求。
[同卡性能和输出验收记录](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/pd-piecewise-fair/native-pair-gpu4/summary.json)

#### 已开始的 span 在后续调度中再切分

最初记录的调度优化。比如一个 8-token prefill span 已完成 CORE，正在等
CODA，而下一次 CODA 还有 7 个空位。调度器可把它拆成 7 和 1，先让前 7 个进入
这次提交。最后 1 个继续保有自己的起点、hidden state 和执行进度。

首个验证应限制在已就绪、没有同范围在途写入的普通 prefill CODA。拆分时按同一
token 区间划分 Worker 的中间状态及 Scheduler 的进度记录，复用已预留的 KV 块。
后 1 个 token 的 CODA 依赖前 7 个在相应层的 KV 和短卷积状态。提交仍按位置连续
推进，只有包含原 prompt 末尾的一段允许采样首 token。需要覆盖后续块回收、取消、
prefix cache 发布和采样语义，并比较多提交一次 CODA 的代价与少等一轮的收益。

#### 下一轮基础性能验证

95% 是同卡配对吞吐和 TTFT 的下限，优化目标约 97%，给运行波动留余量。按相同配置
多波采样，并在重新启动后复测边缘结果，保留每波吞吐、TTFT P50/P95、TPOT/ITL
和逐 token 对照。每项优化先核对原生 vLLM/MRv2 的实现与可复用接口。策略由实际
批布局、就绪依赖及资源选择，W1 和 W3 用来观测。优先看 W1 的逐 op
提交与准备开销，以及 W3 首批和续入请求从入场到末段 CODA 的阶段时间。Profile
用来定位差距，性能比例使用完整热身后的无 profiler 数据。

GPU 4 补采原生和当前 RLT 的完整 W3 NVTX。两边正式波各 32 条，全部有入队、首次
Worker 和首 token 标记，正式窗口各为独占。以下是每个 16 条请求组内的中位数。

| 请求组 | 入队至首次 Worker，原生 / RLT | 首次 Worker 至首 token，原生 / RLT |
|---|---:|---:|
| 首批 16 | 577.46 / 589.18 ms | 124.56 / 205.05 ms |
| 后补 16 | 1.01 / 11.93 ms | 129.57 / 214.97 ms |

两边都是第一个请求四段 prefill、其余 31 条五段。原生记录了 326 次物理 forward，
RLT 为 1670 次，其中 635 次是 P/D 同批。这轮的主要首 token 差距出现在请求开始
执行后，后补组以执行区间约 85 ms 的差距为主。GPU 短窗口用来进一步区分实际计算、
提交间隔和逐 op 输入与 metadata 准备。
[W3 同卡 Host 分段数据](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/trace/w3-final/current-native-host-breakdown.json)

后补请求在首 token 前经过的 prefill Worker forward 中位数是 5 对 25 次。该请求
的相邻 Worker 区间之间累计 31.60 对 120.64 ms。此间隔可能包含 GPU 在算其他请求，
也包含提交和等待，不能当作设备空闲。对应请求本身的 Worker Host 区间累计中位数是
69.50 对 85.93 ms。

GPU 4 短窗比较从首个 Worker 到首批 16 条都取得首 token。原生/RLT 分别经过
66/341 次完整物理 forward，实际 MRv2 输入准备及 attention metadata builder 均为
66/333 次。RLT 的 330 次含 prefill forward 中有 250 次与此前非 CODA forward 的
真实请求、token 起点、长度及 prefill 标志相同；完整 W3 Host 波为 518/665 次。
这些布局是已有输入和 metadata 复用逻辑可进一步核查的场景。需要同时验证请求槽、
KV 块和缓存失效条件。

W1 取双方共同的 37 个完整稳态 decode 位置。MRv2 输入准备和 builder 均每 token
一次。RLT 额外按五个 op 搬运 Graph 输入，Host 区间合计约 0.205 ms/token，FULL
metadata staging 区间约 0.417 ms/token。Host 区间可能与设备计算重叠。
[同卡短窗和输入准备统计](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/trace/w3-final/current-native-device-host-prep.json)

另用 CUDA graph node 粒度采到 W1 的实际图内 kernel。17 个共同稳态 decode 位置的
kernel 覆盖均约 3.72 ms/token。首 token 单例中，从首次 Worker 到首 token，原生与
RLT 分别是 14.84/18.80 ms，其中 kernel 或 memcpy 覆盖为 9.02/8.78 ms。Profile
用来区分设备与提交阶段，延迟验收仍取无 profiler 的正式跑分。
[W1 CUDA 节点与 GPU 覆盖](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/trace/w3-final/current-native-w1-node-breakdown.json)

W3 的同卡 CUDA node 短窗覆盖首批 16 条。从首个 Worker 到第 16 个首 token，
原生/RLT 为 1374.74/1549.36 ms，kernel 或 memcpy 覆盖为 1334.31/1410.95 ms，
没有记录到 kernel 或 memcpy 的区间是 40.43/138.41 ms。与完整 Host 波的逐请求
结果一起看，仍有缩短 op 间提交和准备间隔的空间。
[W3 CUDA node 短窗](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/trace/w3-final/current-native-w3-node-breakdown.json)

#### 相同批布局复用 MRv2 输入与 attention metadata

原生 `GPUModelRunner` 根据请求槽、已计算位置和本步 token 数构建 `InputBatch`，
再 gather 块表与 slot mapping，最后由 `DefaultModelState` 构建 backend metadata。
RLT 原先只在等布局的纯 decode 跨 op 复用这三步。现在沿用原对象和已有失效逻辑，
将复用覆盖到普通 prefill 和 P/D 混批。每行核对 MRv2 的实际顺序、请求槽、位置、
长度及 prefill 状态，整批还核对 `BatchExecutionDescriptor`。确认相同后共享
`InputBatch`、块表、slot mapping 和 attention metadata。深度由 LoopSpan 传给
模型，模型仍选择本轮的 KV。KV 块变化、请求状态变化和采样后触发重新准备；动态
投机、WCPB 及并行等未满足本次复用条件的路径保持原有准备方式。

GPU 7 定点双臂均有 620 个实际 P/D forward。其中 MRv2 `prepare_inputs`、块表
准备和 DefaultModelState builder 各从 620 次降到 151 次，32/32 输出逐 token
一致。GPU 4 的独立无 profiler 对照中，W1 吞吐为 246.82/247.11 tok/s，TTFT
P50/P95 为 13.63/14.45 → 12.90/13.68 ms。W3 为 912.68/918.35 tok/s，TTFT
P50/P95 为 244.16/1348.13 → 240.90/1339.45 ms。新版本 W3 自身三波逐 token
一致，对当轮旧版为 32/32、32/32、30/32，旧版第三波自身也有两条分歧。
[MRv2 调用计数](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/prepared-batch-reuse/builder-summary.json)

#### 同一物理形状跨 op 共享 PIECEWISE 的 SConv staging

继承 vLLM Graph 对固定设备地址的要求。对同一物理形状，启动捕获的逐 op PIECEWISE
图绑定同一份 SConv 设备缓冲。新的 MRv2 metadata 对象进入时刷新一次，后续取到同一
对象复用该次 staging。Attention breakpoint 每次仍读取实际请求布局。FULL 路径
已有按同样原则共享的 metadata 绑定。此项在上一项的独立快照基础上验证。

GPU 7 短 W3 对照中，两组均有 740 次物理 forward 和 640 次 PIECEWISE replay。
共享后 SConv 发生 169 次刷新、471 次复用，PIECEWISE staging Host 区间累计从
64.48 降到 17.65 ms。32/32 输出逐 token 一致。
[SConv staging 计数](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/shared-piecewise-metadata/stage-summary.json)

GPU 4 同卡无 profiler 三波中位数，和上一项独立快照比较如下。

| 负载 | 上一项 / 本项吞吐 | TTFT P50 | TTFT P95 | TPOT P50 | 输出一致 |
|---|---:|---:|---:|---:|---:|
| W1 | 247.11 / 247.32 tok/s | 12.90 / 12.54 ms | 13.68 / 13.23 ms | 3.98 / 3.98 ms | 24/24 |
| W3 | 918.35 / 922.27 tok/s | 240.90 / 240.36 ms | 1339.45 / 1329.34 ms | 14.36 / 14.30 ms | 96/96 |

当前对最近同配置原生 vLLM 同卡基线，W1 吞吐为 94.98%，TTFT P50/P95 得分为
93.30%/93.11%。W3 分别为 90.98%、63.83%/92.47%。两项复用缩小了差距，距离
95% 下限及约 97% 目标仍有距离。固定依赖插件测试 53 通过、2 跳过。
[staging 单变量对照](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/shared-piecewise-metadata/summary.json) ·
[相对原生 vLLM 比例](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/shared-piecewise-metadata/native-vs-latest.json)

#### 连续同布局的执行片段

原生 vLLM 的一次 forward 会把该批的 PRELUDE、各轮 CORE 和 CODA 连续推进。
RLT 完整 W3 Host trace 的 1670 次物理 forward 中，有 1650 次落在连续五个 op
且请求槽、token 起点和长度相同的片段里。Decode 和 P/D 混批都出现这种结构。
这给相邻 op 复用布局和设备缓冲提供了依据。每个 op 仍单独调度、提交和回执，
调度器可以在边界处理新就绪、抢占、采样和回滚。
[连续同布局片段计数](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/trace/w3-final/current-op-chain-opportunities.json)

#### 保持逐 op 独立的设备数据搬运优化

保留 `PRELUDE`、各轮 `CORE` 和 `CODA` 的独立调度与完成边界。上一轮同卡 W3 node
短窗内，两边都做了首批 16 条请求的 65536 个 prompt token。原生与 RLT 单独的
kernel 覆盖为 1333.23/1385.84 ms，计入拷贝后的覆盖为 1334.31/1410.95 ms。
同窗 CUPTI D2D 传输约为 2.43/24.35 GB，RLT 的 Graph 输入 staging 约占 10.58 GB。
最大的矩阵乘、attention prologue 和短卷积 kernel 家族累计耗时接近。此样本采用
上一轮运行快照，绝对 GPU 区间受 node profiler 及实际批布局影响。
[W3 设备 kernel 与拷贝归因](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/trace/w3-final/current-native-w3-kernel-copy-attribution.json)

对同一区间再交叉核对 Host 和 GPU。在上面两条 trace 中，GPU 没有记录到 kernel 或
memcpy 的 40.43/138.41 ms 中，Worker 线程的 kernel/Graph launch CUDA API 分别只
重叠 11.91/17.28 ms。上一版完成 metadata 复用后的 RLT 短窗中，这两个时间为
89.57 ms 和 8.83 ms。后者还与 Worker 的 model layer 范围重叠 67.40 ms，该范围
覆盖 Graph wrapper、分段 replay 和调用层前后的准备，无法单凭重叠定因。各 Host
区间互相嵌套，数字不能相加。
[W3 GPU 活动空档与 Host 区间交叉](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/trace/w3-final/current-native-w3-idle-cpu-overlap.json)

同一短窗中，RLT 额外记录 868 次 `CatArrayBatchedCopy_vectorized`，核耗时累计
约 14.27 ms。源码上 `_InklingState` 在 op 边界按请求保存张量，下一次同布局执行时
又拼接它们。原生保持批内 stack 直至 forward 完成。可以在相同布局的独立 op 间保留
批内中间状态，依据 MRv2 已验证的请求槽、顺序、token 区间和物理形状直接交接。
改变布局或发生抢占、回滚、分组时，将有待消费的行按请求归属恢复为已有逐请求状态。
这条路径首先应只保留一份待交接的批内状态，并让它持有独立于 Graph 输出的内存，
防止下一次 replay 覆盖或某一个等待的请求长期留住整块历史 batch。做第二轮独立
A/B 前分别计数整批交接、换批取行和实际复制字节，覆盖普通 prefill、P/D、纯 decode、
padding 升降档和请求取消。

原生 Inkling `_run_layer()` 只读取已完成的 AttnRes block 和当前 block，再顺序
填入后续 block。图输入 staging 可以只复制这个活跃前缀，回拷也只复制下一 op
需要的前缀。`CODA` 的调用方直接使用输出，后续 op 无需读取该 stack，因此图省去
这次回拷。GPU 4 无 profiler 交换顺序验证，每组每个负载先热身再测三波，W3 吞吐
中位数分别由 914.35 到 919.93 tok/s、915.03 到 921.21 tok/s，提升 0.61% 和
0.68%。TTFT P95 分别由 1332.85 到 1328.91 ms、1333.36 到 1329.21 ms。W1 吞吐
提高 0.23%–0.33%，TTFT P95 两组方向相反。历史同卡上一版 W3 测得 922.27 tok/s，
因此量级接近跨会话波动。两组 W1 共 48/48、W3 共 190/192 与旧版本对应波逐 token
一致，另外两条差异来自旧版本自身不一致的首波。固定依赖插件测试 54 通过、2 跳过。
[GPU 4 先新后旧](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/live-stack-staging/same-session-forward.json) ·
[GPU 4 先旧后新](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/live-stack-staging/same-session-reverse.json)

同卡另采短窗口，按请求、每段 token 区间、op 和批布局配对了 290 次 P/D
Graph forward。图输入 D2D 从 9.58 降到 5.75 GB，Graph launch 后的 D2D 从
11.49 降到 8.48 GB、少 58 次拷贝。Graph launch 和可关联 kernel 数相同。候选 node
trace 短窗有外部 GPU 进程，拷贝量用自有进程同布局计数对照，吞吐与 TTFT 取上面
无 profiler 的独占交换顺序结果。
[CUDA device copy 对照](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/live-stack-staging/matched-pd-device.json)

另外核对 MRv2 的 capture buffer。RLT 纯 decode 的 B1、B8、B15 padding 到 B16
以及回落 B8 中，`query_start_loc`、`seq_lens`、块表、slot mapping 以及短卷积的
`seq_idx`、`query_start` 的设备指针、shape、stride、dtype 与 storage offset 均与
同形状 capture 一致。原生直接使用 `InputBuffers`、`BlockTables` 和短卷积 builder
的常驻缓冲。普通 FULL replay 可以用同一来源，并在底层存储身份或物理布局变化时
走既有路径。P/D 的块表和请求维 offsets 在同 token 图大小下仍会改形状，先沿用
PIECEWISE 的 staging。stack、hidden input、bank 和 tau 仍需要独立的生命周期。
[MRv2 指针与布局对照](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/trace/ptr-borrow-probe/pointer-summary.json)

首 token 还应按请求看每个 op 的就绪、预算选择和下一次实际 GPU 起跑。先前 W3
续入请求从首次 Worker 到首 token 的中位数为 129.57/214.97 ms（原生/RLT），其中
相邻同请求 Worker 段之间为 31.60/120.64 ms。这些区间会与别的请求的 GPU 工作重叠。
下一轮可用相同 token 区间的前后 op 事件区分就绪等待、预算排队和设备空档，避免把
别的请求正在运行的时间算作 kernel 提交开销。

#### 独立 op 的批内状态交接与 FULL 常驻输入

批内状态交接沿用 `PreparedBatch` 对请求槽、顺序、token 起点、长度、prefill 标志和
物理布局的匹配。已匹配的后续 op 直接取得 Inkling 的 stack、prelude、hidden input
和 bank。Graph 输出的 hidden input 复制一次到自有内存，后续 replay 不会覆盖它。
换批前按原顺序还原请求状态，排除 padding。取消、抢占、回滚和 KV 更新走同一失效
路径，继续使用已有的逐请求清理。该机制最多保留一份整批状态，调度和回执保持逐 op。

FULL Graph 从 `InputBuffers`、`BlockTables` 和短卷积 metadata builder 识别常驻
存储，直接绑定同一设备地址的 metadata view、positions 和 is_padding。其余 tensor
保留 staging。Replay 前校验地址、形状、stride、类型以及跨层 metadata 别名，失败
则进入 eager。PIECEWISE 继续使用既有 staging。测试区分同布局保留与换批拆行，也覆盖
地址变化和层间 alias 变化造成的 FULL 拒绝。

GPU 7 上以 `live-stack-staging` 为同卡控制组，分别只打开一项，再测两项合用。
每组先完整热身，再跑三波 W1 C1 与 W3 C16。正式窗口 GPU 独占，采样时钟 2032 MHz。

| 版本 | W1 吞吐 tok/s | W1 TTFT P50 / P95 ms | W3 吞吐 tok/s | W3 TTFT P50 / P95 ms |
|---|---:|---:|---:|---:|
| RLT 控制组 | 249.98 | 12.68 / 13.10 | 925.05 | 239.56 / 1326.14 |
| 只做批内状态交接 | 250.87 | 12.28 / 12.87 | 964.53 | 230.63 / 1293.30 |
| 只做 FULL 缓冲复用 | 254.83 | 12.15 / 12.74 | 930.80 | 238.51 / 1323.55 |
| 两项合用 | 255.08 | 11.84 / 12.15 | 968.60 | 229.93 / 1291.96 |

合用相比控制组，W1/W3 吞吐提升 2.04%/4.71%，W3 TPOT P50 由 14.24 降至 13.60 ms。
每个候选的 W1 三波均 24/24 逐 token 一致。W3 各候选在三个波次间一致，与控制组
逐波对照 94/96，差异同为控制组第三波相对其前两波发生变化的两条请求。固定依赖
插件测试 56 通过、2 跳过。
[单项批内状态交接](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/independent-candidates/packed-2/summary.json) ·
[单项 FULL 缓冲复用](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/independent-candidates/full-1/summary.json) ·
[两项合用](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/independent-candidates/combined-1/summary.json)

短 CUDA node trace 中，W3 的 240 个同布局 P/D 后续 CORE/CODA 都消费了上一段的
整批状态。按布局匹配的 290 个 P/D replay 中，Graph launch 和 Graph node 数相同，
launch 后的 D2D 从 4418 次降到 638 次。W1 的 95 次 FULL op 均使用同一地址的
positions、is_padding 与 48 项 metadata，原有每个输出 token 的 48 次 metadata
D2D 消失。短窗用于核对结构和拷贝次数。性能按上面无 profiler 的同卡 A/B 判断。
[CUDA Graph 与中间状态计数](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/independent-candidates/combined-structural-profile.json)

随后用相同 GPU、KV 和 Graph 参数测原生 vLLM。吞吐比为 RLT/原生，TTFT 比为原生/RLT。

| 负载 | 原生 / RLT 吞吐 tok/s | 吞吐比 | TTFT P50 比 | TTFT P95 比 |
|---|---:|---:|---:|---:|
| W1 C1 | 261.83 / 255.08 | 97.42% | 97.36% | 98.94% |
| W3 C16 | 1015.61 / 968.60 | 95.37% | 66.82% | 95.10% |

W3 P50 TTFT 为原生 153.64 ms、RLT 229.93 ms。该次 W3 吞吐和 P95 TTFT 接近
95% 下限，P50 TTFT 仍有差距。按请求分组，首批 16 条的 TTFT P50 为 721.63/774.48 ms，
续入 16 条为 152.28/219.21 ms。W1 与原生三波 24/24 逐 token 一致，W3 每波 27/32
一致，差异仍集中在此前观察到的五条续入请求。
[同卡原生 vLLM 验收对照](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/independent-candidates/native-vs-combined-gpu7/summary.json)

#### W3 的相邻 prefill chunk 等待

完整 Host 波里，665 次含 prefill 的 RLT 提交有 640 次用满 1024-token 预算。记录的
12 次已预留工作被预算跳过都在下次调度补回，从跳过到补回中位数 2.67 ms，最大约
4.99 ms。按普通固定深度的推进规则重建已预留工作，未观察到其同一 op 有剩余预算却
继续被跳过。这个标记不覆盖尚未预留的下一段，也不记录每次是否因 KV、异步回执或
当前 op 类型无法入批。
[完整 Host 波预算与调度分析](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/independent-candidates/scheduler-observations.json)

首批 16 条的 CUDA node 短窗匹配了 330 个 RLT prefill forward。按请求累计，chunk
内部各 op 间隔中位数约 4.15 ms，相邻 chunk 间隔中位数约 87.06 ms，其中约 85.35 ms
与 GPU kernel 或 memcpy 重叠。同时配对了 63 个 `CODA` 完成、收集输出和下一次
`PRELUDE` 的区间。每条请求累计的分段中位数如下，三列中位数分别统计。

| `CODA` 设备结束到输出收集完成 | 输出收集完成到下一段 `PRELUDE` 入计划 | 入计划到下一段 GPU 起跑 |
|---:|---:|---:|
| 1.03 ms | 63.58 ms | 23.90 ms |

从收集完成到新 `PRELUDE` 入计划的配对区间，累计出现 180 次 CORE、60 次 CODA、
0 次 `PRELUDE`（按请求区间计数，区间可能重合）。在 63 个区间里，62 次在收集完成
之前已经发出其他请求的 `PRELUDE`，其中 60 次包含 prefill、2 次纯 decode。原生
在调度时推进在途 token 位置，允许下一段普通 prefill 更早成为候选。RLT 可以考虑
给已确定无需采样的普通 prefill CODA 增加类似的在途进度，同时保留每个 op 的提交
边界和完成后的 KV 提交。这个改动首先会影响请求间服务顺序，需要结合首批与续入的
TTFT、TPOT 和总体吞吐一起验证。本轮保持现有调度行为。
[请求与 GPU 时间轴](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/independent-candidates/device-request-wait.json)

#### 普通 prefill 的下一段提前入队 A/B

实验只影响普通异步 prefill。确定当前 CODA 无需采样且还有 prompt token 时，允许
下一段 `PRELUDE` 进入 EngineCore 的异步队列。沿用已有在途 CODA 机制，最多提前
一段，并把正确的 token 起点传给 MRv2。前一段 CODA 回执后才推进已提交前缀、发布
prefix cache，并解除下一段 CORE 的依赖。提前分配失败时等待现有工作完成，避免
为提前入队抢占其他请求。取消请求后等已提交的工作都返回再释放 KV。

实验使用 `additional_config.rlt_async_prefill_lookahead=true`，与上一节的两项合用
快照对照。W1 单请求的 prompt 小于 chunk 预算，用作回归参照。W3 同时观察首批
与续入两个请求组、吞吐、TTFT、TPOT 和逐 token 输出。固定依赖下实验版插件测试
58 通过、2 跳过，覆盖 KV 分配失败、回执前的 prefix cache 可见性、取消和末段采样。

GPU 7 第一组先跑旧版，完整热身后三波无 profiler 测得 W3 吞吐 967.95 → 981.13
tok/s、TTFT P50/P95 230.33/1294.41 → 165.61/1237.65 ms。首批和续入的 TTFT P50
分别从 776.36/219.40 降至 726.66/155.02 ms。TPOT P50 13.60 → 13.80 ms，ITL P95
22.42 → 22.81 ms。W1 吞吐波动约 0.29%，24/24 逐 token 一致。
[先旧后新对照](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/independent-candidates/async-prefill-1/summary.json)

倒序复测 W3 吞吐 966.50 → 986.80 tok/s、TTFT P50/P95 231.78/1293.59 →
163.18/1236.65 ms；首批和续入 TTFT P50 分别从 776.15/219.29 降至 726.01/154.21 ms。
对照组第一波为 504.83 tok/s，后两波分别为 966.50、968.39 tok/s。新版本的最低一波
986.74 tok/s，比该轮对照最快波高 1.90%。两组中 TPOT P50 上升约 0.19–0.21 ms，ITL
P95 上升 0.10–0.39 ms。W1 六波逐 token 全部一致；W3 对旧版 170/192 一致，变化均
出现在旧版先前相对原生存在差异的五条续入请求中。实验开关默认关闭。
[先新后旧对照](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/independent-candidates/async-prefill-reverse/summary.json)

独立的 W3 调度探针记录了 127 次不采样的 prefill CODA，其中 126 次在回执之前已提交
下一段 PRELUDE，提前量中位数 9.08 ms，下一段 CORE 均等回执。在预算边界 1024/1025、
两段 2048、8191 prompt + 1 生成 token 的长度边界均完成。1025-token prompt 的停止词
测试只生成一个 token。在第二段 PRELUDE 已提交后主动断连，取消请求的已提交前缀
保持 0，随后在途队列排空，并可用相同 prompt 重发、得到相同输出。
[提前提交记录](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/independent-candidates/prefill-probe/active-gpu7/lookahead.json) ·
[长度、停止与取消探针](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/independent-candidates/prefill-probe/active-gpu7/boundaries.json)

复跑同卡原生 W3 的吞吐为 1015.63 tok/s，TTFT P50/P95 为 153.49/1229.17 ms。
实验版两组分别达到原生吞吐的 96.60%/97.16%，TTFT P50 的 92.68%/94.06%，P95
的 99.32%/99.39%。原生续入 TTFT P50 152.24 ms，实验版为 155.02/154.21 ms。
两组与本次原生逐 token 对照，实验版 173/192，旧版 163/192；首批均一致，差异仍只
出现在原有五条续入请求。W3 P50 在两组都低于 95% 门槛。
[原生 W3 同卡复测](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/independent-candidates/prefill-probe/native-gpu7/summary.json)

## 可变轮次的 CUDA Graph

预热使用固定版本 vLLM `3146a9df4140` 解析后的 `decode_cudagraph_capture_request_sizes`
和 `prefill_cudagraph_capture_token_sizes`。单 token decode 复用
`decode_cudagraph_capture_shapes()`，包括取到配置上限处的最后一个桶。
`prepare_inputs_to_capture()` 仍由 MRv2 构造捕捉用的输入、块表、slot mapping 和
attention metadata。RLT 用 vLLM 的 `CUDAGraphWrapper` 和
`BreakableCUDAGraphWrapper` 分别捕捉 FULL 和 PIECEWISE 的单个 loop 操作。

三轮 VarLoop 的每个 FULL 桶准备 PRELUDE、CODA 和五种 CORE 路径。第一轮和第二轮
各有结束和继续两条路径，第三轮直接结束。请求目标为一轮或两轮时可以复用相应的
结束图。固定轮次模型只准备该配置下的路径。prefill 仍执行模型的最大轮次，按配置
的 token 桶预热 PIECEWISE。预热总量由 vLLM 的桶配置与模型路径决定，删除了原来的
64 张图上限。Worker 只用完整捕捉的 FULL 桶做请求数 padding。

同一个普通 decode CORE 中，比如 A 执行第一轮并结束、B 执行第一轮并继续、C 执行
第二轮并结束，Worker 各自用对应 FULL 图。拆分限于单 token、每个请求一次且能复用
MRv2 输入的子批。块表更新、KV 清零和块复制仍应用一次，调度器按原顺序收到一份
回执。各请求完成 CORE 后，CODA 可以在同一批处理不同结束轮次的请求。图输入只
比较有效维度的 stride，复用 MRv2 buffer 仍检查设备地址和 metadata。

混合 prefill/decode 仍按现有 PIECEWISE 条件匹配，跨 CORE 变体或没有可回放大小时
继续走 eager。Speculation、WCPB、Ouro 和 KV transfer 继续使用原有执行路径。

GPU 4 探针的预热从 60 张增至 109 张。FULL 的 7 个请求数桶（1、2、4、8、16、24、32）
各捕捉 7 张，PIECEWISE 的 12 个 token 桶各捕捉 5 张。引擎初始化记录为 19.12 →
28.11 秒，两次固定 KV 容量均为 291,018 token。W1 C1 和 W3 C1 各 510 次纯 decode
操作全部走 FULL。W3 C16 有 277 次 PRELUDE、1,176 次 CORE、698 次 CODA，全部走 FULL。
旧版同一探针有 410 次纯 decode CORE 和 114 次 CODA 走 eager。新实现的三个 case
共 48 个请求逐位置的 CORE 轮次全部匹配请求深度。逐 token 输出旧版对新版在两个 C1
case 各 8/8 一致，C16 为 30/32 一致。C16 的 32 条输出与旧版关闭混深度算子合批时
逐 token 一致。CPU 插件测试 62 通过、2 跳过。
[旧图路径](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/w1w7-depth/terminal-core-probe/variable-w1w3-gpu4-mixed-on/summary.json) ·
[新图路径](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/w1w7-depth/terminal-core-probe/graph-ab-variable-depth-full-graphs-current-gpu4-mixed-on/summary.json) ·
[实际捕捉清单](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/w1w7-depth/terminal-core-probe/graph-ab-variable-depth-full-graphs-current-gpu4-mixed-on/startup-capture.jsonl)

GPU 4 在 `gpu_memory_utilization=0.40` 的自动 KV 分配下完成 109 张图的预热并返回
生成结果。vLLM 记录图占用 0.94 GiB，扩容后 KV 容量 313,867 token。
关闭 KV 动态扩容的对照也通过，vLLM 的显存估算和正式预热分别捕捉 109 张图，图池
估算 0.97 GiB、正式分配 0.68 GiB，KV 容量 319,829 token。
[自动 KV 验证](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/auto-kv-gpu4/summary.json) ·
[关闭动态扩容验证](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/auto-kv-nonext-gpu4/summary.json)

GPU 7 无探针 A/B 两边均打开 `rlt_mixed_core`、packed prefill、异步调度和 prefill
lookahead。同一请求序列、固定 80 GiB KV、每个 case 完整热身后测三波，表内取三波中位数。
可变轮次请求使用一轮、两轮、三轮、默认交替。W1 C1 和 W3 C1 各 8 请求，W3 C16
为并发 16 的 32 请求。每条生成 128 token。

| 可变轮次 case | 吞吐（tok/s） | TTFT P50（ms） | TTFT P95（ms） | TPOT P50（ms） | ITL P95（ms） |
| --- | ---: | ---: | ---: | ---: | ---: |
| W1 C1 | 106.63 → 323.40 | 12.36 → 12.25 | 13.03 → 12.77 | 7.91 → 3.28 | 17.79 → 3.91 |
| W3 C1 | 100.15 → 274.29 | 81.91 → 80.63 | 83.22 → 82.30 | 7.97 → 3.31 | 17.88 → 3.95 |
| W3 C16 | 197.71 → 394.03 | 512.53 → 449.66 | 3895.13 → 3754.74 | 63.73 → 28.30 | 116.91 → 69.75 |

两边关闭 `rlt_mixed_core` 的对照中，W3 C16 吞吐为 244.72 → 392.41 tok/s。新版本
开启和关闭时吞吐接近，普通 decode 在两种配置下都可以按 CORE 路径回放 FULL。
可变轮次 W3 C16 旧对新逐 token 一致数在开启与关闭时分别为 90/96 与 96/96，两种
新版本配置之间为 96/96。两个可变轮次 C1 case 在两组均各 24/24 一致。

固定三轮作为回归对照。开启 `rlt_mixed_core` 时，W3 C16 吞吐 989.32 → 986.42
tok/s，TTFT P50 159.54 → 159.48 ms，TTFT P95 1236.27 → 1241.65 ms，三个 case
逐 token 一致。关闭时 W3 C16 吞吐 987.74 → 986.73 tok/s，逐 token 一致 88/96，
差异在已知的四条请求上。性能对照为同卡新旧 RLT，本轮维持跨 W1–W7 的既有全量
矩阵暂停状态。
[同卡 A/B 原始与汇总](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/summary.json)

### 最新版本对原生 vLLM 的固定三轮对照

原生 Inkling2-mini 通过模型配置全局设置 `loop_num_passes=3`。RLT 对照也固定三轮，
请求省略 `rlt_target_loops`，且开启 mixed CORE、packed prefill 和异步 prefill
lookahead。两组在 GPU 7 以相同的 80 GiB KV、1024 token 批预算、32 请求上限及相同
请求序列，分别执行完整热身和三波测量。性能比按 RLT 吞吐 / 原生吞吐、原生 TTFT /
RLT TTFT 计算。请求级可变深度另用上一节的 RLT A/B 验证，保持 prefill 三轮，再按请求
调整后续 decode。原生单服务只支持全局深度。

| 固定三轮 | 原生 / RLT 吞吐（tok/s） | 吞吐比 | 原生 / RLT TTFT P50（ms） | TTFT P50 比 | TTFT P95 比 |
| --- | ---: | ---: | ---: | ---: | ---: |
| W1 C1 | 262.19 / 255.61 | 97.49% | 11.55 / 12.27 | 94.12% | 95.82% |
| W3 C1 | 229.34 / 223.99 | 97.67% | 79.94 / 81.22 | 98.42% | 99.06% |
| W3 C16 | 1015.21 / 985.39 | 97.06% | 153.46 / 164.06 | 93.54% | 99.18% |

18 个测量波均采到 GPU 7 独占，双方实际 KV 均为 291,018 token。三个负载的吞吐和
TTFT P95 中位数达到 95% 门槛，W1 C1 和 W3 C16 的 TTFT P50 未达门槛。W3 C16 的
TPOT P50 原生 / RLT 为 13.39 / 13.83 ms，ITL P95 为 22.08 / 22.58 ms。两个 C1
负载各 24/24 逐 token 一致，C16 为 86/96，全部 96 个首 token 一致。
[原生 vLLM 同卡复测](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/latest-native-fixed3/summary.json)

### W3 C16 混轮次吞吐归因

GPU 4 使用同一版 RLT 和同一服务分别跑固定三轮与混轮次。同一批 32 个 4096-token
prompt，每个生成 128 token，并发 16。两组均处理 131,072 个 prefill token，按三轮
共执行 393,216 次 prefill CORE token。后续 decode 的真实 CORE token 访问从 12,192
降至 9,144。无探针吞吐仍由 983.73 降到 385.69 tok/s。

固定三轮的 P/D 混批 CORE 有 369/378 次走 PIECEWISE。混轮次的 365 次中有 300 次走
跨变体 `execute_mixed_core()`，其余 65 次因形状回退 eager。跨变体路径用
`_group_metadata()` 屏蔽其他组的 KV 写入，但把整批 query 及原始请求边界传给每组的
attention，输出只取本组。P/D 混批 CORE 的每层 prefill query token 按调用计从应处理的
335,168 增至 702,981。固定三轮到混轮次的 CUDA event 波次为 4.169 → 10.660 秒，
其中 P/D CORE 的流上时间 2.105 → 6.329 秒，约占新增墙钟时间的 65%。

纯 decode 的 985 → 2,151 次物理操作均走 FULL。CORE 的单次实际请求数中位数从 16
降到 4，622 次逻辑 CORE 被拆成 1,176 次物理回放。短轮次提前进入 CODA，使采样批
变小，整波采样调用由 328 增至 839。纯 decode forward 的流上时间 1.235 → 2.102 秒。

临时对照只把跨变体的 P/D CORE 按组执行，保持调度计划和纯 decode 路径一致。
消除重复 prefill query 后，物理 forward 从 2,852 增至 3,258，带同样记录探针的
吞吐从 383.60 升至 467.02 tok/s。32/32 首 token 一致，31/32 逐 token 一致，全部
请求的 CORE 轮次通过核对。对照调整仅在探针中启用。
[同源 W3 C16 时间与批记录](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/w3-depth-diagnosis-gpu4/summary.json)

### P/D CORE 按实际行执行与 PIECEWISE 尾块回放

Inkling Worker 在同一次 CORE 计划内按轮次、是否终轮和是否生成草稿拆子批。具有相同
执行变体的 prefill 和 decode 仍可同批。每个子批通过 MRv2 构造自己的请求顺序、
query offsets、块表和 slot mapping，因此 attention 对子批的实际 query 执行一次。
同一计划的请求与 KV 更新只提交一次，并按原顺序返回一份回执。逻辑调度顺序沿用
之前的策略。这条路径也适用于 eager，带 KV transfer 的 Worker 保留原路径。

PIECEWISE 按 vLLM 的 `prefill_cudagraph_capture_token_sizes` 捕捉，用最小的可容纳
token 桶回放。Worker 复用 MRv2 的 `BatchExecutionDescriptor`，只补物理 token 数；
请求数和 attention 的 query offsets 保持真实值。MRv2 把 padding 的 slot 填成 `-1`，
采样和中间状态交接只取真实行。若一个桶比单个请求最大上下文更大，预热按 MRv2 的
dummy batch 规则拆成多个合规请求，再捕捉同一个 token 桶。CPU 插件测试 65 通过、
2 跳过。

GPU 4 的短形状探针观察到 1023→1024、1009→1024、17→24 和 3→4 的实际 PIECEWISE
回放。每个受检步覆盖多个 KV 组，padding slot 都为 `-1`，attention query offsets 的
末值与 `num_actual_tokens` 都等于真实 token 数。不同轮次的 decode 与长 prefill 同批
时，每个 CORE 计划分别回放单行 FULL 和实际 prefill 子批的 PIECEWISE。

W3 C16 的 2,298 个有序调度计划与旧版逐项相同。300 个原先走跨变体混合 eager 的逻辑
计划都按实际行集合拆分。带 prefill 的物理 forward 全部走 PIECEWISE，包括 256 个
补齐的 P/D CORE 和 168 个补齐的纯 prefill CORE。结构探针下固定三轮吞吐 983.88 →
984.65 tok/s，混轮次 383.60 → 657.94 tok/s。两组混轮次首 token 32/32 一致，完整序列
29/32 一致，差异出现在历史已见的请求索引，逐位置 CORE 次数全部符合请求目标。
[短形状与整波图回放](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/pd-subgroup-padding/gpu4/pd-subgroup-padding-current/verify/summary.json)

GPU 7 的正式 A/B 两组使用同一个模型、同一请求序列、80 GiB KV 和相同启动参数，
实际 KV 容量均为 291,018 token。四个 case 各预热后测三波，24 个正式波次均采到 GPU
独占与 2032 MHz。GPU 4 计时期间有外部计算进程，受影响的波次没有用于比较。

| Case | 吞吐（tok/s） | TTFT P50 / P95（ms） | TPOT P50（ms） | ITL P95（ms） |
| --- | ---: | ---: | ---: | ---: |
| W1 C1 固定三轮 | 256.30 → 255.36 | 12.53 → 12.32 / 12.87 → 12.73 | 3.83 → 3.85 | 3.92 → 3.94 |
| W1 C1 混轮次 | 324.13 → 322.32 | 12.28 → 12.14 / 12.85 → 12.65 | 3.28 → 3.30 | 3.90 → 3.93 |
| W3 C16 固定三轮 | 988.94 → 998.74 | 160.11 → 155.32 / 1236.63 → 1237.00 | 13.81 → 13.59 | 22.37 → 22.35 |
| W3 C16 混轮次 | 391.06 → 709.98 | 452.61 → 226.31 / 3792.54 → 1618.22 | 28.51 → 16.20 | 70.18 → 29.98 |

正式 A/B 的 144 个首 token 全部一致。W1 固定和混轮次各 24/24 逐 token 一致，W3
固定三轮 93/96 一致，混轮次 87/96 一致。两侧各自的三波输出均稳定，差异落在本轮
记录的三个历史请求索引。W1–W7 全量矩阵维持暂停。
[正式 GPU 7 A/B](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/pd-subgroup-padding/gpu7/paired-pd-subgroup-padding-final.json)

单波诊断在最新版本的同一服务内只切换 PIECEWISE 向上补齐。W3 C16 两次均为相同的
2,298 个有序计划、3,260 个物理子批。开启前 719 个含 prefill 的子批中 175 个走
PIECEWISE、544 个走 eager；开启后 719 个全部实际回放。吞吐从 510.06 变为 705.15
tok/s。首 token 32/32 一致，完整序列 30/32 一致。小批 3→4、17→24 与异构
FULL B1 + 1009→1024 在相同计划下完整序列一致，另一个同批 1009→1024 的例子中，
prefill 请求序列一致，已在生成的请求于第 39 个 token 首次出现差异。
[仅切换 PIECEWISE 补齐的对照](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/pd-subgroup-padding/gpu7/pd-subgroup-padding-final/verify-exact/summary.json)

### 补齐后的 W3 C16 Host 与 GPU 归因

GPU 7 依次运行固定版本原生 vLLM、最新版 RLT 固定三轮、最新版 RLT 混轮次。
同一 W3 C16 请求序列，每波 32 个 4096-token prompt，各生成 128 token，固定
80 GiB KV、1024 token 批预算和 32 请求上限。完整轻量探针采集逻辑计划、物理
forward 与 CUDA Event，波末一次同步。详细 Nsight 另开波次，按首个请求 enqueue
到最后一个请求 finish 取窗，采样若有外部计算进程则剔除该波的时长对照。

| 完整轻量探针 | 原生固定三轮 | RLT 固定三轮 | RLT 混轮次 |
| --- | ---: | ---: | ---: |
| 吞吐（tok/s） | 1014.49 | 994.50 | 688.15 |
| 客户端整波（ms） | 4037.51 | 4118.65 | 5952.15 |
| 逻辑计划 | 326 | 1640 | 2298 |
| 物理 forward | 326 | 1640 | 3260 |
| 采样调用 | 326 | 328 | 839 |

轻量探针采用独占重抓的混轮次波，固定三轮到混轮次多出 1,834 ms。相同 RLT 埋点下，
纯 decode 模型区间的 CORE、CODA、PRELUDE 分别多 1,026、214、53 ms。所有含 prefill
的模型区间合计多 149 ms，采样多 88 ms，模型与采样区间之外多 303 ms。首次混轮次
轻量采样为 637.04 tok/s，纯 decode CORE 增量仍是最大项。CUDA Event 区间包含流等待，
原生从 Worker 入口计，RLT 从模型入口计。正式无探针的 RLT 固定与混轮次吞吐为
998.74 和 709.98 tok/s，对应整波约 1.67 秒差距。

混轮次比 RLT 固定三轮增加 658 个逻辑计划，其中 511 个是 CODA。更多轮次变体在
Worker 再拆出 962 个物理 forward。纯 decode CORE 的真实行数中位数从 16 降到 4，
图补齐后的总行数仍接近（8,436→8,313），分布在 591→1,566 次回放。W3 结构探针中
719 个含 prefill 的子批全部 PIECEWISE，纯 decode 全部 FULL。

两组 RLT 独占的完整 CUDA node 波次为 4.124→6.583 秒。GPU kernel 活动区间并集多
1,405 ms，仅拷贝或 memset 多 59 ms，空档多 996 ms。kernel 数从 329,355 增至
694,325。各类累计 kernel 时间中，GEMM 多约 815 ms，short conv 与 AttnRes 多约
285 ms，attention 多约 239 ms，MoE 路由、量化和合并多约 194 ms。异步活动可重叠，
各类累计值用于定位。D2D 拷贝 14,913→59,401 次，传输 28.49→39.23 GB，累计
23→75 ms。详细 CUDA node 采集的额外时长高于轻量探针，特别是混轮次。

Host 上同一批的复用被打散。实际 MRv2 attention metadata 构造 366→2,586 次，
累计 94.77→607.28 ms。打散后的状态拆行 38→1,747 次，累计 12.03→396.77 ms。
图 metadata staging 累计 42.03→228.30 ms。拆行包含逐请求复制，Host 埋点也可
与 GPU 并行，累计时长不相加。细抓多出的 GPU 空档里约 670 ms 发生在下一项实际
GPU 活动对应的 CUDA API 进入前。显式 UVA wait 累计 100.66→79.58 ms，两组和
GPU 空档的重叠均只有数毫秒。原生 vLLM 的完整 Host NVTX 波是 326 次 metadata
构造、累计 73.11 ms。

原生的完整轻量与 Host NVTX 均覆盖 32 个请求。原生 CUDA node 采集发生 CUPTI 卡顿，
部分区间保留用于查阅，原生 kernel 级整波时长不用于三方比较。首轮 RLT 混轮次 node
采集有外部进程重叠，详细时长只使用之后的 GPU 7 独占重抓。
[可缩放 Host / GPU 时间轴](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/pd-subgroup-padding/final-profile/analysis/host-device.html) ·
[分阶段原始汇总](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/pd-subgroup-padding/final-profile/analysis/profile-summary.json) ·
[相同首批请求的首 token 窗口](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/pd-subgroup-padding/final-profile/matched-first-token-prefix.json)

### 纯 decode 的 CORE 回放、状态交接与 CODA

三个改动通过独立 `additional_config` 或等价 `vllm-rlt serve` 参数控制，默认关闭，按顺序
做累积对照。前三个性能组用固定调度计划核对执行路径，CODA 组再改调度。

- `rlt_merge_decode_core`（`--rlt-merge-decode-core`）在普通 Inkling 纯 decode 计划
  内按 depth 合并终轮和继续轮。主体使用同深度的已有 FULL 终轮图，保持对应 pass 的
  KV，输出按行交付。终轮消费图产出的 CODA 输入，继续轮从更新后的 stack 取 bank 并
  重建下一轮输入。prefill、投机和 KV transfer 沿用各自的执行路径。
- `rlt_decode_state_slots`（`--rlt-decode-state-slots`）在 PreparedBatch 被换掉时把
  `stack`、`prelude`、`inputs`、`bank` 以向量写入按 MRv2 request slot 索引的 GPU
  缓冲。读取用 `InputBatch.idx_mapping` 批量 gather。每个状态在原有 request ID 与
  token 位置映射下发布，重用槽前还保留旧的存活位置。当前批成员和输入描述符完全相同
  时仍直接接力已有的 packed state。MRv2 继续管理 attention metadata 和 KV 块表。
- `rlt_batch_decode_coda`（`--rlt-batch-decode-coda`）在普通 decode 已有可采样的 CODA
  时，允许先执行一个已经就绪的终轮 CORE 计划。预算必须容纳候选和随后的合批。优先
  保留已因预算等待的工作及 prefill 的进度，已经让过一次的 CODA 恢复正常优先级。

第一阶段的 W3 C16 结构对照中，2,298 个有序调度计划逐项相同。403 次 CORE 合并都
回放 FULL，总物理 forward 3,260→2,857，纯 decode CORE 1,566→1,163。32 个请求
逐位置轮数通过。GPU 4 三波无探针对照中 W3 混轮次吞吐 690.15→700.29 tok/s，
TTFT P50/P95 232.71/1672.95→227.30/1625.15 ms。W3 固定三轮吞吐变化 −0.03%，
W1 固定三轮与混轮次分别 +0.03% 和 −0.21%。W3 混轮次首 token 全部一致，逐 token
一致数按波为 31/32、31/32、30/32。
[第一阶段 GPU 4 A/B](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/decode-optimizations/gpu4/paired-merge-core.json)

第二阶段保持全部调度计划和图回放顺序。W3 记录了 955 次 slot 批量写入和 837 次
批量读取，分别覆盖 4,855 和 4,772 行；W1 单请求直接接力原有的 packed state。
GPU 4 的 W3 混轮次吞吐从 700.29 变为 700.12 tok/s，运行组间完整输出一致。
该结果只验证了状态搬运方式的改变，子批组成和 MRv2 的 metadata 构造条件仍一致。
[GPU 槽位 A/B](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/decode-optimizations/gpu4/paired-state-slots.json)

第三阶段结构探针中，W3 C16 发现 200 次有已就绪终轮 CORE 的推迟。逻辑计划从
2,298 减至 2,105，物理 forward 从 2,857 减至 2,620，采样从 839 减至 653。
纯 decode CODA 的平均真实请求数从 4.55 增至 6.16。W1 C1 观测到零次推迟；所有
请求的逐位置 CORE 次数与目标深度一致，仍实际回放 FULL/PIECEWISE。断流取消测试后
MRv2 重用了同一个 slot，后续请求的输出和独立参考完全一致。
[CODA 结构验证](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/decode-optimizations/gpu7/batch-decode-coda/structural/summary.json)

GPU 4 同卡对照使用同一语料、独占 2032 MHz、80 GiB KV，各热身后测三波。W3 C16
混轮次结果如下。

| 配置 | 吞吐（tok/s） | TTFT P50 / P95（ms） | TPOT P50（ms） | ITL P95（ms） |
| --- | ---: | ---: | ---: | ---: |
| 原有运行路径 | 690.15 | 232.71 / 1672.95 | 16.47 | 31.76 |
| 合并 CORE | 700.29 | 227.30 / 1625.15 | 16.55 | 30.05 |
| 合并 CORE + GPU 槽位 | 700.12 | 227.15 / 1624.84 | 16.55 | 30.05 |
| 三项均开启 | 720.18 | 238.40 / 1624.44 | 16.19 | 29.53 |
| 合并 CORE + CODA 合批 | 723.77 | 237.96 / 1624.66 | 16.06 | 28.92 |

三项开启时 W3 固定三轮为 997.28 tok/s。单独增加 CODA 后，W3 混轮次一轮请求的
TPOT P50 从 12.49 升至 13.91 ms，二轮和三轮分别从 15.81、20.01 降至 14.81、19.57 ms。
这反映了让更多终轮 CORE 汇到一起采样的延迟取舍。开启 CODA 的两个版本逐 token
一致；这两组和只开 CORE 的对照中，每波 31/32 个请求完整一致，差异出现在同一个
请求，第七个输出 token 首次分歧。三项均作为独立可选项保留。
[CODA 性能与逐深度数据](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/decode-optimizations/gpu4/paired-batch-coda.json) ·
[CORE 与 CODA 组合](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/decode-optimizations/gpu4/paired-merge-and-coda.json)

首轮分进程比较观察到 W1 约 3.8% 的吞吐下降。GPU 4 用同一服务交错切换 CODA、
各热身后测三波，固定三轮 250.19→250.37 tok/s，混轮次 321.85→321.12 tok/s，
逐 token 全同。固定 TTFT P50/P95 17.51/18.65→17.53/18.92 ms，混轮次
11.84/12.07→12.47/13.06 ms。CPU 插件测试 71 通过、2 跳过。
[单服务 W1 交错复测](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/decode-optimizations/gpu4/batch-decode-coda/hot-w1-check/summary.json)

### W3 混轮次的异步提交与批次碎片

对 CORE 合并与 CODA 合批组合补抓完整 CUDA node。413 个合并 CORE 分别在图回放后执行
两次 `torch.as_tensor(indices, device="cuda")`，将终轮和续轮的 CPU 索引送上 GPU。
826 次 `cudaStreamSynchronize` 全部来自这两处拷贝，Host 累计停留 573.25 ms。阻塞发生在
主流，Host 在准备下一批前先等这一批的 CORE。原有的 UVA 缓冲复用统计使用另一类 CUDA event。

RLT 复用了 MRv2 处理 `idx_mapping` 所用的 `vllm.utils.torch_utils.async_tensor_h2d`。
现在 CPU 只生成一份真实行索引，在 CORE 回放前提交一次 pinned、non-blocking H2D。
续轮与终轮各取 GPU 上的连续 slice，复用原来的状态交接逻辑。新 trace 的 413 个合并 CORE
全部符合每批一次、每真实行 8 字节的拷贝，流同步降至零。两次轻量探针的 2,105 个有序
计划、逐位置轮次和 32 个请求的完整 token ID 一致。
GPU 7 同卡补抓也复现了 413 次异步拷贝和零次流同步。在两次完整 CUDA node 采集里，设备
活动从 4.946 变为 4.953 秒，GPU 空档从 2.120 降到 1.159 秒。该时长包含采集开销。
[修改前 CUDA 拷贝审计](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/pd-subgroup-padding/final-profile/gpu7-merge-coda/variable/merged-core-transfers.json) ·
[同卡修改后 CUDA 拷贝审计](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/pd-subgroup-padding/final-profile/gpu7-async-routes/variable/merged-core-transfers.json)

GPU 7 的正式无探针对照中，两边开启 CORE 合并、CODA 合批，关闭 GPU 槽位，KV 为 80 GiB。
两臂各测三个完整波次，所有计时波独占 GPU，SM 2032 MHz。

| 场景 | 修改前吞吐（tok/s） | 修改后吞吐（tok/s） | 变化 |
| --- | ---: | ---: | ---: |
| W3 C16 混轮次 | 725.11 | 793.95 | +9.49% |
| W3 C16 固定三轮 | 997.91 | 998.87 | +0.10% |
| W1 C1 混轮次 | 322.44 | 323.43 | +0.31% |
| W1 C1 固定三轮 | 255.65 | 255.18 | −0.19% |

W3 混轮次 TTFT P50/P95 从 237.35/1620.64 变为 236.92/1616.01 ms，TPOT P50/P95
从 16.08/23.52 变为 14.40/21.85 ms。W3 混轮次与 W1 固定、混轮次的完整输出分别
96/96、24/24、24/24 一致。W3 固定三轮仍观测到个别请求的完整输出差异，首 token 一致。
CPU 插件测试 72 通过、2 跳过。
[GPU 7 异步索引 A/B](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/decode-optimizations/gpu7/paired-async-routes.json)

混轮次实际执行 9,144 个 decode CORE 行，固定三轮执行 12,192 个，减少 25%。两边 prefill
均执行 393,216 个 CORE 行。就整波而言，减少的 decode 行占固定三轮总 CORE 行数 0.75%。
纯 decode 的物理 CORE 批次从 591 增至 1,124，其中真实行数至多 4 的批次从 48 增至
537。纯 decode CUDA Graph 的补齐后行数为 8,436 和 8,469。混轮次节省的真实行被补齐
抵消，更多小批也增加共享权重算子、元数据构造以及状态交接的调用次数。修复后的同卡
正式计时仍相差约 1.06 秒。

GPU 7 的完整 CUDA node trace 按 Host 提交范围关联设备活动。固定三轮到修复后的混轮次，
纯 decode CORE 的设备活动从 928 增至 1,474 ms，纯 decode CODA 从 146 增至 297 ms，
含 prefill 的 CORE 合计从 2,112 增至 2,241 ms，采样从 77 增至 135 ms。各范围内取
CUDA 活动区间的并集，时间用于归因。纯 decode 只跑 1 行的 CORE 也接近 1 ms，固定
16 行的 CORE 约 1.65 ms。一行回放仍要提交该轮的大部分共享权重 kernel。
[CUDA 活动按 op 汇总](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/pd-subgroup-padding/final-profile/analysis/async-routes-by-op.json) ·
[实际批大小与 GPU 成本](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/pd-subgroup-padding/final-profile/analysis/core-batch-cost-gpu7.json)

固定版本的原生 Inkling2 把所有 pass 放在一次 model forward，用一个请求集合依次执行。
RLT 的调度接口仍独立表达 PRELUDE、CORE 和 CODA。后续执行侧可以参考 MRv2 的
`UBatchSlice` 和 `_slice_input_batch()`，从一个批次派生真实的请求、token 视图，供各
轮次的 attention 和 short-conv 使用对应块表。这样同一个逻辑 CORE 的 QKV 投影和
MoE 可在合并行上执行，KV 相关部分仍仅消费各自的真实行。当前 MRv2 路径在轮次拆分后
分别启动完整的 CORE。整波 1,058 个逻辑 CORE 计划启动 1,573 个物理 CORE，其中 387 个
纯 decode 子批来自含 prefill 的父计划。按一个计划执行一份共享权重计算，理论上可省去
515 份子批的共享算子提交。attention 和 short-conv 仍需按实际行核对 KV 与 query
工作量，实际 kernel 效率和吞吐需要测量。

在旧索引实现上进行的同服务交错实验，从已就绪工作中多推进至多 `num_loops - 1` 批
CORE，再发出已等候的 CODA。允许与已就绪 prefill CORE 同批的一组达到 796.37 tok/s，
默认单批 724.63 tok/s。整波 TTFT P95 从 1618 升至 1623 ms，首批 16 请求的 TTFT
P95 从 1658 升至 1659 ms，后续 16 请求的 TTFT P50 从 149 升至 251 ms。低轮次
请求的 TPOT P50 从 13.81 升至 15.50 ms，高轮次从 19.49 降至 15.65 ms。每波首 token
一致，完整输出 30/32 一致。这组只用于验证批次组成和时延的关系。
[同服务调度 A/B](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/decode-optimizations/gpu7/batch-decode-coda/coda-cohort/summary.json) ·
[前后两组请求的 TTFT](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/decode-optimizations/gpu7/batch-decode-coda/coda-cohort/ttft-cohorts.json)

GPU 7 在异步索引版本上交错复测，三组各三波均独占。首 token 保护组在调度器有已知
请求尚未输出 token 的时段使用原来的单批策略，其余时段允许多推进已就绪 CORE。
这些策略使用同一个热服务和同一份 CUDA Graph。

| 策略 | 吞吐（tok/s） | 全体 TTFT P50 / P95（ms） | 续入 16 个请求 TTFT P50 / P95（ms） |
| --- | ---: | ---: | ---: |
| 原单批策略 | 792.19 | 237 / 1620 | 148 / 235 |
| 多推进 CORE | 868.65 | 336 / 1618 | 251 / 364 |
| 多推进 CORE + 首 token 保护 | 842.41 | 352 / 1619 | 235 / 448 |

首批 16 个请求的 TTFT P95 均为约 1,650–1,660 ms。全体 TTFT P95 对续入请求的
延迟退化不敏感。两组实验的首 token 96/96 一致，完整输出各 90/96 一致。首 token
保护改善续入请求的中位数，但尾延迟继续增加。两组均保留为实验探针。
[异步版调度 A/B](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/decode-optimizations/gpu7/async-merged-core-routes/coda-cohort-first-token-guard/summary.json) ·
[首批、续入及 token 序列核对](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/decode-optimizations/gpu7/async-merged-core-routes/coda-cohort-first-token-guard/cohort-validation.json)

另外复用了原生 `decode_cudagraph_capture_grid(32)`。GPU 7 同一热服务捕捉 165 张图，
交错切换 FULL 可选桶，每个设置独占计时三波，PIECEWISE token 桶保持一致。实际验证
波的 1,124 次纯 decode CORE 均回放 FULL，真实行均为 6,748，物理行从 8,469 降到
7,044（−16.8%）。吞吐 781.01→781.75 tok/s（+0.10%），TTFT P50/P95 从
241.52/1656.20 变为 251.13/1650.34 ms，首 token 一致，完整输出每波 30/32 一致。
变密的桶在本对照中几乎没有吞吐收益。被其它进程占用的 GPU 4 计时波已剔除。
[同服务 FULL 桶 A/B](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/decode-optimizations/gpu7/async-merged-core-routes/decode-grid-interleaved/summary.json) ·
[实际回放宽度](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/decode-optimizations/gpu7/async-merged-core-routes/decode-grid-interleaved/actual-full-replays.json)

### 跨轮次 CORE 合批与有界 CODA 合批

通过 `--rlt-compact-mixed-core` 或 `additional_config.rlt_compact_mixed_core` 启用实验。
MRv2 在每个 `(depth, last_core)` 组内保留原生的 decode/prefill 顺序，准备一份总
`InputBatch`。每组借用 `UBatchSlice` 的请求和 token 视图，再从原生 MRv2 builder 拿到
自己的块表、slot 和 query offsets。各组的 query 长度总和等于计划里的真实 token 数。
第二版只为各组实际访问的 attention 和 short-conv 层构造 metadata。同深度终轮和续轮
使用独立的 builder 缓冲，后构造的 sub-batch 不会覆盖前一个。

每个 CORE 层只调用一次整批 `_project`、输出投影和 MoE。attention 和两次 short-conv
切到该组真实行并访问相应深度的 KV。CORE 输出仍按 `(request_id, token_start)` 交付，
终轮保留 CODA 输入，续轮重建下一轮所需的状态。实验覆盖普通 TP1 的 Inkling 多轮次
批，包含 decode 与 prefill 混批。其它路径仍由各自已有的执行入口处理。

首版结构波覆盖 32 个请求，2,105 个调度计划及逐位置 CORE 次数逐项一致。物理执行
从 2,620 降至 2,281，238 批实际走共享权重的 compact eager，其中 110 批纯 decode、
126 批混入 prefill、2 批纯 prefill。687 个真实子批的 query 长度和 KV slot 均核对，
这些子批使用真实边界。首 token 32/32、完整输出 30/32 一致。
[首版结构波](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/decode-optimizations/gpu4/compact-mixed-core-eager/structural/summary.json)

GPU 7 同卡 CUDA node 对照中，优化过 metadata 的 compact eager 让纯 decode CORE 设备
活动从 1,474 降到 1,233 ms，整波设备活动从 4,953 降到 4,864 ms。整波非设备活动
时间从 1,159 增至 5,317 ms，独立提交的 kernel 从 108,560 增至 217,084。CUDA node
采集放大了 eager 的提交开销，这里用于归因。即使共享权重减少了设备工作，直接把合批
后的 CORE 切到 eager 仍会丢失 CUDA Graph 的调度收益。238 个合批中的 2,856 个 CORE
层调用全部核实 `_project` 和 MoE 各执行一次，输入行数等于真实行数。
[同卡 CUDA 归因](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/pd-subgroup-padding/final-profile/analysis/compact-core-device-comparison.json)

GPU 4 在同一热服务中交错开启和关闭 compact eager，各测三个独占波次。W3 C16
吞吐从 786.11 降至 470.95 tok/s，TTFT P50/P95 从 239/1630 增至 322/2648 ms。
首 token 96/96 一致，完整输出 91/96 一致。W1 C1 混轮次单波检查的完整输出一致。
[compact eager 同服务 A/B](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/decode-optimizations/gpu4/compact-mixed-core-filtered/compact-eager-interleaved/summary.json)

另一个探针复用 vLLM 的 `CUDAGraphWrapper`，在热身波为重复出现的纯 decode 子批布局
捕捉共享 CORE 的 FULL graph。计时只回放这批图，冷布局及含 prefill 的 CORE 沿用
已有的分组图。GPU 7 在同一热服务中各测三波，7 种布局每波实际回放 104 次。候选
吞吐 791.90→780.89 tok/s，TTFT P50/P95 均约 237/1617 ms。96/96 个首 token
一致，完整输出 93/96 一致，差异集中在同一请求的第七个输出 token。这组证明共享 CORE
可以捕捉并回放 FULL，测试范围内仍无端到端收益。探针每次重建子批 metadata 并将其
拷入图的固定地址，还会 staging stack、位置和各组的输入，回放后回拷 stack。FULL 对照
尚无这些环节逐项的设备计时。
[跨轮次 FULL 同服务 A/B](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/decode-optimizations/gpu7/compact-mixed-core-filtered/compact-full-decode/summary.json)

CODA 使用异步索引版本的同一热服务，对照已有的单次让行。实验策略至多允许两次
让行，第二次要求已经就绪且能让更多请求抵达 CODA 的 CORE。有已知的首 token 需求或
prefill 时沿用单次策略。完成保护组额外优先发布将达到生成上限或上下文上限的请求。
GPU 7 九个计时波均独占，三组的首 token 一致。

| CODA 策略 | 吞吐（tok/s） | 续入 16 条 TTFT P50 / P95（ms） | 一轮请求 TPOT P50（ms） |
| --- | ---: | ---: | ---: |
| 单次让行 | 787.92 | 150 / 235 | 12.31 |
| 至多两次，让已就绪终轮 CORE 先行 | 825.91 | 236 / 449 | 13.89 |
| 至多两次 + 完成保护 | 831.57 | 236 / 449 | 13.75 |

完成保护每波触发 32 次，单独保护最后一个 token 仍留下了续入请求的 TTFT 劣化。
进一步要求短轮次的 CODA 不为目标轮次更长的 CORE 多等一次，吞吐 792.33→797.22
tok/s，续入 TTFT P50/P95 从 148/235 变为 152/248 ms。但深度保护每波拦截 119 次，
实际额外让行降为零，这个吞吐差不归因于增加的 CODA 合批。两轮调度实验留在探针。
[第二次让行与完成保护](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/decode-optimizations/gpu7/async-merged-core-routes/coda-cohort-bounded-coda/summary.json)
[限制跨目标轮次等待](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/decode-optimizations/gpu7/async-merged-core-routes/coda-cohort-bounded-depth/summary.json)

### 共享 CORE 的等工作量对照

前述 1,474→1,233 ms 按物理 Worker 的提交范围汇总。合批后部分 decode 子批由纯 decode
范围移入 P/D，直接比较该分项会混入归属变化。对两份 GPU 7 node trace 按请求、token
位置和 CORE 深度逐项匹配，110 个混轮次纯 decode 父计划中有 70 个完全相同，覆盖
1,087 个真实行。原分组图到共享 eager 的结果如下。

| 70 个匹配计划 | 原分组图（ms） | 共享 eager（ms） |
| --- | ---: | ---: |
| GEMM kernel 时间累计 | 109.50 | 74.75 |
| MoE 路由、量化、合并 kernel 时间累计 | 20.71 | 5.90 |
| Tensor 拼接、拷贝等 kernel 时间累计 | 0.98 | 7.67 |
| 全部 kernel 时间累计 | 234.55 | 189.33 |
| GPU kernel 活动区间并集 | 179.22 | 188.68 |

累计时间会计入同时执行的 kernel。原分组图的累计值比并集大约 55 ms，共享 eager 的
差值小于 1 ms。共享权重确实减少了算子累计耗时，eager 的拼接、拷贝、kernel 重叠变化和
Host 提交开销仍需计入端到端。70 个计划的 CUDA node Host 提交范围累计 137→1,893 ms，
此处保留为采集下的定位数据。
[按相同请求、位置和深度匹配的 kernel 数据](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/pd-subgroup-padding/final-profile/analysis/matched-mixed-decode-core.json)

同 70 个计划的 stream 与 CUDA launch 关联表明，原分组路径的重叠都发生在同一个子批、
同一次 CUDA Graph 回放、同一个 profiler stream 内。GPU 上同时有至少两个 kernel 活动的
时间，原分组图为 51.99 ms，共享 eager 为 0.66 ms。Inkling2 attention prologue 用
Programmatic Dependent Launch（PDL）允许相邻的 `_rel_verify_kernel` 先读旧 KV，
需要读取当前步刚产生的 Q 和新 KV 时再同步。原图中匹配到的 1,680 对均实际重叠，
下游启动相对上游结束的中位数为提前 5.47 µs。共享 eager 的 2,520 对中，Host 进入
下游 launch API 时上游均已结束，中位数晚 57.52 µs。该项只比较同一份 node 采集协议下
的实际提交节奏。
[匹配计划的 stream、回放及时间戳](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/pd-subgroup-padding/final-profile/analysis/matched-core-stream-overlap.json)

共享 FULL 再区分两种输入准备。拷贝版沿用每组固定图缓冲；复用版核对 tensor 地址、形状
和 stride 后，借用 MRv2 的块表、slot、序列长度、位置与 short-conv 持久缓冲。单 token
decode 的 query offsets 借用固定 `0..n` 视图。两版运行相同的共享 CORE 图，仍逐子批
构造 metadata、聚合和写回状态，CUDA Graph 实验只覆盖纯 decode，含 prefill 的 CORE
仍使用原分组执行。

首次 GPU 7 同服务三臂各测三波，GPU 均独占。两版缓存相同的 6 种布局，每波实际回放
100–102 次。复用版省去中位数 9,792 份 GPU metadata 字段拷贝。Host metadata 构建
81.58→66.71 ms、staging 50.88→19.82 ms，各项 Host 计时中位数相加约少 43 ms。
分组图、拷贝 FULL、复用 FULL 的吞吐中位数为 754.24、701.53、653.29 tok/s；整波波动
较大，实际回放次数也有小幅差异。两版 FULL 首 token 96/96 一致，逐 token 一致数按波
为 29/32、32/32、32/32。
[MRv2 缓冲复用首轮对照与分项](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/decode-optimizations/gpu7/compact-mixed-core-filtered/compact-full-reuse/summary.json)

复测换用 CPU 192–223，在 GPU 7 同一热服务交错各测三波。两版 FULL 每波命中完全一样的
7 种布局和 104 次回放，9 个计时波均独占 GPU 且 SM 2032 MHz。复用省去每波 9,984 份
GPU metadata 字段拷贝，Host metadata 构建 62.90→44.62 ms，staging 35.28→11.64 ms。
各项 Host 计时中位数相加少 41.83 ms，实测整波少 40.42 ms。复用后吞吐追回共享图相对
原分组图差距的约 57%。

| 实现 | 吞吐（tok/s） | TTFT P50/P95（ms） |
| --- | ---: | ---: |
| 原分组图 | 792.67 | 237.28 / 1616.31 |
| 共享 FULL，复制 metadata | 782.06 | 237.21 / 1615.53 |
| 共享 FULL，复用 MRv2 缓冲 | 788.14 | 237.25 / 1616.06 |

两版共享 FULL 完整输出 96/96 一致。原分组图与共享 FULL 每波首 token 全同，完整输出
31/32 一致，同一个请求从第七个 token 分歧。复用只在探针中启用。剩余约 30 ms 的
整波差距尚未按同一 FULL 布局逐项拆分。
[相同回放覆盖的复测](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/decode-optimizations/gpu7/compact-mixed-core-filtered/compact-full-reuse-cpu192/summary.json)

### C32 动态轮次按轮次分批对照

这项对照问的是生成阶段的 CORE 是否把不同轮次的请求合成一个物理 GPU 批。比如同一计划
有第一轮 6 个、第二轮 5 个、第三轮 5 个，分批路径分别执行，合批路径同一次执行共享
权重。P/D 同批描述同一个 GPU 批里有读取 prompt 的行和正在生成的行，是另一个维度。

旧 `async-prefill-lookahead` 快照的历史矩阵里，W2 C32 的纯 decode 混轮次物理合批关闭和
开启分别为 560.96 和 419.57 tok/s。W4 C32 和自然 EOS 的 W7 C32 完成了关闭组，吞吐
分别为 371.03 和 626.26 tok/s。该历史数据来自不同于当前 FULL 实验的执行版本。
[旧矩阵三波中位数和原始配置](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/w1w7-depth/summary.json)

当前 `compact-mixed-core-filtered` 快照在 GPU 4 测 W2、GPU 7 测 W4。每个 case 采用同一
热服务交错做分批与复用 MRv2 缓冲的共享 FULL 各三波。每波 64 个请求、C32、固定生成
128 token，生成轮次 1/2/3 分别是 16/16/32 个请求，80 GiB KV。两组使用相同的 P/D 路径。
12 个计时波采到的计算进程均来自本轮，SM 为 2032 MHz。

| 负载 | 分批吞吐（tok/s） | 部分布局共享 FULL（tok/s） | TTFT P50/P95，分批→共享（ms） |
| --- | ---: | ---: | ---: |
| W2 C32，128-token prompt | 2796.66 | 2793.63 | 64.75/150.81 → 64.69/148.86 |
| W4 C32，4096-token prompt | 954.99 | 954.56 | 696.39/3307.58 → 696.22/3307.28 |

W2 共享组每波 199 个纯 decode 跨轮次计划中回放 14 个，覆盖 224/6,094 个可合行。
W4 每波 18 个跨轮次计划中回放 8 个，覆盖 190/417 个可合行。两组正式波中均无示例里的
6/5/5 轮次布局。未命中的布局使用原分批图，结果只反映这份图覆盖。两 case 每波首 token
均全同。W4 完整输出全部一致。W2 分批首波的布局与后两波不同，稳定的两波分批输出彼此
一致，对应共享组每波有 62/64 个请求的完整输出一致。W1–W7 全量矩阵继续暂停。
[W2 当前数据](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/decode-optimizations/gpu4/compact-mixed-core-filtered/compact-full-reuse-c32-r1-w2/summary.json) ·
[W4 当前数据](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/decode-optimizations/gpu7/compact-mixed-core-filtered/compact-full-reuse-c32-r1-w4/summary.json)

### W3 C16 关闭 prefill 和 decode 同批

GPU 7 同一热服务交错各三波，W3 C16 动态轮次，每波 32 个 4096-token prompt 和
4,096 个输出 token。关闭组在调度时让一个计划只选择 prefill 或 decode，沿用原来的
op 优先级和异步采样。开启组三个 op 均实际出现两类 token 同一物理批的执行。

| 排批方式 | 吞吐（tok/s） | TTFT P50/P95（ms） |
| --- | ---: | ---: |
| P/D 同批开启 | 770.81 | 239.33 / 1670.08 |
| 调度时分开 prefill 与 decode | 274.08 | 6943.10 / 7209.85 |

关闭组三波的三类 op 混合物理批均为零，decode PRELUDE、CORE、CODA 实际分别提交
4,064、9,144、4,064 个单请求批。所有阶段和轮次的实际 token 总量一致。首 token
全部一致，完整输出各波 30/32、31/32、31/32 一致。这组结果包含当前排队规则在关闭
混批后造成的 decode 批变小，性能差对应这份排批规则。
[调度关掉 P/D 同批的 GPU 7 对照](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/decode-optimizations/gpu7/compact-mixed-core-filtered-pd-isolation/pd-batch-w3/summary.json)

另一次 GPU 7 同服务交错对照保留了原调度器，关闭组只在 Worker 将两类 token 分到
不同 GPU 批。拆开的 CODA 子批逐个异步提交采样，并通过原收集入口合并结果。两臂每波
均有 2,105 个逻辑计划，原有逻辑混批数和各轮次工作量一致。开启与关闭后实际提交为
2,620 和 3,138 次，关闭组三个 op 的 P/D 混合物理批为零。

| Worker 执行方式 | 吞吐（tok/s） | TTFT P50/P95（ms） |
| --- | ---: | ---: |
| P/D 同批开启 | 794.31 | 237.16 / 1613.46 |
| 只分开物理 GPU 批 | 732.53 | 276.96 / 1713.99 |

关闭后三波为 731.12、732.53、732.74 tok/s，吞吐中位数比同服务开启组低 7.78%。
GPU 六个计时波均独占且 SM 2032 MHz。首 token 96/96 一致，完整输出各波 30/32
一致。此前约 990–1000 tok/s 的 W3 C16 数据为固定三轮且 P/D 同批开启。两种关闭方式
均为独立诊断快照。
[保留调度、只拆 GPU 批的 GPU 7 对照](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/decode-optimizations/gpu7/compact-mixed-core-filtered-pd-worker-isolation/pd-batch-w3/summary.json)

### W3 动态轮次达到并超过原生吞吐的验证

固定 Inkling2-mini VarLoop checkpoint 和 vLLM `3146a9df4140`，W3 C16 各波 32 个
4096-token prompt、各生成 128 token。RLT 生成轮次分布为 8 个一轮、8 个两轮、16 个三轮，
prefill 始终三轮。P/D 同批开启，80 GiB KV，比较同卡和同批预算的原生固定三轮。正式
吞吐来自预热后的无 profiler 波，独立结构波核对每个位置的轮次和实际图回放。

GPU 7 的起始门槛是原生固定三轮 1,015.02 tok/s，RLT 动态 790.03 tok/s，预算均为
1,024。GPU 4 在同一 RLT 执行路径上将预算提高到 2,048、4,096、8,192，吞吐中位数
为 903.65、998.62、1,012.32 tok/s。原生同预算也有收益，GPU 4 的 2,048/4,096 为
1,124.76/1,233.68，GPU 7 三波干净的 8,192 原生中位数为 1,242.98。
[原生初始基线](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/timed/native-w3-1000-gpu7/summary.json) ·
[原生 8,192 GPU 7 基线](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/decode-optimizations/w3-1000-gate/gpu7/compact-mixed-core-filtered-same-budget-native8192/native-fixed3-8192/summary.json)

普通 prefill 预留复用 vLLM 的 `long_prefill_token_threshold`。预算保持 1,024，单请求
上限设 512 后实测 742.04 tok/s，TTFT P95 从约 1,618 增至 1,753 ms。基线使用阈值零。

`rlt_merge_prefill_core` 让带 prefill 的计划里同一轮的终轮、续轮共享 CORE，再根据实际
token 行将状态交付给 CODA 或下一轮。按 vLLM 的预热方式补齐 PIECEWISE 终轮变体。
在 1,024 预算下，GPU 7 吞吐从 790.03 增至 840.04 tok/s。独立结构波保持 2,105 个
逻辑计划，减少 260 次物理执行，其中新的 191 个混合 CORE 全部实际回放 PIECEWISE；
纯 decode 的合并 CORE 全部实际回放 FULL。逐位置轮次全部通过。8,192 下两项组合后，
GPU 7 到 1,046.76 tok/s，独立结构波为 1,758 个计划、1,942 次物理执行，实际混批
PRELUDE/CORE/CODA 为 20/50/21 次。将路由重排为连续切片的候选降到 1,022.22 tok/s，
工作区恢复已验证的异步索引路由。
[同轮终轮/续轮合并结构](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/decode-optimizations/w3-1000-gate/gpu7/merge-prefill-core-enabled-structure-r1/rlt-variable-1024/structure.json) ·
[8,192 组合结构](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/decode-optimizations/w3-1000-gate/gpu7/merge-prefill-core-budget8192-combined-structure/rlt-variable-8192/structure.json)

调度诊断让当前 token 的已就绪 CORE 优先于开始下一个 token。遇到有首 token 的
prefill CODA 或预算跳过的工作时沿用其优先级。一条已经到 CODA 的 decode token
最多为已就绪的 CORE 让 `num_loops - 1` 次。GPU 4 同服务三臂在 8,192 预算下为
1,021.47、1,178.55、1,186.12 tok/s，最后一组 96/96 完整输出一致。续入请求 TTFT
P50 从 172.52 增到 276.06 ms，整波 TTFT P95 从 847.96 降到 822.76 ms。生产版本
用 `rlt_finish_decode_token` 单独控制，GPU 7 同预算正式吞吐 1,183.94，GPU 4 在
4,096 为 1,183.40，分别达到同预算原生的约 95.25% 和 95.92%。
[调度三臂对照](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/decode-optimizations/gpu4/merge-prefill-core/coda-cohort-w3-gate/summary.json) ·
[生产调度 8,192 正式波](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/decode-optimizations/w3-1000-gate/gpu7/finish-decode-token-formal/rlt-variable-8192/summary.json)

8,192 结构波中有 44 个 prefill 或 P/D 批实际只需约 4,096 行，却按 8,192 行放入图。
将相同计划投到 vLLM 原生 `prefill_cudagraph_capture_grid`，预计 CORE 的物理行从
487,572 降到 399,252，含 prefill 的三个 op 共少 168,960 行。固定三轮已有的执行形状
更接近稀疏档位，预计少 200 行。该数字是按同一执行计划推算，正式运行对照使用两边同一图档。
[真实形状到原生图尺寸的回算](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/decode-optimizations/w3-1000-gate/8192-prefill-grid-projection.json)

两端改用同一套 vLLM 原生 prefill 图尺寸后，GPU 7 的 RLT 三波为 1,231.92、1,232.29、
1,231.00 tok/s，中位数比同卡稀疏图档提高 4.05%。原生固定三轮为 1,274.10、
1,271.84、1,272.42 tok/s。RLT 达同预算原生的 96.82%，每波还差约 105 ms。
TTFT P95 为 805.15 / 737.83 ms。RLT 两套图档之间逐 token 96/96 一致。跨原生对照
的首 token 96/96 一致，生成同样三轮的 16 个请求每波 15/16 完整一致。

独立 GPU 4 结构波核对 32 个请求的逐位置轮次。1,346 个逻辑计划执行过 1,367 次
真实图回放。P/D 同一物理 forward 在 PRELUDE / CORE / CODA 上出现 17 / 45 / 20 次。
用当前调度的真实形状反算，含 prefill 的批全部补齐到稠密图档共跑 662,440 行，旧图档
需跑 766,208 行。需评估图显存成本。GPU 7 计时波采样的 RLT 显存峰值从 109,384
增至 135,344 MiB，同卡原生稠密图档为 106,378 MiB。
[同卡三波及逐输出对照](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/decode-optimizations/w3-1000-gate/gpu7/formal-8192-graph-comparison.json) ·
[P/D 同批与补齐行实测](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/decode-optimizations/w3-1000-gate/gpu4/finish-decode-token-native-grid-structure/rlt-variable-8192/padding-comparison.json)

#### 回到原预算做验收

按原 1,024 token 调度预算和原始 FULL/PIECEWISE 图档复测，8,192 的结果保留作归因。
GPU 7 开启同轮 CORE 合并和当前 decode token 收尾后，动态 RLT 三波为 928.53、928.47、
928.65 tok/s，TTFT P95 为 1,523.62 ms。同卡重测原生固定三轮为 1,016.25、1,015.67、
1,016.35 tok/s，TTFT P95 为 1,228.93 ms。以新鲜原生中位数计算，当前候选达 91.37%，
整波还差约 381 ms。继续在该预算下复测及消融。
[RLT 原预算三波](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/decode-optimizations/w3-1000-gate/gpu7/finish-decode-token-original1024/rlt-variable-1024/summary.json) ·
[同卡原生原预算三波](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/decode-optimizations/w3-1000-gate/gpu7/compact-mixed-core-filtered-original1024-retake/native-fixed3-1024/summary.json)

GPU 7 的独立结构波有 1,692 个逻辑计划、1,823 次实际图回放。物理 P/D 同批数为
PRELUDE 128、CORE 251、CODA 116，32 个请求的逐位置轮数通过。device event 标出的
首尾区间约 4,423 ms，模型及采样范围外约 126 ms。含 prefill 的 CORE 范围合计约
2,213 ms，纯 decode CORE 约 1,055 ms，纯 decode CODA 约 166 ms，采样约 81 ms。
这些范围包含内部提交间隔。121 个 CORE 计划因已就绪的 prefill 和 decode 处在不同
轮次而产生 131 个额外物理子批，拆出的纯 decode 范围约 158 ms。当前试验让较浅的
CORE 轮次先推进，待后续轮次对齐后同批执行。
[GPU 相位及物理批结构](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/decode-optimizations/w3-1000-gate/gpu7/finish-decode-token-original1024-events/rlt-variable-1024/structure.json)

同轮对齐在 `rlt_finish_decode_token` 内启用。已有预算等待或 prefill 首 token 优先级
维持原顺序。含 prefill 的 CORE 计划按当前选定的轮次取工作，较早的轮次推进后可以
加入下一批。GPU 7 的正式吞吐两轮中位数为 975.10 和 975.37 tok/s，TTFT P95 从
1,523.62 降至 1,279.63 ms。独立结构波的 1,659 个计划均只启动一个物理批，实际 CORE
P/D 从 251 增至 360，轮次验证全过。纯 decode CORE 范围从约 1,055 降至 908 ms。
正式波相对旧调度首 token 96/96 一致，每波 31/32 完整一致。差异集中在同一请求，
第 98 个生成 token 首次分歧。
[同轮对齐设备时间对照](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/decode-optimizations/w3-1000-gate/gpu7/original1024-align-phase-comparison.json)

在该策略上组合复用 MRv2 PIECEWISE 输入缓冲、共享同尺寸图的 stack、缓存原生算法
的 Prelude 混合结果，GPU 7 正式中位数 979.69 tok/s，TTFT P95 1,269.40 ms。随后
倒序复测同轮对齐为 975.37，跨配置完整输出 96/96 一致。试验将原生异步队列深度
从 2 增至 4，在 GPU 4 吞吐从 912.22 降至 759.84 tok/s，仍使用默认 2 验收。
[原预算图与状态组合](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/decode-optimizations/w3-1000-gate/gpu7/align-prefill-with-graph-state-original1024/rlt-variable-1024/summary.json)

借用 MRv2 的 PIECEWISE metadata 在独立结构波有 688 个真实回放，每次六组 SConv
都使用原地址，positions 和 padding 也同地址。回放继续使用该次请求的实际 attention
长度。逐地址核对与全 32 请求的逐位置轮次均通过。原生独立进程式 Worker 在 GPU 4
为 972.61 tok/s，同卡单进程对照为 974.23。单进程把提交放到线程，在 GPU 7 降到
911.84。为同一 PreparedBatch 加的 CORE 直提交实测命中 320 次，GPU 7 仍为
979.72，吞吐与对照基本持平。上述并行执行和直提交实验保留在独立快照。
[PIECEWISE 原地址验证](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/decode-optimizations/w3-1000-gate/gpu4/borrow-piecewise-inputs-original1024-structure/rlt-variable-1024/borrowed-bindings.json) ·
[原预算同卡实验汇总](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/decode-optimizations/w3-1000-gate/original1024-scorecard.json)

CODA 末层 MoE 的诊断分支保留每个请求的 short-conv 尾部，复用原生 MoE padding，
其余计算和 KV 访存仍走原来的算子。GPU 7 是 983.48 tok/s，TTFT P95 为 1,263.73 ms，
与同卡组合基线逐 token 96/96 一致。结构波中 130 个含 prefill 的 CODA 在末层 MoE
保留 1,967 / 132,608 行，涵盖所有生成行和未来卷积用得到的 prompt 行。该候选只在
关闭 prefix cache、投机、KV 传输和路由记录的普通路径试验。
[CODA 末层保留行与图回放](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/decode-optimizations/w3-1000-gate/gpu7/align-prefill-coda-tail-moe-v2-original1024-structure/rlt-variable-1024/coda-moe-tail-evidence.json)

原预算普通混批的输入准备试验把请求索引、query offsets 放在一次异步传输，合并
token IDs、positions、seq lengths、logit 行和 padding 的 GPU 写入。后续仍交给原生
`InputBatch`、块表、metadata 和 sampler。GPU 7 正式三波为 986.85、986.33、986.21
tok/s，TTFT P95 为 1,261.76 ms。同卡原生固定三轮为 1,016.25，候选达到 97.05%，每波
相差约 122 ms。对 CODA 末层对照，96/96 完整 token 序列一致。Triton 解释器的混合批
核对也覆盖了实际 token、位置、请求长度、输出行索引和补齐尾部。
[输入准备融合三波](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/decode-optimizations/w3-1000-gate/gpu7/align-prefill-fused-inputs-v2-original1024/rlt-variable-1024/summary.json)

在同预算内，改成同一次 FA4 处理 P/D 得到 944.49 tok/s。结构波 608 个真实 P/D 批的
5,808 次 attention 都进入该路径。移动 PIECEWISE attention 的断点、复用 FA4 分片计划
和 rebased offsets 的独立尝试也未缩短整波。WCPB 的普通图对照为 210.65 tok/s；扩大
合法图复用后的一个独占波为 718.44，后两波有外部进程。独立结构波确认 2,704 次物理
执行均实际回放，32 条请求的逐位置轮次均通过。该 WCPB 组合仍低于门槛，续测与正式
通路分开保留。自投机与 Decode pipeline 在原配置约 169.76 tok/s，使用异步 draft
CODA 的筛选波为 153.89，与同卡旧版 32/32 完整相同。
[实际 P/D 共用 FA4 的执行证据](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/decode-optimizations/w3-1000-gate/gpu7/align-prefill-joint-fa4-original1024-structure/rlt-variable-1024/joint-fa4-evidence.json)

同一原预算 RLT 架构固定三轮达到 1,006.13 tok/s，TTFT P50 / P95 为
153.92 / 1,226.53 ms。相对原生固定三轮，RLT 固定三轮整波多约 41 ms。保持同一版
RLT 切到动态深度，整波再多约 82 ms。组件事件逐项覆盖到输入准备、attention metadata、
图 staging 和状态交接。含 prefill 的 CORE 中，状态组装范围约 35.89 ms；全部 merged
CORE 的状态分流约 36.25 ms。范围包含内部提交间隔，且属于模型执行范围。
[固定三轮与原生的输出对照](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/decode-optimizations/w3-1000-gate/original1024-fixed3-selective-parity.json) ·
[原预算设备分项](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/decode-optimizations/w3-1000-gate/gpu7/align-prefill-fused-inputs-v2-original1024-components/rlt-variable-1024/structure.json)

分流融合按 MRv2 的真实请求顺序，一次拷贝终轮及续轮所需的 stack、bank 和输入。
续轮批在已有 FULL/PIECEWISE 许可档内生成补齐后的状态。下一 op 的请求、排序、真实
行以及图宽度都一致时，直接取得整批状态。其它批仍按请求行组装。GPU 7 保持原始
图档，正式三波为 1,005.70、1,005.85、1,007.41 tok/s，TTFT P50/P95 为
221.02 / 1,236.35 ms。相对同卡 986.33 提升 1.98%，96/96 完整输出一致。该组在
原始预算下首次超过 1,000 tok/s，相对同卡原生仍差 10.40 tok/s。
[原预算续轮整批复用三波](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/decode-optimizations/w3-1000-gate/gpu7/align-prefill-reuse-routed-batches-original1024/rlt-variable-1024/summary.json)

独立结构波统计到 631 次整批续轮状态复用，其中纯 decode FULL 388 次、P/D
PIECEWISE 234 次、纯 prefill PIECEWISE 9 次。1,659 个计划各有一次真实物理图回放。
三个 op 的 P/D 同批数分别为 126、360、122，32 个请求的逐位置轮数通过。
[整批续轮实际命中与设备分项](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/decode-optimizations/w3-1000-gate/gpu4/align-prefill-reuse-routed-batches-original1024-routed-batch-structure/rlt-variable-1024/structure.json)

以下对照均使用 1,024 token 预算和相同 PIECEWISE 图档。第二组在两边同时使用
vLLM 原生的 FULL decode 请求数图档。原有图档的对照单独保留。

| GPU | FULL 图档 | 运行路径 | 吞吐中位数（tok/s） | TTFT P50/P95（ms） |
| --- | --- | --- | ---: | ---: |
| 4 | 原始 | 原生固定三轮 | 1,014.52 | 153.43 / 1,230.35 |
| 4 | 原始 | 动态 RLT，整批复用、状态准备与 decode split-KV | 1,011.43 | 220.12 / 1,233.46 |
| 4 | 原始 | 动态 RLT，再合并图输入拷贝 | 1,012.29 | 219.92 / 1,236.19 |
| 7 | 原生 | 原生固定三轮 | 1,015.69 | 154.01 / 1,229.30 |
| 7 | 原生 | 动态 RLT，整批复用与状态准备 | 1,012.52 | 220.19 / 1,232.96 |
| 7 | 原生 | 动态 RLT，再限制纯 decode split-KV | 1,014.99 | 219.99 / 1,232.27 |
| 7 | 原生 | 动态 RLT，再合并图输入拷贝 | 1,015.20 | 220.07 / 1,233.03 |

图输入拷贝的三波在两卡都保持 GPU 独占和 2,032 MHz，且分别对本卡上一步的
完整输出 96/96 一致。GPU 7 的同图档吞吐还差 0.49 tok/s。把 CODA 的尾部
掩码并到普通输入 Triton 构造从同图档 split-KV 对照的 1,014.99 降到 1,014.09，
保留于实验快照。split-KV 复用原生 `rel_decode_attention`，限制纯 decode 的 CTA 分片，
保持混合批的 attention 路径。与同卡融合输入基准相比，部分纯 decode 请求的
完整输出存在差异，GPU 4 原图档组合为 93/96，一处可复现的首分歧发生在第 2 个
生成 token。两边改用同一 FULL 图档的 GPU 7 比较仍保留约 66 ms 的 TTFT P50 差距。
[GPU 4 原图档原生](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/decode-optimizations/w3-1000-gate/gpu4/compact-mixed-core-filtered-original1024-clean-r3/native-fixed3-1024/summary.json) ·
[GPU 4 原图档动态 RLT](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/decode-optimizations/w3-1000-gate/gpu4/align-prefill-fuse-graph-copies-original1024/rlt-variable-1024/summary.json) ·
[GPU 7 原生图档原生](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/decode-optimizations/w3-1000-gate/gpu7/compact-mixed-core-filtered-original1024-native-decode-grid/native-fixed3-1024/summary.json) ·
[GPU 7 原生图档动态 RLT](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/decode-optimizations/w3-1000-gate/gpu7/align-prefill-fuse-graph-copies-original1024-native-decode-grid/rlt-variable-1024/summary.json) ·
[CODA 掩码融合消融](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/decode-optimizations/w3-1000-gate/gpu7/align-prefill-fuse-coda-mask-original1024-native-decode-grid/rlt-variable-1024/summary.json)

FULL 图捕获覆盖 1 到 32 个合法请求数时，GPU 7 的相同源码 RLT 为 1,015.44 tok/s，
同图档原生固定三轮为 1,017.06。RLT 比原生请求数图档多捕获 119 个 per-op 图，启动
抓图日志约 22 秒对 20 秒，预热阶段采样显存峰值增加 208 MiB。RLT 两种图档完整输出
96/96 一致。这个对照保留原来的 1,024 预算与 PIECEWISE 图档。
[GPU 7 全尺寸 FULL 动态 RLT](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/decode-optimizations/w3-1000-gate/gpu7/align-prefill-fuse-graph-copies-original1024-exhaustive-full-grid/rlt-variable-1024/summary.json) ·
[GPU 7 全尺寸 FULL 原生](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/decode-optimizations/w3-1000-gate/gpu7/align-prefill-fuse-graph-copies-original1024-exhaustive-full-grid/native-fixed3-1024/summary.json)

复用普通输入已识别的纯 decode 标记与连续张量图键后，GPU 4 原图档筛选快照为
1,013.77 tok/s。应用同一思路后的工作区源码独立冻结重跑为 GPU 4 原图档 1,011.18，
GPU 7 原生请求数图档 1,014.37，两组对其对应 RLT 对照完整输出均为 96/96。
工作区版本保留了回到原生 forward 时的 split-KV 隔离与 Warmup 的配置检查。
开发中以冻结工作区源码的复测数字为准。CODA 只消费上批结果就借用共享 stack、把
终轮不读取的 prelude tensor 移除、提前构造帧、缩小纯 decode CTA 分片均未达到
同卡对应对照的吞吐。没有生成 token 的整批 CODA 复用空采样结果为 1,012.30，
对照是 1,012.29，逐 token 96/96 一致。
[GPU 4 优化筛选快照](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/decode-optimizations/w3-1000-gate/gpu4/align-prefill-direct-graph-layout-original1024/rlt-variable-1024/summary.json) ·
[GPU 4 原档工作区源码](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/decode-optimizations/w3-1000-gate/gpu4/original1024-workspace-graph-layout-original1024/rlt-variable-1024/summary.json) ·
[GPU 7 原生请求数图档工作区源码](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/decode-optimizations/w3-1000-gate/gpu7/original1024-workspace-graph-layout-native-decode-grid/rlt-variable-1024/summary.json)

纯 decode 输入准备现在把 KV slot 写入原生 MRv2 的 slot buffer，复用 MRv2 的
block-table 指针、组内 block size 和禁用位。还原 padding 的同时继续调用原生
`gather_block_tables()`，混批的 KV 路径仍由 MRv2 准备。状态重组保留 CUDA Graph
需要的 stack 容量，只拼接下一 op 会读取的 blocks。两项叠加的工作区源码冻结标识为
`06a6972`。

| GPU | FULL 图档 | 动态 RLT（tok/s） | 同卡原生固定三轮（tok/s） | RLT TTFT P50/P95（ms） | 原生 TTFT P50/P95（ms） |
| --- | --- | ---: | ---: | ---: | ---: |
| 4 | 原始 | 1,012.77 | 1,014.52 | 219.86 / 1,232.22 | 153.43 / 1,230.35 |
| 7 | 原生请求数图档 | 1,016.86 | 1,015.69 | 219.71 / 1,230.51 | 154.01 / 1,229.30 |

GPU 7 RLT 三波为 1,017.02、1,016.84、1,016.86 tok/s。吞吐中位数领先 0.115%，
每波均独占 GPU，SM 为 2,032 MHz。对工作区前一个冻结源码，完整输出 96/96 一致。
TTFT P50 仍多约 66 ms，TTFT P95 多约 1.2 ms。原始 FULL 图档单列，GPU 4 的
组合相对同卡原生还差 1.75 tok/s。两边始终保持 1,024 token 和 80 GiB KV 的预算，
原生请求数图档只改变 FULL，PIECEWISE 继续用原来的 12 档。
[GPU 4 原始 FULL 图档组合优化](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/decode-optimizations/w3-1000-gate/gpu4/original1024-fused-kv-live-state-original1024/rlt-variable-1024/summary.json) ·
[GPU 7 原生 FULL 图档组合优化](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/decode-optimizations/w3-1000-gate/gpu7/original1024-fused-kv-live-state-native-decode-grid/rlt-variable-1024/summary.json) ·
[完整启动参数及源码和语料哈希](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/decode-optimizations/w3-1000-gate/gpu7/original1024-fused-kv-live-state-native-decode-grid/rlt-variable-1024/configuration.json)

原生请求数图档的倒序复跑中，GPU 7 新鲜原生为 1,016.63，RLT 为 1,016.12。RLT
两次的完整输出 96/96 一致。该组合的一次领先未在倒序复跑重现。
[倒序原生摘要](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/decode-optimizations/w3-1000-gate/gpu7/original1024-fused-kv-live-state-native-decode-grid-aba-r1/native-fixed3-1024/summary.json) ·
[倒序 RLT 摘要](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/decode-optimizations/w3-1000-gate/gpu7/original1024-fused-kv-live-state-native-decode-grid-aba-r1/rlt-variable-1024/summary.json)

MRv2 一次 attention metadata 准备中的 SConv 各 KV 组使用同一份 query offsets，
原生 builder 原来分别计算 token 对应请求，以及该请求的起始 token 行。RLT 现在
复用首个原生 builder 的行映射。每组 block table 和 slot mapping 仍由 MRv2 单独
提供，capture 与运行共用这条路径。在工作区前一冻结版本上单独启用，GPU 4 原图档
从 1,011.18 提升到 1,016.33 tok/s，完整输出 96/96 一致。相对此前同卡原生
1,014.52 的增幅为 0.18%。下一步用新鲜原生基线核对同路径叠加 KV 与状态优化。
[原预算原图档 SConv 独立验证](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/decode-optimizations/w3-1000-gate/gpu4/original1024-share-sconv-sequence-original1024/rlt-variable-1024/summary.json)

##### 原预算、原图档的组合验收

最终工作区源码冻结标识为 `b3a13ba`。在 SConv 行映射复用之上，保留纯 decode
输入与 KV slot 融合、只拼接下一 op 会读取的状态块。两边使用原 1,024 token 预算、
原 FULL 和 PIECEWISE 图档、80 GiB KV、TP1、异步调度与 chunked prefill。W3 C16
每波 32 条 4,096 输入 token、128 输出 token。RLT 每条请求的 prefill 为三轮，
decode 分布为 8 条一轮、8 条两轮、16 条三轮。原生以固定三轮运行同一语料。

| GPU | 测量顺序 | 动态 RLT（tok/s） | 原生固定三轮（tok/s） | RLT TTFT P50/P95（ms） | 原生 TTFT P50/P95（ms） |
| --- | --- | ---: | ---: | ---: | ---: |
| 4 | 原生 → RLT | 1,016.19 | 1,013.69 | 219.65 / 1,232.08 | 153.52 / 1,229.85 |
| 7 | RLT → 原生 | 1,018.06 | 1,015.63 | 219.51 / 1,230.60 | 153.39 / 1,228.14 |
| 7 | 原生 → RLT 复测 | 1,019.93 | 1,015.63 | 219.21 / 1,229.90 | 153.39 / 1,228.14 |

GPU 4 原生的三波为 1,013.41、1,013.69、1,014.45 tok/s。RLT 三波为
1,016.19、1,016.58、1,016.09，中位数领先 0.25%。GPU 7 的 RLT 三波为
1,018.10、1,018.06、1,018.06，中位数领先 0.24%。两卡各自的三波 RLT 最低值
都高于本卡原生的三波最高值。正式波无 profiler，GPU 独占且 SM 为 2,032 MHz。
两卡 RLT 完整输出 96/96 一致，且与同图档上一动态版本完整输出 96/96 一致。
GPU 7 的新鲜原生三波为 1,015.63、1,015.56、1,015.76；其后 RLT 复测三波为
1,019.93、1,019.94、1,019.91，中位数领先 0.42%。同卡两次 RLT 的完整输出
96/96 一致。相对于原生，96/96 首 token 一致，两边均为固定三轮的请求每波有
14/16 完整输出一致。TTFT P50 多约 66 ms，TTFT P95 多约 1.8–2.5 ms。
[GPU 4 新鲜原生](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/decode-optimizations/w3-1000-gate/gpu4/original1024-share-sconv-sequence-original1024-aba-r1/native-fixed3-1024/summary.json) ·
[GPU 4 最终 RLT](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/decode-optimizations/w3-1000-gate/gpu4/original1024-share-sconv-fused-kv-original1024/rlt-variable-1024/summary.json) ·
[GPU 7 新鲜原生](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/decode-optimizations/w3-1000-gate/gpu7/original1024-share-sconv-fused-kv-original1024/native-fixed3-1024/summary.json) ·
[GPU 7 最终 RLT](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/decode-optimizations/w3-1000-gate/gpu7/original1024-share-sconv-fused-kv-original1024/rlt-variable-1024/summary.json) ·
[GPU 7 反序复测](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/decode-optimizations/w3-1000-gate/gpu7/original1024-share-sconv-fused-kv-original1024-aba-r1/rlt-variable-1024/summary.json) ·
[最终 RLT 的完整启动参数与哈希](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/decode-optimizations/w3-1000-gate/gpu7/original1024-share-sconv-fused-kv-original1024/rlt-variable-1024/configuration.json) ·
[原预算最终双卡验收汇总](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/decode-optimizations/w3-1000-gate/original1024-b3-gate-summary.json)

GPU 4 独立结构波中，32 条请求的逐 token 位置及轮数全部通过，全波输出逐 token 与
正式波一致。1,659 个物理 forward 全部实际回放 CUDA Graph，其中 1,004 个纯
decode FULL，655 个含 prefill 的 PIECEWISE。实际 P/D 同一物理批在 PRELUDE、
CORE、CODA 分别有 126、360、122 次。805 个首次准备的纯 decode 批全量对比
MRv2 原生 slot buffer，覆盖其中 320 个 graph padding 批，共比较 9,891,840 个
元素，差异为零。六个 SConv 组的请求行映射在每个 forward 都共享原地址，各组的
block table 与 slot mapping 地址分别保留。1,331 批实际准备只调用一次原生 SConv
行映射 helper，328 批复用已有 metadata，无需再次构造。
[最终 CUDA Graph、P/D、请求轮数及原生 slot 逐元素验证](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/decode-optimizations/w3-1000-gate/gpu4/original1024-share-sconv-fused-kv-original1024-structure/rlt-variable-1024/structure.json)

### 终版 W1–W7 与 W3 C16 性能复测

使用与工作区源码逐文件一致的终版快照 `b3a13ba` 和固定 Inkling2-Mini VarLoop 语料。
每个 case 在同卡跑原生固定三轮、RLT 固定三轮和 RLT 请求级可变轮次三组，两个
RLT 组都开启 P/D mixed batch 和混深度 CORE 合批，沿用 W3 C16 吞吐验收的启动参数。
预算为 1,024 token，KV 为 80 GiB，FULL/PIECEWISE 图档沿用原配置。固定长度组忽略
EOS，ShareGPT 两档使用同一批 64 条请求，自然 EOS 且上限 512。每组每档完整预热后
测三波无 profiler，共 72 波。

GPU 4 测 W1、W4、W5、W6，GPU 7 测 W3 C16、W3、W2、W7。动态组固定 25% 一轮、
25% 两轮、50% 三轮，三轮请求中显式指定和省略参数各占一半。固定三轮按
`RLT 吞吐 / 原生吞吐` 和 `原生 TTFT P95 / RLT TTFT P95` 各至少达到 95% 判定，
同时记录 TTFT P50、TPOT、ITL 和各轮次的延迟。
[复测脚本与固定语料配置](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/timed/final_b3_matrix_20261127.py)

72 波均满足 GPU 独占，采样 SM 为 2,032 MHz，三组实际 KV 容量均为 291,018 token。
按三波中位数汇总，固定三轮吞吐 8/8 档过 95%，范围为原生的 98.95%–100.18%。
固定三轮 TTFT P95 通过 7/8 档，W6 为 92.02%。

| Case | GPU | 原生固定三轮（tok/s） | RLT 固定三轮（tok/s） | RLT 可变轮次（tok/s） | 可变相对 RLT 固定 | TTFT P95 原生 / RLT 固定 / RLT 可变（ms） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| W1 | 4 | 260.36 | 257.64 | 326.69 | +26.80% | 12.41 / 12.91 / 12.91 |
| W2 | 7 | 3,840.30 | 3,819.49 | 3,725.85 | −2.45% | 97.53 / 101.33 / 105.30 |
| W3 C1 | 7 | 229.35 | 227.27 | 278.47 | +22.53% | 82.62 / 81.96 / 81.79 |
| W3 C16 | 7 | 1,015.78 | 1,012.37 | 1,017.38 | +0.49% | 1,228.57 / 1,226.35 / 1,236.75 |
| W4 | 4 | 1,212.24 | 1,208.20 | 1,214.14 | +0.49% | 2,551.42 / 2,542.63 / 2,546.53 |
| W5 | 4 | 1,361.76 | 1,360.95 | 1,399.61 | +2.84% | 153.91 / 153.46 / 154.02 |
| W6 | 4 | 1,410.03 | 1,412.55 | 1,355.55 | −4.04% | 70.47 / 76.59 / 84.37 |
| W7 | 7 | 3,741.76 | 3,742.95 | 3,915.73 | +4.62% | 254.52 / 260.60 / 264.34 |

W3 C16 可变轮次三波为 1,016.12、1,017.38、1,018.37 tok/s，均高于本轮原生最快
的 1,015.89，继续通过吞吐门槛。三组 TTFT P50 为 153.58、153.21、220.62 ms。
W2 可变轮次吞吐三波为 3,446.23、3,725.85、3,967.98 tok/s。W6 可变轮次的 TTFT
P95 三波为 1,750.47、84.37、83.88 ms，P99 三波为 2,564.32、1,656.97、1,645.22 ms。
各波均保留。W6 可变轮次每波同一条默认三轮请求在 357 token 达到 EOS，其余到
512。W6 两组固定三轮与 W7 三组的所有请求均到 512。

固定三轮对原生的完整输出一致数，W2 为 83/192、W6 为 22/192、W7 为 30/192。
对应首 token 一致数为 192/192、191/192、192/192。完整逐波结果记录请求/s，TTFT、
TPOT、ITL 的分位数，按深度统计，以及首次 token 分歧位置。
[72 波汇总与逐波原始数据路径](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/w1w7-depth/final-b3-matrix-20261127/summary.json) ·
[运行计划与卡位](../artifacts/plugin2-perf.9pulvo/metadata-graph-opt/w1w7-depth/final-b3-matrix-20261127/plan.json)

### W3 C16 调度优先级复测

在 `rlt_finish_decode_token` 路径中，已就绪 CORE 连续因同轮对齐落选后进入已有
等待优先级，触发次数为 `max(1, num_loops - 1)`。其次优先能交付首 token 的
prefill CODA。普通 decode CODA 在已就绪的 CORE 能于剩余等待额度内补进同一批
且预算放得下时短等。每次选定一个 op 后，继续合入兼容且预算放得下的请求。
同步与异步调度回归覆盖两次轮次冲突后推进，以及等待额度不足时交付 CODA。
插件测试 93 通过、2 跳过。

W3 C16 使用原 1,024 token 预算、原图档和 80 GiB KV，32 个 4,096-token 请求
各生成 128 token，动态组实际轮数为 8/8/16，P/D 同批开启。在 GPU 7 完整预热后
分别跑三波。各正式波 GPU 独占，采样 SM 为 2,032 MHz。

| 运行组 | 吞吐中位数（tok/s） | TTFT P95（ms） |
| --- | ---: | ---: |
| 原生固定三轮 | 1,015.85 | 1,228.46 |
| b3 动态，原策略 | 1,019.69 | 1,230.09 |
| 1→2→3 动态，首轮 | 919.28 | 1,422.07 |
| 1→2→3 动态，原样复测 | 1,019.81 | 1,229.78 |
| 3→2→1 动态，单独启动 | 1,016.59 | 1,232.40 |

保留首轮回退数据。相同源码和参数的复测未重现回退，首轮慢因未定位。另做两组
独立消融，等待上限加首 token 优先在 GPU 7 为 1,018.26 tok/s，等待上限加严格
CODA 让行在 GPU 4 为 1,015.88 tok/s，同卡 b3 为 1,018.10。
[首轮原始正式波](../artifacts/w3-scheduler-priority-20261127/comparison.json) ·
[1→2→3 原样复测](../artifacts/w3-scheduler-priority-20261127/gpu7/candidate/combined-retake/summary.json) ·
[3→2→1 独立正式波](../artifacts/w3-scheduler-priority-20261127/gpu7/candidate/reverse-321/summary.json)

两个独立结构波均为 1,659 个逻辑计划与实际 forward，FULL/PIECEWISE 为
1,004/655，32 条请求逐位置轮数通过。实际 P/D 的 PRELUDE/CORE/CODA 次数分别为
126/360/122。分别对任一结构波中的相同就绪集合评估两种顺序，1,333 次判断仅
一处有分歧。1→2→3 先跑已等待的 CORE，3→2→1 先发首 token CODA。第 3 项
未在该波与前两项争抢同一步。
[1→2→3 同就绪集合检查](../artifacts/w3-scheduler-priority-20261127/gpu4/candidate-structure/priority-123-shadow/priority-shadow.json) ·
[3→2→1 同就绪集合检查](../artifacts/w3-scheduler-priority-20261127/gpu7/candidate-structure/reverse-321/priority-shadow.json)

同一热服务内交替执行两种顺序，每种三波。正序和倒序的吞吐中位数分别为
1,018.76 和 1,018.82 tok/s，三对吞吐差为 −0.007%、−0.003%、+0.134%，
96 条完整输出全部一致。工作区保留正序及其防饥饿逻辑。
[同一热服务六波数据](../artifacts/w3-scheduler-priority-20261127/gpu7/priority-interleaved/summary.json)

#### 动态轮次的费用模型

GPU 4 分别采集原生固定三轮、RLT 固定三轮和 RLT 动态轮次各一波。诊断波
用 CUDA event 量区间，正式吞吐以无插桩波为准。两组 RLT 的逐位置轮数均通过。
三组首 token 均为 32/32 一致；RLT 两组均为三轮的 16 条请求有 14 条完整输出一致。

| RLT 诊断项 | 固定三轮 | 动态轮次 | 变化 |
| --- | ---: | ---: | ---: |
| 实际 decode CORE 行 | 12,192 | 9,144 | −25.0% |
| 纯 decode CORE 提交 | 591 | 600 | +9 |
| 纯 decode CORE 图宽合计 | 8,436 | 6,980 | −17.3% |
| 含 prefill 的 CORE 提交 / 图宽合计 | 393 / 398,208 | 393 / 398,208 | 0 |
| CORE 模型区间（ms） | 3,076.49 | 3,015.57 | −60.91 |
| CODA 模型区间（ms） | 543.66 | 565.58 | +21.92 |
| forward 内、模型外（ms） | 105.93 | 121.18 | +15.25 |
| 采样区间（ms） | 72.61 | 77.75 | +5.14 |
| 完整设备流跨度（ms） | 4,282.16 | 4,256.48 | −25.68 |

到最后一个含 prefill 的 forward 为止，设备流为 3,467.54→3,470.59 ms。
之后的纯 decode 尾部为 814.62→785.89 ms。动态组的净收益主要来自尾部。

在这组长上下文纯 decode 中，先按 CUDA Graph 补齐宽度估计每次 CORE 的模型
时间。用固定组各档的平均耗时和动态组各档的批数，预测动态组为 894.29 ms，
实测 887.64 ms，误差 6.65 ms。固定组 B8 和 B16 平均分别为 1.369 和 1.649 ms。
若两批 B8 在同一物理 CORE 路径上可合，合成 B16 估计节省 1.09 ms。动态组有
145 个实际 12 行的纯 decode CORE 批仍执行 B16。按实际行数等比例折算耗时会
高估收益。该模型以 W3 C16 的一对诊断波校验，其他长度、图模式和模型需要单独标定。

调度比较同一就绪集合中的可执行组批方案。先满足数据依赖、首 token 和等待上限，
在这些约束内，按可减少的物理提交及其图档费用排序。需等待的合批要计入因此
增加的关键路径时间。CORE 只计入 Worker 能实际合入同一物理批的工作。
[分段 Host/Device 时间轴](../artifacts/w3-scheduler-priority-20261127/gpu4/time-attribution/timeline.html) ·
[图档模型和区间汇总](../artifacts/w3-scheduler-priority-20261127/gpu4/time-attribution/analysis.json) ·
[原始采集校验](../artifacts/w3-scheduler-priority-20261127/gpu4/time-attribution/validation.json)

#### 第三条补齐纯 decode CORE 合批

此前等待合批的判断覆盖已就绪的 decode CODA。有 prefill CORE 时，调度可以按
当前轮次对齐；纯 decode 的相邻两轮则可能一起进入一份调度计划，随后由 Worker
拆为两次物理提交。只改变选首条请求的优先级无法把这两批合成一个。

现有 `rlt_finish_decode_token` 与 `rlt_merge_decode_core` 同时开启时，在支持
单 token FULL 合批的普通路径上检查相邻两轮的 CORE。如果较浅的一组执行一轮后
能全部赶上，合批能落在配置的 FULL 图档，且补齐总行数和调度预算允许，就先发
较浅一组。等候的请求进入现有等待优先级，避免再次为该下轮合批让行。已有
CODA 能出 token 时优先使用原有 CODA 规则。同步和异步回归覆盖 4 条第 1 轮、
12 条第 2 轮时先发 4 条、随后发 16 条，以及浅轮会直接结束时不等待。

首次 GPU 启动的冻结副本缺少插件注册元数据，补齐后服务可用。另一次复测中，
Worker 在 UniProc 内把 EngineCore 共享的图模式改成了 NONE，导致调度器无法
开启这条规则。Worker 现在持有本地编译配置来关闭原生整模型图，调度器仍读取
EngineCore 的原始图档配置。已测试 Worker 图模式和共享 KV 配置的所有权。
插件测试 98 通过、2 跳过。

上轮 W3 C16 的 993 个 CORE 调度计划均只含一个当前轮次。145 个 B12 中，
141 个在第 2 轮，其中 130 个紧接同一批请求的第 1 轮 B16，少掉的四条均已
达到请求指定的一轮上限。这类空位无法靠让同一批的请求赶上来补齐。
[B12 成因与轮次校验](../artifacts/w3-scheduler-priority-20261127/gpu4/time-attribution/core-coalescing-opportunities.json)

修复配置后，GPU 4 独立结构波确认调度器读取到 `1, 2, 4, 8, 16, 24, 32`，
与 Worker 实际捕获一致。600 次纯 decode CORE 检查均只有一个当前轮次就绪，
等待合批触发零次。32 条请求逐位置轮数校验通过，1,659 个物理 forward 均回放
CUDA Graph，FULL/PIECEWISE 为 1,004/655。
[第三条触发计数](../artifacts/w3-scheduler-priority-20261127/gpu4/core-coalescing-structure-worker-config/core-defer-counts.json) ·
[结构验收](../artifacts/w3-scheduler-priority-20261127/gpu4/core-coalescing-structure-worker-config/structural-summary.json)

GPU 7 使用同一热服务，原预算 1,024、原 FULL/PIECEWISE 图档及 80 GiB KV，
交错测试开关各三波。正式波均 GPU 独占、SM 2,032 MHz，96 条完整输出一致。

| 新增纯 decode CORE 等待合批 | 吞吐中位数（tok/s） | TTFT P95（ms） |
| --- | ---: | ---: |
| 关 | 1,020.41 | 1,230.51 |
| 开 | 1,020.04 | 1,231.26 |

三对的吞吐差为 −0.016%、+0.027%、−0.157%。这次 W3 没有验证到吞吐收益。
[同服务交错六波数据](../artifacts/w3-scheduler-priority-20261127/gpu7/core-coalescing-interleaved-worker-config/summary.json)

#### 普通 prefill CODA 按剩余预算再切分

`--rlt-split-prefill-coda` 默认为关。开启后仅在普通 Inkling prefill 已完成全部
CORE、没有在途操作、且本次 CODA 存在剩余预算时考虑切分。先选预算内放得下的
完整段，仍有空位时再从一个超出剩余预算的段切出前缀。每个原段最多切一次，
尾段在前缀 CODA 回执后才能入批。

Scheduler 保留已预留的 KV 块，分别提交两段的 token 区间。Worker 在读取完成
CORE 的中间状态时按行取前缀，并保留尾段的视图。已有 MRv2 请求进度和块表用于
计划中的实际 token 位置。前缀不采样，每次 CODA 完成只提交对应的已计算前缀，
只有含原 prompt 最后一个 token 的尾段能交付首 token。

CPU 覆盖 1 个 decode token 加 7 个 prefill token、之后尾段 1 个 token 的执行次序，
验证同步/异步的首 token、KV 复用、prefix cache 提交，以及取消时等在途 CODA
返回后释放。Worker 状态测试验证尾段取到原位置的行。插件测试 103 通过、2 跳过。

W3 C16 动态轮次固定原 1,024-token 预算、原图档、80 GiB KV。GPU 7 同一热服务
做关闭和开启各三波。吞吐中位数 1,018.13→1,018.09 tok/s，TTFT P95
1,229.85→1,229.67 ms。96 条输出逐 token 一致。GPU 4 独立自然诊断中未触发
切分，仍有 1,659 个实际 forward。131 个含 prefill 的 CODA 中 124 个已经
装满 1,024 行，4 个已有 1,023 行。
[W3 正式六波](../artifacts/w3-scheduler-priority-20261127/gpu7/prefill-coda-split-interleaved/summary.json) ·
[自然切分计数](../artifacts/w3-scheduler-priority-20261127/gpu4/prefill-coda-split-diagnostic-retry/normal-cuda.splits.json)

定向 GPU 正确性把一拍的预算从 1,024 临时收紧到 1,023。B 的已就绪 prefill
CODA 长 1,023，A 的 decode CODA 长 1。开启时把 B 的前 1,022 行与 A 合批，
B 的最后一行随后单独回放 PIECEWISE。关闭时 A 先发，B 的 1,023 行整批随后发。
两组各 1,660 个物理 forward，逐位置通过，32 条完整输出逐 token 一致。这个
定向波用于验证真实切分和后续 KV 接续。
[W3 定向 GPU 校验](../artifacts/w3-scheduler-priority-20261127/gpu4/prefill-coda-split-diagnostic-targeted/summary.json)

W7 用原 ShareGPT 语料 64 条、C32、异步调度、动态轮次 16/16/32。输入 16–2,505
token，中位数 84.5；每波输出均为 32,768 token，允许自然 EOS，实际均达到 512
上限。GPU 7 用同一热服务交错开关各三波，原 1,024-token 预算和图档。吞吐
中位数 3,922.27→3,911.80 tok/s，TTFT P50/P95 69.91/261.82→68.45/260.61 ms。
TPOT P95 8.027→8.048 ms，ITL P95 8.008→8.024 ms。跨开关首 token 190/192
一致、完整输出 150/192 一致。在开关各自的重复波内也发生了相当规模的输出差异。
[W7 正式六波](../artifacts/w3-scheduler-priority-20261127/gpu7/prefill-coda-split-W7-interleaved/summary.json)

GPU 4 的独立自然结构波触发一次切分，把一段 11 行的 CODA 分成 5 和 6。
相邻三批的真实行从 1,019/794/235 变为 1,024/1,018/6，图档从
1,024/1,024/256 变为 1,024/1,024/8，共少 248 个图档行。两臂仍各为
23 个含 prefill 的 CODA、5,199 个物理 forward 和 1,040 次采样。两臂逐位置
校验通过，64 条完整输出逐 token 一致。这个 W7 验证了自然负载能触发，吞吐
测试没有显示收益，实验开关保持默认关闭。
[W7 自然结构对照](../artifacts/w3-scheduler-priority-20261127/gpu4/prefill-coda-split-W7-structure/summary.json)

#### 整波吞吐费用模型

物理批按 op、当前 CORE 轮次、图模式和补齐后的图宽归档。用 GPU 4 固定三轮
诊断波标定每档的平均模型区间和 forward 内模型外区间。整波模型以各档批数乘
对应平均费用求和，再计入采样调用数乘平均采样费用及基准未覆盖的流区间。
动态组 39 个档均有固定组标定样本，预测只使用动态组的批数和采样次数。

| W3 C16 动态组诊断项 | 预测（ms） | 实测（ms） |
| --- | ---: | ---: |
| PRELUDE | 471.45 | 466.36 |
| CORE | 3,028.57 | 3,015.57 |
| CODA | 547.44 | 565.58 |
| Forward 内模型外 | 106.24 | 121.18 |
| 采样及未覆盖区间 | 88.47 | 87.79 |
| 整波设备流 | 4,242.16 | 4,256.48 |

整波偏差 −14.32 ms（−0.34%）。相对于固定组，预计设备流吞吐增益 +0.94%，
实测 +0.60%。这是已知批计划条件下的一对诊断波校验，尚未验证跨负载的绝对
时间和请求级 TTFT。CODA 和模型外区间的单项低估分别为 18.14 和 14.95 ms。

等输出 token 数下，要提高吞吐比例 (g)，整波时间需少 (Tg/(1+g))。
从无插桩基准计算，W3 C16 的 +3%/+5% 需要省 117/192 ms，W7 需要
243/398 ms。W7 切分只改变三个相邻 CODA 批，CUDA event 区间合计
6.71→5.85 ms。在其余开销相同的条件下，0.85 ms 对 W7 整波约 0.01%
吞吐，远低于整波 1% 所需的 83 ms。
[整波性能模型脚本](../artifacts/w3-scheduler-priority-20261127/gpu4/time-attribution/performance_model.py) ·
[40 档标定批数、均值及逐项乘积](../artifacts/w3-scheduler-priority-20261127/gpu4/time-attribution/performance-bucket-costs.csv) ·
[标定、校验和阈值](../artifacts/w3-scheduler-priority-20261127/gpu4/time-attribution/performance-model.json)

对应 W3 动态诊断波，采样 77.75 ms，forward 内模型外 121.18 ms，未标记
区间 10.04 ms。采样按真实产出量计，4,096 token 在 C16 至少需 256 次采样，
实际 335 次，以当前每次平均 0.2321 ms 计，压到理论次数约省 18.34 ms。
W7 的 32,768 token 在 C32 至少需 1,024 次，实际 1,040 次，按相同口径约省
3.65 ms。W3 只有 8 次零产出采样，共 1.57 ms。

W3 forward 内模型外的 121.18 ms 中，85.49 ms 属于 PRELUDE，其中
70.28 ms 属于 P/D 混批。间隔 10.04 ms 仅是 CUDA event 标记之外的区间。
含 prefill 的 CORE 模型区间为 2,127.93 ms，其中 387 批使用 PIECEWISE 1,024。
同一诊断波上，吞吐提高 3% 需省约 124 ms。如果收益全来自这 387 批，
每批平均需省 0.32 ms。实际可省的准备、重组和图 staging 时间仍需按组件测量。

W3 C16 动态诊断波按 GPU 区间无重叠分解，总和为 4,256.48 ms。下表的吞吐敏感度
假设该行耗时降低 10%、同波输出和其他区间保持相同。

| 区间 | 纯 prefill（ms） | P/D（ms） | 纯 decode（ms） | 合计（ms） | 整波吞吐敏感度 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PRELUDE 模型 | 10.65 | 315.64 | 140.07 | 466.36 | +1.11% |
| CORE 第 1 轮模型 | 19.62 | 571.10 | 262.72 | 853.44 | +2.05% |
| CORE 第 2 轮模型 | 77.12 | 694.86 | 328.68 | 1,100.67 | +2.65% |
| CORE 第 3 轮模型 | 77.98 | 687.25 | 296.24 | 1,061.46 | +2.56% |
| CODA 模型 | 78.29 | 335.30 | 151.98 | 565.58 | +1.35% |
| PRELUDE forward 内模型外 | 3.48 | 70.28 | 11.73 | 85.49 | +0.20% |
| CORE forward 内模型外 | 0.71 | 10.14 | 14.41 | 25.27 | +0.06% |
| CODA forward 内模型外 | 0.15 | 4.61 | 5.66 | 10.43 | +0.02% |
| 有输出采样 | 0.19 | 28.70 | 47.30 | 76.18 | +0.18% |
| 零输出采样 | 1.57 | 0 | 0 | 1.57 | +0.004% |
| 标记外区间 | — | — | — | 10.04 | +0.02% |

其中截至最后一个含 prefill 的 forward 为 3,470.59 ms，之后纯 decode 785.89 ms。
Host 累计调度 262.41 ms，forward 提交 3,451.01 ms，采样 81.75 ms，回执更新
29.49 ms。Host forward 包含模型区间，Host 与 Device 并行，不能叠到表内。

同卡固定与动态诊断波的续入 16 条里，入队到首个计划中位数 38.82→107.68 ms，
首次 Worker 提交到首 token CODA 回执 117.87→117.95 ms，客户端 TTFT
161.39→230.74 ms。入队到首计划之间经过的已满 PRELUDE 批中位数由 1 增到 4。
这将该波的 TTFT 差异定位到新请求取得首段调度预算之前。
[三组完整分段表](../artifacts/w3-scheduler-priority-20261127/gpu4/time-attribution/performance-breakdown.csv) ·
[Host 逐请求区间及图档分布](../artifacts/w3-scheduler-priority-20261127/gpu4/time-attribution/performance-breakdown.json)

#### PIECEWISE 图档固定步长回算

用已有物理批的真实行数选择能容纳该批的最小图档。只替换 PIECEWISE 档位，
保持 FULL 请求数档、调度预算和每批成员。小档沿用原始列表，超过 128 或 256
后分别每 128 或 256 取一档。原 1,024 预算的 PIECEWISE 档数由 12 增为
16 或 13。与当前 pin 的 vLLM 原生 grid 同时回算。

| 负载 | 原档物理行 | 每 128 档 | 每 256 档 | vLLM 原生 grid |
| --- | ---: | ---: | ---: | ---: |
| W3 C16 动态，预算 1,024 | 663,680 | 663,040 | 663,680 | 662,800 |
| W7 C32 动态，预算 1,024 | 108,544 | 102,272 | 105,984 | 101,952 |
| 历史 W3 C16，预算 8,192，按同一 trace 的稀疏图档作基线 | 766,208 | 659,072 | 662,528 | 662,440 |

W3 原预算 655 个 PIECEWISE 批仅 5 个可通过 128 档缩小，固定与动态均少
640 行。W7 基线 114 个 PIECEWISE 批中，128 档缩小 29 批，256 档缩小
10 批。若模型时间与图宽成比例，W3/W7 的 128 档预计分别省约 2.63/25.69 ms。
这是几何与时间敏感度回算，新档单批 CUDA 耗时和正式吞吐尚无实测。
历史 8,192 预算的 128/256/vLLM 原生档数为 72/41/54。
[图档回算脚本](../artifacts/w3-scheduler-priority-20261127/gpu4/time-attribution/project_prefill_grids.py) ·
[逐负载档位、物理行和阶段计数](../artifacts/w3-scheduler-priority-20261127/gpu4/time-attribution/prefill-grid-projection.json)

W3 C16 实测保留预算 1,024、80 GiB KV 和原来 7 个 FULL 图档。GPU 7
启动时捕捉 128 步长的 16 档超集，在同一热服务内只切换可选的 PIECEWISE 档。
三个档分别完整预热，在三轮平衡顺序中每组测三波，均采样为 GPU 独占和 2,032 MHz。
早期 GPU 3 的一轮含外部进程，保留原记录，正式表采用 GPU 7 的完整重跑。

| PIECEWISE 档 | 吞吐中位数（tok/s） | TTFT P50/P95（ms） | TPOT P95（ms） | ITL P95（ms） |
| --- | ---: | ---: | ---: | ---: |
| 原 12 档 | 1,020.52 | 219.24 / 1,227.99 | 13.321 | 22.124 |
| 每 128，16 档 | 1,019.93 | 219.24 / 1,228.30 | 13.322 | 22.106 |
| 每 256，13 档 | 1,020.41 | 219.17 / 1,227.70 | 13.323 | 22.096 |

两种步长相对原档均逐 token 一致 96/96。GPU 4 独立结构波均为
1,659 次物理 forward，1,004 次 FULL、655 次 PIECEWISE，32 条请求逐位置通过。
128 的五个新档回放为 896，替代原来的 1,024，少 640 个物理行。这五批模型区间
累计 21.05→20.55 ms，256 没有用到新增的 768 档。原预算下没有测到吞吐收益。
GPU 7 的超集捕捉共 161 个 per-op 图，25.21 秒，capture 接口报告净增加约
1.14 GiB，三组计时共享这些图。
[W3 同服务九波](../artifacts/w3-scheduler-priority-20261127/gpu7/w3-piecewise-grid-interleaved/summary.json) ·
[W3 独立图回放及正确性](../artifacts/w3-scheduler-priority-20261127/gpu4/w3-piecewise-grid-structure/summary.json)

#### KV 容量、分块与物理批成本

按 PRELUDE 提交重建每条在途请求的最新预留 token 位置，按最后 decode CODA
回执释放。W3 C16 的峰值为 16 条、67,092 个逻辑 token，W7 为 32 条、27,780
个逻辑 token。若按每请求 64 token 向上对齐，峰值分别为 67,584 和 29,120。
这是逻辑容量重建。Inkling 实际有 16-token、64-token 两类 KV group，实际占用
需按 group 和滑窗统计。80 GiB 配置报告 291,018 token 的 group-aware 容量，
两组逐位置检查通过，当前证据不支持 KV 容量把两组的并发限制在更低水平。

容量估算按 KV group 统计仍保留的物理块数，再乘对应每块字节。改变块大小会同时
改变尾块浪费、块表长度和访问路径。模型耗时另按 op、轮次、图档、实际行数和可见
history 拟合。W7 第 1 轮纯 decode B32 有 1,003 个实际 32 行的批。按 batch 内
逻辑 history 总长度切四分位，第一、第四组的 history 中位数为 11,906/23,946，
CUDA event 模型区间中位数为 1.660/1.699 ms。它反映同图档内的观察相关性，
还没有隔离各层滑窗、真实 KV 字节或 attention kernel 时间。

研究按请求目标轮次保留 KV 时，prefill 依旧需要三轮；可估计普通固定目标轮次
的请求进入 decode 后，不再访问的高轮次 KV group 有多少可释放或停止扩容。
共享层、复用中的 prefix 和在途读写都应计入仍需保留的块。容量收益在现有分配
开始限制入场或触发抢占的区间验证；block/布局收益在同 batch、同图档下比较
attention 时间和块表准备。两类比较均保留 1,024-token 原调度预算。
[W3 原始 trace](../artifacts/w3-scheduler-priority-20261127/gpu4/time-attribution/rlt-dynamic/dynamic-cuda.trace.json) ·
[W7 原始 trace](../artifacts/w3-scheduler-priority-20261127/gpu4/prefill-coda-split-W7-structure/w7on-cuda.trace.json)

进一步按 vLLM 的滑窗回收规则重放已观测的 PRELUDE 预留和 CODA 回执。
按模型层类型重建为 1 个 full KV group、5 个滑窗 1,024 的 KV group 和 6 个短卷积
状态 group，共享物理分配块为 2 MiB。80 GiB 对应 40,960 块，8,192 上限的
单请求理论容量为 1,153 块，复算服务报告的 291,018 token。重放按已提交位置
回收窗口外的完整块。

| 负载 | 原 trace 峰值块数 | 等效 KV 池最低需求（GiB） | 理想轮次分离后的等效需求（GiB） | 峰值可省 |
| --- | ---: | ---: | ---: | ---: |
| W3 C16 | 3,764 | 7.35 | 6.49 | 11.8% |
| W7 C32 | 2,507 | 4.90 | 4.43 | 9.6% |

等效需求计入保留的空块。理想轮次分离假设 prefill 完成后能按层释放正式 decode
不再访问的高轮 KV，且物理块没有分组浪费。这给出固定实测调度轨迹上的容量估算。
实际按现有 group 释放、重新分组后的峰值以及小 KV 池下的吞吐，仍需用 allocator
统计和实测校准。
[分组容量重放脚本](../artifacts/w3-scheduler-priority-20261127/gpu4/time-attribution/kv_capacity_model.py) ·
[重放结果](../artifacts/w3-scheduler-priority-20261127/gpu4/time-attribution/kv-capacity-model.json)

### WCPB 独立验收

同一 4829d2f 源码在 GPU 7 分别启动三臂。W3 C16 每波 32 条 4,096→128，
动态 decode 轮数 8/8/16，预算 1,024、80 GiB KV、原 FULL/PIECEWISE 图档。
每臂完整预热后测三个普通计时波。三臂保留异步、packed prefill、P/D mixed 等
启动参数，关闭自投机、Decode pipeline 和 prefill CODA 切分。普通 512 单独用
long_prefill_token_threshold=512，控制 chunk 上限的影响。

| 运行组 | 吞吐中位数（tok/s） | TTFT P50/P95（ms） | TPOT P95（ms） | ITL P95（ms） |
| --- | ---: | ---: | ---: | ---: |
| 普通 prefill，原配置 | 1,018.51 | 219.35 / 1,229.69 | 13.34 | 22.11 |
| 普通 prefill，chunk 上限 512 | 991.06 | 372.09 / 1,279.66 | 13.36 | 22.14 |
| WCPB，chunk 上限 512 | 217.66 | 570.85 / 3,067.25 | 69.05 | 99.44 |

WCPB 相对同为 512 的普通 prefill 吞吐降低 78.04%。九波均 GPU 独占、SM
2,032 MHz，KV 容量 291,018 token，启动捕捉 133 个 per-op 图。WCPB 与普通
512 对照首 token 96/96、完整输出 87/96 一致，所有请求均返回 128 token。
只开 WCPB、自动 chunk 341 的 GPU 4 单波筛选为 179.05 tok/s、TTFT P95
5,994.31 ms，该波没有纳入三臂同卡统计。
[W3 同卡三臂计时](../artifacts/w3-scheduler-priority-20261127/gpu7/w3-wcpb-comparison/comparison.json) ·
[自动 chunk 单波](../artifacts/w3-scheduler-priority-20261127/gpu4/w3-wcpb-auto-smoke/wcpb-auto/summary.json)

GPU 4 独立 WCPB 512 结构波核对 32 条请求逐位置的循环。20 条请求共 135 次
提前在前一段 CODA 提交前发下一段 PRELUDE，有 27 个 CORE 计划包含同一请求
的多个 chunk。2,287 个逻辑计划拆成 3,358 个物理 forward，1,349 次 FULL、
357 次 PIECEWISE、1,652 次 eager。eager 的图宽均不在对应捕捉档里，含纯
decode 1,063 次、含 prefill 589 次。对照结构波为 1,659 次物理 forward，
全部回放图。该快照在 WCPB 下关闭了普通输入复用和 padding，普通 decode 完成优先、
CODA 合批、非采样 prefill 提前入队与 CODA 尾层优化也有 WCPB guard。
[WCPB 交错与图回放](../artifacts/w3-scheduler-priority-20261127/gpu4/w3-wcpb-structure/wcpb512/summary.json) ·
[eager 实际图宽归因](../artifacts/w3-scheduler-priority-20261127/gpu4/w3-wcpb-structure/wcpb512/eager-widths.json)

#### WCPB 图回放补齐

Worker 先按物理组执行现有 MRv2 位置更新，把请求槽游标指向本组对应的 chunk
起点。按物理批核对每个请求唯一、起点和长度与 MRv2 一致后，使用现有
FULL/PIECEWISE padding。WCPB 每个物理组仍独立准备输入和 metadata，
保留原 CORE 分组、请求依赖和 KV 预留。

初版同时启用批输入复用和同轮终/续合并，结构波 2,720 次全部图回放，但对原
WCPB 首 token 31/32、完整输出 25/32 一致。该版本只作诊断保留。
[初版诊断波](../artifacts/w3-scheduler-priority-20261127/gpu4/w3-wcpb-graph-replay-structure/wcpb512/summary.json)

只补 padding 的版本已覆盖含 prefill 的批，仍有 1,083 个纯 decode 批回退到
eager。这些批的输入张量已对齐到 FULL 档位，但 MRv2 原调用仍以 `NONE` 构造
attention metadata，导致 metadata 的请求维度与录图时不同。现在物理批在
`LoopExecution` 中携带 padding 选择的图模式，Inkling model state 据此调用 MRv2
构造对齐的 metadata，WCPB 仍逐批准备输入和 metadata。
[仅 padding 的结构波](../artifacts/w3-scheduler-priority-20261127/gpu4/w3-wcpb-graph-padding-structure/wcpb512/summary.json)

CPU 测试覆盖 WCPB 下混批 PIECEWISE、纯 decode FULL、padding 尾部清零、
真实位置保留、位置不匹配拒绝复用，以及独立构造 metadata 时沿用正确图模式，
插件套件 105 通过、2 跳过。

补齐 FULL 元数据模式的版本沿用原 W3 C16 预算 1,024 和 80 GiB KV，在两张卡
独立检查物理执行路径。

| WCPB chunk 上限 | 物理 forward | FULL | PIECEWISE | eager | 逐位置通过 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512 | 3,409 | 2,454 | 955 | 0 | 32/32 |
| 自动 341 | 3,490 | 2,295 | 1,195 | 0 | 32/32 |

512 和 341 分别有 137 和 203 次在前一段 CODA 提交前发下一段 PRELUDE，
36 和 104 个 CORE 计划容纳同一请求的多个 chunk，两波均返回 4,096 个 token。
启动捕捉保持 133 个 per-op 图，含 prefill 的批回放 PIECEWISE，纯 decode 回放 FULL。
[512 结构波](../artifacts/w3-scheduler-priority-20261127/gpu4/w3-wcpb-graph-metadata-structure/wcpb512/summary.json) ·
[自动 341 结构波](../artifacts/w3-scheduler-priority-20261127/gpu3/w3-wcpb-graph-metadata-auto341/wcpb-auto/summary.json)

同一服务内固定调度顺序、物理形状和 graph staging，在 wrapper 内切换图回放与
逐算子执行。串行 7 条请求每条 128 token 均逐 token 相同，回放臂的 4,217 个
物理步实际调用 3,937 次 FULL 和 280 次 PIECEWISE，逐算子臂的实际回放为 0。
[同形状输出对照](../artifacts/w3-scheduler-priority-20261127/gpu4/w3-wcpb-c1-staged-replay-pair/summary.json)

接连复测 W3 时观察到一次 CODA 没有为应出 token 的请求返回采样结果。检查到
MRv2 接纳新请求和 RLT 恢复 chunk 位置共用 `num_computed_tokens` 的 UVA 写池。
同一物理批可以轮转两次写槽，GPU 完成事件按每物理批一次退休。逐批位置更新现用
独立的 MRv2 `UvaBufferPool` 暂存请求槽和位置，通过 `index_copy_` 更新同一个
设备位置张量。原生接纳的写池留给 MRv2，两个写入路径各自最多每批轮转一次。
[连续压测的异常日志](../artifacts/w3-scheduler-priority-20261127/gpu3/w3-wcpb-graph-metadata-clean/wcpb512/server.log)

最终版在 GPU 1 的 512 结构波执行 3,358 个物理 forward，2,417 FULL、941
PIECEWISE、0 eager。32 条请求逐位置通过，135 次提前启动下一 chunk。与修复前
512 结构波相比，首 token 32/32、完整输出 31/32 一致。
同卡默认 341 结构波直接记录了全部 3,459 次图回放，2,261 FULL、1,198 PIECEWISE、
0 eager，其中 2,075 个批发生 padding。32 条请求逐位置通过，提前启动下一 chunk
203 次。与旧默认 341 单波相比，首 token 32/32、完整输出 29/32 一致。
[最终 512 结构波](../artifacts/w3-scheduler-priority-20261127/gpu1/w3-wcpb-graph-positions-structure/wcpb512/summary.json) ·
[最终自动 341 结构波](../artifacts/w3-scheduler-priority-20261127/gpu1/w3-wcpb-graph-positions-auto341/wcpb-auto/summary.json)

最终版 512 在 GPU 3 连续六个普通计时波全部返回 4,096 个 token，无漏采样。
六波吞吐为 572.03 / 563.42 / 583.87 / 562.34 / 548.50 / 546.35 tok/s，
中位数 562.88 tok/s，TTFT P95 1,976.87 ms，TPOT P95 30.55 ms，ITL P95 42.78 ms。
对照修复前 GPU 7 的 WCPB512 第 2 波，首 token 192/192、完整序列 184/192 一致。
两个配置的调度轨迹随请求到达顺序变化，GPU 3 六波和此前 GPU 7 基线均为测试进程
独占，SM 2,032 MHz，KV 容量 291,018 token。
[最终 512 六波吞吐](../artifacts/w3-scheduler-priority-20261127/gpu3/w3-wcpb-graph-positions/wcpb512/summary.json)

位置写隔离后插件测试 106 通过、2 跳过。

#### WCPB 开关的同卡时间分解

最终源码在 GPU 1 分别跑普通 512 和 WCPB 512 的独立结构波。两波使用同一输入、
动态 decode 深度、1,024-token 预算、80 GiB KV 与图档，SM 2,032 MHz，测试进程独占。
两波都无 eager 回退，CORE 实算的 prefill 393,216 行和 decode 9,144 行完全相同。

| GPU event 区间 | 普通 512（ms） | WCPB 512（ms） | 增量（ms） |
| --- | ---: | ---: | ---: |
| 纯 decode CORE | 909.70 | 2,165.31 | 1,255.61 |
| 含 prefill 的 CORE | 2,227.19 | 2,931.21 | 704.02 |
| PRELUDE 和 CODA 的模型区间 | 1,096.88 | 1,601.18 | 504.30 |
| 采样 | 70.80 | 180.85 | 110.05 |
| 模型和采样事件包络之间 | 311.28 | 1,371.87 | 1,060.58 |
| 整波设备流跨度 | 4,615.86 | 8,250.42 | 3,634.56 |

WCPB 使物理批由 1,701 增至 3,358。纯 decode CORE 由 618 增至 1,508 批，
每批请求数中位数 11→4，纯 decode CODA 由 207 增至 651 批。普通路径启用了
同轮 CORE 终轮/续轮合并、decode CODA 合批和输入/metadata 复用。当前 WCPB
按不同轮次及终轮/续轮分别执行，关闭普通 decode 收尾策略，并独立准备每批输入。
含 prefill 的 CORE 由 402 增至 599 批，padding 后由 403,200 增至 441,216 行。
这两项 CORE 增量约占整波差距的 54%。

WCPB 确实提前启动了 135 次后续 chunk。它新增的是可调度工作，设备仍在同一条流
执行这些物理批。W3 中这部分提前量不足以抵消组批碎化和每批准备开销。模型和采样
事件间有输入与 metadata 准备等未单列阶段，表中的时间差不能全部归为 GPU 空闲。
Host 调度器区间累计 343.61→442.69 ms，单次中位数均约 80 µs。Host 提交区间
合计 3,755.56→6,923.52 ms。每次提交扣去模型入口内的 Host 区间后，累计
897.63→2,540.04 ms，均值 0.53→0.76 ms/物理批。这部分包含输入和 attention
metadata 准备、位置更新及其他提交工作，当前对照没有单独测出输入复用的贡献。
Host 和设备区间有重叠，不能相加。
[同卡分解数据](../artifacts/w3-scheduler-priority-20261127/gpu1/w3-wcpb-vs-ordinary512-structure.json) ·
[普通 512 结构波](../artifacts/w3-scheduler-priority-20261127/gpu1/w3-ordinary512-positions-structure/ordinary512/summary.json)

#### WCPB 复用输入与 attention metadata

Worker 比较相邻物理批的 MRv2 请求顺序、槽位、chunk 起点、真实长度、prefill 标记
和图 descriptor。完全一致时复用 InputBatch、块表、slot mapping 和 attention
metadata。同一请求切换 chunk、追加 KV 块、请求加入或退出以及 CODA 采样后的更新
都会使旧输入失效。首次准备继续走原生 MRv2，模型中间状态仍按请求和 chunk 保存。
这次保留 WCPB 原来的调度、终轮/续轮分组和逐批位置恢复。

CPU 用例核对相同布局只准备一次，同一槽和长度换 chunk 时重新准备。插件套件
107 通过、2 跳过。

GPU 4 的 W3 C16 结构波使用 512 chunk、1,024-token 预算和 80 GiB KV。
3,409 个物理批中，619 个同时命中 InputBatch、块表/slot mapping 与 attention
metadata，命中率 18.16%。命中包括 540 次 CODA 和 79 次 CORE，MRv2 实际
构造 metadata 2,790 次。2,454 批实际回放 FULL，955 批实际回放 PIECEWISE，
无 eager。32 条请求逐位置通过，仍有 137 次提前启动后续 chunk。与改动前的独立
结构波相比，32/32 首 token、31/32 完整输出一致。
[真实复用计数](../artifacts/w3-scheduler-priority-20261127/gpu4/w3-wcpb-reuse-structure/wcpb512/wavefront-cuda.reuse.json) ·
[图回放和逐位置校验](../artifacts/w3-scheduler-priority-20261127/gpu4/w3-wcpb-reuse-structure/wcpb512/summary.json)

GPU 7 使用同一热服务交替关闭和开启复用，各测三个正式波，GPU 独占且均为
2,032 MHz。保持 WCPB 512、输入、预算与 CUDA Graph 一致。关闭/开启的吞吐
中位数为 659.91/665.55 tok/s，三对增幅 +0.87%/+0.89%/+0.74%。TTFT P95
为 1,772.83/1,772.17 ms，TPOT P95 为 26.27/26.14 ms。96 条请求的完整 token
序列逐条一致。该实验的增量以同服务 A/B 为准。
[输入复用交替六波](../artifacts/w3-scheduler-priority-20261127/gpu7/w3-wcpb-reuse-interleaved/summary.json)

逐计划核对同卡 WCPB 开关的结构波，增加的 1,657 次物理执行来自 586 个新增
调度计划和 1,071 个计划内拆出的 CORE 批。586 个新增计划中，CODA 占 496、
PRELUDE 占 74、CORE 占 16。CORE 拆出的 1,071 批中，654 次来自同一轮的终轮与
续轮分组，417 次来自同一计划含不同轮次。输入复用开启后的独立结构波仍有 1,111 次
计划内 CORE 拆批，拆批计数和每个物理批的执行耗时需要分别比较。

例如，开启复用的计划 2,937 有 16 条 decode 请求处于第 1 轮。其中 12 条继续下一轮，
4 条做 CODA。Worker 分别走补齐为 16 和 4 行的两个 FULL 图。普通路径已有能力
对同轮的两种去向合批，WCPB 当前的 CORE 分组条件把该路径关闭。WCPB 的调度器
也关闭了普通 decode 在有限等待内聚合就绪 CODA 的策略，导致更多小批进入采样。

#### WCPB 适配现有调度

WCPB 使用普通路径已有的首 token CODA 优先、decode 收尾等待上限、混合 CORE
按轮次对齐和纯 decode 图档合批判断。WCPB 的预备工作仍由 `_ready_spans()` 按
前段已完成轮次决定，CODA 保留同一请求的位置顺序。Worker 允许同一轮的终轮和
续轮在同一个物理批执行后分流，每个轮次逐请求核对 MRv2 槽位唯一。同一请求的
不同 chunk 仍可在不同轮次分别执行。现有输入复用和位置缓冲隔离继续生效。

CPU 回归覆盖同步和异步下的 CODA 聚合、首 token 优先、重复跳过后的公平性、
首段之后的 WCPB 交错、同轮 P/D 一起推进、短等后合用同一 FULL 图，以及同一请求
两个 chunk 的 Worker 分组。插件套件 127 通过、2 跳过。

GPU 4 的 W3 C16 两次独立结构波保留 512 chunk、1,024-token 预算、80 GiB KV 与
不同请求的 1/2/3 轮 decode，两个进程各自独占，SM 都是 2,032 MHz。

| 结构指标 | 适配前 | 适配后 |
| --- | ---: | ---: |
| 调度计划 | 2,298 | 1,671 |
| 实际物理批 | 3,409 | 1,671 |
| 纯 decode CORE 批 / 每批请求数中位 | 1,543 / 4 | 597 / 11 |
| 纯 decode CODA 批 / 每批请求数中位 | 652 / 3 | 200 / 16 |
| 采样调用 | 840 | 335 |
| FULL / PIECEWISE / eager | 2,454 / 955 / 0 | 996 / 675 / 0 |
| 先于前段 CODA 启动后段 PRELUDE | 137 | 30 |
| 设备流跨度（ms） | 6,269.36 | 4,229.08 |

两个结构波都执行 393,216 个实际 prefill CORE 行和 9,144 个实际 decode CORE
行，32/32 条逐位置通过。适配后有 388 个纯 decode CORE 与 251 个 P/D CORE
物理批同时有终轮和续轮，CORE 计划全部在同一轮。对照首 token 32/32、完整输出
30/32 一致。提前启动次数随调度变化，适配后保留了跨 chunk 的提前推进。
[适配前结构](../artifacts/w3-scheduler-priority-20261127/gpu4/w3-wcpb-reuse-structure/wcpb512/summary.json) ·
[适配后结构](../artifacts/w3-scheduler-priority-20261127/gpu4/w3-wcpb-current-scheduling-structure/wcpb512/summary.json)

GPU 7 同一热服务中交替切换是否适配普通调度及同轮合批，各测三波。两组均开启
WCPB 512 和输入复用，维持 1,024-token 预算、80 GiB KV、原有图档与动态轮数。
六波均独占 GPU，SM 2,032 MHz，均完成每波 4,096 输出 token。

| 正式指标 | 适配前 | 适配后 |
| --- | ---: | ---: |
| 吞吐中位数（tok/s） | 663.74 | 975.65 |
| TTFT P95（ms） | 1,774.04 | 1,305.14 |
| TPOT P95（ms） | 26.20 | 13.78 |
| ITL P95（ms） | 35.78 | 22.63 |

吞吐提高 46.98%，三对的相对增益为 47.14%/46.98%/46.96%。两组首 token
96/96、完整输出 90/96 一致。前后唯一切换的功能是普通调度策略和同轮合批，
相对关闭 WCPB 的收益另外按同卡普通 512 路径计时比较。
[同热六波 A/B](../artifacts/w3-scheduler-priority-20261127/gpu7/w3-wcpb-current-scheduling-interleaved/summary.json)

同源码、同 GPU 7、相同 512 chunk 上限关闭 WCPB 后，三波吞吐为
992.02/992.03/992.05 tok/s，TTFT P50/P95 为 371.71/1,278.94 ms。适配后 WCPB
的吞吐低 1.65%，按每波 4,096 个输出 token 折合慢 69.34 ms；TTFT P50 低
32.13%，P95 高 2.05%。两配置的首 token 96/96、完整序列 87/96 一致。
[同卡同源码普通 512](../artifacts/w3-scheduler-priority-20261127/gpu7/w3-wcpb-current-scheduling-ordinary512/ordinary512/summary.json)

默认 341 的 GPU 4 结构波实际回放 985 次 FULL、732 次 PIECEWISE，0 eager；
339 个物理批同时命中输入、块表和 metadata 复用，45 次提前启动下一 chunk，
32/32 条逐位置通过。该波只核对执行结构。
[适配后默认 341 结构](../artifacts/w3-scheduler-priority-20261127/gpu4/w3-wcpb-current-scheduling-auto341/wcpb-auto/summary.json)

#### WCPB 跨轮物理合批

对照主线 `4821d47` 的 WCPB 语义，默认 chunk 改为 128，显式 chunk 上限可用于
普通 split prefill 和 WCPB。每条请求在有效窗口内可以一次准备多段相邻 chunk；
MRv2 用一个连续 PRELUDE query 处理相邻段，中间状态按真实 chunk 位置保存。每段
CORE 只推进一轮，前段已完成的对应轮次限制后段可就绪的轮次。已有 decode CODA
收尾、纯 decode 图档判断继续应用。

同一次 CORE 物理执行内，MRv2 的基础请求槽仍由请求拥有，batch 输入按实际
`(请求, chunk 起点, 轮次)` 构造。同一请求的多个区间可复用底层请求槽并拥有
不同的 positions、query offsets 和序列长度。MRv2 的 block table 功能提供原始
KV 映射，Inkling 按每组的轮次使用对应的 attention 和短卷积 KV。所有组共享本层
QKV、输出投影与 MLP。PIECEWISE 启动捕捉按补齐后的总行数建图，运行时按本批
的轮次和位置读取 KV 并分流中间状态。

GPU 4 的 C1 定向样本在 2,048 token prompt、512 chunk 下出现 4 次跨轮回放；
其中一批同时包括同一请求 `[512,1024)` 的第 1 轮与 `[0,512)` 的第 3 轮，
12 个 CORE 层各有一次 1,024 行 QKV 和 MLP。逐轮核对设备位置及 KV 层名，
8 个输出 token 在预热和诊断请求间完全一致。

GPU 7 的 W3 C16 512 结构波共 1,861 个物理批。其中 1,055 FULL、418 常规
PIECEWISE、388 跨轮 PIECEWISE，全部实回放。388 次跨轮 CORE 中，134 次把同一
请求的不同 chunk 放进同一物理批；32 条请求逐位置通过。诊断与同源码预热相比，首
token 32/32、完整输出 28/32 一致。正式吞吐采用关闭诊断探针的波次。
[首版跨轮 C1](../artifacts/w3-scheduler-priority-20261127/gpu4/wcpb-cross-depth-graphs-direct/structural-summary.json) ·
[首版跨轮 W3](../artifacts/w3-scheduler-priority-20261127/gpu7/wcpb-cross-depth-graphs-w3/structural-summary.json)

最终版 PRELUDE 先给各就绪请求准备一段，再把剩余预算用于同一批请求的下一段。
相邻 CORE 层合并上一层结束与下一层输入的图回调，每个跨轮图由 49 个记录段减至
38 个。GPU 3 W3 C16 512 结构波全部 1,789 次物理执行实际回放，其中 1,004 FULL、
430 常规 PIECEWISE、355 跨轮 PIECEWISE。62 个物理 CORE 合入同一请求的不同
chunk/轮次，155 次下一段的 PRELUDE 在前段 CODA 之前在设备上开始，32/32 条
逐位置通过，诊断与热身的完整输出 32/32 一致。355 批共享 4,260 次层级 QKV/MLP，
实际分组调用 attention 11,400 次。
[最终版跨轮执行和 replay 证据](../artifacts/w3-scheduler-priority-20261127/gpu3/wcpb-cross-depth-round-robin-w3/structural-summary.json)

正式 W3 C16 每臂预热后三波，每波 32 条 4,096→128，仍是动态 decode 轮数、
1,024-token 预算和 80 GiB KV。全部样本独占，SM 2,032 MHz。普通路径取同卡的
早一版跨轮实验数据，最终提交的改动仅涉及 WCPB 和跨轮 CORE。

| W3 C16 配置 | 卡 | 吞吐中位（tok/s） | TTFT P50 / P95（ms） |
| --- | ---: | ---: | ---: |
| 普通，原 1,024 预算 | 4 | 1,011.34 | 222.06 / 1,238.09 |
| 普通，512 chunk | 4 | 989.93 | 371.72 / 1,279.48 |
| WCPB，512 chunk | 4 | 545.91 | 587.14 / 2,770.57 |
| 普通，128 chunk | 7 | 938.49 | 739.61 / 1,413.32 |
| WCPB，默认 128 chunk | 7 | 453.36 | 1,856.94 / 3,434.96 |

同 chunk 的两对开关各有 96/96 首 token 一致，完整输出分别 90/96 和 89/96。
两种 WCPB 配置的吞吐分别是同卡对应普通路径的 55.15% 和 48.31%。共享权重的
计算已经合批，attention 和短卷积仍调用每组的原生执行路径。当前使用固定 chunk
上限和按请求位置保存的有界中间状态，原版的自适应 sizing 和预分配 hidden ring
尚未接入。
[WCPB 512 正式三波](../artifacts/w3-scheduler-priority-20261127/gpu4/wcpb-cross-depth-round-robin-formal512/wcpb512/summary.json) ·
[WCPB 128 正式三波](../artifacts/w3-scheduler-priority-20261127/gpu7/wcpb-cross-depth-round-robin-formal128/wcpb-auto/summary.json) ·
[普通 512/原 1,024](../artifacts/w3-scheduler-priority-20261127/gpu4/wcpb-cross-depth-graphs-formal512/comparison.json) ·
[普通 128](../artifacts/w3-scheduler-priority-20261127/gpu7/wcpb-cross-depth-graphs-formal128/ordinary128/summary.json)

插件套件 129 通过、2 跳过，包含重复请求槽的独立位置、原生描述符、窗口内 PRELUDE
轮转和逐轮依赖。

#### 跨轮 WCPB 的 W3 性能归因

对照 512 chunk、1,024-token 预算下的普通 GPU 1 结构波和最终 WCPB GPU 3
结构波。两边的实际 prefill CORE 均为 393,216 行，decode CORE 均为 9,144 行。
含 prefill 的 CORE 物理批为 402→413，总真实行均值 985.19→959.46，补齐后的总行
403,200→415,488。普通路径已经能填满大部分 1,024 档，这个 W3 没有通过跨轮进一步
减少 CORE 提交。

在最终 WCPB 的同一个 GPU 3 结构波内，取补齐宽度 1,024、至少 900 真实行的
含 prefill CORE。40 个常规批平均 1,021 真实行，模型区间中位数 5.818 ms，13 次图
段回放。318 个跨轮批平均 1,018.99 真实行，中位数 12.702 ms，38 次图段回放。
355 个跨轮批的 12 层合计 4,260 次共享 QKV/MLP，并调用了 11,400 次分组
attention。按深度切片、短卷积、混合与中间张量拷贝仍在 graph break 的回调内，
图段的计数和模型区间无法单独给出每一种 kernel 的耗时。

同一组宽度 1,024 的样本继续细分。P/D 批的常规路径为 28 批，平均 1,020.93
行，模型区间中位数 6.058 ms。跨轮 2 组为 88 批，平均 1,017.06 行、中位
12.174 ms；跨轮 3 组为 194 批，平均 1,020.58 行、中位 13.359 ms。纯 prefill
的常规路径和跨轮 2 组分别为 5.763 与 8.584 ms，共 12 与 30 批，平均真实行
1,021.17 与 1,022.70。跨轮分组键包含轮次及终轮/续轮，不能从组数推算实际轮次数。
P/D 三行的 Host 模型区间中位数分别为 3.193、12.119 和 13.344 ms。
Host 和设备事件区间并行，不能用两者的差或和分摊 kernel 耗时。

跨卡独立结构波的含 prefill CORE 为 2,227.19→4,886.42 ms，占设备窗口增量
3,149.04 ms 的约 84%。纯 decode CORE 反而为 909.70→885.22 ms。结构波用于
分段归因，正式吞吐比较使用前面的同卡独占三波。
[普通 512 结构波](../artifacts/w3-scheduler-priority-20261127/gpu1/w3-ordinary512-positions-structure/ordinary512/summary.json) ·
[最终 WCPB 同波内的两类 CORE](../artifacts/w3-scheduler-priority-20261127/gpu3/wcpb-cross-depth-round-robin-w3/structural-summary.json)

#### WCPB 低并发对照

W3 保持 4,096→128、1,024-token 调度预算、80 GiB KV 和原图档。每波 8 条请求，
显式 1/2/3 轮和默认各 2 条。C1、C4 各比较普通原配置、普通 128 chunk 和默认
WCPB 128，在同一张卡运行该并发的各臂。C1 额外匹配 512 chunk。各臂完整预热后
测三波，正式波禁用结构诊断探针。

| 负载 | GPU | 配置 | 吞吐（tok/s） | TTFT P50 / P95（ms） | TPOT P50（ms） |
| --- | ---: | --- | ---: | ---: | ---: |
| W3 C1 | 7 | 普通原配置 | 279.73 | 79.57 / 81.12 | 3.26 |
| W3 C1 | 7 | 普通 128 | 174.23 | 341.44 / 358.69 | 3.28 |
| W3 C1 | 7 | WCPB 128 | 131.92 | 568.55 / 626.29 | 3.31 |
| W3 C4 | 4 | 普通原配置 | 547.65 | 209.97 / 329.79 | 5.73 |
| W3 C4 | 4 | 普通 128 | 455.16 | 492.58 / 498.68 | 4.92 |
| W3 C4 | 4 | WCPB 128 | 300.24 | 989.79 / 993.18 | 5.94 |
| W3 C1 | 3 | 普通 512 | 249.37 | 118.20 / 122.24 | 3.33 |
| W3 C1 | 3 | WCPB 512 | 244.84 | 136.45 / 137.83 | 3.32 |

24 个正式波均为测试进程独占，SM 2,032 MHz，每波 1,024 个输出 token。三项同
chunk 对照的吞吐变化为 C1/128 −24.28%、C4/128 −34.04%、C1/512 −1.82%，
跨开关的完整输出分别 24/24 一致。C1 的 TPOT P50 基本一致，差距主要在 TTFT。
[C1 默认 128](../artifacts/w3-scheduler-priority-20261127/gpu7/wcpb-low-concurrency-c1/comparison.json) ·
[C4 默认 128](../artifacts/w3-scheduler-priority-20261127/gpu4/wcpb-low-concurrency-c4/comparison.json) ·
[C1 512](../artifacts/w3-scheduler-priority-20261127/gpu3/wcpb-low-concurrency-c1-chunk512/comparison.json)

#### WCPB 低并发的执行流与首 token 时间

按原版 wavefront 文档及 `main` 的 Inkling 路径，调度器把不同 chunk、不同轮次的
就绪区间放进一批，每个区间只推进一轮。Worker 把每个区间要读取的 KV 块和新 KV
的写入位置一次传给模型，attention 和短卷积在各层消费整批输入。中间 hidden
state 写回请求的预分配环形缓冲，后续轮次按行读取。原版在终轮后由 Worker 补做
中间区间所需的 CODA KV 写入；最后一个 prompt token 进入生成用的 CODA。
[原版设计](https://github.com/InferMatrixLab/vllm-rlt/blob/4821d47cec92aa1d7d2ec4be5f589d19b2794ae3/docs/split-prefill-wavefront.md) ·
[原版 KV 块表和写入位置](https://github.com/InferMatrixLab/vllm-rlt/blob/4821d47cec92aa1d7d2ec4be5f589d19b2794ae3/vllm_rlt/worker/metadata.py#L233-L314) ·
[原版 Inkling 层循环](https://github.com/InferMatrixLab/vllm-rlt/blob/4821d47cec92aa1d7d2ec4be5f589d19b2794ae3/vllm_rlt/models/inkling2mini.py#L360-L439)

当前插件复现了调度器跨轮组批和每段推进一轮的行为。Worker 按轮次、是否终轮分别
构造 vLLM attention 元数据；共享 QKV、输出投影和 MLP 的部分在整批执行，attention
和短卷积每层仍按组调用。中间 hidden state 从各区间取出后拼接，prefill CODA 仍由
调度器提交。由此，同一个调度批在当前 Worker 内仍有多组层内执行。
[当前 attention 输入](../vllm_rlt/models/inkling_plugin.py#L208) ·
[当前层内分组](../vllm_rlt/models/inkling_mixed_core.py#L306) ·
[当前中间状态拼接](../vllm_rlt/models/inkling_mixed_core.py#L423)

128 chunk 的额外诊断逐并发在同一张卡上运行，两臂都处理 98,304 行真实 prefill
CORE，首 token 和完整输出分别 8/8 一致。C1 的物理 CORE 提交从 768 降到 360，
实际宽度从全部 128 行变为 32 批 128 行、248 批 256 行和 80 批 384 行，384 行的
批补到图档 512。C1 全波没有 1,024 行的 CORE 批。C4 的含 prefill CORE 提交从
213 降到 138，含同批 decode 的平均真实行从 462 升到 713，WCPB 有 64 批达到
1,024 行；含 prefill CORE 的设备事件区间仍从 804.19 升至 1,387.49 ms。

| C1 每条请求从首个设备提交到首采样结束的平均时间 | 普通 128（ms） | WCPB 128（ms） |
| --- | ---: | ---: |
| PRELUDE 模型区间 | 45.35 | 37.44 |
| CORE 模型区间 | 248.22 | 419.15 |
| CODA 模型区间 | 40.20 | 44.17 |
| 采样 | 6.58 | 6.11 |
| 提交内部除模型和采样之外 | 26.59 | 68.38 |
| 提交和采样事件之间 | 13.42 | 9.91 |
| 设备事件总窗口 | 380.35 | 585.17 |

C1 首 token 设备窗口增加 204.81 ms，其中 CORE 增加 170.93 ms，提交内部除模型和
采样之外增加 41.79 ms。328 批跨轮 CORE 都有图段回放。其中无需补齐的 248 批
256 行 CORE 平均用 9.00 ms，普通路径处理同等行数的两批 128 行 CORE 平均共用
5.17 ms。设备事件区间可以包含 Host 提交等待，现有探针无法进一步拆出 attention、
短卷积和拼接各自的耗时。正式吞吐和 TTFT 使用前面的无探针结果。
[C1 时间分解](../artifacts/w3-scheduler-priority-20261127/gpu7/wcpb-low-c1-timeline128/timeline-summary.json) ·
[C1 Host 和设备时间轴](../artifacts/w3-scheduler-priority-20261127/gpu7/wcpb-low-c1-timeline128/first-token-timeline.html) ·
[C4 时间分解](../artifacts/w3-scheduler-priority-20261127/gpu4/wcpb-low-c4-timeline128/timeline-summary.json)

#### WCPB 执行方案评审

目标是把已就绪的不同 chunk、不同轮次带进同一个 CORE 批。每个 chunk 在该批只推进
一轮。Inkling 每层的 attention 和短卷积分别处理整批，CORE 最后一层再把完成终轮
的行交给 CODA、把续轮的行留给后续 CORE。PRELUDE、CORE 和 CODA 各自有独立的
调度和执行入口。方案依据原版设计和下文对应的实现。
[原版 wavefront 文档](https://github.com/InferMatrixLab/vllm-rlt/blob/main/docs/split-prefill-wavefront.md)

**调度和就绪条件**。未命中的 prompt 从缓存前缀末尾切 chunk。PRELUDE 在请求的在途
窗口内准备 chunk，先给每个可运行请求机会，再用剩余预算准备同请求的后续 chunk。
一个 chunk 可以进入第 `d` 轮时，它自己的上一轮输出和本轮的因果 KV 前缀都已就绪。
第一轮以 PRELUDE 的结果为输入。同一条请求的三段 A、B、C，在各轮都能放进预算时，
CORE 可依次发 `A1`、`A2+B1`、`A3+B2+C1`、`B3+C2`、`C3`。每次提交后重新选批。
在 MRv2 里，一个真实请求可以对应多个批内输入行。分别检查真实请求数、批内输入行数
和 token 数，当前 `max_num_seqs` 对批内行容量的限制应显式计入。保持已有的等待公平性、
首 token 优先和 decode 等待上限。先验证固定 chunk。原版可选的自适应模式随后按请求
确定 chunk 大小，在各请求、各轮次获得第一份预算后，再用相邻已就绪的区间补余量。
[原版选批](https://github.com/InferMatrixLab/vllm-rlt/blob/4821d47cec92aa1d7d2ec4be5f589d19b2794ae3/vllm_rlt/core/scheduler.py#L972-L1090)

**批内输入和 KV 地址**。Worker 用 MRv2 的真实请求槽和每段绝对位置准备输入。同一
请求可以在不同轮次各有一段。例中 `A2` 有 128 行、`B1` 有 64 行，传给 attention 的
查询边界是 `query_start_loc=[0,128,192]`。每段另有因果末端 `seq_lens`、对应轮次的
`block_table`，以及每行新 KV 的 `slot_mapping`。同请求同轮的相邻区间在因果依赖及
地址约束满足时可合成一段。短卷积另用各自的块表、每行所属区间和区间起点。批描述
每次 CORE 准备一次，各层复用序列边界和位置；全局 attention、滑窗 attention、
短卷积各自使用对应的块表。保留 vLLM 的 KV 分配、滑窗裁剪、MRv2 块表和 slot mapping
生成。在此基础上按 `(请求槽, 轮次, 绝对位置)` 取批内每段的地址。
[原版地址生成](https://github.com/InferMatrixLab/vllm-rlt/blob/4821d47cec92aa1d7d2ec4be5f589d19b2794ae3/vllm_rlt/worker/metadata.py#L233-L314) ·
[MRv2 块表](https://github.com/vllm-project/vllm/blob/3146a9df41403b8fc30234259444d205974982dd/vllm/v1/worker/gpu/block_table.py#L173-L277)

**需要验证的 vLLM 接口**。Inkling 现有 `Inkling2Attention._attention` 用一个
`pass_idx` 为整批选 KV 张量，`conv_mix` 同样用一个 `owner`。变长 FA4 接口已有查询
边界、每段上下文长度和块表，但它接收一对 K/V 张量。先核对各轮已绑定 KV 和卷积缓存
的底层存储、偏移和 page stride，验证能否构造统一分页视图，并把原块号映射到视图中
的正确页。能映射时可直接复用现有 attention 和卷积内核。无法用同一视图表达的布局
需要扩展对应内核的地址选择，使用每段的页地址。两个方向均复用原缓存、每段的因果
和滑窗语义，避免在每层复制历史 KV。首轮与后续轮的模型分支用逐行信息表达，终轮与
续轮的去向只在 CORE 末端分流。开启 WCPB 时，有 prefill 的跨轮 CORE 应进入相应的
packed prefill 路径。现有 `rlt_packed_prefill` 的名称不足以证明这条路径已经生效。
[vLLM Inkling attention](https://github.com/vllm-project/vllm/blob/3146a9df41403b8fc30234259444d205974982dd/vllm/models/inkling2/nvidia/attention.py#L410-L510) ·
[vLLM Inkling 短卷积](https://github.com/vllm-project/vllm/blob/3146a9df41403b8fc30234259444d205974982dd/vllm/models/inkling2/nvidia/sconv.py#L151-L201) ·
[原版 Inkling CORE](https://github.com/InferMatrixLab/vllm-rlt/blob/4821d47cec92aa1d7d2ec4be5f589d19b2794ae3/vllm_rlt/models/inkling2mini.py#L353-L493)

**状态、CODA 和异步回收**。Scheduler 维护 PRELUDE 已完成位置、每轮已完成位置、
可发布的连续 prompt 前缀。Worker 使用预分配的 hidden state 环形缓冲，按请求槽、
chunk 槽和 chunk 内偏移读写。区间完成最后一轮时，Inkling 中间 prompt token 仍需
CODA 写入边界 KV，只有 prompt 末 token 做产生首 token 的采样。原版在 Worker 内补做
前一种 CODA。插件先通过现有 CODA 入口验证等价的写入、采样和完成语义，再比较中间
CODA 合批的提交成本。发布前缀需保证相关 KV 写入对读者可见。hidden state 槽等 GPU
最后一次读取完成后再回收。继续使用 MRv2/UVA 的缓冲及 stream/event 生命周期。
取消或抢占时以已完成前缀为恢复点，未完成 chunk 重算。保持当前 Inkling 请求级轮数只作用于 decode 的
语义。第一阶段验证当前单卡 Inkling 和每轮独立 KV，组合 PD、投机及并行配置分别
通过兼容性测试后再作为支持范围。
[原版进度与回收](https://github.com/InferMatrixLab/vllm-rlt/blob/4821d47cec92aa1d7d2ec4be5f589d19b2794ae3/vllm_rlt/core/scheduler.py#L1227-L1311) ·
[原版 CODA follow-up](https://github.com/InferMatrixLab/vllm-rlt/blob/4821d47cec92aa1d7d2ec4be5f589d19b2794ae3/vllm_rlt/worker/model_runner.py#L680-L704)

**Mixed batch 和 CUDA Graph**。启用 mixed batch 时，同一 op 可以有 prefill 和
decode。原生 Inkling 可在一次层调用内按实际查询类型运行 prefill 和 decode 的计算
路径。各路径处理自己的完整子批，轮次只影响 KV 地址。纯 decode 保留 FULL 执行；
含 prefill 的 CORE 复用 PIECEWISE warmup、真实 token 数和 padding 档位。编译区的
图按稳定形状捕获，实际的区间长度、轮次、块地址和终轮标记作为运行时输入。padding
不参与 KV、卷积状态写入或采样。验收同时检查图命中和每层实际调用次数。

**验收**。先比较同请求跨 chunk、跨请求、末段不足 chunk、prefix hit、抢占恢复、
异步及 mixed 开关下的结果。核对每段的 KV 写入位置、因果依赖、采样和图回放。再做
同设备同预算同 chunk 的普通 prefill 与 WCPB A/B，并保留普通默认配置为参照。用 W3
C1/C4 验证提交减少后 CORE 时间也下降，用 C16 检查已接近满批时的额外成本，再用 W7
检验不规则输入。正式跑吞吐、TTFT、TPOT 与 ITL，诊断同时记录实际行数、补齐行数、
各层 attention/卷积的调用次数和 CORE、CODA 区间。固定三轮对 vLLM 同配置，动态
轮数对 RLT 同请求轮数列表的开关两组。固定三轮的基础性能继续使用吞吐和 TTFT P95
的 95% 门槛，目标约 97%。WCPB 收益由同 chunk 的正式开关 A/B 验证。

#### 跨轮 packed prefill 接入

锁定的 vLLM 在 Inkling2-mini 实机上使用 LBNHC。固定 80 GiB 的 40,960 个 KV
物理块中，CORE 的 12 层各轮缓存共用一个底层分配，轮次间的地址差都是整页。
attention 页步长为 262,144 字节，各层的页偏移与卷积缓存使用的 KV group 分别解析。
统一分页视图只改变寻址，原始块的分配、回收、前缀和滑窗仍由 vLLM 管理。同一组
块表和写入映射在一次 CORE 内复用。实机页偏移下的合成数据验证覆盖了原生 FA4 的
全局 attention、滑窗 attention 及跨页短卷积，输出与分轮调用逐元素一致。
[实机缓存布局](../artifacts/w3-scheduler-priority-20261127/gpu7/wcpb-native-kv-layout/kv-layout.json) ·
[原生算子对照](../artifacts/w3-scheduler-priority-20261127/gpu7/wcpb-native-kv-layout/native-paged-kernel-smoke.json)

跨轮含 prefill 的 CORE 复用 MRv2 变长元数据及 PIECEWISE 图，整体提交 attention、
两处短卷积和 MLP。首次和后续轮的归一化及消息混合仍按各自行计算。终轮在 CORE
末尾路由给 CODA，续轮的消息状态写回环形缓冲。环形缓冲在启动的内存评估期间分配，
预留空间计入自动 KV 容量评估。Worker 在 CODA 入队最后一次读取后归还槽，后续写入
沿同一条计算流排队。CODA 拆分时保留剩余区间，结束与抢占清理请求的状态。当前以
TP=1、各轮缓存为同一底层分配且页对齐的 Inkling 为 packed 路径，其他布局沿用分组。

GPU 诊断用 2,048 和 2,051 token 的单请求各生成 8 个 token。两组 WCPB 开关的输出
逐 token 一致，核对了输入位置。开关开启时分别有 20 和 21 个跨轮 CORE，均实际回放
PIECEWISE。共计 240/252 次层内 attention、240/252 次 FA4、480/504 次短卷积调用，
每层每批分别为 1、1、2 次。环形缓冲峰值为同请求 3 个 chunk，生成后均归零。
[开启结构诊断](../artifacts/w3-scheduler-priority-20261127/gpu7/wcpb-packed-ring-live-on/summary.json) ·
[关闭结构诊断](../artifacts/w3-scheduler-priority-20261127/gpu4/wcpb-packed-ring-live-off/summary.json)

原预算 `max_num_batched_tokens=1024`、chunk 128、80 GiB KV 下，W3 C1 各组完成同样
98,304 行实际 prefill CORE。下表为 GPU 7 的 CUDA event 诊断数据，三组均为独占结构
波，正式吞吐另外测量。

| 指标 | 普通 prefill | 旧 WCPB | 新 packed WCPB |
| --- | ---: | ---: | ---: |
| 含 prefill 的 CORE 提交次数 | 768 | 360 | 360 |
| 每请求首采样前 CORE 时间（ms） | 248.22 | 419.15 | 290.77 |
| 每请求提交内除模型和采样之外（ms） | 26.59 | 68.38 | 62.54 |
| 每请求提交与采样区间之间（ms） | 13.42 | 9.91 | 10.26 |
| 每请求首个设备提交到首采样结束（ms） | 380.35 | 585.17 | 451.55 |

完全无需补齐的 256 行跨轮 CORE，模型事件平均从 9.003 ms 降到 6.665 ms。384 行
补到 512 的批从 12.992 ms 降到 7.390 ms。相比普通关闭组，新版的首采样设备窗口仍
多 71.20 ms，主要来自 CORE 的 42.55 ms 和提交内模型外的 35.95 ms。两边都使用
异步调度，提交之间的区间减少了 3.16 ms。设备事件区间可能包含 GPU 等待 Host
继续提交。C4 在 GPU 4 观测到 125 个 packed CORE 批，其中
35 批确实同时带有 prefill 和 decode，全部走 PIECEWISE。
[普通与旧 WCPB 时间轴](../artifacts/w3-scheduler-priority-20261127/gpu7/wcpb-low-c1-timeline128/timeline-summary.json) ·
[新版 C1 时间轴](../artifacts/w3-scheduler-priority-20261127/gpu7/wcpb-packed-ring-live-c1-timeline/wcpb-auto/phase-data.json) ·
[新版 C4 混合执行](../artifacts/w3-scheduler-priority-20261127/gpu4/wcpb-packed-ring-live-c4-structure/wcpb-auto/phase-data.json)

W3 正式波每波 8 条 4,096→128 token 请求，显式 1/2/3 轮和默认各两条，禁用诊断探针。
同并发的各组使用同一 GPU、相同 128 chunk、1,024 token 预算和 80 GiB KV。
新版在 C1 和 C4 都已补齐同快照、同卡的开关对照，每组独占三波。两个旧 WCPB 组
来自此前各自同卡的独占对照。测量期间发现其他进程的波均剔除。新版两个并发档的
开关对照均为 24/24 条完整输出一致。

| 负载 | 指标 | 普通 prefill | 旧 WCPB | 新 packed WCPB |
| --- | --- | ---: | ---: | ---: |
| W3 C1 | 吞吐（tok/s） | 186.71 | 131.92 | 139.30 |
| W3 C1 | TTFT P95（ms） | 314.49 | 626.29 | 540.80 |
| W3 C4 | 吞吐（tok/s） | 466.32 | 300.24 | 359.40 |
| W3 C4 | TTFT P95（ms） | 492.41 | 993.18 | 737.97 |

新版 C1 三波为 134.01、139.30、139.57 tok/s，C4 三波为 363.55、359.40、305.62
tok/s。新版追回了一部分 WCPB 的损失，开启后相对同 chunk 的关闭组仍存在吞吐和
TTFT 差距。
[新版 C1 同卡六波](../artifacts/w3-scheduler-priority-20261127/gpu7/wcpb-packed-ring-live-c1-clean/matched-comparison.json) ·
[新版 C4 同卡六波](../artifacts/w3-scheduler-priority-20261127/gpu4/wcpb-packed-ring-live-c4/matched-comparison.json)

在 GPU 4 对三个无需补齐的 256 行 packed CORE 增加了层内范围及 CUDA event。每个
样本都有 12 次 attention、12 次 QKV 准备、12 次 MLP 准备和 38 个图段。三次模型
Host 平均 10.89 ms，其中 QKV 准备 2.75 ms，native attention 3.21 ms，MLP 准备
2.90 ms，层尾除下一层 QKV 外 0.74 ms，图段提交 0.88 ms，批首尾状态搬运合计
0.24 ms。MLP 准备包含第一处短卷积、首轮归一化、续轮消息控制及结果写回。模型外
MRv2 metadata 的 Host 区间平均 0.52 ms，其中页号和 slot 路由 0.35 ms。内层事件
明显提高了绝对耗时。该探针用于找位置，批宽与性能差距比较使用上面的轻量时间轴。
[跨轮 CORE 组件时间](../artifacts/w3-scheduler-priority-20261127/gpu4/wcpb-packed-ring-live-components/component-summary.json)

#### WCPB 的 Nsight Systems 对照

GPU 4 上分别完整预热普通 128 和 packed WCPB 128，用同一 4,096-token prompt 生成
8 个 token，逐 token 一致。两臂为同一源码快照，保留 1,024 token 预算、80 GiB KV、
异步调度及原图档。采集 CUDA Graph 节点、CUDA API 和 NVTX。捕捉前后仅可见本臂
EngineCore。以下表格只统计有 prefill 的 CORE。

| 指标 | 普通 128 | Packed WCPB 128 |
| --- | ---: | ---: |
| 实际 CORE 批 | 96 | 45（41 个跨轮） |
| CUDA kernel 数 | 26,528 | 19,508 |
| 各批 kernel 活跃时间合计（ms） | 214.50 | 152.11 |
| 各批首末 kernel 跨度合计（ms） | 239.47 | 429.41 |
| 各批内无 kernel 的区间合计（ms） | 24.97 | 277.30 |
| Host 模型区间合计（ms） | 218.76 | 395.70 |
| Host attention/卷积 metadata（ms） | 0.14 | 20.83 |
| `cudaGraphLaunch` 调用数 / Host 时间（ms） | 1,248 / 13.76 | 1,610 / 14.32 |
| `cudaEventSynchronize` Host 时间（ms） | 0.42 | 0.20 |

GPU kernel 活跃时间按同一个 CORE 批内所有已关联 kernel 的区间并集计算，批内空档
取该批从第一到最后一个 kernel 的跨度减去活跃时间，不包括批间排队。采集节点和
层内 NVTX 会放大绝对时长，轻量时间轴和无 profiler 的六波分别用于估计差距和验收。

Packed WCPB 确实减少了设备计算，主要的增量在批内继续提交。普通 CORE 每批回放
13 个图段，跨轮批为 38。图启动本身的 Host 时间变化约 0.56 ms。更大的变化是
各个图段之间的 eager 工作。Packed CORE 的 QKV 准备累计 106.76 ms，MLP 准备
112.11 ms，attention 范围 110.71 ms。层尾范围包含下一层 QKV 的嵌套，统计时需去重。
新模型在 CORE 启动 `index_elementwise_kernel` 1,533 次、`vectorized_gather_kernel`
1,302 次、独立 `_rms_norm_kernel` 837 次。设备上的 `convMixKernel` 缩短了约 3.90 ms。
模型外 metadata 增量主要是按轮次准备块表及写入位置，新增页路由的 Host 区间为
13.72 ms。现有数据优先支持把归一化、消息混合和短卷积参数准备搬回可回放的图里，
以及减少按批的页路由提交。

[普通128 Nsight](../artifacts/w3-scheduler-priority-20261127/gpu4/wcpb-packed-nsys-retake/ordinary128/prefill.nsys-rep) ·
[Packed WCPB Nsight](../artifacts/w3-scheduler-priority-20261127/gpu4/wcpb-packed-nsys-retake/wcpb128/prefill.nsys-rep) ·
[CUDA/NVTX 解析](../artifacts/w3-scheduler-priority-20261127/gpu4/wcpb-packed-nsys-retake/prefill-profile-summary.json)

#### WCPB 扩大 PIECEWISE 图覆盖

跨轮 CORE 在启动时按现有 token 图档捕获固定形状的输入和状态。Worker 在回放前更新
真实位置、首轮和续轮的标记、状态、短卷积块表和写入槽位。块表和槽位按相同物理
布局共享，native attention 仍读取实际批的变长输入。卷积、归一化、消息控制与线性层
在可回放段内。首轮输入的 BF16 值暂存在 FP32 缓冲中，续轮隐状态保持原 FP32。
补齐行在卷积的写入槽位标记为 `-1`，输出只取实际行。

原有取行、归一化、写回及清理 padding 的操作合为一个逐行 Triton kernel，复用
vLLM 的 RMSNorm 数学。首轮、续轮张量的行数和 stride 作为运行时参数，回放前的
BF16/FP32 输入整理在启动时预热。混合 dtype 及输出 padding 有单独的 GPU 用例。
随机算子对照在一个布局发现归一化输出有一个 BF16 舍入档的差异。真实模型 2,051
token 请求的 8 个输出 token 与旧 packed 及普通 prefill 完全一致。21 个跨轮 CORE
均实际回放 13 个图段、12 个 attention 切点，覆盖 `131→256` 和 `384→512`。
[逐行处理 GPU 对照](../artifacts/w3-scheduler-priority-20261127/gpu4/wcpb-fused-row-native-parity/result.json) ·
[真实图回放和补齐验证](../artifacts/w3-scheduler-priority-20261127/gpu7/wcpb-expanded-graph-fp32-smoke2051/summary.json)

GPU 7 的 W3 C1 保持输入 4,096、输出 128、chunk 128、预算 1,024、KV 80 GiB、
异步调度及原图档。新图路径的三次正式波均独占，SM 为 2,032 MHz，吞吐分别为
202.34、190.97、193.06 tok/s。与同卡旧 WCPB 及此前普通 128 分别逐序列核对，
均为 24/24 完整输出一致。

| W3 C1 | 旧 WCPB | 新图 WCPB | 此前普通 128 |
| --- | ---: | ---: | ---: |
| 吞吐（tok/s） | 139.30 | 193.06 | 186.71 |
| TTFT P95（ms） | 540.80 | 290.14 | 314.49 |

新图 WCPB 比旧 WCPB 提升 38.6% 吞吐。相对历史同卡普通 128，新图 WCPB 吞吐
提升 3.4%，TTFT P95 降低 7.7%。
[C1 三臂对照及输出校验](../artifacts/w3-scheduler-priority-20261127/gpu7/wcpb-expanded-graph-fp32-c1/matched-graph-comparison.json)

同一冻结源码 `ae73562` 的 W3 C4 在 GPU 1 各测三波，SM 均为 2,032 MHz，完整
输出 24/24 一致。普通 128 的吞吐为 451.94、443.85、449.99 tok/s，新图 WCPB
为 467.73、386.45、457.59 tok/s。中位吞吐为 449.99→457.59（+1.7%），本次波动
幅度大于中位差值。TTFT P95 为 583.54→478.49 ms（−18.0%），TPOT P50 为
4.78→5.18 ms。
[C4 同源码同卡对照](../artifacts/w3-scheduler-priority-20261127/gpu1/wcpb-expanded-graph-fp32-c4-fresh-pair/comparison.json)

随后在 GPU 3 对 `ae73562` 做了同源码的 W3 C1 开关配对。普通 128 吞吐
187.51→193.97 tok/s（+3.45%），TTFT P50/P95 从 304.46/312.42 ms 降到
267.34/301.09 ms。三波均独占，24/24 条输出完全一致。
[C1 同源码同卡对照](../artifacts/w3-scheduler-priority-20261127/gpu3/wcpb-expanded-graph-fp32-c1-fresh-pair/comparison.json)

#### WCPB 从调度到 GPU 的逐拍复盘

取冻结源码 `ae73562` 上 GPU 7 的一条真实请求。输入 2,051 token，输出 8 token，
prefill 每段最多 128 token，三轮，异步调度，预算 1,024，KV 80 GiB。窗口允许同一
请求最多三段在途。完整记录共 92 个物理 forward，含 prefill 的 PRELUDE、CORE、
CODA 分别为 15、25、17 次，另有纯 decode FULL 35 次。接下来用其中连续六拍展示
交错执行。A、B、C 是同一请求从位置 896、1,024、1,152 开始的三段，各 128 token。

| 第几拍 | 这一拍 GPU 做什么 | 真实行 → 图档行 | Host 调度（ms） | Host Worker 模型前准备（ms） | Host Worker 整个提交（ms） | GPU 模型事件起止（ms） |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 26 | CORE，A 第 2 轮、B 第 1 轮 | 256 → 256 | 0.031 | 0.668 | 2.986 | 220.041–223.547 |
| 27 | PRELUDE，提前准备 C | 128 → 128 | 0.097 | 0.616 | 1.707 | 223.603–224.588 |
| 28 | CORE，A 第 3 轮、B 第 2 轮、C 第 1 轮 | 384 → 512 | 0.034 | 0.693 | 3.062 | 225.098–229.294 |
| 29 | CODA，A 收尾 | 128 → 128 | 0.037 | 0.477 | 1.544 | 229.333–230.452 |
| 30 | CORE，B 第 3 轮、C 第 2 轮 | 256 → 256 | 0.032 | 0.670 | 2.845 | 230.689–233.883 |
| 31 | CODA，B 收尾 | 128 → 128 | 0.025 | 0.481 | 1.537 | 233.925–235.064 |

第 27 拍为 C 预留 KV 并做 PRELUDE。第 28 拍前，A 的第 2 轮与 B 的第 1 轮已经
完成，后段可读前段对应轮次的 KV。此时三段进入同一个物理 CORE，共 384 个真实行。
其中一份稠密 hidden 缓冲形状为 `[512, 2304]`，128 个 padding 行不产生新的实际 KV
写入。Attention 处理三个 128 行的真实 query 区间，各自的上下文终点为 1,024、
1,152、1,280。12 层 CORE 共享这一次物理 forward，实测回放 13 个图段，中间有
12 个原生 attention 调用点。A 已到终轮，下一拍进入 CODA；B 和 C 的中间状态入环形
缓冲，继续各自下一轮。A、B 的 CODA 提交中间结果，最后的 3-token 段才在第 57 拍
由 CODA 出首 token。那一拍走 `3→4`，模型事件 1.025 ms，随后采样 0.252 ms。

Host 的整次提交包含模型前准备，两列有包含关系。模型前准备覆盖位置、输入及相关
metadata 的处理，当前记录没有单独的 KV 分配和逐算子区间。这个例子中 Host 在本请求
入场后 235.893 ms 完成第 28 拍排程，第 27 拍的回执在 239.039 ms 才开始处理，排程
比上一拍回执早约 3.15 ms。GPU 模型事件在这六拍依次执行，第 27 拍结束到第 28 拍
开始隔约 0.51 ms。GPU 事件与 Host 绝对时间各有原点。这里可以直接核对 Host 提前
排程、同批三段并入一次 CORE 和相邻 GPU 模型事件的间隔。实际 kernel 忙闲使用
Nsight 数据。此诊断有首次形状长尾，表内这六拍是实测切片，正式 TTFT 另以无探针
结果为准。

[全 92 拍 Host/Device 明细 CSV](../artifacts/w3-scheduler-priority-20261127/gpu7/wcpb-expanded-graph-fp32-smoke2051/wcpb-per-forward.csv) ·
[原始时间轴](../artifacts/w3-scheduler-priority-20261127/gpu7/wcpb-expanded-graph-fp32-smoke2051/packed2051-cuda.trace.json)

对窗口决策有用的两份历史对照分别测到了合批成本和下游等待。

- 较早的同卡 Nsight 对照里，WCPB 开启把 prefill CORE 物理批从 96 降到 45，
  kernel 活跃时间从 214.50 降到 152.11 ms，但各批内部没有 kernel 的区间从
  24.97 变为 277.30 ms。那一版跨轮批仍用 38 个图段，现在上述样本回放 13 个段。
  收益要按减少的物理批和批内实际提交开销一起核算。
- W3 C16 有一组历史 A/B 两边都开 WCPB 512。接入已有调度和同轮合批后，提前发
  后段 PRELUDE 的次数从 137 降到 30，物理 forward 从 3,409 降到 1,671，纯
  decode CORE 批从 1,543 降到 597，正式吞吐从 663.74 升到 975.65 tok/s。
  这组同时改了 CORE/CODA 选择和 Worker 合批，差值不能单独归到窗口开段数。

[Kernel 及批内空档对照](../artifacts/w3-scheduler-priority-20261127/gpu4/wcpb-packed-nsys-retake/prefill-profile-summary.json) ·
[两边都开 WCPB 的同热正式波](../artifacts/w3-scheduler-priority-20261127/gpu7/w3-wcpb-current-scheduling-interleaved/summary.json)

窗口策略的独立对照尚待采集。双方保留同一 WCPB Worker、图档、chunk 大小和 KV 配置，
只切换当前固定窗口与按批决定是否多开一段。在 W3 C1 检查少发了哪些真实 CORE，
在 W4 C32 检查原本同批的 decode 是否被拆出、首 token 和下一 token 各多等多少。
两臂需记录每次开段前的候选集合、PRELUDE 新增 KV 页、实际物理拆批和输出回执。
同卡正式波再对照吞吐、TTFT 和 ITL，量化前移一个 chunk 的净收益。

#### W3 C16 放大 WCPB 批预算

在冻结源码 `ae73562` 对照 `max_num_batched_tokens=1024/2048/4096`，每档配对 WCPB 关/开。
每波 32 条 4,096→128，C16，动态 decode 轮数 8/8/16，共 4,096 个输出 token。
均使用 128-token chunk、异步、P/D 混批、TP1 和 80 GiB KV。大预算组加入相应的
PIECEWISE 图档，FULL decode 的请求数档不变。正式波使用独占 GPU，每臂先完整预热，
再跑三波。独立结构波核对真实 CORE 行、packed 图档和 P/D 分布。

先用同源码调度器重放，保留相同请求和轮数、两个在途提交，回执使用固定输出。
两档各行均执行 393,216 行真实 prefill CORE 和 9,144 行真实 decode CORE。

| 预算 | WCPB | 含 prefill CORE 计划 | 每计划真实行均值 | 纯 decode CORE 计划 | 全部调度计划 | 含 prefill CORE 图档真实行占比 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 关 | 414 | 955 | 663 | 1,797 | 98.04% |
| 1,024 | 开 | 445 | 889 | 650 | 1,951 | 93.29% |
| 2,048 | 关 | 192 | 2,048 | 762 | 1,590 | 100.00% |
| 2,048 | 开 | 245 | 1,607 | 744 | 1,691 | 88.81% |
| 4,096 | 关 | 192 | 2,048 | 762 | 1,590 | 100.00% |
| 4,096 | 开 | 110 | 3,575 | 762 | 1,506 | 100.00% |

2,048 WCPB 有 197 个含 prefill CORE 计划超过 1,024 行，132 个达到 2,048。
含 prefill CORE 比 1,024 WCPB 少 200 次，纯 decode CORE 多 94 次。P/D CORE
计划数由 340 降到 81。普通路径也会从扩大预算受益，2,048 的含 prefill CORE 计划
均达到预算，同时 P/D CORE 计划由 315 降到零。更多 prefill 和 decode 分开执行
会改变整波成本，CPU 重放的逻辑计划数与投影图档需要用物理 forward 校验。

4,096 更适合检验 WCPB 独有的额外组批空间。普通路径每拍通常每条请求只能提供一段，
16 × 128 是 2,048 行。4,096 WCPB 在模拟中有 82 批达到 4,096、另有 28 批
达到 2,048，普通仍为 192 个 2,048 行批。两边的纯 decode CORE 都为 762 批。
如果两个路径的 2,048 行 CORE 耗时相同，那么含 prefill CORE 要有收益，WCPB 的
4,096 行实际耗时需低于普通 2,048 行批的两倍。GPU 对照增加 4,096 预算，保留
2,048 和 4,096 图档，验证这 82 次潜在合并是否出现在实际执行中。

两条历史 W3 C16 trace 的首批都先到一条，另外 15 条在第一次调度之后集中到达。
按同样的到达顺序重放，4,096 档普通和 WCPB 的含 prefill CORE 分别为 213 和
131 个计划。WCPB 仍有 84 个超过 2,048，其中 67 个达到 4,096，真实行均值
1,847→3,003。该重放只复现开头的到达顺序，续入仍由合成回执驱动。

2,048 WCPB 在原档+2,048 的含 prefill CORE 投影 padding 为 49,614 行，加入
1,536 档后回算到 25,550，加入 1,280/1,536/1,792 后为 19,406。图尺寸的收益仍需
验证。同一波对照需要同时看两种路径的吞吐变化和各预算下 WCPB 开关的差值。

固定 128 只用于隔离开关的作用。CPU 另跑了普通路径无额外 chunk 上限的原配置。
4,096 预算下，它有 105 个含 prefill CORE，平均真实 3,752 行，另有 726 个纯 decode
CORE。比较 WCPB 的工程收益时还要以该路径为参照。WCPB 的价值要看保持小 chunk 时
是否仍能得到大批的效率，以及首 token 和生成延迟的变化。

确认 2,048 与 4,096 的跨轮图在启动阶段均捕获为 13 个段。首轮检查错误地把 vLLM
按满 8,192-token 请求折算的容量要求固定为 291,018 token。固定 80 GiB 后，预算
2,048/4,096 的日志折算值为 161,241/87,131。重新测试改为检查保留字节及实际物理
块池，并要求同预算开关两侧物理配置一致。

GPU 启动验证三档均为 85,899,345,920 字节、40,960 个物理 KV 块。GPU 0 独占的
4,096 WCPB 结构波共有 1,553 个物理 forward，32 条请求的逐位置轮次全部通过。
127 个跨轮 CORE 都实际回放了 expanded packed 图，其中 83 个通过 B4096 回放，
均有超过 2,048 个真实行并跨不同轮次。64 个为 `4096→4096`，平均模型事件
16.43 ms；16 个全满的 `2048→2048` 跨轮批平均 9.64 ms。用这两个形状的观察均值
归一到每行，前者低约 15%。这两类批的 KV 历史和成员组成存在差异。135 个含 prefill
CORE 的平均真实行 2,914，padding 为 16,952 行，全部物理 forward 均有图回放。
[4,096 WCPB 物理结构](../artifacts/wcpb-budget-20261127/structure-retake/gpu0/4096/wcpb128/structural-summary.json)

| 预算 | 正式计时 GPU | 普通 128 吞吐（tok/s） | WCPB 128 吞吐（tok/s） | WCPB TTFT P95（ms） |
| ---: | ---: | ---: | ---: | ---: |
| 1,024 | 2 | 935.56 | 790.33 | 1,748.38 |
| 2,048 | 3 | — | 916.51 | 1,324.18 |
| 4,096 | 1 | — | 1,029.82 | 1,105.54 |

1,024 的同卡吞吐对照下降 15.52%，TTFT P95 为 1,417.14→1,748.38 ms，首 token
96/96、完整输出 90/96 一致。2,048 WCPB 三波为 916.51、918.62、845.58 tok/s。
4,096 WCPB 三波为 1,029.82、1,029.99、1,013.32 tok/s。上述预算间为不同 GPU 的
趋势。这三组为动态轮次 WCPB 预算趋势。后续对照以原生 vLLM 与 WCPB 的固定三轮
为主，停止大预算的普通 RLT 补测。
[1,024 普通正式波](../artifacts/wcpb-budget-20261127/formal/gpu2/1024/ordinary128/summary.json) ·
[1,024 WCPB 正式波](../artifacts/wcpb-budget-20261127/formal-retake/gpu2/1024/wcpb128/summary.json) ·
[2,048 WCPB 正式波](../artifacts/wcpb-budget-20261127/formal-retake/gpu3/2048/wcpb128/summary.json) ·
[4,096 WCPB 正式波](../artifacts/wcpb-budget-20261127/formal-retake/gpu1/4096/wcpb128/summary.json)

[调度重放与图档回算](../artifacts/wcpb-budget-20261127/scheduler-replay.json) ·
[首条请求先到的重放](../artifacts/wcpb-budget-20261127/scheduler-replay-first-request-alone.json) ·
[普通路径无额外上限重放](../artifacts/wcpb-budget-20261127/scheduler-replay-ordinary-uncapped.json)

##### 对照 vLLM 在 4,096 预算下的实际批大小

原生 vLLM 保持自己的无额外 chunk 上限入口，WCPB 使用 128-token chunk。两个正式臂
均固定三轮、同模型、TP1、80 GiB KV、C16，并对齐 4,096 预算和 FULL/PIECEWISE 图档。
每个请求都生成 128 token。在结构波中比较实际行、图回放和时间。原生一拍执行完整
模型三轮，RLT 一次 CORE 只推进各段一轮，因此原生 forward 数和 RLT CORE 数分别报告。

同一个 W3 C16 语料在原生 AsyncScheduler 的 CPU 重放中，35 个含 prefill 批有
32 个达到 4,096 行，实际输入总数 131,072。原生有 4,096-token 的请求就能直接填满。
固定三轮的 WCPB 重放有 110 个含 prefill CORE，82 个达到 4,096。它证明小 chunk 能
通过跨轮合出 B4096。W3 C16 这份原生配置也有足够工作凑满。旧同模型、预算 4,096
的原生 GPU 4 计时是 1,233.68 tok/s、TTFT P95 802.25 ms，作为历史参照单列。

[原生与 WCPB 固定三轮的 CPU 结构重放](../artifacts/wcpb-budget-20261127/native-scheduler-replay.json) ·
[历史 vLLM 4,096 正式波](/home/zjy/code/david/b_workspace/vllm-rlt-review/artifacts/plugin2-perf.9pulvo/metadata-graph-opt/variable-graph-ab/decode-optimizations/w3-1000-gate/gpu4/compact-mixed-core-filtered-same-budget-native/native-fixed3-4096/summary.json)

| W3 C16，4,096 固定三轮 | 原生 vLLM | RLT WCPB128 |
| --- | ---: | ---: |
| GPU | 0 | 3 |
| 吞吐（tok/s） | 1,232.49 | 941.53 |
| TTFT P50/P95（ms） | 210.52 / 805.95 | 1,113.33 / 1,308.74 |
| TPOT P50（ms） | 11.12 | 7.49 |

两个 GPU 均为独占 B300。表中取各臂前三波中位数。原生六个独占波在
1,231.40–1,233.83 tok/s，时钟采样为 1,995–2,032 MHz。WCPB 三波为
918.83、1,003.54、941.53 tok/s，时钟采样为 2,032 MHz。每波均生成 4,096 token，
配对首 token 一致 96/96、完整序列一致 91/96。WCPB 吞吐为原生的 76.39%。
原生固定三轮的运行期批宽结构波和同卡复测遇到外部进程，均未纳入对照。现有
固定三轮吞吐来自跨卡独占计时，原生可打满 B4096 的形状结论来自实际调度器重放。
[4,096 固定三轮实测及原始波索引](../artifacts/wcpb-budget-20261127/native-fixed3-4096-cross-gpu.json)

##### 普通路径优先的自适应 WCPB 设计

每一拍先按普通规则选择同一阶段的工作。使用普通路径自己的 chunk 上限和已有的
首 token、decode、防饥饿优先级。若能够形成达到真实 token 预算的物理批，或请求槽
已经用完，本拍停止新增 WCPB chunk。若还有余量，检查现有可运行的 prefill、当前
及后续 CORE 的物理合批形状、空闲 KV 页及中间状态槽。优先利用现有图档的空位，
需要升图档时比较两种排法总耗时。调度只用当前已知的请求与依赖。

例如预算 1,024，普通路径能提交一个有 768 个真实行的 B1024 CORE。假如本次已有的
PRELUDE 能顺手准备两个 128-token 段，它们随后能赶上这类 CORE，原本会单独发的
B256 可以省掉，值得尝试开段。若本来就有 1,024 个兼容行，额外准备两段可能只是
换掉已有成员，把别人的最后一段或下一个 decode token 挪后。纯 decode 的 16 条
已就绪请求可以走 B16 FULL，不能从 1,024 总预算倒推出这个批需要填 1,008 行。

决策与执行分开。Scheduler 只在本拍选的是 PRELUDE 时增加新段，在提交计划前检查
既有请求可加入、连续位置和后续读写依赖。只计算一次普通计划，利用其剩余预算，
避免反复试排导致修改 CODA 等待计数或重复申请 KV。增量候选分配失败时提交原批，
额外开段沿用 vLLM 的 KV 分配器、滑窗回收和 prefix 生命周期，同时避免为试探开段
抢占就绪请求。已发出的段继续按该请求自己的位置和轮次跑完，后段 CORE 仍要等前段
对应轮次完成，CODA 仍按位置提交。

已有 WCPB 与普通路径的不同状态布局需要保留。启动时启用 WCPB 能力、预热两类图，
以每个 chunk 创建时的状态归属和实际物理批布局选择 Worker 路径。第一个版本从
状态已排空的请求边界启用 WCPB 窗口，保留普通 chunk 的原尺寸。普通 chunk 仍走
普通位置处理、输入/状态复用及可用的 CODA 快路径。该窗口已经在跑时，即使负载
变满，也只收紧新段准入，原段的 KV、隐藏状态和图布局随其运行到已有的回执边界。
没有在途段以后即可恢复普通准入。这避免把已执行一半的普通大 chunk 搬入按小 chunk
分配的 WCPB 环形缓冲。需要多次实际提交才能执行的逻辑批，按 Worker 的实际分组
与图覆盖计费。跨路径合批只在已有 Worker 兼容能力覆盖时使用。预热的额外图和环形
缓冲会占显存，应记录其开销及两臂实际的 KV 容量。

先交付只利用原有 PRELUDE 拍的保守策略。允许的额外段数先限为一段；检查候选的真实
行能在后续一个或两个 CORE 步内合并，且预估可以减少物理提交。未来合并仍以实际
就绪、图档和执行路径为准。较小的批也可能没有可提前推进的工作，例如 prompt 尾部
已经全部入批、依赖还在途，或者纯 decode。碰到这些条件直接按原节奏提交。若保守
策略的正式数据表明窗口更大有收益，再评估允许第三段或升档。

用一个能自然填满的场景和一个有真实 prefill 空位且存在可提前推进的段的场景验收。
W3 C16 的 4,096 档用于确认避免退化，因为 vLLM 的普通调度已经能做大批。低负载的
受控小 chunk 实验可用于检查开窗和排空是否有效，与默认原生 vLLM 的性能比较仍用其
正常配置。结构波记录每次普通物理批的真实行、开段原因、KV 与图档，以及少掉的
CORE、增加的纯 decode 提交和首 token/下一 token 延迟。正式计时限定到代表负载，
同波工作量与正确性通过后比较吞吐、TTFT 和 ITL。

##### 自适应 WCPB 实现和 W7 验收

使用 `--rlt-adaptive-wavefront-prefill` 开启自适应模式。`--rlt-prefill-chunk-size`
为 WCPB 单段上限，未设置时取 `min(128, max_num_batched_tokens)`。原普通
prefill 的 `long_prefill_token_threshold` 不受 WCPB 上限影响。两种 WCPB 模式单独启用。

Scheduler 先组普通批，只在该 PRELUDE 尚有真实余量和调度槽时尝试给一条新段加一个
连续后段。前段已在该批被选中、请求在该段前没有其他在途操作，并且前段可以放进
WCPB 中间状态缓冲。后段按剩余预算、剩余 prompt 和单段上限裁剪。已有 decode 时
只填当前 PIECEWISE 图档内的空行。可选后段的 KV 申请失败时保留原计划。两段一同
标记为 WCPB，沿用逐轮 KV 依赖，按请求位置完成 CODA。先组同轮 CORE，有余量再检查
已就绪的 WCPB 跨轮段。Worker 普通批保留普通输入和状态复用、正常 decode 槽及
CODA 尾部优化，WCPB 所有权按实际段保存在环形缓冲。两类图在启动时捕获。

普通路径不设单段额外上限时，所选长段可以消耗本拍剩余预算，留下的未满批可能只剩
已经全部入批的 prompt 尾部。在 W7 直接比较 vLLM 与自适应 RLT 时，两边均明确设
`long_prefill_token_threshold=128`，以同配置观察满批和可提前推进的未满批。
另外保留原生默认无额外上限的 W7 数据作为参照。W7 使用相同 ShareGPT 64 条请求、
C32、固定三轮、预算 1,024、80 GiB KV、TP1、异步和相同图档。

CPU 插件测试 138 通过、3 跳过。覆盖满批保留普通路径、开出一个短段、同批运行两条
依赖就绪的 CORE、异步排空、KV 不足回落，以及有 decode 时图档保持不变。

W7 在同一张独占 GPU 4 上运行。vLLM cap128 的结构波中，含 prefill 的物理批有 7 个
达到 1,024 个真实行，37 个未达到。自适应结构波中，7 个普通 PRELUDE 已占满 1,024 个
真实 token 行，全部保持普通执行。52 个未满的决策点里有 10 个能开新段，共补入 843 个
真实行。其余 42 个没有满足条件的请求，其中 33 个已选请求只剩 prompt 尾段。实测
976 行的混合批补了 48 行，继续复用 1,024 档。10 个跨轮 CORE 物理批均重放了 256
档的 packed CUDA Graph。64 个请求通过逐位置的 PRELUDE、三轮 CORE、CODA 执行
计数检查。这次实现以实际预算、就绪依赖、预留 KV 和图档为准，尚未接入上文提议的
后续 CORE 合批成本预测。

[W7 自适应调度证据](../artifacts/w7-adaptive-wcpb-20261127/adaptive-v2-cap128-structure/adaptive-evidence.json) ·
[W7 自适应执行统计](../artifacts/w7-adaptive-wcpb-20261127/adaptive-v2-cap128-structure/structural-summary.json) ·
[W7 vLLM 执行统计](../artifacts/w7-adaptive-wcpb-20261127/native-cap128-structure/1024/native/structural-summary.json)

W7 正式数据为每组三波无计时插桩的中位数，均使用 GPU 4，记录到的 SM 时钟为
2,032 MHz。每波 64 个请求均输出 512 token，合计 32,768 token。

| W7 C32，普通 prefill 上限 128 | 吞吐（tok/s） | TTFT P95（ms） | TPOT P95（ms） | ITL P95（ms） |
| --- | ---: | ---: | ---: | ---: |
| vLLM | 3,683.61 | 263.28 | 8.51 | 8.50 |
| RLT，自适应关闭 | 3,646.82 | 275.62 | 8.61 | 8.64 |
| RLT，自适应开启 | 3,605.85 | 297.52 | 8.74 | 8.46 |

自适应吞吐达到同配置 vLLM 的 97.89%，相对 RLT 关闭自适应下降 1.12%，三波均低于
关闭侧的各波。开启后 TTFT P95 较关闭增加 21.89 ms。启动时图分配量增加 270 MiB，
KV 仍为相同的 40,960 物理块。vLLM 第一波为 2,781.72 tok/s、TTFT P95 2,937.76 ms，
保留在中位数计算内。原生 vLLM 无普通段上限的参考中位数为 3,721.39 tok/s。

RLT 的两次独立结构波各处理 58,338 个真实 prefill CORE 行，即 19,446 个 prompt token
各跑三轮。含 prefill 的物理 CORE 提交从 129 增至 155。其中纯 prefill 从 9 增至 32，
含 decode 的混合批从 120 增至 123。两波请求到达时机有差异。这是整波结构实测，
可确认开启侧虽然发生了跨轮合批，整波的物理提交数仍未减少，不能从中分离每次开段的成本。

逐 token 比较中，三波 RLT 开启交叉对三波关闭，64 个请求完全相同的中位数为 12，
RLT 关闭侧重复波之间为 22，vLLM 重复波之间为 20。输出长度与阶段执行计数均校验
通过。输出内容仍有差异，逐 token 一致性尚未通过。该特性保持显式启用。

[同配置 vLLM 对比](../artifacts/w7-adaptive-wcpb-20261127/cap128-comparison.json) ·
[RLT 关闭自适应正式结果](../artifacts/w7-adaptive-wcpb-20261127/adaptive-v2-cap128-off-formal/summary.json) ·
[RLT 开关结构波对比](../artifacts/w7-adaptive-wcpb-20261127/cap128-structure-comparison.json) ·
[输出前缀对比](../artifacts/w7-adaptive-wcpb-20261127/prefix-parity.json)

##### 未满批恢复 WCPB 轮转组批

普通调度先选批。普通批用完 token 预算后直接提交。有余量时，PRELUDE 复用固定 WCPB
的轮转组批，给每个可入选的请求补一段，再回到仍有空间的请求，直到耗尽 token、请求槽、
KV 或中间状态槽。单段长度受 `rlt-prefill-chunk-size` 上限约束。已有 decode 时，从
含 prefill 的 PRELUDE 补段只填当前图档内的空位，纯 decode 的 PRELUDE 保留 FULL 路径。
PRELUDE 中相邻的多个段由 MRv2 合成同一请求的查询，仍分别保存中间状态。只有额外
KV 申请成功后才切换普通前段的状态归属。

已进入 WCPB 的窗口可以在已有段完成后滚动补段。CORE 先尊重普通调度的优先级和同轮
组批。仍有空位且有就绪 WCPB 段时，可合入其他请求和轮次的兼容工作。当前 CORE 的物理
元数据保留每请求每轮一条查询及总查询数上限，并继续检查前段同一轮的 KV 是否已就绪。
无必须先执行的工作且本拍仅有少量 prefill CORE 时，允许先补已有窗口，随后再按实际
就绪情况组成 CORE。CODA 仍按请求位置完成。

CPU 插件测试通过 142 项、跳过 3 项。新增覆盖多个请求轮转、连续多轮补段、旧段完成后
滚动补段、WCPB CORE 合入普通请求，以及纯 decode 保留原批。

初次结构波 `source-round-robin-v1` 观察到 459 次为 31 行纯 decode 的批补入 1 个
prefill token。实际执行因此从 FULL 的 decode 路径切到 PIECEWISE，混合 CORE 提交由
关闭侧的 120 增到 1,501 次。该版唯一正式数据波吞吐 2,814.49 tok/s。修复后重新冻结
`source-round-robin-v2`，保留这次数据用于归因。

修复后的 W7 结构波中，5 个普通 PRELUDE 原本就满，占用 1,024 个真实行，均保持普通
执行。未满时开了 8 个窗口，另有 29 次给已打开的窗口补段。纯 decode 批补段次数为零。
一次 635 行的 PRELUDE 在同一批中按顺序为两个请求补了 112 和 128 行。57 个跨轮
CORE 物理批均重放 packed CUDA Graph，每批真实行数中位数为 256。64 个请求的逐位置
阶段执行计数全部通过。

含 prefill 的 CORE 在两次独立结构波中共处理相同的 58,338 个真实 prefill 行。关闭侧
提交 129 批，物理行数 77,568。轮转版提交 167 批，物理行数 73,920。物理行数包括
同批 decode 和 padding，两波请求到达时机不同。实测物理行数降低，提交数仍增加。

正式计时使用同一张 GPU 4、同一 W7 C32 语料和上述阈值 128 配置。vLLM、RLT 关闭、
单段版保留已完成的同配置三波数据。轮转版另跑三次预热后的正式波。

| W7 C32 | 吞吐（tok/s） | TTFT P95（ms） | TPOT P95（ms） | ITL P95（ms） |
| --- | ---: | ---: | ---: | ---: |
| vLLM | 3,683.61 | 263.28 | 8.51 | 8.50 |
| RLT，自适应关闭 | 3,646.82 | 275.62 | 8.61 | 8.64 |
| RLT，单段版 | 3,605.85 | 297.52 | 8.74 | 8.46 |
| RLT，轮转版 | 3,528.49 | 347.73 | 8.72 | 8.72 |

轮转版吞吐为同配置 vLLM 的 95.79%，比 RLT 关闭低 3.24%，比单段版低 2.15%。
轮转版第一波输出 32,768 token，吞吐 3,543.68 tok/s。其余两波各输出 32,516 token，
同一请求在 260 token 时自然结束。所有组均允许 EOS。逐位置阶段计数通过，输出内容
一致性仍待验证。三轮轮转版与三轮关闭版两两比较，每 64 个请求的首 token 相同数中位为
63，完整序列相同数中位为 8。关闭版自身重复波完整序列相同数中位为 22。

[首次轮转结构记录](../artifacts/w7-adaptive-wcpb-round-robin-20261127/round-robin-evidence.json) ·
[修复后轮转结构记录](../artifacts/w7-adaptive-wcpb-round-robin-20261127/round-robin-v2-evidence.json) ·
[轮转版同配置对比](../artifacts/w7-adaptive-wcpb-round-robin-20261127/three-arm-comparison.json) ·
[轮转版输出比对](../artifacts/w7-adaptive-wcpb-round-robin-20261127/prefix-parity.json)

###### W7 提交次数增加的调度归因

两个结构波的 Scheduler 计划数均与 Worker 实际 forward 数一一对应。含 prefill 的
PRELUDE、CORE、CODA 分别从 43、129、43 次增加到 60、167、72 次。CORE 增量中纯
prefill 从 9 到 39 次，P/D 混合从 120 到 128 次。含 prefill 的 CORE CUDA Graph
分段重放次数从 1,677 到 2,171。结构波记录的是物理 forward 与图段，逐个 GPU kernel
的启动数需由设备时间轴另行测量。

轮转版第一条 2,505-token 请求先进入 WCPB。其余 31 条在约 6.2 ms 内已抵达
Scheduler，此后该请求仍单独占用 29 个 CORE，直到约 177 ms 才首次让其他请求进入
PRELUDE。中间 17 次从零基础行启动的 PRELUDE 补段均服务同一个已打开窗口。关闭版
同样先单独推进该请求，3 个 CORE 后在约 13 ms 组成了含其他请求的 1,024 行 PRELUDE。

普通批选择先扫描运行请求，选定 CORE 后，本轮跳过需要 PRELUDE 的等待请求。
WCPB 随后可以把一个有余量的纯 prefill CORE 换成 PRELUDE，但传入的是空候选，
补段列表只包含已开启窗口的请求。普通模式每段 CODA 后有重新配批的机会，窗口滚动
期间持续出现的 CORE 加上仅回填旧窗口的 PRELUDE，让这个机会一直推迟。当前是否
开启或继续窗口的判定关注本拍空行，没有比较同一批已到达请求在普通 PRELUDE 和
WCPB 路径中的组批机会。

固定到达顺序的 CPU 调度复现使用 A 的 20 个 prompt token、B/C/D 各 12 个、单段上限
4、总预算 16。B/C/D 都在 A 首次 PRELUDE 提交后加入。同步与异步结果一致。关闭版让
A 单独提交 3 个 CORE、完成前 4 个 token 后，下一批是 A/B/C/D 各 4 个 token。轮转版
先让 A 独占 7 个 CORE、5 个 CODA 和 2 个补段 PRELUDE，完成全部 20 个 token 后才
给 B/C/D 首轮调度。这一序列固定了到达顺序，复现了请求准入饥饿。

下一版设计应在每次窗口补段之前考虑已到达请求的首段，保持 CODA、decode 和防饥饿
优先级，并在候选间比较实际 CORE 提交数、图档和阶段开销。预估使用实际可入选请求与
同轮 KV 就绪条件，选定后再申请额外 KV。固定顺序复现可作为请求准入回归用例，W7
则比较各阶段物理提交与图段重放次数。

###### WCPB 续批补入其他请求

当有余量的纯 prefill CORE 允许先补 WCPB 窗口时，先为一个已开启的窗口预留确实能
推进的段。成功后才决定发 PRELUDE。其他已到达请求可以加入这个 PRELUDE，依据现有候选
顺序、活跃请求上限、prefill 限流和异步就绪规则，补进首段或下一段。剩余预算交给原
WCPB 轮转。只申请可用 KV；旧窗口无空位或无法预留时继续执行已就绪的 CORE。被让过
一次的 CORE 从现有等待最久优先级重新参与下一拍。

同步与异步回归验证，A 的 WCPB 窗口已满时，单个短请求到达后 A 的 CORE 按原计划执行。
A 的旧段完成并需要补段时，B 可以和 A 共用一个 PRELUDE。插件测试通过 145 项，
跳过 3 项。

中间版 `source-admission-v1` 会先以普通候选决定能否发 PRELUDE。CPU 复现中，B/C/D
均是单 token 请求，B 到达便能单独占一个 PRELUDE。`source-admission-v2` 把是否换拍
交回 WCPB 窗口续批。普通调度本来选中 PRELUDE 时继续按普通规则组批。

中间版的同配置 W7 结构波中，首个含其他请求的 PRELUDE 在约 5.7 ms 提交，给 14 个请求分配
了 1,024 个有效行。前一个轮转版约 177 ms，期间第一条请求单跑 29 次 CORE；本版
单跑为零。含 prefill 的 PRELUDE、CORE、CODA 物理提交为 37、125、54 次。前一轮转版
为 60、167、72 次，关闭版为 43、129、43 次。三次独立结构波均处理 58,338 个真实
prefill CORE 行。49 个跨轮 packed CORE 均重放图，64 条请求通过逐位置阶段计数。

最终版 `source-admission-v2` 的 W7 结构波有 7 次真实 WCPB 窗口补段，其中 2 个满批、
5 个未满批。另外 5 次判断因为窗口没有可预留段，继续执行已就绪的 CORE。首个含其他
请求的 PRELUDE 在 18.36 ms 提交，实际由第一条请求的 128-token 后段和 13 个新请求
的首段组成，合计 1,024 个有效行。此前第一条请求独占 CORE 的次数由 29 降为 4。
含 prefill 的 PRELUDE、CORE、CODA 提交分别为 45、153、65 次，63 个跨轮 packed CORE
均重放图，64 条请求通过逐位置阶段计数。提交数较上一轮转版减少，含 prefill 的 CORE
仍高于关闭版的 129 次。结构波的到达时机与批次分布有差异。

正式计时沿用同卡同配置、每组三波的中位数。最终版为 3,548.93 tok/s、TTFT P95
317.42 ms。吞吐较上一轮转版增加 0.58%，比关闭版低 2.68%，达到同配置 vLLM 的
96.34%。TTFT P95 低于此前轮转版的 347.73 ms，高于关闭版的 275.62 ms 和 vLLM 的
263.28 ms。最终版每波实际输出 32,588 token，同一请求在 332 token 时自然结束。
关闭版与 vLLM 每波输出 32,768 token。各组均启用 EOS。最终版与关闭版的首 token、
完整序列逐请求一致数中位分别为 60/64 和 7/64，输出一致性仍未通过。

[中间版 W7 记录](../artifacts/w7-adaptive-wcpb-admission-20261127/report-v1-interim.md) ·
[最终版 W7 结构证据](../artifacts/w7-adaptive-wcpb-admission-20261127/admission-v2-evidence.json) ·
[最终版同配置性能](../artifacts/w7-adaptive-wcpb-admission-20261127/four-arm-comparison-v2.json) ·
[最终版输出对比](../artifacts/w7-adaptive-wcpb-admission-20261127/prefix-parity-v2.json)

###### WCPB 最终版 W7 的耗时归因

正式三波关闭与开启中位耗时分别为 8.985、9.182 秒。即使开启波的输出少 180 token，
整波仍多用约 197 ms。以下来自独立插桩波，按每个物理 forward 的 CUDA event 区间
累加，只统计带 prefill 的批，混合批包含其中的 decode 行。

| 含 prefill 的阶段 | 关闭批数 | 开启批数 | 关闭（ms） | 开启（ms） | 差值（ms） |
| --- | ---: | ---: | ---: | ---: | ---: |
| PRELUDE | 43 | 45 | 90.53 | 93.99 | +3.45 |
| CORE | 129 | 153 | 549.14 | 665.77 | +116.63 |
| CODA | 43 | 65 | 152.84 | 178.78 | +25.94 |

含 prefill 的 CORE 实际 prompt 计算量均为 58,338 行。物理图行数从 77,568 降到
72,576，每批真实 prefill 行从平均 452 降到 381。物理图档 ≤256 的批从 39 增到 91，
≥512 的批从 90 降到 62。切段后总 PRELUDE prompt 段数 192 → 199，小于 128 token 的
段数 69 → 89。例如第 63 条的 696-token prompt 从 5 个 128 加 1 个 56 改为 3 个 128、
3 个 97 加 1 个 21。含 prefill 的 CORE CUDA Graph 分段重放由 1,677 增到 1,989，
两边均有图覆盖。

首个含新请求的满 PRELUDE 提交后，关闭版的 1,024 行第 1 轮 CORE 一次提交。开启版
计划 5326 也准备了 1,024 行；紧接着 5327 只执行其中 765 行第 1 轮，另有 256 行由
之前准备好的段在第 2、3 轮占用。剩余 259 行第 1 轮在 5328 单独发，使用 512 图档。
两次独立波的首个满批含同样数量的请求和相同的段长度分布，旧段所在位置不同。这一
例说明旧就绪工作与刚准备的一轮在同一个预算争抢行数时，跨轮合批可以打散新同轮批。

确定性 CPU 调度复现固定请求到达拍、输出长度和回执顺序。预算 16、段上限 4，输入
按（到达拍、prompt 长度、输出长度）为 `(0,20,8)`、`(1,12,4)`、`(1,28,2)`、
`(8,17,2)`、`(25,17,8)`。开关两边实际 prefill CORE 都是 282 行，decode CORE 都是
57 行。含 prefill 的 PRELUDE、CORE、CODA 调度计划从 10、30、10 变成 10、31、21，
其中只含 1–8 个 prefill 行的 CORE 从 12 变成 19。该复现只调用 Scheduler，已能看到
更多小批和计划，证实执行端的 CUDA Graph 与异步到达抖动无需参与这一机制。

用关闭版同图档的常规 CORE 均值乘开启版的各档次数估算。关闭版没有 64 档样本，取
开启波常规路径的 64 档均值。关闭版实测 549.14 ms，套用开启版批次分布得到 602.61 ms，
开启版实测 665.77 ms。约 53.46 ms 与批次数和图档分布变化相关，余下 63.16 ms 混有
执行路径与实际批内容变化。用同一开启波常规路径的各档均值替代其中 packed CORE 的
计时，估算差 35.27 ms。在该波 512 和 1,024 档，packed CORE 均值分别为 4.74 与
6.19 ms，普通路径为 4.22 与 5.48 ms。相同图档仍可能有不同的段长与上下文组合。

调度开段依赖当前 PRELUDE 余量、槽、KV 与图档，CORE 只在同轮普通候选选完后加入
当前已就绪的跨轮段。这个决策缺少对后续物理批能否合并的判断。执行端 packed CORE
每次准备独立位置、拼接隐藏状态、生成并上传路由、回写每个段的状态。现有 CUDA Graph
只覆盖其中的执行片段。CODA 当前每个请求按最早的位置推进，后段不会随前段 CODA 入批；
带 WCPB 的 CODA 计划还停用末层 MoE 尾部裁剪。上述实现点均有独立代码路径，分项
耗时仍需单独 A/B。去掉两边各一个超过 60 ms 的 CODA 事件后，CODA 仍多约 23.8 ms。

两次插桩波的物理设备时间轴相差 +218 ms。模型和采样记录区间合计缩短约 84 ms，
未覆盖区间多约 302 ms，其中长于 5 ms 的少数间隔多出约 250 ms，主要邻接纯 decode。
两波输出长度和到达时机不同。这些间隔无法从现有 trace 归因给 WCPB 或精确分摊到正式
吞吐差。约 146 ms 的带 prefill 阶段增量仍可用来定位应优先检查的路径。

[关闭结构波](../artifacts/w7-adaptive-wcpb-20261127/adaptive-v2-cap128-off-structure/structural-summary.json) ·
[开启结构波](../artifacts/w7-adaptive-wcpb-admission-20261127/admission-v2-structure/structural-summary.json)

#### WCPB W1–W7 开关矩阵

矩阵用冻结的 `ae73562`，保持 Inkling2-mini、TP1、1,024 token 预算、80 GiB KV、
异步调度和原 CUDA Graph 档。使用现有的 W1–W7 标准语料、并发、请求数与输出上限。
每个 case 跑固定三轮和请求级动态轮次，动态请求按既有 1/2/3/默认各占四分之一的分配。
每组完整预热，保留三波同卡独占且 SM 2,032 MHz 的正式数据。

对每个 case 和轮次配置，在 P/D mixed batch 开、关下分别对比普通 128 与 WCPB 128。
关闭 P/D mixed batch 的独立测量源码仍允许 prefill 不同轮次互相合批。关闭组在 Worker
按 prefill 和 decode 分开执行，保留调度优先级、逻辑计划和异步 CODA 回执。期望覆盖
28 个同卡 WCPB 开关对照。这一轮保持自投机和 Decode pipeline 关闭。已有低并发 8
请求的 WCPB 对照独立保留，标准 W3 使用 64 请求，其他历史结果也需要完全匹配方可复用。

[矩阵汇总](../artifacts/w3-scheduler-priority-20261127/wcpb-matrix-20261127/matrix-summary.json) ·
[后续自投机参考](https://github.com/InferMatrixLab/vllm-rlt/blob/main/docs/async-speculative.md)

##### 暂停时的结果与归因

暂停时保留 101 波原始记录。99 波属于各自做完三次重复的组，其中 66 波组成 11 个
完整的同卡 WCPB 开关对照。下表只列允许 prefill 和 decode 合批的完整对照，吞吐取
各组三波中位数。

| 负载 | 固定三轮，关闭 → 开启（tok/s） | 请求级动态轮次，关闭 → 开启（tok/s） |
| --- | ---: | ---: |
| W1 C1 | 258.06 → 249.98（−3.13%） | 326.92 → 304.78（−6.77%） |
| W3 C1 | 142.96 → 166.98（+16.80%） | 180.29 → 174.54（−3.19%） |
| W4 C32 | 1,044.24 → 801.06（−23.29%） | 956.61 → 771.52（−19.35%） |
| W7 C32 | 3,459.53 → 2,740.94（−20.77%） | 2,469.01 → 3,563.91（+44.35%） |

- W1 每个 prompt 正好 128 token，单请求只有一个 prefill chunk，无法靠同一请求内
  跨 chunk 合批省掉 CORE 提交。当前开关仍改变输入位置准备和中间状态复用。分开
  执行 prefill 和 decode 时，固定三轮的 +1.09% 落在波间波动内，动态轮次下降 2.95%。
- W3 每个 prompt 有 32 个 chunk，之前的 C1 诊断中，同样的 98,304 行实际 prefill
  CORE 从 768 次提交合成了 360 次。这次 64 请求对照的固定三轮吞吐区间分别为
  137.84–160.82 和 136.00–169.16 tok/s，动态轮次区间同样重叠。分开执行时，固定
  三轮为 161.44 → 154.24 tok/s（−4.46%）。现有数据证实了合批能力，W3 吞吐差值
  的大小和方向仍受波间波动影响。
- W4 的普通路径只需同时有八个就绪的 128-token chunk 即可填满 1,024 token 预算。
  WCPB 改变同一请求可并行推进的 chunk 数以及每批选中的工作，并关闭普通路径对
  大 prefill CODA 批的最后一层 MoE 跳算优化。固定和动态轮次的 WCPB 三波吞吐都
  低于各自所有普通波，TTFT P95 和 TPOT P50 同时上升。当前没有同批次时间轴可把
  W4 的损失分摊到实际组批、MoE 和输入准备。
- W7 动态轮次每波都生成 32,768 token，允许合批时 TPOT P50 从 12.07 降到 8.72 ms，
  TTFT P50 从 103.59 降到 95.25 ms。分开执行的普通路径已经达到 3,740.03 tok/s，
  相应的 WCPB 开启侧还没有成对数据。允许合批时两臂完整输出逐 token 一致的请求为
  18/192，生成长度一致，生成内容的差异仍需定位。+44.35% 是相对本次允许合批的
  普通路径的结果，现有数据无法量化跨轮次 prefill 合批自身贡献了多少。

[暂停覆盖记录](../artifacts/w3-scheduler-priority-20261127/wcpb-matrix-20261127/paused-coverage.json) ·
[同卡对照及原始记录索引](../artifacts/w3-scheduler-priority-20261127/wcpb-matrix-20261127/matrix-summary.json)

### 自适应批量的成本与调度实验

调度比较的是同样的有效工作最终需要多少次物理 forward，以及每次占用设备多久。
以 `op × CORE 轮次 × 执行路径 × 实际图档` 为成本表的键。图档预测线性层等固定形状
算子的工作量。attention 还取决于每个请求的真实 query 行和各 KV group 的有效历史。
对一个 group，可用每个 query 的 `min(实际历史长度, 滑窗)` 之和作工作量特征。
块表与状态重排按实际准备和复用情况计费。Host 与设备重叠的区间只计入关键路径
上暴露的部分。Worker 若把一个计划拆成两次物理 forward，应分别计费。

对两种排法，先校验数据依赖、KV 页、图覆盖和首 token/已有 decode 的等待风险，再
比较本次与受影响的后续物理批的总耗时。一个 160 行的批已经用 B256 图，加 80 个
真实行仍用 B256，但 attention 会处理新增的 80 行。填入这些行能否省一次提交，取决于
这些行原本会单独跑，还是能进入后续已有的批。

| 历史证据 | 实际观察 | 对调度的含义 |
| --- | --- | --- |
| W3 C1，长 prompt | 普通 128 cap → 无额外上限，吞吐 174.23 → 279.73 tok/s，TTFT P95 358.69 → 81.12 ms | 单条 4,096-token prompt 由 32 个 chunk 降到 4 个，逐 chunk 的 PRELUDE、3 次 CORE、CODA 计划合计 160 → 20 |
| W3 C1，历史 CUDA event | 普通 128 与普通 512 两条各 8 请求的结构波有同样 163,840 行 prefill，图均无 padding；物理 prefill forward 为 1,280 与 320，模型事件累计 2,670.18 与 1,347.05 ms | 两条结构波来自不同卡。图内利用率均为 100%，减少提交次数仍有明显空间，正式收益以同卡计时为准 |
| W3 C16，原预算 | 655 个带 prefill 的 PIECEWISE 批，645 个用 B1024，真实 661,108 / 物理 663,680 行（99.61%） | 可继续填充的总行数很少。继续合批须减少实际提交，或让单批本身更便宜 |
| W7 C32，原预算 | 带 prefill 的 114 批行利用率 90.97%，但模型事件合计 521.51 ms，约占设备模型首末事件跨度的 5.9%。纯 decode FULL CORE 有 3,014/3,051 批已经达到图档宽度 | 仅优化这部分 prefill 直接成本，空间有限。缩减 decode 的阶段和提交开销，需要单独用实测验证 |

W3 C16 旧表将一次普通 prefill chunk 的五段模型事件相加，B128、B512、B1024
分别为 12.72、16.96、21.98 ms。若两个 B512 的工作可合法合成同路径 B1024，
估计模型区间节省 `2 × 16.96 − 21.98 = 11.94 ms`。B512 各段只有一份历史
样本。短上下文、不同 KV 分布及不同 Worker 路径应使用各自的实测定价。

KV 容量和 KV 读取成本在模型里各自约束。历史 W3 C16/W7 C32 的峰值重放为
3,764/2,507 块，对应 80 GiB 池中的 40,960 块，容量空间充足。W7 同一个 B32
第 1 轮图档中，batch 逻辑历史中位数 11,906 和 23,946 的两组，模型区间中位数
为 1.660 和 1.699 ms。当前候选用 vLLM 的实际空闲页数给申请新 chunk 和已运行
decode 下一 token 留余量。已有 `KVCacheManager.allocate_slots()` 继续负责真分配，
图的 padding 不申请 KV。

首个实验候选在普通路径上借鉴 vLLM 按输入预算公平放大 chunk 上限的做法。基准上限
为 128，按本拍能开始的 prefill 的剩余长度计算共同上限。仅一个就绪长请求且没有
在途 decode 时，上限可到 1,024；四个等长请求可到约 256；八个 128 已打满则保留
原上限。比较两种对在途 decode 的处理。保守臂只填入根据当前可提交行数估计的原图
档，允许升档的臂按就绪请求填向现有最大图档。Worker 的真实图档和混批组成由独立
结构波核对。等候策略、请求轮次、真实 token 预算和 FULL/PIECEWISE 捕捉表沿用原
配置。实验通过测量入口启用。

用冻结 scheduler 和相同请求语料做 CPU 重放。按并发补请求，最多两份计划在途；用
固定输出回执驱动调度器，使用 vLLM 分配器与重建的 12 个 KV group。投影图档来自原
捕捉表。W7 无额外上限的重放复现历史 GPU 结构波的 114 批和 9,803 行 padding。

| 负载 | 固定 128 | 固定 512 | 无额外上限 | 自适应只填原档 | 自适应允许升档 |
| --- | ---: | ---: | ---: | ---: | ---: |
| W3 C1，含 prefill 计划 / padding 行 | 10,240 / 0 | 2,560 / 0 | 1,280 / 0 | 1,280 / 0 | 1,280 / 0 |
| W4 C32，含 prefill 计划 / padding 行 | 1,430 / 21,276 | 1,335 / 22,293 | 1,315 / 7,134 | 1,365 / 4,236 | 1,310 / 7,814 |
| W7 C32，含 prefill 计划 / padding 行 | 205 / 31,601 | 125 / 15,935 | 114 / 9,803 | 140 / 869 | 115 / 1,161 |

严格保住原图档可能推迟有用行而增加整波提交。W7 允许升档从 140 降到 115 个计划，
只增加 292 行 padding。相对无额外上限，它以一个额外计划换掉 8,642 行 padding。
用历史 W3 的 B512、B1024 五段平均值做一阶估算，每次提交的截距为 2.3883 ms，
每物理行为 0.001960 ms。W7 无额外上限有 108,544 物理行，严格原档 100,768 行，
允许升档 100,448 行。相对无额外上限，严格原档的模型区间差估为
`26 × 2.3883 − 7,776 × 0.001960 = +46.86 ms`，允许升档为
`1 × 2.3883 − 8,096 × 0.001960 = −13.48 ms`。这是套用 W3 大图档定价的
敏感度，不含 decode 排队、混批构成或 KV 访问的变化。适合排序候选，无法直接预测
正式吞吐。W4 允许升档在重放里同时减少五个计划和 4,352 个物理行。

同一重放还检查从请求入场到首 token 之间实际经过的计划数。W4 无额外上限的
P50/P95 为 60/594，允许升档为 170/640。W7 对应为 15/64 与 10/45。不同计划的
实际时间不同，这些计数只标识队列与进度变化。W4 的长 prompt 都在同时推进时，
让更多请求较早开始可能会推迟前一批首 token，需要同时比吞吐和 TTFT。

自适应候选最初在每次调度前扫描就绪工作，W7 允许升档的整波策略 CPU 时间约
73.6 ms。增加纯 decode 早退后，同一重放约 18.4 ms。三种负载的计划数、padding
与逐位置检查结果一致。CPU 时间会与设备运行重叠，使用正式对照确认对吞吐的影响。

W4 在 40,960 块正常池和 6,144 块压缩池的重放都无抢占。压缩池的无额外上限峰值为
6,129 块，两个自适应均为 5,945 块。允许升档在压缩池因页余量回退，含 prefill 计划
由正常池中的 1,310 增至 1,355。CPU 重放校验调度、位置、预算及分配行为，吞吐和
尾延迟由 GPU 对照测定。

GPU 对照使用冻结的 `ae73562`，对比普通固定 128 / 512 / 无额外 chunk 上限，以及
上述自适应策略。每个同服务交错测量保留原 1,024 预算、80 GiB KV、异步和 P/D 混批。
首先用 W3 C1 检验填满和减少提交，用 W4 C32 检验长 prompt 争用下的吞吐与尾延迟，
用 W7 C32 检验真实输入及生成阶段影响。正式比较吞吐、TTFT、TPOT 和逐 token 输出，
独立结构波用于核对物理提交与图档。

| 正式计时，均为请求级动态轮次 | GPU | 配置 | 吞吐中位数（tok/s） | TTFT P50/P95（ms） | TPOT P50（ms） |
| --- | ---: | --- | ---: | ---: | ---: |
| W3 C1 | 3 | 固定 128 | 176.25 | 336.99 / 391.33 | 3.31 |
| W3 C1 | 3 | 固定 512 | 257.46 | 116.87 / 119.84 | 3.27 |
| W3 C1 | 3 | 无额外上限 | 268.48 | 82.46 / 86.33 | 3.27 |
| W3 C1 | 1 | 无额外上限 | 270.92 | 82.00 / 84.59 | 3.27 |
| W3 C1 | 1 | 自适应允许升档 | 265.21 | 81.92 / 88.03 | 3.27 |
| W4 C32 | 1 | 固定 128 | 1,069.93 | 798.24 / 2,928.18 | 20.51 |
| W4 C32 | 1 | 固定 512 | 1,176.88 | 441.37 / 2,632.24 | 21.25 |
| W4 C32 | 1 | 无额外上限 | 1,201.12 | 301.39 / 2,583.22 | 21.48 |
| W4 C32 | 6 | 无额外上限 | 1,212.84 | 274.23 / 2,554.60 | 21.46 |
| W4 C32 | 6 | 自适应允许升档 | 1,134.73 | 908.42 / 2,923.01 | 19.32 |
| W4 C32 | 5 | 无额外上限 | 1,212.47 | 274.84 / 2,547.57 | 21.48 |
| W4 C32 | 5 | 自适应下限 512、允许升档 | 1,192.99 | 461.46 / 2,609.51 | 20.95 |
| W7 C32 | 1 | 固定 128 | 3,795.28 | 60.96 / 289.36 | 8.19 |
| W7 C32 | 1 | 固定 512 | 3,896.67 | 63.44 / 211.46 | 8.00 |
| W7 C32 | 1 | 无额外上限 | 3,920.34 | 73.93 / 266.51 | 7.92 |
| W7 C32 | 5 | 无额外上限 | 3,911.99 | 70.41 / 262.96 | 7.94 |
| W7 C32 | 5 | 自适应只填原档 | 3,874.99 | 64.12 / 209.91 | 8.06 |
| W7 C32 | 5 | 自适应允许升档 | 3,906.34 | 60.88 / 220.24 | 8.03 |

W7 用 GPU 5 做独立同服务四臂复测。

| 配置 | 吞吐三波中位数（tok/s） | TTFT P50/P95（ms） | 相对无额外上限同轮吞吐比中位数 |
| --- | ---: | ---: | ---: |
| 无额外上限 | 3,910.82 | 72.30 / 265.36 | 基线 |
| 固定 512 | 3,891.43 | 66.91 / 211.38 | −0.45% |
| 自适应下限 128、允许升档 | 3,908.19 | 62.21 / 218.77 | −0.08% |
| 自适应下限 512、允许升档 | 3,917.43 | 62.94 / 222.38 | +0.03% |

自适应下限 512 的三波为 3,917.43、3,921.26、3,854.37 tok/s。同轮中位差很小，
整波吞吐尚无可重复的提升证据。四臂的 P95 以固定 512 最低。相对无额外上限的三
个其他臂，各自首 token 一致 192/192，完整输出一致分别为 24、33、40 /192。

W4 的三次允许升档均低于同卡无额外上限的最慢波，TTFT P50 更高。W7 允许升档的
吞吐相对同卡无额外上限约 −0.14%，首 token P95 低 16.2%；严格原档吞吐约 −0.95%，
首 token P95 低 20.2%。W7 三臂每波均生成 32,768 token。分别和无额外上限比，
严格原档/允许升档首 token 都一致 192/192，完整输出一致 24/192、33/192。
W4 下限提高到 512 后，同轮吞吐比中位降幅收窄到 1.85%，三对仍均低于无额外上限。

[W3 静态三臂](../artifacts/adaptive-batch-20261127/static/gpu3/W3/comparison.json) ·
[W3 自适应同卡复测](../artifacts/adaptive-batch-20261127/adaptive-w3/gpu1/W3/comparison.json) ·
[W4 静态三臂](../artifacts/adaptive-batch-20261127/static/gpu1/W4/comparison.json) ·
[W4 自适应同卡](../artifacts/adaptive-batch-20261127/adaptive/gpu6/W4/comparison.json) ·
[W4 下限 512 同卡](../artifacts/adaptive-batch-20261127/adaptive-w4-floor512/gpu5/W4/comparison.json) ·
[W7 静态三臂](../artifacts/adaptive-batch-20261127/static/gpu1/W7/comparison.json) ·
[W7 自适应同卡](../artifacts/adaptive-batch-20261127/adaptive/gpu5/W7/comparison.json) ·
[W7 同卡四臂复测](../artifacts/adaptive-batch-20261127/adaptive-w7-four-way/gpu5/W7/comparison.json)

W4 同服务结构波复现 CPU 的含 prefill 计划数 1,315→1,310，并测出了整波代价。
两臂 64 条请求的逐位置与轮次校验通过，全部物理 forward 回放 CUDA Graph。

| W4 结构波 | 无额外上限 | 自适应允许升档 |
| --- | ---: | ---: |
| 纯 prefill 物理 forward | 47 | 167 |
| prefill/decode 混批物理 forward | 1,268 | 1,143 |
| 纯 decode 物理 forward | 679 | 822 |
| 物理 forward 合计 | 1,994 | 2,132 |
| CUDA 图启动数 | 14,618 | 14,708 |
| 纯 decode CORE 实际行 | 5,121 | 8,101 |
| PURE + mixed 的 decode CORE 实际总行 | 18,288 | 18,288 |
| 纯 decode 模型事件时间（ms） | 856.03 | 1,165.32 |
| 整波设备流跨度（ms） | 7,790.77 | 8,099.59 |

较小的请求级 chunk 更均匀地推进许多长 prompt。首条请求从第 20 个 forward 的
CODA 采出首 token 改为第 125 个。纯 prefill 批增加 120，P/D 混批减少 125，
多出的 143 个纯 decode 提交构成净增的 138 个物理 forward。纯 decode 的 CORE
执行行增加 2,980，全部来自原先与 prefill 同批的行。含 prefill 的物理行和 padding
下降，算子阶段之间的混批机会变化决定了这次整波成本。
[W4 无额外上限执行结构](../artifacts/adaptive-batch-20261127/structure/gpu6/W4/uncapped/structural-summary.json) ·
[W4 允许升档执行结构](../artifacts/adaptive-batch-20261127/structure/gpu6/W4/adaptive-grow/structural-summary.json)

W4 的 64 个 prompt 从 323 段变成 2,017 段。续入 32 条的入队至首 PRELUDE
中位数相近，147.02→141.88 ms，首 PRELUDE 至首采样 CODA 的设备区间中位数
133.78→866.61 ms。纯公平分配让同一请求反复跨批等待，影响首次交付。对同一个
B1024 混合 CORE 图档，prefill 请求数平均从 1.23 到 7.74，full-attention 的
真实 query-key 对数均值从 213 万到 217 万，模型事件均值从 5.76 到 5.99 ms。
同图宽的成员组成和有效 KV 历史变化可改变实耗。用原配置按 `op × 轮次 × 路径 × 图档`
的中位成本回算候选，预测整波模型事件 +132.0 ms，实测 +404.3 ms。方向正确，幅度
低估集中在混批路径。该回算的候选只有 5 批缺同档样本，对用时影响取最近档近似。
[W4 逐请求时间](../artifacts/adaptive-batch-20261127/structure/gpu6/W4/request-timing.json) ·
[分路径定价与同 B1024 的组成](../artifacts/adaptive-batch-20261127/historical-costs.json)

W7 同卡结构波中，允许升档的含 prefill forward 为 114→110，padding 为
9,803→1,131 行。两臂均有 5,085 个纯 decode forward，占全波提交的约 98%。
含 prefill 的模型事件为 522.45→450.65 ms，纯 decode 模型事件为
7,548.88→7,566.72 ms。模型事件合计少约 54 ms，但同一 forward 内模型之外的设备
事件区间多约 78 ms，整波设备流为 8,755.47→8,784.44 ms。因此少量 prefill 提交和
padding 可以改善首 token，整波设备吞吐仍受生成阶段和提交准备约束。
首批 32 条的入队至首次 PRELUDE 中位数 171.05→39.17 ms，TTFT P95 342.74→281.84 ms；
续入 32 条的 TTFT P95 为 92.58→136.64 ms。首批短请求获得较早处理机会，但最长
的首条 2,505-token 请求从 3 段增到 10 段，TTFT 135.22→241.87 ms。
[W7 无额外上限执行结构](../artifacts/adaptive-batch-20261127/structure-w7/gpu6/W7/uncapped/structural-summary.json) ·
[W7 允许升档执行结构](../artifacts/adaptive-batch-20261127/structure-w7/gpu6/W7/adaptive-grow/structural-summary.json) ·
[W7 逐请求时间](../artifacts/adaptive-batch-20261127/structure-w7/gpu6/W7/request-timing.json)

#### 调度策略选择

当前可验证的默认候选沿用无额外 chunk 上限。W3 与 W4 的正式数据均支持这个选择。
在 W7 追求 TTFT P95 时，固定 512 是同卡测试的最低值，吞吐只少约 0.45%。自适应仍
保留在实验入口。以同卡原策略为参照，未达到整波吞吐可重复提升，W7 还需定位跨批
输出差异。实验收齐 69 个正式计时波、4 个独立结构波。

下一版的比较对象应包括原配置生成的批。只在实际存在填充和减少提交机会时，再看
一个按当前图补齐的批以及一个允许升档的批。比较本次和紧随其后的 PRELUDE、CORE、
CODA 的物理提交，含后续独立的 decode。要求请求的首 token 和下一个生成 token 的
等待可接受，再按实测成本选批。对同图档里的批，给将完成 prompt 或能继续原有 P/D
物理合批的工作优先权。相同形状的成本用历史长度、请求数和 metadata 复用进一步区分。
Worker 暖机后报告各 op 实际捕获的图和可执行路径，Scheduler 通过已有 KV 分配器
约束真实新增行，预算保留给在途请求继续生成。

[历史物理批重算](../artifacts/adaptive-batch-20261127/historical-costs.json) ·
[调度重放](../artifacts/adaptive-batch-20261127/scheduler-replay.json) ·
[压缩 KV 池重放](../artifacts/adaptive-batch-20261127/scheduler-replay-kv-pressure.json) ·
[vLLM pin 的公平上限实现](/home/zjy/code/david/tmp/inkling2-mini-perf-20261123/vllm-source/vllm/v1/core/sched/scheduler.py#L761) ·
[自适应实验候选](../artifacts/adaptive-batch-20261127/adaptive_prefill.py) ·
[同服务测试入口](../artifacts/adaptive-batch-20261127/measure_chunks.py)

## KV 代码入口

- `core/loop_scheduler.py` 的 `_reserve_span()` 调用 vLLM `KVCacheManager` 分配块，
  先预留 span。完成最后一轮后，由 `update_from_output()` 推进已计算位置并发布
  prefix cache。抢占和请求结束也通过这个 manager 释放块。
- `worker/loop_model_runner.py` 的 `prepare_attn()` 调用 MRv2 gather 块表并构造
  slot mapping，同一 decode 批在多个 op 间保留这些结果。`_submit_forward()`
  管理异步提交下 UVA staging 缓冲的可复用时机。
- `models/inkling_plugin.py` 把 `LoopSpan.depth` 交给 Inkling 的对应 pass，原生
  attention 根据 pass 选择 KV tensor。`models/inkling_mixed_core.py` 管混深度
  CORE 的分组和每组可更新的 slot。
- `models/ouro_plugin.py` 决定早退后需要补写的 pass，`worker/kv_updates.py`
  调用 vLLM 的 update op 写入目标 slot，并提供供 Ouro 混深度调用的 slot 掩码。

## Loop 调度策略规格

这是下一版 Scheduler 的设计。每条请求按输入位置划分为多块，每一块称为一个
chunk。调度器用就绪队列决定这一拍做什么，从队列中选出能在 Worker 上一次执行的
工作，在可用容量内尽量打满。首 token、生成中的 token 和等待过久的工作都有明确的
优先级。同步和异步调度共用这套选批规则。

### 请求、chunk 和队列

请求持有到达顺序、请求优先级、已提交的 token 位置、输出进度和 KV 生命周期。每个
chunk 记录 `request_id`、本请求内从 `0` 开始递增的 `chunk_id`、token 位置和长度、下一阶段、
下一轮、已完成的轮次，以及进入就绪状态的时间和跳过次数。跨请求比较用 FCFS 或
请求优先级。同一请求内，能执行的工作优先取 `chunk_id` 较小的块。

例如一条 300-token 请求，单段上限为 128，可以依次准备第 1–128、129–256、257–300
个 token，对应的 `chunk_id` 为 `0`、`1`、`2`。实际段长按剩余输入和本拍可分配容量确定。
生成阶段，每个已确定可继续计算的 token 提供自己的下一阶段工作。Prefill 的最后
一块在交付首 token 前仍属于 prefill。

| 就绪队列 | 入队条件 | 执行后怎么推进 |
| --- | --- | --- |
| `PRELUDE` | 新请求能准入，或已有请求有下一块可准备。Prefill 准入同时检查下游窗口、KV、中间状态槽和前置依赖。 | 准备输入。满足第 1 轮依赖的块进入 `CORE[1]`，其余块等对应 KV 就绪。 |
| `CORE[轮次]` | 这块的输入或上一轮输出已就绪，同一请求中更早位置所需的本轮 KV 已就绪。 | 结果进入下一轮候选。达到该请求的目标轮数或模型确定早退后，进入可提交位置的检查。 |
| `CODA` | 这一块做完实际需要的 CORE，且它的起点已轮到该请求提交。 | 按 token 位置推进请求的提交进度。到采样边界时交付 token，随后暴露下一块或下一个 decode token 的候选。 |

队列是请求进度的索引。同一块的同一步工作只占一个就绪位置，提交后移入在途状态。
异步调度可使用执行器已经保证的顺序暴露后续候选。跨块 KV 读取和早退结果仍须满足
各自的就绪条件。在途回执按请求和提交归属更新。请求结束、取消或抢占时，按现有
生命周期回收资源，从有效的已提交位置恢复排队。模型只执行自身支持的阶段。Ouro
从 CORE 开始，达到采样边界时按模型的 CORE 输出契约交付。

### PRELUDE 准入窗口

用一个全局窗口限制**已经提交 PRELUDE、后续 CORE 还没有完成的 prefill 工作量**。
新请求的首块和已有请求的后续块共用窗口。提交 PRELUDE 时按 `真实行数 × 剩余轮数`
占用额度，在途 PRELUDE 也计入。每完成一轮 CORE，归还实际完成的行数额度。确认
早退、取消或抢占后，归还已取消的剩余工作额度。按提交归属更新，每份额度归还一次。
CODA 继续在可以安全回收时释放中间状态槽。新增 prefill 块同时满足窗口、KV 和状态槽
约束，同批 decode 按自身的执行容量检查。

**先检查准入，再应用优先级。** 窗口满时，新的 prefill PRELUDE 留在等待队列，选择
已就绪的 CORE、CODA 或 decode 继续执行。受窗口限制的块暂停累计跳过次数，保留原来
的请求顺序和计数。额度足以准备一批时，按 FCFS 或 priority 恢复批量准入。下游暂时
没有可推进的工作且仍有额度时，可以用可准入的块组较小的 PRELUDE，继续推动依赖。

用两批满 PRELUDE 演示。每批 1,024 行，固定三轮，窗口示例为 6,144 行待执行 CORE。
两批 PRELUDE 提交后窗口占满。即使同时来了大量请求，也先消化下游。CORE 累计完成
3,072 行后，窗口又能容纳一批 1,024 行的三轮 prefill，仍需检查 KV 和状态槽。窗口
大小按 W7 的满批率、待执行 CORE 行数和 TTFT 测量确定，6,144 是示例值。

### 决策顺序和防饥饿

每一拍按下表从上往下，找到第一个有可执行工作的级别。该级别决定本拍必须推进的
工作，后续组批可以用剩余容量加入兼容的其他工作。

| 级别 | 工作 | 同级的选法 |
| --- | --- | --- |
| 1 | 累计被跳过两次的就绪工作，覆盖所有阶段及 decode、prefill | 谁等得最久谁先。若等待时间相同，按请求到达顺序和请求内 `chunk_id` 排。 |
| 2 | 能交付下一个 token 的 decode CODA | 先交付已完成计算的生成 token。 |
| 3 | 就绪的 decode CORE | 给生成中的请求推进一轮，根据 CORE 的批容量和轮次规则选批。 |
| 4 | 就绪的 decode PRELUDE | 准备下一个可计算的生成 token。 |
| 5 | 能交付首 token 的 prefill CODA | 把完成输入计算的请求送到首 token 交付。 |
| 6 | 其余 prefill PRELUDE、CORE、CODA | 优先可直接发的满批。都无法发满时推进已就绪的工作。 |

`fcfs` 用请求到达顺序排列同级候选。`priority` 先选请求配置的更高优先级，同优先级
再按到达顺序。两种模式共用阶段和防饥饿优先级。`priority` 模式在相同请求优先级内
比较满批，选定本拍操作后仍可用其他优先级的兼容候选补空位。

在同一级别、同等请求优先级内，先找可直接发满的候选，其次找跨轮补完后能一次发满
的 CORE，最后选可执行的未满批。多个 CORE 在同一档时选轮次更靠后的。同操作、
同轮次按队列顺序选。多个不同阶段给出同一档候选时，以候选队首的等待时间破平。
等待时间相同，按 CODA、CORE、PRELUDE 排。

Decode 选中的批还有空位时，可以先推进一批能在现有等待额度内追到同一阶段的 decode，
随后合成一次提交。合批判断受真实槽位容量约束，已满的批和已经提权的工作按表中规则
执行。

**防饥饿上限为两次可调度机会。** 一块输入通过准入检查、具备执行条件，而且能放进
一个合法的物理批，本拍因为选择了别的阶段或预算分给其他工作而没入批，记一次。每个
调度决策最多记一次。等依赖、KV、状态槽或 PRELUDE 窗口时暂停计数。第二次被跳过
后，从下一拍进入级别 1。提交的部分清零；如果一次只提交块的一部分，尚未提交的
位置继承原等待顺序和次数。多个不同阶段同时到上限，由最老的就绪工作决定本拍，
其余到上限的工作继续保留最高级别。`2` 是初始策略值，按后续数据验证。

### 同一拍怎样打满

每拍只选一种操作。选批按同一份就绪状态计算候选，同时检查真实 token 行数、Worker
查询槽位、活跃请求数、可用 KV 和中间状态槽。满批按实际执行容量判断。纯 decode
使用其 FULL 图和查询槽位，prefill 与混合批使用实际支持的执行路径和图档。Padding
行只计成本。多个 Worker 提交分别计批。确定候选后再预留需要新增的资源，提交多少
就推进多少。暂时放不下的工作保留原队列位置和等待状态。

**PRELUDE 按准入顺序准备足。** 先让每个符合条件的请求拿到一块，继续按同样的顺序
轮转补可准备的块，直到实际批容量、准入额度用完或候选耗尽。已开始 CORE 的请求可以
在窗口、状态槽和 KV 允许时准备下一块。短请求到达后排队，通过准入后随选中的
PRELUDE 与其他块一起准备。当其他操作已按优先级选中时，短请求继续等自己的机会。

**CORE 先保住已经能一次发出的同轮满批。** 如果没有更高优先级的工作，先从能发满的
同轮队列选批，其中轮次靠后的优先。选中的同轮批有余量时，用其他轮次的就绪块补，
前提是 Worker 支持这组请求和轮次在一次物理提交中执行。已经能发满的另一轮保留自己的
整批。需要先服务更紧迫的工作时，先选它，再从可用候选补，必要时提交小批。队首块
放不下时继续扫描后面的就绪块，把放得下的补进来，队首保留等待次数。若要利用最后
一小段空位，可在 Worker 已支持正确切分和回执的路径上拆块。余下的位置保留就绪状态
与 KV 依赖，并把切分后的 CORE 和 CODA 次数一起算入收益。

**CODA 只从每个请求当前可提交的位置取工作。** 先选本拍最应该交付的 token，再把同
阶段、相容的其他请求合入。长输入的中间块完成 CODA 会推进提交位置，并释放可以
安全回收的中间状态。它可以随已有 CODA 补批，也可以靠满批或等待上限取得一拍。
普通 prefill CODA 仍遵循这两条规则。合入 prefill 前要按 Worker 实际路径核算容量，
已经选中的满 FULL decode 批保持整批执行。

同一个请求允许多块输入同时就绪，跨轮工作按 CORE 规则补入真实空位。这两项就是
WCPB 在调度上的能力。它们使用相同的三阶段队列、优先级和计数。Worker 根据选批结果
选择同轮、跨轮、PURE decode 或混合执行。已经准备好的 chunk 保留输入状态，只在
自身依赖满足后进入相应 CORE 队列。

### 逐拍例子

以下 CORE 例子用 1,024 行预算，除特别说明外，候选都已就绪、优先级相同，Worker
可以把所选成员一次执行。

| 当前候选 | 本拍选择 | 留到后面 |
| --- | --- | --- |
| CORE 第 1 轮 1,024 行，第 3 轮 256 行 | 第 1 轮一次发满 1,024 行。 | 第 3 轮的 256 行保留原进度，记录一次跳过。 |
| CORE 第 1 轮 768 行，第 3 轮 256 行 | 先取第 3 轮，用第 1 轮补到 1,024 行。 | 本批取走的块按各自轮次更新。 |
| CORE 已选 896 行，队首块 192 行，后面有兼容的 128 行 | 用后面的 128 行打满。 | 192 行的块保留原位置，记录一次跳过。 |
| 有一个满 CORE 批，一条新的 8-token 请求可以准入 | 发满 CORE。 | 8-token 请求进 PRELUDE 队列，在后续机会补入。当它累计被跳过两次后，升到最高级别。 |
| A 的后续 128 行已选中 PRELUDE，同时有 B 的 8 行、C 的 64 行能入批 | A、B、C 的 200 行共用一次 PRELUDE，继续按准入顺序填剩余容量。 | 各块根据实际 KV 依赖进入 CORE 队列。 |
| B 能通过 decode CODA 交 token，A 只有中间块 CODA，C 有满 prefill CORE | 先发 B 的 CODA。A 能进实际同批时可以补入。 | C 保留在 CORE 队列。A 和 C 若具备执行条件却被跳过，分别记一次。 |
| A 的 prefill chunk 累计已被跳过两次，还有 decode CORE 就绪 | A 取得最高级别，选它所在的阶段，再补同阶段的兼容工作。 | 仍有其他到上限的工作时，保留其最高级别，后续按等待时间推进。 |
| A 先到，B 后到但请求优先级更高，两者都在同轮，且本拍只能选一块 | `fcfs` 选 A，`priority` 选 B。 | 落选的块保留等待次数，累计两次后两种模式都将它提权。 |
| 两批三轮 prefill 已占满示例窗口，又同时到达大量请求 | 继续消化已就绪的 CORE 和 CODA，照常按优先级处理 decode。 | 新请求保留队列顺序，受窗口限制时暂停计数。腾出一批额度后批量补入。 |

### 验收

先固定请求到达顺序、预算、KV 和回执顺序，逐拍检查单轮满批、高轮优先、跨轮填空、
首 token 和 decode CODA、短请求续入、头部大块放不下，以及恰好两次跳过后提权。
同一输入在 FCFS 和 priority 下验证预期顺序，分别跑同步和异步。校验每个 token
位置完成正确的阶段和轮数，提交进度连续，取消、抢占和晚到回执的资源归属正确。

再在同配置 W7 中观察满 prefill 批、确有余量的 prefill 批、纯 decode 满批、持续到达
和集中到达。检查在途 PRELUDE 的额度已经占用，突发时待执行 CORE 行数维持在窗口内，
窗口满后的跳过计数暂停，完成 CORE 和释放状态槽后能够恢复批量准入。
同时记录调度计划、实际物理 PRELUDE/CORE/CODA 提交、图档、真实行与 padding、首
token 等待以及两次跳过的提权情况。性能以无插桩的吞吐和 TTFT/TPOT/ITL 为准。结构
验收重点看跨轮补批吸收了哪些原本需要单独提交的工作，以及总提交是否下降。正确性
验收记录输出长度、结束原因、逐 token 差异及对照组自身重复波的差异。

### 统一调度的实现与验证

`LoopScheduler` 在同一份请求进度上生成 PRELUDE、分轮 CORE 和 CODA 的就绪队列。
首段和续段共用同一次选批，Worker 按实际位置与轮次决定执行布局。`LoopAsyncScheduler`
继续继承选批流程，沿用可安全提前提交的回执规则。原 WCPB 两个启动选项已移除，
多块预备和跨轮补批由统一调度及设备执行能力共同决定。原有的 decode 抢先执行和
延迟采样开关由固定优先级替代。显式自投机流水保留其 draft 后继与当前 token 的依赖。

沿用 `--scheduling-policy fcfs` 或 `--scheduling-policy priority`。`--rlt-prefill-chunk-size`
可设单段上限，与 vLLM 的 `long_prefill_token_threshold` 一起取有效上限。初始准入窗口
取两批等值的 CORE 行数，`--rlt-prefill-window` 可覆盖。W1、W3、W7 对照统一使用
1,024 行预算且普通 prefill 阈值不额外限制，此时固定三轮、RLT 默认窗口为 6,144 行。

固定 128 条同时到达的 CPU 回放覆盖同步与异步、FCFS 与 priority 四种组合，混用请求
级 1、2、3 轮和不同输入输出长度。四组均完整结束。在 96 行待执行 CORE 的测试窗口下，
峰值分别为 87、88、96、96 行，异步组也包含已提交未回执的工作。

首次 W7 预热在混合跨轮 CORE 遇到普通 decode 的 GPU 状态槽。该状态在单轮 decode
里按请求槽位保存，混合 CORE 原先只接受独立的张量状态。现通过模型适配器取出状态槽
对应行，复制到独立生命周期后再执行，避免后续 decode 复用槽位覆盖仍需保留的输入。
CPU 回归用同一个混合计划验证读出的 decode 行独立于槽复用，并保留原 prefill 状态。

插件测试 112 项通过、3 项跳过。GPU 7 上使用 Inkling2-mini、固定三轮、TP1、FCFS、
异步、1,024 行预算、80 GiB KV，FULL 与 PIECEWISE 捕捉档一致。W1 和 W3 按标准
C1 执行，W7 为 C32。每组预热后跑三轮正式波，表中取中位数。

| 场景 | vLLM（tok/s） | RLT（tok/s） | RLT/vLLM | TTFT P95，vLLM → RLT（ms） | TPOT P95，vLLM → RLT（ms） | ITL P95，vLLM → RLT（ms） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| W1 C1 | 261.31 | 258.60 | 98.96% | 21.68 → 14.95 | 3.75 → 3.80 | 3.84 → 3.87 |
| W3 C1 | 229.00 | 226.06 | 98.72% | 83.81 → 84.66 | 3.77 → 3.82 | 3.86 → 3.90 |
| W7 C32 | 3,738.68 | 2,331.65 | 62.37% | 256.84 → 226.07 | 8.42 → 14.12 | 8.58 → 13.99 |

W1、W3 正式波使用 `source-unified-v3`，W7 使用修正混合 CORE decode 槽后的
`source-unified-v4`。W1 三轮整体逐 token 一致。W3 的三轮与原生配对时，完整一致为
54、53、55 /64，首 token 一致为 61、60、60 /64。W7 完整一致为 4、6、2 /64，首
token 一致为 57、59、59 /64。原生 W3 重复波完整一致，RLT W3 重复波在 54–56 /64。
W7 原生重复波为 55–60 /64，RLT 为 2–4 /64。W7 RLT 第三波有一个请求在 495 token
结束，该波生成 32,751 token，其余两波与原生均为 32,768。原生 W7 第二波的慢点保留
在中位数统计中，其吞吐为 2,768.84 tok/s。W3 和 W7 的输出逐 token 一致性未通过。

独立 W7 结构波通过全部 64 条请求的各阶段、轮次 token 计数。Worker 实际提交 PRELUDE
2,072 次、CORE 6,200 次、CODA 2,070 次。其中纯 decode CORE 有 6,113 批，平均
15.98 个真实行，中位数 9 行；纯 decode CODA 有 2,040 批，同样中位数 9 行。纯 decode
CORE 的主要真实批大小为 1、9、23、31 行，各出现约 1,520 次。小批持续到了生成阶段，
与正式波 TPOT 和 ITL 劣化一致。结构波有 4,210 次计划选入提权工作，单波记录无法分开
提权和阶段优先级各自对小批的影响。14 个跨轮 CORE 批全部实际重放了对应 CUDA Graph。

修复版的独立 W3 结构波验证 8 条请求。观察到 24 次提前准备下一块，跨轮 CORE、packed
CORE 和对应图重放均为零。这 8 条请求没有经过 W7 需要修复的混合 CORE 路径。
W1、W3 的正式性能和 W7 的新修复版结构波仍按各自的冻结源码记录。

[GPU 7 同卡完整对比](../artifacts/unified-scheduler-20261127/comparison.md) ·
[vLLM 与 RLT 的 W1/W3 原始记录](../artifacts/unified-scheduler-20261127/formal-gpu7) ·
[修复后的 W7 原始记录](../artifacts/unified-scheduler-20261127/formal-v4-gpu7/rlt-basic) ·
[修复版 W7 结构](../artifacts/unified-scheduler-20261127/structure-v4-gpu7/W7-summary.json) ·
[修复版 W3 结构](../artifacts/unified-scheduler-20261127/structure-v4-gpu7/W3-summary.json)

### 防止 W7 生成阶段长期拆批

上一版按当前阶段的优先级立即发送就绪 decode。若少量请求先进入 CODA，便先完成并
开始各自的下一 token。另一组仍在 CORE，两组在不同阶段循环，后面的批持续拆开。
W7 结构波的纯 decode CORE 中，1、9、23、31 行的批各重复约 1,520 次。

当选中的 decode 批有空位时，检查同轮数、单 token 的另一组就绪 decode。从其当前阶段
到同一目标阶段若只需一到两拍，并且等待的队列成员都还有对应的跳过额度，先推进能
追上的批。到达同一阶段后一起提交。满批、已经累计跳过两次的工作，以及自投机的
独立依赖继续按各自规则执行。单批上限仍来自真实 token 预算和 Worker 查询槽。
在 `priority` 模式下，只允许同级或更高请求优先级的候选触发追齐。

回归用例以 1 条即将交付 token 的请求和 3 条还差一轮或两轮的请求构造。剩余等待额度
足够时，两组在两次内汇成一个 CODA 批，随后连续保持同批。额度不足时当拍交付。
同步和异步均覆盖。

同配置 GPU 7 的 W7 C32 结构波，每条请求的 PRELUDE、逐轮 CORE 和 CODA token 数
均通过检查。纯 decode CORE 提交由 6,113 降到 3,067，真实批大小中位数从 9 回到 32，
其中 3,044 批有 32 个真实行。纯 decode CODA 由 2,040 降到 1,022。带有提权工作的
计划由 4,210 降到 127。两次结构波独立运行，统计 Worker 的实际物理提交。

| W7 C32，中位数 | 原生 vLLM | 修复前 | 修复后 |
| --- | ---: | ---: | ---: |
| 吞吐（tok/s） | 3,738.68 | 2,331.65 | 3,610.90 |
| TTFT P95（ms） | 256.84 | 226.07 | 383.10 |
| TPOT P95（ms） | 8.42 | 14.12 | 8.84 |

修复后三轮分别为 2,856.83、3,693.10、3,610.90 tok/s，吞吐中位数达到原生的 96.58%，
较修复前提高 54.86%。RLT 和原生各有一次观测到 GPU 长间隔的慢波，均保留在统计中。
修复后第一波比其余两波少输出 4 个 token。输出全序列与原生逐波完全一致为 5、5、4
/64，首 token 一致为 60、58、59 /64。逐 token 一致性仍未通过。插件测试通过 118 项，
跳过 3 项。

[W7 修复后正式对比](../artifacts/w7-decode-realign-20261127/comparison.json) ·
[W7 修复后结构波](../artifacts/w7-decode-realign-20261127/tagged/W7-summary.json)

### 请求级动态 loop 的吞吐复测

以 `0582bb7` 为基准，W7 的两个正式组均使用下文的隔离修补副本。每个场景的固定组
和动态组在同一张卡、同一个 RLT 服务内交替跑，
两组都完成预热后各采三轮正式数据，取各项指标的三轮中位数。Inkling2-mini 使用 TP1、
FCFS、异步调度、mixed batch、1,024 token 调度预算、80 GiB KV 及相同的 FULL 与
PIECEWISE 图档。W1 C1 与 W3 C1 各有 64 条请求，分别输入 128 和 4,096 token，均固定
输出 128 token。动态组各有 16 条请求指定 1、2、3 轮以及使用默认值。默认值为 3，
prefill 始终走 3 轮，请求指定轮数从第二个生成 token 开始生效。

| 场景 | 固定 3 轮（tok/s） | 动态 1/2/3 轮（tok/s） | 吞吐变化 | TTFT P95（ms） | TPOT P50 / P95（ms） | ITL P95（ms） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| W1 C1，GPU 2 | 257.41 | 326.75 | +26.94% | 13.95 → 13.33 | 3.81 / 3.82 → 3.26 / 3.82 | 3.88 → 3.87 |
| W3 C1，GPU 3 | 225.63 | 277.75 | +23.10% | 84.47 → 83.19 | 3.82 / 3.83 → 3.27 / 3.82 | 3.88 → 3.86 |
| W7 C32，GPU 7，隔离修补副本 | 3,645.15 | 1,723.81 | −52.71% | 223.93 → 232.99 | 8.46 / 8.64 → 16.14 / 23.76 | 8.41 → 25.22 |

W1 和 W3 的两组每轮均生成 8,192 token。按实际生成长度和请求上限计算，decode CORE 从每波
24,384 行降到 18,288 行，减少 25%。W1 动态组 1、2、3 轮请求的 TPOT P50 分别为
1.60、2.71、3.81 ms，W3 为 1.60、2.71、3.82 ms。保留三轮的请求仍位于尾部，
因此整体 TPOT P95 近乎持平。独立结构波分别抽取 W1 的 8 条、W3 的 4 条请求，验证了
prefill 的三轮和每个 decode token 的请求指定轮数。W1 的 32 条三轮请求三波均与对应的
固定组逐 token 一致。W3 对应三波均为 25/32 一致，与固定组自身的重复性接近。低轮数
允许生成内容变化，此处测量运行性能。

GPU 7 的 W7 C32 在 `0582bb7` 上完成了固定三轮预热。动态组首轮预热在 prefill 与
decode 同批提交 CODA 时退出。独立诊断复现后发现，packed CORE 覆盖了单请求的状态，
但这些行仍指向前一轮共享的 FP32 状态，CODA 合批时拿到了旧状态。隔离副本清除这处
已失效的引用，用于复测。`0582bb7` 的 W7 动态吞吐无有效结果。

隔离副本按同一组 64 条 ShareGPT 请求测试，每条自然结束或至多生成 512 token。
两组的六轮正式波都生成 32,768 token，均触及长度上限。固定组三轮 2,272.77、
3,645.15、3,700.74 tok/s，动态组三轮 1,730.66、1,723.81、1,176.70 tok/s。
采样期间 GPU 7 只看到测试进程，SM 时钟为 2,032 MHz。保留全部波次，包括慢波。
固定组与动态组的三轮子集逐 token 一致率偏低，固定组重复波的一致率也偏低。

W7 的独立动态结构波逐请求通过轮数校验。19,446 个 prompt token 各执行三轮，decode CORE
各轮的真实行数为 32,704、24,528、16,352，合计 73,584 行，比固定三轮的 98,112 行少
25%。与上节同卡同负载的固定三轮结构波相比，纯 decode CORE 物理提交由 3,067 升至
7,467，批大小中位数从 32 降到 8。CODA 从 1,022 升至 5,398，批大小中位数从 32 降到
4。调度按阶段与轮次建立就绪队列，纯 decode 的不同轮次形成小批，固定深度的追批逻辑也
只合并总轮数相同的请求。另有 43 个带 prefill 的跨轮 packed CORE 批，43 个均重放了图。

两次单独记录执行批大小的测试中，PRELUDE、CORE、CODA 总提交从 5,244 增至 15,698，
新增的 10,454 次中有 10,423 次是纯 decode。固定组三个阶段各有超过 98% 的纯 decode
批达到 32 条，动态组三个阶段都没有 32 条的纯 decode 批。实际补齐后的纯 decode CORE
行数为 97,848 → 89,725，只减少了 8.30%。出现两次被跳过后提权的批为 127 → 13,383。
纯 decode 的跨轮 CORE 在本次配置下不会合批，CODA 就绪队列仍按上一轮 CORE 的轮次
分开。调度批数和 Worker 实际提交数一致，拆分在 Worker 执行前已发生。

使用同一调度器做只运行 CPU 的小例子。8 条同长请求一起到达，每条生成 10 token，
固定三轮的纯 decode 有 45 批，动态组按 2 条一轮、2 条两轮、4 条三轮运行，共 78 批。
第一轮 CORE 是 8 条一起跑，随后先提交两条一轮请求的 CODA，再提交另外六条的第二轮
CORE。两条两轮请求完成 CODA 后，这四条已经开始下一 token 的 PRELUDE，余下四条才
做第三轮 CORE。CPU 例子里临时让所有已就绪的 CODA 进入同一队列，总批数降到 73。
不同请求抵达 CODA 的时间依然错开，合并队列只能处理已经同时就绪的部分。

[W1 原始记录和汇总](../artifacts/dynamic-loop-0582bb7/w1/summary.json) ·
[W1 轮数验证](../artifacts/dynamic-loop-0582bb7/w1/tagged/W1-summary.json) ·
[W3 数据汇总](../artifacts/dynamic-loop-0582bb7/w3/summary.md) ·
[W3 轮数验证](../artifacts/dynamic-loop-0582bb7/w3/diagnostic/depth-validation.json) ·
[W7 原版错误](../artifacts/dynamic-loop-0582bb7/w7/formal/rlt-basic/server.log) ·
[W7 状态诊断](../artifacts/dynamic-loop-0582bb7/w7/diagnostic/rlt-basic/server.log) ·
[W7 隔离修补副本的正式数据](../artifacts/dynamic-loop-0582bb7/w7/repaired-formal-summary.json) ·
[W7 动态轮数与物理批验证](../artifacts/dynamic-loop-0582bb7/w7/repaired-structure/W7-summary.json)

### 实验记录

#### 从最后一轮 CORE 向前补批

实验版本在有指定较短 loop 的 decode 请求就绪时，由未满的最后一轮 CORE 或 CODA 发起
跨轮追批，按每条已就绪请求剩余等待次数检查前面可以推进的 CORE 轮次。
以能补入的请求数最多、需额外执行的轮次最少的组合为先，从最前面的必要轮次开始推进。
每次只提交一个已有就绪批，下次用最新的就绪状态重选。已经凑满的批立即提交，已达到
等待上限的请求沿用防饥饿优先级。三轮模型最多跳过两次提交，模型有多轮 CORE 与 CODA
时按模型轮数减一计算上限。不同请求最后执行的 CORE 轮次可以不同，正常 CODA 共用
就绪队列，每条请求仍保留自己的最后一轮标记和模型状态。

packed CORE 更新单条请求状态时清除它对上一轮整批状态的引用，避免 CODA 复用已失效
的 FP32 状态。CPU 用例验证了跨轮追到同一个 CORE 批、剩余等待额度不足时按时提交、
中间 CORE 不为不同总轮数的请求让出提交，以及 1、2、3 轮请求一起进 CODA。插件测试
123 项通过，3 项跳过。

中间版本在 GPU 7 上的 W7 C32 三轮正式波中，动态吞吐中位数达到 2,579.05 tok/s，
固定三轮降到 2,307.54 tok/s。单独记录执行批时，固定三轮的纯 decode CORE 提交由原版的
3,067 增到 6,128，批大小中位数从 32 降到 12。动态组的纯 decode CORE 和 CODA 分别为
4,378 和 1,578 次，较上次动态组的 7,467 和 5,398 次有减少。由于固定组也受到了新规则的
影响，最终版只在有较短轮数的 decode 就绪时启用这条规则，并单独复测三组。

[中间版本数据](../artifacts/loop-catchup-retest/summary.json) ·
[中间版本固定三轮的批记录](../artifacts/loop-catchup-retest/w7/tagged/fixed-trace.json) ·
[中间版本动态轮数的批记录](../artifacts/loop-catchup-retest/w7/tagged/dynamic-trace.json)

加入固定轮数保护后的实验组，W1、W3 吞吐中位数分别是 257.96 → 327.11 和
223.83 → 276.34 tok/s。W7 是 2,325.49 → 2,462.45 tok/s。固定组三轮分别为
2,309.31、3,700.53、2,325.49 tok/s，动态组为 2,224.93、2,462.45、2,500.83 tok/s。
同 GPU 0 再跑原调度器的两个固定波为 2,871.34、2,159.17 tok/s。W7 固定组波动大，
这次没有测到相对于稳定满批固定组的收益。

[三组复测与同卡固定对照](../artifacts/loop-catchup-retest/preserved/report.md)

### 同一批 decode 一起出 token

从 `0582bb7` 的调度器出发，保留 packed CORE 状态引用的修复。当 PRELUDE 选中的
decode 请求有不同的目标轮数时，记录本次提交的请求和 token 位置。这个记录到各自的
CODA 提交为止。请求结束或发生抢占时，删除已经失效的工作。下一 token 在 PRELUDE
重新选请求，新的请求可以在这一步补进来。

例如一批有 8 条跑一轮、8 条跑两轮、16 条跑三轮，三次 CORE 分别跑 32、24、16 条。
各请求最后一轮的结果保存在自己的状态中。优先选到本批时，先推进同批还需要的 CORE，
然后用一次 CODA 交付 32 个 token。批间仍由现有 FCFS 或请求优先级以及防饥饿规则选批。
多个 CORE 批的优先级与满批状态相同时，先选较深的轮次。同一批内如果实际进度不同，
从仍需执行的最早一轮开始推进。
如果某条可进 CODA 的请求剩下的等待次数不足以覆盖同批剩余 CORE，就提交已就绪的
CODA。三轮模型最多让出两次实际提交。Prefill 按现有 chunk、KV 依赖和预算调度。

CPU 用例覆盖同步、异步下的混合轮数。另验证等待两次的 prefill 可以先提交，以及
因此只剩一次等待机会的短轮请求可以先交付。
插件测试 121 项通过，3 项跳过。

沿用 Inkling2-mini、FCFS、异步、mixed batch、1,024 token 调度预算、80 GiB KV 及
FULL/PIECEWISE 图档。每个场景同卡交替跑固定三轮和动态 1/2/3/default 三对，取中位数。
GPU 7 的 W7 初跑遇到外部进程，完整三对在启动前空闲的 GPU 0 重新执行。

| 场景 | 固定 3 轮（tok/s） | 动态 1/2/3 轮（tok/s） | 吞吐变化 | TTFT P95（ms） | TPOT P95（ms） |
| --- | ---: | ---: | ---: | ---: | ---: |
| W1 C1，GPU 2 | 256.26 | 325.47 | +27.01% | 15.79 → 15.58 | 3.84 → 3.83 |
| W3 C1，GPU 3 | 225.35 | 275.25 | +22.15% | 84.90 → 85.21 | 3.83 → 3.83 |
| W7 C32，GPU 0 | 2,222.97 | 1,446.29 | −34.94% | 251.83 → 332.74 | 19.95 → 33.24 |

W1、W3 每轮返回 8,192 token。W7 固定组三轮为 3,696.67、2,186.78、2,222.97 tok/s，
动态组三轮为 1,535.18、1,446.29、1,190.93 tok/s。动态第一轮有一次 EOS，返回 32,707
token，其余五轮均返回 32,768 token。这版在 W7 上没有达到性能目标。

单独重跑 W7 并记录每次实际执行的批大小。两组 64 条请求全部通过轮数校验，prefill 各轮
都有 19,446 个真实行。动态组 decode 三轮分别为 32,704、24,528、16,352 行，合计比
固定组少 25%。纯 decode 的执行批统计如下。

| 阶段 | 固定组提交次数 | 动态组提交次数 | 固定组每批真实行数中位数 | 动态组每批真实行数中位数 |
| --- | ---: | ---: | ---: | ---: |
| PRELUDE | 1,029 | 3,632 | 32 | 8 |
| CORE | 3,064 | 9,636 | 32 | 6 |
| CODA | 1,021 | 7,910 | 32 | 2 |

PRELUDE 进入时已经出现小批。当前的记录只能维持同一次 PRELUDE 选中的请求，来自其他
提交的请求继续按普通策略运行。W7 的执行批统计说明单次 PRELUDE 内维持分组不足以把
生成阶段保持在大批。逐 token 精确一致性在固定组重复间也较低，轮数校验只检查执行轮数
和各阶段的行数。

[三组同卡对照原始汇总](../artifacts/decode-group-retest/summary.json) ·
[W7 固定组执行批](../artifacts/decode-group-retest/w7-tagged/fixed-summary.json) ·
[W7 动态组执行批](../artifacts/decode-group-retest/w7-tagged/dynamic-summary.json)

### 回退到已上传的固定版调度

调度器和对应测试已恢复到远端 `vllm-rlt-plugin2` 的 `0582bb7`。前两节的合批实验是历史
记录。运行时代码逐文件与该提交对照，唯一差异是 packed CORE 写入新状态时清除旧整批
状态引用的修复，避免 W7 动态轮数在 CODA 读取已失效的 FP32 状态。

### 按执行阶段合并 VarLoop 的就绪工作（历史实验）

基底为 `0582bb7` 加 packed CORE 状态修复。PRELUDE 和 CODA 使用同一阶段的就绪工作。
CODA 队列不再按最后一轮 CORE 拆开，请求的实际轮次仍随执行计划传递。纯 decode CORE
保留同轮满批路径。候选已进入当前最大 decode 图档时继续走普通路径，其他较小的候选
可以把已就绪的不同轮次一起交给 Worker。Worker 复用现有的跨轮 attention、卷积页路由
及分段 CUDA Graph。纯 decode 的输出按请求位置写回现有 MRv2 状态槽。不同轮次的 KV
位置和完成条件分别保留。

这一版实验默认关闭主动 decode 追赶，同轮请求仍可自然合批。CODA 可让出一次提交给直接产生
终轮结果的 CORE。需要当前 CODA 尚有预算、合并后的批能装下，并且预计的图补齐行数
不会超过分两批执行的行数。首版使用最近的纯 decode CORE 提交至回执时间估计下一步
等待，默认上限 2 ms，已有提权工作和已经为合批让路的请求立即按原优先级选批。时间
预算用于作出等待决策，实际完成时延还包含已提交的 GPU 工作。

| 对照 | `rlt_decode_joining` | `rlt_unified_coda` | `rlt_coda_max_wait_ms` | `rlt_cross_depth_decode` |
| --- | ---: | ---: | ---: | ---: |
| 原策略 | true | false | 0 | false |
| 只关闭主动追赶 | false | false | 0 | false |
| CODA 统一和单步聚合 | false | true | 2 | false |
| 再加纯 decode 跨轮 CORE | false | true | 2 | true |

四组仍使用同样的 `rlt_mixed_core`、`rlt_packed_prefill`、`rlt_merge_decode_core` 与
`rlt_decode_state_slots` 设置。新增跨轮选批要求 `rlt_mixed_core=true` 并满足已有的
Inkling 单卡、独立 KV、普通执行限制。保留原策略开关用于同条件基线。

用 W7 的 prompt 长度和 64 个输出 token 做 CPU
调度回放，所有设备回执即时返回。仅关闭主动追赶时，动态组纯 decode 的 CORE/CODA
批数为 995/762。启用统一 CODA 和单步聚合后为 809/339，再开启跨轮 CORE 后为
420/299，CORE 每批实际行数中位数达到 23。这组数据用于验证选批，性能以 GPU 结果为准。

原临时语料文件已不存在。用保留的 ShareGPT 数据和原 64 条请求 ID 重建输入，64 条
分词长度全部与旧记录一致。新语料以自己的 SHA-256 标识，四臂使用同一份输入和分布。
启动使用原 vLLM `3146a9df4`。每臂在自己的 B300 上交替测固定三轮、动态轮数各三波，
下表为输出吞吐的中位数。各臂使用不同卡，固定组也出现过较大的波间波动。

| W7 C32 首版 | 固定三轮（tok/s） | 动态轮数（tok/s） |
| --- | ---: | ---: |
| 原策略 | 2,314.48 | 1,495.07 |
| 仅关闭主动追赶 | 2,326.98 | 1,352.19 |
| CODA 统一和单步聚合 | 2,174.10 | 1,762.74 |
| 加纯 decode 跨轮 CORE | 3,683.72 | 1,693.77 |

首版独立诊断验证 64 条请求的每轮 token 数。动态组纯 decode CORE 提交 4,317 次，
其中 1,272 次跨轮 packed CORE 全部重放分段图，其余 3,045 次走普通图。固定组和动态组
的纯 decode CORE GPU 计时合计为 9.36 s 和 9.59 s。动态 CORE 单次中位数为 2.10 ms，
P95 为 3.61 ms。动态纯 decode CODA 提交从 2,037 增至 3,558 次，计时从 1.52 s 增至
3.17 s。动态组没有触发单步 CODA 等待。每次 packed CORE 的状态读取、图输入准备、
状态槽写回计时中位数分别为 0.118、0.143、0.056 ms。计时区间有嵌套，诊断耗时与
无额外记录的正式测试分别统计。

随后把纯 decode packed CORE 的终轮写回和继续计算的状态按布局分组，继续计算的行按
是否已有上一轮输入再分开。一组一起恢复 PRELUDE 状态、写回已有 MRv2 槽，避免逐条
执行这些操作。新增测试验证不同轮次和不同状态类型下的请求槽与 token 位置。CPU 插件
测试 123 项通过、3 项跳过。

[四臂正式数据](../artifacts/varloop-physical-batch-20261128/runs-v1/summary.json) ·
[首版 W7 固定组三阶段计时](../artifacts/varloop-physical-batch-20261128/runs-v1/full-tagged-profile/profile-0.json) ·
[首版 W7 动态组三阶段计时](../artifacts/varloop-physical-batch-20261128/runs-v1/full-tagged-profile/profile-1.json)

状态槽分组写回版在 GPU 0 用相同配置重跑 W7。2 ms 的动态吞吐中位数为 2,228.99 tok/s，
首版同卡为 1,693.77 tok/s。三波中位差为 +31.6%。两版的写回函数计时覆盖不同操作，
不能用它的计时差给这 31.6% 做分摊。两个版本的固定组均有波间变化。

| W7 C32，状态槽分组写回版 | 固定三轮（tok/s） | 动态轮数（tok/s） | 动态 ITL P50/P95（ms） |
| --- | ---: | ---: | ---: |
| 2 ms，GPU 0 | 3,708.97 | 2,228.99 | 9.92 / 15.95 |
| 6 ms，GPU 0 | 2,355.04 | 2,762.08 | 12.02 / 14.05 |
| 6 ms，GPU 1 | 3,696.78 | 2,752.05 | 12.38 / 13.82 |

同卡从 2 ms 改为 6 ms 后，动态吞吐提高 23.92%，ITL 中位数上升、P95 下降。6 ms 的
GPU 0 固定组三波为 2,355.04、2,303.61、3,699.53 tok/s。GPU 1 的固定组三波为
3,701.47、3,696.78、2,917.12 tok/s。固定组已观察到是否能长期满批的变化，因此以绝对
吞吐作为约束，动态组尚未达到稳定满批的固定组三轮水平。当时的默认值为 2 ms，W7 的
6 ms 结果通过 `rlt_coda_max_wait_ms=6` 显式配置取得。后续单步前瞻已移除该参数。

GPU 4 的独立诊断里，两组配置均通过逐请求轮数校验。动态组选择让 CODA 等一次 CORE 的
次数由 0 增至 1,014。纯 decode CODA 提交由 3,560 降到 2,042，每批真实行中位数由 7
增至 16。纯 decode CORE 总提交由 4,314 降到 3,060，其中跨轮 packed CORE 图重放由
1,273 降到 7。诊断说明在 6 ms 这波中，更多工作已经自然聚到同一轮，且需要单独执行的
CODA 批更少。诊断与正式波各自独立，不能把各项变化直接相加当作吞吐收益。

[状态槽分组写回版和等待预算对照](../artifacts/varloop-physical-batch-20261128/runs-v2/summary.json) ·
[2 ms 动态执行诊断](../artifacts/varloop-physical-batch-20261128/runs-v2/full-tagged-profile/profile-1.json) ·
[6 ms 动态执行诊断](../artifacts/varloop-physical-batch-20261128/runs-v2/wait6-tagged-profile-gpu4/profile-1.json)

### VarLoop 的单步前瞻（当前策略）

按 PRELUDE、共享 CORE 和 CODA 组织就绪工作。CORE 保留同轮满批的普通路径，
在现有 Inkling 跨轮执行条件下可以直接合并已就绪的不同轮次。CODA 统一选批，
仍携带每个请求实际完成的轮次，以便读取对应状态。删除主动 Decode Joining 和试验用的
`rlt_decode_joining`、`rlt_coda_max_wait_ms` 参数。原有 FCFS、请求优先级和两次跳过后的
防饥饿提权继续生效。

当前实现把前瞻用于目标轮数已确定的普通单 token decode。默认应提交 CODA 时，检查
已就绪的 CORE 候选，找出这一轮结束后直接进 CODA 的行。合并时每个请求只占一行，且
总行数同时满足现有的 `max_num_scheduled_tokens` 和 `max_num_seqs`，Worker 就会把这两组
放进一次 CODA Forward。比如已有 8 行 CODA，CORE 能直接产生 16 行，分开发需要两次
CODA Forward，合在一起只需一次。只有待合并的行包含不同目标轮数，才启用这项前瞻。
同目标轮数的请求按默认顺序提交。已有 CODA 提权时也按默认顺序提交。使用 `priority`
策略时，CORE 批首请求也不能低于 CODA 批首请求的优先级。

选中 CORE 后只记录实际提交的后继。下一次非空提交优先兑现这批 CODA，按请求和 token
位置核对状态。即使还有其他 CORE 能继续产生 CODA，也只前瞻一次。同步模式下如果后继
尚未就绪，先执行已经就绪的 CODA。若原请求已失效，则继续正常选批。Prefill、运行时早退
和投机请求仍走各自已有的就绪判定，当前不参与这条前瞻。策略只比较可确定的物理 Forward
次数，不使用 TTFT、ITL、等待时间、图档大小或 GPU 耗时估算。

CPU 插件测试 125 项通过、3 项跳过。用 W7 C32 的 64 条 prompt 回放调度，模拟每条
输出 64 个 token，其他设置保持一致，仅在对照组禁用前瞻。动态组的 122 次重排均在
下一次提交合并了原 CODA 和预计到达的请求。纯 decode CODA 从 388 批降为 253 批，
CORE 从 436 批降为 366 批。固定三轮在开启和关闭前瞻时逐批一致，CODA 均为 189 批，
CORE 均为 498 批。各组均产出 4096 个 token，没有抢占。回放用模拟回执验证批次行为，
吞吐和延迟另用设备测试。

在同一张 B300 上使用相同的 W7 prompt、并发 32、每请求最多 512 个输出 token、
1024 token/批、80 GiB KV、异步调度及 `FULL_AND_PIECEWISE`。固定组每条请求执行三轮，
动态组请求深度分布为 25% 一轮、25% 两轮、50% 三轮，Prefill 都执行三轮。两组各跑三波，
每组 192/192 条请求完成。首版允许同目标轮数重排，固定组三波输出吞吐为 2917.54、
2727.77、2672.30 tok/s，动态组为 3349.80、3373.08、2253.68 tok/s。用同一份源码单独
关闭前瞻后，固定组为 3670.09、3695.16、2345.85 tok/s，动态组为 2109.36、2040.79、
2119.01 tok/s。前瞻提升了动态组吞吐，但固定组吞吐中位数下降 25.68%。首版动态组第三波有
7 个续入请求首 token 等待约 3.71–3.77 s，波级 TTFT P95 达到 3722.78 ms。当前策略据此
保留同目标轮数的默认提交顺序。

当前版本的固定组吞吐为 3677.49、2664.41、3708.89 tok/s，动态组为 3346.20、3360.28、
3376.59 tok/s。两组均完成 192/192，固定组第二波有一条自然停止，其余请求均生成 512 个
token。显卡采样未发现外部进程。下表按三波取中位数，TTFT、ITL 先分别计算每波的 P95。

| W7 C32 | 固定三轮 output tok/s | 动态轮数 output tok/s | 动态 TTFT P95（ms） | 动态 ITL P95（ms） |
| --- | ---: | ---: | ---: | ---: |
| 关前瞻 | 3670.09 | 2109.36 | 226.02 | 17.40 |
| 首版前瞻 | 2727.77 | 3349.80 | 233.60 | 9.37 |
| 当前前瞻 | 3677.49 | 3360.28 | 239.02 | 9.53 |

与关闭前瞻相比，当前版本的动态组吞吐高 59.30%，达到当前固定三轮吞吐的 91.37%。
固定组三波仍有波动，当前与关闭前瞻的中位数接近。这三种策略在同卡依次测试，各策略内
交替运行固定与动态请求，结论限于本次 W7 C32 配置。

[CPU 调度回放](../artifacts/varloop-physical-batch-20261128/runs-one-step/cpu-varloop-replay.json) ·
[W7 单步前瞻完整数据](../artifacts/varloop-physical-batch-20261128/runs-one-step/summary.json)

### 恢复同周期 Decode Joining 的对照

前一轮 3670.09 和 2727.77 tok/s 来自同一份试验源码，两边都已取消 Joining，唯一开关
是 CODA 前瞻。允许相同目标轮数参与前瞻时，固定三轮出现了吞吐下降。将前瞻限定为待合并
行的目标轮数不同时，固定三轮中位数回到 3677.49 tok/s。这组数据不能把首版固定组的下降
归因于取消 Joining。

本次基于已经限定目标轮数的版本，只恢复 `0582bb7` 的 `_join_decode_batches()`。默认选出
候选批后，允许目标轮数相同的 Decode 在原有两次跳过上限内追赶。目标轮数仍用于识别追赶
对象，同一 Depth 的请求照常自然聚合。Unified CODA、跨轮 CORE、单步 CODA 前瞻、Worker、
状态槽和 CUDA Graph 保持原样。用于 GPU 对照的两个源码快照只有 `loop_scheduler.py` 不同。

同步和异步测试覆盖一轮、两轮追赶以及已有跳过时的限制，并验证合并后的请求继续成批
执行。纯跨轮 CORE 和跨轮数 CODA 前瞻的测试仍通过。CPU 插件测试 131 项通过、3 项跳过。

用 W7 C32 的同一语料模拟每条输出 512 个 token，两边都完成 32768 个 token。固定三轮
的纯 Decode CORE 从 4082 增至 4588 批，CODA 从 1533 增至 2038 批，CODA 每批真实行
中位数从 31 降至 10。动态组只触发 5 次 Joining，CORE 均为 3054 批，CODA 从 2045 降到
2041 批。CPU 回放只复现调度和回执依赖，设备吞吐另测。

固定回放的 1532 次 Joining 里，有 1527 次在应当提交 CODA 时改发第 2 轮 CORE。
例如第 88 次调度让 13 条 Decode CODA 等待另 10 条请求推进第 2 轮。接下来先发了已提权的
PRELUDE，原来的 13 条等满两次后先跑了 CODA。那 10 条没有完成第 3 轮，两个群组没有会合。
接下来第 3 轮 CORE 与另一些请求的第 1 轮 CORE 合批，新的小 CODA 批重复触发 Joining。
纯 Decode 跨轮 CORE 计划从 515 增至 1527 批。

GPU 0 上沿用上一版 W7 C32 配置，每组各三波。恢复 Joining 后，固定三轮的输出吞吐
为 2887.22、2228.42、2237.08 tok/s，中位数从上一版 3677.49 降至 2237.08（−39.17%）。
动态组为 3382.47、3367.30、3363.79 tok/s，中位数从 3360.28 升至 3367.30，各组均
完成 192/192 条请求。随后无 Joining 的同卡复测只完成首对正式波，结果为固定 3695.74、
动态 3369.75 tok/s。根据本次实验，代码和测试已回到上一版，保留异目标轮数 CODA 单步前瞻，
撤回 Joining。额外复测和设备执行路径诊断按回退要求停止。

[Joining CPU 回放](../artifacts/varloop-physical-batch-20261128/runs-one-step/cpu-joining-replay.json) ·
[固定三轮逐步调度](../artifacts/varloop-physical-batch-20261128/runs-one-step/cpu-joining-timeline.json) ·
[W7 GPU 数据](../artifacts/varloop-physical-batch-20261128/runs-one-step/summary.json)
