# vLLM-RLT 插件化开发记录

- 工作分支为 `vllm-rlt-plugin2`，起点为主仓 `4821d47`。
- 设计目标为 vLLM `0.34.0rc1.dev1149+g3146a9df4`，本地实际验证使用同一 Git revision `3146a9df4140`。该构建环境的生成版本标签为 `0.26.1rc1.dev6849+g3146a9df4`。
- 每轮记录本轮范围、复现命令、实际结果和待核实项。改动在当前工作区留为 unstaged，供 VS Code 审阅。

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

记录一条待实现的调度优化。比如一个 8-token prefill span 已完成 CORE，正在等
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
