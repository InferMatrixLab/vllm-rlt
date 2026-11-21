# RMSNorm 缓存估算与 FMA 消融：2026-11-14

**建议保留已实现的缓存估算修正；不要全局替换为标量 FMA。** 缓存修正在误选缓存容量的边界上有可复现收益，FP32 H=2049、8 行减少约 4% 延迟；FMA 的收益依场景变化，在若干宽行、大网格场景显著变慢。本次是算子实验，未实现 FMA 到生产、未做新的模型端到端或 10 题质量评测。统计不表示真实流量的加权收益。

## 范围与方法

在空闲物理 GPU 7（NVIDIA B300 SXM6 AC、SM 10.3、148 SM，UUID `GPU-40e00845-d89c-1393-2c32-a2dca3ee9442`）测试。原计划使用 GPU 3，但启动前发现其他作业已占用该卡；全部四臂都在 GPU 7 重新测量，没有拼接旧 GPU 3 绝对时间。版本、构建、环境、命令及哈希见 [manifest.json](manifest.json)；封装验证、计时、扩展精度和 profiler 阶段的 `*-gpu-monitor.jsonl` 前后及周期采样均未发现外部进程。GPU 测试套件的空闲检查发生在启动前和完成后，期间没有连续进程采样。计时未锁频。

同一组输入和**同一预分配输出地址**比较四臂：旧缓存估算、修正缓存估算、修正加双累加链的逐元素标量 FMA、冻结的 vLLM RMSNorm。残差场景 vLLM 在归一化后另外做一次 add，符合生产的先转换到输出 dtype 再加残差语义。FMA 仅改 `accumulate` 的向量平方累加，未改标量首尾、归约、输出乘法和残差逻辑；见 [fma.patch](fma.patch)。四臂是显式 out 的源码衍生封装，图重放时间不等同生产 API 的分配/主机开销。

每形状每臂预热 10 次，随机 capture 次序和每轮臂次序；7 轮、每轮 CUDA event 计时 50 次包含 32 个 op 的 CUDA Graph 重放，报告每 op 微秒的中位数。首先独立运行关键 54 组，然后运行原 420 组（seed 42）加 H=4097/8193、rows=4/1024 的 24 组；补充 seed 3039 的 342 组只做精度检查。输入 dtype 为 FP32/FP16/BF16，均覆盖有无残差。完整每轮数据和顺序分别保存在 [key-raw.jsonl.gz](key-raw.jsonl.gz)、[matrix-raw.jsonl.gz](matrix-raw.jsonl.gz) 和 [extended-raw.jsonl.gz](extended-raw.jsonl.gz)。

实验封装与旧/当前生产扩展分别在 240 组不同 dtype、对齐/偏移、2–4 维及步长下逐元素 bitwise 一致；完整矩阵中当前封装每组都与生产分配接口一致。SM103 上六类代表性内核的指令文本和编码也分别与对应生产扩展一致，见 [sass-equality.json](sass-equality.json)。

## 图重放性能

下表**全部来自同一轮 444 矩阵**，时间单位为 μs/op，值是 7 轮中位数。缓存变化相对旧版，FMA 变化相对修正缓存版；负值为延迟降低。

| FP32 H / rows / 残差 | 旧缓存 | 修正缓存 | 修正+FMA | vLLM + 可选 add | 缓存变化 | FMA 变化 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2049 / 8 / 无 | 2.689 | 2.563 | 2.562 | 2.497 | -4.7% | 0.0% |
| 2049 / 8 / 有 | 3.136 | 3.010 | 3.010 | 3.784 | -4.0% | 0.0% |
| 4096 / 1024 / 无 | 5.316 | 5.319 | 6.612 | 5.247 | 0.0% | +24.3% |
| 4096 / 1024 / 有 | 5.764 | 5.764 | 6.629 | 9.533 | 0.0% | +15.0% |
| 8192 / 4 / 无 | 2.690 | 2.691 | 2.561 | 3.073 | 0.0% | -4.8% |
| 8192 / 4 / 有 | 3.009 | 3.009 | 2.753 | 4.291 | 0.0% | -8.5% |
| 8192 / 1024 / 有 | 11.814 | 11.814 | 15.075 | 19.528 | 0.0% | +27.6% |
| 8193 / 4 / 无 | 3.842 | 3.904 | 3.904 | 3.839 | +1.6% | 0.0% |

例如 H=4096/1024、无残差，修正缓存 7 轮范围为 5.317–5.327 μs，FMA 为 6.607–6.624 μs。独立的关键组也观察到 H=2049/8 缓存收益约 4%、H=4096/1024 FMA 回退约 17%–27%，但其数值没有混入主表。

444 组以逐组中位时间比的几何平均计算：旧/修正为 1.0057，修正/FMA 为 1.0066；后者同时包括明显获益和明显回退，不表示 FMA 稳健。以**时间比** 0.95–1.05 为持平区间，旧/修正为 0 组低于 0.95、438 组在区间内、6 组高于 1.05；修正/FMA 分别为 6/406/32 组。时间比区间并非严格的“延迟 ±5%”。旧/修正的 76 组实际选择不同缓存实例，其他 368 组实例选择相同；其中小耗时形状仍有约 5% 的测量差异，因此不把所有赢格归因于缓存。vLLM 总体对照仅在原始 420 个各臂全部通过精度的形状上计算，vLLM/修正的几何平均时间比为 1.3441，不能外推为模型端到端收益。[summary.json.gz](summary.json.gz) 记录完整分类和各案例样本。

## 正确性及局限

此次 GPU 测试套件 `tests/test_normalization.py tests/test_ouro.py tests/test_activation.py tests/test_native_build.py --run-gpu` 实际 **384 passed**，见 [gpu-tests.txt](gpu-tests.txt)。这与旧版历史测试分开统计。FP32、FP16、BF16 原阈值分别是 rtol=atol 2e-6、2e-3、1e-2，均未放宽。

seed 42 的原 420 组四臂均通过；加入边界后的 444 组中，旧缓存、修正、FMA 各 444/444，通过率不涵盖其他输入。vLLM 在新增 BF16 H=4097、1024 行、有残差时 1 个元素超阈值：实际 -0.171875，参考 -0.15625，修正缓存为 -0.15625；此新 vLLM 差异的底层数值原因尚未单独定位，不纳入 420 组总体比较。

seed 3039 扩展 342 组中，旧缓存、修正和 FMA 均为 **340/342**：BF16、残差 H=4095/1024 和 H=8192/296 各有一个元素误差 0.015625，三臂在这两元素一致；既有 [numeric-diagnosis.json](../rmsnorm-policy-20261110/numeric-diagnosis.json) 和 [numeric-probe.json](../rmsnorm-policy-20261110/numeric-probe.json) 将这两个历史边界定位为平方和舍入跨 BF16 中点、残差抵消。未发现 FMA 独有新增超阈值案例；这不保证全部 BF16 输入满足阈值。vLLM 在该组也有两处失败：H=3073/1024 和 H=8192/296，各一个元素；原始值均保留。

## 内核分支和 FMA 回退证据

短 ops-only/full PyTorch traces 都有 9 个目标内核：FP32 H=2049/8 从 256 线程未对齐 K4 改为 K2；H=4096/1024 均为 256 线程对齐 K0；H=8192/4 均为 512 线程对齐 K4。full 开启 record_shapes、profile_memory、with_stack，轨迹中有输入形状和 Python 函数事件；因本窗口使用预分配输出，没有采到分配事件，不能据此推断完整内存生命周期。profile 非性能基准。首个脚本名 `profile.py` 遮蔽标准模块导致同进程重复 capture，相关 `prior-module-shadow-*` 留档且不计入分析；已更名并重新捕获一次。

SM103 静态资源：H=4096/1024 的 K0 修正版每线程 32 寄存器，FMA 为 34；二者都无 local/stack 溢出、shared 1072 B。H=8192/4 的 K4 两臂均为 40 寄存器、无溢出、shared 1112 B。SASS 证实平方和主体的成对指令变为 FFMA；静态指令数不代表动态工作量。

同一 GPU 预热后单 kernel 的 NCU 诊断（3 次 replay passes，未控制缓存或锁频；原始 CSV 和 [ncu-summary.json](ncu-summary.json) 保留）：

| FP32 H / rows | 版本 | 寄存器/线程声明 → 实际分配 | 寄存器限制 blocks/SM | 活跃 SM 上的 warp 占比 | 全局 load 请求 |
| --- | --- | --- | ---: | ---: | ---: |
| 4096 / 1024 | 修正缓存 | 32 → 32 | 8 | 71.64% | 50,331,648 B |
| 4096 / 1024 | FMA | 34 → 40 | 6 | 64.43% | 50,331,648 B |
| 8192 / 4 | 修正缓存 | 40 → 40 | 3 | 23.89% | 262,144 B |
| 8192 / 4 | FMA | 40 → 40 | 3 | 23.94% | 262,144 B |

H=4096/1024 中声明多 2 个寄存器跨过分配粒度，实际分配从 32 增到 40，资源限制从 8 降到 6 blocks/SM，支持驻留下降参与回退；不同指令、调度和缓存也可能影响结果，无法只凭本消融把全部时间差定量归因于 occupancy。NCU 的缓存状态和 replay 会改变数值，不能用其计时替代无 profiler 的图重放测量。

## 本地复现

本目录发布了压缩原始数据、监控、NCU CSV 和 trace；manifest 保留原实验路径与未压缩内容的哈希。实验脚本及源码在 [experiment-sources.tar.gz](experiment-sources.tar.gz)，解压到 `artifacts/rmsnorm-cache-fma-20261114/`。旧版生产二进制与 vLLM 对照二进制需要按 manifest 的源码提交和环境重建，本包不分发二进制。`README.md` 为原 `REPORT.md` 调整发布链接后的副本，原报告哈希保留在 manifest。

从仓库根目录，使用该仓库对应的 PyTorch/CUDA 环境，先确认物理 GPU 7 空闲并核验 UUID；脚本会自行拒绝外部计算进程。更换 GPU 需先调整 guard 和冻结协议，不能直接拼接绝对耗时。

```bash
PY=/home/zjy/code/david/b_workspace/.b_rdma/bin/python
ART=artifacts/rmsnorm-cache-fma-20261114
CUDA_VISIBLE_DEVICES=7 "$PY" -m pytest tests/test_normalization.py tests/test_ouro.py tests/test_activation.py tests/test_native_build.py --run-gpu -q
"$PY" "$ART/make_variants.py"
CUDA_VISIBLE_DEVICES='' MAX_JOBS=2 "$PY" "$ART/build_one.py" rms_cache_old
CUDA_VISIBLE_DEVICES='' MAX_JOBS=2 "$PY" "$ART/build_one.py" rms_cache_fixed
CUDA_VISIBLE_DEVICES='' MAX_JOBS=2 "$PY" "$ART/build_one.py" rms_cache_fma
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. "$PY" "$ART/verify_wrapper.py" old
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. "$PY" "$ART/verify_wrapper.py" fixed
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. "$PY" "$ART/benchmark.py" key
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. "$PY" "$ART/benchmark.py" matrix
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. "$PY" "$ART/benchmark.py" extended
bash "$ART/run_ncu.sh"
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. "$PY" "$ART/capture_traces.py" ops
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. "$PY" "$ART/capture_traces.py" full
"$PY" "$ART/summarize.py"
```
