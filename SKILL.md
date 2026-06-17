---
name: perfetto-trace-holes
description: 修复 Perfetto 里 cuda-graph torch-profiler trace 同 stream kernel 之间出现的"空洞"（缺 kernel）——根因是 Hopper PDL 让相邻 kernel 执行窗重叠、Perfetto 画不下；用本工具把重叠 kernel 摊到额外行并保留 cudaGraphLaunch 箭头。看 trace 发现莫名空洞、或要判断某间隙是真 GPU 空闲还是渲染假象时用。
allowed-tools: Bash, Read
s_used: 0
s_edited: 2
s_last_used: 2026-06-15
s_last_edited: 2026-06-15
managed_by: agent
---

# /perfetto-trace-holes —— 修复 cuda-graph trace 的 kernel "空洞"

> **agent-managed skill**：可自行更新；改动请在文末「修改历史」留痕。

## 背景（为什么会有空洞）

用 torch profiler + Perfetto（ui.perfetto.dev）看开了 cuda graph 的 trace 时，同一条 GPU stream 上 kernel 之间会出现**空洞**，看着像缺 kernel / GPU 空闲。

- **根因**：Hopper(sm90) 的 **PDL（Programmatic Dependent Launch）** 允许后一个 kernel 的 prologue（grid 拉起 / TMA 预取）与前一个 kernel 的 epilogue 在同一条 stream 上**执行时间窗重叠**（数据依赖仍由 `cudaGridDependencySynchronize` 串行保证）。CUPTI 忠实测出 `next.ts < prev.ts + prev.dur`（**负 gap = 重叠**）。Perfetto 在一条 track 上画不下两个时间重叠的 slice，于是丢掉一个、留出空洞。`deep_gemm` / `flashinfer` / cutlass 的 sm90 kernel 默认带 PDL。
- **所以空洞≠真实 GPU 空闲**，那里其实坐着一个与邻居重叠的 kernel。
- **务必区分**：只有**图内 µs 级小洞（伴随负 gap）**是渲染假象；**ms 级大间隙是真 idle**（attention/sparse-flash、NCCL 通信 all-gather/reduce-scatter/all-reduce、MoE deepep all-to-all 那段全 stream 无 kernel），本工具**不会**也**不该**让它们消失。

## 解法

Tom(fzyzcjy) 的 `convert_to_perfetto_compatible.py`：把重叠 kernel 的 `tid` 加后缀挪到一条额外行显示，空洞处的 kernel 就露出来了。

本仓库版在其上做了**最小改动**（见下方 diff 指针）：
1. 同步把被挪 kernel 对应的 `ac2g` flow（`ph=="f"`，GPU 端）的 tid 一起改掉 → **保留 cudaGraphLaunch 箭头**（原版只改 kernel tid，会丢箭头）；
2. 后缀 `_hack` → `_pdl`（点名根因）。

> 已知局限：补充行用的是**字符串 tid**（`163_pdl`），Perfetto 的 chrome 导入器只对**整数 tid** 绑定 `thread_name`/`thread_sort_index`，所以补充行会出现在轨道**最底、不紧贴父流**。曾试整数 tid 方案让它紧贴父流，数据层正确但 UI 仍未紧贴，ROI 低，已放弃——当前只保「补洞 + 保箭头 + pdl 改名」。

## 何时用

- 在 Perfetto 看 sglang / PCG / 任意 cuda-graph 的 torch trace，发现 kernel 之间有莫名空洞。
- 想确认某个"间隙"到底是真 GPU 空闲、还是 PDL 重叠的渲染假象。

## 怎么用

依赖：`/usr/bin/python3` 需有 `orjson` + `typer`（本机已装；缺则 `/usr/bin/python3 -m pip install orjson typer`）。

```bash
cd .claude/skills/perfetto-trace-holes        # 进入本 skill 目录(相对 repo 根)；或 cd 到 SKILL.md 所在目录
/usr/bin/python3 scripts/convert_to_perfetto_compatible.py "<trace文件名.json.gz>" --dir-data "<trace所在目录>"
# 输出: <trace所在目录>/perfetto-compatible-<trace文件名.json.gz>
```

- `filename` 是**相对 `--dir-data` 的文件名**（沿用原版 main 约定），不是绝对路径；路径含空格时记得引号。
- 把输出的 `perfetto-compatible-*.json.gz` 拖进 https://ui.perfetto.dev 。重叠 kernel 出现在 `<原tid>_pdl` 行（在轨道最底）。
- **内存**：整文件 `orjson.loads` 进内存——1.9GB 解压 trace 峰值约 16.5GB RAM；小内存机器慎用。

### Perfetto 判读小抄

- **负 gap**（后一个 kernel 起点 < 前一个 kernel 终点）= PDL 重叠 = 会被渲染成洞。
- 同一次 `cudaGraphLaunch` 喷出的 kernel **共享同一个 correlation**（一个箭头指向多个 kernel）。
- 点某 kernel 看底部 **Preceding Flows** 的 Slice：`cudaGraphLaunch` = 图内 kernel；`cudaLaunchKernel`/`cuLaunchKernelEx` = eager 单发。
- GPU kernel 在 `pid=<device id>` 进程下、tid=`stream <N>`；CPU 侧（`cudaLaunchKernel` 等）在另一个同名进程的主线程。

## 指针（原码 / diff 单独成文件，本 skill 只给路径）

> 以下路径相对**本 skill 目录**（`.claude/skills/perfetto-trace-holes/`）。

- **改动版脚本（要跑的就是它）**：`scripts/convert_to_perfetto_compatible.py`
- **GitHub 原版（基准）**：`references/original_github.py`
- **改动 diff**：`references/patch.diff`

看 diff：

```bash
cd .claude/skills/perfetto-trace-holes
diff -u references/original_github.py scripts/convert_to_perfetto_compatible.py   # 或直接 cat references/patch.diff
```

## 链接

- **GitHub 代码页**：https://github.com/fzyzcjy/torch_utils/blob/master/src/convert_to_perfetto_compatible/convert_to_perfetto_compatible.py
- **raw（重新拉原版用）**：https://raw.githubusercontent.com/fzyzcjy/torch_utils/master/src/convert_to_perfetto_compatible/convert_to_perfetto_compatible.py
- **知乎背景文（BBuf《记录下 SGLang 开发，编译和 Profile 的几个小技巧》0x1.1 节）**：https://zhuanlan.zhihu.com/p/1939041055208112436
- **相关：合并多 rank trace（看跨 rank AllReduce 起止）**：https://github.com/fzyzcjy/torch_utils/blob/master/src/torch_profile_trace_merger/sglang_profiler_trace_merger.py

## 验证

跑完后产物里：被挪 kernel 与其 `ac2g` flow（`ph=="f"`）应都在 `<原tid>_pdl`；`_pdl` 行上事件数 = 2 × 被挪 kernel 数（kernel + flow 各一份）。本任务实测 GLM-5.1 PCG trace：moved 10761 / flow_reconnect 10761 / `_pdl` 事件 21522。

## 修改历史

| 日期 | 修改 | 理由 |
|------|------|------|
| 2026-06-15 | 新建：背景(PDL→Perfetto 洞) + 用法 + 改动版脚本(原版+flow 回连+`_pdl`) + 原码/diff 指针 + 知乎/GitHub 链接 | 看 PCG trace 时反复需要解释空洞并转换；沉淀为可复用 skill |
| 2026-06-15 | 引用路径由绝对改为相对(相对 skill 目录) | 绝对路径不可移植；规则已同步落进 skill-create |
