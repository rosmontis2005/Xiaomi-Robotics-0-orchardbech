# OrchardBench V1 Closed-Loop Learning Curve Benchmark

本入口比较 pretrained → 1k → 3k → 6k → 10k 的新 simulator seeds 闭环能力。
只改变 weights；所有节点都经过当前 `OrchardPolicy` 和 Orchard action adapter。
不调用 CALVIN evaluator，也不修改训练、数据、模型、controller 或物理参数。

## 固定协议

| 项目 | 值 |
| --- | --- |
| Seeds | 从 3010000 顺序扫描，仅 `OrchardVLAEnv.reset(seed=...)` 成功即可入选，收满 30 |
| 可视化 | 在加载任何模型前冻结上述列表前 5 个 seed；保存 static / wrist MP4 |
| Execution | 每次 `policy.predict(obs, seed=42)`；完整执行 k=0…29；每步 `policy.command(k, obs)` 使用最新 live obs |
| Budget | 900 control steps，30 Hz，约 30 s；成功可提前终止 |
| Grasp / detach | `benchmark_assist` / `detach_force_scale=1.5`；其他 `VLAEnvConfig` 默认值 |
| Success | 环境的 `success`：至少一个已脱离苹果物理上位于桶内 |
| Action | `orchard_cartesian_local_rotvec_width_v1`；[30,32]；active 0:7；local translation / SO(3) rotvec / absolute physical width |
| Prompt / RGB | 复用 `policy_messages` 的 “Pick an apple and place it in the bucket.” 完整模板及 `prepare_rgb`；两路原有相机 |
| Processor / pretrained | `../checkpoints/Xiaomi-Robotics-0-Calvin-ABCD_D` |
| Stats | `/home/rosmontis/Projects/orchardbench/data/orchard_v1_2650/filtered/action_stats.json` |

2650 collection 预留范围为 2010000…3009999；450 collection 预留范围为
1020000…2009999（见 OrchardBench 两批 collection 的 `seed_scan.json`）。入口拒绝
3010000 以下的新 seed，完全不根据训练/val episode 或任何 policy 的表现筛选。

正式检查点目录固定为
`outputs/orchard_v1_2650/project_orchardbench/orchard_v1_2650_frozen_vlm_seed42/`。
仅匹配 `*step=1000.ckpt`、`*step=3000.ckpt`、`*step=6000.ckpt`、
`*step=10000.ckpt`；每个选中节点必须恰好匹配一个文件。缺失或重复直接报错，
与 epoch 编号无关，也不会使用 `last.ckpt` 替代 10k。pretrained 单独使用 HF 目录。

## 运行

以下命令从 `xr0/` 执行。薄 launcher 默认使用 `.venv-orchard/bin/python`，
可以通过 `PYTHON_BIN` 指定已有的兼容环境、`GPU_IDS` 指定单张卡。
当前机器的 Newton 在 OrchardBench 的同 Python 版本 Pixi 环境中；入口仅在
Newton 不可导入时把该环境的 site-packages 追加到搜索路径尾部，保持 XR-0
依赖优先级，不安装或更改环境。统一安装了两套依赖的 Python 可直接执行脚本。
绘图需要 matplotlib；视频使用系统 ffmpeg（libx264），不改 renderer。

```bash
# 只发现选中的五个权重、检查路径/现有 seed 文件并打印配置；不会 reset 或加载模型。
bash scripts/eval_orchard_learning_curve.sh --checkpoints all --dry-run

# 只扫描并冻结 30 个 feasible seeds + 5 个 visual seeds；不加载模型。
bash scripts/eval_orchard_learning_curve.sh --checkpoints all --prepare-seeds

# 正式 benchmark：此命令会执行完整 5 × 30，脚本开发验收不运行此命令。
bash scripts/eval_orchard_learning_curve.sh \
  --checkpoints all --num-seeds 30 --seed-start 3010000 \
  --output outputs/orchard_v1_benchmark

# 1-checkpoint × 1-seed 调试；复用正式冻结列表的首个 seed。
bash scripts/eval_orchard_learning_curve.sh \
  --checkpoints step_1000 --num-seeds 1 \
  --seeds-file outputs/orchard_v1_benchmark/benchmark_v1_30seeds.json \
  --output outputs/orchard_v1_benchmark_smoke

# 可追加 --no-video；也可以指定 --checkpoints pretrained。
```

直接 Python 调用具有相同 CLI：

```bash
.venv-orchard/bin/python tools/eval_orchard_learning_curve.py \
  --checkpoints all --num-seeds 30 --seed-start 3010000 \
  --output outputs/orchard_v1_benchmark --dry-run
```

`--num-seeds N` 仅选择冻结集合的前 N 个，调试时也先冻结完整 30 个和前 5 个
visual seeds。`--seeds-file` 接受本工具生成的冻结文件；显式指定但不存在则报错。
已有 seed 文件永远复用，`--seed-start` 不会重新扫描或替换它。已有文件损坏、
环境配置/源码变化时直接报错。reset 的 stance/home-IK infeasibility 记录原因并跳过；
其他基础设施异常中止扫描，避免硬件/依赖错误改变 cohort。扫描审计为
`benchmark_v1_30seeds.scan.jsonl`，仅完整收齐后写冻结文件。

每次按固定顺序加载一个 checkpoint，跑完选中的相同 seeds 后释放模型和 CUDA
缓存。模型加载失败会为该节点每个 seed 记录 error，然后继续下一个选中节点；
rollout 异常记录该 episode 后继续下一 seed，不 retry。结果目录已有 manifest 或
episodes 时拒绝覆盖。需要重跑时用新 `--output` 并复用原 `--seeds-file`；不做隐式恢复
或跨运行拼接。subset 报告、图表明确标记 `DEBUG SUBSET`，不能充当正式结果。

## 结果与口径

默认输出 `outputs/orchard_v1_benchmark/`：

```text
benchmark_v1_30seeds.json          # frozen seeds + skipped reset reasons + simulator identity
benchmark_visual_seeds.json       # frozen first five + seed-file hash
checkpoint_manifest.json         # only selected weight paths/stat identities; full fixed config/hashes
benchmark.log
episodes.jsonl / episodes.csv
summary.json / benchmark_report.md
learning_curve.png / capability_funnel.png
clipping_rate.png / ik_failure_rate.png / episode_length.png
{checkpoint}/summary.json
{checkpoint}/episodes.jsonl
{checkpoint}/videos/{checkpoint}_seed_{seed}_{outcome}_{static|wrist}.mp4
```

每个完成的 episode 立即追加 JSONL 并更新节点 summary。视频只缓存冻结 visual
seeds 的两路原始 RGB（reset + 每个 control step），episode 结束后按 30 FPS 编码。
视频错误写 `video_errors` 和日志，保持原任务结果。plots 失败保留所有数值结果并记录
`plot_status`。正常退出为 0，episode/model/plot 错误完成后返回 1，配置/路径错误返回 2。

阶段按优先级 `SUCCESS > DETACHED_NOT_PLACED > GRASPED_NOT_DETACHED > NO_GRASP`。
grasp trigger / held fruit ID（包括 ID=0）在整个 rollout 累积；detach 取最大计数，
完全不使用 intended target。只有环境 `success` 为真才算成功。

`termination=success|episode_budget` 的记录构成 **valid denominator**；
`episodes_completed` 与 valid denominator 相同。`termination=error` 的记录保留
`error_type/error_message/error_phase`、已观测诊断，并将 success / capability_stage
设为 null；它们不计入 success/grasp/detach 率和阶段分布。
部分 rollout 的最后阶段放在 `last_observed_capability_stage`，不是普通任务失败。

步数是 env.step 正常返回的次数；replans 是 predict 调用次数。三项能力率以 valid
episodes 为分母，IK/clipping 总率以 valid episodes 的总 control steps 为分母，
不是逐 episode rate 的简单平均。error 的已返回步数另列 `error_control_steps`。
branch-break mean 是 valid episodes 的最大 branch count 均值。
无有效 episode 时率和均值为 null / N/A，不伪装成 0%。报告同时列出 requested、valid、errors。

CPU 协议测试不调用实际模型或模拟器：

```bash
bash -n scripts/eval_orchard_learning_curve.sh
.venv-orchard/bin/python -m unittest discover -s tools -p test_eval_orchard_learning_curve.py -v
```

## 本次开发验收（2026-10-02）

`bash -n`、Python syntax/import、上述 9 项 CPU 协议测试、五节点 `--dry-run` 均通过。
dry-run 无 Torch/Newton import，也不加载模型。当前四个训练文件分别为
`epoch=0-step={1000,3000,6000,10000}.ckpt`，pretrained HF 目录及分片完整。
缺失/重复 checkpoint 的错误分支使用 glob/stat mock 验证，没有伪造任何 checkpoint。

reset-only 扫描得到 `3010000…3010029`，30/30 reset 成功、跳过 0；可视化在模型
加载前冻结为 `3010000…3010004`。与 2650 和 450 两批原始 manifest 的 collection
seeds 均无交集。正式输出目录目前只有冻结 seed/扫描记录，没有正式 rollout 结果。

只执行了一次 `step_1000 × seed 3010000` smoke，输出在
[`outputs/orchard_v1_benchmark_smoke/benchmark_report.md`](../outputs/orchard_v1_benchmark_smoke/benchmark_report.md)：

| 项目 | 观测值 |
| --- | --- |
| Termination / stage | `episode_budget` / `NO_GRASP` |
| Control steps / replans | 900 / 30 |
| Grasp / detach / bucket | false / 0 / 0 |
| Action clipped | 634/900，70.44% |
| IK failed | 314/900，34.89% |
| Valid denominator / runtime errors | 1 / 0 |
| Video | static、wrist 均为 H.264 MP4，192×144、30 FPS、901 帧（含 reset），无写入错误 |
| Plots / reports | 5 张 PNG、JSONL、CSV、aggregate JSON、Markdown report 均生成并校验 |

这是接口与产物链路验收，不是模型能力的 learning-curve 结果。验收前后现有
OrchardPolicy、环境、action contract、stats 文件 SHA-256 一致，四个目标 checkpoint
的文件大小和修改时间一致。没有修改这些文件或任何训练代码。

```text
FULL 5 × 30 BENCHMARK WAS NOT RUN
NO TRAINING WAS STARTED
NO CHECKPOINT WAS MODIFIED
```
