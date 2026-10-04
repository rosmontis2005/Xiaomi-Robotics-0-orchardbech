# Orchard V1 2650：第一轮 frozen-VLM XR-0 post-training

本文件对应独立入口 [`scripts/train_orchard_v1_2650.sh`](../scripts/train_orchard_v1_2650.sh)。准备日期：2026-10-02。**本任务只准备启动入口并执行静态检查 / DRY_RUN；没有启动训练或生成训练 checkpoint。**

## A. Experiment objective

验证 frozen-VLM XR-0 是否能从 replay-verifiable Orchard expert trajectories 中学习出有闭环意义的 action policy。本轮尚未证明策略有效；训练完成后的统一 closed-loop learning curve 才能检验这一目标。

训练继续使用现有 `train_orchardbench.sh → train.sh → tools/train.py → Lightning`，以及 `OrchardBenchDataset` / `OrchardRunner`。数据采集、controller、模型结构、loss、action horizon 和 normalization 均沿用当前实现。

## B. Dataset

只使用第二批 2650 accepted expert cohort 经既有 strict replay 筛选后的数据：

| 项目 | 数量 |
| --- | ---: |
| Raw expert accepted | 2650 |
| Strict replay PASS | 2374 |
| Strict replay FAIL | 276 |
| Filtered train episodes | 2144 |
| Filtered val episodes | 230 |
| Train full windows（30 steps） | 388172 |
| Val full windows（30 steps） | 41892 |

```text
raw provenance:
  /home/rosmontis/Projects/orchardbench/data/orchard_v1_2650/raw
DATASET_ROOT:
  /home/rosmontis/Projects/orchardbench/data/orchard_v1_2650/filtered
ORCHARD_STATS:
  /home/rosmontis/Projects/orchardbench/data/orchard_v1_2650/filtered/action_stats.json
```

训练选择 filtered manifest 中 accepted 的全部 train episodes，`split=train`、`episode_ids=null`。原始 episode IDs / splits 保持不变。filtered annotation 的视频路径仍引用同一第二批 raw 下的视频，因此需保留这些源文件；训练 episode 的选择依据始终是 filtered manifest。

**第一批 450 数据 `/home/rosmontis/Projects/orchardbench/data/orchard_v1_replay_verified` 不参与本轮训练，暂作为 independent holdout。** 不合并、不重编号、不建立 union / multi-root loader，也不使用第一批 stats。

唯一 normalization 文件为上述 `action_stats.json`，来自 2144 个 filtered train episodes 的 388172 个完整窗口。`action_stats_30x32.json` 是 collector preview stats，不用于训练。现有 Dataset 会校验 contract、train source fingerprint 和 normalization shape；本入口不生成 stats。

已核对的 provenance：

```text
train source_sha256:
  02157fb879b339c5fba94ca8a202428d3a84e09b23092ed43b6aa1b953268de3
action_stats.json file SHA-256:
  63dec8bbf2249d505358f17beac21d800bd49dee1192c9e79ddf3e1085c9b371
```

## C. Model initialization

初始化来源是本仓库 README 列出的 `XiaomiRobotics/Xiaomi-Robotics-0-Calvin-ABCD_D`，使用机器上已有的 HF sharded safetensors checkpoint：

```text
/home/rosmontis/Projects/dualsys/Xiaomi-Robotics-0/checkpoints/Xiaomi-Robotics-0-Calvin-ABCD_D
```

`PRETRAINED_CKPT` / `PROCESSOR_PATH` 指向此目录，`VLM_CONFIG_PATH` 指向其 `config.json`；沿用通用 launcher 的 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`。没有下载、切换或加载其他初始化权重。

正式启动时由现有 `OrchardRunner` 严格执行 **weights-only initialization**，`trainer.ckpt_path=null`；使用 **fresh AdamW、fresh ConstantLR，global step 从 0 开始**，不恢复旧 optimizer / scheduler / step。VLM 冻结并保持 eval 模式；当前 trainable 部分沿用 Runner 的 DiT、各投影层、timestep / sink 参数，trainable master weights 保持 FP32，计算使用 bf16 mixed。维持 16 层 DiT，不添加 LoRA。

## D. Action contract

```text
contract:    orchard_cartesian_local_rotvec_width_v1
action:      [30, 32]
state:       [1, 32]
active dims: 0:7（索引 0–6）
```

| Action 维度 | 含义 |
| --- | --- |
| 0:3 | 相对窗口起点 TCP 的 local translation，m |
| 3:6 | 相对同一窗口起点 TCP 的 local SO(3) rotvec，rad |
| 6 | absolute gripper width：两指总目标开度，m |
| 7:32 | 确定性零，action mask 为 0 |

30 个 action 共享窗口起点作为锚点。编码/解码继续复用 `treesim/orchard_action.py`。Normalization 沿用 `(action - mean) / (std + 1e-6)`，mean/std 均为 `[30,32]`，active std floor 为 `1e-4`，inactive mean=0/std=1；state 不归一化。**Deployment 必须使用训练时同一份 stats 和同一 contract**，不能采用 CALVIN 或第一批数据的 action scaling / stats。

## E. Training hyperparameters

入口显式设置固定实验环境变量，并通过既有 launcher 传递固定 Hydra overrides；其他配置复用 `configs/{data,model,trainer}/orchardbench.yaml`。完整 effective config 在每次启动前展开打印。

| 配置 | 本轮值 |
| --- | --- |
| Pretrained | `Xiaomi-Robotics-0-Calvin-ABCD_D`（上面的本地路径） |
| `freeze_vlm` | `true` |
| Precision / accelerator | `bf16-mixed` / `gpu` |
| Optimizer | `torch.optim.AdamW` |
| Learning rate | `1e-5` |
| Betas | `[0.9, 0.95]` |
| Weight decay | `0.1`；沿用 Runner 对 bias / norm 等的 no-decay 分组 |
| AdamW eps / foreach | `1e-8` / `false` |
| Scheduler | fresh `torch.optim.lr_scheduler.ConstantLR(factor=1.0, total_iters=1)`，每 step 调度，无 warmup |
| Gradient clip | norm，`1.0` |
| Batch size | 每 GPU `1` |
| Gradient accumulation | `1` |
| Max steps / max epochs | `10000` / `-1` |
| Seed | `42`（既有 helper 在各 rank 使用 `42 + RANK`） |
| `training_repeat` | `1` |
| `enable_freq` | `false` |
| `async_train` | `false` |
| `prefix_mask_prob` | `0.5`（保留既有值；本轮 sync 路径） |
| DiT layers | `16`（不变） |
| Action / state shape | `[30,32]` / `[1,32]` |
| Dataset split / subset | `train` / `episode_ids=null` |
| Data workers | `2` |
| Validation | `limit_val_batches=0`，`num_sanity_val_steps=0`，`val_check_interval=1.0` |
| Checkpoint interval | 每 `1000` optimizer steps；`save_top_k=-1`、`save_last=true` |
| Resume checkpoint | `trainer.ckpt_path=null` |
| Devices / strategy | 默认 `GPU_IDS=0`、`RESOURCE_GPU=1`，单节点 native `auto`；多 GPU 沿用 launcher 的 `ddp` |
| Logger | W&B offline，project=`orchardbench` |
| Experiment name | `orchard_v1_2650_frozen_vlm_seed42` |
| Output root | `/home/rosmontis/Projects/dualsys/Xiaomi-Robotics-0/xr0/outputs/orchard_v1_2650` |

`GPU_IDS`、`RESOURCE_GPU`、`PYTHON_BIN` 可由环境指定；默认 Python 是 `xr0/.venv-orchard/bin/python`。多 GPU 下全局 batch 为 `RESOURCE_GPU × 1`，推荐首轮使用下方单 GPU 命令。数据、stats、pretrained、output、实验名和训练超参数不接受调用环境覆盖；脚本拒绝额外命令行 / Hydra 参数。`DRY_RUN` 只允许 `0` 或 `1`，拼写错误会退出。没有新增 augmentation、LR sweep、DeepSpeed 或 controller 调整。

## F. Checkpoint plan

后续使用以下节点构建统一 closed-loop learning curve；本任务不实现或运行 evaluation：

| 节点 | 使用的权重 |
| --- | --- |
| 0 / pretrained | 原始 CALVIN pretrained 通过同一 Orchard 初始化/部署路径，使用本轮 stats / contract；不新存 step-0 checkpoint |
| 1k | `step=1000` checkpoint |
| 3k | `step=3000` checkpoint |
| 6k | `step=6000` checkpoint |
| 10k | `step=10000` checkpoint；正常完成时也对应 `last.ckpt` |

既有 `tools/train.py` 的 `ModelCheckpoint(every_n_train_steps=1000, save_top_k=-1, save_last=True)` 会保留每个 1000-step 节点，包含上述四个必需节点；不需要新增 callback。step 指 Lightning global step / optimizer update。只有正常达到相应 step 才会生成该节点。

## G. Exact launch command

**以下正式训练命令留待之后执行，本次没有执行：**

```bash
cd /home/rosmontis/Projects/dualsys/Xiaomi-Robotics-0/xr0

GPU_IDS=0 \
RESOURCE_GPU=1 \
PYTHON_BIN=/home/rosmontis/Projects/dualsys/Xiaomi-Robotics-0/xr0/.venv-orchard/bin/python \
DRY_RUN=0 \
bash scripts/train_orchard_v1_2650.sh
```

配置验收命令（安全，不构建模型或进入训练）：

```bash
cd /home/rosmontis/Projects/dualsys/Xiaomi-Robotics-0/xr0

DRY_RUN=1 \
GPU_IDS=0 \
RESOURCE_GPU=1 \
PYTHON_BIN=/home/rosmontis/Projects/dualsys/Xiaomi-Robotics-0/xr0/.venv-orchard/bin/python \
bash scripts/train_orchard_v1_2650.sh
```

通用 launcher 先执行 `tools/train.py ... --cfg job --resolve`。Hydra 在显示配置后退出，不调用 `main` / `prepare`；`DRY_RUN=1` 随即令 shell 退出，不调用 `scripts/train.sh` / torchrun / `Trainer.fit`。配置展开会 import 现有训练依赖，但不会实例化模型、加载权重或构建 optimizer。缺失数据/stats/权重文件或 Python 时入口直接失败，没有 fallback。非 dry-run 启动发现实验目录已存在也会失败，防止覆盖和隐式续训。

## H. Expected outputs

正常训练时实验目录为：

```text
/home/rosmontis/Projects/dualsys/Xiaomi-Robotics-0/xr0/outputs/orchard_v1_2650/project_orchardbench/orchard_v1_2650_frozen_vlm_seed42
```

预期内容：

- `config.yaml` / `config.py`：既有 helper 保存的 resolved config；同一 helper 仍会写 `xr0/assets/config.py` 便于部署。
- `hydra/.hydra/{config,hydra,overrides}.yaml` 和 `hydra/train.log`：Hydra 配置快照和 Python logging；shell / Lightning 进度也输出到终端，`train.log` 不保证包含全部终端输出。
- `epoch=…-step=1000.ckpt`、`epoch=…-step=2000.ckpt` 等每 1000 steps 的 checkpoint，正常完成后包含 `step=10000.ckpt`。
- `last.ckpt`：最近一次保存的 checkpoint；正常完成 10000 steps 后即 final / last，不另造 `final.ckpt`。

W&B 日志沿用现有 `WandbLogger(save_dir=".")`，位于 `xr0/wandb/offline-run-*`（实验目录之外），包含 train losses / LR 等输出；本入口固定 `WANDB_MODE=offline`。不修改 logger。

保留全部 checkpoint 会增加磁盘占用。后续部署需配对保留原始 `action_stats.json` 和 contract；既有训练程序只在配置中记录 stats 路径，不自动复制/打包 stats。本任务的 DRY_RUN 只显示配置，不创建上述实验目录和训练 checkpoint。

## I. Known limitations

1. 当前训练期间没有真正的 validation dataloader：`OrchardBenchDataModule.val_dataloader()` 返回 `[]`，trainer 已关闭 validation。
2. 230 个 val episodes / 41892 个 val windows 当前主要保留用于后续 offline evaluation，本任务不重构 val loader。
3. Strict replay PASS 不代表 controller 没有 clipping / IK intervention，也不能证明 policy 闭环成功。
4. Training loss 不能代替 closed-loop success；本轮只准备验证实验，尚无策略有效性或泛化结论。
5. 第一轮只训练第二批的 filtered train 数据；2374 replay PASS 包含 train 和 val，并非全部都进入优化。
6. 第一批 450 batch 尚未参与训练，保留为 independent holdout；本任务不制作 holdout evaluator。
7. 多 GPU DDP 沿用现有入口，本次仅验收单 GPU 配置，没有进行分布式训练验证。
8. 本说明中的 cohort 统计属于当前 2650 批次；`TRAINING_NOTES.md` 的历史 smoke 数据量和 action stats 路径不应用于本次实验。

## J. Preparation verification

静态验收核对现有 manifest / annotation SHA-256、2144/230 split、388172/41892 windows、train source fingerprint、stats contract / `[30,32]` shape，以及 4288 个 train 视频路径存在。只读取现有文件，没有重新筛选数据、计算 normalization、解码视频或调用 Dataset `__getitem__`。

`bash -n scripts/train_orchard_v1_2650.sh scripts/train_orchardbench.sh` 通过。上述 `DRY_RUN=1` 命令退出码为 `0`；展开的配置确认 filtered root / stats、`freeze_vlm=true`、`bf16-mixed`、AdamW `lr=1e-5` / `betas=[0.9,0.95]` / `weight_decay=0.1`、clip=1.0、batch=1、max_steps=10000、seed=42、`training_repeat=1`、`enable_freq=false`、`async_train=false`、`save_interval=1000`、`ckpt_path=null`。

对捕获的 YAML 进行静态解析，40 个配置字段断言通过；callback 保留策略通过源码静态检查。验收前后 stats 文件 SHA-256、`assets/config.py` 和两个既有 smoke checkpoint 均未变化；没有新增 checkpoint，也没有创建本实验输出目录。此次准备工作的边界是：

```text
NO TRAINING WAS STARTED
NO FORWARD/BACKWARD WAS RUN
NO OPTIMIZER STEP WAS RUN
NO TRAINING CHECKPOINT WAS GENERATED
```
