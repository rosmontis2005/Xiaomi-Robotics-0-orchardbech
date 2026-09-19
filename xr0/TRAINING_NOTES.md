# OrchardBench XR-0 post-training V1

验证日期：2026-09-18。已执行全量数据审计、两次成功的单步官方训练 smoke、20 步固定样本诊断、离线推理解码和 3 次模拟器控制接口调用。**未运行 3000 步正式训练，也未运行闭环策略评估。**

## 仓库与数据审计

| 项目 | 实际位置 / 结果 |
|---|---|
| OrchardBench HEAD | `26a4b97ddb71aacc0677549b9f4aa0b45399ec18` |
| OrchardBench | `/home/rosmontis/Projects/orchardbench` |
| XR-0 HEAD | `7c316604d43357d10b06d11fae4c39290c447da5` |
| XR-0 训练代码 | `/home/rosmontis/Projects/dualsys/Xiaomi-Robotics-0/xr0`（git 根目录在上一级） |
| 正式数据 | `/home/rosmontis/Projects/data/orchard_autopicker_v1` |
| manifest | `manifest.jsonl`：780 次尝试，300 accepted，270 train / 30 val |
| 帧数 | 每条 113–197 帧；总计 39,527 个时间帧 |
| 视频 | 600 个真实 MP4，ego/static + wrist_left；144×192 RGB，30 Hz；完整解码 79,054 个视频帧 |
| 完整 30 步窗口 | train 27,622；val 3,205；不复制窗口、不采样尾部残缺窗口 |
| 硬件 | RTX 5060 Ti，16,311 MiB VRAM；约 46 GiB 主存 |
| CALVIN 权重 | `../checkpoints/Xiaomi-Robotics-0-Calvin-ABCD_D`，两个 safetensors 分片、index、config、processor/tokenizer 文件，约 8.8 GiB |

采集默认路径并非正式数据路径；实际目录由现有文件、manifest、collection config 与 summary 确认。初始 Orchard 仓库无修改；XR-0 上级仓库已有 `eval_calvin/main.py` 修改及 checkpoint/eval/log 等未跟踪目录，本任务未更改这些既有内容。未 commit/push。

实际完整首条 JSON：`/home/rosmontis/Projects/data/orchard_autopicker_v1/json/train/episode_000001.json`。全部字段及嵌套类型/长度/实例已记录在 `artifacts/orchardbench/episode_schema.json`。核心 schema：

```text
schema: orchard_xr0_single_arm_v0
trajectory_type: success
seed: integer
num_frames: N (首条 151)
record_fps: 30
episode_id: episode_000001
split: train | val
instruction.general[]:
  images: [observations.ego, observations.wrist_left]
  conversations: [{from: human, value: 两个 <image> 的任务文本},
                  {from: gpt, value: <bot></bot>}]
observations.{ego,wrist_left}[]:
  path: MP4 路径; start: 0; end: N; fps: 30; crop_bbox: null
proprios:
  ee_pos: [N,3]       # world metres
  ee_rotm: [N,9]      # row-major world rotation matrix
  arm_joint: [N,7]    # radians
  arm_joint_vel: [N,7]# radians/second
  gripper_pos: [N,1]  # 两指开度之和，metres
actions:
  ee_pos: [N,3]; ee_rotm: [N,9]; arm_joint: [N,7]; gripper_pos: [N,1]
orchardbench: provenance / diagnostics（详见结构 artifact，绝不进入模型输入）
```

每个 `actions[key][t] == proprios[key][t+1]`，最后一个 target 重复最后实测状态；所有 accepted JSON 都按 collector 的 `validate_arrays()` 重新验证。窗口允许包含这个已有 terminal hold target，不伪造额外帧。

可用质量 metadata 包括：fixed-base expert/planner、base pose/timestamps/drift、clean start、完整 REACH→GRASP→PULL→TRANSPORT→DROP→DONE trace、home joints、first-attempt success、grasped/detached/placed、branch/incidental detach、target visibility（初始/GRASP 静态和腕部像素数）、detach frame/force/tension/premature/meaningful-pull diagnostics、detach force multiplier。全量 collector gate 重审通过，包括 premature=false、incidental=0、固定底盘及完整链条。

## 唯一训练/部署契约

契约名：`orchard_cartesian_local_rotvec_width_v1`。唯一数学实现位于 OrchardBench 的 `treesim/orchard_action.py`，XR-0 通过 `PYTHONPATH=$ORCHARD_REPO` 引用它，训练与部署不各自复制编码公式。

记录量是 physical state，未经 VLA normalization。设窗口起点为 t，当前世界 TCP 位姿为 `(p_t,R_t)`，k=0..29 的真实 target 为记录的 `action[t+k]`，即 `S_(t+k+1)`（末帧遵循 terminal hold）。

| 32D state 维度 | 内容 / 单位 |
|---|---|
| 0:3 | 世界坐标 TCP xyz，m |
| 3:6 | 世界 TCP extrinsic xyz Euler，rad；只是 state 表示 |
| 6 | 两指总实测开度，m |
| 7:14 | FR3 joint 1..7，rad；第七关节明确在索引 13 |
| 14:32 | 确定性零 |

state 形状 `[1,32]`，不归一化；不使用关节速度或 privileged metadata。

| 32D action 维度 | 内容 / 单位 |
|---|---|
| 0:3 | `R_t.T @ (p_target - p_t)`，窗口锚点 TCP 局部坐标，m |
| 3:6 | `Log(R_t.T @ R_target)`，SO(3) rotation vector，rad |
| 6 | **绝对** target 总夹爪开度，m；不二值化、不做 delta |
| 7:32 | 确定性零，mask=0，无关节预测目标，无右臂 |

每个窗口锚点固定；30 个动作都相对于同一个 t，而非依次累加。action/action_mask 形状 `[30,32]`，mask 仅 0:7 为 1。两个图像块及固定自然语言任务是唯一视觉/语言输入。模型消息不拼接 target ID、GT fruit pose、segmentation、expert state、IK target 或 visibility。训练和新推理桥均使用相同 95% 中心裁剪、factor-32 resize（此数据为 192×128）及相同 Qwen chat template。

反解（先逐 horizon 反归一化）：

```text
a = z * (std + 1e-6) + mean
p_target = p_t + R_t @ a[0:3]
R_target = R_t @ Exp(a[3:6])
width_target = a[6]
```

对执行时的**实时**观测 `(p_now,R_now)`，生成 native world translation `p_target-p_now`，以及 `Euler_xyz(R_target @ R_now.T)`。旋转向量必须先转换成矩阵，再转换成原生左乘 world Euler delta，不能直接把 rotvec 当 Euler。通过 `env.step(action, gripper_width=width_target)` 控制两指目标各为 width/2。默认 signed-gripper 接口仍用于旧 CALVIN 路径。宽度接口保留机械 `[0,.08]` 裁剪及原有控制器限幅/IK 检查。

`OrchardPolicy.predict(obs)` 使用同一 state/prompt/images、32D mask、训练 stats 和官方 XR0.generate；`policy.command(k,obs)` 返回可直接传入 `env.step(**kwargs)` 的参数。该桥是独立本地接口，不改变原有 generic dual-arm server，也不把 Orchard 权重交给旧 CALVIN smoke worker。

### 为什么必须独立 adapter

官方 `JsonDataset` / `compose_state` 的左臂关节 slice 为 7:13，只容纳 6 joints；默认要求左右臂/三路视图，不能直接承载这套数据。collector 的 preview 还包含 joint delta 和 gripper delta。现有 `CalvinActionAdapter` 则是 world xyz×50、Euler delta×20、累积 target Euler、signed grip。这三种契约均不等同。

本版保留 collector 的局部 SE(3) 结构，取消 joint action objective，夹爪采用可精确恢复的绝对物理开度，并显式增加部署宽度入口。CALVIN checkpoint 仅作权重初始化，**不继承 CALVIN action stats、action scaling 或 gripper convention**。原始 preview stats 不能使用。

## 统计与数据来源

生成 artifact：`artifacts/orchardbench/action_stats.json`。只遍历 accepted **train** 27,622 个完整窗口，直接调用 Dataset 的 `raw_action()` / `encode_window()`。采用 float64 Welford population mean/std，保存 float32 `[30,32]`；active std 下限 `1e-4`，inactive mean=0/std=1。normalize 与官方一致：`(action-mean)/(std+1e-6)`。

artifact 包含契约、全部源 episode IDs、源 JSON SHA-256、样本数、epsilon 和 floor。Dataset 校验契约及源 fingerprint，拒绝旧 preview stats 或与选择集不符的统计文件。重复生成得到逐字节一致的文件。validation 仅读取 train stats；绝不计算自己的统计。

默认使用 canonical accepted train 全集。可选 `data.params.train_datasets.episode_ids=/path/to/ids.json`（JSON 字符串数组）只允许 accepted train IDs；更换 subset 必须用同样的 `--episode-ids` 重新计算 stats。`audit.json` 的 `weak_visibility` 给出 initial static<=2 pixels 且 wrist=0 的 38 条（train 30 / val 8），可供以后生成排除列表；默认没有排除它们。当前 val audit 使用 canonical train stats；未来 ablation 验证需明确选择对应的 train provenance。不存在 rejected 数据混入。

## 初始化、训练配置及范围

官方 README 的 `model.params.model.pretrained` 示例与当前实现不符：实际 `BaseRunner` 消费 `model.params.pretrained`，且仅 `torch.load(...)["module"]`，不能直接加载 HF 目录。新增 `OrchardRunner` 严格加载 HF 分片，或兼容官方 `module`/Lightning `state_dict` 权重文件；所有 key 和 shape 必须匹配。HF index 有 932 个张量，补回**已验证共享 embedding 的 tied lm_head**后严格加载 933 个键。不存在 `strict=False` 静默漏载。

本机没有独立 Qwen3-VL-4B cache，因此新增可选 `vlm_config_path`，从本地 checkpoint 嵌套 VLM config 构建官方训练架构，然后加载全部权重；不下载/回退到另一套权重。默认模型路径不受影响。

训练通过 `scripts/train_orchardbench.sh → scripts/train.sh → tools/train.py → Lightning`。`trainer.ckpt_path=null`；fresh AdamW/fresh ConstantLR，global step 从 0 开始。OrchardRunner 拒绝 trainer resume。`PRETRAINED_CKPT` 可覆盖；使用官方 generic HF pretrained 时也走同样严格加载，没有本机可验证的 generic checkpoint，因此该 fallback **未实测**。若使用转换后的 `.pt`，另设 `PROCESSOR_PATH` 指向兼容 HF 目录，`VLM_CONFIG_PATH` 指向其 config.json。

保守默认：sync、batch=1、LR=1e-5、3000 steps、bf16 mixed、冻结 VLM，训练 DiT/所有投影层与 timestep/sink（278,458,368 parameters）。trainable master weights/Adam moments 保持 FP32，避免低 LR 更新被 bf16 参数舍入抹掉；VLM 权重与计算为 bf16。可设 `FREEZE_VLM=false`，但全量微调在本机未验证，可能超显存。batch、LR、steps、seed、GPU IDs/resource、数据/config、权重、output/experiment 均可覆盖。

官方默认 frequency loss 没有使用 action mask，故 Orchard config **enable_freq=false**。mask 只作用于 Cartesian 7D；已测试改变闲置预测不改变 loss，闲置维度梯度严格为零。`training_repeat=1`，不做 async prefix。

本机原 `xr0-mibot` 环境有 Torch 2.8.0+cu128 / Transformers 4.57.1 / FlashAttention，但没有训练依赖。已创建 `.venv-orchard`（system-site-packages）并安装 `requirements-orchard.txt` 所列依赖；完整版本在 `artifacts/orchardbench/environment.txt`。DeepSpeed 安装因无 CUDA_HOME/nvcc 失败；选择官方 Lightning 的 native auto strategy + torch AdamW，避免 CUDA 扩展编译，未重写训练框架。增加了小型 auto/DDP strategy hook。多 GPU DDP 未实测。W&B 默认为 offline。

## 精确复现命令

以下均在 XR-0 目录执行；除明确标注的正式命令外，都是已执行的 smoke/audit 路径。

```bash
cd /home/rosmontis/Projects/dualsys/Xiaomi-Robotics-0/xr0
export ORCHARD_REPO=/home/rosmontis/Projects/orchardbench
export DATASET_ROOT=/home/rosmontis/Projects/data/orchard_autopicker_v1
export PRETRAINED_CKPT=/home/rosmontis/Projects/dualsys/Xiaomi-Robotics-0/checkpoints/Xiaomi-Robotics-0-Calvin-ABCD_D
export ORCHARD_STATS="$PWD/artifacts/orchardbench/action_stats.json"
export PYTHONPATH="$PWD:$ORCHARD_REPO"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# 首次环境建立（本机已完成；不需要 DeepSpeed）：
/home/rosmontis/miniconda3/envs/xr0-mibot/bin/python -m venv --system-site-packages .venv-orchard
.venv-orchard/bin/pip install -r requirements-orchard.txt

.venv-orchard/bin/python tools/compute_orchard_stats.py \
  --root "$DATASET_ROOT" --output "$ORCHARD_STATS"
.venv-orchard/bin/python tools/compute_orchard_stats.py \
  --root "$DATASET_ROOT" --output /tmp/orchard_stats_repeat.json
cmp "$ORCHARD_STATS" /tmp/orchard_stats_repeat.json

.venv-orchard/bin/python tools/audit_orchardbench.py \
  --root "$DATASET_ROOT" --stats "$ORCHARD_STATS" \
  --orchard-repo "$ORCHARD_REPO" --output artifacts/orchardbench/audit.json
.venv-orchard/bin/python tools/audit_orchard_precision.py \
  --root "$DATASET_ROOT" --stats "$ORCHARD_STATS" --output-dir artifacts/orchardbench

DRY_RUN=1 bash scripts/train_orchardbench.sh
MAX_STEPS=1 EXP_NAME=smoke_final OUTPUT_ROOT="$PWD/artifacts/orchardbench/runs" \
  bash scripts/train_orchardbench.sh \
  model.params.diagnostic_path="$PWD/artifacts/orchardbench/model_smoke_final.json" \
  +trainer.limit_train_batches=1 trainer.save_interval=100000

.venv-orchard/bin/python tools/smoke_orchard_model.py \
  --root "$DATASET_ROOT" --stats "$ORCHARD_STATS" --pretrained "$PRETRAINED_CKPT" \
  --steps 20 --output artifacts/orchardbench/overfit.json

# 实际验证使用第一次成功单步运行保存的 checkpoint：
OMP_NUM_THREADS=1 .venv-orchard/bin/python tools/smoke_orchard_inference.py \
  --weights artifacts/orchardbench/runs/project_orchardbench/smoke_one_step_v3/last.ckpt \
  --processor-path "$PRETRAINED_CKPT" --stats "$ORCHARD_STATS" \
  --episode "$DATASET_ROOT/json/train/episode_000001.json" \
  --output artifacts/orchardbench/inference.json
# 从头复现时可换成上面刚生成的 smoke_final/last.ckpt。

cd "$ORCHARD_REPO"
.pixi/envs/default/bin/python scripts/test_orchard_action_env.py \
  --output /home/rosmontis/Projects/dualsys/Xiaomi-Robotics-0/xr0/artifacts/orchardbench/controller_smoke.json
```

首个正式实验命令（**本任务没有执行**）：

```bash
cd /home/rosmontis/Projects/dualsys/Xiaomi-Robotics-0/xr0
GPU_IDS=0 RESOURCE_GPU=1 \
DATASET_ROOT=/home/rosmontis/Projects/data/orchard_autopicker_v1 \
PRETRAINED_CKPT=/home/rosmontis/Projects/dualsys/Xiaomi-Robotics-0/checkpoints/Xiaomi-Robotics-0-Calvin-ABCD_D \
OUTPUT_ROOT="$PWD/outputs" EXP_NAME=orchard_v1_calvin_cartesian_seed42 \
BATCH_SIZE=1 LR=1e-5 MAX_STEPS=3000 SEED=42 \
bash scripts/train_orchardbench.sh
```

额外 Hydra overrides 放在末尾；`DATA_CONFIG` 选择 data config，`ORCHARD_STATS`/`DATASET_ROOT` 覆盖路径；`PYTHON_BIN` 可指定另一个兼容训练 Python。launcher 开始时打印有效环境路径及完整 resolved Hydra config。

## 实际测试结果

| 检查 | 实测结果 / 证据 |
|---|---|
| 全集质量与视频 | PASS，300 JSON、600 视频、79,054 完整解码帧，`initial_audit.json` / `audit.json` |
| split / shape / finite / mask | 270/30；全体 30,827 窗口 raw/normalized finite、inactive zero；16 个随机样本图像实际解码，action/mask `[30,32]`，state `[1,32]`，恰好两个 image blocks |
| real trajectory round-trip | 300 episodes 多个窗口、所有 horizon；最大 xyz `1.67225e-7 m`、SO(3) `5.49541e-7 rad`、gripper `5.58794e-9 m`；容差 `1e-5 / 1e-5 / 1e-6` |
| bf16 模拟输出 round-trip | 每个 train episode 中间窗口，270 窗口；最大 `0.002876 m / 0.009007 rad / 0.000153 m`，容差 `.005 / .02 / .001`，`bf16_roundtrip.json` |
| stats | `[30,32]`，train only，inactive 0/1，重复生成 byte-identical，`stats_reproducibility.log` |
| CALVIN load | 933 keys strict PASS；不恢复 optimizer/scheduler，`model_smoke_final.json` |
| 官方训练入口 | 默认 2 个 data workers、batch=1、一个 forward/backward/AdamW step；loss `0.48395446`；219 个 gradient tensors finite；global step 0→1 |
| optimizer/scheduler | optimizer state entries 0→219，scheduler epoch 0→1；output weight 最大更新 `1.00732e-5`，更新后参数有限 |
| 显存 | 单步 peak allocated `13,458,257,920 bytes`（约 12.53 GiB）；不是 nvidia-smi 总占用 |
| 20 步诊断 | 一个真实固定窗口、固定 flow timestep/noise；loss `0.5455035→0.0407857`，max output weight change `0.000200778`，所有记录梯度有限；peak allocated 14,076,170,752 bytes；`overfit.json` |
| inactive loss/gradient | PASS，闲置预测 +10000 不改变 loss；对应输出梯度为零 |
| 训练后 checkpoint 推理 | strict load、两路真实图像、finite `[30,32]`、30 个 native commands；`inference.json`；未在模拟器执行这些策略输出 |
| 控制器宽度 API | `.06→.03/.03`、`.09→.04/.04` 并报告 clipping；旧 signed close→0/0；NaN 在控制前拒绝；`controller_smoke.json` |
| 静态检查 | `bash -n` 两个 launcher、两个仓库 `git diff --check` 通过 |

最初两个训练尝试在 Lightning 的 integer `val_check_interval=2000` 与只有 1 个 batch 的 smoke 配置不兼容处停止，尚未执行训练步；Orchard trainer 改为 epoch fraction `1.0` 并关闭 validation 后成功。日志保留，不将这些失败尝试计为通过。

## 已知限制与未做事项

1. **控制器限幅与采集速度并不相同。** 39,527 个唯一记录步骤中，9,619 个 translation 超过原有每轴 .02 m，8,615 个 Euler delta 超过 .05 rad，合并 11,360 个步骤至少一项超限。最大每轴位移约 .123/.098/.097 m，最大旋转约 .171/.168/.263 rad。403 个实测宽度超出 `[0,.08]`，实测范围 `[-.0102711,.0882895] m`。完整数值见 `controller_limits.json`。未裁剪/改写 demonstrations 或训练标签，未提高控制器限幅。round-trip 证明的是**安全裁剪之前**的物理目标和命令一致，不证明限幅后的轨迹可重放；还受 workspace bounds、IK、joint slew、接触及 zero-delta hold-last-setpoint 约束。后续闭环实验必须显式评估这些执行误差。
2. bf16 的毫米级位移/毫弧度量化误差与 float32 数学 round-trip 分开报告，不能用 float32 容差宣称实际 bf16 无损。
3. collector 有 expert assist；新接口测试不证明 contact-grasp policy success。没有闭环 success rate、正式 validation loss 或泛化结论。
4. 20 步诊断固定噪声和单窗口，证明可优化性，不是模型质量证据。3000 步、全 VLM 微调、generic pretrained fallback、多 GPU/DeepSpeed 均未运行。
5. 当前是新增 Orchard 专用本地 inference bridge；旧 `xr0_orchard_smoke.py` 仍明确是 CALVIN transport smoke，旧 generic server 仍是双臂路径。不要将 Orchard 权重/统计交给这些旧解码器。
6. 统计与 trained checkpoint 应成对保留；checkpoint 目录的 resolved config 指向 stats artifact，部署必须传相同契约的 stats。当前未实现模型打包发布。

## 文件范围

OrchardBench 新增 `treesim/orchard_action.py`、`scripts/test_orchard_action_env.py`、`TRAINING_NOTES.md` 指针；修改 `treesim/vla_env.py`（可选连续宽度入口）与 `treesim/xr0_adapter.py`（显式导出新 adapter）。

XR-0 新增：

- `mibot/data/datasets/orchardbench_dataset.py`、`mibot/data/datamodule/orchardbench_datamodule.py`
- `mibot/models/runner/orchard_runner.py`、`mibot/utils/orchard_checkpoint.py`、`mibot/server/orchard_policy.py`
- `configs/{data,model,trainer}/orchardbench.yaml`、`scripts/train_orchardbench.sh`
- `tools/compute_orchard_stats.py`、`tools/audit_orchardbench.py`、`tools/audit_orchard_precision.py`、`tools/smoke_orchard_model.py`、`tools/smoke_orchard_inference.py`
- `requirements-orchard.txt`、`.gitignore`（仅新增本地运行产物规则）、本说明及 `artifacts/orchardbench/` 的统计、schema、provenance、审计/测试结果。

XR-0 小型 integration hooks：`mibot/data/__init__.py` / `mibot/models/__init__.py` 注册；`XR0.py` 本地 config 构建；`cfg_utils.py` auto/DDP strategy；`scripts/train.sh` 使用选定 Python 执行 `torch.distributed.run`，避免继承环境的 torchrun shebang 错用 Python。原 `JsonDataset`、`io.py`、`BaseDataModule`、earphone/官方 trainer config 未改。

未改 expert、planner、tree/fruit/stem physics、camera pose、acceptance gates 或任何已采集 JSON/MP4。运行中的 checkpoint、venv、W&B offline files、Hydra outputs 和官方自动生成 `assets/config.py` 已忽略，未作为源码提交。
