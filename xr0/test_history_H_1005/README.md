# OrchardBench-VLA / XR-0 H 可观测因果 proprio 历史实验

本轮先完成 R03 step4000→8000 的真实 optimizer continuation，再依据后段闭环行为决定 H 的统一 4k/8k 预算。H0/Hrepeat/Hhistory 各自从正式 B8000 开始、fresh AdamW、同 schedule 行顺序和 seed42；不使用 fresh24，不重新把已消费的 heldout12 当 endpoint selection set。

执行协议保持 full30 / 30-target chunk / reach-conditioned / 10mm / 0.08rad / dwell30 / budget900 / benchmark_assist / Euler5。production physics/controller/IK/action/state projector 权重形状均不修改，VLM 保持冻结，两路当前 RGB、prompt 与预处理保持原样。

H 只替换 loader 与 inference 的 state tensor：H0 `[s_t]`，Hrepeat 四份逐值相同当前状态，Hhistory `[s_(t-15), s_(t-5), s_(t-1), s_t]`。时间单位是真实30Hz control step；初期使用已经实际出现的最早 observation padding。Recovery 负 local frame 索引从对应的已保留 student prefix 实测 TCP/width/joints 提取，local≥0 使用实测 teacher proprio；不插值、不伪造 reset。历史 debug 容器只抽取这四类 sensor 字段，所有 fruit/phase/held/category 元数据都不进入 batch。

部署通过本实验的 `ObservedEnv` 只读 hook 调用 unchanged `super.step` 后更新 ring buffer，每个控制步一次，不改变动作执行。训练与部署 token 顺序一致，动态 mask/RoPE 使用原 XR0 能力，无新增网络参数。

`causal_history_check.json`：24 条 recovery prefix 可用，排除0 anchor；`batch_check.json`：一次真实 recovery frame0 collated batch 检查。R03-8k evaluation 后额外进行一个30步输入路径 smoke，单独记为 smoke、不计闭环成绩。各正式训练只进行一次实际 batch generation 后立刻训练。

`late_offline.py` 固定使用原24条 accepted recovery 各1个 TRANSPORT→DROP 前窗口及1个 DROP 窗口，共48个，seed42/43/44；仅作为训练来源机制诊断。原240窗口 val panel 没有 DROP targets，因此不能将此补充诊断称为验证集 DROP 泛化。训练来源、dev8（train2+val6）、历史单场景双seed、最终泛化分别报告。

**已完成**：R03 continuation及H0/Hrepeat/Hhistory全部训练与闭环；未运行H4。主结果在 [final_report.md](final_report.md) / [summary.json](summary.json)。最终进度见各条件 `training/status.json`、`evaluation/status.json`。所有 checkpoint、大视频、逐控制步 traces 留本地且忽略 Git。实际命令与PID保存在 receipts。中断与修复见 `engineering_repairs.jsonl`，不删除失败病例，不择优替换历史闭环。

## Git 归档范围

提交训练/推理/评估脚本、固定 schedule 与 recovery manifest、配置、checkpoint 路径和 SHA256、实际命令 receipts、loss 摘要、逐病例指标、预算/停止决定与最终报告。根目录的 `data_audit.json`、`recovery_manifest.jsonl` 是 `setup_condition.py` 的输入，随各条件的冻结副本一起保留。失败 continuation 尝试只提交 run manifest/status 和修复记录，其逐 update 日志留本地。CSV 保留原始 CRLF 字节。

Checkpoint/optimizer resume、视频/图像、逐控制步 rollout、逐 update 日志、离线逐窗口预测与完整原始报告、cache/tmp/pycache 均不提交；不删除本地实验产物。数据池、B8000、R03 step4000 resume 和 OrchardBench 依赖继续使用配置记录的已有本地路径，Git 归档不包含这些外部数据。

临时启动与报告生成 helper 的执行版本分别归档为 `launch_job.py` / `write_final_report.py`，不依赖 `/tmp` 脚本文件。`assemble_summary.py` 重新汇总需要本地完整 offline report；已提交的 `summary.json` 保留本轮选定离线聚合证据，`write_final_report.py` 可据此重新生成报告。各次实际运行参数见 `*_receipt.json`。
