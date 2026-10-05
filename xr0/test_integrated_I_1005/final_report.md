# OrchardBench-VLA / XR-0 Integrated Training

本轮已结束。真实 R03-8k live late-state supervision 已补齐，但三轮 S8/J8/J16 的合格 strict placement 均为0；停止以“再补一点局部数据”为主的 patch-style continuation。未形成可冻结的完整任务 endpoint，fresh24 保持未使用。

完成 3/3 轮方案对照、9个 endpoint 闭环；本轮实际新增 72000 optimizer updates。三轮全部停止，不开启第四轮。

## 1. Live student → teacher 数据闭环

实际 student source rollouts 144；candidate 124；accepted 72；rejected 52；accepted unique source episodes 36。

| late-state | candidate | accepted |
|---|---:|---:|
| transport_stall | 75 | 49 |
| near_bucket_no_release | 5 | 2 |
| premature_release_precursor | 44 | 21 |

Rejected 原因：`{"teacher_budget": 48, "engineering_exception": 1, "strict_but_unreasonable_release": 1, "collection_interrupted_for_teacher_motion_repair": 1, "interrupted_for_collection_suffix_budget_repair": 1}`。原 observer teacher strict 73；通过合理释放与全部数据门槛的 accepted 72，两者不混用。

唯一 mining student 为 current-only R03-8k。Student 从 reset 按原 full30 reach-conditioned production 路径执行，teacher 直接接管同一个正在运行的 env；prefix regeneration 不参与获得 suffix。Teacher 只初始化私有规划 bookkeeping/IK，初始化前后真实 physics digest 相等；命令经过原 OrchardActionAdapter 和 env.step。Teacher 通过实测关节反馈限制 native TCP 目标，并在桶附近修正实际持果偏移；真实释放仍由 production sustained-open gate 触发。没有 teleport、写 simulator state、伪造 held 或改 controller/physics。

每条 accepted 保存完整双 RGB、实测 proprio、student prefix/chunks、takeover step、teacher suffix/action labels、source checkpoint 和诊断真值。冻结时逐条核对 prefix 最后时间戳/proprio 等于 suffix 第一个观测，prefix-end digest 等于 teacher-init digest，label 为下一步实测状态，timestamps 为30Hz。接受要求 same-fruit held15→detach→合理释放→non-held→stable bucket60，并通过原 physical health gate；失败/中断/invalid attempts 留档但不训练。

工程修复前3条 accepted 在原900总预算内完成；修复后69条从reset配置总预算1800，student最多900、teacher suffix最多900。接管时不修改 env.done/state，两个版本都通过同一物理与合理释放门槛。正式 student evaluator始终900。必要工程修复见 engineering_repairs.jsonl，collector 各版本源码保留并按每条 provenance 记录 SHA。

新数据与旧 teacher-continuation 的区别是起点来自当前 R03-8k 实际到达的 late-state，接管后仍在同一真实 env。旧数据主要是既有成功 recovery 的后段，不提供这种当前闭环 late-state 监督。Near-bucket/no-release 自然仅接受2条，覆盖有限；保持真实 failure distribution，不人工凑类别比例。

Simulator truth 只用于 mining、分类/接管、teacher、审计和诊断。Student loader 仍只形成原 prompt、当前两路 RGB、当前 proprio 和原 action/mask；fruit/bucket xyz、velocity、held/detach/phase、apple ID、seed/debug metadata、history 均不进入 policy input。

## 2. D*、provenance 与冻结 schedule

D* trajectories：`{"original": 128, "existing_R": 24, "live_late": 72}`。Original 为既定128个训练 episodes 的全部合法 full30 task windows，覆盖 early/grasp-pull/transport/drop-done；不是旧 B quotas，也不声称纳入原2650 episodes 的全部数据。Existing R 为已通过原 oracle 的24条真实 R1/R2/R3 student-state recovery onset；live late 为本轮72条合格同-env suffix。

Available full30 windows：`{"original_early": 922, "original_grasp_pull": 7147, "original_transport": 14632, "original_drop_done": 512, "existing_R1": 360, "existing_R2": 180, "existing_R3": 180, "live_onset": 2160, "live_transport": 36510, "live_release": 2520, "live_settle": 2509}`。数据/media/annotations/accepted prefix/code/action-stats/checkpoint hashes 见 frozen.json；manifest 见 dataset_manifest.jsonl。保护 dev/val/historical auxiliary/fresh24 等 mining exclusion；dev train2 仍是原训练数据中的已消费开发 case。

| checkpoint | SHA256 |
|---|---|
| original full10k | `fdb580fe60d4cdeb1d35b2a7ce0ec2d27798396aa717c23d9648866a09f9d109` |
| historical B8000 | `c2a5c22352b916989cae7026ff9af6e06ce8129cf3611a128930f517f8838c31` |
| R03-8k mining | `767ab133d24ac6d024c259e9b773aaca84dc9bd4c5b5a74d43d5d1671e309138` |

B8000 来源为 test_128_episode_AB_1003/arms/B/checkpoints/step_8000_trainable.pt，由 original full10k 继续8000更新得到；其旧 quota 为 reset1600、first1_4=800、REACH1200、GRASP2000、PULL1200、TRANSPORT800、DROP400。本轮不继承该前段偏重采样，只比较其权重 warm-start。

Round1 original/R/live-late=60/15/25；original内部25/35/25/15；live内部onset/transport/release/settle=20/30/35/15。D_A/D_B各8000，seed100501/100502，模型启动前冻结。后续共同分布修订使用独立轮次目录，不覆盖这些原 schedule 或 D*。

## 3. S/J 实验与统一闭环

所有轮次 S 从原 full10k→历史 B8000 trainable weights 开始，fresh AdamW；J 从原 full10k 开始，fresh AdamW，D_A→D_B连续 optimizer/RNG并保留 J8/J16。每轮新增预算 S8=8000、J8=8000、J16=16000；full10k之后累计 S8 与 J16 均约16000。不同轮次不从上轮 endpoint 打补丁。

冻结 VLM BF16，非 VLM master FP32；current two RGB/current proprio、原 prompt/action encoding/action stats、full30 active7 normalized flow coefficient0.5、Euler5、batch1/acc1、AdamW betas(0.9,0.95)/wd0.1/eps1e-8/clip1、同 train seed42、同900-step closed-loop协议均保持。lr默认1e-5，实际每轮 config/receipt/run_manifest/checkpoint SHA/source loss 独立记录。

原 StrictPlacement observer 与 reach loop 保持不变。原结果记为 strict_original_observer；按本轮完整链要求，主 strict 另外核对该原成功链的实际 release 在既有合理区域（fruit-bucket distance≤0.15m且 bucket XY 内）。落桶但 release 超界不晋级。原 raw metrics、qualified booleans、release距离和修正后的诊断标签分别保留，不能把原 observer 结果隐藏或替代主 strict。Dev8 与历史辅助都是已消费开发诊断，不是 fresh test。

### Round 1

首版60/15/25和既定8k/16k预算，严格执行S8/J8/J16。

| endpoint | updates | original/R/live | first500→last500 original/R/live loss | checkpoint SHA256 |
|---|---:|---|---|---|
| S8 | 8000 | 4800/1200/2000 | 0.1378→0.1260 / 0.2235→0.1069 / 0.2846→0.1345 | `a819a9ec7638d519028649854a828b585ebc9234e5fef0d8c1008f68d3800127` |
| J8 | 8000 | 4800/1200/2000 | 0.1973→0.1504 / 0.1878→0.1122 / 0.2527→0.1346 | `faeef8542a73e0f4fda1dd590965c63f511b6f1df027ad3b52fa3b58e108542e` |
| J16 | 16000 | 9600/2400/4000 | 0.1973→0.1110 / 0.1878→0.0966 / 0.2527→0.1091 | `ce4b93411a08fa78a9bdbf33ab42471e4f2cabaea5e2ac0fac3625a11dbf7c42` |

Loss 为训练头/尾各500次 optimizer updates 内按 source 分组的均值；不是事件成功率。所有6个实际训练作业的 frozen VLM 参数 hash 前后一致，详见各 training_summary.json。

- S8 checkpoint：`/home/rosmontis/Projects/dualsys/Xiaomi-Robotics-0/xr0/test_integrated_I_1005/round1/S/training/checkpoints/step_8000_trainable.pt`。
- J8 checkpoint：`/home/rosmontis/Projects/dualsys/Xiaomi-Robotics-0/xr0/test_integrated_I_1005/round1/J/training/checkpoints/step_8000_trainable.pt`。
- J16 checkpoint：`/home/rosmontis/Projects/dualsys/Xiaomi-Robotics-0/xr0/test_integrated_I_1005/round1/J/training/checkpoints/step_16000_trainable.pt`。

- D_A：`schedule_D_A.jsonl`，SHA256 `c0c78d8e94068b975eee6030b415c19f88ec113f9a7d9220719d58bf2029a905`。
- D_B：`schedule_D_B.jsonl`，SHA256 `78730c63cfe940a6ede82f1b2c1dd1085a05cd6d0ee6d4bd2e46cfddc672cd75`。

实际执行 S/J D_A 对齐：`{"actual_D_A_updates": 8000, "identical_S_J_executed_rows": true, "compared_fields": ["step", "source", "group", "window_id"]}`。

| cohort / endpoint | held15 | detach | bucket-neighborhood | reasonable-release | stable-bucket | strict | long-held-stall | premature-release | physical-anomaly | raw observer strict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dev8 / S8 | 7/8 | 7/8 | 0/8 | 0/8 | 1/8 | 0/8 | 3/8 | 3/8 | 0/8 | 1/8 |
| dev8 / J8 | 5/8 | 5/8 | 0/8 | 0/8 | 0/8 | 0/8 | 4/8 | 1/8 | 0/8 | 0/8 |
| dev8 / J16 | 5/8 | 5/8 | 2/8 | 0/8 | 2/8 | 0/8 | 2/8 | 3/8 | 0/8 | 2/8 |
| historical_seed42_43 / S8 | 2/2 | 2/2 | 0/2 | 0/2 | 0/2 | 0/2 | 2/2 | 0/2 | 0/2 | 0/2 |
| historical_seed42_43 / J8 | 2/2 | 2/2 | 0/2 | 0/2 | 0/2 | 0/2 | 1/2 | 1/2 | 0/2 | 0/2 |
| historical_seed42_43 / J16 | 1/2 | 1/2 | 1/2 | 0/2 | 1/2 | 0/2 | 0/2 | 1/2 | 0/2 | 1/2 |

S8 vs J8：strict 0/8 vs 0/8；held15 7 vs 5；arrival 0 vs 0；reasonable release 0 vs 0。

S8 vs J16：strict 0/8 vs 0/8；held15 7 vs 5；arrival 0 vs 2；reasonable release 0 vs 0。

Per-case 原始 steps/chunks/RGB/metrics、cases.json/csv、aggregate.json 与 unified_cases/closed_loop_summary 保存在本轮目录；辅助距离与 event 只解释失败，不替代 strict。

### Round 2

共同 source ratios：`{"original": 0.55, "existing_R": 0.15, "live_late": 0.3}`；original phase：`{"early": 0.25, "grasp_pull": 0.4, "transport": 0.25, "drop_done": 0.1}`；live phase：`{"onset": 0.15, "transport": 0.35, "release": 0.4, "settle": 0.1}`。独立冻结 schedules，seed：`{"D_A": 100511, "D_B": 100512}`；lr=1e-05。

修改依据：Round1 三个 endpoint strict 均0：S8 held7但停滞/释放失败，J8/J16 held5，J16 arrival2/stable2但合理释放0。保持 existing R 比例，对两条路线共同小幅提高 original grasp 与 live transport/release exposure；D* 不增加样本，不按 scene 打补丁。

| endpoint | updates | original/R/live | first500→last500 original/R/live loss | checkpoint SHA256 |
|---|---:|---|---|---|
| S8 | 8000 | 4400/1200/2400 | 0.1437→0.1200 / 0.2399→0.1188 / 0.2838→0.1396 | `b11d7b253bd3730c1edf9e89cc09d4ceddd951a25395bdac23104526532a330e` |
| J8 | 8000 | 4400/1200/2400 | 0.2153→0.1492 / 0.2073→0.1189 / 0.2654→0.1400 | `d91c364029c3753e98f83977f1744b0d914354c36c61bdbe43bbae1a375da266` |
| J16 | 16000 | 8800/2400/4800 | 0.2153→0.1239 / 0.2073→0.1219 / 0.2654→0.1141 | `65a85d6113f4e230afc5af151250cb36e16efa5ac8e8c29c3dbb73539bfa2004` |

Loss 为训练头/尾各500次 optimizer updates 内按 source 分组的均值；不是事件成功率。所有6个实际训练作业的 frozen VLM 参数 hash 前后一致，详见各 training_summary.json。

- S8 checkpoint：`/home/rosmontis/Projects/dualsys/Xiaomi-Robotics-0/xr0/test_integrated_I_1005/round2/S/training/checkpoints/step_8000_trainable.pt`。
- J8 checkpoint：`/home/rosmontis/Projects/dualsys/Xiaomi-Robotics-0/xr0/test_integrated_I_1005/round2/J/training/checkpoints/step_8000_trainable.pt`。
- J16 checkpoint：`/home/rosmontis/Projects/dualsys/Xiaomi-Robotics-0/xr0/test_integrated_I_1005/round2/J/training/checkpoints/step_16000_trainable.pt`。

- D_A：`round2/schedule_D_A.jsonl`，SHA256 `39785a561df69634ffb1c88deda4939aadf5370e93a6df94f142c9760c35528f`。
- D_B：`round2/schedule_D_B.jsonl`，SHA256 `5aaa3778e0ccea16c29c5cf88386d0e4c64c470ef1e1021be30f7281fed3a1c7`。

实际执行 S/J D_A 对齐：`{"actual_D_A_updates": 8000, "identical_S_J_executed_rows": true, "compared_fields": ["step", "source", "group", "window_id"]}`。

| cohort / endpoint | held15 | detach | bucket-neighborhood | reasonable-release | stable-bucket | strict | long-held-stall | premature-release | physical-anomaly | raw observer strict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dev8 / S8 | 7/8 | 7/8 | 0/8 | 0/8 | 0/8 | 0/8 | 6/8 | 0/8 | 0/8 | 0/8 |
| dev8 / J8 | 7/8 | 7/8 | 0/8 | 0/8 | 0/8 | 0/8 | 6/8 | 0/8 | 0/8 | 0/8 |
| dev8 / J16 | 4/8 | 4/8 | 0/8 | 0/8 | 0/8 | 0/8 | 3/8 | 1/8 | 0/8 | 0/8 |
| historical_seed42_43 / S8 | 2/2 | 2/2 | 0/2 | 0/2 | 0/2 | 0/2 | 2/2 | 0/2 | 0/2 | 0/2 |
| historical_seed42_43 / J8 | 2/2 | 2/2 | 0/2 | 0/2 | 0/2 | 0/2 | 1/2 | 1/2 | 0/2 | 0/2 |
| historical_seed42_43 / J16 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |

S8 vs J8：strict 0/8 vs 0/8；held15 7 vs 7；arrival 0 vs 0；reasonable release 0 vs 0。

S8 vs J16：strict 0/8 vs 0/8；held15 7 vs 4；arrival 0 vs 0；reasonable release 0 vs 0。

Per-case 原始 steps/chunks/RGB/metrics、cases.json/csv、aggregate.json 与 unified_cases/closed_loop_summary 保存在本轮目录；辅助距离与 event 只解释失败，不替代 strict。

### Round 3

共同 source ratios：`{"original": 0.55, "existing_R": 0.15, "live_late": 0.3}`；original phase：`{"early": 0.25, "grasp_pull": 0.4, "transport": 0.25, "drop_done": 0.1}`；live phase：`{"onset": 0.15, "transport": 0.35, "release": 0.4, "settle": 0.1}`。逐字节复用 Round2 schedules，不重新采样，seed：`{"D_A": 100511, "D_B": 100512}`；lr=5e-06。

修改依据：Round2 S8/J8 held15=7/8、J16降至4/8，三者strict均0，尽管 original/live loss下降。共同 lr 从1e-5降为5e-6，检验较小更新是否保留抓取并拟合 late监督；完全复用 Round2 样本顺序、预算、初始化与其他参数。此为最后允许的一轮，不声称已证明失败由学习率导致。

| endpoint | updates | original/R/live | first500→last500 original/R/live loss | checkpoint SHA256 |
|---|---:|---|---|---|
| S8 | 8000 | 4400/1200/2400 | 0.1290→0.1063 / 0.2483→0.1160 / 0.3193→0.1447 | `496349b68909b725df877dc2b5807dac40650eb9fca39029aa658b4c5e187170` |
| J8 | 8000 | 4400/1200/2400 | 0.1876→0.1318 / 0.2046→0.1196 / 0.2714→0.1452 | `201cbe3978e6b04f2fe33e7f2663f2e291406e71ab00e51c5985bcd4570d6f2a` |
| J16 | 16000 | 8800/2400/4800 | 0.1876→0.1217 / 0.2046→0.1225 / 0.2714→0.1201 | `dedd42cc66403f04003e702ef07148060663f09f1d7e4763338431a4c9322b20` |

Loss 为训练头/尾各500次 optimizer updates 内按 source 分组的均值；不是事件成功率。所有6个实际训练作业的 frozen VLM 参数 hash 前后一致，详见各 training_summary.json。

- S8 checkpoint：`/home/rosmontis/Projects/dualsys/Xiaomi-Robotics-0/xr0/test_integrated_I_1005/round3/S/training/checkpoints/step_8000_trainable.pt`。
- J8 checkpoint：`/home/rosmontis/Projects/dualsys/Xiaomi-Robotics-0/xr0/test_integrated_I_1005/round3/J/training/checkpoints/step_8000_trainable.pt`。
- J16 checkpoint：`/home/rosmontis/Projects/dualsys/Xiaomi-Robotics-0/xr0/test_integrated_I_1005/round3/J/training/checkpoints/step_16000_trainable.pt`。

- D_A：`round3/schedule_D_A.jsonl`，SHA256 `39785a561df69634ffb1c88deda4939aadf5370e93a6df94f142c9760c35528f`。
- D_B：`round3/schedule_D_B.jsonl`，SHA256 `5aaa3778e0ccea16c29c5cf88386d0e4c64c470ef1e1021be30f7281fed3a1c7`。

实际执行 S/J D_A 对齐：`{"actual_D_A_updates": 8000, "identical_S_J_executed_rows": true, "compared_fields": ["step", "source", "group", "window_id"]}`。

Round2→Round3 的实际训练对照：`{"S": {"updates": 8000, "identical_executed_rows": true, "identical_initialization_and_other_configuration": true, "fresh_optimizer": true, "previous_lr": 1e-05, "current_lr": 5e-06}, "J": {"updates": 16000, "identical_executed_rows": true, "identical_initialization_and_other_configuration": true, "fresh_optimizer": true, "previous_lr": 1e-05, "current_lr": 5e-06}}`。这里只验证变量控制，不推断学习率是失败原因。

| cohort / endpoint | held15 | detach | bucket-neighborhood | reasonable-release | stable-bucket | strict | long-held-stall | premature-release | physical-anomaly | raw observer strict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dev8 / S8 | 6/8 | 6/8 | 0/8 | 0/8 | 0/8 | 0/8 | 5/8 | 0/8 | 0/8 | 0/8 |
| dev8 / J8 | 6/8 | 6/8 | 0/8 | 0/8 | 0/8 | 0/8 | 4/8 | 2/8 | 0/8 | 0/8 |
| dev8 / J16 | 5/8 | 5/8 | 0/8 | 0/8 | 0/8 | 0/8 | 3/8 | 2/8 | 0/8 | 0/8 |
| historical_seed42_43 / S8 | 2/2 | 2/2 | 0/2 | 0/2 | 0/2 | 0/2 | 1/2 | 1/2 | 0/2 | 0/2 |
| historical_seed42_43 / J8 | 2/2 | 2/2 | 0/2 | 0/2 | 0/2 | 0/2 | 1/2 | 1/2 | 0/2 | 0/2 |
| historical_seed42_43 / J16 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |

S8 vs J8：strict 0/8 vs 0/8；held15 6 vs 6；arrival 0 vs 0；reasonable release 0 vs 0。

S8 vs J16：strict 0/8 vs 0/8；held15 6 vs 5；arrival 0 vs 0；reasonable release 0 vs 0。

Per-case 原始 steps/chunks/RGB/metrics、cases.json/csv、aggregate.json 与 unified_cases/closed_loop_summary 保存在本轮目录；辅助距离与 event 只解释失败，不替代 strict。

### 原 observer 落桶但释放不合格的实际事件

| round / endpoint | scene seed / inference seed | min held distance (m) | exact successful-chain release distance (m) | bucket XY | reasonable region |
|---|---|---:|---:|---|---|
| Round1/S8 | 2012485 / 42 | 0.2131 | 0.2081 | True | False |
| Round1/J16 | 2010914 / 42 | 0.2057 | 0.1999 | True | False |
| Round1/J16 | 2014362 / 42 | 0.1200 | 0.1581 | True | False |
| Round1/J16 | 2010600 / 42 | 0.1479 | 0.1506 | True | False |

min held distance 是整条 rollout 的辅助最小值，不能替代实际 release 时的空间条件或完整 strict chain。

## 4. 路线结论、能力交换与停止规则

分类：Case D。真实 R03-8k live late-state supervision 已补齐，但三轮 S8/J8/J16 的合格 strict placement 均为0；停止以“再补一点局部数据”为主的 patch-style continuation。未形成可冻结的完整任务 endpoint，fresh24 保持未使用。

S8 vs J8：Round1 的 held15/detach 为7/8 vs5/8，B warm-start有前段优势；Round2为7/8 vs7/8，Round3为6/8 vs6/8，优势不稳定，所有主strict均0/8。S8 vs J16：Round1 held7/8 vs5/8，但J16 arrival2/8、stable bucket2/8，均因合理释放不成立而主strict0；Round2 held7/8 vs4/8，Round3 held6/8 vs5/8，主strict仍全0。相同新增8k预算和full10k后累计约16k预算两种比较，都没有形成B curriculum在完整任务上的可靠收益，也没有证明direct integrated已成功。按完整strict停止规则选择Case D，不把零成功的并列包装为Case B/C。

存在局部能力交换：Round1 S保留较多抓取，J16部分运果/落桶进步但释放不合格；Round2提高共同grasp与late transport/release曝光，使J8 held从5升至7，但S/J8 long-held stall都增至6，先前部分到桶/落桶消失。J继续16k时grasp降至4且loss继续下降。Round3唯一共同lr减半后S/J8 held均6、J16 held5；相较Round2 J16抓取仅回升1个case，仍弱于其J8，历史辅助J16仍无held。Round3 dev premature-release 为S0/J8 2/J16 2，long-held stall为5/4/3；stall计数降低同时伴随held不足或错误release，不能解释为完整transport成功。九个endpoint合理释放均0，三轮所有source loss头尾均下降，却未与主strict改善同步。正式闭环physical-anomaly全0。以上是反复消费小dev cohort的描述，不能当统计显著性或泛化证明。

本轮已排除或显著削弱的具体工程解释：训练数据完全没有来自当前R03真实late-state的监督（已新增72条/36个source）；teacher suffix必须依赖reset后prefix regeneration（实际同env接管）；接管通过teleport、伪造held、直接写physics/controller获得成功（接管digest与真实native命令边界已检查）；S/J新增8k比较使用不同D_A或复用B optimizer momentum（逐更新样本相同、fresh AdamW）；Round3差异来自新数据/新sampling（逐字节复用并核对实际样本）；history输入或多个mining checkpoints混杂（仅current-only R03）；production physics/controller/action adapter更改或已记录physical-anomaly主导失败（生产文件无改动、正式诊断均0）。这些结论只针对本轮已核查的路径，不能排除所有模拟器或建模问题。

尚存的合理解释：当前两路单帧RGB+proprio可能不足以识别持果偏移、遮挡、速度/动态与release时机，而truth teacher可以利用这些信息；frozen VLM/action-head-only容量或表征不足；normalized flow/Euler5动作拟合与离散release及稳定placement事件目标不一致；本轮自然near-bucket/no-release仅2条、固定R03 mining分布不能覆盖训练后S/J全部新偏差；original task pool为既定128个episodes而非全部2650，任务多样性仍有限；有限训练预算下仍可能未充分拟合关键事件，loss下降不能单独排除underfit。数据类型缺口已补，但不能由本轮证明数据覆盖充分或某一深层假说成立。停止patch-style主线，本轮不实现新观测/history、解冻VLM、RL、目标函数改造等后续方向。

## 5. 最终 endpoint 与 fresh24

没有值得进入 fresh24 的 endpoint；fresh24 未运行。

没有满足完整 strict 的可晋级 endpoint，按用户规则不运行 fresh24；其冻结cohort未用于mining、checkpoint选择、调参或本轮恢复数据。

fresh24只允许最终冻结 endpoint一次，不参与 checkpoint选择、数据调整、mining或后续训练。Round3之后停止，不开启Round4；本轮不实现 history、新观测、解冻VLM、RL或下一层研究方向。Production controller/physics/action adapter/原严格 observer未修改，未重跑Gate0/Gate1。

工程与复现入口：collect_live.py、inputs.py、freeze_data.py、train_integrated.py、evaluate.py、Round2/3的有限修订脚本及config、frozen schedule、receipts、engineering_repairs.jsonl、逐步training_steps和统一closed_loop_summary。
