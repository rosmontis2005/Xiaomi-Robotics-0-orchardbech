# R student-state recovery：三版训练与闭环汇总

完成原始 R01 及两次额外 R 训练—闭环验证。每版各4000次真实 optimizer update，分别保存数据审计、冻结 schedule、checkpoint、离线预测、闭环步骤和视频。测试前按 dev8 strict 规则选择 R01，之后仅做一次 B/R 配对 heldout 验证。

## 结果

| 模型/版本 | dev8 held15 | dev8 detach | dev8 release | dev8 strict | 辅助1场景×2seed strict |
|---|---:|---:|---:|---:|---:|
| B8000 | 6/8 | 6/8 | 4/8 | 0/8 | 0/2 |
| [version_01](version_01/report.md) | 5/8 | 5/8 | 2/8 | 1/8 | 1/2 |
| [version_02](version_02/report.md) | 5/8 | 5/8 | 2/8 | 0/8 | 0/2 |
| [version_03](version_03/report.md) | 7/8 | 7/8 | 1/8 | 0/8 | 1/2 |

| 最终冻结评测 | B held15 | R01 held15 | B strict | R01 strict |
|---|---:|---:|---:|---:|
| heldout12 | 8/12 | 8/12 | 0/12 | 1/12 |
| remaining11 | 7/11 | 7/11 | 0/11 | 0/11 |
| previous_auxiliary1 | 1/1 | 1/1 | 0/1 | 1/1 |

最终完整配对病例、放置事件和置信区间见 [final_holdout12/report.md](final_holdout12/report.md) 与 [JSON](final_holdout12/report.json)。dev8 含2个原训练场景和6个验证场景；两个训练场景没有进入 recovery mining。辅助结果只有一个场景，不能按两个独立测试场景解释。heldout12 是历史挑选过的 oracle replay-PASS val cohort；其中2010600曾做辅助分析，remaining11另报。fresh24 完全未运行、未用于训练或选择。

## 三版数据改动与训练

| 版本 | 原B | R1 | R2 | R3 | teacher-continuation | 可用 recovery windows | 实际unique | 重复抽样 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| version_01 | 2000 | 1000 | 500 | 500 | 0 | 720 | 663 | 1337 |
| version_02 | 3000 | 500 | 250 | 250 | 0 | 401 | 306 | 694 |
| version_03 | 2800 | 500 | 200 | 200 | 300 | 593 | 411 | 789 |

R01 保持首次4000配比；仅使用takeover后前30个anchor，却发现R1曝光863/1000已是PULL，R2/R3未来targets全为TRANSPORT。R02 将原B增到75%，R1仅选真实未held、未detached的近接触anchor；保持第一版30-anchor上限，未更换label。R03 保留这些真实恢复起点，并添加7.5%单独标记的 teacher-continuation 稀疏放置桥接窗口；它们来自原24条成功recovery的后段，不冒充新student-state样本。其未来30targets增加4594个DROP和2110个DONE。后段window放宽首轮anchor限制的理由和检查在 [version_03/design.json](version_03/design.json)；这属于用户本轮授权的数据修改。

原pilot实际尝试29个unique候选、32次attempt（含3次工程重试）：R1 12/12接受，R2 6/6接受，R3 6/11接受、5拒绝。5个拒绝中4个在前缀再生成检查中失败、teacher未开始，1个teacher超出恢复预算。因此实际运行teacher的是25个unique候选，24个成功并通过oracle；不能把前缀失败解释为teacher恢复失败。接受数据覆盖12个unique训练来源，24/24 teacher strict与当前student-controller oracle全轨迹PASS。未以测试场景补数量。R02/R03继续复用这24条固定成功轨迹；新当前R晚期on-policy candidate共4个因同态前缀再生成不一致被拒绝，teacher/oracle未建立，接受0；未加入训练、未放宽gate。两个阈值批次以及每个错误的traceback均在 [onpolicy_R3](onpolicy_R3/candidate_decisions.jsonl)。

所有三版从 original full10k → 正式B8000 step8000 overlay 开始，而不是从上一版R串行继续。逐一校验219个FP32 trainable tensors初始值与B完全一致；fresh AdamW、lr1e-5、betas(0.9,0.95)、wd0.1、eps1e-8、clip1、batch1、acc1、constantLR；frozen VLM BF16、BF16 autocast、原full30 active7 flow loss，未加入L权重。原action_stats始终不变，未clip或重算归一化。R03扩展label的最高active normalized绝对值约4.10，>5计数0；这只是outlier审计，不是对label做阈值处理。模型只读取两路RGB/current proprio/原prompt；teacher与debug truth未进入model batch。

## 离线误差与机制证据

| 模型 | old-val xyz mm | new-val xyz mm | val reset xyz mm | val GRASP xyz mm | val PULL xyz mm | val TRANSPORT xyz mm |
|---|---:|---:|---:|---:|---:|---:|
| B8000 | 94.78 | 72.80 | 40.99 | 97.92 | 57.68 | 212.50 |
| version_01 | 86.00 | 69.52 | 109.64 | 58.14 | 38.05 | 214.05 |
| version_02 | 62.34 | 47.66 | 35.66 | 39.75 | 31.13 | 211.96 |
| version_03 | 66.74 | 49.66 | 42.09 | 38.59 | 33.41 | 222.68 |

离线使用原冻结240-window panel，80 old_train/80 old_val/80 new_val，inference seeds42/43/44，30targets；各目标和noise seed相关，均值不是独立成功率。R01 recovery-frame0在训练来源的R2位置误差259.6→94.2mm、R3 275.0→129.9mm；这只证明局部监督被学到。真实中途接管诊断6个可比较状态：B与R01均strict0/6。两条R2最低bucket距离由B0.827/0.682m改善为R0.095/0.112m，但均未release。第7个seed43接管因width误差0.1286mm超过0.1mm gate保留为unavailable，未算模型失败。源episode是训练来源，不计泛化结果。

R02修复reset/REACH离线退化，但dev8 strict并未改善；R03提升抓持且补入真实成功teacher后段label，仍不能可靠完成运输—释放—稳定留桶。因此不能由单步loss或抓持成功直接宣布R达成整任务目标。另一方面，样本少、late-current-R样本未成功建立，结论也不足以否定更完整的交互式R数据路线。

## 执行与可复现记录

- version_01: 4000 updates，训练PID/session=432423/432423；step4000 SHA `a929fd15078b6cc72881c200b43ae4d26425a021f1346627414b08723a7a1a7d`；schedule SHA `f27c741be8bbe65fdca5d30ab03bf3dae0fd4d1ce71f738d5959b6449d58c548`。
- version_02: 4000 updates，训练PID/session=15141/15141；step4000 SHA `c8e2e3605850682f84113a4e74cf4172ece91ba8719e5d3c89458aadde786766`；schedule SHA `27fb419613bdd1ae73fb58ad8119e0fe3b27cd19008c10f1eb61c3cb8f6cf387`。
- version_03: 4000 updates，训练PID/session=36719/36719；step4000 SHA `5890b0d08f6e396299a47295f2e8b3987aafb477e7bfd571287d875525cb9575`；schedule SHA `3baf11beb93955b10d8c1e1d634707e90553878a23feb03ca55c146fac49258b`。

正式B8000 SHA `c2a5c22352b916989cae7026ff9af6e06ce8129cf3611a128930f517f8838c31`，full10k SHA `fdb580fe60d4cdeb1d35b2a7ce0ec2d27798396aa717c23d9648866a09f9d109`。

各版原始training status中 post_training_evaluation=false 是训练进程自身的字段；本轮另起独立evaluation进程，完成后的status/metrics位于各版evaluation目录。原第一轮的停止点报告保持原样，不覆盖成这次结果。R01原训练状态/步骤/manifest快照在 version_01/original_training_snapshot，checkpoint用哈希引用。

最终闭环PID/session=57220，启动命令与日志见 [final_holdout_receipt.json](../final_holdout_receipt.json)，最终status=COMPLETE24。各额外训练和evaluation命令在独立version目录的receipt与queue记录中。

production controller/action/physics、正式B数据与归一化的哈希保护通过；dev对照使用原冻结D0 B traces，10个reset全部精确一致，GT同条件dev8历史strict8/8；本轮没有重复全部旧Gate。最终B与R都重新执行，12个reset精确配对。所有完成文件SHA检查通过，observer/summary不一致计数为0。

工程失败全部保留：首次CPU helper导入修复发生在任何闭环开始前；中途接管不可再生案例保留为unavailable；晚期composite prefix4次失败未产生label；版本02规划中的可选takeover诊断没有执行，不计为该版证据。辅助seed43的legacy summary字段硬编码42，但实际BoundSeedPolicy调用、step trace与protocol均为43；每版seed_audit.json逐调用验证PASS，原始summary保留并明确解释。

## 可重复性限制

辅助2010600的fresh B与历史B reset和首个预测chunk完全一致，但第一控制步实测TCP已有1.19e-7m差异；长程release结果不同。本轮没有隔离唯一根因，不能声称完整物理rollout位级确定性。最终采用本轮重新配对的B/R结果；历史记录保留且不择优替换，见 [historical_baseline_reproducibility.json](final_holdout12/historical_baseline_reproducibility.json)。部分同态再生成失败被拒绝，也没有通过降低门槛纳入数据。

## 决策

保留R作为有效的局部恢复数据baseline，停止继续仅调配比的R重训。原R01有完整成功个例，R03抓持更强，但目前没有形成可靠整任务放置能力。下一步建议检验“可观测因果历史能否补充任务进度/运动状态”，作为与纯R数据分布调整不同的新方向；这是假设，尚未证明单帧观测导致不可约歧义。详细代码依据、文献、对照和通过标准见 [next_experiment_proposal.md](next_experiment_proposal.md)。本轮没有实施或启动新方向训练，也没有消耗剩余第3次R尝试额度。
