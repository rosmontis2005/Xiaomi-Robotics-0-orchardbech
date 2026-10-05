# OrchardBench-VLA / XR-0：R03 剂量对照与 H 因果 proprio 历史实验

本轮实际完成 R03 step4000→总计8000 optimizer continuation，以及 H0/Hrepeat/Hhistory 各8000 updates的独立训练；每次训练后立即完成固定 dev8＋历史单场景 seed42/43 的10个闭环，再进入下一条件。新增正式闭环共40例，另有一次30控制步部署输入 smoke，不计成绩。H预算选择及H4停止决定分别保存在 `budget_decision.json` / `H4_decision.json`。本轮停止，不运行H4、fresh24、新恢复采集或其他模型路线。

**主要结论：剂量增加改变运输行为，但没有解决完整放置；真实近期 proprio history 没有优于 token-count control 的整体闭环证据。** Hhistory held/detach从控制条件的7/8降到5/8，三种H的dev8 strict都为0/8，合理近桶释放和稳定留桶均为0。不能把其略低的离线误差宣布为模型更好。

## 最重要的结果（dev8）

| model | updates | held15 | detach | release | strict | bucket-progress | main failure |
|---|---:|---:|---:|---:|---:|---|---|
| B8000 | 8000 | 6/8 | 6/8 | 4/8 | 0/8 | 最近0.801m；稳定0步 | 远处/桶外释放4；持果运输停滞2 |
| R03-4k | 4000 | 7/8 | 7/8 | 1/8 | 0/8 | 最近0.597m；稳定0步 | 持果运输停滞5；运输未完成、不释放1 |
| R03-8k | 8000 | 7/8 | 7/8 | 3/8 | 0/8 | 最近0.161m；稳定0步 | 远处/桶外释放3；运输未完成、不释放3 |
| H0 | 8000 | 7/8 | 7/8 | 3/8 | 0/8 | 最近0.259m；稳定0步 | 远处/桶外释放3；持果运输停滞2 |
| Hrepeat | 8000 | 7/8 | 7/8 | 2/8 | 0/8 | 最近0.217m；稳定0步 | 持果运输停滞4；远处/桶外释放2 |
| Hhistory | 8000 | 5/8 | 5/8 | 2/8 | 0/8 | 最近0.216m；稳定0步 | 持果运输停滞3；未稳定抓持3 |

`updates`中R/H是正式B8000之后的追加更新；R03-8k包含已有4000＋本次新增4000。B8000为正式baseline，不是full10k。R01/R02/R03历史上均从同一个B8000独立开始，未互相串行；本轮只有R03-8k沿原R03 optimizer继续。

`release`为boundary observer记录的held→非held事件；本表中的事件都具有held15/detach同果链，但它们的空间位置并不合理。Strict仍严格要求同果held15→detach→release→非held且连续60控制步稳定留桶。Bucket-progress的“最近”是该cohort最小持果距离，不能代表多数病例；完整逐例值如下。所有模型dev8的15cm近桶诊断计数均0/8、非held稳定留桶最大均0步；branch break及当前日志可识别的physical anomaly均0/8。15cm/桶口投影是统一被动诊断，不改变任何physical gate或strict定义。

## 1. R03-4k→8k 是否说明训练不足？

**存在运输未饱和/训练剂量的真实混杂，但不能说训练不足充分解释全部失败，也不能宣布R03-8k的整任务成绩更好。** 2010914从0.855→0.513m，2015751从0.841→0.243m；后者到第899控制步才释放，释放距离0.238m且在桶口XY投影之外。2014362从未held15发展到held/detach并最接近0.161m，随后又漂远。长held末段停滞诊断5→1例，但其中也有更多远处释放。held/detach总数仍7/8；2012485丢失抓持，由2014362补回，不能隐藏此交换。

dev8 strict仍0/8、合理区域release仍0/8、稳定bucket仍0。历史辅助2010600 strict由1/2变0/2；这说明剂量收益并不稳定覆盖整个任务。根据用户给定的“多个病例明显更合理bucket approach且未破坏aggregate held/detach”的预算规则，采用**统一8k H预算**以控制这部分运输剂量混杂；不是因training loss下降或release次数增加而自动选择8k。该决定在任何H训练前冻结。

Continuation从原`version_03/training/checkpoints/latest_resume.pt`加载全部219组AdamW状态（每组step4000）、scheduler、精确匹配step4000权重、Torch/CUDA RNG；LR始终1e-5。Legacy resume未保存Python/numpy RNG，但训练随机forward使用Torch、数据读取RNG-preserved，新增schedule由独立确定性sampler seed1005提前冻结。未将fresh AdamW称为continuation。首次运行因新checkpoint目录遗漏在4500保存时中断；失败500次更新和日志保留在`failed_attempt_checkpoint_directory`，随后从原4000完整状态重跑，最终checkpoint真实optimizer step为8000，不把失败尝试加到最终更新数。

## 2. H0 是否基本复现 current-only R 路线？

**基本复现。** H0与R03-8k的dev8同为held7/detach7/release3/strict0，所有release均远处/桶外；验证TRANSPORT误差203.08与202.77mm、训练来源DROP误差23.09与23.19mm接近。H0第一update loss与原R03完全一致，前几步接近；初始化219个FP32 trainable tensors都精确匹配B8000，fresh AdamW初始状态0。

这不是位级复现：逐病例运输/释放不同，H0辅助seed43成功、R03-8k两seed均失败。小幅训练数值变化和长程物理rollout不可位级确定的历史限制仍存在；不择优替换已有R/B traces，不把一次辅助成功当作泛化。

## 3. Hrepeat 与 H0：token 数/结构有影响吗？

**存在局部运输行为变化，但没有完整放置收益。** Hrepeat与H0同为held/detach7/8，strict均0，近桶合理release均0。Hrepeat将2010914的最低距离从H0的0.810降到0.265m、2011310从0.697降到0.217m，同时2014461从0.259退到0.797m；release3→2且都不合理，辅助strict1/2→0/2。因此新增state token/mask/RoPE/optimization路径本身足以改变部分运输行为；未来收益必须额外优于Hrepeat才能归因于真实时序信息。单seed、小dev不能把这些逐例差异全部确认为稳健的结构因果效应。

## 4. Hhistory 是否真正优于 Hrepeat？

**没有整体闭环优势，本轮H假设未获得支持。** Hhistory held/detach5/8，Hrepeat为7/8；Hhistory额外丢失2014461与2011310稳定抓持，2012485双方都未held15。release都是2/8、全部桶口XY之外；strict都0/8、stable bucket都0。2010914距离0.265→0.216m，2015716 0.435→0.303m，2015751 0.784→0.684m，有局部移动改善，但2014362从0.350退到0.465m，而且丢失的抓持链不能从距离统计里排除来制造“平均更好”。

没有明确真实history优势，故不符合Hanchor或Hhistory延长训练的启动条件，停止在3次H训练；未临时加入图像/命令历史、RNN、RL或新配比救实验。

## 5. 改善在哪个阶段？

R03剂量对照的主要变化在**transport**，并有一个新增grasp/detach病例和一个丢失病例相抵。新增release属于远处释放，未改善稳定放置或strict。Hrepeat相对H0也主要改变transport方向/进度；没有有效release/strict收益。Hhistory在部分已有held病例的transport及离线局部误差略好，但**grasp→pull同果链覆盖下降**，没有release或strict改善，因此不是总体改进。

### 每例最低持果 fruit-bucket distance（m）

| scene seed（均 inference42） | R03-4k | R03-8k | H0 | Hrepeat | Hhistory |
|---|---:|---:|---:|---:|---:|
| 2013322 | 0.823 | 0.757 | 0.777 | 0.755 | 0.639 |
| 2014461 | 0.858 | 0.784 | 0.259 | 0.797 | 未held15 |
| 2010914 | 0.855 | 0.513 | 0.810 | 0.265 | 0.216 |
| 2012485 | 0.597 | 未held15 | 0.782 | 未held15 | 未held15 |
| 2015716 | 0.606 | 0.624 | 0.785 | 0.435 | 0.303 |
| 2014362 | 未held15 | 0.161 | 未held15 | 0.350 | 0.465 |
| 2015751 | 0.841 | 0.243 | 0.738 | 0.784 | 0.684 |
| 2011310 | 0.800 | 0.766 | 0.697 | 0.217 | 未held15 |

“未held15”不是删除失败episode；全部保留且进入分母。逐control-step traces、视频、seed_delivery与history_delivery本地保留；每case的chain/release/stable/failure详见各条件`evaluation/cases.json` / `cases.csv`。

### 每次 dev8 release 的位置

| 模型 | scene seed | release control step | fruit-bucket distance（m） | 桶口XY投影内 | strict |
|---|---:|---:|---:|---|---|
| B8000 | 2012485 | 448 | 0.879 | 否 | 0 |
| B8000 | 2015716 | 215 | 1.077 | 否 | 0 |
| B8000 | 2015751 | 291 | 0.903 | 否 | 0 |
| B8000 | 2011310 | 258 | 1.008 | 否 | 0 |
| R03-4k | 2011310 | 437 | 0.797 | 否 | 0 |
| R03-8k | 2013322 | 442 | 0.765 | 否 | 0 |
| R03-8k | 2015751 | 899 | 0.238 | 否 | 0 |
| R03-8k | 2011310 | 374 | 0.800 | 否 | 0 |
| H0 | 2010914 | 354 | 0.836 | 否 | 0 |
| H0 | 2012485 | 423 | 0.803 | 否 | 0 |
| H0 | 2011310 | 412 | 0.713 | 否 | 0 |
| Hrepeat | 2013322 | 456 | 0.756 | 否 | 0 |
| Hrepeat | 2014362 | 686 | 0.348 | 否 | 0 |
| Hhistory | 2015716 | 477 | 0.299 | 否 | 0 |
| Hhistory | 2015751 | 450 | 0.680 | 否 | 0 |

所有正式dev8 release均在桶口XY投影之外。Hhistory两个release分别约0.299/0.680m，不是合理放置；R03-8k第899步的较近释放也不是成功。最大稳定bucket为0，未放宽预算以等待后续落桶。

## 6. 阴性结果能区分哪种不足？

本轮能直接说的是：**当前8k、同R03数据池、当前两路RGB＋这组近期measured proprio history没有解决主要闭环失败的证据，且降低了稳定抓持覆盖。** 不能由此认定所有历史信息本身无帮助，或证明单帧观测存在/不存在不可约歧义。

这组history包含机器人运动和夹爪测量，不包含fruit swing/fruit velocity；相同measured width也未必恢复生产controller依赖的过去command-width/open计数。缺少关键物体动态仍是合理信息限制，但本轮未做能将它与学习/数据因素分离的干预，因此只是推测。训练来源DROP拟合约22mm而dev仍无合理release，原24条/12来源successful teacher recovery的晚期状态也不等价于当前student真实晚期状态；新late-current-R接受0的覆盖限制仍然存在。8k只控制所测试的剂量，不保证完全饱和；没有历史闭环优势，不能仅因loss还降而继续延长H。

## 训练来源机制、开发集、辅助场景、最终泛化分开报告

| 模型 | train2 held/detach/strict | val6 held/detach/release/strict | 历史单场景×2seed held/release/strict |
|---|---|---|---|
| B8000 | 1/1/0 | 5/5/4/0 | 2/0/0 |
| R03-4k | 2/2/0 | 5/5/1/0 | 2/1/1 |
| R03-8k | 2/2/0 | 5/5/2/0 | 2/0/0 |
| H0 | 2/2/0 | 5/5/3/0 | 2/2/1 |
| Hrepeat | 2/2/0 | 5/5/1/0 | 2/0/0 |
| Hhistory | 1/1/0 | 4/4/2/0 | 2/1/0 |

表中train2、val6、auxiliary分别为原冻结cohort，数字按各列标题顺序，不把auxiliary两个seed当两场景。Hhistory val6 held/detach4/6，H0/Hrepeat均5/6；三者val6 strict均0/6。Auxiliary strict：R03-4k1/2、R03-8k0/2、H0 1/2、Hrepeat0/2、Hhistory0/2。这个2010600是历史观察场景，不是新test。

**最终泛化本轮未测量**：未运行fresh24；未重新把已消费heldout12作为新endpoint selection set。B8000/R03-4k是历史冻结参考；dev8包含train2＋val6，所有H均一个training seed，不能声明总体成功率或泛化提升。训练来源恢复拟合不能代替泛化成绩。

### 离线误差（只解释机制）

| 模型 | old-val xyz mm | new-val xyz mm | val TRANSPORT xyz mm | 训练来源 DROP xyz mm | DROP rotation rad | DROP width mm |
|---|---:|---:|---:|---:|---:|---:|
| R03-4k | 66.74 | 49.66 | 222.68 | 23.80 | 0.0993 | 4.24 |
| R03-8k | 60.27 | 50.97 | 202.77 | 23.19 | 0.0805 | 2.88 |
| H0 | 62.01 | 51.79 | 203.08 | 23.09 | 0.0791 | 2.16 |
| Hrepeat | 59.84 | 51.54 | 201.31 | 22.85 | 0.0744 | 2.20 |
| Hhistory | 54.01 | 45.69 | 197.39 | 21.74 | 0.0768 | 2.69 |

原冻结240窗口面板使用seeds42/43/44；val TRANSPORT按原anchor分组的full30物理位置平均距离。该panel没有DROP targets。新增固定48-window诊断仅来自原24条accepted成功recovery，每条一个TRANSPORT→DROP前1步anchor、一个DROP anchor；未来targets为DROP/DONE，因此表中DROP明确是**训练来源机制**，不是val DROP或新泛化集。Targets、windows、noise seeds相互相关，不是独立成功样本；ground-truth action只用于监督/误差，generation输入action全部置零。

Hhistory验证TRANSPORT相对Hrepeat仅201.31→197.39mm，训练来源DROP22.85→21.74mm；这些小幅离线变化不能覆盖held链7→5且strict仍0的闭环结论。复用旧offline helper输出的`fit_gate`仅保留为拟合字段，不参与本轮endpoint选择。

### 不同 source 的 training loss 趋势

下表为整条8k序列最初500与最后500 updates内各source的平均flow loss；R03-8k拼接原4000＋本次4000真实记录。统计noise和重复窗口存在，不是相同独立验证样本。

| 模型 | original_B | R1 | R2 | R3 | teacher continuation |
|---|---:|---:|---:|---:|---:|
| R03-8k | 0.120→0.103 | 0.138→0.039 | 0.221→0.134 | 0.300→0.091 | 0.206→0.152 |
| H0 | 0.120→0.103 | 0.138→0.039 | 0.223→0.109 | 0.300→0.096 | 0.206→0.161 |
| Hrepeat | 0.129→0.105 | 0.139→0.040 | 0.249→0.119 | 0.303→0.102 | 0.202→0.163 |
| Hhistory | 0.134→0.096 | 0.141→0.040 | 0.302→0.128 | 0.312→0.100 | 0.205→0.124 |

所有8k版本实际source counts均为：original_B5600、R1 1000、R2 400、R3 400、R_continuation_teacher600。R03新增4000单独为2800/500/200/200/300；原R03-4k同此配比。所有更新都是batch1/acc1的真实optimizer.step，未以loop iterations替代。

## 实现与执行保护范围

H0 `[s_t]`；Hrepeat四份逐值相同current；Hhistory `[s_(t-15),s_(t-5),s_(t-1),s_t]`，时间单位30Hz真实control step。普通B初期padding使用最早已真实出现的observation；recovery负local index从对应student source的真实测量prefix取，nonnegative取teacher测量，可跨边界；frame0 timestamp精确对齐takeover控制步。24条prefix全部可用、排除anchor0。只抽取TCP world position/Euler、measured width、7 arm joints；其余state维度0。历史debug容器的category/phase/apple/held/fruit/bucket/seed均不进入batch。

训练与部署token顺序相同；简单16-state deque每个真实control boundary更新，episode reset清空，不能只在replan保存。一次30-control-step smoke实际验证predict@30输入indices `[15,25,29,30]`、31次buffer更新、action有限；不计experiment成绩。每个条件正式训练前只做一个真实batch generation检查后立即训练。XR0已支持动态state length和local mask，因此**没有修改production XR0/action head参数形状或OrchardPolicy/controller/physics/IK/FixedBaseAutoPicker**；实验HistoryPolicy只替换state tensor，ObservedEnv hook只读测量并调用原super.step。

所有H各自从full10k→正式B8000 overlay开始，fresh AdamW，seed42，lr1e-5、betas(0.9,0.95)、wd0.1、eps1e-8、foreachFalse、clip1、batch1/acc1、constantLR、原full30 active7 flow loss（系数0.5）、frozen BF16 VLM/FP32非VLM masters/BF16 autocast；两路当前RGB、原prompt、图像95%crop/resize、action encoding、原stats不改。H schedule/manifest/stats SHA分别为`a92d54ad15d38f4335699e460b68fc544c656fc01428a6bea49b4902359581f6` / `2f857fc7af14bf6ea0225f6fc385f1840209a14ec1d10523ab1f471e52b7b71f` / `63dec8bbf2249d505358f17beac21d800bd49dee1192c9e79ddf3e1085c9b371`，三者一致。

闭环保持full30、30-target、reach-conditioned、10mm/0.08rad、dwell30、budget900、benchmark_assist、detach_force_scale1.5、Euler5、原strict定义。各正式case的reset与冻结B双路RGB/state精确匹配，observer/summary consistency检查通过；history buffer每步更新并写delivery记录。未重复Gate0/Gate1/GT全套。模型不消费diagnostic fruit/bucket几何。

## Checkpoints、命令与归档

| 模型 | checkpoint SHA256 |
|---|---|
| B8000 | `c2a5c22352b916989cae7026ff9af6e06ce8129cf3611a128930f517f8838c31` |
| R03-4k | `5890b0d08f6e396299a47295f2e8b3987aafb477e7bfd571287d875525cb9575` |
| R03-8k | `767ab133d24ac6d024c259e9b773aaca84dc9bd4c5b5a74d43d5d1671e309138` |
| H0 | `007a07b42edf9907b65212619cb142700b4f2b3d13adcf62658e42377627ef7e` |
| Hrepeat | `c5f2597c0667ef248e2ad369fa9566f5f8415d6225e8b43851e549bf684e2831` |
| Hhistory | `13fd59afcc08adbb42c8428948bf52264f59443ecd36dcba8b1f6b7d27479569` |

- B8000: `/home/rosmontis/Projects/dualsys/Xiaomi-Robotics-0/xr0/test_128_episode_AB_1003/arms/B/checkpoints/step_8000_trainable.pt`
- R03-4k: `/home/rosmontis/Projects/dualsys/Xiaomi-Robotics-0/xr0/test_recovery_R_1004/evaluation_followup/version_03/training/checkpoints/step_4000_trainable.pt`
- R03-8k: `/home/rosmontis/Projects/dualsys/Xiaomi-Robotics-0/xr0/test_history_H_1005/r03_extension/training/checkpoints/step_8000_trainable.pt`
- H0: `/home/rosmontis/Projects/dualsys/Xiaomi-Robotics-0/xr0/test_history_H_1005/H0/training/checkpoints/step_8000_trainable.pt`
- Hrepeat: `/home/rosmontis/Projects/dualsys/Xiaomi-Robotics-0/xr0/test_history_H_1005/Hrepeat/training/checkpoints/step_8000_trainable.pt`
- Hhistory: `/home/rosmontis/Projects/dualsys/Xiaomi-Robotics-0/xr0/test_history_H_1005/Hhistory/training/checkpoints/step_8000_trainable.pt`

Full10k base SHA保留为`fdb580fe60d4cdeb1d35b2a7ce0ec2d27798396aa717c23d9648866a09f9d109`。实际命令、PID、schedule、源loss/更新数、protocol和checkpoint paths/full SHA都在receipts、各training/evaluation文件及`summary.json`中。主要实际启动命令如下；队列内各train/eval的完整argv另见`H*_stage*_receipt.json`，R评测见`r03_eval_command.json`。

```bash
/home/rosmontis/Projects/dualsys/Xiaomi-Robotics-0/xr0/.venv-orchard/bin/python -u -B /home/rosmontis/Projects/dualsys/Xiaomi-Robotics-0/xr0/test_history_H_1005/r03_extension/train_R.py --run
```
```bash
/home/rosmontis/Projects/dualsys/Xiaomi-Robotics-0/xr0/.venv-orchard/bin/python -u -B /home/rosmontis/Projects/dualsys/Xiaomi-Robotics-0/xr0/test_history_H_1005/queue_phase_a.py
```
```bash
/home/rosmontis/Projects/dualsys/Xiaomi-Robotics-0/xr0/.venv-orchard/bin/python -u -B /home/rosmontis/Projects/dualsys/Xiaomi-Robotics-0/xr0/test_history_H_1005/run_three.py --updates 8000
```

checkpoint、大视频、逐控制步trace、optimizer resume继续留本地并被Git忽略；未为归档提交这些二进制产物。历史R/B/L脚本、checkpoint、日志未覆盖；一次历史派生case summary误写位置后已全部移回新H参考目录，修复记录保留在`engineering_repairs.jsonl`。工作树tracked production文件未改。

## 7. 当前最合理的下一步与停止点

停止当前proprio-history分支，不选Hhistory作为更优endpoint，不扩展image/command history或架构，也不自动进入fresh24。下一阶段最有依据的候选是：先修复**当前student真实late-state的live takeover/prefix provenance工程路径**，在真实未teleport的student状态取得因果输入与teacher监督，保留原oracle/physical gate，再做有针对性的transport→release状态覆盖对照。它是下一阶段建议，本轮没有采集数据或启动训练；也不能把此建议当成数据覆盖不足已被证明。若后续要检验物体动态信息，应作为独立信息对照，不能把本轮proprio阴性结果直接外推到图像历史。

当前证据最支持的组合是：**训练剂量不足对transport构成真实但不充分的混杂（中等强度开发集证据）；当前student后段数据状态覆盖不足是优先工作解释（间接、偏弱到中等），这组近期proprio历史未形成足够有效的闭环补充（直接阴性对照证据，但不等于所有历史信息无价值）。** 缺少fruit swing等动态信息仍合理，但仅有弱推测，无法与覆盖/优化不足分离。没有证据支持“单纯再加训练步数就能完成放置”，也没有证据支持“真实近期proprio history已经解决主要失败”；所有结论限于本轮固定开发协议，不是最终泛化结论。
