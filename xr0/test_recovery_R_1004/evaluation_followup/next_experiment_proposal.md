# 候选下一方向 H：可观测因果历史与任务进度

这是实验方案，未实现模型或loader改动，也未启动H训练。它与R仅改变训练分布的路线不同：H测试policy可用的信息是否足够表达执行进度。采用理由结合冻结R01的最终配对结果说明；本方案不能被视为H已经有效的证据。最终配对结果已经确认：R01与B均held15=8/12，strict分别1/12和0/12；唯一R成功来自先前辅助场景2010600，remaining11双方strict0/11。因此本轮决策是停止继续仅调R配比，优先检验H假设，且到方案为止。

## 本轮证据能支持什么

R01在原始student真实偏移起点的训练数据上确实学到了恢复：R2 frame0预测位置误差259.6→94.2mm，两个真实held接管的最低bucket距离0.827/0.682→0.095/0.112m。然而接管诊断没有出现strict成功，接近bucket也未release。训练来源的这个诊断支持“原student-state覆盖不足是问题的一部分”，不能作为泛化提升结论。

R01原配比暴露两个可修复的数据问题：reset/REACH退化，以及R1大量anchor已进入PULL；R02提高原B回放、限制R1实际未held anchor，修复离线误差却未增加dev8 strict。R03另补成功teacher恢复轨迹里的稀疏后段桥接监督，实际多出4594个DROP future targets；dev8 held15提高到7/8、detach7/8，但release1/8、strict0/8。增加后段label并未自动改善整个“运输—释放—稳定留桶”过程。

这不足以否定R或DAgger。原pilot只有24条、12个训练来源；新late-current-R状态的4个候选在前缀再生成一致性检查中被拒绝，接受0，没有完成teacher/oracle，因此本轮没有充分检验当前R自身的新状态分布。这个采集限制是明确的剩余工程债；不能把拒绝说成teacher无法恢复，也不能把成功teacher后段窗口说成新on-policy student起点。数据分布问题仍可能存在。但在已完成两次数据修改、主指标没有改善时，再无预设问题地继续调配比的收益证据不足。

## 代码提供的具体假设

1. [orchard_action.py](/home/rosmontis/Projects/orchardbench/treesim/orchard_action.py:25)的`state_vector`仅提供当前world TCP位置3维、Euler姿态3维、measured总宽度1维、arm joints7维；其余维度为零。没有关节速度、近期命令或过去观测。
2. [orchard_policy.py](/home/rosmontis/Projects/dualsys/Xiaomi-Robotics-0/xr0/mibot/server/orchard_policy.py:29)每次predict重建两张当前RGB和单个当前state。VLM KV-cache属于当前调用内部，不是跨执行过程的记忆。full30 action chunk提供未来序列，不能代替过去已经执行的状态。
3. [vla_env.py](/home/rosmontis/Projects/orchardbench/treesim/vla_env.py:316)的`_update_width_intent`根据连续open-width命令计数与前一条commanded width决定gripper intent。闭合判据比较的是前次**命令宽度**而非measured width；打开需要连续5个控制步。相同当前measured width不直接给出这个内部进度。部署端已经知道自己发出的命令，因此可加入合法的因果历史，而不必给policy simulator held/phase真值。需强调：两路RGB也可能提供相关证据；此处并未证明完整当前观测一定发生不可区分的别名。
4. [picker.py](/home/rosmontis/Projects/orchardbench/treesim/picker.py:748)的TRANSPORT→DROP使用真实fruit位置、速度、TCP距离、计时与stalled状态。teacher可以用truth，这是合法监督；student单帧RGB/current proprio未必能可靠推断动态进度。现有teacher还有超时/停滞DROP分支，不能将所有DROP labels都解释为最优安全释放；本轮成功数据经过strict/oracle筛选，但未来应单独审计这些转换的成因。
5. [XR0.py](/home/rosmontis/Projects/dualsys/Xiaomi-Robotics-0/xr0/mibot/models/VLA/XR0.py:407)默认`state_shape=(1,32)`；state projector逐token映射32→DiT，`dit_forward`接收`(B,state_len,D)`，local causal mask也支持动态state length。这提供了复用现有权重的实验入口，仍需验证collate、RoPE位置、checkpoint加载和batch shapes，不能承诺仅改一个配置就正确。

因此H的可检验假设是：**在保持当前动作、控制器、归一化和冻结VLM的情况下，加入实际可观测过去信息，能否让动作头更可靠地区分继续接近、保持抓持、运输与释放的进度？** 这是从代码与结果推断的假设，尚未被本轮数据证明。近期proprio也不直接包含fruit swing速度；如果H只改proprio而失败，不能由此否定图像历史的作用。

## 最小第一组对照

先固定一份已经审计过的R03数据、4000 schedule与所有优化设置。所有实验从同一个正式B8000开始，fresh AdamW，固定训练seed与inference设置。原R03只能作为历史对照，新增H0应承担相同追加训练预算，避免将多训练4000步的收益错误归因于history。

| 条件 | 信息 | 目的 |
|---|---|---|
| H0 | 一个当前state，当前两路RGB | 同预算对照 |
| Hrepeat | 4个完全相同的当前state，当前两路RGB | 控制token数量、位置改变及容量因素 |
| Hhistory | 真实因果proprio：t−15、t−5、t−1、t（30Hz） | 检验近期运动/宽度进度 |

第一组不同时改图像输入、prompt、loss或grasp gate。时间不足和episode刚开始时只允许用实际最早观测做padding，并标记规则，不得使用未来teacher状态或假HOME。时序必须按真实control step对齐；reach-conditioned dwell意味着target index不能用来替代时间。训练与部署使用同一因果buffer更新顺序。

若近期history有效但后段bucket方向仍差，再单独比较“reset观测锚点+t−5、t−1、t”和等长近期history。reset proprio是实际机器人读数，可能帮助记住固定底座坐标参考；它并不是GT bucket/base字段。当前world TCP pose与joint状态原则上也可能经已知运动学推断底座参考，因此reset anchor主要是学习便利性的对照，不能宣称它必然提供不可从当前观测获得的新信息。不要在第一组一次加上reset anchor、视觉记忆和命令历史，让结果失去归因。

输入只允许robot已能读到的measured proprio、现有RGB和自己真实下发的native命令。category、teacher phase、apple ID、held/detached、fruit xyz/velocity、bucket xyz、scene seed、debug state全部禁止进入policy。

对于recovery frame0，历史必须来自对应真实student prefix，且要与takeover实测记录一致。后续teacher期间可用过去的measured proprio，但不能伪造student的连续width command历史：AutoPicker的binary gripper命令与部署adapter连续width语义不同。若测试command history，应在当前native-controller oracle实际执行中记录白名单命令与measured history；先解决训练/部署命令语义一致性。旧debug metadata即使保存了truth，也不可被整体拼入state。另建白名单字段和poison-metadata测试，缺失真实历史的样本不能以reset填充冒充完整prefix。

## 通过标准与反证

仍用full30、30-target chunk、reach-conditioned 10mm/0.08rad、dwell30、900预算、benchmark_assist、原action stats、Euler5。先冻结dev协议，以strict placement为主指标；报告held15、同果detach、release区域、held后长期停滞、异常物理与全部失败候选。辅助seed42/43只用于开发阶段稳定性，不能把一个episode的两个seed当两场景。

Hhistory至少应优于Hrepeat与同预算H0，且不能靠held提升掩盖strict下降。单次dev8的一例成功不够；应在训练cohort之外的预先固定开发场景中检查多个inference seed，避免重演L的seed sensitivity。当前heldout12已经被本轮最终报告消费，未来不能再作为新的未接触选型集；fresh24继续冻结，待下一轮真正确定endpoint后只做最终测试，不用于调参或恢复采样。

需要检查哪些历史跨度实际改善放置，是否只是学习时间/场景捷径。history置乱/截断可作为诊断，但不能将跨episode样本产生的严重OOD失败当作充分因果证据。若Hrepeat同样提升，则优先解释为token/优化因素；若真实历史只降低离线误差但闭环strict仍无改善，则历史信息假设缺少支持。近期proprio对fruit运动信息不足，后续才考虑冻结VLM的稀疏过去图像特征；这属于另一项独立对照。

采集方面，若下一轮仍需要当前R/H late-state数据，优先在**正在运行的真实student env**中暂停并接管teacher，再通过正常student prefix再生成验证；保留物理偏差与拒绝记录，不写teleport式snapshot，也不放宽controller/physics gate来凑数据。这个工程修复不自动解决policy历史问题，两者应分开报告。

## 文献依据与适用范围

[DAgger, Ross et al., 2011](https://proceedings.mlr.press/v15/ross11a.html)指出动作会改变未来观测分布，支持在learner实际到达状态上取得监督。它支持R的动机；本pilot不是充分迭代到收敛的DAgger实现，论文不能用于宣称R已被否定。

[MemoryVLA, Shi et al., ICLR 2026](https://arxiv.org/abs/2508.19236)研究VLA长时任务中的记忆条件化动作生成，支持检验temporal context。这里建议先用现有XR0动作头的少量proprio history做归因明确的对照，而非直接移植完整memory bank；论文其他任务的成功率不能作为OrchardBench收益预测。

[Nguyen et al., CoRL / PMLR 2023](https://proceedings.mlr.press/v205/nguyen23a.html)研究利用fully-observable state expert学习partial-observable策略，提示privileged teacher的成功不等价于学生可由有限观测直接模仿。其方法属于RL，与本项目flow imitation不同；只用于说明值得检验的信息差异，不据此直接启动RL。

[HALO, Shah et al., RSS 2026](https://roboticsproceedings.org/rss22/p010.html)指出历史检索在模仿学习中可能学到伪相关，并因预测与环境交互产生memory drift。这支持设置Hrepeat、真实history对照和严格因果buffer；不是“加入越多历史越好”的依据。

若历史对照最终不支持假设，再分别调查局部视觉几何精度与continuous loss/event outcome之间的差异。目前证据不足以同时解冻VLM、修改控制器、引入GT感知或启动大规模RL；本轮到报告和方案为止。
