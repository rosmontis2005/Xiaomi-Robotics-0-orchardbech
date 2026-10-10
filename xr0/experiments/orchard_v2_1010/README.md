# V2 requested-command 首轮实验

唯一主实验：原始 CALVIN pretrained weights-only → 60,000 实际 AdamW updates；seed42，batch1/acc1，constant1e-5，冻结 VLM BF16，non-VLM FP32 master。实验结束于统一闭环和研究报告，不自动进入下一轮。

执行入口：`bash run_pipeline.sh`，当前 tmux 会话 `orchard_v2_1010`。流水线先完成150个reset可见性审计，冻结配置/场景/来源SHA，再训练、固定面板和闭环。中断后同命令依据完整 rolling optimizer/scheduler/RNG/cursor 恢复；最多499个未提交updates会被归档重放，最终有效step不重复计数。逐行训练日志保留原始loss/MSE/频域loss、裁剪前范数、读取与优化耗时和窗口身份。

数据：1800 train episodes / 434783 windows；200 validation / 48501 windows。原始 accepted manifest 不变。V2 action stats 只用 train全部窗口重新计算；inactive mean0/std1，active std floor1e-4。Dataset使用每episode轻量数值数组缓存（源JSON SHA见manifest）和有界video decoder LRU；保持原RGB95%中心裁剪、processor、prompt和collate。阶段/事件字段只供分组和诊断。

schedule精确60k条：reset5%、earlyREACH15%、GRASP20%、PULL10%、TRANSPORT25%、near-bucket/DROP20%、transition5%。池互斥，组内以均衡episode cycles抽样，再选窗口。训练前唯一一次修正保留在schedule_revision.json，归档初稿前后曝光。GRASP组一半优先取future30含closing的窗口，DROP组一半优先释放命令起点-15/+10步。没有根据模型结果改配比。

固定train/val各256窗，原flow训练分支no_grad计算loss，逐窗重置Python/numpy/CPU/CUDA RNG并恢复外层训练RNG。注意XR0.eval()+return_loss不是训练flow loss，不能替代。每2000更新评估，额外5k；0/2k/5k/10k/20k/40k/60k生成固定种子动作，分别记录anchor与实际target phase、四种horizon物理误差。生成输入action为零，不提供GT prefix。

闭环：16个新原规划器可行seed，从8500000递增选取；G0和底座局部y±0.08m，不重优化。全288条（5 checkpoints×48 + native expert48）。闭环使用4个独立轻量仿真worker，单个共享VLM以batch1串行推理，每个场景各自保存/恢复推理RNG；不同时加载多个VLM。共同有效性独立报告，失败保留。学生每次预测30，仅执行前5条，每条一次native_command→env.step；每块重建测量锚点，不dwell、不专家干预。原StrictPlacement作为主指标；合理释放距离≤0.15m且桶XY内仅独立诊断。学生没有FSM，因此阶段停留/超时是事先固定的被动事件/距离代理，不假称模型内部阶段；不用于终止控制。

数据、checkpoint和逐步仿真日志在本目录本地留存；Git忽略大型产物。`frozen.json`、manifest和training_complete.json记录来源与权重SHA。完成前不得把final_report.md中部分统计解释为60k结果。
