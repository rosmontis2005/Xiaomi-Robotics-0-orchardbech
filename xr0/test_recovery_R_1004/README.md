# R student-state recovery session archive

本目录归档原始R pilot及随后两次额外训练与闭环验证。所有版本4000 updates完成，最终选定R01再与正式B8000进行一次配对heldout12测试；未实施H方向，未运行fresh24。

## 最终结果入口

- [最终汇总](evaluation_followup/summary_report.md)及[机器可读结果](evaluation_followup/summary_report.json)。
- 独立版本：[R01](evaluation_followup/version_01/report.md)、[R02](evaluation_followup/version_02/report.md)、[R03](evaluation_followup/version_03/report.md)。
- [冻结R01/B8000最终heldout配对报告](evaluation_followup/final_holdout12/report.md)与[cases.csv](evaluation_followup/final_holdout12/cases.csv)。
- [下一方向方案](evaluation_followup/next_experiment_proposal.md)：仅提出可观测因果历史H的对照，未改模型或启动训练。

`final_report.json`及`final_report.txt`保留首次任务的历史停止点（启动训练后停止），不代表最终训练仍在运行。最新结论以`evaluation_followup/summary_report.*`为准；原始`experiment_registry.json`也是中途记录。各版本training/status.json和独立evaluation status记录最终完成状态。

严格成功为同果连续held15 → detach → release → 连续60个控制步稳定留桶。原R01 dev8 strict1/8；R02和R03为0/8。最终heldout12 B/R01分别strict0/12和1/12，但R01唯一成功来自先前辅助场景2010600；其余11例双方均0/11。不能把这批历史挑选场景称为随机新场景总体成绩。

## 追踪与本地保留范围

Git保留实验脚本、冻结4000-step schedule和manifest、checkpoint/hash provenance、配置/审计、候选接受与拒绝记录、工程修复记录、训练/启动记录以及最终汇总。局部`.gitignore`在仓库原有实验规则上，仅为这些小型复现记录增加例外。`.gitattributes`保留已执行takeover源码的原始末尾空格及CSV原CRLF，避免格式化改变冻结provenance哈希。

原始sensor/action trajectory、checkpoint和optimizer权重、两路视频、逐控制步/逐chunk日志、离线逐窗口prediction、缓存、临时文件与锁留在本机，未删除，也不进入Git。脚本和报告中的绝对路径、checkpoint及数据SHA保留原始provenance；仅Git checkout不能替代这些外部资产。

`ARCHIVE_INDEX.json`列出此次归档内容的路径、大小及SHA256（索引自身除外），便于定位与校验；候选debug truth仅供teacher/审计，policy loader仍使用原白名单。

## 复现条件和脚本

运行依赖现有XR0 `.venv-orchard`、OrchardBench仿真环境、原`orchard_v1_2650`数据/统计、formal full10k、正式B8000及本地生成的数据。协议文件记录运行时具体路径与SHA。所有版本保持full30/30-target、reach-conditioned、10mm/0.08rad、dwell30、budget900、benchmark_assist与Euler5。

- `mine_student.py`、`recovery_teacher.py`、`collect_recovery.py`：真实B8000学生状态采样、live teacher与原生student-controller oracle。
- `recovery_dataset.py`、`prepare_training.py`、`train_R.py`：白名单mixer、冻结schedule、B8000初始化和训练；R02/R03各自保留所用源码。
- `evaluate_followup.py`、`evaluate_takeovers.py`、`evaluate_final_holdout.py`：开发集、训练来源的机制诊断及最终冻结配对测试。
- `collect_onpolicy_R3.py`、`inspect_composite_prefix.py`：新增晚期候选失败和前缀再生成诊断，接受0，未产生有效新label。
- `launch_job.py`及各receipt：实际后台启动命令、PID/session和日志位置。归档不重新运行这些命令。

`recovery_continuation_diagnostic.py`为准备但未执行的辅助代码；R02计划版evaluation源码不入归档，未算实验结果。`summarize_followup.py`/`finalize_followup_report.py`是历史报告生成脚本；最终报告另有基于candidate_decisions.jsonl的人工校正及辅助场景重叠说明（见summary_report.json的report_assembly_notes），不要直接覆盖最终报告来抹去这些校正。

原pilot29个unique候选、32次attempt：24接受，4在前缀检查中失败且teacher未开始，1个teacher预算失败。真实teacher运行25个unique候选，24个成功并通过oracle。另有新增late-current-R4候选全部在前缀检查中拒绝。不能把工程前缀失败解释为teacher恢复失败。物理长程复跑并非位级确定；历史baseline重现检查与全部限制保留在最终报告。
