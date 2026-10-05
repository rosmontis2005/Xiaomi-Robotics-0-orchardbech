# 冻结 R01 与 B8000：最终配对 heldout 验证

固定 full30、reach-conditioned、10mm/0.08rad、dwell30、budget900、benchmark_assist、Euler5；同果 held15→detach→release→连续60步稳定留桶才算 strict。detach列为episode曾有detached的诊断计数，release列为held→unheld事件（合法chain另见JSON），两者不替代strict身份链。B 与 R 均在本轮重新执行，双路 RGB/proprio reset 12/12 完全一致。

| Cohort | n | B held15 | R held15 | B detach | R detach | B release | R release | B strict | R strict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| heldout12 | 12 | 8 | 8 | 8 | 8 | 5 | 3 | 0 | 1 |
| remaining11 | 11 | 7 | 7 | 7 | 7 | 4 | 2 | 0 | 0 |
| previous_auxiliary1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 1 |

| Scene | prior auxiliary | B held15 | R held15 | B strict | R strict |
|---|---|---:|---:|---:|---:|
| 2010796 | False | 0 | 1 | 0 | 0 |
| 2014018 | False | 1 | 0 | 0 | 0 |
| 2013145 | False | 0 | 0 | 0 | 0 |
| 2010387 | False | 1 | 1 | 0 | 0 |
| 2010143 | False | 1 | 1 | 0 | 0 |
| 2010700 | False | 1 | 0 | 0 | 0 |
| 2010600 | True | 1 | 1 | 0 | 1 |
| 2014304 | False | 0 | 1 | 0 | 0 |
| 2016203 | False | 1 | 1 | 0 | 0 |
| 2016218 | False | 1 | 1 | 0 | 0 |
| 2012745 | False | 0 | 0 | 0 | 0 |
| 2015662 | False | 1 | 1 | 0 | 0 |

按场景配对的 strict 结果与描述性 Wilson 区间见 [report.json](report.json)。这是历史挑选过的12个 val 场景，2010600 已做辅助诊断；其余11个仅在冻结选择后使用。未运行 fresh24，不能把本表称为随机新场景总体成功率。所有失败、视频、步骤与 chunk、模型加载报告均保留；observer 和 summary 一致性检查均通过。
