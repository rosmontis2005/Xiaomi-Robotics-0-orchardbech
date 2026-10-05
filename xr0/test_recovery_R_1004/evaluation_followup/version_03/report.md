# version_03：R4000 测评

Endpoint: `/home/rosmontis/Projects/dualsys/Xiaomi-Robotics-0/xr0/test_recovery_R_1004/evaluation_followup/version_03/training/checkpoints/step_4000_trainable.pt`，step=4000，SHA256=`5890b0d08f6e396299a47295f2e8b3987aafb477e7bfd571287d875525cb9575`。

执行保持 full30 / 30-target chunk / reach-conditioned / 10mm / 0.08rad / dwell30 / budget900 / Euler5 / benchmark_assist。基座、控制器、动作表示、归一化和物理源码未改。

| Cohort | n | B held15 | R held15 | B detach | R detach | B release | R release | B strict | R strict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dev8 | 8 | 6 | 7 | 6 | 7 | 4 | 1 | 0 | 0 |
| dev8_train2 | 2 | 1 | 2 | 1 | 2 | 0 | 0 | 0 | 0 |
| dev8_val6 | 6 | 5 | 5 | 5 | 5 | 4 | 1 | 0 | 0 |
| historical_seed42_43 | 2 | 2 | 2 | 2 | 2 | 0 | 1 | 0 | 1 |

Strict = 同果 held15 → detach → release → unheld 且连续60控制步留桶；legacy in-bucket 不计 strict。B 为原冻结 D0 配对 traces，10/10 reset（含双路RGB哈希）完全一致。GT 原同条件 dev8 strict 8/8，控制源码哈希仍一致；本轮未重复整套 GT gate。

| Scene / RNG | split | B held15 | R held15 | B strict | R strict |
|---|---|---:|---:|---:|---:|
| 2013322 / 42 | train | 1 | 1 | 0 | 0 |
| 2014461 / 42 | train | 0 | 1 | 0 | 0 |
| 2010914 / 42 | val | 1 | 1 | 0 | 0 |
| 2012485 / 42 | val | 1 | 1 | 0 | 0 |
| 2015716 / 42 | val | 1 | 1 | 0 | 0 |
| 2014362 / 42 | val | 0 | 0 | 0 | 0 |
| 2015751 / 42 | val | 1 | 1 | 0 | 0 |
| 2011310 / 42 | val | 1 | 1 | 0 | 0 |
| 2010600 / 42 | val | 1 | 1 | 0 | 1 |
| 2010600 / 43 | val | 1 | 1 | 0 | 0 |

| Offline full30 group | B xyz mm | R xyz mm | B rot rad | R rot rad | B width mm | R width mm |
|---|---:|---:|---:|---:|---:|---:|
| old_train/all | 83.80 | 59.07 | 0.1209 | 0.1027 | 8.54 | 5.79 |
| old_val/all | 94.78 | 66.74 | 0.1560 | 0.1284 | 9.70 | 6.90 |
| new_val/all | 72.80 | 49.66 | 0.1171 | 0.0999 | 7.44 | 5.33 |
| train/reset | 40.77 | 35.24 | 0.0553 | 0.0450 | 4.26 | 3.69 |
| val/reset | 40.99 | 42.09 | 0.0664 | 0.0587 | 5.25 | 4.55 |
| val/phase_REACH | 44.49 | 46.68 | 0.0676 | 0.0692 | 5.08 | 4.87 |
| val/phase_GRASP | 97.92 | 38.59 | 0.1093 | 0.0770 | 9.71 | 5.68 |
| val/phase_PULL | 57.68 | 33.41 | 0.1062 | 0.0575 | 9.71 | 7.62 |
| val/phase_TRANSPORT | 212.50 | 222.68 | 0.5142 | 0.5183 | 12.17 | 7.91 |

训练4000次真实更新。配比 `{'original_B': 2800, 'R2': 200, 'R1': 500, 'R3': 200, 'R_continuation_teacher': 300}`，同24条 recovery（R1=12/R2=6/R3=6），12个来源 episodes。合法 recovery windows=593，实际曝光 unique=411，重复抽样=789。Teacher 和 oracle 原24/24 PASS；本版本复用哈希固定数据，未重新生成 label。

实际 observation anchor phase：
```json
{
  "original_B": {
    "REACH": 1260,
    "GRASP": 700,
    "PULL": 420,
    "TRANSPORT": 280,
    "DROP": 140
  },
  "R2": {
    "TRANSPORT": 200
  },
  "R1": {
    "GRASP": 500
  },
  "R3": {
    "TRANSPORT": 200
  },
  "R_continuation_teacher": {
    "DROP": 100,
    "TRANSPORT": 150,
    "DONE": 50
  }
}
```
实际未来30 target phases：
```json
{
  "original_B": {
    "REACH": 4772,
    "GRASP": 12668,
    "PULL": 50128,
    "TRANSPORT": 10885,
    "DROP": 5209,
    "DONE": 338
  },
  "R2": {
    "TRANSPORT": 6000
  },
  "R1": {
    "GRASP": 654,
    "PULL": 14346
  },
  "R3": {
    "TRANSPORT": 6000
  },
  "R_continuation_teacher": {
    "DROP": 4594,
    "DONE": 2110,
    "TRANSPORT": 2296
  }
}
```

小样本限制：dev8不是最终测试集；历史补充两个seed属于同一episode；离线窗口、目标和噪声seed相关。冻结 heldout12 / fresh24 未参与本版本数据或配比决策。完整 traces、双路视频、metrics、load manifest、训练审计和失败证据保存在本目录。

最终cohort说明：辅助2010600属于历史heldout12，先前已观察，不能声称整批12例从未接触。最终endpoint按dev8 primary规则冻结后才运行完整配对测试，其余11例单独报告；训练来源未含任何heldout，fresh24未运行。见上级summary_report.md。
