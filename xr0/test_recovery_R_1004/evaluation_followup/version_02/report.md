# version_02：R4000 测评

Endpoint: `/home/rosmontis/Projects/dualsys/Xiaomi-Robotics-0/xr0/test_recovery_R_1004/evaluation_followup/version_02/training/checkpoints/step_4000_trainable.pt`，step=4000，SHA256=`c8e2e3605850682f84113a4e74cf4172ece91ba8719e5d3c89458aadde786766`。

执行保持 full30 / 30-target chunk / reach-conditioned / 10mm / 0.08rad / dwell30 / budget900 / Euler5 / benchmark_assist。基座、控制器、动作表示、归一化和物理源码未改。

| Cohort | n | B held15 | R held15 | B detach | R detach | B release | R release | B strict | R strict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dev8 | 8 | 6 | 5 | 6 | 5 | 4 | 2 | 0 | 0 |
| dev8_train2 | 2 | 1 | 1 | 1 | 1 | 0 | 1 | 0 | 0 |
| dev8_val6 | 6 | 5 | 4 | 5 | 4 | 4 | 1 | 0 | 0 |
| historical_seed42_43 | 2 | 2 | 1 | 2 | 1 | 0 | 0 | 0 | 0 |

Strict = 同果 held15 → detach → release → unheld 且连续60控制步留桶；legacy in-bucket 不计 strict。B 为原冻结 D0 配对 traces，10/10 reset（含双路RGB哈希）完全一致。GT 原同条件 dev8 strict 8/8，控制源码哈希仍一致；本轮未重复整套 GT gate。

| Scene / RNG | split | B held15 | R held15 | B strict | R strict |
|---|---|---:|---:|---:|---:|
| 2013322 / 42 | train | 1 | 1 | 0 | 0 |
| 2014461 / 42 | train | 0 | 0 | 0 | 0 |
| 2010914 / 42 | val | 1 | 1 | 0 | 0 |
| 2012485 / 42 | val | 1 | 0 | 0 | 0 |
| 2015716 / 42 | val | 1 | 1 | 0 | 0 |
| 2014362 / 42 | val | 0 | 1 | 0 | 0 |
| 2015751 / 42 | val | 1 | 1 | 0 | 0 |
| 2011310 / 42 | val | 1 | 0 | 0 | 0 |
| 2010600 / 42 | val | 1 | 1 | 0 | 0 |
| 2010600 / 43 | val | 1 | 0 | 0 | 0 |

| Offline full30 group | B xyz mm | R xyz mm | B rot rad | R rot rad | B width mm | R width mm |
|---|---:|---:|---:|---:|---:|---:|
| old_train/all | 83.80 | 52.23 | 0.1209 | 0.0975 | 8.54 | 6.29 |
| old_val/all | 94.78 | 62.34 | 0.1560 | 0.1257 | 9.70 | 6.87 |
| new_val/all | 72.80 | 47.66 | 0.1171 | 0.1005 | 7.44 | 5.50 |
| train/reset | 40.77 | 26.04 | 0.0553 | 0.0363 | 4.26 | 3.14 |
| val/reset | 40.99 | 35.66 | 0.0664 | 0.0466 | 5.25 | 3.38 |
| val/phase_REACH | 44.49 | 40.00 | 0.0676 | 0.0589 | 5.08 | 4.43 |
| val/phase_GRASP | 97.92 | 39.75 | 0.1093 | 0.0699 | 9.71 | 5.89 |
| val/phase_PULL | 57.68 | 31.13 | 0.1062 | 0.0694 | 9.71 | 7.80 |
| val/phase_TRANSPORT | 212.50 | 211.96 | 0.5142 | 0.5361 | 12.17 | 8.66 |

训练4000次真实更新。配比 `{'original_B': 3000, 'R1': 500, 'R3': 250, 'R2': 250}`，同24条 recovery（R1=12/R2=6/R3=6），12个来源 episodes。合法 recovery windows=401，实际曝光 unique=306，重复抽样=694。Teacher 和 oracle 原24/24 PASS；本版本复用哈希固定数据，未重新生成 label。

实际 observation anchor phase：
```json
{
  "original_B": {
    "GRASP": 750,
    "REACH": 1350,
    "PULL": 450,
    "TRANSPORT": 300,
    "DROP": 150
  },
  "R1": {
    "GRASP": 500
  },
  "R3": {
    "TRANSPORT": 250
  },
  "R2": {
    "TRANSPORT": 250
  }
}
```
实际未来30 target phases：
```json
{
  "original_B": {
    "GRASP": 13559,
    "PULL": 53529,
    "REACH": 5125,
    "TRANSPORT": 12261,
    "DROP": 5138,
    "DONE": 388
  },
  "R1": {
    "GRASP": 632,
    "PULL": 14368
  },
  "R3": {
    "TRANSPORT": 7500
  },
  "R2": {
    "TRANSPORT": 7500
  }
}
```

小样本限制：dev8不是最终测试集；历史补充两个seed属于同一episode；离线窗口、目标和噪声seed相关。冻结 heldout12 / fresh24 未参与本版本数据或配比决策。完整 traces、双路视频、metrics、load manifest、训练审计和失败证据保存在本目录。

最终cohort说明：辅助2010600属于历史heldout12，先前已观察，不能声称整批12例从未接触。最终endpoint按dev8 primary规则冻结后才运行完整配对测试，其余11例单独报告；训练来源未含任何heldout，fresh24未运行。见上级summary_report.md。
