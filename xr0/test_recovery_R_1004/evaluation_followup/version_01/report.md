# version_01：R4000 测评

Endpoint: `/home/rosmontis/Projects/dualsys/Xiaomi-Robotics-0/xr0/test_recovery_R_1004/training/checkpoints/step_4000_trainable.pt`，step=4000，SHA256=`a929fd15078b6cc72881c200b43ae4d26425a021f1346627414b08723a7a1a7d`。

执行保持 full30 / 30-target chunk / reach-conditioned / 10mm / 0.08rad / dwell30 / budget900 / Euler5 / benchmark_assist。基座、控制器、动作表示、归一化和物理源码未改。

| Cohort | n | B held15 | R held15 | B detach | R detach | B release | R release | B strict | R strict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dev8 | 8 | 6 | 5 | 6 | 5 | 4 | 2 | 0 | 1 |
| dev8_train2 | 2 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
| dev8_val6 | 6 | 5 | 4 | 5 | 4 | 4 | 2 | 0 | 1 |
| historical_seed42_43 | 2 | 2 | 2 | 2 | 2 | 0 | 1 | 0 | 1 |

Strict = 同果 held15 → detach → release → unheld 且连续60控制步留桶；legacy in-bucket 不计 strict。B 为原冻结 D0 配对 traces，10/10 reset（含双路RGB哈希）完全一致。GT 原同条件 dev8 strict 8/8，控制源码哈希仍一致；本轮未重复整套 GT gate。

| Scene / RNG | split | B held15 | R held15 | B strict | R strict |
|---|---|---:|---:|---:|---:|
| 2013322 / 42 | train | 1 | 0 | 0 | 0 |
| 2014461 / 42 | train | 0 | 1 | 0 | 0 |
| 2010914 / 42 | val | 1 | 1 | 0 | 0 |
| 2012485 / 42 | val | 1 | 0 | 0 | 0 |
| 2015716 / 42 | val | 1 | 1 | 0 | 1 |
| 2014362 / 42 | val | 0 | 0 | 0 | 0 |
| 2015751 / 42 | val | 1 | 1 | 0 | 0 |
| 2011310 / 42 | val | 1 | 1 | 0 | 0 |
| 2010600 / 42 | val | 1 | 1 | 0 | 1 |
| 2010600 / 43 | val | 1 | 1 | 0 | 0 |

| Offline full30 group | B xyz mm | R xyz mm | B rot rad | R rot rad | B width mm | R width mm |
|---|---:|---:|---:|---:|---:|---:|
| old_train/all | 83.80 | 70.79 | 0.1209 | 0.1302 | 8.54 | 7.74 |
| old_val/all | 94.78 | 86.00 | 0.1560 | 0.1772 | 9.70 | 7.99 |
| new_val/all | 72.80 | 69.52 | 0.1171 | 0.1393 | 7.44 | 6.64 |
| train/reset | 40.77 | 127.55 | 0.0553 | 0.2010 | 4.26 | 7.54 |
| val/reset | 40.99 | 109.64 | 0.0664 | 0.1658 | 5.25 | 6.95 |
| val/phase_REACH | 44.49 | 87.47 | 0.0676 | 0.1348 | 5.08 | 6.41 |
| val/phase_GRASP | 97.92 | 58.14 | 0.1093 | 0.1121 | 9.71 | 7.23 |
| val/phase_PULL | 57.68 | 38.05 | 0.1062 | 0.0976 | 9.71 | 8.18 |
| val/phase_TRANSPORT | 212.50 | 214.05 | 0.5142 | 0.5400 | 12.17 | 8.26 |

训练4000次真实更新。配比 `{'original_B': 2000, 'R3': 500, 'R1': 1000, 'R2': 500}`，同24条 recovery（R1=12/R2=6/R3=6），12个来源 episodes。合法 recovery windows=720，实际曝光 unique=663，重复抽样=1337。Teacher 和 oracle 原24/24 PASS；本版本复用哈希固定数据，未重新生成 label。

实际 observation anchor phase：
```json
{
  "original_B": {
    "REACH": 900,
    "TRANSPORT": 200,
    "GRASP": 500,
    "PULL": 300,
    "DROP": 100
  },
  "R3": {
    "TRANSPORT": 500
  },
  "R1": {
    "PULL": 863,
    "GRASP": 137
  },
  "R2": {
    "TRANSPORT": 500
  }
}
```
实际未来30 target phases：
```json
{
  "original_B": {
    "GRASP": 8984,
    "PULL": 35862,
    "REACH": 3412,
    "TRANSPORT": 8073,
    "DROP": 3412,
    "DONE": 257
  },
  "R3": {
    "TRANSPORT": 15000
  },
  "R1": {
    "PULL": 28154,
    "TRANSPORT": 1644,
    "GRASP": 202
  },
  "R2": {
    "TRANSPORT": 15000
  }
}
```

中途 takeover 配对诊断有 6 个可比较状态。它们来自训练来源，不能作为泛化成绩。一个 seed43 前缀因 measured width 相差0.1286mm，超过0.1mm gate，被保留为 unavailable；未放宽 gate，也未计作模型失败。另补同类别首个不同来源 accepted 状态，规则和失败记录在 takeovers/engineering_repairs.jsonl。
```json
{
  "pairs": [
    {
      "candidate": "episode_000023_rng42_R1_s0085",
      "category": "R1",
      "paired_available": true,
      "B_strict": false,
      "R_strict": false,
      "B_reason": "branch_break",
      "R_reason": "budget"
    },
    {
      "candidate": "episode_000329_rng42_R1_s0075",
      "category": "R1",
      "paired_available": true,
      "B_strict": false,
      "R_strict": false,
      "B_reason": "budget",
      "R_reason": "budget"
    },
    {
      "candidate": "episode_000613_rng42_R2_s0192",
      "category": "R2",
      "paired_available": true,
      "B_strict": false,
      "R_strict": false,
      "B_reason": "base_drift",
      "R_reason": "budget"
    },
    {
      "candidate": "episode_000613_rng42_R3_s0465",
      "category": "R3",
      "paired_available": true,
      "B_strict": false,
      "R_strict": false,
      "B_reason": "budget",
      "R_reason": "budget"
    },
    {
      "candidate": "episode_000613_rng43_R3_s0217",
      "category": "R3",
      "paired_available": false,
      "B_strict": null,
      "R_strict": null,
      "B_reason": "takeover_regeneration_unavailable",
      "R_reason": "takeover_regeneration_unavailable"
    },
    {
      "candidate": "episode_000703_rng42_R2_s0203",
      "category": "R2",
      "paired_available": true,
      "B_strict": false,
      "R_strict": false,
      "B_reason": "budget",
      "R_reason": "budget"
    },
    {
      "candidate": "episode_000703_rng42_R3_s0424",
      "category": "R3",
      "paired_available": true,
      "B_strict": false,
      "R_strict": false,
      "B_reason": "budget",
      "R_reason": "budget"
    }
  ],
  "frame0_offline": {
    "R1": {
      "B8000": {
        "position_mae_m": 0.06998021375155736,
        "rotation_mae_rad": 0.12925317427125782,
        "width_mae_m": 0.008887442178092897
      },
      "R4000": {
        "position_mae_m": 0.045911861927529336,
        "rotation_mae_rad": 0.06286775161288305,
        "width_mae_m": 0.004282196382215868
      }
    },
    "R2": {
      "B8000": {
        "position_mae_m": 0.2596338074127415,
        "rotation_mae_rad": 0.5552776600209314,
        "width_mae_m": 0.009098822328572473
      },
      "R4000": {
        "position_mae_m": 0.0941800764097337,
        "rotation_mae_rad": 0.2108687368863225,
        "width_mae_m": 0.0034042493983482323
      }
    },
    "R3": {
      "B8000": {
        "position_mae_m": 0.2749711208132407,
        "rotation_mae_rad": 0.5500311938979673,
        "width_mae_m": 0.013067323869715134
      },
      "R4000": {
        "position_mae_m": 0.12989628088437513,
        "rotation_mae_rad": 0.26896130179918126,
        "width_mae_m": 0.0027492614656997225
      }
    }
  },
  "role": "training-source mechanism diagnostic only; not held-out performance",
  "available_pairs": 6
}
```

小样本限制：dev8不是最终测试集；历史补充两个seed属于同一episode；离线窗口、目标和噪声seed相关。冻结 heldout12 / fresh24 未参与本版本数据或配比决策。完整 traces、双路视频、metrics、load manifest、训练审计和失败证据保存在本目录。

最终cohort说明：辅助2010600属于历史heldout12，先前已观察，不能声称整批12例从未接触。最终endpoint按dev8 primary规则冻结后才运行完整配对测试，其余11例单独报告；训练来源未含任何heldout，fresh24未运行。见上级summary_report.md。
