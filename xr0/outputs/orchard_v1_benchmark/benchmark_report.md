# OrchardBench V1 Closed-Loop Learning Curve Benchmark

Run scope: **FULL 5 x 30**.

30 frozen feasible seeds; visualization seeds are the frozen first five. 30-step full chunks with live observations; 900 control steps at 30 Hz; policy inference seed 42; benchmark-assist grasp; strict physical bucket success.

Evaluated seeds (same order for each selected checkpoint): `[3010000, 3010001, 3010002, 3010003, 3010004, 3010005, 3010006, 3010007, 3010008, 3010009, 3010010, 3010011, 3010012, 3010013, 3010014, 3010015, 3010016, 3010017, 3010018, 3010019, 3010020, 3010021, 3010022, 3010023, 3010024, 3010025, 3010026, 3010027, 3010028, 3010029]`

Rates and capability stages exclude runtime errors. Completed = valid denominator. Clipping/IK rates use the total returned control steps of valid episodes. Mean branch breaks uses each valid episode's maximum observed count. Error episodes retain partial diagnostics in episodes.jsonl.

| checkpoint | requested | valid | errors | success | grasp | detach | clipping | IK fail | mean steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pretrained | 30 | 30 | 0 | 0.00% | 0.00% | 0.00% | 98.25% | 62.48% | 900.0 |
| step_1000 | 30 | 30 | 0 | 0.00% | 0.00% | 0.00% | 72.61% | 26.32% | 900.0 |
| step_3000 | 30 | 30 | 0 | 0.00% | 0.00% | 0.00% | 70.31% | 16.29% | 900.0 |
| step_6000 | 30 | 30 | 0 | 0.00% | 0.00% | 0.00% | 56.37% | 30.75% | 900.0 |
| step_10000 | 30 | 30 | 0 | 0.00% | 0.00% | 0.00% | 71.52% | 15.01% | 900.0 |

| checkpoint | NO_GRASP | GRASPED_NOT_DETACHED | DETACHED_NOT_PLACED | SUCCESS |
| --- | ---: | ---: | ---: | ---: |
| pretrained | 30 | 0 | 0 | 0 |
| step_1000 | 30 | 0 | 0 | 0 |
| step_3000 | 30 | 0 | 0 | 0 |
| step_6000 | 30 | 0 | 0 | 0 |
| step_10000 | 30 | 0 | 0 | 0 |

- pretrained: largest failure stage(s): NO_GRASP (30 each).
- step_1000: largest failure stage(s): NO_GRASP (30 each).
- step_3000: largest failure stage(s): NO_GRASP (30 each).
- step_6000: largest failure stage(s): NO_GRASP (30 each).
- step_10000: largest failure stage(s): NO_GRASP (30 each).

Observed success rate: pretrained 0.00% → step_10000 0.00%.

Video errors: 0.

Plot status: complete.

NO TRAINING WAS STARTED. NO CHECKPOINT WAS MODIFIED.
