# Integrated live-state experiment (2026-10-05)

Only mining student: current-only R03-8k. Production files are unchanged.
`collect_live.py` runs student from reset, pauses before teacher takeover, then
uses the same env and original action adapter/controller until strict success.
Teacher never calls hold/release, writes simulator state, or regenerates a prefix.
Student prefix and teacher suffix are stored separately, including failed attempts.
Truth and phase are diagnostic/teacher only; `inputs.py` whitelists current policy fields.

Collection target: 72 accepted, at least 20 source episodes; freeze minimum 60.
Fixed training source order, inference seeds 42/43/44 and stall eligibility onset
350/550/750 control steps are declared in protocol and collection_plan before mining.
After the suffix-budget repair, collection reset config reserves 1800 control steps: student max 900 plus teacher suffix max 900. Formal S/J evaluator remains 900. This changes only the collection termination horizon, not production control or physics.
Three early accepted samples completed within the original 900 total budget; 69 later accepted samples used 1800. Every accepted collector SHA maps to a preserved source snapshot.
All three conditions are checked on every run. They change only when takeover may
happen, never student behavior. Natural category distribution is reported.

After collection, `freeze_data.py` freezes dataset manifest, available window counts,
media/annotation hashes and D_A/D_B (8k each, independent fixed seeds). Ratios are
60% original / 15% existing R / 25% live late. Original internal ratios are
25% early / 35% grasp-pull / 25% transport / 15% drop-done. No historical B quota.

`run_round1.py` queues S8, J8, J16 training/evaluation with fresh AdamW for S/J.
J continues its optimizer/RNG from D_A to D_B. It stops after Round1 and does not
open another round or consume fresh24. Root agent decides any evidence-supported
Round2/3 and final endpoint. At most three complete scheme revisions are allowed.

Artifacts: collection_summary.json, dataset_manifest.jsonl, dataset_statistics.json,
frozen.json, schedule_D_A.jsonl, schedule_D_B.jsonl, round1/{S,J}/training,
round1/{S8,J8,J16}/evaluation, receipts and final_report.md.

The original StrictPlacement observer and reach loop are reused unchanged. Primary
summary also requires reasonable release of that exact strict-success fruit chain,
as requested; raw observer success is retained as strict_original_observer.

Round2 uses the unchanged D* with common 55/15/30 sampling and independently frozen schedules in round2/. Both routes restart from historical B8000/full10k with fresh AdamW. run_next.py accepts only Round2/3 and stops after the selected comparison; final_report.md reports all rounds.

Round3 reuses Round2 D_A/D_B byte for byte and changes only the common learning rate to 5e-6. The same S8/J8/J16 comparison stops after this final allowed round. Actual optimizer rows and initialization are checked in learning_rate_control.json.

## Git archive

Git tracks collector source snapshots, training/evaluation scripts, protocols,
candidate decisions, source/takeover provenance, the dataset manifest and hashes,
all frozen schedules, executed optimizer-step records, per-case result summaries,
checkpoint references and hashes, execution receipts, and the final Case D report.
The scoped `.gitignore` retains these records without changing production files.

Checkpoints, RGB videos, raw per-frame sensor/action/physics traces, runtime logs,
and caches remain local and ignored. In particular, `recoveries/*/trajectory.json`
and raw student prefixes are referenced by the frozen manifests but are not Git
payloads. Reproducing training requires those local assets plus the original
OrchardBench data and original full10k/B8000 checkpoints at the recorded paths;
a Git checkout alone does not contain the complete dataset or model weights.
