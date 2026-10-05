"""Report existing evaluator results without changing observers or thresholds."""
import bootstrap
import argparse, json, statistics
from collections import Counter
from recovery_common import ROOT, sha, write

FIELDS = {
    'held15': 'held15', 'detach': 'same_fruit_detach',
    'bucket-neighborhood': 'bucket_neighborhood',
    'reasonable-release': 'reasonable_release', 'stable-bucket': 'stable_bucket',
    'strict': 'strict', 'long-held-stall': 'long_held_stall',
    'premature-release': 'premature_release', 'physical-anomaly': 'physical_anomaly',
    'strict_original_observer': 'strict_original_observer',
}

def summarize(round_number):
    directory = ROOT / f'round{round_number}'
    result = dict(round=round_number, code_sha256=sha(__file__), endpoints={}, training_endpoints={})
    for endpoint in ['S8', 'J8', 'J16']:
        arm = endpoint[0]
        updates = int(endpoint[1:]) * 1000
        training = directory / arm / 'training'
        checkpoint = training / 'checkpoints' / f'step_{updates:04d}_trainable.pt'
        if checkpoint.exists():
            records = [json.loads(line) for line in (training / 'training_steps.jsonl').read_text().splitlines()][:updates]
            assert len(records) == updates
            manifest = json.loads((training / 'run_manifest.json').read_text())
            result['training_endpoints'][endpoint] = dict(
                updates=updates, checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                initialization=manifest['provenance']['initialization']['initialization'],
                base_sha256=manifest['provenance']['base_sha256'],
                B_sha256=manifest['provenance']['B_sha256'],
                optimizer=manifest['configuration']['optimizer'],
                fresh_optimizer_state_entries=manifest['fresh_optimizer_state_entries'],
                source_counts=dict(Counter(row['source'] for row in records)),
                loss_by_source={source: {
                    label: statistics.mean(row['flow_loss'] for row in subset if row['source'] == source)
                    for label, subset in [('first500', records[:500]), ('last500', records[-500:])]
                } for source in ['original', 'existing_R', 'live_late']})
        evaluation = directory / endpoint / 'evaluation'
        if not (evaluation / 'aggregate.json').exists():
            continue
        cases = json.loads((evaluation / 'cases.json').read_text())
        protocol = json.loads((evaluation / 'protocol.json').read_text())
        rows = []
        for case in cases:
            row = dict(case)
            row['failure_original_diagnostic'] = row['failure']
            if row['strict_original_observer'] and not row['strict']:
                row['failure'] = 'stable_bucket_but_release_outside_reasonable_region'
            rows.append(row)
        groups = {}
        for cohort in ['dev8', 'dev8_train2', 'dev8_val6', 'historical_seed42_43']:
            subset = [c for c in rows if c['cohort'] == cohort or
                      cohort == 'dev8_train2' and c['cohort'] == 'dev8' and c['split'] == 'train' or
                      cohort == 'dev8_val6' and c['cohort'] == 'dev8' and c['split'] == 'val']
            groups[cohort] = dict(n=len(subset), **{
                name: sum(bool(c[field]) for c in subset) for name, field in FIELDS.items()
            })
        write(evaluation / 'unified_cases.json', rows)
        result['endpoints'][endpoint] = dict(
            checkpoint=protocol['checkpoint'], checkpoint_sha256=protocol['checkpoint_sha256'],
            original_cases_sha256=sha(evaluation / 'cases.json'), cohorts=groups)
    if all(name in result['training_endpoints'] for name in ['S8', 'J8']):
        logs = {
            arm: [json.loads(line) for line in (directory / arm / 'training/training_steps.jsonl').read_text().splitlines()][:8000]
            for arm in ['S', 'J']
        }
        keys = ['step', 'source', 'group', 'window_id']
        assert all([logs['S'][i][key] for key in keys] == [logs['J'][i][key] for key in keys] for i in range(8000))
        result['schedule_parity'] = dict(actual_D_A_updates=8000, identical_S_J_executed_rows=True, compared_fields=keys)
        write(directory / 'schedule_parity.json', result['schedule_parity'])
    if round_number == 3 and all(name in result['training_endpoints'] for name in ['S8', 'J8', 'J16']):
        control = {}
        for arm, updates in [('S', 8000), ('J', 16000)]:
            old = ROOT / 'round2' / arm / 'training'
            new = directory / arm / 'training'
            before = [json.loads(line) for line in (old / 'training_steps.jsonl').read_text().splitlines()]
            after = [json.loads(line) for line in (new / 'training_steps.jsonl').read_text().splitlines()]
            assert len(before) == len(after) == updates
            fields = ['step', 'source', 'group', 'window_id']
            assert all([a[key] for key in fields] == [b[key] for key in fields] for a, b in zip(before, after))
            assert {row['lr'] for row in before} == {1e-5} and {row['lr'] for row in after} == {5e-6}
            prior = json.loads((old / 'run_manifest.json').read_text())
            current = json.loads((new / 'run_manifest.json').read_text())
            for key in ['base_sha256', 'B_sha256', 'dataset_sha256', 'schedule_sha256', 'stats_sha256', 'source_sha256']:
                assert prior['provenance'][key] == current['provenance'][key], key
            assert {k: v for k, v in prior['configuration'].items() if k != 'optimizer'} == {k: v for k, v in current['configuration'].items() if k != 'optimizer'}
            assert {k: v for k, v in prior['configuration']['optimizer'].items() if k != 'lr'} == {k: v for k, v in current['configuration']['optimizer'].items() if k != 'lr'}
            assert prior['fresh_optimizer_state_entries'] == current['fresh_optimizer_state_entries'] == 0
            control[arm] = dict(updates=updates, identical_executed_rows=True, identical_initialization_and_other_configuration=True, fresh_optimizer=True, previous_lr=1e-5, current_lr=5e-6)
        result['round2_lr_control'] = control
        write(directory / 'learning_rate_control.json', control)
    write(directory / 'closed_loop_summary.json', result)
    print(json.dumps(result, ensure_ascii=False))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--round', type=int, choices=[1, 2, 3], default=1)
    summarize(parser.parse_args().round)
