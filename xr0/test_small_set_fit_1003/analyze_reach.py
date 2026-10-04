#!/usr/bin/env python3
"""CPU-only analysis of completed M0 reach rollouts; never modify raw files.

Only rollout_analysis.json and rollout_comparison.csv are replaced on each run.
The summaries describe boundary observations, not the unobserved physics frames
inside a control step. Nearest fruit identities are retained and may change.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXAMPLE_LIMIT = 20
NEAR_THRESHOLDS_M = (.05, .10)


def percentile(values, q):
    values = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    if not values:
        return None
    x = (len(values) - 1) * q
    lo, hi = math.floor(x), math.ceil(x)
    return values[lo] + (values[hi] - values[lo]) * (x - lo)


def distribution(values):
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return dict(count=len(values), min=min(values) if values else None,
                p50=percentile(values, .5), p95=percentile(values, .95),
                max=max(values) if values else None,
                mean=sum(values) / len(values) if values else None)


def rate(count, total):
    return count / total if total else None


def event(row, boundary='after', control_hz=30.):
    after = boundary == 'after'
    return dict(command_control_step=row['control_step'], boundary=boundary,
                completed_control_steps_at_boundary=row['control_step'] - int(not after),
                sim_time_s=row['sim_time'] - (0. if after else 1. / control_hz),
                chunk_id=row['chunk_id'], target_k=row['target_k'])


def grasp_info(info):
    held = any(info.get(k) is not None and info[k] >= 0
               for k in ('held_apple_id_debug', 'grasped_apple_id_debug'))
    return bool(info.get('grasp_assist_triggered', False) or held)


def boundary_summary(steps, boundary, control_hz):
    intents = Counter()
    counts = Counter()
    examples = defaultdict(list)
    nearest_seen = set()
    nearest_transitions = 0
    previous_nearest = None
    minimum_tcp = None
    minimum_palm = None
    first_close = None
    for row in steps:
        snap = row['grasp_' + boundary]
        stamp = event(row, boundary, control_hz)
        intent = snap['gripper_intent']
        intents['negative' if intent < 0 else 'positive' if intent > 0 else 'zero'] += 1
        if intent < 0 and first_close is None:
            first_close = dict(stamp, intent=intent, width_m=snap['measured_width_m'],
                               commanded_width_m=snap['commanded_width_m'])
        contact_apples = snap.get('contact_apples', [])
        nearest = snap.get('nearest_tcp_apple')
        palm_nearest = snap.get('nearest_palm_center_apple')
        if nearest is not None:
            nearest_seen.add(nearest['apple_id'])
            if previous_nearest is not None and previous_nearest != nearest['apple_id']:
                nearest_transitions += 1
            previous_nearest = nearest['apple_id']
            for threshold in NEAR_THRESHOLDS_M:
                counts[f'nearest_tcp_le_{threshold:.2f}m'] += int(nearest['tcp_distance_m'] <= threshold)
            if minimum_tcp is None or nearest['tcp_distance_m'] < minimum_tcp['distance_m']:
                minimum_tcp = dict(stamp, apple_id=nearest['apple_id'],
                                   distance_m=nearest['tcp_distance_m'],
                                   hand_local_position=nearest['hand_local_position'],
                                   world_position=nearest['world_position'])
        if palm_nearest is not None:
            if minimum_palm is None or palm_nearest['palm_center_distance_m'] < minimum_palm['distance_m']:
                minimum_palm = dict(stamp, apple_id=palm_nearest['apple_id'],
                                    distance_m=palm_nearest['palm_center_distance_m'],
                                    hand_local_position=palm_nearest['hand_local_position'],
                                    world_position=palm_nearest['world_position'])
        # The nearest palm-center fruit is not guaranteed to be the only fruit
        # inside the rectangular palm volume. Include all contact fruit as well.
        represented = {a['apple_id']: a for a in contact_apples}
        for candidate in (nearest, palm_nearest):
            if candidate is not None:
                represented[candidate['apple_id']] = candidate
        inside = sorted(a['apple_id'] for a in represented.values() if a['inside_palm_volume'])
        both = sorted(a['apple_id'] for a in contact_apples if a['both_fingers_contact'])
        eligible = snap.get('eligible_apple_ids', [])
        nonclosing = bool(eligible) and intent >= 0.
        sole_intent_block = (nonclosing and snap['assist_enabled'] and snap['held_apple_id'] is None
                             and len(eligible) == 1)
        flags = dict(any_finger_contact=bool(contact_apples),
                     represented_apple_inside_palm=bool(inside),
                     both_fingers_same_apple=bool(both),
                     geometry_eligible=bool(eligible),
                     geometry_eligible_nonclosing=nonclosing,
                     unique_geometry_with_intent_as_only_snapshot_block=sole_intent_block,
                     would_attach_now=bool(snap['would_attach_now']))
        for key, present in flags.items():
            counts[key] += int(present)
            if present and len(examples[key]) < EXAMPLE_LIMIT:
                examples[key].append(dict(stamp, gripper_intent=intent,
                    held_apple_id=snap['held_apple_id'], eligible_apple_ids=eligible,
                    inside_represented_apple_ids=inside, both_finger_apple_ids=both,
                    contact_apple_ids=[a['apple_id'] for a in contact_apples]))
    total = len(steps)
    return dict(samples=total,
        intent_counts={k: intents[k] for k in ('negative', 'positive', 'zero')},
        intent_rates={k: rate(intents[k], total) for k in ('negative', 'positive', 'zero')},
        counts=dict(counts), rates={k: rate(v, total) for k, v in counts.items()},
        first_negative_intent=first_close,
        minimum_tcp_distance=minimum_tcp, minimum_palm_center_distance=minimum_palm,
        nearest_tcp_apple_ids_seen=sorted(nearest_seen), nearest_tcp_apple_id_changes=nearest_transitions,
        examples=dict(examples), example_limit_per_type=EXAMPLE_LIMIT,
        inside_palm_count_scope='represented contact apples plus nearest TCP/palm-center apples; may undercount other noncontact apples')


def analyze_episode(summary, steps, chunks, directory, initial_info=None):
    n = len(steps)
    provider = summary['checkpoint']
    environment = summary['environment']
    hz = environment['sim_hz'] / environment['action_repeat']
    clipping = Counter()
    advance_reasons = Counter()
    dwell_timeout_events = []
    observer_failed_fields = Counter()
    observer_failed_steps = set()
    observer_examples = []
    observer_missing_rows = 0
    recorded_observer_disagreement = 0
    chunk_rows = defaultdict(list)
    first_grasp = None
    first_success = None
    for row in steps:
        info = row['info']
        diag = row['command_diagnostics']
        after = row['grasp_after']
        for key, value in diag['clipping_components'].items():
            clipping[key] += int(bool(value))
        advance_reasons[row['advance_reason']] += 1
        if row['advance_reason'] == 'maximum_dwell':
            dwell_timeout_events.append(dict(event(row, control_hz=hz),
                position_error_m=row['position_error_m'], rotation_error_rad=row['rotation_error_rad'],
                dwell_steps=row['dwell_steps']))
        chunk_rows[row['chunk_id']].append(row)
        recomputed = dict(
            clipping=bool(diag['any_action_clipping']) == bool(info['action_clipped']),
            intent=diag['predicted_gripper_intent'] == after['gripper_intent'],
            open_steps=diag['predicted_width_open_steps'] == after['width_open_steps'])
        recorded = row.get('observer_consistency')
        if recorded is None:
            observer_missing_rows += 1
        elif recorded != recomputed:
            recorded_observer_disagreement += 1
        failed = [k for k, okay in recomputed.items() if not okay]
        if failed:
            observer_failed_steps.add(row['control_step'])
            observer_failed_fields.update(failed)
            if len(observer_examples) < EXAMPLE_LIMIT:
                observer_examples.append(dict(event(row, control_hz=hz), failed_fields=failed))
        if first_grasp is None and grasp_info(info):
            first_grasp = dict(event(row, control_hz=hz),
                held_apple_id=info.get('held_apple_id_debug'),
                grasped_apple_id=info.get('grasped_apple_id_debug'),
                grasp_assist_triggered=bool(info.get('grasp_assist_triggered')))
        if first_success is None and info['success']:
            first_success = event(row, control_hz=hz)
    durations = []
    for chunk in chunks:
        cid = chunk['chunk_id']
        rows = chunk_rows[cid]
        durations.append(dict(chunk_id=cid, control_steps=len(rows), seconds=len(rows) / hz,
            loaded_at_control_step=chunk['at_control_step'], start_k=chunk['start_k'], stop_k=chunk['stop_k'],
            last_executed_k=rows[-1]['target_k'] if rows else None,
            last_advance_reason=rows[-1]['advance_reason'] if rows else None,
            target_advance_events=sum(bool(r['advance']) for r in rows),
            maximum_dwell_observed=max((r['dwell_steps'] for r in rows), default=0),
            prediction_seconds=chunk.get('prediction_seconds')))
    reconstructed = dict(
        control_steps=n,
        ever_grasped=bool(grasp_info(initial_info or {}) or first_grasp is not None),
        success=bool(steps[-1]['info']['success']) if steps else bool((initial_info or {}).get('success')),
        ik_failed_steps=sum(bool(r['info']['ik_failed']) for r in steps),
        action_clipped_steps=sum(bool(r['info']['action_clipped']) for r in steps),
        reached_advances=sum(bool(r['advance'] and r['reached']) for r in steps),
        dwell_timeouts=advance_reasons['maximum_dwell'],
        chunks_loaded=len(chunks),
        replans=0 if provider == 'gt' else len(chunks),
    )
    mismatches = []
    for key, value in reconstructed.items():
        # An execution exception intentionally invalidates summary.success.
        if key == 'success' and summary.get('termination') == 'error':
            continue
        if summary.get(key) != value:
            mismatches.append(dict(field=key, summary=summary.get(key), reconstructed=value))
    if [r['control_step'] for r in steps] != list(range(1, n + 1)):
        mismatches.append(dict(field='control_step_sequence', expected='consecutive 1..N'))
    if sum(x['control_steps'] for x in durations) != n:
        mismatches.append(dict(field='chunk_step_coverage', expected=n))
    if summary.get('chunk_control_steps') != [x['control_steps'] for x in durations]:
        mismatches.append(dict(field='chunk_control_steps', summary=summary.get('chunk_control_steps'),
                               reconstructed=[x['control_steps'] for x in durations]))
    if summary.get('termination') == 'success' and not reconstructed['success']:
        mismatches.append(dict(field='success_termination_without_success'))
    before = boundary_summary(steps, 'before', hz)
    after = boundary_summary(steps, 'after', hz)
    return dict(directory=str(directory), provider=provider, mode=summary['mode'], seed=summary['seed'],
        episode_id=summary['episode_id'], termination=summary['termination'],
        valid_for_outcome=summary['termination'] != 'error',
        success=summary['success'], capability_stage=summary.get('capability_stage'),
        ever_grasped=summary['ever_grasped'],
        max_apple_detached_count=summary['max_apple_detached_count'],
        final_apple_in_bucket_count=summary['final_apple_in_bucket_count'],
        split=summary.get('split'), scene_check_pass=summary.get('scene_check_pass'),
        control_steps=n, replans=summary['replans'],
        chunks_loaded=len(chunks), wall_seconds=summary.get('wall_seconds'),
        first_grasp_info_event=first_grasp, first_success_info_event=first_success,
        advancement=dict(control_step_counts=dict(advance_reasons),
            reached_advances=reconstructed['reached_advances'], dwell_timeouts=len(dwell_timeout_events),
            dwell_timeout_events=dwell_timeout_events),
        clipping=dict(component_steps=dict(clipping), component_rates={k: rate(v, n) for k, v in clipping.items()},
            union_steps=reconstructed['action_clipped_steps'], union_rate=rate(reconstructed['action_clipped_steps'], n),
            note='component rates overlap; joint-command limits are not included in action_clipped'),
        ik_failed_steps=reconstructed['ik_failed_steps'], ik_failed_rate=rate(reconstructed['ik_failed_steps'], n),
        position_error_m=distribution(r['position_error_m'] for r in steps),
        rotation_error_rad=distribution(r['rotation_error_rad'] for r in steps),
        chunk_duration_steps=distribution(c['control_steps'] for c in durations), chunk_durations=durations,
        boundary_observations=dict(before=before, after=after),
        no_grasp_geometry_nonclosing_boundary_evidence=None if summary['termination'] == 'error' else bool(not reconstructed['ever_grasped'] and any(
            x['counts'].get('geometry_eligible_nonclosing', 0) for x in (before, after))),
        no_grasp_intent_only_snapshot_block_evidence=None if summary['termination'] == 'error' else bool(not reconstructed['ever_grasped'] and any(
            x['counts'].get('unique_geometry_with_intent_as_only_snapshot_block', 0) for x in (before, after))),
        observer_consistency=dict(mismatch_fields_total=sum(observer_failed_fields.values()),
            mismatch_steps=len(observer_failed_steps), mismatches_by_field=dict(observer_failed_fields),
            examples=observer_examples, recorded_missing_rows=observer_missing_rows,
            recorded_flags_disagreement_rows=recorded_observer_disagreement),
        summary_consistency=dict(mismatch_count=len(mismatches), mismatches=mismatches, reconstructed=reconstructed))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def csv_row(item):
    after = item['boundary_observations']['after']
    before = item['boundary_observations']['before']
    close = after['first_negative_intent']
    grasp = item['first_grasp_info_event']
    tcp = after['minimum_tcp_distance']
    palm = after['minimum_palm_center_distance']
    row = {k: item[k] for k in ('provider', 'mode', 'seed', 'termination', 'valid_for_outcome', 'success', 'ever_grasped', 'max_apple_detached_count', 'split', 'scene_check_pass',
                                'control_steps', 'replans', 'chunks_loaded', 'ik_failed_rate')}
    row.update(reached_advances=item['advancement']['reached_advances'],
        dwell_timeouts=item['advancement']['dwell_timeouts'],
        clipped_rate=item['clipping']['union_rate'],
        position_error_p50_m=item['position_error_m']['p50'], position_error_p95_m=item['position_error_m']['p95'],
        rotation_error_p50_rad=item['rotation_error_rad']['p50'], rotation_error_p95_rad=item['rotation_error_rad']['p95'],
        chunk_steps_p50=item['chunk_duration_steps']['p50'], chunk_steps_max=item['chunk_duration_steps']['max'],
        first_close_after_step=None if close is None else close['command_control_step'],
        first_grasp_step=None if grasp is None else grasp['command_control_step'],
        min_tcp_distance_after_m=None if tcp is None else tcp['distance_m'],
        min_tcp_apple_id=None if tcp is None else tcp['apple_id'],
        min_palm_distance_after_m=None if palm is None else palm['distance_m'],
        min_palm_apple_id=None if palm is None else palm['apple_id'],
        nearest_apple_id_changes_after=after['nearest_tcp_apple_id_changes'],
        both_fingers_boundary_samples=before['counts'].get('both_fingers_same_apple', 0) + after['counts'].get('both_fingers_same_apple', 0),
        geometry_nonclosing_boundary_samples=before['counts'].get('geometry_eligible_nonclosing', 0) + after['counts'].get('geometry_eligible_nonclosing', 0),
        no_grasp_intent_only_snapshot_block_evidence=item['no_grasp_intent_only_snapshot_block_evidence'],
        observer_mismatch_fields=item['observer_consistency']['mismatch_fields_total'],
        summary_mismatch_count=item['summary_consistency']['mismatch_count'])
    row.update({f'clip_{k}_rate': v for k, v in item['clipping']['component_rates'].items()})
    row.update({f'intent_{k}_after_rate': v for k, v in after['intent_rates'].items()})
    return row



def plot_rollouts(items):
    """Small static chart; numeric diagnostics remain valid if plotting fails."""
    if not items:
        return dict(status='no_completed_episodes')
    import os
    os.environ['MPLCONFIGDIR'] = str(ROOT / 'cache' / 'matplotlib')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as exc:
        return dict(status='unavailable', reason=str(exc))
    seeds = sorted(set(item['seed'] for item in items))
    colors = {('gt', 'reach-conditioned'): '#217a45', ('gt', 'time-indexed'): '#999999',
              ('original_10k', 'reach-conditioned'): '#d98220', ('best', 'reach-conditioned'): '#346bd1'}
    figure, axes = plt.subplots(len(seeds), 2, figsize=(14, max(3.5, len(seeds) * 3.1)),
                                squeeze=False)
    represented = []
    for index, seed in enumerate(seeds):
        left, right = axes[index]
        intent_axis = right.twinx()
        for item in items:
            if item['seed'] != seed:
                continue
            rows = read_jsonl(Path(item['directory']) / 'steps.jsonl')
            if not rows:
                continue
            key = (item['provider'], item['mode'])
            if key not in represented:
                represented.append(key)
            color = colors.get(key, '#854aa3')
            times = [r['sim_time'] for r in rows]
            tcp = [r['grasp_after']['nearest_tcp_apple']['tcp_distance_m'] for r in rows]
            palm = [r['grasp_after']['nearest_palm_center_apple']['palm_volume_distance_m'] for r in rows]
            left.plot(times, tcp, color=color, linewidth=1.2)
            left.plot(times, palm, color=color, linewidth=1., linestyle='--', alpha=.8)
            right.plot(times, [r['measured_width'] for r in rows], color=color, linewidth=1.2)
            right.plot(times, [r['target_width'] for r in rows], color=color,
                       linewidth=1., linestyle='--', alpha=.8)
            intent_axis.plot(times, [r['grasp_after']['gripper_intent'] for r in rows],
                             color=color, linewidth=.8, linestyle=':', alpha=.28)
        left.set_ylabel(f'Seed {seed}\ndistance (m)')
        right.set_ylabel('gripper total width (m)')
        intent_axis.set_ylabel('intent (faint dotted)')
        intent_axis.set_ylim(-1.15, 1.15)
        intent_axis.set_yticks([-1, 0, 1])
        for axis in (left, right):
            axis.set_xlabel('simulation time (s)')
            axis.grid(alpha=.2)
        left.set_title('TCP to nearest fruit (solid); palm-volume gap (dashed)')
        right.set_title('Actual width (solid); predicted target width (dashed)')
    handles = [Line2D([0], [0], color=colors.get(key, '#854aa3'), label=f'{key[0]} / {key[1]}')
               for key in represented]
    figure.legend(handles=handles, loc='upper center', ncol=min(4, len(handles)), bbox_to_anchor=(.5, .975))
    figure.suptitle('Diagnostic boundary snapshots: nearest fruit identity can change', y=.997, fontsize=13)
    figure.text(.02, .006,
        'Palm gap uses the nearest palm-center fruit; it is not a fixed target. Before/after snapshots miss internal physics frames. '
        'Each curve ends at its recorded termination.', fontsize=8)
    figure.tight_layout(rect=(0, .03, 1, .94))
    output = ROOT / 'rollout_diagnostics.png'
    figure.savefig(output, dpi=150)
    plt.close(figure)
    return dict(status='saved', path=str(output), plotted_episode_count=len(items))


def main():
    completed, pending, read_errors = [], [], []
    for directory in sorted((ROOT / 'rollouts').glob('*')):
        if not directory.is_dir():
            continue
        if not (directory / 'summary.json').is_file():
            pending.append(str(directory))
            continue
        try:
            summary = json.loads((directory / 'summary.json').read_text())
            initial_path = directory / 'initial.json'
            initial = json.loads(initial_path.read_text()).get('info', {}) if initial_path.is_file() else {}
            completed.append(analyze_episode(summary, read_jsonl(directory / 'steps.jsonl'),
                read_jsonl(directory / 'chunks.jsonl'), directory, initial))
        except Exception as exc:
            read_errors.append(dict(directory=str(directory), error_type=type(exc).__name__, error=str(exc)))
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), completed_episode_count=len(completed),
        pending_directories=pending, read_errors=read_errors,
        interpretation_limits=[
            'Before/after samples can miss contacts inside action_repeat physics frames; info.grasp_assist_triggered is authoritative.',
            'Boundary sample counts are not independent control steps and before/after may observe the same state twice.',
            'Nearest fruit identities may change; no nearest fruit is treated as the intended target.',
            'Geometric eligibility with nonclosing intent is observed evidence, not proof that intent caused episode failure.',
            'Model tracking error is measured against its predicted target, not against demonstration ground truth.',
            'Palm-inside counts cover logged contact/nearest fruit and may undercount unlogged noncontact fruit.',
        ],
        near_tcp_thresholds_m=list(NEAR_THRESHOLDS_M),
        observer_mismatch_fields_total=sum(x['observer_consistency']['mismatch_fields_total'] for x in completed),
        summary_mismatch_count_total=sum(x['summary_consistency']['mismatch_count'] for x in completed),
        episodes=completed)
    try:
        result['plot'] = plot_rollouts(completed)
    except Exception as exc:
        result['plot'] = dict(status='error', error_type=type(exc).__name__, reason=str(exc))
    (ROOT / 'rollout_analysis.json').write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + '\n')
    csv_rows = [csv_row(item) for item in completed]
    columns = list(dict.fromkeys(k for row in csv_rows for k in row)) or ['provider', 'mode', 'seed']
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows(csv_rows)
    (ROOT / 'rollout_comparison.csv').write_text(stream.getvalue())
    print(json.dumps(dict(completed=len(completed), pending=len(pending), read_errors=read_errors,
        observer_mismatch_fields_total=result['observer_mismatch_fields_total'],
        summary_mismatch_count_total=result['summary_mismatch_count_total'], plot=result['plot'])))


if __name__ == '__main__':
    main()
