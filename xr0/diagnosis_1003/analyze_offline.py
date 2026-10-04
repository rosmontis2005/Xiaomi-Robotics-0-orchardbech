#!/usr/bin/env python3
"""Paired analysis of fixed offline windows; never loads models or starts GPU."""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import bootstrap  # noqa: F401
import hashlib
import json
import numpy as np

OUT = Path(__file__).resolve().parent
LABELS = ('mean_action', 'pretrained', 'step_10000')
METRICS = ('position_error_m', 'rotation_error_rad', 'width_error_m')
HORIZONS = {'all_01_30': slice(0, 30), 'first_01': slice(0, 1),
            'early_01_05': slice(0, 5), 'middle_06_15': slice(5, 15), 'late_16_30': slice(15, 30)}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def phase_groups(window):
    groups = ['all', f"anchor_{window['anchor_phase'].lower()}"]
    if window['label'] == 'reset': groups.append('reset')
    if window['label'] == 'before_grasp': groups.append('before_grasp')
    if window['anchor_phase'] in ('REACH', 'GRASP', 'PULL'): groups.append('front_reach_grasp_pull')
    if window['anchor_phase'] in ('TRANSPORT', 'DROP'): groups.append('transport_drop')
    return groups


def distribution(values):
    a = np.asarray(values, dtype=float)
    return dict(n=len(a), mean=float(a.mean()), median=float(np.median(a)),
                p90=float(np.quantile(a, .9)), min=float(a.min()), max=float(a.max()))


def paired_result(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    delta = a - b
    return dict(windows=len(delta), ten_k_better_windows=int(np.sum(delta < -1e-12)),
                tied_windows=int(np.sum(np.abs(delta) <= 1e-12)), ten_k_worse_windows=int(np.sum(delta > 1e-12)),
                ten_k_minus_reference=distribution(delta),
                ten_k_mean=float(a.mean()), reference_mean=float(b.mean()))


def closure_summary(windows, label):
    gt_has_switch = [w for w in windows if w[label]['gt_closure']['first_close_switch_horizon'] is not None]
    pred_missing = [w for w in gt_has_switch if w[label]['predicted_closure']['first_close_switch_horizon'] is None]
    offsets = [w[label]['close_switch_offset_steps'] for w in gt_has_switch if w[label]['close_switch_offset_steps'] is not None]
    return dict(windows=len(windows), gt_reference_switch_windows=len(gt_has_switch),
                predicted_missing_switch_windows=len(pred_missing), both_have_switch_windows=len(offsets),
                earlier_than_gt_windows=sum(x < 0 for x in offsets), equal_to_gt_windows=sum(x == 0 for x in offsets),
                later_than_gt_windows=sum(x > 0 for x in offsets),
                offset_steps=None if not offsets else distribution(offsets),
                mean_intent_mismatch_fraction=float(np.mean([w[label]['intent_mismatch_fraction'] for w in windows])))


def main():
    reports = {label: OUT / 'offline' / label / 'report.json' for label in LABELS}
    missing = [str(path) for path in reports.values() if not path.is_file()]
    if missing:
        print(json.dumps(dict(status='PENDING', missing_reports=missing, outputs_written=False), indent=2))
        return 2
    targets = [OUT / 'offline_comparison.json', OUT / 'offline_errors.png']
    if any(p.exists() for p in targets):
        raise FileExistsError('Refusing to overwrite existing offline comparison/figure')
    payloads = {k: json.loads(p.read_text()) for k, p in reports.items()}
    assert len({p['selection_sha256'] for p in payloads.values()}) == 1
    assert all(p['windows'] == 80 and len(p['window_metrics']) == 80 for p in payloads.values())
    by_label = {label: {(x['split'], x['episode_id'], x['frame']): x for x in report['window_metrics']}
                for label, report in payloads.items()}
    keys = sorted(by_label[LABELS[0]])
    assert all(set(rows) == set(keys) for rows in by_label.values())
    windows = []
    for key in keys:
        meta = by_label['mean_action'][key]
        record = {k: meta[k] for k in ('split', 'episode_id', 'seed', 'frame', 'label', 'anchor_phase')}
        errors = {}; target_phases = None; gt = None
        for label in LABELS:
            row = by_label[label][key]
            with np.load(row['npz']) as a:
                if gt is None:
                    gt = a['gt_action_physical'].copy(); target_phases = a['target_phase'].copy()
                else:
                    assert np.array_equal(gt, a['gt_action_physical']), f'GT differs: {key}'
                    assert np.array_equal(target_phases, a['target_phase']), f'Phases differ: {key}'
                errors[label] = {metric: a[metric].copy() for metric in METRICS}
                assert all(x.shape == (30,) and np.isfinite(x).all() for x in errors[label].values())
            record[label] = {k: row[k] for k in ('gt_closure', 'predicted_closure', 'close_switch_offset_steps', 'intent_mismatch_fraction')}
            record[label]['errors_by_horizon'] = {k: v.tolist() for k, v in errors[label].items()}
        record['target_phases'] = target_phases.tolist()
        record['_errors'] = errors
        windows.append(record)
    grouped = {}
    for w in windows:
        for split in ('all', w['split']):
            for phase in phase_groups(w):
                grouped.setdefault(f'{split}/{phase}', []).append(w)
    summary = {}
    for group, selected in grouped.items():
        data = dict(windows=len(selected), metrics={}, closure_proxy={label: closure_summary(selected, label) for label in LABELS})
        for metric in METRICS:
            data['metrics'][metric] = {}
            for horizon, sl in HORIZONS.items():
                vals = {label: [float(w['_errors'][label][metric][sl].mean()) for w in selected] for label in LABELS}
                data['metrics'][metric][horizon] = dict(
                    by_model={label: distribution(v) for label, v in vals.items()},
                    ten_k_vs_pretrained=paired_result(vals['step_10000'], vals['pretrained']),
                    ten_k_vs_mean_action=paired_result(vals['step_10000'], vals['mean_action']))
        summary[group] = data
    # Count cases where the 30-step average improves while action 1 gets worse.
    masking = {}
    for reference in ('pretrained', 'mean_action'):
        masking[reference] = {}
        for metric in METRICS:
            rows = []
            for w in windows:
                trained = w['_errors']['step_10000'][metric]; baseline = w['_errors'][reference][metric]
                if trained.mean() < baseline.mean() and trained[0] > baseline[0]:
                    rows.append({k: w[k] for k in ('split', 'episode_id', 'frame', 'label', 'anchor_phase')})
            masking[reference][metric] = dict(count=len(rows), windows=rows)
    curves = {}
    for split in ('train', 'val'):
        selected = [w for w in windows if w['split'] == split]
        curves[split] = {label: {metric: np.mean([w['_errors'][label][metric] for w in selected], axis=0).tolist()
                                for metric in METRICS} for label in LABELS}
    for w in windows: del w['_errors']
    report = dict(status='COMPLETE', paired_windows=80, episodes=8,
                  source_reports={k: dict(path=str(p), sha256=digest(p)) for k, p in reports.items()},
                  selection_sha256=payloads['mean_action']['selection_sha256'],
                  metric_definition='Separate physical error metrics; lower is better. Position Euclidean metres, SO(3) geodesic radians, total-width absolute metres. Per-window means compared on exactly matched windows.',
                  sampling_limit='Purposively selected overlapping windows in 4 train and 4 val episodes. No confidence interval, independent-sample inference or generalization-success claim.',
                  grouping_definition='Phase groups use observation/anchor phase. Target phase is retained per horizon in window records. RESET is separate; REACH/GRASP/PULL front windows are separate from TRANSPORT/DROP.',
                  intent_limit=payloads['mean_action']['width_intent_reference'],
                  grouped=summary, horizon_curves=curves,
                  full_chunk_improvement_but_first_action_worse=masking, windows=windows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    styles = {'mean_action': ('#737373', '--', 'Training mean'),
              'pretrained': ('#D8891A', '-', 'Pretrained'),
              'step_10000': ('#157A6E', '-', '10k')}
    specs = [('position_error_m', 100., 'Position error (cm)'),
             ('rotation_error_rad', 180. / np.pi, 'Orientation error (deg)'),
             ('width_error_m', 1000., 'Gripper width error (mm)')]
    fig, axes = plt.subplots(2, 3, figsize=(13.4, 7.4), sharex=True, constrained_layout=True)
    for ri, split in enumerate(('train', 'val')):
        for ci, (metric, scale, ylabel) in enumerate(specs):
            ax = axes[ri, ci]
            for label in LABELS:
                color, linestyle, name = styles[label]
                ax.plot(np.arange(1, 31), np.array(curves[split][label][metric]) * scale,
                        label=name, color=color, linestyle=linestyle, linewidth=2)
            ax.set_title(f'{split.capitalize()} — 4 episodes / 40 selected windows')
            ax.set_ylabel(ylabel); ax.set_xlim(1, 30); ax.grid(alpha=.22)
            ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
            if ri == 1: ax.set_xlabel('Predicted action horizon (1–30)')
            if ri == 0 and ci == 0: ax.legend(frameon=False, fontsize=9)
    fig.suptitle('Expert-observation action prediction: paired physical errors\nMeans over selected, overlapping windows; no closed-loop success claim', fontsize=13)
    # All inputs validated before writing either output; exclusive mode protects artifacts.
    with targets[1].open('xb') as f: fig.savefig(f, format='png', dpi=170)
    plt.close(fig)
    with targets[0].open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps(dict(status='COMPLETE', comparison=str(targets[0]), figure=str(targets[1]),
                          paired_windows=80), indent=2))
    for split in ('train', 'val'):
        for group in ('all', 'reset', 'front_reach_grasp_pull', 'transport_drop'):
            s = summary[f'{split}/{group}']
            for metric in METRICS:
                a = s['metrics'][metric]['all_01_30']
                print(json.dumps(dict(group=f'{split}/{group}', metric=metric,
                    means={k: v['mean'] for k, v in a['by_model'].items()},
                    ten_k_better_than_mean=a['ten_k_vs_mean_action']['ten_k_better_windows'],
                    ten_k_better_than_pretrained=a['ten_k_vs_pretrained']['ten_k_better_windows'], windows=s['windows'])))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
