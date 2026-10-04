"""CPU-only complete filtered-cohort audit for a reach-conditioned repair plan.

Reads every annotation once; no image decode, simulation, training or edits to
existing artifacts. Uniform exposure expectations describe sampling, not actual
identified batches or independent demonstrations.
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import bootstrap
from bisect import bisect_right
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import statistics
import time

ORCHARD = bootstrap.ORCHARD
XR0 = bootstrap.XR0
DATA = ORCHARD/'data/orchard_v1_2650/filtered'
MANIFEST = DATA/'manifest.jsonl'
HORIZON = 30
DRAWS = 10000


def digest(data):
    return hashlib.sha256(data).hexdigest()


def distribution(values):
    v = sorted(values)
    if not v:
        return {'n':0}
    def quantile(q):
        x=(len(v)-1)*q
        lo=int(x); hi=min(lo+1,len(v)-1)
        return v[lo]+(v[hi]-v[lo])*(x-lo)
    return dict(n=len(v), total=sum(v), mean=statistics.mean(v), minimum=v[0],
                p05=quantile(.05), p25=quantile(.25), median=quantile(.5),
                p75=quantile(.75), p95=quantile(.95), p99=quantile(.99), maximum=v[-1])


def pixels_summary(rows, event):
    out={}
    for view in ['static','wrist','best_view','sum_views']:
        vals=[r['visibility'][event][view] for r in rows]
        valid=[v for v in vals if v is not None]
        bins=Counter()
        for n in valid:
            bins['0' if n==0 else '1_2' if n<=2 else '3_9' if n<10 else '10_49' if n<50 else '50_99' if n<100 else '100_499' if n<500 else '500_plus']+=1
        out[view]=dict(distribution=distribution(valid), missing=len(vals)-len(valid), bins=dict(bins),
                       at_most_2_count=sum(v<=2 for v in valid),
                       positive_but_at_most_2_count=sum(0<v<=2 for v in valid),
                       at_most_10_count=sum(v<=10 for v in valid),
                       at_most_50_count=sum(v<=50 for v in valid))
    out['interpretation']='Visibility truth from collector nearest-hit shape-index camera; initial and GRASP entry only, before model image preprocessing. best_view=max(static,wrist), sum_views sums pixels across two views. Not a perceptual difficulty label.'
    return out


def any_sample_probability(total, count, draws):
    # Exact hypergeometric P(at least one group member in a random permutation prefix).
    if count==0 or draws==0:return 0.
    if draws>=total or count>total-draws:return 1.
    return -math.expm1(sum(math.log1p(-draws/(total-j)) for j in range(count)))


def split_summary(rows, split):
    total=sum(r['full_windows'] for r in rows)
    phase=Counter()
    recorded=Counter()
    target=Counter()
    for r in rows:
        phase.update(r['anchor_phase_windows'])
        recorded.update(r['recorded_phase_frames'])
        target.update(r['target_phase_label_count_over_all_windows'])
    groups=dict(reset_frame0=len(rows), first5_anchor_frames_0_to_4=sum(r['first5_windows'] for r in rows),
                **{f'anchor_phase_{p}':c for p,c in phase.items()})
    exposure={k:dict(windows=c, fraction=c/total, expected_draws_in_10000=DRAWS*c/total)
              for k,c in groups.items()}
    distinct={}
    for label, key in [('reset','reset_windows'),('first5','first5_windows'),('REACH','reach_windows'),('GRASP','grasp_windows')]:
        distinct[label]=sum(any_sample_probability(total,r[key],min(DRAWS,total)) for r in rows)
    visibility={stage:pixels_summary(rows,stage) for stage in ['initial','grasp_entry']}
    best=[r['visibility']['initial']['best_view'] for r in rows]
    weak_ids=[dict(seed=r['seed'],episode_id=r['episode_id'],initial=r['visibility']['initial']) for r in rows
              if r['visibility']['initial']['best_view'] is not None and r['visibility']['initial']['best_view']<=2]
    replay={k:distribution([r['replay'][k] for r in rows if r['replay'].get(k) is not None])
            for k in ['control_steps','IK_failed_steps','clipped_steps','dwell_timeout_count','forced_advancement_count']}
    return dict(episodes=len(rows), full_windows=total, recorded_frames=sum(r['num_frames'] for r in rows),
                frame_count_distribution=distribution([r['num_frames'] for r in rows]),
                expert_duration_s_distribution=distribution([r['expert_duration_s'] for r in rows]),
                full_window_count_distribution=distribution([r['full_windows'] for r in rows]),
                anchor_phase_windows=dict(phase), anchor_phase_fraction={k:v/total for k,v in phase.items()},
                recorded_phase_frames=dict(recorded), target_phase_label_count_over_all_windows=dict(target),
                recorded_REACH_frames_distribution=distribution([r['recorded_phase_frames'].get('REACH',0) for r in rows]),
                recorded_GRASP_frames_distribution=distribution([r['recorded_phase_frames'].get('GRASP',0) for r in rows]),
                expert_REACH_trace_duration_s_distribution=distribution([r['phase_trace_duration_s'].get('REACH',0.) for r in rows]),
                expert_GRASP_trace_duration_s_distribution=distribution([r['phase_trace_duration_s'].get('GRASP',0.) for r in rows]),
                uniform_10000_exposure=dict(groups=exposure,expected_distinct_episodes_with_at_least_one_group_anchor=distinct,
                    sampling='Uniform random permutation of all complete windows; 10000 draws without replacement. Expected count formula is 10000 * group_windows / all_windows.',
                    actual_training_use='train split only; batch 1, accumulation 1, single GPU assumption' if split=='train' else 'No validation dataloader in actual run; actual optimizer exposure is ZERO. This 10000-draw column is hypothetical comparison only.'),
                initial_and_grasp_visibility=visibility, initial_best_view_at_most_2_episodes=weak_ids,
                strict_replay=dict(all_manifest_oracle_success=all(r['oracle_strict_success'] for r in rows),
                    all_grasped=all(r['replay'].get('grasped') for r in rows),
                    all_detached=all(r['replay'].get('detached') for r in rows),
                    termination_counts=dict(Counter(r['replay'].get('termination_reason') for r in rows)),
                    metrics=replay,
                    has_ik_failed_episodes=sum(r['replay'].get('IK_failed_steps',0)>0 for r in rows),
                    has_dwell_timeout_episodes=sum(r['replay'].get('dwell_timeout_count',0)>0 for r in rows)))


def main():
    output=HERE/'dataset_phase_visibility_audit.json'
    if output.exists():raise FileExistsError(f'Refuse to overwrite {output}')
    started=time.monotonic()
    raw_manifest=MANIFEST.read_bytes()
    manifest=[json.loads(l) for l in raw_manifest.decode().splitlines()]
    sources=[XR0/'mibot/data/datasets/orchardbench_dataset.py',XR0/'mibot/data/datamodule/orchardbench_datamodule.py',
             ORCHARD/'log/v1_2650_collection_replay_filter/sources/frozen_gate1_oracle.py']
    before={str(p):digest(p.read_bytes()) for p in sources}
    episodes=[]
    problems=[]
    hashes_ok=0
    for i,m in enumerate(manifest,1):
        path=DATA/m['annotation']
        payload=path.read_bytes()
        t=json.loads(payload)
        n=t['num_frames']; length=n-HORIZON+1
        assert length>0
        timestamps=t['orchardbench']['timestamps']
        trace=t['orchardbench']['fixed_base_expert']['state_trace']
        times=[q['frame']/60 for q in trace]
        assert times==sorted(times)
        phases=[trace[bisect_right(times,ts+1e-8)-1]['state'] for ts in timestamps]
        assert len(phases)==n
        recorded=Counter(phases)
        anchors=Counter(phases[:length])
        targets=Counter()
        # Each action target corresponds to index+1 measured state; terminal repeats.
        # Added as context only; requested main distribution is anchor phase.
        for f in range(length):
            targets.update(phases[min(k+1,n-1)] for k in range(f,f+HORIZON))
        durations=defaultdict(float)
        for k,event in enumerate(trace):
            end=times[k+1] if k+1<len(times) else timestamps[-1]
            durations[event['state']]+=max(0.,end-times[k])
        visibility={}
        for stage in ['initial','grasp_entry']:
            v=m.get('target_visibility',{}).get(stage,{}) or {}
            a=v.get('static_visible_pixels'); b=v.get('wrist_visible_pixels')
            visibility[stage]=dict(static=a,wrist=b,best_view=max(a,b) if a is not None and b is not None else None,
                                   sum_views=a+b if a is not None and b is not None else None)
        correct=digest(payload)==m.get('annotation_sha256')
        hashes_ok+=int(correct)
        if not correct:problems.append(dict(episode_id=m['episode_id'],problem='annotation_sha_mismatch'))
        if t['record_fps']!=30 or any(abs(ts-j/30)>1e-7 for j,ts in enumerate(timestamps)):
            problems.append(dict(episode_id=m['episode_id'],problem='unexpected_timestamps'))
        episodes.append(dict(seed=t['seed'],episode_id=t['episode_id'],split=m['split'],annotation=str(path),
            manifest_line=i,annotation_sha256=digest(payload),is_symlink=path.is_symlink(),
            num_frames=n,full_windows=length,expert_duration_s=timestamps[-1],
            recorded_phase_frames=dict(recorded),anchor_phase_windows=dict(anchors),
            target_phase_label_count_over_all_windows=dict(targets),phase_trace_duration_s=dict(durations),
            reset_windows=1,first5_windows=min(5,length),reach_windows=anchors.get('REACH',0),grasp_windows=anchors.get('GRASP',0),
            visibility=visibility,oracle_strict_success=m['oracle_strict_success'],replay=m['replay']))
        if i%400==0:print(json.dumps(dict(processed=i,total=len(manifest),elapsed_seconds=round(time.monotonic()-started,1))),flush=True)
    splits={split:split_summary([r for r in episodes if r['split']==split],split) for split in ['train','val']}
    assert splits['train']['episodes']==2144 and splits['val']['episodes']==230
    assert splits['train']['full_windows']==388172 and splits['val']['full_windows']==41892
    after={str(p):digest(p.read_bytes()) for p in sources}
    assert before==after
    result=dict(schema='orchard_full_filtered_phase_visibility_audit_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        runtime_seconds=time.monotonic()-started,status='PASS' if not problems else 'CHECK_FAILED',
        scope='All 2144 train + 230 val filtered manifest records and original JSON annotations; CPU only; no new policy, image decoding or replay.',
        manifest_path=str(MANIFEST),manifest_sha256=digest(raw_manifest),source_hashes=before,sources_unchanged=before==after,
        annotations_verified_by_sha=hashes_ok,problems=problems,horizon=HORIZON,uniform_draws=DRAWS,
        definitions=dict(anchor_phase='last expert state_trace event with event.frame/60 <= anchor timestamp + 1e-8; never compare physics frame index to video frame index',
            reset='anchor frame 0, subset of REACH',first5='anchor frame indices 0,1,2,3,4; includes reset, not disjoint',
            duration='REACH/GRASP physical trace interval duration; recorded frame counts independently mapped through timestamps',
            visibility='collector target apple visible pixels at reset and GRASP entry only, not every anchor; no new visibility filtering',
            reach_policy='Reach-conditioned remains the established deployment contract. Original expert observations/next-state labels are retained; replay acceptance is not student-observation relabeling.'),
        replay_contract=dict(mode='reach-conditioned',position_tolerance_m=.01,rotation_tolerance_rad=.08,maximum_dwell_control_steps=30,
            max_control_steps=900,sim_hz=60,action_repeat=2,grasp_mode='benchmark_assist',
            advance='post-step position AND rotation reached OR max dwell; no width tolerance, no passed waypoint; at most one target per control step',
            success='at least one detached apple physically inside bucket; filter retains strict evaluator success',
            other_filters='Expert accepted and integrity-valid; no extra duration/IK/clipping/dwell/smoothness filter; train/val split assigned before replay'),
        split_summaries=splits,episodes=episodes)
    with output.open('x') as f:json.dump(result,f,indent=2,ensure_ascii=False);f.write('\n')
    print(json.dumps(dict(output=str(output),status=result['status'],runtime_seconds=result['runtime_seconds'],
        split_overview={k:{'episodes':v['episodes'],'windows':v['full_windows'],'phases':v['anchor_phase_windows'],
                           'exposure':v['uniform_10000_exposure']['groups'],
                           'initial_best_view_bins':v['initial_and_grasp_visibility']['initial']['best_view']['bins']}
                        for k,v in splits.items()}),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
