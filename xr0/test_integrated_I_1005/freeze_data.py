"""Freeze D* and independent D_A/D_B before any S/J training."""
import bootstrap
import json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from recovery_common import *
from collect_live import summary

I=PROTOCOL['integrated']

def draw(pool,quota,rng):
    sources=defaultdict(list)
    for row in pool:sources[row['source_episode_id']].append(row)
    names=sorted(sources);assert names and quota>=len(names)
    counts=np.full(len(names),quota//len(names));counts[rng.permutation(len(names))[:quota%len(names)]]+=1
    rows=[]
    for source,count in zip(names,counts):
        choices=sources[source]
        required=[r for r in choices if r['frame']==0]
        assert int(count)>=len(required)
        # Every actual takeover receives frame0 supervision in BOTH schedules.
        rows.extend(dict(r) for r in required)
        rows.extend(dict(choices[int(i)]) for i in rng.choice(len(choices),size=int(count)-len(required),replace=True))
    return rows

def freeze():
    if (ROOT/'frozen.json').exists():
        f=json.loads((ROOT/'frozen.json').read_text())
        for path,digest in f['files'].items():assert sha(path)==digest
        print('Already frozen');return
    status=summary();assert status['ready_to_freeze'],status
    protection_check();manifest=[];pools=defaultdict(list)
    def add(t,path,source,episode_id,source_episode_id,frames,group):
        digest=sha(path)
        for frame in frames:
            pools[group].append(dict(source=source,episode_id=episode_id,source_episode_id=source_episode_id,frame=frame,group=group,annotation=str(path),annotation_sha256=digest,window_id=f'{source}/{episode_id}/frame{frame:04d}'))
    original=json.loads((AB/'selection.json').read_text())['episodes_train']
    for ep in original:
        path=Path(ep['annotation']);assert sha(path)==ep['annotation_sha256'];t=json.loads(path.read_text())
        groups=defaultdict(list)
        mapping=dict(reset='early',first1_4='early',REACH='early',GRASP='grasp_pull',PULL='grasp_pull',TRANSPORT='transport',DROP='drop_done')
        for old_group,frames in ep['group_frames'].items():groups[mapping[old_group]].extend(frames)
        assert sorted(f for frames in groups.values() for f in frames)==list(range(t['num_frames']-29))
        manifest.append(dict(source='original',annotation=str(path),annotation_sha256=sha(path),episode_id=ep['episode_id'],source_episode_id=ep['episode_id'],legal_anchors=list(range(t['num_frames']-29)),groups=dict(groups)))
        for group,frames in groups.items():add(t,path,'original',ep['episode_id'],ep['episode_id'],frames,'original_'+group)
    old=[json.loads(l) for l in (XR0/'test_recovery_R_1004/recovery_manifest.jsonl').read_text().splitlines()]
    for d in old:
        path=Path(d['annotation']);assert d['accepted'] and sha(path)==d['annotation_sha256'];t=json.loads(path.read_text());assert t['recovery']['oracle_replay_result']['status']=='PASS'
        manifest.append(dict(d,source='existing_R'));add(t,path,'existing_R',d['candidate_id'],d['source_episode_id'],d['legal_anchors'],'existing_'+d['category'])
    for decision in sorted((ROOT/'recoveries').glob('*/decision.json')):
        d=json.loads(decision.read_text())
        if not d['accepted']:continue
        path=Path(d['annotation']);assert sha(path)==d['annotation_sha256'] and d['same_live_env'] and d['student_checkpoint_sha256']==I['mining_sha256']
        t=json.loads(path.read_text());r=t['recovery'];assert r['same_live_env'] and not r['prefix_regeneration_required'] and r['initialization_audit']['physics_unchanged']
        assert d['teacher_result']['strict']['strict_success'] and d['teacher_result']['reasonable_release']
        prefix_path=Path(r['student_prefix']);prefix=json.loads(prefix_path.read_text())
        assert len(prefix['timestamps'])==r['student_control_step']+1 and prefix['timestamps'][-1]==t['orchardbench']['timestamps'][0]
        for key in t['proprios']:assert np.array_equal(prefix['proprios'][key][-1],t['proprios'][key][0]),(d['candidate_id'],key)
        prefix_meta=json.loads((prefix_path.parent/'complete.json').read_text())
        assert prefix_meta['prefix_end_physics_digest']==r['initialization_audit']['physical_digest_before']
        assert r['source_episode_id'] not in PROTOCOL['protected_episode_ids']
        for k in t['actions']:assert np.array_equal(t['actions'][k],t['proprios'][k][1:]+t['proprios'][k][-1:])
        assert np.allclose(np.diff(t['orchardbench']['timestamps']),1/30,atol=1e-9,rtol=0)
        phases=t['orchardbench']['phase_by_frame'];drop=next(i for i,p in enumerate(phases) if p=='DROP');groups=defaultdict(list)
        for frame in range(t['num_frames']-29):
            group='live_release' if drop-29<=frame<=drop+5 else 'live_onset' if frame<30 else 'live_transport' if frame<drop else 'live_settle'
            groups[group].append(frame)
        manifest.append(dict(d,source='live_late',legal_anchors=list(range(t['num_frames']-29)),groups=dict(groups)))
        for group,frames in groups.items():add(t,path,'live_late',d['candidate_id'],d['source_episode_id'],frames,group)
    # Exact 60/15/25 and 25/35/25/15 within original; simple fixed late exposure.
    quotas=dict(original_early=1200,original_grasp_pull=1680,original_transport=1200,original_drop_done=720,existing_R1=600,existing_R2=300,existing_R3=300,live_onset=400,live_transport=600,live_release=700,live_settle=300)
    # Small trajectories can have onset entirely overlapping release; count available
    # groups explicitly, never synthesize windows or alter quotas after training.
    assert all(pools[k] for k in quotas),{k:len(v) for k,v in pools.items()}
    with (ROOT/'dataset_manifest.jsonl').open('x') as f:
        for row in manifest:f.write(json.dumps(row)+'\n')
    schedules={}
    for name,seed in I['schedule_seeds'].items():
        rng=np.random.default_rng(seed);rows=[]
        for group,quota in quotas.items():rows.extend(draw(pools[group],quota,rng))
        rows=[rows[int(i)] for i in rng.permutation(len(rows))]
        for step,row in enumerate(rows,1):row['step']=step
        assert len(rows)==8000 and Counter(r['source'] for r in rows)==dict(original=4800,existing_R=1200,live_late=2000)
        with (ROOT/f'schedule_{name}.jsonl').open('x') as f:
            for row in rows:f.write(json.dumps(row)+'\n')
        schedules[name]=dict(seed=seed,updates=8000,source_counts=dict(Counter(r['source'] for r in rows)),group_counts=dict(Counter(r['group'] for r in rows)),unique_windows=len({r['window_id'] for r in rows}))
    statistics=dict(collection=status,trajectories_by_source=dict(Counter(r['source'] for r in manifest)),available_windows={k:len(v) for k,v in pools.items()},unique_sources={k:len({r['source_episode_id'] for r in v}) for k,v in pools.items()},quotas=quotas,schedules=schedules,original_source='full legal task windows of frozen 128 training episodes; no old B quotas',existing_R='accepted real student-state R1/R2/R3 onset windows; historical successful suffixes remain within trajectories',live_internal_quota_reason='onset20%, transport30%, release35%, settle15% prevents terminal stationary frames dominating late supervision',policy_fields=['messages','action','action_mask','state'],no_privileged_input=True,every_takeover_frame0_in_both_schedules=True)
    write(ROOT/'dataset_statistics.json',statistics)
    files=[ROOT/'dataset_manifest.jsonl',ROOT/'dataset_statistics.json',ROOT/'schedule_D_A.jsonl',ROOT/'schedule_D_B.jsonl',ROOT/'protocol.json']
    assets={}
    for row in manifest:
        path=Path(row['annotation']);assets[str(path)]=sha(path);t=json.loads(path.read_text())
        if row['source']=='live_late':
            prefix=Path(t['recovery']['student_prefix']);pm=json.loads(prefix.read_text())
            for asset in [prefix,prefix.parent/'steps.json',prefix.parent/'chunks.json',path.parent/'candidate.json',path.parent/'teacher_result.json',*map(Path,pm['observations'].values())]:assets[str(asset)]=sha(asset)
        for v in t['observations'].values():
            video=Path(v[0]['path']);video=video if video.is_absolute() else ORCHARD/'data/orchard_v1_2650/filtered'/video;assets[str(video)]=sha(video)
    for name in ['collect_live.py','collection_v1.py','collection_v2.py','collection_v3.py','collection_v4.py','inputs.py','freeze_data.py','train_integrated.py','evaluate.py']:
        assets[str(ROOT/name)]=sha(ROOT/name)
    for path in [XR0/'test_recovery_R_1004'/name for name in ['recovery_common.py','recovery_teacher.py','collect_recovery.py','checkpoint_io.py']]+[XR0/'test_grasp_1004/evaluation'/name for name in ['strict_metrics.py','reach_loop.py']]:assets[str(path)]=sha(path)
    write(ROOT/'frozen.json',dict(status='FROZEN',time=time.time(),files={str(p):sha(p) for p in files},assets=assets,stats_sha256=sha(STATS),mining_checkpoint_sha256=sha(I['mining_checkpoint']),base_checkpoint_sha256=sha(BASE),B_checkpoint_sha256=sha(ENDPOINT),training_started=False))
    print(json.dumps(statistics))
if __name__=='__main__':freeze()
