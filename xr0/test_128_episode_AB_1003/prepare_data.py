#!/usr/bin/env python3
"""Freeze 128-episode A/B cohorts, exact schedules and comparable CPU eval inputs.

All writes stay in this experiment. No simulator, model construction or training.
"""
import sys
sys.dont_write_bytecode=True
import bootstrap
import argparse
from collections import Counter
import copy
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from decord import VideoReader,cpu
import torch
from treesim.orchard_action import CONTRACT,HORIZON,EPS,encode_window,action_mask,prepare_rgb
from mibot.data.datasets.orchardbench_dataset import OrchardBenchDataset,load_stats
from mibot.data.datamodule.orchardbench_datamodule import OrchardBenchDataModule
from ab_common import phase_metadata,width_intent,cpu_deployment_parity

ROOT=Path(__file__).resolve().parent
XR0=ROOT.parent
ORCHARD=Path('/home/rosmontis/Projects/orchardbench')
DATA=ORCHARD/'data/orchard_v1_2650/filtered'
STATS=DATA/'action_stats.json'
M0=XR0/'test_small_set_fit_1003'
AUDIT=XR0/'diagnosis_1003/reach_plan/dataset_phase_visibility_audit.json'
SEED=20261003
UPDATES=8000
QUOTAS={'reset':1600,'first1_4':800,'REACH':1200,'GRASP':2000,'PULL':1200,'TRANSPORT':800,'DROP':400}
EXCLUDED_VISUAL_TRAIN={'episode_000598','episode_000656'}
EXCLUDED_DIAGNOSTIC_VAL={'episode_000010','episode_000020','episode_000030','episode_000040'}

spec=importlib.util.spec_from_file_location('m0_readonly_selector_helpers',M0/'select_dataset.py')
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load(path):return json.loads(Path(path).read_text())

def simple(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,Path):return str(x)
    raise TypeError(type(x).__name__)

def save(path,obj):
    assert Path(path).resolve().is_relative_to(ROOT)
    with Path(path).open('x') as f:json.dump(obj,f,indent=2,default=simple,allow_nan=False);f.write('\n')


def group_for(traj,frame):
    if frame==0:return 'reset'
    if 1<=frame<=4:return 'first1_4'
    return helper.phase(traj,frame)


def qualified(m):
    return m['accepted'] and m['oracle_strict_success'] and m['replay']['IK_failed_steps']==0 and m['replay']['dwell_timeout_count']==0 and max(m['target_visibility']['initial']['static_visible_pixels'],m['target_visibility']['initial']['wrist_visible_pixels'])>=128


def choose_diverse(candidates,n,initial,rule):
    lo=np.asarray(rule['robust_low_p05']);hi=np.asarray(rule['robust_high_p95'])
    def scale(m):return np.clip((np.asarray(m['_geometry']['selection_features'])-(lo+hi)/2)/np.maximum(hi-lo,1e-9),-1,1)
    remaining=list(candidates);result=[]
    anchors=np.asarray([scale(x) for x in initial]) if initial else np.empty((0,5))
    matrix=np.asarray([scale(x) for x in remaining])
    if len(anchors):minimum=np.min(np.sum((matrix[:,None,:]-anchors[None,:,:])**2,axis=2),axis=1)
    else:
        center=np.median(matrix,axis=0)
        first=min(range(len(remaining)),key=lambda i:(float(np.sum((matrix[i]-center)**2)),remaining[i]['seed']))
        minimum=np.sum((matrix-matrix[first])**2,axis=1)
        result.append(remaining[first]);minimum[first]=-np.inf
    while len(result)<n:
        idx=min(range(len(remaining)),key=lambda i:(-float(minimum[i]),remaining[i]['seed']))
        if not np.isfinite(minimum[idx]):raise ValueError('Not enough distinct qualified candidates')
        result.append(remaining[idx]);minimum=np.minimum(minimum,np.sum((matrix-matrix[idx])**2,axis=1));minimum[idx]=-np.inf
    return result


def episode_record(m,role):
    path=DATA/m['annotation'];t=load(path)
    assert sha(path)==m['annotation_sha256'] and t['seed']==m['seed'] and t['split']==m['split']
    assert t['record_fps']==30 and np.allclose(t['orchardbench']['timestamps'],np.arange(t['num_frames'])/30,atol=1e-7,rtol=0)
    groups={g:[] for g in QUOTAS}
    for f in range(t['num_frames']-29):
        g=group_for(t,f)
        if g not in groups:raise ValueError(f'Unspecified complete-window anchor phase {g}')
        groups[g].append(f)
    windows,events,closure,omissions=helper.windows(t)
    for w in windows:
        w.update(window_id=f"{m['split']}/{m['episode_id']}/frame{w['frame']:04d}",split=m['split'],episode_id=m['episode_id'],
                 seed=m['seed'],annotation=str(path),annotation_sha256=sha(path))
    ep=dict(split=m['split'],episode_id=m['episode_id'],seed=m['seed'],role=role,annotation=str(path),annotation_sha256=sha(path),
        num_frames=t['num_frames'],full_windows=t['num_frames']-29,geometry=m['_geometry'],visibility=m['target_visibility'],
        replay=m['replay'],oracle_strict_success=True,group_frames=groups,group_counts={g:len(fs) for g,fs in groups.items()},
        windows=windows,events=events,closure_reference=closure,window_landmark_omissions=omissions)
    return ep,t


def coverage(ep,t):
    close=ep['closure_reference']['first_close_action_index'];frames=[0,close]
    ims=[];checks={}
    for frame in frames:
        for view in ['ego','wrist_left']:
            path=helper.video_path(t,view);reader=VideoReader(str(path),ctx=cpu(0),num_threads=1)
            assert len(reader)==t['num_frames'] and abs(reader.get_avg_fps()-30)<1e-6
            raw=reader[frame].asnumpy();assert raw.shape==(144,192,3) and raw.dtype==np.uint8
            prepared=prepare_rgb(Image.fromarray(raw));arr=np.asarray(prepared)
            assert prepared.size==(192,128) and arr.std()>1
            ims.append(prepared)
            checks[f'{view}/frame{frame}']=dict(path=str(path),decoded_frame=frame,container_frames=len(reader),fps=reader.get_avg_fps(),
                raw_shape=list(raw.shape),prepared_shape=list(arr.shape),prepared_std=float(arr.std()),prepared_sha256=hashlib.sha256(arr.tobytes()).hexdigest())
    ep['processed_coverage']=checks
    return ims


def save_sheets(rows,role):
    folder=ROOT/'contact_sheets';folder.mkdir(exist_ok=True)
    paths=[]
    for page,start in enumerate(range(0,len(rows),8),1):
        part=rows[start:start+8]
        canvas=Image.new('RGB',(792,len(part)*160+32),(245,245,245));draw=ImageDraw.Draw(canvas)
        draw.text((10,7),'reset static | reset wrist | close-ref static | close-ref wrist; actual prepare_rgb',fill='black')
        for i,(ep,images) in enumerate(part):
            y=32+i*160
            draw.text((10,y),f"{ep['episode_id']} seed{ep['seed']} pixels={ep['geometry']['initial_best_view_pixels']} role={role}",fill='black')
            for j,im in enumerate(images):canvas.paste(im,(12+192*j,y+22))
        path=folder/f'{role}_{page:02d}.png';assert not path.exists();canvas.save(path);paths.append(str(path))
    return paths


def make_schedules(selection):
    directory=ROOT/'schedules';directory.mkdir(exist_ok=False)
    eps=selection['episodes_train'];all_windows=[];group_maps={g:[] for g in QUOTAS}
    for ep in eps:
        for g,frames in ep['group_frames'].items():
            if frames:group_maps[g].append((ep,frames))
            for frame in frames:all_windows.append((ep,frame,g))
    assert len(all_windows)>UPDATES
    rng=np.random.default_rng(SEED)
    indices=rng.permutation(len(all_windows))[:UPDATES]
    A=[all_windows[int(i)] for i in indices]
    rng=np.random.default_rng(SEED)
    B=[];availability={}
    for group,quota in QUOTAS.items():
        choices=group_maps[group];count=len(choices)
        if not count:raise ValueError(f'Required group {group} has zero available episodes/windows')
        base,extra=divmod(quota,count);per_episode=np.full(count,base,dtype=int)
        per_episode[rng.permutation(count)[:extra]]+=1
        for (ep,frames),draws in zip(choices,per_episode):
            B.extend((ep,int(frame),group) for frame in rng.choice(frames,size=int(draws),replace=True))
        availability[group]=dict(quota=quota,available_episodes=count,available_full_windows=sum(len(fs) for _,fs in choices),
            unavailable_episode_ids=[e['episode_id'] for e in eps if not e['group_frames'][group]],
            per_available_episode_draws_min=int(per_episode.min()),per_available_episode_draws_max=int(per_episode.max()))
    B=[B[int(i)] for i in rng.permutation(len(B))]
    arms={}
    for name,draws in [('A',A),('B',B)]:
        path=directory/f'{name}.jsonl';records=[]
        for step,(ep,frame,group) in enumerate(draws,1):
            records.append(dict(step=step,episode_id=ep['episode_id'],seed=ep['seed'],split='train',frame=frame,group=group,
                annotation=ep['annotation'],annotation_sha256=ep['annotation_sha256'],window_id=f"train/{ep['episode_id']}/frame{frame:04d}"))
        with path.open('x') as f:
            for r in records:f.write(json.dumps(r)+'\n')
        counts=Counter(r['group'] for r in records)
        if name=='B':assert dict(counts)==QUOTAS
        assert len(records)==UPDATES and {r['episode_id'] for r in records}<=set(selection['train_episode_ids'])
        arms[name]=dict(path=str(path),sha256=sha(path),steps=len(records),unique_windows=len({r['window_id'] for r in records}),
            group_counts=dict(counts),episode_draw_counts=dict(Counter(r['episode_id'] for r in records)),
            group_episode_draw_counts={g:dict(Counter(r['episode_id'] for r in records if r['group']==g)) for g in QUOTAS})
    meta=dict(schema='orchard_128_ab_schedules_v1',selection_sha256=sha(ROOT/'selection.json'),schedule_seed=SEED,total_legal_windows=len(all_windows),
        group_definition='Mutually exclusive: frame0 reset; frame1..4 first1_4; remaining complete anchors assigned by expert phase at anchor timestamp. No phase fabricated or quota substituted.',
        A='Window-uniform random permutation of every legal full window, first 8000 without replacement; same shuffle-epoch interpretation as production Dataset.',
        B='Exact phase quotas; within group every eligible episode receives floor(quota/n) or ceil(quota/n), randomized remainder; within episode its legal group frames uniformly sampled with replacement; final group sequence shuffled.',
        comparison_limit='B jointly changes group weights and episode weights relative to A. Duplicate exposure is recorded; not solely a phase-weight intervention.',
        quota_requested=QUOTAS,availability=availability,arms=arms)
    save(ROOT/'schedule_meta.json',meta)
    return meta


def prepare_cohort():
    selection_path=ROOT/'selection.json'
    if selection_path.exists():raise FileExistsError('Cohort already frozen; refusing overwrite')
    m0=load(M0/'selection.json');assert m0['selection_revision']==2
    _,mean,std=load_stats(STATS);assert sha(STATS)==m0['stats_sha256']
    rows=[json.loads(l) for l in (DATA/'manifest.jsonl').read_text().splitlines()]
    byid={r['episode_id']:r for r in rows}
    for m in rows:m['_geometry']=helper.candidate_geometry(m)
    oldtrain=[byid[e] for e in m0['train_episode_ids']]
    assert len(oldtrain)==8 and all(qualified(m) for m in oldtrain)
    traincand=[m for m in rows if m['split']=='train' and qualified(m) and m['episode_id'] not in set(m0['train_episode_ids'])|EXCLUDED_VISUAL_TRAIN]
    extra=choose_diverse(traincand,120,oldtrain,m0['selection_rule']['geometric_spread']['train'])
    excluded_val=set(m0['val_episode_ids'])|EXCLUDED_DIAGNOSTIC_VAL
    vc=[m for m in rows if m['split']=='val' and qualified(m) and m['episode_id'] not in excluded_val]
    newval=choose_diverse(vc,8,[],m0['selection_rule']['geometric_spread']['val'])
    newids={m['episode_id'] for m in newval}
    holdcand=[m for m in vc if m['episode_id'] not in newids]
    heldout=choose_diverse(holdcand,12,newval,m0['selection_rule']['geometric_spread']['val'])
    sets={};sheet_paths={};decoded=[]
    for role,chosen in [('train',oldtrain+extra),('new_val',newval),('heldout',heldout)]:
        records=[];sheets=[]
        for i,m in enumerate(chosen,1):
            ep,t=episode_record(m,role);images=coverage(ep,t);records.append(ep);sheets.append((ep,images))
            if i%16==0 or i==len(chosen):print(json.dumps(dict(role=role,prepared=i,total=len(chosen))),flush=True)
        sets[role]=records;sheet_paths[role]=save_sheets(sheets,role)
    eval_sets={f'old_{split}':[copy.deepcopy(w) for w in m0['windows'] if w['split']==split] for split in ['train','val']}
    eval_sets['new_val']=[copy.deepcopy(w) for e in sets['new_val'] for w in e['windows']]
    assert all(len(v)==80 for v in eval_sets.values())
    dev=copy.deepcopy(m0['closed_loop_scenes'])
    scene_keys=['split','episode_id','seed','annotation','annotation_sha256','visibility','replay','geometry','oracle_strict_success']
    dev.extend({k:sets['new_val'][i][k] for k in scene_keys} for i in [0,2,4,6])
    hold=[{k:e[k] for k in scene_keys} for e in sets['heldout']]
    allseeds=[e['seed'] for role in sets.values() for e in role]
    assert len(allseeds)==len(set(allseeds))==148
    assert {e['episode_id'] for e in sets['train']} >= set(m0['train_episode_ids'])
    group_windows={g:sum(e['group_counts'][g] for e in sets['train']) for g in QUOTAS}
    selection=dict(schema='orchard_128_ab_selection_v1',created_utc=datetime.now(timezone.utc).isoformat(),contract=CONTRACT,horizon=30,
        dataset_root=str(DATA),stats_path=str(STATS),stats_sha256=sha(STATS),source_manifest=str(DATA/'manifest.jsonl'),source_manifest_sha256=sha(DATA/'manifest.jsonl'),
        source_m0_selection=str(M0/'selection.json'),source_m0_selection_sha256=sha(M0/'selection.json'),source_m0_cache=str(M0/'input_cache.pt'),
        source_m0_cache_sha256=sha(M0/'input_cache.pt'),selection_rule=dict(visibility_raw_best_min=128,replay_strict_success=True,IK_failed_steps=0,dwell_timeouts=0,
            preserve_m0_train_ids=m0['train_episode_ids'],excluded_visual_train_ids=sorted(EXCLUDED_VISUAL_TRAIN),excluded_previous_val_ids=sorted(excluded_val),
            geometric_rule='deterministic greedy farthest point using original M0 robust geometric feature scales; original eight train retained; no model performance consulted',
            qualified_extra_train_candidates=len(traincand),qualified_new_val_candidates=len(vc),new_val_and_heldout_disjoint=True),
        phase_mapping='physics trace frame/60 vs recorded timestamps, tolerance 1e-7; not physics index vs video index',
        train_episode_ids=[e['episode_id'] for e in sets['train']],episodes_train=sets['train'],episodes_new_val=sets['new_val'],episodes_heldout=sets['heldout'],
        evaluation_sets=eval_sets,development_scenes=dev,closed_loop_scenes=dev,heldout_scenes=hold,
        group_full_window_counts=group_windows,contact_sheets=sheet_paths,
        observability_limits='Raw selected-target visible pixel gate precedes crop. Actual prepare_rgb reset/closure sheets and nonblank checks supplied; this is not a segmentation-based post-crop visibility guarantee, exact target identification or model perception proof.',
        heldout_limits='12 independent curated observable historical replay-PASS val episodes; not natural new-seed population testing. No model outcome consulted. Excluded from training and all offline panels.',
        replay_status='Historical GT clean for all. Four old M0 development scenes previously validated; new four development and 12 heldout live GT control-path revalidation pending root scheduling, not silently counted as current passes.')
    save(selection_path,selection)
    schedules=make_schedules(selection)
    report=dict(status='PASS',selection_sha256=sha(selection_path),train_episodes=128,m0_train_retained=8,added_train=120,
        old_train_windows_exact=m0['windows'][:80]==eval_sets['old_train'],old_val_windows_exact=[w for w in m0['windows'] if w['split']=='val']==eval_sets['old_val'],
        new_val_episodes=8,new_val_windows=80,heldout_episodes=12,development_scenes=8,all_new_roles_seed_disjoint=True,
        all_annotations_hashes_match=True,video_metadata_and_sampled_frame_validation='296 videos: len/FPS checked, reset and closure frame decoded through actual prepare_rgb; not full video re-decoding',
        processed_coverage_frames=148*4,group_full_window_counts=group_windows,schedule_quota_counts=schedules['arms']['B']['group_counts'],
        A_unique_windows=schedules['arms']['A']['unique_windows'],B_unique_windows=schedules['arms']['B']['unique_windows'],
        actual_processed_image_sheets=sheet_paths,manual_visual_review='Pending root spot-check; no exhaustive visual-identifiability claim',cuda_initialized=torch.cuda.is_initialized())
    assert report['old_train_windows_exact'] and report['old_val_windows_exact'] and not report['cuda_initialized']
    save(ROOT/'selection_validation.json',report)
    print(json.dumps(dict(selection_sha256=report['selection_sha256'],group_windows=group_windows,availability=schedules['availability'],sheets=sheet_paths)),flush=True)


def prepare_eval_cache():
    path=ROOT/'eval_cache.pt'
    if path.exists():raise FileExistsError('Evaluation input cache already exists')
    selection=load(ROOT/'selection.json');m0_validation=load(M0/'cpu_validation.json')
    assert sha(M0/'input_cache.pt')==selection['source_m0_cache_sha256']==m0_validation['input_cache_sha256']
    old=torch.load(M0/'input_cache.pt',map_location='cpu',weights_only=True,mmap=True)
    assert old['provenance']['selection_sha256']==selection['source_m0_selection_sha256'] and old['provenance']['stats_sha256']==selection['stats_sha256']
    _,mean,std=load_stats(STATS);assert torch.equal(old['mean'],torch.from_numpy(mean)) and torch.equal(old['std'],torch.from_numpy(std))
    samples=[]
    for original in old['samples']:
        item=dict(original);item['meta']=dict(original['meta']);item['meta']['evaluation_set']='old_'+item['meta']['split']
        samples.append(item)
    dm=OrchardBenchDataModule(dict(processor_path=str(XR0.parent/'checkpoints/Xiaomi-Robotics-0-Calvin-ABCD_D')))
    params=dict(train_datasets=dict(root=str(DATA),split='val',episode_ids=None,stats_path=str(STATS),action_length=30,batch_size=1))
    ds=OrchardBenchDataset(params)
    lookup={(str(Path(p).resolve()),frame):i for i,(p,frame) in enumerate(ds.samples)}
    parity=None
    for j,original in enumerate(selection['evaluation_sets']['new_val']):
        meta=dict(original);f=meta['frame'];annotation=Path(meta['annotation']);assert sha(annotation)==meta['annotation_sha256']
        traj=load(annotation);index=lookup[(str(annotation.resolve()),f)]
        batch=dm.collate_fn([ds[index]])
        assert all(isinstance(v,torch.Tensor) and v.device.type=='cpu' for v in batch.values())
        gt=encode_window(traj,f);assert np.array_equal(batch['action'][0].numpy(),(gt-mean)/(std+EPS))
        assert torch.equal(batch['action_mask'][0],torch.from_numpy(action_mask()))
        anchor_phase,target_phases,target_times=phase_metadata(traj,f)
        _,intent=width_intent(np.asarray(traj['actions']['gripper_pos'])[:f,0])
        meta.update(evaluation_set='new_val',anchor_phase=anchor_phase,target_phase_by_horizon=target_phases,target_times_s=target_times.tolist(),
            dataset_index=index,anchor_position=traj['proprios']['ee_pos'][f],anchor_rotation=np.asarray(traj['proprios']['ee_rotm'][f]).reshape(3,3).tolist(),
            initial_width_intent_reference=intent)
        sample=dict(meta=meta,batch={k:v.contiguous() for k,v in batch.items()},gt_action_physical=torch.from_numpy(gt))
        if parity is None:parity=cpu_deployment_parity(dm,sample,traj)
        samples.append(sample)
        if (j+1)%20==0:print(json.dumps(dict(new_val_eval_cached=j+1,total=80)),flush=True)
    assert len(samples)==240 and not torch.cuda.is_initialized()
    for a,b in zip(old['samples'],samples[:160]):
        assert torch.equal(a['gt_action_physical'],b['gt_action_physical'])
        assert all(torch.equal(v,b['batch'][k]) for k,v in a['batch'].items())
        assert {k:v for k,v in b['meta'].items() if k!='evaluation_set'}==a['meta']
    provenance=dict(selection_sha256=sha(ROOT/'selection.json'),stats_sha256=sha(STATS),source_m0_selection_sha256=selection['source_m0_selection_sha256'],
        source_m0_cache_sha256=selection['source_m0_cache_sha256'],normalization_source_fingerprint=ds.stats['source_sha256'],
        source_files={str(p):sha(p) for p in [Path(__file__),M0/'select_dataset.py',ORCHARD/'treesim/orchard_action.py',XR0/'mibot/data/datasets/orchardbench_dataset.py']})
    payload=dict(schema='orchard_128_ab_eval_input_cache_v1',provenance=provenance,samples=samples,mean=torch.from_numpy(mean),std=torch.from_numpy(std))
    with path.open('xb') as f:torch.save(payload,f)
    report=dict(status='PASS',selection_sha256=provenance['selection_sha256'],eval_cache_path=str(path),eval_cache_sha256=sha(path),eval_cache_bytes=path.stat().st_size,
        panel_counts=dict(Counter(s['meta']['evaluation_set'] for s in samples)),old_160_batches_gt_and_metadata_exact_except_evaluation_set=True,
        new80_full_val_dataset_then_index=True,dataset_to_deployment_tensor_parity=parity,cuda_initialized=False,stats_sha256=sha(STATS))
    save(ROOT/'cache_validation.json',report)
    save(ROOT/'data_validation.json',dict(status='PASS',selection_validation=load(ROOT/'selection_validation.json'),cache_validation=report))
    print(json.dumps(report),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--cohort-only',action='store_true');p.add_argument('--cache-eval',action='store_true');args=p.parse_args()
    if args.cache_eval:prepare_eval_cache()
    else:
        prepare_cohort()
        if not args.cohort_only:prepare_eval_cache()


if __name__=='__main__':main()
