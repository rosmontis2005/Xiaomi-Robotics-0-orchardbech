#!/usr/bin/env python3
"""Freeze M0 observable/clean episodes and real windows; CPU validation only."""
import sys
sys.dont_write_bytecode=True
import bootstrap
import ast
from bisect import bisect_right
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation
from decord import VideoReader, cpu
from treesim.orchard_action import CONTRACT, HORIZON, EPS, encode_window, decode_targets, prepare_rgb

OUT=Path(__file__).resolve().parent
XR0=OUT.parent
ORCHARD=Path('/home/rosmontis/Projects/orchardbench')
DATA=ORCHARD/'data/orchard_v1_2650/filtered'
STATS=DATA/'action_stats.json'
AUDIT=XR0/'diagnosis_1003/reach_plan/dataset_phase_visibility_audit.json'
EXCLUDED_VAL={'episode_000010','episode_000020','episode_000030','episode_000040'}
VISUAL_REPLACEMENTS={'episode_000656':'episode_000613','episode_000598':'episode_001129'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path,data):
    assert path.resolve().is_relative_to(OUT)
    with path.open('x') as f:json.dump(data,f,indent=2,allow_nan=False);f.write('\n')

def native(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:native(v) for k,v in x.items()}
    if isinstance(x,list):return [native(v) for v in x]
    return x


def phase_events(t):
    ts=np.asarray(t['orchardbench']['timestamps'])
    out=[]
    for e in t['orchardbench']['fixed_base_expert']['state_trace']:
        when=e['frame']/60.
        assert abs(when-e['sim_time'])<=.000051
        f=int(np.searchsorted(ts,when-1e-7,side='left'))
        out.append(dict(phase=e['state'],physics_frame=e['frame'],event_time_s=when,recorded_frame=min(f,len(ts)-1)))
    return out


def phase(t,f):
    tr=t['orchardbench']['fixed_base_expert']['state_trace']
    idx=bisect_right([r['frame']/60 for r in tr],t['orchardbench']['timestamps'][f]+1e-7)-1
    return tr[max(0,idx)]['state']


def width_method():
    path=ORCHARD/'treesim/vla_env.py'
    tree=ast.parse(path.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='OrchardVLAEnv')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_update_width_intent')
    ns={};exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),ns)
    return ns['_update_width_intent']


UPDATE=width_method()


def closure_frame(t):
    obj=SimpleNamespace(config=SimpleNamespace(gripper_open_width=.075,gripper_open_steps=5,gripper_close_deadband=.001),
                        _gripper=0.,_width_open_steps=0,_gripper_width_command=.04)
    rows=[]
    for f,raw in enumerate(np.asarray(t['actions']['gripper_pos']).ravel()):
        w=float(np.clip(raw,0,.08));old=obj._gripper
        UPDATE(obj,w)
        rows.append(dict(frame=f,width_raw_m=float(raw),width_clipped_m=w,intent_before=old,intent_after=obj._gripper))
        obj._gripper_width_command=w
    switches=[r['frame'] for r in rows if r['intent_before']>=0 and r['intent_after']<0]
    if not switches:raise ValueError('No measurable width-intent closure transition in accepted episode')
    return switches[0],rows


def windows(t):
    ev=phase_events(t);starts={r['phase']:r['recorded_frame'] for r in ev}
    g,p,tr=[starts[s] for s in ['GRASP','PULL','TRANSPORT']]
    c,width_history=closure_frame(t)
    desired=[('reset',0),('early_reach',2),('before_grasp',g-1),('grasp_entry',g),
             ('grasp_middle',(g+p-1)//2),('before_close',c-1),('close_transition',c),
             ('after_close',c+2),('early_pull',p+3),('transport_entry',tr)]
    chosen=[];omissions=[];last=t['num_frames']-HORIZON
    for label,f in desired:
        if not 0<=f<=last:
            omissions.append(dict(label=label,requested_frame=f,reason='no_complete_window'));continue
        duplicate=next((r for r in chosen if r['frame']==f),None)
        if duplicate:
            duplicate['labels'].append(label)
            omissions.append(dict(label=label,requested_frame=f,reason='coincides_with_existing_real_window',existing_label=duplicate['label']))
            continue
        chosen.append(dict(frame=f,label=label,labels=[label],selection_origin='primary_requested_keyframe'))
    # Duplicate phase landmarks are not invented or shifted. Fill remaining slots
    # with explicitly named real frames, prioritizing nearby front-stage observations.
    backups=[('early_frame1',1),('early_frame3',3),('early_frame4',4),('grasp_after_entry',g+1),
             ('grasp_before_exit',p-1),('close_next_frame',c+1),('pull_entry',p),('late_reach',g-2),
             ('pull_late',p+5),('transport_after_entry',tr+2)]
    for label,f in backups:
        if len(chosen)>=10:break
        if 0<=f<=last and not any(r['frame']==f for r in chosen):
            chosen.append(dict(frame=f,label=label,labels=[label],selection_origin='actual_frame_fallback_for_duplicate_or_unavailable_landmark'))
    assert len(chosen)==10 and len({r['frame'] for r in chosen})==10
    chosen.sort(key=lambda r:r['frame'])
    for r in chosen:
        f=r['frame']
        r.update(time_s=t['orchardbench']['timestamps'][f],anchor_phase=phase(t,f),
                 target_phase_by_horizon=[phase(t,min(k+1,t['num_frames']-1)) for k in range(f,f+HORIZON)],
                 observation_width_m=float(t['proprios']['gripper_pos'][f][0]),
                 first_target_width_m=float(t['actions']['gripper_pos'][f][0]))
    return chosen,ev,dict(first_close_action_index=c,method='unchanged production width-intent method applied to measured target widths at one target per expert 30Hz step; not reach physical timing',
                         width_history=width_history),omissions


def candidate_geometry(m):
    tcp=np.asarray(m['state_trace'][0]['tcp_world'],dtype=float)
    fruit=np.asarray(m['stance']['selected_apple_initial_world_pose'],dtype=float)
    delta=fruit-tcp;dist=float(np.linalg.norm(delta));bearing=math.atan2(delta[1],delta[0])
    best=max(m['target_visibility']['initial']['static_visible_pixels'],m['target_visibility']['initial']['wrist_visible_pixels'])
    features=[math.cos(bearing),math.sin(bearing),dist,float(fruit[2]),math.log1p(best)]
    return dict(target_world=fruit.tolist(),reset_tcp_world_rounded=tcp.tolist(),target_delta_world_rounded=delta.tolist(),
                tcp_to_target_distance_m=dist,approach_world_bearing_rad=bearing,target_height_m=float(fruit[2]),
                initial_best_view_pixels=best,base_xy_yaw=m['stance']['selected_base_pose_xy_yaw'],selection_features=features)


def diverse_select(candidates,count=8):
    feature=np.asarray([r['_geometry']['selection_features'] for r in candidates])
    lo=np.quantile(feature,.05,axis=0);hi=np.quantile(feature,.95,axis=0)
    scaled=np.clip((feature-(lo+hi)/2)/np.maximum(hi-lo,1e-9),-1,1)
    # Representative central first episode, then deterministic farthest-point spread.
    central=np.median(scaled,axis=0)
    first=min(range(len(candidates)),key=lambda i:(float(np.sum((scaled[i]-central)**2)),candidates[i]['seed']))
    indices=[first]
    while len(indices)<count:
        options=[i for i in range(len(candidates)) if i not in indices]
        nxt=min(options,key=lambda i:(-float(np.min(np.sum((scaled[i]-scaled[indices])**2,axis=1))),candidates[i]['seed']))
        indices.append(nxt)
    return [candidates[i] for i in indices],dict(features=['cos_world_approach_bearing','sin_world_approach_bearing','tcp_target_distance','fruit_height','log1p_initial_best_view_pixels'],
        robust_low_p05=lo.tolist(),robust_high_p95=hi.tolist(),selection='central representative then farthest point; each feature robust-normalized and clipped; seed tie-break; no model results')


def video_path(t,view):
    p=Path(t['observations'][view][0]['path'])
    if not p.is_absolute():p=DATA/p
    if not p.exists():p=DATA/'videos'/p.name
    return p


def contact_sheet(ep,prepared,folder):
    close=ep['closure_reference']['first_close_action_index']
    frames=[0,max(0,close-1),min(close+2,ep['num_frames']-1)]
    labels=['reset','before close reference','after close reference']
    panel_w,panel_h=384,256
    sheet=Image.new('RGB',(panel_w*2+24,(panel_h+35)*3+45),(245,245,245));draw=ImageDraw.Draw(sheet)
    draw.text((12,8),f"{ep['split']} {ep['episode_id']} seed {ep['seed']} | prepared 192x128 (display 2x)",fill=(0,0,0))
    for row,(f,label) in enumerate(zip(frames,labels)):
        y=38+row*(panel_h+35)
        draw.text((12,y),f"{label}, frame {f}, phase {phase(ep['_traj'],f)}",fill=(0,0,0))
        for col,view in enumerate(['ego','wrist_left']):
            im=prepared[(view,f)]
            sheet.paste(im.resize((panel_w,panel_h),Image.Resampling.NEAREST),(12+col*panel_w,y+22))
    path=folder/f"{ep['split']}_{ep['episode_id']}.png"
    if path.exists():raise FileExistsError(path)
    sheet.save(path)
    return str(path),frames


def main():
    selection_path=OUT/'selection.json';validation_path=OUT/'selection_validation.json'
    if selection_path.exists() or validation_path.exists():raise FileExistsError('M0 selection already exists; will not overwrite')
    folder=OUT/'contact_sheets';folder.mkdir(exist_ok=False)
    raw=(DATA/'manifest.jsonl').read_bytes();manifest=[json.loads(l) for l in raw.decode().splitlines()]
    audit=json.loads(AUDIT.read_text());audited={(r['split'],r['episode_id']):r for r in audit['episodes']}
    stat=json.loads(STATS.read_text());mean=np.asarray(stat['mean']);std=np.asarray(stat['std'])
    assert stat['contract']==CONTRACT and stat['source_split']=='train' and mean.shape==std.shape==(30,32)
    assert np.isfinite(mean).all() and np.isfinite(std).all() and (std>0).all()
    assert np.array_equal(mean[:,7:],np.zeros((30,25))) and np.array_equal(std[:,7:],np.ones((30,25)))
    assert sha(STATS)=='63dec8bbf2249d505358f17beac21d800bd49dee1192c9e79ddf3e1085c9b371'
    selected=[];selection_rules={};candidate_counts={}
    for split in ['train','val']:
        candidates=[]
        for m in manifest:
            if m['split']!=split or not m['oracle_strict_success'] or not m['accepted']:continue
            if split=='val' and m['episode_id'] in EXCLUDED_VAL:continue
            if m['replay']['IK_failed_steps'] or m['replay']['dwell_timeout_count']:continue
            g=candidate_geometry(m)
            if g['initial_best_view_pixels']<128:continue
            if not audited[(split,m['episode_id'])]['oracle_strict_success']:continue
            m['_geometry']=g;candidates.append(m)
        candidate_counts[split]=len(candidates)
        chosen,rule=diverse_select(candidates)
        selection_rules[split]=rule;selected.extend(chosen)
    # Revision 2: root reviewed actual prepare_rgb inputs before any model run.
    # Keep six original train and all eight val episodes; replace only two ambiguous/border cases.
    for i, m in enumerate(selected):
        if m['split']=='train' and m['episode_id'] in VISUAL_REPLACEMENTS:
            replacement=next(r for r in manifest if r['episode_id']==VISUAL_REPLACEMENTS[m['episode_id']] and r['split']=='train')
            replacement['_geometry']=candidate_geometry(replacement)
            assert replacement['_geometry']['initial_best_view_pixels']>=160
            assert replacement['oracle_strict_success'] and replacement['replay']['IK_failed_steps']==0 and replacement['replay']['dwell_timeout_count']==0
            selected[i]=replacement
    episodes=[];checks=[];flat=[];overview_rows={'train':[],'val':[]}
    for order,m in enumerate(selected):
        path=DATA/m['annotation'];t=json.loads(path.read_text());n=t['num_frames']
        assert sha(path)==m['annotation_sha256'] and t['split']==m['split'] and t['seed']==m['seed']
        assert t['record_fps']==30 and len(t['orchardbench']['timestamps'])==n
        assert np.allclose(t['orchardbench']['timestamps'],np.arange(n)/30,atol=1e-7,rtol=0)
        wins,events,close,omissions=windows(t)
        ep=dict(split=m['split'],episode_id=m['episode_id'],seed=m['seed'],annotation=str(path),annotation_sha256=sha(path),
            num_frames=n,selection_order_within_split=order%8,visibility=m['target_visibility'],replay=m['replay'],
            oracle_strict_success=m['oracle_strict_success'],geometry=m['_geometry'],events=events,closure_reference=close,
            windows=wins,window_landmark_omissions=omissions,_traj=t)
        gt_position_error=gt_rotation_error=gt_width_error=next_state_error=0.
        for key in ['ee_pos','ee_rotm','arm_joint','gripper_pos']:
            a=np.asarray(t['actions'][key]);p=np.asarray(t['proprios'][key])
            assert np.isfinite(a).all() and np.isfinite(p).all() and len(a)==len(p)==n
            next_state_error=max(next_state_error,float(np.max(np.abs(a[:-1]-p[1:]))),float(np.max(np.abs(a[-1]-p[-1]))))
        assert next_state_error<1e-7
        decode_checks=[];prepared={};video_info={}
        sheet_frames=[0,max(0,close['first_close_action_index']-1),min(close['first_close_action_index']+2,n-1)]
        for view in ['ego','wrist_left']:
            vp=video_path(t,view);reader=VideoReader(str(vp),ctx=cpu(0),num_threads=1)
            assert len(reader)==n and abs(reader.get_avg_fps()-30)<1e-6
            frames=reader.get_batch(list(range(n))).asnumpy()
            assert frames.shape==(n,144,192,3) and frames.dtype==np.uint8
            for f in sorted(set(sheet_frames+[w['frame'] for w in wins])):
                im=prepare_rgb(Image.fromarray(frames[f]));assert im.size==(192,128)
                prepared[(view,f)]=im
            video_info[view]=dict(path=str(vp),sha256=sha(vp),decoded_frames=len(frames),fps=reader.get_avg_fps(),
                raw_shape=list(frames.shape[1:]),prepared_size=[192,128],
                decoded_reset_rgb_sha256=hashlib.sha256(frames[0].tobytes()).hexdigest(),
                prepared_reset_rgb_sha256=hashlib.sha256(np.asarray(prepared[(view,0)]).tobytes()).hexdigest(),
                decoded_reset_std=float(frames[0].std()),full_video_decode_verified=True,
                raw_reset_hash_comparison='Decoded compressed RGB hash is not expected to equal simulator raw RGB hash.')
            del frames,reader
        for w in wins:
            f=w['frame'];a=encode_window(t,f);normalized=(a-mean)/(std+EPS);denorm=normalized*(std+EPS)+mean
            r=np.asarray(t['proprios']['ee_rotm'][f]).reshape(3,3)
            xyz,rot,width=decode_targets(denorm,t['proprios']['ee_pos'][f],r)
            poserr=float(np.max(np.linalg.norm(xyz-np.asarray(t['actions']['ee_pos'][f:f+HORIZON]),axis=1)))
            roterr=float(np.max(Rotation.from_matrix(rot@np.asarray(t['actions']['ee_rotm'][f:f+HORIZON]).reshape(30,3,3).transpose(0,2,1)).magnitude()))
            widerr=float(np.max(np.abs(width-np.asarray(t['actions']['gripper_pos'][f:f+HORIZON]).ravel())))
            assert poserr<2e-7 and roterr<2e-7 and widerr<1e-8 and np.isfinite(normalized).all()
            w.update(window_id=f"{ep['split']}/{ep['episode_id']}/frame{f:04d}",split=ep['split'],episode_id=ep['episode_id'],
                seed=ep['seed'],annotation=str(path),annotation_sha256=ep['annotation_sha256'],
                initial_best_view_pixels=ep['geometry']['initial_best_view_pixels'],
                prepared_observation_sha256={view:hashlib.sha256(np.asarray(prepared[(view,f)]).tobytes()).hexdigest() for view in ['ego','wrist_left']})
            flat.append(dict(w))
            gt_position_error=max(gt_position_error,poserr);gt_rotation_error=max(gt_rotation_error,roterr);gt_width_error=max(gt_width_error,widerr)
        ep['videos']=video_info
        ep['contact_sheet'],ep['contact_sheet_frames']=contact_sheet(ep,prepared,folder)
        reference=close['first_close_action_index']
        if ('ego',reference) not in prepared:
            for view in ['ego','wrist_left']:
                reader=VideoReader(video_info[view]['path'],ctx=cpu(0),num_threads=1)
                prepared[(view,reference)]=prepare_rgb(Image.fromarray(reader[reference].asnumpy()))
        overview_rows[ep['split']].append((f"{ep['episode_id']} s{ep['seed']} vis={ep['geometry']['initial_best_view_pixels']} fclose={reference}",
            [prepared[('ego',0)],prepared[('wrist_left',0)],prepared[('ego',reference)],prepared[('wrist_left',reference)]]))
        ep.pop('_traj')
        checks.append(dict(split=ep['split'],episode_id=ep['episode_id'],seed=ep['seed'],status='PASS',
            windows=len(wins),next_state_max_abs_error=next_state_error,roundtrip_max_position_m=gt_position_error,
            roundtrip_max_rotation_rad=gt_rotation_error,roundtrip_max_width_m=gt_width_error,
            anchor_phase_counts=dict(Counter(w['anchor_phase'] for w in wins)),window_landmark_omissions=omissions,
            full_video_frame_count=[video_info[v]['decoded_frames'] for v in ['ego','wrist_left']]))
        episodes.append(ep)
        print(json.dumps(dict(selected=order+1,split=ep['split'],episode_id=ep['episode_id'],seed=ep['seed'],best_pixels=ep['geometry']['initial_best_view_pixels'],frames=[w['frame'] for w in wins])),flush=True)
    overviews={}
    for split,rows in overview_rows.items():
        image=Image.new('RGB',(4*192+24,8*(128+32)+32),(245,245,245));draw=ImageDraw.Draw(image)
        draw.text((12,8),'reset static | reset wrist | close-reference static | close-reference wrist (actual prepare_rgb)',fill='black')
        for i,(title,ims) in enumerate(rows):
            y=32+i*160;draw.text((12,y),title,fill='black')
            for j,im in enumerate(ims):image.paste(im,(12+j*192,y+22))
        op=folder/f'overview_{split}.png';image.save(op);overviews[split]=str(op)
    loop=[]
    for split in ['train','val']:
        group=[e for e in episodes if e['split']==split]
        for e in [group[0],group[4]]:
            loop.append({k:e[k] for k in ['split','episode_id','seed','annotation','annotation_sha256','visibility','replay','geometry','oracle_strict_success']})
    assert len(episodes)==16 and len(flat)==160
    assert len({e['seed'] for e in episodes})==16 and len({w['window_id'] for w in flat})==160
    source_files=[ORCHARD/'treesim/orchard_action.py',ORCHARD/'treesim/vla_env.py',XR0/'mibot/data/datasets/orchardbench_dataset.py']
    selection=dict(schema='orchard_m0_small_set_selection_v1',selection_revision=2,visual_review=dict(reviewer='root',decision='approved for M0 before any model run',replacements=VISUAL_REPLACEMENTS,reason='actual processed image visual ambiguity / border truncation in two initial train samples',replacement_rule='first two deterministic farthest-point candidates from original robust geometry scale, conditioned on six retained train episodes, with raw best-view pixels >=160',archive=str(OUT/'pre_visual_review'),initial_selection_sha256=sha(OUT/'pre_visual_review/selection.json'),limits='Root reviewed two overviews and six original/replacement single-episode sheets. Acceptance does not assert targets are centered, unoccluded, or precise fruit IDs visually confirmed; 613 lower-border truncation, 1129 partial leaf occlusion, val210/2430 border/multiple-fruit limitations retained.'),created_utc=datetime.now(timezone.utc).isoformat(),contract=CONTRACT,horizon=30,model_seed=42,
        dataset_root=str(DATA),stats_path=str(STATS),source_stats=str(STATS),stats_sha256=sha(STATS),
        source_manifest=str(DATA/'manifest.jsonl'),source_manifest_sha256=hashlib.sha256(raw).hexdigest(),
        source_audit=str(AUDIT),source_audit_sha256=sha(AUDIT),source_sha256={str(p):sha(p) for p in source_files},
        selection_rule=dict(initial_best_view_pixels_minimum=128,strict_replay_pass=True,maximum_recorded_IK_failed_steps=0,
            maximum_recorded_dwell_timeouts=0,excluded_previous_val_episode_ids=sorted(EXCLUDED_VAL),candidate_counts=candidate_counts,
            geometric_spread=selection_rules,model_results_consulted=False),
        phase_mapping='physics state_trace frame/60 mapped to original recorded 30Hz timestamps; no frame-index mixing',
        window_rule='10 unique actual complete windows per episode: fixed early/GRASP/closure landmarks, with explicit deduplication and named real-frame fallbacks; no fabricated phases or labels',
        width_intent_proxy_limits='One measured target per expert 30Hz step. Not actual expert intent and not reach-conditioned hold duration.',
        train_episode_ids=[e['episode_id'] for e in episodes if e['split']=='train'],val_episode_ids=[e['episode_id'] for e in episodes if e['split']=='val'],
        episodes=episodes,windows=flat,closed_loop_scenes=loop,contact_sheet_overviews=overviews,
        observability_limits='Raw target-visible-pixel gate precedes 95% crop; prepared sheets are for human inspection. Pixel count is not a proof of perceptual identifiability. Other valid apples may exist; selection is only an easy-case development cohort, not natural test-set filtering.')
    write(selection_path,native(selection))
    validation=dict(status='PASS',created_utc=datetime.now(timezone.utc).isoformat(),selection_path=str(selection_path),selection_sha256=sha(selection_path),
        episodes=16,train_episodes=8,val_episodes=8,train_windows=80,val_windows=80,total_unique_windows=160,
        train_val_seed_disjoint=True,excluded_previous_val_episode_ids_absent=True,stats_unchanged=True,
        stats_sha256=sha(STATS),contract=CONTRACT,full_video_decode_count=32,
        phase_anchor_counts={split:dict(Counter(w['anchor_phase'] for w in flat if w['split']==split)) for split in ['train','val']},
        checks=checks,contact_sheet_count=18,selection_revision=2,manual_visual_review='root accepted M0 after reviewing two overviews and six single-episode sheets before model execution; see selection.visual_review limitations')
    write(validation_path,native(validation))
    print(json.dumps(dict(selection=str(selection_path),validation=str(validation_path),closed_loop_seeds=[e['seed'] for e in loop],overviews=overviews)),flush=True)


if __name__=='__main__':main()
