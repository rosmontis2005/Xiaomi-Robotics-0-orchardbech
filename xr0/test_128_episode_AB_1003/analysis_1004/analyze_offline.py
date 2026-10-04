"""CPU-only independent analysis of already saved A/B generation panels.
Never runs a model, consumes closed-loop/heldout outcomes, or changes frozen inputs.
"""
import os, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.dont_write_bytecode=True
os.environ.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLCONFIGDIR=str(HERE/'matplotlib_cache'))
import argparse, bisect, collections, csv, hashlib, json
import numpy as np
from scipy.spatial.transform import Rotation

SETS=['old_train','old_val','new_val']
METRICS=['position_error_m','rotation_error_rad','width_error_m']
BANDS={'h1':(0,1),'h1_5':(0,5),'h6_15':(5,15),'h16_30':(15,30),'full':(0,30)}
STEPS=[0,2000,4000,8000]
PHASES=['REACH','GRASP','PULL','TRANSPORT','DROP']
SELECTION=json.loads((ROOT/'selection.json').read_text())
EXPECTED={w['window_id']:dict(w,evaluation_set=s) for s,ws in SELECTION['evaluation_sets'].items() for w in ws}
ANNOTATIONS={}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def dump(path,obj):
    assert path.resolve().is_relative_to(HERE)
    with path.open('x') as f:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n')

def annotation(path):
    if path not in ANNOTATIONS:ANNOTATIONS[path]=json.loads(Path(path).read_text())
    return ANNOTATIONS[path]

def target_phases(w):
    tr=annotation(w['annotation']);times=np.array(tr['orchardbench']['timestamps']);trace=tr['orchardbench']['fixed_base_expert']['state_trace']
    events=[t['frame']/60 for t in trace];f=w['frame'];inds=np.arange(f+1,f+31).clip(max=len(times)-1)
    labels=[trace[max(0,bisect.bisect_right(events,float(times[i])+1e-7)-1)]['state'] for i in inds]
    return labels,times[inds]

def stats(a):
    a=np.asarray(a,dtype=float)
    return dict(count=int(a.size),mean=float(a.mean()),p50=float(np.quantile(a,.5)),p90=float(np.quantile(a,.9)),max=float(a.max()))

def load_eval(arm,step):
    path=ROOT/f'arms/{arm}/evaluations/step_{step:04d}/report.json'
    if not path.exists():return None
    report=json.loads(path.read_text());rows=report['window_metrics'];assert len(rows)==240
    assert report['seeds']==[42,43,44] and report['generated_from_zero_action'] and not report['ground_truth_prefix_used']
    assert report['selection_sha256']==sha(ROOT/'selection.json')
    data={};sources={str(path.relative_to(ROOT)):sha(path)};maxerr={m:0. for m in METRICS}
    for row in rows:
        w=EXPECTED[row['window_id']];assert sha(w['annotation'])==w['annotation_sha256'];assert all(row[k]==w[k] for k in ['evaluation_set','episode_id','seed','frame','anchor_phase','split'])
        assert row['window_id'] not in data
        with np.load(row['npz']) as z:arrays={k:z[k] for k in z.files}
        assert arrays['normalized_prediction'].shape==(3,30,32)
        assert arrays['prediction_seeds'].tolist()==[42,43,44]
        assert np.array_equal(arrays['horizon'],np.arange(1,31))
        assert all(np.isfinite(v).all() for v in arrays.values() if v.dtype.kind in 'fci')
        assert np.count_nonzero(arrays['normalized_prediction'][:,:,7:])==0
        ph,times=target_phases(w);assert arrays['target_phase'].tolist()==ph and np.array_equal(arrays['target_time_s'],times)
        # Recompute physical errors from independently saved world poses/widths.
        recompute={
          'position_error_m':np.linalg.norm(arrays['predicted_world_position']-arrays['gt_world_position'][None],axis=-1),
          'rotation_error_rad':Rotation.from_matrix((arrays['gt_world_rotation'][None].transpose(0,1,3,2)@arrays['predicted_world_rotation']).reshape(-1,3,3)).magnitude().reshape(3,30),
          'width_error_m':np.abs(arrays['predicted_width_raw_m']-arrays['gt_width_m'][None])}
        for m,v in recompute.items():
            diff=float(np.abs(v-arrays[m]).max());maxerr[m]=max(maxerr[m],diff);assert diff<1e-7
            for band,(lo,hi) in BANDS.items():
                assert np.isclose(v[:,lo:hi].mean(),row['metrics'][band][m]['mean'],atol=2e-8,rtol=0)
        data[row['window_id']]=dict(meta=row,arrays=arrays)
        sources[str(Path(row['npz']).relative_to(ROOT))]=sha(row['npz'])
    assert set(data)==set(EXPECTED)
    return dict(report=report,data=data,sources=sources,error_recomputation_max_abs=maxerr)

def selectors(meta,z):
    yield 'all',np.ones(30,bool)
    if meta['frame']==0:yield 'reset',np.ones(30,bool)
    if meta['frame']<5:yield 'first5_anchor_frames',np.ones(30,bool)
    yield 'anchor_'+meta['anchor_phase'],np.ones(30,bool)
    for ph in PHASES:
        mask=z['target_phase']==ph
        if mask.any():
            yield 'target_'+ph,mask
            if meta['frame']==0:yield 'reset_target_'+ph,mask
            yield 'anchor_'+meta['anchor_phase']+'_target_'+ph,mask

def aggregate(ev):
    raw=collections.defaultdict(list);counts=collections.defaultdict(set)
    closure={};horizon={}
    for wid,item in ev['data'].items():
        w,z=item['meta'],item['arrays'];panel=w['evaluation_set'];ep=w['episode_id']
        for group,base in selectors(w,z):
            for band,(lo,hi) in BANDS.items():
                mask=base.copy();mask[:lo]=False;mask[hi:]=False
                if not mask.any():continue
                for m in METRICS:
                    key=(panel,group,band,m,ep);raw[key].append(z[m][:,mask].reshape(-1));counts[key].add(wid)
    # Two fixed val panels remain separately reportable; their 16 disjoint scenes also form a combined validation view.
    for key,values in list(raw.items()):
        if key[0] in ['old_val','new_val']:
            merged=('val_combined',)+key[1:];raw[merged].extend(values);counts[merged].update(counts[key])
    per_episode={};combined=collections.defaultdict(list);episodes=collections.defaultdict(dict);wc=collections.defaultdict(int)
    for (panel,group,band,m,ep),arrays in raw.items():
        a=np.concatenate(arrays);key='/'.join((panel,group,band,m));rec=stats(a);rec['windows']=len(counts[(panel,group,band,m,ep)])
        per_episode.setdefault(key,{})[ep]=rec;combined[key].append(a);episodes[key][ep]=rec['mean'];wc[key]+=rec['windows']
    summary={}
    for key,arrays in combined.items():
        summary[key]=stats(np.concatenate(arrays));summary[key].update(episodes=len(episodes[key]),windows=wc[key],episode_equal_weight_mean=float(np.mean(list(episodes[key].values()))))
    for panel in SETS+['val_combined']:
        allowed=['old_val','new_val'] if panel=='val_combined' else [panel]
        rows=[x for x in ev['data'].values() if x['meta']['evaluation_set'] in allowed]
        for group in ['all','reset']:
            selected=[x for x in rows if group=='all' or x['meta']['frame']==0]
            cps=[c for x in selected for c in x['meta']['closure_proxy']];has=[c for c in cps if c['gt_first_close_switch_horizon'] is not None]
            offs=[c['close_switch_offset_targets'] for c in has if c['close_switch_offset_targets'] is not None]
            widths=np.concatenate([x['arrays']['predicted_width_raw_m'].reshape(-1) for x in selected])
            closure[panel+'/'+group]=dict(episodes=len(set(x['meta']['episode_id'] for x in selected)),windows=len(selected),window_seed_predictions=len(cps),gt_switch_predictions=len(has),missing_predictions=sum(c['missing_gt_close_switch'] for c in has),within_2_targets=sum(abs(x)<=2 for x in offs),early_more_than_2=sum(x< -2 for x in offs),late_more_than_2=sum(x>2 for x in offs),offset_mean_targets=float(np.mean(offs)) if offs else None,offset_abs_mean_targets=float(np.mean(np.abs(offs))) if offs else None,width_intent_mismatch_fraction=float(np.mean([c['width_intent_mismatch_fraction'] for c in cps])),raw_width_outside_0_08_count=int(((widths<0)|(widths>.08)).sum()),raw_width_targets=len(widths))
        horizon[panel]={m:[float(np.stack([x['arrays'][m] for x in rows]).mean(axis=(0,1))[i]) for i in range(30)] for m in METRICS}
    return dict(summary=summary,per_episode=per_episode,closure_proxy=closure,horizon_curves=horizon,fit_gate=ev['report']['fit_gate'])

def paired(before,after):
    result={};rng=np.random.default_rng(1004)
    for key in sorted(set(before['per_episode'])&set(after['per_episode'])):
        a,b=before['per_episode'][key],after['per_episode'][key];assert set(a)==set(b)
        ids=sorted(a);old=np.array([a[x]['mean'] for x in ids]);new=np.array([b[x]['mean'] for x in ids]);delta=new-old
        boot=delta[rng.integers(0,len(ids),size=(10000,len(ids)))].mean(1)
        result[key]=dict(episodes=len(ids),before_episode_mean=float(old.mean()),after_episode_mean=float(new.mean()),delta_after_minus_before=float(delta.mean()),percent_change=float(100*delta.mean()/old.mean()) if old.mean() else None,improved=int((delta< -1e-12).sum()),worsened=int((delta>1e-12).sum()),unchanged=int((abs(delta)<=1e-12).sum()),descriptive_scene_bootstrap_95_interval=np.quantile(boot,[.025,.975]).tolist(),per_episode_delta={ep:float(d) for ep,d in zip(ids,delta)})
    return result

def training_audit(arm):
    schedule=[json.loads(x) for x in (ROOT/f'schedules/{arm}.jsonl').read_text().splitlines()]
    selected={e['episode_id']:e for e in SELECTION['episodes_train']};held=set(x['episode_id'] for x in SELECTION['heldout_scenes']);val=set(w['episode_id'] for s in ['old_val','new_val'] for w in SELECTION['evaluation_sets'][s]);assert not(set(selected)&(held|val))
    for i,w in enumerate(schedule):
        assert w['step']==i+1 and w['split']=='train' and w['episode_id'] in selected
        ep=selected[w['episode_id']];assert w['annotation']==ep['annotation'] and w['annotation_sha256']==ep['annotation_sha256']
        assert w['frame'] in ep['group_frames'][w['group']]
    source_checks={};mf=json.loads((ROOT/f'arms/{arm}/run_manifest.json').read_text());prov=mf['provenance']
    for f,expected in prov['source_sha256'].items():source_checks[f]=sha(f)==expected
    assert all(source_checks.values())
    for f,key in [('selection.json','selection_sha256'),(f'schedules/{arm}.jsonl','schedule_sha256'),('eval_cache.pt','eval_cache_sha256')]:assert sha(ROOT/f)==prov[key]
    assert sha(SELECTION['stats_path'])==SELECTION['stats_sha256']==prov['stats_sha256']
    assert mf['initial_load_report']['non_vlm_source_fp32_tensor_exact_equal'];assert mf['initial_optimizer_state_entries']==0
    logs_path=ROOT/f'arms/{arm}/training_steps.jsonl';logs=[json.loads(x) for x in logs_path.read_text().splitlines()] if logs_path.exists() else []
    for i,row in enumerate(logs):
        assert all(row[k]==schedule[i][k] for k in ['step','window_id','episode_id','frame','group','split'])
        assert row['gradients_finite'] and np.isfinite([row['flow_loss'],row['gradient_norm_before_clip'],row['lr']]).all()
    counts=collections.Counter(w['group'] for w in schedule);episode=collections.Counter(w['episode_id'] for w in schedule)
    out=dict(scheduled_updates=len(schedule),actual_logged_updates=len(logs),actual_rows_match_schedule=True,all_logged_loss_gradients_finite=True,heldout_or_val_optimizer_rows=0,unique_windows=len(set(w['window_id'] for w in schedule)),train_episodes=len(episode),group_exposures=dict(counts),episode_exposure_min=min(episode.values()),episode_exposure_max=max(episode.values()),frozen_source_hashes_match=source_checks,source_fp32_initialization_exact=True,initial_optimizer_entries=0,loss_note='Per-update loss across arms is not comparable without accounting for different sampled targets and training trajectory. These bins are descriptive only.')
    out['source_annotation_hashes_match']=all(sha(e['annotation'])==e['annotation_sha256'] for e in selected.values())
    assert out['source_annotation_hashes_match']
    out['old_train_window_exposure_by_checkpoint']={}
    for n in [2000,4000,8000]:
        exposures=collections.Counter(w['window_id'] for w in schedule[:n]);panel=SELECTION['evaluation_sets']['old_train'];seen=[w for w in panel if exposures[w['window_id']]]
        out['old_train_window_exposure_by_checkpoint'][str(n)]=dict(exact_anchor_windows_seen=len(seen),panel_windows=80,reset_windows_seen=sum(w['frame']==0 for w in seen),reset_windows=8,total_draws_on_panel_windows=sum(exposures[w['window_id']] for w in panel),per_window_draws={w['window_id']:exposures[w['window_id']] for w in panel})
    out['loss_bins_by_training_group']={}
    for lo in range(0,len(logs),1000):
        rows=logs[lo:lo+1000];out['loss_bins_by_training_group'][f'{lo+1}-{lo+len(rows)}']={g:dict(count=len(v),mean_flow_loss=float(np.mean(v))) for g in PHASES+['reset','first1_4'] if (v:=[r['flow_loss'] for r in rows if r['group']==g])}
    summary_path=ROOT/f'arms/{arm}/training_summary.json'
    if summary_path.exists():
        summary=json.loads(summary_path.read_text());assert summary['completed_updates']==len(logs)==8000
        assert summary['val_optimizer_updates']==0 and summary['frozen_vlm_unchanged'] and summary['all_logged_gradients_finite']
        assert summary['frozen_vlm_parameter_sha256_before']==summary['frozen_vlm_parameter_sha256_after']
        assert dict(counts)==summary['group_counts']
        out['completion_evidence']={k:summary[k] for k in ['status','completed_updates','frozen_vlm_unchanged','frozen_vlm_parameter_sha256_before','frozen_vlm_parameter_sha256_after','trainable_master_dtype','output_weight_max_absolute_update','best_step','primary_endpoint_step','training_seconds','evaluation_seconds','process_elapsed_s']}
        out['completion_evidence']['scope']='Runtime summary records before/after hashes and update magnitude; analysis verifies consistency but does not independently load model weights.'
    return out

def plots(analyses,outdir):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'A':'#d07828','B':'#197991'}
    fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    for col,panel in enumerate(SETS):
        for arm in ['A','B']:
            steps=[s for s in STEPS if f'{arm}{s}' in analyses]
            for row,band in enumerate(['full','h1_5']):
                vals=[100*analyses[f'{arm}{s}']['summary'][f'{panel}/reset/{band}/position_error_m']['mean'] for s in steps]
                axes[row,col].plot(steps,vals,'o-',label=arm,color=colors[arm]);axes[row,col].set_title(f'{panel}: reset {band}');axes[row,col].set_ylabel('Mean position error (cm)');axes[row,col].set_xlabel('Additional optimizer updates');axes[row,col].grid(alpha=.25);axes[row,col].legend()
    fig.suptitle('Reset full-chunk error and first-five-target error describe different behavior')
    fig.savefig(outdir/'reset_trajectories.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    for col,panel in enumerate(SETS):
        for row,step in enumerate([4000,8000]):
            for arm in ['A','B']:
                key=f'{arm}{step}'
                if key not in analyses:continue
                y=np.array(analyses[key]['horizon_curves'][panel]['position_error_m'])*100
                axes[row,col].plot(np.arange(1,31),y,label=key,color=colors[arm])
            axes[row,col].set_title(f'{panel}: step {step}');axes[row,col].set_xlabel('Predicted target index');axes[row,col].set_ylabel('Mean position error (cm)');axes[row,col].grid(alpha=.25);axes[row,col].legend()
    fig.suptitle('Fixed expert observations, all ten windows per scene; not rollout tracking error')
    fig.savefig(outdir/'horizon_position.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for ax,panel in zip(axes,SETS):
        for phase in ['REACH','GRASP','PULL','TRANSPORT']:
            steps=[s for s in STEPS if f'B{s}' in analyses]
            vals=[100*analyses[f'B{s}']['summary'][f'{panel}/target_{phase}/full/position_error_m']['mean'] for s in steps]
            ax.plot(steps,vals,'o-',label=phase)
        ax.set_title(panel);ax.set_xlabel('B optimizer updates');ax.set_ylabel('Mean position error (cm)');ax.legend();ax.grid(alpha=.25)
    fig.suptitle('Phase of each future target, not window anchor phase; descriptive pooled targets')
    fig.savefig(outdir/'B_target_phase_position.png',dpi=170);plt.close(fig)
    if 'A8000' in analyses and 'B8000' in analyses:
        phases=['REACH','GRASP','PULL','TRANSPORT'];positions=np.arange(4)
        for metric,bands,scale,unit,filename in [
            ('position_error_m',['h1_5','full'],100,'cm','A_B_target_phase_position.png'),
            ('rotation_error_rad',['h1_5','full'],1,'rad','A_B_target_phase_rotation.png'),
            ('width_error_m',['h1_5','full'],1000,'mm','A_B_target_phase_width.png')]:
            fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
            for col,panel in enumerate(SETS):
                for row,band in enumerate(bands):
                    ax=axes[row,col]
                    for offset,name,color in [(-.25,'A8000',colors['A']),(0,'B8000',colors['B']),(.25,'B4000','#93c8d4')]:
                        values=[scale*analyses[name]['summary'][f'{panel}/target_{ph}/{band}/{metric}']['mean'] for ph in phases]
                        ax.bar(positions+offset,values,width=.24,label=name+(' secondary' if name=='B4000' else ''),color=color)
                    ax.set_xticks(positions,phases,rotation=20);ax.set_title(f'{panel}, {band}');ax.set_ylabel(f'Mean error ({unit})');ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
            fig.suptitle('Future-target phase, fixed offline inputs; pooled target entries (not independent scenes)')
            fig.savefig(outdir/filename,dpi=170);plt.close(fig)


def report_text(result):
    a=result['evaluations'];lines=['Offline analysis of saved predictions (CPU only)', 'Primary comparison: A8000 vs B8000. B4000 is train-panel-selected secondary evidence.', 'Each original panel contains 8 scenes × 10 fixed windows × 3 noise seeds; val_combined has16 disjoint scenes. Seeds/overlapping targets are not independent scenes.', 'Position: Euclidean m; rotation: SO(3) radians; width: total-opening absolute m; all before controller clipping.', 'first5_anchor_frames means observation frame 0..4; h1_5 means first 5 future targets. These are different axes.', 'Anchor-phase metrics may include targets from later phases. target_PHASE uses independently checked expert timestamps.', 'Coverage limitation: 240×30=7200 target slots contain REACH246, GRASP936, PULL5298, TRANSPORT720 and DROP0 (before3noise replication). DROP anchors are also0. These panels cannot validate late-release/DROP prediction; full30 is not full-task coverage.', 'old_train means episodes used by training; not all80 exact anchor windows were drawn. See training_audit exposure counts.', 'Exact pooled counts and scene means/deltas/95% descriptive bootstrap intervals are in JSON. Curated n=8 scenes per panel; intervals are not natural-population guarantees.', 'FP32 baseline uses corrected direct FP32 initialization from original 10k checkpoint. Historic M0 initial BF16 roundtrip differs and is not this paired baseline.', 'No heldout model outcomes read; no checkpoint selected by validation; full30 production loss preserved.', 'One training run/seed42 per arm. The3 inference noise seeds are repeated predictions, not independent training replicates; scene bootstrap does not quantify training-run uncertainty.', 'A is window-uniform without replacement; B combines phase quotas + episode balancing + replacement. Isolating phase-only causality is unsupported.', '']
    for panel in SETS+['val_combined']:
        lines.append(panel+' (pooled mean position cm: all/full, reset/full, reset/h1_5, target_GRASP/full, target_PULL/full)')
        for arm in ['A','B']:
            for step in STEPS:
                k=f'{arm}{step}'
                if k not in a:continue
                keys=['all/full','reset/full','reset/h1_5','target_GRASP/full','target_PULL/full'];v=[100*a[k]['summary'][f'{panel}/{g}/position_error_m']['mean'] for g in keys]
                lines.append('  '+k+': '+', '.join(f'{x:.3f}' for x in v))
        lines.append('')
    lines.extend(['Rotation rad / width mm (all windows, full 30 targets):'])
    for panel in SETS:
        for name in ['A0','A8000','B0','B4000','B8000']:
            if name not in a:continue
            metrics=a[name]['summary']
            rot=metrics[f'{panel}/all/full/rotation_error_rad']['mean'];width=1000*metrics[f'{panel}/all/full/width_error_m']['mean']
            lines.append(f'  {panel} {name}: rotation {rot:.5f} rad, width {width:.3f} mm')
    lines.append('Interpretation boundary: any 4k→8k worsening shared by train and val is not sufficient evidence of classic train-improves/val-worsens overfitting; it may reflect horizon/phase tradeoffs. Offline prediction gains do not establish contact, grasp, or success.')
    lines.append('Width-close timing is a target-sequence proxy at 30Hz, not actual reach-conditioned execution timing.')
    lines.append('Baseline exact comparison: '+json.dumps(result['A0_vs_B0_array_parity']))
    lines.append('Available evaluations: '+', '.join(a))
    return '\n'.join(lines)+'\n'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',required=True);args=ap.parse_args();out=HERE/args.name;out.mkdir(exist_ok=False)
    loaded={};analyses={};sources={};audits={}
    for arm in ['A','B']:
        if (ROOT/f'arms/{arm}/run_manifest.json').exists():audits[arm]=training_audit(arm)
        for step in STEPS:
            ev=load_eval(arm,step)
            if ev is None:continue
            key=f'{arm}{step}';loaded[key]=ev;analyses[key]=aggregate(ev);sources.update(ev['sources'])
            print('Analyzed',key,flush=True)
    reference=loaded['B0']['data']
    phase_counts=collections.Counter(ph for x in reference.values() for ph in x['arrays']['target_phase'].tolist())
    assert phase_counts==dict(REACH=246,GRASP=936,PULL=5298,TRANSPORT=720) and phase_counts['DROP']==0
    for ev in loaded.values():
        for wid,item in ev['data'].items():
            for key,a in reference[wid]['arrays'].items():
                if key.startswith('gt_') or key in ['target_phase','target_time_s','horizon']:
                    b=item['arrays'][key];assert a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
    parity={'status':'PENDING'}
    if 'A0' in loaded and 'B0' in loaded:
        n=0
        for wid,a in loaded['A0']['data'].items():
            b=loaded['B0']['data'][wid];assert set(a['arrays'])==set(b['arrays'])
            for k,v in a['arrays'].items():
                other=b['arrays'][k]
                assert v.dtype==other.dtype and v.shape==other.shape and v.tobytes()==other.tobytes(),(wid,k)
                n+=1
        parity=dict(status='PASS',windows=240,noise_seeds=[42,43,44],array_comparisons=n,all_arrays_bit_exact=True,note='Dtype, shape, and raw array bytes identical; not zip-file byte identity. Covers predictions, physical targets, errors, phases, and all saved fields.')
    comparisons={}
    for before,after in [('B0','B8000'),('B4000','B8000'),('B2000','B8000'),('A0','A8000'),('A4000','A8000'),('A8000','B8000'),('A4000','B4000')]:
        if before in analyses and after in analyses:comparisons[f'{before}_to_{after}']=paired(analyses[before],analyses[after])
    result=dict(schema='orchard_ab_independent_offline_analysis_v1',read_only_inputs=True,GPU_used=False,selection_sha256=sha(ROOT/'selection.json'),eval_cache_sha256=sha(ROOT/'eval_cache.pt'),script_sha256=sha(__file__),primary_endpoint=8000,observational_unit='Scene/episode; average its available fixed windows/targets and three noise seeds before paired uncertainty.',bootstrap=dict(repetitions=10000,seed=1004,method='Paired scene resampling with replacement, percentile 95%; descriptive only, curated 8 scenes per original panel /16 in val_combined.'),no_heldout_model_outcomes_read=True,offline_phase_coverage=dict(target_slots_before_noise_replication=dict(phase_counts),DROP_targets=0,DROP_anchors=0,late_release_prediction_evaluated=False),all_ground_truth_fields_identical_across_checkpoints=True,evaluations=analyses,paired_comparisons=comparisons,A0_vs_B0_array_parity=parity,training_audit=audits,input_source_sha256=sources,error_recomputation={k:v['error_recomputation_max_abs'] for k,v in loaded.items()})
    (out/'analyze_offline_source.py').write_text(Path(__file__).read_text())
    dump(out/'offline_analysis.json',result)
    (out/'offline_report.txt').write_text(report_text(result))
    with (out/'summary.csv').open('x',newline='') as f:
        wr=csv.writer(f);wr.writerow(['checkpoint','panel','group','horizon','metric','count','windows','episodes','pooled_mean','p90','episode_equal_weight_mean'])
        for name,agg in analyses.items():
            for key,v in agg['summary'].items():wr.writerow([name,*key.split('/'),v['count'],v['windows'],v['episodes'],v['mean'],v['p90'],v['episode_equal_weight_mean']])
    plots(analyses,out)
    dump(out/'validation.json',dict(status='PASS',evaluation_panels=len(analyses),windows_per_panel=240,A0_B0=parity,all_saved_errors_recomputed=True,all_target_phases_checked_against_source_timestamps=True,all_source_hashes_unchanged=True,all_annotation_and_original_stats_hashes_match=True,no_train_val_or_heldout_optimizer_leak=True,output_files=[str(x.relative_to(ROOT)) for x in out.iterdir()]))
    print('PASS',out,flush=True)

if __name__=='__main__':main()
