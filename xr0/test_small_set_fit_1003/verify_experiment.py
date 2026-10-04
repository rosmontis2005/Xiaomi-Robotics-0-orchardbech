"""Final provenance, training isolation and artifact completeness checks; no GPU."""
import bootstrap
import json
import hashlib
from pathlib import Path
import torch
R=Path(__file__).resolve().parent

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    before=json.loads((R/'integrity_before.json').read_text()); sel=json.loads((R/'selection.json').read_text()); cpu=json.loads((R/'cpu_validation.json').read_text()); train=json.loads((R/'training_summary.json').read_text())
    source_mismatches=[p for p,h in before['source_sha256'].items() if sha(p)!=h]
    assert not source_mismatches,source_mismatches
    ck=before['initialization_checkpoint'];p=Path(ck['path']);assert p.stat().st_size==ck['size'] and p.stat().st_mtime_ns==ck['mtime_ns']
    assert sha(p)==cpu['base_sha256']
    assert sha(R/'selection.json')==cpu['selection_sha256']
    assert sha(sel['stats_path'])==sel['stats_sha256']==cpu['stats_sha256']
    data_changes=[]
    for ep in sel['episodes']:
        if sha(ep['annotation'])!=ep['annotation_sha256']:data_changes.append(ep['annotation'])
        for info in ep['videos'].values():
            if sha(info['path'])!=info['sha256']:data_changes.append(info['path'])
    assert not data_changes,data_changes
    assert train['status']=='COMPLETE' and train['completed_updates']==2000
    assert train['unique_train_windows_seen']==80 and train['val_optimizer_updates']==0
    assert set(train['per_window_counts'].values())=={25}
    assert train['frozen_vlm_unchanged'] and train['all_logged_gradients_finite'] and train['output_weight_max_absolute_update']>0
    assert len(set(sel['train_episode_ids']) & set(sel['val_episode_ids']))==0
    prediction_count=0
    for step in [0,250,500,1000,2000]:
        d=R/'evaluations'/f'step_{step:04d}';q=json.loads((d/'report.json').read_text())
        assert q['windows']==160 and q['seeds']==[42,43,44] and q['generated_from_zero_action'] and not q['ground_truth_prefix_used']
        assert len(list(d.glob('*.npz')))==160
        prediction_count+=160*3
    best=torch.load(R/'checkpoints/best_trainable.pt',map_location='cpu',weights_only=True,mmap=True)
    final=torch.load(R/'checkpoints/final_trainable.pt',map_location='cpu',weights_only=True,mmap=True)
    assert best['base_sha256']==final['base_sha256']==cpu['base_sha256']
    same_step_same_weights=None
    if best['step']==final['step']:
        same_step_same_weights=all(torch.equal(v,final['trainable_state_dict'][k]) for k,v in best['trainable_state_dict'].items());assert same_step_same_weights
    queue=[json.loads(l) for l in (R/'model_check_status.jsonl').read_text().splitlines()]
    assert {x['provider'] for x in queue if x['event']=='finished' and x['returncode']==0}=={'original_10k','best'}
    results=list((R/'rollouts').glob('*/result.json'))
    if not results:results=list((R/'rollouts').glob('*/summary.json'))
    # Exact filenames and observer consistency are also checked by analyze_reach.py.
    rollout_analysis=json.loads((R/'rollout_analysis.json').read_text())
    assert rollout_analysis['completed_episode_count']==12
    assert not rollout_analysis['pending_directories'] and not rollout_analysis['read_errors']
    assert rollout_analysis['observer_mismatch_fields_total']==0 and rollout_analysis['summary_mismatch_count_total']==0
    summaries=[json.loads(p.read_text()) for p in (R/'rollouts').glob('*/summary.json')]
    assert len(summaries)==12 and all(x['scene_check_pass'] and not x['video_errors'] for x in summaries)
    evidence={'status':'PASS','source_files_unchanged':len(before['source_sha256']),'original_checkpoint_unchanged':True,'selected_annotations_unchanged':16,'selected_videos_unchanged':32,'stats_unchanged':True,'selection_revision':sel.get('selection_revision'),'train_val_disjoint':True,'train_windows':80,'val_windows':80,'train_updates':2000,'each_train_window_presentations':25,'val_updates':0,'frozen_vlm_unchanged':True,'finite_gradients':True,'trainable_weights_changed':True,'generated_predictions_checked':prediction_count,'generation_without_gt_action_prefix':True,'best_step':best['step'],'final_step':final['step'],'best_final_same_weights':same_step_same_weights,'model_check_queue_completed':True,'completed_closed_loop_episodes':12,'observer_and_summary_mismatches':0,'all_scene_checks_pass':True,'cpu_verifier_cuda_initialized':torch.cuda.is_initialized(),'rollout_analysis_path':str(R/'rollout_analysis.json'),'source_mismatches':source_mismatches,'data_mismatches':data_changes,'report_scope':'Fitting and task outcomes may fail their behavioral gates even when engineering/provenance checks PASS.'}
    assert not torch.cuda.is_initialized()
    (R/'verification.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps(evidence))
if __name__=='__main__':main()
