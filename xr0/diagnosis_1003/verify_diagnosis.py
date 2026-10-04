"""Read-only evidence consistency checks; writes only diagnosis verification.json."""
import bootstrap
import hashlib
import json
from pathlib import Path
root=bootstrap.ROOT
plan=json.loads((root/'plan.json').read_text())
expected=[(provider,mode,seed) for seed in [2010027,2010042,2010067]
          for provider,mode in [('gt','reach-conditioned'),('gt','time-indexed'),('step_10000','reach-conditioned')]]
missing=[f'{p}_{m}_{s}' for p,m,s in expected if not (root/'rollouts'/f'{p}_{m}_{s}'/'summary.json').exists()]
if missing:
    print(json.dumps(dict(status='PENDING',missing=missing)));raise SystemExit(2)
checks={}
checks['production_sources_and_stats_unchanged']=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in plan['source_sha256_before'].items())
checkpoint=plan['checkpoint_identity'];st=Path(checkpoint['path']).stat()
checks['checkpoint_size_mtime_unchanged']=st.st_size==checkpoint['size'] and st.st_mtime_ns==checkpoint['mtime_ns']
rows=[];initial=[]
for provider,mode,seed in expected:
 d=root/'rollouts'/f'{provider}_{mode}_{seed}'
 summary=json.loads((d/'summary.json').read_text())
 steps=[json.loads(x) for x in (d/'steps.jsonl').read_text().splitlines()]
 chunks=[json.loads(x) for x in (d/'chunks.jsonl').read_text().splitlines()]
 cc=dict(completed_without_error=summary['termination']!='error',step_count_matches=len(steps)==summary['control_steps'],
         indices_contiguous=[x['control_step'] for x in steps]==list(range(1,len(steps)+1)),
         within_budget=len(steps)<=900,observer_all_consistent=all(all(x['observer_consistency'].values()) for x in steps),
         clipping_matches=sum(x['info']['action_clipped'] for x in steps)==summary['action_clipped_steps'],
         ik_matches=sum(x['info']['ik_failed'] for x in steps)==summary['ik_failed_steps'],
         chunk_count_matches=len(chunks)==summary['chunks_loaded'],video_errors_absent=not summary['video_errors'])
 if provider=='step_10000': cc['predict_count_matches']=len(chunks)==summary['replans']
 checks[f'{provider}/{mode}/{seed}']=all(cc.values())
 rows.append(dict(provider=provider,mode=mode,seed=seed,checks=cc))
 a=json.loads((d/'initial.json').read_text())
 b=json.loads((root/'rollouts'/f'gt_reach-conditioned_{seed}'/'initial.json').read_text())
 initial.append(dict(provider=provider,mode=mode,seed=seed,obs_equal=a['obs']==b['obs'],info_equal=a['info']==b['info']))
checks['all_paired_resets_identical']=all(x['obs_equal'] and x['info_equal'] for x in initial)
for name in ['mean_action','pretrained','step_10000']:
 report=json.loads((root/'offline'/name/'report.json').read_text())
 checks[f'offline_{name}_80_windows']=report['windows']==80 and len(list((root/'offline'/name).glob('*.npz')))==80
result=dict(status='PASS' if all(checks.values()) else 'CHECK_FAILED',checks=checks,rollouts=rows,
            paired_resets=initial,expected_rollouts=len(expected),source_files_checked=len(plan['source_sha256_before']),
            checkpoint_identity_check='Size and mtime only; original checkpoint is read-only throughout diagnostics',
            protocol_amendment=str(root/'protocol_amendment.json'))
(root/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(status=result['status'],checks=checks),indent=2))
