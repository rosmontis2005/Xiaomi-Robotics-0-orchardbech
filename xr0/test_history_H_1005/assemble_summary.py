"""Assemble final evidence without re-evaluating or selecting checkpoints."""
import json,hashlib
from pathlib import Path
root=Path(__file__).resolve().parent;xr=root.parent;R03=xr/'test_recovery_R_1004/evaluation_followup/version_03'
def load(p):return json.loads(Path(p).read_text())
def offline(folder,step):
 p=folder/f'evaluations/step_{step:04d}/report.json'
 if not p.exists():return None
 r=load(p);groups=r['groups'];return dict(scope='fixed240 original windows; seeds42/43/44, not independent success samples',windows=r['windows'],sets=r['evaluation_sets'],groups={k:dict(position_mae_m=v['full']['position_error_m']['mean'],rotation_mae_rad=v['full']['rotation_error_rad']['mean'],width_mae_m=v['full']['width_error_m']['mean']) for k,v in groups.items() if k in ['old_train/all','old_val/all','new_val/all','val/reset','val/phase_GRASP','val/phase_PULL','val/phase_TRANSPORT']})
models={}
models['B8000']=dict(updates=8000,checkpoint=load(root/'protocol.json')['baseline'],cohorts=load(root/'historical_B8000_reference/aggregate.json'),cases=load(root/'historical_B8000_reference/cases.json'),scope='Historical frozen D0 matching dev8 and single auxiliary scene; not a new test')
models['R03-4k']=dict(updates=4000,checkpoint=dict(path=str(R03/'training/checkpoints/step_4000_trainable.pt'),sha256='5890b0d08f6e396299a47295f2e8b3987aafb477e7bfd571287d875525cb9575'),cohorts=load(root/'historical_R03_4k_reference/aggregate.json'),cases=load(root/'historical_R03_4k_reference/cases.json'),offline=offline(R03/'evaluation/offline',4000),late_offline=load(root/'r03_extension/evaluation/late_offline_R03_4k.json'),scope='Existing R03 4k, historical development results')
for name,folder in [('R03-8k','r03_extension'),('H0','H0'),('Hrepeat','Hrepeat'),('Hhistory','Hhistory'),('Hanchor','Hanchor')]:
 path=root/folder
 if not (path/'evaluation/aggregate.json').exists():
  if name=='Hanchor':continue
  raise RuntimeError(f'{name} incomplete; refuse final summary')
 training=load(path/'training/summary.json');step=training['total_actual_updates'];assert training['all_real_updates'] and training['status']['status']=='COMPLETE'
 assert load(path/'evaluation/status.json')['status']=='COMPLETE'
 ck=training['checkpoint'];ck['path']=str((root.parent.parent/ck['path']).resolve()) if not Path(ck['path']).is_absolute() else ck['path']
 # Prefer authoritative absolute checkpoint path from evaluator protocol.
 ck['path']=load(path/'evaluation/protocol.json')['checkpoint']
 models[name]=dict(updates=step,checkpoint=ck,training=training,cohorts=load(path/'evaluation/aggregate.json'),cases=load(path/'evaluation/cases.json'),offline=offline(path/'evaluation/offline',step),late_offline=load(path/'evaluation/late_offline.json'),scope='Development dev8 (train2+val6) and historical single auxiliary scene, not final generalization')
result=dict(status='COMPLETE',models=models,H_budget=load(root/'budget_decision.json'),protocol=load(root/'protocol.json'),causal_input_check=load(root/'causal_history_check.json'),batch_check=load(root/'batch_check.json'),deploy_smoke=load(root/'r03_extension/evaluation/history_inference_smoke.json'),engineering_repairs=[json.loads(l) for l in (root/'engineering_repairs.jsonl').read_text().splitlines()],optional_H4=load(root/'H4_decision.json'),fresh24_run=False,heldout12_reused_for_selection=False,final_generalization_conclusion='Not measured this round; development results only',execution_receipts={p.name:load(p) for p in root.glob('*_receipt.json')})
(root/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps({m:d['cohorts']['dev8'] for m,d in models.items()}))
