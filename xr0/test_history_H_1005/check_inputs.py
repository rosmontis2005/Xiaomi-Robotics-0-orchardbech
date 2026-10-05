"""Only checks needed to avoid an invalid causal-history experiment."""
import bootstrap,json
from pathlib import Path
import numpy as np
import torch
from history_inputs import CausalHistory,HistoryBuffer,proprio_at,OFFSETS
from recovery_common import ROOT,XR0,write,sha
from scipy.spatial.transform import Rotation
R03=XR0/'test_recovery_R_1004/evaluation_followup/version_03'
manifest=[json.loads(l) for l in (R03/'recovery_manifest.jsonl').read_text().splitlines()]
history=CausalHistory();examples=[]
for row in manifest:
 path=row['annotation'];t=history.trajectory(path);prefix,takeover=history.prefix(t)
 for f in [0,1,min(16,t['num_frames']-30)]:
  states=history.states(path,f,'Hhistory')
  expected=np.stack([proprio_at(t,f+d) if f+d>=0 else prefix[max(0,takeover+f+d)] for d in OFFSETS])
  assert np.array_equal(states,expected)
  assert np.array_equal(history.states(path,f,'Hrepeat'),np.repeat(proprio_at(t,f)[None],4,axis=0))
  examples.append(dict(episode=t['episode_id'],frame=f,global_anchor=takeover+f,global_indices=[max(0,takeover+f+d) for d in OFFSETS],crosses_prefix=bool(f<15)))
# One ordinary B start: causal padding and deployment parity over 20 actual observations.
row=json.loads((R03/'schedule_R.jsonl').read_text().splitlines()[0]);path=row['annotation']
if row['source']!='original_B':path=next(json.loads(l)['annotation'] for l in (R03/'schedule_R.jsonl').read_text().splitlines() if json.loads(l)['source']=='original_B')
t=history.trajectory(path);buf=HistoryBuffer('Hhistory')
for f in range(20):
 p=t['proprios'];obs=dict(tcp_pos_world=np.asarray(p['ee_pos'][f]),tcp_quat_world=Rotation.from_matrix(np.asarray(p['ee_rotm'][f]).reshape(3,3)).as_quat(),gripper_width=float(p['gripper_pos'][f][0]),joint_pos=np.asarray(p['arm_joint'][f]))
 buf.observe(obs,f);assert np.allclose(buf.state(),history.states(path,f,'Hhistory'),atol=2e-6,rtol=0)
report=dict(status='PASS',recovery_trajectories=len(manifest),excluded_anchors=0,offsets=OFFSETS,padding='earliest actual observation',prefix='recorded student measured proprio; teacher samples only at local index>=0',prefix_audit=history.prefix_audit,causal_examples=examples,ordinary_B_padding_and_deployment_parity=True,no_model_input_fields_beyond_proprio=True)
write(ROOT/'causal_history_check.json',report);print(json.dumps({k:report[k] for k in ['status','recovery_trajectories','excluded_anchors','ordinary_B_padding_and_deployment_parity']}))
