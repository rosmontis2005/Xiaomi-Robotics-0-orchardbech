"""48 fixed teacher-continuation windows: training-source mechanism, not generalization."""
import bootstrap,json
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from recovery_common import XR0
from recovery_dataset import RecoveryDataset
from history_inputs import CausalHistory
from treesim.orchard_action import encode_window,decode_targets

def evaluate_late(policy,mode,output):
 output=Path(output)
 if output.exists():return json.loads(output.read_text())
 manifest=XR0/'test_recovery_R_1004/evaluation_followup/version_03/recovery_manifest.jsonl';dataset=RecoveryDataset(manifest);history=CausalHistory();rows=[];phase_values=defaultdict(list)
 with torch.inference_mode():
  for row in dataset.rows:
   t=dataset.episodes[row['annotation']]
   for group,f in [('TRANSPORT',row['teacher_continuation_anchors_by_group']['transport_to_drop'][-1]),('DROP',row['teacher_continuation_anchors_by_group']['DROP_transition'][0])]:
    sample=dataset.from_trajectory(t,f);sample['state']=torch.from_numpy(history.states(row['annotation'],f,mode));batch=policy.collate_late([sample]);gt=encode_window(t,f);p=t['proprios'];anchor_pos=p['ee_pos'][f];anchor_rot=np.asarray(p['ee_rotm'][f]).reshape(3,3);gp,gr,gw=decode_targets(gt,anchor_pos,anchor_rot);values=[]
    for seed in [42,43,44]:
     data={k:v.to(policy.device) for k,v in batch.items()};data['action']=torch.zeros_like(data['action'])
     with torch.random.fork_rng(devices=[0]):
      torch.manual_seed(seed)
      with torch.autocast('cuda',dtype=torch.bfloat16):z=policy.model.generate(data)[0].float().cpu().numpy()
     physical=z*(dataset.std+1e-6)+dataset.mean;physical[:,7:]=0;pp,pr,pw=decode_targets(physical,anchor_pos,anchor_rot)
     err=np.stack([np.linalg.norm(pp-gp,axis=1),Rotation.from_matrix(gr.transpose(0,2,1)@pr).magnitude(),np.abs(pw-gw)],axis=1);values.append(err)
     for i,phase in enumerate(t['orchardbench']['phase_by_frame'][f+1:f+31]):phase_values[phase].append(err[i])
    arr=np.stack(values);rows.append(dict(candidate_id=row['candidate_id'],source_episode=row['source_episode_id'],origin_category=row['category'],anchor_frame=f,anchor_phase=group,position_mae_m=float(arr[:,:,0].mean()),rotation_mae_rad=float(arr[:,:,1].mean()),width_mae_m=float(arr[:,:,2].mean())))
 result=dict(scope='TRAINING-SOURCE teacher-continuation mechanism diagnostic; no generalization claim',windows=len(rows),seeds=[42,43,44],rows=rows,by_target_phase={phase:dict(target_seed_count=len(v),position_mae_m=float(np.array(v)[:,0].mean()),rotation_mae_rad=float(np.array(v)[:,1].mean()),width_mae_m=float(np.array(v)[:,2].mean())) for phase,v in phase_values.items()})
 output.write_text(json.dumps(result,indent=2));return result
