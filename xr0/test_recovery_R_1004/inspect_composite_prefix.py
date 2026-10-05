import bootstrap
import json,os
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from recovery_common import *
from treesim.orchard_action import OrchardActionAdapter
OUT=ROOT/'evaluation_followup/onpolicy_R3';results=[]
for file in sorted((OUT/'student_sources').glob('*/candidate.json')):
 c=json.loads(file.read_text());folder=Path(c['source_folder']);meta=json.loads((folder/'complete.json').read_text());chunks=json.loads((folder/'chunks.json').read_text());rows=json.loads((folder/'steps.json').read_text());env=environment();obs,info=env.reset(seed=c['scene_seed']);adapter=OrchardActionAdapter();loaded=-1
 try:
  for r in rows:
   if r['chunk']!=loaded:
    x=chunks[r['chunk']];adapter.targets=(np.asarray(x['position']),np.asarray(x['rotation']),np.asarray(x['width']));loaded=r['chunk']
   obs,_,_,_,_=env.step(**adapter.to_native(r['k'],obs))
  actual=truth(env);expected=c['truth'];difference={k:float(np.max(np.abs(np.asarray(actual[k])-expected[k]))) for k in ['joint_pos','tcp_position','tcp_quaternion','width','fruit_pose','base_pose','palm_local_fruit','fruit_velocity']};q=np.abs(np.asarray(actual['joint_pos'])-expected['joint_pos']);difference['per_joint_pos_abs']=q.tolist();difference['tcp_rotation_geodesic_rad']=float((Rotation.from_quat(actual['tcp_quaternion'])*Rotation.from_quat(expected['tcp_quaternion']).inv()).magnitude());record=dict(candidate=c['candidate_id'],actual=actual,expected=expected,difference=difference);write(file.parent/'prefix_diagnostic.json',record);results.append(record);print(json.dumps(dict(candidate=c['candidate_id'],difference=difference)),flush=True)
 finally:env.close()
write(OUT/'prefix_consistency_audit.json',results)
