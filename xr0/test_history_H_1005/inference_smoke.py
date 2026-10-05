"""30-control-step input-path smoke, excluded from experiment metrics."""
import numpy as np
from history_inputs import HistoryBuffer

def rollout_smoke(policy,env_class,config,scene):
 original_mode=policy.history.mode;policy.history=HistoryBuffer('Hhistory');env=env_class(config)
 try:
  obs,info=env.reset(seed=scene['seed']);policy.reset_history(obs);prediction=policy.predict(obs,seed=42);assert prediction.shape==(30,32) and np.isfinite(prediction).all()
  k=0;dwell=0
  from scipy.spatial.transform import Rotation
  for step in range(1,31):
   obs,_,_,_,info=env.step(**policy.adapter.to_native(k,obs));policy.observe_control_step(obs,step);dwell+=1;p,r,w=policy.adapter.targets
   pe=np.linalg.norm(p[k]-obs['tcp_pos_world']);re=(Rotation.from_matrix(r[k])*Rotation.from_quat(obs['tcp_quat_world']).inv()).magnitude()
   if (pe<=.01 and re<=.08) or dwell>=30:k+=1;dwell=0
  prediction=policy.predict(obs,seed=42);assert np.isfinite(prediction).all()
  assert policy.history_calls[-1]['token_control_steps']==[15,25,29,30]
  return dict(status='PASS',scope='Input-path smoke only; no experiment success inference',control_steps=30,mode='Hhistory',state_shape=[1,4,32],actions_finite=True,calls=policy.history_calls,buffer_updates=policy.history.updates)
 finally:env.close();policy.history=HistoryBuffer(original_mode)
