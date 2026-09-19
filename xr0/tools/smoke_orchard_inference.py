"""Offline trained-checkpoint -> two real RGB views -> Orchard native command."""
import argparse
import json
from pathlib import Path
import numpy as np
from decord import VideoReader
from scipy.spatial.transform import Rotation
from mibot.server.orchard_policy import OrchardPolicy

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('weights','processor-path','stats','episode','output'):p.add_argument('--'+arg,required=True)
    args=p.parse_args()
    t=json.loads(Path(args.episode).read_text());frame=0
    obs=dict(tcp_pos_world=np.array(t['proprios']['ee_pos'][frame]),
             tcp_quat_world=Rotation.from_matrix(np.array(t['proprios']['ee_rotm'][frame]).reshape(3,3)).as_quat(),
             joint_pos=np.r_[t['proprios']['arm_joint'][frame],0.,0.],
             gripper_width=t['proprios']['gripper_pos'][frame][0])
    for name,key in (('rgb_static','ego'),('rgb_wrist','wrist_left')):
        obs[name]=VideoReader(t['observations'][key][0]['path'],num_threads=1)[frame].asnumpy()
    policy=OrchardPolicy(args.weights,args.processor_path,args.stats)
    z=policy.predict(obs)
    cmds=[policy.command(k,obs) for k in range(30)]
    assert z.shape==(30,32) and np.isfinite(z).all() and not z[:,7:].any()
    assert all(np.isfinite(c['action']).all() and np.isfinite(c['gripper_width']) for c in cmds)
    report=dict(status='PASS',checkpoint=policy.load_report,shape=list(z.shape),finite=True,
                executed_in_simulator=False,first_native_action=cmds[0]['action'].tolist(),
                first_gripper_width=cmds[0]['gripper_width'])
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
