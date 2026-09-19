"""Quantization round-trip and existing controller limit compatibility audit."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from treesim.orchard_action import decode_targets, encode_window, EPS
from mibot.data.datasets.orchardbench_dataset import load_stats, episode_files, read_episode


def main(args):
    _,mean,std=load_stats(args.stats)
    errors=np.zeros(3);deltas=[];widths=[];windows=0
    for split in ('train','val'):
        for path in episode_files(args.root,split):
            t=read_episode(path);p,a=t['proprios'],t['actions']
            r=np.array(p['ee_rotm']).reshape(-1,3,3)
            rt=np.array(a['ee_rotm']).reshape(-1,3,3)
            delta=Rotation.from_matrix(rt@r.transpose(0,2,1)).as_euler('xyz')
            deltas.append(np.c_[np.array(a['ee_pos'])-p['ee_pos'],delta]);widths.extend(np.array(a['gripper_pos'])[:,0])
            if split!='train':continue
            frame=(t['num_frames']-30)//2;raw=encode_window(t,frame)
            z=torch.from_numpy((raw-mean)/(std+EPS)).bfloat16().float().numpy()
            pos,rot,width=decode_targets(z*(std+EPS)+mean,p['ee_pos'][frame],p['ee_rotm'][frame])
            sl=slice(frame,frame+30)
            error=[np.linalg.norm(pos-np.array(a['ee_pos'][sl]),axis=1).max(),
                   Rotation.from_matrix(rot.transpose(0,2,1)@rt[sl]).magnitude().max(),
                   np.max(abs(width-np.array(a['gripper_pos'][sl])[:,0]))]
            errors=np.maximum(errors,error);windows+=1
    d=np.concatenate(deltas);w=np.array(widths)
    quant=dict(normalized_output_dtype='bf16',episodes=windows,windows=windows,
               roundtrip_max=errors.tolist(),tolerances=[.005,.02,.001])
    assert np.all(errors<quant['tolerances'])
    limits=dict(steps=len(d),translation_exceeds=int(np.any(abs(d[:,:3])>.02,axis=1).sum()),
                rotation_exceeds=int(np.any(abs(d[:,3:])>.05,axis=1).sum()),
                either_exceeds=int((np.any(abs(d[:,:3])>.02,axis=1)|np.any(abs(d[:,3:])>.05,axis=1)).sum()),
                max_abs_per_axis=np.max(abs(d),axis=0).tolist(),gripper_min=float(w.min()),gripper_max=float(w.max()),
                gripper_out_of_bounds=int(((w<0)|(w>.08)).sum()))
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    for name,value in [('bf16_roundtrip',quant),('controller_limits',limits)]:
        (out/(name+'.json')).write_text(json.dumps(value,indent=2)+'\n');print(name,json.dumps(value))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True);p.add_argument('--stats',required=True);p.add_argument('--output-dir',required=True)
    main(p.parse_args())
