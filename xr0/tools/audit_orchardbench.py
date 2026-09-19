"""Reaudit every accepted asset, then test actual Dataset and deployment round-trip."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation
from treesim.orchard_action import OrchardActionAdapter, decode_targets, encode_window, EPS
from mibot.data.datasets.orchardbench_dataset import OrchardBenchDataset, load_stats, read_episode


def audit(args):
    # Read-only collector validation is the source of truth for acceptance provenance.
    sys.path.insert(0, str(Path(args.orchard_repo) / 'scripts'))
    from collect_autopicker_dataset import validate_arrays, validate_videos
    root = Path(args.root)
    rows = [json.loads(x) for x in (root / 'manifest.jsonl').read_text().splitlines()]
    accepted = [r for r in rows if r['accepted']]
    assert len({r['seed'] for r in accepted}) == len(accepted)
    assert {str(root / r['annotation']) for r in accepted} == {str(p) for p in (root / 'json').glob('*/*.json')}
    counts = dict(Counter(r['split'] for r in accepted))
    stats, mean, std = load_stats(args.stats)
    report = dict(accepted=len(accepted), splits=counts, expected_300_270_30=(len(accepted),counts.get('train'),counts.get('val')) == (300,270,30),
                  videos=[], weak_visibility=[], roundtrip_max=[0., 0., 0.], controller_limit_exceedances=0)
    rng = np.random.default_rng(42)
    for i, row in enumerate(accepted):
        t = read_episode(str(root / row['annotation']))
        validate_arrays(t)
        videos = validate_videos(t)
        report['videos'].append(dict(episode=t['episode_id'], frames=t['num_frames'], views=videos))
        vis = t['orchardbench']['target_visibility']['initial']
        if vis['static_visible_pixels'] <= 2 and vis['wrist_visible_pixels'] == 0:
            report['weak_visibility'].append(t['episode_id'])
        n = t['num_frames']
        for frame in sorted({0, (n-30)//2, n-30, *rng.integers(0,n-29,size=3).tolist()}):
            raw = encode_window(t, frame)
            z = (raw - mean) / (std + EPS)
            obs = dict(tcp_pos_world=np.array(t['proprios']['ee_pos'][frame]),
                       tcp_quat_world=Rotation.from_matrix(np.array(t['proprios']['ee_rotm'][frame]).reshape(3,3)).as_quat(),
                       joint_pos=np.r_[t['proprios']['arm_joint'][frame],0.,0.],
                       gripper_width=t['proprios']['gripper_pos'][frame][0])
            adapter = OrchardActionAdapter()
            adapter.set_chunk(z, mean, std, obs)
            assert np.array_equal(adapter.state(obs)[0,7:14],np.asarray(t['proprios']['arm_joint'][frame],dtype=np.float32))
            for k in range(30):
                # Execute relative to the measured state at EACH future step, with
                # the original chunk anchor fixed. No sequential-delta shortcut.
                j=frame+k
                current=np.array(t['proprios']['ee_pos'][j])
                rcur=np.array(t['proprios']['ee_rotm'][j]).reshape(3,3)
                now=dict(tcp_pos_world=current,tcp_quat_world=Rotation.from_matrix(rcur).as_quat())
                cmd=adapter.to_native(k,now)
                a=cmd['action']
                recovered=current+a[:3]
                recovered_r=Rotation.from_euler('xyz',a[3:6]).as_matrix() @ rcur
                expected_r=np.array(t['actions']['ee_rotm'][j]).reshape(3,3)
                error=[np.linalg.norm(recovered-t['actions']['ee_pos'][j]),
                       Rotation.from_matrix(recovered_r.T @ expected_r).magnitude(),
                       abs(cmd['gripper_width']-t['actions']['gripper_pos'][j][0])]
                report['roundtrip_max']=np.maximum(report['roundtrip_max'],error).tolist()
                report['controller_limit_exceedances']+=int(np.any(abs(a[:3])>.02) or np.any(abs(a[3:6])>.05))
        if (i+1)%50==0:print('Audited',i+1,flush=True)
    assert np.all(np.array(report['roundtrip_max']) < [1e-5,1e-5,1e-6])
    for split in ('train','val'):
        ds=OrchardBenchDataset({'train_datasets':dict(root=str(root),split=split,stats_path=args.stats)})
        for i in rng.integers(0,len(ds),size=8):
            sample=ds[int(i)]
            assert set(sample)=={'messages','action','action_mask','state'}
            assert tuple(sample['action'].shape)==tuple(sample['action_mask'].shape)==(30,32)
            assert tuple(sample['state'].shape)==(1,32)
            assert all(np.isfinite(sample[k].numpy()).all() for k in ('action','action_mask','state'))
            assert not sample['action'][:,7:].any() and not sample['action_mask'][:,7:].any()
            assert sample['action_mask'][:,:7].all() and not sample['state'][:,14:].any()
            assert sum(c['type']=='image' for m in sample['messages'] for c in m['content'])==2
        # Every window passes finite/inactive checks without expensive video IO.
        for i in range(len(ds)):
            a=ds.raw_action(i)
            assert np.isfinite((a-mean)/(std+EPS)).all() and not a[:,7:].any()
        report[split+'_windows']=len(ds)
    assert stats['full_window_samples']==report['train_windows']
    report.update(status='PASS', random_decoded_samples=16, normalization_shape=list(mean.shape),
                  total_frames=sum(x['frames'] for x in report['videos']),
                  total_decoded_video_frames=sum(sum(v['decoded_frames'] for v in x['views'].values()) for x in report['videos']))
    Path(args.output).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('videos','weak_visibility')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True);p.add_argument('--stats',required=True)
    p.add_argument('--orchard-repo',required=True);p.add_argument('--output',required=True)
    audit(p.parse_args())
