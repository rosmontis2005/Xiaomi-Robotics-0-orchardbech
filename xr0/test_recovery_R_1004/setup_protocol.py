import json,hashlib
from pathlib import Path
X=Path('/home/rosmontis/Projects/dualsys/Xiaomi-Robotics-0/xr0'); R=X/'test_recovery_R_1004'; A=X/'test_128_episode_AB_1003';O=Path('/home/rosmontis/Projects/orchardbench')
s=json.loads((A/'selection.json').read_text());m=json.loads((X/'test_small_set_fit_1003/selection.json').read_text()); f=json.loads((X/'test_grasp_1004/data/fresh24_heldout.json').read_text())
protected=s['development_scenes']+s['heldout_scenes']+s['episodes_new_val']+s['episodes_heldout']+f['scenes']+[e for e in m['episodes'] if e['split']=='val']
ids={e['episode_id'] for e in protected};seeds={e['seed'] for e in protected}
manifest=[json.loads(l) for l in (O/'data/orchard_v1_2650/filtered/manifest.jsonl').read_text().splitlines()]
ids.update(e['episode_id'] for e in manifest if e['split']!='train');seeds.update(e['seed'] for e in manifest if e['split']!='train')
ids.update(s['selection_rule']['excluded_previous_val_ids'])
source=[{k:e[k] for k in ('episode_id','seed','split','annotation','annotation_sha256')} for e in s['episodes_train'] if e['episode_id'] not in ids and e['seed'] not in seeds]
assert all(e['split']=='train' for e in source)
files=[O/p for p in ['scripts/collect_autopicker_dataset.py','treesim/fixed_base_picker.py','treesim/picker.py','treesim/orchard_action.py','treesim/vla_env.py','treesim/fruit.py']]+[X/'mibot/data/datasets/orchardbench_dataset.py']+[A/p for p in ['train_ab.py','config_ab.json','checkpoint_io.py','prepare_data.py','selection.json']]
hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
protocol=dict(schema='orchard_student_state_recovery_R_v1',source_episodes=source,protected_episode_ids=sorted(ids),protected_scene_seeds=sorted(seeds),baseline=json.loads((R/'baseline_provenance.json').read_text()),production_sha256=hashes,student=dict(targets_per_chunk=30,position_tolerance_m=.01,rotation_tolerance_rad=.08,max_dwell=30,budget=900,inference_seeds=[42,43],grasp_mode='benchmark_assist',detach_force_scale=1.5),pilot_targets=dict(R1=12,R2=6,R3=6),mechanism_pilot_min_each=2,teacher_budget_control_steps=900,replay_total_episode_budget=900,acceptance='same planned fruit held15 -> detached -> release -> unheld in strict bucket60; no breaks/incidental detach/base drift/numerical failure; all accepted trajectories oracle replay PASS',anchor_rule='0..min(29,num_frames-30), frame0 included; full30 next measured-state labels',debug_usage='NOT POLICY INPUT',mining_rules=dict(R1='no held; planned apple attached; TCP distance <=0.09m and closing intent or finger contact; best contact/geometry candidate per episode',R2='same planned apple held >=15 boundaries, detached preferred, >0.25m bucket distance; earliest stable transport boundary',R3='same detached fruit held >=15; predicted imminent sustained open away from bucket OR >=350 steps with progress over previous60 steps <0.03m; never duplicate R2 step'),bounded_source_rule='deterministic original B cohort order, protected scenes removed; initial4 then at most36 seed42 scenes; seed43 only for missing categories; max2 candidates/category/source for retries, no quota-driven gate changes')
(R/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
for name in ['bootstrap.py','checkpoint_io.py']:(R/name).write_text((A/name).read_text())
print(json.dumps(dict(eligible_sources=len(source),excluded_train=len(s['episodes_train'])-len(source),first_sources=source[:4]),indent=2))
t=json.loads(Path(source[0]['annotation']).read_text());print('motion',t['orchardbench'].get('arm_motion_profile'))
