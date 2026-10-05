"""R03-8k live takeover. Truth is confined to mining, teacher and diagnostics.
Every physical control step (both sides of boundary) calls production env.step.
No reset/replay/restore after takeover. Failed attempts remain recorded.
"""
import bootstrap
import argparse, fcntl, json, os, time, traceback
from collections import Counter
import numpy as np
from scipy.spatial.transform import Rotation
from recovery_common import *
from recovery_teacher import LiveRecoveryTeacher
from collect_recovery import DIMS, measured, healthy
from treesim.orchard_action import OrchardActionAdapter

I=PROTOCOL['integrated']; CHECKPOINT=Path(I['mining_checkpoint']); CODE_SHA256=sha(Path(__file__))

def record(obs, states, images, times):
    values=dict(ee_pos=obs['tcp_pos_world'],ee_rotm=Rotation.from_quat(obs['tcp_quat_world']).as_matrix().ravel(),arm_joint=obs['joint_pos'][:7],arm_joint_vel=obs['joint_vel'][:7],gripper_pos=[obs['gripper_width']])
    for key,value in values.items(): states[key].append(np.asarray(value).tolist())
    for j,key in enumerate(['rgb_static','rgb_wrist']): images[j].append(obs[key].copy())
    times.append(float(obs['sim_time']))

def save_record(folder, states, images, times):
    encoder=module(ORCHARD/'scripts/collect_autopicker_dataset.py','i_encoder')
    for j,key in enumerate(['ego','wrist_left']): encoder.encode(folder/f'{key}.mp4',images[j])
    write(folder/'proprio.json',dict(proprios=states,timestamps=times,observations={key:str(folder/f'{key}.mp4') for key in ['ego','wrist_left']}))

def geometry(env,t):
    cp,cr=env._chassis_pose();local=cr.inv().apply(np.asarray(t['fruit_pose'])[:3]-cp)
    from treesim import robot
    return bool(t['fruit_bucket_distance']<=.15 and abs(local[0]-robot._BUCKET_CENTER_X)<robot._BUCKET_HALF and abs(local[1])<robot._BUCKET_HALF)

def teacher_live(env,obs,strict,candidate,folder):
    """Existing expert TRANSPORT IK goal, expressed through native production control.
    Inherited expert initialization is read-only. Never call picker.update, hold,
    release, _slew_arm or assign physical/production controller state.
    """
    source=json.loads(Path(candidate['scene']['annotation']).read_text())
    before=physics_digest(env)
    profile=dict(source['orchardbench']['arm_motion_profile'])
    if profile['active_phases'] is not None:profile['active_phases']=tuple(profile['active_phases'])
    picker=LiveRecoveryTeacher(env,'R3',profile)
    assert physics_digest(env)==before and picker.initial_state=='TRANSPORT'
    from treesim import robot
    cp,cr=env._chassis_pose()
    over=cp+cr.apply([robot._BUCKET_CENTER_X,0.,robot._CHASSIS_Z+robot._CHASSIS[2]+robot._BUCKET_WALL_H+.11])
    goal,_=picker._ik_to(over)
    import newton
    # Follow the existing expert motion limiter in its PRIVATE joint command
    # bookkeeping. Only FK TCP targets are passed to the production controller.
    def target_pose():
        for _ in range(env.config.action_repeat):
            picker._q_cmd[:7]=picker.arm_motion.step(goal[:7],legacy_step=.045,active=True,vmax_rad_s=picker.arm_motion.profile.phase_vmax_rad_s.get('TRANSPORT') if picker.arm_motion.profile.phase_vmax_rad_s else None)
        measured_q=env._obs['joint_pos'].astype(float)
        # Keep every target near the real measured arm; an open-loop private
        # command plan can outrun native pose IK and become clipped/unreachable.
        proposal=picker._q_cmd.copy()
        proposal[:7]=measured_q[:7]+np.clip(proposal[:7]-measured_q[:7],-.012,.012)
        for scale in [1.,.5,.25,.125,.0625]:
            planned=measured_q+scale*(proposal-measured_q)
            q=picker.ik.model.joint_q.numpy();q[picker.ik._arm_q]=planned
            picker.ik.model.joint_q.assign(q)
            newton.eval_fk(picker.ik.model,picker.ik.model.joint_q,picker.ik.model.joint_qd,picker.ik._state)
            pose=picker.ik._state.body_q.numpy()[picker.ik.tcp]
            position=cp+cr.apply(pose[:3]);rotation=(cr*Rotation.from_quat(pose[3:])).as_matrix()
            delta_rotation=(Rotation.from_matrix(rotation)*Rotation.from_quat(env._obs['tcp_quat_world']).inv()).as_euler('xyz')
            if np.max(np.abs(position-env._obs['tcp_pos_world']))<env.config.max_translation*.8 and np.max(np.abs(delta_rotation))<env.config.max_rotation*.8:break
        return position,rotation
    # FK here belongs to the expert's private IK model, never the live simulator.
    pose=picker.ik._state.body_q.numpy()[picker.ik.tcp]
    target_position=cp+cr.apply(pose[:3]);target_rotation=(cr*Rotation.from_quat(pose[3:])).as_matrix()
    adapter=OrchardActionAdapter();adapter.targets=(np.repeat(target_position[None],30,0),np.repeat(target_rotation[None],30,0),np.zeros(30))
    states={k:[] for k in DIMS};images=[[],[]];times=[];phases=[];rows=[]
    record(obs,states,images,times);phases.append('TRANSPORT');phase='TRANSPORT';settled=0;reason='teacher_budget';base=np.asarray(candidate['truth']['base_pose']);fruit=candidate['truth']['fruit_id']
    started=time.monotonic();release_region=False
    for step in range(candidate['step']+1,901):
        pre=truth(env)
        target_position,target_rotation=target_pose()
        adapter.targets=(np.repeat(target_position[None],30,0),np.repeat(target_rotation[None],30,0),np.full(30,.08 if phase=='DROP' else 0.))
        if phase=='TRANSPORT':
            ready=geometry(env,pre) and pre['fruit_bucket_distance']<=.13 and np.linalg.norm(np.asarray(pre['fruit_velocity'])[:3])<.35 and np.linalg.norm(obs['tcp_pos_world']-target_position)<.07
            settled=settled+1 if ready else 0
            if settled>=15: phase='DROP';adapter.targets[2][:]=.08
        command=adapter.to_native(0,obs)
        obs,_,_,truncated,info=env.step(**command)
        record(obs,states,images,times);phases.append(phase)
        outcome=update_strict(env,strict,step);t=truth(env);bad=healthy(env,base,fruit)
        if env._held is None and strict.release_events and strict.release_events[-1]['step']==step:
            release_region=geometry(env,t)
        rows.append(dict(control_step=step,local_frame=len(times)-1,phase=phase,command=command,truth=t,strict=outcome,failure=bad,ik_failed=info['ik_failed']))
        if bad: reason=bad;break
        if outcome['strict_success']: reason='strict_success' if release_region else 'strict_but_unreasonable_release';break
        if phase=='TRANSPORT' and env._held!=fruit:reason='lost_hold_before_release';break
        if truncated:break
    success=reason=='strict_success' and len(times)>=31
    save_record(folder,states,images,times)
    trajectory=dict(schema='orchard_xr0_recovery_R_v1',trajectory_type='recovery_success' if success else 'recovery_rejected',episode_id=candidate['candidate_id'],split='train',seed=candidate['scene_seed'],num_frames=len(times),record_fps=30,proprios=states,actions={k:states[k][1:]+states[k][-1:] for k in DIMS if k!='arm_joint_vel'},observations={key:[dict(path=str(folder/f'{key}.mp4'))] for key in ['ego','wrist_left']},recovery=dict(debug_usage='NOT POLICY INPUT',recovery_category=candidate['category'],source_episode_id=candidate['source_episode_id'],collection_code_sha256=CODE_SHA256,student_checkpoint=str(CHECKPOINT),student_checkpoint_sha256=I['mining_sha256'],student_inference_seed=candidate['inference_seed'],student_control_step=candidate['step'],student_prefix=str(Path(candidate['source_folder'])/'proprio.json'),student_steps=str(Path(candidate['source_folder'])/'steps.json'),takeover=candidate['truth'],teacher_initial_state=picker.initial_state,initialization_audit=picker.initialization_audit,teacher_final_result=reason,same_live_env=True,prefix_regeneration_required=False,execution='production OrchardActionAdapter.to_native -> env.step'),orchardbench=dict(debug_usage='NOT POLICY INPUT',timestamps=times,phase_by_frame=phases))
    result=dict(status='PASS' if success else 'REJECT',reason=reason,strict=strict.summary(),reasonable_release=release_region,frames=len(times),wall_seconds=time.monotonic()-started,same_live_env=True,initialization_audit=picker.initialization_audit)
    write(folder/'teacher_steps.json',rows);write(folder/'teacher_result.json',result);write(folder/'trajectory.json',trajectory)
    return trajectory,result

def mine(policy,scene,seed,stall_after):
    run_id=f"{scene['episode_id']}_rng{seed}"
    folder=ROOT/'student_sources'/run_id
    attempt=0
    while (folder/'error.json').exists():
        attempt+=1;folder=ROOT/'student_sources'/f'{run_id}_retry{attempt}'
    run_id=folder.name;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists(): return
    env=environment();states={k:[] for k in DIMS};images=[[],[]];times=[];rows=[];chunks=[];distances=[];strict=StrictPlacement();heldrun=0;previous=None;reason='student_budget';candidate=None
    try:
        obs,info=env.reset(seed=scene['seed']);base=env.sim.body_q_np()[env.chassis].copy();record(obs,states,images,times)
        write(folder/'initial.json',dict(scene=scene,inference_seed=seed,truth=truth(env),physics_digest=physics_digest(env)))
        k=30;dwell=0;chunk=-1
        for step in range(1,901):
            if k==30:
                normalized=policy.predict(obs,seed=seed);p,r,w=policy.adapter.targets;k=0;dwell=0;chunk+=1
                chunks.append(dict(chunk=chunk,at_step=step-1,normalized=normalized,position=p,rotation=r,width=w))
            pre=truth(env);fruit=pre['fruit_id'];category=None
            eligible=env._held==fruit and heldrun>=15 and pre['detached'][fruit]
            if eligible:
                if pre['fruit_bucket_distance']>.15 and w[k]>=env.config.gripper_open_width and env._width_open_steps>=3:
                    category='premature_release_precursor'
                elif pre['fruit_bucket_distance']<=.30 and heldrun>=60 and (step>=750 or (len(distances)>=61 and min(distances[-61:-1])-distances[-1]<.01)):
                    category='near_bucket_no_release'
                elif step-1>=stall_after and heldrun>=90 and len(distances)>=61 and pre['fruit_bucket_distance']>.30 and min(distances[-61:-1])-distances[-1]<.03:
                    category='transport_stall'
            if category:
                cid=f'{run_id}_{category}_s{step-1:04d}'
                candidate=dict(candidate_id=cid,category=category,step=step-1,chunk=chunk,k=k,scene=scene,scene_seed=scene['seed'],source_episode_id=scene['episode_id'],source_folder=str(folder),inference_seed=seed,student_checkpoint_sha256=I['mining_sha256'],truth=pre)
                reason='live_takeover';break
            command=policy.adapter.to_native(k,obs);obs,_,_,truncated,info=env.step(**command);dwell+=1
            t=truth(env);record(obs,states,images,times);outcome=update_strict(env,strict,step)
            heldrun=heldrun+1 if env._held is not None and env._held==previous else 1 if env._held is not None else 0;previous=env._held;distances.append(t['fruit_bucket_distance'])
            adv,pe,re=advance(obs,p,r,k,dwell);bad=healthy(env,base,fruit)
            rows.append(dict(step=step,chunk=chunk,k=k,dwell=dwell,advance=adv,command=command,truth=t,held_run=heldrun,strict=outcome,physical_failure=bad))
            if adv:k+=1;dwell=0
            if bad:reason=bad;break
            if outcome['strict_success']:reason='student_strict_success';break
            if truncated:break
        write(folder/'steps.json',rows);write(folder/'chunks.json',chunks);save_record(folder,states,images,times)
        prefix_digest=physics_digest(env)
        if candidate:
            recovery=ROOT/'recoveries'/candidate['candidate_id'];recovery.mkdir(parents=True,exist_ok=True);write(recovery/'candidate.json',candidate)
            assert physics_digest(env)==prefix_digest,'Recording mutated live physics'
            traj,result=teacher_live(env,obs,strict,candidate,recovery)
            decision=dict(candidate_id=candidate['candidate_id'],category=candidate['category'],source_episode_id=scene['episode_id'],inference_seed=seed,student_control_step=candidate['step'],accepted=result['status']=='PASS',teacher_status=result['status'],reason=result['reason'],annotation=str(recovery/'trajectory.json'),annotation_sha256=sha(recovery/'trajectory.json'),num_frames=traj['num_frames'],same_live_env=True,collection_code_sha256=CODE_SHA256,student_checkpoint_sha256=I['mining_sha256'],student_prefix_sha256=sha(folder/'proprio.json'),student_steps_sha256=sha(folder/'steps.json'),teacher_result=result)
            write(recovery/'decision.json',decision);append(ROOT/'candidate_decisions.jsonl',decision);print('DECISION',json.dumps({k:decision[k] for k in ['candidate_id','accepted','reason']}),flush=True)
        write(folder/'complete.json',dict(scene=scene,inference_seed=seed,student_sha256=I['mining_sha256'],student_steps=len(rows),termination=reason,candidate=None if candidate is None else candidate['candidate_id'],prefix_end_physics_digest=prefix_digest))
    except Exception as e:
        if candidate:
            recovery=ROOT/'recoveries'/candidate['candidate_id'];decision=dict(candidate_id=candidate['candidate_id'],category=candidate['category'],source_episode_id=scene['episode_id'],accepted=False,reason='engineering_exception',error=str(e),traceback=traceback.format_exc());write(recovery/'decision.json',decision);append(ROOT/'candidate_decisions.jsonl',decision)
        write(folder/'error.json',dict(error=str(e),traceback=traceback.format_exc()));raise
    finally:env.close()

def summary():
    decisions=[json.loads(p.read_text()) for p in sorted((ROOT/'recoveries').glob('*/decision.json'))];accepted=[r for r in decisions if r['accepted']];completed=list((ROOT/'student_sources').glob('*/complete.json'))
    result=dict(source_rollouts=len(completed),candidate=len(decisions),accepted=len(accepted),rejected=len(decisions)-len(accepted),unique_source_episodes=len({r['source_episode_id'] for r in accepted}),accepted_categories=dict(Counter(r['category'] for r in accepted)),candidate_categories=dict(Counter(r['category'] for r in decisions)),rejected_reasons=dict(Counter(r['reason'] for r in decisions if not r['accepted'])),teacher_strict_success=sum(r.get('teacher_result',{}).get('strict',{}).get('strict_success',False) for r in decisions),mining_checkpoint=str(CHECKPOINT),mining_sha256=I['mining_sha256'],time=time.time(),ready_to_freeze=len(accepted)>=I['minimum_accepted'] and len({r['source_episode_id'] for r in accepted})>=I['minimum_unique_sources'])
    write(ROOT/'collection_summary.json',result);return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    protection_check();assert sha(CHECKPOINT)==I['mining_sha256']
    plan_path=ROOT/'collection_plan.json'
    plan=[dict(scene=scene,seed=seed,stall_after=I['stall_min_steps'][i]) for scene in PROTOCOL['source_episodes'][:I['max_source_episodes']] for i,seed in enumerate(I['mining_inference_seeds'])]
    if plan_path.exists():assert json.loads(plan_path.read_text())==plan
    else:write(plan_path,plan)
    lock=(XR0/'test_recovery_R_1004/.gpu_training.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    import torch
    from mibot.server.orchard_policy import OrchardPolicy
    from checkpoint_io import load_trainable_overlay
    torch.set_num_threads(1);policy=OrchardPolicy(str(BASE),str(PROCESSOR),str(STATS));report=load_trainable_overlay(policy.model,CHECKPOINT,base_checkpoint=BASE);assert report['checkpoint_sha256']==I['mining_sha256'];write(ROOT/'mining_load_manifest.json',report)
    for job in plan[:1] if args.smoke else plan:
        if summary()['accepted']>=I['target_accepted'] and summary()['unique_source_episodes']>=I['minimum_unique_sources']:break
        mine(policy,job['scene'],job['seed'],job['stall_after']);print('COLLECTION',json.dumps(summary()),flush=True)
    protection_check();print('FINISHED',json.dumps(summary()),flush=True)
if __name__=='__main__':main()
