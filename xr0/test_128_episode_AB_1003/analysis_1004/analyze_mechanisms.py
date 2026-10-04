#!/usr/bin/env python3
"""CPU-only paired dev8 failure-mechanism audit; never supplies model/control inputs."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import bootstrap
import argparse
from collections import Counter
import json
import time
from datetime import datetime,timezone
import numpy as np
from checkpoint_io import file_sha256
ROOT=bootstrap.ROOT
OUT=ROOT/'analysis_1004'
RUN=ROOT/'closed_loop_1004'


def read(path):return json.loads(Path(path).read_text())
def rows(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
def dist(values):
    v=np.asarray(values,dtype=float)
    return None if not len(v) else dict(count=len(v),mean=float(v.mean()),p95=float(np.quantile(v,.95)),max=float(v.max()))

def stamp(row,boundary='after'):
    s=row['grasp_'+boundary]
    return dict(control_step=row['control_step'],completed_control_steps=row['control_step']-(boundary=='before'),boundary=boundary,
                chunk_id=row['chunk_id'],target_k=row['target_k'],intent=s['gripper_intent'],target_width_m=row['target_width'],
                actual_width_m=s['measured_width_m'],commanded_width_m=s['commanded_width_m'],held_apple_id=s['held_apple_id'])

def analyze(job):
    d=RUN/'rollouts'/job['job_id'];completion=read(d/'completed.json');s=read(d/'summary.json');m=read(d/'metrics.json');ss=rows(d/'steps.jsonl');cc=rows(d/'chunks.jsonl')
    assert completion['status']=='COMPLETE'
    for name,digest in completion['files_sha256'].items():assert file_sha256(d/name)==digest,(d,name)
    assert len(ss)==s['control_steps'] and [r['control_step'] for r in ss]==list(range(1,len(ss)+1))
    assert s['cohort']=='dev' and s['checkpoint_step']==8000 and s['actual_inference_seed']==42
    first=next((r['control_step'] for r in ss if r['info']['grasp_assist_triggered'] or r['grasp_after']['held_apple_id'] is not None),None)
    approach=[]
    for row in ss:
        for boundary in ('before','after'):
            a=stamp(row,boundary)
            if first is not None and a['completed_control_steps']>=first:continue
            snapshot=row['grasp_'+boundary];a.update(planned_apple=snapshot.get('planned_apple'),nearest_apple=snapshot.get('nearest_tcp_apple'),
                eligible_apple_ids=snapshot['eligible_apple_ids'],would_attach_now=snapshot['would_attach_now'],
                geometry_eligible_any=bool(snapshot['eligible_apple_ids']),expected_finger_bodies=snapshot['expected_finger_bodies'])
            approach.append(a)
    def closest(key,items=approach):
        valid=[x for x in items if x[key] is not None]
        return min(valid,key=lambda x:x[key]['tcp_distance_m']) if valid else None
    planned=closest('planned_apple');nearest=closest('nearest_apple')
    eligible=[x for x in approach if x['geometry_eligible_any']]
    nonclosing=[x for x in eligible if x['intent']>=0];closing=[x for x in eligible if x['intent']<0]
    firstik=next((r['control_step'] for r in ss if r['info']['ik_failed']),None)
    chunks=[]
    for c in cc:
        sub=[r for r in ss if r['chunk_id']==c['chunk_id']];ap=[x for x in approach if x['chunk_id']==c['chunk_id']]
        cp=closest('planned_apple',ap);cn=closest('nearest_apple',ap)
        if not sub:continue
        chunks.append(dict(chunk_id=c['chunk_id'],loaded_at_control_step=c['at_control_step'],first_control_step=sub[0]['control_step'],last_control_step=sub[-1]['control_step'],
            control_steps=len(sub),targets_executed=sorted({r['target_k'] for r in sub}),advance_reasons=dict(Counter(r['advance_reason'] for r in sub)),
            IK_failed_steps=sum(bool(r['info']['ik_failed']) for r in sub),tracking_position_error_m=dist([r['position_error_m'] for r in sub]),
            tracking_rotation_error_rad=dist([r['rotation_error_rad'] for r in sub]),minimum_planned_approach=cp,minimum_nearest_approach=cn,
            target_width_min_m=min(r['target_width'] for r in sub),target_width_max_m=max(r['target_width'] for r in sub),
            negative_intent_control_steps=sum(r['grasp_after']['gripper_intent']<0 for r in sub),
            positive_intent_control_steps=sum(r['grasp_after']['gripper_intent']>0 for r in sub),
            held_control_steps=sum(r['grasp_after']['held_apple_id'] is not None for r in sub)))
    zero=chunks[0];zero_rows=[r for r in ss if r['chunk_id']==zero['chunk_id']]
    reasons=zero['advance_reasons'];all_reached=(reasons.get('reached',0)==30 and reasons.get('maximum_dwell',0)==0)
    gt=rows(RUN/'rollouts'/f"gt_reach-conditioned_{job['scene']['seed']}"/'chunks.jsonl')[0]
    positional=np.linalg.norm(np.asarray(cc[0]['target_positions'])-np.asarray(gt['target_positions']),axis=1)
    assert np.isclose(positional.mean(),m['first_chunk_teacher_error']['full30']['position_error_m']['mean'])
    post=[r for r in ss if first is not None and r['control_step']>=first]
    releases=[stamp(r) for r in ss if r['grasp_before']['held_apple_id'] is not None and r['grasp_after']['held_apple_id'] is None]
    postdata=None
    if post:
        postdata=dict(first_grasp_step=first,first_detach_step=m.get('first_detach_control_step'),
            same_grasped_fruit_detach_verified=m.get('same_grasped_fruit_detach_verified'),
            first_IK_after_grasp=next((r['control_step'] for r in post if r['info']['ik_failed']),None),IK_steps_after_grasp=sum(bool(r['info']['ik_failed']) for r in post),
            max_target_width_m=max(r['target_width'] for r in post),max_commanded_width_m=max(r['grasp_after']['commanded_width_m'] for r in post),
            positive_intent_steps=sum(r['grasp_after']['gripper_intent']>0 for r in post),release_count=len(releases),release_events=releases,
            final_held_apple_id=ss[-1]['grasp_after']['held_apple_id'],held15=m.get('same_fruit_held_at_least_15_steps'))
    final_approach=approach[-1] if approach else None
    tags=[];explanations=[]
    if first is None:
        if not eligible:
            tags.append('no_observed_grasp_geometry')
            explanations.append('未观察到任一果实同时满足掌内与双指接触条件；需同时查看最近任意果实和专家规划果实，不能将规划果ID设为成功门槛。')
        elif nonclosing and not closing:
            tags.append('geometry_observed_without_closing_at_boundaries')
            explanations.append('边界快照曾显示抓持几何条件，但对应intent未为负；这是时序证据，不能等同连续物理子步内唯一阻碍。')
        elif closing:
            tags.append('eligible_and_closing_snapshot_without_recorded_grasp')
            explanations.append('边界快照存在eligible且closing，仍无实际抓持；需核对快照/子步时序，不直接宣称控制bug。')
        if all_reached and zero['IK_failed_steps']==0:
            tags.append('all_first_chunk_predicted_goals_reached_without_IK_or_forced_advance')
            explanations.append('首chunk30个预测目标均按到达推进，无IK/强制超时；首段未抓不能仅解释为不给执行器等待时间。预测几何与夹爪配合仍须区分。')
        else:
            tags.append('first_chunk_tracking_or_advancement_interventions')
        if firstik is not None and firstik>zero['last_control_step']:
            tags.append('IK_begins_after_first_chunk')
            explanations.append(f'首个IK失败在step{firstik}，晚于首chunk结束step{zero["last_control_step"]}；不能用全程IK比例解释之前已发生的接近偏差。')
        if planned and final_approach and final_approach.get('planned_apple'):
            if final_approach['planned_apple']['tcp_distance_m']>planned['planned_apple']['tcp_distance_m']+.10:
                tags.append('moves_farther_from_planned_fruit_after_closest_approach')
                explanations.append('最近接近后最终离规划果实超过最小距离10cm；这是观察到的后续偏离，尚未隔离导致偏离的原因。')
    else:
        tags.append('actual_grasp')
        if s['success']:tags.append('benchmark_success')
        elif postdata['release_count']==0 and postdata['final_held_apple_id'] is not None:
            tags.append('grasped_but_still_held_without_release')
            if postdata['positive_intent_steps']==0:tags.append('no_positive_open_intent_after_grasp')
        if postdata['first_IK_after_grasp'] is not None:tags.append('IK_after_grasp')
        explanations.append('抓持与抓后阶段分别报告；抓后的高IK比例不能回溯否定已经发生的抓持，也不能将其与未抓组直接作为运输能力排名。')
    return dict(arm=job['arm'],seed=job['scene']['seed'],episode_id=job['scene']['episode_id'],split=job['scene']['split'],directory=str(d),
        outcome=dict(success=s['success'],actual_grasp=first is not None,grasp_step=first,held15=m.get('same_fruit_held_at_least_15_steps'),
            detach_step=m.get('first_detach_control_step'),any_detach=m.get('any_detach'),same_grasped_fruit_detach_verified=m.get('same_grasped_fruit_detach_verified')),
        first_chunk=dict(teacher_target_error=m['first_chunk_teacher_error'],execution=zero,all30_goals_reached_without_forced_advance=all_reached),
        approach=dict(minimum_planned=planned,minimum_nearest=nearest,first300_minimum_planned=closest('planned_apple',[x for x in approach if x['completed_control_steps']<=300]),
            geometry_eligible_boundaries=len(eligible),eligible_nonclosing_boundaries=len(nonclosing),eligible_closing_boundaries=len(closing),
            first_eligible_nonclosing=nonclosing[0] if nonclosing else None,first_eligible_closing=closing[0] if closing else None,
            first_negative_intent=m.get('first_negative_intent_boundary'),last_approach_snapshot=final_approach),
        first_IK_failure_control_step=firstik,total_IK_failed_steps=s['ik_failed_steps'],post_grasp=postdata,chunks=chunks,
        mechanism_tags=tags,interpretation=explanations,observer_consistency=m['observer_consistency'],summary_consistency=m['summary_consistency'],
        evidence_sha256={name:file_sha256(d/name) for name in ('steps.jsonl','chunks.jsonl','metrics.json','summary.json','completed.json')})


def text_summary(records):
    lines=['A8000/B8000 dev8配对失败机制审计（seed42，仅被动CPU分析）','',
      '首chunk目标相对GT误差与实际追踪自身目标的误差是不同量；规划果ID仅用于诊断，任何合法果实抓持均计入能力。closest只统计第一次抓持前，避免持果后果实贴近TCP污染接近距离。',
      'geometry/intent来自控制边界快照，不等同完整物理子步轨迹；不据此修改任何抓持门槛。','']
    for seed in sorted({r['seed'] for r in records}):
        lines.append(f'scene {seed}:')
        for r in [r for r in records if r['seed']==seed]:
            z=r['first_chunk'];e=z['execution'];p=r['approach']['minimum_planned'];n=r['approach']['minimum_nearest'];o=r['outcome']
            lines.append(f"  {r['arm']}: grasp={o['actual_grasp']} @ {o['grasp_step']}, detach={o['detach_step']}, success={o['success']}; firstchunk GT target position mean={1000*z['teacher_target_error']['full30']['position_error_m']['mean']:.2f}mm; own-target tracking mean/p95={1000*e['tracking_position_error_m']['mean']:.2f}/{1000*e['tracking_position_error_m']['p95']:.2f}mm; reached30={z['all30_goals_reached_without_forced_advance']}; firstIK={r['first_IK_failure_control_step']}")
            if p:lines.append(f"    nearest planned before grasp={1000*p['planned_apple']['tcp_distance_m']:.2f}mm @step{p['control_step']}/{p['boundary']}/chunk{p['chunk_id']}/k{p['target_k']}; handlocal={np.round(np.asarray(p['planned_apple']['hand_local_position'])*1000,3).tolist()}mm; intent={p['intent']}; inside={p['planned_apple']['inside_palm_volume']}; both_fingers={p['planned_apple']['both_fingers_contact']}")
            if n:lines.append(f"    nearest ANY fruit before grasp={1000*n['nearest_apple']['tcp_distance_m']:.2f}mm (apple{n['nearest_apple']['apple_id']}); eligible boundaries={r['approach']['geometry_eligible_boundaries']}, nonclosing={r['approach']['eligible_nonclosing_boundaries']}, closing={r['approach']['eligible_closing_boundaries']}")
            if r['post_grasp']:lines.append('    post_grasp='+json.dumps(r['post_grasp'],ensure_ascii=False))
            lines.append('    tags: '+', '.join(r['mechanism_tags']))
            lines.extend('    '+x for x in r['interpretation'])
        lines.append('')
    lines+=['仅16个开发回合，未做参数/检查点选择；heldout汇总另由独立统计分析负责。首chunkGT误差仅衡量示教拟合，不代表只有示教路径才合法。']
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--wait',action='store_true');parser.add_argument('--arm',choices=('A','B'));args=parser.parse_args()
    protocol=read(RUN/'protocol.json');arms=(args.arm,) if args.arm else ('A','B')
    jobs=[j for arm in arms for j in protocol['jobs'][arm] if j['scene']['cohort']=='dev'];expected=len(jobs)
    last_pending=None
    while True:
        missing=[j['job_id'] for j in jobs if not (RUN/'rollouts'/j['job_id']/'completed.json').exists()]
        if not missing:break
        if last_pending!=len(missing):
            print(json.dumps(dict(status='PENDING',complete=expected-len(missing),expected=expected)),flush=True);last_pending=len(missing)
        if not args.wait:return
        time.sleep(30)
    records=[analyze(j) for j in jobs]
    if args.arm:
        print(json.dumps(dict(status='ARM_COMPLETE_PENDING_PAIRED',arm=args.arm,records=records),ensure_ascii=False));return
    payload=dict(schema='orchard_ab_dev_mechanism_audit_v1',status='COMPLETE',created_utc=datetime.now(timezone.utc).isoformat(),
        primary_endpoint_step=8000,inference_seed=42,cohort='dev',episodes_per_arm=8,protocol_sha256=file_sha256(RUN/'protocol.json'),
        records=records,classification_is_posthoc_diagnostic_only=True,no_new_control_or_grasp_gates=True,
        limits=['Only eight fixed development scenes per arm and one noise seed here.','Boundary snapshots cannot establish all internal physics-substep contact/intent timing.','GT target errors and actual tracking errors describe different comparisons.','Planned apple is diagnostic only; any fruit grasp counts.','Postgrasp errors are not comparable to untested stages in no-grasp episodes.'])
    destination=OUT/'mechanism_audit.json';txt=OUT/'mechanism_audit.txt'
    assert not destination.exists() and not txt.exists(),'Refuse overwrite completed audit'
    destination.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n');txt.write_text(text_summary(records))
    print(json.dumps(dict(status='COMPLETE',records=len(records),json=str(destination),text=str(txt))),flush=True)

if __name__=='__main__':main()
