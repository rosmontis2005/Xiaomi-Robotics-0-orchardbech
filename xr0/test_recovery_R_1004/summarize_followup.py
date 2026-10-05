"""Produce version reports without changing frozen datasets or evaluation traces."""
import bootstrap
import json,hashlib
from pathlib import Path
from collections import Counter
ROOT=bootstrap.ROOT;XR0=bootstrap.XR0;D0=XR0/'test_grasp_1004/evaluation';AB=XR0/'test_128_episode_AB_1003'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as s:
  for b in iter(lambda:s.read(8388608),b''):h.update(b)
 return h.hexdigest()
def summarize(version):
 folder=ROOT/'evaluation_followup'/version;ev=folder if version=='version_01' else folder/'evaluation';data=ROOT if version=='version_01' else folder
 if not (ev/'status.json').exists() or read(ev/'status.json')['status']!='COMPLETE':return
 protocol=read(ev/'protocol.json');groups={};cases=[]
 for j in protocol['jobs']:
  r=ev/'rollouts'/j['job_id'];b=D0/'rollouts'/j['baseline_job_id'];a=read(r/'metrics.json');bm=read(b/'metrics.json');s=read(r/'summary.json');bs=read(b/'summary.json')
  completed=read(r/'completed.json');assert completed['baseline_reset_exact']
  for name,h in completed['files_sha256'].items():assert sha(r/name)==h
  main=j['stage']=='R30';cohort='dev8' if main else 'historical_seed42_43'
  case=dict(cohort=cohort,seed=j['scene']['seed'],split=j['scene']['split'],episode_id=j['scene']['episode_id'],inference_seed=j['inference_seed'],baseline_reset_exact=True,rollout=str(r),baseline=str(b),B_held15=bm['same_fruit_held_at_least_15_steps'],R_held15=a['same_fruit_held_at_least_15_steps'],B_detached=bs['max_apple_detached_count']>0,R_detached=s['max_apple_detached_count']>0,B_release=bm['observed_held_to_none_release_count']>0,R_release=a['observed_held_to_none_release_count']>0,B_strict=bs['strict_success'],R_strict=s['strict_success'],R_legacy=s['legacy_success_ever'],R_branch_breaks=s['max_branch_break_count'],R_ik_failure_rate=s['ik_failed_rate'],R_replans=s['replans'],R_steps=s['control_steps'],R_release_events=s['strict_placement']['release_events'],R_strict_events=s['strict_placement']['strict_success_events'],R_max_stable_bucket_steps=max([x['max_stable_bucket_steps'] for x in s['strict_placement']['fruit_chains'].values()],default=0))
  cases.append(case)
 for name in ['dev8','dev8_train2','dev8_val6','historical_seed42_43']:
  subset=[c for c in cases if c['cohort']==name] if name in ['dev8','historical_seed42_43'] else [c for c in cases if c['cohort']=='dev8' and c['split']==('train' if name=='dev8_train2' else 'val')]
  groups[name]=dict(n=len(subset),unique_episodes=len(set(c['episode_id'] for c in subset)),**{f'{model}_{metric}':sum(c[f'{model}_{metric}'] for c in subset) for model in ['B','R'] for metric in ['held15','detached','release','strict']},R_branch_break_episodes=sum(c['R_branch_breaks']>0 for c in subset))
 offline=read(ev/'offline/evaluations/step_4000/report.json');baseoff=read(AB/'arms/B/evaluations/step_8000/report.json');off={}
 for key in ['old_train/all','old_val/all','new_val/all','train/reset','val/reset','val/phase_REACH','val/phase_GRASP','val/phase_PULL','val/phase_TRANSPORT']:
  off[key]={metric:dict(B=baseoff['groups'][key]['full'][metric]['mean'],R=offline['groups'][key]['full'][metric]['mean']) for metric in ['position_error_m','rotation_error_rad','width_error_m']}
 audit=read(data/'data_audit.json');training=read(ev/'training_audit.json')
 report=dict(version=version,status='COMPLETE',checkpoint=protocol['checkpoint'],checkpoint_sha256=protocol['checkpoint_sha256'],step=4000,baseline_checkpoint=read(ROOT/'protocol.json')['baseline'],evaluation_protocol=str(ev/'protocol.json'),groups=groups,cases=cases,offline=off,training=training,data={k:audit[k] for k in ['accepted_counts','unique_source_episodes','recovery_available_unique_windows','recovery_schedule_unique_windows','repeat_draws','schedule_counts','actual_anchor_phase_counts_schedule','future_target_phase_counts_schedule','schedule_sha256','manifest_sha256']},limitations=['dev8 includes2 original-train episodes excluded from recovery sources and6 validation episodes; 8 is small','historical auxiliary is one scene with2 inference seeds, not2 independent scenes','No heldout12/fresh24 used for tuning at this point','Offline mean errors pool correlated targets and three noise seeds; not independent task-success estimates'],production_sources_unchanged=True,paired_reset_checks='10/10 exact')
 tdir=ROOT/'evaluation_followup/version_01/takeovers'
 if version=='version_01' and (tdir/'results.json').exists():
  diagnostic=read(tdir/'results.json');pairs=[]
  for cid in sorted({r['candidate'] for r in diagnostic}):
   match={r['model']:r for r in diagnostic if r['candidate']==cid};valid=len(match)==2 and all(r['strict_success'] is not None for r in match.values());pairs.append(dict(candidate=cid,category=next(iter(match.values()))['category'],paired_available=valid,B_strict=match.get('B8000',{}).get('strict_success'),R_strict=match.get('R4000',{}).get('strict_success'),B_reason=match.get('B8000',{}).get('reason'),R_reason=match.get('R4000',{}).get('reason')))
  roff={}
  for cat in ['R1','R2','R3']:
   roff[cat]={}
   for model in ['B8000','R4000']:
    rr=[r for r in read(tdir/f'offline_{model}.json') if r['category']==cat and r['frame']==0];roff[cat][model]={m:sum(x[m] for x in rr)/len(rr) for m in ['position_mae_m','rotation_mae_rad','width_mae_m']}
  report['takeover_diagnostic']=dict(pairs=pairs,frame0_offline=roff,role='training-source mechanism diagnostic only; not held-out performance',available_pairs=sum(p['paired_available'] for p in pairs))
 (folder/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
 text=[f'# {version}：R4000 测评', '',f"Endpoint: `{protocol['checkpoint']}`，step=4000，SHA256=`{protocol['checkpoint_sha256']}`。",'', '执行保持 full30 / 30-target chunk / reach-conditioned / 10mm / 0.08rad / dwell30 / budget900 / Euler5 / benchmark_assist。基座、控制器、动作表示、归一化和物理源码未改。', '', '| Cohort | n | B held15 | R held15 | B detach | R detach | B release | R release | B strict | R strict |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for name,g in groups.items():text.append('| '+name+' | '+' | '.join(str(g[k]) for k in ['n','B_held15','R_held15','B_detached','R_detached','B_release','R_release','B_strict','R_strict'])+' |')
 text += ['', 'Strict = 同果 held15 → detach → release → unheld 且连续60控制步留桶；legacy in-bucket 不计 strict。B 为原冻结 D0 配对 traces，10/10 reset（含双路RGB哈希）完全一致。GT 原同条件 dev8 strict 8/8，控制源码哈希仍一致；本轮未重复整套 GT gate。', '', '| Scene / RNG | split | B held15 | R held15 | B strict | R strict |','|---|---|---:|---:|---:|---:|']
 for c in cases:text.append(f"| {c['seed']} / {c['inference_seed']} | {c['split']} | {int(c['B_held15'])} | {int(c['R_held15'])} | {int(c['B_strict'])} | {int(c['R_strict'])} |")
 text += ['', '| Offline full30 group | B xyz mm | R xyz mm | B rot rad | R rot rad | B width mm | R width mm |','|---|---:|---:|---:|---:|---:|---:|']
 for key,v in off.items():text.append(f"| {key} | {v['position_error_m']['B']*1000:.2f} | {v['position_error_m']['R']*1000:.2f} | {v['rotation_error_rad']['B']:.4f} | {v['rotation_error_rad']['R']:.4f} | {v['width_error_m']['B']*1000:.2f} | {v['width_error_m']['R']*1000:.2f} |")
 text += ['',f"训练4000次真实更新。配比 `{audit['schedule_counts']}`，同24条 recovery（R1=12/R2=6/R3=6），12个来源 episodes。合法 recovery windows={audit['recovery_available_unique_windows']}，实际曝光 unique={audit['recovery_schedule_unique_windows']}，重复抽样={audit['repeat_draws']}。Teacher 和 oracle 原24/24 PASS；本版本复用哈希固定数据，未重新生成 label。",'', '实际 observation anchor phase：', '```json',json.dumps(audit['actual_anchor_phase_counts_schedule'],indent=2),'```', '实际未来30 target phases：', '```json',json.dumps(audit['future_target_phase_counts_schedule'],indent=2),'```']
 if 'takeover_diagnostic' in report:
  d=report['takeover_diagnostic'];text += ['',f"中途 takeover 配对诊断有 {d['available_pairs']} 个可比较状态。它们来自训练来源，不能作为泛化成绩。一个 seed43 前缀因 measured width 相差0.1286mm，超过0.1mm gate，被保留为 unavailable；未放宽 gate，也未计作模型失败。另补同类别首个不同来源 accepted 状态，规则和失败记录在 takeovers/engineering_repairs.jsonl。",'```json',json.dumps(d,indent=2),'```']
 text += ['', '小样本限制：dev8不是最终测试集；历史补充两个seed属于同一episode；离线窗口、目标和噪声seed相关。冻结 heldout12 / fresh24 未参与本版本数据或配比决策。完整 traces、双路视频、metrics、load manifest、训练审计和失败证据保存在本目录。']
 (folder/'report.md').write_text('\n'.join(text)+'\n')
 return report
if __name__=='__main__':
 reports=[r for version in ['version_01','version_02','version_03','version_04'] if (r:=summarize(version)) is not None]
 (ROOT/'evaluation_followup/versions_summary.json').write_text(json.dumps(reports,indent=2,ensure_ascii=False)+'\n');print(json.dumps({r['version']:r['groups'] for r in reports},ensure_ascii=False))
