"""CPU-only report assembly after the frozen final paired evaluation completes."""
from pathlib import Path
import json,hashlib,math,csv,time
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'evaluation_followup'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2))
def wilson(k,n):
 z=1.959963984540054;p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;r=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
 return [max(0,c-r),min(1,c+r)]
versions=read(OUT/'versions_summary.json');selected=read(OUT/'selected_endpoint.json');hold=OUT/'final_holdout12';assert read(hold/'status.json')['status']=='COMPLETE';protocol=read(hold/'protocol.json');assert selected['sha256']==protocol['checkpoint_sha256'];assert protocol['code_sha256']==sha(ROOT/'evaluate_final_holdout.py')
cases=[]
for scene in protocol['scenes']:
 rows={}
 for label in ['B8000','R_selected']:
  folder=hold/'rollouts'/f'{label}_targets30_rng42_reach-conditioned_{scene["seed"]}'
  complete=read(folder/'completed.json');assert complete['status']=='COMPLETE'
  for name,h in complete['files_sha256'].items():assert sha(folder/name)==h
  s=read(folder/'summary.json');m=read(folder/'metrics.json');assert s['scene_check_pass'];assert m['observer_consistency']['mismatch_fields_total']==0;assert m['summary_consistency']['mismatch_count']==0
  calls=read(folder/'seed_delivery.json')['calls'];assert len(calls)==s['replans'];assert all(c['actual_inference_seed']==42 for c in calls)
  rows[label]=dict(held15=bool(m['same_fruit_held_at_least_15_steps']),detached=bool(s['max_apple_detached_count']>0),release=bool(s['strict_placement']['release_events']),strict=bool(s['strict_success']),strict_events=s['strict_placement']['strict_success_events'],release_events=s['strict_placement']['release_events'],branch_breaks=s['max_branch_break_count'],steps=s['control_steps'],replans=s['replans'],termination=s['termination'],ik_failure_rate=s['ik_failed_rate'],path=str(folder))
 b=hold/'rollouts'/f'B8000_targets30_rng42_reach-conditioned_{scene["seed"]}'/'initial.json';r=hold/'rollouts'/f'R_selected_targets30_rng42_reach-conditioned_{scene["seed"]}'/'initial.json';assert read(b)['obs']==read(r)['obs']
 cases.append(dict(scene_seed=scene['seed'],episode_id=scene['episode_id'],split=scene['split'],previously_used_auxiliary=scene['seed']==2010600,paired_reset_exact=True,B=rows['B8000'],R=rows['R_selected']))
def summarize(xs):
 result=dict(n=len(xs),unique_episodes=len({x['episode_id'] for x in xs}))
 for label in ['B','R']:
  result[label]={k:sum(bool(x[label][k]) for x in xs) for k in ['held15','detached','release','strict']}
  result[label]['branch_break_episodes']=sum(x[label]['branch_breaks']>0 for x in xs)
  result[label]['strict_wilson95']=wilson(result[label]['strict'],len(xs))
 both=sum(x['B']['strict'] and x['R']['strict'] for x in xs);b_only=sum(x['B']['strict'] and not x['R']['strict'] for x in xs);r_only=sum(x['R']['strict'] and not x['B']['strict'] for x in xs);n=b_only+r_only
 result['paired_strict']=dict(both=both,B_only=b_only,R_only=r_only,neither=len(xs)-both-b_only-r_only,exact_two_sided_mcnemar_p=min(1.,2*sum(math.comb(n,k) for k in range(min(b_only,r_only)+1))/2**n) if n else 1.)
 return result
final=dict(status='COMPLETE',selected_endpoint=selected,protocol=protocol,groups={'heldout12':summarize(cases),'remaining11':summarize([x for x in cases if not x['previously_used_auxiliary']]),'previous_auxiliary1':summarize([x for x in cases if x['previously_used_auxiliary']])},cases=cases,paired_reset_checks='12/12 exact',limits=['Historical curated replay-PASS val scenes, not fresh random scenes.','One scene2010600 already used as auxiliary; remaining11 reported separately.','Endpoint frozen using dev8 before these runs; no further tuning.','Small sample and one primary inference seed; no claim of significance or population success.'])
write(hold/'report.json',final)
md=['# 冻结 R01 与 B8000：最终配对 heldout 验证','', '固定 full30、reach-conditioned、10mm/0.08rad、dwell30、budget900、benchmark_assist、Euler5；同果 held15→detach→release→连续60步稳定留桶才算 strict。detach列为episode曾有detached的诊断计数，release列为held→unheld事件（合法chain另见JSON），两者不替代strict身份链。B 与 R 均在本轮重新执行，双路 RGB/proprio reset 12/12 完全一致。','', '| Cohort | n | B held15 | R held15 | B detach | R detach | B release | R release | B strict | R strict |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for group,x in final['groups'].items():md.append(f"| {group} | {x['n']} | {x['B']['held15']} | {x['R']['held15']} | {x['B']['detached']} | {x['R']['detached']} | {x['B']['release']} | {x['R']['release']} | {x['B']['strict']} | {x['R']['strict']} |")
md+=['','| Scene | prior auxiliary | B held15 | R held15 | B strict | R strict |','|---|---|---:|---:|---:|---:|']
for x in cases:md.append(f"| {x['scene_seed']} | {x['previously_used_auxiliary']} | {int(x['B']['held15'])} | {int(x['R']['held15'])} | {int(x['B']['strict'])} | {int(x['R']['strict'])} |")
md+=['','按场景配对的 strict 结果与描述性 Wilson 区间见 [report.json](report.json)。这是历史挑选过的12个 val 场景，2010600 已做辅助诊断；其余11个仅在冻结选择后使用。未运行 fresh24，不能把本表称为随机新场景总体成功率。所有失败、视频、步骤与 chunk、模型加载报告均保留；observer 和 summary 一致性检查均通过。']
(hold/'report.md').write_text('\n'.join(md)+'\n')
with (hold/'cases.csv').open('w') as f:
 writer=csv.writer(f);writer.writerow(['scene_seed','episode_id','prior_aux','B_held15','R_held15','B_strict','R_strict','B_release','R_release'])
 for x in cases:writer.writerow([x['scene_seed'],x['episode_id'],x['previously_used_auxiliary'],x['B']['held15'],x['R']['held15'],x['B']['strict'],x['R']['strict'],x['B']['release'],x['R']['release']])
audit=read(ROOT/'data_audit.json');onpolicy=[]
p=OUT/'onpolicy_R3/candidate_decisions.jsonl'
if p.exists():onpolicy=[json.loads(s) for s in p.read_text().splitlines() if s.strip()]
assert len(onpolicy)==4 and not any(x['accepted'] for x in onpolicy)
aggregate=dict(status='COMPLETE',created_unix=time.time(),versions=versions,final=final,original_data_audit=str(ROOT/'data_audit.json'),additional_cycles=2,maximum_additional_cycles=3,onpolicy_attempts=onpolicy,decision='Preserve R as a local-recovery baseline; stop ratio-only R retries. Propose causal observable history H as a new hypothesis, no implementation or training in this task.',fresh24_run=False,new_direction_training_started=False)
write(OUT/'summary_report.json',aggregate)
md=['# R student-state recovery：三版训练与闭环汇总','', '完成原始 R01 及两次额外 R 训练—闭环验证。每版各4000次真实 optimizer update，分别保存数据审计、冻结 schedule、checkpoint、离线预测、闭环步骤和视频。测试前按 dev8 strict 规则选择 R01，之后仅做一次 B/R 配对 heldout 验证。','', '## 结果','', '| 模型/版本 | dev8 held15 | dev8 detach | dev8 release | dev8 strict | 辅助1场景×2seed strict |','|---|---:|---:|---:|---:|---:|','| B8000 | 6/8 | 6/8 | 4/8 | 0/8 | 0/2 |']
for v in versions:
 g=v['groups']['dev8'];h=v['groups']['historical_seed42_43'];md.append(f"| [{v['version']}]({v['version']}/report.md) | {g['R_held15']}/8 | {g['R_detached']}/8 | {g['R_release']}/8 | {g['R_strict']}/8 | {h['R_strict']}/2 |")
md+=['', '| 最终冻结评测 | B held15 | R01 held15 | B strict | R01 strict |','|---|---:|---:|---:|---:|']
for group,x in final['groups'].items():md.append(f"| {group} | {x['B']['held15']}/{x['n']} | {x['R']['held15']}/{x['n']} | {x['B']['strict']}/{x['n']} | {x['R']['strict']}/{x['n']} |")
md+=['','最终完整配对病例、放置事件和置信区间见 [final_holdout12/report.md](final_holdout12/report.md) 与 [JSON](final_holdout12/report.json)。dev8 含2个原训练场景和6个验证场景；两个训练场景没有进入 recovery mining。辅助结果只有一个场景，不能按两个独立测试场景解释。heldout12 是历史挑选过的 oracle replay-PASS val cohort；其中2010600曾做辅助分析，remaining11另报。fresh24 完全未运行、未用于训练或选择。','', '## 三版数据改动与训练','', '| 版本 | 原B | R1 | R2 | R3 | teacher-continuation | 可用 recovery windows | 实际unique | 重复抽样 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for v in versions:
 d=v['data'];c=d['schedule_counts'];md.append(f"| {v['version']} | {c['original_B']} | {c['R1']} | {c['R2']} | {c['R3']} | {c.get('R_continuation_teacher',0)} | {d['recovery_available_unique_windows']} | {d['recovery_schedule_unique_windows']} | {d['repeat_draws']} |")
md+=['','R01 保持首次4000配比；仅使用takeover后前30个anchor，却发现R1曝光863/1000已是PULL，R2/R3未来targets全为TRANSPORT。R02 将原B增到75%，R1仅选真实未held、未detached的近接触anchor；保持第一版30-anchor上限，未更换label。R03 保留这些真实恢复起点，并添加7.5%单独标记的 teacher-continuation 稀疏放置桥接窗口；它们来自原24条成功recovery的后段，不冒充新student-state样本。其未来30targets增加4594个DROP和2110个DONE。后段window放宽首轮anchor限制的理由和检查在 [version_03/design.json](version_03/design.json)；这属于用户本轮授权的数据修改。','', '原29个实际运行teacher的unique候选：R1 12/12接受，R2 6/6接受，R3 6/11接受、5拒绝；覆盖12个unique训练来源。24/24 teacher strict与当前student-controller oracle全轨迹PASS。未以测试场景补数量。R02/R03继续复用这24条固定成功轨迹；新当前R晚期on-policy candidate共4个因同态前缀再生成不一致被拒绝，teacher/oracle未建立，接受0；未加入训练、未放宽gate。两个阈值批次以及每个错误的traceback均在 [onpolicy_R3](onpolicy_R3/candidate_decisions.jsonl)。','', '所有三版从 original full10k → 正式B8000 step8000 overlay 开始，而不是从上一版R串行继续。逐一校验219个FP32 trainable tensors初始值与B完全一致；fresh AdamW、lr1e-5、betas(0.9,0.95)、wd0.1、eps1e-8、clip1、batch1、acc1、constantLR；frozen VLM BF16、BF16 autocast、原full30 active7 flow loss，未加入L权重。原action_stats始终不变，未clip或重算归一化。R03扩展label的最高active normalized绝对值约4.10，>5计数0；这只是outlier审计，不是对label做阈值处理。模型只读取两路RGB/current proprio/原prompt；teacher与debug truth未进入model batch。','', '## 离线误差与机制证据','', '| 模型 | old-val xyz mm | new-val xyz mm | val reset xyz mm | val GRASP xyz mm | val PULL xyz mm | val TRANSPORT xyz mm |','|---|---:|---:|---:|---:|---:|---:|']
keys=['old_val/all','new_val/all','val/reset','val/phase_GRASP','val/phase_PULL','val/phase_TRANSPORT'];b=versions[0]['offline'];md.append('| B8000 | '+' | '.join(f"{b[k]['position_error_m']['B']*1000:.2f}" for k in keys)+' |')
for v in versions:md.append('| '+v['version']+' | '+' | '.join(f"{v['offline'][k]['position_error_m']['R']*1000:.2f}" for k in keys)+' |')
md+=['','离线使用原冻结240-window panel，80 old_train/80 old_val/80 new_val，inference seeds42/43/44，30targets；各目标和noise seed相关，均值不是独立成功率。R01 recovery-frame0在训练来源的R2位置误差259.6→94.2mm、R3 275.0→129.9mm；这只证明局部监督被学到。真实中途接管诊断6个可比较状态：B与R01均strict0/6。两条R2最低bucket距离由B0.827/0.682m改善为R0.095/0.112m，但均未release。第7个seed43接管因width误差0.1286mm超过0.1mm gate保留为unavailable，未算模型失败。源episode是训练来源，不计泛化结果。','', 'R02修复reset/REACH离线退化，但dev8 strict并未改善；R03提升抓持且补入真实成功teacher后段label，仍不能可靠完成运输—释放—稳定留桶。因此不能由单步loss或抓持成功直接宣布R达成整任务目标。另一方面，样本少、late-current-R样本未成功建立，结论也不足以否定更完整的交互式R数据路线。','', '## 执行与可复现记录','']
for v in versions:
 t=v['training']['status'];md += [f"- {v['version']}: {t['completed_updates']} updates，训练PID/session={t['pid']}/{t.get('session',t['pid'])}；step4000 SHA `{v['checkpoint_sha256']}`；schedule SHA `{v['data']['schedule_sha256']}`。"]
md+=['', f"正式B8000 SHA `{versions[0]['baseline_checkpoint']['sha256']}`，full10k SHA `{versions[0]['baseline_checkpoint']['base_sha256']}`。", '', '各版原始training status中 post_training_evaluation=false 是训练进程自身的字段；本轮另起独立evaluation进程，完成后的status/metrics位于各版evaluation目录。原第一轮的停止点报告保持原样，不覆盖成这次结果。R01原训练状态/步骤/manifest快照在 version_01/original_training_snapshot，checkpoint用哈希引用。', '', f"最终闭环PID/session={read(ROOT/'final_holdout_receipt.json')['pid']}，启动命令与日志见 [final_holdout_receipt.json](../final_holdout_receipt.json)，最终status=COMPLETE24。各额外训练和evaluation命令在独立version目录的receipt与queue记录中。", '', 'production controller/action/physics、正式B数据与归一化的哈希保护通过；dev对照使用原冻结D0 B traces，10个reset全部精确一致，GT同条件dev8历史strict8/8；本轮没有重复全部旧Gate。最终B与R都重新执行，12个reset精确配对。所有完成文件SHA检查通过，observer/summary不一致计数为0。', '', '工程失败全部保留：首次CPU helper导入修复发生在任何闭环开始前；中途接管不可再生案例保留为unavailable；晚期composite prefix4次失败未产生label；版本02规划中的可选takeover诊断没有执行，不计为该版证据。辅助seed43的legacy summary字段硬编码42，但实际BoundSeedPolicy调用、step trace与protocol均为43；每版seed_audit.json逐调用验证PASS，原始summary保留并明确解释。', '', '## 可重复性限制', '', '辅助2010600的fresh B与历史B reset和首个预测chunk完全一致，但第一控制步实测TCP已有1.19e-7m差异；长程release结果不同。本轮没有隔离唯一根因，不能声称完整物理rollout位级确定性。最终采用本轮重新配对的B/R结果；历史记录保留且不择优替换，见 [historical_baseline_reproducibility.json](final_holdout12/historical_baseline_reproducibility.json)。小量同态再生成失败被拒绝，也没有通过降低门槛纳入数据。', '', '## 决策', '', '保留R作为有效的局部恢复数据baseline，停止继续仅调配比的R重训。原R01有完整成功个例，R03抓持更强，但目前没有形成可靠整任务放置能力。下一步建议检验“可观测因果历史能否补充任务进度/运动状态”，作为与纯R数据分布调整不同的新方向；这是假设，尚未证明单帧观测导致不可约歧义。详细代码依据、文献、对照和通过标准见 [next_experiment_proposal.md](next_experiment_proposal.md)。本轮没有实施或启动新方向训练，也没有消耗剩余第3次R尝试额度。']
(OUT/'summary_report.md').write_text('\n'.join(md)+'\n')
print(json.dumps(final['groups'],ensure_ascii=False));print('SUMMARY_COMPLETE',OUT/'summary_report.md')
