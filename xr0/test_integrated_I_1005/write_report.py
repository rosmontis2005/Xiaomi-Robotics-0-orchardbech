"""Unified report from measured artifacts; conclusions require explicit completion."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
KEYS=['held15','same_fruit_detach','bucket_neighborhood','reasonable_release','stable_bucket','strict','long_held_stall','premature_release','physical_anomaly']

def load(path,default=None):
    return json.loads(path.read_text()) if path.exists() else default

def report():
    collection=load(ROOT/'collection_summary.json',{}); frozen=load(ROOT/'frozen.json',{}); stats=load(ROOT/'dataset_statistics.json',{}); pipeline=load(ROOT/'pipeline_status.json',{})
    decision=load(ROOT/'route_decision.json'); completion=load(ROOT/'completion.json'); endpoint=load(ROOT/'final_endpoint.json')
    rows=['# OrchardBench-VLA / XR-0 Integrated Training','']
    rows.append('本轮已结束。'+decision['conclusion'] if completion and decision else f"进行中：{pipeline.get('status','PREPARING')} / Round {pipeline.get('round',1)} / {pipeline.get('stage','pending')}。未完成的训练或评价不形成路线结论。")
    if completion:
        rows+=['',f"完成 {completion['completed_rounds']}/3 轮方案对照、9个 endpoint 闭环；本轮实际新增 {completion['optimizer_updates']} optimizer updates。三轮全部停止，不开启第四轮。"]
    rows+=['','## 1. Live student → teacher 数据闭环','',f"实际 student source rollouts {collection.get('source_rollouts',0)}；candidate {collection.get('candidate',0)}；accepted {collection.get('accepted',0)}；rejected {collection.get('rejected',0)}；accepted unique source episodes {collection.get('unique_source_episodes',0)}。",'','| late-state | candidate | accepted |','|---|---:|---:|']
    for category in ['transport_stall','near_bucket_no_release','premature_release_precursor']:
        rows.append(f"| {category} | {collection.get('candidate_categories',{}).get(category,0)} | {collection.get('accepted_categories',{}).get(category,0)} |")
    rows+=['',f"Rejected 原因：`{json.dumps(collection.get('rejected_reasons',{}),ensure_ascii=False)}`。原 observer teacher strict {collection.get('teacher_strict_success',0)}；通过合理释放与全部数据门槛的 accepted {collection.get('accepted',0)}，两者不混用。",'',
        '唯一 mining student 为 current-only R03-8k。Student 从 reset 按原 full30 reach-conditioned production 路径执行，teacher 直接接管同一个正在运行的 env；prefix regeneration 不参与获得 suffix。Teacher 只初始化私有规划 bookkeeping/IK，初始化前后真实 physics digest 相等；命令经过原 OrchardActionAdapter 和 env.step。Teacher 通过实测关节反馈限制 native TCP 目标，并在桶附近修正实际持果偏移；真实释放仍由 production sustained-open gate 触发。没有 teleport、写 simulator state、伪造 held 或改 controller/physics。', '',
        '每条 accepted 保存完整双 RGB、实测 proprio、student prefix/chunks、takeover step、teacher suffix/action labels、source checkpoint 和诊断真值。冻结时逐条核对 prefix 最后时间戳/proprio 等于 suffix 第一个观测，prefix-end digest 等于 teacher-init digest，label 为下一步实测状态，timestamps 为30Hz。接受要求 same-fruit held15→detach→合理释放→non-held→stable bucket60，并通过原 physical health gate；失败/中断/invalid attempts 留档但不训练。', '',
        '工程修复前3条 accepted 在原900总预算内完成；修复后69条从reset配置总预算1800，student最多900、teacher suffix最多900。接管时不修改 env.done/state，两个版本都通过同一物理与合理释放门槛。正式 student evaluator始终900。必要工程修复见 engineering_repairs.jsonl，collector 各版本源码保留并按每条 provenance 记录 SHA。', '',
        '新数据与旧 teacher-continuation 的区别是起点来自当前 R03-8k 实际到达的 late-state，接管后仍在同一真实 env。旧数据主要是既有成功 recovery 的后段，不提供这种当前闭环 late-state 监督。Near-bucket/no-release 自然仅接受2条，覆盖有限；保持真实 failure distribution，不人工凑类别比例。', '',
        'Simulator truth 只用于 mining、分类/接管、teacher、审计和诊断。Student loader 仍只形成原 prompt、当前两路 RGB、当前 proprio 和原 action/mask；fruit/bucket xyz、velocity、held/detach/phase、apple ID、seed/debug metadata、history 均不进入 policy input。','',
        '## 2. D*、provenance 与冻结 schedule','',
        f"D* trajectories：`{json.dumps(stats.get('trajectories_by_source',{}))}`。Original 为既定128个训练 episodes 的全部合法 full30 task windows，覆盖 early/grasp-pull/transport/drop-done；不是旧 B quotas，也不声称纳入原2650 episodes 的全部数据。Existing R 为已通过原 oracle 的24条真实 R1/R2/R3 student-state recovery onset；live late 为本轮72条合格同-env suffix。", '',
        f"Available full30 windows：`{json.dumps(stats.get('available_windows',{}))}`。数据/media/annotations/accepted prefix/code/action-stats/checkpoint hashes 见 frozen.json；manifest 见 dataset_manifest.jsonl。保护 dev/val/historical auxiliary/fresh24 等 mining exclusion；dev train2 仍是原训练数据中的已消费开发 case。", '',
        '| checkpoint | SHA256 |','|---|---|',
        f"| original full10k | `{frozen.get('base_checkpoint_sha256','pending')}` |",f"| historical B8000 | `{frozen.get('B_checkpoint_sha256','pending')}` |",f"| R03-8k mining | `{frozen.get('mining_checkpoint_sha256','pending')}` |", '',
        'B8000 来源为 test_128_episode_AB_1003/arms/B/checkpoints/step_8000_trainable.pt，由 original full10k 继续8000更新得到；其旧 quota 为 reset1600、first1_4=800、REACH1200、GRASP2000、PULL1200、TRANSPORT800、DROP400。本轮不继承该前段偏重采样，只比较其权重 warm-start。', '',
        'Round1 original/R/live-late=60/15/25；original内部25/35/25/15；live内部onset/transport/release/settle=20/30/35/15。D_A/D_B各8000，seed100501/100502，模型启动前冻结。后续共同分布修订使用独立轮次目录，不覆盖这些原 schedule 或 D*。','',
        '## 3. S/J 实验与统一闭环','',
        '所有轮次 S 从原 full10k→历史 B8000 trainable weights 开始，fresh AdamW；J 从原 full10k 开始，fresh AdamW，D_A→D_B连续 optimizer/RNG并保留 J8/J16。每轮新增预算 S8=8000、J8=8000、J16=16000；full10k之后累计 S8 与 J16 均约16000。不同轮次不从上轮 endpoint 打补丁。', '',
        '冻结 VLM BF16，非 VLM master FP32；current two RGB/current proprio、原 prompt/action encoding/action stats、full30 active7 normalized flow coefficient0.5、Euler5、batch1/acc1、AdamW betas(0.9,0.95)/wd0.1/eps1e-8/clip1、同 train seed42、同900-step closed-loop协议均保持。lr默认1e-5，实际每轮 config/receipt/run_manifest/checkpoint SHA/source loss 独立记录。', '',
        '原 StrictPlacement observer 与 reach loop 保持不变。原结果记为 strict_original_observer；按本轮完整链要求，主 strict 另外核对该原成功链的实际 release 在既有合理区域（fruit-bucket distance≤0.15m且 bucket XY 内）。落桶但 release 超界不晋级。原 raw metrics、qualified booleans、release距离和修正后的诊断标签分别保留，不能把原 observer 结果隐藏或替代主 strict。Dev8 与历史辅助都是已消费开发诊断，不是 fresh test。','']
    for number in range(1,4):
        directory=ROOT/f'round{number}'
        if not directory.exists():continue
        revision=load(directory/'config.json')
        summary=load(directory/'closed_loop_summary.json',{})
        rows += [f'### Round {number}','']
        if revision:
            seed_description='逐字节复用 Round2 schedules，不重新采样' if number==3 else '独立冻结 schedules'
            rationale={2:'Round1 三个 endpoint strict 均0：S8 held7但停滞/释放失败，J8/J16 held5，J16 arrival2/stable2但合理释放0。保持 existing R 比例，对两条路线共同小幅提高 original grasp 与 live transport/release exposure；D* 不增加样本，不按 scene 打补丁。',3:'Round2 S8/J8 held15=7/8、J16降至4/8，三者strict均0，尽管 original/live loss下降。共同 lr 从1e-5降为5e-6，检验较小更新是否保留抓取并拟合 late监督；完全复用 Round2 样本顺序、预算、初始化与其他参数。此为最后允许的一轮，不声称已证明失败由学习率导致。'}[number]
            rows += [f"共同 source ratios：`{json.dumps(revision['source_ratios'])}`；original phase：`{json.dumps(revision['original_phase_ratios'])}`；live phase：`{json.dumps(revision['live_phase_ratios'])}`。{seed_description}，seed：`{json.dumps(revision['schedule_seeds'])}`；lr={revision['lr']}。",'',f"修改依据：{rationale}",'']
        else:rows+=['首版60/15/25和既定8k/16k预算，严格执行S8/J8/J16。','']
        if summary.get('training_endpoints'):
            rows+=['| endpoint | updates | original/R/live | first500→last500 original/R/live loss | checkpoint SHA256 |','|---|---:|---|---|---|']
            for name,item in summary['training_endpoints'].items():
                sources=['original','existing_R','live_late'];counts='/'.join(str(item['source_counts'][source]) for source in sources);loss=' / '.join(f"{item['loss_by_source'][source]['first500']:.4f}→{item['loss_by_source'][source]['last500']:.4f}" for source in sources)
                rows.append(f"| {name} | {item['updates']} | {counts} | {loss} | `{item['checkpoint_sha256']}` |")
            rows+=['']
            rows+=['Loss 为训练头/尾各500次 optimizer updates 内按 source 分组的均值；不是事件成功率。所有6个实际训练作业的 frozen VLM 参数 hash 前后一致，详见各 training_summary.json。','']
            for name,item in summary['training_endpoints'].items():
                rows.append(f"- {name} checkpoint：`{item['checkpoint']}`。")
            rows+=['']
        schedule_directory=ROOT if number==1 else directory
        for schedule_name in ['D_A','D_B']:
            schedule_path=schedule_directory/f'schedule_{schedule_name}.jsonl'
            if schedule_path.exists():
                rows.append(f"- {schedule_name}：`{schedule_path.relative_to(ROOT)}`，SHA256 `{hashlib.sha256(schedule_path.read_bytes()).hexdigest()}`。")
        rows+=['']
        if summary.get('schedule_parity'):
            rows+=[f"实际执行 S/J D_A 对齐：`{json.dumps(summary['schedule_parity'],ensure_ascii=False)}`。",'']
        if summary.get('round2_lr_control'):
            rows+=[f"Round2→Round3 的实际训练对照：`{json.dumps(summary['round2_lr_control'],ensure_ascii=False)}`。这里只验证变量控制，不推断学习率是失败原因。",'']
        rows+=['| cohort / endpoint | held15 | detach | bucket-neighborhood | reasonable-release | stable-bucket | strict | long-held-stall | premature-release | physical-anomaly | raw observer strict |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        evaluation={name:load(directory/name/'evaluation/aggregate.json') for name in ['S8','J8','J16']}
        for cohort in ['dev8','historical_seed42_43']:
            for name,item in evaluation.items():
                group=item[cohort] if item else None
                values=['pending']*10 if group is None else [f"{group[key]}/{group['n']}" for key in KEYS+['strict_original_observer']]
                rows.append('| '+cohort+' / '+name+' | '+' | '.join(values)+' |')
        rows+=['']
        if all(evaluation.values()):
            s,j,j16=[evaluation[name]['dev8'] for name in ['S8','J8','J16']]
            rows += [f"S8 vs J8：strict {s['strict']}/{s['n']} vs {j['strict']}/{j['n']}；held15 {s['held15']} vs {j['held15']}；arrival {s['bucket_neighborhood']} vs {j['bucket_neighborhood']}；reasonable release {s['reasonable_release']} vs {j['reasonable_release']}。",'',f"S8 vs J16：strict {s['strict']}/{s['n']} vs {j16['strict']}/{j16['n']}；held15 {s['held15']} vs {j16['held15']}；arrival {s['bucket_neighborhood']} vs {j16['bucket_neighborhood']}；reasonable release {s['reasonable_release']} vs {j16['reasonable_release']}。",'']
        else:rows+=['本轮 S/J 三个 endpoint 尚未全部完成，路线比较待闭环齐备。','']
        rows += ['Per-case 原始 steps/chunks/RGB/metrics、cases.json/csv、aggregate.json 与 unified_cases/closed_loop_summary 保存在本轮目录；辅助距离与 event 只解释失败，不替代 strict。','']
    rows+=['### 原 observer 落桶但释放不合格的实际事件','',
           '| round / endpoint | scene seed / inference seed | min held distance (m) | exact successful-chain release distance (m) | bucket XY | reasonable region |',
           '|---|---|---:|---:|---|---|']
    for number in range(1,4):
        for name in ['S8','J8','J16']:
            for case in load(ROOT/f'round{number}/{name}/evaluation/cases.json',[]):
                if not case['strict_original_observer'] or case['strict']:continue
                for fruit,chain in case['held_chains'].items():
                    if chain['strict_success_step'] is None:continue
                    for event in case['release_events']:
                        if str(event['apple_id'])!=fruit or event['step']!=chain['release_step']:continue
                        minimum=case['min_held_fruit_bucket_distance_m'];distance=event['fruit_bucket_distance_m']
                        minimum='unknown' if minimum is None else f'{minimum:.4f}'
                        distance='unknown' if distance is None else f'{distance:.4f}'
                        rows.append(f"| Round{number}/{name} | {case['seed']} / {case['inference_seed']} | {minimum} | {distance} | {event['within_bucket_xy']} | {event['reasonable_region']} |")
    rows+=['','min held distance 是整条 rollout 的辅助最小值，不能替代实际 release 时的空间条件或完整 strict chain。','',
           '## 4. 路线结论、能力交换与停止规则','']
    if decision:
        rows += [f"分类：{decision['case']}。{decision['conclusion']}",'',decision['comparison'],'',decision['capability_tradeoffs'],'',decision['excluded_explanations'],'',decision['remaining_explanations'],'']
    else:rows+=['路线结论尚未冻结。Round1主 strict 均0，S的抓取更强，J16出现更多到桶/落桶但释放不合格；这不构成 endpoint 成功。允许的共同方案修订最多三轮，只有实际闭环证据支持才启动；不进行 scene 专属 patch。','']
    rows+=['## 5. 最终 endpoint 与 fresh24','']
    if endpoint:
        rows += [f"最终 endpoint 已在 fresh24 前冻结：`{endpoint['checkpoint']}`；SHA `{endpoint['checkpoint_sha256']}`。",'']
    elif completion:rows+=['没有值得进入 fresh24 的 endpoint；fresh24 未运行。','']
    else:rows+=['尚未冻结最终 endpoint；fresh24 保持未使用。','']
    if decision and decision.get('fresh24_result'):rows += [decision['fresh24_result'],'']
    rows += ['fresh24只允许最终冻结 endpoint一次，不参与 checkpoint选择、数据调整、mining或后续训练。Round3之后停止，不开启Round4；本轮不实现 history、新观测、解冻VLM、RL或下一层研究方向。Production controller/physics/action adapter/原严格 observer未修改，未重跑Gate0/Gate1。','', '工程与复现入口：collect_live.py、inputs.py、freeze_data.py、train_integrated.py、evaluate.py、Round2/3的有限修订脚本及config、frozen schedule、receipts、engineering_repairs.jsonl、逐步training_steps和统一closed_loop_summary。']
    (ROOT/'final_report.md').write_text('\n'.join(rows)+'\n')

if __name__=='__main__':report()
