import bootstrap,json
from collections import Counter
from recovery_common import ROOT,write,protection_check
rows=[json.loads(p.read_text()) for p in sorted((ROOT/'recoveries').glob('*/decision.json'))];good=[r for r in rows if r['accepted']];counts=Counter(r['category'] for r in good)
assert all(counts[c]>=2 for c in ['R1','R2','R3']),counts
checks=[]
for r in good:
 directory=ROOT/'recoveries'/r['candidate_id'];teacher=json.loads((directory/'teacher_result.json').read_text());oracle=json.loads((directory/'oracle_result.json').read_text())
 assert teacher['status']==oracle['status']=='PASS' and teacher['strict']['strict_success'] and oracle['strict']['strict_success']
 assert teacher['initialization_audit']['physics_unchanged'] and oracle['same_student_controller'] and oracle['total_episode_steps']<=900
 checks.append(dict(candidate_id=r['candidate_id'],category=r['category'],teacher_strict_success=True,oracle_strict_success=True,teacher_no_state_reset=True,takeover_regeneration=oracle['takeover_physics_max_absolute_differences'],oracle_total_episode_steps=oracle['total_episode_steps']))
protection_check();write(ROOT/'mechanism_gate.json',dict(status='PASS',category_counts=dict(counts),teacher_and_student_controller_recovery_pass=True,checks=checks));print('Mechanism PASS',dict(counts))
