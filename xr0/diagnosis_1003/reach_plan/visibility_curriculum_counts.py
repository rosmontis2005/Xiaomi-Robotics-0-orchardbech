"""Derive exact raw-visibility curriculum counts without filtering any dataset."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib
import json
HERE=Path(__file__).resolve().parent
SOURCE=HERE/'dataset_phase_visibility_audit.json'
OUTPUT=HERE/'visibility_curriculum_counts.json'
if OUTPUT.exists():raise FileExistsError(OUTPUT)
raw=SOURCE.read_bytes()
d=json.loads(raw)
out={}
for split in ['train','val']:
    rows=[r for r in d['episodes'] if r['split']==split]
    n=len(rows)
    s=d['split_summaries'][split]['initial_and_grasp_visibility']['initial']['best_view']['distribution']
    counts={}
    for t in [2,16,64,128]:
        weak=[r for r in rows if r['visibility']['initial']['best_view']<=t]
        easy=[r for r in rows if r['visibility']['initial']['best_view']>=t]
        counts[str(t)]={'at_most_count':len(weak),'at_most_fraction':len(weak)/n,
            'at_least_count':len(easy),'at_least_fraction':len(easy)/n,
            'at_least_and_zero_replay_IK_and_dwell':sum(r['replay']['IK_failed_steps']==0 and r['replay']['dwell_timeout_count']==0 for r in easy)}
    special=[r for r in rows if r['visibility']['initial']['wrist']==0 and 0<r['visibility']['initial']['static']<=2]
    initial_weak=[r for r in rows if r['visibility']['initial']['best_view']<=2]
    out[split]={'episodes':n,'best_view_quantiles':{k:s[k] for k in ['minimum','p05','p25','median','p75','p95']},
        'thresholds':counts,'wrist_zero_static_1_or_2_count':len(special),'wrist_zero_static_1_or_2_fraction':len(special)/n,
        'initial_best_view_at_most2_later_grasp_entry_over50_count':sum(r['visibility']['grasp_entry']['best_view']>50 for r in initial_weak)}
result={'schema':'orchard_visibility_curriculum_counts_v1','source':str(SOURCE),
    'source_sha256':hashlib.sha256(raw).hexdigest(),'split_summaries':out,
    'diagnostic_cases':[{k:r[k] for k in ['seed','visibility']} for r in d['episodes'] if r['seed'] in [2010027,2010042,2010067]],
    'limits':[
        'No data is removed or recollected by this diagnostic. The existing train/val split is unchanged.',
        'The <= and >= columns both include the threshold value; their counts need not be complementary.',
        'Raw visibility is for the privileged expert-selected fruit only. It does not establish post-crop visibility, learned perceptual observability, or absence of other graspable visible fruit.',
        'Thresholds are candidate training curricula and evaluation strata, not permission to discard natural new-seed test outcomes.',
        'The generic pick-an-apple task may admit multiple targets; disagreement with one expert trajectory alone is not proof of an invalid action.'
    ]}
with OUTPUT.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(OUTPUT)
