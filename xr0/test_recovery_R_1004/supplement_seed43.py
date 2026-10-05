"""Bounded seed43 supplement only for the missing R3 category."""
import bootstrap,json,sys,subprocess,os
from collections import Counter
from recovery_common import ROOT,PROTOCOL,write
from mine_student import mine
from recovery_common import make_policy
rows=[json.loads(p.read_text()) for p in (ROOT/'recoveries').glob('*/decision.json')];counts=Counter(r['category'] for r in rows if r['accepted']);assert counts['R1']==12 and counts['R2']==6 and counts['R3']<6
# Only original train episodes already demonstrated actual R3 with seed42.
ids=['episode_000613','episode_000974','episode_000703','episode_001606']
scenes=[next(e for e in PROTOCOL['source_episodes'] if e['episode_id']==i) for i in ids]
write(ROOT/'seed43_supplement_protocol.json',dict(status='AUTHORIZED_SCOPE',reason='R3 seed42 pool accepted4/6; preserve all failures; fixed seed43 supplement on four original B train sources with actual seed42 R3 behavior, selected before seed43 outcomes',source_episode_ids=ids,inference_seed=43,additional_seed_count=1,category_to_collect='R3',max_scenes=4,acceptance_unchanged=True))
policy=make_policy()
for scene in scenes:mine(policy,scene,43)
policy=None
import gc,torch
gc.collect();torch.cuda.empty_cache()
subprocess.run([sys.executable,'-u','-B',str(ROOT/'collect_recovery.py'),'--per-category','6','--category','R3'],cwd=ROOT,check=True)
write(ROOT/'seed43_supplement_complete.json',dict(status='COMPLETE',pid=os.getpid()))
