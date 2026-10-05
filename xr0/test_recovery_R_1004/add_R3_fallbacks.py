"""One bounded alternative actual late-stall start, all selection revisions logged."""
import bootstrap,json
from recovery_common import ROOT,append,write
for folder in sorted((ROOT/'student_sources').iterdir()):
 chosen=json.loads((folder/'selected.json').read_text());r3=[c for c in chosen if c['category']=='R3']
 if len(r3)!=1:continue
 if (ROOT/'recoveries'/r3[0]['candidate_id']/'decision.json').exists() and json.loads((ROOT/'recoveries'/r3[0]['candidate_id']/'decision.json').read_text())['accepted']:continue
 eligible=json.loads((folder/'eligible_candidates.json').read_text())['R3'];used={c['step'] for c in chosen};first=r3[0]['step']
 alternative=[c for c in eligible if c['step'] not in used and c['step']<=600 and abs(c['step']-first)>=30]
 if not alternative:continue
 alternative.sort(key=lambda c:(c['step']))
 c=dict(alternative[0]);c.update({k:r3[0][k] for k in ['source_episode_id','scene_seed','inference_seed','source_folder']});c['candidate_id']=f"{c['source_episode_id']}_rng{c['inference_seed']}_R3_s{c['step']:04d}"
 chosen.append(c);write(folder/'selected.json',chosen);append(ROOT/'candidate_selection_revisions.jsonl',dict(source=str(folder),original_candidate=r3[0]['candidate_id'],added_candidate=c['candidate_id'],reason='Bounded second actual R3 state after long-prefix failure or insufficient remaining900-step budget; earliest qualifying state<=600 separated>=30steps; physical and success gates unchanged'))
 print('ADDED',c['candidate_id'])
