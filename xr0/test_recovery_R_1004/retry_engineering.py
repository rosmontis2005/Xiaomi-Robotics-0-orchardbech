import bootstrap,json,time
from recovery_common import ROOT,append,sha
for folder in sorted((ROOT/'recoveries').iterdir()):
 p=folder/'decision.json'
 if not p.exists():continue
 d=json.loads(p.read_text())
 if d.get('reason')!='engineering_exception' or 'Prefix native command drift' not in d.get('error',''):continue
 archive=folder/f"decision_prefix_assertion_{time.time_ns()}.json";p.replace(archive)
 append(ROOT/'engineering_repairs.jsonl',dict(candidate_id=folder.name,original_failure=str(archive),repair='Intermediate feedback native-command equality is diagnostic; full physical takeover tests unchanged',collect_source_sha256=sha(ROOT/'collect_recovery.py')))
 print('RETRY',folder.name)
