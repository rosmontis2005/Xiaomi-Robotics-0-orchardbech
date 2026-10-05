import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('condition',choices=['H0','Hrepeat','Hhistory','Hanchor']);p.add_argument('--updates',type=int,choices=[4000,8000],required=True);a=p.parse_args();root=Path(__file__).resolve().parent;folder=root/a.condition;folder.mkdir(exist_ok=True)
for name in ['protocol.json','recovery_manifest.jsonl','data_audit.json']:
 target=folder/name
 if target.exists():assert target.read_bytes()==(root/name).read_bytes()
 else:target.write_bytes((root/name).read_bytes())
schedule=root/f'schedule_H_{a.updates}.jsonl';target=folder/'schedule_R.jsonl'
if target.exists():assert target.read_bytes()==schedule.read_bytes()
else:target.write_bytes(schedule.read_bytes())
print(json.dumps(dict(condition=a.condition,updates=a.updates,schedule=str(target))))
