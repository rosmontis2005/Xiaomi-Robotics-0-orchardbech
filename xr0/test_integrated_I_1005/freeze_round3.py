"""Final common LR contrast; reuse Round2 schedules byte for byte."""
import bootstrap
import json, shutil, time
from recovery_common import ROOT, sha, write, append

def freeze():
    assert json.loads((ROOT/'pipeline_status.json').read_text())['status']=='ROUND2_COMPLETE'
    directory=ROOT/'round3';previous=ROOT/'round2'
    assert not directory.exists(), 'Round3 already prepared; never regenerate'
    evidence={}
    for endpoint in ['S8','J8','J16']:
        path=previous/endpoint/'evaluation/aggregate.json'
        evidence[endpoint]=dict(path=str(path),sha256=sha(path),dev8=json.loads(path.read_text())['dev8'])
    assert all(item['dev8']['strict']==0 for item in evidence.values())
    assert evidence['J16']['dev8']['held15']<evidence['J8']['dev8']['held15']
    configuration=json.loads((previous/'config.json').read_text())
    configuration.update(round=3,lr=5e-6,evidence=evidence,
        rationale='Round2: S8/J8 held15=7/8 and strict0; J16 held15=4/8 and strict0 despite original/live loss decreasing. Test the hypothesis that a smaller update rate preserves acquisition while fitting late supervision. Change ONLY common lr1e-5->5e-6; same D*, same exact Round2 D_A/D_B rows, same budgets and initialization. This is the final allowed round, not a claim that LR caused the failures.',
        reused_fixed_schedules_from_round=2,no_new_data=True,fresh24_used=False,
        maximum_rounds=3,stop_after_this_round=True)
    directory.mkdir();write(directory/'config.json',configuration)
    copied=[]
    for name in ['schedule_D_A.jsonl','schedule_D_B.jsonl','schedule_statistics.json']:
        shutil.copyfile(previous/name,directory/name)
        assert sha(previous/name)==sha(directory/name)
        copied.append(directory/name)
    files=[directory/'config.json',*copied]
    code=[ROOT/name for name in ['freeze_round3.py','train_integrated_next.py','evaluate_next.py','run_next.py']]
    write(directory/'frozen.json',dict(status='FROZEN',round=3,time=time.time(),training_started=False,
        dataset_manifest_sha256=sha(ROOT/'dataset_manifest.jsonl'),original_freeze_sha256=sha(ROOT/'frozen.json'),
        previous_round_freeze_sha256=sha(previous/'frozen.json'),
        files={str(path):sha(path) for path in files+code},
        reused_schedule_sha256={name:sha(previous/name) for name in ['schedule_D_A.jsonl','schedule_D_B.jsonl']}))
    append(ROOT/'round_decisions.jsonl',dict(round=3,reason=configuration['rationale'],configuration=str(directory/'config.json'),frozen_sha256=sha(directory/'frozen.json'),fresh24_used=False,final_allowed_round=True))
    print(json.dumps(dict(round=3,lr=configuration['lr'],schedule_reused=True,stop_after_round3=True)))

if __name__=='__main__':freeze()
