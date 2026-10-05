"""One evidence-driven common sampling revision over the unchanged frozen D*."""
import bootstrap
import json, time
from collections import Counter, defaultdict
import numpy as np
from recovery_common import ROOT, sha, write, append
from freeze_data import draw

def freeze():
    directory = ROOT / 'round2'
    assert not directory.exists(), 'Round2 already prepared; never regenerate its schedules'
    evidence = {}
    for endpoint in ['S8', 'J8', 'J16']:
        path = ROOT / 'round1' / endpoint / 'evaluation' / 'aggregate.json'
        evidence[endpoint] = dict(path=str(path), sha256=sha(path), dev8=json.loads(path.read_text())['dev8'])
    assert all(item['dev8']['strict'] == 0 for item in evidence.values())
    # Five percentage points move from original to genuine current late states.
    quotas = dict(original_early=1100, original_grasp_pull=1760,
                  original_transport=1100, original_drop_done=440,
                  existing_R1=600, existing_R2=300, existing_R3=300,
                  live_onset=360, live_transport=840, live_release=960, live_settle=240)
    seeds = dict(D_A=100511, D_B=100512)
    configuration = dict(round=2, source_ratios=dict(original=.55, existing_R=.15, live_late=.30),
        original_phase_ratios=dict(early=.25, grasp_pull=.40, transport=.25, drop_done=.10),
        live_phase_ratios=dict(onset=.15, transport=.35, release=.40, settle=.10),
        quotas=quotas, schedule_seeds=seeds, updates_per_schedule=8000, lr=1e-5,
        rationale='Round1 strict0 for all endpoints; S8 held7 but stall/release failures; J8/J16 held5, J16 arrival2/stable2 but reasonable release0. Preserve R, slightly raise original grasp exposure and true late transport/release exposure for BOTH routes.',
        evidence=evidence, dataset_manifest_sha256=sha(ROOT/'dataset_manifest.jsonl'),
        original_freeze_sha256=sha(ROOT/'frozen.json'),
        initialization='S=B8000 fresh AdamW; J=full10k fresh AdamW; J8->J16 continuous optimizer',
        no_new_data=True, fresh24_used=False)
    manifest = [json.loads(line) for line in (ROOT/'dataset_manifest.jsonl').read_text().splitlines()]
    pools = defaultdict(list)
    for item in manifest:
        source = item['source']; episode = item['episode_id'] if source == 'original' else item['candidate_id']
        if source == 'existing_R':
            groups = {'existing_'+item['category']: item['legal_anchors']}
        elif source == 'original':
            groups = {'original_'+group: frames for group,frames in item['groups'].items()}
        else: groups = item['groups']
        for group, frames in groups.items():
            for frame in frames:
                pools[group].append(dict(source=source, episode_id=episode,
                    source_episode_id=item['source_episode_id'], frame=frame, group=group,
                    annotation=item['annotation'], annotation_sha256=item['annotation_sha256'],
                    window_id=f'{source}/{episode}/frame{frame:04d}'))
    directory.mkdir()
    write(directory/'config.json', configuration)
    statistics = {}
    for name,seed in seeds.items():
        rng = np.random.default_rng(seed); rows=[]
        for group,quota in quotas.items(): rows.extend(draw(pools[group],quota,rng))
        rows=[rows[int(i)] for i in rng.permutation(len(rows))]
        for step,row in enumerate(rows,1): row['step']=step
        assert len(rows)==8000 and Counter(row['source'] for row in rows)==dict(original=4400,existing_R=1200,live_late=2400)
        with (directory/f'schedule_{name}.jsonl').open('x') as stream:
            for row in rows: stream.write(json.dumps(row)+'\n')
        statistics[name]=dict(seed=seed,updates=8000,source_counts=dict(Counter(row['source'] for row in rows)),group_counts=dict(Counter(row['group'] for row in rows)),unique_windows=len({row['window_id'] for row in rows}))
    write(directory/'schedule_statistics.json', statistics)
    files=[directory/'config.json', directory/'schedule_statistics.json', *[directory/f'schedule_{name}.jsonl' for name in seeds]]
    code=[ROOT/name for name in ['freeze_round2.py','train_integrated_next.py','evaluate_next.py','run_next.py']]
    write(directory/'frozen.json',dict(status='FROZEN',round=2,time=time.time(),training_started=False,
        dataset_manifest_sha256=sha(ROOT/'dataset_manifest.jsonl'),original_freeze_sha256=sha(ROOT/'frozen.json'),
        files={str(path):sha(path) for path in files+code}))
    append(ROOT/'round_decisions.jsonl',dict(round=2,reason=configuration['rationale'],configuration=str(directory/'config.json'),frozen_sha256=sha(directory/'frozen.json'),fresh24_used=False))
    print(json.dumps(statistics))

if __name__=='__main__':freeze()
