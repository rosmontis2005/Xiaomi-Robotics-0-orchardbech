#!/usr/bin/env python3
"""Actual RGB audit sheets and independent auxiliary CPU input cache."""
import sys,os,json,hashlib,ast,copy
from pathlib import Path
from types import SimpleNamespace
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent;XR0=ROOT.parent.parent;ORCHARD=Path('/home/rosmontis/Projects/orchardbench');DATA=ORCHARD/'data/orchard_v1_2650/filtered';AB=XR0/'test_128_episode_AB_1003'
sys.path[:0]=[str(XR0),str(ORCHARD)]
for k,rel in {'TMPDIR':'tmp','MPLCONFIGDIR':'cache/matplotlib','XDG_CACHE_HOME':'cache/xdg','HF_HOME':'cache/huggingface','HF_HUB_CACHE':'cache/huggingface/hub','TORCH_HOME':'cache/torch'}.items():
 p=ROOT/rel;p.mkdir(parents=True,exist_ok=True);os.environ[k]=str(p)
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',WANDB_MODE='disabled')
import numpy as np
from PIL import Image,ImageDraw
from decord import VideoReader,cpu
import torch
from scipy.spatial.transform import Rotation
from treesim.orchard_action import EPS,encode_window,action_mask,prepare_rgb,OrchardActionAdapter
from mibot.data.datasets.orchardbench_dataset import OrchardBenchDataset,load_stats,policy_messages
from mibot.data.datamodule.orchardbench_datamodule import OrchardBenchDataModule

def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,obj):
 with (ROOT/name).open('x') as f:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n')
def videos(t):
 out={}
 for view in ['ego','wrist_left']:
  p=Path(t['observations'][view][0]['path']);p=p if p.is_absolute() else DATA/p
  if not p.exists():p=DATA/'videos'/p.name
  v=VideoReader(str(p),ctx=cpu(0),num_threads=1);assert len(v)==t['num_frames'] and abs(v.get_avg_fps()-30)<1e-6;out[view]=v
 return out

def review():
 spec=load(ROOT/'review_24_manifest.json');index=load(ROOT/'event_index.json')['episodes'];sheets=[];checks=[];rows=[]
 for ep in spec['episodes']:
  t=load(ep['annotation']);assert sha(ep['annotation'])==ep['annotation_sha256'];readers=videos(t);images=[];captions=[];e=index['train/'+ep['episode_id']]
  for f in ep['frames']:
   for view in ['ego','wrist_left']:
    raw=readers[view][f].asnumpy();assert raw.shape==(144,192,3);im=prepare_rgb(Image.fromarray(raw));a=np.asarray(im);assert a.shape==(128,192,3) and a.std()>1;images.append(im);captions.append(f"f{f},t={t['orchardbench']['timestamps'][f]:.3f},w={t['proprios']['gripper_pos'][f][0]*1000:.1f}mm {view}");checks.append(dict(episode_id=ep['episode_id'],frame=f,view=view,prepared_sha256=hashlib.sha256(a.tobytes()).hexdigest(),image_std=float(a.std())))
  rows.append((ep,images,captions))
 folder=ROOT/'event_review_sheets';folder.mkdir(exist_ok=False)
 for page,start in enumerate(range(0,len(rows),6),1):
  part=rows[start:start+6];canvas=Image.new('RGB',(1170,30+len(part)*185),'white');draw=ImageDraw.Draw(canvas);draw.text((8,7),'Event proxy: first PULL. Three times: interval start / PULL / interval end; each pair static + wrist. True model crops.',fill='black')
  for r,(ep,ims,caps) in enumerate(part):
   y=30+r*185;draw.text((8,y),f"{ep['episode_id']} PULL={ep['event_time_s']:.3f}s close-target minus PULL={ep['first_close_minus_pull_s']:+.3f}s",fill='black')
   for j,(im,cap) in enumerate(zip(ims,caps)):
    x=8+j*192;draw.text((x,y+19),cap,fill='black');canvas.paste(im,(x,y+36))
  p=folder/f'events_{page:02d}.png';canvas.save(p);sheets.append(str(p))
 save('event_visual_decode.json',dict(status='PASS_DECODE_REVIEW_PENDING',scenes=24,frames=144,sheets=sheets,checks=checks,limitations='Prepared crop shown, no post-crop target segmentation or student held ground truth.'))
 print(json.dumps(dict(event='review_sheets_ready',paths=sheets)),flush=True)

def width_intent_method():
 tree=ast.parse((ORCHARD/'treesim/vla_env.py').read_text());c=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='VLAEnvConfig');names=('gripper_open_width','gripper_open_steps','gripper_close_deadband');const={x.target.id:ast.literal_eval(x.value) for x in c.body if isinstance(x,ast.AnnAssign) and isinstance(x.target,ast.Name) and x.target.id in names};e=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='OrchardVLAEnv');fn=copy.deepcopy(next(x for x in e.body if isinstance(x,ast.FunctionDef) and x.name=='_update_width_intent'));ns={};exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'width_intent_reference','exec'),ns);return const,ns['_update_width_intent']

def cache():
 panel=load(ROOT/'auxiliary_panel.json');stat,mean,std=load_stats(DATA/'action_stats.json');dm=OrchardBenchDataModule(dict(processor_path=str(XR0.parent/'checkpoints/Xiaomi-Robotics-0-Calvin-ABCD_D')));ds={};lookup={};samples=[];parity=None;constants,method=width_intent_method()
 for split in ['train','val']:
  d=OrchardBenchDataset(dict(train_datasets=dict(root=str(DATA),split=split,episode_ids=None,stats_path=str(DATA/'action_stats.json'),action_length=30,batch_size=1)));ds[split]=d;lookup[split]={(str(Path(p).resolve()),f):i for i,(p,f) in enumerate(d.samples)}
 for j,w in enumerate(panel['windows']):
  meta=dict(w);f=w['frame'];t=load(w['annotation']);assert sha(w['annotation'])==w['annotation_sha256'];idx=lookup[w['split']][(str(Path(w['annotation']).resolve()),f)];batch=dm.collate_fn([ds[w['split']][idx]]);gt=encode_window(t,f);assert np.array_equal(batch['action'][0].numpy(),(gt-mean)/(std+EPS));assert torch.equal(batch['action_mask'][0],torch.from_numpy(action_mask()));assert all(isinstance(v,torch.Tensor) and v.device.type=='cpu' for v in batch.values())
  obj=SimpleNamespace(config=SimpleNamespace(**constants),_gripper=0.,_width_open_steps=0,_gripper_width_command=.04)
  for raw in np.asarray(t['actions']['gripper_pos'])[:f,0]:
   width=float(np.clip(raw,0,.08));method(obj,width);obj._gripper_width_command=width
  intent=dict(intent=float(obj._gripper),open_steps=int(obj._width_open_steps),previous_width_command_m=float(obj._gripper_width_command))
  meta.update(dataset_index=idx,anchor_position=t['proprios']['ee_pos'][f],anchor_rotation=np.asarray(t['proprios']['ee_rotm'][f]).reshape(3,3).tolist(),initial_width_intent_reference=intent)
  sample=dict(meta=meta,batch={k:v.contiguous() for k,v in batch.items()},gt_action_physical=torch.from_numpy(gt));samples.append(sample)
  if parity is None:
   p=t['proprios'];obs=dict(tcp_pos_world=np.asarray(p['ee_pos'][f]),tcp_quat_world=Rotation.from_matrix(np.asarray(p['ee_rotm'][f]).reshape(3,3)).as_quat(),joint_pos=np.r_[p['arm_joint'][f],0.,0.],gripper_width=float(p['gripper_pos'][f][0]));readers=videos(t);images=[prepare_rgb(Image.fromarray(readers[v][f].asnumpy())) for v in ['ego','wrist_left']];dep=dict(messages=policy_messages(images),state=torch.from_numpy(OrchardActionAdapter.state(obs)),action=torch.zeros(30,32),action_mask=torch.from_numpy(action_mask()));actual=dm.collate_fn([dep]);expected=dict(batch);expected['action']=torch.zeros_like(expected['action']);assert set(actual)==set(expected);parity={}
   for k in actual:
    a,b=actual[k],expected[k];assert a.shape==b.shape and a.dtype==b.dtype;exact=torch.equal(a,b);assert exact or(k=='state' and torch.allclose(a,b,atol=2e-6,rtol=0));parity[k]=dict(exact_equal=exact,max_difference=float((a.float()-b.float()).abs().max()))
  if (j+1)%24==0:print(json.dumps(dict(event='aux_cache_progress',windows=j+1,total=len(panel['windows']))),flush=True)
 assert not torch.cuda.is_initialized()
 provenance=dict(panel_path=str(ROOT/'auxiliary_panel.json'),panel_sha256=sha(ROOT/'auxiliary_panel.json'),event_index_sha256=sha(ROOT/'event_index.json'),stats_sha256=sha(DATA/'action_stats.json'),old_selection_sha256=panel['provenance']['old_selection_sha256'],normalization_source_fingerprint=stat['source_sha256'],script_sha256=sha(__file__))
 out=ROOT/'aux_eval_cache.pt'
 with out.open('xb') as f:torch.save(dict(schema='orchard_auxiliary_eval_input_cache_v1',provenance=provenance,samples=samples,mean=torch.from_numpy(mean),std=torch.from_numpy(std)),f)
 save('aux_cache_validation.json',dict(status='PASS',windows=len(samples),cache_path=str(out),cache_sha256=sha(out),bytes=out.stat().st_size,provenance=provenance,panel_counts=panel['counts'],original_240_unchanged=True,original_240_cache_reference=str(AB/'eval_cache.pt'),dataset_to_deployment_tensor_parity=parity,all_action_masks_unchanged=True,labels_model_inputs=False,cuda_initialized=False,no_checkpoint_selection=True))
 print(json.dumps(dict(event='aux_cache_ready',path=str(out),bytes=out.stat().st_size)),flush=True)
if __name__=='__main__':
 review();cache()
