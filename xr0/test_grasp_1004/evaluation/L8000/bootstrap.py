"""Isolated evaluation caches; existing training and simulation sources are read-only."""
import importlib.util
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
EXPERIMENT=ROOT.parents[1]
XR0=EXPERIMENT.parent
AB=XR0/'test_128_episode_AB_1003'
ORCHARD=Path('/home/rosmontis/Projects/orchardbench')
sys.dont_write_bytecode=True
sys.path[:0]=[str(ROOT),str(ROOT.parent),str(AB),str(XR0),str(ORCHARD)]
for key,relative in {'TMPDIR':'tmp','NEWTON_CACHE_PATH':'cache/newton','WARP_CACHE_PATH':'cache/warp','CUDA_CACHE_PATH':'cache/cuda','TORCHINDUCTOR_CACHE_DIR':'cache/inductor','TRITON_CACHE_DIR':'cache/triton','MPLCONFIGDIR':'cache/matplotlib','XDG_CACHE_HOME':'cache/xdg','NUMBA_CACHE_DIR':'cache/numba','HF_HOME':'cache/huggingface','HF_HUB_CACHE':'cache/huggingface/hub','TORCH_HOME':'cache/torch'}.items():
 path=ROOT/relative;path.mkdir(parents=True,exist_ok=True);os.environ[key]=str(path)
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',WANDB_MODE='disabled')
if importlib.util.find_spec('newton') is None:
 path=ORCHARD/f'.pixi/envs/default/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages'
 if path.is_dir():sys.path.append(str(path))
