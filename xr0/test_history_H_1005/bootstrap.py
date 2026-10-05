"""Isolated 128-episode A/B runtime paths and caches; production sources stay unchanged."""
import importlib.util
import os
import sys
from pathlib import Path
EXPERIMENT_ROOT = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('H_RUN_DIR',str(EXPERIMENT_ROOT))).resolve()
assert ROOT.is_relative_to(EXPERIMENT_ROOT)
XR0 = Path('/home/rosmontis/Projects/dualsys/Xiaomi-Robotics-0/xr0')
ORCHARD = Path('/home/rosmontis/Projects/orchardbench')
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT), str(XR0), str(ORCHARD)]
for key, relative in {
    'TMPDIR': 'tmp', 'NEWTON_CACHE_PATH': 'cache/newton', 'WARP_CACHE_PATH': 'cache/warp',
    'CUDA_CACHE_PATH': 'cache/cuda', 'TORCHINDUCTOR_CACHE_DIR': 'cache/inductor',
    'TRITON_CACHE_DIR': 'cache/triton', 'MPLCONFIGDIR': 'cache/matplotlib',
    'XDG_CACHE_HOME': 'cache/xdg', 'NUMBA_CACHE_DIR': 'cache/numba',
    'HF_HOME': 'cache/huggingface', 'HF_HUB_CACHE': 'cache/huggingface/hub',
    'TORCH_HOME': 'cache/torch',
}.items():
    path = Path('/home/rosmontis/Projects/dualsys/Xiaomi-Robotics-0/xr0/test_recovery_R_1004') / relative if key in {'NEWTON_CACHE_PATH','WARP_CACHE_PATH','CUDA_CACHE_PATH'} else ROOT / relative
    path.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(path)
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
    PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false',
    OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', WANDB_MODE='disabled')
if importlib.util.find_spec('newton') is None:
    site = ORCHARD / f'.pixi/envs/default/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages'
    if site.is_dir():
        sys.path.append(str(site))

sys.path.append(str(XR0/'test_recovery_R_1004/evaluation_followup/version_03'))
