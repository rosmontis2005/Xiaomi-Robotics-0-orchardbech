"""Local diagnostic runtime; no production source changes."""
import os
import sys
import importlib.util
from pathlib import Path
ROOT = Path(__file__).resolve().parent
XR0 = ROOT.parent
ORCHARD = Path('/home/rosmontis/Projects/orchardbench')
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT), str(XR0), str(ORCHARD)]
for key, relative in {
    'TMPDIR': 'tmp', 'NEWTON_CACHE_PATH': 'cache/newton', 'WARP_CACHE_PATH': 'cache/warp',
    'CUDA_CACHE_PATH': 'cache/cuda', 'TORCHINDUCTOR_CACHE_DIR': 'cache/inductor',
    'TRITON_CACHE_DIR': 'cache/triton', 'MPLCONFIGDIR': 'cache/matplotlib',
    'XDG_CACHE_HOME': 'cache/xdg', 'NUMBA_CACHE_DIR': 'cache/numba',
}.items():
    p = ROOT / relative
    p.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(p)
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                  PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false',
                  OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
if importlib.util.find_spec('newton') is None:
    site = ORCHARD / f'.pixi/envs/default/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages'
    if not site.is_dir():
        raise RuntimeError(f'Missing existing simulator packages: {site}')
    sys.path.append(str(site))
