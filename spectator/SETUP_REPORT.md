# Xiaomi Robotics 0 CALVIN GUI Setup Report

Repository commit: 7c316604d43357d10b06d11fae4c39290c447da5 (7c31660 Add Papers with Code badges)
GPU: NVIDIA GeForce RTX 5060 Ti, 16311 MiB
NVIDIA driver: 580.173.02

Conda env 1: xr0-mibot
Python: 3.12.13
PyTorch: 2.8.0+cu128
CUDA runtime: 12.8
Transformers: 4.57.1
FlashAttention: 2.8.3

Conda env 2: xr0-calvin
Python: 3.10.20
Torch: 1.13.1+cu117
torchvision: 0.14.1+cu117
NumPy: 1.23.0
Hydra: 1.1.1
PyTorch Lightning: 1.8.6
CALVIN commit: fa03f01f19c65920e18cf37398a9ce859274af76
calvin_env commit: 797142c588c21e76717268b7b430958dbd13bf48

Dataset: /home/rosmontis/Projects/dualsys/calvin/dataset/task_D_D (read-only use, 168G)
Checkpoint HF ID: XiaomiRobotics/Xiaomi-Robotics-0-Calvin-ABCD_D
Local checkpoint path: /home/rosmontis/Projects/dualsys/Xiaomi-Robotics-0/checkpoints/Xiaomi-Robotics-0-Calvin-ABCD_D
Checkpoint size: 8.9G
Model split / task id: abcd / calvin_abcd_orig
D_D verification: no public XiaomiRobotics/Xiaomi-Robotics-0-Calvin-D_D; HfApi public enumeration returned only ABC_D and ABCD_D. The downloaded processor supports calvin_abcd_orig and raises KeyError for calvin_d_orig.

Server command: spectator/run_server.sh
GUI command: spectator/run_calvin_gui.sh
Server endpoint: localhost:10086

GUI opened: yes; native X11 Direct GLX, OpenGL 3.3, NVIDIA renderer
Server connected: yes
Sequence started: yes
Sequence: turn_on_led -> open_drawer -> move_slider_right -> lift_blue_block_table -> place_in_slider
Completed subtasks: 5/5
Failure subtask if any: none
Server inference requests: 34
GIF directory: /home/rosmontis/Projects/dualsys/Xiaomi-Robotics-0/logs/calvin_gui/visualize
BF16 GPU memory after inference: 9520 MiB across server compute processes (9276 MiB parent + 244 MiB child); total board use was 9877 MiB including desktop.

Files modified:
- eval_calvin/main.py: minimal opt-in GUI/backend/sleep patch.
- eval_calvin/flower_vla_calvin/pyhash-0.9.3/src/pybind11/include/pybind11/attr.h: added <cstdint> for current GCC compatibility.

Files created:
- checkpoints/Xiaomi-Robotics-0-Calvin-ABCD_D/
- eval_calvin/calvin/
- eval_calvin/flower_vla_calvin/
- logs/calvin_gui/
- spectator/run_server.sh
- spectator/run_calvin_gui.sh
- spectator/SETUP_REPORT.md

Known warnings/issues:
- libxrender1 runtime is installed; libxrender-dev is not installed because sudo requires a password. Manual optional command: sudo apt-get update && sudo apt-get install -y libxrender1 libxrender-dev
- CALVIN emits the expected legacy Gym maintenance warning.
- Full calvin_models dependency installation failed at training-only MulticoreTSNE CMake generation. Installed calvin_models --no-deps and pinned evaluator runtime dependencies manually; required imports and the complete GUI sequence passed.
- pip check intentionally reports omitted training-only cmake, MulticoreTSNE, plotly, sentence-transformers, and wandb.
- Xiaomi requires networkx 3.4.2, while tacto urdfpy metadata pins networkx 2.2; native GUI and the complete sequence passed with 3.4.2.
- The official result-pickle save path rereads unshuffled sequences after evaluation, so its stored sequence metadata does not match the shuffled debug sequence. evaluator.log and the five GIF filenames record the actually executed sequence above.
- No push or commit was performed.
