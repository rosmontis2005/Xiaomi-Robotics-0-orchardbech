#!/usr/bin/env python3
"""OrchardBench V1 Closed-Loop Learning Curve Benchmark.

Only the five named weight states are supported. --dry-run is stdlib-only;
--prepare-seeds freezes the reset-only cohort without loading any model.
"""
from __future__ import annotations

import argparse
import ast
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
from importlib.metadata import PackageNotFoundError, version
import json
import logging
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile


XR0 = Path(__file__).resolve().parents[1]
ORCHARD = XR0.parents[2] / "orchardbench"
PRETRAINED = XR0.parent / "checkpoints/Xiaomi-Robotics-0-Calvin-ABCD_D"
CHECKPOINT_DIR = XR0 / ("outputs/orchard_v1_2650/project_orchardbench/"
                        "orchard_v1_2650_frozen_vlm_seed42")
STATS = ORCHARD / "data/orchard_v1_2650/filtered/action_stats.json"
DEFAULT_OUTPUT = XR0 / "outputs/orchard_v1_benchmark"
BENCHMARK = "OrchardBench V1 Closed-Loop Learning Curve Benchmark"
CHECKPOINTS = ("pretrained", "step_1000", "step_3000", "step_6000", "step_10000")
STAGES = ("NO_GRASP", "GRASPED_NOT_DETACHED", "DETACHED_NOT_PLACED", "SUCCESS")
CONTRACT = "orchard_cartesian_local_rotvec_width_v1"
SEED_COUNT, SEED_START, INFERENCE_SEED, CHUNK, BUDGET = 30, 3010000, 42, 30, 900
ENV_OVERRIDES = dict(max_control_steps=BUDGET, detach_force_scale=1.5,
                     grasp_mode="benchmark_assist")
LOG = logging.getLogger("orchard_benchmark")


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=2, allow_nan=False) + "\n"
    if exclusive:
        # Frozen seeds and run manifests must never be silently replaced.
        with path.open("x") as stream:
            stream.write(data)
    else:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(data)
        temporary.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_digest(root):
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def environment_snapshot():
    """Read dataclass defaults without importing Torch, Newton or a renderer."""
    tree = ast.parse((ORCHARD / "treesim/vla_env.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "VLAEnvConfig")
    defaults = {n.target.id: ast.literal_eval(n.value) for n in cls.body
                if isinstance(n, ast.AnnAssign) and n.value is not None}
    # JSON normalization also converts tuple bounds to lists.
    return json.loads(json.dumps(defaults | ENV_OVERRIDES))


def fixed_config():
    prompt_source = XR0 / "mibot/data/datasets/orchardbench_dataset.py"
    tree = ast.parse(prompt_source.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "policy_messages")
    prompt = next(ast.literal_eval(n.value) for n in function.body
                  if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "prompt"
                                                       for t in n.targets))
    stats = read_json(STATS)
    if stats["contract"] != CONTRACT or stats["source_split"] != "train":
        raise ValueError(f"Wrong Orchard 2650 stats contract/source: {STATS}")
    for name in ("mean", "std"):
        rows = stats[name]
        if len(rows) != CHUNK or any(len(row) != 32 for row in rows):
            raise ValueError(f"Stats {name} must have shape [30,32]")
        if any(not math.isfinite(x) or (name == "std" and x <= 0) for row in rows for x in row):
            raise ValueError(f"Invalid stats {name}")
        if any(x != (0 if name == "mean" else 1) for row in rows for x in row[7:]):
            raise ValueError(f"Invalid inactive stats dimensions: {name}")
    for name in ("config.json", "processor_config.json", "preprocessor_config.json",
                 "tokenizer_config.json", "tokenizer.json", "chat_template.jinja"):
        if not (PRETRAINED / name).is_file():
            raise FileNotFoundError(PRETRAINED / name)
    processor_files = {p.name: sha256(p) for p in sorted(PRETRAINED.iterdir())
                       if p.suffix in (".json", ".jinja", ".txt", ".py")}
    return dict(benchmark=BENCHMARK, contract=CONTRACT, action_shape=[CHUNK, 32],
                active_dims="0:7", processor_path=str(PRETRAINED), processor_files=processor_files,
                stats_path=str(STATS), stats_sha256=sha256(STATS), prompt=prompt,
                cameras=["rgb_static", "rgb_wrist"],
                image_preparation="treesim.orchard_action.prepare_rgb (95% center crop)",
                execution="30-step full chunk; command(k, live_obs) on every control step",
                inference_seed=INFERENCE_SEED, environment=environment_snapshot(),
                success_definition="at least one detached apple physically inside the bucket",
                simulator_source_sha256=source_digest(ORCHARD / "treesim"),
                policy_source_sha256=source_digest(XR0 / "mibot"))


def discover_checkpoints(names, directory=CHECKPOINT_DIR, pretrained=PRETRAINED):
    records, errors = [], []
    for name in names:
        if name not in CHECKPOINTS:
            raise ValueError(f"Unsupported checkpoint: {name}")
        if name == "pretrained":
            path = Path(pretrained)
            try:
                index = read_json(path / "model.safetensors.index.json")["weight_map"]
                shards = [path / s for s in sorted(set(index.values()))]
                if not shards or any(not p.is_file() for p in shards):
                    raise FileNotFoundError("Missing HF safetensors shards")
                files = shards
            except (OSError, ValueError, KeyError) as exc:
                errors.append(f"pretrained: {path}: {exc}")
                continue
        else:
            step = int(name.removeprefix("step_"))
            pattern = f"*step={step}.ckpt"
            matches = sorted(p for p in Path(directory).glob(pattern) if p.is_file())
            if len(matches) != 1:
                errors.append(f"{name}: expected exactly one {directory}/{pattern}; "
                              f"found {len(matches)}: {[str(p) for p in matches]}")
                continue
            path, files = matches[0], matches
        records.append(dict(checkpoint=name, checkpoint_path=str(path.resolve()),
                            files=[dict(path=str(p.resolve()), size_bytes=p.stat().st_size,
                                        mtime_ns=p.stat().st_mtime_ns) for p in files]))
    return records, errors


def seed_protocol(config):
    return dict(environment=config["environment"],
                simulator_source_sha256=config["simulator_source_sha256"])


def inspect_seeds(path, config):
    data = read_json(path)
    seeds = data.get("seeds", [])
    if (data.get("schema") != "orchard_benchmark_v1_reset_seeds"
            or len(seeds) != SEED_COUNT or len(set(seeds)) != SEED_COUNT
            or any(type(s) is not int or s < SEED_START for s in seeds)
            or seeds != sorted(seeds)):
        raise ValueError(f"{path}: expected 30 unique, ordered frozen reset seeds >= {SEED_START}")
    if data.get("selection") != "OrchardVLAEnv.reset only" or data.get("protocol") != seed_protocol(config):
        raise ValueError(f"{path}: reset protocol/source mismatch; refusing to replace frozen seeds")
    return data


def configure_runtime():
    # The local XR-0 venv contains Torch; Orchard's same-version Pixi env contains
    # Newton. Append its packages only when Newton is absent, preserving XR-0's
    # package precedence. A unified environment needs no fallback.
    sys.path[:0] = [str(XR0), str(ORCHARD)]
    if importlib.util.find_spec("newton") is None:
        version = f"python{sys.version_info.major}.{sys.version_info.minor}"
        site = ORCHARD / ".pixi/envs/default/lib" / version / "site-packages"
        if not site.is_dir():
            raise RuntimeError("Newton is unavailable; use the Orchard/XR-0 runtime via the shell launcher")
        sys.path.append(str(site))
        LOG.info("Using installed simulator packages: %s", site)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def runtime_versions():
    versions = {}
    for package in ("torch", "numpy", "scipy", "newton", "warp-lang", "mujoco", "mujoco-warp",
                    "transformers", "flash-attn", "matplotlib"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def make_env():
    from treesim.vla_env import OrchardVLAEnv, VLAEnvConfig
    config = VLAEnvConfig(**ENV_OVERRIDES)
    if json.loads(json.dumps(asdict(config))) != environment_snapshot():
        raise RuntimeError("Imported simulator config differs from the frozen source")
    return OrchardVLAEnv(config)


def prepare_seeds(path, config, start, env_factory=make_env):
    if path.exists():
        return inspect_seeds(path, config)
    path.parent.mkdir(parents=True, exist_ok=True)
    seeds, skipped = [], []
    env = env_factory()
    try:
        with path.with_suffix(".scan.jsonl").open("a") as audit:
            seed = start
            while len(seeds) < SEED_COUNT:
                try:
                    env.reset(seed=seed)  # No step, policy, intended target or expert.
                except Exception as exc:
                    row = dict(seed=seed, error_type=type(exc).__name__, error_message=str(exc))
                    # Only scene/reset infeasibility is a skip. Infrastructure
                    # errors abort scanning so they cannot change the cohort.
                    infeasible = isinstance(exc, RuntimeError) and str(exc).startswith(
                        ("No feasible fixed-base stance for seed=", "Home pose IK failed:"))
                    row["status"] = "infeasible" if infeasible else "error"
                    audit.write(json.dumps(row) + "\n")
                    audit.flush()
                    if not infeasible:
                        raise
                    skipped.append(row)
                    LOG.info("Reset infeasible seed=%s: %s", seed, exc)
                else:
                    seeds.append(seed)
                    audit.write(json.dumps(dict(seed=seed, status="feasible")) + "\n")
                    audit.flush()
                    LOG.info("Frozen cohort candidate %s/%s: seed=%s", len(seeds), SEED_COUNT, seed)
                finally:
                    env.close()
                seed += 1
    finally:
        env.close()
    data = dict(schema="orchard_benchmark_v1_reset_seeds", benchmark=BENCHMARK,
                created_utc=datetime.now(timezone.utc).isoformat(), seed_start=start,
                selection="OrchardVLAEnv.reset only", protocol=seed_protocol(config),
                seeds=seeds, skipped=skipped)
    write_json(path, data, exclusive=True)
    return data


def freeze_visual_seeds(output, seed_path, seeds, *, dry_run=False):
    data = dict(selection="first five frozen benchmark seeds, before any model rollout",
                benchmark_seeds_sha256=sha256(seed_path), seeds=seeds[:5])
    path = output / "benchmark_visual_seeds.json"
    if path.exists():
        if read_json(path) != data:
            raise ValueError(f"{path}: frozen visualization seeds mismatch")
    elif not dry_run:
        write_json(path, data, exclusive=True)
    return data


def capability_stage(row):
    if row["success"]:
        return "SUCCESS"
    if row["max_apple_detached_count"] > 0:
        return "DETACHED_NOT_PLACED"
    return "GRASPED_NOT_DETACHED" if row["ever_grasped"] else "NO_GRASP"


def new_episode(checkpoint, seed):
    return dict(checkpoint=checkpoint["checkpoint"], checkpoint_path=checkpoint["checkpoint_path"],
                seed=seed, success=False, capability_stage=None, termination=None,
                control_steps=0, replans=0, ever_grasped=False, max_apple_detached_count=0,
                final_apple_in_bucket_count=0, max_branch_break_count=0,
                ik_failed_steps=0, ik_failed_rate=None, action_clipped_steps=0,
                action_clipped_rate=None, videos={}, video_errors=[])


def observe_info(row, info, *, stepped=False):
    held = any(info.get(key) is not None and info[key] >= 0
               for key in ("held_apple_id_debug", "grasped_apple_id_debug"))
    row["ever_grasped"] |= bool(info.get("grasp_assist_triggered", False) or held)
    row["max_apple_detached_count"] = max(row["max_apple_detached_count"], int(info["apple_detached_count"]))
    row["max_branch_break_count"] = max(row["max_branch_break_count"], int(info["branch_break_count"]))
    row["final_apple_in_bucket_count"] = int(info["apple_in_bucket_count"])
    row["success"] = bool(info["success"])
    if stepped:
        row["ik_failed_steps"] += int(bool(info["ik_failed"]))
        row["action_clipped_steps"] += int(bool(info["action_clipped"]))


def mark_error(row, exc, phase):
    row.update(termination="error", error_type=type(exc).__name__, error_message=str(exc),
               error_phase=phase, last_observed_capability_stage=capability_stage(row),
               success=None, capability_stage=None)
    LOG.error("%s seed=%s %s: %s: %s", row["checkpoint"], row["seed"], phase, type(exc).__name__, exc)


class EpisodeVideo:
    """Buffer only the two small RGB streams for a preselected visual seed."""

    def __init__(self, enabled):
        self.frames = {key: [] for key in ("rgb_static", "rgb_wrist")} if enabled else {}
        self.errors = []

    def capture(self, obs):
        for key in list(self.frames):
            try:
                self.frames[key].append(obs[key].copy())
            except Exception as exc:
                self.errors.append(dict(camera=key, error_type=type(exc).__name__, error_message=str(exc)))
                del self.frames[key]

    def finish(self, row, directory):
        outcome = ("ERROR" if row["termination"] == "error" else
                   "SUCCESS" if row["success"] else f"FAILURE_{row['capability_stage']}")
        for key, frames in self.frames.items():
            if not frames:
                continue
            path = directory / f"{row['checkpoint']}_seed_{row['seed']}_{outcome}_{key.removeprefix('rgb_')}.mp4"
            proc = None
            try:
                if not shutil.which("ffmpeg"):
                    raise FileNotFoundError("ffmpeg not found on PATH")
                directory.mkdir(parents=True, exist_ok=True)
                height, width, channels = frames[0].shape
                if channels != 3:
                    raise ValueError("Expected RGB video frames")
                with tempfile.TemporaryFile() as errors:
                    proc = subprocess.Popen([
                        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                        "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{width}x{height}",
                        "-framerate", "30", "-i", "pipe:0", "-an", "-c:v", "libx264",
                        "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-threads", "1",
                        str(path)], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors)
                    try:
                        for frame in frames:
                            proc.stdin.write(frame.tobytes())
                        proc.stdin.close()
                    except BrokenPipeError:
                        pass
                    code = proc.wait(timeout=30)
                    errors.seek(0)
                    message = errors.read().decode(errors="replace")
                    if code:
                        raise RuntimeError(f"ffmpeg exit={code}: {message}")
                row["videos"][key] = str(path.resolve())
            except Exception as exc:
                self.errors.append(dict(camera=key, error_type=type(exc).__name__, error_message=str(exc)))
                path.unlink(missing_ok=True)
            finally:
                if proc is not None:
                    if proc.poll() is None:
                        proc.kill()
                        proc.wait()
                    if proc.stdin and not proc.stdin.closed:
                        proc.stdin.close()
        row["video_errors"] = self.errors
        for error in self.errors:
            LOG.error("Video error %s seed=%s: %s", row["checkpoint"], row["seed"], error)
        self.frames.clear()


def run_episode(policy, checkpoint, seed, directory, video=False, env_factory=make_env):
    row = new_episode(checkpoint, seed)
    recorder, env, phase = EpisodeVideo(video), None, "reset"
    try:
        env = env_factory()
        obs, info = env.reset(seed=seed)
        observe_info(row, info)
        recorder.capture(obs)
        done = False
        while not done:
            phase = "predict"
            row["replans"] += 1
            chunk = policy.predict(obs, seed=INFERENCE_SEED)
            if tuple(chunk.shape) != (CHUNK, 32):
                raise ValueError(f"Expected action shape [30,32], got {chunk.shape}")
            for k in range(CHUNK):
                phase = "command"
                command = policy.command(k, obs)  # obs is always the latest live observation.
                phase = "step"
                obs, _, terminated, truncated, info = env.step(**command)
                row["control_steps"] += 1
                observe_info(row, info, stepped=True)
                recorder.capture(obs)
                done = bool(terminated or truncated)
                if done:
                    if row["success"]:
                        row["termination"] = "success"
                    elif truncated and row["control_steps"] == BUDGET:
                        row["termination"] = "episode_budget"
                    else:
                        raise RuntimeError("Unexpected environment termination before the 900-step budget")
                    break
                if row["control_steps"] >= BUDGET:
                    raise RuntimeError("Environment did not truncate at the fixed 900-step budget")
            LOG.info("%s seed=%s steps=%s replans=%s", row["checkpoint"], seed,
                     row["control_steps"], row["replans"])
        row["capability_stage"] = capability_stage(row)
    except Exception as exc:
        mark_error(row, exc, phase)
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                if row["termination"] != "error":
                    mark_error(row, exc, "environment_close")
                else:
                    row["cleanup_error"] = f"{type(exc).__name__}: {exc}"
    steps = row["control_steps"]
    for metric in ("ik_failed", "action_clipped"):
        row[f"{metric}_rate"] = row[f"{metric}_steps"] / steps if steps else None
    # Video failures are auxiliary and must never change the task result.
    try:
        recorder.finish(row, directory / "videos")
    except Exception as exc:
        row["video_errors"].append(dict(error_type=type(exc).__name__, error_message=str(exc)))
        LOG.error("Video finalization error %s seed=%s: %s", row["checkpoint"], seed, exc)
    return row


def aggregate(rows, requested):
    valid = [r for r in rows if r["termination"] in ("success", "episode_budget")]
    errors = [r for r in rows if r["termination"] == "error"]
    n = len(valid)
    steps = sum(r["control_steps"] for r in valid)
    result = dict(episodes_requested=requested, episodes_recorded=len(rows), episodes_completed=n,
                  valid_denominator=n, errors=len(errors), total_control_steps=steps,
                  error_control_steps=sum(r["control_steps"] for r in errors),
                  mean_control_steps=statistics.mean(r["control_steps"] for r in valid) if n else None,
                  median_control_steps=statistics.median(r["control_steps"] for r in valid) if n else None,
                  mean_branch_break_count=statistics.mean(r["max_branch_break_count"] for r in valid) if n else None,
                  capability_stages={stage: sum(r["capability_stage"] == stage for r in valid) for stage in STAGES},
                  video_error_count=sum(len(r["video_errors"]) for r in rows))
    for key, count in (("success", sum(bool(r["success"]) for r in valid)),
                       ("grasp", sum(r["ever_grasped"] for r in valid)),
                       ("detach", sum(r["max_apple_detached_count"] > 0 for r in valid))):
        result[f"{key}_count"] = count
        result[f"{key}_rate"] = count / n if n else None
    for key in ("ik_failed", "action_clipped"):
        count = sum(r[f"{key}_steps"] for r in valid)
        result[f"{key}_steps"] = count
        result[f"{key}_rate"] = count / steps if steps else None
    return result


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def write_report(output, summary):
    def percent(value):
        return "N/A" if value is None else f"{value:.2%}"

    lines = [f"# {BENCHMARK}", "", f"Run scope: **{summary['scope']}**.", "",
             "30 frozen feasible seeds; visualization seeds are the frozen first five. "
             "30-step full chunks with live observations; 900 control steps at 30 Hz; "
             "policy inference seed 42; benchmark-assist grasp; strict physical bucket success.", "",
             f"Evaluated seeds (same order for each selected checkpoint): `{summary['evaluated_seeds']}`", "",
             "Rates and capability stages exclude runtime errors. Completed = valid denominator. "
             "Clipping/IK rates use the total returned control steps of valid episodes. "
             "Mean branch breaks uses each valid episode's maximum observed count. "
             "Error episodes retain partial diagnostics in episodes.jsonl.", "",
             "| checkpoint | requested | valid | errors | success | grasp | detach | clipping | IK fail | mean steps |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, values in summary["checkpoints"].items():
        mean = values["mean_control_steps"]
        cells = [name, str(values["episodes_requested"]), str(values["valid_denominator"]), str(values["errors"])]
        cells += [percent(values[f"{k}_rate"]) for k in ("success", "grasp", "detach", "action_clipped", "ik_failed")]
        cells.append("N/A" if mean is None else f"{mean:.1f}")
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "| checkpoint | " + " | ".join(STAGES) + " |", "| --- | ---: | ---: | ---: | ---: |"]
    for name, values in summary["checkpoints"].items():
        counts = values["capability_stages"]
        lines.append("| " + name + " | " + " | ".join(str(counts[s]) for s in STAGES) + " |")
    lines.append("")
    for name, values in summary["checkpoints"].items():
        counts = values["capability_stages"]
        peak = max(counts[s] for s in STAGES[:-1])
        if peak:
            modes = ", ".join(s for s in STAGES[:-1] if counts[s] == peak)
            lines.append(f"- {name}: largest failure stage(s): {modes} ({peak} each).")
    first = summary["checkpoints"].get("pretrained", {}).get("success_rate")
    last = summary["checkpoints"].get("step_10000", {}).get("success_rate")
    if first is not None and last is not None:
        lines += ["", f"Observed success rate: pretrained {percent(first)} → step_10000 {percent(last)}."]
    lines += ["", f"Video errors: {sum(v['video_error_count'] for v in summary['checkpoints'].values())}.",
              "", f"Plot status: {summary.get('plot_status', 'pending')}.", ""]
    if summary["scope"] != "FULL 5 x 30":
        lines += ["**FULL 5 × 30 BENCHMARK WAS NOT RUN. This subset is not the benchmark result.**", ""]
    lines += ["NO TRAINING WAS STARTED. NO CHECKPOINT WAS MODIFIED.", ""]
    (output / "benchmark_report.md").write_text("\n".join(lines))


def write_plots(output, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    names = list(summary["checkpoints"])
    labels = ["pretrained" if n == "pretrained" else f"{int(n.split('_')[1]) // 1000}k" for n in names]
    values = [summary["checkpoints"][n] for n in names]
    for metric, filename, label in (
        ("success_rate", "learning_curve.png", "Success rate (valid episodes)"),
        ("action_clipped_rate", "clipping_rate.png", "Action clipped rate (valid control steps)"),
        ("ik_failed_rate", "ik_failure_rate.png", "IK failure rate (valid control steps)"),
        ("mean_control_steps", "episode_length.png", "Mean control steps (valid episodes)"),
    ):
        fig, ax = plt.subplots(figsize=(7, 4))
        y = [v[metric] if v[metric] is not None else float("nan") for v in values]
        ax.plot(labels, y, marker="o")
        ax.set(xlabel="Checkpoint", ylabel=label, title=summary["scope"])
        if metric.endswith("rate"):
            ax.set_ylim(0, 1)
            ax.yaxis.set_major_formatter(PercentFormatter(1))
        else:
            ax.set_ylim(0, BUDGET)
        ax.grid(axis="y", alpha=.25)
        fig.tight_layout()
        fig.savefig(output / filename, dpi=150)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    bottoms = [0.] * len(values)
    for stage, color in zip(STAGES, ("#9ca3af", "#e6ad49", "#5ba5cf", "#58a56b")):
        heights = [v["capability_stages"][stage] / v["valid_denominator"] if v["valid_denominator"] else 0.
                   for v in values]
        ax.bar(labels, heights, bottom=bottoms, label=stage, color=color)
        bottoms = [a + b for a, b in zip(bottoms, heights)]
    for i, v in enumerate(values):
        ax.text(i, 1.015, f"valid={v['valid_denominator']}; errors={v['errors']}", ha="center", fontsize=8)
    ax.set(xlabel="Checkpoint", ylabel="Fraction of valid episodes", ylim=(0, 1.1), title=summary["scope"])
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.17), ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "capability_funnel.png", dpi=150)
    plt.close(fig)


def load_policy(checkpoint):
    """Do not retain exception tracebacks (and their partially loaded models)."""
    try:
        from mibot.server.orchard_policy import OrchardPolicy
        return OrchardPolicy(checkpoint["checkpoint_path"], str(PRETRAINED), str(STATS)), None
    except Exception as exc:
        return None, dict(error_type=type(exc).__name__, error_message=str(exc), error_phase="model_load")


def evaluate(checkpoints, config, seeds, visuals, output, no_video):
    all_rows, summaries = [], {}
    with (output / "episodes.jsonl").open("x") as root_stream:
        for checkpoint in checkpoints:
            name = checkpoint["checkpoint"]
            directory = output / name
            directory.mkdir(exist_ok=True)
            rows = []
            LOG.info("Loading %s: %s", name, checkpoint["checkpoint_path"])
            policy, load_error = load_policy(checkpoint)
            try:
                with (directory / "episodes.jsonl").open("x") as stream:
                    for seed in seeds:
                        if load_error:
                            row = new_episode(checkpoint, seed)
                            row.update(termination="error", success=None, **load_error)
                            LOG.error("%s seed=%s model load error: %s", name, seed, load_error)
                        else:
                            row = run_episode(policy, checkpoint, seed, directory,
                                              video=not no_video and seed in visuals)
                        rows.append(row)
                        all_rows.append(row)
                        encoded = json.dumps(row, allow_nan=False) + "\n"
                        for dest in (stream, root_stream):
                            dest.write(encoded)
                            dest.flush()
                        write_json(directory / "summary.json", aggregate(rows, len(seeds)))
                        LOG.info("Result %s seed=%s: %s %s", name, seed, row["termination"], row["capability_stage"])
                summaries[name] = aggregate(rows, len(seeds))
            finally:
                del policy
                gc.collect()
                if "torch" in sys.modules:
                    sys.modules["torch"].cuda.empty_cache()
    scope = "FULL 5 x 30" if tuple(summaries) == CHECKPOINTS and len(seeds) == SEED_COUNT else "DEBUG SUBSET"
    summary = dict(benchmark=BENCHMARK, scope=scope, config=config, evaluated_seeds=seeds,
                   visualization_seeds=visuals, checkpoints=summaries,
                   status="completed_with_errors" if any(s["errors"] for s in summaries.values()) else "completed")
    write_csv(output / "episodes.csv", all_rows)
    write_json(output / "summary.json", summary)
    write_report(output, summary)
    try:
        write_plots(output, summary)
        summary["plot_status"] = "complete"
    except Exception as exc:
        summary["plot_status"] = f"error: {type(exc).__name__}: {exc}"
        LOG.error("Plot generation failed: %s", summary["plot_status"])
    write_json(output / "summary.json", summary)
    write_report(output, summary)
    return 1 if summary["status"] != "completed" or summary["plot_status"] != "complete" else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", default=["all"],
                        choices=("all", *CHECKPOINTS), help="Named states; executed in canonical order")
    parser.add_argument("--num-seeds", type=int, default=SEED_COUNT,
                        help="Evaluate the first N frozen seeds; always freeze all 30 before any rollout")
    parser.add_argument("--seed-start", type=int, default=SEED_START)
    parser.add_argument("--seeds-file", type=Path, help="Reuse a frozen 30-seed file produced by this script")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-video", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Read-only paths/discovery/seed inspection; no model or reset")
    mode.add_argument("--prepare-seeds", action="store_true", help="Freeze 30 reset-feasible seeds and 5 visual seeds; no model")
    args = parser.parse_args(argv)
    if not 1 <= args.num_seeds <= SEED_COUNT or args.seed_start < SEED_START:
        parser.error("num-seeds must be 1..30; seed-start must be >=3010000 (outside collection namespaces)")
    if "all" in args.checkpoints and args.checkpoints != ["all"]:
        parser.error("Use --checkpoints all alone")
    names = [n for n in CHECKPOINTS if args.checkpoints == ["all"] or n in args.checkpoints]
    output = args.output.resolve()
    seed_path = (args.seeds_file or output / "benchmark_v1_30seeds.json").resolve()
    config = fixed_config()
    checkpoints, errors = discover_checkpoints(names)
    data = inspect_seeds(seed_path, config) if seed_path.exists() else None
    if args.seeds_file and data is None:
        raise FileNotFoundError(f"Explicit --seeds-file does not exist: {seed_path}")
    visuals = freeze_visual_seeds(output, seed_path, data["seeds"], dry_run=True) if data else None
    preview = dict(config=config, checkpoints=checkpoints, checkpoint_errors=errors,
                   seed_file=str(seed_path), seed_status="frozen; reuse" if data else "missing; reset scan required",
                   frozen_seeds=data["seeds"] if data else None,
                   evaluated_seeds=data["seeds"][:args.num_seeds] if data else None,
                   visualization=visuals, num_seeds=args.num_seeds, seed_start_for_new_scan=args.seed_start,
                   output=str(output), video=not args.no_video)
    print(json.dumps(preview, indent=2), flush=True)
    if errors:
        raise FileNotFoundError("Checkpoint discovery failed:\n" + "\n".join(errors))
    if args.dry_run:
        return 0
    if (output / "checkpoint_manifest.json").exists() or (output / "episodes.jsonl").exists():
        raise FileExistsError(f"Run results already exist in {output}; use a new --output and reuse --seeds-file")
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(output / "benchmark.log")])
    configure_runtime()
    data = prepare_seeds(seed_path, config, args.seed_start)
    # Store an identical local copy when a smoke/subset reuses the formal cohort.
    local_seeds = output / "benchmark_v1_30seeds.json"
    if local_seeds.resolve() != seed_path:
        if local_seeds.exists():
            if local_seeds.read_bytes() != seed_path.read_bytes():
                raise ValueError(f"{local_seeds}: refusing to replace existing frozen seeds")
        else:
            with local_seeds.open("xb") as dest:
                dest.write(seed_path.read_bytes())
    visuals = freeze_visual_seeds(output, local_seeds, data["seeds"])
    if args.prepare_seeds:
        LOG.info("Frozen %s feasible seeds and 5 visual seeds. No model was loaded.", SEED_COUNT)
        return 0
    manifest = dict(benchmark=BENCHMARK, config=config, checkpoints=checkpoints,
                    created_utc=datetime.now(timezone.utc).isoformat(),
                    evaluated_seeds=data["seeds"][:args.num_seeds], visualization_seeds=visuals["seeds"],
                    seed_file_sha256=sha256(local_seeds), script_sha256=sha256(Path(__file__)),
                    python_executable=sys.executable, python_version=sys.version,
                    runtime_versions=runtime_versions(),
                    cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", "all visible"),
                    last_checkpoint_policy="Never substitute last.ckpt for step=10000.ckpt")
    write_json(output / "checkpoint_manifest.json", manifest, exclusive=True)
    return evaluate(checkpoints, config, manifest["evaluated_seeds"], visuals["seeds"], output, args.no_video)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"BENCHMARK ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        sys.exit(2)
