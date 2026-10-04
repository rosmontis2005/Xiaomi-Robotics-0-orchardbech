"""CPU-only protocol regression checks; no real checkpoint, reset or rollout.

Run: python -m unittest discover -s tools -p test_eval_orchard_learning_curve.py
"""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import eval_orchard_learning_curve as bench


CHECKPOINT = dict(checkpoint="pretrained", checkpoint_path="synthetic-test-only")


class FakeEnv:
    def __init__(self, success_at=None, fail_at=None):
        self.success_at, self.fail_at = success_at, fail_at
        self.tick = 0

    def info(self):
        return dict(success=self.tick == self.success_at,
                    apple_detached_count=int(self.tick >= 4), branch_break_count=int(self.tick >= 10),
                    apple_in_bucket_count=int(self.tick == self.success_at),
                    grasp_assist_triggered=self.tick == 1,
                    held_apple_id_debug=0 if self.tick == 2 else None,
                    grasped_apple_id_debug=None, ik_failed=self.tick % 2 == 0,
                    action_clipped=self.tick % 3 == 0)

    def reset(self, *, seed):
        self.tick = 0
        return dict(tick=0), self.info()

    def step(self, action):
        if self.tick == self.fail_at:
            raise RuntimeError("synthetic simulator error")
        self.tick += 1
        return dict(tick=self.tick), 0., self.tick == self.success_at, self.tick == bench.BUDGET, self.info()

    def close(self):
        pass


class LivePolicy:
    def __init__(self):
        self.predictions, self.commands = [], []

    def predict(self, obs, seed):
        self.predictions.append((obs["tick"], seed))
        return SimpleNamespace(shape=(30, 32))

    def command(self, index, obs):
        assert obs["tick"] == len(self.commands), "Stale observation passed to command"
        assert index == len(self.commands) % 30, "Chunk was skipped or replanned early"
        self.commands.append((index, obs["tick"]))
        return dict(action=None)


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.output = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.logging = patch.object(bench.LOG, "disabled", True)
        self.logging.start()
        self.addCleanup(self.logging.stop)

    def rollout(self, **env_kwargs):
        policy = LivePolicy()
        row = bench.run_episode(policy, CHECKPOINT, bench.SEED_START, self.output,
                                env_factory=lambda: FakeEnv(**env_kwargs))
        return row, policy

    def test_full_chunks_live_observation_and_fixed_rng(self):
        row, policy = self.rollout(success_at=31)
        self.assertEqual(policy.predictions, [(0, 42), (30, 42)])
        self.assertEqual([k for k, _ in policy.commands], list(range(30)) + [0])
        self.assertEqual((row["termination"], row["capability_stage"]), ("success", "SUCCESS"))
        self.assertEqual((row["control_steps"], row["replans"]), (31, 2))
        self.assertTrue(row["ever_grasped"])  # The final info contains no held fruit/trigger.
        self.assertEqual(row["max_apple_detached_count"], 1)
        self.assertEqual(row["ik_failed_steps"], 15)
        self.assertEqual(row["action_clipped_steps"], 10)

    def test_fixed_900_step_budget(self):
        row, policy = self.rollout()
        self.assertEqual(row["termination"], "episode_budget")
        self.assertEqual(row["capability_stage"], "DETACHED_NOT_PLACED")
        self.assertEqual((len(policy.commands), len(policy.predictions)), (900, 30))
        self.assertEqual(policy.predictions, [(t, 42) for t in range(0, 900, 30)])

    def test_stage_priority_and_held_zero(self):
        row = bench.new_episode(CHECKPOINT, bench.SEED_START)
        self.assertEqual(bench.capability_stage(row), "NO_GRASP")
        env = FakeEnv()
        env.tick = 2
        bench.observe_info(row, env.info())
        self.assertTrue(row["ever_grasped"])
        self.assertEqual(bench.capability_stage(row), "GRASPED_NOT_DETACHED")
        row["max_apple_detached_count"] = 1
        self.assertEqual(bench.capability_stage(row), "DETACHED_NOT_PLACED")
        row["success"] = True
        self.assertEqual(bench.capability_stage(row), "SUCCESS")

    def test_errors_excluded_and_step_rates_weighted(self):
        success, _ = self.rollout(success_at=31)
        failure, _ = self.rollout()
        error, _ = self.rollout(fail_at=45)
        self.assertEqual(error["termination"], "error")
        self.assertIsNone(error["success"])
        self.assertIsNone(error["capability_stage"])
        self.assertEqual(error["last_observed_capability_stage"], "DETACHED_NOT_PLACED")
        values = bench.aggregate([success, error, failure], 3)
        self.assertEqual((values["valid_denominator"], values["episodes_completed"], values["errors"]), (2, 2, 1))
        self.assertEqual(values["success_rate"], .5)
        self.assertEqual(values["grasp_rate"], 1.)
        self.assertEqual(values["detach_rate"], 1.)
        self.assertEqual(values["total_control_steps"], 931)
        self.assertEqual(values["error_control_steps"], 45)
        self.assertAlmostEqual(values["ik_failed_rate"], 465 / 931)
        self.assertAlmostEqual(values["action_clipped_rate"], 310 / 931)
        self.assertEqual(sum(values["capability_stages"].values()), 2)
        empty = bench.aggregate([error], 1)
        self.assertIsNone(empty["success_rate"])
        self.assertIsNone(empty["ik_failed_rate"])

    def test_prediction_and_video_errors(self):
        with patch.object(LivePolicy, "predict", side_effect=FloatingPointError("synthetic nonfinite prediction")):
            row, _ = self.rollout()
        self.assertEqual(row["termination"], "error")
        self.assertEqual(row["error_phase"], "predict")
        self.assertEqual(row["control_steps"], 0)
        with patch.object(bench.EpisodeVideo, "finish", side_effect=OSError("synthetic encoder failure")):
            row, _ = self.rollout(success_at=1)
        self.assertTrue(row["success"])
        self.assertEqual(row["termination"], "success")
        self.assertEqual(len(row["video_errors"]), 1)

    def test_discovery_missing_duplicate_and_epoch_independence(self):
        # Glob/stat mocks only: never create or alter a checkpoint on disk.
        records, errors = bench.discover_checkpoints(["step_1000"], directory=self.output)
        self.assertFalse(records)
        self.assertIn("found 0", errors[0])
        paths = [self.output / f"epoch={n}-step=1000.ckpt" for n in (73, 81)]
        with patch.object(Path, "glob", return_value=paths), patch.object(Path, "is_file", return_value=True):
            records, errors = bench.discover_checkpoints(["step_1000"], directory=self.output)
        self.assertFalse(records)
        self.assertIn("found 2", errors[0])
        with (patch.object(Path, "glob", return_value=paths[:1]) as glob,
              patch.object(Path, "is_file", return_value=True),
              patch.object(Path, "stat", return_value=SimpleNamespace(st_size=7, st_mtime_ns=9))):
            records, errors = bench.discover_checkpoints(["step_1000"], directory=self.output)
        self.assertFalse(errors)
        glob.assert_called_once_with("*step=1000.ckpt")
        self.assertEqual(records[0]["checkpoint_path"], str(paths[0]))

    def test_seed_freezing_reuse_and_visual_prefix(self):
        config = dict(environment={"max_control_steps": 900}, simulator_source_sha256="synthetic")
        path = self.output / "benchmark_v1_30seeds.json"
        attempted = []

        class ResetOnly:
            def reset(self, *, seed):
                attempted.append(seed)
                if seed == bench.SEED_START:
                    raise RuntimeError(f"No feasible fixed-base stance for seed={seed}: test fixture")

            def close(self):
                pass

        data = bench.prepare_seeds(path, config, bench.SEED_START, ResetOnly)
        self.assertEqual(data["seeds"], list(range(bench.SEED_START + 1, bench.SEED_START + 31)))
        self.assertEqual(len(attempted), 31)
        self.assertEqual(data["skipped"][0]["status"], "infeasible")
        before = path.read_bytes()
        again = bench.prepare_seeds(path, config, bench.SEED_START + 900,
                                   lambda: self.fail("Existing seed file must not reset or scan"))
        self.assertEqual(again, data)
        self.assertEqual(path.read_bytes(), before)
        visual = bench.freeze_visual_seeds(self.output, path, data["seeds"])
        self.assertEqual(visual["seeds"], data["seeds"][:5])
        with self.assertRaisesRegex(ValueError, "visualization seeds mismatch"):
            bench.freeze_visual_seeds(self.output, path, list(reversed(data["seeds"])))
        changed = config | {"simulator_source_sha256": "changed"}
        with self.assertRaisesRegex(ValueError, "protocol/source mismatch"):
            bench.inspect_seeds(path, changed)

    def test_scanning_infrastructure_error_is_not_a_skip(self):
        class BrokenReset:
            def reset(self, *, seed):
                raise RuntimeError("CUDA out of memory")

            def close(self):
                pass

        path = self.output / "seeds.json"
        with self.assertRaisesRegex(RuntimeError, "CUDA out of memory"):
            bench.prepare_seeds(path, {}, bench.SEED_START, BrokenReset)
        self.assertFalse(path.exists())
        self.assertEqual(json.loads(path.with_suffix(".scan.jsonl").read_text())["status"], "error")

    def test_load_failure_records_all_seeds_and_continues(self):
        checkpoints = [CHECKPOINT, CHECKPOINT | {"checkpoint": "step_1000"}]
        error = dict(error_type="RuntimeError", error_message="synthetic load failure", error_phase="model_load")
        with patch.object(bench, "load_policy", return_value=(None, error)) as load, patch.object(bench, "write_plots"):
            code = bench.evaluate(checkpoints, {}, [bench.SEED_START, bench.SEED_START + 1], [], self.output, True)
        self.assertEqual(code, 1)
        self.assertEqual(load.call_count, 2)
        rows = [json.loads(s) for s in (self.output / "episodes.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 4)
        summary = json.loads((self.output / "summary.json").read_text())
        self.assertEqual(summary["scope"], "DEBUG SUBSET")
        for values in summary["checkpoints"].values():
            self.assertEqual(values["errors"], 2)
            self.assertEqual(values["valid_denominator"], 0)
            self.assertIsNone(values["success_rate"])
        self.assertIn("FULL 5 × 30 BENCHMARK WAS NOT RUN", (self.output / "benchmark_report.md").read_text())


if __name__ == "__main__":
    unittest.main()
