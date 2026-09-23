import copy
import io
import json
import shlex
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pipeline.config import Setup
from pipeline.job_manager import JobManager, JobResult, submit_stage
from pipeline.main import DEFAULT_SETUP, build_parser, main


class StageJobTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.repository = self.root / "repo with spaces"
        (self.repository / "pipeline").mkdir(parents=True)
        (self.repository / "pipeline" / "main.py").touch()
        (self.repository / "pipeline" / "dlc_sbatch_superanimal.sh").touch()
        self.session = self.root / "session with spaces"
        self.session.mkdir()
        self.sandbox = self.root / "sandbox"
        self.sandbox.mkdir()
        self.data = copy.deepcopy(Setup.load(DEFAULT_SETUP).data)
        self.data["hosts"]["hpc"].update(repository_dir=str(self.repository),
                                          sandbox=str(self.sandbox))
        self.config = self.root / "custom setup.json"
        self.config.write_text(json.dumps(self.data), encoding="utf-8")

    def test_every_stage_has_job_flag(self):
        for stage in ("run", "prepare", "downsample", "calibrate", "labels", "evaluate",
                      "pose", "convert", "filter", "triangulate", "label-3d"):
            self.assertTrue(build_parser().parse_args([stage, str(self.session), "--job"]).job)

    def test_cpu_submission_quotes_worker_and_omits_gpu_resources(self):
        arguments = [str(self.session), "--config", str(self.config), "--environment", "hpc"]
        with patch("pipeline.job_manager._is_windows", return_value=False), \
             patch("pipeline.job_manager.subprocess.run", return_value=SimpleNamespace(stdout="123;cluster\n")) as run:
            result = submit_stage(self.data, "calibrate", arguments)
        command = run.call_args.args[0]
        self.assertNotIn("--partition", command)
        self.assertNotIn("--gres", command)
        script = run.call_args.kwargs["input"]
        worker = shlex.split(script.split("exec ", 1)[1])
        self.assertEqual(worker[1:], ["-m", "pipeline.main", "calibrate", *arguments])
        self.assertNotIn("--job", worker)
        self.assertEqual(result.job_id, "123")
        self.assertTrue(result.output_log.endswith("calibrate-123.out"))
        self.assertTrue((self.repository / "logs" / "submission-123.json").exists())

    def test_cpu_resources_and_each_submission_is_new(self):
        self.data["slurm"]["cpu"] = {"partition": "cpu", "cpus": 8, "memory": "32G", "time": "02:00:00"}
        state = self.session / "pipeline_state.json"
        state.write_text('{"pose_job_id": "99"}')
        with patch("pipeline.job_manager._is_windows", return_value=False), \
             patch("pipeline.job_manager.subprocess.run", return_value=SimpleNamespace(stdout="124")) as run:
            for _ in range(2):
                submit_stage(self.data, "triangulate", [str(self.session)])
        self.assertEqual(run.call_count, 2)
        command = run.call_args.args[0]
        for flag, value in (("--partition", "cpu"), ("--cpus-per-task", "8"),
                            ("--mem", "32G"), ("--time", "02:00:00")):
            self.assertEqual(command[command.index(flag) + 1], value)
        self.assertEqual(json.loads(state.read_text()), {"pose_job_id": "99"})

    def test_early_dispatch_and_forwarding(self):
        for stage, extras in (("run", []), ("prepare", []),
                              ("evaluate", ["--labels", str(self.root / "manual labels")]),
                              ("labels", ["--frame-index", "27", "--dlc-project", str(self.root / "project")])):
            with self.subTest(stage=stage), \
                 patch("pipeline.main.submit_stage", return_value=JobResult("submitted", "", "123")) as submit, \
                 patch("pipeline.main.FileHandler", side_effect=AssertionError("processing on login node")), \
                 patch("pipeline.main.Setup.materialize", side_effect=AssertionError("materialized on login node")), \
                 redirect_stdout(io.StringIO()):
                main([stage, str(self.session), "--job", "--environment", "hpc",
                      "--config", str(self.config), "--experiment-dir", str(self.root), *extras])
                args = submit.call_args.args[2]
                self.assertEqual(args[0], str(self.session))
                self.assertEqual(args[args.index("--config") + 1], str(self.config))
                self.assertEqual(args[args.index("--experiment-dir") + 1], str(self.root))
                self.assertNotIn("--job", args)
                if stage == "run":
                    self.assertIn("--submit", args)
                for extra in extras:
                    self.assertIn(extra, args)

    def test_gpu_uses_custom_paths_and_preserves_state(self):
        state = self.session / "pipeline_state.json"
        state.write_text('{"other": 42}')
        with patch("pipeline.job_manager._is_windows", return_value=False), \
             patch("pipeline.job_manager.subprocess.run", return_value=SimpleNamespace(stdout="125")) as run, \
             patch("pipeline.main.FileHandler", side_effect=AssertionError("processing")), \
             redirect_stdout(io.StringIO()) as output:
            main(["pose", str(self.session), "--job", "--environment", "hpc", "--config", str(self.config)])
        command = run.call_args.args[0]
        self.assertEqual(command[-1], str(self.session))
        exports = next(arg for arg in command if arg.startswith("--export="))
        self.assertIn(f"PIPELINE_SETUP={self.config}", exports)
        self.assertEqual(json.loads(state.read_text()), {"other": 42, "pose_job_id": "125"})
        self.assertIn("squeue -j 125", output.getvalue())
        self.assertIn("scancel 125", output.getvalue())
        self.assertIn("Output log:", output.getvalue())

    def test_failed_submissions_do_not_record_jobs(self):
        for error in (FileNotFoundError(), subprocess.CalledProcessError(1, "sbatch", stderr="denied")):
            with patch("pipeline.job_manager._is_windows", return_value=False), \
                 patch("pipeline.job_manager.subprocess.run", side_effect=error), \
                 self.assertRaises(RuntimeError):
                submit_stage(self.data, "prepare", [str(self.session)])
        self.assertFalse(list((self.repository / "logs").glob("submission-*.json")))

    def test_gpu_failure_does_not_change_state(self):
        state = self.session / "pipeline_state.json"
        state.write_text('{"other": 42}')
        manager = JobManager(self.root, self.data, self.session, setup_path=self.config)
        with patch("pipeline.job_manager._is_windows", return_value=False), \
             patch("pipeline.job_manager.subprocess.run", return_value=SimpleNamespace(stdout="invalid")), \
             self.assertRaises(RuntimeError):
            manager.submit()
        self.assertEqual(json.loads(state.read_text()), {"other": 42})

    def test_worker_command_executes_stage_without_resubmission(self):
        with patch("pipeline.job_manager._is_windows", return_value=False), \
             patch("pipeline.job_manager.subprocess.run", return_value=SimpleNamespace(stdout="126")) as run:
            submit_stage(self.data, "downsample", [str(self.session), "--config", str(self.config),
                                                   "--environment", "hpc"])
        worker = shlex.split(run.call_args.kwargs["input"].split("exec ", 1)[1])
        with patch("pipeline.main.submit_stage", side_effect=AssertionError("recursive submission")), \
             patch("pipeline.main.VideoProcessor.downsample", return_value=[]) as downsample, \
             redirect_stdout(io.StringIO()):
            main(worker[3:])
        downsample.assert_called_once_with(self.session)

    def test_config_validates_cpu_resources_and_python(self):
        for cpu in ({"cpus": 0}, {"cpus": True}, {"memory": 32}, {"unknown": "x"}):
            self.data["slurm"]["cpu"] = cpu
            self.config.write_text(json.dumps(self.data))
            with self.assertRaises(ValueError):
                Setup.load(self.config)
        self.data["slurm"]["cpu"] = {}
        self.data["hosts"]["hpc"]["python_executable"] = "relative/python"
        self.config.write_text(json.dumps(self.data))
        with self.assertRaises(ValueError):
            Setup.load(self.config)

    def test_configured_interpreter_is_used(self):
        python = self.root / "host env" / "python"
        python.parent.mkdir()
        python.touch()
        self.data["hosts"]["hpc"]["python_executable"] = str(python)
        with patch("pipeline.job_manager._is_windows", return_value=False), \
             patch("pipeline.job_manager.subprocess.run", return_value=SimpleNamespace(stdout="127")) as run:
            submit_stage(self.data, "calibrate", [str(self.session)])
        worker = shlex.split(run.call_args.kwargs["input"].split("exec ", 1)[1])
        self.assertEqual(worker[0], str(python))

    def test_local_and_missing_input_rejected_before_submission(self):
        for environment, source in (("local", self.session), ("hpc", self.root / "missing")):
            with patch("pipeline.main.submit_stage") as submit, redirect_stderr(io.StringIO()), \
                 self.assertRaises(SystemExit):
                main(["calibrate", str(source), "--job", "--environment", environment,
                      "--config", str(self.config)])
            submit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
