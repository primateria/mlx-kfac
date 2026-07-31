import os
from pathlib import Path
import shutil
import subprocess
import sys


def test_two_rank_conditional_validation_and_recovery():
    root = Path(__file__).resolve().parents[1]
    launcher = shutil.which("mlx.launch")
    if launcher is None:
        launcher = Path(sys.executable).with_name("mlx.launch")
    assert Path(launcher).is_file(), "mlx.launch is not installed"
    worker = root / "tests" / "distributed_safety_worker.py"
    environment = os.environ.copy()
    result = subprocess.run(
        [
            str(launcher),
            "-n",
            "2",
            "--backend",
            "ring",
            "--env",
            f"PYTHONPATH={root / 'src'}",
            "env",
            sys.executable,
            str(worker),
        ],
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("distributed_safety=ok") == 2
