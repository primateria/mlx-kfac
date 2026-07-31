import os
from pathlib import Path
import subprocess
import sys


def test_two_rank_unequal_batch_factor_aggregation():
    root = Path(__file__).resolve().parents[1]
    launcher = root / ".venv" / "bin" / "mlx.launch"
    worker = root / "tests" / "distributed_worker.py"
    environment = os.environ.copy()
    result = subprocess.run(
        [
            str(launcher),
            "-n",
            "2",
            "--backend",
            "ring",
            "--env",
            "PYTHONPATH=src",
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
    assert result.stdout.count("distributed_kfac=ok") == 2
