import os
from pathlib import Path
import shutil
import subprocess
import sys


def test_two_rank_conditional_validation_and_recovery(tmp_path):
    root = Path(__file__).resolve().parents[1]
    launcher = shutil.which("mlx.launch")
    if launcher is None:
        launcher = Path(sys.executable).with_name("mlx.launch")
    assert Path(launcher).is_file(), "mlx.launch is not installed"
    worker = root / "tests" / "distributed_safety_worker.py"
    environment = os.environ.copy()
    environment["MLX_KFAC_MARKER_DIR"] = str(tmp_path)
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
    markers = {path.name for path in tmp_path.glob("rank-*.ok")}
    assert markers == {"rank-0.ok", "rank-1.ok"}, result.stdout + result.stderr
