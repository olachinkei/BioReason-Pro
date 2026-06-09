import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_launch(*args: str) -> str:
    result = subprocess.run(
        [sys.executable, "k8s/launch.py", "--tag", "contract-r1", "--dry_run", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def test_sunk_student_job_contract() -> None:
    rendered = run_launch("--gate_steps", "1", "--continue_steps", "0", "--max_val_samples", "2")

    assert "kind: Job" in rendered
    assert "schedulerName: tenant-slurm-staging-slurm-scheduler" in rendered
    assert 'sunk.coreweave.com/partition: "h100"' in rendered
    assert 'sunk.coreweave.com/exclusive: "none"' in rendered
    assert "terminationGracePeriodSeconds: 10" in rendered
    assert 'nvidia.com/gpu: "8"' in rendered
    assert "gpu.nvidia.com/model" in rendered
    assert "python train.py --mode 'gate'" in rendered
    assert "--gate_steps 1" in rendered
    assert "--continue_steps 0" in rendered
    assert "--max_val_samples 2" in rendered


def test_multiple_students_render_distinct_jobs() -> None:
    rendered = run_launch("--n_students", "2", "--names", "frieren,fern")

    assert "name: bioreason-contract-r1-frieren" in rendered
    assert "name: bioreason-contract-r1-fern" in rendered
    assert "student: frieren" in rendered
    assert "student: fern" in rendered
