import subprocess
import os
import pytest


def test_train_script_dry_run():
    # 1. Retrieve the path to python running pytest
    import sys
    venv_python = sys.executable

    train_script = os.path.join(os.getcwd(), "train.py")
    assert os.path.exists(train_script), "train.py script not found."

    # 2. Execute train.py with dry_run override using subprocess
    cmd = [venv_python, train_script, "dry_run=true", "training.epochs=1"]
    
    # Run the process
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=os.getcwd()
    )

    # 3. Print stdout/stderr in case of failure for debugging
    print("STDOUT:")
    print(result.stdout)
    print("STDERR:")
    print(result.stderr)

    # 4. Assert exit code is 0 (success)
    assert result.returncode == 0, f"train.py execution failed with exit code {result.returncode}"
    
    # 5. Check if checkpoint folder and best_model.pth was created and clean it up
    checkpoint_path = os.path.join(os.getcwd(), "checkpoints", "best_model.pth")
    assert os.path.exists(checkpoint_path), "best_model.pth checkpoint was not saved."

    # Cleanup checkpoints created during dry-run testing
    import shutil
    shutil.rmtree(os.path.join(os.getcwd(), "checkpoints"), ignore_errors=True)
