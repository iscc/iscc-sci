import subprocess
import sys


def get_sci_command():
    """Get the command to run the SCT CLI"""
    return [sys.executable, "-m", "iscc_sci.cli"]


def test_cli_no_args():
    result = subprocess.run(get_sci_command(), capture_output=True, text=True)
    assert result.returncode == 0
    assert "Generate Semantic" in result.stdout


def test_cli_generate_sci(img_path):
    result = subprocess.run([*get_sci_command(), str(img_path)], capture_output=True, text=True)
    assert result.returncode == 0
    assert "ISCC:" in result.stdout


def test_cli_debug_mode(img_path):
    result = subprocess.run(
        [*get_sci_command(), str(img_path), "-d"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "DEBUG" in result.stderr
