import subprocess
import sys
import json


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


def test_cli_embedding_flag(img_path):
    result = subprocess.run(
        [*get_sci_command(), str(img_path), "-e"], capture_output=True, text=True
    )
    assert result.returncode == 0

    # Parse the output as JSON
    output_json = json.loads(result.stdout)

    # Check that the JSON contains the expected fields
    assert "iscc" in output_json
    assert "features" in output_json

    # Check the structure of the features field
    features = output_json["features"]
    assert isinstance(features, list)
    assert len(features) == 1

    feature_set = features[0]
    assert feature_set["maintype"] == "semantic"
    assert feature_set["subtype"] == "image"
    assert feature_set["version"] == 0
    assert "embedding" in feature_set
