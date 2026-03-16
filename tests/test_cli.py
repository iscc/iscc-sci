"""Tests for the iscc-sci CLI."""

import json

from iscc_sci.cli import main


def test_cli_no_args(capsys, monkeypatch):
    monkeypatch.setattr("sys.argv", ["iscc-sci"])
    main()
    captured = capsys.readouterr()
    assert "Generate Semantic" in captured.out


def test_cli_generate_sci(capsys, monkeypatch, img_path):
    monkeypatch.setattr("sys.argv", ["iscc-sci", str(img_path)])
    main()
    captured = capsys.readouterr()
    assert "ISCC:" in captured.out


def test_cli_bits_option(capsys, monkeypatch, img_path):
    monkeypatch.setattr("sys.argv", ["iscc-sci", str(img_path), "-b", "64"])
    main()
    captured = capsys.readouterr()
    output = captured.out.strip()
    assert output.startswith("ISCC:")
    # 64-bit code is shorter than default 256-bit
    assert len(output) < 60


def test_cli_debug_mode(capsys, monkeypatch, img_path):
    monkeypatch.setattr("sys.argv", ["iscc-sci", str(img_path), "-d"])
    main()
    captured = capsys.readouterr()
    assert "ISCC:" in captured.out


def test_cli_embedding_flag(capsys, monkeypatch, img_path):
    monkeypatch.setattr("sys.argv", ["iscc-sci", str(img_path), "-e"])
    main()
    captured = capsys.readouterr()
    output_json = json.loads(captured.out)
    assert "iscc" in output_json
    assert "features" in output_json
    features = output_json["features"]
    assert isinstance(features, list)
    assert len(features) == 1
    feature_set = features[0]
    assert feature_set["maintype"] == "semantic"
    assert feature_set["subtype"] == "image"
    assert feature_set["version"] == 0
    assert "embedding" in feature_set


def test_cli_glob_no_match(capsys, monkeypatch, tmp_path):
    pattern = str(tmp_path / "*.nonexistent")
    monkeypatch.setattr("sys.argv", ["iscc-sci", pattern])
    main()
    captured = capsys.readouterr()
    assert captured.out == ""


def test_cli_glob_skips_directories(capsys, monkeypatch, tmp_path):
    sub_dir = tmp_path / "subdir"
    sub_dir.mkdir()
    pattern = str(tmp_path / "*")
    monkeypatch.setattr("sys.argv", ["iscc-sci", pattern])
    main()
    captured = capsys.readouterr()
    assert "ISCC:" not in captured.out
