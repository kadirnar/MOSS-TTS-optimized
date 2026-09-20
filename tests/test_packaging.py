import hashlib
import importlib.resources
import json
import subprocess
import sys

import pytest

from moss_tts.calibration import read_dataset


def test_bundled_runtime_resources():
    root = importlib.resources.files("moss_tts")
    assets = root / "_assets"
    assert json.loads((assets / "packing.json").read_text())["codebooks"] == 32
    for name in (
        "qkv_cluster_bundle_v6",
        "qkv_cluster_bundle_history_v1",
        "gateup_exact_bundle_v1",
    ):
        folder = assets / name
        manifest = json.loads((folder / "manifest.json").read_text())
        assert manifest["complete"]
        assert len(manifest["binaries"]) == 2
        for record in manifest["binaries"].values():
            binary = (folder / (record["prefix"] + ".cubin")).read_bytes()
            assert hashlib.sha256(binary).hexdigest() == record["cubin_sha256"]
    for name in ("attention_history.cu", "attention_pdl.cu", "attention_native.cu"):
        assert (root / "_runtime" / name).is_file()


def test_cli_help_without_loading_model():
    for command in ([], ["synthesize"], ["serve"], ["calibrate"]):
        result = subprocess.run(
            [sys.executable, "-m", "moss_tts", *command, "--help"],
            capture_output=True,
            text=True,
            check=True,
        )
        assert "usage:" in result.stdout


def test_calibration_dataset_requires_heldout_split(tmp_path):
    reference = tmp_path / "reference.wav"
    reference.touch()
    dataset = tmp_path / "examples.jsonl"
    train = {"text": "Hello", "language": "English", "reference": "reference.wav"}
    dataset.write_text(json.dumps(train) + "\n")
    with pytest.raises(ValueError, match="held-out"):
        read_dataset(dataset)
    with dataset.open("a") as output:
        output.write(json.dumps({**train, "split": "validation"}) + "\n")
    rows = read_dataset(dataset)
    assert rows[0]["reference"] == str(reference)
    assert rows[1]["split"] == "validation"
