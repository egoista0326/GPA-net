from __future__ import annotations

import importlib
import builtins
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.dont_write_bytecode = True


RELEASE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = RELEASE_ROOT.parent
BANNED_LOCAL_MARKERS = ("python_pipeline/.venv", "/Users/", "~/", "~/Downloads")


def test_package_name_import_does_not_require_eager_torch_import(monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(RELEASE_ROOT / "src"))
    sys.modules.pop("gpanet", None)
    sys.modules.pop("gpanet.model", None)
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch" or name.startswith("torch."):
            raise ModuleNotFoundError("blocked torch import")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    gpanet = importlib.import_module("gpanet")

    assert gpanet.__name__ == "gpanet"


def test_gpanet_package_imports_with_public_model_surface(monkeypatch) -> None:
    pytest.importorskip("torch", reason="public model surface requires torch")
    monkeypatch.syspath_prepend(str(RELEASE_ROOT / "src"))

    gpanet = importlib.import_module("gpanet")

    assert gpanet.__name__ == "gpanet"
    assert gpanet.__version__
    assert gpanet.DualAttentionBaselineModel.__name__ == "DualAttentionBaselineModel"


def _clone_release_tree(tmp_path: Path, name: str = "gpa-net-release") -> Path:
    clone_root = tmp_path / name
    shutil.copytree(
        RELEASE_ROOT,
        clone_root,
        ignore=shutil.ignore_patterns(".pytest_cache", "__pycache__", "*.pyc"),
    )
    return clone_root


def _run_release_validator(release_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(release_root / "scripts" / "validate_release.py"), *args],
        cwd=release_root,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _parse_validation_summary(completed: subprocess.CompletedProcess[str]) -> dict[str, object]:
    output = completed.stdout.strip()
    summary_text = output.split("\nPASS release validation", 1)[0].strip()
    return json.loads(summary_text)


def test_release_validator_passes_in_filtered_standalone_clone(tmp_path: Path) -> None:
    clone_root = _clone_release_tree(tmp_path)

    completed = _run_release_validator(clone_root)
    summary = _parse_validation_summary(completed)

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert summary["status"] == "PASS"
    assert not summary["errors"]


def test_release_validator_accepts_repository_root_name(tmp_path: Path) -> None:
    clone_root = _clone_release_tree(tmp_path, "gpa-net")

    completed = _run_release_validator(clone_root)
    summary = _parse_validation_summary(completed)

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert summary["status"] == "PASS"
    assert not summary["errors"]


def test_release_validator_can_opt_into_source_provenance_checks(tmp_path: Path) -> None:
    clone_root = _clone_release_tree(tmp_path)

    completed = _run_release_validator(
        clone_root,
        "--check-source-provenance",
        "--project-root",
        str(PROJECT_ROOT),
    )
    summary = _parse_validation_summary(completed)

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert summary["status"] == "PASS"
    assert summary["check_source_provenance"] is True


def test_reproduction_wrapper_help_is_release_local() -> None:
    completed = subprocess.run(
        [sys.executable, str(RELEASE_ROOT / "scripts" / "reproduce_gpa_net.py"), "--help"],
        cwd=RELEASE_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    help_text = completed.stdout + completed.stderr

    assert "raw data is not bundled" in help_text.lower()
    assert "docs/source_reproduce_commands.yaml" in help_text
    for marker in BANNED_LOCAL_MARKERS:
        assert marker not in help_text


def test_data_contract_documents_schema_and_no_bundled_data_policy() -> None:
    data_contract = (RELEASE_ROOT / "docs" / "data_contract.md").read_text(encoding="utf-8")
    data_readme = (RELEASE_ROOT / "data" / "README.md").read_text(encoding="utf-8")
    combined = data_contract + "\n" + data_readme

    for token in (
        "feature_001",
        "feature_096",
        "feature_097",
        "subject_id",
        "trial_id",
        "trial_key",
        "condition_label",
        "abnormal_type",
        "source_file",
        "pressure_source_file",
        "gait_cycle_index",
        "UpperAsymmetry",
        "LowerAsymmetry",
        "TotalAsymmetry",
        "Classification",
        "forefoot pressure-balance asymmetry (UpperAsymmetry)",
        "rearfoot pressure-balance asymmetry (LowerAsymmetry)",
        "entire-foot pressure-balance asymmetry (TotalAsymmetry)",
        "Raw data are not bundled",
        "Processed*",
        "dataset.zip",
        "License pending",
        "Dataset access pending permission verification",
    ):
        assert token in combined


def test_source_reproduce_commands_are_sanitized_release_local_commands() -> None:
    manifest = (RELEASE_ROOT / "docs" / "source_reproduce_commands.yaml").read_text(
        encoding="utf-8"
    )

    assert "python scripts/reproduce_gpa_net.py" in manifest
    assert "raw data" in manifest.lower()
    for marker in BANNED_LOCAL_MARKERS:
        assert marker not in manifest


def test_reproduction_guide_uses_release_root_public_validation_commands() -> None:
    reproduction = (RELEASE_ROOT / "docs" / "reproduction.md").read_text(encoding="utf-8")

    assert "python scripts/validate_release.py" in reproduction
    assert "pytest tests/test_package_contract.py -q" in reproduction
    assert "python gpa-net-release/scripts/validate_release.py" not in reproduction
    assert "python_pipeline/tests" not in reproduction
