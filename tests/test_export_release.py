from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


RELEASE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = RELEASE_ROOT.parent
EXPORTER_PATH = RELEASE_ROOT / "scripts" / "export_release.py"
ASSET_EXPORTER_PATH = RELEASE_ROOT / "scripts" / "export_release_assets.py"

EXPECTED_ALLOWLIST = {
    "python_pipeline/phase3/data.py": "src/gpanet/data.py",
    "python_pipeline/phase3/targets.py": "src/gpanet/targets.py",
    "python_pipeline/phase3/model.py": "src/gpanet/model.py",
    "python_pipeline/registry/reproduce_commands.yaml": "docs/source_reproduce_commands.yaml",
}


def _load_exporter():
    assert EXPORTER_PATH.is_file(), f"Missing exporter: {EXPORTER_PATH}"
    spec = importlib.util.spec_from_file_location("export_release_under_test", EXPORTER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_asset_exporter():
    assert ASSET_EXPORTER_PATH.is_file(), f"Missing asset exporter: {ASSET_EXPORTER_PATH}"
    spec = importlib.util.spec_from_file_location(
        "export_release_assets_under_test",
        ASSET_EXPORTER_PATH,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_fake_sources(project_root: Path) -> None:
    for source in EXPECTED_ALLOWLIST:
        path = project_root / source
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".yaml":
            path.write_text(
                """
commands:
  L4_FUS02_reproduce:
    scheme_id: L4_FUS02
    description: Historical source command with local virtualenv paths.
    command: python_pipeline/.venv/bin/python -m python_pipeline.phase8_6.l4_centered --output-root python_pipeline/results/local
    status: run
    artifacts:
      - python_pipeline/results/phase8_6/local_artifact.csv
""".lstrip(),
                encoding="utf-8",
            )
        else:
            path.write_text(f"# fake source for {source}\n", encoding="utf-8")

    leak = project_root / "python_pipeline" / "phase3" / "secret_raw_data.csv"
    leak.write_text("must not be copied\n", encoding="utf-8")


def test_allowlist_export_copies_only_approved_files_and_sanitizes_commands(tmp_path: Path) -> None:
    exporter = _load_exporter()
    fake_project = tmp_path / "project"
    release_root = fake_project / "gpa-net-release"
    _write_fake_sources(fake_project)

    exporter.export_release(project_root=fake_project, release_root=release_root)

    copied = {
        path.relative_to(release_root).as_posix()
        for path in release_root.rglob("*")
        if path.is_file()
    }
    assert set(EXPECTED_ALLOWLIST.values()) <= copied
    assert "src/gpanet/secret_raw_data.csv" not in copied
    assert exporter.ALLOWLIST == EXPECTED_ALLOWLIST

    command_manifest = (release_root / "docs" / "source_reproduce_commands.yaml").read_text(
        encoding="utf-8"
    )
    assert "python scripts/reproduce_gpa_net.py" in command_manifest
    for banned in ("python_pipeline/.venv", "/Users/", "~/"):
        assert banned not in command_manifest


@pytest.mark.parametrize(
    "destination",
    [
        "/tmp/outside.py",
        "../outside.py",
        "src/../outside.py",
        "~/outside.py",
        r"src\\gpanet\\data.py",
    ],
)
def test_release_destination_rejects_unsafe_strings(destination: str) -> None:
    exporter = _load_exporter()

    with pytest.raises(ValueError):
        exporter.resolve_release_destination(RELEASE_ROOT, destination)


def test_environment_and_requirements_are_portable_and_pinned() -> None:
    environment = (RELEASE_ROOT / "environment.yml").read_text(encoding="utf-8")
    requirements = (RELEASE_ROOT / "requirements.txt").read_text(encoding="utf-8")
    combined = environment + "\n" + requirements

    for token in (
        "name: gpa-net",
        "python=3.11",
        "numpy==1.24.3",
        "pandas==2.0.3",
        "scipy==1.11.1",
        "scikit-learn==1.3.0",
        "PyYAML==6.0",
        "pytest==7.4.0",
        "matplotlib==3.10.8",
        "seaborn==0.13.2",
        "torch==2.5.1",
        "Pillow==12.2.0",
        "pypdfium2==5.7.0",
        "markdown-it-py==4.0.0",
    ):
        assert token in combined

    for banned in (
        "python_pipeline/.venv",
        "/Users/",
        "~/Downloads",
        "numpy>=2",
        "pandas>=3",
        "torch>=2.11",
    ):
        assert banned not in combined


def test_training_curve_band_is_seed_level_after_fold_averaging() -> None:
    asset_exporter = _load_asset_exporter()
    frame = asset_exporter.pd.DataFrame(
        [
            {"seed": 1, "epoch": 1, "heldout_loss": 0.0},
            {"seed": 1, "epoch": 1, "heldout_loss": 10.0},
            {"seed": 2, "epoch": 1, "heldout_loss": 4.0},
            {"seed": 2, "epoch": 1, "heldout_loss": 6.0},
        ]
    )

    curve = asset_exporter._seed_averaged_epoch_curve(frame, "heldout_loss")

    assert curve.loc[0, "mean"] == 5.0
    assert curve.loc[0, "low"] == 5.0
    assert curve.loc[0, "high"] == 5.0
