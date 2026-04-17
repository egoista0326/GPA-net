from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RELEASE_ROOT = PROJECT_ROOT / "gpa-net-release"

ALLOWLIST = {
    "python_pipeline/phase3/data.py": "src/gpanet/data.py",
    "python_pipeline/phase3/targets.py": "src/gpanet/targets.py",
    "python_pipeline/phase3/model.py": "src/gpanet/model.py",
    "python_pipeline/registry/reproduce_commands.yaml": "docs/source_reproduce_commands.yaml",
}

BANNED_TOP_LEVEL = {
    ".planning",
    ".pytest_cache",
    ".venv",
    "Erin",
    "GPA_Net__2_0_",
    "Processed",
    "Processed_canonical",
    "Processed_canonical_smoke",
    "Processed_matlab",
    "__pycache__",
    "dataset",
    "dataset.zip",
    "interview",
    "my matlab script",
    "my python file",
    "paper writing",
    "ppt",
    "questions",
}
BANNED_TOP_LEVEL_PREFIXES = ("Processed",)
FORBIDDEN_LOCAL_MARKERS = ("python_pipeline/.venv", "/Users/", "~/", "~/Downloads")


def _require_relative_to(path: Path, parent: Path, label: str) -> None:
    try:
        path.relative_to(parent)
    except ValueError as exc:
        raise ValueError(f"{label} escapes expected root: {path}") from exc


def resolve_project_source(project_root: Path, source: str) -> Path:
    if "\\" in source or source.startswith("~") or ".." in Path(source).parts:
        raise ValueError(f"Unsafe source path: {source}")

    source_path = Path(source)
    resolved_root = project_root.resolve()
    resolved = (resolved_root / source_path).resolve()
    _require_relative_to(resolved, resolved_root, "source")
    return resolved


def resolve_release_destination(release_root: Path, destination: str) -> Path:
    if "\\" in destination:
        raise ValueError(f"Destination must use POSIX separators: {destination}")

    destination_path = Path(destination)
    if destination_path.is_absolute():
        raise ValueError(f"Destination must be relative: {destination}")
    if destination.startswith("~") or ".." in destination_path.parts:
        raise ValueError(f"Unsafe release destination: {destination}")

    resolved_root = release_root.resolve()
    resolved = (resolved_root / destination_path).resolve()
    _require_relative_to(resolved, resolved_root, "destination")
    return resolved


def _release_artifact_name(command_id: str, artifact: str) -> str:
    name = Path(artifact).name
    if not name or name in {".", ".."}:
        name = "artifacts"
    return f"results/{command_id}/{name}"


def sanitize_reproduce_manifest(source_text: str) -> str:
    payload = yaml.safe_load(source_text) or {}
    source_commands = payload.get("commands", {})
    if not isinstance(source_commands, dict):
        raise ValueError("Expected a top-level 'commands' mapping in reproduction manifest")

    release_commands: dict[str, dict[str, Any]] = {}
    for command_id, entry in source_commands.items():
        if not isinstance(entry, dict):
            raise ValueError(f"Command entry must be a mapping: {command_id}")

        artifacts = entry.get("artifacts") or []
        if not isinstance(artifacts, list):
            raise ValueError(f"Command artifacts must be a list: {command_id}")

        release_commands[str(command_id)] = {
            "scheme_id": entry.get("scheme_id", ""),
            "description": entry.get("description", ""),
            "command": f"python scripts/reproduce_gpa_net.py --recipe {command_id}",
            "status": "release-wrapper",
            "source_status": entry.get("status", ""),
            "expected_artifacts": [
                _release_artifact_name(str(command_id), str(artifact))
                for artifact in artifacts
            ],
        }

    release_payload = {
        "schema_version": 1,
        "data_policy": "Raw data is not bundled; provide a CSV matching docs/data_contract.md.",
        "commands": release_commands,
    }
    sanitized = yaml.safe_dump(release_payload, sort_keys=False, allow_unicode=False)
    leaked = [marker for marker in FORBIDDEN_LOCAL_MARKERS if marker in sanitized]
    if leaked:
        raise ValueError(f"Sanitized reproduction manifest still contains local markers: {leaked}")
    return sanitized


def validate_no_banned_entries(release_root: Path) -> None:
    if not release_root.exists():
        return

    forbidden = sorted(
        path.name
        for path in release_root.iterdir()
        if path.name in BANNED_TOP_LEVEL
        or any(path.name.startswith(prefix) for prefix in BANNED_TOP_LEVEL_PREFIXES)
    )
    if forbidden:
        raise RuntimeError(f"Release contains forbidden top-level entries: {forbidden}")


def export_release(
    project_root: str | Path = PROJECT_ROOT,
    release_root: str | Path | None = None,
) -> None:
    project_root = Path(project_root).resolve()
    release_root = Path(release_root).resolve() if release_root else project_root / "gpa-net-release"

    validate_no_banned_entries(release_root)
    for source, destination in ALLOWLIST.items():
        source_path = resolve_project_source(project_root, source)
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing allowlisted source file: {source}")

        destination_path = resolve_release_destination(release_root, destination)
        destination_path.parent.mkdir(parents=True, exist_ok=True)

        if source == "python_pipeline/registry/reproduce_commands.yaml":
            sanitized = sanitize_reproduce_manifest(source_path.read_text(encoding="utf-8"))
            destination_path.write_text(sanitized, encoding="utf-8")
        else:
            shutil.copy2(source_path, destination_path)

    validate_no_banned_entries(release_root)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the GPA-Net release tree from an explicit allowlist."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Source project root. Defaults to the parent of gpa-net-release/.",
    )
    parser.add_argument(
        "--release-root",
        type=Path,
        default=None,
        help="Destination release root. Defaults to PROJECT_ROOT/gpa-net-release.",
    )
    args = parser.parse_args()

    export_release(project_root=args.project_root, release_root=args.release_root)
    release_root = args.release_root or args.project_root / "gpa-net-release"
    print(f"Exported allowlisted files into {Path(release_root).resolve()}")


if __name__ == "__main__":
    main()
