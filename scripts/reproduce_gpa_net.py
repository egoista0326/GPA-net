from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml


RELEASE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = RELEASE_ROOT / "src"
MANIFEST_PATH = RELEASE_ROOT / "docs" / "source_reproduce_commands.yaml"
DATA_POLICY = (
    "Raw data is not bundled. Dataset access pending permission verification; "
    "provide a CSV matching docs/data_contract.md."
)

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def resolve_release_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or value.startswith("~") or "\\" in value or ".." in path.parts:
        raise ValueError(f"Path must stay inside the release tree: {value}")
    resolved = (RELEASE_ROOT / path).resolve()
    resolved.relative_to(RELEASE_ROOT.resolve())
    return resolved


def load_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Missing release command manifest: {MANIFEST_PATH}")
    return yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8")) or {}


def command_ids(manifest: dict[str, Any]) -> list[str]:
    commands = manifest.get("commands", {})
    if not isinstance(commands, dict):
        return []
    return sorted(str(command_id) for command_id in commands)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Release-local GPA-Net reproduction wrapper. Raw data is not bundled; "
            "see docs/data_contract.md before running a recipe."
        ),
        epilog=(
            "Recipes are listed in docs/source_reproduce_commands.yaml. "
            "Use --list-recipes to print the available release-local entrypoints."
        ),
    )
    parser.add_argument("--list-recipes", action="store_true", help="List available recipes.")
    parser.add_argument("--recipe", help="Recipe key from docs/source_reproduce_commands.yaml.")
    parser.add_argument(
        "--data",
        default="data/canonical_dataset.csv",
        help="Release-relative canonical CSV path. Raw data is not bundled.",
    )
    parser.add_argument(
        "--output",
        default="results/reproduction",
        help="Release-relative output directory for generated artifacts.",
    )
    parser.add_argument(
        "--check-data",
        action="store_true",
        help="Validate the CSV against the release data contract before printing the recipe.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    manifest = load_manifest()
    recipes = command_ids(manifest)

    if args.list_recipes:
        for recipe in recipes:
            print(recipe)
        return 0

    if not args.recipe:
        parser.print_help()
        print(f"\n{DATA_POLICY}")
        return 0

    if args.recipe not in recipes:
        parser.error(f"Unknown recipe '{args.recipe}'. Available recipes: {', '.join(recipes)}")

    data_path = resolve_release_path(args.data)
    output_path = resolve_release_path(args.output)

    if args.check_data:
        from gpanet.data import load_canonical_dataset

        load_canonical_dataset(data_path)

    command = manifest["commands"][args.recipe]
    print(f"Recipe: {args.recipe}")
    print(f"Description: {command.get('description', '')}")
    print(f"Data CSV: {data_path.relative_to(RELEASE_ROOT)}")
    print(f"Output directory: {output_path.relative_to(RELEASE_ROOT)}")
    print(DATA_POLICY)
    print("Release command:")
    print(command.get("command", f"python scripts/reproduce_gpa_net.py --recipe {args.recipe}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
