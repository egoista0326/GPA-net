from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError


INFERRED_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RELEASE_ROOT = Path(__file__).resolve().parents[1]

D02_BANNED_NAMES = {
    ".DS_Store",
    ".planning",
    ".pytest_cache",
    ".venv",
    "feature-pca-visualization.jpg",
    "Erin",
    "__pycache__",
    "dataset",
    "dataset.zip",
    "interview",
    "my matlab script",
    "my python file",
    "paper writing",
    "ppt",
    "questions",
    "reviews",
    "run_review_loop.py",
    "training-curves.jpg",
}
D02_BANNED_PREFIXES = ("Processed",)
D02_BANNED_SUFFIXES = (".log", ".svg", ".tmp")
D03_PAPER_FOLDER_NAME = "GPA_Net__2_0_"
PAPER_FOLDER_REFERENCE = "GPA_Net__2_0_/"
LOCAL_PATH_MARKERS = ("/Users/", "~/Downloads", "~/", "python_pipeline/.venv")
POLICY_EXEMPT_LINE_MARKERS = (
    "LOCAL_PATH_MARKERS",
    "FORBIDDEN_LOCAL_MARKERS",
    "PAPER_FOLDER_REFERENCE",
    "READ_ME_BAD_PATH_MARKERS",
)

PUBLIC_TEXT_FILES = ("README.md", "environment.yml", "requirements.txt")
PUBLIC_TEXT_DIRS = ("docs", "scripts", "src", "data")
TEXT_SUFFIXES = {".csv", ".json", ".md", ".py", ".rst", ".toml", ".txt", ".yaml", ".yml"}

REQUIRED_SOURCE_FILES = (
    "src/gpanet/__init__.py",
    "src/gpanet/data.py",
    "src/gpanet/targets.py",
    "src/gpanet/model.py",
)

DEPENDENCY_MANIFESTS = ("environment.yml", "requirements.txt")
PINNED_DEPENDENCIES = {
    "numpy": "numpy==1.24.3",
    "pandas": "pandas==2.0.3",
    "scipy": "scipy==1.11.1",
    "scikit-learn": "scikit-learn==1.3.0",
    "pyyaml": "PyYAML==6.0",
    "pytest": "pytest==7.4.0",
    "matplotlib": "matplotlib==3.10.8",
    "seaborn": "seaborn==0.13.2",
    "torch": "torch==2.5.1",
    "pillow": "Pillow==12.2.0",
    "pypdfium2": "pypdfium2==5.7.0",
    "markdown-it-py": "markdown-it-py==4.0.0",
}
REQUIRED_ENVIRONMENT_PIN = "python=3.11"
BLOCKED_DEPENDENCY_SNIPPETS = ("numpy>=2", "pandas>=3", "torch>=2.11")

FIGURE_MANIFEST_RELATIVE = Path("assets") / "figures" / "figure_manifest.csv"
FIGURE_ROOT_RELATIVE = Path("assets") / "figures"
SOURCE_LIST_SEPARATOR = ";"
REQUIRED_FIGURE_MANIFEST_COLUMNS = {
    "figure_id",
    "title",
    "asset_path",
    "asset_sha256",
    "source_path",
    "source_sha256",
    "width_px",
    "height_px",
    "role",
    "readme_candidate",
}
UNSUPPORTED_README_IMAGE_SUFFIXES = {".svg", ".pdf", ".png", ".gif", ".webp"}
READ_ME_BAD_PATH_MARKERS = ("/Users/", "~/")
MIN_README_IMAGE_WIDTH = 1200
MIN_README_IMAGE_HEIGHT = 500
MAX_README_IMAGE_BYTES = 5_000_000

CLAIM_SCOPE_MARKDOWN_FILES = ("README.md",)
CLAIM_SCOPE_MARKDOWN_DIRS = ("docs", "data")
CLAIM_CAVEAT_RE = re.compile(
    r"\b("
    r"not|no|cannot|can't|does not|doesn't|do not|don't|"
    r"unsupported|bounded|scope|support|support-only|diagnostic-only|"
    r"plausibility|exploratory|caveat"
    r")\b",
    flags=re.I,
)
UNCAVEATED_CLAIM_PATTERNS = {
    "single-trial replacement": re.compile(
        r"\b(single[- ]trial replacement|replace[s]? single[- ]trial|"
        r"replacement for single[- ]trial)\b",
        flags=re.I,
    ),
    "architecture superiority": re.compile(
        r"\b(architecture superiority|architecturally superior|"
        r"dual[- ]attention (?:architecture )?(?:is|was |proves? )?superior)\b",
        flags=re.I,
    ),
    "matched-search architecture proof": re.compile(
        r"\b(same[- ]HPO|equal[- ]search|matched[- ]search|"
        r"fair architecture (?:proof|comparison)|architecture proof)\b",
        flags=re.I,
    ),
    "causal proof": re.compile(r"\bcausal proof\b", flags=re.I),
    "causally explains": re.compile(r"\bcausally explains?\b", flags=re.I),
    "diagnostic override": re.compile(
        r"\bdiagnostic variant outperforms GPA-Net\b",
        flags=re.I,
    ),
    "diagnostic superiority": re.compile(
        r"\b(diagnostic superiority|diagnostically superior|superior diagnostic)\b",
        flags=re.I,
    ),
    "best overall": re.compile(r"\bbest overall\b", flags=re.I),
}

REVIEW_DIR_RELATIVE = Path("docs") / "reviews"
REVIEW_PROMPTS_RELATIVE = REVIEW_DIR_RELATIVE / "reviewer_prompts.md"
REVIEW_ACTIONS_RELATIVE = REVIEW_DIR_RELATIVE / "review_actions.csv"
REVIEW_CLOSEOUT_RELATIVE = REVIEW_DIR_RELATIVE / "review_closeout.md"
REVIEW_ROLE_LOG_DIR_RELATIVE = REVIEW_DIR_RELATIVE / "role_logs"
REVIEW_REQUIRED_ROLE_LOGS = (
    "visual_design.md",
    "report_flow.md",
    "result_correctness.md",
    "claim_validity.md",
    "repo_structure.md",
    "markdown_rendering.md",
    "english_terminology.md",
)
REVIEW_ACTIONS_HEADER = [
    "finding_id",
    "severity",
    "role",
    "round",
    "target_path",
    "summary",
    "required_fix",
    "fix_status",
    "fix_summary",
    "verified_by",
]
REVIEW_BLOCKING_SEVERITIES = {"blocker", "high"}
REVIEW_MEDIUM_SEVERITY = "medium"
REVIEW_RESOLVED_STATUSES = {
    "fixed",
    "closed_no_change_required",
    "accepted_bounded_risk",
    "pass",
}
REVIEW_REMEDIATION_MARKERS = ("## Applied Fixes", "## Validation Evidence")


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    path: str | None = None

    def as_dict(self) -> dict[str, str]:
        payload = {"code": self.code, "message": self.message}
        if self.path:
            payload["path"] = self.path
        return payload


def _relative(path: Path, release_root: Path) -> str:
    try:
        return path.relative_to(release_root).as_posix()
    except ValueError:
        return path.as_posix()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _strip_dependency_line(line: str) -> str:
    stripped = line.split("#", 1)[0].strip()
    if stripped.startswith("-"):
        stripped = stripped[1:].strip()
    return stripped.strip("\"'")


def _dependency_name(spec: str) -> str | None:
    match = re.match(r"^([A-Za-z0-9_.-]+)(?:\s*(?:==|=|>=|<=|>|<|~=|!=).*)?$", spec)
    if not match:
        return None
    return match.group(1).lower()


def _has_version_operator(spec: str) -> bool:
    return bool(re.search(r"(==|=|>=|<=|>|<|~=|!=)", spec))


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_manifest_values(value: str) -> list[str]:
    return [item.strip() for item in value.split(SOURCE_LIST_SEPARATOR) if item.strip()]


def _is_policy_exempt_line(line: str) -> bool:
    return any(marker in line for marker in POLICY_EXEMPT_LINE_MARKERS)


def _iter_public_text_paths(release_root: Path) -> list[Path]:
    paths: set[Path] = set()
    for relative in PUBLIC_TEXT_FILES:
        path = release_root / relative
        if path.is_file():
            paths.add(path)

    for relative_dir in PUBLIC_TEXT_DIRS:
        directory = release_root / relative_dir
        if not directory.is_dir():
            continue
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
                if _is_relative_to(path.resolve(), (release_root / REVIEW_DIR_RELATIVE).resolve()):
                    continue
                paths.add(path)

    return sorted(paths)


def _iter_claim_scope_paths(release_root: Path) -> list[Path]:
    paths: set[Path] = set()
    for relative in CLAIM_SCOPE_MARKDOWN_FILES:
        path = release_root / relative
        if path.is_file():
            paths.add(path)
    for relative_dir in CLAIM_SCOPE_MARKDOWN_DIRS:
        directory = release_root / relative_dir
        if not directory.is_dir():
            continue
        for path in directory.rglob("*.md"):
            if path.is_file():
                if _is_relative_to(path.resolve(), (release_root / REVIEW_DIR_RELATIVE).resolve()):
                    continue
                paths.add(path)
    return sorted(paths)


def validate_release_root(release_root: Path) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    resolved = release_root.resolve()

    if not resolved.exists():
        return [
            ValidationIssue(
                "ROOT",
                f"Release root does not exist: {resolved}",
                str(resolved),
            )
        ]
    if not resolved.is_dir():
        issues.append(ValidationIssue("ROOT", "Release root is not a directory", str(resolved)))
    if resolved == INFERRED_PROJECT_ROOT.resolve():
        issues.append(
            ValidationIssue(
                "ROOT",
                "Release root points at the parent workspace instead of the clean release tree",
                str(resolved),
            )
        )
    return issues


def validate_banned_entries(release_root: Path) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for path in release_root.rglob("*"):
        name = path.name
        relative = _relative(path, release_root)
        if name == D03_PAPER_FOLDER_NAME and path.is_dir():
            issues.append(
                ValidationIssue(
                    "D-03",
                    f"Full paper folder must not ship in release: {relative}",
                    relative,
                )
            )
            continue
        if name in D02_BANNED_NAMES or any(
            name.startswith(prefix) for prefix in D02_BANNED_PREFIXES
        ) or path.suffix.lower() in D02_BANNED_SUFFIXES:
            issues.append(
                ValidationIssue(
                    "D-02",
                    f"Banned release entry found: {relative}",
                    relative,
                )
            )
    return issues


def validate_public_text(release_root: Path) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for path in _iter_public_text_paths(release_root):
        relative = _relative(path, release_root)
        try:
            lines = _read_text(path).splitlines()
        except UnicodeDecodeError as exc:
            issues.append(
                ValidationIssue(
                    "TEXT",
                    f"Public text file is not valid UTF-8: {exc}",
                    relative,
                )
            )
            continue

        for line_number, line in enumerate(lines, start=1):
            if _is_policy_exempt_line(line):
                continue
            if PAPER_FOLDER_REFERENCE in line:
                issues.append(
                    ValidationIssue(
                        "D-03",
                        f"{relative}:{line_number} references {PAPER_FOLDER_REFERENCE}",
                        relative,
                    )
                )
            for marker in LOCAL_PATH_MARKERS:
                if marker in line:
                    issues.append(
                        ValidationIssue(
                            "LOCAL-PATH",
                            f"{relative}:{line_number} contains local path token {marker}",
                            relative,
                        )
                    )

    manifest = release_root / "assets" / "figures" / "figure_manifest.csv"
    if manifest.is_file():
        relative = _relative(manifest, release_root)
        for line_number, line in enumerate(_read_text(manifest).splitlines(), start=1):
            for marker in LOCAL_PATH_MARKERS:
                if marker in line:
                    issues.append(
                        ValidationIssue(
                            "LOCAL-PATH",
                            f"{relative}:{line_number} contains local path token {marker}",
                            relative,
                        )
                    )

    return issues


def validate_claim_scope(release_root: Path) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for path in _iter_claim_scope_paths(release_root):
        relative = _relative(path, release_root)
        lines = _read_text(path).splitlines()
        for line_number, line in enumerate(lines, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            for label, pattern in UNCAVEATED_CLAIM_PATTERNS.items():
                if pattern.search(stripped) and not CLAIM_CAVEAT_RE.search(stripped):
                    issues.append(
                        ValidationIssue(
                            "CLAIM-SCOPE",
                            f"{relative}:{line_number} contains uncaveated {label}: {stripped}",
                            relative,
                        )
                    )
                    break
    return issues


def validate_dependencies(release_root: Path) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    manifest_specs: dict[str, list[str]] = {}

    for relative in DEPENDENCY_MANIFESTS:
        path = release_root / relative
        if not path.is_file():
            issues.append(
                ValidationIssue(
                    "DEPENDENCY",
                    f"Missing dependency manifest: {relative}",
                    relative,
                )
            )
            continue

        specs: list[str] = []
        for line in _read_text(path).splitlines():
            spec = _strip_dependency_line(line)
            if not spec or spec.endswith(":") or spec in {"conda-forge", "pip"}:
                continue
            if spec.startswith(("channels:", "dependencies:", "name:")):
                continue
            specs.append(spec)

        manifest_specs[relative] = specs
        compact_specs = [_compact(spec) for spec in specs]

        for blocked in BLOCKED_DEPENDENCY_SNIPPETS:
            if any(blocked in spec for spec in compact_specs):
                issues.append(
                    ValidationIssue(
                        "DEPENDENCY",
                        f"{relative} contains unsupported dependency drift: {blocked}",
                        relative,
                    )
                )

        for spec, compact_spec in zip(specs, compact_specs):
            name = _dependency_name(spec)
            if not name or name not in PINNED_DEPENDENCIES:
                continue
            expected = _compact(PINNED_DEPENDENCIES[name])
            if name == "torch" and not _has_version_operator(spec):
                issues.append(
                    ValidationIssue(
                        "DEPENDENCY",
                        f"{relative} contains unbounded torch dependency: {spec}",
                        relative,
                    )
                )
            elif compact_spec != expected:
                issues.append(
                    ValidationIssue(
                        "DEPENDENCY",
                        f"{relative} contains dependency drift: {spec}; expected {PINNED_DEPENDENCIES[name]}",
                        relative,
                    )
                )

    environment_specs = {_compact(spec) for spec in manifest_specs.get("environment.yml", [])}
    if _compact(REQUIRED_ENVIRONMENT_PIN) not in environment_specs:
        issues.append(
            ValidationIssue(
                "DEPENDENCY",
                f"environment.yml must include {REQUIRED_ENVIRONMENT_PIN}",
                "environment.yml",
            )
        )

    for relative in DEPENDENCY_MANIFESTS:
        specs = {_compact(spec) for spec in manifest_specs.get(relative, [])}
        if not specs:
            continue
        for expected in PINNED_DEPENDENCIES.values():
            if _compact(expected) not in specs:
                issues.append(
                    ValidationIssue(
                        "DEPENDENCY",
                        f"{relative} must include measured stack pin {expected}",
                        relative,
                    )
                )

    return issues


def validate_required_source_files(release_root: Path) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for relative in REQUIRED_SOURCE_FILES:
        if not (release_root / relative).is_file():
            issues.append(
                ValidationIssue(
                    "SOURCE",
                    f"Missing required source file: {relative}",
                    relative,
                )
            )
    return issues


def _path_has_basic_escape(value: str) -> bool:
    path = Path(value)
    return (
        not value
        or path.is_absolute()
        or value.startswith("~")
        or "\\" in value
        or ".." in path.parts
    )


def _validate_manifest_asset_path(
    row: dict[str, str],
    release_root: Path,
    figure_root: Path,
) -> tuple[list[ValidationIssue], Path | None]:
    issues: list[ValidationIssue] = []
    asset_path = row.get("asset_path", "").strip()
    figure_id = row.get("figure_id", "<unknown>")
    readme_candidate = row.get("readme_candidate", "").strip().lower() == "true"
    suffix = Path(asset_path).suffix.lower()

    if any(marker in asset_path for marker in READ_ME_BAD_PATH_MARKERS):
        issues.append(
            ValidationIssue(
                "FIGURE-MANIFEST",
                f"README candidate asset path contains forbidden marker: {asset_path}",
                asset_path,
            )
        )
    if _path_has_basic_escape(asset_path):
        issues.append(
            ValidationIssue(
                "FIGURE-MANIFEST",
                f"Figure asset path must be safe and relative: {asset_path}",
                asset_path,
            )
        )
        return issues, None
    if suffix in UNSUPPORTED_README_IMAGE_SUFFIXES or suffix != ".jpg":
        issues.append(
            ValidationIssue(
                "FIGURE-MANIFEST",
                f"README-facing figure assets must be .jpg, got {asset_path}",
                asset_path,
            )
        )
    if readme_candidate and not asset_path.startswith("assets/figures/"):
        issues.append(
            ValidationIssue(
                "FIGURE-MANIFEST",
                f"README candidate asset path must start with assets/figures/: {asset_path}",
                asset_path,
            )
        )

    asset = (release_root / asset_path).resolve()
    if not _is_relative_to(asset, figure_root):
        issues.append(
            ValidationIssue(
                "FIGURE-MANIFEST",
                f"Figure asset escapes release figure directory: {asset_path}",
                asset_path,
            )
        )
        return issues, None
    if not asset.is_file():
        issues.append(
            ValidationIssue(
                "FIGURE-MANIFEST",
                f"Manifest asset is missing for {figure_id}: {asset_path}",
                asset_path,
            )
        )
        return issues, None
    return issues, asset


def _validate_manifest_sources(
    row: dict[str, str],
) -> tuple[list[ValidationIssue], list[tuple[str, str]]]:
    issues: list[ValidationIssue] = []
    figure_id = row.get("figure_id", "<unknown>")
    source_paths = _split_manifest_values(row.get("source_path", ""))
    source_hashes = _split_manifest_values(row.get("source_sha256", ""))
    if not source_paths or len(source_paths) != len(source_hashes):
        return [
            ValidationIssue(
                "FIGURE-MANIFEST",
                f"Manifest row must have matching source_path/source_sha256 values: {figure_id}",
                figure_id,
            )
        ], []

    validated_entries: list[tuple[str, str]] = []
    for source_path, expected_hash in zip(source_paths, source_hashes):
        if _path_has_basic_escape(source_path):
            issues.append(
                ValidationIssue(
                    "FIGURE-MANIFEST",
                    f"Source path must be safe and project-relative: {source_path}",
                    source_path,
                )
            )
            continue
        validated_entries.append((source_path, expected_hash))
    return issues, validated_entries


def _validate_manifest_source_provenance(
    source_entries: list[tuple[str, str]],
    *,
    figure_id: str,
    project_root: Path,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    resolved_project_root = project_root.resolve()

    for source_path, expected_hash in source_entries:
        source = (resolved_project_root / source_path).resolve()
        if not _is_relative_to(source, resolved_project_root):
            issues.append(
                ValidationIssue(
                    "FIGURE-MANIFEST",
                    f"Source path escapes project root: {source_path}",
                    source_path,
                )
            )
            continue
        if not source.is_file():
            issues.append(
                ValidationIssue(
                    "FIGURE-MANIFEST",
                    f"Manifest source is missing for {figure_id}: {source_path}",
                    source_path,
                )
            )
            continue
        actual_hash = _sha256_file(source)
        if expected_hash != actual_hash:
            issues.append(
                ValidationIssue(
                    "FIGURE-MANIFEST",
                    f"Source SHA-256 is stale for {source_path}: {expected_hash} != {actual_hash}",
                    source_path,
                )
            )
    return issues


def validate_figure_manifest(
    release_root: Path,
    *,
    check_source_provenance: bool = False,
    project_root: Path | None = None,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    manifest = release_root / FIGURE_MANIFEST_RELATIVE
    figure_root = (release_root / FIGURE_ROOT_RELATIVE).resolve()
    resolved_project_root = None
    if check_source_provenance:
        resolved_project_root = (project_root or INFERRED_PROJECT_ROOT).resolve()

    if not manifest.is_file():
        return [
            ValidationIssue(
                "FIGURE-MANIFEST",
                f"Missing figure manifest: {FIGURE_MANIFEST_RELATIVE.as_posix()}",
                FIGURE_MANIFEST_RELATIVE.as_posix(),
            )
        ]

    with manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing_columns = sorted(REQUIRED_FIGURE_MANIFEST_COLUMNS - columns)
        if missing_columns:
            return [
                ValidationIssue(
                    "FIGURE-MANIFEST",
                    f"Figure manifest missing required columns: {missing_columns}",
                    FIGURE_MANIFEST_RELATIVE.as_posix(),
                )
            ]
        rows = list(reader)

    if not rows:
        return [
            ValidationIssue(
                "FIGURE-MANIFEST",
                "Figure manifest must contain at least one row",
                FIGURE_MANIFEST_RELATIVE.as_posix(),
            )
        ]

    seen_assets: set[str] = set()
    for row in rows:
        figure_id = row.get("figure_id", "<unknown>")
        asset_path = row.get("asset_path", "").strip()
        role = row.get("role", "").strip()
        readme_candidate = row.get("readme_candidate", "").strip().lower()
        if readme_candidate not in {"true", "false"}:
            issues.append(
                ValidationIssue(
                    "FIGURE-MANIFEST",
                    f"readme_candidate must be true or false for {figure_id}",
                    figure_id,
                )
            )

        if asset_path in seen_assets:
            issues.append(
                ValidationIssue(
                    "FIGURE-MANIFEST",
                    f"Duplicate manifest asset path: {asset_path}",
                    asset_path,
                )
            )
        seen_assets.add(asset_path)

        path_issues, asset = _validate_manifest_asset_path(row, release_root, figure_root)
        issues.extend(path_issues)
        source_issues, source_entries = _validate_manifest_sources(row)
        issues.extend(source_issues)
        if check_source_provenance and resolved_project_root is not None:
            issues.extend(
                _validate_manifest_source_provenance(
                    source_entries,
                    figure_id=figure_id,
                    project_root=resolved_project_root,
                )
            )
        if asset is None:
            continue

        actual_asset_hash = _sha256_file(asset)
        expected_asset_hash = row.get("asset_sha256", "").strip()
        if expected_asset_hash != actual_asset_hash:
            issues.append(
                ValidationIssue(
                    "FIGURE-MANIFEST",
                    f"Asset SHA-256 is stale for {asset_path}: {expected_asset_hash} != {actual_asset_hash}",
                    asset_path,
                )
            )

        try:
            width = int(row.get("width_px", ""))
            height = int(row.get("height_px", ""))
        except ValueError:
            issues.append(
                ValidationIssue(
                    "FIGURE-MANIFEST",
                    f"Figure dimensions must be integers for {figure_id}",
                    figure_id,
                )
            )
            continue

        try:
            with Image.open(asset) as image:
                image_format = image.format
                image_mode = image.mode
                image_size = image.size
        except (OSError, UnidentifiedImageError) as exc:
            issues.append(
                ValidationIssue(
                    "FIGURE-MANIFEST",
                    f"Figure asset is not a readable image: {asset_path}: {exc}",
                    asset_path,
                )
            )
            continue

        if image_format != "JPEG" or image_mode != "RGB":
            issues.append(
                ValidationIssue(
                    "FIGURE-MANIFEST",
                    f"README assets must be RGB JPEGs: {asset_path} ({image_format}, {image_mode})",
                    asset_path,
                )
            )
        if image_size != (width, height):
            issues.append(
                ValidationIssue(
                    "FIGURE-MANIFEST",
                    f"Manifest dimensions are stale for {asset_path}: {width}x{height} != {image_size[0]}x{image_size[1]}",
                    asset_path,
                )
            )
        if readme_candidate == "true":
            if width < MIN_README_IMAGE_WIDTH or height < MIN_README_IMAGE_HEIGHT:
                issues.append(
                    ValidationIssue(
                        "FIGURE-MANIFEST",
                        f"README candidate image is too small: {asset_path} ({width}x{height})",
                        asset_path,
                    )
                )
            if asset.stat().st_size >= MAX_README_IMAGE_BYTES and role != "mandatory_readme_anchor":
                issues.append(
                    ValidationIssue(
                        "FIGURE-MANIFEST",
                        f"README candidate image exceeds size limit: {asset_path}",
                        asset_path,
                    )
                )

    return issues


def _has_review_closeout(release_root: Path) -> bool:
    return (release_root / REVIEW_CLOSEOUT_RELATIVE).is_file()


def _review_is_in_final_mode(release_root: Path, final_validation: bool) -> bool:
    if final_validation:
        return True
    closeout = release_root / REVIEW_CLOSEOUT_RELATIVE
    if not closeout.is_file():
        return False
    closeout_text = _read_text(closeout)
    return all(marker in closeout_text for marker in REVIEW_REMEDIATION_MARKERS)


def validate_review_loop(release_root: Path, *, final_validation: bool) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    prompts = release_root / REVIEW_PROMPTS_RELATIVE
    actions = release_root / REVIEW_ACTIONS_RELATIVE
    closeout = release_root / REVIEW_CLOSEOUT_RELATIVE
    role_log_dir = release_root / REVIEW_ROLE_LOG_DIR_RELATIVE

    review_artifacts_present = any(
        path.exists() for path in (prompts, actions, closeout, role_log_dir)
    )
    if not review_artifacts_present:
        return issues

    if not prompts.is_file():
        issues.append(
            ValidationIssue(
                "REVIEW-STRUCTURE",
                f"Missing reviewer prompts: {REVIEW_PROMPTS_RELATIVE.as_posix()}",
                REVIEW_PROMPTS_RELATIVE.as_posix(),
            )
        )

    if not actions.is_file():
        issues.append(
            ValidationIssue(
                "REVIEW-ACTIONS",
                f"Missing review action ledger: {REVIEW_ACTIONS_RELATIVE.as_posix()}",
                REVIEW_ACTIONS_RELATIVE.as_posix(),
            )
        )
        return issues

    with actions.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        actual_header = reader.fieldnames or []
        if actual_header != REVIEW_ACTIONS_HEADER:
            issues.append(
                ValidationIssue(
                    "REVIEW-ACTIONS",
                    "review_actions.csv has an invalid header",
                    REVIEW_ACTIONS_RELATIVE.as_posix(),
                )
            )
            return issues
        action_rows = list(reader)

    if not closeout.is_file():
        return issues

    closeout_text = _read_text(closeout)
    for required_section in (
        "## Loop Execution",
        "## Round Coverage",
        "## Findings By Severity",
        "## Remediation Handoff",
    ):
        if required_section not in closeout_text:
            issues.append(
                ValidationIssue(
                    "REVIEW-CLOSEOUT",
                    f"review_closeout.md is missing required section {required_section}",
                    REVIEW_CLOSEOUT_RELATIVE.as_posix(),
                )
            )

    for relative_name in REVIEW_REQUIRED_ROLE_LOGS:
        path = role_log_dir / relative_name
        if not path.is_file():
            issues.append(
                ValidationIssue(
                    "REVIEW-LOG",
                    f"Missing review role log: {REVIEW_ROLE_LOG_DIR_RELATIVE.joinpath(relative_name).as_posix()}",
                    REVIEW_ROLE_LOG_DIR_RELATIVE.joinpath(relative_name).as_posix(),
                )
            )

    if issues:
        return issues

    if not _review_is_in_final_mode(release_root, final_validation):
        return issues

    unresolved = []
    for row in action_rows:
        severity = row.get("severity", "").strip().lower()
        fix_status = row.get("fix_status", "").strip().lower()
        fix_summary = (row.get("fix_summary") or "").strip()
        verified_by = (row.get("verified_by") or "").strip()
        finding_id = row.get("finding_id", "<missing>")

        if severity in REVIEW_BLOCKING_SEVERITIES:
            if fix_status not in {"fixed", "closed_no_change_required"}:
                unresolved.append(f"{finding_id} ({severity})")
                continue
            if not fix_summary or not verified_by:
                issues.append(
                    ValidationIssue(
                        "REVIEW-ACTIONS",
                        f"review_actions.csv blocker/high finding lacks fix evidence: {finding_id}",
                        REVIEW_ACTIONS_RELATIVE.as_posix(),
                    )
                )
        elif severity == REVIEW_MEDIUM_SEVERITY:
            if fix_status not in {
                "fixed",
                "accepted_bounded_risk",
                "closed_no_change_required",
            }:
                issues.append(
                    ValidationIssue(
                        "REVIEW-ACTIONS",
                        f"review_actions.csv medium finding has invalid fix_status: {finding_id}",
                        REVIEW_ACTIONS_RELATIVE.as_posix(),
                    )
                )
                continue
            if not fix_summary or not verified_by:
                issues.append(
                    ValidationIssue(
                        "REVIEW-ACTIONS",
                        f"review_actions.csv medium finding lacks fix evidence: {finding_id}",
                        REVIEW_ACTIONS_RELATIVE.as_posix(),
                    )
                )

    if unresolved:
        offenders = ", ".join(
            unresolved
        )
        issues.append(
            ValidationIssue(
                "REVIEW-ACTIONS",
                f"review_actions.csv contains unresolved blocker/high findings: {offenders}",
                REVIEW_ACTIONS_RELATIVE.as_posix(),
            )
        )

    return issues


def validate_release(
    release_root: Path,
    *,
    final_validation: bool = False,
    check_source_provenance: bool = False,
    project_root: Path | None = None,
) -> list[ValidationIssue]:
    root_issues = validate_release_root(release_root)
    if root_issues:
        return root_issues

    resolved = release_root.resolve()
    issues: list[ValidationIssue] = []
    issues.extend(validate_banned_entries(resolved))
    issues.extend(validate_public_text(resolved))
    issues.extend(validate_claim_scope(resolved))
    issues.extend(validate_dependencies(resolved))
    issues.extend(validate_required_source_files(resolved))
    issues.extend(
        validate_figure_manifest(
            resolved,
            check_source_provenance=check_source_provenance,
            project_root=project_root,
        )
    )
    return issues


def build_summary(
    release_root: Path,
    issues: list[ValidationIssue],
    *,
    final_validation: bool,
    check_source_provenance: bool,
    project_root: Path | None,
) -> dict[str, object]:
    ok = not issues
    checks = [
        "root-boundary",
        "D-02-banned-entries",
        "D-03-paper-folder",
        "public-local-paths",
        "claim-scope",
        "dependency-pins",
        "required-source-files",
        "figure-manifest",
    ]
    if check_source_provenance:
        checks.append("figure-source-provenance")

    summary: dict[str, object] = {
        "ok": ok,
        "status": "PASS" if ok else "FAIL",
        "release_root": str(release_root.resolve()),
        "final_validation": final_validation,
        "check_source_provenance": check_source_provenance,
        "checks": checks,
        "errors": [issue.as_dict() for issue in issues],
    }
    if check_source_provenance and project_root is not None:
        summary["project_root"] = str(project_root.resolve())
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate the clean GPA-Net release tree."
    )
    parser.add_argument(
        "--release-root",
        type=Path,
        default=DEFAULT_RELEASE_ROOT,
        help="Release tree root. Defaults to the parent directory of this script.",
    )
    parser.add_argument(
        "--final-validation",
        action="store_true",
        help="Enable final validation mode, including unresolved review-action blockers/highs.",
    )
    parser.add_argument(
        "--check-source-provenance",
        action="store_true",
        help=(
            "Opt into maintainer-only figure source provenance checks against the canonical "
            "workspace sources referenced by assets/figures/figure_manifest.csv."
        ),
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        help=(
            "Canonical workspace root containing the pipeline outputs and original figure "
            "source files referenced by assets/figures/figure_manifest.csv. Only used with "
            "--check-source-provenance."
        ),
    )
    args = parser.parse_args(argv)

    release_root = args.release_root.resolve()
    if args.project_root and not args.check_source_provenance:
        parser.error("--project-root requires --check-source-provenance")

    project_root = args.project_root.resolve() if args.project_root else None
    issues = validate_release(
        release_root,
        final_validation=args.final_validation,
        check_source_provenance=args.check_source_provenance,
        project_root=project_root,
    )
    summary = build_summary(
        release_root,
        issues,
        final_validation=args.final_validation,
        check_source_provenance=args.check_source_provenance,
        project_root=project_root,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if issues:
        return 1
    print("PASS release validation")
    return 0


if __name__ == "__main__":
    sys.exit(main())
