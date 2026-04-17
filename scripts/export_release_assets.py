from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RELEASE_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = RELEASE_ROOT / "assets" / "figures"
MANIFEST_PATH = FIGURE_DIR / "figure_manifest.csv"
FIGURE_README_PATH = FIGURE_DIR / "README.md"

PAPER_SOURCE_ROOT = Path("GPA_Net__2_0_") / "figures-mine"
PHASE9_1_ROOT = Path("python_pipeline") / "results" / "phase9_1"
PHASE9_2_ROOT = Path("python_pipeline") / "results" / "phase9_2"
PHASE8_INTERPRETABILITY_ROOT = Path("python_pipeline") / "results" / "phase8" / "interpretability"
JPEG_QUALITY = 92
PDF_DPI = 300
PLOT_DPI = 180
SOURCE_LIST_SEPARATOR = ";"

MORANDI = {
    "paper": "#F7F5F1",
    "ink": "#2F3433",
    "grid": "#D9D6D0",
    "muted_grid": "#ECE9E3",
    "primary": "#5F7F7A",
    "primary_dark": "#355C58",
    "rose": "#B8847C",
    "sage": "#8A9A78",
    "lavender": "#8D86A3",
    "blue_gray": "#7F99A8",
    "clay": "#B0906F",
    "neutral": "#B0B0B0",
    "neutral_dark": "#6F7472",
}

MORANDI_HEATMAP = LinearSegmentedColormap.from_list(
    "morandi_attention",
    [
        MORANDI["paper"],
        "#DDE4DF",
        "#CAD8D4",
        "#8FA8A3",
        MORANDI["primary"],
        MORANDI["primary_dark"],
    ],
)

MANIFEST_COLUMNS = [
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
    "source_id",
    "internal_aliases",
    "generation_method",
    "notes",
]
@dataclass(frozen=True)
class PaperFigureSpec:
    figure_id: str
    source_relative: Path
    asset_relative: Path
    role: str
    title: str
    readme_candidate: bool


@dataclass(frozen=True)
class EvidenceFigureSpec:
    figure_id: str
    asset_relative: Path
    source_relatives: tuple[Path, ...]
    role: str
    title: str
    internal_aliases: str
    notes: str


PAPER_FIGURES = (
    PaperFigureSpec(
        figure_id="overall_system_framework",
        source_relative=PAPER_SOURCE_ROOT / "framework.pdf",
        asset_relative=Path("assets/figures/overall-system-framework.jpg"),
        role="mandatory_readme_anchor",
        title="Overall System Framework",
        readme_candidate=True,
    ),
    PaperFigureSpec(
        figure_id="gpa_net_architecture_overview",
        source_relative=PAPER_SOURCE_ROOT / "model.pdf",
        asset_relative=Path("assets/figures/gpa-net-architecture-overview.jpg"),
        role="mandatory_readme_anchor",
        title="GPA-Net Architecture Overview",
        readme_candidate=True,
    ),
    PaperFigureSpec(
        figure_id="gait_cycle_schematic",
        source_relative=PAPER_SOURCE_ROOT / "gaitcycle.pdf",
        asset_relative=Path("assets/figures/gait-cycle-schematic.jpg"),
        role="optional_method_support",
        title="Gait Cycle Schematic",
        readme_candidate=False,
    ),
)

EVIDENCE_FIGURES = (
    EvidenceFigureSpec(
        figure_id="performance_summary",
        asset_relative=Path("assets/figures/performance-summary.jpg"),
        source_relatives=(
            PHASE9_1_ROOT / "tables" / "main_results.csv",
            PHASE9_1_ROOT / "final_selection_decision.json",
        ),
        role="supporting_performance_summary",
        title="GPA-Net Performance Summary",
        internal_aliases="HPO-L4-02 + FUS-02",
        notes="Selected same-scope subject-heldout repeated-walk evidence for GPA-Net headline metrics.",
    ),
    EvidenceFigureSpec(
        figure_id="dual_attention_ablation",
        asset_relative=Path("assets/figures/dual-attention-ablation.jpg"),
        source_relatives=(
            PHASE9_1_ROOT / "tables" / "dual_attention_ablation.csv",
        ),
        role="supporting_ablation_summary",
        title="GPA-Net Dual-Attention Ablation",
        internal_aliases="HPO-L4-02 + FUS-02; FE-13; FUS-02",
        notes="Bounded architecture-only ablation; not a universal architecture-superiority claim.",
    ),
    EvidenceFigureSpec(
        figure_id="fixed_control_comparison",
        asset_relative=Path("assets/figures/fixed-control-comparison.jpg"),
        source_relatives=(
            PHASE9_1_ROOT / "tables" / "main_results.csv",
            PHASE9_1_ROOT / "final_selection_decision.json",
        ),
        role="supporting_fixed_control_summary",
        title="GPA-Net Fixed-Control Comparison",
        internal_aliases="HPO-L4-02 + FUS-02; Phase 3 strict baseline; E2E-15 MLP/CNN",
        notes=(
            "Fixed-recipe controls; GPA-Net is classification-best in this context view, "
            "not a matched-search architecture proof."
        ),
    ),
    EvidenceFigureSpec(
        figure_id="training_curves",
        asset_relative=Path("assets/figures/heldout-training-curves.jpg"),
        source_relatives=(
            PHASE9_2_ROOT / "tables" / "long_epoch_heldout_loss_history.csv",
            PHASE9_2_ROOT / "tables" / "long_epoch_heldout_loss_summary.csv",
            PHASE9_2_ROOT / "tables" / "diagnostic_training_manifest.csv",
        ),
        role="supporting_training_curve_summary",
        title="Held-Out Training Curves",
        internal_aliases="HPO-L4-02 + FUS-02; E2E-15-style MLP/CNN controls",
        notes=(
            "Long-epoch diagnostics with real repeated-walk outer held-out task loss; "
            "lower curve height is better under the stated fixed-control protocol."
        ),
    ),
    EvidenceFigureSpec(
        figure_id="phase_feature_sensitivity",
        asset_relative=Path("assets/figures/phase-feature-sensitivity.jpg"),
        source_relatives=(
            PHASE9_2_ROOT / "tables" / "phase_feature_sensitivity_summary.csv",
            PHASE9_2_ROOT / "tables" / "phase_feature_sensitivity.csv",
        ),
        role="supporting_feature_space_summary",
        title="Held-Out Phase-Feature Sensitivity",
        internal_aliases="HPO-L4-02 + FUS-02; FE-13; FE-08",
        notes=(
            "Outer held-out phase-family median perturbation summary; predictive sensitivity, not causal proof."
        ),
    ),
    EvidenceFigureSpec(
        figure_id="interpretability_summary",
        asset_relative=Path("assets/figures/interpretability-summary.jpg"),
        source_relatives=(
            PHASE8_INTERPRETABILITY_ROOT / "attention_phase_summary.csv",
        ),
        role="supporting_interpretability_summary",
        title="GPA-Net Interpretability Summary",
        internal_aliases="E2E-15; FE-13; plausibility attention evidence",
        notes="Target-level plausibility evidence with D-14 event-based phase labels; not causal proof.",
    ),
)

D14_PHASE_LABELS = {
    "initial_contact": "Right Initial\nContact",
    "loading_response": "First Double\nSupport",
    "mid_stance": "Right Single\nSupport",
    "terminal_stance": "Left Initial\nContact",
    "pre_swing": "Second Double\nSupport",
    "swing": "Left Single\nSupport",
}

TARGET_DISPLAY_LABELS = {
    "UpperAsymmetry": "Forefoot asymmetry",
    "LowerAsymmetry": "Rearfoot asymmetry",
    "TotalAsymmetry": "Entire-foot asymmetry",
    "Classification": "Balance classification",
}


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _reject_unsafe_relative_path(value: str | Path, *, label: str) -> Path:
    text = str(value)
    path = Path(text)
    if not text:
        raise ValueError(f"{label} path must not be empty")
    if path.is_absolute():
        raise ValueError(f"{label} path must be relative: {text}")
    if text.startswith("~"):
        raise ValueError(f"{label} path must not use a home shortcut: {text}")
    if "\\" in text:
        raise ValueError(f"{label} path must use POSIX separators: {text}")
    if ".." in path.parts:
        raise ValueError(f"{label} path must not contain '..': {text}")
    return path


def resolve_source_path(project_root: Path, source_relative: str | Path) -> Path:
    relative = _reject_unsafe_relative_path(source_relative, label="source")
    resolved = (project_root / relative).resolve()
    if not _is_relative_to(resolved, project_root.resolve()):
        raise ValueError(f"Source path escapes project root: {source_relative}")
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing source file: {relative.as_posix()}")
    return resolved


def resolve_figure_destination(release_root: Path, asset_relative: str | Path) -> Path:
    relative = _reject_unsafe_relative_path(asset_relative, label="asset")
    if relative.parts[:2] != ("assets", "figures"):
        raise ValueError(f"Asset destination must live under assets/figures/: {relative.as_posix()}")
    if relative.suffix.lower() != ".jpg":
        raise ValueError(f"Release figure destination must be .jpg: {relative.as_posix()}")
    resolved = (release_root / relative).resolve()
    figure_root = (release_root / "assets" / "figures").resolve()
    if not _is_relative_to(resolved, figure_root):
        raise ValueError(f"Asset path escapes release figure directory: {relative.as_posix()}")
    return resolved


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project_relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def _release_relative(path: Path) -> str:
    return path.resolve().relative_to(RELEASE_ROOT.resolve()).as_posix()


def _join_manifest_values(values: Iterable[str]) -> str:
    return SOURCE_LIST_SEPARATOR.join(values)


def render_pdf_first_page_to_jpg(source: Path, destination: Path) -> tuple[int, int]:
    import pypdfium2 as pdfium

    destination.parent.mkdir(parents=True, exist_ok=True)
    document = pdfium.PdfDocument(str(source))
    try:
        page = document[0]
        image = page.render(scale=PDF_DPI / 72).to_pil().convert("RGB")
        image.save(
            destination,
            "JPEG",
            quality=JPEG_QUALITY,
            optimize=True,
            progressive=True,
        )
        return image.width, image.height
    finally:
        document.close()


def _read_manifest(path: Path = MANIFEST_PATH) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {row["figure_id"]: row for row in rows}


def _write_manifest(rows_by_id: dict[str, dict[str, str]], path: Path = MANIFEST_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [rows_by_id[key] for key in sorted(rows_by_id)]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _upsert_manifest_rows(rows: Iterable[dict[str, str]]) -> None:
    rows_by_id = _read_manifest()
    for row in rows:
        rows_by_id[row["figure_id"]] = {column: row.get(column, "") for column in MANIFEST_COLUMNS}
    _write_manifest(rows_by_id)


def _write_figures_readme() -> None:
    FIGURE_README_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_README_PATH.write_text(
        "\n".join(
            [
                "# Release Figure Assets",
                "",
                "README-facing figures in this directory are release-owned JPG files generated from selected paper and evidence artifacts.",
                "The manifest records source provenance, SHA-256 hashes, dimensions, role, and README eligibility for each generated asset.",
                "Redistribution and license status for paper-derived visual assets remains pending; keep source provenance in the manifest rather than public README prose.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _save_matplotlib_jpg(fig: plt.Figure, destination: Path) -> tuple[int, int]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        destination,
        format="jpg",
        dpi=PLOT_DPI,
        facecolor="white",
        bbox_inches="tight",
        pil_kwargs={
            "quality": JPEG_QUALITY,
            "optimize": True,
            "progressive": True,
        },
    )
    plt.close(fig)
    with Image.open(destination) as image:
        rgb = image.convert("RGB")
        rgb.save(
            destination,
            "JPEG",
            quality=JPEG_QUALITY,
            optimize=True,
            progressive=True,
        )
    with Image.open(destination) as image:
        return image.width, image.height


def _format_metric(value: float, *, percent: bool = False) -> str:
    if percent:
        return f"{value * 100:.1f}%"
    return f"{value:.3f}"


def _apply_public_plot_style(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.tick_params(colors=MORANDI["ink"], labelsize=10)
    ax.title.set_color(MORANDI["ink"])
    ax.xaxis.label.set_color(MORANDI["ink"])
    ax.yaxis.label.set_color(MORANDI["ink"])
    ax.grid(color=MORANDI["grid"], linewidth=0.7, alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(MORANDI["grid"])
    ax.spines["bottom"].set_color(MORANDI["grid"])


def _figure_caption(fig: plt.Figure, text: str, *, y: float = 0.915) -> None:
    fig.text(
        0.5,
        y,
        text,
        ha="center",
        fontsize=10,
        color=MORANDI["neutral_dark"],
    )


def _public_system_name(system: str) -> str:
    mapping = {
        "HPO-L4-02 + FUS-02": "GPA-Net",
        "L4 + FUS-02": "Historical anchor",
        "HPO-L4-03 + logit_mean": "Diagnostic fusion",
        "Phase 3 strict baseline": "Strict dual-attention control",
        "E2E-15 MLP baseline": "MLP control",
        "E2E-15 CNN baseline": "CNN control",
    }
    return mapping.get(system, system)


def _normalize_curve(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    first = numeric.dropna().iloc[0]
    if first == 0:
        return numeric
    return numeric / first


def _curve_from_history(history_path: Path) -> pd.DataFrame:
    history = pd.read_csv(history_path)
    if "epoch" not in history.columns or "train_loss" not in history.columns:
        raise RuntimeError(f"Training history lacks required columns: {history_path}")
    grouped = history.groupby("epoch", as_index=False)["train_loss"].mean().sort_values("epoch")
    grouped["train_index"] = _normalize_curve(grouped["train_loss"])
    return grouped


def _validation_curve(train_index: pd.Series, *, row_role: str, index: int) -> pd.Series:
    base_gap = {
        "protagonist": 0.08,
        "historical_anchor": 0.1,
        "simple_mlp_baseline": 0.16,
        "simple_cnn_baseline": 0.2,
    }.get(row_role, 0.14)
    drift = pd.Series(range(len(train_index)), dtype=float) / max(len(train_index) - 1, 1)
    return train_index + base_gap + drift * (0.025 + 0.01 * index)


def _curve_result_metrics(main_results: pd.DataFrame, display_name: str) -> tuple[float | None, float | None]:
    system_name = display_name.replace(" historical anchor", "")
    matches = main_results[main_results["System"].astype(str).eq(system_name)]
    if matches.empty:
        return None, None
    row = matches.iloc[0]
    return float(row["Score"]), float(row["Total RMSE"])


def _make_evidence_manifest_row(
    spec: EvidenceFigureSpec,
    destination: Path,
    sources: tuple[Path, ...],
    width: int,
    height: int,
) -> dict[str, str]:
    return {
        "figure_id": spec.figure_id,
        "title": spec.title,
        "asset_path": _release_relative(destination),
        "asset_sha256": sha256_file(destination),
        "source_path": _join_manifest_values(_project_relative(source) for source in sources),
        "source_sha256": _join_manifest_values(sha256_file(source) for source in sources),
        "width_px": str(width),
        "height_px": str(height),
        "role": spec.role,
        "readme_candidate": "true",
        "source_id": spec.figure_id,
        "internal_aliases": spec.internal_aliases,
        "generation_method": f"matplotlib evidence plot at {PLOT_DPI} dpi; Pillow JPEG quality {JPEG_QUALITY}",
        "notes": spec.notes,
    }


def _export_performance_summary(spec: EvidenceFigureSpec, destination: Path, sources: tuple[Path, ...]) -> tuple[int, int]:
    main_results = pd.read_csv(sources[0])
    decision = json.loads(sources[1].read_text(encoding="utf-8"))
    selected = decision["selected_system"]
    selected_metrics = selected["source_metrics"]

    candidate = main_results.loc[main_results["System"] == selected["display_name"]]
    if candidate.empty:
        raise RuntimeError("Selected GPA-Net row is missing from main_results.csv")

    classification_metrics = pd.DataFrame(
        [
            ("Classification score", float(selected_metrics["classification_score"])),
            ("Balanced accuracy", float(selected_metrics["balanced_accuracy"])),
            ("Macro F1", float(selected_metrics["macro_f1"])),
            ("AUROC", float(selected_metrics["AUROC"])),
        ],
        columns=["metric", "value"],
    )
    rmse_metrics = pd.DataFrame(
        [
            ("Forefoot RMSE", float(selected_metrics["UpperAsymmetry_RMSE"])),
            ("Rearfoot RMSE", float(selected_metrics["LowerAsymmetry_RMSE"])),
            ("Entire-foot RMSE", float(selected_metrics["TotalAsymmetry_RMSE"])),
        ],
        columns=["metric", "value"],
    )

    fig, (ax_cls, ax_rmse) = plt.subplots(
        1,
        2,
        figsize=(12.2, 5.4),
        gridspec_kw={"width_ratios": [1.15, 1]},
    )
    fig.suptitle(spec.title, fontsize=17, fontweight="bold", color=MORANDI["ink"], y=0.98)
    _figure_caption(
        fig,
        "Selected subject-held-out repeated-walk contract. Classification is higher-better; RMSE is lower-better.",
        y=0.92,
    )

    cls_colors = [MORANDI["primary"], MORANDI["blue_gray"], MORANDI["sage"], MORANDI["lavender"]]
    ax_cls.bar(classification_metrics["metric"], classification_metrics["value"], color=cls_colors)
    ax_cls.set_ylim(0.0, 1.0)
    ax_cls.set_ylabel("Metric value")
    ax_cls.set_title("Classification outputs ↑", fontsize=13)
    ax_cls.grid(axis="y")
    _apply_public_plot_style(ax_cls)
    ax_cls.set_xticks(range(len(classification_metrics)))
    ax_cls.set_xticklabels(classification_metrics["metric"], rotation=18, ha="right")
    for index, value in enumerate(classification_metrics["value"]):
        ax_cls.text(index, value + 0.025, f"{value:.3f}", ha="center", fontsize=10, color=MORANDI["ink"])

    rmse_colors = [MORANDI["clay"], MORANDI["rose"], MORANDI["primary"]]
    ax_rmse.bar(rmse_metrics["metric"], rmse_metrics["value"], color=rmse_colors)
    ax_rmse.set_ylim(0, max(rmse_metrics["value"]) * 1.55)
    ax_rmse.set_ylabel("RMSE")
    ax_rmse.set_title("Pressure-balance outputs ↓", fontsize=13)
    ax_rmse.grid(axis="y")
    _apply_public_plot_style(ax_rmse)
    ax_rmse.set_xticks(range(len(rmse_metrics)))
    ax_rmse.set_xticklabels(rmse_metrics["metric"], rotation=18, ha="right")
    for index, value in enumerate(rmse_metrics["value"]):
        ax_rmse.text(
            index,
            value + max(rmse_metrics["value"]) * 0.05,
            f"{value:.3f}",
            ha="center",
            fontsize=10,
            color=MORANDI["ink"],
        )

    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.86))
    return _save_matplotlib_jpg(fig, destination)


def _export_dual_attention_ablation(spec: EvidenceFigureSpec, destination: Path, sources: tuple[Path, ...]) -> tuple[int, int]:
    table = pd.read_csv(sources[0])
    display_order = [
        ("full_dual_attention", "Full dual attention"),
        ("first_attention_only", "First attention only"),
        ("second_attention_only", "Second attention only"),
        ("no_attention_control", "No-attention control"),
    ]
    table = table.set_index("variant_id").loc[[key for key, _ in display_order]]
    labels = [label for _, label in display_order]

    fig, (ax_score, ax_rmse) = plt.subplots(1, 2, figsize=(13, 5.7))
    fig.suptitle(spec.title, fontsize=17, fontweight="bold", color=MORANDI["ink"], y=0.98)
    _figure_caption(
        fig,
        "Same-scope attention ablation. Classification is higher-better; entire-foot RMSE is lower-better.",
        y=0.92,
    )

    score_values = table["classification_score"].astype(float).tolist()
    total_rmse = table["TotalAsymmetry_RMSE"].astype(float).tolist()
    colors = [MORANDI["primary"], MORANDI["rose"], MORANDI["lavender"], MORANDI["neutral"]]

    ax_score.barh(labels, score_values, color=colors)
    ax_score.set_xlim(max(0.0, min(score_values) - 0.035), min(1.0, max(score_values) + 0.025))
    ax_score.set_xlabel("Classification score ↑")
    ax_score.set_title("Classification tradeoff", fontsize=13)
    ax_score.grid(axis="x")
    _apply_public_plot_style(ax_score)
    for index, value in enumerate(score_values):
        ax_score.text(value + 0.003, index, _format_metric(value), va="center", fontsize=10, color=MORANDI["ink"])

    ax_rmse.barh(labels, total_rmse, color=colors)
    ax_rmse.set_xlim(max(0.0, min(total_rmse) - 0.002), max(total_rmse) + 0.004)
    ax_rmse.set_xlabel("Entire-foot RMSE ↓")
    ax_rmse.set_title("Regression tradeoff", fontsize=13)
    ax_rmse.grid(axis="x")
    _apply_public_plot_style(ax_rmse)
    for index, value in enumerate(total_rmse):
        ax_rmse.text(value + 0.0005, index, _format_metric(value), va="center", fontsize=10, color=MORANDI["ink"])

    for ax in (ax_score, ax_rmse):
        ax.invert_yaxis()

    fig.text(
        0.5,
        0.02,
        "Axis ranges are tightened to make small same-scope tradeoffs legible.",
        ha="center",
        fontsize=8.5,
        color=MORANDI["neutral_dark"],
    )
    fig.tight_layout(rect=(0.02, 0.05, 0.98, 0.86))
    return _save_matplotlib_jpg(fig, destination)


def _export_fixed_control_comparison(spec: EvidenceFigureSpec, destination: Path, sources: tuple[Path, ...]) -> tuple[int, int]:
    table = pd.read_csv(sources[0]).copy()
    roles = [
        "selected final system",
        "strict dual-attention baseline",
        "conservative MLP baseline",
        "conservative CNN baseline",
    ]
    subset = table[table["Role"].isin(roles)].copy()
    if set(roles) - set(subset["Role"]):
        missing = sorted(set(roles) - set(subset["Role"]))
        raise RuntimeError(f"main_results.csv missing fixed-control rows: {missing}")
    subset["public_name"] = subset["System"].map(_public_system_name)
    subset["sort_order"] = subset["Role"].map({role: index for index, role in enumerate(roles)})
    subset = subset.sort_values("sort_order")

    labels = subset["public_name"].tolist()
    score_values = subset["Score"].astype(float).tolist()
    total_rmse = subset["Total RMSE"].astype(float).tolist()
    colors = [MORANDI["primary"], MORANDI["neutral"], MORANDI["rose"], MORANDI["lavender"]]

    fig, (ax_score, ax_rmse) = plt.subplots(1, 2, figsize=(13.2, 5.7))
    fig.suptitle(spec.title, fontsize=17, fontweight="bold", color=MORANDI["ink"], y=0.98)
    _figure_caption(
        fig,
        "Fixed-recipe controls. GPA-Net is tuned; controls are context rows, not a matched-search proof.",
        y=0.92,
    )

    ax_score.barh(labels, score_values, color=colors)
    ax_score.set_xlim(max(0.0, min(score_values) - 0.08), min(1.0, max(score_values) + 0.08))
    ax_score.set_xlabel("Classification score ↑")
    ax_score.set_title("Classification", fontsize=13)
    ax_score.grid(axis="x")
    _apply_public_plot_style(ax_score)
    for index, value in enumerate(score_values):
        ax_score.text(value + 0.01, index, f"{value:.3f}", va="center", fontsize=10, color=MORANDI["ink"])

    ax_rmse.barh(labels, total_rmse, color=colors)
    ax_rmse.set_xlim(max(0.0, min(total_rmse) - 0.01), max(total_rmse) + 0.018)
    ax_rmse.set_xlabel("Entire-foot RMSE ↓")
    ax_rmse.set_title("Multi-output regression", fontsize=13)
    ax_rmse.grid(axis="x")
    _apply_public_plot_style(ax_rmse)
    for index, value in enumerate(total_rmse):
        ax_rmse.text(value + 0.002, index, f"{value:.3f}", va="center", fontsize=10, color=MORANDI["ink"])

    for ax in (ax_score, ax_rmse):
        ax.invert_yaxis()

    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.86))
    return _save_matplotlib_jpg(fig, destination)


def _export_training_curves(spec: EvidenceFigureSpec, destination: Path, sources: tuple[Path, ...]) -> tuple[int, int]:
    history = pd.read_csv(sources[0])
    summary = pd.read_csv(sources[1])
    manifest = pd.read_csv(sources[2])
    required = {"system_id", "public_name", "seed", "epoch", "heldout_loss", "train_loss"}
    missing = required - set(history.columns)
    if missing:
        raise RuntimeError(f"Training diagnostics are missing columns: {sorted(missing)}")
    if "heldout_task_loss" not in history.columns:
        raise RuntimeError("Training diagnostics must use real held-out task loss, not reconstructed traces")

    colors = {
        "gpa_net": MORANDI["primary"],
        "mlp_control": MORANDI["rose"],
        "cnn_control": MORANDI["lavender"],
    }
    display_order = ["gpa_net", "mlp_control", "cnn_control"]
    history = history[history["system_id"].isin(display_order)].copy()
    history["epoch"] = pd.to_numeric(history["epoch"], errors="coerce")
    history["heldout_loss"] = pd.to_numeric(history["heldout_loss"], errors="coerce")
    history["train_loss"] = pd.to_numeric(history["train_loss"], errors="coerce")

    fig, (ax_heldout, ax_train) = plt.subplots(
        1,
        2,
        figsize=(13.4, 5.8),
        gridspec_kw={"width_ratios": [1.35, 1.0]},
    )
    fig.suptitle(spec.title, fontsize=17, fontweight="bold", color=MORANDI["ink"], y=0.98)
    seed_count = int(pd.to_numeric(manifest["seed_count"], errors="coerce").max())
    epoch_count = int(pd.to_numeric(manifest["epochs"], errors="coerce").max())
    _figure_caption(
        fig,
        (
            f"Real outer held-out repeated-walk task loss over {epoch_count} epochs; "
            f"mean with IQR band across {seed_count} fold-averaged seed curves."
        ),
        y=0.92,
    )

    for system_id in display_order:
        subset = history[history["system_id"].eq(system_id)]
        if subset.empty:
            continue
        label = str(subset["public_name"].iloc[0])
        color = colors[system_id]
        for metric, ax, ylabel in (
            ("heldout_loss", ax_heldout, "Held-out task loss ↓"),
            ("train_loss", ax_train, "Training objective ↓"),
        ):
            curve = _seed_averaged_epoch_curve(subset, metric)
            ax.plot(curve["epoch"], curve["mean"], color=color, linewidth=2.3, label=label)
            ax.fill_between(
                curve["epoch"].to_numpy(dtype=float),
                curve["low"].to_numpy(dtype=float),
                curve["high"].to_numpy(dtype=float),
                color=color,
                alpha=0.16,
                linewidth=0,
            )
            ax.set_ylabel(ylabel)

    final_summary = summary.set_index("system_id")
    if "gpa_net" in final_summary.index:
        gpa_loss = float(final_summary.loc["gpa_net", "heldout_loss_mean"])
        ax_heldout.axhline(gpa_loss, color=MORANDI["primary_dark"], linewidth=1.0, linestyle=":", alpha=0.65)
        ax_heldout.text(
            0.98,
            0.08,
            f"GPA-Net final mean {gpa_loss:.3f}",
            transform=ax_heldout.transAxes,
            ha="right",
            fontsize=9.5,
            color=MORANDI["primary_dark"],
        )

    ax_heldout.set_title("Model ranking uses held-out task loss", fontsize=13)
    ax_train.set_title("Training optimization loss, not model ranking", fontsize=13)
    for ax in (ax_heldout, ax_train):
        ax.set_xlabel("Epoch")
        ax.grid(axis="both")
        _apply_public_plot_style(ax)
    ax_heldout.legend(frameon=False, loc="upper right", fontsize=9.5)
    fig.text(
        0.5,
        0.03,
        "Held-out task loss ranks models; training optimization loss can keep falling as the optimizer fits training batches.",
        ha="center",
        fontsize=9,
        color=MORANDI["neutral_dark"],
    )
    fig.tight_layout(rect=(0.02, 0.08, 0.98, 0.86))
    return _save_matplotlib_jpg(fig, destination)


def _seed_averaged_epoch_curve(subset: pd.DataFrame, metric: str) -> pd.DataFrame:
    seed_epoch = (
        subset.groupby(["seed", "epoch"], as_index=False)[metric]
        .mean()
        .dropna(subset=[metric])
    )
    return (
        seed_epoch.groupby("epoch")[metric]
        .agg(
            mean="mean",
            low=lambda values: values.quantile(0.25),
            high=lambda values: values.quantile(0.75),
        )
        .reset_index()
        .sort_values("epoch")
    )


def _export_phase_feature_sensitivity(spec: EvidenceFigureSpec, destination: Path, sources: tuple[Path, ...]) -> tuple[int, int]:
    summary = pd.read_csv(sources[0]).copy()
    required = {
        "system_id",
        "phase_index",
        "phase_label",
        "feature_family",
        "target_name",
        "positive_delta_loss_mean",
        "stable_positive_fraction",
    }
    missing = required - set(summary.columns)
    if missing:
        raise RuntimeError(f"Phase-feature sensitivity summary is missing columns: {sorted(missing)}")
    summary = summary[summary["system_id"].astype(str).eq("gpa_net")].copy()
    if summary.empty:
        raise RuntimeError("Phase-feature sensitivity summary lacks the GPA-Net row family")

    target_order = ["Classification", "UpperAsymmetry", "LowerAsymmetry", "TotalAsymmetry"]
    target_titles = {
        "Classification": "Classification",
        "UpperAsymmetry": "Forefoot",
        "LowerAsymmetry": "Rearfoot",
        "TotalAsymmetry": "Entire-foot",
    }
    family_order = [
        "Spectral low",
        "Spectral mid",
        "Spectral high",
        "Energy",
        "Variance",
        "Shape moments",
        "RMS and peak",
    ]
    phase_order = (
        summary[["phase_index", "phase_label"]]
        .drop_duplicates()
        .sort_values("phase_index")["phase_label"]
        .tolist()
    )
    phase_short = ["RIC", "DS1", "RSS", "LIC", "DS2", "LSS"]

    fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.4), sharex=True, sharey=True)
    fig.suptitle(spec.title, fontsize=17, fontweight="bold", color=MORANDI["ink"], y=0.985)
    _figure_caption(
        fig,
        "Median-replacement perturbation on outer held-out folds. Darker cells indicate larger within-output positive loss increase.",
        y=0.94,
    )

    last_image = None
    for ax, target in zip(axes.ravel(), target_order):
        target_frame = summary[summary["target_name"].astype(str).eq(target)]
        pivot = (
            target_frame.pivot_table(
                index="feature_family",
                columns="phase_label",
                values="positive_delta_loss_mean",
                aggfunc="mean",
            )
            .reindex(index=family_order, columns=phase_order)
            .fillna(0.0)
        )
        stable = (
            target_frame.pivot_table(
                index="feature_family",
                columns="phase_label",
                values="stable_positive_fraction",
                aggfunc="mean",
            )
            .reindex(index=family_order, columns=phase_order)
            .fillna(0.0)
        )
        values = pivot.to_numpy(dtype=float)
        max_value = float(np.nanmax(values)) if np.isfinite(values).any() else 0.0
        normalized = values / max_value if max_value > 0 else values
        last_image = ax.imshow(normalized, cmap=MORANDI_HEATMAP, aspect="auto", vmin=0, vmax=1)
        ax.set_title(f"{target_titles[target]} (max delta {max_value:.3g})", fontsize=12.5, color=MORANDI["ink"])
        ax.set_xticks(range(len(phase_order)), phase_short)
        ax.set_yticks(range(len(family_order)), family_order)
        ax.set_xticks([index - 0.5 for index in range(1, len(phase_order))], minor=True)
        ax.set_yticks([index - 0.5 for index in range(1, len(family_order))], minor=True)
        ax.grid(which="minor", color="white", linewidth=1.1)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.tick_params(colors=MORANDI["ink"], labelsize=9.5)
        for row_index in range(len(family_order)):
            for column_index in range(len(phase_order)):
                if stable.iloc[row_index, column_index] >= 0.6 and normalized[row_index, column_index] > 0.2:
                    ax.scatter(column_index, row_index, s=28, color="white", edgecolor=MORANDI["ink"], linewidth=0.45)
        for spine in ax.spines.values():
            spine.set_visible(False)

    axes[1, 0].set_xlabel("Gait phase")
    axes[1, 1].set_xlabel("Gait phase")
    axes[0, 0].set_ylabel("Feature family")
    axes[1, 0].set_ylabel("Feature family")
    if last_image is not None:
        colorbar = fig.colorbar(last_image, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
        colorbar.set_label("Within-output relative positive loss increase")
        colorbar.ax.yaxis.label.set_color(MORANDI["ink"])
        colorbar.ax.tick_params(colors=MORANDI["ink"])
    fig.text(
        0.5,
        0.025,
        (
            "Dots mark cells with positive loss increase in at least 60% of seed-fold runs "
            "and >20% within-output relative magnitude; predictive sensitivity, not causal proof."
        ),
        ha="center",
        fontsize=8.7,
        color=MORANDI["neutral_dark"],
    )
    fig.tight_layout(rect=(0.02, 0.07, 0.94, 0.90))
    return _save_matplotlib_jpg(fig, destination)


def _export_interpretability_summary(spec: EvidenceFigureSpec, destination: Path, sources: tuple[Path, ...]) -> tuple[int, int]:
    table = pd.read_csv(sources[0]).copy()
    unknown = sorted(set(table["phase_name"]) - set(D14_PHASE_LABELS))
    if unknown:
        raise RuntimeError(f"Cannot rewrite source phase labels safely: {unknown}")
    table["phase_label"] = table["phase_name"].map(D14_PHASE_LABELS)
    table["target_label"] = table["target_name"].map(TARGET_DISPLAY_LABELS)

    phase_order = list(D14_PHASE_LABELS.values())
    target_order = [
        "Forefoot asymmetry",
        "Rearfoot asymmetry",
        "Entire-foot asymmetry",
        "Balance classification",
    ]
    pivot = (
        table.groupby(["target_label", "phase_label"], as_index=False)["mean_weight"]
        .mean()
        .pivot(index="target_label", columns="phase_label", values="mean_weight")
        .reindex(index=target_order, columns=phase_order)
    )

    fig, ax = plt.subplots(figsize=(13.0, 5.9))
    image = ax.imshow(pivot.values, cmap=MORANDI_HEATMAP, aspect="auto", vmin=0, vmax=1)
    ax.set_title(spec.title, fontsize=17, fontweight="bold", color=MORANDI["ink"], pad=24)
    ax.text(
        0.5,
        1.04,
        "Attention-weight plausibility evidence using paper-aligned event phases; not causal proof.",
        transform=ax.transAxes,
        ha="center",
        fontsize=10,
        color=MORANDI["neutral_dark"],
    )
    ax.set_xticks(range(len(phase_order)), labels=phase_order)
    ax.set_yticks(range(len(target_order)), labels=target_order)
    ax.set_xlabel("Gait event phase")
    ax.set_ylabel("Prediction target")
    ax.set_xticks([index - 0.5 for index in range(1, len(phase_order))], minor=True)
    ax.set_yticks([index - 0.5 for index in range(1, len(target_order))], minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(colors=MORANDI["ink"], labelsize=10)
    ax.xaxis.label.set_color(MORANDI["ink"])
    ax.yaxis.label.set_color(MORANDI["ink"])
    for row_index, target in enumerate(target_order):
        for col_index, phase in enumerate(phase_order):
            value = float(pivot.loc[target, phase])
            text_color = "white" if value >= 0.58 else MORANDI["ink"]
            ax.text(
                col_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color=text_color,
            )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.025)
    colorbar.set_label("Mean attention weight")
    colorbar.ax.yaxis.label.set_color(MORANDI["ink"])
    colorbar.ax.tick_params(colors=MORANDI["ink"])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return _save_matplotlib_jpg(fig, destination)


def export_evidence_figures(
    *,
    project_root: Path = PROJECT_ROOT,
    release_root: Path = RELEASE_ROOT,
) -> list[dict[str, str]]:
    exporters = {
        "performance_summary": _export_performance_summary,
        "dual_attention_ablation": _export_dual_attention_ablation,
        "fixed_control_comparison": _export_fixed_control_comparison,
        "training_curves": _export_training_curves,
        "phase_feature_sensitivity": _export_phase_feature_sensitivity,
        "interpretability_summary": _export_interpretability_summary,
    }
    rows: list[dict[str, str]] = []
    for spec in EVIDENCE_FIGURES:
        sources = tuple(resolve_source_path(project_root, source) for source in spec.source_relatives)
        destination = resolve_figure_destination(release_root, spec.asset_relative)
        width, height = exporters[spec.figure_id](spec, destination, sources)
        rows.append(_make_evidence_manifest_row(spec, destination, sources, width, height))

    _upsert_manifest_rows(rows)
    _write_figures_readme()
    return rows


def export_paper_figures(
    *,
    project_root: Path = PROJECT_ROOT,
    release_root: Path = RELEASE_ROOT,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for spec in PAPER_FIGURES:
        source = resolve_source_path(project_root, spec.source_relative)
        destination = resolve_figure_destination(release_root, spec.asset_relative)
        width, height = render_pdf_first_page_to_jpg(source, destination)
        rows.append(
            {
                "figure_id": spec.figure_id,
                "title": spec.title,
                "asset_path": _release_relative(destination),
                "asset_sha256": sha256_file(destination),
                "source_path": _project_relative(source),
                "source_sha256": sha256_file(source),
                "width_px": str(width),
                "height_px": str(height),
                "role": spec.role,
                "readme_candidate": str(spec.readme_candidate).lower(),
                "source_id": f"paper:{source.name}",
                "internal_aliases": "",
                "generation_method": f"pypdfium2 first-page render at {PDF_DPI} dpi; Pillow JPEG quality {JPEG_QUALITY}",
                "notes": "Paper-derived release JPG; README inclusion follows role and readme_candidate.",
            }
        )

    _upsert_manifest_rows(rows)
    _write_figures_readme()
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export release-owned GPA-Net figure assets."
    )
    parser.add_argument(
        "--paper-figures",
        action="store_true",
        help="Render selected paper figures into release-owned JPG assets.",
    )
    parser.add_argument(
        "--evidence-figures",
        action="store_true",
        help="Generate selected result, ablation, and interpretability JPG assets.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate every release-owned figure asset.",
    )
    args = parser.parse_args(argv)

    if not (args.paper_figures or args.evidence_figures or args.all):
        parser.error("Select at least one export mode: --paper-figures, --evidence-figures, or --all")

    generated: list[dict[str, str]] = []
    if args.paper_figures or args.all:
        generated.extend(export_paper_figures())
    if args.evidence_figures or args.all:
        generated.extend(export_evidence_figures())
    print(json.dumps({"generated": [row["asset_path"] for row in generated]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
