from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


METADATA_COLUMNS = [
    "subject_id",
    "trial_id",
    "trial_key",
    "condition_label",
    "abnormal_type",
    "source_file",
    "pressure_source_file",
    "gait_cycle_index",
]
TRIAL_GROUP_COLUMNS = [
    "subject_id",
    "trial_id",
    "trial_key",
    "condition_label",
    "abnormal_type",
    "source_file",
    "pressure_source_file",
]
PHASE_FEATURE_COLUMNS = [f"feature_{idx:03d}" for idx in range(1, 97)]
SCALAR_FEATURE_COLUMN = "feature_097"
PRESSURE_COLUMNS = [
    "leftUpper_combined",
    "rightUpper_combined",
    "leftLower_combined",
    "rightLower_combined",
    "lefttotal",
    "righttotal",
]
REQUIRED_COLUMNS = METADATA_COLUMNS + PHASE_FEATURE_COLUMNS + [SCALAR_FEATURE_COLUMN] + PRESSURE_COLUMNS


def load_canonical_dataset(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Canonical dataset is missing required columns: {missing_columns}")

    return df


def split_phase_features(df: pd.DataFrame) -> tuple[list[np.ndarray], np.ndarray]:
    phase_inputs: list[np.ndarray] = []

    for phase_idx in range(6):
        start = phase_idx * 16
        end = start + 16
        columns = PHASE_FEATURE_COLUMNS[start:end]
        phase_inputs.append(df.loc[:, columns].to_numpy(dtype=np.float32))

    scalar_feature = df.loc[:, [SCALAR_FEATURE_COLUMN]].to_numpy(dtype=np.float32)
    return phase_inputs, scalar_feature


def dataset_root(project_root: str | Path) -> Path:
    return Path(project_root) / "Processed_canonical" / "datasets" / "canonical_dataset.csv"
