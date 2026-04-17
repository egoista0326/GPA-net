from __future__ import annotations

import numpy as np
import pandas as pd


EPSILON = 1e-10


def attach_targets(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()

    upper_left = enriched["leftUpper_combined"].to_numpy(dtype=float)
    upper_right = enriched["rightUpper_combined"].to_numpy(dtype=float)
    lower_left = enriched["leftLower_combined"].to_numpy(dtype=float)
    lower_right = enriched["rightLower_combined"].to_numpy(dtype=float)
    total_left = enriched["lefttotal"].to_numpy(dtype=float)
    total_right = enriched["righttotal"].to_numpy(dtype=float)

    enriched["UpperAsymmetry"] = np.abs(upper_left - upper_right) / (
        upper_left + upper_right + EPSILON
    )
    enriched["LowerAsymmetry"] = np.abs(lower_left - lower_right) / (
        lower_left + lower_right + EPSILON
    )
    enriched["TotalAsymmetry"] = np.abs(total_left - total_right) / (
        total_left + total_right + EPSILON
    )
    enriched["Classification"] = (enriched["condition_label"].str.lower() == "abnormal").astype(int)

    return enriched
