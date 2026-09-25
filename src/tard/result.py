from collections.abc import Hashable
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DistortionResult:
    distortion: pd.DataFrame
    signed_change: pd.DataFrame
    nodes: list[Hashable]
    scales: list[int]

    @property
    def global_distortion(self) -> float:
        """Mean distortion over all node-scale cells that are not NaN."""
        values = self.distortion.to_numpy()
        valid_values = values[~np.isnan(values)]
        if valid_values.size == 0:
            return float("nan")
        return float(valid_values.mean())
