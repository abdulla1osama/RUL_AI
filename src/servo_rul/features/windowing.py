from typing import Literal

import numpy as np
import pandas as pd

def create_sliding_windows(
    df: pd.DataFrame,
    feature_columns: list[str],
    window_size: int,
    target_column: str = "rul_capped",
) -> tuple[np.ndarray, np.ndarray]:
    windows: list[np.ndarray] = []
    targets: list[float] = []

    if window_size < 1:
        raise ValueError(f"window_size must be >= 1, got {window_size}")

    for _, unit_df in df.groupby("unit_id"):
        unit_df = unit_df.sort_values("cycle")

        feature_values = unit_df[feature_columns].to_numpy(dtype=np.float32)
        target_values = unit_df[target_column].to_numpy(dtype=np.float32)

        for end_idx in range(len(unit_df)):
            start_idx = max(0, end_idx - window_size + 1)

            window = feature_values[start_idx : end_idx + 1]

            real_mask = np.ones(
                (len(window), 1),
                dtype=np.float32,
            )

            window = np.concatenate(
                [window, real_mask],
                axis=1,
            )

            if len(window) < window_size:
                pad_count = window_size - len(window)

                pad = np.zeros(
                    (pad_count, len(feature_columns) + 1),
                    dtype=np.float32,
                )

                window = np.concatenate(
                    [pad, window],
                    axis=0,
                )

            target = target_values[end_idx]

            windows.append(window)
            targets.append(target)

    if not windows:
        raise ValueError("No windows were created. Check window_size and dataframe.")

    X = np.stack(windows)
    y = np.array(targets, dtype=np.float32)

    return X, y


def flatten_windows(X: np.ndarray) -> np.ndarray:
    if X.ndim != 3:
        raise ValueError(f"Expected X with 3 dimensions, got shape {X.shape}")

    n_samples = X.shape[0]
    return X.reshape(n_samples, -1)


def prepare_model_input(X: np.ndarray,model_type: Literal["linear", "mlp", "cnn1d"],) -> np.ndarray:
    
    if model_type in {"linear", "mlp"}:
        return flatten_windows(X)

    if model_type == "cnn1d":
        return X

    raise ValueError(f"Unknown model_type: {model_type}")