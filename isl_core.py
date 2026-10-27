"""Core helpers for the sign language translator prototype."""

import numpy as np

FEATURE_COUNT = 42


def normalize_landmarks(points):
    """Normalize 21 hand landmarks into 42 numeric features."""
    arr = np.asarray(points, dtype=float)

    if arr.shape != (21, 2):
        raise ValueError(
            "Expected 21 hand landmarks with x and y coordinates."
        )

    # Use the wrist as the origin.
    arr = arr - arr[0]

    # Scale coordinates to reduce size differences.
    scale = float(np.max(np.abs(arr)))

    if scale > 0:
        arr = arr / scale

    return arr.flatten().tolist()


def feature_columns():
    """Return names for the 42 landmark features."""
    return [
        f"{axis}{i}"
        for i in range(21)
        for axis in ("x", "y")
    ]


def valid_training_data(frame):
    """Check whether data contains at least two gesture labels."""
    return (
        frame is not None
        and not frame.empty
        and "label" in frame.columns
        and all(
            column in frame.columns
            for column in feature_columns()
        )
        and frame["label"].nunique() >= 2
    )
