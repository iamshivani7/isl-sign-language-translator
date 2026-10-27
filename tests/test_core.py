
import numpy as np
import pandas as pd
import pytest

from isl_core import (
    FEATURE_COUNT,
    feature_columns,
    normalize_landmarks,
    valid_training_data,
)


def test_normalize_returns_42_values():
    points = [[i / 20, i / 40] for i in range(21)]
    result = normalize_landmarks(points)

    assert len(result) == FEATURE_COUNT
    assert result[0] == 0.0
    assert all(np.isfinite(result))


def test_normalize_rejects_wrong_shape():
    with pytest.raises(ValueError):
        normalize_landmarks([[0, 0], [1, 1]])


def test_feature_columns_count():
    assert len(feature_columns()) == 42


def test_valid_training_data_needs_two_labels():
    columns = ["label"] + feature_columns()
    frame = pd.DataFrame(columns=columns)

    assert not valid_training_data(frame)

    row = {"label": "A", **{column: 0.0 for column in feature_columns()}}
    two_labels = pd.DataFrame([
        row,
        {**row, "label": "B"},
    ])

    assert valid_training_data(two_labels)
