"""Training, loading and persistence for the JA4+ Random Forest model."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
except ModuleNotFoundError as exc:
    np = None
    pd = None
    RandomForestClassifier = Any
    _ML_IMPORT_ERROR = exc
else:
    _ML_IMPORT_ERROR = None

from current.services.loaders import load_database_records
from current.core.models import DatabaseRecord, Observation
from current.services.feature_parser import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    records_to_training_samples,
)


@dataclass(slots=True)
class RandomForestBundle:
    """Serializable model bundle with encoders and label metadata."""

    classifier: RandomForestClassifier
    categorical_maps: dict[str, dict[str, int]]
    label_to_index: dict[str, int]
    index_to_label: dict[int, str]
    label_to_category: dict[str, str | None]
    feature_columns: list[str]


class JA4RandomForestModel:
    """Wrapper around a Random Forest classifier for JA4+ data."""

    def __init__(self, bundle: RandomForestBundle):
        self.bundle = bundle

    @classmethod
    def train(
        cls,
        records: list[DatabaseRecord],
        *,
        random_state: int = 42,
        n_estimators: int = 200,
    ) -> "JA4RandomForestModel":
        """Train a Random Forest model from normalized records."""
        _ensure_ml_dependencies()
        samples = records_to_training_samples(records)
        if not samples:
            raise ValueError("Training data does not contain any labeled samples.")

        frame = pd.DataFrame(samples)
        categorical_maps: dict[str, dict[str, int]] = {}

        for column in CATEGORICAL_FEATURES:
            categories = sorted(frame[column].astype(str).unique().tolist())
            categorical_maps[column] = {value: index for index, value in enumerate(categories)}
            frame[column] = frame[column].astype(str).map(categorical_maps[column]).astype(int)

        for column in NUMERIC_FEATURES:
            frame[column] = frame[column].fillna(0).astype(int)

        labels = sorted(frame["label"].astype(str).unique().tolist())
        label_to_index = {label: index for index, label in enumerate(labels)}
        index_to_label = {index: label for label, index in label_to_index.items()}
        frame["target"] = frame["label"].astype(str).map(label_to_index).astype(int)

        label_to_category: dict[str, str | None] = {}
        for _, row in frame[["label", "category"]].drop_duplicates().iterrows():
            label_to_category[str(row["label"])] = row["category"] if pd.notna(row["category"]) else None

        classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=1,
        )
        classifier.fit(
            frame[FEATURE_COLUMNS],
            frame["target"],
            sample_weight=frame["weight"],
        )

        return cls(
            RandomForestBundle(
                classifier=classifier,
                categorical_maps=categorical_maps,
                label_to_index=label_to_index,
                index_to_label=index_to_label,
                label_to_category=label_to_category,
                feature_columns=list(FEATURE_COLUMNS),
            )
        )

    @classmethod
    def train_from_json(
        cls,
        dataset_path: str | Path,
        *,
        random_state: int = 42,
        n_estimators: int = 200,
    ) -> "JA4RandomForestModel":
        """Train a model from one of the repository JSON datasets."""
        records = load_database_records(dataset_path)
        return cls.train(records, random_state=random_state, n_estimators=n_estimators)

    @classmethod
    def load(cls, model_path: str | Path) -> "JA4RandomForestModel":
        """Load a previously saved model from disk."""
        path = Path(model_path)
        with path.open("rb") as handle:
            bundle = pickle.load(handle)
        if not isinstance(bundle, RandomForestBundle):
            raise TypeError(f"Unexpected model bundle type in {model_path}")
        return cls(bundle)

    def save(self, model_path: str | Path) -> Path:
        """Persist the model bundle to disk."""
        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self.bundle, handle)
        return path

    def predict_probabilities(self, encoded_row: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Predict class probabilities for an already encoded row."""
        _ensure_ml_dependencies()
        probabilities = self.bundle.classifier.predict_proba(encoded_row)[0]
        class_ids = self.bundle.classifier.classes_
        return probabilities, class_ids


def _ensure_ml_dependencies() -> None:
    """Raise a clear error when optional ML dependencies are unavailable."""
    if _ML_IMPORT_ERROR is not None:
        missing_name = getattr(_ML_IMPORT_ERROR, "name", "optional ML dependency")
        raise ModuleNotFoundError(
            "Random Forest features require optional dependencies "
            f"that are not installed: {missing_name}."
        ) from _ML_IMPORT_ERROR
