"""Shared interface for every model.

A model is a scikit-learn ``Pipeline`` of two steps:

  ``transform`` -> imputation, encoding and scaling (from ``Transformation``)
  ``model``     -> the estimator itself (made by the subclass)

Because the transformation sits inside the pipeline, it is refitted on the
training rows only: in ``fit``, in every cross-validation fold, and in every
hyper-parameter search candidate. Nothing is learned from the test set.

To add a model, subclass ``BaseModel``, set ``name`` (the key under ``models:``
in config.yaml) and implement ``_make_estimator``.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline

from src.pipeline.transformation import Transformation

logger = logging.getLogger(__name__)

# Pipeline step names. Search grids use "model__<param>" for the estimator.
TRANSFORM_STEP = "transform"
MODEL_STEP = "model"


class BaseModel(ABC):
    """Base class: builds, tunes, fits, scores and saves one model.

    Args:
        transformation: Builds the imputation/encoding/scaling step.
        params: Estimator hyper-parameters (passed to ``_make_estimator``).
        param_grid: Candidate values per parameter, used when tuning.
        cv_folds: Number of folds for cross-validation and tuning.
        scoring: scikit-learn scoring name, e.g. ``"balanced_accuracy"``.
        random_state: Seed for the folds and the search.
    """

    #: Key of this model under ``models:`` in config.yaml (set by subclasses).
    name: str = "base"
    #: Scaler to use instead of the configured one (``None`` keeps the config).
    scaler_override: Optional[str] = None

    def __init__(
        self,
        transformation: Transformation,
        params: Optional[Dict[str, Any]] = None,
        param_grid: Optional[Dict[str, List[Any]]] = None,
        cv_folds: int = 5,
        scoring: str = "balanced_accuracy",
        random_state: int = 42,
    ) -> None:
        if cv_folds < 2:
            raise ValueError(f"cv_folds must be at least 2, got {cv_folds}.")
        self.transformation = transformation
        self.params = dict(params or {})
        self.param_grid = dict(param_grid or {})
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.random_state = random_state

        self.pipeline: Pipeline = self._build_pipeline()
        self.best_params_: Optional[Dict[str, Any]] = None

    @classmethod
    def from_config(
        cls, config: Dict[str, Any], transformation: Transformation
    ) -> "BaseModel":
        """Create a model from the ``models`` section of config.yaml."""
        models_cfg = config.get("models", {})
        model_cfg = models_cfg.get(cls.name, {})
        return cls(
            transformation=transformation,
            params=model_cfg.get("params", {}),
            param_grid=model_cfg.get("param_grid", {}),
            cv_folds=int(models_cfg.get("cv_folds", 5)),
            scoring=models_cfg.get("scoring", "balanced_accuracy"),
            random_state=int(models_cfg.get("random_state", 42)),
        )

    # ------------------------------------------------------------------ #
    # To implement in each model
    # ------------------------------------------------------------------ #
    @abstractmethod
    def _make_estimator(self, **params: Any) -> BaseEstimator:
        """Return a new, unfitted estimator built from ``params``."""

    # ------------------------------------------------------------------ #
    # Building
    # ------------------------------------------------------------------ #
    def _build_pipeline(self) -> Pipeline:
        return Pipeline(
            steps=[
                (TRANSFORM_STEP, self.transformation.build(self.scaler_override)),
                (MODEL_STEP, self._make_estimator(**self.params)),
            ]
        )

    def _cv(self) -> StratifiedKFold:
        return StratifiedKFold(
            n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
        )

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        tuning: Optional[Dict[str, Any]] = None,
    ) -> "BaseModel":
        """Fit on the training split, tuning hyper-parameters first if asked.

        Args:
            X_train: Training features (after feature engineering and split).
            y_train: Training target.
            tuning: The ``models.tuning`` config section. Tuning runs only when
                ``enabled`` is true and the model has a ``param_grid``.
        """
        tuning = tuning or {}
        if tuning.get("enabled") and self.param_grid:
            self._tune(X_train, y_train, tuning)
        else:
            self.pipeline.fit(X_train, y_train)
        return self

    def _tune(
        self, X_train: pd.DataFrame, y_train: pd.Series, tuning: Dict[str, Any]
    ) -> None:
        """Search the param grid with cross-validation on the training data."""
        method = tuning.get("method", "random")
        grid = {f"{MODEL_STEP}__{key}": values for key, values in self.param_grid.items()}

        if method == "grid":
            search: Union[GridSearchCV, RandomizedSearchCV] = GridSearchCV(
                self.pipeline,
                grid,
                cv=self._cv(),
                scoring=self.scoring,
                n_jobs=-1,
            )
        elif method == "random":
            n_candidates = int(np.prod([len(values) for values in grid.values()]))
            n_iter = min(int(tuning.get("n_iter", 10)), n_candidates)
            search = RandomizedSearchCV(
                self.pipeline,
                grid,
                n_iter=n_iter,
                cv=self._cv(),
                scoring=self.scoring,
                n_jobs=-1,
                random_state=self.random_state,
            )
        else:
            raise ValueError(f"Unknown tuning method '{method}'. Use 'grid' or 'random'.")

        search.fit(X_train, y_train)
        # refit=True (default): best_estimator_ is refitted on all training rows.
        self.pipeline = search.best_estimator_
        self.best_params_ = {
            key.removeprefix(f"{MODEL_STEP}__"): value
            for key, value in search.best_params_.items()
        }
        logger.info(
            "%s tuning (%s) best %s=%.4f with %s",
            self.name,
            method,
            self.scoring,
            search.best_score_,
            self.best_params_,
        )

    # ------------------------------------------------------------------ #
    # Prediction and scoring
    # ------------------------------------------------------------------ #
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict the class of each row."""
        return self.pipeline.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities (one column per class)."""
        return self.pipeline.predict_proba(X)

    def cross_validate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Cross-validate on the training data to check bias and variance.

        The pipeline is cloned, so the fitted model is left untouched.

        Returns:
            Dict with ``train_score``, ``cv_score`` and ``cv_std``. A train
            score far above the CV score suggests overfitting; both low and
            close together suggests underfitting.
        """
        results = cross_validate(
            clone(self.pipeline),
            X,
            y,
            cv=self._cv(),
            scoring=self.scoring,
            return_train_score=True,
            n_jobs=-1,
        )
        return {
            "train_score": float(np.mean(results["train_score"])),
            "cv_score": float(np.mean(results["test_score"])),
            "cv_std": float(np.std(results["test_score"])),
        }

    def feature_importances(self) -> Optional[pd.Series]:
        """Importance of each transformed feature, or ``None`` if unsupported."""
        return None

    def feature_names(self) -> List[str]:
        """Names of the columns the estimator sees (after transformation)."""
        return list(self.pipeline.named_steps[TRANSFORM_STEP].get_feature_names_out())

    # ------------------------------------------------------------------ #
    # Saving
    # ------------------------------------------------------------------ #
    def save(self, path: Union[str, Path]) -> Path:
        """Save the fitted pipeline with joblib and return the path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)
        logger.info("Saved %s to %s.", self.name, path)
        return path

    @staticmethod
    def load(path: Union[str, Path]) -> Pipeline:
        """Load a saved pipeline (it can ``predict`` on raw split features)."""
        return joblib.load(path)
