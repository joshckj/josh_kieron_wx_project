"""Fitted transformations: imputation, encoding and scaling.

Everything here *learns from data* (medians, means, categories), so it must be
fitted on the training rows only. ``Transformation.build`` returns an unfitted
scikit-learn ``ColumnTransformer``; each model puts it inside its own
``Pipeline`` so it is refitted correctly in every cross-validation fold and
during hyper-parameter search.

Scaling options (see course notes, "Feature Scaling"):
  * ``standard``: (x - mean) / std
  * ``minmax``:   (x - min) / (max - min)
  * ``robust``:   uses median and IQR, less sensitive to outliers
  * ``none``:     leave values as they are (tree models do not need scaling)
"""

import logging
from typing import Any, Dict, List, Optional

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)

logger = logging.getLogger(__name__)

_VALID_IMPUTER_STRATEGIES = {"mean", "median", "most_frequent"}
_VALID_SCALERS = {"standard", "minmax", "robust", "none"}


class _NumericSelector:
    """Pick numeric/bool columns, skipping id, timestamp and target.

    A class (not a lambda) so a fitted pipeline can be saved with joblib.
    """

    def __init__(self, excluded: List[str]) -> None:
        self.excluded = tuple(excluded)

    def __call__(self, X: pd.DataFrame) -> List[str]:
        numeric = X.select_dtypes(include=["number", "bool"]).columns
        return [col for col in numeric if col not in self.excluded]


class Transformation:
    """Builds the imputation + encoding + scaling step.

    Args:
        id_column: Machine identifier. One-hot encoded if ``encode_machine_id``
            is True, otherwise dropped before modelling.
        timestamp_column: Never used as a model input (dropped).
        target_column: Never transformed (safety net; it should already be
            separated from the features).
        imputer_strategy: ``"mean"``, ``"median"`` or ``"most_frequent"``.
        add_missing_indicator: Add a 0/1 column for each column that had
            missing values in the training data.
        scaler: ``"standard"``, ``"minmax"``, ``"robust"`` or ``"none"``.
        encode_machine_id: One-hot encode ``id_column`` when True.
    """

    def __init__(
        self,
        id_column: str = "machine_id",
        timestamp_column: str = "timestamp",
        target_column: str = "maintenance_required",
        imputer_strategy: str = "median",
        add_missing_indicator: bool = False,
        scaler: str = "standard",
        encode_machine_id: bool = False,
    ) -> None:
        if imputer_strategy not in _VALID_IMPUTER_STRATEGIES:
            raise ValueError(
                f"Invalid imputer_strategy '{imputer_strategy}'. "
                f"Expected one of {sorted(_VALID_IMPUTER_STRATEGIES)}."
            )
        if scaler not in _VALID_SCALERS:
            raise ValueError(
                f"Invalid scaler '{scaler}'. Expected one of {sorted(_VALID_SCALERS)}."
            )
        self.id_column = id_column
        self.timestamp_column = timestamp_column
        self.target_column = target_column
        self.imputer_strategy = imputer_strategy
        self.add_missing_indicator = add_missing_indicator
        self.scaler = scaler
        self.encode_machine_id = encode_machine_id

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Transformation":
        """Create a ``Transformation`` from the parsed ``config.yaml``."""
        try:
            target = config["features"]["target"]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                "Invalid config structure for Transformation.from_config."
            ) from exc
        tr_cfg = config.get("transformation", {})
        return cls(
            id_column=config.get("preprocessing", {}).get("id_column", "machine_id"),
            timestamp_column=config.get("data", {}).get(
                "timestamp_column", "timestamp"
            ),
            target_column=target,
            imputer_strategy=tr_cfg.get("imputer_strategy", "median"),
            add_missing_indicator=bool(tr_cfg.get("add_missing_indicator", False)),
            scaler=tr_cfg.get("scaler", "standard"),
            encode_machine_id=bool(tr_cfg.get("encode_machine_id", False)),
        )

    @staticmethod
    def _make_scaler(name: str) -> Any:
        if name == "standard":
            return StandardScaler()
        if name == "minmax":
            return MinMaxScaler()
        if name == "robust":
            return RobustScaler()
        return "passthrough"

    def build(self, scaler: Optional[str] = None) -> ColumnTransformer:
        """Return a new, unfitted ``ColumnTransformer``.

        Args:
            scaler: Override the configured scaler for one model, e.g. a tree
                model passes ``"none"``. ``None`` keeps the configured one.
        """
        scaler_name = self.scaler if scaler is None else scaler
        if scaler_name not in _VALID_SCALERS:
            raise ValueError(
                f"Invalid scaler '{scaler_name}'. Expected one of {sorted(_VALID_SCALERS)}."
            )

        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy=self.imputer_strategy,
                        add_indicator=self.add_missing_indicator,
                    ),
                ),
                ("scaler", self._make_scaler(scaler_name)),
            ]
        )
        excluded = [self.id_column, self.timestamp_column, self.target_column]
        transformers = [("num", numeric_pipeline, _NumericSelector(excluded))]

        if self.encode_machine_id:
            transformers.append(
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    [self.id_column],
                )
            )

        # remainder="drop" removes the timestamp (and machine_id when not encoded).
        transformer = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            verbose_feature_names_out=False,
        )
        transformer.set_output(transform="pandas")
        return transformer
