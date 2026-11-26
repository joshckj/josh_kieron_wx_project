"""Data preprocessing module for machine sensor readings.

Provides a configurable ``Preprocessor`` class that handles:
  1. Stateless cleaning (dropping leakage columns, fixing sign-flipped
     vibration readings, deduplicating, parsing timestamps, and sorting).
  2. Train/test splitting (stratified or chronological) before any stateful
     fitting to prevent data leakage.
  3. Stateful transformations fit strictly on training data (missing-value
     imputation, optional categorical encoding of ``machine_id``, and scaling)
     via ``scikit-learn``'s ``ColumnTransformer`` with pandas DataFrame output.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

logger = logging.getLogger(__name__)

_VALID_SPLIT_STRATEGIES = {"stratified", "time"}
_VALID_IMPUTER_STRATEGIES = {"mean", "median", "most_frequent"}
_VALID_SCALERS = {"standard", "robust", "none"}


class Preprocessor:
    """Cleans, splits, imputes, encodes, and scales machine sensor data.

    Designed to work cleanly with a downstream ``feature_engineering`` step:
      ``clean()`` -> ``split()`` -> (optional feature engineering) ->
      ``fit_transform(X_train)`` / ``transform(X_test)``.

    Args:
        target_column: Name of the target column.
        leakage_columns: Columns to drop during ``clean()`` to prevent leakage.
        id_column: Machine identifier column (treated as categorical, never
            scaled as a numeric quantity).
        timestamp_column: Datetime column used for sorting and time-based splits.
        drop_leakage: Whether to drop ``leakage_columns`` in ``clean()``.
        fix_negative_vibration: Whether to convert negative ``vibration``
            readings using ``abs()`` in ``clean()``.
        drop_duplicates: Whether to drop exact duplicate rows in ``clean()``.
        split_strategy: Splitting method (``"stratified"`` or ``"time"``).
        test_size: Fraction of rows reserved for the test set (0 < x < 1).
        random_state: Random seed for reproducible stratified splits.
        imputer_strategy: ``SimpleImputer`` strategy for numeric columns
            (``"mean"``, ``"median"``, or ``"most_frequent"``).
        add_missing_indicator: Whether to append binary missingness indicator
            columns during imputation.
        scaler: Scaling method for numeric columns (``"standard"``,
            ``"robust"``, or ``"none"``).
        encode_machine_id: If True, one-hot encode ``id_column``; if False,
            drop ``id_column`` before modeling.
    """

    def __init__(
        self,
        target_column: str = "maintenance_required",
        leakage_columns: Optional[List[str]] = None,
        id_column: str = "machine_id",
        timestamp_column: str = "timestamp",
        drop_leakage: bool = True,
        fix_negative_vibration: bool = True,
        drop_duplicates: bool = True,
        split_strategy: str = "stratified",
        test_size: float = 0.2,
        random_state: int = 42,
        imputer_strategy: str = "median",
        add_missing_indicator: bool = False,
        scaler: str = "standard",
        encode_machine_id: bool = False,
    ) -> None:
        if split_strategy not in _VALID_SPLIT_STRATEGIES:
            raise ValueError(
                f"Invalid split_strategy '{split_strategy}'. "
                f"Expected one of {sorted(_VALID_SPLIT_STRATEGIES)}."
            )
        if not 0.0 < test_size < 1.0:
            raise ValueError(
                f"test_size must be between 0 and 1, got {test_size}."
            )
        if imputer_strategy not in _VALID_IMPUTER_STRATEGIES:
            raise ValueError(
                f"Invalid imputer_strategy '{imputer_strategy}'. "
                f"Expected one of {sorted(_VALID_IMPUTER_STRATEGIES)}."
            )
        if scaler not in _VALID_SCALERS:
            raise ValueError(
                f"Invalid scaler '{scaler}'. "
                f"Expected one of {sorted(_VALID_SCALERS)}."
            )

        self.target_column = target_column
        self.leakage_columns = list(leakage_columns or [])
        self.id_column = id_column
        self.timestamp_column = timestamp_column
        self.drop_leakage = drop_leakage
        self.fix_negative_vibration = fix_negative_vibration
        self.drop_duplicates = drop_duplicates
        self.split_strategy = split_strategy
        self.test_size = test_size
        self.random_state = random_state
        self.imputer_strategy = imputer_strategy
        self.add_missing_indicator = add_missing_indicator
        self.scaler = scaler
        self.encode_machine_id = encode_machine_id

        self._transformer: Optional[ColumnTransformer] = None
        self._numeric_features: List[str] = []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Preprocessor":
        """Create a ``Preprocessor`` from the parsed ``config.yaml`` dictionary.

        Args:
            config: Full configuration dictionary containing ``data``,
                ``features``, and ``preprocessing`` sections.

        Returns:
            Configured ``Preprocessor`` instance.
        """
        try:
            data_cfg = config.get("data", {})
            feat_cfg = config["features"]
            prep_cfg = config.get("preprocessing", {})
            split_cfg = prep_cfg.get("split", {})
        except (KeyError, TypeError) as exc:
            raise ValueError(
                "Invalid config structure for Preprocessor.from_config."
            ) from exc

        return cls(
            target_column=feat_cfg["target"],
            leakage_columns=feat_cfg.get("exclude_leakage", []),
            id_column=prep_cfg.get("id_column", "machine_id"),
            timestamp_column=data_cfg.get("timestamp_column", "timestamp"),
            drop_leakage=prep_cfg.get("drop_leakage", True),
            fix_negative_vibration=prep_cfg.get("fix_negative_vibration", True),
            drop_duplicates=prep_cfg.get("drop_duplicates", True),
            split_strategy=split_cfg.get("strategy", "stratified"),
            test_size=float(split_cfg.get("test_size", 0.2)),
            random_state=int(split_cfg.get("random_state", 42)),
            imputer_strategy=prep_cfg.get("imputer_strategy", "median"),
            add_missing_indicator=bool(prep_cfg.get("add_missing_indicator", False)),
            scaler=prep_cfg.get("scaler", "standard"),
            encode_machine_id=bool(prep_cfg.get("encode_machine_id", False)),
        )

    # ------------------------------------------------------------------ #
    # Stage 1: Stateless Cleaning (safe before splitting)
    # ------------------------------------------------------------------ #
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply stateless row-level cleaning to a raw DataFrame.

        Args:
            df: Raw machine readings DataFrame.

        Returns:
            Cleaned copy of the DataFrame.
        """
        if df.empty:
            raise ValueError("Input DataFrame is empty; cannot clean data.")

        cleaned = df.copy()

        # Drop derived columns that leak the target before any downstream work.
        if self.drop_leakage and self.leakage_columns:
            to_drop = [col for col in self.leakage_columns if col in cleaned.columns]
            if to_drop:
                cleaned = cleaned.drop(columns=to_drop)
                logger.info("Dropped leakage columns: %s", to_drop)

        # Negative vibration readings mirror the positive distribution (sign-flip
        # sensor artifact), so taking abs() restores true vibration magnitude.
        if self.fix_negative_vibration and "vibration" in cleaned.columns:
            neg_count = int((cleaned["vibration"] < 0).sum())
            if neg_count > 0:
                cleaned["vibration"] = cleaned["vibration"].abs()
                logger.info("Fixed %d negative vibration readings with abs().", neg_count)

        if self.drop_duplicates:
            before = len(cleaned)
            cleaned = cleaned.drop_duplicates()
            dropped = before - len(cleaned)
            if dropped > 0:
                logger.info("Dropped %d exact duplicate rows.", dropped)

        if self.timestamp_column in cleaned.columns:
            cleaned[self.timestamp_column] = pd.to_datetime(
                cleaned[self.timestamp_column], errors="raise"
            )

        # Sort deterministically by machine and time for downstream rolling/time ops.
        sort_cols = [
            col
            for col in (self.id_column, self.timestamp_column)
            if col in cleaned.columns
        ]
        if sort_cols:
            cleaned = cleaned.sort_values(sort_cols).reset_index(drop=True)

        return cleaned

    # ------------------------------------------------------------------ #
    # Stage 2: Train / Test Split (before any stateful fitting)
    # ------------------------------------------------------------------ #
    def split(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split a cleaned DataFrame into train and test feature/target sets.

        Args:
            df: Cleaned DataFrame containing ``target_column``.

        Returns:
            Tuple of ``(X_train, X_test, y_train, y_test)``.
        """
        if self.target_column not in df.columns:
            raise ValueError(
                f"Target column '{self.target_column}' not found in DataFrame."
            )

        if self.split_strategy == "time":
            if self.timestamp_column not in df.columns:
                raise ValueError(
                    f"Time split requested, but '{self.timestamp_column}' is missing."
                )
            ordered = df.sort_values(self.timestamp_column).reset_index(drop=True)
            split_idx = int(len(ordered) * (1.0 - self.test_size))
            train_df = ordered.iloc[:split_idx]
            test_df = ordered.iloc[split_idx:]
            X_train = train_df.drop(columns=[self.target_column])
            y_train = train_df[self.target_column]
            X_test = test_df.drop(columns=[self.target_column])
            y_test = test_df[self.target_column]
            return X_train, X_test, y_train, y_test

        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            stratify=y,
            random_state=self.random_state,
        )
        return X_train, X_test, y_train, y_test

    # ------------------------------------------------------------------ #
    # Stage 3: Stateful Fit & Transform (imputation, encoding, scaling)
    # ------------------------------------------------------------------ #
    def _build_scaler(self) -> Any:
        """Return the configured scikit-learn scaler or 'passthrough'."""
        if self.scaler == "standard":
            return StandardScaler()
        if self.scaler == "robust":
            return RobustScaler()
        return "passthrough"

    def _prepare_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        """Prepare input features by casting id_column to string if encoded."""
        prepared = X.copy()
        if self.encode_machine_id and self.id_column in prepared.columns:
            # Cast integer IDs to string so OneHotEncoder treats them as labels.
            prepared[self.id_column] = prepared[self.id_column].astype(str)
        return prepared

    def fit(
        self, X_train: pd.DataFrame, y_train: Optional[pd.Series] = None
    ) -> "Preprocessor":
        """Fit imputation, encoding, and scaling transformers on training data.

        Args:
            X_train: Training feature DataFrame (after optional feature engineering).
            y_train: Unused; accepted for scikit-learn API compatibility.

        Returns:
            Fitted ``Preprocessor`` instance.
        """
        if X_train.empty:
            raise ValueError("X_train is empty; cannot fit Preprocessor.")

        prepared = self._prepare_frame(X_train)

        # Exclude identifier, timestamp, and target columns from numeric scaling.
        excluded = {self.id_column, self.timestamp_column, self.target_column}
        self._numeric_features = [
            col
            for col in prepared.select_dtypes(include=["number", "bool"]).columns
            if col not in excluded
        ]
        if not self._numeric_features:
            raise ValueError(
                "No numeric feature columns found to preprocess in X_train."
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
                ("scaler", self._build_scaler()),
            ]
        )

        transformers: List[Tuple[str, Any, List[str]]] = [
            ("num", numeric_pipeline, self._numeric_features)
        ]

        if self.encode_machine_id:
            if self.id_column not in prepared.columns:
                raise ValueError(
                    f"encode_machine_id=True, but '{self.id_column}' is missing."
                )
            cat_pipeline = Pipeline(
                steps=[
                    (
                        "encoder",
                        OneHotEncoder(
                            handle_unknown="ignore", sparse_output=False
                        ),
                    )
                ]
            )
            transformers.append(("cat", cat_pipeline, [self.id_column]))

        # remainder="drop" drops raw timestamp (and machine_id when not encoded).
        self._transformer = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            verbose_feature_names_out=False,
        )
        self._transformer.set_output(transform="pandas")
        self._transformer.fit(prepared, y_train)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform a feature DataFrame using the fitted transformers.

        Args:
            X: Feature DataFrame to transform.

        Returns:
            Transformed DataFrame with preserved column names.
        """
        if self._transformer is None:
            raise RuntimeError(
                "Preprocessor must be fitted before calling transform()."
            )
        prepared = self._prepare_frame(X)
        return self._transformer.transform(prepared)

    def fit_transform(
        self, X_train: pd.DataFrame, y_train: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """Fit on ``X_train`` and return the transformed training DataFrame.

        Args:
            X_train: Training feature DataFrame.
            y_train: Unused; accepted for scikit-learn API compatibility.

        Returns:
            Transformed training DataFrame.
        """
        return self.fit(X_train, y_train).transform(X_train)
