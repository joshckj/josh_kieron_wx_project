"""Data preprocessing module for machine sensor readings.

Provides a configurable ``Preprocessor`` class that handles the *stateless*
steps that are safe to run before splitting:
  1. Cleaning (dropping leakage columns, fixing sign-flipped vibration
     readings, deduplicating, parsing timestamps, and sorting).
  2. Train/test splitting (stratified or chronological).

Anything that learns from the data (imputation, encoding, scaling) lives in
``transformation.py`` and is fitted on the training rows only.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

_VALID_SPLIT_STRATEGIES = {"stratified", "time"}


class Preprocessor:
    """Cleans and splits machine sensor data.

    Pipeline order: ``clean()`` -> feature engineering -> ``split()`` ->
    ``Transformation`` (fitted on the training split only).

    Args:
        target_column: Name of the target column.
        leakage_columns: Columns to drop during ``clean()`` to prevent leakage.
        id_column: Machine identifier column.
        timestamp_column: Datetime column used for sorting and time-based splits.
        drop_leakage: Whether to drop ``leakage_columns`` in ``clean()``.
        fix_negative_vibration: Whether to convert negative ``vibration``
            readings using ``abs()`` in ``clean()``.
        drop_duplicates: Whether to drop exact duplicate rows in ``clean()``.
        split_strategy: Splitting method (``"stratified"`` or ``"time"``).
        test_size: Fraction of rows reserved for the test set (0 < x < 1).
        random_state: Random seed for reproducible stratified splits.
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
