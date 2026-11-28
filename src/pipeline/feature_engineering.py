"""Feature engineering module for machine sensor readings.

Provides a configurable ``FeatureEngineer`` class that adds derived columns to
a *cleaned* DataFrame:
  1. Threshold flags (e.g. ``temperature_over_90``): 1 when a reading is at or
     above a cut-off, else 0. Tree models find cut-offs on their own, but linear
     models such as Logistic Regression cannot, so these flags help them a lot.
  2. Rolling-window means per machine (e.g. ``temperature_roll5_mean``): the
     average of a sensor over that machine's last N readings.
  3. Optional dropping of columns (e.g. weak sensors) once the features above
     have been built.

Every step is *stateless*: a row's new values depend only on that row, or on
earlier rows of the same machine. Nothing is fitted and the target is never
used, so ``transform`` is safe to call before the train/test split.
Everything is switched on or off, and pointed at different columns, from
``config.yaml`` so the same class works on another dataset.
"""

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Adds threshold flags and rolling-window features to cleaned readings.

    Args:
        thresholds: Mapping of column name to cut-off, e.g.
            ``{"temperature": 90, "vibration": 80}``. Used for threshold flags.
        add_threshold_flags: Whether to add one ``<col>_over_<cutoff>`` flag per
            entry in ``thresholds``.
        add_rolling_features: Whether to add rolling-window mean columns.
        rolling_columns: Columns to compute rolling means for.
        rolling_windows: Window sizes, in number of readings (e.g. ``[3, 5, 10]``).
        drop_columns: Columns to remove at the very end (after the features
            above are built, so a dropped column can still feed a feature).
        id_column: Entity identifier. Rolling windows are computed separately
            for each value, so one machine's readings never mix with another's.
            Use ``None`` if the dataset is a single series.
        timestamp_column: Column that defines the order of readings. If it is
            missing, the existing row order is used.
    """

    def __init__(
        self,
        thresholds: Optional[Dict[str, float]] = None,
        add_threshold_flags: bool = True,
        add_rolling_features: bool = False,
        rolling_columns: Optional[List[str]] = None,
        rolling_windows: Optional[List[int]] = None,
        drop_columns: Optional[List[str]] = None,
        id_column: Optional[str] = "machine_id",
        timestamp_column: str = "timestamp",
    ) -> None:
        thresholds = dict(thresholds or {})
        rolling_columns = list(rolling_columns or [])
        rolling_windows = list(rolling_windows or [])

        if add_threshold_flags and not thresholds:
            raise ValueError(
                "add_threshold_flags=True, but no thresholds were given."
            )
        for col, cutoff in thresholds.items():
            if isinstance(cutoff, bool) or not isinstance(cutoff, (int, float)):
                raise ValueError(
                    f"Threshold for '{col}' must be a number, got {cutoff!r}."
                )
        if add_rolling_features:
            if not rolling_columns:
                raise ValueError(
                    "add_rolling_features=True, but rolling_columns is empty."
                )
            if not rolling_windows:
                raise ValueError(
                    "add_rolling_features=True, but rolling_windows is empty."
                )
        for window in rolling_windows:
            if isinstance(window, bool) or not isinstance(window, int) or window < 1:
                raise ValueError(
                    f"Rolling windows must be positive integers, got {window!r}."
                )

        self.thresholds = thresholds
        self.add_threshold_flags = add_threshold_flags
        self.add_rolling_features = add_rolling_features
        self.rolling_columns = rolling_columns
        self.rolling_windows = rolling_windows
        self.drop_columns = list(drop_columns or [])
        self.id_column = id_column
        self.timestamp_column = timestamp_column

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FeatureEngineer":
        """Create a ``FeatureEngineer`` from the parsed ``config.yaml`` dictionary.

        Thresholds are read from ``features.thresholds`` (shared with the rest
        of the pipeline); everything else comes from ``feature_engineering``.

        Args:
            config: Full configuration dictionary.

        Returns:
            Configured ``FeatureEngineer`` instance.
        """
        try:
            thresholds = config["features"].get("thresholds", {})
            fe_cfg = config.get("feature_engineering", {})
            id_column = config.get("preprocessing", {}).get("id_column", "machine_id")
            timestamp_column = config.get("data", {}).get(
                "timestamp_column", "timestamp"
            )
        except (KeyError, AttributeError, TypeError) as exc:
            raise ValueError(
                "Invalid config structure for FeatureEngineer.from_config."
            ) from exc

        return cls(
            thresholds=thresholds,
            add_threshold_flags=bool(fe_cfg.get("add_threshold_flags", True)),
            add_rolling_features=bool(fe_cfg.get("add_rolling_features", False)),
            rolling_columns=fe_cfg.get("rolling_columns", []),
            rolling_windows=fe_cfg.get("rolling_windows", []),
            drop_columns=fe_cfg.get("drop_columns", []),
            id_column=id_column,
            timestamp_column=timestamp_column,
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add the configured features to a cleaned DataFrame.

        Args:
            df: Cleaned DataFrame (output of ``Preprocessor.clean``).

        Returns:
            New DataFrame with the extra columns added and ``drop_columns``
            removed. The input is not modified and row order is unchanged.
        """
        if df.empty:
            raise ValueError("Input DataFrame is empty; cannot engineer features.")

        out = df.copy()
        if self.add_threshold_flags:
            out = self._add_threshold_flags(out)
        if self.add_rolling_features:
            out = self._add_rolling_features(out)
        return self._drop_columns(out)

    # ------------------------------------------------------------------ #
    # Steps
    # ------------------------------------------------------------------ #
    @staticmethod
    def _require_columns(df: pd.DataFrame, columns: List[str], purpose: str) -> None:
        """Raise a clear error if any of ``columns`` is missing from ``df``."""
        missing = [col for col in columns if col not in df.columns]
        if missing:
            raise ValueError(
                f"Cannot build {purpose}: column(s) {missing} not found. "
                f"Columns available: {sorted(df.columns)}"
            )

    def _add_threshold_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add a 0/1 column per threshold: 1 when the value is >= the cut-off."""
        self._require_columns(df, list(self.thresholds), "threshold flags")

        for col, cutoff in self.thresholds.items():
            name = f"{col}_over_{cutoff}"
            # A missing reading (NaN) compares as False, so its flag is 0. This
            # is deliberate: the imputed median (~75 C for temperature) would
            # also be below the cut-off. Use preprocessing.add_missing_indicator
            # if you need to know the reading was missing.
            df[name] = (df[col] >= cutoff).astype(int)
            logger.info("Added threshold flag '%s'.", name)
        return df

    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling means of each column over each window, per machine."""
        self._require_columns(df, self.rolling_columns, "rolling features")
        if self.id_column is not None:
            self._require_columns(df, [self.id_column], "rolling features")

        # Rolling windows need time order. Sort a view (not df itself) so the
        # caller's row order is kept; results are matched back by index.
        sort_cols = [
            col
            for col in (self.id_column, self.timestamp_column)
            if col is not None and col in df.columns
        ]
        if self.timestamp_column not in df.columns:
            logger.warning(
                "Timestamp column '%s' not found; rolling windows use the "
                "existing row order.",
                self.timestamp_column,
            )
        ordered = df.sort_values(sort_cols, kind="stable") if sort_cols else df

        # Group by machine so one machine's window never includes another's rows.
        # With no id_column the whole frame is treated as one series.
        groups = ordered.groupby(self.id_column) if self.id_column else None

        for col in self.rolling_columns:
            series = groups[col] if groups is not None else ordered[col]
            for window in self.rolling_windows:
                name = f"{col}_roll{window}_mean"
                # Windows look backwards only (the default; never center=True),
                # so a row only sees its own and earlier readings.
                # min_periods=1 gives the first rows a mean of what exists so far
                # instead of NaN.
                rolled = series.rolling(window, min_periods=1).mean()
                if groups is not None:
                    # Drop the machine_id level so the result lines up with df's index.
                    rolled = rolled.reset_index(level=0, drop=True)
                df[name] = rolled
                logger.info("Added rolling feature '%s'.", name)
        return df

    def _drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop ``drop_columns`` (missing ones are skipped with a warning)."""
        if not self.drop_columns:
            return df
        present = [col for col in self.drop_columns if col in df.columns]
        absent = [col for col in self.drop_columns if col not in df.columns]
        if absent:
            logger.warning(
                "drop_columns not found (already dropped by the loader?): %s", absent
            )
        if present:
            df = df.drop(columns=present)
            logger.info("Dropped columns: %s", present)
        return df
