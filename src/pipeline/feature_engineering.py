"""Feature engineering module for machine sensor readings.

Provides a configurable ``FeatureEngineer`` class that adds derived columns to
a *cleaned* DataFrame:
  1. Threshold flags (e.g. ``temperature_over_90``): 1 when a reading is at or
     above a cut-off, else 0. Tree models find cut-offs on their own, but linear
     models such as Logistic Regression cannot, so these flags help them a lot.
  2. Rolling-window means per machine (e.g. ``temperature_roll5_mean``): the
     average of a sensor over that machine's last N readings.
  3. Fixed-edge bins (e.g. ``temperature_bin``): the bin number of a reading,
     using edges written in the config, so nothing is learned from the data.
  4. Optional dropping of columns (e.g. weak sensors) once the features above
     have been built.
  5. Column selection: keep only the chosen raw sensors, the columns added
     above, and the id, timestamp and target columns. It runs last, so a
     sensor that is not chosen can still feed a feature built earlier.

The names of the columns added by the last ``transform`` call are kept in
``added_columns``.

Every step is *stateless*: a row's new values depend only on that row, or on
earlier rows of the same machine. Nothing is fitted and the target is never
used, so ``transform`` is safe to call before the train/test split.
Everything is switched on or off, and pointed at different columns, from
``config.yaml`` so the same class works on another dataset.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
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
        bin_edges: Mapping of column name to a list of increasing edges, e.g.
            ``{"temperature": [60, 75, 90]}``. Adds ``<col>_bin`` holding the
            bin number (0 = below the first edge). Missing readings stay NaN.
        drop_columns: Columns to remove at the very end (after the features
            above are built, so a dropped column can still feed a feature).
        id_column: Entity identifier. Rolling windows are computed separately
            for each value, so one machine's readings never mix with another's.
            Use ``None`` if the dataset is a single series.
        timestamp_column: Column that defines the order of readings. If it is
            missing, the existing row order is used.
        sensor_columns: Raw sensor columns to keep, e.g. ``["temperature",
            "vibration"]``. When given, every other raw column is dropped at
            the end (added columns, id, timestamp and target are always kept).
            Leave empty to keep all columns.
        target_column: Target column, always kept by the column selection.
    """

    def __init__(
        self,
        thresholds: Optional[Dict[str, float]] = None,
        add_threshold_flags: bool = True,
        add_rolling_features: bool = False,
        rolling_columns: Optional[List[str]] = None,
        rolling_windows: Optional[List[int]] = None,
        bin_edges: Optional[Dict[str, List[float]]] = None,
        drop_columns: Optional[List[str]] = None,
        id_column: Optional[str] = "machine_id",
        timestamp_column: str = "timestamp",
        sensor_columns: Optional[List[str]] = None,
        target_column: Optional[str] = None,
    ) -> None:
        thresholds = dict(thresholds or {})
        rolling_columns = list(rolling_columns or [])
        rolling_windows = list(rolling_windows or [])
        bin_edges = {col: list(edges) for col, edges in (bin_edges or {}).items()}

        for col, edges in bin_edges.items():
            numeric = all(
                isinstance(e, (int, float)) and not isinstance(e, bool) for e in edges
            )
            if not edges or not numeric or sorted(set(edges)) != edges:
                raise ValueError(
                    f"Bin edges for '{col}' must be a non-empty list of "
                    f"increasing numbers, got {edges!r}."
                )

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
        self.bin_edges = bin_edges
        self.drop_columns = list(drop_columns or [])
        self.id_column = id_column
        self.timestamp_column = timestamp_column
        self.sensor_columns = list(sensor_columns or [])
        self.target_column = target_column
        self.added_columns: List[str] = []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FeatureEngineer":
        """Create a ``FeatureEngineer`` from the parsed ``config.yaml`` dictionary.

        Thresholds, the sensor list and the target are read from ``features``
        (shared with the rest of the pipeline); everything else comes from
        ``feature_engineering``.

        Args:
            config: Full configuration dictionary.

        Returns:
            Configured ``FeatureEngineer`` instance.
        """
        try:
            thresholds = config["features"].get("thresholds", {})
            sensor_columns = config["features"].get("sensors", [])
            target_column = config["features"].get("target")
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
            bin_edges=fe_cfg.get("binning", {}),
            drop_columns=fe_cfg.get("drop_columns", []),
            id_column=id_column,
            timestamp_column=timestamp_column,
            sensor_columns=sensor_columns,
            target_column=target_column,
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add the configured features to a cleaned DataFrame.

        Args:
            df: Cleaned DataFrame (output of ``Preprocessor.clean``).

        Returns:
            New DataFrame with the extra columns added, ``drop_columns``
            removed, and (if ``sensor_columns`` is set) only the chosen
            columns kept. The input is not modified and row order is unchanged.
        """
        if df.empty:
            raise ValueError("Input DataFrame is empty; cannot engineer features.")

        out = df.copy()
        before = set(out.columns)
        if self.add_threshold_flags:
            out = self._add_threshold_flags(out)
        if self.bin_edges:
            out = self._add_bins(out)
        if self.add_rolling_features:
            out = self._add_rolling_features(out)
        out = self._drop_columns(out)
        # Keep only the columns that are new and still present after dropping.
        self.added_columns = [col for col in out.columns if col not in before]
        if self.sensor_columns:
            out = self._select_columns(out)
        return out

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
            # 90.0 is written as 90 so the name stays temperature_over_90.
            label = int(cutoff) if float(cutoff).is_integer() else cutoff
            name = f"{col}_over_{label}"
            # A missing reading (NaN) compares as False, so its flag is 0. This
            # is deliberate: the imputed median (~75 C for temperature) would
            # also be below the cut-off. Use preprocessing.add_missing_indicator
            # if you need to know the reading was missing.
            df[name] = (df[col] >= cutoff).astype(int)
            logger.info("Added threshold flag '%s'.", name)
        return df

    def _add_bins(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ``<col>_bin``: the bin number of each reading, from fixed edges."""
        self._require_columns(df, list(self.bin_edges), "bins")

        for col, edges in self.bin_edges.items():
            name = f"{col}_bin"
            # Bin i holds values in [edges[i-1], edges[i]); 0 is below the first edge.
            binned = pd.Series(np.digitize(df[col], edges), index=df.index, dtype=float)
            df[name] = binned.where(df[col].notna())  # a missing reading has no bin
            logger.info("Added bin feature '%s'.", name)
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

    def _select_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep the chosen sensors, the added columns, and the id/time/target."""
        missing = [col for col in self.sensor_columns if col not in df.columns]
        if len(missing) == len(self.sensor_columns):
            raise ValueError(
                f"None of the sensor columns {self.sensor_columns} exist. "
                f"Columns available: {sorted(df.columns)}"
            )
        if missing:
            logger.warning("Sensor column(s) not found, skipped: %s", missing)

        wanted = set(self.sensor_columns) | set(self.added_columns)
        wanted |= {
            col
            for col in (self.id_column, self.timestamp_column, self.target_column)
            if col is not None
        }
        kept = [col for col in df.columns if col in wanted]
        dropped = [col for col in df.columns if col not in wanted]
        if dropped:
            logger.info("Column selection dropped: %s", dropped)
        return df[kept].copy()

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
