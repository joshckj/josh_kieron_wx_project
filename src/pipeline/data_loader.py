"""Data ingestion: build a SQLite database from the raw CSV and read it back.

The CSV is loaded into SQLite once (``build_database``); every later run reads
the data with SQL (``load``) instead of touching the CSV again.

The build is safe against two problems:
  * Interrupted builds: rows go into a temporary table that only replaces the
    real table once every chunk has loaded, so a crash never leaves a
    half-filled table behind.
  * Stale data: the CSV's modified time and size are stored in a small
    ``_build_info`` table, so the database is rebuilt automatically if the CSV
    changes.
"""
import logging
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Small table that remembers which CSV version each data table was built from.
BUILD_INFO_TABLE = "_build_info"


class DataLoader:
    """Ingests machine readings from a SQLite database.

    Args:
        csv_path: Path to the raw CSV file.
        db_path: Path of the SQLite database file (created if missing).
        table_name: Table that holds the readings.
        chunksize: Rows per insert when building the database.
        timestamp_column: Column parsed as datetime when loading.
        required_columns: Columns that must exist in the dataset. ``load``
            raises a ``ValueError`` listing any that are missing. Extra columns
            are always allowed. Leave as None to skip the check.
    """

    def __init__(
        self,
        csv_path: str,
        db_path: str,
        table_name: str = "machine_readings",
        chunksize: int = 20_000,
        timestamp_column: str = "timestamp",
        required_columns: Optional[List[str]] = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.db_path = Path(db_path)
        self.table_name = table_name
        self.chunksize = chunksize
        self.timestamp_column = timestamp_column
        self.required_columns = list(required_columns or [])
        # Rows are loaded here first, then swapped in once the load succeeds.
        self._tmp_table = f"{table_name}_tmp"

    @classmethod
    def from_config(cls, config: dict) -> "DataLoader":
        """Create a loader from the ``data`` section of config.yaml."""
        data_cfg = config["data"]
        return cls(
            csv_path=data_cfg["csv_path"],
            db_path=data_cfg["db_path"],
            table_name=data_cfg.get("table_name", "machine_readings"),
            chunksize=data_cfg.get("csv_chunksize", 20_000),
            timestamp_column=data_cfg.get("timestamp_column", "timestamp"),
            required_columns=data_cfg.get("required_columns"),
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _table_exists(self, conn: sqlite3.Connection, name: str) -> bool:
        """Return True if a table called ``name`` exists in the database."""
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (name,),
        ).fetchone()
        return row is not None

    def _csv_signature(self) -> tuple:
        """Fingerprint of the CSV (modified time, size) used to detect changes."""
        stat = self.csv_path.stat()
        return (stat.st_mtime, stat.st_size)

    def _is_stale(self, conn: sqlite3.Connection) -> bool:
        """Return True if the table is missing or was built from a different CSV."""
        if not self._table_exists(conn, self.table_name):
            return True
        if not self._table_exists(conn, BUILD_INFO_TABLE):
            return True  # built before we tracked the CSV, so rebuild to be safe

        row = conn.execute(
            f"SELECT csv_mtime, csv_size FROM {BUILD_INFO_TABLE} WHERE table_name=?",
            (self.table_name,),
        ).fetchone()
        return row is None or tuple(row) != self._csv_signature()

    def _validate_columns(self, conn: sqlite3.Connection) -> None:
        """Raise a clear error if any required column is missing from the table.

        Checks the table itself (not the query result), so a query that selects
        only some columns still passes as long as the dataset has them all.
        """
        if not self.required_columns:
            return

        # PRAGMA table_info returns one row per column; index 1 is the name.
        rows = conn.execute(f'PRAGMA table_info("{self.table_name}")').fetchall()
        present = {row[1] for row in rows}
        missing = [col for col in self.required_columns if col not in present]
        if missing:
            raise ValueError(
                f"Dataset {self.csv_path} is missing required column(s): {missing}. "
                f"Columns found: {sorted(present)}"
            )

    def _load_csv_into_tmp(self, conn: sqlite3.Connection) -> int:
        """Stream the CSV into the temporary table chunk by chunk.

        Returns the number of rows loaded.
        """
        conn.execute(f'DROP TABLE IF EXISTS "{self._tmp_table}"')  # clear leftovers
        rows = 0
        for chunk in pd.read_csv(self.csv_path, chunksize=self.chunksize):
            # pandas commits once per chunk, which keeps inserts fast.
            chunk.to_sql(self._tmp_table, conn, if_exists="append", index=False)
            rows += len(chunk)
        return rows

    def _swap_in_tmp(self, conn: sqlite3.Connection) -> None:
        """Replace the real table with the temporary one in a single transaction."""
        conn.execute("BEGIN")
        try:
            conn.execute(f'DROP TABLE IF EXISTS "{self.table_name}"')
            conn.execute(f'ALTER TABLE "{self._tmp_table}" RENAME TO "{self.table_name}"')
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS {BUILD_INFO_TABLE} "
                "(table_name TEXT PRIMARY KEY, csv_mtime REAL, csv_size INTEGER)"
            )
            conn.execute(
                f"INSERT OR REPLACE INTO {BUILD_INFO_TABLE} VALUES (?, ?, ?)",
                (self.table_name, *self._csv_signature()),
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")  # keep the old table if the swap fails
            raise

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def build_database(self, overwrite: bool = False) -> None:
        """Load the CSV into the SQLite table.

        Skipped when the table already exists and the CSV has not changed,
        unless ``overwrite`` is True.

        Raises:
            FileNotFoundError: If the CSV does not exist.
        """
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # closing() makes sure the file is released (matters on Windows).
        with closing(sqlite3.connect(self.db_path)) as conn:
            if not overwrite and not self._is_stale(conn):
                logger.info("Table '%s' is up to date, skipping build.", self.table_name)
                return

            try:
                rows = self._load_csv_into_tmp(conn)
                self._swap_in_tmp(conn)
            except Exception:
                # Drop the partial table so the next run starts clean.
                conn.execute(f'DROP TABLE IF EXISTS "{self._tmp_table}"')
                raise
            logger.info("Loaded %d rows into %s.", rows, self.db_path)

    def load(self, query: Optional[str] = None) -> pd.DataFrame:
        """Read readings from SQLite into a DataFrame.

        Args:
            query: Optional SQL to run. Defaults to selecting the whole table.

        Raises:
            ValueError: If a required column is missing from the dataset.
        """
        self.build_database()  # no-op if the database is already up to date
        query = query or f'SELECT * FROM "{self.table_name}"'
        with closing(sqlite3.connect(self.db_path)) as conn:
            self._validate_columns(conn)  # fail early with a clear message
            df = pd.read_sql_query(query, conn)

        # SQLite stores timestamps as text, so convert back to datetime.
        if self.timestamp_column in df.columns:
            df[self.timestamp_column] = pd.to_datetime(df[self.timestamp_column])
        logger.info("Loaded %d rows, %d columns.", *df.shape)
        return df
