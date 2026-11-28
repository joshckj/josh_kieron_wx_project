"""Entry point for the pipeline.

Run from anywhere with ``python -m src.main`` (from the project root).

Order of the steps:
  1. data_loader: load the readings from SQLite
  2. preprocessing.clean: stateless cleaning
  3. feature_engineering: row-level features
  4. data_loader.select_columns: keep the chosen columns
  5. preprocessing.split: train/test split
  6. transformation: impute / encode / scale, fitted on the train split only
  7. models: train and check (model_1 only for now)
"""

import logging
import os
from typing import Any, Dict, Optional

from sklearn.metrics import balanced_accuracy_score

from src.models.model_1 import GradientBoostingModel
from src.pipeline.data_loader import DataLoader
from src.pipeline.feature_engineering import FeatureEngineer
from src.pipeline.preprocessing import Preprocessor
from src.pipeline.transformation import Transformation
from src.utils.config_loader import PROJECT_ROOT, load_config
from src.utils.logger import setup_logging

logger = logging.getLogger(__name__)

SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"


def run(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Run the whole pipeline and return the main results."""
    # Config paths such as data/... are relative to the project root.
    os.chdir(PROJECT_ROOT)

    config = load_config(config_path)
    setup_logging(config.get("logging", {}).get("level", "INFO"))

    loader = DataLoader.from_config(config)
    preprocessor = Preprocessor.from_config(config)
    engineer = FeatureEngineer.from_config(config)
    transformation = Transformation.from_config(config)

    # 1-3. Load, clean, and add row-level features.
    raw = loader.load()
    cleaned = preprocessor.clean(raw)
    featured = engineer.transform(cleaned)

    # 4. Choose the columns to use (sensors, engineered ones, and the keys).
    keep = (
        config["features"].get("sensors", [])
        + engineer.added_columns
        + [preprocessor.id_column, preprocessor.timestamp_column]
        + [preprocessor.target_column]
    )
    selected = loader.select_columns(featured, keep)

    # 5. Split before anything is fitted.
    X_train, X_test, y_train, y_test = preprocessor.split(selected)
    logger.info(
        "Split: train %s, test %s; positive rate train %.3f, test %.3f.",
        X_train.shape,
        X_test.shape,
        y_train.mean(),
        y_test.mean(),
    )

    # 6. Fit the transformation on train only, then check the result.
    transformer = transformation.build()
    train_ready = transformer.fit_transform(X_train)
    test_ready = transformer.transform(X_test)
    logger.info(
        "Transformed: train %s, test %s; NaN left: train %d, test %d.",
        train_ready.shape,
        test_ready.shape,
        int(train_ready.isna().sum().sum()),
        int(test_ready.isna().sum().sum()),
    )
    logger.info("Model input columns: %s", list(train_ready.columns))

    # 7. Train model_1 and check how it does (full evaluation comes later).
    models_cfg = config.get("models", {})
    model = GradientBoostingModel.from_config(config, transformation)
    model.fit(X_train, y_train, tuning=models_cfg.get("tuning"))

    scores = model.cross_validate(X_train, y_train)
    test_score = float(balanced_accuracy_score(y_test, model.predict(X_test)))
    logger.info(
        "%s balanced accuracy: train(CV) %.4f | CV %.4f (+/- %.4f) | test %.4f",
        model.name,
        scores["train_score"],
        scores["cv_score"],
        scores["cv_std"],
        test_score,
    )

    model.save(SAVED_MODELS_DIR / f"{model.name}.joblib")
    # TODO: add model_2 / model_3 here when they are implemented.
    return {"model": model, "cv": scores, "test_score": test_score}


if __name__ == "__main__":
    run()
