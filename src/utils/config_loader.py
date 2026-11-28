"""Load ``config.yaml`` into a plain dictionary."""
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

# Folder that holds config/, data/, src/ (two levels above this file).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

# Sections every other module expects to find.
REQUIRED_SECTIONS = ("data", "features")


def load_config(path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Read the YAML config file.

    Args:
        path: Config file to read. Defaults to ``config/config.yaml`` in the
            project root, wherever the script is run from.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is empty or a required section is missing.
    """
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(f"Config file {config_path} is empty or not a mapping.")

    missing = [name for name in REQUIRED_SECTIONS if name not in config]
    if missing:
        raise ValueError(f"Config file {config_path} is missing section(s): {missing}")
    return config
