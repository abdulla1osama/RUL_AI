from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    _validate_relative_path(config["data"]["root"], "data.root")
    _validate_relative_path(config["data"]["split"], "data.split")
    _validate_relative_path(config["output"]["dir"], "output.dir")

    return config


def _validate_relative_path(path_value: str, field_name: str) -> None:
    path_obj = Path(path_value)

    if path_obj.is_absolute():
        raise ValueError(
            f"{field_name} must use a relative path, got: {path_value}"
        )