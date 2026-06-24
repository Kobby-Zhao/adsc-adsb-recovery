
import os
from copy import deepcopy
from typing import Any, Dict

import pandas as pd
import yaml


def load_settings(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_parquet(df: pd.DataFrame, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    df.to_parquet(path, index=False)


def write_csv(df: pd.DataFrame, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=False)


def write_schema_markdown(path: str, columns: Dict[str, str], title: str) -> None:
    ensure_dir(os.path.dirname(path))
    lines = [f"# {title}", "", "| 列名 | 含义 |", "| --- | --- |"]
    for name, desc in columns.items():
        lines.append(f"| {name} | {desc} |")
    content = "\n".join(lines) + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
