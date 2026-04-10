"""Config loading and lightweight override helpers."""

from __future__ import annotations

import json
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, MutableMapping, Optional

_ENV_PATTERN = re.compile(r"\$\{env:([^,}]+)(?:,([^}]*))?\}")


def load_config(path: str, overrides: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """Load JSON or YAML config and apply dotted-key overrides.

    YAML support is provided by PyYAML. JSON remains available with only the
    standard library, which keeps lightweight tests and config tooling usable in
    minimal Python environments.
    """

    cfg_path = Path(path)
    suffix = cfg_path.suffix.lower()
    text = cfg_path.read_text(encoding="utf-8")

    if suffix == ".json":
        cfg = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "PyYAML is required to load YAML configs. Install with `pip install -e .` "
                "or use a JSON config."
            ) from exc
        cfg = yaml.safe_load(text)
    else:
        raise ValueError(f"Unsupported config extension: {suffix!r}")

    if cfg is None:
        cfg = {}
    if not isinstance(cfg, dict):
        raise TypeError(f"Top-level config must be a mapping, got {type(cfg)!r}")

    cfg = resolve_env_vars(cfg)
    for override in overrides or []:
        key, value = parse_override(override)
        set_by_dotted_key(cfg, key, value)
    return cfg


def parse_override(override: str) -> Any:
    """Parse ``a.b=value`` into a key path and a JSON-ish value."""

    if "=" not in override:
        raise ValueError(f"Override must use key=value form, got {override!r}")
    key, raw_value = override.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError("Override key cannot be empty")

    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError:
        value = raw_value
    return key, value


def set_by_dotted_key(cfg: MutableMapping[str, Any], dotted_key: str, value: Any) -> None:
    """Set a nested config value using ``foo.bar.0.baz`` style paths."""

    parts = dotted_key.split(".")
    cursor: Any = cfg
    for part in parts[:-1]:
        if isinstance(cursor, list):
            cursor = cursor[int(part)]
            continue
        if part not in cursor or not isinstance(cursor[part], (dict, list)):
            cursor[part] = {}
        cursor = cursor[part]

    last = parts[-1]
    if isinstance(cursor, list):
        cursor[int(last)] = value
    else:
        cursor[last] = value


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Return ``base`` recursively merged with ``override``."""

    merged = deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def resolve_env_vars(value: Any) -> Any:
    """Resolve ``${env:NAME,default}`` placeholders in string config values."""

    if isinstance(value, dict):
        return {k: resolve_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve_env_vars(v) for v in value]
    if isinstance(value, str):
        return _ENV_PATTERN.sub(_replace_env, value)
    return value


def _replace_env(match: re.Match[str]) -> str:
    name = match.group(1)
    default = match.group(2)
    if name in os.environ:
        return os.environ[name]
    if default is not None:
        return default
    raise KeyError(f"Environment variable {name!r} is not set and no default was provided")
