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
    cfg = expand_dataset_registry(cfg)
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


def expand_dataset_registry(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Expand dataset registry references inside config mappings.

    Supported helpers:

    - ``dataset_key``: expand one dataset entry into a single-dataset mapping.
    - ``dataset_keys``: expand several dataset entries into ``datasets``.
    - ``dataset_group``: expand one top-level dataset group into ``datasets``.
    - ``dataset_groups``: expand several top-level dataset groups into ``datasets``.
    """

    registry = cfg.get("dataset_registry")
    if registry is None:
        return cfg
    if not isinstance(registry, dict):
        raise TypeError("dataset_registry must be a mapping")

    groups = cfg.get("dataset_groups", {})
    if groups is None:
        groups = {}
    if not isinstance(groups, dict):
        raise TypeError("dataset_groups must be a mapping when provided")

    return _expand_dataset_node(cfg, registry=registry, groups=groups, is_root=True)


def _replace_env(match: re.Match[str]) -> str:
    name = match.group(1)
    default = match.group(2)
    if name in os.environ:
        return os.environ[name]
    if default is not None:
        return default
    raise KeyError(f"Environment variable {name!r} is not set and no default was provided")


def _expand_dataset_node(
    value: Any,
    *,
    registry: Dict[str, Any],
    groups: Dict[str, Any],
    is_root: bool = False,
) -> Any:
    if isinstance(value, list):
        return [
            _expand_dataset_node(item, registry=registry, groups=groups, is_root=False)
            for item in value
        ]
    if not isinstance(value, dict):
        return value

    special_keys = {"dataset_key", "dataset_keys", "dataset_group", "dataset_groups"}
    expanded = {}
    for key, item in value.items():
        if key in special_keys:
            continue
        if is_root and key in {"dataset_registry", "dataset_groups"}:
            expanded[key] = item
            continue
        expanded[key] = _expand_dataset_node(item, registry=registry, groups=groups, is_root=False)

    dataset_key = None if is_root else value.get("dataset_key")
    dataset_keys = [] if is_root else _resolve_dataset_key_list(value, groups)

    if dataset_key is not None and dataset_keys:
        raise ValueError("Use either dataset_key or dataset_group(s)/dataset_keys, not both")

    if dataset_key is not None:
        entry = _lookup_dataset_entry(str(dataset_key), registry)
        return deep_merge(entry, expanded)

    if dataset_keys:
        merged = dict(expanded)
        datasets = [
            _expand_dataset_node(_lookup_dataset_entry(key, registry), registry=registry, groups=groups)
            for key in dataset_keys
        ]
        explicit = merged.get("datasets", [])
        if explicit:
            if not isinstance(explicit, list):
                raise TypeError("Expanded 'datasets' field must be a list")
            datasets.extend(explicit)
        merged["datasets"] = datasets
        return merged

    return expanded


def _resolve_dataset_key_list(value: Dict[str, Any], groups: Dict[str, Any]) -> list[str]:
    keys: list[str] = []

    dataset_group = value.get("dataset_group")
    if dataset_group is not None:
        keys.extend(_lookup_dataset_group(str(dataset_group), groups))

    dataset_groups = value.get("dataset_groups")
    if dataset_groups is not None:
        if isinstance(dataset_groups, str):
            dataset_groups = [dataset_groups]
        if not isinstance(dataset_groups, list):
            raise TypeError("dataset_groups must be a string or list of strings")
        for group_name in dataset_groups:
            keys.extend(_lookup_dataset_group(str(group_name), groups))

    dataset_keys = value.get("dataset_keys")
    if dataset_keys is not None:
        if isinstance(dataset_keys, str):
            dataset_keys = [dataset_keys]
        if not isinstance(dataset_keys, list):
            raise TypeError("dataset_keys must be a string or list of strings")
        keys.extend(str(item) for item in dataset_keys)

    return keys


def _lookup_dataset_entry(name: str, registry: Dict[str, Any]) -> Dict[str, Any]:
    if name not in registry:
        available = ", ".join(sorted(registry)) or "<empty>"
        raise KeyError(f"Unknown dataset registry key {name!r}. Available: {available}")
    entry = registry[name]
    if not isinstance(entry, dict):
        raise TypeError(f"dataset_registry[{name!r}] must be a mapping")
    return deepcopy(entry)


def _lookup_dataset_group(name: str, groups: Dict[str, Any]) -> list[str]:
    if name not in groups:
        available = ", ".join(sorted(groups)) or "<empty>"
        raise KeyError(f"Unknown dataset group {name!r}. Available: {available}")
    keys = groups[name]
    if isinstance(keys, str):
        return [keys]
    if not isinstance(keys, list):
        raise TypeError(f"dataset_groups[{name!r}] must be a string or list of strings")
    return [str(item) for item in keys]
