"""Small registry utilities used across BasicSTFM.

The framework intentionally keeps the registry simple: each component declares a
string name, configs select that name through a ``type`` field, and the remaining
keys become constructor arguments.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Dict, Iterable, Optional, TypeVar

T = TypeVar("T")


class Registry:
    """Map string names to callables/classes and build them from config dicts."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._items: Dict[str, Callable[..., Any]] = {}

    def __contains__(self, key: str) -> bool:
        return key in self._items

    def __repr__(self) -> str:
        keys = ", ".join(sorted(self._items))
        return f"Registry(name={self.name!r}, items=[{keys}])"

    def keys(self) -> Iterable[str]:
        return self._items.keys()

    def get(self, key: str) -> Callable[..., Any]:
        try:
            return self._items[key]
        except KeyError as exc:
            available = ", ".join(sorted(self._items)) or "<empty>"
            raise KeyError(
                f"{key!r} is not registered in registry {self.name!r}. "
                f"Available: {available}"
            ) from exc

    def register(
        self,
        name: Optional[str] = None,
        *,
        force: bool = False,
    ) -> Callable[[T], T]:
        """Register an object.

        Usage:
            @MODELS.register()
            class MyModel: ...

            @LOSSES.register("masked_mae")
            class MaskedMAE: ...
        """

        def decorator(obj: T) -> T:
            key = name or getattr(obj, "__name__", None)
            if not key:
                raise ValueError(f"Cannot infer registry name for object {obj!r}")
            if key in self._items and not force:
                raise KeyError(f"{key!r} is already registered in {self.name!r}")
            self._items[key] = obj  # type: ignore[assignment]
            return obj

        return decorator

    def build(self, cfg: Dict[str, Any], **extra_kwargs: Any) -> Any:
        """Instantiate a registered component from a config dict.

        Supported forms:
            {"type": "Foo", "a": 1}
            {"type": "Foo", "params": {"a": 1}}

        Direct keys outside ``params`` override same-named values inside
        ``params``. ``extra_kwargs`` override both.
        """

        if not isinstance(cfg, dict):
            raise TypeError(f"Registry.build expects a dict, got {type(cfg)!r}")
        if "type" not in cfg:
            raise KeyError(f"Config for registry {self.name!r} must contain a 'type' key")

        cfg = deepcopy(cfg)
        obj_type = cfg.pop("type")
        params = cfg.pop("params", {}) or {}
        if not isinstance(params, dict):
            raise TypeError("'params' must be a mapping when provided")
        params.update(cfg)
        params.update(extra_kwargs)
        return self.get(str(obj_type))(**params)


MODELS = Registry("models")
DATAMODULES = Registry("datamodules")
TASKS = Registry("tasks")
LOSSES = Registry("losses")
METRICS = Registry("metrics")
TRAINERS = Registry("trainers")
