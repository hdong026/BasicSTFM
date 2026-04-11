"""Command line interface for BasicSTFM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, Optional

from .builders import import_builtin_components, import_custom_modules
from .config import load_config
from .engines.stage import StagePlan
from .registry import TRAINERS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BasicSTFM experiment runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("train", "dry-run", "print-config"):
        sub = subparsers.add_parser(name)
        sub.add_argument("config", help="Path to a JSON/YAML experiment config")
        sub.add_argument(
            "--cfg-options",
            nargs="*",
            default=[],
            help="Override config values, e.g. trainer.device=cpu pipeline.stages.0.epochs=1",
        )

    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    _prepare_import_paths(args.config)
    cfg: Dict[str, Any] = load_config(args.config, overrides=args.cfg_options)

    if args.command == "print-config":
        print(json.dumps(cfg, indent=2, sort_keys=True))
        return

    if args.command == "dry-run":
        print(json.dumps(StagePlan.from_config(cfg).describe(), indent=2))
        return

    import_builtin_components()
    import_custom_modules(cfg.get("custom_imports", []))

    trainer_cfg = dict(cfg.get("trainer", {}))
    trainer_type = trainer_cfg.pop("type", "MultiStageTrainer")
    trainer = TRAINERS.build(
        {"type": trainer_type, "cfg": cfg, **trainer_cfg},
        dry_run=args.command == "dry-run",
    )
    trainer.run()


def _prepare_import_paths(config_path: str) -> None:
    """Add common local paths so ``custom_imports`` works in editable repo usage."""

    candidates = []
    cwd = Path.cwd().resolve()
    candidates.append(cwd)

    cfg_path = Path(config_path).expanduser()
    if not cfg_path.is_absolute():
        cfg_path = (cwd / cfg_path).resolve()
    if cfg_path.exists():
        candidates.append(cfg_path.parent)
        for parent in cfg_path.parents:
            candidates.append(parent)
            if (parent / "pyproject.toml").exists() or (parent / "setup.py").exists():
                break

    for path in candidates:
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


if __name__ == "__main__":
    main()
