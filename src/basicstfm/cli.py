"""Command line interface for BasicSTFM."""

from __future__ import annotations

import argparse
import json
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


if __name__ == "__main__":
    main()
