#!/usr/bin/env python3
"""One-off generator: P0 full-eval 12x12 -> 96x96 copies (input_len/output_len only)."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "configs" / "cross_domain"

SOURCES = [
    "dpm_stfm_cross_domain_sharded_transfer_p0_full_eval.yaml",
    "dpm_v2_cross_domain_sharded_transfer_p0_full_eval.yaml",
    "dpm_v3_cross_domain_sharded_transfer_p0_full_eval.yaml",
    "dpm_v5_cross_domain_sharded_transfer_p0_full_eval.yaml",
    "dpm_v5_cross_domain_sharded_transfer_p0_full_eval_rob0.yaml",
    "opencity_cross_domain_sharded_transfer_p0_full_eval.yaml",
    "factost_cross_domain_sharded_transfer_p0_full_eval.yaml",
    "unist_cross_domain_sharded_transfer_p0_full_eval.yaml",
    "dpm_stfm_v4_cross_domain_e2e_transfer_p0_full_eval.yaml",
]


def bump_lengths(text: str) -> str:
    text = re.sub(
        r"^(\s*input_len:\s*)12\s*$", r"\g<1>96", text, flags=re.MULTILINE
    )
    text = re.sub(
        r"^(\s*output_len:\s*)12\s*$", r"\g<1>96", text, flags=re.MULTILINE
    )
    return text


def bump_experiment_and_workdir(text: str) -> str:
    def repl_exp(m: re.Match[str]) -> str:
        name = m.group(1)
        if name.endswith("_96"):
            return m.group(0)
        return f"experiment_name: {name}_96"

    text = re.sub(
        r"^experiment_name:\s*(\S+)\s*$", repl_exp, text, flags=re.MULTILINE
    )

    def repl_wd(m: re.Match[str]) -> str:
        indent_key = m.group(1)
        path = m.group(2)
        if path.rstrip("/").endswith("_96"):
            return m.group(0)
        return f"{indent_key}{path.rstrip('/')}_96"

    text = re.sub(r"^(\s*work_dir:\s*)(\S+)\s*$", repl_wd, text, flags=re.MULTILINE)
    return text


def header_note(text: str) -> str:
    lines = text.splitlines()
    if not lines:
        return text
    if "96" in lines[0] and ("96->96" in lines[0] or "P0 full-eval" in lines[0] and "96" in lines[0]):
        return text
    tag = "96->96 cross-domain sharded P0 full-eval (all stages); copy of 12x12 recipe."
    if lines[0].startswith("#"):
        lines[0] = f"{lines[0]}  {tag}"
    else:
        lines.insert(0, f"# {tag}")
    return "\n".join(lines) + "\n"


def main() -> None:
    for name in SOURCES:
        src = ROOT / name
        if not src.is_file():
            raise SystemExit(f"Missing source: {src}")
        stem = src.stem
        out = ROOT / f"{stem}_96.yaml"
        text = src.read_text(encoding="utf-8")
        text = bump_lengths(text)
        text = bump_experiment_and_workdir(text)
        text = header_note(text)
        out.write_text(text, encoding="utf-8")
        print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
