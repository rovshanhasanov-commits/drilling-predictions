"""Regenerate llm/prompts/constraints.md from Master_Data_With_ID.csv.

Run this once after preprocessing (or whenever master data changes).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config import load_config, resolve  # noqa: E402


def main():
    cfg = load_config()
    master_csv = resolve(cfg, cfg["data"]["master_csv"])
    out_path = REPO_ROOT / cfg["llm"]["prompts_dir"] / cfg["llm"]["constraints_file"]

    df = pd.read_csv(master_csv, usecols=["Phase", "Phase_Step", "Major_Ops_Code", "Operation"])
    df = df.dropna(subset=["Phase"])

    lines = ["# Domain Constraints", "",
             "Valid combinations observed in historical data.", ""]

    lines.append("## Phase -> Valid Phase_Steps\n")
    for phase in sorted(df["Phase"].dropna().unique()):
        steps = sorted(df.loc[df["Phase"] == phase, "Phase_Step"].dropna().unique())
        lines.append(f"- **{phase}**: {', '.join(steps)}")
    lines.append("")

    lines.append("## Phase_Step -> Valid Major_Ops_Codes\n")
    for step in sorted(df["Phase_Step"].dropna().unique()):
        codes = sorted(df.loc[df["Phase_Step"] == step, "Major_Ops_Code"].dropna().unique())
        lines.append(f"- **{step}**: {', '.join(codes)}")
    lines.append("")

    lines.append("## Major_Ops_Code -> Valid Operations\n")
    for code in sorted(df["Major_Ops_Code"].dropna().unique()):
        ops = sorted(df.loc[df["Major_Ops_Code"] == code, "Operation"].dropna().unique())
        lines.append(f"- **{code}**: {', '.join(ops)}")
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Constraints written to {out_path}")


if __name__ == "__main__":
    main()
