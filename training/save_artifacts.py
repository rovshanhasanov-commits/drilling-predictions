"""Save the trained model bundle in the layout expected by inference/load.py.

`model_config.json` is the source of truth at eval/inference time. It captures
the full effective training config (after Colab override-cell mods), code
state (git SHA), environment versions, dataset fingerprints, active feature
lists, and run metadata. See improvements/Learning rate changes and Eval fixes.md
§2e for the spec.
"""

from __future__ import annotations

import json
import pickle
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def save_bundle(
    model_dir: Path,
    training_model,
    encoder_model,
    decoder_step_model,
    strategy_data_dir: Path,
    cfg: dict,
    *,
    run_metadata: dict | None = None,
    n_classes: dict | None = None,
    cat_input_cols: list[str] | None = None,
    numeric_cols: list[str] | None = None,
):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    training_model.save(model_dir / "training_model.keras")
    encoder_model.save(model_dir / "encoder_model.keras")
    decoder_step_model.save(model_dir / "decoder_step_model.keras")

    shutil.copy(Path(strategy_data_dir) / "encoders.pkl", model_dir / "encoders.pkl")
    shutil.copy(Path(strategy_data_dir) / "config.json",  model_dir / "data_config.json")

    enc_path = Path(strategy_data_dir) / "encoders.pkl"
    encoders_pkl = None
    if enc_path.exists():
        with open(enc_path, "rb") as f:
            encoders_pkl = pickle.load(f)

    model_config = _build_model_config(
        cfg=cfg,
        encoders_pkl=encoders_pkl,
        run_metadata=run_metadata,
        n_classes=n_classes,
        cat_input_cols=cat_input_cols,
        numeric_cols=numeric_cols,
    )

    with open(model_dir / "model_config.json", "w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=2, default=str)

    print(f"Saved model bundle to {model_dir}/")
    for fn in sorted(model_dir.iterdir()):
        size_mb = fn.stat().st_size / 1e6
        print(f"  {fn.name} ({size_mb:.1f} MB)")


def _build_model_config(
    cfg: dict,
    encoders_pkl: dict | None,
    run_metadata: dict | None,
    n_classes: dict | None,
    cat_input_cols: list[str] | None,
    numeric_cols: list[str] | None,
) -> dict:
    """Assemble the rich model_config.json payload."""
    # Strip the internal _repo_root key so the dump is portable.
    effective_cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}

    out: dict = {
        "schema_version": 2,
        "effective_cfg": effective_cfg,
        "code_state": _git_state(),
        "environment": _environment_info(),
        "dataset_fingerprints": _dataset_fingerprints(cfg),
    }

    # Active feature lists used at training time.
    out["features"] = {
        "cat_input_cols": list(cat_input_cols) if cat_input_cols else None,
        "numeric_cols":   list(numeric_cols)   if numeric_cols   else None,
    }

    # Encoder classes per target head + bin metadata.
    if encoders_pkl is not None:
        target_encoders = encoders_pkl.get("target_encoders", {}) or {}
        out["target_classes"] = {
            head: list(le.classes_) for head, le in target_encoders.items()
        }
        if encoders_pkl.get("bin_edges") is not None:
            out["bin_edges"] = list(encoders_pkl["bin_edges"])
        if encoders_pkl.get("bin_centers") is not None:
            out["bin_centers"] = dict(encoders_pkl["bin_centers"])
        out["eoo_token"] = encoders_pkl.get("eoo_token")

    if n_classes is not None:
        out["n_classes"] = {k: int(v) for k, v in n_classes.items()}

    if run_metadata is not None:
        out["run_metadata"] = run_metadata

    return out


def _git_state() -> dict:
    """Capture git SHA, branch, and dirty flag. Best-effort — never raises."""
    repo_root = Path(__file__).resolve().parent.parent
    out: dict = {"sha": None, "branch": None, "dirty": None}
    try:
        out["sha"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        pass
    try:
        out["branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        pass
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo_root, stderr=subprocess.DEVNULL,
        ).decode()
        out["dirty"] = bool(status.strip())
    except Exception:
        pass
    return out


def _environment_info() -> dict:
    """Python and key package versions — read at save time, never raises."""
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for pkg in ("tensorflow", "numpy", "pandas", "sklearn", "yaml"):
        try:
            mod = __import__(pkg)
            info[pkg] = getattr(mod, "__version__", None)
        except Exception:
            info[pkg] = None
    return info


def _dataset_fingerprints(cfg: dict) -> dict:
    """Per-split file metadata: path, size, mtime. Hashes intentionally skipped
    (slow on big parquets); add later if needed."""
    from datetime import datetime
    out: dict = {}
    repo_root = Path(cfg.get("_repo_root") or Path(__file__).resolve().parent.parent)
    rel_output = cfg.get("data", {}).get("output_dir")
    strategy = cfg.get("training", {}).get("embedding_strategy")
    if not rel_output or not strategy:
        return out
    strategy_dir = (repo_root / rel_output / strategy).resolve()
    for split in ("train", "val", "test"):
        path = strategy_dir / f"df_{split}.parquet"
        if not path.exists():
            out[split] = {"path": str(path), "exists": False}
            continue
        st = path.stat()
        out[split] = {
            "path":  str(path),
            "size":  int(st.st_size),
            "mtime": datetime.utcfromtimestamp(st.st_mtime).isoformat(timespec="seconds") + "Z",
        }
    return out
