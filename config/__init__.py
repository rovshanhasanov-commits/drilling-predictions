"""Configuration loader. Call load_config() from any stage to get a resolved dict."""

from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parent / "pipeline.yaml"
REPO_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: Path | str | None = None) -> dict:
    p = Path(path) if path else CONFIG_PATH
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_repo_root"] = str(REPO_ROOT)
    return cfg


def resolve(cfg: dict, rel_path: str) -> Path:
    return (Path(cfg["_repo_root"]) / rel_path).resolve()


def model_folder_name(cfg: dict) -> str:
    """Config-derived subfolder name so each distinct training setup has its own bundle.

    Format: seq2seq_N{seq_len}_K{n_future}_T{n_targets}_lr{lr}_p{patience}_{strategy}
      N        = encoder context window length
      K        = prediction horizon (number of future steps)
      T        = number of target heads (hierarchy levels + duration if enabled)
      lr       = learning rate (`:g` format — `0.0001` stays fixed, `1e-5` becomes scientific)
      p        = early stopping patience (epochs)
      strategy = categorical embedding strategy
    """
    t = cfg["training"]
    return (
        f"seq2seq_N{t['sequence_length']}"
        f"_K{t['n_future']}"
        f"_T{len(t['target_variables'])}"
        f"_lr{t['learning_rate']:g}"
        f"_p{t['early_stopping_patience']}"
        f"_{t['embedding_strategy']}"
    )


def get_model_dir(cfg: dict) -> Path:
    """Full resolved path to the model bundle for the current config.

    Both training (to save) and inference (to load) should call this — guarantees
    they agree on where artifacts live.
    """
    return resolve(cfg, cfg["training"]["model_dir"]) / model_folder_name(cfg)
