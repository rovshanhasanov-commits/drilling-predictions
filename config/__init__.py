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


def model_folder_name(cfg: dict, timestamp: str | None = None) -> str:
    """Config-derived subfolder name so each distinct training setup has its own bundle.

    Format varies by `training.lr_schedule`:

    Plateau:
      seq2seq_N{N}_K{K}_T{T}_lr{lr}_p{patience}_plateau_lrp{lr_patience}_{strategy}_{ts}

    Cosine warm restarts:
      seq2seq_N{N}_K{K}_T{T}_lr{lr}_p{patience}_cosineWR_T0{t0}_M{tmult}_{strategy}_{ts}

    Common fields:
      N        = encoder context window length
      K        = prediction horizon (number of future steps)
      T        = number of target heads
      lr       = initial learning rate (`:g` format)
      p        = early stopping patience (epochs)
      strategy = categorical embedding strategy
      ts       = training-start timestamp (`YYYYMMDD_HHMM`); appended only when provided

    `timestamp` is set by training to make every run a sibling folder. Eval / inference
    should pass the bundle path directly via `--model-dir` rather than rederive a name.
    """
    t = cfg["training"]
    base = (
        f"seq2seq_N{t['sequence_length']}"
        f"_K{t['n_future']}"
        f"_T{len(t['target_variables'])}"
        f"_lr{t['learning_rate']:g}"
        f"_p{t['early_stopping_patience']}"
    )

    schedule = t.get("lr_schedule", "plateau")
    if schedule == "cosine_restarts":
        sched_part = (
            f"_cosineWR"
            f"_T0{int(t.get('cosine_t_0', 50))}"
            f"_M{int(t.get('cosine_t_mult', 2))}"
        )
    else:
        sched_part = f"_plateau_lrp{int(t.get('lr_patience', 0))}"

    name = f"{base}{sched_part}_{t['embedding_strategy']}"
    if timestamp:
        name = f"{name}_{timestamp}"
    return name


def get_model_dir(cfg: dict, timestamp: str | None = None) -> Path:
    """Full resolved path to the model bundle for the current config.

    Pass `timestamp` at training time so each run gets its own sibling folder.
    Eval / inference should rely on the explicit folder path on disk rather than
    re-deriving a name (the timestamp won't round-trip).
    """
    return resolve(cfg, cfg["training"]["model_dir"]) / model_folder_name(cfg, timestamp)
