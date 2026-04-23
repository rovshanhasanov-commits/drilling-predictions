"""Save the trained model bundle in the layout expected by inference/load.py."""

import json
import shutil
from pathlib import Path


def save_bundle(
    model_dir: Path,
    training_model,
    encoder_model,
    decoder_step_model,
    strategy_data_dir: Path,
    model_config: dict,
):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    training_model.save(model_dir / "training_model.keras")
    encoder_model.save(model_dir / "encoder_model.keras")
    decoder_step_model.save(model_dir / "decoder_step_model.keras")

    shutil.copy(Path(strategy_data_dir) / "encoders.pkl", model_dir / "encoders.pkl")
    shutil.copy(Path(strategy_data_dir) / "config.json",  model_dir / "data_config.json")

    with open(model_dir / "model_config.json", "w") as f:
        json.dump(model_config, f, indent=2, default=str)

    print(f"Saved model bundle to {model_dir}/")
    for fn in sorted(model_dir.iterdir()):
        size_mb = fn.stat().st_size / 1e6
        print(f"  {fn.name} ({size_mb:.1f} MB)")
