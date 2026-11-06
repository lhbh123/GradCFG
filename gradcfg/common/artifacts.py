"""Resolve historical gradient naming conventions without rewriting results."""
from pathlib import Path
import json


def gradient_path(checkpoint_dir, epoch):
    directory = Path(checkpoint_dir)
    candidates = [directory / f"gradients_epoch_{epoch}.pt", directory / f"gradients_{epoch}.pt"]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("Missing captured gradient; expected one of: " + ", ".join(map(str, candidates)))


def prepare_run_directory(args, default_directory):
    directory = Path(getattr(args, "output_dir", None) or default_directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")
    return directory
