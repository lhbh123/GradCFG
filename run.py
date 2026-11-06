"""Run GradCFG workflows from the repository root, with lazy imports."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys


COMMANDS = {'train': ('gradcfg.training.finetune', 'Fine-tune and record gradients'), 'attack': ('gradcfg.attacks.gradcfg', 'GradCFG full-batch reconstruction'), 'attack-lowmem': ('gradcfg.attacks.lowmem', 'GradCFG two-pass reconstruction'), 'sample': ('gradcfg.sampling.sample', 'Optional Inv-Rev'), 'compare': ('gradcfg.evaluation.compare', 'SSIM / PSNR / LPIPS'), 'download': ('gradcfg.tools.download_model', 'Download diffusion weights'), 'images': ('gradcfg.tools.images', 'Image layout tools'), 'decode': ('gradcfg.tools.decode_latents', 'Decode latent tensors')}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Commands:\n" + "\n".join(f"  {name:23} {desc}" for name, (_, desc) in COMMANDS.items())
        + "\n\nUse: python run.py COMMAND --help. ",
    )
    parser.add_argument("command", nargs="?")
    parser.add_argument("--list", action="store_true", help="List workflows without importing ML dependencies")
    if not argv or argv[0] in ("-h", "--help", "--list"):
        parser.print_help()
        return
    raw = argv.pop(0)
    name = raw
    if name not in COMMANDS:
        parser.error(f"Unknown command: {raw}")
    # Every historical relative data/checkpoint path is relative to this directory.
    os.chdir(Path(__file__).resolve().parent)
    if "--config" in argv:
        index = argv.index("--config")
        if index + 1 >= len(argv):
            parser.error("--config requires a JSON file")
        path = Path(argv[index + 1])
        del argv[index:index + 2]
        with path.open(encoding="utf-8") as handle:
            config = json.load(handle)
        if not isinstance(config, dict) or not all(isinstance(key, str) and key.startswith("--") for key in config):
            parser.error("Config must map complete --option names to values")
        config_args = []
        for key, value in config.items():
            if value is None or value is False:
                continue
            config_args.append(key)
            if value is not True:
                config_args.extend(map(str, value if isinstance(value, list) else [value]))
        argv = config_args + argv
    sys.argv = [f"run.py {name}", *argv]
    runpy.run_module(COMMANDS[name][0], run_name="__main__")


if __name__ == "__main__":
    main()
