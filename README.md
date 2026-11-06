# GradCFG

Research implementation of **GradCFG: Gradient Inversion of Classifier-Free Guidance Diffusion Models**.

GradCFG jointly optimizes image latents, a continuous text-conditioning embedding, noise, and continuous timesteps to match an observed training gradient. Text-conditioning recovery does not reconstruct discrete prompt tokens. Inv-Rev is an optional second stage using model-disparity guidance.

This release contains the core method, low-memory implementation, gradient capture, Inv-Rev, and evaluation utilities. The default `attack` uses the full-batch implementation. It does not include baseline or ablation experiments.

## Setup

See [environment and data setup](docs/setup.md). Activate your Python environment before running commands; scripts do not activate a server-specific Conda installation.

```bash
python run.py download --model-id segmind/tiny-sd --output sd_model/tiny-sd --endpoint https://hf-mirror.com
```

Place your own images under `data/backpack/` (or `data/<label>/`). The demo selects up to five sorted images, resizes them to 512 × 512 and uses the folder label as their shared text condition. Use images you are authorized to process. Model weights, input images, private gradients, and reconstruction outputs are not included.

## End-to-end demo

```bash
# Default full-batch GradCFG: requires substantial GPU memory.
bash scripts/run_demo.sh

# Low-memory alternative, including optional Inv-Rev.
ATTACK=attack-lowmem RUN_ID=demo_lowmem RUN_INV_REV=1 bash scripts/run_demo.sh

# Short execution check, not a quality benchmark.
ATTACK=attack-lowmem RUN_ID=smoke TRAIN_EPOCHS=2 ATTACK_EPOCHS=2 bash scripts/run_demo.sh
```

`MODEL_PATH`, `DATA_DIR`, `LABEL`, `BATCH_SIZE`, `RUN_ID`, `TRAIN_EPOCHS`, and `ATTACK_EPOCHS` are configurable environment variables. Existing run directories are protected from accidental reuse. Training no longer restricts capture-time timestep sampling. This changes newly captured gradients compared with earlier restricted-range runs. Training records the model and first batch gradient before the epoch-0 parameter update, then continues fine-tuning. The attack uses this recorded model and gradient. `RUN_INV_REV=1` additionally uses the final fine-tuned model.

For first-stage reconstruction alone, `TRAIN_EPOCHS=1` is sufficient to capture the epoch-0 signal. The default 200 training epochs also prepare the optional fine-tuned model; changing this setting changes the model used by Inv-Rev.

## Individual stages

```bash
python run.py train --config configs/train.json
python run.py attack --config configs/gradcfg.json --output_dir outputs/reconstruction
# Substitute attack-lowmem for attack when GPU memory is limited.
python run.py --list
python run.py sample --help
```

Command-line parameters override JSON configuration. Both attack implementations default to 4001 iterations, noise reset every 100 iterations, and TV every 5 iterations with weight 0.5. Sparse TV is not mathematically equivalent to TV at every iteration; use `--tv_every 1 --tv_lambda 0.1` to select that setting explicitly. Timesteps are optimized within the configured normalized range, default 0.001–0.999. All training batches, including gradient-capture batches, sample integer timesteps uniformly from the full scheduler range [0, 1000). The continuous attack uses a minimum normalized time of 0.001 for numerical stability; timestep 0 is therefore represented approximately near this boundary.

Both implementations save latent and text tensors and decoded image strips every 100 iterations and at the final iteration (zero-based indices). `--save_every 0` saves only the final state. The default attack reports timing and peak CUDA memory every 10 iterations; the low-memory attack uses the same logging interval. Results and `run_config.json` go to `--output_dir`.

The low-memory algorithm accumulates batch parameter gradients and then backpropagates the aggregate matching objective sample by sample. It trades extra computation for memory. Prior five-image TinySD runs used approximately 18 GiB allocated memory with this implementation, while full-batch runs required substantially more than 24 GiB. These are reference observations, not hardware guarantees; first run the short check on your device. The default implementation enables cuDNN benchmarking, so fixed seeds alone do not guarantee bitwise reproducibility.

## Optional Inv-Rev and access assumptions

Inv-Rev requires the recovered latent and embedding, the model at the observed training step, and an additional fine-tuned model for disparity guidance. Access to this additional model must be stated when reporting results; it is not assumed to follow from observing a single gradient. Report first-stage reconstruction separately. `sample --mode cfg` supports a single-model path without a fine-tuned checkpoint.

## Evaluation

```bash
python run.py compare --reference path/to/reference.png --images path/to/reconstruction.png \
  --protocol standard --out-csv outputs/metrics.csv
```

Use corresponding individual images for per-image evaluation. This command does not perform batch permutation matching. Standard metrics are grayscale SSIM, RGB PSNR, and AlexNet LPIPS with inputs scaled to [-1, 1]. See [metric definitions](docs/metrics.md). LPIPS may download public AlexNet weights on first use. A legacy metric convention is available for compatibility, but must not be mixed with standard scores.

## Validation and scope

```bash
python -m unittest discover -s tests -v
```

Tests check aggregate-gradient and regularization derivatives on small models without downloading diffusion weights. Release packaging passed Python compilation and shell syntax checks. Four low-memory numerical tests passed during preparation; subsequent checks, including the added full-batch regression tests, were interrupted by this server's memory limit and require rerunning in the target environment. Full GPU experiments have not been rerun for this packaging release. The shared-prompt fine-tuning example uses a diffusion denoising objective without DreamBooth prior-preservation loss; it should not be described as a full prior-preserving DreamBooth implementation.

Use this code for authorized privacy evaluation. Reconstruction quality depends on observed parameters, batch size, model state and optimization settings; recovery is not guaranteed. No blanket software license is granted in this initial release; licensing and third-party provenance should be resolved before redistribution under an open-source license.
