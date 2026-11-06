# Environment and inputs

Python 3.11 is recommended. Install PyTorch and torchvision wheels compatible with your NVIDIA driver first. For example, for a CUDA 12.1 compatible driver:

```bash
conda create -n gradcfg python=3.11 -y
conda activate gradcfg
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

The CUDA wheel example is an installation option, not a claim of full experiment reproduction. The development environment uses Python 3.11 / PyTorch 2.9.1. Diffusers 0.33.1, transformers 4.51.3 and huggingface-hub 0.30.2 are pinned to the development versions. Record `pip freeze` alongside experimental results.

```bash
export HF_ENDPOINT=https://hf-mirror.com
python run.py download --model-id segmind/tiny-sd --revision cad0bd7495fa6c4bcca01b19a723dc91627fe84f --output sd_model/tiny-sd
```

Use `--endpoint https://huggingface.co` for the official endpoint. The default release targets TinySD and SD 1.x-style 512-pixel images / 64-pixel latents; arbitrary diffusion architectures are not supported automatically. Follow the model provider's license and access terms.

Place at least `BATCH_SIZE` images in `data/<label>/`. Training and attack must use the same batch size. Training uses the label string as a shared prompt. The attack initializes a continuous text embedding rather than receiving the original prompt as its target condition. Checkpoint lookup uses `ckpts/model_ckpt/<model_name>/<label_name>/`.

Neither training data nor recovered embeddings are bundled. Generated checkpoints, gradients, logs and outputs are ignored by Git. Keep enough disk space for model states and gradients; full UNet tensors can each be over 1 GB.
