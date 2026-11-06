# Image metrics

Use `python run.py compare` with corresponding reference and reconstructed images.

| Metric | Standard protocol | Legacy compatibility protocol |
| --- | --- | --- |
| SSIM | RGB to grayscale, data range 255 | BGR grayscale conversion applied to RGB input |
| PSNR | RGB, data range 255 | Same |
| LPIPS | AlexNet, RGB normalized to [-1, 1] | RGB normalized to [0, 1] |

Inputs are uint8 RGB. Different dimensions are resized to the minimum height and width using OpenCV. The command does not match batch permutations or split image strips. For per-image results, align sample identities first and evaluate individual images. Do not treat a metric on an entire strip as a mean of per-image metrics.

The standard protocol is the recommended convention. Legacy is provided only to reproduce earlier metric conventions, including their normalization differences. Compare results only when reference processing, correspondence, resizing and protocol are the same. Missing dependencies cause errors; `--skip-lpips` explicitly leaves LPIPS empty. LPIPS uses public AlexNet weights and may download them on first use.
