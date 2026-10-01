# WGAN-GP Image Generation

A PyTorch learning implementation of a **Wasserstein GAN with gradient penalty** for 64 × 64 RGB images. A generator upsamples a 100-dimensional noise vector; a convolutional critic assigns scores to real and generated images.

[Architecture and cleanup notes](docs/architecture.md) · [Original submitted code](archive/original-code)

## What is included

| Location | Contents |
|---|---|
| [src/wgan_gp/generator.py](src/wgan_gp/generator.py) | Transposed-convolution generator with BatchNorm and Tanh output |
| [src/wgan_gp/critic.py](src/wgan_gp/critic.py) | Convolutional critic with InstanceNorm and LeakyReLU |
| [src/wgan_gp/gradient_penalty.py](src/wgan_gp/gradient_penalty.py) | Gradient norm penalty on interpolated images |
| [src/wgan_gp/train.py](src/wgan_gp/train.py) | Explicit CLI training, local images, metrics, samples and checkpoint output |
| [archive/original-code/](archive/original-code) | All five original Python files, unchanged |

## Run

Install compatible PyTorch and torchvision versions for your Python/CPU/GPU environment. The original dependency lockfile was not supplied; the requirements list is unpinned.

```sh
python -m venv .venv
# Activate the environment, then:
python -m pip install -r requirements.txt
PYTHONPATH=src python -m wgan_gp.train --help
PYTHONPATH=src python -m wgan_gp.train --data-dir data/images --epochs 50
```

For Windows PowerShell, set `$env:PYTHONPATH="src"` before running `python -m wgan_gp.train ...`. Organize local images into subfolders as expected by torchvision `ImageFolder`. The loader resizes to 64 × 64 and scales pixels to approximately [-1, 1].

With PyTorch installed, a short synthetic-data check is available:

```sh
PYTHONPATH=src python -m wgan_gp.train --smoke-test --batch-size 8 --max-steps 2 --epochs 1
```

Each run writes `metrics.csv`, sample grids and `checkpoint.pt` to its output directory. Use a fresh output directory for each experiment. The synthetic check verifies execution only; it does not measure generated-image quality.

## Evidence and limitations

The attachment contained code only: no model checkpoint, training log, generated samples or evaluation results. Syntax and the dependency-free CLI help were checked during cleanup. PyTorch/torchvision were unavailable, so no training or gradient test was executed here. **No FID, convergence or image-quality score is claimed.**

The cleaned entry point fixes the original critic/function import mismatches, handles actual batch sizes, explicitly enables interpolation gradients, separates critic and generator updates, and avoids starting a remote dataset download when imported. It uses local images instead of the historical Deep Lake Places205 loader. This is a changed implementation, not a reproduction of an undocumented historical run.

## Reference

Gulrajani et al., [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028), 2017. The paper introduces the gradient-penalty approach. The submitted source is preserved with its original comments; no new authorship or licensing claim is added.
