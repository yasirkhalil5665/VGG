## Rainbow Colors CNN Classifier

A beginner-friendly PyTorch project for practicing CNNs on a small, simple image dataset. Built with a TinyVGG architecture to classify images into the 7 rainbow colors: **red, orange, yellow, green, blue, indigo, violet**.

## Project Overview

This project exists as a **learning sandbox for practicing CNNs with a small dataset** — a low-stakes way to build and debug a full PyTorch image-classification pipeline (transforms, `DataLoader`, model, training loop, evaluation) without the confounding difficulty of a "real" hard problem like distinguishing food or animal photos. It's intentionally simple: color is one of the easiest visual signals a CNN can learn, so once the pipeline works correctly the model should train fast and hit very high accuracy — making it easy to tell whether a bug is in your *code* or your *model*.

The dataset is small and synthetic on purpose (700 training images, 175 test images), so a full training run only takes a couple of minutes even on CPU — ideal for quickly experimenting with architecture changes and seeing the effect right away.

**Final result: 98–100% test accuracy**, achieved after two key fixes made while practicing/debugging:
- Adding `ReLU` activations between convolutional layers (non-linearity was initially missing)
- Using `RandomHorizontalFlip` instead of `TrivialAugmentWide` for data augmentation — flipping preserves color identity, while `TrivialAugmentWide`'s color/contrast jitter was actively distorting the very signal the model needs to classify on

If you're new to CNNs, this repo is a good starting point to fork and experiment with: try removing the `ReLU` layers yourself and watch what happens, swap in different augmentations, change `hidden_units`, or add more conv blocks — the small dataset means you'll see results within minutes.

## Dataset

```
rainbow_colors/
├── train/
│   ├── red/       (100 images)
│   ├── orange/    (100 images)
│   ├── yellow/    (100 images)
│   ├── green/     (100 images)
│   ├── blue/      (100 images)
│   ├── indigo/    (100 images)
│   └── violet/    (100 images)
└── test/
    └── (same 7 classes, 25 images each)
```
- 128×128 JPG images, `ImageFolder`-compatible structure
- 700 training images, 175 test images total
- Synthetically generated: base color + Gaussian noise + random shapes + brightness jitter

## Model Architecture

TinyVGG — a small CNN with two convolutional blocks followed by a linear classifier:

```
Input (3, 64, 64)
 → Conv2d → ReLU → Conv2d → ReLU → MaxPool2d
 → Conv2d → ReLU → Conv2d → ReLU → MaxPool2d
 → Flatten → Linear(→ 7 classes)
```

## Repository Structure

```
.
├── CNN_Rainbow.ipynb     # Main notebook: data loading, model, training, evaluation
├── rainbow_colors/       # Dataset (train/test folders)
├── README.md
└── requirements.txt
```

## Setup

```bash
git clone https://github.com/yasirkhalil5665/CNN.git
cd CNN
pip install -r requirements.txt
```

## Usage

Open and run `CNN_Rainbow.ipynb` in Jupyter:

```bash
jupyter notebook CNN_Rainbow.ipynb
```

## Key Learnings

- **Non-linearity matters**: removing `ReLU` between conv layers collapses a multi-layer conv stack into an equivalent single linear operation, capping model capacity regardless of depth.
- **Augmentation should match the task**: augmentations that alter color/contrast can hurt a color-classification task even though they typically help general image classification — always sanity-check that an augmentation doesn't distort the actual signal being classified.
- **Small datasets need honest evaluation**: with only 175 test images, single-run accuracy has real variance; results were confirmed by re-running training multiple times with different random weight initializations.

## Why This Is a Good Beginner Project

- **Fast feedback loop** — full training runs in minutes, even on CPU, so you can test an idea and see the result immediately instead of waiting hours.
- **Easy to isolate bugs** — since the task itself is simple, if accuracy is low it's almost always a code/architecture issue (missing activation, bad augmentation, wrong batch size), not a "the problem is just hard" issue. That makes it a great place to build debugging instincts before tackling harder datasets.
- **Small enough to read end-to-end** — the whole pipeline fits in one notebook you can actually read top to bottom and understand every line of.
- **Room to experiment safely** — try breaking things on purpose (remove ReLU, change augmentation, shrink the dataset further) and see the effect, without burning hours of compute per experiment.

## Requirements

See `requirements.txt`. Core dependencies: PyTorch, torchvision, torchinfo, NumPy, Matplotlib.
