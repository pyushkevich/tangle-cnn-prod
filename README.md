# Tangle-CNN Package

This package contains Python scripts to train and apply weakly supervised segmentation models based on the WildCat algorithm (Durand, CVPR 2017) in the context of detection and segmentation of pathological inclusions, such as neurofibrillary tangles, in digital pathology images in neurodegenerative disorders. For additional details, please read (Yushkevich et al., Brain 2021), (Ravikumar et al., Nature Communications, 2024) and (Denning et al., Acta Neuropathologica, 2024).

## Overview

Tangle-CNN is a deep learning toolkit for weakly supervised detection and segmentation of pathological features in histology whole-slide images. The package implements a WildCat-based approach that enables training models with image-level labels and applying them to generate pixel-level density maps of pathological structures.

Key features:
- Weakly supervised learning using WildCat architecture
- Support for multi-class classification of pathological features
- Whole-slide image processing capabilities
- Training with bounding box constraints and multi-resolution support
- Flexible data augmentation options

## Installation

### Requirements
- Python >= 3.8
- CUDA-capable GPU (recommended for training and inference)

### Install from source

```bash
git clone https://github.com/pyushkevich/tangle-cnn-prod.git
cd tangle-cnn-prod
pip install -e .
```

### Dependencies

The package requires the following Python libraries:
- PyTorch >= 1.6.0
- torchvision >= 0.7.0
- numpy
- scipy
- matplotlib
- openslide-python
- SimpleITK
- Pillow
- pandas
- parse

These dependencies will be automatically installed during package installation.

## Usage

The package provides a command-line interface with four main subcommands:

### 1. Training a Model

Train a WildCat model on labeled image patches:

```bash
python -m tangle_cnn train \
  --expdir /path/to/experiment \
  --epochs 50 \
  --batch 16 \
  --kmax 0.02 \
  --kmin 0.0 \
  --alpha 0.7 \
  --nmaps 4
```

**Key training options:**
- `--expdir`: Experiment directory containing `patches/train` and `patches/val` subdirectories
- `--epochs`: Number of training epochs (default: 50)
- `--batch`: Batch size (default: 16)
- `--kmax`, `--kmin`, `--alpha`: WildCat hyperparameters
- `--nmaps`: Number of maps per class
- `--bbox`: Use bounding box constraints (requires manifest file)
- `--scale`: Scale patches to target resolution for multi-resolution training
- `--color-jitter`: Apply color jitter augmentation
- `--lr`: Learning rate (default: 0.01)
- `--resume`: Resume from saved checkpoint

### 2. Validating a Model

Evaluate a trained model on test data:

```bash
python -m tangle_cnn validate \
  --expdir /path/to/experiment \
  --batch 16 \
  --target test
```

**Validation options:**
- `--expdir`: Experiment directory with trained model
- `--batch`: Batch size for evaluation
- `--target`: Dataset to evaluate ('test' or other)
- `--manifest`: Manifest file (if training used --bbox or --scale)

### 3. Applying to Whole-Slide Images

Apply trained model to generate density maps for whole-slide images:

```bash
python -m tangle_cnn apply \
  --slide /path/to/slide.svs \
  --modeldir /path/to/trained/model \
  --output /path/to/output_density.nii.gz \
  --window 4096 \
  --shrink 4
```

**Apply options:**
- `--slide`: Input whole-slide image path
- `--reader`: Image reader to use ('openslide', 'sitk', or 'pillow')
- `--modeldir`: Directory containing trained model
- `--output`: Output density map file (.nii.gz format)
- `--window`: Window size for slide scanning (default: 4096)
- `--shrink`: Downsampling factor for output (default: 4)
- `--region`: Process only a specific region [x, y, w, h]

### 4. Applying to Image Patches

Apply model to pre-extracted image patches:

```bash
python -m tangle_cnn patch_apply \
  --input /path/to/patches \
  --modeldir /path/to/trained/model \
  --outdir /path/to/output \
  --batch 16
```

**Patch apply options:**
- `--input`: Input directory or directories containing patches
- `--modeldir`: Directory containing trained model
- `--outdir`: Output directory for density maps
- `--outstat`: Output file for density statistics
- `--batch`: Batch size (default: 16)
- `--scale`: Resolution scaling factor for multi-resolution inference

### Getting System Information

Check PyTorch and CUDA configuration:

```bash
python -m tangle_cnn info
```

## Data Organization

### Training Data Structure

For training, organize your data as follows:

```
experiment_dir/
├── patches/
│   ├── train/
│   │   ├── class1/
│   │   │   ├── image1.png
│   │   │   └── image2.png
│   │   └── class2/
│   │       ├── image1.png
│   │       └── image2.png
│   └── val/
│       ├── class1/
│       └── class2/
└── models/
    ├── config.json
    └── wildcat_upsample.dat
```

### Manifest File Format

When using `--bbox` or `--scale` options, provide a CSV manifest with columns:
- For bounding boxes: `filename`, `x`, `y`, `width`, `height`
- For scaling: `filename`, `mpp_x`, `mpp_y` (microns per pixel)

## References

If you use this package, please cite the following publications:

1. Yushkevich et al., "Deep learning analysis of tau pathology in postmortem human hippocampus" *Brain*, 2021
2. Ravikumar et al., "Deep learning methods detect tau pathology in early Alzheimer's disease using multimodal digitized slides" *Nature Communications*, 2024
3. Denning et al., "Spatial quantification of tau pathology in progressive supranuclear palsy" *Acta Neuropathologica*, 2024
4. Durand et al., "WILDCAT: Weakly Supervised Learning of Deep ConvNets for Image Classification, Pointwise Localization and Segmentation" *CVPR*, 2017

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE.txt](LICENSE.txt) file for details.

## Authors

- Paul Yushkevich (pyushkevich@gmail.com)

## Contributing

For bug reports and feature requests, please use the GitHub issue tracker at:
https://github.com/pyushkevich/tangle-cnn-prod/issues
