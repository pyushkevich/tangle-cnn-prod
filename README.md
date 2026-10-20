# Tangle-CNN Package

This package contains Python scripts to train and apply weakly supervised segmentation models based on the WildCat algorithm ([Durand et al., CVPR 2017](https://doi.org/10.1109/CVPR.2017.631)) in the context of detection and segmentation of pathological inclusions, such as neurofibrillary tangles, in digital pathology images in neurodegenerative disorders. For additional details, please read [Yushkevich et al., Brain 2021](https://doi.org/10.1093/brain/awab262), [Ravikumar et al., Nature Communications 2024](https://doi.org/10.1038/s41467-024-49205-0) and [Denning et al., Acta Neuropathologica 2024](https://doi.org/10.1007/s00401-024-02789-9).

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

### Install from PyPI

```bash
pip install tangle-cnn
```

### Install from source

To use the latest development version, or to modify the code, install from a clone of the repository in editable mode:

```bash
git clone --recursive https://github.com/pyushkevich/tangle-cnn-prod.git
cd tangle-cnn-prod
pip install -e .
```

The `wildcat_pytorch` code is included as a git submodule, so `--recursive` is required. If you already cloned without it, run `git submodule update --init --recursive` before installing.

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

The package provides a command-line interface with five subcommands:

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
- `--color-jitter B C S H`: Apply color jitter augmentation with the given brightness, contrast, saturation and hue (see `torchvision.transforms.ColorJitter`)
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
- `--target`: Subdirectory of `patches/` to evaluate (default: `test`)
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
- `--manifest`: CSV file with `slide` and `output` columns, to process multiple slides in one run

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
- `--outstat`: Output file for density histograms
- `--shrink`: Downsampling factor for output (default: 4)
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
│   ├── val/
│   │   ├── class1/
│   │   └── class2/
│   └── test/          # used by 'validate' (default --target)
│       ├── class1/
│       └── class2/
└── models/            # created by 'train'
    ├── config.json
    └── wildcat_upsample.dat
```

### Manifest File Format

When using `--bbox` or `--scale` options, provide a CSV manifest via `--manifest`. Patch filenames must be integer IDs (e.g. `1234.png`), and the manifest must contain an `id` column matching them, plus:
- For bounding boxes (`--bbox`): `w`, `h` — width and height of a box centered in the patch, in pixels
- For scaling (`--scale`): `mpp_x`, `mpp_y` — pixel size of the patch, in the same units as the `--scale` target resolution

## References

If you use this package, please cite the following publications:

1. Yushkevich PA, et al. "Three-dimensional mapping of neurofibrillary tangle burden in the human medial temporal lobe." *Brain*. 2021;144(9):2784–2797. [doi:10.1093/brain/awab262](https://doi.org/10.1093/brain/awab262)
2. Ravikumar S, et al. "Postmortem imaging reveals patterns of medial temporal lobe vulnerability to tau pathology in Alzheimer's disease." *Nature Communications*. 2024;15(1):4803. [doi:10.1038/s41467-024-49205-0](https://doi.org/10.1038/s41467-024-49205-0)
3. Denning AE, et al. "Association of quantitative histopathology measurements with antemortem medial temporal lobe cortical thickness in the Alzheimer's disease continuum." *Acta Neuropathologica*. 2024;148(1):37. [doi:10.1007/s00401-024-02789-9](https://doi.org/10.1007/s00401-024-02789-9)
4. Durand T, Mordan T, Thome N, Cord M. "WILDCAT: Weakly Supervised Learning of Deep ConvNets for Image Classification, Pointwise Localization and Segmentation." *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*. 2017:5957–5966. [doi:10.1109/CVPR.2017.631](https://doi.org/10.1109/CVPR.2017.631)

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

The bundled submodules are distributed under their own licenses: `tangle_cnn/wildcat_pytorch` (MIT) and `tangle_cnn/deepcluster` (CC BY-NC 4.0, non-commercial use only).

## Authors

- Paul Yushkevich (pyushkevich@gmail.com)

## Contributing

For bug reports and feature requests, please use the GitHub issue tracker at:
https://github.com/pyushkevich/tangle-cnn-prod/issues
