# Low-Light Image Enhancement - Dataset Curation & Preprocessing

Mini Project 1: Dataset Curation & Preprocessing for Low-Light Image Enhancement

Tools for collecting, preprocessing, and preparing low-light image datasets.

## Project Overview

Features:
- **Dataset Collection**: Download and organize low-light datasets (LOL, ExDark, Nighttime Driving)
- **Preprocessing**: Resizing, normalization, and data augmentation
- **Dataset Splitting**: Train/test split with proper organization
- **Visualization**: Generate examples and statistics

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Dataset Setup

Download low-light datasets:
- **LOL Dataset**: https://daooshee.github.io/BMVC2018website/
- **ExDark Dataset**: https://github.com/cs-chan/Exclusively-Dark-Image-Dataset
- **Nighttime Driving Dataset**: Search for "Nighttime Driving dataset"

Extract to `lol_dataset/`, `datasets/ExDark/`, or `datasets/NighttimeDriving/` respectively.

### Preprocess Dataset

For LOL dataset:
```bash
python process_lol_dataset.py --lol_dir lol_dataset --output_dir processed_dataset
```

For custom datasets:
```bash
python dataset_preprocessing.py --input_dir datasets/combined --output_dir processed_dataset
```

**Parameters:**
- `--input_dir`: Input directory
- `--output_dir`: Output directory
- `--target_size`: Resize dimensions (default: 512 512)
- `--split_ratio`: Train/test ratio (default: 0.8)
- `--no_normalize`: Skip normalization
- `--no_augment`: Skip augmentation

### Visualize Results

```bash
python visualize_dataset.py \
    --dataset_dir processed_dataset \
    --output_dir visualizations \
    --num_samples 8
```

## Dataset Structure

```
processed_dataset/
├── train/
│   ├── image1.jpg
│   ├── image1_aug1.jpg
│   ├── image1_aug2.jpg
│   └── ...
├── test/
│   ├── image100.jpg
│   ├── image101.jpg
│   └── ...
├── visualizations/
│   ├── preprocessing_examples_train.png
│   ├── preprocessing_examples_test.png
│   ├── dataset_statistics.png
│   └── sample_grid.png
└── dataset_stats.json
```

## Preprocessing

1. **Resizing**: 512×512 pixels (bilinear interpolation)
2. **Normalization**: Standard normalization (zero mean, unit variance), scaled to [0, 255]
3. **Augmentation**: Synthetic darkening with factors (0.2, 0.3, 0.4) and gamma (1.8, 2.0, 2.2)
4. **Split**: 80/20 train/test (random seed 42)

## Output Files

- `dataset_stats.json`: Dataset statistics
- `preprocessing_examples_train.png`, `preprocessing_examples_test.png`: Preprocessing examples
- `dataset_statistics.png`: Statistical analysis
- `sample_grid.png`: Sample image grid

## Troubleshooting

**"No images found"**: Verify input directory contains image files and path is correct.

**Out of Memory**: Reduce `--target_size` or use `--no_augment`.

**Slow Processing**: Disable augmentation or reduce image resolution.

## References

- LOL Dataset: https://daooshee.github.io/BMVC2018website/
- ExDark Dataset: https://github.com/cs-chan/Exclusively-Dark-Image-Dataset
- Retinex Theory: Original paper on Retinex theory
- Dual-Tree Complex Wavelet Transform: Springer 2025 reference

## License

Educational use only.

