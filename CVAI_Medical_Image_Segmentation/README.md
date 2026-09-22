# CVAI — Medical Diagnostic Image Segmentation

## Project Aim

Compare a compact U-Net baseline with an Attention U-Net for polyp segmentation and test whether MC-dropout uncertainty tracks segmentation error.

## Dataset
Kvasir-SEG contains 1,000 colonoscopy images with corresponding segmentation masks. See the [official dataset page](https://datasets.simula.no/kvasir-seg/).

## Experimental Configuration
- Seeds: 7, 42, 123
- Split: 800 train / 100 validation / 100 test
- Input: 96×96 RGB
- Batch size: 16
- Epochs: 10
- Optimiser: Adam, learning rate 0.001
- Loss: binary cross-entropy + soft Dice
- Threshold: 0.5
- MC-dropout passes: 12

## Repository Contents

- [Submitted assignment report](report/Rio_Roy_CVAI_Assignment_Final.pdf)
- [Experiment notebook](code/Rio_Roy_CVAI_Kvasir_Experiment.ipynb)
- [Model-training script](code/train_models.py)
- [Publication analysis script](code/publication_analysis.py)
- [Evaluation results](outputs/results.json)
- [Training history](outputs/history.json)
- [Dataset split](outputs/split.json)
- [Seed-42 confusion counts](outputs/confusion_counts_seed42.json)

## Reproducing the Analysis

Install the dependencies imported by the scripts, download Kvasir-SEG from the official source, and configure the dataset path in the training workflow. Run `code/train_models.py` first, followed by `code/publication_analysis.py`. The notebook provides an additional route for inspecting the saved outputs and figures.

> The JSON files in `outputs/` preserve the reported experiment results and validation evidence.
