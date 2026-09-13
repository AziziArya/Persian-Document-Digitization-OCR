# v1.0.0 - Initial Public Release

## Overview

First public release of the Persian & English Document OCR system — an end-to-end
pipeline for recognizing printed and handwritten text in Persian and English documents.

## Features

- End-to-end OCR pipeline: preprocessing → deskew → line segmentation → recognition → structured output
- CRNN + CTC architecture with a shared Persian/English character set
- Support for both printed and handwritten text
- Synthetic dataset generation pipeline (real, freely-licensed fonts)
- Documented dataset preparation, training, and evaluation notebooks
- Full inference pipeline (`src/inference.py`) producing structured JSON output
- Tkinter GUI demo for interactive testing

## Model Note

The checkpoint included in this release demonstrates pipeline correctness and the
end-to-end inference flow (data loading → training → CTC decoding → structured output).
It is **not** a production-accuracy model. For real-world use, train on a full-scale
dataset for substantially more epochs — see the **Training** section of the README and
`notebooks/03_model_training_evaluation.ipynb` for GPU (recommended) and CPU workflows.

## Technical Stack

- Python
- TensorFlow / Keras
- OpenCV
- Jupyter Notebook
