# Persian & English Document OCR

An end-to-end Optical Character Recognition (OCR) pipeline for **Persian and English** documents, supporting both **printed and handwritten** text at the **word and sentence** level. The system combines a classical computer-vision preprocessing pipeline (deskewing, binarization, line segmentation) with a **CRNN + CTC** deep learning model to convert full document images into structured, machine-readable text.

پشتیبانی از فارسی و انگلیسی، متن چاپی و دست‌نویس، در سطح کلمه و جمله.

---

## Overview

Digitizing real-world documents — forms, notes, scanned pages — is hard when the text is Persian, handwritten, or a mix of both with English. Most off-the-shelf OCR tools are tuned for clean, printed Latin text and struggle with Persian's cursive, right-to-left script, or with handwriting of either language.

This project provides a complete, transparent pipeline that:

- **Input**: a photo or scan of a document (single or multi-line, Persian and/or English, printed or handwritten)
- **Output**: structured JSON containing the recognized text per line, plus the reconstructed full-document text

**Target use cases**: digitizing forms and notes, building searchable archives of scanned documents, and as a reference implementation / learning resource for CRNN+CTC-based OCR systems.

## Features

- 🇮🇷 Persian OCR — 🇬🇧 English OCR — single shared model and character set
- Printed **and** handwritten text recognition
- Multi-line, full-document processing (not just isolated words)
- Synthetic dataset generation using real, freely-licensed fonts (no large external dataset required to get started)
- Image preprocessing pipeline: denoising, binarization, automatic deskew
- Classical line-segmentation (projection-profile based) with artifact merging
- CRNN (CNN + BiLSTM) encoder with CTC decoding for variable-length text
- Structured JSON output (line-level text, bounding boxes, reconstructed full text)
- Tkinter desktop GUI demo

## Architecture

```
        Image Document
              │
              ▼
        Preprocessing
              │
              ▼
     Deskew / Binarization
              │
              ▼
       Line Segmentation
              │
              ▼
        CRNN Encoder
              │
              ▼
        CTC Decoder
              │
              ▼
       Extracted Text
```

A single shared model handles both scripts via a combined ~128-character set (Persian letters, Latin letters, digits, punctuation), rather than requiring separate models or a language-detection step.

## Project Structure

```
persian-english-ocr/
├── docs/
│   └── ARCHITECTURE.md           # full technical design document
├── notebooks/
│   ├── 01_dataset_preparation.ipynb        # synthetic + real dataset generation
│   ├── 02_preprocessing_and_segmentation.ipynb
│   ├── 03_model_training_evaluation.ipynb  # model, training, CTC decode, CER
│   └── 04_inference_and_gui.ipynb          # end-to-end inference demo
├── src/                           # reusable modules (imported by the notebooks)
│   ├── data_pipeline.py           # character set, label encode/decode, batching
│   ├── preprocessing.py           # binarization, deskew
│   ├── segmentation.py            # line segmentation
│   ├── model.py                   # CRNN architecture (baseline/improved/regularized)
│   ├── train.py / train_steps.py / cache_data.py   # training entry points
│   ├── decode.py                  # CTC greedy decoding + CER metric
│   └── inference.py               # full document → text pipeline
├── gui/app.py                     # Tkinter desktop demo
├── fonts/                         # freely-licensed fonts used for synthetic data
├── data/                          # datasets (raw/processed)
├── models/                        # model checkpoints
├── samples/                       # sample document images
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/AziziArya/Persian-Document-Digitization-OCR.git
cd Persian-Document-Digitization-OCR
pip install -r requirements.txt
```

## Quick Start

1. Install the requirements (above).
2. Run the notebooks in order (`01` → `04`) — each is self-contained and documents its own steps and outputs.
3. To try the desktop demo:
   ```bash
   python gui/app.py
   ```
   Load a document image; the recognized text and a structured JSON result are produced.

## Training

Notebook `03_model_training_evaluation.ipynb` documents two training paths:

- **GPU workflow (recommended)** — using a free Google Colab GPU runtime. Mount the project on Google Drive, install requirements, and run the training cells with a larger batch size and epoch count. Substantially faster than CPU training.
- **CPU fallback workflow** — for machines without a GPU. Uses `src/cache_data.py` and `src/train_steps.py`, which train in small, resumable chunks (a fixed number of batches per run) so training can be spread across multiple short sessions instead of one long uninterrupted run. This is slower — expect CPU training on a full dataset to take considerably longer than on a GPU — but requires no special hardware.

## Inference

`src/inference.py` exposes `run_ocr_on_document(image_path, model)`, which runs the full pipeline (preprocessing → deskew → line segmentation → CRNN+CTC recognition → reconstruction) on a single document image and returns a dictionary with per-line text, bounding boxes, and the reconstructed full text. See `notebooks/04_inference_and_gui.ipynb` for a worked example, and `gui/app.py` for a GUI wrapper around the same function.

## Dataset

- **Printed text (Persian & English)**: generated synthetically using real, freely-licensed fonts (e.g. Vazirmatn, Scheherazade, DejaVu) rendered over frequency-ranked word lists — no large external dataset is required to get started.
- **Handwritten text**: synthesized using cursive/script-style fonts as a stand-in, optionally augmented with real handwriting data if available:
  - Persian handwritten digits (optional, user-supplied)
  - [IAM Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database) for English (optional)

See `notebooks/01_dataset_preparation.ipynb` for the full generation pipeline and how to plug in additional real datasets.

## Model Status

**The checkpoint included in `models/` is a pipeline-validation checkpoint, not a production-accuracy model.** It was trained briefly on a small sample to confirm that the full pipeline — data loading, CTC loss, training loop, checkpointing, greedy decoding, and inference — works correctly end-to-end.

- ✅ It validates that training and inference flow correctly with no structural or numerical errors.
- ❌ It has not been trained long enough, or on enough data, to produce reliable transcriptions.

For real-world accuracy, train on a full-scale dataset (see `notebooks/01`) for substantially more epochs, ideally on a GPU (see the **Training** section above).

## Limitations

- The bundled checkpoint's recognition accuracy is not representative of the architecture's ceiling — see **Model Status**.
- Handwriting realism currently relies partly on synthetic cursive fonts rather than large-scale real handwriting corpora.
- Line segmentation uses a classical projection-profile method, which works well on reasonably clean scans but is not as robust as a learned segmentation model on heavily skewed, noisy, or curved documents.
- No language model / spell-correction post-processing is applied to decoded text.
- Not yet benchmarked against standard OCR datasets or commercial OCR engines.

## Future Roadmap

- Larger-scale, higher-quality handwriting datasets (Persian and English)
- Transformer-based OCR architectures as an alternative to CRNN+CTC
- Integrated language modeling / spell-correction for post-processing
- Learned (rather than classical) line and paragraph segmentation
- Pretrained, production-ready model release

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Author

Created by **AziziArya**

- Website: [aryahub.ir](https://aryahub.ir/)
- GitHub: [github.com/AziziArya](https://github.com/AziziArya)
