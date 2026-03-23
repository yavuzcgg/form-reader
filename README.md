# FormReader

Automatic data extraction from handwritten production tracking forms using OCR and NLP. Developed under the TÜBİTAK 3501 Career Program.

## Overview

Manufacturing facilities still rely on handwritten forms to record production data. FormReader reads these forms and extracts:

- **Downtime durations** (minutes)
- **Downtime reasons** (handwritten Turkish text)
- **Operator names**
- **Date/time information**
- **Production quantities**

## Architecture

The system uses a hybrid **TrOCR** architecture — a Vision Encoder paired with a Turkish-optimized text decoder:

```
┌──────────────────────────────────────────────────┐
│  ENCODER                    DECODER              │
│  TrOCR (ViT)                BERTurk              │
│  microsoft/trocr-base       dbmdz/bert-base      │
│  -handwritten               -turkish-cased       │
│                                                  │
│  Image → Visual Features → Turkish Tokens → Text │
└──────────────────────────────────────────────────┘
```

**Why this combination?** TrOCR's encoder is pre-trained on handwriting recognition (IAM dataset), while BERTurk's decoder natively handles Turkish characters (ş, ğ, ü, ı, ö, ç) as single tokens rather than byte-level BPE fragments.

## Accuracy Targets

| Field | Method | Target |
|-------|--------|--------|
| Numeric data (duration, quantity) | TrOCR + Tesseract | **99%** |
| Handwritten Turkish text | TrOCR fine-tuned | **85%** |
| Downtime reason classification | NLP (spaCy/BERT) | **95%** |

## Project Structure

```
FormReader/
├── src/
│   ├── ocr_engine.py          # TrOCR inference engine
│   ├── trainer.py             # Model fine-tuning script
│   ├── data_generator.py      # Synthetic data generator
│   └── FormReader_Main.ipynb  # Main experiment notebook
├── data/                      # Training data (not tracked)
│   ├── fonts/                 # Handwriting fonts (.ttf)
│   ├── synthetic/             # Generated synthetic images
│   └── real_forms/            # Real form images for testing
├── models/                    # Trained models (not tracked)
├── experiments/               # Experiment logs (not tracked)
└── docs/                      # Documentation (not tracked)
```

## Setup

### Requirements

- Python 3.9+
- CUDA-compatible GPU (recommended)
- PyTorch with CUDA support

```bash
# Install dependencies
pip install -r requirements.txt

# Verify GPU
python -c "import torch; print(torch.cuda.is_available())"
```

## Usage

### 1. Generate Synthetic Training Data

```python
from src.data_generator import generate_dataset

generate_dataset(
    font_dir="data/fonts",
    output_dir="data/synthetic",
    count=10000
)
```

The generator creates images across 5 categories with augmentation (rotation, noise, blur, contrast/brightness variation, ink color variation):

| Category | Share | Examples |
|----------|-------|---------|
| Numbers | 30% | `140`, `1300`, `%85`, `0,5` |
| Turkish text | 30% | `Merkezi yağlama arızası`, `Ayşe Yılmaz` |
| Dates | 20% | `06.03.2023`, `08:30`, `Salı` |
| Mixed | 15% | `Hat-3`, `10 LT`, `A1 Arıza` |
| Technical codes | 5% | `S25C-925103023031` |

### 2. Fine-tune the Model

```python
from src.trainer import train_trocr

train_trocr(
    dataset_dir="data/synthetic",
    csv_path="data/synthetic/metadata.csv",
    output_model_dir="models/trocr-turkish-handwritten/deney005",
    epochs=10,
    batch_size=8,
    learning_rate=2e-5
)
```

Key training features:
- **Smart Embedding Init**: Turkish character weights initialized from similar Latin characters (`ş` ← `s`)
- **Unicode NFC Normalization**: Ensures consistent Turkish character encoding
- **Encoder Freeze**: ViT encoder is frozen; only the decoder learns Turkish

### 3. Run Inference

```python
from src.ocr_engine import OCREngine

engine = OCREngine()

# Single image
text = engine.predict("image.jpg")

# With confidence score
text, confidence = engine.predict("image.jpg", return_confidence=True)

# Batch
results = engine.predict_batch(["img1.jpg", "img2.jpg"])
```

## Training Notes

| Parameter | Default | Notes |
|-----------|---------|-------|
| `epochs` | 10 | ~40 min/epoch on RTX GPUs |
| `batch_size` | 8 | Lower to 4 if VRAM limited |
| `learning_rate` | 2e-5 | Range: 2e-5 to 5e-5 |

**Windows users**: `dataloader_num_workers=0` and `predict_with_generate=False` are set by default to prevent multiprocessing and VRAM overflow issues.

## References

- [TrOCR: Transformer-based Optical Character Recognition](https://arxiv.org/abs/2109.10282)
- [TrOCR for Spanish](https://arxiv.org/abs/2407.06950) — multilingual extension approach
- [BERTurk](https://huggingface.co/dbmdz/bert-base-turkish-cased)
- [VIPI Paper](https://arxiv.org/abs/2112.14569) — smart embedding initialization

## License

This project is developed under TÜBİTAK 3501 Career Development Program.
