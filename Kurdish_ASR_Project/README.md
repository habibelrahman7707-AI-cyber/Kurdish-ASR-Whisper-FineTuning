# Kurdish ASR Pipeline (Whisper Fine-Tuning)

A high-performance pipeline for fine-tuning OpenAI's Whisper model on Kurdish audio data (Bahdini and Sorani dialects). 
This project automates data extraction (from `.rar`/`.zip` archives), metadata parsing (Excel files), text normalization, and batched log-Mel feature extraction to massively speed up ASR training on local or cloud GPUs.

## 🚀 Key Features
- **Batched Feature Extraction**: Pre-processes thousands of audio files 32x faster using vectorized `WhisperProcessor`.
- **PyArrow Dataset Optimization**: Prevents `large_string` mapping crashes commonly found in Pandas-to-HuggingFace conversions.
- **Automated Text Normalization**: Cleans Kurdish and Arabic scripts, standardizes characters (`ي` → `ی`), and removes diacritics.
- **Idempotent Data Loading**: Automatically copies and extracts `.rar` archives safely, matching audio paths to your Excel metadata instantly.
- **Seq2Seq Trainer**: Full integration with HuggingFace `Seq2SeqTrainer` including `bf16`/`fp16` mixed precision and `jiwer` WER metric evaluation.

## 📁 Project Structure
```
Kurdish_ASR_Project/
├── config.py              # Hyperparameters and path configurations
├── utils.py               # Text normalization and indexing functions
├── data_loader.py         # Archive extraction and Pandas dataframe normalization
├── dataset.py             # DataFrame to HuggingFace Dataset conversion
├── preprocessor.py        # Batched feature extraction & tokenization logic
├── train.py               # Seq2Seq Trainer and WER metrics configuration
├── main.py                # Pipeline entry point
└── requirements.txt       # Dependencies
```

## ⚙️ Installation
1. Install system prerequisites (e.g., `unrar` for RAR extraction):
   ```bash
   sudo apt-get install unrar
   ```
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Dataset Preparation
Place your data in the `data/` folder (or adjust paths in `config.py`):
```
data/
├── archives/          # Place your .rar or .zip files containing audio here
├── bahdini.xlsx       # Metadata for Bahdini dialect
└── sorani.xlsx        # Metadata for Sorani dialect
```

## 🎯 Usage
To start the end-to-end pipeline (extraction → preprocessing → training):
```bash
python main.py
```
Checkpoints and logs will be saved to `./workspace/output`.
