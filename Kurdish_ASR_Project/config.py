import os
import torch
from pathlib import Path

# === Seed & Device ===
SEED = 42
BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
FP16 = torch.cuda.is_available() and not BF16

# === Paths ===
# We use standard local paths instead of Google Drive for generic deployment.
# Change these paths if you are running on Google Colab.
WORK_ROOT      = Path('./workspace')
ARCHIVES_DIR   = Path('./data/archives')               # Place your .rar/.zip files here
LOCAL_RARS_DIR = WORK_ROOT / 'rars'                    # Temp directory for processing
EXTRACT_DIR    = WORK_ROOT / 'extracted'               # Extracted audio lives here
FEATURES_DIR   = WORK_ROOT / 'features'                # Cached log-Mel features
OUTPUT_DIR     = WORK_ROOT / 'output'                  # Model checkpoints

# Metadata Excel files
BAHDINI_XLSX   = Path('./data/bahdini.xlsx')
SORANI_XLSX    = Path('./data/sorani.xlsx')

# === Model Settings ===
MODEL_ID       = 'openai/whisper-small'
LANGUAGE_KMR   = 'persian'  # Whisper lacks Kurdish; Persian is used as a phonetic script placeholder
TASK           = 'transcribe'

# === Training Hyperparameters ===
TRAIN_BATCH_SIZE       = 16
EVAL_BATCH_SIZE        = 8
GRAD_ACCUM_STEPS       = 2
LEARNING_RATE          = 1e-5
WARMUP_STEPS           = 200
NUM_TRAIN_EPOCHS       = 3
WEIGHT_DECAY           = 0.01
LOGGING_STEPS          = 25
SAVE_STEPS             = 500
EVAL_STEPS             = 500
SAVE_TOTAL_LIMIT       = 3
GRADIENT_CHECKPOINTING = False

# === Space Management ===
DELETE_LOCAL_RAR_AFTER_EXTRACT = True
DELETE_AUDIO_AFTER_CACHE       = True

def create_dirs():
    for p in [WORK_ROOT, LOCAL_RARS_DIR, EXTRACT_DIR, FEATURES_DIR, OUTPUT_DIR]:
        p.mkdir(parents=True, exist_ok=True)
