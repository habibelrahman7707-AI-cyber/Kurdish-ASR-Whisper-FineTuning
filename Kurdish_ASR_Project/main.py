import os
import torch
import pandas as pd
from datasets import DatasetDict
from transformers import WhisperProcessor, WhisperForConditionalGeneration

import config
from utils import build_filename_index
from data_loader import extract_archives, normalize_schema, clean_dataframe, AUDIO_EXTENSIONS
from dataset import df_to_dataset
from preprocessor import WhisperBatchPreprocessor
from train import DataCollatorSpeechSeq2SeqWithPadding, get_trainer

def main():
    config.create_dirs()
    print("Extracting archives...")
    extract_counts = extract_archives(
        archives_dir=config.ARCHIVES_DIR,
        local_rars_dir=config.LOCAL_RARS_DIR,
        extract_dir=config.EXTRACT_DIR,
        delete_local_after=config.DELETE_LOCAL_RAR_AFTER_EXTRACT
    )
    print("Extraction counts:", extract_counts)

    AUDIO_INDEX = build_filename_index(config.EXTRACT_DIR, AUDIO_EXTENSIONS)

    df_bah = pd.read_excel(config.BAHDINI_XLSX)
    df_sor = pd.read_excel(config.SORANI_XLSX)
    df_bah.columns = [str(c).strip() for c in df_bah.columns]
    df_sor.columns = [str(c).strip() for c in df_sor.columns]

    df_bah_norm = normalize_schema(df_bah, dialect='bahdini')
    df_sor_norm = normalize_schema(df_sor, dialect='sorani')

    df_bah_clean = clean_dataframe(df_bah_norm, AUDIO_INDEX, dialect='bahdini')
    df_sor_clean = clean_dataframe(df_sor_norm, AUDIO_INDEX, dialect='sorani')

    df_all = pd.concat([df_bah_clean, df_sor_clean], ignore_index=True)
    df_all = df_all.sample(frac=1.0, random_state=config.SEED).reset_index(drop=True)
    df_all['transcript_len'] = df_all['transcript'].str.len()

    print(f"Total rows after merge: {len(df_all)}")
    if len(df_all) == 0:
        raise RuntimeError("No data found! Check if audio files were extracted and metadata filenames match.")

    # 90/10 Split
    def stratified_split(df, val_frac=0.1, seed=config.SEED):
        train_parts, val_parts = [], []
        for _, sub in df.groupby('dialect'):
            sub = sub.sample(frac=1.0, random_state=seed)
            n_val = max(1, int(len(sub) * val_frac))
            val_parts.append(sub.iloc[:n_val])
            train_parts.append(sub.iloc[n_val:])
        train_df = pd.concat(train_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
        val_df = pd.concat(val_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
        return train_df, val_df

    train_df, val_df = stratified_split(df_all, val_frac=0.1, seed=config.SEED)
    
    print("Building datasets...")
    train_ds = df_to_dataset(train_df, sampling_rate=16000).rename_column('audio_path', 'audio')
    val_ds = df_to_dataset(val_df, sampling_rate=16000).rename_column('audio_path', 'audio')
    raw_datasets = DatasetDict({'train': train_ds, 'validation': val_ds})

    print(f"Loading Whisper processor: {config.MODEL_ID}")
    processor = WhisperProcessor.from_pretrained(config.MODEL_ID)
    
    print(f"Loading Whisper model: {config.MODEL_ID}")
    model = WhisperForConditionalGeneration.from_pretrained(
        config.MODEL_ID,
        torch_dtype=torch.bfloat16 if config.BF16 else (torch.float16 if config.FP16 else torch.float32),
        low_cpu_mem_usage=True,
    )
    
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    
    preprocessor = WhisperBatchPreprocessor(processor)

    print("Preprocessing splits (Extracting Log-Mel features)...")
    train_ds_proc = raw_datasets['train'].map(
        preprocessor.prepare_dataset_batched, batched=True, batch_size=config.TRAIN_BATCH_SIZE, num_proc=1,
        remove_columns=[c for c in raw_datasets['train'].column_names if c not in ['input_features', 'labels']]
    )
    val_ds_proc = raw_datasets['validation'].map(
        preprocessor.prepare_dataset_batched, batched=True, batch_size=config.TRAIN_BATCH_SIZE, num_proc=1,
        remove_columns=[c for c in raw_datasets['validation'].column_names if c not in ['input_features', 'labels']]
    )

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    trainer = get_trainer(model, train_ds_proc, val_ds_proc, data_collator, processor, config)

    print("Starting training...")
    trainer.train()

    print(f"Saving final model to {config.OUTPUT_DIR}...")
    trainer.save_model(str(config.OUTPUT_DIR))
    processor.save_pretrained(str(config.OUTPUT_DIR))
    print("Pipeline Complete!")

if __name__ == "__main__":
    main()
