import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

def compute_metrics(pred, tokenizer):
    metric = evaluate.load("jiwer")
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    
    pred_str = [x.strip() for x in pred_str]
    label_str = [x.strip() for x in label_str]

    valid_preds, valid_labels = [], []
    for p, l in zip(pred_str, label_str):
        if l:
            valid_preds.append(p)
            valid_labels.append(l)

    wer = 100 * metric.compute(predictions=valid_preds, references=valid_labels) if valid_labels else 100.0
    return {"wer": wer}

def get_trainer(model, train_ds_proc, val_ds_proc, data_collator, processor, config):
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(config.OUTPUT_DIR),
        per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=config.GRAD_ACCUM_STEPS,
        learning_rate=config.LEARNING_RATE,
        warmup_steps=config.WARMUP_STEPS,
        max_steps=config.NUM_TRAIN_EPOCHS * (len(train_ds_proc) // config.TRAIN_BATCH_SIZE),
        gradient_checkpointing=config.GRADIENT_CHECKPOINTING,
        fp16=config.FP16,
        bf16=config.BF16,
        evaluation_strategy="steps",
        per_device_eval_batch_size=config.EVAL_BATCH_SIZE,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=config.SAVE_STEPS,
        eval_steps=config.EVAL_STEPS,
        logging_steps=config.LOGGING_STEPS,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_ds_proc,
        eval_dataset=val_ds_proc,
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, processor.tokenizer),
        tokenizer=processor.feature_extractor,
    )
    return trainer
