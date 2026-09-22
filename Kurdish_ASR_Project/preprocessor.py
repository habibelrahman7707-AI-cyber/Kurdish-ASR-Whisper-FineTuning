from typing import Dict, Any

class WhisperBatchPreprocessor:
    def __init__(self, processor):
        self.processor = processor

    def prepare_dataset_batched(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        audios = batch['audio']
        audio_arrays = [a['array'] for a in audios]
        sr = audios[0]['sampling_rate'] if audios else 16000

        batch['input_features'] = self.processor.feature_extractor(
            audio_arrays,
            sampling_rate=sr,
        ).input_features
        
        batch['labels'] = self.processor.tokenizer(batch['transcript']).input_ids
        return batch
