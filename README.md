Developed a high-performance Automatic Speech Recognition (ASR) pipeline to fine-tune OpenAI’s Whisper model specifically for underrepresented Kurdish dialects (Sorani and Bahdini).
Engineered a batched feature extraction system utilizing HuggingFace's WhisperProcessor, achieving a 32x speedup in generating log-Mel spectrograms from raw audio files.
Resolved complex memory and dataset casting bottlenecks by implementing PyArrow optimizations to handle massive text-audio datasets without large_string mapping crashes.
Automated deep data normalization processes to clean Arabic and Kurdish scripts, remove diacritics, and map thousands of audio files to metadata seamlessly using dynamic indexing.
Utilized mixed-precision training (BF16/FP16) with the HuggingFace Seq2SeqTrainer to maximize GPU utilization and rigorously evaluated the model using the jiwer Word Error Rate (WER) metric.
