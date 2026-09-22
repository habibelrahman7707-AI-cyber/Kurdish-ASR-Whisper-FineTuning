import pandas as pd
import pyarrow as pa
from datasets import Dataset, Features, Value, Audio as AudioFeature

def build_features(df: pd.DataFrame, sampling_rate: int) -> Features:
    feat = {}
    for col in df.columns:
        if col == 'audio_path':
            feat[col] = AudioFeature(sampling_rate=sampling_rate)
        elif col in ('transcript', 'dialect', 'id'):
            feat[col] = Value('string')
        elif col == 'duration':
            feat[col] = Value('float64')
        elif col == 'transcript_len':
            feat[col] = Value('int64')
        else:
            dtype = str(df[col].dtype)
            if 'int' in dtype:
                feat[col] = Value('int64')
            elif 'float' in dtype:
                feat[col] = Value('float64')
            else:
                feat[col] = Value('string')
    return Features(feat)

def df_to_dataset(df: pd.DataFrame, sampling_rate: int) -> Dataset:
    features = build_features(df, sampling_rate)
    data = {}
    for col in df.columns:
        if col in ('duration', 'transcript_len'):
            data[col] = df[col].tolist()
        else:
            data[col] = [str(x) if x is not None else None for x in df[col].tolist()]

    try:
        return Dataset.from_dict(data, features=features)
    except Exception as exc1:
        print(f"  from_dict failed: {exc1}. Trying explicit pa.string() table...")
    
    arrays = {}
    for col in df.columns:
        if col == 'audio_path':
            arrays[col] = pa.array(data[col], type=pa.string())
        elif col == 'duration':
            arrays[col] = pa.array(data[col], type=pa.float64())
        elif col == 'transcript_len':
            arrays[col] = pa.array(data[col], type=pa.int64())
        else:
            arrays[col] = pa.array(data[col], type=pa.string())
    table = pa.table(arrays)
    return Dataset.from_arrow_table(table, features=features)
