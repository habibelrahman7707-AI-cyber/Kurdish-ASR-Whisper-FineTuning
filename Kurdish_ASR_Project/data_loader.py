import shutil
import zipfile
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional

from utils import safe_delete, normalize_kurdish_text

try:
    import rarfile
    rarfile.UNRAR_TOOL = shutil.which('unrar') or 'unrar'
except ImportError:
    rarfile = None

AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.opus', '.aac'}
ARCHIVE_EXTENSIONS = {'.rar', '.zip'}

def extract_archives(archives_dir: Path, local_rars_dir: Path, extract_dir: Path, delete_local_after: bool = True) -> Dict[str, int]:
    archives = sorted([p for p in archives_dir.iterdir() if p.is_file() and p.suffix.lower() in ARCHIVE_EXTENSIONS])
    if not archives:
        print(f"⚠️  No .rar/.zip files found in {archives_dir}")
        return {}

    results: Dict[str, int] = {}
    for i, drive_ar in enumerate(archives, 1):
        dest = extract_dir / drive_ar.stem
        dest.mkdir(parents=True, exist_ok=True)
        
        existing_audio = [p for p in dest.rglob('*') if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS]
        if existing_audio:
            results[drive_ar.name] = len(existing_audio)
            continue

        local_ar = local_rars_dir / drive_ar.name
        try:
            shutil.copy2(drive_ar, local_ar)
            if local_ar.suffix.lower() == '.rar':
                with rarfile.RarFile(local_ar, 'r') as rf:
                    rf.extractall(dest)
            else:
                with zipfile.ZipFile(local_ar, 'r') as zf:
                    zf.extractall(dest)

            audio_count = sum(1 for p in dest.rglob('*') if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS)
            results[drive_ar.name] = audio_count
        except Exception as exc:
            print(f"❌ EXTRACT FAILED ({exc})")
            results[drive_ar.name] = 0
            if local_ar.exists():
                safe_delete(local_ar)
            continue

        if delete_local_after and local_ar.exists():
            safe_delete(local_ar)

    return results

def resolve_audio_path_via_index(raw: Any, index: Dict[str, List[str]]) -> Optional[str]:
    if not isinstance(raw, str) or not raw.strip():
        return None
    basename = Path(raw.strip()).name
    candidates = index.get(basename)
    if candidates:
        for c in candidates:
            if Path(c).exists():
                return c
    return None

def normalize_schema(df: pd.DataFrame, dialect: Optional[str] = None) -> pd.DataFrame:
    COLUMN_ALIASES = {
        'audio_path': 'audio_path', 'audio': 'audio_path', 'file': 'audio_path',
        'filename': 'audio_path', 'file_name': 'audio_path', 'path': 'audio_path',
        'transcript': 'transcript', 'text': 'transcript', 'sentence': 'transcript',
        'duration': 'duration', 'duration_sec': 'duration'
    }
    rename_map = {}
    for col in df.columns:
        key = str(col).strip().lower().replace(' ', '_')
        if key in COLUMN_ALIASES:
            rename_map[col] = COLUMN_ALIASES[key]
    df = df.rename(columns=rename_map)

    for required in ['audio_path', 'transcript', 'dialect', 'duration']:
        if required not in df.columns:
            df[required] = pd.NA

    if dialect is not None:
        mask = df['dialect'].isna() | (df['dialect'].astype(str).str.strip() == '')
        df.loc[mask, 'dialect'] = dialect

    return df[['audio_path', 'transcript', 'dialect', 'duration']]

def clean_dataframe(df: pd.DataFrame, index: Dict[str, List[str]], dialect: str) -> pd.DataFrame:
    df = df.copy()
    df = df.loc[:, ~df.columns.str.contains('^Unnamed', regex=True)]
    df['audio_path'] = df['audio_path'].apply(lambda x: resolve_audio_path_via_index(x, index))
    df['transcript'] = df['transcript'].apply(normalize_kurdish_text)
    
    df = df.dropna(subset=['audio_path', 'transcript'])
    df = df[df['transcript'].str.len() >= 1]
    df = df[df['audio_path'].astype(str).str.len() > 0]
    return df.reset_index(drop=True)
