import os
import re
import time
import shutil
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any

# === Text Normalization ===
ARABIC_DIACRITICS = re.compile(r'[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]')
TATWEEL            = '\u0640'
CONTROL_CHARS      = re.compile(r'[\x00-\u001F\u007F-\u009F]')
CHAR_UNIFY = {
    'ي': 'ی', 'ك': 'ک', 'ە': 'ە', 'أ': 'ا', 'إ': 'ا',
    'آ': 'ا', 'ٱ': 'ا', 'ة': 'ە',
}

def normalize_kurdish_text(text: Any) -> str:
    if not isinstance(text, str):
        return ''
    text = text.strip()
    if not text:
        return ''
    text = text.replace(TATWEEL, '')
    text = ARABIC_DIACRITICS.sub('', text)
    for src, dst in CHAR_UNIFY.items():
        text = text.replace(src, dst)
    text = CONTROL_CHARS.sub('', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# === File System Utilities ===
def human_size(n: float) -> str:
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if abs(n) < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"

def safe_delete(path: Path) -> None:
    path = Path(path).resolve()
    if path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path, ignore_errors=True)

def build_filename_index(root: Path, extensions: set) -> Dict[str, List[str]]:
    print(f"Building filename index for {root} ...")
    t0 = time.time()
    index: Dict[str, List[str]] = defaultdict(list)
    n_files = 0
    for p in root.rglob('*'):
        if not p.is_file():
            continue
        if p.suffix.lower() not in extensions:
            continue
        index[p.name].append(str(p.resolve()))
        n_files += 1
    elapsed = time.time() - t0
    print(f"  Indexed {n_files} audio files in {elapsed:.1f}s")
    return dict(index)
