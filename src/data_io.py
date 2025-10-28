from pathlib import Path
import pandas as pd
from src.config import RAW, INTERIM, PROCESSED, FIGS

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def ensure_dirs():
    for p in [RAW, INTERIM, PROCESSED, FIGS]:
        ensure_dir(p)

def save_csv(df: pd.DataFrame, path: Path):
    ensure_dir(path.parent)
    df.to_csv(path, index=False)
