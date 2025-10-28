from src.data_io import ensure_dirs, save_csv
from src.text_clean import basic_clean
import pandas as pd
from pathlib import Path

RAW = Path("data/raw")
PROC = Path("data/processed")

def load_and_unify():
    a = pd.read_csv(RAW / "Suicide_Detection.csv")
    b = pd.read_csv(RAW / "Suicide_Ideation_Dataset(Twitter-based).csv")

    # normalizar para (text, label, source)
    df_a = a.rename(columns={"class": "label"})[["text", "label"]].copy()
    df_a["source"] = "DatasetA"

    df_b = b.rename(columns={"Tweet": "text", "Suicide": "label"})[["text", "label"]].copy()
    df_b["source"] = "DatasetB"

    # --- Mapeamento ampliado e função robusta de normalização ---
    maplab = {
        "suicide": 1,
        "non-suicide": 0,
        "Suicide": 1,
        "Non-Suicide": 0,
        "Suicide post": 1,
        "Not Suicide post": 0,
        "Potential Suicide post": 1,
        "Suicide-related": 1,
        "Not Suicide-related": 0,
        "Yes": 1,
        "No": 0,
        1: 1,
        0: 0
    }

    def normalize_label(x):
        """Normaliza os valores de rótulo para 0 ou 1."""
        if isinstance(x, str):
            x = x.strip()
            return maplab.get(x, 0)  # assume 0 se não estiver no mapa
        elif pd.notna(x):
            try:
                return int(x)
            except:
                return 0
        return 0

    # aplicar normalização
    df_a["label"] = df_a["label"].apply(normalize_label)
    df_b["label"] = df_b["label"].apply(normalize_label)

    # --- limpeza simples dos textos ---
    for df in (df_a, df_b):
        df["text"] = df["text"].astype(str).str.strip()
        df["text_clean"] = df["text"].map(basic_clean)

    # --- concatenar e remover duplicados ---
    df = pd.concat([df_a, df_b], ignore_index=True)
    df = df.drop_duplicates(subset=["text_clean"]).reset_index(drop=True)

    return df


if __name__ == "__main__":
    ensure_dirs()
    df = load_and_unify()
    save_csv(df, PROC / "unified.csv")
    print(f"Unificado: {df.shape[0]} linhas, salvo em data/processed/unified.csv")
