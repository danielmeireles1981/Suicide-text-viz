import pandas as pd
from src.features import build_numeric_features
from src.data_io import save_csv
from pathlib import Path

PROC = Path("data/processed")

def run():
    print("🚀 Gerando colunas de features numéricas...")

    df = pd.read_csv(PROC / "unified.csv")

    # aplicar função e transformar a lista de dicionários em DataFrame
    feat_dicts = df["text_clean"].apply(build_numeric_features)
    feat_df = pd.DataFrame(list(feat_dicts))  # <- aqui está a diferença crucial

    # juntar o dataframe original com as novas colunas
    out = pd.concat([df, feat_df], axis=1)

    # salvar arquivo final
    save_csv(out, PROC / "unified_with_features.csv")
    print(f"✅ Features salvas em data/processed/unified_with_features.csv")
    print(f"✅ Colunas adicionadas: {list(feat_df.columns)}")

if __name__ == "__main__":
    run()
