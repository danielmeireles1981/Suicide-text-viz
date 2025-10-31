from src.data_io import ensure_dir, save_csv
from src.text_clean import basic_clean
import pandas as pd
from pathlib import Path

RAW = Path("data/raw")
PROC = Path("data/processed")

# --- Configuração dos Datasets ---
# Adicione ou modifique esta lista para incluir novos datasets.
# Você precisa especificar o nome do arquivo e os nomes das colunas de texto e rótulo.
DATASET_CONFIG = [
    {
        "filename": "Suicide_Detection.csv",
        "text_col": "text",
        "label_col": "class",
        "source_name": "DatasetA"
    },
    {
        "filename": "Suicide_Ideation_Dataset(Twitter-based).csv",
        "text_col": "Tweet",
        "label_col": "Suicide",
        "source_name": "DatasetB"
    },
    {
        "filename": "twitter-suicidal_data.csv",
        "text_col": "tweet",         # ATENÇÃO: Verifique se o nome da coluna de texto é 'tweet'
        "label_col": "intention",    # ATENÇÃO: Verifique se o nome da coluna de rótulo é 'intention'
        "source_name": "DatasetC"
    },
    {
        "filename": "data_raw_translated_en.csv",
        "text_col": "traducido",
        "label_col": "class",
        "source_name": "DatasetD"
    }
]

def load_and_unify():
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

    all_dfs = []
    print("🔎 Procurando e processando datasets em data/raw/...")
    for config in DATASET_CONFIG:
        filepath = RAW / config["filename"]
        if not filepath.exists():
            print(f"⚠️  Aviso: Arquivo '{config['filename']}' não encontrado. Pulando.")
            continue

        print(f"  -> Processando '{config['filename']}'...")
        temp_df = pd.read_csv(filepath)

        # --- Validação das colunas ---
        # Verifica se as colunas configuradas existem no DataFrame
        required_cols = [config["text_col"], config["label_col"]]
        missing_cols = [col for col in required_cols if col not in temp_df.columns]

        if missing_cols:
            print(f"❌ Erro em '{config['filename']}': Coluna(s) não encontrada(s): {missing_cols}.")
            print(f"   Colunas disponíveis no arquivo: {list(temp_df.columns)}")
            print("   Por favor, corrija 'text_col' ou 'label_col' na configuração DATASET_CONFIG e tente novamente.")
            continue # Pula para o próximo arquivo
        
        # Renomear colunas para o padrão ('text', 'label')
        temp_df = temp_df.rename(columns={
            config["text_col"]: "text",
            config["label_col"]: "label"
        })
        
        temp_df["source"] = config["source_name"]
        temp_df["label"] = temp_df["label"].apply(normalize_label)
        all_dfs.append(temp_df[["text", "label", "source"]])

    # --- limpeza simples dos textos ---
    for df in all_dfs:
        df["text"] = df["text"].astype(str).str.strip()
        df["text_clean"] = df["text"].map(basic_clean)

    # --- concatenar e remover duplicados ---
    df = pd.concat(all_dfs, ignore_index=True)
    df = df.drop_duplicates(subset=["text_clean"]).reset_index(drop=True)

    return df

if __name__ == "__main__":
    ensure_dir(PROC)
    df = load_and_unify()
    save_csv(df, PROC / "unified.csv")
    print(f"Unificado: {df.shape[0]} linhas, salvo em data/processed/unified.csv")
