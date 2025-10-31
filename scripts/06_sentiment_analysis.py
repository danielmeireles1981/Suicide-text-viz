import sys
from pathlib import Path
# Adiciona o diretório raiz do projeto ao sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from textblob import TextBlob
from src.data_io import save_csv

PROC = Path("data/processed")

def analyze_sentiment(text):
    """
    Analisa um texto e retorna polaridade, subjetividade e um rótulo de sentimento.
    Otimizado para ser chamado uma única vez por texto.
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0, 0.0, 'Neutro'

    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    subjectivity = analysis.sentiment.subjectivity

    if polarity > 0:
        label = 'Positivo'
    elif polarity < 0:
        label = 'Negativo'
    else:
        label = 'Neutro'
    
    return polarity, subjectivity, label

def run():
    print("🚀 Iniciando Análise de Sentimento...")
    df = pd.read_csv(PROC / "unified_with_features.csv")

    df["text_clean"] = df["text_clean"].fillna("").astype(str)

    # Otimização: Aplicar a análise uma vez e descompactar os resultados em 3 colunas
    sentiments = df['text_clean'].apply(analyze_sentiment)
    df[['sentiment_polarity', 'sentiment_subjectivity', 'sentiment_label']] = pd.DataFrame(sentiments.tolist(), index=df.index)

    save_csv(df, PROC / "unified_with_features.csv") # Sobrescreve com as novas colunas
    print(f"✅ Análise de sentimento concluída e dados salvos em: {PROC / 'unified_with_features.csv'}")

if __name__ == "__main__":
    run()