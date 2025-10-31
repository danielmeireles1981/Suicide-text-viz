import pandas as pd
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from src.data_io import ensure_dir
from pathlib import Path

PROC = Path("data/processed")
N_TOPICS = 10  # Número de tópicos que queremos encontrar
N_TOP_WORDS = 15 # Número de palavras para descrever cada tópico
MAX_SAMPLES_FOR_LDA = 100000 # Limita o número de amostras para acelerar o LDA

def run_topic_modeling():
    """
    Executa a modelagem de tópicos LDA nos textos limpos.
    """
    print("🚀 Iniciando Modelagem de Tópicos (LDA)...")
    df = pd.read_csv(PROC / "unified_with_features.csv")

    # Garantir que o texto seja uma string limpa
    df["text_clean"] = df["text_clean"].fillna("").astype(str)
    texts = df["text_clean"][df["text_clean"].str.strip() != ""]

    if texts.empty:
        print("⚠️ Nenhum texto válido encontrado para a modelagem de tópicos.")
        return

    # --- Amostragem para performance ---
    # LDA é muito lento em datasets grandes. Uma amostra é suficiente.
    if len(texts) > MAX_SAMPLES_FOR_LDA:
        print(f"⚠️  Dataset muito grande ({len(texts)}). Usando uma amostra de {MAX_SAMPLES_FOR_LDA} para o LDA.")
        texts = texts.sample(MAX_SAMPLES_FOR_LDA, random_state=42)

    # Vetorização com contagem de palavras (melhor para LDA)
    # Limitamos o vocabulário para focar nas palavras mais relevantes
    vectorizer = CountVectorizer(
        max_df=0.8, min_df=20, stop_words='english', ngram_range=(1, 1)
    )
    X = vectorizer.fit_transform(texts)
    print(f"📊 Matriz de contagem criada com formato: {X.shape}")

    # Treinamento do modelo LDA
    lda = LatentDirichletAllocation(
        n_components=N_TOPICS,
        random_state=42,
        learning_method='online', # Eficiente para datasets grandes
        n_jobs=-1
    )
    lda.fit(X)
    print("✅ Modelo LDA treinado.")

    # Extração e salvamento dos tópicos
    feature_names = vectorizer.get_feature_names_out()
    topics = {}
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-N_TOP_WORDS - 1:-1]]
        topics[f"Tópico {topic_idx + 1}"] = top_words

    # Salvar os tópicos em um arquivo JSON para usar na aplicação
    ensure_dir(PROC)
    with open(PROC / "topics.json", "w", encoding="utf-8") as f:
        json.dump(topics, f, indent=2, ensure_ascii=False)

    print(f"💾 Tópicos salvos em: {PROC / 'topics.json'}")
    print("🎉 Modelagem de tópicos concluída com sucesso!")


if __name__ == "__main__":
    run_topic_modeling()
