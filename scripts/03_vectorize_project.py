import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
import umap
from src.data_io import save_csv
from pathlib import Path

PROC = Path("data/processed")

def run():
    print("🚀 Iniciando vetorização e projeções...")
    df = pd.read_csv(PROC / "unified_with_features.csv")

    # --- Corrigir NaN e garantir strings válidas ---
    df["text_clean"] = df["text_clean"].fillna("").astype(str)
    df = df[df["text_clean"].str.strip() != ""].reset_index(drop=True)
    print(f"✅ Total de registros válidos: {df.shape[0]}")

    # --- TF-IDF com limitação de vocabulário ---
    # reduz o vocabulário (menos memória, melhor performance)
    tfidf = TfidfVectorizer(min_df=10, max_df=0.7, ngram_range=(1, 1))
    X = tfidf.fit_transform(df["text_clean"])
    print(f"📊 Matriz TF-IDF criada com formato: {X.shape}")

    # --- Projeção 2D com TruncatedSVD (PCA esparso, mais leve) ---
    print("⚙️  Gerando SVD (PCA esparso) 2D...")
    svd = TruncatedSVD(n_components=2, random_state=42)
    pca2 = svd.fit_transform(X)
    pca_df = pd.DataFrame(pca2, columns=["pca1", "pca2"])
    pca_df["idx"] = range(len(pca_df))
    save_csv(pca_df, PROC / "pca2_sample.csv")
    print("✅ Projeção SVD (PCA esparso) salva em pca2_sample.csv")

    # --- UMAP 2D (estrutura global dos dados) ---
    print("⚙️  Gerando projeção UMAP 2D (pode demorar alguns minutos)...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        n_jobs=-1  # Usa todos os núcleos de CPU para acelerar o processo
    )
    umap2 = reducer.fit_transform(X)
    umap_df = pd.DataFrame(umap2, columns=["umap1", "umap2"])
    save_csv(pd.concat([df[["label", "source"]], umap_df], axis=1), PROC / "umap2_full.csv")
    print("✅ Projeção UMAP salva em umap2_full.csv")

    # --- t-SNE (visualização local em pequena amostra) ---
    print("⚙️  Gerando projeção t-SNE (amostragem reduzida)...")
    n_ts = min(2000, X.shape[0])  # reduzir para poupar memória
    tsne2 = TSNE(
        n_components=2,
        init="random",
        learning_rate="auto",
        perplexity=30,
        random_state=42
    ).fit_transform(X[:n_ts].toarray().astype("float32"))
    tsne_df = pd.DataFrame(tsne2, columns=["tsne1", "tsne2"])
    tsne_df["idx"] = range(n_ts)
    save_csv(tsne_df, PROC / "tsne2_sample.csv")
    print("✅ Projeção t-SNE salva em tsne2_sample.csv")

    # --- Salvar o vetorizador TF-IDF ---
    joblib.dump(tfidf, PROC / "tfidf_vectorizer.joblib")
    print("💾 Vetorizador TF-IDF salvo em tfidf_vectorizer.joblib")
    print("🎉 Vetores e projeções gerados com sucesso!")

if __name__ == "__main__":
    run()
