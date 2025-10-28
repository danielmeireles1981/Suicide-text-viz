import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from src.data_io import ensure_dir

PROC = Path("data/processed")
FIGS = Path("reports/figures")
ensure_dir(FIGS)

# --- Funções de plotagem ---

def plot_balance(df):
    """Gráfico de barras mostrando o balanceamento de classes."""
    if "label" not in df.columns:
        print("⚠️ Coluna 'label' não encontrada, pulando gráfico de balanceamento.")
        return
    ax = df["label"].value_counts().sort_index().plot(kind="bar", color=["#8888ff", "#ff6666"])
    ax.set_title("Balanceamento de classes (0=controle, 1=ideação)")
    ax.set_xlabel("label")
    ax.set_ylabel("contagem")
    plt.tight_layout()
    plt.savefig(FIGS / "balanceamento_classes.png", dpi=150)
    plt.close()
    print("✅ Gráfico de balanceamento salvo.")

def plot_corr(df):
    """Heatmap de correlação das features numéricas."""
    num_cols = ["len", "n_hash", "n_mention", "n_exc", "n_q", "n_url_like", "upper_ratio", "label"]
    cols_exist = [c for c in num_cols if c in df.columns]

    if len(cols_exist) < 2:
        print("⚠️ Colunas numéricas não encontradas, pulando heatmap de correlação.")
        return

    corr = df[cols_exist].corr()
    plt.figure(figsize=(7, 5))
    sns.heatmap(corr, annot=False, cmap="vlag", center=0)
    plt.title("Correlação entre variáveis numéricas")
    plt.tight_layout()
    plt.savefig(FIGS / "correlacao.png", dpi=150)
    plt.close()
    print("✅ Heatmap de correlação salvo.")

def plot_umap():
    """Gráficos das projeções UMAP coloridos por label e por source."""
    umap_file = PROC / "umap2_full.csv"
    if not umap_file.exists():
        print("⚠️ Arquivo umap2_full.csv não encontrado, pulando UMAP.")
        return

    umap_df = pd.read_csv(umap_file)
    for color_by in ["label", "source"]:
        if color_by not in umap_df.columns:
            continue
        plt.figure(figsize=(6, 5))
        sns.scatterplot(
            data=umap_df,
            x="umap1", y="umap2",
            hue=color_by, s=10, alpha=0.6, linewidth=0
        )
        plt.title(f"UMAP 2D colorido por {color_by}")
        plt.legend(markerscale=2, bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(FIGS / f"umap_{color_by}.png", dpi=180)
        plt.close()
        print(f"✅ Gráfico UMAP colorido por {color_by} salvo.")

# --- Execução principal ---
if __name__ == "__main__":
    print("📊 Gerando gráficos...")
    csv_path = PROC / "unified_with_features.csv"

    if not csv_path.exists():
        csv_path = PROC / "unified.csv"
        print("⚠️ Arquivo unified_with_features.csv não encontrado. Usando unified.csv.")

    df = pd.read_csv(csv_path)
    plot_balance(df)
    plot_corr(df)
    plot_umap()
    print(f"🎨 Figuras salvas em: {FIGS}")
    print("🎉 Gráficos gerados com sucesso!")