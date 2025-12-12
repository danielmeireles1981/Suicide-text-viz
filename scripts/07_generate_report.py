import base64
import json
from pathlib import Path
import pandas as pd
from datetime import datetime
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import numpy as np

# --- Caminhos ---
BASE_DIR = Path(__file__).resolve().parent.parent
PROC = BASE_DIR / "data" / "processed"
FIGS = BASE_DIR / "reports" / "figures"
REPORTS = BASE_DIR / "reports"

# --- Função auxiliar: converter imagem em base64 ---
def img_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# --- Nuvem de palavras ---
def generate_wordcloud(df, label_value, output_name):
    subset = df[df["label"] == label_value]["text_clean"].dropna()
    text = " ".join(subset)
    wc = WordCloud(width=800, height=400, background_color="white", colormap="plasma").generate(text)
    path = FIGS / output_name
    wc.to_file(path)
    return img_to_base64(path)

# --- Top palavras por classe ---
def top_tfidf_terms_by_class(df, label_value, top_n=20):
    subset = df[df["label"] == label_value]["text_clean"].dropna()
    if subset.empty:
        return pd.DataFrame(columns=["termo", "peso"])
    vectorizer = TfidfVectorizer(max_df=0.8, min_df=5, stop_words="english")
    X = vectorizer.fit_transform(subset)
    means = np.asarray(X.mean(axis=0)).ravel()
    vocab = np.array(vectorizer.get_feature_names_out())
    top_idx = means.argsort()[::-1][:top_n]
    return pd.DataFrame({"termo": vocab[top_idx], "peso": means[top_idx]})

# --- Geração de interpretação automática ---
def gerar_interpretacao(df, top0, top1):
    texto = []

    # Diferença de proporção
    prop_suicida = (df["label"] == 1).mean() * 100
    texto.append(f"O conjunto total contém aproximadamente {prop_suicida:.1f}% de mensagens com indícios de ideação suicida.")

    # Comprimento médio
    if "len" in df.columns:
        len0 = df[df["label"] == 0]["len"].mean()
        len1 = df[df["label"] == 1]["len"].mean()
        if len1 > len0:
            texto.append(f"Mensagens suicidas tendem a ser mais longas ({len1:.1f} caracteres em média) do que as não suicidas ({len0:.1f}).")
        else:
            texto.append(f"Mensagens suicidas são geralmente mais curtas ({len1:.1f} vs {len0:.1f} caracteres em média).")

    # Palavras distintivas
    if not top0.empty and not top1.empty:
        top_terms_0 = ", ".join(top0["termo"].head(5))
        top_terms_1 = ", ".join(top1["termo"].head(5))
        texto.append(
            f"As palavras mais típicas em mensagens não suicidas são <b>{top_terms_0}</b>, "
            f"enquanto em mensagens com ideação suicida predominam <b>{top_terms_1}</b>."
        )

    texto.append("Esses padrões sugerem diferenças linguísticas relevantes entre os grupos, "
                 "possibilitando o uso de modelos de machine learning supervisionados para predição futura "
                 "ou análises psicossociais mais profundas.")

    return " ".join(texto)

# --- Gráfico de Sentimento ---
def generate_sentiment_chart(df):
    if "sentiment_label" not in df.columns:
        return None
    
    counts = df["sentiment_label"].value_counts()
    order = ["Negativo", "Neutro", "Positivo"]
    labels = [x for x in order if x in counts.index]
    values = [counts[x] for x in labels]
    colors = ["#d62728" if x == "Negativo" else "#7f7f7f" if x == "Neutro" else "#2ca02c" for x in labels]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color=colors, alpha=0.8, edgecolor='black')
    plt.title("Distribuição de Sentimento dos Textos")
    plt.xlabel("Sentimento")
    plt.ylabel("Contagem")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    
    path = FIGS / "sentiment_dist.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return img_to_base64(path)

# --- Relatório HTML ---
if __name__ == "__main__":
    print("🧩 Gerando relatório final com interpretação automática...")

    df = pd.read_csv(PROC / "unified_with_features.csv")

    # Estatísticas
    n_total = len(df)
    n_ideation = (df["label"] == 1).sum()
    n_non = (df["label"] == 0).sum()
    n_datasetA = (df["source"] == "DatasetA").sum()
    n_datasetB = (df["source"] == "DatasetB").sum()
    avg_len = df["len"].mean() if "len" in df.columns else None

    # Carregar gráficos
    img_balance = img_to_base64(FIGS / "balanceamento_classes.png")
    img_corr = img_to_base64(FIGS / "correlacao.png")
    img_umap_label = img_to_base64(FIGS / "umap_label.png")
    img_umap_source = img_to_base64(FIGS / "umap_source.png")
    img_sentiment = generate_sentiment_chart(df)

    # Nuvens e TF-IDF
    print("☁️  Gerando nuvens de palavras e top termos...")
    wc_0 = generate_wordcloud(df, 0, "wordcloud_class0.png")
    wc_1 = generate_wordcloud(df, 1, "wordcloud_class1.png")
    top0 = top_tfidf_terms_by_class(df, 0)
    top1 = top_tfidf_terms_by_class(df, 1)

    interpretacao = gerar_interpretacao(df, top0, top1)

    # Tabelas TF-IDF
    def df_to_html_table(df, color):
        if df.empty:
            return "<p><i>Sem dados disponíveis.</i></p>"
        rows = "".join(
            f"<tr><td>{r.termo}</td><td style='color:{color}; font-weight:bold;'>{r.peso:.4f}</td></tr>"
            for r in df.itertuples()
        )
        return f"<table style='border-collapse:collapse; width:80%; margin:auto;'><tr style='background:#f0f4f9;'><th>Termo</th><th>Peso TF-IDF Médio</th></tr>{rows}</table>"

    table0_html = df_to_html_table(top0, "#2a4b8d")
    table1_html = df_to_html_table(top1, "#b03060")

    # --- Textos Explicativos (Baseados no app.py) ---
    txt_balance = """
    <div style="margin-top:15px; padding:15px; background:#f4f6f9; border-left:4px solid #2a4b8d; border-radius:4px; font-size:0.95em;">
        <b>Interpretação Técnica:</b><br>
        A distribuição de classes apresenta um desequilíbrio significativo. Em contextos de detecção de risco psicossocial, o desbalanceamento pode induzir o modelo a um viés majoritário, comprometendo a sensibilidade ou a especificidade da detecção. É crucial aplicar técnicas de reamostragem ou pesos de classe durante o treinamento para garantir que o modelo aprenda a distinguir padrões sutis na classe minoritária, evitando generalizações excessivas.
    </div>
    """

    txt_corr = """
    <div style="margin-top:15px; padding:15px; background:#f4f6f9; border-left:4px solid #2a4b8d; border-radius:4px; font-size:0.95em;">
        <b>Análise de Features:</b><br>
        O mapa de calor indica correlações fracas entre as variáveis estruturais (metadados do texto) e a classe alvo. Isso demonstra que características superficiais, como o comprimento da mensagem ou contagem de caracteres, não são preditores confiáveis isoladamente para ideação suicida. A detecção eficaz depende, portanto, da análise semântica profunda e do contexto linguístico, justificando o uso de embeddings complexos e modelos de NLP avançados.
    </div>
    """

    txt_umap = """
    <div style="margin-top:15px; padding:15px; background:#f4f6f9; border-left:4px solid #2a4b8d; border-radius:4px; font-size:0.95em;">
        <b>Análise de Variedade (Manifold Learning):</b><br>
        A projeção UMAP revela a estrutura latente dos dados textuais:
        <ul style="margin-top:5px; margin-bottom:0;">
            <li><b>Separação de Classes:</b> Observa-se uma distinção topológica entre textos de controle e ideação, embora existam regiões de fronteira difusa, indicando ambiguidade semântica em certos casos.</li>
            <li><b>Viés de Domínio (Source Bias):</b> A forte clusterização baseada na fonte dos dados evidencia que cada origem possui uma "assinatura" linguística própria. Isso alerta para o risco de o modelo aprender características do dataset em vez do fenômeno clínico, exigindo estratégias de validação robustas.</li>
        </ul>
    </div>
    """

    txt_sentiment = ""
    if img_sentiment:
        txt_sentiment = f"""
        <div class="card">
            <h2>Análise de Sentimento</h2>
            <img src="data:image/png;base64,{img_sentiment}"/>
            <div style="margin-top:15px; padding:15px; background:#f4f6f9; border-left:4px solid #2a4b8d; border-radius:4px; font-size:0.95em;">
                <b>Análise de Polaridade:</b><br>
                A predominância de textos com polaridade negativa é consistente com a natureza do corpus. Contudo, a existência de segmentos classificados como neutros ou positivos evidencia a limitação de abordagens puramente baseadas em léxicos de sentimento. A ideação suicida pode ser expressa através de resignação calma (falso neutro) ou ironia, exigindo modelos capazes de inferir intenção pragmática além da polaridade superficial.
            </div>
        </div>
        """

    # --- Tópicos (LDA) ---
    html_topics = ""
    topics_path = PROC / "topics.json"
    if topics_path.exists():
        with open(topics_path, "r", encoding="utf-8") as f:
            topics_data = json.load(f)
        
        if topics_data:
            topics_list_html = ""
            for topic, words in topics_data.items():
                badges = "".join([f"<span style='background:#e1e4e8; color:#24292e; padding:2px 8px; margin:2px; border-radius:12px; font-size:0.85em; display:inline-block; border:1px solid #d1d5da;'>{w}</span>" for w in words])
                topics_list_html += f"<div style='margin-bottom:15px; break-inside: avoid;'><b>{topic}</b><br><div style='margin-top:5px;'>{badges}</div></div>"

            html_topics = f"""
            <div class="card">
                <h2>Modelagem de Tópicos (LDA)</h2>
                <div style="column-count: 2; column-gap: 20px;">{topics_list_html}</div>
                <div style="margin-top:15px; padding:15px; background:#f4f6f9; border-left:4px solid #2a4b8d; border-radius:4px; font-size:0.95em;">
                    <b>Interpretação Semântica:</b><br>
                    A Modelagem de Tópicos Latentes (LDA) identificou agrupamentos de palavras que co-ocorrem frequentemente, revelando os temas subjacentes do corpus. Tópicos contendo termos relacionados a sentimentos diretos indicam expressão de sofrimento, enquanto outros podem revelar estressores contextuais (escola, família, trabalho). Essa análise permite contextualizar a ideação suicida além da classificação binária.
                </div>
            </div>
            """

    # HTML
    html = f"""
<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<title>Relatório Analítico — Detecção de Ideação Suicida</title>
<style>
body {{font-family:'Segoe UI',Arial,sans-serif;background:linear-gradient(180deg,#eef2f8,#fff);margin:0;color:#222;}}
header {{background:#2a4b8d;color:white;text-align:center;padding:25px 0;box-shadow:0 2px 5px rgba(0,0,0,0.2);}}
section {{padding:40px;max-width:1100px;margin:0 auto;}}
.card {{background:white;border-radius:10px;box-shadow:0 0 10px rgba(0,0,0,0.1);padding:20px;margin-bottom:30px;transition:transform .2s;}}
.card:hover{{transform:scale(1.01);}}
.stats{{display:flex;flex-wrap:wrap;justify-content:space-around;text-align:center;}}
.stat{{flex:1 1 200px;background:#f8f9ff;margin:10px;padding:15px;border-radius:8px;box-shadow:0 0 5px rgba(0,0,0,0.05);}}
.stat b{{display:block;font-size:1.4em;color:#2a4b8d;}}
img{{display:block;margin:25px auto;border-radius:8px;max-width:95%;box-shadow:0 0 10px rgba(0,0,0,0.1);}}
h2{{color:#2a4b8d;text-align:center;}}
h3{{text-align:center;}}
table{{border:1px solid #ccc;margin-top:15px;}}
th,td{{border:1px solid #ccc;padding:5px 10px;text-align:center;}}
footer{{text-align:center;padding:20px;background:#2a4b8d;color:white;font-size:0.9em;margin-top:60px;}}
.analysis{{font-size:1.05em;line-height:1.6;margin:30px auto;background:#fefefe;padding:20px 30px;border-left:6px solid #2a4b8d;border-radius:8px;box-shadow:0 0 5px rgba(0,0,0,0.1);}}
</style>
</head>
<body>
<header>
<h1>📊 Relatório Analítico — Detecção de Ideação Suicida</h1>
<p>Gerado em {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</p>
</header>

<section>
<div class="card">
<h2>Resumo dos Dados</h2>
<div class="stats">
<div class="stat"><b>{n_total:,}</b>Total de registros</div>
<div class="stat"><b>{n_ideation:,}</b>Ideação suicida (classe 1)</div>
<div class="stat"><b>{n_non:,}</b>Não suicida (classe 0)</div>
<div class="stat"><b>{n_datasetA:,}</b>Dataset A</div>
<div class="stat"><b>{n_datasetB:,}</b>Dataset B</div>
<div class="stat"><b>{f"{avg_len:.2f}" if avg_len else "N/D"}</b>Tamanho médio do texto</div>
</div>
</div>

<div class="card"><h2>Distribuição de Classes</h2><img src="data:image/png;base64,{img_balance}"/>{txt_balance}</div>
<div class="card"><h2>Correlação entre Variáveis Numéricas</h2><img src="data:image/png;base64,{img_corr}"/>{txt_corr}</div>
<div class="card"><h2>Projeções UMAP</h2><img src="data:image/png;base64,{img_umap_label}"/><img src="data:image/png;base64,{img_umap_source}"/>{txt_umap}</div>
{txt_sentiment}
{html_topics}
<div class="card"><h2>Nuvens de Palavras</h2><h3>Classe 0 — Não Suicida</h3><img src="data:image/png;base64,{wc_0}"/><h3>Classe 1 — Ideação Suicida</h3><img src="data:image/png;base64,{wc_1}"/></div>
<div class="card"><h2>Top 20 Palavras por Classe</h2><h3>Classe 0 — Não Suicida</h3>{table0_html}<h3>Classe 1 — Ideação Suicida</h3>{table1_html}</div>
<div class="card"><h2>Análise Interpretativa</h2><div class="analysis">{interpretacao}</div></div>
</section>

<footer>
Relatório gerado automaticamente pelo pipeline <b>Suicide-Text-Viz</b><br>
C:\\Developer\\Suicide-text-viz
</footer>
</body>
</html>
"""

    REPORTS.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS / "report.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"✅ Relatório final salvo em: {out_path}")
    print("Abra o arquivo no navegador para visualizar.")
    