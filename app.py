import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import json

# --- Configuração da Página ---
st.set_page_config(
    page_title="Análise de Textos sobre Ideação Suicida",
    page_icon="📊",
    layout="wide"
)

# --- Caminhos (ajuste conforme a estrutura do seu projeto) ---
PROC_PATH = Path("data/processed")
FIGS_PATH = Path("reports/figures")

# --- Funções de Cache para Carregar Dados (melhora a performance) ---
@st.cache_data
def load_data(file_path):
    """Carrega um arquivo CSV de forma segura."""
    if file_path.exists():
        return pd.read_csv(file_path)
    return None

@st.cache_data
def load_json(file_path):
    """Carrega um arquivo JSON de forma segura."""
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

# --- Título e Introdução ---
st.title("📊 Análise e Visualização de Textos sobre Ideação Suicida")
st.markdown("""
Esta aplicação apresenta os resultados da análise de múltiplos datasets públicos contendo textos
relacionados à ideação suicida. Explore os gráficos abaixo para entender a estrutura dos dados.
""")

# --- Carregar os dados ---
df_features = load_data(PROC_PATH / "unified_with_features.csv")
df_umap = load_data(PROC_PATH / "umap2_full.csv")
topics = load_json(PROC_PATH / "topics.json")

if df_features is None or df_umap is None:
    st.error(
        "Arquivos de dados não encontrados! "
        "Por favor, execute o pipeline de dados primeiro (`scripts/01_...` a `scripts/06_...`)."
    )
    st.stop()


# --- Layout com Colunas ---
col1, col2 = st.columns(2)

with col1:
    st.header("Balanceamento de Classes")
    balance_img_path = FIGS_PATH / "balanceamento_classes.png"
    if balance_img_path.exists():
        st.image(str(balance_img_path), use_container_width=True)
    else:
        st.warning("Imagem do balanceamento não encontrada.")
    st.info(
        """
        **O que isso significa?**
        Imagine que você quer ensinar uma máquina a diferenciar maçãs de laranjas, mas você mostra 100 fotos de maçãs e apenas 10 de laranjas. A máquina pode ficar "viciada" em dizer "maçã". Da mesma forma, nosso conjunto de dados tem muito mais textos sobre ideação suicida (label 1). Isso é um **ponto de atenção**, pois um modelo de IA pode tender a classificar tudo como "ideação" por ter visto mais exemplos.
        """
    )

with col2:
    st.header("Correlação de Features")
    corr_img_path = FIGS_PATH / "correlacao.png"
    if corr_img_path.exists():
        st.image(str(corr_img_path), use_container_width=True)
    else:
        st.warning("Imagem da correlação não encontrada.")
    st.info(
        """
        **O que isso significa?**
        Este "mapa de calor" mostra se características simples, como o tamanho do texto ou o número de "!", têm relação com o fato de ele ser sobre ideação suicida. As cores claras, próximas de zero, indicam uma **correlação muito fraca**. Ou seja, saber que um texto é longo ou curto não nos ajuda a classificá-lo. A verdadeira pista está no **significado das palavras**, e não nessas contagens superficiais.
        """
    )

st.divider()

# --- Gráfico UMAP Interativo ---
st.header("Projeção UMAP Interativa")
st.markdown("Passe o mouse sobre os pontos para ver detalhes. Use a legenda para filtrar.")
st.info(
    """
    **Conclusão Final: A História Contada pelo UMAP**

    Este mapa é a principal ferramenta para entendermos a estrutura do nosso universo de textos. Ele revela duas verdades cruciais:

    1.  **O Problema é Complexo:** Ao colorir por `label`, vemos que os textos de "ideação" e "controle" formam "continentes" distintos, mas suas fronteiras são muito misturadas. Isso prova que a linguagem usada em ambos os casos é, muitas vezes, parecida, tornando a tarefa de separação um desafio que vai além de simples palavras-chave.

    2.  **Cada Dataset tem um "Sotaque":** Ao colorir por `source`, notamos que os textos se agrupam fortemente de acordo com sua origem (Dataset A, B, C, D). Isso é um achado crítico: cada dataset tem um "dialeto" ou estilo de escrita próprio, provavelmente devido à plataforma de onde foi extraído (Reddit, Twitter, etc.) ou ao método de coleta.

    **A Grande Conclusão:** A combinação desses dois pontos nos leva à principal conclusão do projeto: o maior desafio não é apenas diferenciar "ideação" de "controle", mas fazê-lo **sem que o modelo de IA "trapaceie" aprendendo o "sotaque" de cada fonte**. Por exemplo, se um dataset é majoritariamente de "controle" e tem um estilo único, um modelo preguiçoso pode aprender a regra perigosa: *"Se o texto tem esse sotaque, então não é ideação"*. Este **"viés de fonte" (source bias)** é a principal armadilha a ser evitada, e o UMAP a torna visível e inegável.
    """
)

color_option = st.selectbox(
    "Colorir projeção por:",
    ("label", "source")
)

fig_umap = px.scatter(
    df_umap,
    x="umap1",
    y="umap2",
    color=color_option,
    hover_data={"umap1": False, "umap2": False, color_option: True},
    title=f"Projeção UMAP 2D colorida por {color_option}",
    labels={'color': color_option}
)
fig_umap.update_traces(marker=dict(size=5, opacity=0.7))
fig_umap.update_layout(height=700)  # Aumenta a altura do gráfico
st.plotly_chart(fig_umap, use_container_width=True)


# --- Amostra dos Dados ---
st.header("Amostra dos Dados Processados")
st.markdown("Abaixo uma amostra aleatória dos dados unificados e com features.")
st.caption("A coluna `text_clean` contém o texto após a limpeza básica, usada para a vetorização.")
st.dataframe(df_features.sample(10, random_state=42))

st.divider()

# --- Análise de Sentimento ---
st.header("Análise de Sentimento")
st.markdown(
    "Esta seção apresenta a distribuição do sentimento (positivo, neutro, negativo) dos textos, "
    "calculado usando a biblioteca `TextBlob`. Isso nos ajuda a entender o tom geral das mensagens."
)

if 'sentiment_label' in df_features.columns:
    sentiment_counts = df_features['sentiment_label'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentimento', 'Contagem']

    # Definir uma ordem para as categorias e cores
    sentiment_order = ['Negativo', 'Neutro', 'Positivo']
    sentiment_colors = {'Negativo': 'red', 'Neutro': 'gray', 'Positivo': 'green'}

    fig_sentiment = px.bar(
        sentiment_counts,
        x='Sentimento',
        y='Contagem',
        title='Distribuição de Sentimento dos Textos',
        color='Sentimento',
        color_discrete_map=sentiment_colors,
        category_orders={'Sentimento': sentiment_order}
    )
    st.plotly_chart(fig_sentiment, use_container_width=True)
    st.info(
        """
        **O que isso significa?**
        Como esperado, a grande maioria dos textos tem um tom **negativo**. O interessante é a presença de textos classificados como **neutros** ou até **positivos**. Isso pode indicar mensagens de apoio, postagens com humor negro, ou expressões de resignação que não carregam um tom explicitamente negativo, mostrando a diversidade de formas como o tema é abordado.
        """
    )

    # --- Cruzamento de Sentimento com Classe ---
    st.subheader("Análise de Sentimento por Classe")
    sentiment_class_counts = df_features.groupby(['sentiment_label', 'label']).size().reset_index(name='Contagem')
    sentiment_class_counts['Classe'] = sentiment_class_counts['label'].map({0: 'Controle', 1: 'Ideação Suicida'})

    fig_cross_sentiment = px.bar(
        sentiment_class_counts,
        x='sentiment_label',
        y='Contagem',
        color='Classe',
        barmode='group',
        title='Distribuição de Sentimento Agrupada por Classe',
        labels={'sentiment_label': 'Sentimento do Texto', 'Contagem': 'Número de Textos'},
        category_orders={'sentiment_label': sentiment_order},
        color_discrete_map={'Controle': '#8888ff', 'Ideação Suicida': '#ff6666'}
    )
    st.plotly_chart(fig_cross_sentiment, use_container_width=True)
    st.info(
        """
        **O que isso significa?**
        Este gráfico confirma que textos sobre ideação suicida são, em sua maioria, negativos. Porém, ele revela algo crucial: uma parte significativa desses textos é classificada como **neutra ou positiva**. Isso mostra que a ideação suicida nem sempre é expressa com palavras de tristeza ou raiva. Às vezes, ela pode vir em forma de planejamento calmo ou até em textos que parecem otimistas superficialmente, o que torna a detecção automática um desafio ainda maior.
        """
    )

else:
    st.warning("Dados de sentimento não encontrados. Execute o script `06_sentiment_analysis.py`.")

st.divider()

# --- Análise de Tópicos (LDA) ---
st.header("Análise de Tópicos (LDA)")
st.markdown(
    "A Modelagem de Tópicos é uma técnica de IA que encontra grupos de palavras semanticamente relacionadas (tópicos) "
    "que ocorrem frequentemente juntas nos textos. É como descobrir as 'receitas' de assuntos que compõem nosso conjunto de dados."
)

if topics:
    for topic_name, words in topics.items():
        with st.expander(f"**{topic_name}**"):
            # Formatando as palavras em badges coloridos para melhor visualização
            word_html = "".join([
                f'<span style="background-color: #e0e0e0; border-radius: 5px; padding: 3px 8px; margin: 3px; display: inline-block;">{word}</span>'
                for word in words
            ])
            st.markdown(word_html, unsafe_allow_html=True)
    st.info(
        """
        **O que esses tópicos significam?**
        Cada "tópico" é um conjunto de palavras que a IA descobriu que aparecem frequentemente juntas. Eles representam os principais temas discutidos nos textos. Por exemplo, um tópico com palavras como `[vida, morrer, quero, dor, fim]` está claramente relacionado a expressões diretas de ideação suicida. Outro tópico com `[escola, trabalho, amigos, família]` pode representar os estressores da vida cotidiana que são mencionados nos textos. Analisar esses temas nos ajuda a entender o contexto em que a ideação suicida é discutida.
        """
    )
else:
    st.warning("Arquivo de tópicos (`topics.json`) não encontrado. Execute o script `05_topic_modeling.py`.")

st.divider()

# --- Conclusão Geral e Próximos Passos ---
st.header("Conclusão Geral do Projeto e Próximos Passos")
st.markdown(
    """
    A pergunta que surge da análise UMAP é: *"Se os datasets são tão diferentes, a análise é inválida?"*

    **A resposta é o oposto.** A principal conclusão deste projeto **não é um fracasso**, mas sim um **sucesso analítico crucial**: a descoberta e visualização do **"viés de fonte" (source bias)**.

    - **O que descobrimos:** Descobrimos que os datasets não formam um grupo homogêneo. Eles têm "sotaques" distintos. A relação entre eles é de maior ou menor semelhança estilística, e não de uniformidade.

    - **Por que isso é valioso:** Essa descoberta nos protege de criar um modelo de IA ingênuo e perigoso. Um modelo treinado em todos esses dados juntos aprenderia a "trapacear", usando o "sotaque" de cada fonte para adivinhar a classe, em vez de entender o conteúdo real sobre ideação suicida. Ele teria um bom desempenho em dados de teste *do mesmo conjunto*, mas falharia catastroficamente no mundo real.

    **Próximos Passos Sugeridos:**
    1.  **Modelagem Consciente do Viés:** Em vez de um único modelo, poderíamos treinar modelos separados para cada "dialeto" ou usar técnicas avançadas de *Domain Adaptation* para ensinar um modelo a ignorar o "sotaque".
    2.  **Validação Robusta:** Criar um conjunto de validação com dados de uma fonte completamente nova seria a única maneira de medir verdadeiramente a capacidade de generalização de um modelo.
    """
)
