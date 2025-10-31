import time

# Importa as funções principais de cada script do pipeline
from scripts.s01_unify_datasets import load_and_unify
from scripts.s02_build_features import run as run_build_features
from scripts.s03_vectorize_project import run as run_vectorize
from scripts.s04_make_plots import run as run_make_plots
from scripts.s05_topic_modeling import run_topic_modeling
from scripts.s06_sentiment_analysis import run as run_sentiment_analysis

def main():
    """
    Executa o pipeline completo de processamento de dados e geração de análises.
    """
    start_time = time.time()
    print("=================================================")
    print("🚀 INICIANDO PIPELINE COMPLETO DE ANÁLISE DE DADOS 🚀")
    print("=================================================\n")

    # Dicionário de etapas para facilitar a execução e o logging
    pipeline_steps = {
        "1. Unificação de Datasets": load_and_unify,
        "2. Construção de Features": run_build_features,
        "3. Vetorização e Projeção": run_vectorize,
        "4. Análise de Sentimento": run_sentiment_analysis,
        "5. Modelagem de Tópicos": run_topic_modeling,
        "6. Geração de Gráficos": run_make_plots,
    }

    for name, step_func in pipeline_steps.items():
        step_start_time = time.time()
        print(f"\n--- Etapa {name} ---\n")
        step_func()
        step_duration = time.time() - step_start_time
        print(f"\n--- Etapa {name} concluída em {step_duration:.2f} segundos ---")

    total_duration = time.time() - start_time
    print(f"\n======================================================")
    print(f"🎉 PIPELINE COMPLETO CONCLUÍDO COM SUCESSO! 🎉")
    print(f"   Tempo total de execução: {total_duration:.2f} segundos.")
    print(f"======================================================\n")

if __name__ == "__main__":
    main()