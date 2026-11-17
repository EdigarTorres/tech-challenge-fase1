"""
Interface Médica de Apoio à Decisão Clínica (Streamlit)
Tech Challenge - Fase 1: Diagnóstico Assistido por IA com Interpretabilidade SHAP
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
FIGURES_DIR = os.path.join(BASE_DIR, "reports", "figures")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

st.set_page_config(
    page_title="Sistema de Apoio ao Diagnóstico Médico - IA",
    page_icon="🏥",
    layout="wide",
)


@st.cache_resource
def load_artifacts():
    """Carrega o modelo exportado e metadados; se não existirem, executa o pipeline."""
    model_path = os.path.join(MODELS_DIR, "modelo_final.joblib")
    all_models_path = os.path.join(MODELS_DIR, "todos_modelos.joblib")
    meta_path = os.path.join(MODELS_DIR, "metadata.json")

    if not (os.path.exists(model_path) and os.path.exists(meta_path)):
        from pipeline import train_and_evaluate_all
        train_and_evaluate_all()

    model = joblib.load(model_path)
    all_models = joblib.load(all_models_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    test_df = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "test.csv"))
    return model, all_models, metadata, test_df


model, all_models, metadata, test_df = load_artifacts()

# Cabeçalho Principal
st.title("🏥 Sistema Inteligente de Suporte à Triagem Oncológica (Câncer de Mama)")
st.markdown(
    """
    **Pós Tech - Tech Challenge (Fase 1)** | **Apoio à Decisão Clínica com Machine Learning e Interpretabilidade (SHAP)**  
    > ⚠️ **Aviso Clínico Importante:** Este sistema atua exclusivamente como **ferramenta de triagem e segunda opinião automatizada** para priorização hospitalar. **A decisão diagnóstica e conduta terapêutica final são sempre soberanas do médico responsável.**
    """
)

tab_triagem, tab_metricas, tab_eda = st.tabs([
    "🩺 Triagem de Paciente & Explicação SHAP",
    "📊 Avaliação de Modelos & Foco Clínico (Recall)",
    "🔍 Exploração de Dados (EDA)",
])

with tab_triagem:
    st.subheader("Simulador de Triagem de Exame Citopatológico (PAAF de Mama)")

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown("#### 1. Selecionar Perfil ou Ajustar Biomarcadores")
        preset = st.selectbox(
            "Carregar caso clínico de referência (Conjunto de Teste):",
            options=[
                "Paciente Real #1 (Caso Maligno - Alto Risco)",
                "Paciente Real #2 (Caso Benigno - Baixo Risco)",
                "Caso Limítrofe (Suspeita Moderada - Triagem Prioritária)",
            ],
        )

        malign_cases = test_df[test_df["target"] == 1]
        benign_cases = test_df[test_df["target"] == 0]

        if "Maligno" in preset:
            base_row = malign_cases.iloc[0].to_dict()
        elif "Benigno" in preset:
            base_row = benign_cases.iloc[0].to_dict()
        else:
            base_row = metadata["feature_medians"].copy()

        selected_Model_name = st.selectbox(
            "Algoritmo de Classificação:",
            options=list(all_models.keys()),
            index=2,  # Random Forest por padrão
        )
        active_pipeline = all_models[selected_Model_name]

        threshold = st.slider(
            "Limiar Clínico de Sensibilidade (Threshold de Probabilidade):",
            min_value=0.15,
            max_value=0.85,
            value=0.40,
            step=0.05,
            help="Em triagem médica, reduzir o limiar (ex: 0.35 - 0.40) aumenta o Recall (Sensibilidade), reduzindo o risco de Falsos Negativos.",
        )

        st.markdown("##### Principais Parâmetros Morfológicos (Ajustáveis)")
        key_features = [
            "worst radius",
            "worst concave points",
            "worst perimeter",
            "worst area",
            "mean concave points",
            "mean concavity",
            "mean radius",
            "mean texture",
        ]

        user_input = metadata["feature_medians"].copy()
        for k, v in base_row.items():
            if k in user_input:
                user_input[k] = float(v)

        for feat in key_features:
            min_val = float(metadata["feature_mins"][feat])
            max_val = float(metadata["feature_maxs"][feat])
            default_val = float(user_input[feat])
            user_input[feat] = st.slider(
                f"{feat}",
                min_value=min_val,
                max_value=max_val,
                value=default_val,
                step=(max_val - min_val) / 100.0,
            )

    with col_right:
        input_df = pd.DataFrame([user_input])[metadata["feature_names"]]
        prob_maligno = float(active_pipeline.predict_proba(input_df)[0, 1])
        pred_class = 1 if prob_maligno >= threshold else 0

        st.markdown("#### 2. Resultado da Análise Automatizada")
        m_col1, m_col2, m_col3 = st.columns(3)

        with m_col1:
            st.metric(
                label="Probabilidade de Malignidade",
                value=f"{prob_maligno * 100:.1f}%",
                delta="Acima do Limiar" if pred_class == 1 else "Abaixo do Limiar",
                delta_color="inverse" if pred_class == 1 else "normal",
            )
        with m_col2:
            st.metric(
                label="Classificação Sugerida",
                value="🔴 SUSPEITA MALIGNA" if pred_class == 1 else "🟢 PADRÃO BENIGNO",
            )
        with m_col3:
            prioridade = (
                "ALTA (Encaminhar Biópsia/Oncologia)"
                if prob_maligno >= 0.60
                else ("MÉDIA (Revisão Médica Prioritária)" if prob_maligno >= threshold else "ROTINA (Acompanhamento Padrão)")
            )
            st.metric(label="Prioridade na Fila de Triagem", value=prioridade)

        st.markdown("---")
        st.markdown("#### 3. Explicabilidade Clínica Individual (SHAP Waterfall Plot)")
        st.caption(
            "O gráfico abaixo mostra como cada característica celular empurrou a predição para maior risco de malignidade (vermelho) ou menor risco (azul)."
        )

        # Explicação local SHAP usando o modelo Random Forest
        rf_pipe = all_models["Random Forest"]
        prep = rf_pipe.named_steps["preprocessor"]
        clf = rf_pipe.named_steps["classifier"]
        input_scaled = pd.DataFrame(
            prep.transform(input_df),
            columns=input_df.columns,
        )
        explainer = shap.TreeExplainer(clf)
        sv = explainer(input_scaled)
        sv_malign = sv[:, :, 1] if len(sv.shape) == 3 else sv

        fig_shap, _ = plt.subplots(figsize=(9, 5))
        shap.plots.waterfall(sv_malign[0], max_display=10, show=False)
        plt.title("Fatores Determinantes para este Paciente (SHAP)", fontsize=11, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig_shap)
        plt.close(fig_shap)

with tab_metricas:
    st.subheader("Comparação de Modelos e Discussão Crítica das Métricas")
    st.markdown(
        """
        ### 🧠 Por que o **Recall (Sensibilidade)** é a métrica mais crítica em Diagnóstico Médico?
        Em um sistema de triagem oncológica:
        - **Falso Negativo (FN):** O paciente possui um tumor **Maligno**, mas o modelo classifica como **Benigno**. Consequência: atraso crítico no tratamento oncológico, com risco direto à vida.
        - **Falso Positivo (FP):** O paciente possui um nódulo **Benigno**, mas o sistema sinaliza alerta para revisão detalhada. Consequência: solicitação de exame complementar/revisão pelo patologista.
        
        Portanto, priorizamos maximizar o **Recall** da classe positiva (`Maligno = 1`), utilizando o **F1-Score** e o **ROC-AUC** para garantir que a precisão permaneça alta sem gerar fadiga de alertas na equipe clínica.
        """
    )

    if "cv_metrics" in metadata:
        st.markdown("#### Resultados na Validação Cruzada Estratificada (5-Fold Stratified K-Fold no Treino)")
        st.dataframe(pd.DataFrame(metadata["cv_metrics"]), use_container_width=True)

    col_v, col_t = st.columns(2)
    with col_v:
        st.markdown("#### Resultados na Validação (15%)")
        st.dataframe(pd.DataFrame(metadata["validation_metrics"]), use_container_width=True)
    with col_t:
        st.markdown("#### Resultados Finais no Teste (15% - Isolado)")
        st.dataframe(pd.DataFrame(metadata["test_metrics"]), use_container_width=True)

    st.markdown("---")
    g_col1, g_col2 = st.columns(2)
    with g_col1:
        st.image(os.path.join(FIGURES_DIR, "04_matrizes_confusao_teste.png"), caption="Matrizes de Confusão no Conjunto de Teste")
        st.image(os.path.join(FIGURES_DIR, "06_feature_importance_rf.png"), caption="Feature Importance Nativo (Random Forest)")
    with g_col2:
        st.image(os.path.join(FIGURES_DIR, "05_curva_roc_comparativa.png"), caption="Curvas ROC Comparativas no Conjunto de Teste")
        st.image(os.path.join(FIGURES_DIR, "07_shap_summary_plot.png"), caption="SHAP Summary Plot Global")

with tab_eda:
    st.subheader("Exploração de Dados (EDA) - Breast Cancer Wisconsin")
    e_col1, e_col2 = st.columns(2)
    with e_col1:
        st.image(os.path.join(FIGURES_DIR, "01_distribuicao_classes.png"), caption="Balanceamento entre Casos Benignos e Malignos")
    with e_col2:
        st.image(os.path.join(FIGURES_DIR, "03_matriz_correlacao.png"), caption="Matriz de Correlação das Top 15 Variáveis")
    st.image(os.path.join(FIGURES_DIR, "02_boxplots_principais_variaveis.png"), caption="Distribuição e Separação das Variáveis por Diagnóstico")
