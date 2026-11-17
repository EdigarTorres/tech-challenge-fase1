"""
Script gerador dos Notebooks Jupyter (.ipynb) solicitados na estrutura do projeto:
  - notebooks/01_eda_e_preprocessamento.ipynb
  - notebooks/02_modelagem_e_avaliacao_shap.ipynb
"""

import os
import nbformat as nbf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOKS_DIR = os.path.join(BASE_DIR, "notebooks")
os.makedirs(NOTEBOOKS_DIR, exist_ok=True)


def create_notebook_01():
    nb = nbf.v4.new_notebook()
    cells = [
        nbf.v4.new_markdown_cell(
            "# 01 - Exploração de Dados (EDA) e Pipeline de Pré-processamento\n"
            "**Tech Challenge - Fase 1 | Sistema de Suporte ao Diagnóstico de Câncer de Mama**\n\n"
            "### 1. Contexto Clínico e Definição do Problema\n"
            "O objetivo deste estudo é construir a base de um sistema inteligente de apoio à triagem médica utilizando o dataset público **Breast Cancer Wisconsin (Diagnostic)**.\n"
            "- **Variável Alvo (`target`):** Recodificada para o padrão clínico onde `1 = Maligno (Presença da doença)` e `0 = Benigno (Ausência de malignidade)`.\n"
            "- **Impacto Hospitalar:** Automatizar a análise inicial de parâmetros morfológicos de núcleos celulares (obtidos por Punção Aspirativa por Agulha Fina - PAAF) permite priorizar pacientes de alto risco na fila de patologia e oncologia, reduzindo erros diagnósticos."
        ),
        nbf.v4.new_code_cell(
            "import os\n"
            "import sys\n"
            "import pandas as pd\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n\n"
            "sys.path.append(os.path.abspath('../src'))\n"
            "from pipeline import load_and_prepare_data, split_dataset, build_preprocessor\n\n"
            "df = load_and_prepare_data()\n"
            "print(f'Dimensões do Dataset: {df.shape}')\n"
            "df.head()"
        ),
        nbf.v4.new_markdown_cell(
            "### 2. Inspeção de Tipos, Valores Ausentes e Estatísticas Descritivas"
        ),
        nbf.v4.new_code_cell(
            "print('Valores ausentes por coluna:', df.isnull().sum().sum())\n"
            "print('\\nDistribuição das classes:')\n"
            "print(df['diagnostico'].value_counts(normalize=True).mul(100).round(2).astype(str) + '%')\n"
            "df.describe().T.head(12)"
        ),
        nbf.v4.new_markdown_cell(
            "### 3. Visualização de Distribuições e Matriz de Correlação\n"
            "Analisamos a separabilidade das classes e a colinearidade entre atributos geométricos (`radius`, `perimeter`, `area`, `concavity`)."
        ),
        nbf.v4.new_code_cell(
            "from IPython.display import Image, display\n"
            "display(Image('../reports/figures/01_distribuicao_classes.png'))\n"
            "display(Image('../reports/figures/02_boxplots_principais_variaveis.png'))\n"
            "display(Image('../reports/figures/03_matriz_correlacao.png'))"
        ),
        nbf.v4.new_markdown_cell(
            "### 4. Divisão Estratificada (Treino 70% / Validação 15% / Teste 15%) e Pipeline sem Data Leakage\n"
            "A separação ocorre **antes** de qualquer cálculo de mediana (`SimpleImputer`) ou padronização (`StandardScaler`)."
        ),
        nbf.v4.new_code_cell(
            "X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df)\n"
            "print(f'Treino: {X_train.shape} | Validação: {X_val.shape} | Teste: {X_test.shape}')\n\n"
            "preprocessor = build_preprocessor()\n"
            "X_train_processed = preprocessor.fit_transform(X_train)\n"
            "print('Pipeline ajustado com sucesso no conjunto de treino!')"
        ),
    ]
    nb["cells"] = cells
    with open(os.path.join(NOTEBOOKS_DIR, "01_eda_e_preprocessamento.ipynb"), "w", encoding="utf-8") as f:
        nbf.write(nb, f)


def create_notebook_02():
    nb = nbf.v4.new_notebook()
    cells = [
        nbf.v4.new_markdown_cell(
            "# 02 - Modelagem, Avaliação Crítica e Interpretabilidade com SHAP\n"
            "**Tech Challenge - Fase 1 | Comparação de Algoritmos e Explicabilidade Clínica**\n\n"
            "### 1. Treinamento de Múltiplos Modelos de Classificação\n"
            "Avaliamos três famílias de algoritmos acoplados ao Pipeline de pré-processamento:\n"
            "1. **Regressão Logística** (Baseline linear interpretável com `StandardScaler`)\n"
            "2. **Árvore de Decisão** (Regras de corte explícitas)\n"
            "3. **Random Forest** (Ensemble não-linear de alta performance)"
        ),
        nbf.v4.new_code_cell(
            "import os\n"
            "import sys\n"
            "import pandas as pd\n"
            "from IPython.display import Image, display\n\n"
            "sys.path.append(os.path.abspath('../src'))\n"
            "from pipeline import train_and_evaluate_all\n\n"
            "df_val, df_test = train_and_evaluate_all()\n"
            "print('=== TABELA COMPARATIVA: CONJUNTO DE VALIDAÇÃO (15%) ===')\n"
            "display(df_val)\n"
            "print('=== TABELA COMPARATIVA: CONJUNTO DE TESTE (15%) ===')\n"
            "display(df_test)"
        ),
        nbf.v4.new_markdown_cell(
            "### 2. Justificativa Clínica da Métrica: Por que priorizar o Recall?\n"
            "- **Falso Negativo (FN):** Classificar um tumor **Maligno** como **Benigno** é o erro mais grave na medicina diagnóstica, pois retarda o início do tratamento oncológico.\n"
            "- **Escolha Técnica:** Priorizamos o **Recall (Sensibilidade)** da classe positiva (`Maligno = 1`) combinado ao **F1-Score** e **ROC-AUC** para manter alta precisão sem gerar alarmes falsos excessivos."
        ),
        nbf.v4.new_code_cell(
            "display(Image('../reports/figures/04_matrizes_confusao_teste.png'))\n"
            "display(Image('../reports/figures/05_curva_roc_comparativa.png'))"
        ),
        nbf.v4.new_markdown_cell(
            "### 3. Interpretabilidade do Modelo: Feature Importance e SHAP\n"
            "Para que o médico confie no suporte à decisão, o modelo não pode ser uma caixa-preta. Utilizamos **Feature Importance** e **SHAP (SHapley Additive exPlanations)** para explicar tanto o comportamento global do algoritmo quanto o diagnóstico individual de cada paciente."
        ),
        nbf.v4.new_code_cell(
            "display(Image('../reports/figures/06_feature_importance_rf.png'))\n"
            "display(Image('../reports/figures/07_shap_summary_plot.png'))\n"
            "display(Image('../reports/figures/08_shap_waterfall_paciente.png'))"
        ),
        nbf.v4.new_markdown_cell(
            "### 4. Discussão Crítica: O Modelo Pode Ser Utilizado na Prática Clínica?\n"
            "**Sim, como ferramenta de triagem e apoio à decisão (CDSS - Clinical Decision Support System), nunca em substituição ao médico:**\n"
            "1. **Priorização de Fila (Triagem Inteligente):** Exames com probabilidade elevada de malignidade ganham prioridade máxima na fila de revisão do médico patologista/oncologista.\n"
            "2. **Segunda Opinião Explicável:** Ao exibir o gráfico **SHAP Waterfall** ao lado do laudo preliminar, o médico visualiza imediatamente quais variáveis morfológicas (`worst radius`, `worst concave points`, `worst perimeter`) motivaram o alerta.\n"
            "3. **Soberania Médica:** O médico correlaciona o escore preditivo com anamnese, exame físico, histórico familiar e biópsia histopatológica antes de definir o diagnóstico definitivo."
        ),
    ]
    nb["cells"] = cells
    with open(os.path.join(NOTEBOOKS_DIR, "02_modelagem_e_avaliacao_shap.ipynb"), "w", encoding="utf-8") as f:
        nbf.write(nb, f)


if __name__ == "__main__":
    create_notebook_01()
    create_notebook_02()
    print(f"Notebooks criados com sucesso em: {NOTEBOOKS_DIR}")
