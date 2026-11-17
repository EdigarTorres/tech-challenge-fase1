import os
import json
import joblib

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # Permite salvar gráficos em disco sem abrir janelas pop-up

import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)

# Define estilo visual limpo para os gráficos e semente fixa de aleatoriedade
sns.set_theme(style="whitegrid", palette="muted")
RANDOM_STATE = 42

# Descobre dinamicamente o caminho da pasta raiz do projeto (independente do computador)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
FIGURES_DIR = os.path.join(BASE_DIR, "reports", "figures")
METRICS_DIR = os.path.join(BASE_DIR, "reports", "metrics")


def ensure_directories():
    """Garante que todas as pastas de saída existam antes de salvar arquivos."""
    for d in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, FIGURES_DIR, METRICS_DIR]:
        os.makedirs(d, exist_ok=True)


def load_and_prepare_data() -> pd.DataFrame:
    """
    Carrega o dataset Breast Cancer Wisconsin (Diagnostic).

    Recodifica o alvo para a convenção médica padrão:

      1 = Maligno (Presença da doença - classe positiva)
      0 = Benigno (Ausência de malignidade - classe negativa)
    """
    ensure_directories()

    csv_path = os.path.join(DATA_RAW_DIR, "breast_cancer_wisconsin.csv")

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        raw = load_breast_cancer()
        df = pd.DataFrame(raw.data, columns=raw.feature_names)

        # No sklearn original: 0 = Malignant, 1 = Benign. Invertemos com np.where:
        df["target"] = np.where(raw.target == 0, 1, 0)
        df["diagnostico"] = df["target"].map({1: "Maligno", 0: "Benigno"})
        df.to_csv(csv_path, index=False)

    return df


def perform_eda(df: pd.DataFrame):
    """
    Executa a Análise Exploratória de Dados (EDA) e salva 3 gráficos fundamentais:

      1. Proporção entre casos Benignos e Malignos (Balanceamento)
      2. Boxplots das 8 variáveis morfológicas mais importantes (Outliers e Separação)
      3. Matriz de Correlação de Pearson (Colinearidade e associação com o diagnóstico)
    """
    ensure_directories()

    feature_cols = [c for c in df.columns if c not in ("target", "diagnostico")]

    # Gráfico 1: Distribuição de Classes
    plt.figure(figsize=(7, 5))

    ax = sns.countplot(
        data=df,
        x="diagnostico",
        hue="diagnostico",
        palette={"Benigno": "#2ecc71", "Maligno": "#e74c3c"},
        legend=False,
    )

    plt.title("Distribuição do Diagnóstico (Benigno vs Maligno)", fontsize=13, fontweight="bold")
    plt.xlabel("Diagnóstico Clínico")
    plt.ylabel("Quantidade de Pacientes")

    for p in ax.patches:
        ax.annotate(
            f"{int(p.get_height())} ({p.get_height()/len(df)*100:.1f}%)",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    plt.ylim(0, len(df) * 0.75)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "01_distribuicao_classes.png"), dpi=150)
    plt.close()

    # Gráfico 2: Boxplots das principais variáveis por diagnóstico
    top_features = [
        "mean radius",
        "mean texture",
        "mean perimeter",
        "mean area",
        "mean concavity",
        "worst radius",
        "worst perimeter",
        "worst concave points",
    ]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()

    for i, col in enumerate(top_features):
        sns.boxplot(
            data=df,
            x="diagnostico",
            y=col,
            hue="diagnostico",
            palette={"Benigno": "#2ecc71", "Maligno": "#e74c3c"},
            ax=axes[i],
            legend=False,
        )
        axes[i].set_title(col, fontsize=11, fontweight="bold")
        axes[i].set_xlabel("")

    plt.suptitle("Distribuição e Outliers das Principais Variáveis por Diagnóstico", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "02_boxplots_principais_variaveis.png"), dpi=150)
    plt.close()

    # Gráfico 3: Matriz de Correlação (Top 15 variáveis mais correlacionadas com o target)
    corr_matrix = df[feature_cols + ["target"]].corr()

    top_corr_cols = (
        corr_matrix["target"]
        .abs()
        .sort_values(ascending=False)
        .head(16)
        .index.tolist()
    )

    plt.figure(figsize=(13, 10))

    sns.heatmap(
        df[top_corr_cols].corr(),
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        linewidths=0.5,
    )

    plt.title("Matriz de Correlação - Top 15 Variáveis Mais Associadas ao Diagnóstico", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "03_matriz_correlacao.png"), dpi=150)
    plt.close()

    return corr_matrix