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

def split_dataset(df: pd.DataFrame):
    """
    Separa os dados de forma estratificada em:

      - Treino: 70% (398 registros)
      - Validação: 15% (85 registros)
      - Teste: 15% (86 registros)

    A separação é feita antes do pré-processamento para evitar data leakage.
    """
    feature_cols = [c for c in df.columns if c not in ("target", "diagnostico")]
    X = df[feature_cols].copy()
    y = df["target"].copy()

    # Separa 70% para treino e 30% para validação e teste.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # Divide o conjunto temporário igualmente entre validação e teste.
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )

    ensure_directories()
    X_train.assign(target=y_train).to_csv(
        os.path.join(DATA_PROCESSED_DIR, "train.csv"), index=False
    )
    X_val.assign(target=y_val).to_csv(
        os.path.join(DATA_PROCESSED_DIR, "val.csv"), index=False
    )
    X_test.assign(target=y_test).to_csv(
        os.path.join(DATA_PROCESSED_DIR, "test.csv"), index=False
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_preprocessor() -> Pipeline:
    """
    Cria o pipeline de pré-processamento numérico:
      1. Substitui valores faltantes pela mediana.
      2. Padroniza os valores usando média 0 e desvio-padrão 1.
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

def build_models() -> dict:
    """
    Instancia 3 arquiteturas distintas de Machine Learning acopladas ao Pipeline:

    1. Regressão Logística (modelo linear paramétrico)
    2. Árvore de Decisão (modelo não linear baseado em regras de corte)
    3. Random Forest (ensemble de 200 árvores via bagging)
    """
    return {
        "Regressão Logística": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Árvore de Decisão": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    DecisionTreeClassifier(
                        max_depth=4,
                        min_samples_leaf=5,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=6,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def evaluate_model(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    dataset_name: str,
    model_name: str,
) -> dict:
    """
    Avalia um modelo treinado, retornando métricas globais e a decomposição
    clínica da matriz de confusão (TP, TN, FP e FN).
    """
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        "Modelo": model_name,
        "Conjunto": dataset_name,
        "Accuracy": round(accuracy_score(y, y_pred), 4),
        "Precision": round(
            precision_score(y, y_pred, zero_division=0), 4
        ),
        "Recall (Sensibilidade)": round(
            recall_score(y, y_pred, zero_division=0), 4
        ),
        "Especificidade": round(
            tn / (tn + fp) if (tn + fp) > 0 else 0.0, 4
        ),
        "F1-Score": round(f1_score(y, y_pred, zero_division=0), 4),
        "ROC-AUC": round(roc_auc_score(y, y_proba), 4),
        "Falsos Negativos (FN)": int(fn),
        "Falsos Positivos (FP)": int(fp),
        "Verdadeiros Positivos (TP)": int(tp),
        "Verdadeiros Negativos (TN)": int(tn),
    }


def run_cross_validation(
    models: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> list:
    """
    Executa validação cruzada estratificada em 5 dobras no conjunto
    de treinamento para medir a estabilidade estatística.
    """
    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    cv_results = []

    for name, pipeline in models.items():
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=skf,
            scoring=scoring,
            n_jobs=-1,
        )
        cv_results.append(
            {
                "Modelo": name,
                "CV Accuracy (Média)": round(
                    float(np.mean(scores["test_accuracy"])), 4
                ),
                "CV Precision (Média)": round(
                    float(np.mean(scores["test_precision"])), 4
                ),
                "CV Recall (Média)": round(
                    float(np.mean(scores["test_recall"])), 4
                ),
                "CV F1-Score (Média)": round(
                    float(np.mean(scores["test_f1"])), 4
                ),
                "CV ROC-AUC (Média)": round(
                    float(np.mean(scores["test_roc_auc"])), 4
                ),
            }
        )

    return cv_results


class BreastCancerPredictor:
    """
    Carrega o modelo persistido (.joblib), valida as colunas de entrada
    e retorna as previsões.
    """

    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = os.path.join(MODELS_DIR, "modelo_final.joblib")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Modelo não encontrado em: {model_path}. "
                "Execute src/pipeline.py primeiro."
            )

        self.model = joblib.load(model_path)

        meta_path = os.path.join(MODELS_DIR, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as file:
                self.metadata = json.load(file)
            self.expected_features = self.metadata["feature_names"]
        else:
            self.expected_features = list(
                load_breast_cancer().feature_names
            )

    def validate_input(self, df: pd.DataFrame) -> pd.DataFrame:
        """Garante que o DataFrame contenha todas as variáveis exigidas."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                "Os dados de entrada devem ser um pandas.DataFrame."
            )

        missing = [
            column
            for column in self.expected_features
            if column not in df.columns
        ]
        if missing:
            raise ValueError(f"Colunas obrigatórias ausentes: {missing}")

        return df[self.expected_features].copy()

    def predict(self, df: pd.DataFrame, threshold: float = 0.50) -> list:
        """Retorna o rótulo clínico previsto e as probabilidades."""
        X_valid = self.validate_input(df)
        probas = self.model.predict_proba(X_valid)[:, 1]
        results = []

        for prob in probas:
            is_malignant = float(prob) >= threshold
            results.append(
                {
                    "diagnostico_previsto": (
                        "Maligno" if is_malignant else "Benigno"
                    ),
                    "target_previsto": 1 if is_malignant else 0,
                    "probabilidade_maligno": round(float(prob), 4),
                    "probabilidade_benigno": round(1.0 - float(prob), 4),
                }
            )

        return results

def train_and_evaluate_all():
    """
    Orquestra todo o ciclo de vida:

    1. Carregamento e EDA
    2. Split 70/15/15
    3. Validação cruzada 5-Fold e treinamento
    4. Avaliação em validação e teste
    5. Gráficos de matriz de confusão, curva ROC, feature importance e SHAP
    6. Persistência dos modelos (.joblib) e metadados (.json)
    """
    ensure_directories()

    df = load_and_prepare_data()
    perform_eda(df)

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df)
    models = build_models()

    cv_results = run_cross_validation(models, X_train, y_train)
    validation_results = []
    test_results = []
    trained_models = {}

    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        trained_models[name] = pipeline

        validation_results.append(
            evaluate_model(pipeline, X_val, y_val, "Validação", name)
        )
        test_results.append(
            evaluate_model(pipeline, X_test, y_test, "Teste", name)
        )

    df_cv_results = pd.DataFrame(cv_results)
    df_val_results = pd.DataFrame(validation_results)
    df_test_results = pd.DataFrame(test_results)

    # Exporta CSVs de métricas para reports/metrics/
    df_cv_results.to_csv(
        os.path.join(METRICS_DIR, "cross_validation_metrics.csv"),
        index=False,
    )
    df_val_results.to_csv(
        os.path.join(METRICS_DIR, "validation_metrics.csv"),
        index=False,
    )
    df_test_results.to_csv(
        os.path.join(METRICS_DIR, "test_metrics.csv"),
        index=False,
    )

    # Gráfico 4: Matrizes de Confusão Lado a Lado no Teste
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    for idx, (name, pipeline) in enumerate(trained_models.items()):
        y_pred = pipeline.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Benigno (0)", "Maligno (1)"],
            yticklabels=["Benigno (0)", "Maligno (1)"],
            ax=axes[idx],
            annot_kws={"size": 13, "weight": "bold"},
        )
        axes[idx].set_title(
            f"{name}\n(Conjunto de Teste)",
            fontsize=12,
            fontweight="bold",
        )
        axes[idx].set_xlabel("Predição do Modelo")
        axes[idx].set_ylabel("Diagnóstico Real")

    plt.suptitle(
        "Matrizes de Confusão no Conjunto de Teste "
        "(Foco em Minimizar Falsos Negativos)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURES_DIR, "04_matrizes_confusao_teste.png"),
        dpi=150,
    )
    plt.close()

    # Gráfico 5: Curva ROC Comparativa no Teste
    plt.figure(figsize=(8, 6))

    for name, pipeline in trained_models.items():
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_val = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, lw=2.2, label=f"{name} (AUC = {auc_val:.4f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1.2, label="Aleatório (AUC = 0.5000)")
    plt.xlabel("Taxa de Falsos Positivos (1 - Especificidade)")
    plt.ylabel("Taxa de Verdadeiros Positivos (Recall / Sensibilidade)")
    plt.title(
        "Curva ROC Comparativa - Conjunto de Teste",
        fontsize=13,
        fontweight="bold",
    )
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURES_DIR, "05_curva_roc_comparativa.png"),
        dpi=150,
    )
    plt.close()

    # Gráfico 6: Feature Importance Nativo da Random Forest
    rf_pipeline = trained_models["Random Forest"]
    rf_clf = rf_pipeline.named_steps["classifier"]

    importances = (
        pd.Series(rf_clf.feature_importances_, index=X_train.columns)
        .sort_values(ascending=True)
        .tail(15)
    )

    plt.figure(figsize=(9, 6))
    importances.plot(kind="barh", color="#3498db")
    plt.title(
        "Top 15 Variáveis Mais Importantes "
        "(Feature Importance - Random Forest)",
        fontsize=13,
        fontweight="bold",
    )
    plt.xlabel("Importância Relativa (Redução de Impureza de Gini)")
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURES_DIR, "06_feature_importance_rf.png"),
        dpi=150,
    )
    plt.close()

    # Gráficos 7 e 8: Interpretabilidade com SHAP
    preprocessor = rf_pipeline.named_steps["preprocessor"]
    X_test_scaled = pd.DataFrame(
        preprocessor.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )

    explainer = shap.TreeExplainer(rf_clf)
    shap_values = explainer(X_test_scaled)

    # Seleciona os valores SHAP referentes à classe 1 (Maligno)
    if len(shap_values.shape) == 3:
        shap_values_maligno = shap_values[:, :, 1]
    else:
        shap_values_maligno = shap_values

    # 7a. Summary Plot Global (Beeswarm)
    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values_maligno,
        X_test_scaled,
        show=False,
        max_display=15,
    )
    plt.title(
        "SHAP Summary Plot - Impacto Global das Variáveis no Risco Maligno",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURES_DIR, "07_shap_summary_plot.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    # 7b. Waterfall Plot Individual (paciente maligno do conjunto de teste)
    malign_indices = np.where(y_test.values == 1)[0]
    sample_idx = int(malign_indices[0])

    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(
        shap_values_maligno[sample_idx],
        max_display=12,
        show=False,
    )
    plt.title(
        f"Explicação Individual SHAP (Waterfall) - "
        f"Paciente #{sample_idx} (Diagnóstico: Maligno)",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURES_DIR, "08_shap_waterfall_paciente.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    # Persistência dos modelos e metadados
    joblib.dump(
        rf_pipeline,
        os.path.join(MODELS_DIR, "modelo_final.joblib"),
    )
    joblib.dump(
        trained_models,
        os.path.join(MODELS_DIR, "todos_modelos.joblib"),
    )

    metadata = {
        "feature_names": list(X_train.columns),
        "feature_medians": X_train.median().to_dict(),
        "feature_mins": X_train.min().to_dict(),
        "feature_maxs": X_train.max().to_dict(),
        "cv_metrics": cv_results,
        "validation_metrics": validation_results,
        "test_metrics": test_results,
    }

    with open(
        os.path.join(MODELS_DIR, "metadata.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)

    return df_val_results, df_test_results


if __name__ == "__main__":
    print(
        "Iniciando pipeline de EDA, pré-processamento, validação cruzada, "
        "treinamento, avaliação e SHAP..."
    )
    val_df, test_df = train_and_evaluate_all()

    print("\n=== RESULTADOS NA VALIDAÇÃO (15%) ===")
    print(val_df.to_string(index=False))

    print("\n=== RESULTADOS FINAIS NO TESTE (15%) ===")
    print(test_df.to_string(index=False))

    print(f"\nModelos salvos em: {MODELS_DIR}")
    print(f"Gráficos salvos em: {FIGURES_DIR}")

    