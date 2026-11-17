# FIAP Pós Tech — Tech Challenge (Fase 1)
## Inteligência Artificial e Machine Learning Aplicados ao Apoio ao Diagnóstico Médico

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-orange.svg)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-purple.svg)](https://shap.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Clinical%20App-red.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-blue.svg)](https://www.docker.com/)

---

## 1. Sobre o projeto

Este projeto foi desenvolvido como entregável oficial do **Tech Challenge — Fase 1** da **Pós Tech (FIAP)**, com o propósito de construir a base de um **Sistema Inteligente de Suporte à Decisão Clínica (CDSS — *Clinical Decision Support System*)** voltado à triagem automatizada de exames médicos.

Em hospitais universitários e centros oncológicos com volume crescente de exames citopatológicos e prontuários digitalizados, atrasos na fila de análise podem comprometer o prognóstico de pacientes graves. Nossa solução aplica **Machine Learning supervisionado** e **Inteligência Artificial Explicável (XAI com SHAP)** para classificar nódulos mamários como **Benignos (`0`)** ou **Malignos (`1`)** a partir de parâmetros morfométricos extraídos de exames de Punção Aspirativa por Agulha Fina (PAAF).

A arquitetura do projeto contempla:
- **Análise Exploratória de Dados (EDA)** detalhada com visualizações estatísticas e estudo de multicolinearidade;
- **Pipeline de Pré-processamento** isolado contra vazamento de dados (*Data Leakage*);
- **Treinamento, Validação Cruzada Estratificada (*Stratified K-Fold*) e Comparação** de múltiplos algoritmos de classificação;
- **Avaliação Clínica Crítica**, priorizando a redução de **Falsos Negativos** por meio da métrica de **Recall (Sensibilidade)** combinada ao **F1-Score** e **ROC-AUC**;
- **Interpretabilidade Global e Local (SHAP)** para eliminar o efeito "caixa-preta" e fornecer transparência à equipe médica;
- **Persistência do Modelo (`joblib`)** e **Interface Web Interativa em Streamlit** containerizada via **Docker** para simulação de triagem hospitalar em tempo real.

---

## 2. Objetivos

- Explorar, limpar e compreender a distribuição estatística e o comportamento biológico das variáveis do exame citopatológico de mama.
- Identificar padrões geométricos, assimetrias, *outliers* e correlações lineares entre os atributos celulares e o diagnóstico oncológico.
- Construir um pipeline reprodutível de engenharia de atributos e pré-processamento (`SimpleImputer` + `StandardScaler`) livre de *Data Leakage*.
- Implementar, validar e comparar três famílias distintas de algoritmos de aprendizado supervisionado (**Regressão Logística**, **Árvore de Decisão** e **Random Forest**).
- Defender tecnicamente a escolha das métricas de avaliação sob a ótica médica (**Recall / Sensibilidade** como métrica prioritária para evitar Falsos Negativos).
- Explicar tanto o funcionamento global do modelo quanto o porquê de cada predição individual utilizando **Feature Importance** e **SHAP (*SHapley Additive exPlanations*)**.
- Disponibilizar um sistema interativo em **Streamlit** pronto para execução local ou em container **Docker**.

---

## 3. Dataset

### Breast Cancer Wisconsin (Diagnostic) Dataset

Foi utilizado o consagrado conjunto de dados clínicos **Breast Cancer Wisconsin (Diagnostic)**, disponibilizado publicamente pela UCI Machine Learning Repository / Kaggle e integrado à biblioteca `scikit-learn`.

- **Fonte Oficial (Kaggle):** [https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
- **Origem Clínica:** As características são calculadas a partir de imagens digitalizadas de Punção Aspirativa por Agulha Fina (PAAF) de massas mamárias, descrevendo propriedades morfológicas dos núcleos celulares presentes na amostra.

#### Características do Dataset

| Informação | Quantidade / Detalhe |
| :--- | :--- |
| **Total de Registros (Pacientes)** | 569 |
| **Características Preditivas (Features Numéricas)** | 30 variáveis contínuas |
| **Valores Ausentes Originais** | 0 (Pipeline preparado com imputação por mediana para produção) |
| **Diagnósticos Benignos (`target = 0`)** | 357 pacientes (62,74%) |
| **Diagnósticos Malignos (`target = 1`)** | 212 pacientes (37,26%) |

#### Grupos de Variáveis Morfológicas (Calculadas em Média `mean`, Erro Padrão `error` e Pior Valor `worst`)

1. **`radius` (Raio):** Média das distâncias do centro até os pontos do perímetro celular.
2. **`texture` (Textura):** Desvio padrão dos valores da escala de cinza na imagem citológica.
3. **`perimeter` (Perímetro):** Contorno total do núcleo celular.
4. **`area` (Área):** Superfície ocupada pelo núcleo celular.
5. **`smoothness` (Suavidade):** Variação local nos comprimentos dos raios.
6. **`compactness` (Compacidade):** Calculada como $\frac{\text{perímetro}^2}{\text{área}} - 1.0$.
7. **`concavity` (Concavidade):** Severidade das porções côncavas do contorno celular.
8. **`concave points` (Pontos Côncavos):** Número de reentrâncias côncavas no contorno celular.
9. **`symmetry` (Simetria):** Grau de simetria nuclear.
10. **`fractal dimension` (Dimensão Fractal):** Aproximação de irregularidade da borda ("coastline approximation").

> **Nota de Padronização Clínica:** No pacote bruto do `scikit-learn`, a classe `0` representa maligno e `1` representa benigno. Em nosso pipeline ([`load_and_prepare_data`](file:///C:/Users/ediga/.gemini/antigravity/scratch/tech-challenge-fase1/src/pipeline.py#L45-L64)), **invertemos explicitamente a codificação para o padrão epidemiológico (`1 = Maligno / Doente` e `0 = Benigno / Saudável`)**, garantindo que as métricas de **Recall**, **Precision**, **F1-Score** e os gráficos **SHAP** reflitam diretamente o risco de câncer.

---

## 4. Tecnologias utilizadas

| Tecnologia | Versão Base | Aplicação no Projeto |
| :--- | :---: | :--- |
| **Python** | 3.10 / 3.11 | Linguagem principal de desenvolvimento e ciência de dados |
| **Pandas** | `>= 2.0.0` | Manipulação tabular, limpeza e exportação de dados e métricas |
| **NumPy** | `>= 1.24.0` | Operações vetoriais, matrizes e cálculos numéricos |
| **Scikit-learn** | `>= 1.3.0` | Pipelines de pré-processamento, divisão estratificada, modelos de ML e métricas |
| **SHAP** | `>= 0.44.0` | Inteligência Artificial Explicável (`TreeExplainer`, `summary_plot`, `waterfall`) |
| **Matplotlib** | `>= 3.7.0` | Geração e exportação de figuras científicas de alta resolução |
| **Seaborn** | `>= 0.12.0` | Visualizações estatísticas (boxplots, distribuições e matriz de correlação) |
| **Streamlit** | `>= 1.30.0` | Interface web interativa para simulação médica e apoio à decisão |
| **Joblib** | `>= 1.3.0` | Serialização e persistência dos pipelines e modelos treinados |
| **Jupyter / NBFormat** | `>= 1.0.0` | Notebooks interativos para demonstração passo a passo de EDA e modelagem |
| **Docker** | Engine 24+ | Empacotamento e execução reprodutível de todo o ecossistema |

---

## 5. Estrutura do projeto

O projeto segue uma arquitetura modular de Ciência de Dados e Engenharia de Machine Learning:

```text
tech-challenge-fase1/
│
├── data/
│   ├── raw/
│   │   └── breast_cancer_wisconsin.csv       # Dataset bruto gerado automaticamente
│   └── processed/
│       ├── train.csv                         # Conjunto de Treinamento (70% - 398 registros)
│       ├── val.csv                           # Conjunto de Validação (15% - 85 registros)
│       └── test.csv                          # Conjunto de Teste isolado (15% - 86 registros)
│
├── notebooks/
│   ├── 01_eda_e_preprocessamento.ipynb       # Notebook documentado de EDA e Pipeline
│   └── 02_modelagem_e_avaliacao_shap.ipynb   # Notebook de Modelagem, Métricas e SHAP
│
├── src/
│   ├── pipeline.py                           # Orquestrador completo (EDA, Split, Treino, SHAP, Predictor)
│   ├── app.py                                # Aplicação Web Interativa em Streamlit para Triagem Médica
│   └── generate_notebooks.py                 # Gerador automatizado dos notebooks Jupyter
│
├── models/
│   ├── modelo_final.joblib                   # Pipeline campeão persistido (Preprocessor + Random Forest)
│   ├── todos_modelos.joblib                  # Dicionário contendo todos os modelos treinados
│   └── metadata.json                         # Estatísticas de treino e métricas consolidadas
│
├── reports/
│   ├── figures/                              # Gráficos exportados automaticamente pelo pipeline
│   │   ├── 01_distribuicao_classes.png
│   │   ├── 02_boxplots_principais_variaveis.png
│   │   ├── 03_matriz_correlacao.png
│   │   ├── 04_matrizes_confusao_teste.png
│   │   ├── 05_curva_roc_comparativa.png
│   │   ├── 06_feature_importance_rf.png
│   │   ├── 07_shap_summary_plot.png
│   │   └── 08_shap_waterfall_paciente.png
│   ├── metrics/                              # Tabelas CSV com resultados de CV, Validação e Teste
│   │   ├── cross_validation_metrics.csv
│   │   ├── validation_metrics.csv
│   │   └── test_metrics.csv
│   └── relatorio_tecnico.md                  # Relatório técnico completo pronto para exportação em PDF
│
├── tests/
│   └── test_predictor.py                     # Testes unitários automatizados (dados, split e inferência)
│
├── .dockerignore
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 6. Análise exploratória dos dados (EDA)

A etapa de exploração de dados é executada pela função [`perform_eda`](file:///C:/Users/ediga/.gemini/antigravity/scratch/tech-challenge-fase1/src/pipeline.py#L67-L138) e documentada no notebook `01_eda_e_preprocessamento.ipynb`, abrangendo:

1. **Inspeção de Integridade e Tipos:** Verificação das 30 variáveis contínuas (`float64`) e confirmação de inexistência de registros corrompidos ou duplicados.
2. **Análise de Balanceamento de Classes:**
   - **Benigno (`0`):** 357 casos (62,7%)
   - **Maligno (`1`):** 212 casos (37,3%)
   - *Implicação Técnica:* Existe um desbalanceamento moderado na razão de ~1,68:1. Para evitar que os classificadores favoreçam a classe majoritária (Benigna), aplicamos **estratificação no split** (`stratify=y`) e **ponderação inversamente proporcional às frequências de classe** (`class_weight='balanced'`).
3. **Distribuições e *Outliers* (`02_boxplots_principais_variaveis.png`):**
   - Tumores malignos exibem medianas e intervalos interquartis substancialmente superiores em variáveis de dimensão (`mean radius`, `worst radius`, `worst perimeter`, `worst area`) e irregularidade de contorno (`mean concavity`, `worst concave points`).
   - Há presença de *outliers* naturais em casos severos de malignidade, motivo pelo qual optamos pela **mediana** como estratégia de imputação no pipeline.
4. **Análise de Correlação e Multicolinearidade (`03_matriz_correlacao.png`):**
   - Forte correlação positiva (> `0.77`) entre o diagnóstico maligno (`target`) e as variáveis `worst concave points` (`0.79`), `worst perimeter` (`0.78`), `mean concave points` (`0.78`) e `worst radius` (`0.78`).
   - Alta colinearidade geométrica (> `0.95`) entre `radius`, `perimeter` e `area`, visto que perímetro ($2\pi r$) e área ($\pi r^2$) são funções diretas do raio.

---

## 7. Pré-processamento e Pipeline (Prevenção de *Data Leakage*)

Um dos erros mais comuns em projetos acadêmicos é aplicar normalização ou imputação na base inteira antes de dividir os dados, causando vazamento de informação (*Data Leakage*) do conjunto de teste para o treinamento.

Em nossa implementação ([`split_dataset`](file:///C:/Users/ediga/.gemini/antigravity/scratch/tech-challenge-fase1/src/pipeline.py#L141-L166) e [`build_preprocessor`](file:///C:/Users/ediga/.gemini/antigravity/scratch/tech-challenge-fase1/src/pipeline.py#L169-L181)):

### 7.1. Divisão Estratificada dos Dados

Os dados são divididos **antes** de qualquer transformação matemática, preservando exatamente a proporção de 37,3% de casos malignos em todas as partições:

| Conjunto | Registros | Proporção | Papel no Fluxo de Machine Learning |
| :--- | :---: | :---: | :--- |
| **Treinamento (`train.csv`)** | 398 | 70% | Ajuste (`fit`) dos transformadores, Validação Cruzada 5-Fold e treinamento dos modelos |
| **Validação (`val.csv`)** | 85 | 15% | Comparação intermediária de hiperparâmetros e calibragem de limiar clínico |
| **Teste (`test.csv`)** | 86 | 15% | Avaliação final estritamente isolada (dados nunca vistos pelos modelos) |
| **Total** | **569** | **100%** | — |

### 7.2. Pipeline Scikit-Learn

Todas as transformações são encapsuladas em um objeto `Pipeline` oficial do `scikit-learn`:
1. **`SimpleImputer(strategy="median")`:** Caso um novo exame em produção contenha alguma medição ausente (`NaN`), substitui pela mediana aprendida exclusivamente no conjunto de treino (robusta a valores extremos).
2. **`StandardScaler()`:** Padroniza cada variável para média $\mu = 0$ e desvio padrão $\sigma = 1$ com base apenas na média e variância do conjunto de treino, garantindo convergência rápida e pesos comparáveis na Regressão Logística.

---

## 8. Modelos implementados

Foram implementadas e comparadas três técnicas distintas de aprendizado supervisionado em [`build_models`](file:///C:/Users/ediga/.gemini/antigravity/scratch/tech-challenge-fase1/src/pipeline.py#L184-L238):

### 8.1. Regressão Logística (`LogisticRegression`)
- **Justificativa:** Modelo linear paramétrico clássico em bioestatística e epidemiologia médica. Serve como excelente *baseline* de alta interpretabilidade e baixa variância em espaços padronizados via `StandardScaler`.
- **Configuração:** `class_weight="balanced"`, `max_iter=1000`, `random_state=42`.

### 8.2. Árvore de Decisão (`DecisionTreeClassifier`)
- **Justificativa:** Algoritmo não-paramétrico baseado em regras hierárquicas de corte (*if-then-else*), simulando fluxogramas de triagem clínica.
- **Configuração:** `max_depth=4`, `min_samples_leaf=5` (poda preventiva contra *overfitting*), `class_weight="balanced"`, `random_state=42`.

### 8.3. Random Forest (`RandomForestClassifier`)
- **Justificativa:** Comitê (*Ensemble* via *Bagging*) de 200 árvores de decisão descorrelacionadas. Captura interações não-lineares complexas entre atributos morfológicos, reduz drasticamente a variância de uma árvore isolada e possui integração exata e ultrarrápida com o `TreeExplainer` da biblioteca **SHAP**.
- **Configuração:** `n_estimators=200`, `max_depth=6`, `min_samples_leaf=2`, `class_weight="balanced"`, `random_state=42`.

---

## 9. Avaliação dos modelos e Escolha Crítica da Métrica

### 🧠 A Mente Clínica: Por que o **Recall (Sensibilidade)** é Prioritário?

Na medicina diagnóstica, os erros de classificação possuem consequências profundamente assimétricas:

- **Falso Negativo (FN — Erro Crítico):** O paciente possui um tumor **Maligno (`1`)**, mas o algoritmo o classifica como **Benigno (`0`)**. Esse paciente pode deixar de ser priorizado para biópsia confirmatória, atrasando o tratamento oncológico e colocando sua vida em risco.
- **Falso Positivo (FP — Alerta Conservador):** O paciente possui um nódulo **Benigno (`0`)**, mas o modelo emite um alerta de suspeita maligna. O paciente passa por revisão médica prioritária pelo patologista, que descarta a malignidade com segurança.

Portanto, a métrica principal de defesa deste projeto é o **Recall (Sensibilidade)** da classe positiva (`Maligno`), acompanhado pelo **F1-Score** (média harmônica entre *Precision* e *Recall*) e pelo **ROC-AUC** (capacidade discriminativa global em diferentes limiares de probabilidade) para evitar excesso de alarmes falsos.

### Métricas Calculadas

- **Accuracy:** Proporção total de diagnósticos corretos.
- **Precision (Valor Preditivo Positivo):** Proporção de alertas malignos que são verdadeiramente malignos ($\frac{TP}{TP + FP}$).
- **Recall (Sensibilidade):** Proporção de casos reais de câncer detectados pelo modelo ($\frac{TP}{TP + FN}$).
- **Especificidade:** Proporção de casos benignos corretamente identificados ($\frac{TN}{TN + FP}$).
- **F1-Score:** Equilíbrio harmônico entre *Precision* e *Recall* ($2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$).
- **ROC-AUC:** Área sob a curva ROC (varia de `0.5` a `1.0`).

### Desempenho Comparativo Esperado (Conjunto de Teste — 86 registros: 54 Benignos e 32 Malignos)

| Modelo | Accuracy | Precision | Recall (Sensibilidade) | F1-Score | ROC-AUC | Falsos Negativos (FN) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Regressão Logística** | ~97,67% | ~96,88% | **~96,88%** | ~96,88% | ~99,54% | $\le 1$ |
| **Random Forest** | ~96,51% | ~96,77% | **~93,75% - 96,88%** | ~95,24% | ~99,31% | $\le 2$ |
| **Árvore de Decisão** | ~91,86% | ~87,88% | **~90,63%** | ~89,23% | ~94,10% | 3 |

*(As tabelas exatas geradas a cada execução ficam salvas em `reports/metrics/cross_validation_metrics.csv`, `validation_metrics.csv` e `test_metrics.csv`, e podem ser consultadas diretamente na aba **Avaliação de Modelos** da interface Streamlit).*

---

## 10. Interpretabilidade (Feature Importance e SHAP)

Em ambientes hospitalares, modelos preditivos do tipo "caixa-preta" enfrentam forte rejeição clínica. Um médico não pode basear uma conduta apenas em uma porcentagem sem saber **quais características celulares** levaram àquela conclusão. Por isso, implementamos duas camadas de interpretabilidade:

### 10.1. Feature Importance Nativo (`06_feature_importance_rf.png`)
Extrai a redução média de impureza de Gini proporcionada por cada variável na **Random Forest**. As 5 variáveis de maior importância global identificadas são:
1. `worst perimeter` (Perímetro nos núcleos mais alterados)
2. `worst radius` (Raio nos núcleos mais alterados)
3. `worst concave points` (Pontos côncavos nos núcleos mais alterados)
4. `mean concave points` (Média de pontos côncavos)
5. `worst area` (Área nos núcleos mais alterados)

### 10.2. SHAP — *SHapley Additive exPlanations*
Baseado na Teoria dos Jogos Cooperativos de Lloyd Shapley, o SHAP calcula a contribuição marginal exata de cada variável para a predição:

1. **Explicação Global (`07_shap_summary_plot.png` — *Beeswarm Plot*):**
   - Demonstra a direção do efeito biológico: pontos vermelhos (valores elevados de `worst radius`, `worst concave points`, `worst perimeter` e `worst area`) concentram-se fortemente no lado positivo do eixo SHAP, comprovando que o aumento de tamanho e irregularidade côncava eleva drasticamente a probabilidade de malignidade.
2. **Explicação Individual por Paciente (`08_shap_waterfall_paciente.png` — *Waterfall Plot*):**
   - Mostra como o modelo partiu do valor esperado base $E[f(X)]$ e somou/subtraiu as contribuições de cada exame daquele paciente específico até atingir o escore final $f(x)$. Na aplicação Streamlit, esse gráfico é gerado dinamicamente ao vivo para qualquer paciente avaliado!

---

## 11. Persistência, Preditor e Aplicação Web Interativa (Streamlit)

### 11.1. Persistência e Classe `BreastCancerPredictor`
- O pipeline final treinado é serializado em `models/modelo_final.joblib` (junto com todos os modelos em `models/todos_modelos.joblib` e metadados em `models/metadata.json`).
- A classe [`BreastCancerPredictor`](file:///C:/Users/ediga/.gemini/antigravity/scratch/tech-challenge-fase1/src/pipeline.py#L286-L328) encapsula o carregamento do artefato `.joblib`, valida se o `DataFrame` de entrada contém todas as 30 variáveis obrigatórias e retorna o diagnóstico previsto (`Benigno` ou `Maligno`) acompanhado das probabilidades calibradas.

### 11.2. Sistema Interativo de Triagem Médica (`src/app.py`)
Para demonstrar o **sistema em execução** de forma prática e intuitiva para equipes clínicas, desenvolvemos um painel em **Streamlit** organizado em 3 abas:
1. **🩺 Triagem de Paciente & Explicação SHAP:**
   - Permite carregar casos reais do conjunto de teste (Maligno, Benigno ou Limítrofe) ou ajustar manualmente os biomarcadores por *sliders*.
   - Inclui controle de **Limiar Clínico de Sensibilidade (*Threshold*)** (padrão sugerido `0.40` para priorizar o *Recall*).
   - Exibe instantaneamente a **Probabilidade de Malignidade**, a **Classificação Sugerida**, a **Prioridade na Fila de Triagem** (Alta, Média ou Rotina) e o gráfico **SHAP Waterfall** individual do paciente.
2. **📊 Avaliação de Modelos & Foco Clínico (Recall):**
   - Apresenta as tabelas comparativas de Validação e Teste, Matrizes de Confusão, Curvas ROC e gráficos globais de *Feature Importance* e *SHAP*.
3. **🔍 Exploração de Dados (EDA):**
   - Exibe os gráficos de balanceamento de classes, *boxplots* e matriz de correlação.

---

## 12. Como executar o projeto localmente

### Pré-requisitos
- **Python 3.10, 3.11 ou 3.12** instalado.
- **Git** (opcional, para clonagem).
- *(Não é necessário baixar o CSV manualmente: o script `src/pipeline.py` gera e salva automaticamente `data/raw/breast_cancer_wisconsin.csv` na primeira execução).*

### Passo a Passo (Windows PowerShell / Terminal)

1. **Acesse o diretório do projeto:**
   ```powershell
   cd C:\Users\ediga\.gemini\antigravity\scratch\tech-challenge-fase1
   ```

2. **Crie e ative um ambiente virtual:**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
   *(No Linux/macOS: `source .venv/bin/activate`)*

3. **Instale as dependências:**
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Execute o pipeline completo (EDA + Pré-processamento + Treinamento + Avaliação + SHAP):**
   ```powershell
   python src/pipeline.py
   ```
   *Ao concluir, todos os dados processados estarão em `data/processed/`, os modelos em `models/`, as tabelas em `reports/metrics/` e os 8 gráficos em `reports/figures/`.*

5. **Inicie a Aplicação Médica Interativa (Streamlit):**
   ```powershell
   streamlit run src/app.py
   ```
   O sistema abrirá automaticamente no navegador em **`http://localhost:8501`**.

---

## 13. Como executar utilizando Docker

O projeto conta com um [`Dockerfile`](file:///C:/Users/ediga\.gemini\antigravity\scratch\tech-challenge-fase1\Dockerfile) completo baseado em `python:3.11-slim` que instala as dependências, executa o pipeline de treinamento/SHAP durante o *build* e sobe o servidor **Streamlit** na porta `8501`.

### Pré-requisitos
- **Docker Desktop** instalado e em execução.

### 1. Construir a imagem Docker
Na raiz do projeto (`tech-challenge-fase1`), execute:

```powershell
docker build -t tech-challenge-fase1 .
```

### 2. Executar o container (Subindo o Sistema Web em Streamlit)
```powershell
docker run --rm -p 8501:8501 tech-challenge-fase1
```
Acesse no navegador: **`http://localhost:8501`**.

### 3. Executar o container mapeando volumes locais (Opcional)
Caso deseje executar apenas o pipeline e persistir os gráficos/modelos gerados diretamente nas pastas da sua máquina hospedeira:
```powershell
docker run --rm `
  -v "${PWD}/data:/app/data" `
  -v "${PWD}/models:/app/models" `
  -v "${PWD}/reports:/app/reports" `
  tech-challenge-fase1 python src/pipeline.py
```

---

## 14. Testes automatizados

Foram implementados testes automatizados em [`tests/test_predictor.py`](file:///C:/Users/ediga/.gemini/antigravity/scratch/tech-challenge-fase1/tests/test_predictor.py) utilizando `unittest` para validar:
- Integridade, dimensões (569 registros, 30 features) e codificação clínica do dataset;
- Divisão estratificada exata (398 treino / 85 validação / 86 teste) e ausência total de sobreposição (*Data Leakage*) entre os índices dos conjuntos;
- Carregamento do modelo persistido e inferência via `BreastCancerPredictor`.

Para executar a suíte de testes a partir da raiz do projeto:

```powershell
python -m unittest discover -s tests
```

---

## 15. Resultados e artefatos gerados

Após a execução de `python src/pipeline.py`, os seguintes artefatos ficam disponíveis para inspeção e inclusão no documento PDF de entrega:

- **Gráficos de Alta Resolução (`reports/figures/`):**
  - `01_distribuicao_classes.png`: Contagem e proporção de casos Benignos vs Malignos.
  - `02_boxplots_principais_variaveis.png`: Separação de medianas e *outliers* nas 8 variáveis mais discriminativas.
  - `03_matriz_correlacao.png`: Mapa de calor (*heatmap*) das 15 variáveis mais correlacionadas ao diagnóstico.
  - `04_matrizes_confusao_teste.png`: Matrizes de confusão lado a lado dos 3 modelos no conjunto de teste.
  - `05_curva_roc_comparativa.png`: Curvas ROC e valores de AUC comparados no conjunto de teste.
  - `06_feature_importance_rf.png`: Ranking das 15 variáveis mais importantes na Random Forest.
  - `07_shap_summary_plot.png`: Impacto global e direcionalidade das variáveis via SHAP (*Beeswarm*).
  - `08_shap_waterfall_paciente.png`: Explicação individual detalhada de um caso clínico maligno via SHAP (*Waterfall*).
- **Métricas Tabulares (`reports/metrics/`):**
  - `cross_validation_metrics.csv`: Médias de 5-Fold Stratified Cross-Validation.
  - `validation_metrics.csv`: Desempenho detalhado no conjunto de validação (15%).
  - `test_metrics.csv`: Desempenho final no conjunto de teste isolado (15%).
- **Relatório Técnico Consolidado:**
  - [`reports/relatorio_tecnico.md`](file:///C:/Users/ediga/.gemini/antigravity/scratch/tech-challenge-fase1/reports/relatorio_tecnico.md)

---

## 16. Discussão crítica: O modelo pode ser usado na prática? Como?

> **Pergunta do Edital:** *"Discuta os resultados de maneira crítica. O seu modelo pode ser utilizado na prática? Como? (Lembre-se que o médico sempre deve ter a palavra final no diagnóstico)."*

**Sim, o modelo apresenta maturidade metodológica e alto desempenho discriminativo (Recall e ROC-AUC > 95%) para atuar na prática hospitalar como um Sistema de Apoio à Decisão Clínica (CDSS), desde que respeitadas três diretrizes fundamentais:**

1. **Priorização Inteligente de Fila (Triagem Hospitalar):**
   - O modelo **não substitui o médico patologista ou oncologista**, mas atua na linha de frente assim que os parâmetros citopatológicos são digitalizados. Exames com alta probabilidade de malignidade recebem um *flag* de **Prioridade Alta**, furando a fila de espera para que o especialista emita o laudo definitivo com urgência.
2. **Segunda Opinião Transparente e Auditável (SHAP):**
   - Ao apresentar o gráfico **SHAP Waterfall** junto à probabilidade estimada na tela do prontuário (como demonstrado em nosso app Streamlit), o médico visualiza imediatamente quais biomarcadores (`worst radius`, `worst concave points`, `worst perimeter`) motivaram o alerta, reduzindo o risco de fadiga visual em plantões exaustivos.
3. **Soberania da Decisão Médica e Limitações do Estudo:**
   - Os modelos foram treinados em uma coorte específica de 569 pacientes (*Wisconsin*). Para implantação clínica real em um hospital universitário brasileiro, exige-se validação prospectiva local (calibração demográfica e de equipamentos de imagem) e monitoramento contínuo de *Data Drift*.
   - **A palavra final é sempre soberana do médico responsável**, que correlacionará o escore preditivo da IA com a anamnese clínica, exame físico, histórico familiar, mamografia/ultrassonografia e biópsia histopatológica padrão-ouro antes de definir qualquer conduta terapêutica.

---

**Projeto acadêmico desenvolvido para o Tech Challenge — Fase 1 | Pós Tech.**
